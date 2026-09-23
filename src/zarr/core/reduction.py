"""
Chunked, memory-bounded reductions for zarr arrays.

This module implements sum, mean and variance reductions over one or more
axes of a zarr array without materializing the whole array in memory. Data
is processed one stored block (shard, or chunk when sharding is not used) at
a time, so the working set scales with the block size and the size of the
reduced result, not with the total number of rows.

Moments (count / mean / M2) are combined across blocks with the pairwise
update of Chan et al. (1979), which is numerically stable: the variance is
never computed by subtracting two large, nearly equal sums of squares.

Blocks that have not been written to storage are treated as if they were
filled with the fill value of the array and therefore contribute to the
sample count; they are *not* treated as missing observations.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from zarr.core.array import AsyncArray

from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

type AnyAsyncArray = AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]
__all__ = ["async_reduce_mean", "async_reduce_sum", "async_reduce_variance"]

NaNPolicy = Literal["propagate", "omit"]

_ReduceOp = Literal["sum", "mean", "variance"]

_AnyNDArray = np.ndarray[Any, np.dtype[Any]]


def _normalize_nan_policy(nan_policy: str) -> NaNPolicy:
    if nan_policy not in ("propagate", "omit"):
        msg = f"nan_policy must be 'propagate' or 'omit', got {nan_policy!r}."
        raise ValueError(msg)
    return cast("NaNPolicy", nan_policy)


def _normalize_axes(axis: int | Iterable[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, (int, np.integer)):
        axis = (int(axis),)
    return normalize_axis_tuple(tuple(axis), ndim)


def _read_block_boundaries(
    array: AnyAsyncArray,
) -> list[list[tuple[int, int]]]:
    """
    Compute, per axis, the (start, stop) pairs of the blocks used for reading.

    Reads happen at shard granularity when sharding is used (so each stored
    object is decoded at most once) and at chunk granularity otherwise.
    Boundary blocks are clipped to the array extent.
    """
    shape = array.shape
    try:
        steps: tuple[int, ...] | None = array.shards or array.chunks
    except NotImplementedError:
        # Rectilinear chunk grid without a regular chunk shape.
        steps = None
    boundaries: list[list[tuple[int, int]]] = []
    if steps is not None:
        for size, step in zip(shape, steps, strict=True):
            boundaries.append([(start, min(start + step, size)) for start in range(0, size, step)])
    else:
        for size, sizes in zip(shape, array.read_chunk_sizes, strict=True):
            starts = itertools.accumulate(sizes, initial=0)
            boundaries.append(
                [
                    (start, min(start + length, size))
                    for start, length in zip(starts, sizes, strict=True)
                ]
            )
    return boundaries


async def _iter_blocks(
    array: AnyAsyncArray, axes: tuple[int, ...]
) -> AsyncIterator[tuple[_AnyNDArray, tuple[slice, ...]]]:
    """
    Yield the data of the array one read block at a time as NumPy arrays.

    Each block is paired with the selection (in keepdims shape) that locates
    the reduced statistics of the block within the global accumulators.
    """
    boundaries = _read_block_boundaries(array)
    for block_index in itertools.product(*(range(len(b)) for b in boundaries)):
        spans = [boundaries[dim][i] for dim, i in enumerate(block_index)]
        selection = tuple(slice(start, stop) for start, stop in spans)
        out_selection = tuple(
            slice(0, 1) if dim in axes else slice(start, stop)
            for dim, (start, stop) in enumerate(spans)
        )
        yield np.asarray(await array.getitem(selection)), out_selection


def _supports_nan(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.inexact)


def _default_sum_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    # Mirror the type promotion of np.sum (integers are promoted to the
    # platform default integer, floats keep their precision).
    return cast("np.dtype[Any]", np.zeros(1, dtype=dtype).sum().dtype)


def _real_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    # The real-valued counterpart of dtype (used for M2 accumulators).
    return np.empty((), dtype=dtype).real.dtype


@dataclass
class _MomentsAccumulator:
    """Running count/mean/M2 over blocks, merged with the update of Chan et al."""

    count: _AnyNDArray  # int64, keepdims shape
    mean: _AnyNDArray  # accumulator dtype, keepdims shape
    m2: _AnyNDArray  # real accumulator dtype, keepdims shape

    @classmethod
    def empty(cls, shape: tuple[int, ...], mean_dtype: np.dtype[Any]) -> _MomentsAccumulator:
        return cls(
            count=np.zeros(shape, dtype=np.int64),
            mean=np.zeros(shape, dtype=mean_dtype),
            m2=np.zeros(shape, dtype=_real_dtype(mean_dtype)),
        )

    def update(
        self,
        count_b: _AnyNDArray,
        mean_b: _AnyNDArray,
        m2_b: _AnyNDArray,
        selection: tuple[slice, ...],
    ) -> None:
        count_a = self.count[selection]
        mean_a = self.mean[selection]
        m2_a = self.m2[selection]
        n = count_a + count_b
        nonempty = n > 0
        # Where a block contributes no observations (possible with
        # nan_policy='omit'), its mean/M2 are zero and the ratios below are 0,
        # leaving the running values unchanged.
        delta = np.where(nonempty, mean_b - mean_a, 0)
        ratio = np.divide(count_b, n, out=np.zeros(n.shape, dtype=mean_a.dtype), where=nonempty)
        mean_a[...] = mean_a + delta * ratio
        m2_a[...] = m2_a + m2_b + np.abs(delta) ** 2 * count_a * ratio
        count_a[...] = n


def _block_moments(
    block: _AnyNDArray,
    axes: tuple[int, ...],
    mean_dtype: np.dtype[Any],
    nan_omit: bool,
) -> tuple[_AnyNDArray, _AnyNDArray, _AnyNDArray]:
    """Count, mean and M2 of one block, reduced over axes with keepdims."""
    valid: _AnyNDArray | None = None
    if nan_omit:
        valid = ~np.isnan(block)
        count = valid.sum(axis=axes, keepdims=True, dtype=np.int64)
        clean = np.where(valid, block, 0).astype(mean_dtype, copy=False)
    else:
        count = np.full(
            tuple(1 if i in axes else s for i, s in enumerate(block.shape)),
            math.prod(block.shape[i] for i in axes),
            dtype=np.int64,
        )
        clean = block.astype(mean_dtype, copy=False)
    total = clean.sum(axis=axes, keepdims=True)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    centered = clean - mean
    if valid is not None:
        centered = np.where(valid, centered, 0)
    m2 = (np.abs(centered) ** 2).sum(axis=axes, keepdims=True, dtype=_real_dtype(mean_dtype))
    return count, mean, m2


def _finalize(
    result: _AnyNDArray,
    axes: tuple[int, ...],
    keepdims: bool,
    out: _AnyNDArray | None,
) -> _AnyNDArray | np.generic:
    if not keepdims:
        result = result.squeeze(axis=axes)
    if out is not None:
        if tuple(out.shape) != result.shape:
            msg = (
                f"Shape of out argument {tuple(out.shape)} does not match the "
                f"reduction result shape {result.shape}."
            )
            raise ValueError(msg)
        out[...] = result
        return out
    if result.shape == ():
        return result[()]
    return result


async def _async_reduce(
    array: AnyAsyncArray,
    op: _ReduceOp,
    *,
    axis: int | Iterable[int] | None,
    keepdims: bool,
    dtype: np.dtype[Any] | None,
    ddof: int,
    nan_policy: NaNPolicy,
    out: _AnyNDArray | None,
) -> _AnyNDArray | np.generic:
    if ddof < 0:
        msg = f"ddof must be non-negative, got {ddof}."
        raise ValueError(msg)

    native_dtype = cast("np.dtype[Any]", array.dtype)
    axes = _normalize_axes(axis, array.ndim)
    keepdims_shape = tuple(1 if i in axes else s for i, s in enumerate(array.shape))

    if op == "sum":
        acc_dtype = np.dtype(dtype) if dtype is not None else _default_sum_dtype(native_dtype)
    else:
        acc_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
        if not np.issubdtype(acc_dtype, np.inexact):
            msg = (
                f"The accumulator dtype for {op} must be a floating point or "
                f"complex dtype, got {acc_dtype}."
            )
            raise ValueError(msg)

    nan_omit = nan_policy == "omit" and _supports_nan(native_dtype)

    result: _AnyNDArray
    if op == "sum":
        total = np.zeros(keepdims_shape, dtype=acc_dtype)
        async for block, out_selection in _iter_blocks(array, axes):
            if nan_omit:
                block = np.where(np.isnan(block), 0, block)
            total[out_selection] += block.sum(axis=axes, keepdims=True, dtype=acc_dtype)
        result = total
    else:
        moments = _MomentsAccumulator.empty(keepdims_shape, acc_dtype)
        async for block, out_selection in _iter_blocks(array, axes):
            moments.update(
                *_block_moments(block, axes, acc_dtype, nan_omit), selection=out_selection
            )
        if op == "mean":
            # Empty reductions (count == 0) are defined as NaN.
            result = np.where(moments.count > 0, moments.mean, np.nan)
        else:
            denominator = moments.count - ddof
            # Variance of fewer than ddof + 1 observations is defined as NaN.
            result = np.divide(
                moments.m2,
                denominator,
                out=np.full_like(moments.m2, np.nan),
                where=denominator > 0,
            )
    return _finalize(np.asarray(result), axes, keepdims, out)


async def async_reduce_sum(
    array: AnyAsyncArray,
    axis: int | Iterable[int] | None = None,
    *,
    keepdims: bool = False,
    dtype: np.dtype[Any] | None = None,
    nan_policy: NaNPolicy = "propagate",
    out: _AnyNDArray | None = None,
) -> _AnyNDArray | np.generic:
    """See AsyncArray.sum."""
    return await _async_reduce(
        array,
        "sum",
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        ddof=0,
        nan_policy=_normalize_nan_policy(nan_policy),
        out=out,
    )


async def async_reduce_mean(
    array: AnyAsyncArray,
    axis: int | Iterable[int] | None = None,
    *,
    keepdims: bool = False,
    dtype: np.dtype[Any] | None = None,
    nan_policy: NaNPolicy = "propagate",
    out: _AnyNDArray | None = None,
) -> _AnyNDArray | np.generic:
    """See AsyncArray.mean."""
    return await _async_reduce(
        array,
        "mean",
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        ddof=0,
        nan_policy=_normalize_nan_policy(nan_policy),
        out=out,
    )


async def async_reduce_variance(
    array: AnyAsyncArray,
    axis: int | Iterable[int] | None = None,
    *,
    ddof: int = 0,
    keepdims: bool = False,
    dtype: np.dtype[Any] | None = None,
    nan_policy: NaNPolicy = "propagate",
    out: _AnyNDArray | None = None,
) -> _AnyNDArray | np.generic:
    """See AsyncArray.variance."""
    return await _async_reduce(
        array,
        "variance",
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        ddof=ddof,
        nan_policy=_normalize_nan_policy(nan_policy),
        out=out,
    )
