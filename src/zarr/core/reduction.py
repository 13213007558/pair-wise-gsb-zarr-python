"""Streaming chunk-wise reductions (sum, mean, variance) over Zarr arrays.

The functions in this module compute reductions without materializing the whole
array in memory: chunks are read one at a time, reduced to per-chunk sufficient
statistics (count, sum, and a sum of squared deviations), and those statistics
are merged across chunks.  Variance is combined with the numerically stable
parallel algorithm of Chan et al. (a generalization of Welford's method), so
the result never relies on subtracting two large, nearly equal sums of squares.
The working set therefore scales with the chunk size (plus the reduced output),
not with the total number of rows of the array.

Results are independent of the chunk layout up to floating-point rounding:
for well-scaled data they agree with the equivalent NumPy reduction to within
a small multiple of machine epsilon.  For data with a large mean and a small
variance, the relative error of the variance grows like
``eps * (mean / std) ** 2`` (as it does for any floating-point algorithm that
centers on a computed mean), while remaining orders of magnitude more accurate
than a naive sum-of-squares computation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from numpy.exceptions import AxisError

from zarr.core.buffer import default_buffer_prototype
from zarr.core.indexing import BasicIndexer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr.core.array import AsyncArray

__all__ = ["NanPolicy", "reduce_array"]

type NanPolicy = Literal["propagate", "omit"]
"""How to treat NaN values during a reduction.

- "propagate": any NaN in the input poisons the output cell it contributes
  to (NumPy's default behavior).
- "omit": NaN values are excluded from the statistics, as with
  np.nansum / np.nanmean / np.nanvar.
"""

type ReductionOp = Literal["sum", "mean", "variance"]

_REDUCIBLE_KINDS = "biufc"


def _parse_nan_policy(nan_policy: str) -> NanPolicy:
    if nan_policy not in ("propagate", "omit"):
        msg = f"nan_policy must be 'propagate' or 'omit', got {nan_policy!r}"
        raise ValueError(msg)
    return cast("NanPolicy", nan_policy)


def _normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    """Normalize an axis specification to a sorted tuple of unique non-negative ints."""
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, (int, np.integer)):
        raw_axes = (int(axis),)
    else:
        raw_axes = tuple(int(a) for a in axis)
    axes: list[int] = []
    for ax in raw_axes:
        if ax < -ndim or ax >= ndim:
            raise AxisError(ax, ndim=ndim)
        ax %= ndim
        if ax in axes:
            msg = "duplicate value in 'axis'"
            raise ValueError(msg)
        axes.append(ax)
    return tuple(sorted(axes))


def _default_accumulator_dtype(dtype: np.dtype[Any], op: ReductionOp) -> np.dtype[Any]:
    """Pick the default accumulator dtype for a reduction.

    For "sum" this follows NumPy's default promotion rules (e.g. int8 sums
    accumulate in the platform default integer, float32 in float32).
    For "mean" and "variance" the accumulator is at least float64 /
    complex128 so that merging statistics across many chunks stays
    numerically stable, even when the stored dtype has lower precision.
    """
    if op == "sum":
        return np.empty(0, dtype=dtype).sum().dtype
    if dtype.kind == "c":
        return np.dtype(np.result_type(dtype, np.complex128))
    return np.dtype(np.result_type(dtype, np.float64))


def _resolve_accumulator_dtype(
    dtype: np.dtype[Any], op: ReductionOp, dtype_like: npt.DTypeLike | None
) -> np.dtype[Any]:
    if dtype_like is None:
        return _default_accumulator_dtype(dtype, op)
    acc_dtype = np.dtype(dtype_like)
    if op == "sum":
        if acc_dtype.kind not in _REDUCIBLE_KINDS:
            msg = f"accumulator dtype {acc_dtype} is not supported for sum"
            raise TypeError(msg)
    elif acc_dtype.kind not in "fc":
        msg = f"accumulator dtype for {op} must be a floating or complex dtype, got {acc_dtype}"
        raise TypeError(msg)
    return acc_dtype


def _chunk_statistics(
    chunk: npt.NDArray[Any],
    axes: tuple[int, ...],
    acc_dtype: np.dtype[Any],
    m2_dtype: np.dtype[Any],
    omit_nan: bool,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Compute (count, sum, M2) of a chunk, reduced over "axes".

    M2 is the sum of squared deviations from the chunk mean.  The returned
    arrays have the chunk's shape with "axes" removed.
    """
    expand = tuple(np.newaxis if dim in axes else slice(None) for dim in range(chunk.ndim))
    if omit_nan:
        mask = ~np.isnan(chunk)
        count = mask.sum(axis=axes, dtype=np.int64)
        total = np.nansum(chunk, axis=axes, dtype=acc_dtype)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count
        # Cells with no valid observations have an undefined mean, but every
        # contributing value is masked out below, so they contribute 0 to M2.
        centered = np.where(mask, chunk - mean[expand], 0)
        m2 = (np.abs(centered) ** 2).sum(axis=axes, dtype=m2_dtype)
    else:
        count_n = math.prod(chunk.shape[dim] for dim in axes)
        kept_shape = tuple(chunk.shape[dim] for dim in range(chunk.ndim) if dim not in axes)
        count = np.full(kept_shape, count_n, dtype=np.int64)
        total = chunk.sum(axis=axes, dtype=acc_dtype)
        mean = total / count_n
        centered = chunk - mean[expand]
        m2 = (np.abs(centered) ** 2).sum(axis=axes, dtype=m2_dtype)
    return count, total, m2


def _combine_statistics(
    count_a: npt.NDArray[Any],
    total_a: npt.NDArray[Any],
    m2_a: npt.NDArray[Any],
    count_b: npt.NDArray[Any],
    total_b: npt.NDArray[Any],
    m2_b: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Merge two sets of (count, sum, M2) statistics.

    Uses the parallel combination formula of Chan et al.: with
    delta = mean_b - mean_a, the combined M2 is
    M2_a + M2_b + delta**2 * (n_a * n_b) / (n_a + n_b).  This is
    numerically stable even when the values have a large mean and a small
    variance, unlike subtracting raw sums of squares.
    """
    count = count_a + count_b
    total = total_a + total_b
    both_nonempty = (count_a > 0) & (count_b > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = total_b / count_b - total_a / count_a
        weight = (count_a * count_b) / count
        correction = np.abs(delta) ** 2 * weight
    # Where either side is empty the correction is undefined (0/0); the merged
    # statistics are then just the other side's, so the correction is 0.
    correction = np.where(both_nonempty, correction, 0.0)
    m2 = m2_a + m2_b + correction
    return count, total, m2


def _finalize(
    op: ReductionOp,
    count: npt.NDArray[Any],
    total: npt.NDArray[Any],
    m2: npt.NDArray[Any],
    ddof: int,
    acc_dtype: np.dtype[Any],
    m2_dtype: np.dtype[Any],
) -> npt.NDArray[Any]:
    """Turn merged (count, sum, M2) statistics into the reduction result."""
    if op == "sum":
        # The additive identity: an empty reduction sums to 0.
        return total.astype(acc_dtype, copy=False)
    if op == "mean":
        empty = count == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            result = total / count
        if empty.any():
            warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=4)
            result = np.where(empty, np.nan, result)
        return result.astype(acc_dtype, copy=False)
    denom = count - ddof
    invalid = denom <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        result = m2 / denom
    if invalid.any():
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=4)
        result = np.where(invalid, np.nan, result)
    return result.astype(m2_dtype, copy=False)


async def reduce_array(
    array: AsyncArray[Any],
    op: ReductionOp,
    *,
    axis: int | Sequence[int] | None = None,
    dtype: npt.DTypeLike | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
    out: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any] | Any:
    """Compute a streaming reduction over a Zarr array, chunk by chunk.

    Parameters
    ----------
    array : AsyncArray
        The array to reduce.
    op : {"sum", "mean", "variance"}
        The reduction to compute.
    axis : int or Sequence[int] or None, optional
        Axis or axes along which to reduce.  The default, axis=None,
        reduces over all axes.
    dtype : dtype-like, optional
        Data type of the accumulator (and, for floating point results, of the
        output).  Defaults to NumPy's promotion rules for "sum" and to at
        least float64 / complex128 for "mean" and "variance".
    ddof : int, default=0
        Delta degrees of freedom for "variance": the divisor is
        N - ddof where N is the number of (non-omitted) observations.
    keepdims : bool, default=False
        If True, reduced axes are left in the result with size 1.
    nan_policy : {"propagate", "omit"}, default="propagate"
        Whether NaN values poison the result ("propagate") or are excluded
        from the statistics ("omit").
    out : numpy.ndarray, optional
        Array into which the result is written.  It must have the shape of the
        reduced result (after applying keepdims).

    Returns
    -------
    numpy.ndarray or scalar
        The reduced result, or "out" if it was provided.

    Notes
    -----
    Chunks that have not been written to the store are read as the array's
    fill_value and are counted as observations, exactly as if the fill
    value had been written to every cell.
    """
    # Imported here to avoid a circular import at module load time.
    from zarr.core.array import _get_selection, _iter_chunk_regions

    nan_policy = _parse_nan_policy(nan_policy)
    if ddof < 0:
        msg = f"ddof must be non-negative, got {ddof}"
        raise ValueError(msg)
    axes = _normalize_axes(axis, array.ndim)
    in_dtype = np.dtype(array.dtype)
    if in_dtype.kind not in _REDUCIBLE_KINDS:
        msg = f"{op} is not supported for arrays with dtype {in_dtype}"
        raise TypeError(msg)
    acc_dtype = _resolve_accumulator_dtype(in_dtype, op, dtype)
    m2_dtype = np.empty((), dtype=acc_dtype).real.dtype

    shape = array.shape
    kept_shape = tuple(shape[dim] for dim in range(array.ndim) if dim not in axes)

    count = np.zeros(kept_shape, dtype=np.int64)
    total = np.zeros(kept_shape, dtype=acc_dtype)
    m2 = np.zeros(kept_shape, dtype=m2_dtype)

    # NaNs can only occur in floating point and complex data; for other dtypes
    # "omit" degrades gracefully to "propagate".
    omit_nan = nan_policy == "omit" and in_dtype.kind in "fc"
    # Unwritten chunks represent fill_value and must be counted as
    # observations, so force the fill-on-missing read path.
    config = replace(array.config, read_missing_chunks=True)
    prototype = default_buffer_prototype()

    for region in _iter_chunk_regions(array):
        indexer = BasicIndexer(region, shape=shape, chunk_grid=array._chunk_grid)
        chunk = await _get_selection(
            array.store_path,
            array.metadata,
            array.codec_pipeline,
            config,
            array._chunk_grid,
            indexer,
            prototype=prototype,
        )
        chunk_stats = _chunk_statistics(np.asarray(chunk), axes, acc_dtype, m2_dtype, omit_nan)
        # The chunk spans a sub-region of the kept (non-reduced) axes, so its
        # statistics are merged into the matching slice of the accumulators.
        kept_index = tuple(region[dim] for dim in range(array.ndim) if dim not in axes)
        merged = _combine_statistics(
            count[kept_index], total[kept_index], m2[kept_index], *chunk_stats
        )
        count[kept_index], total[kept_index], m2[kept_index] = merged

    result = _finalize(op, count, total, m2, ddof, acc_dtype, m2_dtype)
    if keepdims:
        final_shape = tuple(1 if dim in axes else shape[dim] for dim in range(array.ndim))
        result = result.reshape(final_shape)
    else:
        result = result.reshape(kept_shape)

    if out is not None:
        out_array = np.asarray(out)
        if out_array.shape != result.shape:
            msg = (
                f"shape of out argument doesn't match. "
                f"Expected {result.shape}, got {out_array.shape}"
            )
            raise ValueError(msg)
        out_array[...] = result
        return out
    if result.shape == ():
        return result[()]
    return result
