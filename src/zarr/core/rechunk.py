"""Copy an existing Zarr array into a new chunk layout under a decoded-memory budget.

This module implements :func:`rechunk_array`, the engine behind the public
``zarr.rechunk`` entry point.  The copy is planned so that the decoded working
memory never exceeds a caller-supplied budget:

* The array is processed in rectangular *copy units*.  A unit is always a
  whole number of target write chunks (shards when the target is sharded,
  otherwise chunks), so writes never have to read-modify-write an interior
  target chunk.
* While a unit is in flight, at most one source read chunk (a shard when the
  source is sharded) and one target write chunk are decoded on top of the unit
  buffer itself.  The planner grows the unit along each axis only while
  ``unit_bytes + max(read_chunk_bytes, write_chunk_bytes) <= max_mem``.
* If even the minimal necessary unit -- one source chunk plus one target
  chunk -- does not fit in ``max_mem``, a ``ValueError`` is raised before the
  target array is created.
* Units never span more of an axis than the budget allows, so the copy never
  assembles an entire axis (let alone the whole array) in memory unless the
  budget explicitly permits it.
* Trailing edge chunks, shards, and regions of the source that were never
  written (fill regions) all take part in the planning.  Units or chunks whose
  decoded content equals the fill value are not written to the target store
  (``write_empty_chunks=False`` semantics), so pure fill is never forced to
  materialize as real chunks.

Failure handling: the target array is created with a ``False`` value for the
:data:`RECHUNK_COMPLETE_ATTR` attribute, which is only flipped to ``True``
after every unit has been copied.  A target left behind by a failed copy is
therefore recognizable as incomplete via :func:`rechunk_is_complete`.  The
source array is never modified or deleted; removing it is deliberately not
part of this interface.
"""

from __future__ import annotations

import itertools
import math
import operator
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from zarr.core.array import (
    DEFAULT_FILL_VALUE,
    Array,
    AsyncArray,
    _parse_keep_array_attr,
    init_array,
)
from zarr.core.array_spec import parse_array_config
from zarr.core.config import config as _zarr_config
from zarr.storage import MemoryStore
from zarr.storage._common import make_store_path

if TYPE_CHECKING:
    from zarr.core.array import CompressorsLike, FiltersLike, SerializerLike
    from zarr.core.array_spec import ArrayConfigLike
    from zarr.storage import StoreLike
    from zarr.types import AnyArray, AnyAsyncArray

__all__ = ["RECHUNK_COMPLETE_ATTR", "rechunk_array", "rechunk_is_complete"]

RECHUNK_COMPLETE_ATTR = "_rechunk_complete"
"""Attribute key recording whether a rechunked target array was fully copied."""


def rechunk_is_complete(array: AnyArray | AnyAsyncArray) -> bool:
    """Return True if ``array`` is a rechunk target whose copy ran to completion.

    Arrays that were not produced by :func:`rechunk_array`, and targets left
    behind by a failed copy, both report ``False``.
    """
    return bool(array.attrs.get(RECHUNK_COMPLETE_ATTR, False))


def _normalize_grid_spec(value: int | tuple[int, ...], *, name: str, ndim: int) -> tuple[int, ...]:
    """Validate a chunk/shard specification as a tuple of ``ndim`` positive ints."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = (int(value),) * ndim
    if isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int or a sequence of ints, got {value!r}.")
    try:
        result = tuple(operator.index(v) for v in value)
    except TypeError:
        raise TypeError(f"{name} must be an int or a sequence of ints, got {value!r}.") from None
    if len(result) != ndim:
        raise ValueError(
            f"{name} must have one entry per array dimension ({ndim}), got {result!r}."
        )
    if any(v <= 0 for v in result):
        raise ValueError(f"{name} entries must be positive integers, got {result!r}.")
    return result


def _plan_copy_unit(
    *,
    shape: tuple[int, ...],
    write_granularity: tuple[int, ...],
    temp_bytes: int,
    itemsize: int,
    max_mem: int,
) -> tuple[int, ...]:
    """Choose the copy-unit shape, a per-axis multiple of ``write_granularity``.

    The unit starts at a single target write chunk and is grown axis by axis
    while ``unit_bytes + temp_bytes <= max_mem``.  ``temp_bytes`` reserves room
    for the largest transient decoded buffer (one source read chunk or one
    target write chunk, whichever is bigger) that must coexist with the unit.
    """
    unit = list(write_granularity)
    unit_bytes = math.prod(unit) * itemsize
    if unit_bytes + temp_bytes > max_mem:
        raise ValueError(
            "max_mem is too small for this rechunking: the minimal copy unit "
            f"(one target write chunk of {unit_bytes} bytes plus one transient "
            f"chunk buffer of {temp_bytes} bytes) requires "
            f"{unit_bytes + temp_bytes} bytes, but max_mem is {max_mem}."
        )
    budget_elements = (max_mem - temp_bytes) // itemsize
    for axis in range(len(unit)):
        cover = math.ceil(shape[axis] / write_granularity[axis]) * write_granularity[axis]
        others = math.prod(unit[other] for other in range(len(unit)) if other != axis)
        if others == 0:
            continue
        max_len = budget_elements // others
        grown = (max_len // write_granularity[axis]) * write_granularity[axis]
        unit[axis] = min(cover, max(unit[axis], grown))
    return tuple(unit)


def _is_all_fill(data: Any, fill_value: Any) -> bool:
    """Best-effort check that a decoded region consists purely of the fill value."""
    if fill_value is None:
        return False
    try:
        return bool(np.array_equal(data, fill_value, equal_nan=True))
    except (TypeError, ValueError):
        return False


async def rechunk_array(
    source: AnyArray | AnyAsyncArray,
    chunks: int | tuple[int, ...],
    max_mem: int,
    *,
    shards: int | tuple[int, ...] | None = None,
    filters: FiltersLike | Literal["keep"] = "keep",
    compressors: CompressorsLike | Literal["keep"] = "keep",
    serializer: SerializerLike | Literal["keep"] = "keep",
    store: StoreLike | None = None,
    name: str | None = None,
    overwrite: bool = False,
    storage_options: dict[str, Any] | None = None,
    config: ArrayConfigLike | None = None,
) -> AnyAsyncArray:
    """Copy ``source`` into a new array with a different chunk layout.

    The new array keeps the source's shape, dtype, fill value, zarr format,
    dimension names, and attributes; only the chunk grid (and optionally the
    shard grid and the codecs) change.  The copy is planned so that the decoded
    working memory stays within ``max_mem`` bytes; see the module docstring for
    the exact memory model.

    Parameters
    ----------
    source : Array | AsyncArray
        The array to copy.  It is only read, never modified or deleted.
    chunks : int or tuple[int, ...]
        Chunk shape of the new array.  An int is broadcast to every dimension.
    max_mem : int
        Maximum decoded working memory in bytes.  This is a memory budget, not
        a task count: the copy unit is grown to use as much of the budget as
        possible, and a ``ValueError`` is raised before anything is written if
        even one source chunk plus one target chunk cannot fit.
    shards : int or tuple[int, ...], optional
        Shard shape of the new array.  Each entry must be a multiple of the
        corresponding ``chunks`` entry.  The default of ``None`` creates an
        unsharded array.
    filters, compressors, serializer : "keep" or codec specification, optional
        Codec configuration of the new array.  ``"keep"`` (the default)
        retains the source's configuration; pass any value accepted by
        :func:`zarr.create_array` to change it, e.g. a different compressor.
    store : StoreLike, optional
        Destination store.  Defaults to a new in-memory store.
    name : str, optional
        Path of the new array within ``store``.
    overwrite : bool, default False
        Whether to overwrite an existing array at the destination.
    storage_options : dict, optional
        Passed to the store constructor when ``store`` is a URL.
    config : ArrayConfigLike, optional
        Runtime configuration for the new array.

    Returns
    -------
    AsyncArray
        The newly created array.  Its ``_rechunk_complete`` attribute is
        ``True`` only if every value was copied; see
        :func:`rechunk_is_complete`.

    Notes
    -----
    The source array is intentionally left in place; deleting it after a
    successful rechunk is the caller's responsibility and not part of this
    interface.
    """
    if isinstance(source, Array):
        source_sync = source
    elif isinstance(source, AsyncArray):
        source_sync = Array(source)
    else:
        raise TypeError(f"source must be a zarr Array or AsyncArray, got {type(source)!r}.")
    source_async = source_sync._async_array

    try:
        max_mem = operator.index(max_mem)
    except TypeError:
        raise TypeError(f"max_mem must be an integer number of bytes, got {max_mem!r}.") from None
    if max_mem <= 0:
        raise ValueError(f"max_mem must be a positive number of bytes, got {max_mem}.")

    shape = source_async.shape
    ndim = len(shape)
    chunks_parsed = _normalize_grid_spec(chunks, name="chunks", ndim=ndim)
    shards_parsed: tuple[int, ...] | None = None
    if shards is not None:
        shards_parsed = _normalize_grid_spec(shards, name="shards", ndim=ndim)
        for axis, (shard_size, chunk_size) in enumerate(
            zip(shards_parsed, chunks_parsed, strict=True)
        ):
            if shard_size % chunk_size != 0:
                raise ValueError(
                    f"shards must be a multiple of chunks along every dimension; "
                    f"axis {axis} has shard size {shard_size} and chunk size {chunk_size}."
                )

    itemsize = source_async.dtype.itemsize

    # Memory model: while a copy unit is in flight, the largest transient
    # decoded buffer alongside it is one source read chunk (a whole shard when
    # the source is sharded) during reads, or one target write chunk (a whole
    # shard when the target is sharded) during edge-chunk merges on writes.
    # Reads and writes happen in separate phases, so only the larger of the
    # two has to be reserved.  Sizes are clipped to the array extent so that
    # trailing edge chunks and shards are accounted for at their real size.
    read_granularity = tuple(max(sizes) if sizes else 1 for sizes in source_async.write_chunk_sizes)
    write_granularity = shards_parsed if shards_parsed is not None else chunks_parsed
    read_temp_bytes = (
        math.prod(min(gran, extent) for gran, extent in zip(read_granularity, shape, strict=True))
        * itemsize
    )
    write_temp_bytes = (
        math.prod(min(gran, extent) for gran, extent in zip(write_granularity, shape, strict=True))
        * itemsize
    )
    temp_bytes = max(read_temp_bytes, write_temp_bytes)

    # Raises before the target exists when the minimal unit cannot fit.
    unit = _plan_copy_unit(
        shape=shape,
        write_granularity=write_granularity,
        temp_bytes=temp_bytes,
        itemsize=itemsize,
        max_mem=max_mem,
    )

    (
        chunks_resolved,
        shards_resolved,
        filters_resolved,
        compressors_resolved,
        serializer_resolved,
        fill_value_resolved,
        order_resolved,
        zarr_format_resolved,
        chunk_key_encoding_resolved,
        dimension_names_resolved,
        attributes_resolved,
    ) = _parse_keep_array_attr(
        data=source_sync,
        chunks=chunks_parsed,
        shards=shards_parsed,
        filters=filters,
        compressors=compressors,
        serializer=serializer,
        fill_value=DEFAULT_FILL_VALUE,
        order=None,
        zarr_format=None,
        chunk_key_encoding=None,
        dimension_names=None,
        attributes=None,
    )
    attributes = dict(attributes_resolved or {})
    # Mark the target as incomplete up front: any target left behind by a
    # failed copy is recognizable through `rechunk_is_complete`.
    attributes[RECHUNK_COMPLETE_ATTR] = False

    store_path = await make_store_path(
        store if store is not None else MemoryStore(),
        path=name,
        mode="a",
        storage_options=storage_options,
    )
    target = await init_array(
        store_path=store_path,
        shape=shape,
        dtype=source_sync._async_array._zdtype,
        chunks=chunks_resolved,
        shards=shards_resolved,
        filters=filters_resolved,
        compressors=compressors_resolved,
        serializer=serializer_resolved,
        fill_value=fill_value_resolved,
        order=order_resolved,
        zarr_format=zarr_format_resolved,
        attributes=attributes,
        chunk_key_encoding=chunk_key_encoding_resolved,
        dimension_names=dimension_names_resolved,
        overwrite=overwrite,
        config=parse_array_config(config),
    )

    fill_value = source_async.metadata.fill_value
    if all(extent > 0 for extent in shape):
        ranges = [range(0, extent, step) for extent, step in zip(shape, unit, strict=True)]
        # Chunk-level concurrency is pinned to 1 so that the transient
        # decoded buffers stay within the budget the caller paid for;
        # `async.concurrency` would otherwise multiply them.
        with _zarr_config.set({"async.concurrency": 1}):
            for starts in itertools.product(*ranges):
                region = tuple(
                    slice(start, min(start + step, extent))
                    for start, step, extent in zip(starts, unit, shape, strict=True)
                )
                data = await source_async.getitem(region)
                if _is_all_fill(data, fill_value):
                    # Pure fill must not be forced to materialize as real chunks.
                    continue
                await target.setitem(region, data)

    await target.update_attributes({RECHUNK_COMPLETE_ATTR: True})
    return target
