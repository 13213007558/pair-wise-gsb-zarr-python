"""Copy a Zarr array into a new chunk layout within a decoded-working-memory budget.

This module implements :func:`rechunk_array`, an entry point that copies an
existing Zarr array to a new location with a different (regular) chunk grid,
optionally swapping the compressor.  The copy never changes the shape, dtype,
fill value, or values of the array.

Design notes
------------
* The caller supplies a *decoded working memory* budget in bytes (``max_mem``),
  not a task count.  At any moment the copy holds at most one decoded source
  read-chunk and one decoded target chunk (the staging buffer), so the minimal
  necessary unit is ``max_source_chunk_bytes + max_target_chunk_bytes``.  If
  that unit does not fit in ``max_mem`` the function raises *before* anything
  is written to the destination.
* The copy walks target chunks one at a time and, for each of them, reads only
  the intersections with overlapping source chunks.  It never assembles a whole
  axis (or any other multi-chunk slab) into a single in-memory array.
* Sharded sources are planned at inner-chunk granularity
  (:attr:`AsyncArray.read_chunk_sizes`), boundary ("edge") chunks are clipped
  to the array extent, and regions of the source that were never written read
  back as the fill value.  Target chunks whose content is entirely the fill
  value are never materialized in the store.
* The array metadata document is written *last*.  A destination whose copy
  failed (or is still running) therefore has no array metadata and cannot be
  opened as an array -- that is how an incomplete copy is recognized.  Removing
  the old array (or the orphaned chunks of a failed copy) is deliberately not
  part of this interface.
"""

from __future__ import annotations

import dataclasses
import math
from bisect import bisect_right
from itertools import product as iter_product
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.abc.codec import BytesBytesCodec
from zarr.core.array import (
    Array,
    AsyncArray,
    _parse_chunk_encoding_v2,
    _parse_chunk_encoding_v3,
    _prepare_overwrite,
    _sharding_codec,
)
from zarr.core.array_spec import ArrayConfig
from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import product
from zarr.core.indexing import BasicIndexer
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.v3 import RegularChunkGridMetadata
from zarr.storage._common import make_store_path

if TYPE_CHECKING:
    from zarr.core.array import CompressorsLike
    from zarr.core.metadata import ArrayMetadata
    from zarr.storage import StoreLike

__all__ = ["minimum_rechunk_memory", "rechunk_array"]


def _parse_chunks(chunks: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if not isinstance(chunks, tuple):
        chunks = tuple(chunks)
    if len(chunks) != ndim:
        raise ValueError(
            f"chunks must have one entry per array dimension. "
            f"Got {len(chunks)} entries for a {ndim}-dimensional array."
        )
    for c in chunks:
        if not isinstance(c, int | np.integer) or c < 1:
            raise ValueError(f"each chunk size must be a positive integer. Got {c!r}.")
    return tuple(int(c) for c in chunks)


def minimum_rechunk_memory(source: AsyncArray[Any] | Array, chunks: tuple[int, ...]) -> int:
    """
    The minimal decoded working memory, in bytes, needed to copy ``source``
    onto the chunk layout ``chunks``.

    This is the sum of the largest decoded source read-chunk (the inner chunk
    when the source is sharded) and the largest decoded target chunk.  Both
    the source chunk being decoded and the target chunk being assembled occupy
    the budget at the same time.
    """
    if isinstance(source, Array):
        source = source.async_array
    chunks_parsed = _parse_chunks(chunks, source.ndim)
    itemsize = source.dtype.itemsize
    read_sizes = source.read_chunk_sizes
    source_unit = product(tuple(max(sizes) for sizes in read_sizes)) * itemsize
    target_unit = product(chunks_parsed) * itemsize
    return source_unit + target_unit


def _target_metadata(
    source_metadata: ArrayMetadata,
    chunks: tuple[int, ...],
    compressors: CompressorsLike,
) -> ArrayMetadata:
    """Build the destination metadata: new chunk grid, everything else preserved."""
    if isinstance(source_metadata, ArrayV3Metadata):
        sharding_codec = _sharding_codec(source_metadata)
        # Unwrap sharding: the destination is a plain regular-chunked array.
        # The codecs below the sharding codec describe the per-chunk encoding.
        base_codecs = (
            tuple(sharding_codec.codecs)
            if sharding_codec is not None
            else tuple(source_metadata.codecs)
        )
        if compressors == "auto":
            codecs = base_codecs
        else:
            kept = tuple(c for c in base_codecs if not isinstance(c, BytesBytesCodec))
            _, _, new_compressors = _parse_chunk_encoding_v3(
                compressors=compressors,
                filters="auto",
                serializer="auto",
                dtype=source_metadata.data_type,
            )
            codecs = (*kept, *new_compressors)
        return dataclasses.replace(
            source_metadata,
            chunk_grid=RegularChunkGridMetadata(chunk_shape=chunks),
            codecs=codecs,
        )
    if isinstance(source_metadata, ArrayV2Metadata):
        if compressors == "auto":
            compressor = source_metadata.compressor
        else:
            _, compressor = _parse_chunk_encoding_v2(
                compressor=compressors,
                filters=source_metadata.filters,
                dtype=source_metadata.dtype,
            )
        return dataclasses.replace(source_metadata, chunks=chunks, compressor=compressor)
    raise TypeError(f"Unsupported array metadata type: {type(source_metadata)!r}")


def _chunk_start_offsets(sizes_per_dim: tuple[tuple[int, ...], ...]) -> list[tuple[int, ...]]:
    """Prefix offsets of each chunk along every dimension (len == nchunks + 1)."""
    offsets: list[tuple[int, ...]] = []
    for sizes in sizes_per_dim:
        running = [0]
        for size in sizes:
            running.append(running[-1] + size)
        offsets.append(tuple(running))
    return offsets


async def rechunk_array(
    source: AsyncArray[Any] | Array,
    dest: StoreLike,
    *,
    chunks: tuple[int, ...],
    compressors: CompressorsLike = "auto",
    max_mem: int,
    path: str | None = None,
    overwrite: bool = False,
    storage_options: dict[str, Any] | None = None,
) -> AsyncArray[Any]:
    """
    Copy ``source`` to ``dest`` with a new chunk layout.

    The copy preserves the shape, dtype, fill value, and every value of the
    source array; only the chunk grid (and optionally the compressor) changes.

    Parameters
    ----------
    source : AsyncArray | Array
        The array to copy.  It may be sharded and may have unwritten
        (implicit fill-value) regions; both are handled by the copy plan.
    dest : StoreLike
        Where to write the new array (anything accepted by
        :func:`zarr.storage.make_store_path`).
    chunks : tuple[int, ...]
        The chunk shape of the new array.  It does not need to divide the
        array shape or relate to the source chunking in any way.
    compressors : CompressorsLike, optional
        Compressor(s) for the new array.  The default, ``"auto"``, keeps the
        source array's compressor(s).  Pass e.g. ``None`` (or ``()``) to store
        chunks uncompressed.
    max_mem : int
        Decoded working-memory budget in bytes.  This is *not* a task count:
        at most one decoded source read-chunk plus one decoded target chunk
        are held at any time, and the copy raises :class:`ValueError` before
        writing anything if that minimal unit does not fit in ``max_mem``.
    path : str, optional
        Path of the new array within ``dest``.
    overwrite : bool, optional
        If True, delete any pre-existing node at the destination first.
    storage_options : dict, optional
        Passed to :func:`zarr.storage.make_store_path` for ``dest``.

    Returns
    -------
    AsyncArray
        The completed new array.

    Notes
    -----
    The destination's array metadata is written only after every chunk has
    been copied.  If the copy fails partway through, the destination holds
    orphaned chunk keys but *no* array metadata, so it cannot be opened as an
    array -- that is how an incomplete copy is recognized.  Cleaning up such a
    destination (or deleting the old array) is left to the caller; retry with
    ``overwrite=True`` to replace an incomplete destination.
    """
    if isinstance(source, Array):
        source = source.async_array
    chunks_parsed = _parse_chunks(chunks, source.ndim)

    # Plan the memory budget before touching the destination.  Both the source
    # chunk being decoded and the target chunk being assembled count against
    # the budget; the minimal necessary unit is one of each.
    required = minimum_rechunk_memory(source, chunks_parsed)
    if required > max_mem:
        raise ValueError(
            f"max_mem={max_mem} bytes is too small to rechunk this array: the "
            f"minimal necessary unit is {required} bytes (one decoded source "
            f"chunk plus one decoded target chunk)."
        )

    store_path = await make_store_path(dest, path=path, mode="w", storage_options=storage_options)
    await _prepare_overwrite(
        store_path, zarr_format=source.metadata.zarr_format, overwrite=overwrite
    )

    metadata = _target_metadata(source.metadata, chunks_parsed, compressors)
    # Pure-fill chunks must not be materialized; unwritten source regions must
    # read back as the fill value.  Pin both behaviors so the copy does not
    # depend on the ambient configuration.
    target_config = ArrayConfig.from_dict(
        {"write_empty_chunks": False, "read_missing_chunks": True}
    )
    target: AsyncArray[Any] = AsyncArray(
        metadata=metadata, store_path=store_path, config=target_config
    )
    reader = source.with_config({"read_missing_chunks": True})

    prototype = default_buffer_prototype()
    shape = source.shape
    dtype = source.dtype
    ndim = source.ndim
    fill_value = metadata.fill_value
    if fill_value is None:
        # Zarr format 2 permits a null fill value; reads substitute the
        # dtype's default scalar, so the staging buffer must match.
        fill_value = metadata.dtype.default_scalar()

    read_sizes = reader.read_chunk_sizes
    read_offsets = _chunk_start_offsets(read_sizes)

    # Walk the target grid one chunk at a time.  For each target chunk, read
    # only its intersections with the overlapping source read-chunks directly
    # into a view of the staging buffer -- no whole-axis assembly, and at most
    # one source chunk plus the staging buffer alive at any moment.
    grid_dims = tuple(math.ceil(shape[d] / chunks_parsed[d]) for d in range(ndim))
    for target_coords in iter_product(*(range(n) for n in grid_dims)):
        region_start = tuple(target_coords[d] * chunks_parsed[d] for d in range(ndim))
        region_stop = tuple(min(region_start[d] + chunks_parsed[d], shape[d]) for d in range(ndim))
        region_shape = tuple(region_stop[d] - region_start[d] for d in range(ndim))

        staging = prototype.nd_buffer.create(
            shape=region_shape, dtype=dtype, order="C", fill_value=fill_value
        )

        per_dim_indices = []
        for d in range(ndim):
            offsets = read_offsets[d]
            first = bisect_right(offsets, region_start[d]) - 1
            last = bisect_right(offsets, region_stop[d] - 1) - 1
            per_dim_indices.append(range(first, last + 1))

        for source_indices in iter_product(*per_dim_indices):
            box = []
            offset = []
            for d in range(ndim):
                i = source_indices[d]
                lo = max(region_start[d], read_offsets[d][i])
                hi = min(region_stop[d], read_offsets[d][i + 1])
                box.append(slice(lo, hi))
                offset.append(slice(lo - region_start[d], hi - region_start[d]))
            indexer = BasicIndexer(tuple(box), shape=shape, chunk_grid=reader._chunk_grid)
            await reader._get_selection(indexer, prototype=prototype, out=staging[tuple(offset)])

        if staging.all_equal(fill_value):
            # Pure fill-value chunk: leave it unwritten so it stays implicit.
            continue

        region = tuple(slice(region_start[d], region_stop[d]) for d in range(ndim))
        await target._set_selection(
            BasicIndexer(region, shape=shape, chunk_grid=target._chunk_grid),
            staging.as_numpy_array(),
            prototype=prototype,
        )

    # Metadata goes last: its presence is what marks the copy as complete.
    await target._save_metadata(metadata, ensure_parents=True)
    return target
