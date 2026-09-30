from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from zarr.codecs.bytes import BytesCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.chunk_utils import chunk_is_empty, fill_value_or_default
from zarr.core.codec_pipeline import AsyncChunkTransform
from zarr.core.config import config
from zarr.core.metadata.v3 import create_chunk_grid_metadata
from zarr.storage._common import StorePath, make_store_path

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.array import AnyAsyncArray
    from zarr.core.metadata import ArrayMetadata
    from zarr.storage import StoreLike


@dataclass(frozen=True, slots=True)
class RechunkResult:
    """Statistics from a budget-controlled rechunk copy."""

    chunks_completed: int
    chunks_written: int
    chunks_empty: int
    peak_bytes: int


class _PeakMeter:
    __slots__ = ("current", "peak")

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def add(self, size: int) -> None:
        self.current += size
        self.peak = max(self.peak, self.current)

    def remove(self, size: int) -> None:
        self.current -= size


class _TrackedBuffer:
    __slots__ = ("buffer", "meter", "size")

    def __init__(self, buffer: Buffer, meter: _PeakMeter, budget: int) -> None:
        self.buffer = buffer
        self.size = len(buffer)
        self.meter = meter
        _require_encoded_budget(meter, self.size, budget)
        meter.add(self.size)

    def __enter__(self) -> Buffer:
        return self.buffer

    def __exit__(self, *_args: object) -> None:
        self.meter.remove(self.size)


def _require_encoded_budget(meter: _PeakMeter, encoded_size: int, budget: int) -> None:
    if meter.current + encoded_size > budget:
        raise MemoryError(
            f"a {encoded_size}-byte encoded chunk exceeds the remaining working-memory "
            f"budget of {budget - meter.current} bytes (total budget {budget} bytes)."
        )


def _itemsize(metadata: ArrayMetadata) -> int:
    itemsize = metadata.dtype.to_native_dtype().itemsize
    if itemsize <= 0:
        raise ValueError(
            "rechunk requires a fixed-size dtype; variable-length dtypes have no "
            "statically bounded working memory."
        )
    return itemsize


def _chunk_spec(array: AnyAsyncArray, shape: tuple[int, ...], config: ArrayConfig) -> ArraySpec:
    return ArraySpec(
        shape=shape,
        dtype=array.metadata.dtype,
        fill_value=array.metadata.fill_value,
        config=config,
        prototype=default_buffer_prototype(),
    )


def _async_transform(array: AnyAsyncArray) -> AsyncChunkTransform:
    return AsyncChunkTransform(codecs=tuple(array.codec_pipeline))


def _chunk_key(array: AnyAsyncArray, coords: tuple[int, ...]) -> str:
    key = array.metadata.encode_chunk_key(coords)
    return f"{array.store_path.path}/{key}".strip("/")


def _codec_work_bytes(array: AnyAsyncArray, itemsize: int) -> int:
    """Conservative raw working space for one pass through the array codecs."""

    largest = max(
        (math.prod(spec.codec_shape) for spec in array._chunk_grid),
        default=0,
    ) * itemsize
    # This is the decoded array seen by the array-to-bytes codec. The byte
    # serializer itself is counted by the caller through the encoded-size sum.
    spec = _chunk_spec(
        array,
        max(array._chunk_grid, key=lambda item: math.prod(item.codec_shape)).codec_shape,
        array.config,
    )
    for codec in tuple(array.codec_pipeline):
        spec = codec.resolve_metadata(spec)
        if spec.dtype.to_native_dtype().itemsize > 0:
            largest = max(
                largest,
                math.prod(spec.shape) * spec.dtype.to_native_dtype().itemsize,
            )
    return largest


def _validate(source: AnyAsyncArray, target: AnyAsyncArray) -> None:
    if source.store_path == target.store_path:
        raise ValueError("source and target must be different arrays")
    if source.metadata.shape != target.metadata.shape:
        raise ValueError(
            f"rechunk requires identical shapes: source {source.metadata.shape!r}, "
            f"target {target.metadata.shape!r}"
        )
    if source.metadata.dtype.to_native_dtype() != target.metadata.dtype.to_native_dtype():
        raise ValueError(
            f"rechunk requires identical dtypes: source {source.metadata.dtype!r}, "
            f"target {target.metadata.dtype!r}"
        )
    _itemsize(source.metadata)

    native_dtype = source.metadata.dtype.to_native_dtype()
    source_fill = np.asarray(
        fill_value_or_default(_chunk_spec(source, (), source.config)), dtype=native_dtype
    ).view("V")
    target_fill = np.asarray(
        fill_value_or_default(_chunk_spec(target, (), target.config)), dtype=native_dtype
    ).view("V")
    if not np.array_equal(source_fill, target_fill):
        raise ValueError(
            "rechunk requires identical fill values: "
            f"source {source.metadata.fill_value!r}, target {target.metadata.fill_value!r}"
        )

    source_last_coords = tuple(value - 1 for value in source._chunk_grid.grid_shape)
    target_last_coords = tuple(value - 1 for value in target._chunk_grid.grid_shape)
    source_last = source._chunk_grid[source_last_coords]
    target_last = target._chunk_grid[target_last_coords]
    if source_last is None or target_last is None:
        raise IndexError("Cannot inspect terminal chunks")
    source_dimensions = source._chunk_grid.dimensions
    target_dimensions = target._chunk_grid.dimensions
    for axis, (source_index, target_index) in enumerate(
        zip(source_last_coords, target_last_coords, strict=True)
    ):
        source_is_partial = (
            source_dimensions[axis].chunk_size(source_index)
            != source_dimensions[axis].data_size(source_index)
        )
        target_is_partial = (
            target_dimensions[axis].chunk_size(target_index)
            != target_dimensions[axis].data_size(target_index)
        )
        if source_is_partial or target_is_partial:
            if source_last.shape[axis] != target_last.shape[axis]:
                raise ValueError(
                    "rechunk requires identical non-full edge chunks: "
                    f"axis {axis} source edge is {source_last.shape[axis]} but target "
                    f"edge is {target_last.shape[axis]}"
                )
    if getattr(source.metadata, "sharding_codec", None) is not None or getattr(
        target.metadata, "sharding_codec", None
    ) is not None:
        raise NotImplementedError("budget-controlled rechunk does not support sharded arrays yet.")
    if not target.store.supports_writes or not target.store.supports_deletes:
        raise ValueError("The target store must support writes and deletes.")


def _region_coords(grid: ChunkGrid, region: tuple[slice, ...]):
    ranges = []
    for axis, bounds in enumerate(region):
        dimension = grid.dimensions[axis]
        ranges.append(
            range(
                dimension.index_to_chunk(bounds.start),
                dimension.index_to_chunk(bounds.stop - 1) + 1,
            )
        )
    return itertools.product(*ranges)


def _intersection(left: tuple[slice, ...], right: tuple[slice, ...]):
    out = []
    for a, b in zip(left, right, strict=True):
        start = max(a.start, b.start)
        stop = min(a.stop, b.stop)
        if start >= stop:
            return None
        out.append(slice(start, stop, 1))
    return tuple(out)


def _relative(region: tuple[slice, ...], origin: tuple[slice, ...]) -> tuple[slice, ...]:
    return tuple(
        slice(bounds.start - base.start, bounds.stop - base.start, 1)
        for bounds, base in zip(region, origin, strict=True)
    )


def _all_edges(grid: ChunkGrid, axis: int) -> tuple[int, ...]:
    dimension = grid.dimensions[axis]
    return tuple(
        dimension.chunk_offset(index) + dimension.data_size(index)
        for index in range(dimension.nchunks)
    )


def _union_edges(source: AnyAsyncArray, target: AnyAsyncArray) -> tuple[tuple[int, ...], ...]:
    axes = []
    for axis in range(source.ndim):
        edges = {0, source.shape[axis]}
        edges.update(_all_edges(source._chunk_grid, axis))
        edges.update(_all_edges(target._chunk_grid, axis))
        axes.append(tuple(sorted(edges)))
    return tuple(axes)


async def _list_source_keys(source: AnyAsyncArray) -> set[tuple[int, ...]]:
    prefix = "" if source.store_path.path == "" else f"{source.store_path.path}/"
    keys: set[tuple[int, ...]] = set()
    async for key in source.store.list_prefix(prefix):
        relative = key.removeprefix(prefix)
        try:
            coords = source.metadata.chunk_key_encoding.decode_chunk_key(relative)
            if len(coords) != source.ndim or source._chunk_grid[coords] is None:
                raise ValueError(relative)
        except (ValueError, NotImplementedError):
            continue
        else:
            keys.add(coords)
    return keys


async def _delete_target_chunks(target: AnyAsyncArray) -> None:
    prefix = "" if target.store_path.path == "" else f"{target.store_path.path}/"
    async for key in target.store.list_prefix(prefix):
        relative = key.removeprefix(prefix)
        try:
            coords = target.metadata.chunk_key_encoding.decode_chunk_key(relative)
        except (ValueError, NotImplementedError, IndexError):
            continue
        if len(coords) == target.ndim and target._chunk_grid[coords] is not None:
            await target.store.delete(key)


async def _decode_chunk(
    array: AnyAsyncArray,
    coords: tuple[int, ...],
    config: ArrayConfig,
    meter: _PeakMeter,
    budget: int,
) -> NDBuffer:
    spec = array._chunk_grid[coords]
    if spec is None:
        raise IndexError(f"Invalid chunk coordinates {coords}")
    encoded = await array.store.get(_chunk_key(array, coords), default_buffer_prototype())
    if encoded is None:
        raise KeyError(coords)
    with _TrackedBuffer(encoded, meter, budget):
        decoded = await _async_transform(array).decode_chunk(
            encoded, _chunk_spec(array, spec.codec_shape, config)
        )
        meter.add(_nd_size(decoded))
        return decoded


async def _persist_target(
    target: AnyAsyncArray,
    coords: tuple[int, ...],
    chunk: NDBuffer,
    config: ArrayConfig,
    meter: _PeakMeter,
    budget: int,
) -> bool:
    spec = target._chunk_grid[coords]
    if spec is None:
        raise IndexError(f"Invalid target chunk coordinates {coords}")
    chunk_spec = _chunk_spec(target, spec.codec_shape, config)
    if chunk.shape != chunk_spec.shape or chunk_is_empty(chunk, chunk_spec):
        await target.store.delete(_chunk_key(target, coords))
        return False
    encoded = await _async_transform(target).encode_chunk(chunk, chunk_spec)
    if encoded is None:
        await target.store.delete(_chunk_key(target, coords))
        return False
    with _TrackedBuffer(encoded, meter, budget):
        await target.store.set(_chunk_key(target, coords), encoded)
    return True


def _nd_size(buffer: NDBuffer) -> int:
    return buffer.dtype.itemsize * math.prod(buffer.shape)


def _max_intersecting_source_chunks(source: AnyAsyncArray, target: AnyAsyncArray) -> int:
    result = 1
    for axis in range(source.ndim - 1, -1, -1):
        source_dimension = source._chunk_grid.dimensions[axis]
        target_dimension = target._chunk_grid.dimensions[axis]
        axis_count = 0
        for index in range(target_dimension.nchunks):
            region = slice(
                target_dimension.chunk_offset(index),
                target_dimension.chunk_offset(index) + target_dimension.data_size(index),
            )
            first = source_dimension.index_to_chunk(region.start)
            last = source_dimension.index_to_chunk(region.stop - 1)
            axis_count = max(axis_count, last - first + 1)
        result *= axis_count
    return result

def _write_config(array: AnyAsyncArray) -> ArrayConfig:
    return replace(array.config, write_empty_chunks=False)


async def _copy_direct(
    source: AnyAsyncArray,
    target: AnyAsyncArray,
    *,
    work_memory_budget: int,
) -> RechunkResult:
    meter = _PeakMeter()
    config = _write_config(target)
    source_config = _write_config(source)
    present = await _list_source_keys(source)
    active_sources: dict[tuple[int, ...], NDBuffer] = {}
    written = 0
    empty = 0

    try:
        for target_coords in target._chunk_grid.all_chunk_coords():
            target_spec_for_release = target._chunk_grid[target_coords]
            if target_spec_for_release is None:
                raise IndexError(f"Invalid target chunk coordinates {target_coords}")
            overlapping_sources = tuple(
                _region_coords(source._chunk_grid, target_spec_for_release.slices)
            )
            first_overlapping_source = overlapping_sources[0]
            for old_source_coords in list(active_sources):
                if old_source_coords < first_overlapping_source:
                    old_source = active_sources.pop(old_source_coords)
                    meter.remove(_nd_size(old_source))

            target_spec = target_spec_for_release
            if not any(coords in present for coords in overlapping_sources):
                await target.store.delete(_chunk_key(target, target_coords))
                empty += 1
                continue

            output: NDBuffer | None = None
            saw_non_fill = False
            try:
                for source_coords in overlapping_sources:
                    if source_coords not in present:
                        continue
                    source_spec = source._chunk_grid[source_coords]
                    if source_spec is None:
                        raise IndexError(f"Invalid source chunk coordinates {source_coords}")
                    decoded = active_sources.get(source_coords)
                    if decoded is None:
                        decoded = await _decode_chunk(
                            source,
                            source_coords,
                            source_config,
                            meter,
                            work_memory_budget,
                        )
                        active_sources[source_coords] = decoded

                    source_chunk_spec = _chunk_spec(
                        source, source_spec.codec_shape, source_config
                    )
                    if decoded.all_equal(fill_value_or_default(source_chunk_spec)):
                        continue
                    overlap = _intersection(source_spec.slices, target_spec.slices)
                    if overlap is None:
                        continue
                    saw_non_fill = True
                    if output is None:
                        output = default_buffer_prototype().nd_buffer.create(
                            shape=target_spec.codec_shape,
                            dtype=target.metadata.dtype.to_native_dtype(),
                            order=config.order,
                            fill_value=fill_value_or_default(
                                _chunk_spec(target, target_spec.codec_shape, config)
                            ),
                        )
                        meter.add(_nd_size(output))
                    output[_relative(overlap, target_spec.slices)] = decoded[
                        _relative(overlap, source_spec.slices)
                    ]
            except Exception:
                if output is not None:
                    meter.remove(_nd_size(output))
                raise

            if not saw_non_fill:
                await target.store.delete(_chunk_key(target, target_coords))
                empty += 1
                meter.remove(_nd_size(output))
                continue

            try:
                if output is None:
                    raise RuntimeError("unreachable")
                if await _persist_target(
                    target, target_coords, output, config, meter, work_memory_budget
                ):
                    written += 1
                else:
                    empty += 1
            finally:
                meter.remove(_nd_size(output))
    except Exception:
        for buffer in active_sources.values():
            meter.remove(_nd_size(buffer))
        raise

    for buffer in active_sources.values():
        meter.remove(_nd_size(buffer))
    return RechunkResult(written + empty, written, empty, meter.peak)


async def _copy_via_temp(
    source: AnyAsyncArray,
    target: AnyAsyncArray,
    *,
    temp_store: Store,
    work_memory_budget: int,
) -> RechunkResult:
    from zarr.core.array import AsyncArray

    edges = _union_edges(source, target)
    chunk_sizes = tuple(
        tuple(axis_edges[index + 1] - axis_edges[index] for index in range(len(axis_edges) - 1))
        for axis_edges in edges
    )
    temp_path_name = f".zarr-rechunk-{uuid.uuid4().hex}"
    temp_path = StorePath(temp_store, temp_path_name)
    temp_config = replace(target.config, write_empty_chunks=False)
    meter = _PeakMeter()
    source_config = _write_config(source)
    target_config = temp_config
    written = 0
    empty = 0
    temp_array: AnyAsyncArray | None = None

    try:
        await temp_store._ensure_open()
        await temp_path.delete_dir()
        chunk_grid = ChunkGrid.from_sizes(source.shape, chunk_sizes)
        with config.set({"array.rectilinear_chunks": True}):
            temp_metadata = AsyncArray._create_metadata_v3(
                shape=source.shape,
                dtype=source.metadata.dtype,
                chunk_grid=create_chunk_grid_metadata(chunk_grid),
                fill_value=target.metadata.fill_value,
                codecs=(BytesCodec(endian=None),),
                attributes={},
            )
        temp_array = AsyncArray(
            metadata=temp_metadata, store_path=temp_path, config=temp_config
        )
        await temp_array._save_metadata(temp_metadata, ensure_parents=False)
        present_keys = await _list_source_keys(source)

        for source_coords in source._chunk_grid.all_chunk_coords():
            if source_coords not in present_keys:
                continue
            source_spec = source._chunk_grid[source_coords]
            if source_spec is None:
                raise IndexError(f"Invalid source chunk coordinates {source_coords}")
            source_encoded = await source.store.get(
                _chunk_key(source, source_coords), default_buffer_prototype()
            )
            if source_encoded is None:
                raise KeyError(source_coords)
            with _TrackedBuffer(source_encoded, meter, work_memory_budget):
                decoded = await _async_transform(source).decode_chunk(
                    source_encoded,
                    _chunk_spec(source, source_spec.codec_shape, source_config),
                )
            meter.add(_nd_size(decoded))
            try:
                for cell_coords in _region_coords(chunk_grid, source_spec.slices):
                    cell_region = chunk_grid[cell_coords]
                    if cell_region is None:
                        continue
                    overlap = _intersection(source_spec.slices, cell_region.slices)
                    cell_spec = temp_array._chunk_grid[cell_coords]
                    if cell_spec is None:
                        raise IndexError(f"Invalid intermediate chunk coordinates {cell_coords}")
                    cell = decoded[_relative(overlap, source_spec.slices)]
                    cell_array_spec = _chunk_spec(temp_array, cell_spec.codec_shape, temp_config)
                    if chunk_is_empty(cell, cell_array_spec):
                        continue
                    encoded = await _async_transform(temp_array).encode_chunk(
                        cell, cell_array_spec
                    )
                    if encoded is None:
                        continue
                    with _TrackedBuffer(encoded, meter, work_memory_budget):
                        await temp_array.store.set(_chunk_key(temp_array, cell_coords), encoded)
            finally:
                meter.remove(_nd_size(decoded))

        temp_prefix = "" if temp_array.store_path.path == "" else f"{temp_array.store_path.path}/"
        cell_present = await _list_source_keys(temp_array)
        covered_targets = {
            target_coords
            for cell_coords in cell_present
            for target_coords in _region_coords(
                target._chunk_grid, temp_array._chunk_grid[cell_coords].slices
            )
        }

        all_targets = set(target._chunk_grid.all_chunk_coords())
        for target_coords in all_targets - covered_targets:
            await target.store.delete(_chunk_key(target, target_coords))
        empty += len(all_targets - covered_targets)

        for target_coords in target._chunk_grid.all_chunk_coords():
            if target_coords not in covered_targets:
                continue
            target_spec = target._chunk_grid[target_coords]
            if target_spec is None:
                raise IndexError(f"Invalid target chunk coordinates {target_coords}")
            cell_coords_for_target = tuple(_region_coords(chunk_grid, target_spec.slices))
            chunk = default_buffer_prototype().nd_buffer.create(
                shape=target_spec.codec_shape,
                dtype=target.metadata.dtype.to_native_dtype(),
                order=target_config.order,
                fill_value=fill_value_or_default(
                    _chunk_spec(target, target_spec.codec_shape, target_config)
                ),
            )
            meter.add(_nd_size(chunk))
            try:
                for cell_coords in cell_coords_for_target:
                    encoded_cell = await temp_store.get(
                        f"{temp_prefix}{temp_array.metadata.encode_chunk_key(cell_coords)}",
                        default_buffer_prototype(),
                    )
                    if encoded_cell is None:
                        continue
                    cell_region = chunk_grid[cell_coords]
                    if cell_region is None:
                        continue
                    cell_spec_shape = temp_array._chunk_grid[cell_coords]
                    if cell_spec_shape is None:
                        raise IndexError(f"Invalid intermediate chunk coordinates {cell_coords}")
                    with _TrackedBuffer(encoded_cell, meter, work_memory_budget):
                        decoded_cell = await _async_transform(temp_array).decode_chunk(
                            encoded_cell,
                            _chunk_spec(temp_array, cell_spec_shape.codec_shape, temp_config),
                        )
                        overlap = _intersection(cell_region.slices, target_spec.slices)
                        if overlap is not None:
                            chunk[_relative(overlap, target_spec.slices)] = decoded_cell
                        meter.add(_nd_size(decoded_cell))
                    meter.remove(_nd_size(decoded_cell))
                if await _persist_target(
                    target, target_coords, chunk, target_config, meter, work_memory_budget
                ):
                    written += 1
                else:
                    empty += 1
            finally:
                meter.remove(_nd_size(chunk))
    finally:
        if temp_store.supports_deletes and temp_store.supports_listing:
            await temp_path.delete_dir()

    return RechunkResult(written + empty, written, empty, meter.peak)


async def rechunk(
    source: AnyAsyncArray,
    target: AnyAsyncArray,
    *,
    work_memory_budget: int,
    temp_store: StoreLike | None = None,
) -> RechunkResult:
    """Copy an existing array into a compatible array with a different chunk grid.

    The operation streams complete target chunks in lexicographic order and
    caches only source chunks that can still intersect a pending target
    chunk. Each stored source chunk is therefore decoded at most once. When
    ``temp_store`` is provided, data is first written to an intermediate grid
    whose boundaries divide both grids, then read once into each final target
    chunk.

    The preflight budget includes the largest source and target raw chunks,
    the array-array codec stage with the largest decoded footprint, and the
    retained-source factor for the direct path. Bytes-to-bytes codecs with
    variable output size cannot be bounded from metadata alone; the source
    compressed bytes retained during decoding are included in the measured
    peak.

    Empty target chunks are normalized to missing keys; no complete
    all-fill target chunk is allocated. On failure, any temporary prefix
    created by this operation is removed. The target is not transactional:
    chunks completed before the failure remain visible.
    """

    if work_memory_budget < 0:
        raise ValueError("work_memory_budget must be non-negative")
    _validate(source, target)
    itemsize = _itemsize(source.metadata)
    source_work = _codec_work_bytes(source, itemsize)
    target_work = _codec_work_bytes(target, itemsize)
    largest_source = max(
        (math.prod(spec.codec_shape) for spec in source._chunk_grid), default=0
    ) * itemsize
    largest_target = max(
        (math.prod(spec.codec_shape) for spec in target._chunk_grid), default=0
    ) * itemsize
    if temp_store is None:
        max_active = _max_intersecting_source_chunks(source, target)
        direct_bytes = max(
            largest_source + source_work,
            largest_target + target_work,
            max_active * largest_source + target_work,
            max_active * largest_source + largest_target + target_work,
        )
        if direct_bytes > work_memory_budget:
            raise MemoryError(
                f"rechunk requires at least {direct_bytes} bytes of working memory, "
                f"but work_memory_budget is {work_memory_budget} bytes."
            )
        await _delete_target_chunks(target)
        return await _copy_direct(
            source,
            target,
            work_memory_budget=work_memory_budget,
        )

    store = await make_store_path(temp_store, mode="a")
    if not store.store.supports_writes or not store.store.supports_deletes:
        raise ValueError("temp_store must support writes, deletes, and listing.")
    edges = _union_edges(source, target)
    max_cell_bytes = (
        math.prod(
            max(
                axis_edges[index + 1] - axis_edges[index]
                for index in range(len(axis_edges) - 1)
            )
            for axis_edges in edges
        )
        * itemsize
    )
    temp_bytes = max(
        largest_source + source_work + max_cell_bytes,
        largest_target + target_work + max_cell_bytes,
        largest_target + 2 * max_cell_bytes,
    )
    if temp_bytes > work_memory_budget:
        raise MemoryError(
            f"rechunk with a temporary store requires at least {temp_bytes} bytes of working "
            f"memory, but work_memory_budget is {work_memory_budget} bytes."
        )
    await _delete_target_chunks(target)
    return await _copy_via_temp(
        source,
        target,
        temp_store=store.store,
        work_memory_budget=work_memory_budget,
    )
