from __future__ import annotations

import gc
import math

import numpy as np
import pytest

import zarr
import zarr.api.asynchronous
from zarr.codecs import GzipCodec, ZstdCodec
from zarr.core.buffer import cpu
from zarr.core.rechunk import RECHUNK_COMPLETE_ATTR, rechunk_is_complete
from zarr.registry import fully_qualified_name, register_ndbuffer
from zarr.storage import MemoryStore


@pytest.fixture
def source_2d() -> zarr.Array:
    data = np.arange(100 * 100, dtype="i4").reshape(100, 100)
    source = zarr.create_array({}, shape=data.shape, chunks=(17, 11), dtype="i4", fill_value=0)
    source[:] = data
    return source


def test_basic_2d(source_2d: zarr.Array) -> None:
    target = zarr.rechunk(source_2d, (8, 23), max_mem=1 << 16)
    assert target.chunks == (8, 23)
    assert target.shape == source_2d.shape
    assert target.dtype == source_2d.dtype
    assert target.fill_value == source_2d.fill_value
    np.testing.assert_array_equal(target[:], source_2d[:])
    assert rechunk_is_complete(target)
    assert target.attrs[RECHUNK_COMPLETE_ATTR] is True


@pytest.mark.parametrize("new_chunks", [(8, 23, 5), (3, 7, 25), (31, 2, 4), (5, 13, 7)])
def test_3d_nondivisible_chunks(new_chunks: tuple[int, int, int]) -> None:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 100, size=(31, 47, 25), dtype="i2")
    source = zarr.create_array({}, shape=data.shape, chunks=(17, 11, 9), dtype="i2", fill_value=0)
    source[:] = data
    target = zarr.rechunk(source, new_chunks, max_mem=1 << 16)
    assert target.chunks == new_chunks
    np.testing.assert_array_equal(target[:], data)
    assert rechunk_is_complete(target)


def test_max_mem_too_small_raises_before_target_created(source_2d: zarr.Array) -> None:
    store = MemoryStore()
    # one source chunk is 17*11*4 = 748 bytes, one target chunk is 8*23*4 = 736 bytes
    with pytest.raises(ValueError, match="max_mem is too small"):
        zarr.rechunk(source_2d, (8, 23), max_mem=100, store=store)
    # the failure happened during planning, so no target array exists
    assert zarr.core.sync.sync(_store_keys(store)) == []


async def _store_keys(store: MemoryStore) -> list[str]:
    return [key async for key in store.list()]


def test_max_mem_exact_minimum(source_2d: zarr.Array) -> None:
    # source chunk 748 bytes + target chunk 736 bytes
    target = zarr.rechunk(source_2d, (8, 23), max_mem=748 + 736)
    np.testing.assert_array_equal(target[:], source_2d[:])
    with pytest.raises(ValueError, match="max_mem is too small"):
        zarr.rechunk(source_2d, (8, 23), max_mem=748 + 736 - 1)


@pytest.mark.parametrize("bad_max_mem", [0, -1, 2.5, "1000"])
def test_max_mem_invalid(source_2d: zarr.Array, bad_max_mem: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        zarr.rechunk(source_2d, (8, 23), max_mem=bad_max_mem)  # type: ignore[arg-type]


def test_invalid_grid_specs(source_2d: zarr.Array) -> None:
    with pytest.raises(ValueError, match="one entry per array dimension"):
        zarr.rechunk(source_2d, (8, 23, 5), max_mem=1 << 16)
    with pytest.raises(ValueError, match="positive integers"):
        zarr.rechunk(source_2d, (8, 0), max_mem=1 << 16)
    with pytest.raises(ValueError, match="multiple of chunks"):
        zarr.rechunk(source_2d, (8, 23), shards=(9, 23), max_mem=1 << 16)


def test_sharded_source_and_target() -> None:
    rng = np.random.default_rng(2)
    data = rng.integers(0, 100, size=(40, 30), dtype="i4")
    source = zarr.create_array(
        {}, shape=data.shape, chunks=(5, 6), shards=(10, 12), dtype="i4", fill_value=0
    )
    source[:] = data

    target = zarr.rechunk(source, (8, 23), max_mem=1 << 12)
    assert target.chunks == (8, 23)
    assert target.shards is None
    np.testing.assert_array_equal(target[:], data)

    sharded_target = zarr.rechunk(source, (4, 5), shards=(8, 10), max_mem=1 << 12)
    assert sharded_target.chunks == (4, 5)
    assert sharded_target.shards == (8, 10)
    np.testing.assert_array_equal(sharded_target[:], data)


def test_fill_regions_not_materialized() -> None:
    source = zarr.create_array({}, shape=(20, 20), chunks=(5, 5), dtype="i4", fill_value=7)
    source[0:5, 0:5] = 1
    target = zarr.rechunk(source, (3, 4), max_mem=1 << 12)
    np.testing.assert_array_equal(target[:], source[:])
    assert target.fill_value == 7
    # only the 2x2 target chunks overlapping the written corner may exist
    assert target.nchunks_initialized == 4


def test_all_fill_source_writes_no_chunks() -> None:
    source = zarr.create_array({}, shape=(10, 10), chunks=(4, 4), dtype="i4", fill_value=3)
    target = zarr.rechunk(source, (3, 3), max_mem=1 << 12)
    assert target.nchunks_initialized == 0
    np.testing.assert_array_equal(target[:], np.full((10, 10), 3, dtype="i4"))
    assert rechunk_is_complete(target)


def test_compressor_change_and_keep(source_2d: zarr.Array) -> None:
    changed = zarr.rechunk(source_2d, (8, 23), max_mem=1 << 16, compressors=[GzipCodec(level=1)])
    assert isinstance(changed.compressors[0], GzipCodec)
    np.testing.assert_array_equal(changed[:], source_2d[:])

    source = zarr.create_array(
        {},
        shape=(50, 50),
        chunks=(17, 11),
        dtype="i4",
        fill_value=0,
        compressors=[ZstdCodec(level=3)],
    )
    source[:] = np.arange(2500, dtype="i4").reshape(50, 50)
    kept = zarr.rechunk(source, (8, 23), max_mem=1 << 16)
    assert isinstance(kept.compressors[0], ZstdCodec)
    np.testing.assert_array_equal(kept[:], source[:])


def test_v2_source_stays_v2() -> None:
    data = np.arange(2500, dtype="i4").reshape(50, 50)
    source = zarr.create_array(
        {}, shape=data.shape, chunks=(17, 11), dtype="i4", fill_value=0, zarr_format=2
    )
    source[:] = data
    target = zarr.rechunk(source, (8, 23), max_mem=1 << 16)
    assert target.metadata.zarr_format == 2
    assert target.chunks == (8, 23)
    np.testing.assert_array_equal(target[:], data)


def test_failed_copy_leaves_recognizable_target(source_2d: zarr.Array) -> None:
    class FailOnChunkStore(MemoryStore):
        async def set(self, key: str, value: object) -> None:
            if key != "zarr.json":
                raise RuntimeError("simulated store failure")
            await super().set(key, value)  # type: ignore[arg-type]

    store = FailOnChunkStore()
    with pytest.raises(RuntimeError, match="simulated store failure"):
        zarr.rechunk(source_2d, (8, 23), max_mem=1 << 16, store=store)
    broken = zarr.open_array(store=store)
    assert not rechunk_is_complete(broken)
    assert broken.attrs[RECHUNK_COMPLETE_ATTR] is False


def test_source_is_not_modified(source_2d: zarr.Array) -> None:
    before = source_2d[:]
    nchunks_before = source_2d.nchunks_initialized
    zarr.rechunk(source_2d, (8, 23), max_mem=1 << 16)
    np.testing.assert_array_equal(source_2d[:], before)
    assert source_2d.nchunks_initialized == nchunks_before
    assert RECHUNK_COMPLETE_ATTR not in dict(source_2d.attrs)


def test_regular_array_reports_incomplete(source_2d: zarr.Array) -> None:
    assert not rechunk_is_complete(source_2d)


def test_zero_size_axis() -> None:
    source = zarr.create_array({}, shape=(0, 10), chunks=(4, 4), dtype="i4", fill_value=0)
    target = zarr.rechunk(source, (3, 3), max_mem=1 << 12)
    assert target.shape == (0, 10)
    assert target.chunks == (3, 3)
    assert rechunk_is_complete(target)


async def test_async_api(source_2d: zarr.Array) -> None:
    target = await zarr.api.asynchronous.rechunk(source_2d._async_array, (8, 23), max_mem=1 << 16)
    assert target.chunks == (8, 23)
    np.testing.assert_array_equal(
        await target.getitem(...), await source_2d._async_array.getitem(...)
    )


class _AllocRecordingNDBuffer(cpu.NDBuffer):
    """An NDBuffer that tracks the peak size of simultaneously live allocations."""

    live = 0
    peak = 0

    def _track(self) -> None:
        size = int(self.as_numpy_array().nbytes)
        self._recorded_size = size
        cls = type(self)
        cls.live += size
        cls.peak = max(cls.peak, cls.live)

    @classmethod
    def empty(cls, shape, dtype, order="C") -> _AllocRecordingNDBuffer:  # type: ignore[no-untyped-def]
        self = super().empty(shape, dtype, order)
        self._track()
        return self

    @classmethod
    def create(cls, *, shape, dtype, order="C", fill_value=None) -> _AllocRecordingNDBuffer:  # type: ignore[no-untyped-def]
        self = super().create(shape=shape, dtype=dtype, order=order, fill_value=fill_value)
        self._track()
        return self

    def __del__(self) -> None:
        type(self).live -= getattr(self, "_recorded_size", 0)


register_ndbuffer(_AllocRecordingNDBuffer)


@pytest.mark.parametrize(
    ("new_chunks", "budget"),
    [((8, 23, 5), 1 << 16), ((3, 7, 25), 1 << 15), ((31, 2, 4), 1 << 14)],
)
def test_peak_memory_within_budget(new_chunks: tuple[int, int, int], budget: int) -> None:
    rng = np.random.default_rng(1)
    data = rng.integers(0, 100, size=(31, 47, 25), dtype="i2")
    source = zarr.create_array({}, shape=data.shape, chunks=(17, 11, 9), dtype="i2", fill_value=0)
    source[:] = data

    with zarr.config.set({"ndbuffer": fully_qualified_name(_AllocRecordingNDBuffer)}):
        _AllocRecordingNDBuffer.live = 0
        _AllocRecordingNDBuffer.peak = 0
        gc.collect()
        target = zarr.rechunk(source, new_chunks, max_mem=budget)
        peak = _AllocRecordingNDBuffer.peak

    assert peak <= budget
    # the copy must never assemble a whole axis (let alone the array) in memory
    # unless the budget allows it: here a single axis slab is 47*25*2 = 2350..
    # 31*47*2 = 2914 bytes at minimum, but the full array is 72850 bytes.
    assert peak < math.prod(data.shape) * data.dtype.itemsize
    np.testing.assert_array_equal(target[:], data)
