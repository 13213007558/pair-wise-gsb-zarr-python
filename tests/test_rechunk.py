"""Tests for ``zarr.rechunk_array``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.codecs import GzipCodec
from zarr.core.buffer import cpu
from zarr.core.rechunk import minimum_rechunk_memory
from zarr.core.sync import sync as _sync
from zarr.errors import ArrayNotFoundError
from zarr.registry import fully_qualified_name, register_ndbuffer
from zarr.storage import MemoryStore


class RecordingNDBuffer(cpu.NDBuffer):
    """An NDBuffer that records live and peak allocated bytes."""

    current_bytes: int = 0
    peak_bytes: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.current_bytes = 0
        cls.peak_bytes = 0

    @classmethod
    def _track(cls, buf: RecordingNDBuffer) -> RecordingNDBuffer:
        nbytes = int(buf.as_numpy_array().nbytes)
        cls.current_bytes += nbytes
        cls.peak_bytes = max(cls.peak_bytes, cls.current_bytes)
        buf._recording_nbytes = nbytes  # type: ignore[attr-defined]
        return buf

    @classmethod
    def create(cls, *, shape: Any, dtype: Any, order: Any = "C", fill_value: Any = None) -> Any:
        buf = super().create(shape=shape, dtype=dtype, order=order, fill_value=fill_value)
        return cls._track(buf)

    @classmethod
    def empty(cls, shape: Any, dtype: Any, order: Any = "C") -> Any:
        return cls._track(super().empty(shape=shape, dtype=dtype, order=order))

    def __del__(self) -> None:
        nbytes = getattr(self, "_recording_nbytes", 0)
        if nbytes:
            type(self).current_bytes -= nbytes


def _data(shape: tuple[int, ...], dtype: str, seed: int = 0) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return rng.integers(0, 100, size=shape).astype(dtype)
    return rng.random(shape).astype(dtype)


@pytest.mark.parametrize(
    ("shape", "src_chunks", "dst_chunks"),
    [
        ((40, 35, 27), (17, 11, 9), (8, 23, 13)),
        ((53, 41, 30), (16, 16, 16), (7, 9, 11)),
        ((29, 31, 37), (5, 6, 7), (11, 9, 13)),
        ((24, 18, 12), (24, 18, 12), (5, 7, 5)),
        ((20, 20, 20), (3, 3, 3), (20, 20, 20)),
    ],
)
def test_roundtrip_nondivisible_3d(
    shape: tuple[int, ...], src_chunks: tuple[int, ...], dst_chunks: tuple[int, ...]
) -> None:
    """Several mutually non-divisible 3D chunk layouts, compared element-wise."""
    data = _data(shape, "f8")
    source = zarr.create_array(
        MemoryStore(), shape=shape, chunks=src_chunks, dtype="f8", fill_value=1.5
    )
    source[:] = data

    result = zarr.rechunk_array(source, MemoryStore(), chunks=dst_chunks, max_mem=10**7)

    assert result.chunks == dst_chunks
    assert result.shape == shape
    assert result.dtype == np.dtype("f8")
    assert result.fill_value == 1.5
    np.testing.assert_array_equal(result[:], data)
    # the source array is untouched
    np.testing.assert_array_equal(source[:], data)
    assert source.chunks == src_chunks


async def _collect_keys(store: MemoryStore) -> list[str]:
    return [k async for k in store.list()]


def test_max_mem_below_minimum_raises_before_writing() -> None:
    source = zarr.create_array(MemoryStore(), shape=(20, 20), chunks=(6, 6), dtype="f8")
    source[:] = _data((20, 20), "f8")
    dest = MemoryStore()
    required = minimum_rechunk_memory(source, (7, 7))
    assert required == (6 * 6 + 7 * 7) * 8

    with pytest.raises(ValueError, match="too small"):
        zarr.rechunk_array(source, dest, chunks=(7, 7), max_mem=required - 1)

    # nothing was written to the destination
    assert _sync(_collect_keys(dest)) == []


def test_max_mem_equal_to_minimum_succeeds() -> None:
    shape = (20, 20)
    data = _data(shape, "f8")
    source = zarr.create_array(MemoryStore(), shape=shape, chunks=(6, 6), dtype="f8")
    source[:] = data
    required = minimum_rechunk_memory(source, (7, 7))
    result = zarr.rechunk_array(source, MemoryStore(), chunks=(7, 7), max_mem=required)
    np.testing.assert_array_equal(result[:], data)


def test_peak_memory_stays_within_budget() -> None:
    """A recording buffer verifies the decoded working memory never exceeds max_mem.

    The array is much larger than max_mem along every axis, so any
    whole-axis (or larger) intermediate would blow the budget.
    """
    shape = (120, 90, 80)
    src_chunks = (17, 11, 13)
    dst_chunks = (8, 23, 9)
    data = _data(shape, "f8")
    source = zarr.create_array(MemoryStore(), shape=shape, chunks=src_chunks, dtype="f8")
    source[:] = data

    max_mem = minimum_rechunk_memory(source, dst_chunks)
    register_ndbuffer(RecordingNDBuffer)
    RecordingNDBuffer.reset()
    with zarr.config.set({"ndbuffer": fully_qualified_name(RecordingNDBuffer)}):
        result = zarr.rechunk_array(source, MemoryStore(), chunks=dst_chunks, max_mem=max_mem)

    assert RecordingNDBuffer.peak_bytes > 0  # the recording buffer was exercised
    assert RecordingNDBuffer.peak_bytes <= max_mem
    np.testing.assert_array_equal(result[:], data)


def test_fill_regions_are_not_materialized() -> None:
    """Unwritten source regions participate in planning but pure-fill target
    chunks must not be written as real chunks."""
    shape = (30, 30, 30)
    fill_value = 5
    source = zarr.create_array(
        MemoryStore(), shape=shape, chunks=(10, 10, 10), dtype="i4", fill_value=fill_value
    )
    sub = np.arange(12 * 13 * 14, dtype="i4").reshape(12, 13, 14)
    source[2:14, 3:16, 5:19] = sub

    result = zarr.rechunk_array(source, MemoryStore(), chunks=(7, 11, 9), max_mem=10**6)

    expected = np.full(shape, fill_value, dtype="i4")
    expected[2:14, 3:16, 5:19] = sub
    np.testing.assert_array_equal(result[:], expected)
    # only the 2 x 2 x 3 target chunks overlapping the written region may exist
    assert result.nchunks_initialized == 12
    assert result.nchunks_initialized < result.nchunks


def test_sharded_source() -> None:
    """Shards participate in planning at inner-chunk granularity."""
    shape = (40, 36)
    data = _data(shape, "f4")
    source = zarr.create_array(
        MemoryStore(), shape=shape, chunks=(7, 5), shards=(21, 20), dtype="f4"
    )
    source[:] = data

    result = zarr.rechunk_array(source, MemoryStore(), chunks=(9, 9), max_mem=10**6)

    assert result.shards is None
    assert result.chunks == (9, 9)
    np.testing.assert_array_equal(result[:], data)


def test_compressor_override() -> None:
    data = _data((20, 20), "f8")
    source = zarr.create_array(MemoryStore(), shape=(20, 20), chunks=(6, 6), dtype="f8")
    source[:] = data

    kept = zarr.rechunk_array(source, MemoryStore(), chunks=(7, 7), max_mem=10**6)
    assert kept.compressors == source.compressors

    changed = zarr.rechunk_array(
        source, MemoryStore(), chunks=(7, 7), compressors=[GzipCodec(level=1)], max_mem=10**6
    )
    assert changed.compressors == (GzipCodec(level=1),)
    np.testing.assert_array_equal(changed[:], data)

    uncompressed = zarr.rechunk_array(
        source, MemoryStore(), chunks=(7, 7), compressors=None, max_mem=10**6
    )
    assert uncompressed.compressors == ()
    np.testing.assert_array_equal(uncompressed[:], data)


def test_zarr_format_2_source() -> None:
    shape = (25, 22)
    data = _data(shape, "i8")
    source = zarr.create_array(
        MemoryStore(),
        shape=shape,
        chunks=(8, 7),
        dtype="i8",
        zarr_format=2,
        fill_value=-1,
    )
    source[:] = data

    result = zarr.rechunk_array(source, MemoryStore(), chunks=(6, 11), max_mem=10**6)

    assert result.metadata.zarr_format == 2
    assert result.chunks == (6, 11)
    assert result.fill_value == -1
    np.testing.assert_array_equal(result[:], data)


class _FaultyStore(MemoryStore):
    """A store that starts raising after a fixed number of ``set`` calls."""

    def __init__(self, fail_after: int = 0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_after = fail_after
        self._sets = 0

    async def set(self, key: str, value: Any) -> None:
        self._sets += 1
        if self._sets > self.fail_after:
            raise RuntimeError("simulated store failure")
        await super().set(key, value)


def test_failed_copy_leaves_recognizably_incomplete_target() -> None:
    data = _data((20, 20), "i4")
    source = zarr.create_array(MemoryStore(), shape=(20, 20), chunks=(5, 5), dtype="i4")
    source[:] = data

    dest = _FaultyStore(fail_after=3)
    with pytest.raises(RuntimeError, match="simulated store failure"):
        zarr.rechunk_array(source, dest, chunks=(7, 7), max_mem=10**6)

    # the incomplete copy has no array metadata, so it cannot be opened
    with pytest.raises(ArrayNotFoundError):
        zarr.open_array(dest, mode="r")

    # retrying with overwrite=True completes the copy
    dest.fail_after = 10**9
    result = zarr.rechunk_array(source, dest, chunks=(7, 7), max_mem=10**6, overwrite=True)
    np.testing.assert_array_equal(result[:], data)


def test_chunks_validation() -> None:
    source = zarr.create_array(MemoryStore(), shape=(10, 10), chunks=(3, 3), dtype="f8")
    with pytest.raises(ValueError, match="one entry per array dimension"):
        zarr.rechunk_array(source, MemoryStore(), chunks=(4, 4, 4), max_mem=10**6)
    with pytest.raises(ValueError, match="positive integer"):
        zarr.rechunk_array(source, MemoryStore(), chunks=(4, 0), max_mem=10**6)
