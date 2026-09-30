from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import zarr
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs.bytes import BytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer
from zarr.core.common import JSON
from zarr.storage import MemoryStore

@dataclass(frozen=True)
class CountingCodec(BytesBytesCodec):
    is_fixed_size = True

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "test-rechunk-counting"}

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        CountingCodec.decode_count += 1
        CountingCodec.decoded_bytes += len(chunk_bytes)
        return chunk_bytes

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return self._decode_single(chunk_bytes, chunk_spec)

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_bytes

    def _encode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_bytes

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length


@dataclass(frozen=True)
class FailingCodec(BytesBytesCodec):
    is_fixed_size = True
    fail_after: int

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "test-rechunk-failing",
            "configuration": {"fail_after": self.fail_after},
        }

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_bytes

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_bytes

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        FailingCodec.encode_count += 1
        if FailingCodec.encode_count > self.fail_after:
            raise RuntimeError("boom")
        return chunk_bytes

    def _encode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return self._encode_single(chunk_bytes, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length


def _create_array(
    store: MemoryStore | zarr.StorePath,
    *,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    codecs: tuple[object, ...],
    dtype: str = "uint8",
    fill_value: int = 0,
) -> zarr.Array:
    return zarr.create_array(
        store,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
        serializer=codecs[0],
        compressors=codecs[1:],
        overwrite=True,
        config={"write_empty_chunks": False},
    )


def _write_chunks(array: zarr.Array, shape: tuple[int, int], chunks: tuple[int, int]) -> None:
    for row in range(0, shape[0], chunks[0]):
        for column in range(0, shape[1], chunks[1]):
            rows, columns = np.meshgrid(
                np.arange(row, row + chunks[0], dtype=np.int64),
                np.arange(column, column + chunks[1], dtype=np.int64),
                indexing="ij",
            )
            block = ((rows * shape[1] + columns) % 251 + 1).astype(np.uint8)
            array[row : row + chunks[0], column : column + chunks[1]] = block


def test_rechunk_direct_decodes_each_source_chunk_once() -> None:
    shape = (4096, 16)
    source_chunks = (4096, 16)
    target_chunks = (32, 2048)
    source_store = MemoryStore()
    target_store = MemoryStore()
    source = _create_array(
        source_store,
        shape=shape,
        chunks=source_chunks,
        codecs=(BytesCodec(endian=None), CountingCodec()),
    )
    _write_chunks(source, shape, source_chunks)
    target = _create_array(
        target_store,
        shape=shape,
        chunks=target_chunks,
        codecs=(BytesCodec(endian=None),),
    )

    CountingCodec.decode_count = 0
    CountingCodec.decoded_bytes = 0
    result = zarr.rechunk(source, target, work_memory_budget=10_000_000)

    assert CountingCodec.decode_count == 1
    assert result.chunks_written == 128
    assert result.chunks_empty == 0
    assert result.peak_bytes < 8_000_000 + 2 * 64 * 1024
    np.testing.assert_array_equal(target[:], source[:])


def test_rechunk_interleaved_direct_counting_codec() -> None:
    shape = (128, 128)
    source_chunks = (128, 16)
    target_chunks = (32, 128)
    source_store = MemoryStore()
    target_store = MemoryStore()
    source = _create_array(
        source_store,
        shape=shape,
        chunks=source_chunks,
        codecs=(BytesCodec(endian=None), CountingCodec()),
    )
    _write_chunks(source, shape, source_chunks)
    target = _create_array(
        target_store,
        shape=shape,
        chunks=target_chunks,
        codecs=(BytesCodec(endian=None),),
    )

    CountingCodec.decode_count = 0
    result = zarr.rechunk(source, target, work_memory_budget=1_000_000)

    assert CountingCodec.decode_count == 8
    assert result.chunks_written == 4
    np.testing.assert_array_equal(target[:], source[:])


def test_rechunk_temp_store_decodes_each_source_chunk_once_and_cleans_up() -> None:
    shape = (4096, 16)
    source_store = MemoryStore()
    target_store = MemoryStore()
    temp_store = MemoryStore()
    source = _create_array(
        source_store,
        shape=shape,
        chunks=(4096, 16),
        codecs=(BytesCodec(endian=None), CountingCodec()),
    )
    _write_chunks(source, shape, (4096, 16))
    target = _create_array(
        target_store,
        shape=shape,
        chunks=(32, 2048),
        codecs=(BytesCodec(endian=None),),
    )

    CountingCodec.decode_count = 0
    result = zarr.rechunk(
        source, target, work_memory_budget=300_000, temp_store=temp_store
    )

    assert CountingCodec.decode_count == 1
    assert result.chunks_written == 128
    assert result.peak_bytes < 300_000
    np.testing.assert_array_equal(target[:], source[:])
    assert temp_store._store_dict == {}


def test_rechunk_rejects_budget_before_target_mutation() -> None:
    shape = (4096, 16)
    source_store = MemoryStore()
    target_store = MemoryStore()
    source = _create_array(
        source_store, shape=shape, chunks=(4096, 16), codecs=(BytesCodec(endian=None),)
    )
    target = _create_array(
        target_store, shape=shape, chunks=(32, 2048), codecs=(BytesCodec(endian=None),)
    )
    target[:] = 1
    target_keys_before = set(target_store._store_dict)

    with pytest.raises(MemoryError, match="working memory"):
        zarr.rechunk(source, target, work_memory_budget=100_000)

    assert set(target_store._store_dict) == target_keys_before


def test_rechunk_validates_metadata_and_edge_chunks() -> None:
    source_store = MemoryStore()
    target_store = MemoryStore()
    source = _create_array(
        source_store, shape=(10, 10), chunks=(4, 10), codecs=(BytesCodec(endian=None),)
    )
    target = _create_array(
        target_store,
        shape=(10, 10),
        chunks=(4, 10),
        codecs=(BytesCodec(endian=None),),
        fill_value=9,
    )

    with pytest.raises(ValueError, match="fill values"):
        zarr.rechunk(source, target, work_memory_budget=1_000_000)

    different_dtype = _create_array(
        target_store,
        shape=(10, 10),
        chunks=(4, 10),
            dtype="int8",
        codecs=(BytesCodec(endian=None),),
    )
    with pytest.raises(ValueError, match="dtypes"):
        zarr.rechunk(source, different_dtype, work_memory_budget=1_000_000)

    wrong_target = _create_array(
        target_store, shape=(12, 10), chunks=(4, 10), codecs=(BytesCodec(endian=None),)
    )
    with pytest.raises(ValueError, match="shapes"):
        zarr.rechunk(source, wrong_target, work_memory_budget=1_000_000)

    bad_edge = _create_array(
        target_store, shape=(10, 10), chunks=(6, 10), codecs=(BytesCodec(endian=None),)
    )
    with pytest.raises(ValueError, match="edge chunks"):
        zarr.rechunk(source, bad_edge, work_memory_budget=1_000_000)


def test_rechunk_does_not_materialize_empty_targets() -> None:
    shape = (8, 8)
    source_store = MemoryStore()
    target_store = MemoryStore()
    source = _create_array(
        source_store, shape=shape, chunks=(4, 8), codecs=(BytesCodec(endian=None),)
    )
    source[:4, :4] = 1
    target = _create_array(
        target_store, shape=shape, chunks=(8, 4), codecs=(BytesCodec(endian=None),)
    )

    result = zarr.rechunk(source, target, work_memory_budget=1_000_000)

    assert result.chunks_completed == 2
    assert result.chunks_written == 1
    assert result.chunks_empty == 1
    assert "c/0/1" not in target_store._store_dict
    np.testing.assert_array_equal(target[:], source[:])


def test_rechunk_rectilinear_temp_grid() -> None:
    source_store = MemoryStore()
    target_store = MemoryStore()
    temp_store = MemoryStore()
    source = _create_array(
        source_store,
        shape=(10, 10),
        chunks=(6, 4),
        codecs=(BytesCodec(endian=None),),
    )
    source[:] = np.arange(100, dtype=np.uint8).reshape(10, 10)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        target = zarr.create_array(
            target_store,
            shape=(10, 10),
            chunks=((6, 4), (4, 4, 2)),
            dtype="uint8",
            serializer=BytesCodec(endian=None),
            compressors=(),
            overwrite=True,
            config={"write_empty_chunks": False},
        )

    result = zarr.rechunk(
        source, target, work_memory_budget=1_000_000, temp_store=temp_store
    )

    assert result.chunks_completed == 6
    np.testing.assert_array_equal(target[:], source[:])
    assert temp_store._store_dict == {}


def test_rechunk_failure_cleans_temp_and_preserves_chunk_visibility() -> None:
    shape = (128, 128)
    source_store = MemoryStore()
    target_store = MemoryStore()
    temp_store = MemoryStore()
    FailingCodec.encode_count = 0
    source = _create_array(
        source_store, shape=shape, chunks=(64, 128), codecs=(BytesCodec(endian=None),)
    )
    source[:] = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    target = _create_array(
        target_store,
        shape=shape,
        chunks=(32, 128),
        codecs=(BytesCodec(endian=None), FailingCodec(fail_after=1)),
    )

    with pytest.raises(RuntimeError, match="boom"):
        zarr.rechunk(source, target, work_memory_budget=1_000_000, temp_store=temp_store)

    assert temp_store._store_dict == {}
    # The copy deletes the old output up front; completed chunks written before the
    # failure remain visible, but there is no attempt to provide a transaction.
    target_keys = {
        key for key in target_store._store_dict if not key.endswith("zarr.json")
    }
    assert len(target_keys) == 1
