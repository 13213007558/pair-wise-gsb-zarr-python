import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

import zarr
from zarr.codecs import ZstdCodec
from zarr.codecs.sharding import (
    MAX_UINT_64,
    IndexLocation,
    ShardCompactionReport,
    ShardIndexCorruptionError,
    ShardingCodec,
    _ShardIndex,
    compact_shard,
    compact_shard_bytes,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.core.indexing import lexicographic_order_coords

DATA = np.arange(16, dtype=np.uint8).reshape(4, 4)


def _create_array(path: Path, index_location: IndexLocation) -> zarr.Array:
    return zarr.create_array(
        store=str(path),
        shape=(4, 4),
        chunks=(2, 2),
        shards={"shape": (4, 4), "index_location": index_location},
        dtype="uint8",
        compressors=ZstdCodec(),
        fill_value=0,
    )


def _sharding_codec(arr: zarr.Array) -> ShardingCodec:
    codec = next(c for c in arr.metadata.codecs if isinstance(c, ShardingCodec))
    assert isinstance(codec, ShardingCodec)
    return codec


def _chunks_per_shard(arr: zarr.Array) -> tuple[int, ...]:
    assert arr.shards is not None
    return tuple(s // c for s, c in zip(arr.shards, arr.chunks, strict=True))


def _decode_index(data: bytes, codec: ShardingCodec, cps: tuple[int, ...]) -> _ShardIndex:
    index_size = codec._shard_index_size(cps)
    if codec.index_location == "start":
        index_bytes = data[:index_size]
    else:
        index_bytes = data[len(data) - index_size :]
    buf = default_buffer_prototype().buffer.from_bytes(index_bytes)
    return codec._decode_shard_index_sync(buf, cps)


def _replace_index(
    data: bytes, codec: ShardingCodec, cps: tuple[int, ...], offsets_and_lengths: np.ndarray
) -> bytes:
    new_index = codec._encode_shard_index_sync(_ShardIndex(cps, offsets_and_lengths)).to_bytes()
    index_size = codec._shard_index_size(cps)
    if codec.index_location == "start":
        return new_index + data[index_size:]
    return data[: len(data) - index_size] + new_index


def _insert_holes(data: bytes, codec: ShardingCodec, cps: tuple[int, ...], gap: int = 6) -> bytes:
    """Rebuild a shard with unreferenced gap bytes after every chunk payload."""
    index = _decode_index(data, codec, cps)
    index_size = codec._shard_index_size(cps)
    cursor = index_size if codec.index_location == "start" else 0
    offsets_and_lengths = index.offsets_and_lengths.copy()
    pieces: list[bytes] = []
    for coords in lexicographic_order_coords(cps):
        offset, length = (int(v) for v in index.offsets_and_lengths[coords])
        if (offset, length) == (MAX_UINT_64, MAX_UINT_64):
            continue
        pieces.append(data[offset : offset + length])
        pieces.append(b"\x00" * gap)
        offsets_and_lengths[coords] = (cursor, length)
        cursor += length + gap
    return _assemble(codec, cps, offsets_and_lengths, b"".join(pieces))


def _assemble(
    codec: ShardingCodec, cps: tuple[int, ...], offsets_and_lengths: np.ndarray, payload: bytes
) -> bytes:
    new_index = codec._encode_shard_index_sync(_ShardIndex(cps, offsets_and_lengths)).to_bytes()
    if codec.index_location == "start":
        return new_index + payload
    return payload + new_index


def _fragment_digests(
    data: bytes, codec: ShardingCodec, cps: tuple[int, ...]
) -> dict[tuple[int, ...], str]:
    index = _decode_index(data, codec, cps)
    digests: dict[tuple[int, ...], str] = {}
    for coords in lexicographic_order_coords(cps):
        offset, length = (int(v) for v in index.offsets_and_lengths[coords])
        if (offset, length) == (MAX_UINT_64, MAX_UINT_64):
            continue
        digests[coords] = hashlib.sha256(data[offset : offset + length]).hexdigest()
    return digests


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_compact_shard_with_holes(tmp_path: Path, index_location: IndexLocation) -> None:
    src_dir = tmp_path / "src.zarr"
    arr = _create_array(src_dir, index_location)
    arr[:] = DATA
    codec = _sharding_codec(arr)
    cps = _chunks_per_shard(arr)
    shard_path = src_dir / "c" / "0" / "0"

    gapped = _insert_holes(shard_path.read_bytes(), codec, cps, gap=6)
    assert len(gapped) > shard_path.stat().st_size
    shard_path.write_bytes(gapped)
    # the artificially gapped shard is still a valid shard
    np.testing.assert_array_equal(zarr.open_array(src_dir, mode="r")[:], DATA)
    source_digests = _fragment_digests(gapped, codec, cps)
    assert len(source_digests) == 4

    dst_dir = tmp_path / "dst.zarr"
    shutil.copytree(src_dir, dst_dir)
    dst_shard = dst_dir / "c" / "0" / "0"
    report = compact_shard(shard_path, dst_shard, codec=codec, chunks_per_shard=cps)

    assert isinstance(report, ShardCompactionReport)
    assert report.chunks_present == 4
    assert report.chunks_empty == 0
    assert report.bytes_reclaimed == 4 * 6
    assert report.compacted_size < report.source_size
    assert dst_shard.stat().st_size == report.compacted_size
    # the source shard is read-only for the tool: bytes are unchanged
    assert shard_path.read_bytes() == gapped
    # every live compressed fragment is byte-identical after compaction
    assert _fragment_digests(dst_shard.read_bytes(), codec, cps) == source_digests
    # normal zarr reads return the same array values
    np.testing.assert_array_equal(zarr.open_array(dst_dir, mode="r")[:], DATA)


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_compact_shard_preserves_empty_markers(
    tmp_path: Path, index_location: IndexLocation
) -> None:
    src_dir = tmp_path / "src.zarr"
    arr = _create_array(src_dir, index_location)
    arr[0:2, 0:2] = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    codec = _sharding_codec(arr)
    cps = _chunks_per_shard(arr)
    shard_path = src_dir / "c" / "0" / "0"

    gapped = _insert_holes(shard_path.read_bytes(), codec, cps)
    compacted, report = compact_shard_bytes(gapped, codec=codec, chunks_per_shard=cps)

    assert report.chunks_present == 1
    assert report.chunks_empty == 3
    new_index = _decode_index(compacted, codec, cps)
    for coords in [(0, 1), (1, 0), (1, 1)]:
        assert tuple(int(v) for v in new_index.offsets_and_lengths[coords]) == (
            MAX_UINT_64,
            MAX_UINT_64,
        )
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[0:2, 0:2] = [[1, 2], [3, 4]]
    dst_dir = tmp_path / "dst.zarr"
    shutil.copytree(src_dir, dst_dir)
    (dst_dir / "c" / "0" / "0").write_bytes(compacted)
    np.testing.assert_array_equal(zarr.open_array(dst_dir, mode="r")[:], expected)


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_compact_shard_partial_overlap_is_corruption(
    tmp_path: Path, index_location: IndexLocation
) -> None:
    src_dir = tmp_path / "src.zarr"
    arr = _create_array(src_dir, index_location)
    arr[:] = DATA
    codec = _sharding_codec(arr)
    cps = _chunks_per_shard(arr)
    shard_path = src_dir / "c" / "0" / "0"

    index = _decode_index(shard_path.read_bytes(), codec, cps)
    offsets_and_lengths = index.offsets_and_lengths.copy()
    offset0, length0 = (int(v) for v in offsets_and_lengths[0, 0])
    assert length0 > 1
    # chunk (0, 1) partially overlaps chunk (0, 0)
    offsets_and_lengths[0, 1] = (offset0 + 1, length0)
    shard_path.write_bytes(_replace_index(shard_path.read_bytes(), codec, cps, offsets_and_lengths))

    target = tmp_path / "out.shard"
    with pytest.raises(ShardIndexCorruptionError) as excinfo:
        compact_shard(shard_path, target, codec=codec, chunks_per_shard=cps)
    # the offending inner-chunk coordinates are reported before any output exists
    assert (0, 0) in excinfo.value.entries
    assert (0, 1) in excinfo.value.entries
    assert "(0, 0)" in str(excinfo.value)
    assert "(0, 1)" in str(excinfo.value)
    assert not target.exists()


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_compact_shard_out_of_bounds_range_is_corruption(
    tmp_path: Path, index_location: IndexLocation
) -> None:
    src_dir = tmp_path / "src.zarr"
    arr = _create_array(src_dir, index_location)
    arr[:] = DATA
    codec = _sharding_codec(arr)
    cps = _chunks_per_shard(arr)
    shard_path = src_dir / "c" / "0" / "0"

    index = _decode_index(shard_path.read_bytes(), codec, cps)
    offsets_and_lengths = index.offsets_and_lengths.copy()
    # chunk (1, 1) points past the end of the shard
    offsets_and_lengths[1, 1] = (shard_path.stat().st_size, 4)
    shard_path.write_bytes(_replace_index(shard_path.read_bytes(), codec, cps, offsets_and_lengths))

    target = tmp_path / "out.shard"
    with pytest.raises(ShardIndexCorruptionError) as excinfo:
        compact_shard(shard_path, target, codec=codec, chunks_per_shard=cps)
    assert excinfo.value.entries == ((1, 1),)
    assert "(1, 1)" in str(excinfo.value)
    assert not target.exists()


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_compact_shard_index_checksum_mismatch(
    tmp_path: Path, index_location: IndexLocation
) -> None:
    src_dir = tmp_path / "src.zarr"
    arr = _create_array(src_dir, index_location)
    arr[:] = DATA
    codec = _sharding_codec(arr)
    cps = _chunks_per_shard(arr)
    shard_path = src_dir / "c" / "0" / "0"

    data = bytearray(shard_path.read_bytes())
    index_size = codec._shard_index_size(cps)
    # flip a byte inside the index region without updating the checksum
    data[0 if index_location == "start" else len(data) - index_size] ^= 0xFF
    shard_path.write_bytes(bytes(data))

    target = tmp_path / "out.shard"
    with pytest.raises(ValueError, match="checksum"):
        compact_shard(shard_path, target, codec=codec, chunks_per_shard=cps)
    assert not target.exists()


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_compact_shard_identical_range_sharing(
    tmp_path: Path, index_location: IndexLocation
) -> None:
    src_dir = tmp_path / "src.zarr"
    arr = _create_array(src_dir, index_location)
    arr[:] = DATA
    codec = _sharding_codec(arr)
    cps = _chunks_per_shard(arr)
    shard_path = src_dir / "c" / "0" / "0"

    index = _decode_index(shard_path.read_bytes(), codec, cps)
    offsets_and_lengths = index.offsets_and_lengths.copy()
    # chunk (1, 1) references the exact same byte range as chunk (0, 0):
    # legal per the sharding format, and the payload is stored once
    offsets_and_lengths[1, 1] = offsets_and_lengths[0, 0]
    shard_path.write_bytes(_replace_index(shard_path.read_bytes(), codec, cps, offsets_and_lengths))

    compacted, report = compact_shard_bytes(
        shard_path.read_bytes(), codec=codec, chunks_per_shard=cps
    )
    assert report.shared_ranges == 1
    assert report.chunks_present == 4
    shared_length = int(offsets_and_lengths[0, 0][1])
    assert report.bytes_reclaimed == shared_length

    new_index = _decode_index(compacted, codec, cps)
    assert tuple(int(v) for v in new_index.offsets_and_lengths[0, 0]) == tuple(
        int(v) for v in new_index.offsets_and_lengths[1, 1]
    )
    digests = _fragment_digests(compacted, codec, cps)
    assert digests[(0, 0)] == digests[(1, 1)]

    dst_dir = tmp_path / "dst.zarr"
    shutil.copytree(src_dir, dst_dir)
    (dst_dir / "c" / "0" / "0").write_bytes(compacted)
    expected = DATA.copy()
    expected[2:4, 2:4] = DATA[0:2, 0:2]
    np.testing.assert_array_equal(zarr.open_array(dst_dir, mode="r")[:], expected)


def test_compact_shard_rejects_in_place(tmp_path: Path) -> None:
    src_dir = tmp_path / "src.zarr"
    arr = _create_array(src_dir, "end")
    arr[:] = DATA
    codec = _sharding_codec(arr)
    cps = _chunks_per_shard(arr)
    shard_path = src_dir / "c" / "0" / "0"
    original = shard_path.read_bytes()

    with pytest.raises(ValueError, match="different paths"):
        compact_shard(shard_path, shard_path, codec=codec, chunks_per_shard=cps)
    assert shard_path.read_bytes() == original
