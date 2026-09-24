from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.codecs import GzipCodec, ZstdCodec
from zarr.core.sync import sync
from zarr.integrity import (
    IntegrityManifest,
    _canonical_bytes,
    _canonical_dtype,
    compute_manifest,
    verify_manifest,
)
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.types import AnyArray


def _make_array(
    path: Path,
    data: np.ndarray[Any, Any],
    *,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None = None,
    compressors: Any = "auto",
    zarr_format: int = 3,
) -> AnyArray:
    store = LocalStore(str(path))
    arr = zarr.create_array(
        store=store,
        shape=data.shape,
        chunks=chunks,
        shards=shards,
        dtype=data.dtype,
        compressors=compressors,
        zarr_format=zarr_format,
    )
    arr[:] = data
    return arr


def test_manifest_independent_of_compressor_and_sharding(tmp_path: Path) -> None:
    """Same logical data, different encodings -> identical chunk digests."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(8, 8))

    arr_plain = _make_array(tmp_path / "plain", data, chunks=(4, 4), compressors=[GzipCodec()])
    arr_sharded = _make_array(
        tmp_path / "sharded",
        data,
        chunks=(2, 2),
        shards=(4, 4),
        compressors=[ZstdCodec()],
    )

    manifest_plain = compute_manifest(arr_plain)
    manifest_sharded = compute_manifest(arr_sharded)

    # Both arrays share the same logical chunk grid (4, 4); only the encoding differs.
    assert manifest_plain.array_metadata["chunks"] == [4, 4]
    assert manifest_sharded.array_metadata["chunks"] == [4, 4]
    assert set(manifest_plain.chunks) == set(manifest_sharded.chunks)
    for coords, record in manifest_plain.chunks.items():
        assert record.digest == manifest_sharded.chunks[coords].digest

    # Cross-verification: a manifest from one encoding validates the other.
    assert verify_manifest(arr_sharded, manifest_plain).ok
    assert verify_manifest(arr_plain, manifest_sharded).ok


def test_canonical_bytes_normalize_nan_zero_and_byteorder() -> None:
    dtype = _canonical_dtype(np.dtype("<f8"))

    # negative zero hashes like positive zero
    assert _canonical_bytes(np.array([-0.0, 1.5]), dtype) == _canonical_bytes(
        np.array([0.0, 1.5]), dtype
    )

    # non-canonical NaN payloads hash like the canonical quiet NaN
    weird_nan = np.array([np.uint64(0x7FF0000000000001).view(np.float64), 1.0])
    assert np.isnan(weird_nan[0])
    assert _canonical_bytes(weird_nan, dtype) == _canonical_bytes(np.array([np.nan, 1.0]), dtype)

    # big-endian values hash like their little-endian equivalents
    values = np.array([1.0, -2.5, np.nan, -0.0])
    be = values.astype(">f8")
    assert _canonical_bytes(be, _canonical_dtype(be.dtype)) == _canonical_bytes(values, dtype)

    # complex: normalization applies to real and imaginary parts
    cvals = np.array([complex(-0.0, np.nan), 1.0 + 2.0j])
    cdtype = _canonical_dtype(np.dtype("<c16"))
    assert _canonical_bytes(cvals, cdtype) == _canonical_bytes(
        np.array([complex(0.0, np.nan), 1.0 + 2.0j]), cdtype
    )

    # non-contiguous input hashes like its contiguous equivalent
    base = np.arange(12.0).reshape(3, 4)
    assert _canonical_bytes(base[:, ::2], dtype) == _canonical_bytes(
        np.ascontiguousarray(base[:, ::2]), dtype
    )


def test_edge_chunk_padding_excluded(tmp_path: Path) -> None:
    """Only valid elements of a tail chunk participate in the digest."""
    data = np.arange(6.0)
    arr = _make_array(tmp_path / "edge", data, chunks=(4,), compressors=[])
    manifest = compute_manifest(arr)
    assert set(manifest.chunks) == {(0,), (1,)}
    assert verify_manifest(arr, manifest).ok

    # the tail chunk digest covers only the 2 valid elements
    from zarr.integrity import _digest_chunk

    dtype = _canonical_dtype(arr.dtype)
    expected = _digest_chunk(dtype, (2,), _canonical_bytes(data[4:6], dtype))
    assert manifest.chunks[(1,)].digest == expected


def test_partial_and_full_verification(tmp_path: Path) -> None:
    """Local report covers only the selection; full verification finds all changes."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(8, 8))

    arr_a = _make_array(tmp_path / "a", data, chunks=(4, 4), compressors=[GzipCodec()])
    arr_b = _make_array(tmp_path / "b", data, chunks=(4, 4), compressors=[ZstdCodec()])

    manifest = compute_manifest(arr_a)
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    loaded = IntegrityManifest.load(manifest_path)
    assert loaded == manifest

    # identical logical data under a different encoding verifies cleanly
    assert verify_manifest(arr_b, loaded).ok

    # modify one chunk inside the selection and one outside of it
    arr_b[0:4, 0:4] = -1.0  # chunk (0, 0), selected
    arr_b[4:8, 4:8] = -2.0  # chunk (1, 1), outside the selection

    partial = verify_manifest(arr_b, loaded, selection=np.s_[0:4, 0:4])
    assert partial.selection_only
    assert not partial.ok
    assert partial.changed == [(0, 0)]
    assert partial.missing == []
    assert set(partial.results) == {(0, 0)}  # out-of-range chunk not even read

    full = verify_manifest(arr_b, loaded)
    assert not full.selection_only
    assert full.changed == [(0, 0), (1, 1)]
    assert full.missing == []
    assert "changed" in full.summary()


def test_missing_chunk_and_implicit_fill(tmp_path: Path) -> None:
    """Distinguish data changed, chunk became missing, and implicit fill."""
    arr = zarr.create_array(
        store=LocalStore(str(tmp_path / "arr")),
        shape=(4, 4),
        chunks=(2, 2),
        dtype="float64",
        compressors=[],
    )
    arr[0:2, 0:2] = 1.0  # chunk (0, 0) stored
    arr[2:4, 2:4] = 2.0  # chunk (1, 1) stored
    # chunks (0, 1) and (1, 0) are never written -> implicit fill

    manifest = compute_manifest(arr)
    assert manifest.chunks[(0, 0)].implicit_fill is False
    assert manifest.chunks[(1, 1)].implicit_fill is False
    assert manifest.chunks[(0, 1)].implicit_fill is True
    assert manifest.chunks[(1, 0)].implicit_fill is True
    assert verify_manifest(arr, manifest).ok

    # a chunk that had data becomes missing
    key = arr.metadata.encode_chunk_key((0, 0))
    sync(arr.store.delete(key))
    report = verify_manifest(arr, manifest)
    assert not report.ok
    assert report.missing == [(0, 0)]
    assert report.changed == []
    # implicit-fill chunks are still absent and still fine
    assert report.results[(0, 1)].status == "ok"
    assert report.results[(1, 0)].status == "ok"

    # changing stored data reports "changed", not "missing"
    arr[2:4, 2:4] = 3.0
    report = verify_manifest(arr, manifest)
    assert report.changed == [(1, 1)]
    assert report.missing == [(0, 0)]


def test_metadata_incompatible(tmp_path: Path) -> None:
    data = np.arange(16.0).reshape(4, 4)
    arr = _make_array(tmp_path / "arr", data, chunks=(2, 2), compressors=[])
    manifest = compute_manifest(arr)

    other_shape = _make_array(
        tmp_path / "other", np.arange(24.0).reshape(4, 6), chunks=(2, 2), compressors=[]
    )
    report = verify_manifest(other_shape, manifest)
    assert not report.metadata_compatible
    assert not report.ok
    assert report.results == {}
    assert any("shape" in issue for issue in report.metadata_issues)

    other_dtype = zarr.create_array(
        store=LocalStore(str(tmp_path / "other_dtype")),
        shape=(4, 4),
        chunks=(2, 2),
        dtype="int32",
        compressors=[],
    )
    report = verify_manifest(other_dtype, manifest)
    assert not report.metadata_compatible
    assert any("dtype" in issue for issue in report.metadata_issues)


def test_manifest_roundtrip_json(tmp_path: Path) -> None:
    data = np.arange(12.0).reshape(3, 4)
    arr = _make_array(tmp_path / "arr", data, chunks=(2, 2), compressors=[])
    manifest = compute_manifest(arr)

    restored = IntegrityManifest.from_json(manifest.to_json())
    assert restored == manifest
    assert manifest.array_metadata["dtype"] == "<f8"
    assert manifest.array_metadata["shape"] == [3, 4]
    assert manifest.array_metadata["chunks"] == [2, 2]

    with pytest.raises(ValueError, match="format version"):
        IntegrityManifest.from_json('{"format_version": 999, "array": {}, "chunks": {}}')
