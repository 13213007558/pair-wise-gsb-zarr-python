from __future__ import annotations

import pathlib

import numpy as np

import zarr
from zarr.codecs import BytesCodec, GzipCodec, ZstdCodec
from zarr.core.sync import sync
from zarr.integrity import (
    IntegrityManifest,
    async_create_manifest,
    async_verify_manifest,
    canonical_bytes,
    create_manifest,
    verify_manifest,
)
from zarr.storage import MemoryStore


def _sample() -> np.ndarray:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(10, 9)).astype(np.float64)
    data[0, 0] = np.nan
    data[1, 1] = -0.0
    data[9, 8] = np.nan
    return data


def test_manifest_invariant_to_compressor_and_shard_layout() -> None:
    data = _sample()

    store_zstd = MemoryStore()
    arr_zstd = zarr.create_array(
        store_zstd,
        shape=data.shape,
        chunks=(4, 5),
        dtype="float64",
        compressors=[ZstdCodec(level=1)],
        fill_value=0.0,
    )
    arr_zstd[:] = data

    store_sharded = MemoryStore()
    arr_sharded = zarr.create_array(
        store_sharded,
        shape=data.shape,
        chunks=(4, 5),
        shards=(8, 10),
        dtype="float64",
        compressors=[GzipCodec(level=1)],
        serializer=BytesCodec(endian="big"),
        fill_value=0.0,
    )
    arr_sharded[:] = data

    store_plain = MemoryStore()
    arr_plain = zarr.create_array(
        store_plain,
        shape=data.shape,
        chunks=(4, 5),
        dtype="float64",
        compressors=None,
        fill_value=0.0,
    )
    arr_plain[:] = data

    manifest_zstd = create_manifest(arr_zstd)
    manifest_sharded = create_manifest(arr_sharded)
    manifest_plain = create_manifest(arr_plain)

    assert manifest_sharded.to_dict() == manifest_zstd.to_dict()
    assert manifest_plain.to_dict() == manifest_zstd.to_dict()
    # Boundary chunks are clipped; codec padding never enters the manifest.
    assert manifest_zstd.chunk_sizes == ((4, 4, 2), (5, 4))


def test_canonical_bytes_are_host_independent() -> None:
    nan_payloads = np.array([1.5, np.nan, np.nan], dtype="<f8")
    nan_payloads_bits = nan_payloads.view(np.uint64)
    nan_payloads_bits[1] = 0x7FF0000000000001  # signaling NaN with payload
    nan_payloads_bits[2] = 0xFFF8000000000ABC  # negative quiet NaN with payload
    nan_canonical = np.array([1.5, np.nan, np.nan], dtype="<f8")
    nan_canonical_bits = nan_canonical.view(np.uint64)
    nan_canonical_bits[1] = 0x7FF8000000000000
    nan_canonical_bits[2] = 0x7FF8000000000000
    assert canonical_bytes(nan_payloads) == canonical_bytes(nan_canonical)

    signed_zero = np.array([0.0, -0.0], dtype="<f8")
    positive_zero = np.array([0.0, 0.0], dtype="<f8")
    assert canonical_bytes(signed_zero) == canonical_bytes(positive_zero)

    little = np.arange(4, dtype="<i4")
    big = np.arange(4, dtype=">i4")
    assert canonical_bytes(little) == canonical_bytes(big)
    assert canonical_bytes(np.asfortranarray(little)) == canonical_bytes(little)

    complex_values = np.array([1 + 2j, np.nan - 0j, 0 - 0j], dtype="<c16")
    complex_other = np.array([1 + 2j, np.nan + 0j, 0 + 0j], dtype=">c16")
    assert canonical_bytes(complex_values) == canonical_bytes(complex_other)


async def test_partial_and_full_verification_find_different_changes() -> None:
    data = np.arange(24, dtype="int32").reshape(6, 4)
    store = MemoryStore()
    arr = zarr.create_array(
        store,
        shape=data.shape,
        chunks=(2, 2),
        dtype="int32",
        compressors=None,
        fill_value=0,
    )
    arr[:] = data
    manifest = await async_create_manifest(arr)

    arr[0:2, 0:2] = -1  # selected chunk (0, 0)
    arr[4:6, 2:4] = -2  # chunk (2, 1), outside the partial selection

    selection = (slice(0, 2), slice(0, 2))
    local = await async_verify_manifest(arr, manifest, selection=selection)
    assert not local.ok
    assert local.checked_coords == ((0, 0),)
    assert {chunk.coords: chunk.status for chunk in local.chunks} == {
        (0, 0): "changed",
    }

    full = verify_manifest(arr, manifest)
    assert not full.ok
    assert [chunk.coords for chunk in full.changed] == [(0, 0), (2, 1)]
    assert full.missing == ()


def test_implicit_fill_is_recorded_and_missing_is_distinct() -> None:
    store = MemoryStore()
    arr = zarr.create_array(
        store,
        shape=(6,),
        chunks=(2,),
        dtype="int32",
        compressors=None,
        fill_value=0,
    )
    arr[0:2] = [1, 2]
    manifest = create_manifest(arr)
    assert [(record.coords, record.stored) for record in manifest.chunks] == [
        ((0,), True),
        ((1,), False),
        ((2,), False),
    ]

    initial = verify_manifest(arr, manifest)
    assert initial.ok

    sync((arr.store_path / arr.metadata.encode_chunk_key((0,))).delete())
    after_delete = verify_manifest(arr, manifest)
    assert not after_delete.ok
    assert [chunk.coords for chunk in after_delete.missing] == [(0,)]
    assert after_delete.changed == ()


def test_sharded_inner_chunk_implicit_fill_is_recorded() -> None:
    store = MemoryStore()
    arr = zarr.create_array(
        store,
        shape=(6,),
        chunks=(2,),
        shards=(4,),
        dtype="int32",
        compressors=None,
        fill_value=0,
    )
    arr[0:2] = [1, 2]
    manifest = create_manifest(arr)
    # Inner chunk (0,) is stored; (1,) is elided inside shard 0; shard 1 is absent.
    assert [(record.coords, record.stored) for record in manifest.chunks] == [
        ((0,), True),
        ((1,), False),
        ((2,), False),
    ]
    assert verify_manifest(arr, manifest).ok


def test_metadata_incompatibility_is_reported_not_mixed_with_chunks() -> None:
    source = MemoryStore()
    src = zarr.create_array(
        source, shape=(6,), chunks=(2,), dtype="int32", fill_value=0, compressors=None
    )
    src[:] = np.arange(6, dtype="int32")
    manifest = create_manifest(src)

    target = MemoryStore()
    dst = zarr.create_array(
        target, shape=(6,), chunks=(2,), dtype="float64", fill_value=0.0
    )
    dst[:] = np.arange(6, dtype="float64")
    report = verify_manifest(dst, manifest)
    assert not report.metadata_compatible
    assert report.chunks == ()
    assert any("Dtype mismatch" in message for message in report.metadata_errors)


def test_manifest_save_load_roundtrip(tmp_path: pathlib.Path) -> None:
    store = MemoryStore()
    arr = zarr.create_array(
        store, shape=(4,), chunks=(2,), dtype="int32", fill_value=0, compressors=None
    )
    arr[:] = [1, 2, 3, 4]
    manifest = create_manifest(arr)
    path = tmp_path / "integrity.json"
    manifest.save(path)
    loaded = IntegrityManifest.load(path)
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.verify(arr).ok


def test_v2_array_manifest_and_missing_chunk() -> None:
    store = MemoryStore()
    arr = zarr.create_array(
        store,
        shape=(4,),
        chunks=(2,),
        dtype="int32",
        fill_value=0,
        zarr_format=2,
    )
    arr[0:2] = [5, 6]
    manifest = create_manifest(arr)
    assert verify_manifest(arr, manifest).ok
    sync((arr.store_path / arr.metadata.encode_chunk_key((0,))).delete())
    report = verify_manifest(arr, manifest)
    assert [chunk.coords for chunk in report.missing] == [(0,)]
