"""Saveable logical-chunk integrity manifests for Zarr arrays.

The manifest summarizes decoded array data at logical-chunk granularity.
Compressors, filters, byte-oriented serialization, and shard placement are
not part of a digest: two arrays with identical logical chunks produce the
same manifest even when their encoded storage differs.

Digest canonicalization is independent of the host machine:

* elements are serialized in C order with explicit little-endian byte order;
* boundary chunks cover only elements inside the array shape, never codec
  padding;
* every NaN is normalized to the positive quiet NaN for its dtype;
* negative zero is normalized to positive zero.

Variable-length, object, and structured dtypes are not supported.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from zarr.core.array import Array, AsyncArray, _sharding_codec
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import ceildiv
from zarr.core.indexing import BasicIndexer
from zarr.core.sync import sync

__all__ = [
    "ChunkRecord",
    "ChunkVerification",
    "IntegrityManifest",
    "VerificationReport",
    "async_create_manifest",
    "async_verify_manifest",
    "create_manifest",
    "verify_manifest",
]

MANIFEST_VERSION = 1
DIGEST_ALGORITHM = "sha256"

STATUS_CHANGED = "changed"
STATUS_MATERIALIZED = "materialized"
STATUS_MISSING = "missing"
STATUS_OK = "ok"

_CANONICAL_NAN = {
    2: 0x7E00,
    4: 0x7FC00000,
    8: 0x7FF8000000000000,
}


@dataclass(frozen=True)
class ChunkRecord:
    """Digest and original storage state of one logical chunk."""

    coords: tuple[int, ...]
    digest: str
    stored: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "coords": list(self.coords),
            "digest": self.digest,
            "stored": self.stored,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkRecord:
        return cls(
            coords=tuple(int(value) for value in data["coords"]),
            digest=str(data["digest"]),
            stored=bool(data["stored"]),
        )


@dataclass(frozen=True)
class IntegrityManifest:
    """A portable, encoding-independent summary of a complete logical array.

    The chunk_sizes field contains valid, boundary-clipped edge lengths for
    each dimension. Codecs and the storage shard grid are deliberately omitted.
    """

    shape: tuple[int, ...]
    dtype: str
    chunk_sizes: tuple[tuple[int, ...], ...]
    fill_value: str | None
    chunks: tuple[ChunkRecord, ...]
    zarr_format: int | None = None
    manifest_version: int = MANIFEST_VERSION
    digest_algorithm: str = DIGEST_ALGORITHM

    def chunk_coords(self) -> tuple[tuple[int, ...], ...]:
        return tuple(record.coords for record in self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "digest_algorithm": self.digest_algorithm,
            "zarr_format": self.zarr_format,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "chunk_sizes": [list(sizes) for sizes in self.chunk_sizes],
            "fill_value": self.fill_value,
            "chunks": [record.to_dict() for record in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrityManifest:
        chunks = tuple(ChunkRecord.from_dict(record) for record in data["chunks"])
        chunk_sizes = tuple(
            tuple(int(size) for size in sizes) for sizes in data["chunk_sizes"]
        )
        fill_value = data.get("fill_value")
        return cls(
            shape=tuple(int(value) for value in data["shape"]),
            dtype=str(data["dtype"]),
            chunk_sizes=chunk_sizes,
            fill_value=None if fill_value is None else str(fill_value),
            chunks=chunks,
            zarr_format=None if data.get("zarr_format") is None else int(data["zarr_format"]),
            manifest_version=int(data.get("manifest_version", MANIFEST_VERSION)),
            digest_algorithm=str(data.get("digest_algorithm", DIGEST_ALGORITHM)),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> IntegrityManifest:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> IntegrityManifest:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def verify(self, array: Array | AsyncArray, selection: Any = None) -> VerificationReport:
        return verify_manifest(array, self, selection=selection)

    async def async_verify(
        self, array: Array | AsyncArray, selection: Any = None
    ) -> VerificationReport:
        return await async_verify_manifest(array, self, selection=selection)


@dataclass(frozen=True)
class ChunkVerification:
    """Result for one logical chunk touched by a verification selection."""

    coords: tuple[int, ...]
    status: Literal["changed", "materialized", "missing", "ok"]
    expected_digest: str
    actual_digest: str
    expected_stored: bool
    actual_stored: bool

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "coords": list(self.coords),
            "status": self.status,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "expected_stored": self.expected_stored,
            "actual_stored": self.actual_stored,
        }


@dataclass(frozen=True)
class VerificationReport:
    """Verification outcome, including logical metadata compatibility."""

    chunks: tuple[ChunkVerification, ...]
    metadata_errors: tuple[str, ...] = ()

    @property
    def metadata_compatible(self) -> bool:
        return not self.metadata_errors

    @property
    def ok(self) -> bool:
        return self.metadata_compatible and all(chunk.ok for chunk in self.chunks)

    @property
    def checked_coords(self) -> tuple[tuple[int, ...], ...]:
        return tuple(chunk.coords for chunk in self.chunks)

    @property
    def changed(self) -> tuple[ChunkVerification, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.status == STATUS_CHANGED)

    @property
    def missing(self) -> tuple[ChunkVerification, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.status == STATUS_MISSING)

    @property
    def materialized(self) -> tuple[ChunkVerification, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.status == STATUS_MATERIALIZED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_compatible": self.metadata_compatible,
            "metadata_errors": list(self.metadata_errors),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


def create_manifest(array: Array | AsyncArray) -> IntegrityManifest:
    """Create a saveable integrity manifest synchronously."""
    return sync(async_create_manifest(array))


def verify_manifest(
    array: Array | AsyncArray,
    manifest: IntegrityManifest,
    selection: Any = None,
) -> VerificationReport:
    """Verify all chunks, or only chunks intersecting a basic selection."""
    return sync(async_verify_manifest(array, manifest, selection=selection))


async def async_create_manifest(array: Array | AsyncArray) -> IntegrityManifest:
    """Create an integrity manifest from decoded logical data."""
    arr = _as_async_array(array)
    dtype = np.dtype(arr.dtype)
    _require_supported_dtype(dtype)

    shape = tuple(int(size) for size in arr.shape)
    chunk_sizes = tuple(tuple(int(size) for size in axis) for axis in arr.read_chunk_sizes)
    logical_grid = ChunkGrid.from_sizes(shape, chunk_sizes)
    stored_coords = await _stored_logical_chunks(arr)

    records: list[ChunkRecord] = []
    for coords in sorted(logical_grid.all_chunk_coords()):
        chunk_spec = logical_grid[coords]
        if chunk_spec is None:  # pragma: no cover - iterator yields valid coords
            raise IndexError(f"Chunk coordinates {coords} are out of bounds.")
        data = np.asarray(await arr.getitem(chunk_spec.slices))
        records.append(
            ChunkRecord(
                coords=coords,
                digest=digest_elements(data),
                stored=coords in stored_coords,
            )
        )

    return IntegrityManifest(
        shape=shape,
        dtype=canonical_dtype(dtype),
        chunk_sizes=chunk_sizes,
        fill_value=canonical_scalar_hex(arr.metadata.fill_value, dtype),
        chunks=tuple(records),
        zarr_format=int(arr.metadata.zarr_format),
    )


async def async_verify_manifest(
    array: Array | AsyncArray,
    manifest: IntegrityManifest,
    selection: Any = None,
) -> VerificationReport:
    """Asynchronously verify a manifest against current decoded storage."""
    arr = _as_async_array(array)
    dtype = np.dtype(arr.dtype)

    metadata_errors = _metadata_errors(arr, dtype, manifest)
    if metadata_errors:
        return VerificationReport(chunks=(), metadata_errors=tuple(metadata_errors))

    shape = tuple(int(size) for size in arr.shape)
    logical_grid = ChunkGrid.from_sizes(shape, manifest.chunk_sizes)
    selected_coords = _selection_chunk_coords(shape, manifest.chunk_sizes, selection)
    stored_coords = await _stored_logical_chunks(arr)
    expected = {record.coords: record for record in manifest.chunks}

    results: list[ChunkVerification] = []
    for coords in selected_coords:
        record = expected[coords]
        chunk_spec = logical_grid[coords]
        if chunk_spec is None:  # pragma: no cover - coords come from the grid
            raise IndexError(f"Chunk coordinates {coords} are out of bounds.")
        data = np.asarray(await arr.getitem(chunk_spec.slices))
        actual_digest = digest_elements(data)
        actual_stored = coords in stored_coords

        status: Literal["changed", "materialized", "missing", "ok"]
        if record.stored and not actual_stored:
            status = STATUS_MISSING
        elif not record.stored and actual_stored:
            status = STATUS_MATERIALIZED if actual_digest == record.digest else STATUS_CHANGED
        elif actual_digest != record.digest:
            status = STATUS_CHANGED
        else:
            status = STATUS_OK

        results.append(
            ChunkVerification(
                coords=coords,
                status=status,
                expected_digest=record.digest,
                actual_digest=actual_digest,
                expected_stored=record.stored,
                actual_stored=actual_stored,
            )
        )

    return VerificationReport(chunks=tuple(results))


def digest_elements(values: np.ndarray) -> str:
    """Return the stable SHA-256 digest of decoded array elements."""
    return hashlib.sha256(canonical_bytes(values)).hexdigest()


def canonical_bytes(values: np.ndarray) -> bytes:
    """Serialize decoded elements without host-layout dependence."""
    array = np.ascontiguousarray(np.asarray(values))
    _require_supported_dtype(array.dtype)
    array = _little_endian(array)
    if array.dtype.kind in {"f", "c"}:
        array = _canonical_floats(array)
    return np.ascontiguousarray(array).tobytes(order="C")


def canonical_dtype(dtype: np.dtype[Any]) -> str:
    """Return the stable dtype descriptor used by a manifest."""
    parsed = np.dtype(dtype)
    if parsed.kind in {"O", "V"}:
        _raise_unsupported_dtype(parsed)
    if parsed.kind in {"b", "i", "u", "f", "c", "m", "M", "U"}:
        return str(parsed.newbyteorder("<"))
    return str(parsed)


def canonical_scalar_hex(value: Any, dtype: np.dtype[Any]) -> str | None:
    """Canonicalize a scalar fill value and return its element bytes as hex."""
    if value is None:
        return None
    scalar = np.asarray(value, dtype=dtype)
    if scalar.shape != ():
        raise ValueError("Fill value must be a scalar.")
    return canonical_bytes(scalar).hex()


def _metadata_errors(
    arr: AsyncArray,
    dtype: np.dtype[Any],
    manifest: IntegrityManifest,
) -> list[str]:
    errors: list[str] = []
    if manifest.manifest_version != MANIFEST_VERSION:
        errors.append(
            f"Unsupported manifest version {manifest.manifest_version}; "
            f"expected {MANIFEST_VERSION}."
        )
    if manifest.digest_algorithm != DIGEST_ALGORITHM:
        errors.append(
            f"Unsupported digest algorithm {manifest.digest_algorithm!r}; "
            f"expected {DIGEST_ALGORITHM!r}."
        )

    current_shape = tuple(int(size) for size in arr.shape)
    if current_shape != tuple(manifest.shape):
        errors.append(
            f"Shape mismatch: array has {current_shape}, manifest has {tuple(manifest.shape)}."
        )

    try:
        _require_supported_dtype(dtype)
        current_dtype = canonical_dtype(dtype)
    except NotImplementedError:
        current_dtype = str(dtype)
    if current_dtype != manifest.dtype:
        errors.append(
            f"Dtype mismatch: array has {current_dtype!r}, manifest has {manifest.dtype!r}."
        )

    current_chunks = tuple(
        tuple(int(size) for size in axis) for axis in arr.read_chunk_sizes
    )
    if current_chunks != tuple(manifest.chunk_sizes):
        errors.append(
            "Logical chunk shape mismatch: "
            f"array has {current_chunks}, manifest has {tuple(manifest.chunk_sizes)}."
        )

    try:
        current_fill = canonical_scalar_hex(arr.metadata.fill_value, dtype)
    except (TypeError, ValueError, NotImplementedError):
        current_fill = repr(arr.metadata.fill_value)
    if current_fill != manifest.fill_value:
        errors.append(
            "Fill value mismatch: "
            f"array has {arr.metadata.fill_value!r}, manifest records {manifest.fill_value!r}."
        )

    grid = ChunkGrid.from_sizes(current_shape, current_chunks)
    current_coords = set(grid.all_chunk_coords())
    manifest_coords = {record.coords for record in manifest.chunks}
    missing = sorted(manifest_coords - current_coords)
    unexpected = sorted(current_coords - manifest_coords)
    if missing:
        errors.append(f"Manifest contains chunks absent from the logical grid: {missing}.")
    if unexpected:
        errors.append(f"Current array has chunks absent from the manifest: {unexpected}.")

    return errors


def _selection_chunk_coords(
    shape: tuple[int, ...],
    chunk_sizes: tuple[tuple[int, ...], ...],
    selection: Any,
) -> tuple[tuple[int, ...], ...]:
    grid = ChunkGrid.from_sizes(shape, chunk_sizes)
    if selection is None:
        return tuple(sorted(grid.all_chunk_coords()))
    indexer = BasicIndexer(selection, shape, grid)
    coords = {projection.chunk_coords for projection in indexer}
    return tuple(sorted(coords))


async def _stored_logical_chunks(arr: AsyncArray) -> set[tuple[int, ...]]:
    """Return logical chunks physically represented in storage.

    A non-sharded chunk is stored when its key exists. A sharded inner chunk
    is stored only when its shard exists and the shard index marks that entry
    as present. An all-empty shard therefore reports every inner chunk as
    implicit fill.
    """
    metadata = arr.metadata
    store_path = arr.store_path
    sharding_codec = _sharding_codec(metadata)

    if sharding_codec is None:
        grid = arr._chunk_grid
        coords_list = list(grid.all_chunk_coords())
        exists = await asyncio.gather(
            *[(store_path / metadata.encode_chunk_key(coords)).exists() for coords in coords_list]
        )
        return {coords for coords, present in zip(coords_list, exists, strict=True) if present}

    stored: set[tuple[int, ...]] = set()
    shard_grid = arr._chunk_grid
    logical_shape = arr._chunk_grid_shape
    for shard_coords in shard_grid.all_chunk_coords():
        shard_key = metadata.encode_chunk_key(shard_coords)
        shard_buffer = await (store_path / shard_key).get()
        if shard_buffer is None:
            continue
        shard_spec = shard_grid[shard_coords]
        if shard_spec is None:  # pragma: no cover - coords come from the grid
            raise IndexError(f"Shard coordinates {shard_coords} are out of bounds.")
        chunks_per_shard = tuple(
            ceildiv(shard_size, inner_size)
            for shard_size, inner_size in zip(
                shard_spec.codec_shape, sharding_codec.chunk_shape, strict=True
            )
        )
        index_size = sharding_codec._shard_index_size(chunks_per_shard)
        if sharding_codec.index_location == "start":
            index_bytes = shard_buffer[:index_size]
        else:
            index_bytes = shard_buffer[len(shard_buffer) - index_size :]
        shard_index = sharding_codec._decode_shard_index_sync(index_bytes, chunks_per_shard)
        presence = shard_index.get_full_chunk_map()
        for local_coords in np.ndindex(*chunks_per_shard):
            if not presence[local_coords]:
                continue
            logical_coords = tuple(
                shard_index_axis * per_shard + local
                for shard_index_axis, per_shard, local in zip(
                    shard_coords, chunks_per_shard, local_coords, strict=True
                )
            )
            if all(
                coord < extent
                for coord, extent in zip(logical_coords, logical_shape, strict=True)
            ):
                stored.add(logical_coords)
    return stored


def _as_async_array(array: Array | AsyncArray) -> AsyncArray:
    if isinstance(array, Array):
        return array.async_array
    return array


def _require_supported_dtype(dtype: np.dtype[Any]) -> None:
    if dtype.kind in {"O", "V"}:
        _raise_unsupported_dtype(dtype)


def _raise_unsupported_dtype(dtype: np.dtype[Any]) -> None:
    raise NotImplementedError(
        f"Integrity manifests do not support dtype {dtype!r}: object, structured, "
        "and variable-length dtypes have no encoding-independent canonical form."
    )


def _little_endian(array: np.ndarray) -> np.ndarray:
    """Return the array values with explicit little-endian byte order."""
    dtype = array.dtype
    if dtype.kind == "U":
        codepoints = array.view(np.uint32)
        return codepoints.astype(codepoints.dtype.newbyteorder("<"), copy=False)
    if dtype.kind == "S":
        return array
    try:
        return array.astype(dtype.newbyteorder("<"), copy=False)
    except TypeError:  # pragma: no cover - defensive for exotic dtypes
        return array


def _canonical_floats(array: np.ndarray) -> np.ndarray:
    """Normalize NaN payloads and negative zero in a little-endian array."""
    if array.dtype.kind == "c":
        float_dtype = np.dtype("<f4") if array.dtype.itemsize == 8 else np.dtype("<f8")
        floats = array.view(float_dtype)
        canonical = _canonical_float_array(floats)
        return canonical.view(array.dtype)
    return _canonical_float_array(array)


def _canonical_float_array(array: np.ndarray) -> np.ndarray:
    itemsize = array.dtype.itemsize
    if itemsize not in _CANONICAL_NAN:
        raise NotImplementedError(
            f"Integrity manifests do not support floating dtype {array.dtype!r}."
        )
    int_dtype = np.dtype(f"<u{itemsize}")
    canonical_nan = np.array(_CANONICAL_NAN[itemsize], dtype=int_dtype).view(array.dtype)
    result = array.copy()
    result[np.isnan(result)] = canonical_nan
    result = np.where(result == 0, array.dtype.type(0.0), result)
    return result
