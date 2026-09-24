"""Content integrity manifests for Zarr arrays.

This module provides a saveable, content-addressed description of the logical
data stored in a Zarr array. A :class:`IntegrityManifest` records one digest
per logical chunk, computed from *decoded* values (dtype, logical shape and
the valid elements of each chunk), so the digest of a chunk does not depend on
the compressor chain, the shard layout, or the host NumPy memory layout.

Manifests can be serialized to JSON and later used to verify an array, either
in full or restricted to the chunks touched by a basic selection. Verification
distinguishes three situations per chunk: the data changed, the chunk became
missing from the store, or the array metadata is no longer compatible with the
manifest. No repair is attempted; verification is read-only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from zarr.core.chunk_grids import ChunkGrid
from zarr.core.indexing import BasicIndexer
from zarr.core.sync import sync

if TYPE_CHECKING:
    from zarr.core.metadata import ArrayMetadata
    from zarr.types import AnyArray

__all__ = [
    "MANIFEST_FORMAT_VERSION",
    "ChunkRecord",
    "ChunkVerification",
    "IntegrityManifest",
    "VerificationReport",
    "compute_manifest",
    "verify_manifest",
]

MANIFEST_FORMAT_VERSION = 1

_DIGEST_NAMESPACE = b"zarr-integrity-manifest-v1\x00"

ChunkStatus = Literal["ok", "changed", "missing"]


def _canonical_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return a host-independent equivalent of ``dtype``.

    Multi-byte dtypes are normalized to little-endian so that the digest does
    not depend on the byte order the array was stored or decoded with.
    """
    dtype = np.dtype(dtype)
    if dtype.fields is not None:
        raise NotImplementedError(
            f"structured dtypes are not supported by integrity manifests, got {dtype!r}"
        )
    if dtype.hasobject:
        raise NotImplementedError(
            f"object dtypes are not supported by integrity manifests, got {dtype!r}"
        )
    if dtype.byteorder in (">", "="):
        return dtype.newbyteorder("<")
    return dtype


def _normalize_float_parts(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Canonicalize NaN payloads and negative zero in a float array."""
    # Replace every NaN (any payload, either sign, signaling or quiet) with
    # the canonical quiet NaN of this dtype.
    arr = np.where(np.isnan(arr), np.nan, arr)
    # IEEE 754: (-0.0) + (+0.0) == +0.0, so this folds negative zero onto
    # positive zero without changing any other value.
    return arr + np.zeros((), dtype=arr.dtype)


def _canonical_bytes(values: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> bytes:
    """Serialize decoded values to a canonical, host-independent byte string.

    The result depends only on the logical values: little-endian byte order,
    C-contiguous layout, canonical NaN payloads and no negative zero.
    """
    arr = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    kind = arr.dtype.kind
    if kind == "f":
        arr = _normalize_float_parts(arr)
    elif kind == "c":
        normalized = np.empty(arr.shape, dtype=arr.dtype)
        normalized.real = _normalize_float_parts(arr.real)
        normalized.imag = _normalize_float_parts(arr.imag)
        arr = normalized
    return arr.tobytes(order="C")


def _digest_chunk(dtype: np.dtype[Any], shape: tuple[int, ...], data: bytes) -> str:
    """Compute the content digest of one logical chunk."""
    hasher = hashlib.sha256()
    hasher.update(_DIGEST_NAMESPACE)
    hasher.update(dtype.str.encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(json.dumps(list(shape)).encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(data)
    return hasher.hexdigest()


def _chunk_region(
    coords: tuple[int, ...], shape: tuple[int, ...], chunks: tuple[int, ...]
) -> tuple[slice, ...]:
    """Return the array-region slice covered by a logical chunk.

    The region is clipped to the array shape, so padding bytes of an edge
    chunk never participate in the digest.
    """
    return tuple(
        slice(coord * chunk, min((coord + 1) * chunk, dim))
        for coord, dim, chunk in zip(coords, shape, chunks, strict=True)
    )


def _iter_chunk_coords(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> itertools.product[tuple[int, ...]]:
    return itertools.product(
        *(range(-(-dim // chunk)) for dim, chunk in zip(shape, chunks, strict=True))
    )


def _chunk_grid(metadata: ArrayMetadata) -> ChunkGrid:
    """Return the array's chunk grid, requiring a regular grid."""
    chunk_grid = ChunkGrid.from_metadata(metadata)
    if not chunk_grid.is_regular:
        raise NotImplementedError(
            f"integrity manifests require a regular chunk grid, got {type(chunk_grid)!r}"
        )
    return chunk_grid


def _logical_chunks(metadata: ArrayMetadata) -> tuple[int, ...]:
    """Return the shape of the addressable storage chunks (shards, if sharded).

    This is the chunk grid of the array metadata, i.e. the granularity at
    which chunk keys are assigned. Inner chunks of a sharding codec are an
    encoding detail and do not affect the manifest.
    """
    return _chunk_grid(metadata).chunk_shape


def _coords_key(coords: tuple[int, ...]) -> str:
    return ".".join(str(c) for c in coords)


def _parse_coords_key(key: str) -> tuple[int, ...]:
    return tuple(int(part) for part in key.split("."))


def _array_metadata_summary(array: AnyArray) -> dict[str, Any]:
    """Extract the manifest-relevant, JSON-serializable array metadata."""
    metadata = array.metadata
    dtype = _canonical_dtype(array.dtype)
    return {
        "zarr_format": metadata.zarr_format,
        "shape": list(metadata.shape),
        "chunks": list(_logical_chunks(metadata)),
        "dtype": dtype.str,
        "fill_value": metadata.to_dict().get("fill_value"),
    }


def _metadata_issues(manifest_metadata: dict[str, Any], array: AnyArray) -> list[str]:
    """Compare manifest metadata against an array, returning human-readable issues."""
    actual = _array_metadata_summary(array)
    return [
        f"{key}: manifest has {manifest_metadata.get(key)!r}, array has {actual[key]!r}"
        for key in ("zarr_format", "shape", "chunks", "dtype", "fill_value")
        if manifest_metadata.get(key) != actual[key]
    ]


def _chunk_exists(array: AnyArray, metadata: ArrayMetadata, coords: tuple[int, ...]) -> bool:
    """Check whether a chunk (or its containing shard) exists in the store."""
    key = metadata.encode_chunk_key(coords)
    return bool(sync(array.store.exists(key)))


@dataclass(frozen=True)
class ChunkRecord:
    """Manifest entry for one logical chunk."""

    coords: tuple[int, ...]
    digest: str
    implicit_fill: bool
    """True when the chunk had no stored representation and decoded to the
    fill value. For sharded arrays this reflects the containing shard."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "implicit_fill": self.implicit_fill,
        }


@dataclass(frozen=True)
class ChunkVerification:
    """Verification outcome for one logical chunk."""

    coords: tuple[int, ...]
    status: ChunkStatus
    expected_digest: str
    actual_digest: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "coords": list(self.coords),
            "status": self.status,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
        }


@dataclass
class IntegrityManifest:
    """A saveable, content-addressed description of an array's logical data."""

    array_metadata: dict[str, Any]
    chunks: dict[tuple[int, ...], ChunkRecord] = field(default_factory=dict)
    format_version: int = MANIFEST_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "digest_algorithm": "sha256",
            "array": self.array_metadata,
            "chunks": {
                _coords_key(coords): record.to_dict()
                for coords, record in sorted(self.chunks.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrityManifest:
        format_version = data.get("format_version")
        if format_version != MANIFEST_FORMAT_VERSION:
            raise ValueError(
                f"unsupported manifest format version {format_version!r}; "
                f"expected {MANIFEST_FORMAT_VERSION}"
            )
        chunks = {
            _parse_coords_key(key): ChunkRecord(
                coords=_parse_coords_key(key),
                digest=str(record["digest"]),
                implicit_fill=bool(record["implicit_fill"]),
            )
            for key, record in data["chunks"].items()
        }
        return cls(
            array_metadata=dict(data["array"]),
            chunks=chunks,
            format_version=format_version,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> IntegrityManifest:
        return cls.from_dict(json.loads(payload))

    def save(self, path: str | Path) -> None:
        """Write the manifest to ``path`` as JSON."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> IntegrityManifest:
        """Load a manifest previously written with :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class VerificationReport:
    """Result of verifying an array against an :class:`IntegrityManifest`."""

    metadata_compatible: bool
    metadata_issues: list[str] = field(default_factory=list)
    results: dict[tuple[int, ...], ChunkVerification] = field(default_factory=dict)
    selection_only: bool = False

    @property
    def ok(self) -> bool:
        """True when the metadata is compatible and every checked chunk matches."""
        return self.metadata_compatible and all(
            result.status == "ok" for result in self.results.values()
        )

    @property
    def changed(self) -> list[tuple[int, ...]]:
        """Coordinates of chunks whose stored data no longer matches the manifest."""
        return sorted(
            coords for coords, result in self.results.items() if result.status == "changed"
        )

    @property
    def missing(self) -> list[tuple[int, ...]]:
        """Coordinates of chunks that had data in the manifest but are now absent."""
        return sorted(
            coords for coords, result in self.results.items() if result.status == "missing"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_compatible": self.metadata_compatible,
            "metadata_issues": list(self.metadata_issues),
            "selection_only": self.selection_only,
            "ok": self.ok,
            "chunks": {
                _coords_key(coords): result.to_dict()
                for coords, result in sorted(self.results.items())
            },
        }

    def summary(self) -> str:
        if not self.metadata_compatible:
            return "metadata incompatible: " + "; ".join(self.metadata_issues)
        scope = "selection" if self.selection_only else "full array"
        if self.ok:
            return f"ok: {len(self.results)} chunk(s) verified ({scope})"
        return (
            f"failed ({scope}): {len(self.changed)} changed "
            f"{self.changed}, {len(self.missing)} missing {self.missing}"
        )


def compute_manifest(array: AnyArray) -> IntegrityManifest:
    """Compute an integrity manifest for every logical chunk of ``array``.

    Digests are computed from decoded values, so two arrays holding the same
    logical data produce the same manifest even if they use different
    compressors or shard layouts.
    """
    metadata = array.metadata
    shape = tuple(metadata.shape)
    chunks = _logical_chunks(metadata)
    dtype = _canonical_dtype(array.dtype)
    records: dict[tuple[int, ...], ChunkRecord] = {}
    for coords in _iter_chunk_coords(shape, chunks):
        region = _chunk_region(coords, shape, chunks)
        values = np.asarray(array[region])
        region_shape = tuple(dim.stop - dim.start for dim in region)
        digest = _digest_chunk(dtype, region_shape, _canonical_bytes(values, dtype))
        records[coords] = ChunkRecord(
            coords=coords,
            digest=digest,
            implicit_fill=not _chunk_exists(array, metadata, coords),
        )
    return IntegrityManifest(array_metadata=_array_metadata_summary(array), chunks=records)


def _selection_chunk_coords(selection: Any, array: AnyArray) -> set[tuple[int, ...]]:
    """Resolve a basic selection to the set of logical chunk coordinates it touches."""
    metadata = array.metadata
    indexer = BasicIndexer(selection, shape=tuple(metadata.shape), chunk_grid=_chunk_grid(metadata))
    return {projection.chunk_coords for projection in indexer}


def verify_manifest(
    array: AnyArray,
    manifest: IntegrityManifest,
    selection: Any = None,
) -> VerificationReport:
    """Verify ``array`` against ``manifest``.

    Parameters
    ----------
    array
        The array to check.
    manifest
        A manifest previously computed with :func:`compute_manifest`.
    selection
        Optional basic selection (slices/ints, as in ``array[selection]``).
        When given, only the logical chunks overlapping the selection are
        verified; chunks outside the selection are not read.

    Returns
    -------
    VerificationReport
        Per-chunk outcomes distinguishing ``"changed"`` (digest mismatch) and
        ``"missing"`` (chunk absent from the store although the manifest
        recorded data for it), plus metadata compatibility. Verification is
        read-only; no repair is attempted.
    """
    issues = _metadata_issues(manifest.array_metadata, array)
    if issues:
        return VerificationReport(metadata_compatible=False, metadata_issues=issues)

    metadata = array.metadata
    shape = tuple(metadata.shape)
    chunks = _logical_chunks(metadata)
    dtype = _canonical_dtype(array.dtype)

    if selection is None:
        coords_to_check = set(manifest.chunks)
        selection_only = False
    else:
        coords_to_check = _selection_chunk_coords(selection, array) & set(manifest.chunks)
        selection_only = True

    results: dict[tuple[int, ...], ChunkVerification] = {}
    for coords in sorted(coords_to_check):
        record = manifest.chunks[coords]
        if not _chunk_exists(array, metadata, coords):
            if record.implicit_fill:
                results[coords] = ChunkVerification(
                    coords=coords,
                    status="ok",
                    expected_digest=record.digest,
                    actual_digest=None,
                )
            else:
                results[coords] = ChunkVerification(
                    coords=coords,
                    status="missing",
                    expected_digest=record.digest,
                    actual_digest=None,
                )
            continue
        region = _chunk_region(coords, shape, chunks)
        values = np.asarray(array[region])
        region_shape = tuple(dim.stop - dim.start for dim in region)
        actual_digest = _digest_chunk(dtype, region_shape, _canonical_bytes(values, dtype))
        results[coords] = ChunkVerification(
            coords=coords,
            status="ok" if actual_digest == record.digest else "changed",
            expected_digest=record.digest,
            actual_digest=actual_digest,
        )
    return VerificationReport(
        metadata_compatible=True,
        results=results,
        selection_only=selection_only,
    )
