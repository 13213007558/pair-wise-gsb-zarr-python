"""Tests for Array.scatter_reduce / AsyncArray.scatter_reduce."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype
    from zarr.core.buffer.core import Buffer


class CountingDict(MemoryStore):
    counter: Counter[tuple[str, str]]

    @classmethod
    async def open(cls) -> CountingDict:
        store = await super().open()
        store.counter = Counter()
        return store

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__getitem__", key_suffix] += 1
        return await super().get(key, prototype, byte_range)

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__setitem__", key_suffix] += 1
        return await super().set(key, value, byte_range)

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__getitem__", key_suffix] += 1
        return super().get_sync(key, prototype=prototype, byte_range=byte_range)

    def set_sync(self, key: str, value: Buffer) -> None:
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__setitem__", key_suffix] += 1
        return super().set_sync(key, value)

    def io_counts(self) -> tuple[int, int]:
        """Total (reads, writes) of chunk payloads since the counter was reset."""
        reads = sum(v for (op, _), v in self.counter.items() if op == "__getitem__")
        writes = sum(v for (op, _), v in self.counter.items() if op == "__setitem__")
        return reads, writes


def _reference(
    data: npt.NDArray[Any],
    coords: tuple[npt.NDArray[Any], ...],
    values: npt.ArrayLike,
    op: str,
) -> npt.NDArray[Any]:
    """Apply the same batch of updates to a numpy copy with ufunc.at semantics."""
    expected = data.copy()
    ufunc = {"add": np.add, "min": np.minimum, "max": np.maximum}[op]
    ufunc.at(expected, coords, np.asarray(values, dtype=data.dtype))
    return expected


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("op", ["add", "min", "max"])
def test_scatter_reduce_duplicate_coordinates_int(zarr_format: int, op: str) -> None:
    """Duplicate coordinates are merged within the batch, then with the stored values."""
    data = np.arange(10, dtype="i4")
    z = zarr.create_array({}, data=data, chunks=(3,), zarr_format=zarr_format)
    coords = (np.array([0, 1, 1, 4, 4, 4, 9]),)
    values = [10, 1, 2, 5, 3, 1, 7]
    z.scatter_reduce(coords, values, op=op)
    assert_array_equal(z[:], _reference(data, coords, values, op))


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("op", ["add", "min", "max"])
def test_scatter_reduce_duplicate_coordinates_float(zarr_format: int, op: str) -> None:
    """Float dtype: duplicates and existing values combine per the op."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(12,)).astype("f8")
    z = zarr.create_array({}, data=data, chunks=(4,), zarr_format=zarr_format)
    coords = (np.array([3, 3, 0, 11, 11, 11, 5]),)
    values = rng.normal(size=(7,))
    z.scatter_reduce(coords, values, op=op)
    assert_array_equal(z[:], _reference(data, coords, values, op))


@pytest.mark.parametrize("op", ["add", "min", "max"])
def test_scatter_reduce_order_independent(op: str) -> None:
    """Reordering the batch must not change the result (no update is dropped)."""
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 20, size=50)
    values = rng.integers(-5, 5, size=50)
    results = []
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(coords.size)
        z = zarr.full(shape=(20,), fill_value=2, store={}, chunks=(3,), dtype="i8")
        z.scatter_reduce((coords[perm],), values[perm], op=op)
        results.append(z[:])
    assert_array_equal(results[0], results[1])
    assert_array_equal(results[1], results[2])


def test_scatter_reduce_negative_indices_and_unsorted() -> None:
    """Negative indices wrap around and coordinates may be unsorted."""
    data = np.arange(8, dtype="i4")
    z = zarr.create_array({}, data=data, chunks=(2,))
    coords = (np.array([-1, 0, -8, 5, -1]),)
    values = [1, 10, 100, 1000, 2]
    z.scatter_reduce(coords, values, op="add")
    assert_array_equal(z[:], _reference(data, coords, values, "add"))


def test_scatter_reduce_multidim_coordinates() -> None:
    """Multi-dimensional coordinate arrays select points by broadcasting."""
    data = np.arange(16, dtype="i4").reshape(4, 4)
    z = zarr.create_array({}, data=data, chunks=(2, 2))
    coords = (np.array([[0, 1], [3, 3]]), np.array([[0, 1], [3, 2]]))
    values = np.array([[1, 2], [3, 4]])
    z.scatter_reduce(coords, values, op="add")
    assert_array_equal(z[:], _reference(data, coords, values, "add"))


def test_scatter_reduce_scalar_broadcast() -> None:
    """A scalar value is broadcast to every selected point."""
    z = zarr.zeros(shape=(6,), chunks=(2,), dtype="i4")
    z.scatter_reduce(([0, 3, 3, 5],), 2, op="add")
    assert_array_equal(z[:], [2, 0, 0, 4, 0, 2])


def test_scatter_reduce_repeated_calls_accumulate() -> None:
    """Each call merges with the values stored by previous calls."""
    z = zarr.zeros(shape=(4,), chunks=(2,), dtype="i4")
    z.scatter_reduce(([1, 1],), [1, 2], op="add")
    z.scatter_reduce(([1],), [4], op="add")
    assert_array_equal(z[:], [0, 7, 0, 0])


async def test_scatter_reduce_async() -> None:
    """The async API applies the same merge semantics."""
    z = zarr.zeros(shape=(4,), chunks=(2,), dtype="i4")
    await z.async_array.scatter_reduce(([0, 0, 3],), [1, 2, 3], op="add")
    assert_array_equal(z[:], [3, 0, 0, 3])


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_scatter_reduce_out_of_bounds_writes_nothing(zarr_format: int) -> None:
    """An out-of-bounds coordinate rejects the call before any chunk is written."""
    store = await CountingDict.open()
    z = zarr.create_array(store=store, shape=(6,), chunks=(2,), dtype="i4", zarr_format=zarr_format)
    z[:] = np.arange(6)
    store.counter.clear()
    with pytest.raises(IndexError):
        z.scatter_reduce(([0, 6],), [1, 1], op="add")
    reads, writes = store.io_counts()
    assert writes == 0
    assert reads == 0
    assert_array_equal(z[:], np.arange(6))


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        ([1.5, 2.5], "i4"),  # float to int
        ([1, 2**40], "i4"),  # int out of range for the dtype
        ([1 + 2j, 3], "f8"),  # complex to real
        (["a", "b"], "i4"),  # non-numeric
    ],
)
async def test_scatter_reduce_unsafe_conversion_writes_nothing(
    value: list[Any], dtype: str
) -> None:
    """Values that cannot be safely converted reject the call before any write."""
    store = await CountingDict.open()
    z = zarr.create_array(store=store, shape=(4,), chunks=(2,), dtype=dtype)
    z[:] = np.arange(4).astype(dtype)
    store.counter.clear()
    with pytest.raises(TypeError):
        z.scatter_reduce(([0, 1],), value, op="add")
    assert_array_equal(z[:], np.arange(4).astype(dtype))
    _, writes = store.io_counts()
    assert writes == 0


def test_scatter_reduce_value_shape_mismatch() -> None:
    z = zarr.zeros(shape=(4,), chunks=(2,), dtype="i4")
    with pytest.raises(ValueError, match="scatter"):
        z.scatter_reduce(([0, 1, 2],), [1, 2], op="add")
    assert_array_equal(z[:], np.zeros(4, dtype="i4"))


def test_scatter_reduce_invalid_op() -> None:
    z = zarr.zeros(shape=(4,), chunks=(2,), dtype="i4")
    with pytest.raises(ValueError, match="invalid scatter op"):
        z.scatter_reduce(([0],), [1], op="mul")  # type: ignore[arg-type]


@pytest.mark.parametrize("op", ["min", "max"])
def test_scatter_reduce_min_max_complex_raises(op: str) -> None:
    z = zarr.zeros(shape=(4,), chunks=(2,), dtype="c16")
    with pytest.raises(TypeError, match="complex"):
        z.scatter_reduce(([0],), [1 + 1j], op=op)


async def test_scatter_reduce_chunk_io_counts() -> None:
    """Each touched chunk is read and written exactly once, no matter how many
    points fall into it."""
    store = await CountingDict.open()
    z = zarr.create_array(store=store, shape=(10,), chunks=(5,), dtype="i4")
    z[:] = 0
    store.counter.clear()

    # many points, all within a single chunk
    z.scatter_reduce(([0, 1, 1, 2, 3, 3, 3, 4],), [1] * 8, op="add")
    assert store.io_counts() == (1, 1)

    # points spread over two chunks -> two reads and two writes
    store.counter.clear()
    z.scatter_reduce(([0, 9, 9],), [1, 2, 3], op="add")
    assert store.io_counts() == (2, 2)


async def test_scatter_reduce_sharded_single_shard_rewrite() -> None:
    """Points in different inner chunks of the same shard trigger one shard
    read and one shard write per call."""
    store = await CountingDict.open()
    z = zarr.create_array(store=store, shape=(8,), chunks=(2,), shards=(4,), dtype="i4")
    z[:] = 0
    store.counter.clear()

    # points 0 and 2 live in different inner chunks of shard 0
    z.scatter_reduce(([0, 2, 2],), [1, 2, 3], op="add")
    assert store.io_counts() == (1, 1)
    assert_array_equal(z[:], [1, 0, 5, 0, 0, 0, 0, 0])

    # touching both shards -> one read+write each
    store.counter.clear()
    z.scatter_reduce(([0, 7],), [1, 1], op="add")
    assert store.io_counts() == (2, 2)


async def test_scatter_reduce_noop_writes_nothing() -> None:
    """A merge that leaves a chunk at its fill value stores no chunk."""
    store = await CountingDict.open()
    z = zarr.create_array(store=store, shape=(4,), chunks=(2,), dtype="i4", fill_value=0)
    store.counter.clear()
    z.scatter_reduce(([0, 1],), [0, 0], op="add")
    _, writes = store.io_counts()
    assert writes == 0
    assert_array_equal(z[:], np.zeros(4, dtype="i4"))


def test_scatter_reduce_matches_numpy_reference() -> None:
    """Random batches with duplicates across chunk boundaries match ufunc.at."""
    rng = np.random.default_rng(1234)
    data = rng.integers(-50, 50, size=(9, 7)).astype("i8")
    coords = (rng.integers(0, 9, size=40), rng.integers(0, 7, size=40))
    values = rng.integers(-10, 10, size=40)
    for op in ["add", "min", "max"]:
        zz = zarr.create_array({}, data=data, chunks=(4, 3))
        zz.scatter_reduce(coords, values, op=op)
        assert_array_equal(zz[:], _reference(data, coords, values, op))


def test_vindex_assignment_still_last_write_wins() -> None:
    """Plain vindex assignment keeps its existing behavior for duplicates."""
    z = zarr.zeros(shape=(4,), chunks=(2,), dtype="i4")
    z.vindex[[0, 0, 1]] = [1, 2, 3]
    assert_array_equal(z[:], [2, 3, 0, 0])
