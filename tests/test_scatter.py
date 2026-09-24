from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype
    from zarr.core.buffer.core import Buffer


class CountingStore(MemoryStore):
    """MemoryStore that tallies get/set calls per key."""

    def __init__(self) -> None:
        super().__init__()
        self.counter: Counter[tuple[str, str]] = Counter()

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        self.counter["get", key] += 1
        return await super().get(key, prototype, byte_range)

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        self.counter["set", key] += 1
        return await super().set(key, value, byte_range)

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        self.counter["get", key] += 1
        return super().get_sync(key, prototype=prototype, byte_range=byte_range)

    def set_sync(self, key: str, value: Buffer) -> None:
        self.counter["set", key] += 1
        return super().set_sync(key, value)

    def op_count(self, op: str) -> int:
        return sum(count for (counted_op, _), count in self.counter.items() if counted_op == op)


def _scatter_reference(
    a: np.ndarray[Any, Any],
    selection: tuple[Any, ...],
    values: Any,
    op: str,
) -> np.ndarray[Any, Any]:
    """Reference scatter result computed with the equivalent NumPy ufunc.at."""
    ufunc = {"add": np.add, "min": np.minimum, "max": np.maximum}[op]
    expected = a.copy()
    ufunc.at(expected, selection, values)
    return expected


@pytest.mark.parametrize("op", ["add", "min", "max"])
@pytest.mark.parametrize("dtype", ["i4", "f8"])
def test_scatter_repeated_coordinates(op: str, dtype: str) -> None:
    """Repeated coordinates are merged within the batch, then with existing data."""
    a = np.arange(12, dtype=dtype).reshape(3, 4)
    z = zarr.create_array(store={}, shape=a.shape, dtype=a.dtype, chunks=(2, 2))
    z[...] = a

    rows = np.array([0, 2, 0, 1, 2, 0])
    cols = np.array([1, 3, 1, 0, 3, 2])
    values = np.array([1, 5, 2, 7, 3, 4], dtype=dtype)
    z.scatter((rows, cols), values, op=op)

    assert_array_equal(z[...], _scatter_reference(a, (rows, cols), values, op))


@pytest.mark.parametrize("op", ["add", "min", "max"])
def test_scatter_grouping_order_does_not_matter(op: str) -> None:
    """The same updates in a different order must give the same result."""
    coords = [3, 0, 3, 1, 0, 3]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shuffled = [5, 0, 3, 1, 4, 2]

    results = []
    for sel_coords, sel_values in [
        (coords, values),
        ([coords[i] for i in shuffled], [values[i] for i in shuffled]),
    ]:
        z = zarr.create_array(store={}, shape=(4,), dtype="f8", chunks=(2,))
        z[...] = [10.0, 20.0, 30.0, 40.0]
        z.scatter((sel_coords,), sel_values, op=op)
        results.append(z[...])
    assert_array_equal(results[0], results[1])


def test_scatter_negative_and_unordered_indices() -> None:
    """Negative indices wrap around and unordered coordinates are handled."""
    a = np.arange(6, dtype="i4")
    z = zarr.create_array(store={}, shape=a.shape, dtype=a.dtype, chunks=(2,))
    z[...] = a

    coords = [-1, 0, -6, 3]
    values = [10, 20, 30, 40]
    z.scatter((coords,), values, op="add")

    assert_array_equal(z[...], _scatter_reference(a, (coords,), values, "add"))


def test_scatter_multidimensional_coordinates() -> None:
    """Multi-dimensional coordinate arrays broadcast like NumPy fancy indexing."""
    a = np.arange(9, dtype="f8").reshape(3, 3)
    z = zarr.create_array(store={}, shape=a.shape, dtype=a.dtype, chunks=(2, 2))
    z[...] = a

    rows = np.array([[0, 2], [1, 2]])
    cols = np.array([[0, 1], [2, 2]])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    z.scatter((rows, cols), values, op="add")

    assert_array_equal(z[...], _scatter_reference(a, (rows, cols), values, "add"))


@pytest.mark.parametrize("op", ["add", "min", "max"])
def test_scatter_scalar_broadcast(op: str) -> None:
    """A scalar value is broadcast to all coordinates."""
    a = np.arange(5, dtype="f8")
    z = zarr.create_array(store={}, shape=a.shape, dtype=a.dtype, chunks=(2,))
    z[...] = a

    z.scatter(([0, 2, 4],), 10.0, op=op)

    assert_array_equal(z[...], _scatter_reference(a, ([0, 2, 4],), 10.0, op))


def test_scatter_vindex_add_min_max() -> None:
    """The vindex add/min/max helpers match scatter with the same op."""
    z = zarr.create_array(store={}, shape=(5,), dtype="i4", chunks=(2,))
    z[...] = [1, 2, 3, 4, 5]
    z.vindex.add([0, 0, 4], [1, 2, 10])
    z.vindex.min([1, 4], [0, 1])
    z.vindex.max([2], 100)
    assert_array_equal(z[...], [4, 0, 100, 4, 1])


def test_scatter_fill_value_for_missing_chunks() -> None:
    """Chunks that were never written combine updates with the fill value."""
    z = zarr.create_array(store={}, shape=(4,), dtype="i4", chunks=(2,), fill_value=5)
    z.scatter(([1, 1, 3],), [1, 2, 10], op="add")
    assert_array_equal(z[...], [5, 8, 5, 15])


def test_scatter_sharded() -> None:
    """Scatter works on sharded arrays, combining points within each shard."""
    a = np.arange(8, dtype="i4")
    z = zarr.create_array(store={}, shape=a.shape, dtype=a.dtype, chunks=(2,), shards=(4,))
    z[...] = a

    coords = [0, 1, 1, 7, 6]
    values = [1, 2, 3, 10, 20]
    z.scatter((coords,), values, op="add")

    assert_array_equal(z[...], _scatter_reference(a, (coords,), values, "add"))


def test_scatter_zarr_format_2() -> None:
    """Scatter works on zarr format 2 arrays."""
    a = np.arange(6, dtype="i4")
    z = zarr.create_array(store={}, shape=a.shape, dtype=a.dtype, chunks=(2,), zarr_format=2)
    z[...] = a
    z.scatter(([0, 0, 5],), [1, 2, 10], op="add")
    assert_array_equal(z[...], _scatter_reference(a, ([0, 0, 5],), [1, 2, 10], "add"))


async def test_scatter_async() -> None:
    """The asynchronous scatter variant matches the synchronous one."""
    a = np.arange(6, dtype="f8")
    z = zarr.create_array(store={}, shape=a.shape, dtype=a.dtype, chunks=(2,))
    z[...] = a
    await z.async_array.scatter(([0, 0, 5],), [1.0, 2.0, 10.0], op="add")
    assert_array_equal(z[...], _scatter_reference(a, ([0, 0, 5],), [1.0, 2.0, 10.0], "add"))


def test_scatter_invalid_op() -> None:
    z = zarr.create_array(store={}, shape=(4,), dtype="i4", chunks=(2,))
    with pytest.raises(ValueError, match="invalid scatter operation"):
        z.scatter(([0],), [1], op="mul")  # type: ignore[arg-type]


def test_scatter_out_of_bounds_writes_nothing() -> None:
    """An out-of-bounds coordinate raises before any chunk is written."""
    store = CountingStore()
    z = zarr.create_array(store=store, shape=(4,), dtype="i4", chunks=(2,))
    z[...] = [1, 2, 3, 4]
    before = z[...].copy()
    store.counter.clear()

    with pytest.raises(IndexError):
        z.scatter(([0, 4],), [10, 10], op="add")
    with pytest.raises(IndexError):
        z.scatter(([-5],), [10], op="add")

    assert_array_equal(z[...], before)
    assert store.op_count("set") == 0


def test_scatter_unsafe_dtype_writes_nothing() -> None:
    """Values that cannot be cast safely raise before any chunk is written."""
    store = CountingStore()
    z = zarr.create_array(store=store, shape=(4,), dtype="i4", chunks=(2,))
    z[...] = [1, 2, 3, 4]
    before = z[...].copy()
    store.counter.clear()

    # float values cannot be combined into an integer array
    with pytest.raises(TypeError, match="cannot apply scatter operation"):
        z.scatter(([0, 2],), [0.5, 1.5], op="add")

    assert_array_equal(z[...], before)
    assert store.op_count("set") == 0


def test_scatter_bad_value_shape() -> None:
    z = zarr.create_array(store={}, shape=(4,), dtype="i4", chunks=(2,))
    with pytest.raises(ValueError, match="could not broadcast"):
        z.scatter(([0, 1, 2],), [1, 2], op="add")


def test_scatter_fields_not_supported() -> None:
    z = zarr.create_array(store={}, shape=(4,), dtype="i4", chunks=(2,))
    with pytest.raises(IndexError, match="fields"):
        z.scatter(("field", [0]), [1], op="add")


def test_scatter_store_read_write_counts() -> None:
    """Each touched chunk is read once and written once, even with many points."""
    store = CountingStore()
    z = zarr.create_array(store=store, shape=(8,), dtype="i4", chunks=(2,))
    z[...] = np.arange(8)
    store.counter.clear()

    # points hit chunks 0, 1 and 3; chunk 2 is untouched
    z.scatter(([0, 1, 1, 3, 6, 7, 7],), [1, 2, 3, 4, 5, 6, 7], op="add")

    assert store.op_count("get") == 3
    assert store.op_count("set") == 3


def test_scatter_store_read_write_counts_sharded() -> None:
    """With shards, many points in one shard still cost one read and one write."""
    store = CountingStore()
    z = zarr.create_array(store=store, shape=(8,), dtype="i4", chunks=(2,), shards=(4,))
    z[...] = np.arange(8)
    store.counter.clear()

    # points hit inner chunks 0 and 1 of shard 0, and inner chunk 3 of shard 1
    z.scatter(([0, 1, 1, 3, 6],), [1, 2, 3, 4, 5], op="add")

    assert store.op_count("get") == 2
    assert store.op_count("set") == 2
