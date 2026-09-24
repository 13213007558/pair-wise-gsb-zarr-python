from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr import Array
from zarr.core.window import axis_runs, parse_window_modes, reflect_index
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr.core.buffer import Buffer, BufferPrototype


def _pad_and_slice(
    data: np.ndarray,
    origin: Sequence[int],
    shape: Sequence[int],
    modes: str | Sequence[str],
) -> np.ndarray:
    """Reference implementation: np.pad each axis, then slice the window out."""
    if isinstance(modes, str):
        modes = (modes,) * data.ndim
    result = data
    for ax, mode in enumerate(modes):
        pad = [(0, 0)] * data.ndim
        pad[ax] = (
            max(0, -origin[ax]),
            max(0, origin[ax] + shape[ax] - data.shape[ax]),
        )
        result = np.pad(result, pad, mode=mode)
    sel = tuple(
        slice(o + max(0, -o), o + max(0, -o) + s) for o, s in zip(origin, shape, strict=True)
    )
    return result[sel]


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("mode", ["constant", "reflect", "wrap"])
@pytest.mark.parametrize(
    ("origin", "shape"),
    [
        ((0, 0), (4, 5)),  # exact array
        ((1, 1), (2, 2)),  # fully inside
        ((-2, -3), (7, 9)),  # top-left corner window
        ((2, 3), (5, 6)),  # bottom-right corner window
        ((-1, 3), (4, 5)),  # mixed
        ((10, 10), (3, 3)),  # fully outside
        ((-3, 0), (3, 5)),  # only left overhang
    ],
)
def test_window_matches_numpy_pad(
    zarr_format: int, mode: str, origin: tuple[int, int], shape: tuple[int, int]
) -> None:
    data = np.arange(20, dtype="i4").reshape(4, 5)
    z = zarr.create_array(
        {},
        shape=data.shape,
        chunks=(3, 3),
        dtype=data.dtype,
        zarr_format=zarr_format,
    )
    z[:] = data
    got = z.get_window(origin, shape, mode=mode)
    expected = _pad_and_slice(data, origin, shape, mode)
    assert got.shape == shape
    assert got.dtype == data.dtype
    assert_array_equal(got, expected)


@pytest.mark.parametrize(
    "modes",
    [
        ("constant", "reflect"),
        ("reflect", "wrap"),
        ("wrap", "constant"),
    ],
)
def test_window_per_axis_modes(modes: tuple[str, str]) -> None:
    data = np.arange(20, dtype="i4").reshape(4, 5)
    z = zarr.create_array({}, shape=data.shape, chunks=(2, 2), dtype=data.dtype)
    z[:] = data
    origin = (-2, -3)
    shape = (8, 10)
    got = z.get_window(origin, shape, mode=modes)
    expected = _pad_and_slice(data, origin, shape, modes)
    assert got.shape == shape
    assert_array_equal(got, expected)


@pytest.mark.parametrize("mode", ["reflect", "wrap"])
def test_window_halo_larger_than_axis(mode: str) -> None:
    # halo exceeds several multiples of the axis length on both sides
    data = np.arange(3, dtype="i4")
    z = zarr.create_array({}, shape=data.shape, chunks=(2,), dtype=data.dtype)
    z[:] = data
    origin = (-7,)
    shape = (17,)
    got = z.get_window(origin, shape, mode=mode)
    expected = _pad_and_slice(data, origin, shape, mode)
    assert got.shape == shape
    assert_array_equal(got, expected)


@pytest.mark.parametrize("mode", ["reflect", "wrap"])
def test_window_axis_length_one(mode: str) -> None:
    data = np.array([[1, 2, 3]], dtype="i4")  # shape (1, 3)
    z = zarr.create_array({}, shape=data.shape, chunks=(1, 2), dtype=data.dtype)
    z[:] = data
    origin = (-4, -1)
    shape = (9, 5)
    got = z.get_window(origin, shape, mode=mode)
    expected = _pad_and_slice(data, origin, shape, mode)
    assert got.shape == shape
    assert_array_equal(got, expected)


def test_window_empty_axis_constant() -> None:
    z = zarr.create_array({}, shape=(0, 3), chunks=(2, 2), dtype="i4", fill_value=9)
    got = z.get_window((-2, 0), (4, 3), mode="constant")
    assert got.shape == (4, 3)
    assert_array_equal(got, np.full((4, 3), 9, dtype="i4"))


@pytest.mark.parametrize("mode", ["reflect", "wrap"])
def test_window_empty_axis_raises(mode: str) -> None:
    z = zarr.create_array({}, shape=(0, 3), chunks=(2, 2), dtype="i4")
    with pytest.raises(ValueError, match="length 0"):
        z.get_window((-2, 0), (4, 3), mode=mode)


@pytest.mark.parametrize("mode", ["constant", "reflect", "wrap"])
def test_window_zero_length_window(mode: str) -> None:
    # a zero-length window needs no data, even on an empty axis
    z = zarr.create_array({}, shape=(0, 3), chunks=(2, 2), dtype="i4")
    got = z.get_window((0, 0), (0, 2), mode=mode)
    assert got.shape == (0, 2)
    assert got.dtype == np.dtype("i4")


def test_window_is_read_only_and_shape_preserved() -> None:
    data = np.arange(6, dtype="i4")
    z = zarr.create_array({}, shape=data.shape, chunks=(2,), dtype=data.dtype, fill_value=0)
    z[:] = data
    got = z.get_window((100,), (3,))
    assert got.shape == (3,)
    assert_array_equal(got, np.zeros(3, dtype="i4"))
    assert z.shape == data.shape
    assert_array_equal(z[:], data)


def test_window_default_fill_value() -> None:
    z = zarr.create_array({}, shape=(3,), chunks=(2,), dtype="f4", fill_value=np.nan)
    z[:] = np.arange(3, dtype="f4")
    got = z.get_window((-1,), (5,))
    assert got.dtype == np.dtype("f4")
    assert np.isnan(got[0])
    assert np.isnan(got[-1])
    assert_array_equal(got[1:4], np.arange(3, dtype="f4"))


def test_window_constant_value_cast_to_dtype() -> None:
    z = zarr.create_array({}, shape=(3,), chunks=(2,), dtype="f4", fill_value=0)
    z[:] = np.arange(3, dtype="f4")
    got = z.get_window((-2,), (6,), constant_value=1.5)
    assert got.dtype == np.dtype("f4")
    assert_array_equal(got[:2], np.array([1.5, 1.5], dtype="f4"))

    zi = zarr.create_array({}, shape=(3,), chunks=(2,), dtype="i4", fill_value=0)
    zi[:] = np.arange(3, dtype="i4")
    got_i = zi.get_window((-1,), (5,), constant_value=7)
    assert got_i.dtype == np.dtype("i4")
    assert got_i[0] == np.int32(7)


class CountingStore(MemoryStore):
    counter: Counter[str]

    @classmethod
    async def open(cls) -> CountingStore:
        store = await super().open()
        store.counter = Counter()
        return store

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: Any = None,
    ) -> Buffer | None:
        self.counter[key] += 1
        return await super().get(key, prototype, byte_range)


async def _counting_array(
    shape: tuple[int, ...], chunks: tuple[int, ...], data: np.ndarray
) -> tuple[Array, CountingStore]:
    store = await CountingStore.open()
    z = zarr.create_array(store, shape=shape, chunks=chunks, dtype=data.dtype)
    z[:] = data
    store.counter.clear()
    return z, store


async def test_window_reads_only_needed_chunks() -> None:
    data = np.arange(36, dtype="i4").reshape(6, 6)
    z, store = await _counting_array((6, 6), (3, 3), data)

    # window fully inside chunk (1, 1)
    z.get_window((3, 3), (3, 3))
    assert {k for k in store.counter if k.startswith("c/")} == {"c/1/1"}

    store.counter.clear()
    # window overlapping 4 chunks, with constant overhang
    z.get_window((2, 2), (5, 5), mode="constant")
    assert {k for k in store.counter if k.startswith("c/")} == {
        "c/0/0",
        "c/0/1",
        "c/1/0",
        "c/1/1",
    }

    store.counter.clear()
    # fully out-of-bounds constant window reads no chunks at all
    got = z.get_window((10, 10), (3, 3), mode="constant")
    assert got.shape == (3, 3)
    assert not [k for k in store.counter if k.startswith("c/")]


async def test_window_reuses_repeated_source_regions() -> None:
    # wrap window spanning several periods of a single-chunk axis:
    # the chunk must be fetched exactly once
    data = np.arange(4, dtype="i4")
    z, store = await _counting_array((4,), (4,), data)
    got = z.get_window((-4,), (16,), mode="wrap")
    assert_array_equal(got, np.tile(data, 4))
    gets = [k for k in store.counter if k.startswith("c/")]
    assert gets == ["c/0"]

    # reflect window: forward and backward runs share the same source region
    store.counter.clear()
    got = z.get_window((0,), (12,), mode="reflect")
    expected = _pad_and_slice(data, (0,), (12,), "reflect")
    assert_array_equal(got, expected)
    gets = [k for k in store.counter if k.startswith("c/")]
    assert gets == ["c/0"]


async def test_window_async() -> None:
    data = np.arange(6, dtype="i4")
    z = zarr.create_array({}, shape=data.shape, chunks=(2,), dtype=data.dtype)
    z[:] = data
    got = await z.async_array.get_window((-2,), (8,), mode="wrap")
    assert_array_equal(got, _pad_and_slice(data, (-2,), (8,), "wrap"))


def test_window_invalid_mode() -> None:
    z = zarr.create_array({}, shape=(3, 3), chunks=(2, 2), dtype="i4")
    with pytest.raises(ValueError, match="Invalid window mode"):
        z.get_window((0, 0), (2, 2), mode="edge")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one per array dimension"):
        z.get_window((0, 0), (2, 2), mode=("constant",))


def test_window_invalid_origin_and_shape() -> None:
    z = zarr.create_array({}, shape=(3, 3), chunks=(2, 2), dtype="i4")
    with pytest.raises(ValueError, match="origin"):
        z.get_window((0,), (2, 2))
    with pytest.raises(ValueError, match="shape"):
        z.get_window((0, 0), (2,))
    with pytest.raises(ValueError, match=">= 0"):
        z.get_window((0, 0), (2, -1))
    with pytest.raises(TypeError, match="integers"):
        z.get_window((0.5, 0), (2, 2))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integers"):
        z.get_window((0, 0), (2, 1.5))  # type: ignore[arg-type]


def test_reflect_index() -> None:
    # numpy reflect semantics: edge element is not repeated
    assert [reflect_index(i, 4) for i in range(-4, 8)] == [2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]


def test_axis_runs_constant() -> None:
    runs = axis_runs(-2, 6, 3, "constant", 0)
    assert [(r.out_start, r.length, r.src_start, r.src_step) for r in runs] == [
        (0, 2, None, 1),
        (2, 3, 0, 1),
        (5, 1, None, 1),
    ]


def test_axis_runs_wrap_repeats_full_period() -> None:
    runs = axis_runs(0, 8, 4, "wrap", 0)
    # two full periods -> two identical source runs
    assert [(r.out_start, r.length, r.src_start) for r in runs] == [(0, 4, 0), (4, 4, 0)]


def test_parse_window_modes() -> None:
    assert parse_window_modes("wrap", 2) == ("wrap", "wrap")
    assert parse_window_modes(("constant", "reflect"), 2) == ("constant", "reflect")
    with pytest.raises(ValueError, match="Invalid window mode"):
        parse_window_modes("edge", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one per array dimension"):
        parse_window_modes(("wrap",), 2)
