"""Tests for windowed reads (``Array.get_window`` / ``AsyncArray.get_window``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.core.window import normalize_window, normalize_window_modes, plan_window_axis
from zarr.storage import MemoryStore, WrapperStore

if TYPE_CHECKING:
    from zarr.abc.buffer import Buffer
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype


class RecordingStore(WrapperStore[MemoryStore]):
    """A store that records the keys of all read requests."""

    def __init__(self, store: MemoryStore) -> None:
        super().__init__(store)
        self.get_keys: list[str] = []

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        self.get_keys.append(key)
        return await super().get(key, prototype=prototype, byte_range=byte_range)


def make_array(
    data: npt.NDArray[Any],
    chunks: tuple[int, ...],
    *,
    fill_value: Any = 0,
    store: Any = None,
) -> zarr.Array:
    if store is None:
        store = {}
    z = zarr.create_array(
        store,
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        fill_value=fill_value,
    )
    z[:] = data
    return z


def pad_reference(
    data: npt.NDArray[Any],
    window: tuple[slice, ...],
    mode: str,
    fill_value: Any = 0,
) -> npt.NDArray[Any]:
    """Reference implementation: pad with NumPy, then slice the requested window."""
    left = [max(-s.start, 0) if s.start is not None else 0 for s in window]
    right = []
    for s, n in zip(window, data.shape, strict=True):
        stop = n if s.stop is None else s.stop
        right.append(max(stop - n, 0))
    pad_width = list(zip(left, right, strict=True))
    if mode == "constant":
        padded = np.pad(data, pad_width, mode="constant", constant_values=fill_value)
    else:
        padded = np.pad(data, pad_width, mode=mode)
    slices = []
    for i, s in enumerate(window):
        start = 0 if s.start is None else s.start
        stop = data.shape[i] if s.stop is None else s.stop
        slices.append(slice(start + left[i], stop + left[i]))
    return padded[tuple(slices)]


def test_normalize_window_rejects_non_unit_stride() -> None:
    with pytest.raises(ValueError, match="unit-stride"):
        normalize_window((slice(0, 10, 2),), 1)
    with pytest.raises(ValueError, match="unit-stride"):
        normalize_window((slice(10, 0, -1),), 1)


def test_normalize_window_rejects_fancy_indexing() -> None:
    with pytest.raises(TypeError, match="slices"):
        normalize_window((1,), 1)
    with pytest.raises(TypeError, match="slices"):
        normalize_window((np.array([0, 1]),), 1)
    with pytest.raises(TypeError, match="slice"):
        normalize_window(1, 1)  # type: ignore[arg-type]


def test_normalize_window_too_many_axes() -> None:
    with pytest.raises(IndexError, match="too many indices"):
        normalize_window((slice(None), slice(None)), 1)


def test_normalize_window_pads_missing_axes() -> None:
    assert normalize_window((slice(1, 2),), 3) == (slice(1, 2), slice(None), slice(None))


def test_normalize_window_modes() -> None:
    assert normalize_window_modes("wrap", 2) == ("wrap", "wrap")
    assert normalize_window_modes(["reflect", "constant"], 2) == ("reflect", "constant")
    with pytest.raises(ValueError, match="invalid extension mode"):
        normalize_window_modes("edge", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one per axis"):
        normalize_window_modes(("wrap", "wrap"), 3)


def test_plan_window_axis_rejects_inverted_window() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        plan_window_axis(slice(5, 2), 10, "constant")


@pytest.mark.parametrize("mode", ["constant", "reflect", "wrap"])
def test_window_1d_matches_numpy_pad(mode: str) -> None:
    data = np.arange(7, dtype="int64")
    z = make_array(data, chunks=(3,))
    for start, stop in [(-9, 0), (-9, 3), (-2, 5), (0, 7), (3, 12), (7, 20), (-9, 20)]:
        window = (slice(start, stop),)
        expected = pad_reference(data, window, mode)
        result = z.get_window(window, mode=mode)
        assert result.shape == (stop - start,)
        assert_array_equal(result, expected)


@pytest.mark.parametrize("mode", ["constant", "reflect", "wrap"])
@pytest.mark.parametrize(
    "corner",
    [
        ((-3, 2), (-4, 3)),  # top-left
        ((-3, 2), (4, 12)),  # top-right
        ((5, 13), (-4, 3)),  # bottom-left
        ((5, 13), (4, 12)),  # bottom-right
    ],
)
def test_window_2d_corners_match_numpy_pad(
    mode: str, corner: tuple[tuple[int, int], tuple[int, int]]
) -> None:
    """2D corner windows compare elementwise against slicing a NumPy-padded array."""
    data = np.arange(8 * 9, dtype="int64").reshape(8, 9)
    z = make_array(data, chunks=(3, 4))
    window = tuple(slice(*c) for c in corner)
    expected = pad_reference(data, window, mode)
    result = z.get_window(window, mode=mode)
    assert result.shape == expected.shape
    assert_array_equal(result, expected)


def test_window_per_axis_modes() -> None:
    data = np.arange(4 * 5, dtype="int64").reshape(4, 5)
    z = make_array(data, chunks=(2, 2))
    window = (slice(-2, 5), slice(-1, 7))
    result = z.get_window(window, mode=("reflect", "wrap"))
    # reference: reflect rows, then wrap columns
    rows = np.pad(data, ((2, 1), (0, 0)), mode="reflect")
    expected = np.pad(rows, ((0, 0), (1, 2)), mode="wrap")
    assert_array_equal(result, expected)


def test_window_inside_bounds_matches_getitem() -> None:
    data = np.arange(20, dtype="int64").reshape(4, 5)
    z = make_array(data, chunks=(2, 2))
    window = (slice(1, 3), slice(2, 5))
    assert_array_equal(z.get_window(window), data[1:3, 2:5])
    assert_array_equal(z.get_window(window, mode="reflect"), data[1:3, 2:5])
    assert_array_equal(z.get_window(window, mode="wrap"), data[1:3, 2:5])


def test_window_result_coordinates_follow_request() -> None:
    """The result shape always matches the requested window, never a clipped one."""
    data = np.arange(5, dtype="int64")
    z = make_array(data, chunks=(2,))
    result = z.get_window((slice(-3, 9),))
    assert result.shape == (12,)
    # interior values land at their window coordinates
    assert_array_equal(result[3:8], data)


def test_window_does_not_modify_array() -> None:
    data = np.arange(6, dtype="int64")
    z = make_array(data, chunks=(2,))
    z.get_window((slice(-10, 10),), mode="wrap")
    assert z.shape == data.shape
    assert_array_equal(z[:], data)


def test_window_empty() -> None:
    data = np.arange(6, dtype="int64")
    z = make_array(data, chunks=(2,))
    for mode in ("constant", "reflect", "wrap"):
        result = z.get_window((slice(2, 2),), mode=mode)
        assert result.shape == (0,)
        assert result.dtype == data.dtype


def test_window_axis_length_zero() -> None:
    z = zarr.create_array({}, shape=(0,), chunks=(4,), dtype="int64", fill_value=5)
    result = z.get_window((slice(-2, 3),), mode="constant")
    assert_array_equal(result, np.full(5, 5, dtype="int64"))
    for mode in ("reflect", "wrap"):
        with pytest.raises(ValueError, match="empty axis"):
            z.get_window((slice(-2, 3),), mode=mode)
    # an empty window on an empty axis is fine
    assert z.get_window((slice(0, 0),), mode="wrap").shape == (0,)


def test_window_axis_length_one() -> None:
    data = np.array([42], dtype="int64")
    z = make_array(data, chunks=(1,))
    for mode in ("reflect", "wrap"):
        result = z.get_window((slice(-3, 4),), mode=mode)
        assert_array_equal(result, np.full(7, 42, dtype="int64"))
        assert_array_equal(result, pad_reference(data, (slice(-3, 4),), mode))


def test_window_halo_larger_than_axis() -> None:
    """Halos spanning several multiples of the axis length are handled."""
    data = np.arange(3, dtype="int64")
    z = make_array(data, chunks=(2,))
    window = (slice(-10, 11),)
    for mode in ("reflect", "wrap"):
        result = z.get_window(window, mode=mode)
        assert_array_equal(result, pad_reference(data, window, mode))


def test_window_fill_value_default_and_dtype() -> None:
    data = np.arange(4, dtype="uint8")
    z = make_array(data, chunks=(2,), fill_value=7)
    result = z.get_window((slice(-2, 6),))
    assert result.dtype == np.dtype("uint8")
    assert_array_equal(result, [7, 7, 0, 1, 2, 3, 7, 7])

    # an explicit fill value is cast to the array dtype
    result = z.get_window((slice(-2, 6),), fill_value=200)
    assert result.dtype == np.dtype("uint8")
    assert result[0] == np.array(200, dtype="uint8")

    # float arrays keep their dtype
    zf = zarr.create_array({}, shape=(3,), chunks=(2,), dtype="float32", fill_value=np.nan)
    zf[:] = [1.0, 2.0, 3.0]
    result = zf.get_window((slice(-1, 4),))
    assert result.dtype == np.dtype("float32")
    assert np.isnan(result[0])
    assert np.isnan(result[-1])


def test_window_only_reads_needed_chunks() -> None:
    data = np.arange(10, dtype="int64")
    store = RecordingStore(MemoryStore())
    z = make_array(data, chunks=(2,), store=store)

    store.get_keys.clear()
    z.get_window((slice(2, 6),))
    assert sorted(store.get_keys) == ["c/1", "c/2"]

    # a constant halo does not trigger extra chunk reads
    store.get_keys.clear()
    z.get_window((slice(-3, 4),), mode="constant")
    assert sorted(store.get_keys) == ["c/0", "c/1"]


def test_window_reuses_repeated_source_chunks() -> None:
    """Wrap/reflect windows read each needed chunk at most once."""
    data = np.arange(10, dtype="int64")
    store = RecordingStore(MemoryStore())
    z = make_array(data, chunks=(3,), store=store)

    store.get_keys.clear()
    result = z.get_window((slice(-11, 21),), mode="wrap")
    assert_array_equal(result, pad_reference(data, (slice(-11, 21),), "wrap"))
    # every chunk is needed, but none is fetched twice
    assert sorted(store.get_keys) == ["c/0", "c/1", "c/2", "c/3"]

    store.get_keys.clear()
    result = z.get_window((slice(-11, 21),), mode="reflect")
    assert_array_equal(result, pad_reference(data, (slice(-11, 21),), "reflect"))
    assert sorted(store.get_keys) == ["c/0", "c/1", "c/2", "c/3"]


async def test_async_get_window() -> None:
    data = np.arange(6, dtype="int64")
    z = zarr.create_array({}, shape=data.shape, chunks=(2,), dtype=data.dtype)
    z[:] = data
    result = await z.async_array.get_window((slice(-2, 8),), mode="reflect")
    assert_array_equal(result, pad_reference(data, (slice(-2, 8),), "reflect"))


def test_window_zarr_format_2() -> None:
    data = np.arange(6, dtype="int64")
    z = zarr.create_array(
        {}, shape=data.shape, chunks=(2,), dtype=data.dtype, zarr_format=2, fill_value=-1
    )
    z[:] = data
    result = z.get_window((slice(-2, 8),), mode="constant")
    assert_array_equal(result, pad_reference(data, (slice(-2, 8),), "constant", -1))
    result = z.get_window((slice(-2, 8),), mode="wrap")
    assert_array_equal(result, pad_reference(data, (slice(-2, 8),), "wrap"))
