"""
Tests for the read-only virtual concatenation view (zarr.experimental.concat).
"""

import json

import numpy as np
import pytest

import zarr
from zarr.codecs import GzipCodec, ZstdCodec
from zarr.experimental import ConcatViewInvalidError, VirtualConcatArray
from zarr.storage import LocalStore, LoggingStore, MemoryStore


def _make_month(
    store: MemoryStore | LocalStore | LoggingStore,
    data: np.ndarray,
    chunks: tuple[int, ...],
    **kwargs: object,
) -> zarr.Array:
    arr = zarr.create_array(store, shape=data.shape, chunks=chunks, dtype=data.dtype, **kwargs)
    arr[:] = data
    return arr


@pytest.fixture
def month_data():
    # three "months" of 4 time steps and 2 spatial points each
    return [np.arange(8).reshape(4, 2) + 100 * i for i in range(3)]


@pytest.fixture
def months(month_data):
    # different chunk sizes, compressors, and fill values per component,
    # each in its own store
    chunks = [(2, 2), (3, 2), (4, 2)]
    compressors = [ZstdCodec(level=1), GzipCodec(level=1), "auto"]
    fill_values = [0, -1, 999]
    arrays = []
    for data, chunk, compressor, fill_value in zip(
        month_data, chunks, compressors, fill_values, strict=True
    ):
        arrays.append(
            _make_month(
                MemoryStore(),
                data,
                chunk,
                compressors=compressor,
                fill_value=fill_value,
            )
        )
    return arrays


@pytest.fixture
def view(months):
    return VirtualConcatArray(months, axis=0)


@pytest.fixture
def reference(month_data):
    return np.concatenate(month_data, axis=0)


def test_properties(view, months):
    assert view.shape == (12, 2)
    assert view.ndim == 2
    assert view.dtype == np.dtype("int64")
    assert view.size == 24
    assert len(view) == 12
    assert view.axis == 0
    assert view.components == tuple(months)
    assert view.read_only
    # components keep their own chunking
    assert [c.chunks for c in view.components] == [(2, 2), (3, 2), (4, 2)]


def test_negative_axis(months, reference):
    view = VirtualConcatArray(months, axis=-2)
    assert view.axis == 0
    np.testing.assert_array_equal(view[:], reference)


class TestBasicSelection:
    def test_full(self, view, reference):
        np.testing.assert_array_equal(view[:], reference)
        np.testing.assert_array_equal(view[...], reference)

    def test_slice_crossing_boundary(self, view, reference):
        np.testing.assert_array_equal(view[2:7], reference[2:7])
        np.testing.assert_array_equal(view[3:9], reference[3:9])
        np.testing.assert_array_equal(view[2:7, :], reference[2:7, :])

    def test_slice_with_step(self, view, reference):
        np.testing.assert_array_equal(view[1:11:2], reference[1:11:2])
        np.testing.assert_array_equal(view[::3], reference[::3])

    def test_negative_slice(self, view, reference):
        np.testing.assert_array_equal(view[-5:], reference[-5:])
        np.testing.assert_array_equal(view[:-7], reference[:-7])

    def test_int(self, view, reference):
        np.testing.assert_array_equal(view[5], reference[5])
        np.testing.assert_array_equal(view[0], reference[0])
        np.testing.assert_array_equal(view[-1], reference[-1])

    def test_scalar(self, view, reference):
        assert view[5, 1] == reference[5, 1]
        assert view[-1, -1] == reference[-1, -1]

    def test_other_axes(self, view, reference):
        np.testing.assert_array_equal(view[:, 0], reference[:, 0])
        np.testing.assert_array_equal(view[2:7, 1], reference[2:7, 1])

    def test_out_of_bounds(self, view):
        with pytest.raises(IndexError):
            view[12]
        with pytest.raises(IndexError):
            view[-13]
        with pytest.raises(IndexError):
            view[0, 2]

    def test_too_many_indices(self, view):
        with pytest.raises(IndexError, match="too many indices"):
            view[0, 0, 0]

    def test_fancy_index_rejected_in_basic_selection(self, view):
        with pytest.raises(IndexError, match="oindex"):
            view[[0, 1]]


class TestOrthogonalSelection:
    def test_int_array_crossing_boundary(self, view, reference):
        idx = [2, 3, 4, 5, 6]
        np.testing.assert_array_equal(view.oindex[idx], reference[idx])

    def test_duplicates_preserved(self, view, reference):
        idx = [3, 3, 4, 4, 4, 11]
        np.testing.assert_array_equal(view.oindex[idx], reference[idx])

    def test_unordered(self, view, reference):
        idx = [11, 0, 7, 3, 8]
        np.testing.assert_array_equal(view.oindex[idx], reference[idx])

    def test_negative_entries(self, view, reference):
        idx = [-1, -12, 5]
        np.testing.assert_array_equal(view.oindex[idx], reference[idx])

    def test_bool_mask(self, view, reference):
        mask = np.zeros(12, dtype=bool)
        mask[[1, 4, 10]] = True
        np.testing.assert_array_equal(view.oindex[mask], reference[mask])

    def test_int_array_on_other_axis(self, view, reference):
        np.testing.assert_array_equal(
            view.oindex[[2, 5, 9], [0, 1]], reference[[2, 5, 9]][:, [0, 1]]
        )
        np.testing.assert_array_equal(view.oindex[2:9, [1, 0]], reference[2:9][:, [1, 0]])

    def test_mixed_int_and_array(self, view, reference):
        np.testing.assert_array_equal(view.oindex[5, [0, 1]], reference[5][[0, 1]])

    def test_out_of_bounds(self, view):
        with pytest.raises(IndexError):
            view.oindex[[12]]
        with pytest.raises(IndexError):
            view.oindex[[-13]]

    def test_bool_mask_wrong_length(self, view):
        with pytest.raises(IndexError, match="boolean index"):
            view.oindex[np.zeros(5, dtype=bool)]


class TestEmptyComponent:
    @pytest.fixture
    def view_with_empty(self, month_data):
        arrays = [
            _make_month(MemoryStore(), month_data[0], (2, 2)),
            _make_month(MemoryStore(), month_data[1][:0], (3, 2)),  # 0 time steps
            _make_month(MemoryStore(), month_data[2], (4, 2)),
        ]
        return VirtualConcatArray(arrays, axis=0)

    @pytest.fixture
    def reference_without_empty(self, month_data):
        return np.concatenate([month_data[0], month_data[2]], axis=0)

    def test_shape(self, view_with_empty):
        assert view_with_empty.shape == (8, 2)

    def test_no_phantom_elements(self, view_with_empty, reference_without_empty):
        np.testing.assert_array_equal(view_with_empty[:], reference_without_empty)
        np.testing.assert_array_equal(view_with_empty[3:5], reference_without_empty[3:5])
        np.testing.assert_array_equal(
            view_with_empty.oindex[[3, 4, 7]], reference_without_empty[[3, 4, 7]]
        )

    def test_empty_component_never_read(self, view_with_empty):
        # re-open the empty component behind a logging store to prove no reads
        empty_store = LoggingStore(MemoryStore())
        _make_month(empty_store, np.zeros((0, 2), dtype="int64"), (3, 2))
        components = list(view_with_empty.components)
        components[1] = zarr.open_array(store=empty_store)
        view = VirtualConcatArray(components, axis=0)
        empty_store.counter.clear()
        view[:]
        assert dict(empty_store.counter) == {}


class TestValidation:
    def test_dtype_mismatch(self, month_data):
        a = _make_month(MemoryStore(), month_data[0], (2, 2))
        b = _make_month(MemoryStore(), month_data[1].astype("float32"), (2, 2))
        with pytest.raises(ValueError, match="dtype"):
            VirtualConcatArray([a, b], axis=0)

    def test_shape_mismatch_off_axis(self, month_data):
        a = _make_month(MemoryStore(), month_data[0], (2, 2))
        b = _make_month(MemoryStore(), month_data[1].reshape(2, 4), (2, 4))
        with pytest.raises(ValueError, match="axis"):
            VirtualConcatArray([a, b], axis=0)

    def test_ndim_mismatch(self, month_data):
        a = _make_month(MemoryStore(), month_data[0], (2, 2))
        b = _make_month(MemoryStore(), month_data[1].ravel(), (4,))
        with pytest.raises(ValueError, match="dimensions"):
            VirtualConcatArray([a, b], axis=0)

    def test_no_components(self):
        with pytest.raises(ValueError, match="at least one"):
            VirtualConcatArray([])

    def test_bad_axis(self, months):
        with pytest.raises(ValueError, match="axis"):
            VirtualConcatArray(months, axis=2)

    def test_read_only(self, view):
        with pytest.raises(ValueError, match="read-only"):
            view[0] = 0


class TestInvalidation:
    def test_resize_invalidates_view(self, view, months):
        months[0].resize((6, 2))
        with pytest.raises(ConcatViewInvalidError, match="component 0"):
            view[:]
        with pytest.raises(ConcatViewInvalidError, match="component 0"):
            view.oindex[[0, 1]]

    def test_shrink_invalidates_view(self, view, months):
        months[2].resize((2, 2))
        with pytest.raises(ConcatViewInvalidError, match="component 2"):
            view[5]

    def test_error_message_shows_shapes(self, view, months):
        months[1].resize((5, 2))
        with pytest.raises(ConcatViewInvalidError, match=r"(4, 2).*(5, 2)"):
            view[:]


class TestSerialization:
    @pytest.fixture
    def local_view(self, tmp_path, month_data):
        arrays = []
        for i, data in enumerate(month_data):
            store = LocalStore(root=tmp_path / f"month{i}")
            arrays.append(_make_month(store, data, (2 + i, 2)))
        return VirtualConcatArray(arrays, axis=0)

    def test_round_trip(self, local_view, reference):
        config = local_view.to_dict()
        # the configuration must be JSON-serializable
        payload = json.dumps(config)
        restored = VirtualConcatArray.from_json(payload)
        assert restored.shape == local_view.shape
        assert restored.dtype == local_view.dtype
        np.testing.assert_array_equal(restored[:], reference)
        np.testing.assert_array_equal(restored.oindex[[3, 3, 4]], reference[[3, 3, 4]])

    def test_to_json_from_json(self, local_view, reference):
        restored = VirtualConcatArray.from_json(local_view.to_json())
        np.testing.assert_array_equal(restored[2:7], reference[2:7])

    def test_config_records_no_credentials(self, local_view):
        config = local_view.to_dict()
        for entry in config["components"]:
            assert set(entry["store"]) <= {"type", "root", "class"}

    def test_external_stores_must_be_resupplied(self, view, reference):
        # MemoryStores cannot be described by a plain locator; the caller
        # re-supplies them (e.g. with fresh credentials) via stores=
        config = view.to_dict()
        assert all(e["store"]["type"] == "external" for e in config["components"])
        with pytest.raises(ValueError, match="re-supply"):
            VirtualConcatArray.from_dict(config)
        stores = [c.store for c in view.components]
        restored = VirtualConcatArray.from_dict(config, stores=stores)
        np.testing.assert_array_equal(restored[:], reference)

    def test_from_dict_store_count_mismatch(self, view):
        with pytest.raises(ValueError, match="stores"):
            VirtualConcatArray.from_dict(view.to_dict(), stores=[])

    def test_from_dict_detects_shape_change(self, local_view, tmp_path):
        config = local_view.to_dict()
        arr = zarr.open_array(store=LocalStore(root=tmp_path / "month0"), mode="a")
        arr.resize((7, 2))
        with pytest.raises(ConcatViewInvalidError, match="component 0"):
            VirtualConcatArray.from_dict(config)


class TestCrossBoundaryReads:
    """Acceptance: three chunk sizes, five cross-boundary time points."""

    @pytest.fixture
    def logged_months(self, month_data):
        stores = []
        arrays = []
        for data, chunks in zip(month_data, [(2, 2), (3, 2), (4, 2)], strict=True):
            backing = MemoryStore()
            _make_month(backing, data, chunks)
            store = LoggingStore(backing)
            stores.append(store)
            arrays.append(zarr.open_array(store=store))
        return stores, arrays

    def test_only_adjacent_months_are_read(self, logged_months, reference):
        stores, arrays = logged_months
        view = VirtualConcatArray(arrays, axis=0)
        for store in stores:
            store.counter.clear()

        result = view.oindex[[2, 3, 4, 5, 6]]
        np.testing.assert_array_equal(result, reference[[2, 3, 4, 5, 6]])

        # the selection spans months 0 and 1; month 2 must not be touched at all
        assert stores[0].counter["get"] > 0
        assert stores[1].counter["get"] > 0
        assert dict(stores[2].counter) == {}

    def test_basic_slice_only_reads_overlapping_months(self, logged_months, reference):
        stores, arrays = logged_months
        view = VirtualConcatArray(arrays, axis=0)
        for store in stores:
            store.counter.clear()

        np.testing.assert_array_equal(view[6:9], reference[6:9])
        assert dict(stores[0].counter) == {}
        assert stores[1].counter["get"] > 0
        assert stores[2].counter["get"] > 0
