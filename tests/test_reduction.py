"""Tests for the chunked streaming reductions (sum, mean, variance)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import zarr
import zarr.core.array
from zarr.core.reduction import reduce_array


def make_array(data: np.ndarray, chunks: tuple[int, ...], **kwargs: object) -> zarr.Array:
    arr = zarr.create_array(store={}, shape=data.shape, chunks=chunks, dtype=data.dtype, **kwargs)
    arr[:] = data
    return arr


class TestNumPyParity:
    """Reductions match NumPy on dense arrays with ragged end chunks."""

    @pytest.fixture
    def data(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.normal(size=(11, 7, 5))

    @pytest.mark.parametrize("chunks", [(4, 3, 2), (11, 7, 5), (1, 1, 1), (5, 7, 3)])
    @pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2), (1, 2), (0, 1, 2), -1])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_sum(self, data: np.ndarray, chunks: tuple[int, ...], axis, keepdims: bool) -> None:
        arr = make_array(data, chunks)
        expected = data.sum(axis=axis, keepdims=keepdims)
        result = arr.sum(axis=axis, keepdims=keepdims)
        assert np.shape(result) == np.shape(expected)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("chunks", [(4, 3, 2), (11, 7, 5), (5, 7, 3)])
    @pytest.mark.parametrize("axis", [None, 0, (0, 2), -2])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_mean(self, data: np.ndarray, chunks: tuple[int, ...], axis, keepdims: bool) -> None:
        arr = make_array(data, chunks)
        expected = data.mean(axis=axis, keepdims=keepdims)
        result = arr.mean(axis=axis, keepdims=keepdims)
        assert np.shape(result) == np.shape(expected)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("chunks", [(4, 3, 2), (11, 7, 5), (5, 7, 3)])
    @pytest.mark.parametrize("axis", [None, 0, (0, 2), -2])
    @pytest.mark.parametrize("ddof", [0, 1, 3])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_variance(
        self, data: np.ndarray, chunks: tuple[int, ...], axis, ddof: int, keepdims: bool
    ) -> None:
        arr = make_array(data, chunks)
        expected = data.var(axis=axis, ddof=ddof, keepdims=keepdims)
        result = arr.variance(axis=axis, ddof=ddof, keepdims=keepdims)
        assert np.shape(result) == np.shape(expected)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_integer_sum_dtype_and_exactness(self) -> None:
        data = np.arange(20, dtype=np.int8).reshape(4, 5)
        arr = make_array(data, (3, 2))
        result = arr.sum(axis=0)
        expected = data.sum(axis=0)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)

    def test_complex(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(9, 4)) + 1j * rng.normal(size=(9, 4))
        arr = make_array(data, (4, 3))
        np.testing.assert_allclose(arr.mean(axis=0), data.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(arr.variance(axis=1), data.var(axis=1), rtol=1e-12)
        assert arr.variance(axis=1).dtype == np.float64


class TestNumericalStability:
    """Large bias + small fluctuation: naive E[x^2] - E[x]^2 fails, the
    chunked Welford/Chan merge must not."""

    def test_large_bias_small_variance(self) -> None:
        rng = np.random.default_rng(7)
        data = 1e8 + rng.normal(scale=1e-2, size=(1000, 8))
        arr = make_array(data, (37, 8))
        expected = data.var(axis=0)
        result = arr.variance(axis=0)
        # With a mean/std ratio of 1e10, float64 rounding in any chunked
        # algorithm limits agreement with np.var to ~1e-6 relative; the
        # documented tolerance here is 1e-5.
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-12)
        # The naive "sum of squares minus squared sum" algorithm catastrophically
        # loses precision here, so it cannot reproduce the reference this closely.
        naive = (data**2).sum(axis=0) / len(data) - (data.sum(axis=0) / len(data)) ** 2
        assert not np.allclose(naive, expected, rtol=1e-5, atol=1e-12)

    def test_chunk_layout_invariance(self) -> None:
        rng = np.random.default_rng(3)
        data = 1e6 + rng.normal(size=(500, 10))
        reference = data.var(axis=0)
        for chunks in [(1, 10), (7, 5), (128, 10), (500, 10), (33, 3)]:
            arr = make_array(data, chunks)
            np.testing.assert_allclose(arr.variance(axis=0), reference, rtol=1e-9)
            np.testing.assert_allclose(arr.mean(axis=0), data.mean(axis=0), rtol=1e-12)
            np.testing.assert_allclose(arr.sum(axis=0), data.sum(axis=0), rtol=1e-12, atol=1e-6)


class TestSparseFillValue:
    """Unwritten chunks represent fill_value and count as observations."""

    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_unwritten_chunks_count_as_fill_value(self, zarr_format: int) -> None:
        arr = zarr.create_array(
            store={},
            shape=(10, 6),
            chunks=(4, 4),
            dtype="f8",
            fill_value=3.5,
            zarr_format=zarr_format,
        )
        arr[0:2, 0:2] = 1.0
        full = arr[:]
        np.testing.assert_allclose(arr.sum(axis=0), full.sum(axis=0), rtol=1e-12)
        np.testing.assert_allclose(arr.mean(axis=1), full.mean(axis=1), rtol=1e-12)
        np.testing.assert_allclose(arr.variance(), full.var(), rtol=1e-12)

    def test_fill_value_in_sample_count(self) -> None:
        # Only one chunk written, all values equal to the fill value:
        # variance must be exactly 0, not "missing data".
        arr = zarr.create_array(store={}, shape=(8,), chunks=(4,), dtype="f8", fill_value=2.0)
        arr[4:] = 2.0
        assert arr.variance() == 0.0
        assert arr.mean() == 2.0
        assert arr.sum() == 16.0

    def test_nan_fill_value_omit(self) -> None:
        arr = zarr.create_array(store={}, shape=(8,), chunks=(4,), dtype="f8", fill_value=np.nan)
        arr[:4] = [1.0, 2.0, 3.0, 4.0]
        # Unwritten chunks are NaN; with "omit" they drop out of the statistics.
        assert arr.sum(nan_policy="omit") == 10.0
        assert arr.mean(nan_policy="omit") == 2.5
        assert arr.variance(nan_policy="omit") == 1.25
        # With "propagate" they poison the result.
        assert math.isnan(arr.mean())


class TestNaNPolicies:
    @pytest.fixture
    def data(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        data = rng.normal(size=(10, 6))
        data[2, 3] = np.nan
        data[7, :] = np.nan
        return data

    def test_propagate(self, data: np.ndarray) -> None:
        arr = make_array(data, (3, 4))
        np.testing.assert_allclose(arr.sum(axis=0), data.sum(axis=0), rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(arr.mean(axis=1), data.mean(axis=1), rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            arr.variance(axis=0), data.var(axis=0), rtol=1e-12, equal_nan=True
        )

    def test_omit(self, data: np.ndarray) -> None:
        arr = make_array(data, (3, 4))
        np.testing.assert_allclose(
            arr.sum(axis=0, nan_policy="omit"), np.nansum(data, axis=0), rtol=1e-12
        )
        # Row 7 is all-NaN, so both NumPy and zarr warn about the empty slice.
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            expected_mean = np.nanmean(data, axis=1)
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            result_mean = arr.mean(axis=1, nan_policy="omit")
        np.testing.assert_allclose(result_mean, expected_mean, rtol=1e-12)
        np.testing.assert_allclose(
            arr.variance(axis=0, nan_policy="omit", ddof=1),
            np.nanvar(data, axis=0, ddof=1),
            rtol=1e-12,
        )

    def test_omit_all_nan_slice(self) -> None:
        arr = zarr.create_array(store={}, shape=(6,), chunks=(2,), dtype="f8", fill_value=np.nan)
        assert arr.sum(nan_policy="omit") == 0.0
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            assert math.isnan(arr.mean(nan_policy="omit"))
        with pytest.warns(RuntimeWarning, match="Degrees of freedom <= 0"):
            assert math.isnan(arr.variance(nan_policy="omit"))
        # Propagating NaN over an all-NaN slice yields NaN without a warning.
        assert math.isnan(arr.mean())

    def test_invalid_nan_policy(self) -> None:
        arr = zarr.create_array(store={}, shape=(4,), chunks=(2,), dtype="f8")
        with pytest.raises(ValueError, match="nan_policy"):
            arr.sum(nan_policy="raise")


class TestEmptyReductions:
    def test_zero_size_axis(self) -> None:
        arr = zarr.create_array(store={}, shape=(0, 4), chunks=(2, 2), dtype="f8")
        np.testing.assert_array_equal(arr.sum(axis=0), np.zeros(4))
        assert arr.sum() == 0.0
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            np.testing.assert_array_equal(arr.mean(axis=0), np.full(4, np.nan))
        with pytest.warns(RuntimeWarning, match="Degrees of freedom <= 0"):
            np.testing.assert_array_equal(arr.variance(axis=0), np.full(4, np.nan))

    def test_ddof_exceeds_sample_count(self) -> None:
        arr = zarr.create_array(store={}, shape=(3,), chunks=(2,), dtype="f8")
        arr[:] = [1.0, 2.0, 3.0]
        with pytest.warns(RuntimeWarning, match="Degrees of freedom <= 0"):
            assert math.isnan(arr.variance(ddof=3))
        with pytest.raises(ValueError, match="ddof"):
            arr.variance(ddof=-1)


class TestOutParameter:
    def test_out(self) -> None:
        data = np.arange(24.0).reshape(6, 4)
        arr = make_array(data, (2, 3))
        out = np.empty(4)
        result = arr.sum(axis=0, out=out)
        assert result is out
        np.testing.assert_array_equal(out, data.sum(axis=0))

    def test_out_with_keepdims(self) -> None:
        data = np.arange(24.0).reshape(6, 4)
        arr = make_array(data, (2, 3))
        out = np.empty((1, 4))
        arr.mean(axis=0, keepdims=True, out=out)
        np.testing.assert_allclose(out, data.mean(axis=0, keepdims=True))

    def test_out_scalar(self) -> None:
        arr = zarr.create_array(store={}, shape=(4,), chunks=(2,), dtype="f8")
        arr[:] = [1.0, 2.0, 3.0, 4.0]
        out = np.empty(())
        result = arr.sum(out=out)
        assert result is out
        assert out[()] == 10.0

    def test_out_shape_mismatch(self) -> None:
        arr = zarr.create_array(store={}, shape=(6, 4), chunks=(2, 3), dtype="f8")
        with pytest.raises(ValueError, match="out"):
            arr.sum(axis=0, out=np.empty(5))


class TestAccumulatorDtype:
    def test_sum_dtype_override(self) -> None:
        data = np.arange(12, dtype=np.int32).reshape(3, 4)
        arr = make_array(data, (2, 2))
        result = arr.sum(axis=0, dtype=np.float64)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, data.sum(axis=0))

    def test_mean_dtype_override(self) -> None:
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        arr = make_array(data, (2, 2))
        result = arr.mean(axis=0, dtype=np.float32)
        assert result.dtype == np.float32

    def test_invalid_accumulator_dtype(self) -> None:
        arr = zarr.create_array(store={}, shape=(4,), chunks=(2,), dtype="f8")
        with pytest.raises(TypeError, match="accumulator dtype"):
            arr.mean(dtype=np.int64)

    def test_non_numeric_dtype(self) -> None:
        arr = zarr.create_array(store={}, shape=(4,), chunks=(2,), dtype="str")
        with pytest.raises(TypeError, match="not supported"):
            arr.sum()


class TestAxisHandling:
    def test_duplicate_axis(self) -> None:
        arr = zarr.create_array(store={}, shape=(4, 4), chunks=(2, 2), dtype="f8")
        with pytest.raises(ValueError, match="duplicate"):
            arr.sum(axis=(0, 0))

    def test_axis_out_of_range(self) -> None:
        arr = zarr.create_array(store={}, shape=(4, 4), chunks=(2, 2), dtype="f8")
        with pytest.raises(Exception, match="axis"):
            arr.sum(axis=2)

    def test_negative_axes(self) -> None:
        data = np.arange(24.0).reshape(6, 4)
        arr = make_array(data, (2, 3))
        np.testing.assert_allclose(arr.sum(axis=-1), data.sum(axis=-1))
        np.testing.assert_allclose(arr.mean(axis=(0, -1)), data.mean())


class TestWorkingSet:
    """The working set scales with the chunk size, not the number of rows."""

    @pytest.mark.parametrize("n_rows", [256, 4096])
    def test_reads_bounded_by_chunk_size(
        self, monkeypatch: pytest.MonkeyPatch, n_rows: int
    ) -> None:
        chunks = (16, 64)
        rng = np.random.default_rng(0)
        data = rng.normal(size=(n_rows, 64))
        arr = make_array(data, chunks)

        read_sizes: list[int] = []
        real_get_selection = zarr.core.array._get_selection

        async def spy(*args: object, **kwargs: object) -> object:
            indexer = args[5]
            read_sizes.append(math.prod(indexer.shape))
            return await real_get_selection(*args, **kwargs)

        monkeypatch.setattr(zarr.core.array, "_get_selection", spy)
        np.testing.assert_allclose(arr.sum(axis=0), data.sum(axis=0), rtol=1e-12)

        # One read per chunk, and no read ever exceeds a single chunk, no
        # matter how many rows the array has.
        assert len(read_sizes) == arr.nchunks
        assert max(read_sizes) == math.prod(chunks)
        assert max(read_sizes) * data.dtype.itemsize < arr.nbytes


class TestAsyncAPI:
    async def test_reduce_array_async(self) -> None:
        data = np.arange(24.0).reshape(6, 4)
        arr = make_array(data, (2, 3))
        result = await reduce_array(arr.async_array, "variance", axis=0, ddof=1)
        np.testing.assert_allclose(result, data.var(axis=0, ddof=1), rtol=1e-12)

    async def test_async_array_methods(self) -> None:
        data = np.arange(24.0).reshape(6, 4)
        arr = make_array(data, (2, 3))
        np.testing.assert_allclose(await arr.async_array.sum(axis=1), data.sum(axis=1), rtol=1e-12)
        np.testing.assert_allclose(await arr.async_array.mean(), data.mean())
        np.testing.assert_allclose(
            await arr.async_array.variance(axis=0), data.var(axis=0), rtol=1e-12
        )
