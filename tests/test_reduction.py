"""Tests for chunked sum/mean/variance reductions on zarr arrays."""

from __future__ import annotations

import math

import numpy as np
import pytest

import zarr
from zarr.core.array import AsyncArray
from zarr.storage import MemoryStore


def _make_array(
    data: np.ndarray,
    chunks: tuple[int, ...],
    *,
    fill_value: float = 0.0,
    write: bool = True,
) -> zarr.Array:
    arr = zarr.create_array(
        store=MemoryStore(),
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        fill_value=fill_value,
    )
    if write:
        arr[:] = data
    return arr


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), -1, (0, -1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_matches_numpy(axis, keepdims):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(30, 17))
    arr = _make_array(data, chunks=(7, 5))
    result = arr.sum(axis=axis, keepdims=keepdims)
    expected = data.sum(axis=axis, keepdims=keepdims)
    assert np.asarray(result).shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_mean_matches_numpy(axis, keepdims):
    rng = np.random.default_rng(43)
    data = rng.normal(size=(30, 17))
    arr = _make_array(data, chunks=(7, 5))
    result = arr.mean(axis=axis, keepdims=keepdims)
    expected = data.mean(axis=axis, keepdims=keepdims)
    assert np.asarray(result).shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), -1])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("ddof", [0, 1, 3])
def test_variance_matches_numpy(axis, keepdims, ddof):
    rng = np.random.default_rng(44)
    data = rng.normal(size=(30, 17))
    arr = _make_array(data, chunks=(7, 5))
    result = arr.variance(axis=axis, ddof=ddof, keepdims=keepdims)
    expected = data.var(axis=axis, ddof=ddof, keepdims=keepdims)
    assert np.asarray(result).shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_variance_large_bias_small_spread():
    # A naive sum-of-squares-minus-square-of-sum implementation loses all
    # precision here; the merged update must stay accurate.
    rng = np.random.default_rng(0)
    data = 1e9 + rng.normal(0.0, 1.0, size=(500, 40))
    arr = _make_array(data, chunks=(53, 40))
    result = arr.variance(axis=0)
    expected = data.var(axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    result_all = arr.variance()
    np.testing.assert_allclose(result_all, data.var(), rtol=1e-6)


def test_all_nan():
    data = np.full((9, 4), np.nan)
    arr = _make_array(data, chunks=(4, 4))
    assert np.isnan(arr.mean())
    assert np.isnan(arr.variance())
    assert np.isnan(arr.sum())
    # With omit, no observations remain: mean/variance are NaN, sum is 0.
    assert np.isnan(arr.mean(nan_policy="omit"))
    assert np.isnan(arr.variance(nan_policy="omit"))
    assert arr.sum(nan_policy="omit") == 0.0


def test_nan_propagate_vs_omit():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(20, 6))
    data[3, 2] = np.nan
    data[7, 2] = np.nan
    data[0, 5] = np.nan
    arr = _make_array(data, chunks=(6, 4))
    # propagate: any column containing a NaN reduces to NaN
    mean_p = arr.mean(axis=0)
    assert np.isnan(mean_p[2])
    assert np.isnan(mean_p[5])
    assert not np.isnan(mean_p[0])
    var_p = arr.variance(axis=0)
    assert np.isnan(var_p[2])
    assert np.isnan(var_p[5])
    sum_p = arr.sum(axis=0)
    assert np.isnan(sum_p[2])
    # omit: matches the np.nan* functions
    np.testing.assert_allclose(
        arr.mean(axis=0, nan_policy="omit"), np.nanmean(data, axis=0), rtol=1e-12
    )
    np.testing.assert_allclose(
        arr.variance(axis=0, nan_policy="omit"), np.nanvar(data, axis=0), rtol=1e-10
    )
    np.testing.assert_allclose(
        arr.sum(axis=0, nan_policy="omit"), np.nansum(data, axis=0), rtol=1e-12
    )


def test_sparse_fill_value_counts_as_observation():
    fill = 3.0
    shape = (20, 8)
    arr = zarr.create_array(
        store=MemoryStore(), shape=shape, chunks=(6, 8), dtype="f8", fill_value=fill
    )
    # Only write one interior chunk; the rest stays on the fill value.
    written = np.arange(48.0).reshape(6, 8)
    arr[6:12, :] = written
    expected = np.full(shape, fill)
    expected[6:12, :] = written
    assert arr.sum() == expected.sum()
    assert arr.mean() == expected.mean()
    np.testing.assert_allclose(arr.variance(), expected.var(), rtol=1e-12)
    np.testing.assert_allclose(arr.sum(axis=1), expected.sum(axis=1), rtol=1e-12)
    np.testing.assert_allclose(arr.variance(axis=1), expected.var(axis=1), rtol=1e-12)


def test_ragged_edge_chunks():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(23, 11))
    arr = _make_array(data, chunks=(8, 5))
    np.testing.assert_allclose(arr.sum(axis=0), data.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(arr.mean(axis=1), data.mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(arr.variance(axis=0), data.var(axis=0), rtol=1e-10)
    np.testing.assert_allclose(arr.variance(axis=1, ddof=1), data.var(axis=1, ddof=1), rtol=1e-10)


@pytest.mark.parametrize("chunks", [(1, 40), (7, 40), (50, 40), (500, 40), (33, 17), (500, 1)])
def test_chunk_layout_invariance(chunks):
    rng = np.random.default_rng(3)
    data = rng.normal(size=(500, 40))
    arr = _make_array(data, chunks=chunks)
    np.testing.assert_allclose(arr.sum(axis=0), data.sum(axis=0), rtol=1e-9)
    np.testing.assert_allclose(arr.mean(axis=0), data.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(arr.variance(axis=0), data.var(axis=0), rtol=1e-9)
    np.testing.assert_allclose(arr.variance(), data.var(), rtol=1e-9)


def test_out_parameter():
    data = np.arange(24.0).reshape(4, 6)
    arr = _make_array(data, chunks=(2, 3))
    out = np.empty(4)
    result = arr.mean(axis=1, out=out)
    assert result is out
    np.testing.assert_allclose(out, data.mean(axis=1))
    with pytest.raises(ValueError, match="out argument"):
        arr.mean(axis=1, out=np.empty(3))


def test_empty_reduction():
    arr = zarr.create_array(store=MemoryStore(), shape=(0, 3), chunks=(4, 3), dtype="f8")
    np.testing.assert_array_equal(arr.sum(axis=0), np.zeros(3))
    assert np.all(np.isnan(arr.mean(axis=0)))
    assert np.all(np.isnan(arr.variance(axis=0)))
    assert arr.sum() == 0.0
    assert np.isnan(arr.mean())
    assert np.isnan(arr.variance())


def test_ddof_exceeding_sample_count_is_nan():
    data = np.arange(6.0)
    arr = _make_array(data, chunks=(2,))
    assert np.isnan(arr.variance(ddof=6))
    assert np.isnan(arr.variance(ddof=10))
    np.testing.assert_allclose(arr.variance(ddof=5), data.var(ddof=5))


def test_integer_sum_dtype_promotion():
    data = np.arange(10, dtype=np.int32)
    arr = _make_array(data, chunks=(3,))
    result = arr.sum()
    assert np.asarray(result).dtype == np.dtype(np.int64)
    assert result == data.sum()


def test_accumulator_dtype_override():
    data = np.arange(10.0)
    arr = _make_array(data, chunks=(3,))
    assert np.asarray(arr.mean(dtype=np.dtype(np.float32))).dtype == np.float32
    assert np.asarray(arr.variance(dtype=np.dtype(np.float32))).dtype == np.float32
    assert np.asarray(arr.sum(dtype=np.dtype(np.float32))).dtype == np.float32
    with pytest.raises(ValueError, match="accumulator dtype"):
        arr.mean(dtype=np.dtype(np.int32))
    with pytest.raises(ValueError, match="accumulator dtype"):
        arr.variance(dtype=np.dtype(np.int32))


def test_invalid_arguments():
    arr = _make_array(np.arange(6.0), chunks=(2,))
    with pytest.raises(ValueError, match="nan_policy"):
        arr.sum(nan_policy="ignore")
    with pytest.raises(ValueError, match="ddof"):
        arr.variance(ddof=-1)
    with pytest.raises(ValueError, match="axis"):
        arr.sum(axis=(0, 0))
    with pytest.raises(IndexError, match="axis"):
        arr.sum(axis=5)


def test_scalar_result_for_full_reduction():
    data = np.arange(6.0)
    arr = _make_array(data, chunks=(2,))
    assert np.isscalar(arr.sum())
    assert np.isscalar(arr.mean())
    assert np.isscalar(arr.variance())
    keep = arr.sum(keepdims=True)
    assert np.asarray(keep).shape == (1,)


def test_working_set_bounded_by_chunk_size(monkeypatch):
    # The reduction must never materialize more than one block at a time:
    # record every read and check each is at most one chunk.
    shape = (400, 300)
    chunks = (20, 25)
    arr = _make_array(np.ones(shape), chunks=chunks)
    read_sizes: list[int] = []
    original = AsyncArray.getitem

    async def spy(self, selection, *, prototype=None) -> np.ndarray:
        read_sizes.append(math.prod(s.stop - s.start for s in selection))
        return await original(self, selection, prototype=prototype)

    monkeypatch.setattr(AsyncArray, "getitem", spy)
    arr.sum(axis=0)
    assert max(read_sizes) == chunks[0] * chunks[1]
    assert len(read_sizes) == math.ceil(shape[0] / chunks[0]) * math.ceil(shape[1] / chunks[1])


async def test_async_array_methods():
    data = np.arange(24.0).reshape(4, 6)
    arr = _make_array(data, chunks=(2, 3))
    async_arr = arr.async_array
    np.testing.assert_allclose(await async_arr.sum(axis=0), data.sum(axis=0))
    np.testing.assert_allclose(await async_arr.mean(axis=1), data.mean(axis=1))
    np.testing.assert_allclose(await async_arr.variance(), data.var())


def test_sharded_array():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(16, 8))
    arr = zarr.create_array(
        store=MemoryStore(), shape=(16, 8), chunks=(4, 4), shards=(8, 8), dtype="f8"
    )
    arr[:] = data
    np.testing.assert_allclose(arr.sum(axis=0), data.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(arr.mean(axis=1), data.mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(arr.variance(axis=0), data.var(axis=0), rtol=1e-10)
