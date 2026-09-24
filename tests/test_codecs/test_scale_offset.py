from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import zarr
from tests.conftest import Expect, ExpectFail
from zarr.codecs.scale_offset import (
    ScaleOffset,
    _decode,
    _decode_fits_natively,
    _encode,
)
from zarr.core.buffer.core import default_buffer_prototype
from zarr.storage._memory import MemoryStore

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(input=ScaleOffset(), output={"name": "scale_offset"}, id="default"),
        Expect(
            input=ScaleOffset(offset=5),
            output={"name": "scale_offset", "configuration": {"offset": 5}},
            id="offset-only",
        ),
        Expect(
            input=ScaleOffset(scale=0.1),
            output={"name": "scale_offset", "configuration": {"scale": 0.1}},
            id="scale-only",
        ),
        Expect(
            input=ScaleOffset(offset=5, scale=0.1),
            output={"name": "scale_offset", "configuration": {"offset": 5, "scale": 0.1}},
            id="both",
        ),
    ],
    ids=lambda c: c.id,
)
def test_to_dict(case: Expect[ScaleOffset, dict[str, Any]]) -> None:
    """to_dict produces the expected JSON structure."""
    assert case.input.to_dict() == case.output


@pytest.mark.parametrize(
    "case",
    [
        Expect(input={"name": "scale_offset"}, output=(0, 1), id="no-config"),
        Expect(
            input={"name": "scale_offset", "configuration": {"offset": 3, "scale": 2}},
            output=(3, 2),
            id="with-config",
        ),
    ],
    ids=lambda c: c.id,
)
def test_from_dict(case: Expect[dict[str, Any], tuple[int | float, int | float]]) -> None:
    """from_dict deserializes configuration with correct values and defaults."""
    codec = ScaleOffset.from_dict(case.input)
    expected_offset, expected_scale = case.output
    assert codec.offset == expected_offset
    assert codec.scale == expected_scale


def test_serialization_roundtrip() -> None:
    """to_dict followed by from_dict produces an equal codec."""
    original = ScaleOffset(offset=7, scale=0.5)
    restored = ScaleOffset.from_dict(original.to_dict())
    assert original == restored


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input={"offset": [1, 2]},
            exception=TypeError,
            id="list-offset",
            msg="offset must be a number or string",
        ),
        ExpectFail(
            input={"scale": [1, 2]},
            exception=TypeError,
            id="list-scale",
            msg="scale must be a number or string",
        ),
    ],
    ids=lambda c: c.id,
)
def test_construction_rejects_non_numeric(case: ExpectFail[dict[str, Any]]) -> None:
    """Non-numeric offset or scale is rejected at construction time."""
    with case.raises():
        ScaleOffset(**case.input)


@pytest.mark.parametrize(
    "case",
    [
        Expect(input={"offset": 5, "scale": 2}, output=(5, 2), id="int"),
        Expect(input={"offset": 0.5, "scale": 0.1}, output=(0.5, 0.1), id="float"),
    ],
    ids=lambda c: c.id,
)
def test_construction_accepts_numeric(
    case: Expect[dict[str, Any], tuple[int | float, int | float]],
) -> None:
    """Integer and float values are accepted for both parameters."""
    codec = ScaleOffset(**case.input)
    assert codec.offset == case.output[0]
    assert codec.scale == case.output[1]


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "offset", "scale"),
    [
        ("float64", 10.0, 0.1),
        ("float32", 5.0, 2.0),
        ("int32", 0, 1),
    ],
    ids=["float64", "float32", "int32-identity"],
)
def test_encode_decode_roundtrip(dtype: str, offset: float, scale: float) -> None:
    """Data survives encode → decode."""

    arr = zarr.create_array(
        store={},
        shape=(100,),
        dtype=dtype,
        chunks=(100,),
        filters=[ScaleOffset(offset=offset, scale=scale)],
        compressors=None,
        fill_value=0,
    )
    data = np.arange(100, dtype=dtype)
    arr[:] = data
    np.testing.assert_array_almost_equal(arr[:], data)  # type: ignore[arg-type]


def test_fill_value_transformed() -> None:
    """Fill value is transformed through the encode formula and read back correctly."""
    arr = zarr.create_array(
        store={},
        shape=(10,),
        dtype="float64",
        chunks=(10,),
        filters=[ScaleOffset(offset=5, scale=2)],
        compressors=None,
        fill_value=10.0,
    )
    # fill_value=10.0, encode: (10 - 5) * 2 = 10.0 stored
    # Reading back without writing should return the original fill value
    np.testing.assert_array_equal(arr[:], np.full(10, 10.0))


def test_identity_is_noop() -> None:
    """Default codec (offset=0, scale=1) is a no-op."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(50,),
        dtype="float64",
        chunks=(50,),
        filters=[ScaleOffset()],
        compressors=None,
        fill_value=0,
    )
    data = np.arange(50, dtype="float64")
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


def test_rejects_complex_dtype() -> None:
    """Complex dtypes are rejected at array creation time."""

    with pytest.raises(ValueError, match="only supports integer and floating-point"):
        zarr.create_array(
            store={},
            shape=(10,),
            dtype="complex128",
            chunks=(10,),
            filters=[ScaleOffset(offset=1, scale=2)],
            compressors=None,
            fill_value=0,
        )


def test_uint64_large_value_roundtrip() -> None:
    """uint64 values above 2**63 must survive encode+decode (spec requires uint64 support)."""

    arr = zarr.create_array(
        store={},
        shape=(3,),
        dtype="uint64",
        chunks=(3,),
        filters=[ScaleOffset(offset=0, scale=1)],
        compressors=None,
        fill_value=0,
    )
    # Value above int64.max (2**63 - 1) — would wrap if we used int64 as wide dtype.
    data = np.array([0, 2**63, 2**64 - 1], dtype="uint64")
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


def test_float_nan_inf_preserved() -> None:
    """NaN and Inf are representable in float dtypes per IEEE 754 and must pass through."""

    arr = np.array([1.0, np.nan, np.inf, -np.inf], dtype="float64")
    encoded = _encode(arr, np.float64(0.0), np.float64(2.0))
    np.testing.assert_array_equal(encoded[[0]], np.array([2.0]))
    assert np.isnan(encoded[1])
    assert encoded[2] == np.inf
    assert encoded[3] == -np.inf
    decoded = _decode(encoded, np.float64(0.0), np.float64(2.0), scale_repr=2.0)
    np.testing.assert_array_equal(decoded[[0]], np.array([1.0]))
    assert np.isnan(decoded[1])


def test_uint64_encode_rejects_underflow() -> None:
    """uint64 underflow during encode raises rather than silently wrapping."""

    arr = zarr.create_array(
        store={},
        shape=(3,),
        dtype="uint64",
        chunks=(3,),
        filters=[ScaleOffset(offset=100, scale=1)],
        compressors=None,
        fill_value=100,
    )
    with pytest.raises(ValueError, match="outside the range of dtype uint64"):
        arr[:] = np.array([100, 50, 200], dtype="uint64")


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [
        ("int32", 0),
        ("int32", "0"),
        ("float64", 0.0),
        ("float64", "0.0"),
        ("float64", "0x0000000000000000"),
    ],
    ids=["int-numeric", "int-string", "float-numeric", "float-string", "float-hex"],
)
def test_rejects_zero_scale(dtype: str, scale: object) -> None:
    """scale=0 is rejected (destroys data and breaks decode division).

    A string ``scale`` is a documented input, so every spelling of zero has to be
    rejected the same way as the numeric one.
    """

    with pytest.raises(ValueError, match="scale must be non-zero"):
        zarr.create_array(
            store={},
            shape=(10,),
            dtype=dtype,
            chunks=(10,),
            filters=[ScaleOffset(offset=0, scale=scale)],
            compressors=None,
            fill_value=0,
        )


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input={"dtype": "int32", "offset": 1.5, "scale": 1},
            exception=ValueError,
            id="float-offset-for-int",
            msg="offset value 1.5 is not representable",
        ),
        ExpectFail(
            input={"dtype": "int32", "offset": 0, "scale": 0.5},
            exception=ValueError,
            id="float-scale-for-int",
            msg="scale value 0.5 is not representable",
        ),
        ExpectFail(
            input={"dtype": "int16", "offset": "NaN", "scale": 1},
            exception=ValueError,
            id="nan-offset-for-int",
            msg="offset value 'NaN' is not representable",
        ),
    ],
    ids=lambda c: c.id,
)
def test_rejects_unrepresentable_scale_offset(case: ExpectFail[dict[str, Any]]) -> None:
    """Scale/offset values that can't be represented in the array dtype are rejected."""

    with case.raises():
        zarr.create_array(
            store={},
            shape=(10,),
            dtype=case.input["dtype"],
            chunks=(10,),
            filters=[ScaleOffset(offset=case.input["offset"], scale=case.input["scale"])],
            compressors=None,
            fill_value=0,
        )


def test_dtype_preservation() -> None:
    """Integer scale/offset arithmetic preserves the array dtype when division is exact."""

    arr = zarr.create_array(
        store={},
        shape=(10,),
        dtype="int8",
        chunks=(10,),
        filters=[ScaleOffset(offset=1, scale=2)],
        compressors=None,
        fill_value=0,
    )
    data = np.arange(10, dtype="int8")
    arr[:] = data
    # encode=(x-1)*2 is always divisible by scale=2, so decode is exact
    np.testing.assert_array_equal(arr[:], data)


async def test_integer_decode_rejects_non_exact_division() -> None:
    """Decoding an integer array raises when the stored value isn't divisible by scale."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(3,),
        dtype="int8",
        chunks=(3,),
        filters=[ScaleOffset(offset=0, scale=2)],
        compressors=None,
        fill_value=0,
    )
    # Write raw encoded bytes directly so we can inject a value that isn't divisible by scale.
    # Array layout: int8 [2, 3, 4]; 3 % 2 != 0, so decode must fail.

    buf = default_buffer_prototype().buffer.from_bytes(np.array([2, 3, 4], dtype="int8").tobytes())
    await arr.store_path.store.set("c/0", buf)
    with pytest.raises(ValueError, match="non-zero remainder"):
        arr[:]


def test_encode_rejects_signed_integer_overflow() -> None:
    """Encoding raises when (value - offset) * scale exceeds the target integer range."""
    arr = zarr.create_array(
        store={},
        shape=(3,),
        dtype="int8",
        chunks=(3,),
        filters=[ScaleOffset(offset=0, scale=100)],
        compressors=None,
        fill_value=0,
    )
    # 2 * 100 = 200, outside int8 range [-128, 127]
    with pytest.raises(ValueError, match="outside the range of dtype int8"):
        arr[:] = np.array([0, 1, 2], dtype="int8")


def test_encode_rejects_unsigned_integer_underflow() -> None:
    """Encoding raises when value - offset underflows an unsigned dtype."""
    arr = zarr.create_array(
        store={},
        shape=(3,),
        dtype="uint8",
        chunks=(3,),
        filters=[ScaleOffset(offset=10, scale=1)],
        compressors=None,
        fill_value=10,
    )
    # 5 - 10 = -5, outside uint8 range [0, 255]
    with pytest.raises(ValueError, match="outside the range of dtype uint8"):
        arr[:] = np.array([10, 5, 20], dtype="uint8")


def test_float32_dtype_preserved() -> None:
    """float32 arrays survive encode+decode without being promoted to float64."""
    arr = np.arange(100, dtype="float32")
    offset = np.float32(5.0)
    scale = np.float32(0.25)
    encoded = _encode(arr, offset, scale)
    assert encoded.dtype == np.dtype("float32")
    decoded = _decode(encoded, offset, scale, scale_repr=0.25)
    assert decoded.dtype == np.dtype("float32")


def test_float_encode_rejects_wider_scalar() -> None:
    """A float64 scalar passed with a float32 array must not silently widen the result."""
    arr = np.arange(10, dtype="float32")
    # A numpy float64 scalar (not a Python float — NEP 50 exempts those) mixed with a
    # float32 ndarray promotes to float64. The codec must reject that.
    with pytest.raises(ValueError, match="changed dtype from float32 to float64"):
        _encode(arr, np.float64(5.0), np.float64(0.25))


def test_float_decode_rejects_wider_scalar() -> None:
    """A float64 scalar passed with a float32 array must not silently widen on decode."""
    arr = np.arange(10, dtype="float32")
    with pytest.raises(ValueError, match="changed dtype from float32 to float64"):
        _decode(arr, np.float64(5.0), np.float64(0.25), scale_repr=0.25)


async def test_decode_rejects_integer_overflow_on_offset_add() -> None:
    """Decoding raises when quotient + offset overflows the target integer dtype."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(3,),
        dtype="int8",
        chunks=(3,),
        filters=[ScaleOffset(offset=100, scale=1)],
        compressors=None,
        fill_value=0,
    )
    # encoded=100 → decoded = 100/1 + 100 = 200, outside int8 range
    buf = default_buffer_prototype().buffer.from_bytes(
        np.array([0, 50, 100], dtype="int8").tobytes()
    )
    await arr.store_path.store.set("c/0", buf)
    with pytest.raises(ValueError, match="outside the range of dtype int8"):
        arr[:]


def test_decode_fits_natively_negative_scale() -> None:
    """_decode_fits_natively handles negative scale by swapping bounds."""
    # For a negative scale, x // scale flips the relationship between min/max.
    # The function should use info.max // scale as the lower bound and info.min // scale
    # as the upper bound.
    dtype = np.dtype("int16")
    # scale=-2 inverts; offset=0 means range is just q_lo..q_hi
    assert _decode_fits_natively(dtype, offset=0, scale=-2) is True
    # An offset that pushes the range out of bounds returns False
    assert _decode_fits_natively(dtype, offset=100000, scale=-2) is False


async def test_decode_int_widened_path() -> None:
    """When _decode_fits_natively returns False, decode falls through to the widened path."""
    # For uint32 with offset near max, q_hi + offset can exceed uint32 if computed in target dtype.
    # The widened path uses int64 arithmetic and range-checks the result.
    # We bypass encode by writing raw bytes directly to the store.
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(3,),
        dtype="uint32",
        chunks=(3,),
        # offset large enough that _decode_fits_natively returns False
        filters=[ScaleOffset(offset=2**31, scale=1)],
        compressors=None,
        # fill_value must be >= offset to avoid uint32 underflow during encode
        fill_value=2**31,
    )
    # Encoded values that, when added to offset, stay within uint32
    buf = default_buffer_prototype().buffer.from_bytes(
        np.array([0, 100, 1000], dtype="uint32").tobytes()
    )
    await arr.store_path.store.set("c/0", buf)
    expected = np.array([2**31, 2**31 + 100, 2**31 + 1000], dtype="uint32")
    np.testing.assert_array_equal(arr[:], expected)


def test_compute_encoded_size() -> None:
    """compute_encoded_size returns the input byte length unchanged (codec is fixed-size)."""
    codec = ScaleOffset(offset=0, scale=1)
    # The chunk_spec argument is unused; pass any sentinel
    assert codec.compute_encoded_size(input_byte_length=100, _chunk_spec=None) == 100  # type: ignore[arg-type]
    assert codec.compute_encoded_size(input_byte_length=0, _chunk_spec=None) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integer overflow exactness, checked against a Python-int reference
# ---------------------------------------------------------------------------

_INT_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


def _boundary_values(dtype: np.dtype[Any]) -> list[int]:
    """Values on both sides of a dtype's boundaries, as unbounded Python ints."""
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    candidates = [lo, lo + 1, lo // 2, -2, -1, 0, 1, 2, hi // 2, hi - 1, hi]
    return sorted({v for v in candidates if lo <= v <= hi})


def _representable_params(dtype: np.dtype[Any]) -> list[tuple[int, int]]:
    """(offset, scale) pairs representable in ``dtype``, stressing signed +, -, *, //."""
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    pairs = [(0, 1), (1, 1), (0, 2), (1, 2), (-1, 1), (-1, 2), (0, -1), (1, -2), (3, 7), (-5, 3)]
    return [(o, s) for o, s in pairs if lo <= o <= hi and lo <= s <= hi]


@pytest.mark.parametrize("dtype_name", _INT_DTYPES)
def test_encode_int_matches_python_reference(dtype_name: str) -> None:
    """_encode agrees with unbounded Python-int arithmetic at the dtype boundaries.

    When the true result leaves the dtype's range the codec must raise (never return a
    wrapped array); when it fits, the codec must return it exactly (never falsely reject).
    """
    dtype = np.dtype(dtype_name)
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    values = _boundary_values(dtype)
    arr = np.array(values, dtype=dtype)
    original = arr.copy()
    for offset, scale in _representable_params(dtype):
        expected = [(v - offset) * scale for v in values]
        if all(lo <= e <= hi for e in expected):
            out = _encode(arr, dtype.type(offset), dtype.type(scale))
            assert out.dtype == dtype
            assert list(out.tolist()) == expected
        else:
            with pytest.raises(ValueError, match="outside the range of dtype"):
                _encode(arr, dtype.type(offset), dtype.type(scale))
        # A failed or successful encode must never modify the input in place.
        np.testing.assert_array_equal(arr, original)


@pytest.mark.parametrize("dtype_name", _INT_DTYPES)
def test_decode_int_matches_python_reference(dtype_name: str) -> None:
    """_decode agrees with unbounded Python-int arithmetic at the dtype boundaries.

    Encoded inputs are exactly divisible by scale, so decode must reproduce the
    Python-computed quotient + offset exactly, or raise if it leaves the dtype's range.
    """
    dtype = np.dtype(dtype_name)
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    for offset, scale in _representable_params(dtype):
        quotients = _boundary_values(dtype)
        encoded = sorted({q * scale for q in quotients if lo <= q * scale <= hi})
        arr = np.array(encoded, dtype=dtype)
        original = arr.copy()
        # e == q * scale exactly, so Python floor division recovers q for any sign.
        expected = [e // scale + offset for e in encoded]
        if all(lo <= x <= hi for x in expected):
            out = _decode(arr, dtype.type(offset), dtype.type(scale), scale_repr=scale)
            assert out.dtype == dtype
            assert list(out.tolist()) == expected
        else:
            with pytest.raises(ValueError, match="outside the range of dtype"):
                _decode(arr, dtype.type(offset), dtype.type(scale), scale_repr=scale)
        np.testing.assert_array_equal(arr, original)


def test_encode_int64_overflow_does_not_wrap() -> None:
    """Regression: int64 encode must not wrap when (x - offset) * scale exceeds int64.

    The widened path itself computes in int64, so for int64 inputs an overflowing
    intermediate used to wrap *before* the range check and silently pass. Here
    2**62 * 2 = 2**63 is exactly one past int64 max.
    """
    arr = np.array([2**62], dtype="int64")
    original = arr.copy()
    with pytest.raises(ValueError, match="outside the range of dtype int64"):
        _encode(arr, np.int64(0), np.int64(2))
    np.testing.assert_array_equal(arr, original)


def test_encode_int64_accepts_values_that_fit_despite_failed_static_proof() -> None:
    """A dtype-wide proof failure must not reject data whose true results fit in int64."""
    # scale=2**40 overflows int64 for large inputs, so no static proof can pass — but
    # these particular values are small enough that the true results are representable.
    arr = np.array([1, -1, 2, 2**22], dtype="int64")
    out = _encode(arr, np.int64(0), np.int64(2**40))
    expected = np.array([2**40, -(2**40), 2**41, 2**62], dtype="int64")
    np.testing.assert_array_equal(out, expected)
    # One step further and the true result leaves int64: must raise, not wrap.
    with pytest.raises(ValueError, match="outside the range of dtype int64"):
        _encode(np.array([2**23], dtype="int64"), np.int64(0), np.int64(2**40))


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_decode_rejects_min_divided_by_minus_one(dtype_name: str) -> None:
    """min // -1 overflows every signed dtype; decode must raise a stable ValueError.

    numpy's own ``//`` wraps here (and only emits a RuntimeWarning), so this guards
    against the codec delegating to numpy arithmetic without a range proof.
    """
    dtype = np.dtype(dtype_name)
    arr = np.array([np.iinfo(dtype).min, 0, 4], dtype=dtype)
    original = arr.copy()
    with warnings.catch_warnings():
        # The overflow must surface as ValueError, never as a numpy RuntimeWarning.
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="outside the range of dtype"):
            _decode(arr, dtype.type(0), dtype.type(-1), scale_repr=-1)
    np.testing.assert_array_equal(arr, original)


def test_provably_safe_int_arrays_do_not_use_object_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provably safe integer arrays keep the vectorized paths.

    Converting every integer array to per-element Python objects would be correct but
    ~10x slower; the native/widened paths must still be used when the static proofs hold.
    """
    from zarr.codecs import scale_offset as scale_offset_module

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("object path used for a provably safe case")

    monkeypatch.setattr(scale_offset_module, "_encode_int_object", _forbidden)
    monkeypatch.setattr(scale_offset_module, "_decode_int_object", _forbidden)
    for dtype_name, offset, scale in [
        ("int8", 1, 2),
        ("int32", 1, 2),
        ("uint8", 0, 2),
        ("uint32", 0, 3),
        ("int64", 0, 1),
        ("uint64", 0, 1),
    ]:
        dtype = np.dtype(dtype_name)
        arr = np.array([0, 1, 2, 3], dtype=dtype)
        encoded = _encode(arr, dtype.type(offset), dtype.type(scale))
        decoded = _decode(encoded, dtype.type(offset), dtype.type(scale), scale_repr=scale)
        np.testing.assert_array_equal(decoded, arr)
