"""Read-only virtual concatenation of Zarr arrays along a single axis.

A common layout for long-running archives is one Zarr array per time slice
(e.g. one array per month, possibly in different stores).  VirtualConcatArray
presents such a collection as a single logical array whose concatenation axis
(e.g. time) is the concatenation of the component axes.  Opening the view only
inspects array metadata -- no pixel data or chunks are copied -- and reads are
delegated to the component arrays, so components may live in different stores
and use different chunk shapes, compressors, or fill values.  Only the
components that overlap a requested selection are ever read.

Components must share the same dtype and the same shape on every axis except
the concatenation axis.  The view records each component's shape when it is
created and raises ConcatViewInvalidError if a component is resized
afterwards, instead of silently shifting the logical indexes of later
components.

The view configuration is serializable via VirtualConcatArray.to_dict and
VirtualConcatArray.to_json.  Serialized configurations never contain
credentials; stores that cannot be described by a plain locator are recorded
as placeholders and must be re-supplied by the caller through the "stores"
argument of VirtualConcatArray.from_dict and VirtualConcatArray.from_json.
"""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from zarr.core.array import Array
    from zarr.storage._common import StoreLike


class ConcatViewInvalidError(ValueError):
    """Raised when a component of a virtual concatenation view no longer matches
    the shape recorded when the view was created.

    Rebuild the view from the current component arrays instead of relying on
    logical indexes computed from the old shapes.
    """


def _expand_ellipsis(selection: Any, ndim: int) -> list[Any]:
    """Normalize a user selection to a list of exactly ndim items."""
    if not isinstance(selection, tuple):
        selection = (selection,)
    items = list(selection)
    n_ellipsis = sum(1 for item in items if item is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if any(item is None for item in items):
        raise IndexError("newaxis (None) is not supported by VirtualConcatArray")
    if n_ellipsis:
        pos = next(i for i, item in enumerate(items) if item is Ellipsis)
        n_fill = ndim - (len(items) - 1)
        if n_fill < 0:
            raise IndexError(
                f"too many indices for array: view is {ndim}-dimensional, "
                f"but {len(items) - 1} were indexed"
            )
        items = items[:pos] + [slice(None)] * n_fill + items[pos + 1 :]
    if len(items) > ndim:
        raise IndexError(
            f"too many indices for array: view is {ndim}-dimensional, but {len(items)} were indexed"
        )
    return items + [slice(None)] * (ndim - len(items))


def _is_int(item: Any) -> bool:
    return isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))


def _normalize_int_array(item: Any, dim_len: int) -> npt.NDArray[np.intp]:
    arr = np.asarray(item)
    if arr.ndim != 1:
        raise IndexError("only one-dimensional index arrays are supported")
    arr = arr.astype(np.intp)
    arr = np.where(arr < 0, arr + dim_len, arr)
    if arr.size and bool(((arr < 0) | (arr >= dim_len)).any()):
        raise IndexError(f"index is out of bounds for axis with size {dim_len}")
    return arr


def _as_slice_if_regular(local: npt.NDArray[np.intp]) -> slice | npt.NDArray[np.intp]:
    """Collapse a local index array to a slice when it is a regular range.

    Reading a contiguous (or evenly strided) range through a slice lets the
    component array use its fast contiguous-read path instead of fancy indexing.
    """
    if local.size == 1:
        return slice(int(local[0]), int(local[0]) + 1)
    step = int(local[1]) - int(local[0])
    if step > 0 and bool((np.diff(local) == step).all()):
        return slice(int(local[0]), int(local[-1]) + 1, step)
    return local


def _store_to_config(store: Any) -> dict[str, Any]:
    """Serialize a store to a plain locator, never including credentials."""
    from zarr.storage import LocalStore

    if isinstance(store, LocalStore):
        return {"type": "local", "root": str(store.root)}
    return {
        "type": "external",
        "class": f"{type(store).__module__}.{type(store).__qualname__}",
    }


class _OIndex:
    """Orthogonal selection accessor, mirroring zarr.Array.oindex."""

    def __init__(self, view: VirtualConcatArray) -> None:
        self._view = view

    def __getitem__(self, selection: Any) -> npt.NDArray[Any]:
        return self._view.get_orthogonal_selection(selection)


class VirtualConcatArray:
    """A read-only view concatenating several Zarr arrays along one axis.

    Parameters
    ----------
    arrays : sequence of zarr.Array
        The component arrays, in concatenation order.  They must all have the
        same number of dimensions, the same dtype, and the same shape on every
        axis except the concatenation axis.  Components may differ in chunking,
        compressors, and fill value, and may live in different stores.
        Components with a zero-length concatenation axis are allowed and
        contribute no elements.
    axis : int, optional
        The axis along which to concatenate (default 0).

    Notes
    -----
    The view is read-only.  The shape of every component is recorded at
    construction time; if a component is resized afterwards, any read raises
    ConcatViewInvalidError rather than silently shifting indexes.
    """

    def __init__(self, arrays: Sequence[Array], axis: int = 0) -> None:
        components = tuple(arrays)
        if not components:
            raise ValueError("VirtualConcatArray requires at least one component array")
        ndim = components[0].ndim
        axis = operator.index(axis)
        if axis < 0:
            axis += ndim
        if not 0 <= axis < ndim:
            raise ValueError(f"axis {axis} is out of bounds for {ndim}-dimensional arrays")

        dtype0 = components[0].dtype
        shape0 = components[0].shape
        for i, arr in enumerate(components):
            if arr.ndim != ndim:
                raise ValueError(
                    f"component {i} has {arr.ndim} dimensions, but component 0 has {ndim}"
                )
            if arr.dtype != dtype0:
                raise ValueError(
                    f"component {i} has dtype {arr.dtype}, "
                    f"but component 0 has dtype {dtype0}; "
                    "all components of a VirtualConcatArray must share a dtype"
                )
            for dim in range(ndim):
                if dim != axis and arr.shape[dim] != shape0[dim]:
                    raise ValueError(
                        f"component {i} has shape {arr.shape}, but component 0 has "
                        f"shape {shape0}; shapes may only differ along axis {axis}"
                    )

        self._components = components
        self._axis = axis
        self._shapes: tuple[tuple[int, ...], ...] = tuple(tuple(a.shape) for a in components)
        offsets = [0]
        for shape in self._shapes:
            offsets.append(offsets[-1] + shape[axis])
        self._offsets: tuple[int, ...] = tuple(offsets)

    @property
    def components(self) -> tuple[Array, ...]:
        """The component arrays, in concatenation order."""
        return self._components

    @property
    def axis(self) -> int:
        """The concatenation axis."""
        return self._axis

    @property
    def ndim(self) -> int:
        return self._components[0].ndim

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._components[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        shape = list(self._components[0].shape)
        shape[self._axis] = self._offsets[-1]
        return tuple(shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def read_only(self) -> bool:
        return True

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return (
            f"<VirtualConcatArray shape={self.shape} dtype={self.dtype} "
            f"axis={self._axis} components={len(self._components)} read-only>"
        )

    def _check_fresh(self) -> None:
        for i, component in enumerate(self._components):
            current = tuple(component.shape)
            if current != self._shapes[i]:
                raise ConcatViewInvalidError(
                    f"component {i} of this VirtualConcatArray changed shape from "
                    f"{self._shapes[i]} to {current} after the view was created; "
                    "the view is invalid because logical indexes along the "
                    "concatenation axis may have shifted. Rebuild the view from "
                    "the current component arrays."
                )

    def __getitem__(self, selection: Any) -> npt.NDArray[Any]:
        return self.get_basic_selection(selection)

    def __setitem__(self, selection: Any, value: Any) -> None:
        raise ValueError("VirtualConcatArray is read-only")

    @property
    def oindex(self) -> _OIndex:
        """Orthogonal (fancy) selection, like zarr.Array.oindex."""
        return _OIndex(self)

    def get_basic_selection(self, selection: Any = Ellipsis) -> npt.NDArray[Any]:
        """Retrieve data using a basic selection (ints and slices)."""
        return self._read(selection, orthogonal=False)

    def get_orthogonal_selection(self, selection: Any) -> npt.NDArray[Any]:
        """Retrieve data using an orthogonal selection.

        Each dimension may be indexed by an int, a slice, a one-dimensional
        integer array, or a Boolean array.  Integer arrays keep their order and
        any duplicate entries, even when they span component boundaries.
        """
        return self._read(selection, orthogonal=True)

    def _read(self, selection: Any, *, orthogonal: bool) -> npt.NDArray[Any]:
        self._check_fresh()
        items = _expand_ellipsis(selection, self.ndim)
        shape = self.shape
        axis = self._axis

        out_shape: list[int] = []
        concat_out_pos: int | None = None
        global_indices: npt.NDArray[np.intp] | None = None
        concat_scalar: int | None = None

        for dim, (item, dim_len) in enumerate(zip(items, shape, strict=True)):
            is_concat = dim == axis
            if _is_int(item):
                idx = int(item)
                if idx < 0:
                    idx += dim_len
                if not 0 <= idx < dim_len:
                    raise IndexError(
                        f"index {int(item)} is out of bounds for axis {dim} with size {dim_len}"
                    )
                items[dim] = idx
                if is_concat:
                    concat_scalar = idx
            elif isinstance(item, slice):
                if is_concat:
                    global_indices = np.arange(*item.indices(dim_len), dtype=np.intp)
                    concat_out_pos = len(out_shape)
                    out_shape.append(len(global_indices))
                else:
                    out_shape.append(len(range(*item.indices(dim_len))))
            else:
                if not orthogonal:
                    raise IndexError(
                        "arrays and other fancy indexes are not supported in basic "
                        "selections; use the .oindex accessor for orthogonal indexing"
                    )
                arr = np.asarray(item)
                if arr.dtype == bool:
                    if arr.ndim != 1 or arr.shape[0] != dim_len:
                        raise IndexError(
                            f"boolean index for axis {dim} does not match the axis size {dim_len}"
                        )
                    if is_concat:
                        global_indices = np.nonzero(arr)[0].astype(np.intp)
                        concat_out_pos = len(out_shape)
                        out_shape.append(len(global_indices))
                    else:
                        items[dim] = arr
                        out_shape.append(int(arr.sum()))
                elif arr.dtype.kind in "iu":
                    normalized = _normalize_int_array(arr, dim_len)
                    if is_concat:
                        global_indices = normalized
                        concat_out_pos = len(out_shape)
                        out_shape.append(len(normalized))
                    else:
                        items[dim] = normalized
                        out_shape.append(len(normalized))
                else:
                    raise IndexError(
                        f"unsupported index of dtype {arr.dtype} for axis {dim}; "
                        "expected an int, slice, integer array, or Boolean array"
                    )

        if concat_scalar is not None:
            comp_idx = int(np.searchsorted(self._offsets, concat_scalar, side="right")) - 1
            items[axis] = concat_scalar - self._offsets[comp_idx]
            return self._components[comp_idx].get_orthogonal_selection(tuple(items))

        if global_indices is None or concat_out_pos is None:  # pragma: no cover
            raise RuntimeError("concatenation axis selection was not resolved")
        out = np.empty(tuple(out_shape), dtype=self.dtype)
        if global_indices.size:
            for comp_idx, component in enumerate(self._components):
                start = self._offsets[comp_idx]
                stop = self._offsets[comp_idx + 1]
                if start == stop:
                    # Empty components contribute no elements and are never read.
                    continue
                mask = (global_indices >= start) & (global_indices < stop)
                if not mask.any():
                    continue
                positions = np.nonzero(mask)[0]
                local = global_indices[positions] - start
                items[axis] = _as_slice_if_regular(local)
                part = component.get_orthogonal_selection(tuple(items))
                out_index: list[Any] = [slice(None)] * len(out_shape)
                out_index[concat_out_pos] = positions
                out[tuple(out_index)] = part
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize the view configuration to a plain dictionary.

        The result contains only JSON-compatible values and no credentials.
        Stores that cannot be described by a plain locator are recorded as
        placeholders; re-supply them through the stores argument of from_dict.
        """
        return {
            "axis": self._axis,
            "dtype": np.dtype(self.dtype).str,
            "components": [
                {
                    "store": _store_to_config(component.store),
                    "path": component.path,
                    "shape": list(component.shape),
                }
                for component in self._components
            ],
        }

    def to_json(self, **json_kwargs: Any) -> str:
        """Serialize the view configuration to a JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        *,
        stores: Sequence[StoreLike] | None = None,
    ) -> VirtualConcatArray:
        """Rebuild a view from a configuration produced by to_dict.

        Parameters
        ----------
        config : dict
            The serialized configuration.
        stores : sequence of StoreLike, optional
            Stores to use for the components, in order.  Use this to re-supply
            stores that need authentication credentials, which are never
            included in the serialized configuration.  When omitted, only
            components whose store was serialized as a plain locator (e.g. a
            local directory) can be reopened.
        """
        from zarr.api.synchronous import open_array
        from zarr.storage import LocalStore

        entries = config["components"]
        if stores is not None:
            stores = list(stores)
            if len(stores) != len(entries):
                raise ValueError(f"expected {len(entries)} stores, got {len(stores)}")

        arrays: list[Array] = []
        for i, entry in enumerate(entries):
            if stores is not None:
                store: Any = stores[i]
            else:
                store_config = entry["store"]
                if store_config["type"] == "local":
                    store = LocalStore(root=store_config["root"], read_only=True)
                else:
                    raise ValueError(
                        f"component {i}: its store "
                        f"({store_config.get('class', 'unknown')}) was serialized "
                        "as a placeholder without credentials; re-supply it via "
                        "the stores argument"
                    )
            array = open_array(store=store, path=entry.get("path", ""), mode="r")
            recorded = entry.get("shape")
            if recorded is not None and tuple(array.shape) != tuple(recorded):
                raise ConcatViewInvalidError(
                    f"component {i} has shape {tuple(array.shape)}, but the "
                    f"serialized configuration recorded shape {tuple(recorded)}; "
                    "the component changed after the view configuration was created"
                )
            arrays.append(array)
        return cls(arrays, axis=config["axis"])

    @classmethod
    def from_json(
        cls,
        data: str,
        *,
        stores: Sequence[StoreLike] | None = None,
    ) -> VirtualConcatArray:
        """Rebuild a view from a JSON string produced by to_json."""
        return cls.from_dict(json.loads(data), stores=stores)


__all__ = ["ConcatViewInvalidError", "VirtualConcatArray"]
