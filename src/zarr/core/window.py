from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

type WindowExtensionMode = Literal["constant", "reflect", "wrap"]
"""How window coordinates outside the array bounds are resolved."""

_WINDOW_EXTENSION_MODES: tuple[WindowExtensionMode, ...] = ("constant", "reflect", "wrap")

type Window = slice | tuple[slice, ...]


@dataclass(frozen=True)
class WindowAxisPlan:
    """Plan for resolving one axis of a windowed read.

    Attributes
    ----------
    nitems
        Number of output elements along this axis (the window length).
    read_start, read_stop
        Bounding box, in absolute array coordinates, of the source data that
        must be read from the store along this axis.
    out_start
        Offset into the output axis where the read data is placed. Only
        meaningful when ``take_indices`` is None.
    take_indices
        For "reflect" and "wrap" modes, indices (relative to ``read_start``)
        that map each output position onto the read bounding box. None when the
        read data maps contiguously onto the output (the "constant" mode, and
        in-bounds degenerate cases of the other modes).
    """

    nitems: int
    read_start: int
    read_stop: int
    out_start: int
    take_indices: npt.NDArray[np.intp] | None

    @property
    def read_len(self) -> int:
        return self.read_stop - self.read_start

    @property
    def covers_window(self) -> bool:
        """True if the read data fills the whole window along this axis."""
        return self.take_indices is None and self.read_len == self.nitems and self.out_start == 0


def normalize_window(window: Window, ndim: int) -> tuple[slice, ...]:
    """Normalize a window specification to a tuple of unit-stride slices, one per axis.

    Unlike regular indexing, negative slice bounds are *not* interpreted relative to
    the end of the array; they address coordinates before the start of the array.
    """
    if isinstance(window, slice):
        window = (window,)
    elif not isinstance(window, tuple):
        raise TypeError(f"window must be a slice or a tuple of slices, got {type(window)!r}")
    if len(window) > ndim:
        raise IndexError(f"too many indices for window; expected at most {ndim}, got {len(window)}")
    normalized: list[slice] = []
    for item in window:
        if not isinstance(item, slice):
            raise TypeError(
                "window entries must be slices; "
                f"fancy indexing is not supported, got {type(item)!r}"
            )
        if item.step not in (None, 1):
            raise ValueError(f"only unit-stride windows are supported, got step={item.step!r}")
        normalized.append(item)
    # pad missing axes with full-axis windows
    normalized.extend([slice(None)] * (ndim - len(normalized)))
    return tuple(normalized)


def normalize_window_modes(
    mode: WindowExtensionMode | tuple[WindowExtensionMode, ...] | list[WindowExtensionMode],
    ndim: int,
) -> tuple[WindowExtensionMode, ...]:
    """Normalize the per-axis extension mode specification."""
    if isinstance(mode, str):
        modes: list[WindowExtensionMode] = [mode] * ndim
    else:
        modes = list(mode)
        if len(modes) != ndim:
            raise ValueError(f"expected {ndim} extension modes (one per axis), got {len(modes)}")
    for m in modes:
        if m not in _WINDOW_EXTENSION_MODES:
            raise ValueError(
                "invalid extension mode {m!r}; expected one of {_WINDOW_EXTENSION_MODES}"
            )
    return tuple(modes)


def _reflect_indices(positions: npt.NDArray[np.intp], dim_len: int) -> npt.NDArray[np.intp]:
    """Map (possibly out-of-bounds) coordinates to source coordinates by reflection.

    The edge element is not repeated, matching ``numpy.pad(..., mode="reflect")``.
    """
    if dim_len == 1:
        return np.zeros_like(positions)
    period = 2 * dim_len - 2
    r = positions % period
    return np.where(r < dim_len, r, period - r)


def plan_window_axis(dim_sel: slice, dim_len: int, mode: WindowExtensionMode) -> WindowAxisPlan:
    """Compute the read plan for one axis of a window.

    Window coordinates are absolute: coordinate 0 is the first element of the
    array, negative coordinates lie before the array start, and coordinates
    >= ``dim_len`` lie past the array end.
    """
    start = 0 if dim_sel.start is None else dim_sel.start
    stop = dim_len if dim_sel.stop is None else dim_sel.stop
    nitems = stop - start
    if nitems < 0:
        raise ValueError(
            f"window stop ({stop}) must be greater than or equal to window start ({start})"
        )
    if nitems == 0:
        return WindowAxisPlan(0, 0, 0, 0, None)

    if mode == "constant":
        read_start = max(start, 0)
        read_stop = min(stop, dim_len)
        if read_stop < read_start:
            read_stop = read_start
        return WindowAxisPlan(nitems, read_start, read_stop, read_start - start, None)

    # reflect / wrap modes need actual source data for every output position
    if dim_len == 0:
        raise ValueError(
            f"cannot extend an empty axis with mode {mode!r}; "
            "use mode 'constant' or an empty window instead"
        )
    positions = np.arange(start, stop, dtype=np.intp)
    if mode == "wrap":
        source = positions % dim_len
    else:
        source = _reflect_indices(positions, dim_len)
    read_start = int(source.min())
    read_stop = int(source.max()) + 1
    take_indices = source - read_start
    out_start = read_start - start
    if read_stop - read_start == nitems and np.array_equal(
        take_indices, np.arange(nitems, dtype=np.intp)
    ):
        # the window maps contiguously onto the array; no take needed
        return WindowAxisPlan(nitems, read_start, read_stop, out_start, None)
    return WindowAxisPlan(nitems, read_start, read_stop, out_start, take_indices)


def plan_window(
    window: Window,
    shape: tuple[int, ...],
    mode: WindowExtensionMode
    | tuple[WindowExtensionMode, ...]
    | list[WindowExtensionMode] = "constant",
) -> tuple[WindowAxisPlan, ...]:
    """Compute per-axis read plans for a windowed read."""
    slices = normalize_window(window, len(shape))
    modes = normalize_window_modes(mode, len(shape))
    return tuple(
        plan_window_axis(dim_sel, dim_len, dim_mode)
        for dim_sel, dim_len, dim_mode in zip(slices, shape, modes, strict=True)
    )
