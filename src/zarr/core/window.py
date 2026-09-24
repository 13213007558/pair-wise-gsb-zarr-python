from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

type WindowMode = Literal["constant", "reflect", "wrap"]
"""How window positions outside the array bounds are resolved.

- "constant": out-of-bounds positions are filled with a constant value.
- "reflect": positions are mirrored about the array edges without repeating
  the edge element (like numpy.pad with mode="reflect").
- "wrap": positions repeat periodically with the axis length (like numpy.pad
  with mode="wrap").
"""

_WINDOW_MODES: tuple[WindowMode, ...] = ("constant", "reflect", "wrap")


class AxisRun(NamedTuple):
    """A maximal run of consecutive window positions along one axis.

    A run either maps to a contiguous, monotonic (step +1 or -1) slice of the
    source array, or denotes positions outside the array that must be filled
    with the constant value (src_start is None).
    """

    out_start: int
    """First position of the run in window (output) coordinates."""
    length: int
    """Number of window positions covered by the run."""
    src_start: int | None
    """Source index of the first position of the run, or None for a fill run."""
    src_step: int
    """Step between consecutive source indices of the run (+1 or -1)."""


def parse_window_modes(
    mode: WindowMode | Sequence[WindowMode], ndim: int
) -> tuple[WindowMode, ...]:
    """Normalize a single mode or per-axis modes to a tuple of length ndim."""
    if isinstance(mode, str):
        if mode not in _WINDOW_MODES:
            raise ValueError(f"Invalid window mode {mode!r}; expected one of {_WINDOW_MODES}.")
        return (mode,) * ndim
    modes = tuple(mode)
    if len(modes) != ndim:
        raise ValueError(
            f"Expected {ndim} window mode(s), one per array dimension, got {len(modes)}."
        )
    for m in modes:
        if m not in _WINDOW_MODES:
            raise ValueError(f"Invalid window mode {m!r}; expected one of {_WINDOW_MODES}.")
    return modes


def reflect_index(index: int, axis_len: int) -> int:
    """Map index into [0, axis_len) with reflect (half-sample) semantics.

    The edge element is not repeated, matching numpy.pad mode="reflect".
    Requires axis_len >= 2; arbitrarily large (positive or negative)
    out-of-bounds offsets are reduced modulo the reflection period
    2 * axis_len - 2.
    """
    period = 2 * axis_len - 2
    j = index % period
    return j if j < axis_len else period - j


def axis_runs(
    origin: int, length: int, axis_len: int, mode: WindowMode, axis: int
) -> list[AxisRun]:
    """Decompose one window axis into runs of source slices or fill regions.

    Parameters
    ----------
    origin : int
        The first window position, in array coordinates. May be negative or
        beyond the end of the axis.
    length : int
        The number of window positions along this axis. Must be >= 0.
    axis_len : int
        The length of the array along this axis.
    mode : WindowMode
        How out-of-bounds positions are resolved.
    axis : int
        The axis index, used for error messages.

    Returns
    -------
    list of AxisRun
        Runs covering [0, length) in window coordinates, in order.
    """
    if length < 0:
        raise ValueError(f"Window shape entries must be >= 0, got {length} for axis {axis}.")
    if length == 0:
        return []
    if mode == "constant":
        runs: list[AxisRun] = []
        stop = origin + length
        if origin < 0:
            pre = min(stop, 0) - origin
            runs.append(AxisRun(0, pre, None, 1))
        lo = max(origin, 0)
        hi = min(stop, axis_len)
        if lo < hi:
            runs.append(AxisRun(lo - origin, hi - lo, lo, 1))
        if stop > axis_len:
            post_start = max(origin, axis_len)
            runs.append(AxisRun(post_start - origin, stop - post_start, None, 1))
        return runs
    # reflect and wrap sample actual array elements, so the axis must be non-empty
    if axis_len == 0:
        raise ValueError(
            f"Cannot use window mode {mode!r} on axis {axis} with length 0; "
            "there are no elements to sample. Use mode='constant' instead."
        )
    if axis_len == 1:
        # the single element is the only possible source for every position
        return [AxisRun(0, length, 0, 1)]
    if mode == "reflect":
        mapped = [reflect_index(i, axis_len) for i in range(origin, origin + length)]
    else:
        mapped = [i % axis_len for i in range(origin, origin + length)]
    # split the mapped indices into maximal contiguous runs, i.e. runs where
    # consecutive source indices differ by exactly +1 or exactly -1; a wrap
    # boundary (a jump of 1 - axis_len) always starts a new run
    runs = []
    start = 0
    while start < length:
        step = 1
        end = start + 1
        if end < length:
            delta = mapped[start + 1] - mapped[start]
            if delta in (1, -1):
                step = delta
                end = start + 2
                while end < length and mapped[end] - mapped[end - 1] == delta:
                    end += 1
        runs.append(AxisRun(start, end - start, mapped[start], step))
        start = end
    return runs


def iter_axis_run_products(axis_runs: Sequence[Sequence[AxisRun]]) -> Iterator[tuple[AxisRun, ...]]:
    """Iterate the Cartesian product of per-axis runs.

    Each yielded tuple contains one AxisRun per axis and describes a single
    hyper-rectangular block of the output window.
    """
    return itertools.product(*axis_runs)
