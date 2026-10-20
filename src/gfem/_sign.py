"""The reference coefficient of a mode's sign (or phase) convention."""

from __future__ import annotations

import numpy as np

# Coefficients this close to the largest magnitude (relative) tie with it.
TIE_RTOL = 1e-12


def largest_entry(x: np.ndarray) -> int:
    """Index of the largest-magnitude entry of ``x``, ties to the lowest index.

    Symmetric structures give modes whose largest coefficients agree in
    magnitude up to rounding, with opposite signs; a plain ``argmax`` then
    picks one by rounding, so the sign of the mode would depend on the CPU.
    Every entry within `TIE_RTOL` of the maximum ties, and the lowest index
    among them wins.
    """
    magnitude = np.abs(x)
    return int(np.argmax(magnitude >= (1 - TIE_RTOL) * magnitude.max()))
