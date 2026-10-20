"""One-pole (complex Lorentzian) fit of swept network parameters.

Near an isolated resonance every entry of S is

    S(f) = N(f) / (f - p),    p = f0 (1 + i / (2 Q)) (or its conjugate),

with N a low-degree polynomial carrying the residue and the background of
the other modes. All entries share the pole p, the complex resonance
frequency of the loaded structure, i.e. the eigenvalue of the same
operator. The fit is the Sanathanan-Koerner iteration: the linearized
problem S (f - p) = N is solved by least squares, reweighted by the
previous 1 / |f - p| until p settles, which is the least-squares fit of
the rational model.
"""

from __future__ import annotations

import dataclasses

import numpy as np

#: Sanathanan-Koerner iterations at most, and the pole change that stops them
#: (relative to the half span of the grid).
MAX_ITERATIONS = 100
POLE_TOLERANCE = 1e-14


@dataclasses.dataclass(frozen=True)
class Resonance:
    """A fitted resonance.

    Attributes:
        pole: The complex resonance frequency (Hz).
        frequency: Its real part (Hz).
        q: Loaded quality factor, Re p / (2 |Im p|).
        residual: Relative RMS misfit of the rational model over the data.
        iterations: Sanathanan-Koerner iterations used.
    """

    pole: complex
    frequency: float
    q: float
    residual: float
    iterations: int


def fit_resonance(
    frequency: np.ndarray, s: np.ndarray, *, degree: int = 2
) -> Resonance:
    """Fit one pole shared by every entry of ``s`` over ``frequency``.

    Args:
        frequency: ``(n_f,)`` Hz.
        s: ``(n_f, ...)`` complex samples; trailing axes are entries that
            share the pole.
        degree: Degree of each entry's numerator polynomial (2: the
            residue plus a linear background).
    """
    f = np.asarray(frequency, float)
    data = np.asarray(s, complex).reshape(len(f), -1)
    center, half = 0.5 * (f[0] + f[-1]), 0.5 * (f[-1] - f[0])
    x = (f - center) / half
    powers = np.vander(x, degree + 1, increasing=True)  # (n_f, d + 1)
    n_f, n_e = data.shape
    m = degree + 1
    pole = complex(x[np.abs(data).max(axis=1).argmax()])
    weight = np.ones(n_f)
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):  # noqa: B007
        # unknowns: p, then each entry's numerator coefficients
        a = np.zeros((n_f * n_e, 1 + n_e * m), complex)
        b = np.zeros(n_f * n_e, complex)
        for e in range(n_e):
            rows = slice(e * n_f, (e + 1) * n_f)
            a[rows, 0] = data[:, e] * weight
            a[rows, 1 + e * m : 1 + (e + 1) * m] = powers * weight[:, None]
            b[rows] = data[:, e] * x * weight
        scale = np.linalg.norm(a, axis=0)
        scale[scale == 0] = 1.0
        solution = np.linalg.lstsq(a / scale, b, rcond=None)[0] / scale
        new = complex(solution[0])
        weight = 1.0 / np.abs(x - new)
        done = abs(new - pole) <= POLE_TOLERANCE * max(1.0, abs(new))
        pole = new
        if done:
            break
    numerators = powers @ solution[1:].reshape(n_e, m).T  # (n_f, n_e)
    model = numerators / (x - pole)[:, None]
    residual = float(np.linalg.norm(model - data) / np.linalg.norm(data))
    physical = center + half * pole
    return Resonance(
        pole=physical,
        frequency=float(physical.real),
        q=float(physical.real / (2 * abs(physical.imag))),
        residual=residual,
        iterations=iterations,
    )
