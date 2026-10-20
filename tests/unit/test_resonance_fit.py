"""The one-pole (complex Lorentzian) resonance fit used on swept S-parameters."""

import numpy as np
import pytest
from resonance import fit_resonance


def _response(frequency, pole, residues, background):
    """S entries: one pole plus a slowly varying background, (n_f, 2, 2)."""
    x = frequency[:, None, None]
    slope, offset = background
    return residues / (x - pole) + offset + slope * (x - frequency.mean()) / 1e9


@pytest.mark.parametrize(
    ("f0", "q"),
    [(16.114e9, 85.1), (21.7958e9, 2392.0)],
)
def test_fit_resonance_recovers_frequency_and_loaded_q(f0, q):
    width = f0 / q  # full linewidth
    frequency = np.linspace(f0 - 5 * width, f0 + 5 * width, 201)
    pole = f0 + 1j * f0 / (2 * q)
    rng = np.random.default_rng(0)
    residues = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) * width
    background = (
        rng.normal(size=(2, 2)) * 1e-3 + 0.2j,
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)),
    )
    s = _response(frequency, pole, residues, background)

    fit = fit_resonance(frequency, s)

    assert fit.frequency == pytest.approx(f0, rel=1e-10)
    assert fit.q == pytest.approx(q, rel=1e-8)
    assert fit.residual < 1e-10


def test_fit_resonance_pole_sign_does_not_matter():
    f0, q = 10e9, 300.0
    frequency = np.linspace(f0 * 0.98, f0 * 1.02, 201)
    s = 0.1 / (frequency - (f0 - 1j * f0 / (2 * q))) * 1e8 + 0.5
    fit = fit_resonance(frequency, s)
    assert fit.frequency == pytest.approx(f0, rel=1e-10)
    assert fit.q == pytest.approx(q, rel=1e-8)


def test_fit_resonance_reports_a_poor_fit():
    frequency = np.linspace(9e9, 11e9, 201)
    two_poles = 1e7 / (frequency - 9.6e9 - 1e7j) + 1e7 / (frequency - 10.4e9 - 1e7j)
    assert fit_resonance(frequency, two_poles).residual > 1e-2
