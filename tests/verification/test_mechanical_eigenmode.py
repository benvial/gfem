r"""Mechanical eigenmodes of a 2D section against closed forms.

- A thickness-extensional plate of a 6mm crystal (c axis along the thickness
  :math:`h`, the faces of constant :math:`x` clamped along :math:`x`, :math:`q =
  0`), open faces: :math:`f_1 = \bar v / (2h)` with :math:`\bar v` the
  piezoelectrically stiffened velocity; both faces grounded:
  :math:`\tan X / X = 1/k_t^2`, :math:`X = \omega h / (2\bar v)`. The 1D
  thickness solution is exact in the finite-element space up to the mesh, so
  the relative error is the mesh's: measured at order 3 for element sizes 0.25,
  0.125 and 0.0625 of the thickness, 8.3e-7, 9.2e-9, 1.1e-10 (open) and 7.7e-7,
  8.6e-9, 1.1e-10 (shorted), a factor 80 to 90 per halving (the asymptotic factor of
  a frequency at order 3 is 2^6 = 64). The gate is 1e-9 on the finest mesh.

Runtime about 15 s on one thread.
"""

import mechanical_cases as mc
import numpy as np
import piezo_cases as pc
import pytest
from scipy.optimize import brentq

import gfem

THICKNESS = pc.PLATE_THICKNESS * 1e-6
CLAMPED = {"left": "x", "right": "x"}


def _frequency(model, target, **problem):
    result = gfem.solve(
        model,
        gfem.MechanicalEigenmode(target, clamped=CLAMPED, **problem),
        options=gfem.SolverOptions(order=3),
    )
    return float(result.frequency[0])


def _exact():
    form = pc.plate_form("TE")
    v, k2 = float(np.real(form["v"])), float(np.real(form["k2"]))
    x = brentq(lambda x: np.tan(x) / x - 1 / k2, 0.1, np.pi / 2 - 1e-9, xtol=1e-15)
    return v / (2 * THICKNESS), x * v / (np.pi * THICKNESS)


@pytest.fixture(scope="module")
def errors(tmp_path_factory):
    directory = tmp_path_factory.mktemp("plate")
    open_f, shorted_f = _exact()
    out = {}
    for maxh in (0.125, 0.0625):
        mesh = mc.piezo_plate_mesh(
            directory / f"{maxh}.msh", pc.PLATE_WIDTH, pc.PLATE_THICKNESS, maxh
        )
        model = mc.piezo_plate_model(mesh, "TE")
        out[maxh] = (
            abs(_frequency(model, open_f) / open_f - 1),
            abs(_frequency(model, shorted_f, ground=["bot", "top"]) / shorted_f - 1),
        )
    return out


@pytest.mark.parametrize("which", [0, 1], ids=["open", "shorted"])
def test_a_thickness_extensional_plate_follows_its_closed_form(errors, which):
    assert errors[0.0625][which] < 1e-9


@pytest.mark.parametrize("which", [0, 1], ids=["open", "shorted"])
def test_the_plate_frequency_converges_at_the_asymptotic_order(errors, which):
    # 2^6 = 64 asymptotically, 81 measured on the finer pair; 30 leaves room
    assert errors[0.125][which] / errors[0.0625][which] > 30
