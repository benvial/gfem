r"""The elastic layer on 3D bodies against the leaky column's closed form.

The column is the matched layer-on-substrate of `elastic_pml_cases`, whose
thickness-shear mode :math:`u_x(z)` has the exact complex frequency
:math:`\omega_1 = (c_1/h)[\pi + i\,\mathrm{artanh}\,\zeta]` (Q = 157.07). Kuhn
tetrahedra, so every platform solves the same mesh and only round-off differs.

- The column stretched along z, order 2, :math:`h = 0.2` and :math:`0.1` um
  (7.3 k and 53 k unknowns): relative error of :math:`\omega` :math:`9.22\times10^{-5}`
  and :math:`6.28\times10^{-6}`, a rate of :math:`2^{3.88}`. Gates: the fine
  error below :math:`1.3\times10^{-5}` (twice the measured) and a rate of at
  least :math:`2^3`.
- The same column with a layer 0.6 um thick on each x and y side and in the
  corners (25 layer regions, stretched along x, y and z in the corners), order 4,
  :math:`h = 0.3` um (133 k unknowns), with a complex shift (a rough estimate
  of Q): the mode does not depend on x or y, so the side layers leave the
  column's mode, to the interpolation of the stretch on tetrahedra. Measured
  :math:`1.39\times10^{-8}` (the column alone at the same order and mesh:
  :math:`1.95\times10^{-8}`); the gate is :math:`3\times10^{-8}`.
"""

import elastic_pml_cases as ec
import numpy as np
import pytest

import gfem

#: The quality factor the complex shift expects: a rounded estimate of 157.07.
Q_SHIFT = 150.0


def _error(column, order, **kwargs):
    problem = gfem.MechanicalEigenmode(
        column.target, pml=column.pml, clamped=column.clamped, **kwargs
    )
    result = gfem.solve(column.model, problem, options=gfem.SolverOptions(order=order))
    assert complex(result.omega[0]).imag > 0  # a radiating mode decays
    return abs(complex(result.omega[0]) / column.omega - 1)


def test_a_column_stretched_along_z_converges_at_the_rate_of_the_order(tmp_path):
    coarse = _error(ec.leaky_column(tmp_path / "h2.msh", 0.2, dim=3), 2)
    fine = _error(ec.leaky_column(tmp_path / "h1.msh", 0.1, dim=3), 2)
    # measured 9.22e-5 and 6.28e-6; the gate is twice the fine error
    assert fine < 1.3e-5
    # the measured rate is 2^3.88 (the eigenvalue of order 2 converges as h^4);
    # the gate is 2^3
    assert coarse / fine >= 2**3


@pytest.mark.parametrize("side_layers", [False, True], ids=["column", "side layers"])
def test_side_and_corner_layers_leave_the_columns_mode(tmp_path, side_layers):
    column = ec.leaky_column(
        tmp_path / "sides.msh", 0.3, dim=3, side_layers=side_layers
    )
    assert len(column.pml.regions) == (25 if side_layers else 1)
    error = _error(column, 4, target_quality_factor=Q_SHIFT)
    # measured 1.95e-8 for the column and 1.39e-8 with the side layers; the
    # discretization error of the mode, which the side layers do not raise
    assert error < 3e-8
    assert np.isfinite(error)
