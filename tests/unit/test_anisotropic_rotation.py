"""A model turned as a whole by a rotation gives the same results.

Copy B of a model has its nodes turned by a generic rotation Q (0.7 rad
about (1, 2, 3)), its port directions turned with them, and its materials
given as global matrices ``Q C Q^T`` (the matrix input, which derives the
principal axes again). A rigid rotation of the node array gives the same
discrete problem, so S, omega and C agree to round-off: every tensor term
is exercised at faces and axes along no coordinate direction (the tensor
mass, curl-curl and conductivity pieces, the first-order absorbing
admittance ``T^T Y T`` beside the crystal, the lumped ports, the
electrostatic permittivity).
"""

import anisotropy
import numpy as np
import pytest
from gradient_cases import STACK_PROBLEM, capacitor, layered_line

import gfem

Q = anisotropy.rodrigues((1, 2, 3), 0.7)
OPTIONS = gfem.SolverOptions(linear_solver="direct", eigen_tolerance=1e-12)

#: A tilted biaxial crystal with per-axis eps_r, mu_r, tan_delta and conductivity.
CRYSTAL = gfem.Material(
    eps_r=(3.0, 5.0, 6.0),
    mu_r=(1.5, 1.0, 1.2),
    tan_delta=(0.01, 0.02, 0.03),
    conductivity=(0.02, 0.05, 0.03),
    axes=((1 / 3, 2 / 3, 2 / 3), (2 / 3, 1 / 3, -2 / 3), (2 / 3, -2 / 3, 1 / 3)),
)
#: A crystal on the coordinate axes: diagonal in A, a full matrix in B.
UNIAXIAL = gfem.Material(eps_r=(2.0, 2.0, 3.0), tan_delta=(1e-3, 1e-3, 2e-3))


def _pair(model, turned, materials):
    """``model`` with ``materials``; ``turned`` with them as matrices turned by Q."""
    return model.replace(materials=materials), turned.replace(
        materials={r: anisotropy.rotated(m, Q) for r, m in materials.items()}
    )


@pytest.fixture(scope="module")
def lines(tmp_path_factory):
    return _pair(
        layered_line(tmp_path_factory.mktemp("line"), absorbing_order=1),
        layered_line(
            tmp_path_factory.mktemp("rotated_line"), absorbing_order=1, rotation=Q
        ),
        {"sub": CRYSTAL, "cover": UNIAXIAL},
    )


def test_a_rotated_line_gives_the_same_s_parameters(lines):
    line, turned = lines
    problem = gfem.Driven(frequency=[10e9, 20e9])
    s = np.asarray(gfem.solve(line, problem, options=OPTIONS).s.values)
    s_turned = np.asarray(gfem.solve(turned, problem, options=OPTIONS).s.values)
    # measured on MN5: max |dS| 2.4e-15 of max |S| (round-off: the bound
    # leaves room for other nodes' rounding)
    np.testing.assert_allclose(s_turned, s, rtol=0, atol=2e-14 * np.abs(s).max())


def test_a_rotated_line_gives_the_same_eigenfrequency(lines):
    line, turned = lines
    problem = gfem.Eigenmode(target=15e9)
    omega = complex(gfem.solve(line, problem, options=OPTIONS).omega[0])
    omega_turned = complex(gfem.solve(turned, problem, options=OPTIONS).omega[0])
    # measured on MN5: 7.2e-16 relative (a strongly damped mode: the side
    # walls absorb the line's field, Im omega > Re omega)
    np.testing.assert_allclose(omega_turned, omega, rtol=1e-14)


def test_a_rotated_capacitor_gives_the_same_capacitance(tmp_path_factory):
    plate, turned = _pair(
        capacitor(tmp_path_factory.mktemp("plate")),
        capacitor(tmp_path_factory.mktemp("rotated_plate"), rotation=Q),
        {"oxide": CRYSTAL, "nitride": UNIAXIAL},
    )
    c = np.asarray(gfem.solve(plate, STACK_PROBLEM, options=OPTIONS).capacitance)
    c_turned = np.asarray(
        gfem.solve(turned, STACK_PROBLEM, options=OPTIONS).capacitance
    )
    # measured on MN5: max |dC| 5.9e-16 of max |C|
    np.testing.assert_allclose(c_turned, c, rtol=0, atol=1e-14 * np.abs(c).max())
