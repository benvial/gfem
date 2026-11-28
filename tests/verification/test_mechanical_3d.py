r"""Mechanical eigenmodes of 3D bodies against closed forms, refined, and a cantilever.

The boxes are 1 x 0.8 x 0.6 um on a structured Kuhn grid
(`mechanical_cases.box`) at 5 and 10 cells per um, solved through `gfem.solve`
at order 3 (5.2 k and 40 k unknowns; the unit tier runs the same grids at
order 2). The references are the standing waves of a box with sliding walls
(`mechanical_cases.box_frequencies`, isotropic and cubic silicon) and the Lamb
modes of a plate cut to a box (`mechanical_cases.lamb_box_frequencies`).
Largest relative error of the frequency at 5 and 10 cells per um, and the gates
(each at least twice the error it bounds):

- isotropic roller box, five lowest modes: :math:`6.5\times10^{-5}` and
  :math:`1.1\times10^{-6}`, gates :math:`1.3\times10^{-4}` and
  :math:`2.2\times10^{-6}`;
- cubic silicon roller box, five lowest modes: :math:`7.5\times10^{-5}` and
  :math:`1.3\times10^{-6}`, gates :math:`1.5\times10^{-4}` and
  :math:`2.6\times10^{-6}`;
- Lamb box, A(1,0), A(0,1) and A(1,1): :math:`9.0\times10^{-5}` and
  :math:`1.7\times10^{-6}`, gates :math:`1.8\times10^{-4}` and
  :math:`3.4\times10^{-6}`.

Each mode's error falls at least :math:`2^5` times when the edge is halved
(:math:`2^{5.76}` to :math:`2^{5.92}` measured, the rate of order 3 being
:math:`2^6` for an eigenvalue before the asymptote).

The cantilever of the resonators recipe, 100 x 10 x 2 um clamped at its root,
at order 2 and a 1 um mesh (47 k unknowns), has its first frequency within 1 %
of Euler-Bernoulli's (measured +0.38 %: the 3D clamp against the shear and the
rotary inertia of a beam 50 times longer than thick).
"""

import importlib.util
import sys
from pathlib import Path

import mechanical_cases as mc
import numpy as np
import pytest

import gfem

SIZE = (1.0, 0.8, 0.6)
ISOTROPIC = (mc.LAMBDA + 2 * mc.MU, mc.LAMBDA, mc.MU)
OPTIONS = gfem.SolverOptions(order=3)

# (target Hz, modes, clamped, material, closed form, coarse gate, fine gate)
CASES = {
    "isotropic": (
        3.2e9,
        5,
        mc.ROLLERS,
        mc.material(),
        lambda: mc.box_frequencies(SIZE, ISOTROPIC, mc.RHO, 5),
        1.3e-4,
        2.2e-6,
    ),
    "silicon": (
        4.5e9,
        5,
        mc.ROLLERS,
        mc.SILICON,
        lambda: mc.box_frequencies(SIZE, mc.SI_C, mc.SI_RHO, 5),
        1.5e-4,
        2.6e-6,
    ),
    "lamb": (
        1.6e9,
        3,
        mc.LAMB_ROLLERS,
        mc.material(),
        lambda: mc.lamb_box_frequencies(SIZE, 3),
        1.8e-4,
        3.4e-6,
    ),
}


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    path = tmp_path_factory.mktemp("box")
    return {c: mc.box(path / f"box{c}.msh", SIZE, c) for c in (5, 10)}


@pytest.mark.parametrize("case", list(CASES))
def test_the_box_modes_converge_at_the_rate_of_order_three(meshes, case):
    target, count, clamped, material, closed_form, gate_coarse, gate_fine = CASES[case]
    exact = np.asarray(closed_form())
    errors = []
    for cells in (5, 10):
        model = gfem.Model(meshes[cells], materials={"body": material})
        problem = gfem.MechanicalEigenmode(target, n_modes=count, clamped=clamped)
        found = np.sort(
            np.asarray(gfem.solve(model, problem, options=OPTIONS).frequency)
        )
        errors.append(np.abs(found - exact) / exact)
    coarse, fine = errors
    assert coarse.max() < gate_coarse
    assert fine.max() < gate_fine
    # the measured rates are at least 2^5.76 per mode; the gate is 2^5
    assert (coarse / fine >= 2**5).all()


def _cantilever():
    """The recipe's mesh helper, imported from the docs folder."""
    path = Path(__file__).parents[2] / "docs" / "recipes" / "resonator_meshes.py"
    spec = importlib.util.spec_from_file_location("resonator_meshes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["resonator_meshes"] = module
    spec.loader.exec_module(module)
    return module.cantilever()


def test_the_cantilever_fundamental_is_within_one_percent_of_euler_bernoulli():
    length, thickness, modulus, density = 100e-6, 2e-6, 70e9, 2500.0
    beta = 1.875104068711961
    f_eb = (
        beta**2
        / (2 * np.pi)
        * thickness
        / length**2
        * np.sqrt(modulus / (12 * density))
    )
    material = gfem.Material(
        stiffness=gfem.isotropic_stiffness(modulus, 0.25), mass_density=density
    )
    model = gfem.Model(_cantilever(), materials={"beam": material})
    problem = gfem.MechanicalEigenmode(170e3, clamped=["anchor"])
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=2))
    # measured +0.38 % (beam theory leaves out the 3D clamp, which stiffens the
    # root, against the shear and rotary inertia, which soften it); the gate is 1 %
    assert abs(float(result.frequency[0]) / f_eb - 1) < 1e-2
