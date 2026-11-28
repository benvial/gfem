r"""Anchor loss of an in-plane cantilever on a support against the rigid-root constant.

`elastic_pml_cases.anchored_beam` at :math:`L/h = 10, 20, 40`, order 3, with the
support ended by a perfectly matched layer and the search seeded by
:math:`Q_t = 1.583308\,(L/h)^3`. The rigid-root theory gives
:math:`Q \to C\,(L/h)^3`, :math:`C = 1.583308`, with an :math:`O(h/L)` correction
from the root's compliance, so :math:`C(L/h)` falls towards it.

Measured on MN5 at ``maxh`` 0.5, 0.25 and 0.125 um (12 k to 190 k unknowns),
:math:`C` at the finest mesh and its steps from the coarser ones:

====  =========  ===========  ===========
L/h   C (0.125)  step 0.25    step 0.5
====  =========  ===========  ===========
10    1.658055   0.003674     0.011027
20    1.614549   0.001465     0.005163
40    1.597523   0.000855     0.002561
====  =========  ===========  ===========

The finest mesh is converged to 0.1 % of :math:`C` at most (the step is
:math:`O(h)`, so the remaining error is about the last step), far below the
gaps the gates read: the deviations :math:`|C - 1.583308|` are 0.0747, 0.0312
and 0.0142, and :math:`2C_{40} - C_{20}` is 1.5805 (0.18 % below).
"""

import elastic_pml_cases as ec
import numpy as np
import pytest

import gfem

RATIOS = (10.0, 20.0, 40.0)
#: Q of L/h = 10 at ``maxh`` 0.125 um measured on MN5 (job 49486955) with the
#: same idiom: 1658.0549. The platforms differ by the netgen mesh and the
#: solver's round-off only: two meshings of the same geometry gave Q 1647.0285
#: and 1647.0392 at ``maxh`` 0.5 (1e-2 apart), so 0.5 (3e-4 relative, a third
#: of the 1.7 step to the next mesh) holds with margin.
Q10_MN5 = 1658.0549


@pytest.fixture(scope="module")
def constants(tmp_path_factory):
    path = tmp_path_factory.mktemp("anchor")
    found = {}
    for ratio in RATIOS:
        anchor = ec.anchored_beam(path / f"beam{ratio:g}.msh", ratio, 0.125)
        problem = gfem.MechanicalEigenmode(
            anchor.target,
            pml=gfem.PML(["layer"]),
            target_quality_factor=anchor.quality,
        )
        result = gfem.solve(anchor.model, problem, options=gfem.SolverOptions(order=3))
        omega = complex(result.omega[0])
        found[ratio] = omega.real / (2 * omega.imag) / ratio**3
    return found


def test_the_anchor_constant_falls_towards_the_rigid_root_value(constants):
    deviation = [abs(constants[r] - ec.ANCHOR_CONSTANT) for r in RATIOS]
    assert deviation[0] > deviation[1] > deviation[2]
    # the deviations are 0.0747, 0.0312 and 0.0142: each at least 2x the next
    # (the O(h/L) trend halves them), 2.4 and 2.2 measured
    assert deviation[0] > 2 * deviation[1]
    assert deviation[1] > 2 * deviation[2]


def test_the_extrapolated_constant_is_the_rigid_root_one_within_five_percent(
    constants,
):
    # a deviation that falls as h/L extrapolates to 2 C_40 - C_20: 1.5805, 0.18 %
    # from the rigid root; the gate is 5 %
    extrapolated = 2 * constants[40.0] - constants[20.0]
    assert abs(extrapolated / ec.ANCHOR_CONSTANT - 1) < 0.05


def test_the_ratio_ten_value_is_the_one_measured_on_the_large_machine(constants):
    q = constants[10.0] * 10.0**3
    assert np.isfinite(q)
    assert abs(q - Q10_MN5) < 0.5
