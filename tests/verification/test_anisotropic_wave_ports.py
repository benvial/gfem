"""Driven wave ports in anisotropic guides: WR-28 filled with a crystal.

A `MaterialField` with `axes` beside a wave port reads the same tensors as
the `Material` it holds, in the port mode cross-section and in the 3D
operator, so its S equals the material's.
"""

import numpy as np
import pytest
from anisotropy import wr28

import gfem

LENGTH = 10.0  # mm
FREQUENCY = 16e9  # single mode (TE20 propagates at 20 GHz)
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")


@pytest.fixture(scope="module")
def guide_mesh(tmp_path_factory):
    return wr28(tmp_path_factory.mktemp("wr28"), length=LENGTH, cells=(8, 4, 12))


def _s(mesh, fill):
    model = gfem.Model(
        mesh,
        materials={"fill": fill},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    return np.asarray(
        gfem.solve(model, gfem.Driven(FREQUENCY), options=OPTIONS).s.values
    )


@pytest.mark.parametrize(
    "material",
    [
        pytest.param(
            gfem.Material(eps_r=(2.0, 4.0, 3.0), tan_delta=(1e-3, 2e-3, 0.0)),
            id="eps",
        ),
        pytest.param(
            gfem.Material(eps_r=(2.0, 4.0, 3.0), mu_r=(1.0, 2.0, 1.5)),
            id="eps-mu",
            marks=pytest.mark.skip(
                reason="an anisotropic mu_r at a wave port needs ticket 05's "
                "wave-port Robin tensor, not on this branch"
            ),
        ),
    ],
)
def test_a_uniform_field_beside_the_ports_gives_its_material_s_s(guide_mesh, material):
    expected = _s(guide_mesh, material)
    s = _s(guide_mesh, gfem.MaterialField.uniform("fill", material))
    # measured on MN5: 6.9e-15 (the port frame rounds the field's tensors
    # differently)
    np.testing.assert_allclose(s, expected, rtol=0, atol=2e-14)
