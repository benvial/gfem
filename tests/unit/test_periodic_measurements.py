"""Measurements on periodic models: `gfem.power_flow` and `interface_participation`.

A cell periodic along x and y between PEC plates, fed by a lumped port on
an interior sheet: a lossless "core" (a slab around the port and a bottom
layer under the whole cell) and a lossy rest. The power leaving any side
that holds the port and only lossless elements is the power the lossy rest
absorbs, to solver precision, both for plain periodicity and under a Bloch
wave vector. The side surfaces cross the pairs' surfaces, and one side
touches x_min but not x_max, so the measurement must read edges and DOFs
across the pairs (ticket 10 Q4, Q6).
"""

import math
import re

import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem._engine.driven import DrivenOperator

MM = 1e-3
AX, AY, H = 8.0, 2.0, 2.0  # mm
SHEET = 3.0  # the port sheet, mm
SLAB = (2.0, 4.0)  # the lossless slab around the port, mm
BOTTOM = 1.0  # the lossless bottom layer, mm
EPS, TAN_DELTA = 2.0, 0.05
FREQUENCY = 20e9
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
#: The Bloch wave vector, rad/m.
K = (0.3 * math.pi / (AX * MM), 0.2 * math.pi / (AY * MM), 0.0)
#: Relative error of the power identity (ticket 06).
POWER_RTOL = 1e-12

# A Bloch driven solve is right once ticket 03 declares the operator general.
BLOCH = pytest.param(
    K,
    id="bloch",
    marks=pytest.mark.skipif(
        not hasattr(DrivenOperator, "symmetry"),
        reason="Bloch driven solves need the general operator of ticket 03",
    ),
)


def _in_core(x, z):
    return ((x > SLAB[0]) & (x < SLAB[1])) | (z < BOTTOM)


@pytest.fixture(scope="module")
def cell(tmp_path_factory):
    edges = tuple(
        np.linspace(0.0, s, n + 1) for s, n in zip((AX, AY, H), (8, 2, 4), strict=True)
    )
    raw = sm.periodic_cell(
        edges,
        lambda c: "core" if _in_core(c[0], c[2]) else "lossy",
        sheets=(("sheet", SHEET),),
    )
    path = sm.write_msh22(tmp_path_factory.mktemp("cell") / "cell.msh", raw)
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(eps_r=EPS),
            "lossy": gfem.Material(eps_r=EPS, tan_delta=TAN_DELTA),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
        ports=[gfem.LumpedPort("p1", "sheet", "+z", impedance=50.0)],
    )


def _field(model, wave_vector):
    problem = gfem.Driven(
        frequency=FREQUENCY, excite="p1", keep_fields=True, wave_vector=wave_vector
    )
    return gfem.solve(model, problem, options=OPTIONS).e_field(FREQUENCY)


def _tetrahedron_rule(n):
    """Collapsed Gauss rule on the unit tetrahedron, exact to degree 2n - 3."""
    t, w = np.polynomial.legendre.leggauss(n)
    t, w = (t + 1) / 2, w / 2
    u, v, s = (a.ravel() for a in np.meshgrid(t, t, t, indexing="ij"))
    weight = np.einsum("i,j,k->ijk", w, w, w).ravel() * (1 - u) ** 2 * (1 - v)
    points = np.stack([u, v * (1 - u), s * (1 - u) * (1 - v)], axis=1)
    return points, weight


def _absorbed(e, mesh):
    """``1/2 ω eps0 eps_r tanδ ∫ |E|^2`` over the lossy region, from point values."""
    ref, weight = _tetrahedron_rule(4)  # |E|^2 of order 2: degree 4
    p = mesh.nodes[mesh.tetrahedra[mesh.elements("lossy")]]  # (t, 4, 3)
    jac = np.transpose(p[:, 1:] - p[:, :1], (0, 2, 1))  # (t, 3, 3)
    points = p[:, :1] + np.einsum("tij,qj->tqi", jac, ref)
    values = np.asarray(e.at(points.reshape(-1, 3))).reshape(*points.shape[:2], 3)
    density = (np.abs(values) ** 2).sum(axis=-1)
    total = np.abs(np.linalg.det(jac)) @ (density @ weight)
    omega = 2 * np.pi * FREQUENCY
    return 0.5 * omega * const.epsilon_0 * EPS * TAN_DELTA * total


@pytest.mark.parametrize("wave_vector", [pytest.param(None, id="plain"), BLOCH])
def test_power_flow_on_a_periodic_cell_is_the_absorbed_power(cell, wave_vector):
    e = _field(cell, wave_vector)
    absorbed = _absorbed(e, cell.mesh)
    assert absorbed > 0.01

    # the bottom layer's top face crosses both pairs' surfaces
    crossing = float(gfem.power_flow(e, "core", model=cell))
    assert crossing == pytest.approx(absorbed, rel=POWER_RTOL)

    # the strip under x > SLAB[1] left out: the side touches x_min, not x_max
    def touching(c):
        x, z = c[:, 0] / MM, c[:, 2] / MM
        return ((x > SLAB[0]) & (x < SLAB[1])) | ((z < BOTTOM) & (x < SLAB[1]))

    assert float(gfem.power_flow(e, touching, model=cell)) == pytest.approx(
        absorbed, rel=POWER_RTOL
    )
    # the complement takes the same power in
    back = float(gfem.power_flow(e, "lossy", model=cell))
    assert back == pytest.approx(-absorbed, rel=POWER_RTOL)


def test_interface_participation_refuses_a_paired_surface(cell):
    e = _field(cell, None)
    modes = gfem.EigenmodeResult(
        omega=np.array([2 * np.pi * FREQUENCY + 0j]),
        error=np.zeros(1),
        e_fields=(e,),
        materials=cell.materials,
    )
    for name, pair in (
        ("x_max", "('x_min', 'x_max')"),
        ("y_min", "('y_min', 'y_max')"),
    ):
        message = (
            f"'{name}' is a surface of periodic pair {pair}, which is interior "
            "to the periodic cell: it holds no interface layer."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            gfem.interface_participation(modes, name, 1e-9, 10.0)
    # an unpaired surface still measures
    p = gfem.interface_participation(modes, "z_min", 1e-9, 10.0, kind="MS")
    assert np.isfinite(np.asarray(p.participation)).all()
