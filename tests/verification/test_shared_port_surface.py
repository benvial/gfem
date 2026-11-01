"""A shared port surface couples TE10 to TE20 as mode matching says.

A reduced-height rectangular guide (22.86 x 5 mm: TE01 is cut off at 30
GHz, so only TE_n0 modes take part) holds an eps_r = 4 slab against one
wall, x in [0, 8] mm, 6 mm long, between 40 mm empty leads. The slab is
asymmetric in x, so it converts TE10 into TE20; at 14.5 and 16 GHz exactly
TE10 and TE20 propagate in the leads (TE20 is cut off at 13.1 GHz, TE30 at
19.7 GHz). Each end is one shared port surface carrying both modes, its
reference planes moved to the junctions by each mode's own offset.

The reference is a sine-basis mode matching (`mode_matching`), exact up to
truncation (it moves by 4e-6 from 160 to 320 sine modes). The slab is
smooth at the metal, so the mesh error has no edge singularity to fight;
TE30's evanescent tail is down to 7e-5 at the ports.
"""

import numpy as np
import pytest
import structured_meshes as sm
from mode_matching import s_matrix

import gfem

A, B, SLAB, INSERT, LEAD = 22.86, 5.0, 8.0, 6.0, 40.0  # mm
EPS = 4.0
F = (14.5e9, 16e9)
PORTS = ("in_TE10", "in_TE20", "out_TE10", "out_TE20")
END = 2 * LEAD + INSERT


def _mesh(tmp_path):
    def lines(*pieces):
        return np.unique(np.concatenate([np.linspace(*p) for p in pieces]))

    edges = (
        lines((0, SLAB, 9), (SLAB, A, 16)),  # cells of about 1 mm
        np.linspace(0, B, 3),
        lines((0, LEAD, 21), (LEAD, LEAD + INSERT, 7), (LEAD + INSERT, END, 21)),
    )
    eps = 1e-9

    def volume(c):
        return "slab" if c[0] < SLAB and LEAD < c[2] < LEAD + INSERT else "air"

    def surface(c):
        x, y, z = c
        if abs(z) < eps:
            return "port1"
        if abs(z - END) < eps:
            return "port2"
        if min(x, A - x, y, B - y) < eps:
            return "walls"
        return ""

    path = sm.write_msh22(
        tmp_path / "insert.msh", sm.structured(edges, volume, surface)
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _model(mesh):
    # paths along y at x = a/4, where sin(m pi x / a) > 0 for m = 1, 2: the
    # reference's polarity, so every entry compares as a complex number
    def port(name):
        surface, z = ("port1", 0.0) if name.startswith("in") else ("port2", END)
        x = A / 4 * 1e-3
        return gfem.WavePort(
            name,
            surface,
            mode=int(name[-2]),
            offset=LEAD * 1e-3,
            voltage_path=((x, 0.0, z * 1e-3), (x, B * 1e-3, z * 1e-3)),
        )

    return gfem.Model(
        mesh,
        materials={"slab": gfem.Material(eps_r=EPS), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[port(name) for name in PORTS],
    )


@pytest.fixture(scope="module")
def insert(tmp_path_factory):
    model = _model(_mesh(tmp_path_factory.mktemp("insert")))
    results = {}
    for order in (2, 3):
        options = gfem.SolverOptions(order=order, linear_solver="direct")
        problem = gfem.Driven(F, keep_fields=order == 3)
        results[order] = gfem.solve(model, problem, options=options)
    return model, results


@pytest.fixture(scope="module")
def reference():
    return np.array(
        [
            s_matrix(f, a=A * 1e-3, slab=SLAB * 1e-3, eps=EPS, length=INSERT * 1e-3)
            for f in F
        ]
    )


def test_a_slab_insert_converts_te10_to_te20_as_mode_matching_says(insert, reference):
    _, results = insert
    errors = [
        np.abs(np.asarray(results[order].s.values) - reference).max()
        for order in (2, 3)
    ]
    print(f"\nslab insert: max |S - S_ref| at orders 2, 3: {errors}")  # noqa: T201
    assert errors[0] > errors[1]
    assert errors[1] < 2e-3
    s = np.asarray(results[3].s.values)
    assert np.abs(s[:, PORTS.index("out_TE20"), PORTS.index("in_TE10")]).min() > 0.5


def test_a_slab_insert_is_lossless_and_reciprocal(insert):
    _, results = insert
    s = np.asarray(results[3].s.values)
    # every propagating mode is requested: S is unitary
    unitarity = np.einsum("fki,fkj->fij", s.conj(), s) - np.eye(len(PORTS))
    print(  # noqa: T201
        f"\nslab insert: max |S^H S - I| {np.abs(unitarity).max():.2e}, "
        f"max |S - S^T| {np.abs(s - np.swapaxes(s, 1, 2)).max():.2e}"
    )
    assert np.abs(unitarity).max() < 2e-3
    assert np.abs(s - np.swapaxes(s, 1, 2)).max() < 1e-8


def test_the_power_through_each_lead_is_the_modal_power(insert):
    model, results = insert
    result = results[3]
    s = np.asarray(result.s.values)
    col = PORTS.index("in_TE10")
    errors = []
    for k, f in enumerate(F):
        field = result.e_field(f, "in_TE10")
        # through a plane 20 mm into the out lead: what both modes carry out
        out = float(gfem.power_flow(field, lambda c: c[:, 2] < 0.066, model=model))
        # through a plane 20 mm into the in lead: incident minus reflected
        into = float(gfem.power_flow(field, lambda c: c[:, 2] < 0.020, model=model))
        errors += [
            abs(out - (np.abs(s[k, 2:, col]) ** 2).sum()),
            abs(into - (1 - (np.abs(s[k, :2, col]) ** 2).sum())),
        ]
    print(f"\nslab insert: power_flow - modal power {errors}")  # noqa: T201
    assert max(errors) < 5e-4
