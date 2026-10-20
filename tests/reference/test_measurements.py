"""Measurements on the QPDK coupled CPW resonator (``eigenmode/r20``).

The resonator is a PEC sheet (``SUPERCONDUCTOR_pec``) on anisotropic
sapphire under vacuum; the gaps of the CPW are the sapphire-vacuum
interface. The three thin-layer losses of Wenner et al. live there:
metal-air on the sheet's vacuum face, metal-substrate on its sapphire face,
substrate-air on the gaps.

- Palace's reference runs configure no ``Dielectric`` postprocessing, so
  there is no Palace interface participation to compare with: the parity
  test runs only when a rerun adds it (``config.json`` with
  ``Boundaries.Postprocessing.Dielectric`` and ``surface-Q.csv``), and
  skips otherwise. Palace bisects elements next to the sheet (see
  ``test_eigenmode.py``), so that comparison is loose.
- Independent check: each participation is recomputed from `Field.at`
  probes (point location, not the face kernel's element choice) at a
  6-point degree-4 rule on every triangle, each point moved 1e-7 of the
  triangle size into the wanted side, with the sapphire tensor applied by
  hand for metal-substrate.
- Golden baselines: gfem's own values on this mesh, for regressions.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.constants as const

import gfem

THREADS = 4
T, EPS_LAYER, TAN_DELTA = 3e-9, 10.0, 2e-3
SHEET, GAPS = "SUPERCONDUCTOR_pec", "sapphire__vacuum"
CASES = {"MA": SHEET, "MS": SHEET, "SA": GAPS, "default": GAPS}
#: gfem on r20, per mode (7.68 GHz, 16.67 GHz): participation per kind.
GOLDEN: dict[str, list[float]] = {}

# Strang-Fix 6-point rule, exact to degree 4: barycentrics and weights (sum 1)
_A, _B = 0.445948490915965, 0.091576213509771
_RULE = np.array(
    [
        [_A, _A, 1 - 2 * _A],
        [_A, 1 - 2 * _A, _A],
        [1 - 2 * _A, _A, _A],
        [_B, _B, 1 - 2 * _B],
        [_B, 1 - 2 * _B, _B],
        [1 - 2 * _B, _B, _B],
    ]
)
_WEIGHTS = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3) * 2


@pytest.fixture(scope="module")
def case():
    ref = os.environ.get("GFEM_REFERENCE_DIR")
    if not ref:
        pytest.skip("GFEM_REFERENCE_DIR not set (needs the eigenmode Palace runs)")
    return Path(ref) / "eigenmode" / "r20"


def _model(case: Path) -> tuple[gfem.Model, dict]:
    config = json.loads((case / "config.json").read_text())
    mesh = gfem.Mesh.from_gmsh(case / "palace.msh", length_unit="um")
    name = {(r.dim, r.tag): r.name for r in mesh.regions}
    materials = {}
    for m in config["Domains"]["Materials"]:
        material = gfem.Material(
            eps_r=m["Permittivity"],
            mu_r=m.get("Permeability", 1.0),
            tan_delta=float(np.mean(m.get("LossTan", 0.0))),
            axes=m.get("MaterialAxes"),
        )
        for tag in m["Attributes"]:
            materials[name[3, tag]] = material
    bnd = config["Boundaries"]
    ports = [
        gfem.LumpedPort(
            f"p{p['Index']}",
            tuple(name[2, e["Attributes"][0]] for e in p["Elements"]),
            tuple(e["Direction"] for e in p["Elements"]),
            impedance=p["R"],
        )
        for p in sorted(bnd["LumpedPort"], key=lambda p: p["Index"])
    ]
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC(name[2, t]) for t in bnd["PEC"]["Attributes"]],
        ports=ports,
    )
    return model, config


@pytest.fixture(scope="module")
def solved(case):
    model, config = _model(case)
    options = gfem.SolverOptions(order=config["Solver"]["Order"], threads=THREADS)
    eigen = config["Solver"]["Eigenmode"]
    problem = gfem.Eigenmode(target=eigen["Target"] * 1e9, n_modes=eigen["N"] + 1)
    start = time.perf_counter()
    modes = gfem.solve(model, problem, options=options)
    wall = time.perf_counter() - start
    return model, config, modes, wall


@pytest.fixture(scope="module")
def measured(solved):
    _, _, modes, wall = solved
    out, times = {}, {}
    for kind, surface in CASES.items():
        start = time.perf_counter()
        out[kind] = gfem.interface_participation(
            modes, surface, T, EPS_LAYER, kind=kind, tan_delta=TAN_DELTA
        )
        times[kind] = time.perf_counter() - start
    print(  # noqa: T201
        f"r20 solve {wall:.1f} s; measurements "
        + ", ".join(f"{k} {t:.2f} s" for k, t in times.items())
        + "\n  "
        + "\n  ".join(f"{k}: {np.asarray(p.participation)}" for k, p in out.items())
    )
    return out, times, wall


def _side_tensor(model: gfem.Model, region: str) -> np.ndarray:
    material = model.materials[region]
    eps = np.broadcast_to(np.asarray(material.eps_r, float), (3,))
    axes = np.eye(3) if material.axes is None else np.asarray(material.axes)
    return np.einsum("k,ki,kj->ij", eps, axes, axes)


def _probe_participation(model, modes, surface, kind):
    """The layer energy from probes just inside the wanted side of each face."""
    mesh = model.mesh
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    tri = mesh.elements(surface)
    nodes = np.asarray(mesh.nodes)
    v = nodes[mesh.triangles[tri]]
    cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / (2 * area[:, None])
    size = np.sqrt(area)
    sides = mesh.triangle_neighbours[tri]
    region = np.array(
        [[names[mesh.tetrahedron_tags[t]] for t in pair] for pair in sides]
    )
    want = "sapphire" if kind == "MS" else "vacuum"
    # every face of these surfaces has one sapphire and one vacuum side
    assert (np.sort(region, axis=1) == ["sapphire", "vacuum"]).all()
    side = np.where(region[:, 0] == want, sides[:, 0], sides[:, 1])
    centroid = nodes[mesh.tetrahedra[side]].mean(axis=1)
    toward = np.sign(np.einsum("ti,ti->t", centroid - v.mean(axis=1), normal))
    points = np.einsum("qk,tki->tqi", _RULE, v)
    points += (1e-7 * size * toward)[:, None, None] * normal[:, None, :]
    eps_s = _side_tensor(model, "sapphire")
    energy = []
    for field in modes.e_fields:
        e = np.asarray(field.at(points.reshape(-1, 3))).reshape(points.shape)
        if kind == "MS":
            e = np.einsum("ij,tqj->tqi", eps_s, e)
        en = np.einsum("tqi,ti->tq", e, normal)
        if kind in {"MA", "MS"}:
            density = np.abs(en) ** 2 / EPS_LAYER
        else:
            et = e - en[..., None] * normal[:, None, :]
            density = (
                EPS_LAYER * (np.abs(et) ** 2).sum(-1) + np.abs(en) ** 2 / EPS_LAYER
            )
        energy.append(
            0.5 * const.epsilon_0 * T * (density * _WEIGHTS * area[:, None]).sum()
        )
    return np.array(energy) / gfem.conventions.MODE_ENERGY


@pytest.mark.parametrize("kind", ["MA", "MS", "SA"])
def test_face_kernel_matches_probes_from_inside_each_side(solved, measured, kind):
    model, _, modes, _ = solved
    expected = _probe_participation(model, modes, CASES[kind], kind)
    np.testing.assert_allclose(measured[0][kind].participation, expected, rtol=1e-5)


def test_loss_and_sanity_of_the_three_interfaces(measured):
    out = measured[0]
    for kind in ("MA", "MS", "SA"):
        p = np.asarray(out[kind].participation)
        assert ((p > 1e-8) & (p < 1e-2)).all(), (kind, p)
        np.testing.assert_allclose(out[kind].inverse_q, p * TAN_DELTA, rtol=1e-15)


def test_measurements_cost_little_next_to_the_solve(measured):
    _, times, wall = measured
    assert max(times.values()) < 0.2 * wall


@pytest.mark.skipif(not GOLDEN, reason="golden baselines not recorded yet")
def test_interface_participation_golden_baseline(measured):
    out = measured[0]
    for kind, values in GOLDEN.items():
        np.testing.assert_allclose(out[kind].participation, values, rtol=1e-6)


def test_interface_participation_parity_with_palace(solved, case):
    model, config, modes, _ = solved
    dielectric = config["Boundaries"].get("Postprocessing", {}).get("Dielectric")
    table = case / "output" / "palace" / "surface-Q.csv"
    if not dielectric or not table.exists():
        pytest.skip(
            "the Palace run has no Dielectric postprocessing (no surface-Q.csv); "
            "rerun Palace with one to compare"
        )
    names = {r.tag: r.name for r in model.mesh.regions if r.dim == 2}
    rows = np.loadtxt(table, delimiter=",", skiprows=1, ndmin=2)
    header = table.read_text().splitlines()[0].split(",")
    eig = np.loadtxt(
        case / "output" / "palace" / "eig.csv", delimiter=",", skiprows=1, ndmin=2
    )
    f_palace = eig[:, 1] * 1e9  # Re f, GHz
    f_gfem = np.asarray(modes.frequency)
    for d in dielectric:
        surfaces = [names[a] for a in d["Attributes"]]
        kind = {"default": "default"}.get(d.get("Type", "Default").lower(), d["Type"])
        p = gfem.interface_participation(
            modes,
            surfaces,
            d["Thickness"] * config["Model"]["L0"],
            d["Permittivity"],
            kind=kind,
        )
        column = next(
            k
            for k, h in enumerate(header)
            if h.strip().startswith(f"p_surf[{d['Index']}]")
        )
        for row, f in zip(rows, f_palace, strict=False):
            j = int(np.argmin(np.abs(f_gfem - f)))
            if abs(f_gfem[j] - f) > 1e-3 * f:
                continue
            assert math.isclose(float(p.participation[j]), row[column], rel_tol=0.05)
