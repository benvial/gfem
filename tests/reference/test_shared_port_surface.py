"""TE0 and TM0 on one port surface of a straight silicon strip.

The 3D bend's stack (tools/bend3d_mesh.py, ``kind="straight"``, the full
height): a 0.5 x 0.22 um Si strip in SiO2 at 1.55 um, its core straight
along x from port1 to port2, with a second-order absorbing boundary on the
other faces and PEC port window edges, as tools/bend3d_run.py builds the
bend. Each end carries two wave ports on the same surface: TE0 (rank 1, E
along y) and TM0 (rank 2, E along z).

The strip is uniform and the full stack is symmetric under z -> -z, so each
mode is transmitted alone with ``exp(-gamma L)``: no reflection and no
TE0-TM0 conversion, up to the mesh's discretization. The mirror half model
cannot carry this case: its z = 0 plane is a PMC or a PEC, which admits
only one of TE0 and TM0.

The mesh is coarser and the strip shorter than the bend run's straight
check (h = 0.07 um, on the half model). The full stack at h = 0.1 um over
the 5 um box needs more than the 16 GB of an 8-core MareNostrum 5 job, even
with BDDC; a direct solve at h = 0.15 um over 2 um takes 14 GB. At
h = 0.15 um the reflections were 1.6e-3 and the TE0 phase error 0.046 rad;
at h = 0.1 um (BDDC + GMRES) both are below the gates. One solve takes
about 18 min and 11 GiB on 8 cores.
"""

import bend3d_mesh as bm
import numpy as np
import pytest
import structured_meshes as sm
from scipy.constants import c

import gfem

UM = 1e-6
F0 = c / (1.55 * UM)
N_CORE, N_CLADDING = 3.4757, 1.444024  # tools/bend3d_run.py at 1.55 um
H = 0.1  # um
LENGTH = 2.0  # um, port to port
L = LENGTH * UM
MODES = {"TE0": 1, "TM0": 2}
ENDS = {"in": "port1", "out": "port2"}
PORTS = tuple(f"{end}_{m}" for end in ENDS for m in MODES)
# GMRES at 1e-10: reciprocity and the zero conversion sit at that level
OPTIONS = gfem.SolverOptions(order=2, linear_solver="bddc_gmres", tolerance=1e-10)


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    """The model; the design square's tetrahedra become core or cladding."""
    nodes, tets, names, tris, tri_names = bm.bend3d(
        H, half=False, kind="straight", length=LENGTH
    )
    centroids = nodes[tets].mean(axis=1)
    names = [
        ("core" if abs(p[1]) < bm.CORE_W / 2 else "cladding") if n == "design" else n
        for n, p in zip(names, centroids, strict=True)
    ]
    path = tmp_path_factory.mktemp("strip") / "strip.msh"
    sm.write_msh22(path, (nodes, tets, names, tris, tri_names))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "cladding": gfem.Material(eps_r=N_CLADDING**2),
            "core": gfem.Material(eps_r=N_CORE**2),
        },
        boundaries=[gfem.Absorbing("outer", order=2)],
        ports=[
            gfem.WavePort(f"{end}_{m}", surface, rank, pec_edges=("outer",))
            for end, surface in ENDS.items()
            for m, rank in MODES.items()
        ],
    )


@pytest.fixture(scope="module")
def result(strip):
    return gfem.solve(strip, gfem.Driven(F0, excite=None), options=OPTIONS)


def _s(result):
    return np.asarray(result.s.values)[0]


def test_te0_and_tm0_are_the_first_two_ranks(strip):
    centre = np.array([-L / 2, 0.0, 0.0])  # the core's centre on port1
    for m, rank in MODES.items():
        mode = gfem.solve(strip, gfem.PortMode("in_TE0", F0, mode=rank))
        e = np.abs(np.asarray(mode.e_field.at(centre))[0])
        along = {"TE0": 1, "TM0": 2}[m]  # TE0 along y, TM0 along z
        assert e[along] > 30 * np.delete(e, along).max()  # measured 90 (TE0)
        # the 500 x 220 nm strip's indices; measured 2.4448 and 1.7708
        expected = {"TE0": 2.445, "TM0": 1.771}[m]
        assert abs(float(np.real(np.asarray(mode.n_eff)[0])) - expected) < 0.01


def test_each_mode_crosses_the_strip_with_its_own_phase(result):
    s = _s(result)
    for m in MODES:
        i, j = PORTS.index(f"in_{m}"), PORTS.index(f"out_{m}")
        gamma = complex(np.asarray(result.port_modes[f"in_{m}"].gamma)[0])
        for t in (s[j, i], s[i, j]):
            assert abs(t) >= 0.999  # measured 0.99992 (TE0), 0.99925 (TM0)
            # measured 0.0117 rad (TE0) and 0.0035 rad (TM0)
            assert abs(np.angle(t * np.exp(gamma * L))) < 0.02


def test_no_reflection_and_no_conversion_between_te0_and_tm0(result):
    s = np.abs(_s(result))
    mode = [p[-3:] for p in PORTS]
    n = len(PORTS)
    assert max(s[i, i] for i in range(n)) < 1e-3  # measured 4.9e-4
    # zero by the stack's z -> -z symmetry, which the mesh keeps: measured
    # 8.0e-11, GMRES's level
    conversion = [s[i, j] for i in range(n) for j in range(n) if mode[i] != mode[j]]
    assert max(conversion) < 3e-10


def test_the_strip_is_reciprocal(result):
    s = _s(result)
    assert np.abs(s - s.T).max() < 3e-10  # measured 9.1e-11


def test_each_port_has_the_exact_port_mode(strip, result):
    for name in PORTS:
        exact = gfem.solve(strip, gfem.PortMode(name, F0))
        np.testing.assert_allclose(
            np.asarray(result.port_modes[name].n_eff),
            np.asarray(exact.n_eff),
            rtol=1e-12,
        )
