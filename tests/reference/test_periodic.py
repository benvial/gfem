"""Periodic cells against Palace 0.18.1: Bloch bands (R1) and a Bloch two-port (R2).

Both cases are physics cases, not parity: Palace applies the Bloch wave
vector as a shift of the curl on periodic DOFs, gfem as a phase on the
paired DOFs, so the two discrete operators differ and agreement is bounded
by Palace's own discretization error on these meshes. Each case directory
of ``periodic/`` holds the mesh Palace ran (built by
`periodic_cases`), its ``config.json`` and ``output/``; the generation
scripts are kept with the effort's notes. Palace's FloquetWaveVector is in
rad per L0 (mm), gfem's ``wave_vector`` in rad/m.

- R1, the Bragg line C3 (order 3, 10 cells): the 2 lowest Bloch bands at
  k = pi/a lossless (both edges of the gap) and at k = pi/2a with
  tan delta = 0.01. Palace is 1e-7..3e-7 from the transfer-matrix roots
  there, gfem 1e-9.
- R2, the Bloch line two-port C1 (order 2, 16 cells) at k = +0.3 pi/a,
  both ports excited. Palace reports the port quantities of the periodic
  envelope u = E exp(+i k·r), gfem those of E, so
  S_ij^Palace = exp(+i k·(r_i - r_j)) S_ij^gfem (i the output port, j the
  excited one, r_p the port sheet's x). This pins that both codes share
  the sign of the Bloch phase.
"""

import json
import math
from pathlib import Path

import numpy as np
import periodic_cases as pc
import pytest

import gfem

#: Relative agreement with Palace on a complex band frequency (R1).
BLOCH_EIGEN_AGREEMENT = 1e-6
#: Agreement with Palace on S, after the envelope factor (R2).
BLOCH_S_AGREEMENT = 1e-4

BRAGG = ("bragg_k1_lossless", "bragg_k05_lossy")
LINE = "bloch_line_p2"


@pytest.fixture(scope="module")
def reference(reference_dir):
    return reference_dir / "periodic"


def _config(case: Path) -> dict:
    config = json.loads((case / "config.json").read_text())
    assert config["Model"]["L0"] == pc.MM
    return config


def _wave_vector(config: dict) -> tuple[float, float, float]:
    """The Bloch wave vector in rad/m from Palace's (rad per L0)."""
    k = config["Boundaries"]["Periodic"]["FloquetWaveVector"]
    return tuple(v / config["Model"]["L0"] for v in k)


def _palace_bands(case: Path, n: int) -> np.ndarray:
    """The ``n`` lowest of Palace's complex band frequencies, Hz."""
    eig = np.loadtxt(case / "output" / "eig.csv", delimiter=",", skiprows=1, ndmin=2)
    f = (eig[:, 1] + 1j * eig[:, 2]) * 1e9
    return f[np.argsort(f.real)][:n]


def _palace_s(case: Path) -> np.ndarray:
    """Palace's S (S[i, j]: port i, port j excited) from ``output/port-S.csv``."""
    path = case / "output" / "port-S.csv"
    header = [h.strip() for h in path.read_text().splitlines()[0].split(",")]
    (row,) = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    column = {h: n for n, h in enumerate(header)}
    s = np.zeros((2, 2), dtype=complex)
    for i, j in np.ndindex(2, 2):
        db = row[column[f"|S[{i + 1}][{j + 1}]| (dB)"]]
        deg = row[column[f"arg(S[{i + 1}][{j + 1}]) (deg.)"]]
        s[i, j] = 10 ** (db / 20) * np.exp(1j * math.radians(deg))
    return s


@pytest.mark.parametrize("name", BRAGG)
def test_bragg_line_bands_match_palace(reference, name):
    case = reference / name
    config = _config(case)
    (low, high) = config["Domains"]["Materials"]
    assert (low["Permittivity"], high["Permittivity"]) == (
        pc.C3_EPS_LOW,
        pc.C3_EPS_HIGH,
    )
    assert low["LossTan"] == high["LossTan"]
    mesh = gfem.Mesh.from_gmsh(case / "c3.msh", length_unit="mm")
    eigen = config["Solver"]["Eigenmode"]
    problem = gfem.Eigenmode(
        target=eigen["Target"] * 1e9,
        n_modes=2,
        wave_vector=_wave_vector(config),
    )
    options = gfem.SolverOptions(order=config["Solver"]["Order"])
    result = gfem.solve(pc.c3(mesh, low["LossTan"]), problem, options=options)
    f = np.sort_complex(np.asarray(result.omega) / (2 * math.pi))
    palace = _palace_bands(case, 2)
    assert np.abs(f - palace).max() <= BLOCH_EIGEN_AGREEMENT * np.abs(palace).min()


def test_bloch_line_s_matches_palace_through_the_envelope_factor(reference):
    case = reference / LINE
    config = _config(case)
    k = _wave_vector(config)
    assert k == pytest.approx(pc.C1_K, rel=1e-12)
    (material,) = config["Domains"]["Materials"]
    assert (material["Permittivity"], material["LossTan"]) == (
        pc.C1_EPS,
        pc.C1_TAN_DELTA,
    )
    driven = config["Solver"]["Driven"]
    assert driven["MinFreq"] * 1e9 == pc.C1_FREQUENCY
    mesh = gfem.Mesh.from_gmsh(case / "c1.msh", length_unit="mm")
    options = gfem.SolverOptions(order=config["Solver"]["Order"])
    result = gfem.solve(
        pc.c1(mesh), gfem.Driven(pc.C1_FREQUENCY, wave_vector=k), options=options
    )
    (s,) = np.asarray(result.s.values)
    r = np.array([pc.C1_X1, pc.C1_X2]) * pc.MM
    envelope = np.exp(1j * k[0] * (r[:, None] - r[None, :]))
    palace = _palace_s(case)
    assert np.abs(palace - envelope * s).max() <= BLOCH_S_AGREEMENT
    # the factor is not a detail: without it S21 and S12 disagree
    assert np.abs(palace - s).max() > 100 * BLOCH_S_AGREEMENT
