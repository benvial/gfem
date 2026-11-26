r"""Bent port mode cases: exact references and the structured sections that meet them.

Three families have exact (Bessel) solutions of the bent guide at real
frequency:

- ``wr90``: the H-plane bend of a vacuum WR-90 rectangular guide, PEC all
  round, the TE10-like mode, $\nu$ real;
- ``line``: a bent parallel-plate line with $E$ in the bend plane (PEC walls
  at $R \mp a/2$, PMC planes in $y$), a TEM-like whispering-gallery mode,
  $\nu$ real;
- ``slab``: a leaky bent dielectric slab (core 2.0, cladding 1.444, width
  0.6 um, 1.55 um, constant indices) between a PEC wall at $x = -1.5$ um and
  a perfectly matched layer on $[2.5, 3.5]$ um; $\nu = -i\gamma R$ complex
  (Ey: PEC planes in $y$; Hy: PMC planes).

The references are 30-60 digit Bessel/Hankel roots of the matching
determinant (``mpmath``), derivatives by a 5-point finite difference of the
root at 60 digits, so every literal is exact to the digits written.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import structured_meshes as sm

import gfem

C0 = 299792458.0

# ---- WR-90 H-plane: a = 22.86 mm (bend plane), b = 10.16 mm, 10 GHz --------------
WR90_A, WR90_B = 22.86, 10.16  # mm
WR90_F = 10e9  # Hz
#: $R$ (mm) -> ($\nu$, $n_g$, $\beta_2$ in s^2/m): $\nu$ and $n_g$ from the Bessel
#: root of J_nu(k0 r1) Y_nu(k0 r2) - Y_nu(k0 r1) J_nu(k0 r2) = 0 at 50 digits;
#: $\beta_2$ from a 60-digit 5-point difference of the root in k0.
WR90 = {
    20.0: (3.119479248175464139985494, 1.37948746042206, -2.39607596556203e-20),
    30.0: (4.717263705740480610170115, 1.34703629616968, -4.01031527520825e-20),
    60.0: (9.47945670452375776105301, 1.32987043118888, -4.98154433929138e-20),
}

# ---- E-in-plane parallel-plate line: a = 10 mm, b = 2.5 mm, 10 GHz ---------------
LINE_A, LINE_B = 10.0, 2.5  # mm
LINE_F = 10e9
#: $R$ (mm) -> $\nu$ of the TEM-like mode, from the Bessel root of the PEC
#: pair at $R \mp a/2$ (50 digits).
LINE = {
    10.0: 2.164036486250767491691,
    20.0: 4.224985258252037378612,
    40.0: 8.399936802955517966746,
}

# ---- leaky slab: 2.0 / 1.444, w = 0.6 um, lambda = 1.55 um -----------------------
SLAB = {"nco": 2.0, "ncl": 1.444, "w": 0.6, "x0": -1.5, "xp": 2.5, "d": 1.0}
SLAB_F = C0 / 1.55e-6
#: ($R$ in um, polarization) -> $\nu$ with the PEC wall at $x = -1.5$ um (45-digit
#: root of the 4x4 matching determinant; the open-slab root differs by less
#: than 3.4e-6).
SLAB_NU = {
    (3, "Ey"): 22.7539258235505 - 0.0567834637758j,
    (3, "Hy"): 22.02396730780945 - 0.142198291152j,
    (5, "Ey"): 37.47458292979335 - 0.00528818775788j,
    (5, "Hy"): 36.19966850858682 - 0.0261154878394j,
    (8, "Ey"): 59.7149372403172 - 9.18743995062e-5j,
    (8, "Hy"): 57.61370811017967 - 0.00140098463565j,
}
#: ($R$ = 5 um, polarization) -> ($n_g$, $\beta_2$ in ps^2/m), wall column, from a
#: 5-point difference of the root in k0 at 30 digits.
SLAB_DISPERSION = {
    "Ey": (2.036182759171 + 0.002636507265j, 0.29331478 - 0.06907944j),
    "Hy": (2.048175251604 + 0.012985834292j, 0.65045147 - 0.32121204j),
}
#: ($R$ in um, polarization) -> exact $\mathrm{d}\gamma/\mathrm{d}R$ in 1/um^2
#: (open slab, implicit-function derivative of the root).
SLAB_DGAMMA_DR = {
    (5, "Hy"): -0.00578480250041 - 0.0252844751058j,
    (3, "Ey"): -0.0266710314495 - 0.0883628119408j,
}
#: ($R$ in um, polarization) -> junction $\tau^2$ between the straight mode of the
#: same section and the bent mode (30 digits).
SLAB_TAU2 = {
    (3, "Ey"): 0.924686545007 + 0.0398612203497j,
    (5, "Ey"): 0.966735072404 + 0.00466915543199j,
    (8, "Ey"): 0.988246863195 + 9.35239407337e-5j,
    (3, "Hy"): 0.92121329611 + 0.0845199388674j,
    (5, "Hy"): 0.953106322777 + 0.0189360834048j,
    (8, "Hy"): 0.982133746489 + 0.00121415293946j,
}
#: Whispering-gallery slab, w = 2 um, R = 2 um ($n_\mathrm{eff} = 2.549 >
#: n_\mathrm{co}$): $\nu$ at 8 digits (the continuation recipe's pick).
WGM_NU = {"Ey": 20.66859418 - 0.06335274j, "Hy": 20.20448391 - 0.10560599j}
WGM_W = 2.0


def _seg(parts):
    """Grid lines from ``(start, end, cell)`` parts."""
    out = []
    for a, b, h in parts:
        n = max(1, round(np.ceil((b - a) / h - 1e-9)))
        out.append(np.linspace(a, b, n + 1))
    return np.unique(np.round(np.concatenate(out), 12))


def wr90_mesh(path: Path, nx: int = 6, ny: int = 2) -> Path:
    """WR-90 a x b in mm, PEC ``walls``, port ``p1`` at z = 0 (1 mm long)."""
    a, b = WR90_A, WR90_B
    edges = (np.linspace(0, a, nx + 1), np.linspace(0, b, ny + 1), np.array([0, 1.0]))
    eps = 1e-9

    def surface(c):
        if abs(c[2]) < eps:
            return "p1"
        if abs(c[0]) < eps or abs(c[0] - a) < eps:
            return "walls"
        if abs(c[1]) < eps or abs(c[1] - b) < eps:
            return "walls"
        return ""

    return sm.write_msh22(path, sm.structured(edges, lambda _: "air", surface))


def line_mesh(path: Path, nx: int = 8, ny: int = 1) -> Path:
    """Parallel-plate line a x b in mm: PEC ``walls`` at x = 0, a; PMC (free) in y."""
    a, b = LINE_A, LINE_B
    edges = (np.linspace(0, a, nx + 1), np.linspace(0, b, ny + 1), np.array([0, 1.0]))
    eps = 1e-9

    def surface(c):
        if abs(c[2]) < eps:
            return "p1"
        if abs(c[0]) < eps or abs(c[0] - a) < eps:
            return "walls"
        return ""

    return sm.write_msh22(path, sm.structured(edges, lambda _: "air", surface))


def slab_mesh(
    path: Path,
    pol: str,
    hcell: float = 0.1,
    *,
    xp: float | None = None,
    d: float | None = None,
    x0: float | None = None,
    hcore: float | None = None,
    w: float | None = None,
    inner_d: float = 0.0,
    outer_wall: bool = True,
    height: float = 0.1,
    dz: float = 0.1,
) -> Path:
    """The bent-slab section, x in [x0, xp + d] um, core ``|x| < w/2``.

    The layer lies beyond ``xp``. ``pol`` "Ey": PEC planes in y; "Hy": PMC
    (free). PEC on x = x0 and on the outer face (``outer_wall=False``: the
    outer face is free, i.e. PMC). ``d = 0`` makes no outer layer (the outer
    wall is the face at ``xp``); ``inner_d > 0`` adds a layer on
    ``[x0 - inner_d, x0]`` and moves the inner wall out to it.
    """
    xp = SLAB["xp"] if xp is None else xp
    d = SLAB["d"] if d is None else d
    x0 = SLAB["x0"] if x0 is None else x0
    w = SLAB["w"] if w is None else w
    hcore = hcell if hcore is None else hcore
    parts = [(x0, -w / 2, hcell), (-w / 2, w / 2, hcore), (w / 2, xp, hcell)]
    if inner_d > 0:
        parts.insert(0, (x0 - inner_d, x0, hcell))
    if d > 0:
        parts.append((xp, xp + d, hcell))
    xs = _seg(parts)
    xmin = xs[0]
    edges = (xs, np.array([0.0, height]), np.array([0.0, dz]))
    eps = 1e-9
    xmax = xs[-1]

    def volume(c):
        if (d > 0 and c[0] > xp) or (inner_d > 0 and c[0] < x0):
            return "pml"
        return "core" if abs(c[0]) < w / 2 else "clad"

    def surface(c):
        if abs(c[2]) < eps:
            return "p1"
        if abs(c[0] - xmin) < eps or (outer_wall and abs(c[0] - xmax) < eps):
            return "pec"
        if pol == "Ey" and (abs(c[1]) < eps or abs(c[1] - height) < eps):
            return "pec"
        return ""

    return sm.write_msh22(path, sm.structured(edges, volume, surface))


def wr90_model(path: Path) -> gfem.Model:
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "p1")],
    )


line_model = wr90_model


def slab_model(
    path: Path, r0: float = 1e-10, *, nco: float | None = None
) -> gfem.Model:
    mesh = gfem.Mesh.from_gmsh(path)
    clad = gfem.Material(eps_r=SLAB["ncl"] ** 2)
    nco = SLAB["nco"] if nco is None else nco
    materials = {"core": gfem.Material(eps_r=nco**2), "clad": clad}
    pml = None
    if "pml" in mesh.volumes:
        materials["pml"] = clad
        pml = gfem.PML("pml", target_reflection=r0)
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "p1")],
        pml=pml,
    )


def wr90_bend(radius_mm: float) -> gfem.Bend:
    """The bend about the guide's centre line (x = a/2, in the section's plane)."""
    return gfem.Bend(radius_mm * 1e-3, outward="+x", at=(WR90_A / 2 * 1e-3, 0.0, 0.0))


def line_bend(radius_mm: float) -> gfem.Bend:
    return gfem.Bend(radius_mm * 1e-3, outward="+x", at=(LINE_A / 2 * 1e-3, 0.0, 0.0))


def slab_bend(radius_um: float) -> gfem.Bend:
    return gfem.Bend(radius_um * 1e-6, outward="+x", at=(0.0, 0.0, 0.0))


def solve_mode(model, frequency, bend=None, *, order=3, **kwargs):
    """``gfem.solve`` of one bent (or straight) port mode, 1 thread, warnings kept."""
    options = gfem.SolverOptions(order=order, threads=1)
    problem = gfem.PortMode("p1", (frequency,), bend=bend, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnabsorbedBendWarning)
        return gfem.solve(model, problem, options=options)


def nu_of(result, radius_m: float) -> complex:
    r"""$\nu = -i\gamma R$ of the first frequency (``gamma`` is in 1/m)."""
    return -1j * complex(np.asarray(result.gamma)[0]) * radius_m
