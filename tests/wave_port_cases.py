r"""Guides closed by wave ports, shared by the eigenmode wave port tests.

Every guide but H is metallic, uniform along z, with layers that fill its
whole cross-section, closed by a short (PEC at z = 0) and/or wave ports. A
guide mode (TE, TM or TEM, cutoff :math:`k_c`) then never couples to
another, so each mode family is an exact transmission line along z and its
complex resonances solve a scalar equation (`line_root`). Time goes as
:math:`e^{+i\omega t}`: a decaying mode has :math:`\mathrm{Im}\,f > 0`, and
each port takes the outgoing branch of its propagation constant.

The rectangular guides are WR-90 (22.86 x 10.16 mm, TE10 cutoff 6.557 GHz)
on mesh R6: 6 cells across a, 2 across b, 0.5 cells per mm along z, layer
interfaces on grid planes (`ported_guide`). Layers are listed from z = 0 (a
short, or the port "portl") to the port "port" as (name, eps_r or
`gfem.Material`, thickness mm). Roots are in GHz, to about 1e-12 relative.

- A: short, eps_r = 4 10 mm, air 5 mm, port (Q 5.5).
- A3: short, eps_r = 4 30 mm, air 5 mm, port: the root just above the air
  guide's cutoff (Q 24).
- A4: short, air 20, eps_r = 10 2.4, air 9.9, eps_r = 10 2.4, air 3 mm, port:
  two windows, a partial reflector (Q 869).
- B3: a port on each end, three eps_r = 10 windows (Q 52); the ports carry
  0.943972 and 0.056028 of the outgoing power.
- C: A3's stack on a guide 5 mm high whose one port surface carries TE10 and
  TE20; TE20 resonances need the surface's modal correction.
- L: short, eps_r = 4 10 mm, then 5 mm of
  :math:`\varepsilon = \mathrm{diag}(3,\ 2.25(1 - 0.01i),\ 5)` up to the
  port, a lossy anisotropic port section (TE10 sees only
  :math:`\varepsilon_{yy}`).
- D: a quarter coax (radii 0.3 and 1 mm, PMC symmetry planes), short,
  eps_r = 4 10 mm, air 5 mm, port: TEM, a closed form (`coax_roots`).
- E: a circular guide (R = 10 mm), short, eps_r = 4 10 mm, air 5 mm, port on
  TM01 (its third mode, :math:`E_n \ne 0`). The straight-sided mesh has a
  polygonal wall, so its reference takes the port's discrete cutoff.
- H: WR-90 with an eps_r = 6 block (0.35 a < x < 0.65 a, y < b/2) along its
  length, short, a full eps_r = 10 window 2.4 mm thick at 20 mm, port 3 mm
  further: a hybrid, dispersive port mode with no exact value.
"""

import math

import numpy as np
import structured_meshes as sm

import gfem

C0 = 299792458.0
A_MM, B_MM = 22.86, 10.16

CASE_A = (("dielectric", 4.0, 10.0), ("air", 1.0, 5.0))
CASE_A3 = (("dielectric", 4.0, 30.0), ("air", 1.0, 5.0))
CASE_A4 = (
    ("air", 1.0, 20.0),
    ("window", 10.0, 2.4),
    ("air", 1.0, 9.9),
    ("window", 10.0, 2.4),
    ("air", 1.0, 3.0),
)
CASE_B3 = (
    ("air_l", 1.0, 3.0),
    ("window_1", 10.0, 2.4),
    ("air_1", 1.0, 20.0),
    ("window_2", 10.0, 2.4),
    ("air_2", 1.0, 9.9),
    ("window_3", 10.0, 2.4),
    ("air", 1.0, 3.0),
)
C_B_MM = 5.0  # case C's guide height
CASE_L = (
    ("dielectric", 4.0, 10.0),
    (
        "section",
        gfem.Material(eps_r=(3.0, 2.25, 5.0), tan_delta=(0.0, 0.01, 0.0)),
        5.0,
    ),
)

ROOT_A = 11.665906948910 + 1.062461078093j
ROOT_A3 = 7.033666935873 + 0.147969608374j
ANTI_BOUND_A3 = 4.619035589541
ROOT_A4 = 9.968822933448 + 0.005732947875j
ROOT_B3 = 9.974259423230 + 0.096704835502j
SHARE_B3 = {"left": 0.943972, "right": 0.056028}
ROOT_C10 = 14.123326299100 + 0.380179415083j
ROOT_C20 = 15.216633733515 + 0.207573093830j
ROOT_L = 11.669452414901 + 2.116334667323j

TWO_PORTS = (gfem.WavePort("left", "portl"), gfem.WavePort("right", "port"))
SHARED_PORTS = (gfem.WavePort("te10", "port"), gfem.WavePort("te20", "port", mode=2))


def ghz(omega):
    """Complex frequencies in GHz from complex angular frequencies."""
    return np.asarray(omega) / (2 * math.pi) / 1e9


def nearest_error(result, root):
    """``(|f - root| / |root|, j)`` of the mode ``j`` nearest ``root`` (GHz)."""
    f = ghz(result.omega)
    j = int(np.argmin(np.abs(f - root)))
    return abs(f[j] - root) / abs(root), j


def im_error(result, j, root):
    """``|Im f - Im root| / Im root`` of mode ``j``."""
    return abs(ghz(result.omega[j]).imag - root.imag) / root.imag


def _z_edges(layers, nz_per_mm):
    tops = np.cumsum([t for _, _, t in layers])
    zs = [np.zeros(1)]
    for (_, _, t), z1 in zip(layers, tops, strict=True):
        n = max(1, round(t * nz_per_mm))
        zs.append(z1 - t + np.linspace(0, t, n + 1)[1:])
    return tops, np.concatenate(zs)


def ported_guide(
    path, layers, *, left_port, b_mm=B_MM, nx=6, nz_per_mm=0.5, ny=2, mirror=False
):
    """``layers`` from z = 0 (a short, or the port ``portl``) to ``port``.

    PEC surfaces are "walls" (the short at z = 0 among them). ``mirror``: the
    cells of the upper half in z mirror the lower half's, so a symmetric
    stack gives a mesh symmetric under the reflection that swaps the ports.
    """
    tops, z = _z_edges(layers, nz_per_mm)
    edges = (np.linspace(0, A_MM, nx + 1), np.linspace(0, b_mm, ny + 1), z)
    length = tops[-1]

    def volume(c):
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        if abs(c[2] - length) < 1e-9:
            return "port"
        if left_port and abs(c[2]) < 1e-9:
            return "portl"
        if any(abs(c[2] - z) < 1e-9 for z in tops[:-1]):
            return ""
        return "walls"

    if isinstance(mirror, bool):
        mirror = (False, False, mirror)
    mesh = sm.structured(edges, volume, surface, mirror=mirror)
    sm.write_msh22(path, mesh)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _materials(layers):
    return {
        name: eps if isinstance(eps, gfem.Material) else gfem.Material(eps_r=eps)
        for name, eps, _ in layers
    }


WALLS = (gfem.PEC("walls"),)


def ported_model(mesh, layers, ports, boundaries=WALLS):
    """The guide of ``layers`` on ``mesh`` with ``ports``."""
    return gfem.Model(
        mesh,
        materials=_materials(layers),
        boundaries=list(boundaries),
        ports=list(ports),
    )


# --- D: coax, TEM --------------------------------------------------------------

COAX_RADII_MM = (0.3, 1.0)


def coax_guide(path, layers, *, n_r=2, n_theta=4, nz_per_mm=0.5):
    """A quarter coax with ``layers``: PEC "walls", PMC "symmetry", "port"."""
    inner, outer = COAX_RADII_MM
    tops, z = _z_edges(layers, nz_per_mm)
    edges = (
        np.linspace(inner, outer, n_r + 1),
        np.linspace(0, np.pi / 2, n_theta + 1),
        z,
    )
    length = tops[-1]

    def volume(c):
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        if abs(c[2] - length) < 1e-9:
            return "port"
        if any(abs(c[2] - z) < 1e-9 for z in tops[:-1]):
            return ""
        if abs(c[1]) < 1e-9 or abs(c[1] - np.pi / 2) < 1e-9:
            return "symmetry"
        return "walls"

    def mapping(p):
        r, t, z = p.T
        return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)

    sm.write_msh22(path, sm.structured(edges, volume, surface, mapping))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def coax_roots(eps, length_mm, modes):
    r"""Roots (GHz) of a TEM line: short, ``eps`` over ``length_mm``, port in air.

    :math:`\tan(n k_0 L) = i n` with :math:`n = \sqrt{\varepsilon_r} > 1`,
    so :math:`n k_0 L = m\pi + \pi/2 + i\,\mathrm{artanh}(1/n)`; the air
    between the layer and the port moves no root (the port is exact).
    """
    n = math.sqrt(eps)
    k = [
        (m * math.pi + math.pi / 2 + 1j * math.atanh(1 / n)) / (n * length_mm * 1e-3)
        for m in modes
    ]
    return np.array(k) * C0 / (2 * math.pi) / 1e9


# --- E: circular guide, TM01 ----------------------------------------------------

CIRCLE_R_MM = 10.0


def circular_guide(path, layers, *, n=16, nz_per_mm=0.5):
    """A circular guide of radius 10 mm: a square grid of ``n`` x ``n`` cells
    mapped to the disk, PEC "walls", "port" at the far end."""
    tops, z = _z_edges(layers, nz_per_mm)
    edges = (np.linspace(-1, 1, n + 1), np.linspace(-1, 1, n + 1), z)
    length = tops[-1]

    def mapping(p):
        u, v, z = p.T
        r = CIRCLE_R_MM
        return np.stack(
            [r * u * np.sqrt(1 - v**2 / 2), r * v * np.sqrt(1 - u**2 / 2), z], axis=1
        )

    def volume(c):
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        if abs(c[2] - length) < 1e-9:
            return "port"
        if any(abs(c[2] - z) < 1e-9 for z in tops[:-1]):
            return ""
        return "walls"

    sm.write_msh22(path, sm.structured(edges, volume, surface, mapping))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def line_root(family, kc, layers, guess_ghz):
    r"""The root (GHz) nearest ``guess_ghz`` of a shorted line of ``layers``.

    ``layers``: (eps_r, thickness mm) from the short to the port, the port in
    the last layer's medium. Each layer carries :math:`V'' = \gamma^2 V`,
    :math:`\gamma^2 = k_c^2 - \varepsilon_r k_0^2`, with wave impedance
    :math:`i k_0/\gamma` (TE) or :math:`\gamma/(i k_0 \varepsilon_r)` (TM);
    the port takes the outgoing branch
    :math:`\gamma = i\sqrt{\varepsilon_r k_0^2 - k_c^2}`, and the root solves
    :math:`V = Z_p I` by Newton's method from :math:`(V, I) = (0, 1)` at the
    short.
    """

    def impedance(k0, gamma, eps):
        return gamma / (1j * k0 * eps) if family == "TM" else 1j * k0 / gamma

    def residual(k0):
        v, i = 0j, 1 + 0j
        for eps, t in layers:
            g = np.sqrt(kc**2 - eps * k0**2 + 0j)
            z = impedance(k0, g, eps)
            ch, sh = np.cosh(g * t * 1e-3), np.sinh(g * t * 1e-3)
            v, i = v * ch - z * i * sh, -v / z * sh + i * ch
        eps = layers[-1][0]
        gp = 1j * np.sqrt(eps * k0**2 - kc**2 + 0j)
        return v - impedance(k0, gp, eps) * i

    k = 2 * math.pi * guess_ghz * 1e9 / C0
    for _ in range(100):
        h = 1e-7 * abs(k)
        step = residual(k) / ((residual(k + h) - residual(k - h)) / (2 * h))
        k -= step
        if abs(step) < 1e-15 * abs(k):
            break
    assert abs(residual(k)) < 1e-10
    return complex(k * C0 / (2 * math.pi) / 1e9)


# --- H: hybrid port section -----------------------------------------------------

H_Z_MM = (0.0, 20.0, 22.4, 25.4)  # short, window, window end, port


def hybrid_guide(path, gap_mm=None):
    """Case H's mesh: "air", "block", "window"; PEC "walls", "port".

    ``gap_mm`` is the guide's length from the window to the port (3 mm, two
    cells, by default), 1.5 mm cells beyond.
    """
    a, b = A_MM, B_MM
    xs = np.unique(
        np.concatenate(
            [
                np.linspace(0, 0.35 * a, 3),
                np.linspace(0.35 * a, 0.65 * a, 3),
                np.linspace(0.65 * a, a, 3),
            ]
        )
    )
    z = H_Z_MM if gap_mm is None else (*H_Z_MM[:3], H_Z_MM[2] + gap_mm)
    cells = 2 if gap_mm is None else max(2, round(gap_mm / 1.5))
    zs = np.unique(
        np.concatenate(
            [
                np.linspace(z[0], z[1], 11),
                np.linspace(z[1], z[2], 2),
                np.linspace(z[2], z[3], cells + 1),
            ]
        )
    )

    def volume(c):
        if z[1] < c[2] < z[2]:
            return "window"
        if 0.35 * a < c[0] < 0.65 * a and c[1] < b / 2:
            return "block"
        return "air"

    def surface(c):
        if abs(c[2] - z[3]) < 1e-9:
            return "port"
        on_wall = c[2] < 1e-9 or min(c[0], a - c[0], c[1], b - c[1]) < 1e-9
        return "walls" if on_wall else ""

    mesh = sm.structured((xs, np.linspace(0, b, 3), zs), volume, surface)
    sm.write_msh22(path, mesh)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def hybrid_model(mesh):
    """Case H on ``mesh``, one wave port "p"."""
    return gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "block": gfem.Material(eps_r=6.0),
            "window": gfem.Material(eps_r=10.0),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p", "port")],
    )
