r"""Blocks for the mode cascade tests.

The synthetic part builds blocks without any FEM: a `gfem.CascadeResult` (the
same kind of block `gfem.cascade` takes from a driven solve) made of an S
matrix, the propagation constants of its section ports and their polarity
arrows. A Fresnel junction between two guides of propagation constants
:math:`\gamma_1`, :math:`\gamma_2` has :math:`S = \begin{pmatrix} r & t \\ t &
-r \end{pmatrix}`, :math:`r = (\gamma_1 - \gamma_2)/(\gamma_1 + \gamma_2)`,
:math:`t^2 = 1 - r^2`.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import structured_meshes as sm

import gfem
from gfem.results import NetworkParameters, PortModeResult

C0 = 299792458.0
#: Two frequencies, Hz.
FREQUENCY = (1.0e14, 1.1e14)
#: The +z polarity arrow of a line port: a point and a direction.
UP = ((0.0, 0.0), (0.0, 0.0, 1.0))
DOWN = ((0.0, 0.0), (0.0, 0.0, -1.0))


def k0(frequency: Sequence[float] = FREQUENCY) -> np.ndarray:
    """The free-space wave number at each frequency, 1/m."""
    return 2 * np.pi * np.asarray(frequency) / C0


def synthetic_block(
    ports: Sequence[str],
    values,
    *,
    gamma: Mapping[str, object] | None = None,
    polarity: Mapping[str, object] | None = None,
    frequency: Sequence[float] = FREQUENCY,
    z_ref: float = 50.0,
    polarization: str | None = "Ez",
    offsets: tuple[tuple[str, float], ...] = (),
    **mode_fields,
) -> gfem.CascadeResult:
    """A block of ``ports`` with S ``values`` (``(n_freq, n, n)``) and no FEM.

    ``gamma`` gives a port mode its propagation constant (a scalar or one
    value per frequency); ``polarity`` its arrow (default: ``UP`` on every
    port with a ``gamma``). ``mode_fields`` set other `PortModeResult` fields
    of every such port (``pml``, ``Z_PV``, ``e_field``).
    """
    n_f = len(frequency)
    values = jnp.asarray(values)
    ports = tuple(ports)
    modes = {}
    for name, g in (gamma or {}).items():
        rank = int(name.split("@")[1]) if "@" in name else 1
        arrow = (polarity or {}).get(name, UP) if polarity is not None else UP
        fields = {"Z_PV": jnp.full(n_f, jnp.nan), "polarity": arrow, **mode_fields}
        modes[name] = PortModeResult(
            port=name,
            mode=rank,
            frequency=jnp.asarray(frequency),
            gamma=jnp.broadcast_to(jnp.asarray(g, dtype=complex), (n_f,)),
            residual=jnp.zeros(n_f),
            solved=jnp.ones(n_f, dtype=bool),
            **fields,
        )
    s = NetworkParameters(
        values=values,
        z_ref=jnp.full((n_f, len(ports)), z_ref),
        frequency=tuple(frequency),
        ports=ports,
        excited=ports,
    )
    return gfem.CascadeResult(
        s=s, port_modes=modes, polarization=polarization, offsets=offsets
    )


def section_ports(section: str, n: int) -> list[str]:
    """The names ``section@1 .. section@n``."""
    return [f"{section}@{m}" for m in range(1, n + 1)]


def evanescent(n: int, frequency: Sequence[float] = FREQUENCY) -> np.ndarray:
    """:math:`\\gamma` of a mode far below cutoff, :math:`\\alpha = (2 + n) k_0`."""
    return (2.0 + n) * k0(frequency)


def fresnel_pair(
    gamma_outer, gamma_slab, *, section: str = "s", n_pad: int = 1
) -> tuple[gfem.CascadeResult, gfem.CascadeResult]:
    r"""Two Fresnel junctions bounding a slab: ports (a, s@1) and (s@1, b).

    The slab's mode is ``s@1`` (propagation constant ``gamma_slab``); the
    section keeps ``n_pad`` evanescent modes beyond it that couple to
    nothing (their rows of S are zero), so that the last kept mode cannot
    propagate. The first block is the outer guide into the slab,
    :math:`r = (\gamma_1 - \gamma_2)/(\gamma_1+\gamma_2)`; the second is the
    slab into the outer guide.
    """
    g1, g2 = (
        np.broadcast_to(gamma_outer, (len(FREQUENCY),)),
        np.broadcast_to(gamma_slab, (len(FREQUENCY),)),
    )
    r = (g1 - g2) / (g1 + g2)
    t = np.sqrt(1 - r**2 + 0j)
    n = 1 + n_pad
    names = section_ports(section, n)
    gamma = {names[0]: g2} | {names[m]: evanescent(m + 1) for m in range(1, n)}
    first = np.zeros((len(FREQUENCY), 1 + n, 1 + n), complex)
    second = first.copy()
    # block one: ports (a, s@1, ...); block two: (s@1, ..., b)
    first[:, 0, 0], first[:, 0, 1], first[:, 1, 0], first[:, 1, 1] = r, t, t, -r
    second[:, 0, 0], second[:, 0, n], second[:, n, 0], second[:, n, n] = -r, t, t, r
    a = synthetic_block(["a", *names], first, gamma=gamma)
    b = synthetic_block([*names, "b"], second, gamma=gamma)
    return a, b


def airy(gamma_outer, gamma_slab, length) -> tuple[np.ndarray, np.ndarray]:
    r"""Reflection and transmission of a slab of length :math:`\ell` in the outer guide.

    :math:`\rho = r(1 - p)/(1 - r^2 p)`, :math:`\tau = (1 - r^2)\,e^{-\gamma_2
    \ell}/(1 - r^2 p)` with :math:`p = e^{-2\gamma_2 \ell}`.
    """
    g1, g2 = np.asarray(gamma_outer), np.asarray(gamma_slab)
    r = (g1 - g2) / (g1 + g2)
    p = np.exp(-2 * g2 * length)
    return r * (1 - p) / (1 - r**2 * p), (1 - r**2) * np.exp(-g2 * length) / (
        1 - r**2 * p
    )


def random_block(
    rng: np.random.Generator,
    ports: Sequence[str],
    *,
    kind: str = "passive",
    dead: Sequence[str] = (),
    gamma: Mapping[str, object] | None = None,
    **kwargs,
) -> gfem.CascadeResult:
    """A reciprocal block of random S: symmetric, passive or unitary.

    ``passive`` scales a complex symmetric matrix to spectral norm 0.9;
    ``unitary`` is :math:`Q Q^\\mathsf{T}` of a random unitary :math:`Q`, over
    the ports that are not ``dead`` (the rows and columns of a dead port are
    zero, a mode that couples to nothing).
    """
    n_f, n = len(FREQUENCY), len(ports)
    live = [i for i, p in enumerate(ports) if p not in dead]
    out = np.zeros((n_f, n, n), complex)
    for f in range(n_f):
        m = len(live)
        z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        if kind == "passive":
            sym = (z + z.T) / 2
            sub = 0.9 * sym / np.linalg.norm(sym, 2)
        else:
            q, _ = np.linalg.qr(z)
            sub = q @ q.T
        out[f][np.ix_(live, live)] = sub
    return synthetic_block(ports, out, gamma=gamma, **kwargs)


# --- FEM blocks: a width step in a box -------------------------------------------


def lines(marks, h):
    """Sorted unique nodes of a 1D grid through ``marks``, spacing at most ``h``."""
    pieces = [
        np.linspace(a, b, max(2, math.ceil((b - a) / h) + 1))
        for a, b in itertools.pairwise(marks)
    ]
    return np.unique(np.round(np.concatenate(pieces), 12))


def box_mesh(path, xs, ys, region, *, ports=("left", "right")):
    """Structured triangles on ``xs`` x ``ys`` (um), PEC ``side`` top and bottom.

    ``region(xc, yc)`` names a cell; the end lines are ``ports``.
    """
    nx, ny = len(xs), len(ys)
    nodes = np.c_[np.repeat(xs, ny), np.tile(ys, nx), np.zeros(nx * ny)]

    def idx(i, j):
        return i * ny + j

    tris, names = [], []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, d, e = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            name = region(0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1]))
            tris += [(a, b, d), (a, d, e)]
            names += [name, name]
    segs = [(idx(i, 0), idx(i + 1, 0)) for i in range(nx - 1)]
    segs += [(idx(i, ny - 1), idx(i + 1, ny - 1)) for i in range(nx - 1)]
    segs += [(idx(0, j), idx(0, j + 1)) for j in range(ny - 1)]
    segs += [(idx(nx - 1, j), idx(nx - 1, j + 1)) for j in range(ny - 1)]
    seg_names = (
        ["side"] * (2 * (nx - 1)) + [ports[0]] * (ny - 1) + [ports[1]] * (ny - 1)
    )
    mesh = (nodes, np.array(tris), names, np.array(segs), seg_names)
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, mesh), length_unit="um")


#: The step: a box of width ``A`` (um) whose core ``[Y0, Y1]`` widens to
#: ``[Y0, Y2]`` between ``X_STEP`` and ``X_BACK``; block A ends at the plane
#: ``X_A`` after the first step, block B starts at ``X_B`` before the second
#: (the two planes sit 0.5 um apart, in different meshes).
A, Y0, Y1, Y2 = 2.0, 0.75, 1.25, 1.65
X_STEP, X_BACK, X_A, X_B, X_END = 1.0, 2.0, 1.25, 1.75, 3.0
LAMBDA = 1.55  # um
F_STEP = C0 / (LAMBDA * 1e-6)
EPS_CORE, EPS_CLAD = 2.0**2, 1.444**2
#: Where each kept mode's polarity is read, um: off the nodes of the first six.
POLARITY_Y = (1.193, 1.515, 0.470, 0.307, 1.777, 1.817)
STEP_OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=1)


def _step_y(h, shift):
    """The transverse nodes; ``shift`` moves the cladding nodes (another mesh)."""
    y = lines([0.0, Y0, Y1, Y2, A], h)
    if shift:
        inner = (y > 0) & (y < A) & ~np.isin(y, [Y0, Y1, Y2])
        y = y.copy()
        y[inner] += shift * np.diff(y).min() * np.sin(7.3 * y[inner])
    return y


def step_meshes(directory: Path, *, h: float = 0.1, shift: float = 0.0):
    """The meshes ``(A, B)`` of the step's two blocks, B's transverse nodes shifted.

    A spans ``[0, X_A]`` with its plane ``mid`` on the right; B spans
    ``[X_B, X_END]`` with ``mid`` on the left. With ``shift`` the plane of B
    is not conforming to the plane of A.
    """
    mesh_a = box_mesh(
        directory / "a.msh",
        lines([0.0, X_STEP, X_A], h),
        _step_y(h, 0.0),
        lambda x, y: "core" if Y0 < y < (Y1 if x < X_STEP else Y2) else "clad",
        ports=("left", "mid"),
    )
    mesh_b = box_mesh(
        directory / "b.msh",
        lines([X_B, X_BACK, X_END], h),
        _step_y(h, shift),
        lambda x, y: "core" if Y0 < y < (Y2 if x < X_BACK else Y1) else "clad",
        ports=("mid", "right"),
    )
    return mesh_a, mesh_b


def polarity_points(x_um: float, ranks: Sequence[int], y=None) -> dict:
    """``{rank: (point, "+z")}`` reading each mode at ``y`` (default `POLARITY_Y`)."""
    y = POLARITY_Y if y is None else y
    return {m: ((x_um * 1e-6, y[m - 1] * 1e-6), "+z") for m in ranks}


def step_models(
    meshes,
    *,
    n: int,
    polarity: tuple[Mapping | None, Mapping | None] = (None, None),
    eps_core=EPS_CORE,
    tan_delta: float = 0.0,
    absorbing: bool = False,
):
    """The models ``(A, B)``: a lead port, ``n`` kept modes of section ``s``.

    Block A has the exposed port ``in`` on the left, block B ``out`` on the
    right; ``polarity`` is each block's ``kept_modes`` polarity. The walls are
    PEC, or absorbing lines (an open guide) with ``absorbing``.
    """
    design = not isinstance(eps_core, float)
    core = gfem.Material(eps_r=EPS_CORE if design else eps_core, tan_delta=tan_delta)
    if design:  # a design value: a differentiable field
        core = dataclasses.replace(
            gfem.MaterialField.uniform("core", core), eps_r=eps_core
        )
    mats = {"core": core, "clad": gfem.Material(eps_r=EPS_CLAD)}
    wall = gfem.Absorbing("side") if absorbing else gfem.PEC("side")
    mesh_a, mesh_b = meshes
    ports_a = [
        gfem.WavePort("in", "left"),
        *gfem.WavePort.kept_modes("s", "mid", n, polarity=polarity[0]),
    ]
    ports_b = [
        *gfem.WavePort.kept_modes("s", "mid", n, polarity=polarity[1]),
        gfem.WavePort("out", "right"),
    ]
    models = (
        gfem.Model(mesh_a, materials=mats, boundaries=[wall], ports=ports_a),
        gfem.Model(mesh_b, materials=mats, boundaries=[wall], ports=ports_b),
    )
    if not design:
        return models
    return tuple(m.replace(mesh=jax.lax.stop_gradient(m.mesh)) for m in models)


def solve_step(models, polarization: str = "Ez", frequency: float = F_STEP):
    """The two blocks solved (every port excited), without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return tuple(
            gfem.solve(
                m,
                gfem.Driven(frequency, polarization=polarization),
                options=STEP_OPTIONS,
            )
            for m in models
        )
