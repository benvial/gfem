"""Run the 3D bend topology optimization and write its run record.

Usage: python tools/bend3d_run.py STAGE --out DIR [--profile full|smoke]
[--record DIR] [--bddc]

The 2D bend tutorial's recipe (docs/tutorials/bend.ipynb) on a real stack: a
0.8 x 0.4 um Si3N4 strip in SiO2 (meshes: tools/bend3d_mesh.py), solved on
the mirror half model z >= 0 with a PMC at z = 0, |S21|^2 of the TE0 mode at
1.55 um. The run needs a 16-core / 64 GiB Linux machine and the ``mkl``
extra (about 1 h); the docs page ``docs/tutorials/bend3d.ipynb`` reads its
record and solves nothing, and refuses a record whose script hashes differ
from these files', so editing either forces a rerun. ``just bend3d-data``
runs, in ``uv run --extra mkl --group docs``::

    python tools/bend3d_run.py all --out WORK
    python tools/bend3d_run.py merge --out WORK --record docs/tutorials/data

Each stage runs in its own process (``all`` runs them in order, each as a
subprocess), saves its outputs in ``DIR/stages`` and reads its
predecessors', so a failed stage reruns alone:

- ``optimize``: 42 Adam iterations on the h = 0.1 um mirror half model
  (PARDISO, about 23-28 GB); the history, snapshots and designs.
- ``half``: forward re-solves on that mesh: the last iterate, ``final``
  (fields kept: field cuts and fluxes, port n_eff), the eroded, nominal and
  dilated variants, the arc (fields kept) and corner designs, and the
  straight guide.
- ``spectrum``: ``final`` over 13 wavelengths, 1.2 to 1.8 um.
- ``full``: ``final`` on the full stack (about 53 GB).
- ``box``: ``final`` in a 7 x 7 um box (about 50 GB; ``--bddc`` if it does
  not fit, the one stage the flag applies to: ``all --bddc`` passes it to
  ``box`` only).
- ``fine``: ``final`` on the h = 0.07 um mesh, BDDC + GMRES at 1e-6.
- ``merge``: the run record, ``bend3d.json`` (provenance, settings,
  observations; no result) and ``bend3d.npz`` (every solver output), into
  ``--record`` (default ``DIR``). It refuses a record over 1 MB, or one whose
  stages ran at another commit, with other scripts or packages, on another
  machine (a rerun in another container of the same machine merges), or
  with ``src/`` or ``tools/`` uncommitted.

``--profile smoke`` runs every stage at h = 0.4 um (``fine`` at 0.25 um),
with 2 iterations and 3 wavelengths: a complete record in minutes on a
laptop, for development only.

The commit comes from ``git`` in a checkout, else from the
``GFEM_RUN_COMMIT`` environment variable (a ``git archive`` export); the host
label from ``GFEM_RUN_HOST``, else the host name.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import bend3d_mesh as bm
import jax
import jax.numpy as jnp
import numpy as np
import structured_meshes as sm
from scipy.constants import c, mu_0

import gfem
from gfem import design

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("tools/bend3d_run.py", "tools/bend3d_mesh.py")
CODE = ("src", "tools")  # committed, for a record

um = 1e-6
WAVELENGTH = 1.55 * um
F0 = c / WAVELENGTH  # 193.4 THz
PORTS = ("p1", "p2")


# --- the stack ---------------------------------------------------------------


def sellmeier(wavelength_um, terms):
    """The index n with n^2 = 1 + sum B l^2 / (l^2 - C^2), (B, C um) in ``terms``."""
    l2 = wavelength_um**2
    return float(np.sqrt(1 + sum(b * l2 / (l2 - cc**2) for b, cc in terms)))


#: Sellmeier terms: LPCVD Si3N4 (Luke et al., Opt. Lett. 40, 4823 (2015)) and
#: fused silica (Malitson, J. Opt. Soc. Am. 55, 1205 (1965)).
SI3N4_TERMS = ((3.0249, 0.1353406), (40314.0, 1239.842))
SIO2_TERMS = ((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161))
N_CORE = sellmeier(WAVELENGTH / um, SI3N4_TERMS)  # 1.996280
N_CLADDING = sellmeier(WAVELENGTH / um, SIO2_TERMS)  # 1.444024
CORE = gfem.Material(eps_r=N_CORE**2, name="Si3N4")
CLADDING = gfem.Material(eps_r=N_CLADDING**2, name="SiO2")
INDICES = {
    "core": ("Si3N4", N_CORE, "Luke et al., Opt. Lett. 40, 4823 (2015)", SI3N4_TERMS),
    "cladding": ("SiO2", N_CLADDING, "Malitson, JOSA 55, 1205 (1965)", SIO2_TERMS),
}

# --- the recipe (the 2D tutorial's) -----------------------------------------

PITCH = 0.05 * um  # 60 x 60 pixels over the 3 x 3 um design square
FILTER_RADIUS = 0.3 * um
LEARNING_RATE = 0.1
THRESHOLD = 0.5
ROBUST_BETA, ROBUST_DELTA = 64.0, 0.1
BOX_7UM = 3.5  # um, the half-width of the larger box


@dataclasses.dataclass(frozen=True)
class Profile:
    """The size of a run.

    Attributes:
        h: The in-plane cell (um) of the optimization and of every re-solve
            but ``fine``.
        h_fine: The in-plane cell (um) of the ``fine`` re-solve.
        betas: The projection sharpness of each continuation step.
        steps_per_beta: The Adam iterations per beta.
        n_wavelengths: The spectrum's wavelengths, uniform over 1.2-1.8 um.
    """

    h: float
    h_fine: float
    betas: tuple[float, ...]
    steps_per_beta: int
    n_wavelengths: int

    @property
    def iterations(self):
        """The number of optimization iterations."""
        return len(self.betas) * self.steps_per_beta

    @property
    def wavelengths(self):
        """The spectrum's wavelengths (m)."""
        return np.linspace(1.2, 1.8, self.n_wavelengths) * um


PROFILES = {
    "full": Profile(
        h=0.1,
        h_fine=0.07,
        betas=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        steps_per_beta=7,
        n_wavelengths=13,
    ),
    "smoke": Profile(
        h=0.4, h_fine=0.25, betas=(1.0, 2.0), steps_per_beta=1, n_wavelengths=3
    ),
}


def model(mesh, *, half=True):
    """The bend on the stack: SiO2, the two Si3N4 arms, the design volume.

    A second-order absorbing boundary on every outer face, the top and bottom
    included; on the mirror half model a PMC on the plane z = 0, where the
    TE-like fields are even. A TE0 wave port on each arm, over a 4.0 um wide
    window through the whole height, with PEC window edges.
    """
    return gfem.Model(
        mesh,
        materials={"cladding": CLADDING, "core": CORE, "design": CLADDING},
        boundaries=[
            gfem.Absorbing("outer", order=2),
            *([gfem.PMC("mirror")] if half else []),
        ],
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )


def solver_options(stage):
    """The solver of each stage: direct (PARDISO), or BDDC + GMRES on h = 0.07.

    The direct solve keeps 4 GiB for the gradient: at 1.1M DOF one solve's
    retained entry is about 1.8 GB, and a smaller budget evicts it, so the
    backward pass would factorize again.
    """
    if stage == "fine":
        return gfem.SolverOptions(linear_solver="bddc_gmres", tolerance=1e-6)
    return gfem.SolverOptions(linear_solver="direct", retain_memory=4 * 2**30)


def layer_grid(mesh):
    """The design volume's pixels: 0.05 um over its 3 x 3 um footprint."""
    return design.LayerGrid(mesh, "design", pitch=PITCH)


def density_filter(grid):
    """The Helmholtz filter of radius 0.3 um on the pixels."""
    return design.helmholtz_filter(grid, radius=FILTER_RADIUS)


def filtered(rho, blur):
    """Design parameters -> filtered density: mirror about y = x, filter."""
    return blur(0.5 * (rho + rho.T))


def physical(rho, beta, blur):
    """Design parameters -> physical density: mirror about y = x, filter, project."""
    return design.project(filtered(rho, blur), beta)


def material(density, grid):
    """The design volume's permittivity: 0 is SiO2, 1 is Si3N4, linear between."""
    return gfem.MaterialField.interpolate(grid.to_elements(density), CLADDING, CORE)


def missed_power(bend, grid, blur):
    """``loss(rho, beta)``: 1 - |S21|^2 at 1.55 um, aux (|S21|^2, core fraction)."""
    problem = gfem.Driven(frequency=F0, excite=None)
    options = solver_options("optimize")

    def loss(rho, beta):
        density = physical(rho, beta, blur)
        bent = bend.with_material("design", material(density, grid))
        t = jnp.abs(gfem.solve(bent, problem, options=options).s["p2", "p1"][0]) ** 2
        return 1 - t, (t, jnp.mean(density))

    return loss


# --- the hand-drawn designs --------------------------------------------------

GUIDE = bm.CORE_W / 2 * um  # the guides' half-width, 0.4 um
EDGE = -bm.DESIGN * um  # the arms enter the design square at -1.5 um


def area_fraction(inside, grid, samples=8):
    """Per pixel (rows y, columns x): the fraction of its area where ``inside``."""
    offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * grid.pitch
    xs = np.asarray(grid.x)[None, :, None, None] + offsets[None, None, None, :]
    ys = np.asarray(grid.y)[:, None, None, None] + offsets[None, None, :, None]
    return np.broadcast_to(inside(xs, ys), (*grid.shape, samples, samples)).mean(
        axis=(2, 3)
    )


def arc_design(grid):
    """A quarter ring 0.8 um wide, radius 1.5 um about (-1.5, -1.5) um."""

    def inside(x, y):
        r = np.hypot(x - EDGE, y - EDGE)
        return (np.abs(r + EDGE) <= GUIDE) & (x > EDGE) & (y > EDGE)

    return area_fraction(inside, grid)


def corner_design(grid):
    """The two arms extended until they meet in a sharp L."""

    def inside(x, y):
        return ((np.abs(y) <= GUIDE) & (x <= GUIDE)) | (
            (np.abs(x) <= GUIDE) & (y <= GUIDE)
        )

    return area_fraction(inside, grid)


def straight_design(grid):
    """The straight guide's core through the design square, |y| < 0.4 um."""
    return area_fraction(lambda _, y: np.abs(y) <= GUIDE, grid)


# --- measurements on a kept field --------------------------------------------

CUT_STEP = 0.025 * um
#: Sample centres across the 5 um box (x and y) and up the half stack (z).
CUT_ALONG = -bm.BOX * um + CUT_STEP * (
    np.arange(round(2 * bm.BOX * um / CUT_STEP)) + 0.5
)
CUT_HEIGHT = CUT_STEP * (np.arange(round(bm.Z_TOP * um / CUT_STEP)) + 0.5)
CUT_IN_X = CUT_OUT_Y = -2.0 * um  # across each arm, between port and design
FLUX_UP_Z = 1.7 * um  # 0.1 um under the top face


def s_matrix(s):
    """S over ("p1", "p2"), complex128, ``(frequency, to, from)``."""
    return np.moveaxis(
        np.array([[np.asarray(s[a, b]) for b in PORTS] for a in PORTS]), -1, 0
    ).astype(np.complex128)


def _plane(axis, value, u, v):
    """Points (len(u), len(v), 3) on coordinate ``axis`` = ``value``."""
    uu, vv = np.meshgrid(u, v, indexing="ij")
    others = [k for k in range(3) if k != axis]
    points = np.zeros((*uu.shape, 3))
    points[..., axis], points[..., others[0]], points[..., others[1]] = value, uu, vv
    return points


def _sample(field, points):
    return np.asarray(field.at(points.reshape(-1, 3))).reshape(points.shape)


def _magnitude(e):
    return (np.linalg.norm(e, axis=-1) / 1e6).astype(np.float16)  # |E|, MV/m


def _flux(field, points, axis):
    """Midpoint-rule time-averaged Poynting flux along +``axis`` (W)."""
    e, b = _sample(field, points), _sample(field.b(), points)
    s = 0.5 * np.real(np.cross(e, np.conj(b / mu_0)))[..., axis]
    return float(s.sum() * CUT_STEP**2)


def field_maps(result):
    """The cuts and fluxes of a kept solve, port p1 fed (mirror half model).

    |E| (MV/m, float16) on the core mid-plane z = 0 (rows y, columns x), on
    the output arm's cross-section y = -2.0 um and on the input arm's
    x = -2.0 um (rows across, columns z); the Poynting flux through x = -2.0
    um (+x) and through z = 1.7 um over the whole box (+z).
    """
    e = result.e_field(F0, excite="p1")
    top = _plane(2, 0.0, CUT_ALONG, CUT_ALONG)[..., [1, 0, 2]]  # rows y, columns x
    out = _plane(1, CUT_OUT_Y, CUT_ALONG, CUT_HEIGHT)
    into = _plane(0, CUT_IN_X, CUT_ALONG, CUT_HEIGHT)
    up = _plane(2, FLUX_UP_Z, CUT_ALONG, CUT_ALONG)
    return {
        "cut_top": _magnitude(_sample(e, top)),
        "cut_out": _magnitude(_sample(e, out)),
        "cut_in": _magnitude(_sample(e, into)),
        "flux_in": _flux(e, into, 0),
        "flux_up": _flux(e, up, 2),
    }


@contextlib.contextmanager
def gmres_iterations():
    """Collect NGSolve's GMRES iterations, one entry per solved column.

    It counts every NGSolve ``GMResSolver`` solve in the block: one per
    right-hand side that BDDC + GMRES solves on the operator without the
    wave ports' low-rank term, so each column of that term (Woodbury) as
    well as each excitation: more entries than excited ports.
    """
    import ngsolve.solvers

    counts, base = [], ngsolve.solvers.GMResSolver

    class Counted(base):
        def Solve(self, *args, **kwargs):  # noqa: N802 (NGSolve's name)
            out = super().Solve(*args, **kwargs)
            counts.append(int(self.iterations))
            return out

    ngsolve.solvers.GMResSolver = Counted
    try:
        yield counts
    finally:
        ngsolve.solvers.GMResSolver = base


# --- the stages ---------------------------------------------------------------


#: The stages ``--bddc`` applies to: the 7 um box, if its direct solve does
#: not fit (``fine`` always runs BDDC + GMRES).
BDDC_STAGES = ("box",)
BDDC = gfem.SolverOptions(linear_solver="bddc_gmres", tolerance=1e-6)


def check_mesh(name, bend, grid, *, all_sampled):
    """Refuse a mesh whose model or design pixels are not the run's.

    Raises:
        RuntimeError: An exterior face of ``bend`` has no boundary condition
            (a default PMC), the layer grid is not 60 x 60, or, with
            ``all_sampled``, a pixel is sampled by no element.
    """
    if bend.default_pmc:
        raise RuntimeError(f"{name}: default PMC on {bend.default_pmc}")
    if tuple(grid.shape) != (60, 60):
        raise RuntimeError(f"{name}: layer grid {tuple(grid.shape)}, not (60, 60)")
    dead = int((~np.asarray(grid.sampled)).sum())
    if all_sampled and dead:
        raise RuntimeError(f"{name}: {dead} of {grid.sampled.size} pixels unsampled")


def solve_design(bend, grid, density, options, *, frequency=F0, keep=False):
    """``bend`` with ``density`` in its design volume, both ports excited."""
    return gfem.solve(
        bend.with_material("design", material(density, grid)),
        gfem.Driven(frequency=frequency, excite=None, keep_fields=keep),
        options=options,
    )


class Stage:
    """One stage's run: its work directory, profile and what it records."""

    def __init__(self, name, out, profile, *, bddc=False):
        """Stage ``name`` in work directory ``out``, with ``PROFILES[profile]``.

        Raises:
            ValueError: ``bddc`` for a stage not in `BDDC_STAGES`.
        """
        if bddc and name not in BDDC_STAGES:
            raise ValueError(f"--bddc applies to {', '.join(BDDC_STAGES)} only")
        self.name, self.out, self.profile_name = name, Path(out), profile
        self.profile, self.bddc = PROFILES[profile], bddc
        self.meshes, self.options, self.observations = {}, {}, {}

    def mesh(self, h, **kwargs):
        """``(mesh, name)`` of a `bend3d_mesh.bend3d` mesh, built once per DIR."""
        path = self.out / "meshes" / bm.mesh_name(h, **kwargs)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = sm.write_msh22(
                path.with_suffix(".part.msh"), bm.bend3d(h, **kwargs)
            )
            Path(partial).replace(path)
        mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
        self.meshes[path.stem] = {
            "h_um": h,
            "half": kwargs["half"],
            "kind": kwargs.get("kind", "bend"),
            "box_um": kwargs.get("box", bm.BOX),
            "tets": len(mesh.tetrahedra),
        }
        return mesh, path.stem

    def design_mesh(self, h, *, half=True, **kwargs):
        """``(model, layer grid, name)`` on a `mesh`, after `check_mesh`.

        Every pixel must be sampled on the full profile only: the smoke
        profile's coarse cells leave most unsampled.
        """
        mesh, name = self.mesh(h, half=half, **kwargs)
        bend, grid = model(mesh, half=half), layer_grid(mesh)
        check_mesh(name, bend, grid, all_sampled=self.profile_name == "full")
        return bend, grid, name

    def dof(self, name, result):
        """Record the DOF of mesh ``name`` from a solve that kept its fields."""
        self.meshes[name]["dof"] = int(result.e_fields[0].coefficients.shape[0])

    def solver(self):
        """This stage's `solver_options` (BDDC + GMRES at 1e-6 with ``--bddc``)."""
        options = BDDC if self.bddc else solver_options(self.name)
        self.options = dataclasses.asdict(options)
        return options

    def load(self, stage):
        """The arrays a predecessor ``stage`` saved in this work directory."""
        info, arrays = load_stage(self.out, stage)
        if info["profile"] != self.profile_name:
            raise RuntimeError(
                f"{stage} ran with --profile {info['profile']}, this stage with "
                f"--profile {self.profile_name}: rerun {stage} or change profile."
            )
        return arrays

    def final(self):
        """The optimization's thresholded design ``final``, as floats."""
        return self.load("optimize")["final"].astype(float)


def solve_final(stage, h, *, half=True, **mesh_kwargs):
    """``final`` re-solved on another mesh (fields kept, for its DOF)."""
    bend, grid, name = stage.design_mesh(h, half=half, **mesh_kwargs)
    result = solve_design(bend, grid, stage.final(), stage.solver(), keep=True)
    stage.dof(name, result)
    return s_matrix(result.s)[0]


def optimize(stage):
    """Adam on the raw pixels from a uniform 0.5, beta doubling 1 to 32."""
    import optax  # the docs group's, as in the 2D tutorial

    profile = stage.profile
    bend, grid, _ = stage.design_mesh(profile.h)
    blur = density_filter(grid)
    stage.solver()  # records solver_options("optimize"), which missed_power uses
    value_and_grad = jax.jit(
        jax.value_and_grad(missed_power(bend, grid, blur), has_aux=True)
    )
    n = profile.iterations
    snapshot_at = [round(i * (n - 1) / 3) for i in range(4)]  # 0, 14, 27, 41
    rho = rho_start = jnp.full(grid.shape, 0.5)
    optimizer = optax.adam(LEARNING_RATE)
    state = optimizer.init(rho)
    history, snapshots = [], {}
    for beta in profile.betas:
        for _ in range(profile.steps_per_beta):
            k, tic = len(history), time.perf_counter()
            (_, (t, core)), g = value_and_grad(rho, beta)
            if k in snapshot_at:
                snapshots[k] = np.asarray(physical(rho, beta, blur))
            updates, state = optimizer.update(g, state)
            rho = jnp.clip(optax.apply_updates(rho, updates), 0.0, 1.0)
            t, core = float(t), float(core)
            history.append((t, core, beta, time.perf_counter() - tic))
            print(
                f"iteration {k}: beta = {beta:g}, |S21|^2 = {t:.4f}, "
                f"core fraction = {core:.3f}, {history[-1][3]:.1f} s",
                flush=True,
            )
    last = np.asarray(physical(rho, profile.betas[-1], blur))
    t_history, core_history, beta_history, seconds_history = np.array(history).T
    return {
        "t_history": t_history,
        "core_history": core_history,
        "beta_history": beta_history,
        "seconds_history": seconds_history,
        "rho_start": np.asarray(rho_start, np.float64),
        "rho_final": np.asarray(rho, np.float64),
        "snapshots": np.stack([snapshots[k] for k in snapshot_at]).astype(np.float32),
        "snapshot_iterations": np.array(snapshot_at),
        "last": last.astype(np.float32),
        "final": (last > THRESHOLD).astype(np.uint8),
    }


def half(stage):
    """Forward re-solves on the optimization's mesh, the mirror half model."""
    profile, opt = stage.profile, stage.load("optimize")
    bend, grid, name = stage.design_mesh(profile.h)
    blur = density_filter(grid)
    options = stage.solver()
    rho = jnp.asarray(opt["rho_final"])
    variants = design.robust(filtered(rho, blur), beta=ROBUST_BETA, delta=ROBUST_DELTA)
    arc, corner = arc_design(grid), corner_design(grid)
    out = {}
    final_result = solve_design(bend, grid, stage.final(), options, keep=True)
    stage.dof(name, final_result)
    maps = field_maps(final_result)
    out |= {
        "s_final": s_matrix(final_result.s)[0],
        "n_eff": np.array(
            [complex(np.asarray(final_result.port_modes[p].n_eff)[0]) for p in PORTS]
        ),
        "cut_top_final": maps["cut_top"],
        "cut_out_final": maps["cut_out"],
        "cut_in_final": maps["cut_in"],
        "flux_in": np.float64(maps["flux_in"]),
        "flux_up": np.float64(maps["flux_up"]),
    }
    del final_result
    arc_result = solve_design(bend, grid, arc, options, keep=True)
    maps = field_maps(arc_result)
    out |= {
        "s_arc": s_matrix(arc_result.s)[0],
        "cut_top_arc": maps["cut_top"],
        "cut_out_arc": maps["cut_out"],
    }
    del arc_result
    designs = {
        "last": physical(rho, profile.betas[-1], blur),
        "eroded": variants.eroded,
        "nominal": variants.nominal,
        "dilated": variants.dilated,
        "corner": corner,
    }
    for key, density in designs.items():
        out[f"s_{key}"] = s_matrix(solve_design(bend, grid, density, options).s)[0]
    straight_bend, straight_grid, straight_name = stage.design_mesh(
        profile.h, kind="straight"
    )
    straight = solve_design(
        straight_bend,
        straight_grid,
        straight_design(straight_grid),
        options,
        keep=True,
    )
    stage.dof(straight_name, straight)
    out["s_straight"] = s_matrix(straight.s)[0]
    return out | {
        "eroded": np.asarray(variants.eroded, np.float32),
        "nominal": np.asarray(variants.nominal, np.float32),
        "dilated": np.asarray(variants.dilated, np.float32),
        "arc": arc.astype(np.float32),
        "corner": corner.astype(np.float32),
        "cut_x": CUT_ALONG.astype(np.float32),
        "cut_y": CUT_ALONG.astype(np.float32),
        "cut_z": CUT_HEIGHT.astype(np.float32),
    }


def spectrum(stage):
    """``final`` over the band, both ports excited at every wavelength."""
    profile = stage.profile
    bend, grid, _ = stage.design_mesh(profile.h)
    frequency = c / profile.wavelengths
    result = solve_design(
        bend, grid, stage.final(), stage.solver(), frequency=frequency
    )
    return {
        "spectrum_wavelengths": profile.wavelengths,
        "spectrum_s": s_matrix(result.s),
    }


def full(stage):
    """``final`` on the full stack: checks the mirror half model."""
    return {"s_final_full": solve_final(stage, stage.profile.h, half=False)}


def box(stage):
    """``final`` in the 7 x 7 um box: bounds the in-plane cladding margin."""
    return {"s_final_box7": solve_final(stage, stage.profile.h, box=BOX_7UM)}


def fine(stage):
    """``final`` on the finer mesh: the quoted mesh error."""
    with gmres_iterations() as counts:
        s = solve_final(stage, stage.profile.h_fine)
    stage.observations["gmres_iterations"] = counts
    return {"s_final_h007": s}


RUNS = {
    "optimize": optimize,
    "half": half,
    "spectrum": spectrum,
    "full": full,
    "box": box,
    "fine": fine,
}
STAGES = tuple(RUNS)


# --- provenance and observations ---------------------------------------------


def run_commit(root):
    """``(commit, dirty paths under src/ and tools/)`` of the code that runs.

    In a git checkout at ``root``: ``git rev-parse HEAD`` and ``git status
    --porcelain -- src tools``. Elsewhere (a ``git archive`` export): the
    ``GFEM_RUN_COMMIT`` environment variable, with no dirty paths.

    Raises:
        RuntimeError: ``root`` is no checkout and ``GFEM_RUN_COMMIT`` is unset.
    """
    executable = shutil.which("git")

    def git(*args):
        return subprocess.run(
            [executable, "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    top = None
    if executable:
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            top = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
    if top == Path(root).resolve():
        status = git("status", "--porcelain", "--untracked-files=all", "--", *CODE)
        dirty = [line[3:] for line in status.splitlines()]
        return git("rev-parse", "HEAD").strip(), dirty
    commit = os.environ.get("GFEM_RUN_COMMIT")
    if not commit:
        raise RuntimeError(
            f"{root} is not a git checkout: set GFEM_RUN_COMMIT to the commit "
            "this tree was exported from (git archive)."
        )
    return commit, []


def clean_commit(root):
    """The commit of `run_commit`, which must have clean ``src/`` and ``tools/``.

    Raises:
        RuntimeError: ``src/`` or ``tools/`` has uncommitted changes, or no
            commit is known (see `run_commit`).
    """
    commit, dirty = run_commit(root)
    if dirty:
        raise RuntimeError(
            f"src/ or tools/ differ from commit {commit}: {', '.join(dirty)}. "
            "Commit them and rerun, so the record names the code that ran."
        )
    return commit


def sha256s():
    """The sha256 of the run script and the mesh builder, by path."""
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SCRIPTS}


def _read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _cpu_model():
    """/proc/cpuinfo's model name, else the cpuid brand (``cpu_brand`` on PATH)."""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name") and "unknown" not in line.lower():
            return line.split(":", 1)[1].strip()
    if helper := shutil.which("cpu_brand"):  # gVisor hides the model in /proc
        out = subprocess.run([helper], capture_output=True, text=True, check=False)
        return out.stdout.strip().split("\t")[-1].strip() or "unknown"
    return platform.processor() or "unknown"


def _physical_cores():
    cores, cpu = set(), {}
    for line in [*_read("/proc/cpuinfo").splitlines(), ""]:
        if ":" in line:
            key, value = (s.strip() for s in line.split(":", 1))
            cpu[key] = value
        elif cpu:
            cores.add((cpu.get("physical id"), cpu.get("core id")))
            cpu = {}
    return len(cores) if (None, None) not in cores else os.cpu_count()


def _memory_limit():
    """The cgroup memory limit in bytes, else the physical memory."""
    total = next(
        (
            int(line.split()[1]) * 1024
            for line in _read("/proc/meminfo").splitlines()
            if line.startswith("MemTotal:")
        ),
        None,
    )
    group = _read("/proc/self/cgroup").strip().partition("::")[2]
    for path in (
        f"/sys/fs/cgroup{group}/memory.max",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        value = _read(path).strip()
        if value.isdigit() and (total is None or int(value) < total):
            return int(value)
    return total


def _os_name():
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return platform.platform()


def environment():
    """The machine and package versions a stage ran with."""
    from importlib import metadata

    from gfem._linalg.direct import pardiso_available

    def version(dist):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None

    affinity = getattr(os, "sched_getaffinity", None)
    return {
        "versions": {
            "gfem": gfem.__version__,
            "python": platform.python_version(),
            "ngsolve": version("ngsolve"),
            "jax": jax.__version__,
            "jaxlib": version("jaxlib"),
            "numpy": np.__version__,
            "optax": version("optax"),
            "mkl": version("mkl"),
        },
        "pardiso_available": bool(pardiso_available()),
        "os": _os_name(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "physical_cores": _physical_cores(),
        "threads": len(affinity(0)) if affinity else os.cpu_count(),
        "memory_limit_bytes": _memory_limit(),
        "host": os.environ.get("GFEM_RUN_HOST") or platform.node(),
    }


class PeakMemory:
    """The peak RSS of this process and its descendants, sampled every 0.2 s."""

    def __init__(self):
        """Start sampling, in a daemon thread."""
        self._peak = 0
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        while True:
            self._peak = max(self._peak, self._tree())
            time.sleep(0.2)

    @staticmethod
    def _tree():
        total, todo = 0, [os.getpid()]
        while todo:
            pid = todo.pop()
            for line in _read(f"/proc/{pid}/status").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
            with contextlib.suppress(OSError):
                for task in Path(f"/proc/{pid}/task").iterdir():
                    todo += map(int, _read(task / "children").split())
        return total

    @property
    def peak_bytes(self):
        """The largest sample, or this process's own peak if larger."""
        import resource  # Unix only

        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # kB
        return max(self._peak, self._tree(), own)


def _utc():
    return datetime.now(UTC).isoformat(timespec="seconds")


def _stage_files(out, stage):
    folder = Path(out) / "stages"
    return folder / f"{stage}.json", folder / f"{stage}.npz"


def load_stage(out, stage):
    """``(info, arrays)`` a stage saved in the work directory ``out``."""
    info_path, npz_path = _stage_files(out, stage)
    if not info_path.exists():
        raise FileNotFoundError(
            f"no {stage} outputs in {out}: run "
            f"`python tools/bend3d_run.py {stage} --out {out}` first."
        )
    with np.load(npz_path) as npz:
        return json.loads(info_path.read_text()), {k: npz[k] for k in npz.files}


def command_line(argv):
    """The command that reruns ``argv`` (``sys.argv``) from the repository root."""
    return shlex.join(["python", "tools/bend3d_run.py", *argv[1:]])


def run_stage(name, out, profile, *, bddc=False):
    """Run one stage and save its arrays and record in ``out/stages``."""
    memory, start, tic = PeakMemory(), _utc(), time.perf_counter()
    (commit, dirty), hashes = run_commit(ROOT), sha256s()
    info_path, npz_path = _stage_files(out, name)
    info_path.unlink(missing_ok=True)  # a stage's record exists only once it is done
    stage = Stage(name, out, profile, bddc=bddc)
    arrays = RUNS[name](stage)
    info = {
        "stage": name,
        "profile": profile,
        "command": command_line(sys.argv),
        "start_utc": start,
        "end_utc": _utc(),
        "commit": commit,
        "dirty": dirty,
        "sha256": hashes,
        "environment": environment(),
        "meshes": stage.meshes,
        "solver_options": stage.options,
        "observations": {
            "wall_s": round(time.perf_counter() - tic, 1),
            "peak_rss_bytes": memory.peak_bytes,
            **stage.observations,
        },
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    with npz_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    info_path.write_text(json.dumps(info, indent=1) + "\n")
    observed = info["observations"]
    print(
        f"{name}: {observed['wall_s']} s, "
        f"peak RSS {observed['peak_rss_bytes'] / 2**30:.2f} GiB"
    )


# --- the run record -----------------------------------------------------------

RECORD = ("bend3d.json", "bend3d.npz")
#: The caps on the run record: both files together, and each one (bytes).
#: With these values the first binds; the second is the tutorial data's
#: per-file cap, kept so lowering it alone still refuses.
MAX_RECORD_BYTES = 1_000_000
MAX_FILE_BYTES = 3_000_000

RECORD_KEYS = ("profile", "provenance", "settings", "observations")
PROVENANCE_KEYS = (
    "commit",
    "versions",
    "pardiso_available",
    "os",
    "kernel",
    "cpu_model",
    "physical_cores",
    "threads",
    "memory_limit_bytes",
    "host",
    "stages",
    "sha256",
)
SETTINGS_KEYS = (
    "f0_hz",
    "wavelength_um",
    "spectrum_wavelengths_um",
    "indices",
    "geometry",
    "meshes",
    "order",
    "solver_options",
    "schedule",
    "filter_radius_um",
    "pitch_um",
    "symmetry",
    "threshold",
    "robust",
    "cuts",
    "fluxes",
)
#: The numeric plane positions (um) the page reads, by settings entry.
PLANE_KEYS = {
    "geometry": ("port_window_half_width_um",),
    "cuts": ("cut_in_x_um", "cut_out_y_um"),
    "fluxes": ("flux_in_x_um", "flux_up_z_um"),
}
NPZ_KEYS = (
    *("t_history", "core_history", "beta_history", "seconds_history"),
    *("rho_start", "rho_final", "snapshots", "snapshot_iterations", "last", "final"),
    *("eroded", "nominal", "dilated", "arc", "corner"),
    *("s_last", "s_final", "s_final_full", "s_final_h007", "s_final_box7"),
    *("s_eroded", "s_nominal", "s_dilated", "s_arc", "s_corner", "s_straight"),
    *("n_eff", "spectrum_wavelengths", "spectrum_s", "flux_in", "flux_up"),
    *("cut_top_final", "cut_top_arc", "cut_out_final", "cut_out_arc"),
    *("cut_in_final", "cut_x", "cut_y", "cut_z"),
)


def observation_keys(stage):
    """The observations recorded for ``stage``: wall time, peak memory, ..."""
    keys = ("wall_s", "peak_rss_bytes")
    return (*keys, "gmres_iterations") if stage == "fine" else keys


def _um(length):
    """A length in m as a number of um, rounded off its float noise."""
    return round(length / um, 9)


def settings(profile, infos):
    """The record's settings: the model, the meshes, the solvers, the recipe."""
    meshes = {}
    for stage, info in infos.items():
        for name, entry in info["meshes"].items():
            merged = meshes.setdefault(name, {"stages": []})
            merged.update(entry)
            merged["stages"].append(stage)
    plane = {"step_um": CUT_STEP / um, "rule": "midpoint"}
    return {
        "f0_hz": F0,
        "wavelength_um": WAVELENGTH / um,
        "spectrum_wavelengths_um": [round(v / um, 6) for v in profile.wavelengths],
        "indices": {
            region: {"material": name, "n": n, "source": source, "sellmeier": terms}
            for region, (name, n, source, terms) in INDICES.items()
        },
        "geometry": {
            "core_width_um": bm.CORE_W,
            "core_thickness_um": bm.CORE_T,
            "oxide_above_and_below_um": bm.Z_TOP - bm.CORE_T / 2,
            "z_um": [-bm.Z_TOP, bm.Z_TOP],
            "box_half_width_um": bm.BOX,
            "box_7um_half_width_um": BOX_7UM,
            "design_half_width_um": bm.DESIGN,
            "guides": "input |y| < 0.4 along x < -1.5, output |x| < 0.4 along y < -1.5",
            "port_window_half_width_um": bm.PORT,
            "port_windows": "p1 x = -2.5, |y| < 2.0; p2 y = -2.5, |x| < 2.0; "
            "|z| < 1.8 (z >= 0 on the mirror half model)",
            "straight_guide_length_um": 2 * bm.BOX,
            "mirror_half_model": "z >= 0, PMC on z = 0",
        },
        "meshes": meshes,
        "order": solver_options("optimize").order,
        "solver_options": {
            stage: info["solver_options"] for stage, info in infos.items()
        },
        "schedule": {
            "betas": list(profile.betas),
            "steps_per_beta": profile.steps_per_beta,
            "iterations": profile.iterations,
            "optimizer": "optax.adam",
            "learning_rate": LEARNING_RATE,
            "clip": [0.0, 1.0],
            "start": 0.5,
            "loss": "1 - |S21|^2 at f0",
        },
        "filter_radius_um": FILTER_RADIUS / um,
        "pitch_um": PITCH / um,
        "symmetry": "y = x, 0.5 * (rho + rho.T)",
        "threshold": THRESHOLD,
        "robust": {"beta": ROBUST_BETA, "delta": ROBUST_DELTA},
        "cuts": {
            "quantity": "|E| (MV/m, float16), p1 fed with 1 W incident, mirror "
            "half model (z >= 0)",
            "step_um": CUT_STEP / um,
            "cut_in_x_um": _um(CUT_IN_X),
            "cut_out_y_um": _um(CUT_OUT_Y),
            "axes": "cut_x, cut_y across the box, cut_z up the stack (m, centres)",
            "top": {"plane": "z = 0 (core mid-plane)", "shape": ["y", "x"]},
            "out": {"plane": f"y = {CUT_OUT_Y / um:g} um", "shape": ["x", "z"]},
            "in": {"plane": f"x = {CUT_IN_X / um:g} um", "shape": ["y", "z"]},
        },
        "fluxes": {
            "quantity": "0.5 Re(E x conj(B) / mu_0) (W), p1 fed with 1 W "
            "incident, mirror half model value (z >= 0)",
            "in": {**plane, "plane": f"x = {CUT_IN_X / um:g} um", "towards": "+x"},
            "up": {**plane, "plane": f"z = {FLUX_UP_Z / um:g} um", "towards": "+z"},
            "extent": "the cut points of cut_in (y, z) and of the box (x, y)",
            "flux_in_x_um": _um(CUT_IN_X),
            "flux_up_z_um": _um(FLUX_UP_Z),
        },
    }


def _same_keys(what, got, wanted):
    if set(got) != set(wanted):
        missing, extra = set(wanted) - set(got), set(got) - set(wanted)
        raise ValueError(
            f"{what}: missing {sorted(missing)}, unexpected {sorted(extra)}"
        )


def write_record(record, arrays, dest):
    """Write ``bend3d.json`` and ``bend3d.npz`` into ``dest``, after checks.

    Raises:
        ValueError: A key is missing or unexpected, or the files would pass
            `MAX_FILE_BYTES` each or `MAX_RECORD_BYTES` together (nothing is
            written then).
    """
    _same_keys("record", record, RECORD_KEYS)
    _same_keys("provenance", record["provenance"], PROVENANCE_KEYS)
    _same_keys("settings", record["settings"], SETTINGS_KEYS)
    for entry, keys in PLANE_KEYS.items():
        if missing := [k for k in keys if k not in record["settings"][entry]]:
            raise ValueError(f"settings {entry}: missing {missing}")
    _same_keys("observations", record["observations"], STAGES)
    for stage, seen in record["observations"].items():
        _same_keys(f"observations of {stage}", seen, observation_keys(stage))
    _same_keys("arrays", arrays, NPZ_KEYS)
    text = (json.dumps(record, indent=1) + "\n").encode()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payloads = dict(zip(RECORD, (text, buffer.getvalue()), strict=True))
    sizes = {name: len(payload) for name, payload in payloads.items()}
    for name, size in sizes.items():
        if size > MAX_FILE_BYTES:
            raise ValueError(f"{name} would be {size} bytes, over {MAX_FILE_BYTES}")
    if sum(sizes.values()) > MAX_RECORD_BYTES:
        raise ValueError(
            f"the run record would be {sum(sizes.values())} bytes ({sizes}), "
            f"over {MAX_RECORD_BYTES}"
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (dest / name).write_bytes(payload)


#: What `merge` compares across stages: the environment but the host label
#: and the kernel, which change from one container of a machine to the next.
MACHINE_KEYS = (
    "versions",
    "pardiso_available",
    "os",
    "cpu_model",
    "physical_cores",
    "threads",
    "memory_limit_bytes",
)


def mismatches(info, first, commit, hashes):
    """How a stage's ``info`` differs from the run `merge` writes: [] if not.

    The run is at ``commit`` with scripts of sha256 ``hashes``, the profile
    and machine (`MACHINE_KEYS`) of the optimization's ``first``.
    """
    machine = {k: first["environment"][k] for k in MACHINE_KEYS}
    return [
        f"{what} {got!r}, not {wanted!r}"
        for what, got, wanted in (
            ("commit", info["commit"], commit),
            ("uncommitted", info["dirty"], []),
            ("scripts", info["sha256"], hashes),
            ("profile", info["profile"], first["profile"]),
            ("machine", {k: info["environment"][k] for k in MACHINE_KEYS}, machine),
        )
        if got != wanted
    ]


def merge(out, dest):
    """Write the run record from the stages saved in ``out``, into ``dest``.

    Raises:
        RuntimeError: ``src/`` or ``tools/`` is uncommitted, or a stage ran
            at another commit, with other scripts, another profile or on
            another machine or package versions than the optimization (see
            `mismatches`).
    """
    loaded = {stage: load_stage(out, stage) for stage in STAGES}
    infos = {stage: info for stage, (info, _) in loaded.items()}
    commit, hashes = clean_commit(ROOT), sha256s()
    first = infos["optimize"]
    for stage, info in infos.items():
        if problems := mismatches(info, first, commit, hashes):
            raise RuntimeError(f"rerun {stage}: it ran with {'; '.join(problems)}")
    arrays = {}
    for _, stage_arrays in loaded.values():
        arrays |= stage_arrays
    record = {
        "profile": first["profile"],
        "provenance": {
            "commit": commit,
            **first["environment"],
            "stages": {
                stage: {k: info[k] for k in ("start_utc", "end_utc", "command")}
                for stage, info in infos.items()
            },
            "sha256": hashes,
        },
        "settings": settings(PROFILES[first["profile"]], infos),
        "observations": {stage: info["observations"] for stage, info in infos.items()},
    }
    write_record(record, arrays, dest)
    for name in RECORD:
        print(f"{Path(dest) / name}: {(Path(dest) / name).stat().st_size} bytes")


def main(argv=None):
    """Run a stage, the merge, or ``all`` of them in order (one process each)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=(*STAGES, "merge", "all"))
    parser.add_argument("--out", type=Path, required=True, help="the work directory")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument(
        "--record", type=Path, help="merge: the record's directory (default --out)"
    )
    parser.add_argument(
        "--bddc",
        action="store_true",
        help="box (all: box only): BDDC + GMRES at 1e-6, if the direct solve "
        "does not fit",
    )
    args = parser.parse_args(argv)
    if args.bddc and args.stage not in (*BDDC_STAGES, "all"):
        parser.error(f"--bddc applies to {', '.join(BDDC_STAGES)} (or all) only")
    if args.stage == "all":
        record = ["--record", str(args.record)] if args.record else []
        for stage in (*STAGES, "merge"):
            command = [sys.executable, __file__, stage, "--out", str(args.out)]
            command += ["--profile", args.profile, *record]
            command += ["--bddc"] if args.bddc and stage in BDDC_STAGES else []
            subprocess.run(command, check=True)
    elif args.stage == "merge":
        merge(args.out, args.record or args.out)
    else:
        run_stage(args.stage, args.out, args.profile, bddc=args.bddc)


if __name__ == "__main__":
    main()
