"""Port modes over a frequency grid: the guarded port mode sweep.

A wave port's mode over the whole output grid comes from a small reduced
model instead of one exact solve per frequency:

1. **Exact solves** factor the shifted 2D pencil with SuperLU and run ARPACK
   shift-invert at ``sigma = -1.1 (n_max k0)^2``. At the lowest frequency
   the requested mode is picked in rank order: converged, propagating
   Ritz pairs below the largest bulk wavenumber (the formulation's spurious
   modes sit above it), by decreasing ``Re 1 / (lam - sigma)``. On a
   section that cuts a perfectly matched layer, PML modes (layer energy
   share above 1/2) are dropped too, and the request widens until enough
   candidates remain.
2. **Reduced solves** are Rayleigh-Ritz on the affine pencil projected onto
   a real orthonormal basis of the real and imaginary parts of every exact
   mode.
3. **The guard** is the backward error ``|A x - lam B x| / (|A x| +
   |lam| |B x|)`` of every lifted reduced pair, computed exactly from a QR
   factor of the projected pencil pieces (``O(n^2)`` per frequency). A
   frequency passes at ``max(tolerance, 10 x the backward error of the
   exact solve at the nearest sample)``: the second term is the
   formulation's low-frequency accuracy floor.
4. **Enrichment**: an exact solve at the worst frequency, until every
   frequency passes. With at most three frequencies, each is solved exactly.

Across frequencies the mode is **tracked** by its overlap (L2 inner product)
with the neighbouring tabulated mode, never by ranking, which follows the
physical mode through crossings and ignores spurious Ritz values. Several
modes of one port surface share the sweep: every exact solve and the reduced
basis serve all of them, the guard passes a frequency when every mode passes,
and two modes never follow the same candidate. Each mode is normalized to
1 W. Its phase is fixed at the lowest frequency by a quadrant integral of its
tangential magnetic field and is continuous after it (the overlap with the
previous frequency's mode is real and positive); with a voltage path,
``Re V > 0`` at every frequency; with a polarity, the field at the port's
point along its direction is real and positive at every frequency instead.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants
import scipy.linalg
from jax.interpreters import partial_eval

from gfem import _mpi, _pml, _progress
from gfem._boundaries import (
    FloquetPort,
    Polarity,
    WavePort,
    lead_port,
    wave_port_label,
    wave_port_surfaces,
)
from gfem._checks import is_traced
from gfem._dim2 import refuse_anisotropic
from gfem._engine import dispersion
from gfem._engine import terms as driven_terms
from gfem._engine.cache import Entry, discretization
from gfem._engine.line_mode import LineModeSpace, ModeSpace, mode_space
from gfem._engine.line_mode_sensitivity import line_integrated
from gfem._engine.mode_derivative import (
    ModeDerivative,
    gmres_on,
    mode_derivative,
    piece_series,
    rayleigh_series,
    right_hand_side,
    slope,
)
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.port_mode_normalization import normalize_rows
from gfem._engine.port_mode_sensitivity import section_inputs
from gfem._engine.port_mode_shape import chain_to_nodes
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import BEND_RADIUS_KIND, Term, is_thermal_value
from gfem._engine.solve_request import SolveRequest
from gfem._linalg import report
from gfem._linalg.superlu import (
    EIGS_TOLERANCE,
    ShiftInvert,
    backward_error,
)
from gfem._line_section import LineSection, line_elements, line_section
from gfem._port_section import (
    PLANE_RTOL,
    PortSection,
    anisotropic_absorbing,
    outer_edges_conduct,
    outer_sides,
    port_geometry,
    port_section,
)
from gfem._sweep import port_mode_pass
from gfem._text import hertz, hertz_band
from gfem.errors import (
    LayerClearanceWarning,
    NonDifferentiableError,
    PortModeError,
    PortModeTrackingWarning,
    UnabsorbedBendWarning,
)
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import PortMode
from gfem.results import PortModeField, PortModeResult

__all__ = [
    "PortModeEntry",
    "PortModeRule",
    "PortModeSweep",
    "path_voltage",
    "port_mode_sweep",
    "rows_of",
    "sweeps_on",
    "swept_once",
]

# The shift is this factor above the largest bulk (Re eps)(Re mu).
_TARGET = 1.1
# A Ritz pair propagates when Re beta exceeds this fraction of k0 n_max, and
# exceeds |Im beta| (Re beta^2 > 0: above cutoff; below it a lossy mode, or a
# complex pair of a lossless one, keeps a small Re beta).
_EVANESCENT = 1e-6
# Ritz pairs with a larger backward error are unconverged (ARPACK returns some
# from the formulation's large null space below cutoff); converged ones sit
# near 1e-14, or near 1e-6 at the low-frequency accuracy floor.
_UNCONVERGED = 1e-3
# A reduced pair passes at this multiple of the nearest exact backward error.
_FLOOR = 10.0
# Two candidates whose overlaps with the tracked mode differ by less than
# this cannot be told apart.
_AMBIGUOUS = 0.05
# Relative distance of two propagation constants that are one degenerate mode.
_DEGENERATE = 1e-8
#: How to fix a degenerate requested mode, ending its error.
_DEGENERATE_REMEDY = (
    "Select one polarization with a symmetry plane through the guide's axis "
    "(a mirror half model), or break the cross-section's symmetry with a "
    "WavePort(pec_edges=...) wall."
)
# A surface impedance on the port's edges (the internal inductance of a
# conductor, an inductive sheet) slows the mode beyond the bulk wavenumber;
# propagating candidates may then reach this multiple of it.
_SLOW_WAVE = 2.0
# A candidate with more of its electric energy in a perfectly matched layer
# is a PML mode (measured: guided modes at most 0.29, layer modes at least
# 0.986); a picked mode with more than `_CLEARANCE` warns.
_PML_SHARE = 0.5
_CLEARANCE = 1e-2
# With at most this many frequencies every one is solved exactly.
_EXACT_ONLY = 3
# A bent mode is the candidate of the bent pencil that overlaps the straight
# guide's closed mode most, when that overlap is at least `_BEND_OVERLAP` and
# leads the runner-up by `_BEND_MARGIN`; otherwise the radius is reached in
# 2, 4, ... equal steps of 1/R, up to `_BEND_STEPS`. All three are fitted:
# a single step lands the guided cases (overlap 0.53-0.97, runner-up 0.1 at
# most), while a whispering-gallery mode's best overlap falls to 0.43 against
# 0.39 for its neighbour, and the steps of a 2-step walk reach 0.59 and 0.92.
_BEND_OVERLAP = 0.5
_BEND_MARGIN = 0.1
_BEND_STEPS = 32
# Two eigenvalues this close (relative) are one mode found twice.
_SAME_MODE = 1e-6
# A mode derivative is solved to the eigensolver's own tolerance: the mode is
# good to no better, so a tighter solve buys nothing. A mode at the
# low-frequency floor is good only to its backward error, so its derivative
# is solved to that (the preconditioned system cannot reach 1e-10 there).
_DERIVATIVE_RTOL = EIGS_TOLERANCE
# A new mode adds its part orthogonal to the basis unless that part is below
# this fraction of the mode (rounding noise). It must be tiny: a smooth mode
# has |A x| << |A| |x|, so a part of 1e-8 still carries a backward error ~1e-6.
_NEW_DIRECTION = 1e-12
# The fields of a `PortModeSweep` that differ between the modes of one surface.
_PER_MODE = ("gamma", "residual", "floor", "coords", "voltage", "power")
# A polarity point's field below this fraction of the port's peak fixes no
# sign: the exact node of an odd mode reads about 1e-3 on a coarse mesh, a
# point 10 nm off it about 7e-2.
_POLARITY_GUARD = 1e-2
# A polarity direction whose in-plane part is below this is normal to the
# port plane: it reads no field, refused before the sweep.
_NORMAL_ARROW = 1e-6
# The per-mode fields of a `PortModeSweep` that only a sweep asked for its
# modes' dispersion carries.
_DISPERSION_FIELDS = ("n_g", "beta_2", "beta_3")
# Gauss-Legendre rule for voltage line integrals: points per piece, pieces.
_GAUSS = np.polynomial.legendre.leggauss(4)
_PIECES = 32
_Z0 = scipy.constants.mu_0 * scipy.constants.c


@dataclasses.dataclass(frozen=True, eq=False)
class PortModeSweep:
    r"""One wave port's mode over a frequency grid.

    The field at ``frequency[i]`` is ``basis @ coords[i]`` (full coefficient
    vector of ``(E_t, E_n)`` on ``space``, V/m at 1 W). The output basis is
    real and fixed; for a planar port it splits into a tangential and a
    normal block, so the field is affine in ``coords`` even though the
    pencil vector is ``(gamma E_t, E_n)``. Only this output basis is split:
    the sweep's reduced model holds the real and imaginary parts of the
    pencil vectors. A line port's field is its scalar mode ``u`` (Ez in V/m
    or Hz in A/m, at 1 W/m), and its basis is the reduced one.

    Attributes:
        space: The discrete port mode problem.
        frequency: ``(n_f,)`` Hz, in the order given.
        gamma: ``(n_f,)`` propagation constants ``alpha + i beta``, 1/m.
        residual: ``(n_f,)`` backward error of each frequency's mode.
        solved: ``(n_f,)`` True where the mode is an exact solve.
        floor: ``(n_f,)`` the residual each frequency had to reach.
        basis: ``(ndof, m)`` real.
        coords: ``(n_f, m)`` complex.
        voltage: ``(n_f,)`` complex voltage along the port's voltage path,
            V; NaN without one.
        power: ``(n_f,)`` complex power of the mode through the port, W, of
            modulus 1.
        lam: ``(n_f,)`` the eigenvalue :math:`\gamma^2` of each frequency's pencil
            pair, per local unit squared; ``None`` on a sweep rebuilt from
            arrays that lack it.
        y: ``(n_f, m)`` the pencil vector of each frequency in the reduced
            basis ``v``; ``None`` likewise.
        v: ``(n_free, m)`` the real reduced basis of pencil vectors;
            ``None`` likewise.
        n_g: ``(n_f,)`` complex group index
            :math:`(c/i)\,\mathrm{d}\gamma/\mathrm{d}\omega`; ``None`` for a
            mode the sweep was not asked the dispersion of.
        beta_2: ``(n_f,)`` :math:`(1/i)\,\mathrm{d}^2\gamma/\mathrm{d}\omega^2`,
            s²/m, or ``None``.
        beta_3: ``(n_f,)`` :math:`(1/i)\,\mathrm{d}^3\gamma/\mathrm{d}\omega^3`,
            s³/m, or ``None``.
    """

    space: ModeSpace
    frequency: np.ndarray
    gamma: np.ndarray
    residual: np.ndarray
    solved: np.ndarray
    floor: np.ndarray
    basis: np.ndarray
    coords: np.ndarray
    voltage: np.ndarray
    power: np.ndarray
    lam: np.ndarray | None = None
    y: np.ndarray | None = None
    v: np.ndarray | None = None
    n_g: np.ndarray | None = None
    beta_2: np.ndarray | None = None
    beta_3: np.ndarray | None = None

    @property
    def z_pv(self) -> np.ndarray:
        """Complex power-voltage impedance ``|V|^2 / (2 conj(P))``, ohm."""
        return np.abs(self.voltage) ** 2 / (2.0 * np.conj(self.power))

    @property
    def z_ref(self) -> np.ndarray:
        """The wave port's real reference impedance ``|V|^2 / (2 |P|)``, ohm."""
        return np.abs(self.voltage) ** 2 / 2.0


@dataclasses.dataclass(eq=False)
class PortModeEntry(Entry):
    """What a port mode forward keeps for its backward: the surface's sweeps.

    Attributes:
        sweeps: One `PortModeSweep` per mode of the surface (they share the
            basis).
    """

    sweeps: list[PortModeSweep] = dataclasses.field(default_factory=list)

    @property
    def nbytes(self) -> int:
        """Host memory held, in bytes."""
        first = self.sweeps[0] if self.sweeps else None
        if first is None:
            return 0
        shared = int(first.basis.nbytes) + int(np.asarray(first.v).nbytes)
        return shared + sum(
            int(s.coords.nbytes + np.asarray(s.y).nbytes) for s in self.sweeps
        )

    def free(self, *, return_plans: bool = False) -> None:
        """Drop the sweeps."""
        del return_plans
        self.sweeps = []


class PortModeRule:
    """Host-side solve of a `PortMode` problem on a `Model`.

    Construction runs the static checks; `forward` extracts the port's
    cross-section and runs the port mode sweep over its frequencies. Under
    `jax.grad`, ``gamma`` (so ``n_eff``), ``Z_PV`` and the field (``field``,
    the full DOF vector of each frequency, read through
    `gfem.results.PortModeField.at`) differentiate with respect to the
    material, law, sheet and `Impedance` inputs of the section and the face's
    node coordinates (`gfem._sweep.port_mode_pass`; a line port's mode
    differentiates material field values only). The field's node derivative
    is the Lagrangian one, at fixed DOFs, and an objective that depends on the
    phase of a mode's field raises; a point read (``at``) does not
    differentiate with respect to the nodes. ``residual`` and
    the group index and dispersion (``n_g``, ``beta_2``, ``beta_3``) raise
    `gfem.NonDifferentiableError` and a thermal property gets an exact zero.
    """

    physics = "em"
    form = "port mode"
    #: Material properties the port mode problem reads.
    reads = frozenset({"eps_r", "mu_r", "tan_delta", "conductivity"})
    #: The outputs a cotangent may reach.
    differentiable_outputs = frozenset({"gamma", "Z_PV", "field"})
    #: An output's refusal reads the wanted inputs (`nondifferentiable_output`).
    input_aware = True

    def __init__(
        self,
        model: Model,
        problem: PortMode,
        options: SolverOptions,
        *,
        comm: Any = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.PortMode`).

        ``comm`` is the resolved communicator of a distributed solve: every result
        is the one sweep rank 0 ran.

        The options' ``device``, ``linear_solver`` and ``eigen_solver`` are
        ignored: the 2D port mode problem always runs SuperLU and SciPy on
        the CPU, as it does inside a driven solve.

        Raises:
            ValueError: ``problem.port`` is not a wave port of the model.
        """
        if not isinstance(problem, PortMode):
            raise TypeError(f"Expected gfem.PortMode, got {problem!r}.")
        refuse_anisotropic(model)
        if problem.bend is not None and model.mesh.dim == 2:
            raise NotImplementedError(
                f"PortMode({problem.port!r}): a bend is a 3D port mode problem; a "
                "line port on a 2D mesh has none in this version."
            )
        self.problem = problem
        self.options = options
        self.comm = comm
        self.port = _wave_port(model, problem.port)
        self.mode = self.port.mode if problem.mode is None else problem.mode
        # its own mode: the surface's joint sweep, with the surface's wave
        # ports in model order, as a driven problem runs it; a bent mode has
        # its own section, whatever the surface's other ports see
        if problem.mode is None and problem.bend is None:
            self._ports = list(wave_port_surfaces(model.ports)[self.port.surface])
            self._index = self._ports.index(self.port)
        else:
            # the polarity's arrow is drawn on the port's own mode: another
            # mode takes the default sign
            own = self.mode == self.port.mode
            self._ports = [
                self.port if own else dataclasses.replace(self.port, polarity=None)
            ]
            self._index = 0
        self._model = model
        self._explicit: dict[tuple, Any] = {}
        self._radius: float | None = None
        self._nodes: np.ndarray | None = None
        dispersion.warn_fitted_range(
            model,
            problem.frequency,
            "port mode solve",
            comm=comm,
            regions=_section_regions(model, self.port),
        )

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output (one value per frequency).

        ``field`` is the mode's full DOF vector on the port's section, of a
        static width: that of the discrete space built from the mesh alone.
        """
        n = (len(self.problem.frequency),)
        return {
            "gamma": jax.ShapeDtypeStruct(n, jnp.complex128),
            "Z_PV": jax.ShapeDtypeStruct(n, jnp.complex128),
            "residual": jax.ShapeDtypeStruct(n, jnp.float64),
            "solved": jax.ShapeDtypeStruct(n, jnp.bool_),
            "field": jax.ShapeDtypeStruct(
                (*n, self._field_width(model)), jnp.complex128
            ),
        } | {
            name: jax.ShapeDtypeStruct(n, jnp.complex128) for name in _DISPERSION_FIELDS
        }

    def _field_width(self, model: Model) -> int:
        if model.mesh.dim == 2:
            # an H1 line of n_seg segments: n_seg * order + 1 DOFs
            n_seg = len(line_elements(model, self.port)[0])
            return n_seg * self.options.order + 1
        return self._geometry(model).space.ndof

    def _geometry(self, model: Model) -> PortModeSpace | LineModeSpace:
        """The port's discrete space from the mesh alone (`port_geometry`)."""
        if model.mesh.dim == 2:
            section = line_section(model, self.port, geometry=True)
            return LineModeSpace(
                section,
                self.options.order,
                self.problem.polarization,  # ty: ignore[invalid-argument-type]
            )
        section = port_geometry(model, lead_port(self._ports))
        return PortModeSpace(section, self.options.order, geometry=True)

    def role(self, kind: tuple) -> str:
        """``"read"`` if the port mode problem depends on input ``kind``.

        The electromagnetic material properties, the node coordinates and
        any boundary value are read; thermal properties and thermal boundary
        values are not.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        return "unread" if is_thermal_value(kind) else "read"

    def unsupported_input(self, kind: tuple) -> str | None:
        """Why a read input cannot be differentiated here, or ``None``.

        A line port's mode differentiates the material field values only
        (as a 2D driven solve does): its nodes, law parameters and boundary
        values are refused. A wave port's face is differentiated by the
        section's shape kernel.
        """
        if self._model.mesh.dim == 2:
            from gfem._engine.driven2d import _NO_GRADIENTS

            return None if kind[0] == "material" else _NO_GRADIENTS
        if kind[0] == "material":
            return anisotropic_absorbing(
                self._model, lead_port(self._ports), region=kind[1]
            )
        if kind[0] == "nodes":
            return anisotropic_absorbing(self._model, lead_port(self._ports))
        return None

    def nondifferentiable_output(self, name: str, kinds: Any = ()) -> str | None:
        """The error message for a cotangent on output ``name``, or ``None``.

        ``residual`` is a diagnostic. The group index and dispersion have no
        backward pass: their gradient reads the mode derivative, which the
        pass of ``gamma`` does not differentiate.
        """
        del kinds  # the field's node derivative exists: no input refuses it
        if name == "residual":
            return (
                f"gfem cannot differentiate the port mode output {name!r} of wave "
                f"port {self.port.name!r}: it is the backward error of the mode "
                "solve, a diagnostic, not a physical quantity. Wrap it in "
                "jax.lax.stop_gradient to use its value without its derivative."
            )
        if name in _DISPERSION_FIELDS:
            return (
                f"gfem cannot differentiate the port mode output {name!r} of wave "
                f"port {self.port.name!r}: the group index and dispersion (n_g, "
                "beta_2, beta_3 and D) are not differentiable in this version; "
                "gamma, n_eff and Z_PV are, and a central difference in "
                "frequency of the gamma rows of one PortMode gives n_g and beta_2 "
                "that differentiate (the 'Shape gradients of mode figures' "
                "recipe). Wrap it in jax.lax.stop_gradient to use its value "
                "without its derivative."
            )
        if name in self.differentiable_outputs:
            return None
        return (
            f"gfem cannot differentiate the port mode output {name!r}. Wrap it in "
            "jax.lax.stop_gradient to use its value without its derivative."
        )

    def forward(
        self, model: Model, *, keep: bool, request: SolveRequest | None = None
    ) -> tuple[dict, Entry | None]:
        """Run the port mode sweep.

        Args:
            model: The model, with concrete leaves.
            keep: Whether a backward pass will read the sweep.
            request: Ignored: a port mode sweep reads no per-call input.

        Returns:
            The per-frequency arrays of `output_specs` plus ``"sweep"`` (the
            `PortModeSweep`), and the entry a backward reads (or ``None``).
            Only the requested port's mode carries its dispersion, also on a
            shared surface.
        """
        del request
        ports = self._ports
        modes = [self.mode] if len(ports) == 1 else [p.mode for p in ports]
        lead = lead_port(ports)
        # the layer meets its target at the lowest frequency, or its pinned one
        stretch = _pml.stretch(model, self.problem.frequency)
        if stretch is not None:
            stretch.warn_thin()
        section = port_section(model, lead, stretch, self.problem.bend)
        if isinstance(section, PortSection) and section.bend is not None:
            self._warn_unabsorbed(section)
        with task_manager(self.options.threads, self.comm):
            space = mode_space(section, self.options.order, self.problem.polarization)
            sweeps, _ = swept_once(
                self.comm,
                space,
                self.problem.frequency,
                modes,
                tolerance=self.options.port_mode_tolerance,
                ports=ports,
                pairs=keep,
                dispersive_modes=(self._index,),
            )
        sweep = sweeps[self._index]
        field = sweep.coords @ sweep.basis.T
        outputs = {
            "gamma": sweep.gamma,
            "Z_PV": sweep.z_pv,
            "residual": sweep.residual,
            "solved": sweep.solved,
            "field": field,
            "sweep": sweep,
        } | {name: getattr(sweep, name) for name in _DISPERSION_FIELDS}
        entry = PortModeEntry("em", sweeps=sweeps) if keep else None
        return outputs, entry

    def _warn_unabsorbed(self, section: PortSection) -> None:
        """Warn `UnabsorbedBendWarning` for a bent section with no absorber."""
        if outer_sides(section) or outer_edges_conduct(section):
            return
        warnings.warn(
            f"WavePort {self.port.name!r}: the bent port mode has no perfectly "
            "matched layer on its outer side (along outward) and its outer edge "
            "is not a perfect conductor, so the field the bend radiates "
            "reflects from that edge: gamma and bend_loss are those of the "
            "closed cavity, not of the open bend. Add a gfem.PML beyond the "
            "caustic on the outer side.",
            UnabsorbedBendWarning,
            stacklevel=_mpi.outside_gfem(),
        )

    def backward(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset,
    ) -> list:
        """The cotangents of the section's coefficients as owner-tetrahedron integrals.

        The 2D work runs on rank 0 alone (an adjoint failure is an agreed
        `PortModeError`); only rank 0 returns pairs and explicit inputs. The
        gradient by each input kind of the section that is not a material
        (laws, sheet fills, `Impedance` values) is kept for `explicit_inputs`.
        """
        self._explicit = {}
        self._nodes = None
        pairs: list = []
        nodes = ("nodes",) in wanted
        radius = BEND_RADIUS_KIND in wanted
        # the other ranks hold an exact zero: every rank returns the kind
        self._radius = 0.0 if radius else None

        def compute() -> dict[str, np.ndarray]:
            pairs.extend(
                self._pass(model, entry, cotangents, nodes=nodes, radius=radius)
            )
            return {}

        _mpi.on_root(self.comm, compute, results=(PortModeError,))
        return pairs if _mpi.rank(self.comm) == 0 else []

    def _pass(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        nodes: bool = False,
        radius: bool = False,
    ) -> list:
        sweeps = entry.sweeps  # ty: ignore[unresolved-attribute]
        sweep = sweeps[self._index]
        # the port as swept: another mode than its own has no polarity
        port = self._ports[self._index]
        space = sweep.space
        section = space.section
        rule = (
            None
            if port.voltage_path is None
            else _voltage_rule(section, np.asarray(port.voltage_path, float), port.name)
        )
        polarity = (
            None
            if port.polarity is None
            else _polarity_row(space, _polarity_rule(space, port.name, port.polarity))
        )
        done = port_mode_pass.mode_pass(
            space,
            frequency=sweep.frequency,
            lam=sweep.lam,
            y=sweep.y,
            v=sweep.v,
            solved=sweep.solved,
            wavenumbers=lambda f: _wavenumbers(section, f),
            volts=None if rule is None else _voltage_row(space, rule),
            polarity=polarity,
            ct_gamma=cotangents.get("gamma"),
            ct_z_pv=cotangents.get("Z_PV"),
            ct_field=cotangents.get("field"),
            who=f"WavePort {self.port.name!r}",
            nodes=nodes,
            voltage_rule=rule,
            radius=radius,
        )
        if radius:
            self._radius = done.radius
        if isinstance(space, LineModeSpace):
            self._explicit = {}
            return [
                line_integrated(section, done.section, len(model.mesh.volume_elements))  # ty: ignore[invalid-argument-type]
            ]
        if done.points is not None:
            self._nodes = chain_to_nodes(
                section,
                model.mesh.triangles,
                done.points,
                len(model.mesh.imported_nodes),
            )
        cot: Any = done.section
        item, kinds = section_inputs(
            model, section, cot, len(model.mesh.volume_elements)
        )
        out = [item]
        explicit: dict[tuple, Any] = dict(done.laws) | kinds
        self._explicit = explicit
        return out

    def explicit_inputs(self) -> dict[tuple, Any]:
        """The gradients by input kind the section computed itself."""
        return self._explicit

    def explicit_problem(self) -> dict[tuple, np.ndarray]:
        """The gradient by the bend's radius, from the 2D forms (rank 0 only)."""
        if self._radius is None:
            return {}
        return {BEND_RADIUS_KIND: np.asarray(self._radius)}

    def explicit_nodes(self) -> np.ndarray | None:
        """The node cotangent of the port face, ``(n_nodes, 3)`` per metre.

        The section kernel's (`gfem._engine.port_mode_shape`), chained to the
        face's nodes: the mode has no 3D volume term, so it is all of the
        gradient by the node coordinates. ``None`` without a node cotangent.
        """
        return self._nodes

    def terms(self, model: Model, *, nodes: bool) -> list:
        """The driven operator's material terms, for the owner-tetrahedron integrals."""
        del nodes
        built = discretization(model.mesh, model._periodic, self.comm).built  # noqa: SLF001
        if model.mesh.dim == 2:
            from gfem._engine.driven2d import scalar_pieces

            pieces, latent = scalar_pieces(
                model,
                built,
                self.options.order,
                self.problem.polarization,  # ty: ignore[invalid-argument-type]
            )
            keep = {"stiffness", "mass", "conductivity", "absorbing"}
        else:
            pieces, latent = driven_terms.pieces(model, built, self.options.order, ())
            keep = {"mass", "conductivity", "curl_curl", "absorbing"}
        return [
            Term(
                p.name,
                p.kind,  # ty: ignore[invalid-argument-type]
                coefficient=p.coefficient,
                region=p.region,
                derivatives=p.derivatives,
                owner=p.owner,
                position=p.position,
            )
            for p in (*pieces, *latent)
            if p.name in keep
        ]

    def lumped_terms(self, model: Model) -> list:
        """No piece moves with a lumped value: the section's mode reads none."""
        del model
        return []

    def result(self, outputs: dict) -> PortModeResult:
        """The public result from the forward outputs.

        Without ``"sweep"`` (a solve traced by JAX) the field is the traced
        DOF vector of ``"field"``, located from the mesh alone when a point is
        read.
        """
        sweep: PortModeSweep | None = outputs.get("sweep")
        polarization = self.problem.polarization
        quantity, unit = _FIELD_NAMES[polarization]
        if sweep is not None:
            field = PortModeField(
                sweep.space, sweep.basis, sweep.coords, self.problem.bend
            )
        else:
            field = PortModeField._on_vector(  # noqa: SLF001
                self.port.name,
                self._locate,
                jnp.asarray(outputs["field"]),
                quantity=quantity,
                unit=unit,
                traced_nodes=is_traced(self._model.mesh.nodes),
                bend=self.problem.bend,
            )
        hz = polarization == "Hz"
        return PortModeResult(
            port=self.port.name,
            mode=self.mode,
            frequency=jnp.asarray(self.problem.frequency),
            gamma=jnp.asarray(outputs["gamma"]),
            Z_PV=jnp.asarray(outputs["Z_PV"]),
            residual=jnp.asarray(outputs["residual"]),
            solved=jnp.asarray(outputs["solved"]),
            e_field=None if hz else field,
            h_field=field if hz else None,
            pml=self._model.pml,
            bend=self.problem.bend,
            n_g=jnp.asarray(outputs["n_g"]),
            beta_2=jnp.asarray(outputs["beta_2"]),
            beta_3=jnp.asarray(outputs["beta_3"]),
            allow_evanescent=self.port.allow_evanescent,
            polarity=self.port.polarity,
        )

    def _locate(self, *, point_read: bool = False) -> PortModeSpace | LineModeSpace:
        """The port's space from the mesh, for the traced field.

        With the node coordinates traced by differentiation the space is built
        at their values: a field's DOF vector moves with them at fixed DOFs.

        Args:
            point_read: The space is for a point read (`PortModeField.at`).

        Raises:
            NonDifferentiableError: A point is read while the node coordinates
                are traced by differentiation: a point value on a moving
                section is not available.
            TypeError: They are traced by `jax.jit`.
        """
        model = self._model
        nodes = model.mesh.nodes
        if is_traced(nodes):
            staged = partial_eval.DynamicJaxprTracer
            if any(isinstance(x, staged) for x in jax.tree.leaves(nodes)):
                raise TypeError(_AT_STAGED)
            if point_read:
                raise NonDifferentiableError(_AT_NODES)
            from gfem.optomechanics import _host_nodes

            model = model.replace(mesh=model.mesh.with_nodes(_host_nodes(model.mesh)))
        return self._geometry(model)


_AT_NODES = (
    "PortModeField.at reads the mode at fixed points while gfem differentiates the "
    "node coordinates: the derivative of a point value on a moving section is not "
    "available in this version. Differentiate gamma, n_eff or Z_PV, or wrap the "
    "node coordinates in jax.lax.stop_gradient."
)
_AT_STAGED = (
    "PortModeResult.e_field.at needs concrete node coordinates to locate the "
    "points on the section; inside jax.jit the model's mesh is traced. Pass the "
    "model as a closed-over constant, or read the field outside jax.jit."
)


def _section_regions(model: Model, port: WavePort) -> set[str]:
    """The volume regions of the solve domain beside ``port``'s surface.

    The regions whose materials the port's cross-section reads, found from
    the mesh's adjacency alone, so no section is built.
    """
    mesh = model.mesh
    topology = mesh._topology  # noqa: SLF001
    active = model._domain.active  # noqa: SLF001
    nb = topology.surface_neighbours[mesh.elements(port.surface)]
    cells = np.unique(nb[(nb >= 0) & active[np.maximum(nb, 0)]])
    names = mesh._region_names()  # noqa: SLF001
    return {names[int(tag)] for tag in np.unique(topology.volume_tags[cells])}


def _wave_port(model: Model, name: str) -> WavePort:
    for p in model.ports:
        if p.name == name:
            if not isinstance(p, WavePort):
                remedy = (
                    " A Floquet port's orders are plane waves with no mode solve: "
                    "Model.propagating_orders lists them, and "
                    "DrivenResult.diffraction_orders gives each order's gamma and "
                    "angles."
                    if isinstance(p, FloquetPort)
                    else ""
                )
                raise ValueError(
                    f"PortMode(port={name!r}): {name!r} is a {type(p).__name__}; the "
                    f"port mode problem needs a WavePort.{remedy}"
                )
            return p
    waves = [p.name for p in model.ports if isinstance(p, WavePort)]
    known = ", ".join(repr(w) for w in waves) if waves else "none"
    raise ValueError(
        f"PortMode(port={name!r}): the model has no port named {name!r}. Its wave "
        f"ports: {known}."
    )


#: A port mode field's quantity and unit by the problem's polarization.
_FIELD_NAMES = {None: ("E", "V/m"), "Ez": ("Ez", "V/m"), "Hz": ("Hz", "A/m")}


# --- the sweep ------------------------------------------------------------------


def port_mode_sweep(
    space: ModeSpace,
    frequency: tuple[float, ...] | np.ndarray,
    modes: list[int],
    *,
    tolerance: float,
    ports: list[WavePort],
    dispersive_modes: Sequence[int] = (),
) -> list[PortModeSweep]:
    """The guarded port mode sweep of one port surface (see the module docstring).

    Every exact solve serves all requested modes, and the reduced model
    shares one basis; each mode is tracked by its own overlap, and two modes
    never follow the same candidate. One mode is today's single-port sweep.

    Args:
        space: The surface's discrete port mode problem.
        frequency: The output grid, Hz (any order).
        modes: Which modes at the lowest frequency, in rank order (distinct).
        tolerance: Backward error a reduced mode must reach.
        ports: The wave port of each mode (its name for messages, its
            voltage path or polarity).
        dispersive_modes: The modes (indices into ``modes``) whose group index and
            dispersion the sweep reports; the others carry ``None``.

    Returns:
        One `PortModeSweep` per mode, sharing ``space``, ``basis`` and
        ``solved``.

    Raises:
        PortModeError: A mode is below cutoff somewhere in the band, or is
            degenerate, or two candidates cannot be told apart, or two modes
            follow the same candidate.
        ValueError: A frequency lies on a lossless pole of a law in the
            cross-section, or a law region there has Re eps <= 0 (a metal),
            checked before any exact solve.

    Warns:
        PortModeTrackingWarning: A tracked mode stops being its ``mode``-th
            in rank order inside the band (a crossing).
    """
    f_given = np.asarray(frequency, float)
    section = space.section
    section.check_frequencies(f_given)
    rules = [
        None
        if p.voltage_path is None
        else _voltage_rule(
            planar(space).section, np.asarray(p.voltage_path, float), p.name
        )
        for p in ports
    ]
    arrows = [
        None if p.polarity is None else _polarity_rule(space, p.name, p.polarity)
        for p in ports
    ]
    order = np.argsort(f_given, kind="stable")
    sweep = _Greedy(
        space,
        f_given[order],
        modes,
        tolerance,
        ports,
        dispersive_modes=dispersive_modes,
    )
    sweep.run()
    sweep.dispersion_pass()
    sweep.log()
    basis, coords, gamma, voltage, power = sweep.normalized(rules, arrows)
    reduced = sweep.reduced.v
    out = np.empty_like(order)
    out[order] = np.arange(len(order))

    def derived(k: int) -> dict[str, np.ndarray]:
        if k not in sweep.dispersive_modes:
            return {}
        rows = sweep.dispersion_rows(k)
        return {name: rows[n, out] for n, name in enumerate(_DISPERSION_FIELDS)}

    return [
        PortModeSweep(
            space=space,
            frequency=f_given,
            gamma=gamma[k, out],
            residual=sweep.residual[k, out],
            solved=sweep.solved[out],
            floor=sweep.floor[k, out],
            basis=basis,
            coords=coords[k, out],
            voltage=voltage[k, out],
            power=power[k, out],
            lam=sweep.lam[k, out],
            y=sweep.rows[k][out],
            v=reduced,
            **derived(k),
        )
        for k in range(len(modes))
    ]


def swept_once(
    comm: Any,
    space: ModeSpace,
    frequency: tuple[float, ...] | np.ndarray,
    modes: list[int],
    *,
    tolerance: float,
    ports: list[WavePort],
    dispersive_modes: Sequence[int] = (),
    derived: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None,
    pairs: bool = False,
) -> tuple[list[PortModeSweep], dict[str, np.ndarray]]:
    """`port_mode_sweep` once, on rank 0 of ``comm``, with its arrays on every rank.

    A `PortModeError` and the sweep's warnings reach every rank, and each
    rank's sweeps stand on its own ``space``. Collective, on every rank of ``comm``.

    Args:
        comm: The resolved communicator.
        space: The surface's discrete port mode problem, built on every rank.
        frequency: The output grid, Hz.
        modes: As `port_mode_sweep`.
        tolerance: As `port_mode_sweep`.
        ports: As `port_mode_sweep`.
        dispersive_modes: As `port_mode_sweep`; a mode it leaves out has NaN rows in
            the dispersion arrays whenever another mode carries them.
        derived: More arrays read off the sweep's basis, computed beside it and
            sent with it, since an assembled form may round differently per rank.
        pairs: Also send each mode's eigenvalue and pencil vector rows and the
            reduced basis (a backward pass reads them).

    Returns:
        The sweeps and ``derived``'s arrays.
    """

    def compute() -> dict[str, np.ndarray]:
        sweeps = port_mode_sweep(
            space,
            frequency,
            modes,
            tolerance=tolerance,
            ports=ports,
            dispersive_modes=dispersive_modes,
        )
        first = sweeps[0]
        arrays: dict[str, np.ndarray] = {
            name: np.stack([getattr(sweep, name) for sweep in sweeps])
            for name in _PER_MODE
        }
        if dispersive_modes:
            missing = np.full(len(first.frequency), complex(math.nan, math.nan))
            for name in _DISPERSION_FIELDS:
                rows = [getattr(sweep, name) for sweep in sweeps]
                arrays[name] = np.stack([missing if r is None else r for r in rows])
        arrays |= {"basis": first.basis, "solved": first.solved}
        if pairs:
            arrays |= {
                "lam": np.stack([np.asarray(sweep.lam) for sweep in sweeps]),
                "y": np.stack([np.asarray(sweep.y) for sweep in sweeps]),
                "v": np.asarray(first.v),
            }
        return arrays | (derived(first.basis) if derived is not None else {})

    # ValueError: a voltage path or a polarity point off the port, on replicated
    # inputs
    arrays = _mpi.on_root(comm, compute, results=(PortModeError, ValueError))
    return sweeps_on(space, frequency, arrays), arrays


def sweeps_on(
    space: ModeSpace,
    frequency: tuple[float, ...] | np.ndarray,
    arrays: dict[str, np.ndarray],
) -> list[PortModeSweep]:
    """The sweeps that stand on ``space`` and on `swept_once`'s ``arrays``.

    What a rank does with the arrays rank 0 sent, and a frequency worker with
    the rows its parent handed it (`rows_of`): no port mode solve.

    Args:
        space: The surface's discrete port mode problem, built here.
        frequency: The frequencies of the arrays' rows, Hz.
        arrays: `swept_once`'s arrays, or their rows at ``frequency``.

    Returns:
        One `PortModeSweep` per mode.
    """
    return [
        PortModeSweep(
            space=space,
            frequency=np.asarray(frequency, float),
            basis=arrays["basis"],
            solved=arrays["solved"],
            **{name: arrays[name][k] for name in _PER_MODE},
            **(
                {"lam": arrays["lam"][k], "y": arrays["y"][k], "v": arrays["v"]}
                if "lam" in arrays
                else {}
            ),
            **{name: arrays[name][k] for name in _DISPERSION_FIELDS if name in arrays},
        )
        for k in range(len(arrays["gamma"]))
    ]


def rows_of(
    arrays: dict[str, np.ndarray], rows: Sequence[int] | np.ndarray
) -> dict[str, np.ndarray]:
    """`swept_once`'s ``arrays`` at the grid indices ``rows``.

    The per-frequency arrays are sliced; the basis and what was derived from
    it are the whole sweep's, so a block of the grid solves on the modes the
    whole grid's sweep found.
    """
    rows = np.asarray(rows, int)
    out = dict(arrays)
    for name in _PER_MODE + _DISPERSION_FIELDS:
        if name in arrays:
            out[name] = arrays[name][:, rows]
    out["solved"] = arrays["solved"][rows]
    return out


@dataclasses.dataclass(frozen=True)
class _Candidates:
    """The converged propagating eigenpairs of one exact solve, in rank order."""

    lam: np.ndarray
    vecs: np.ndarray
    residual: np.ndarray
    #: Each candidate's layer energy share; ``None`` reads as zero (no layer).
    share: np.ndarray | None = None
    #: The solve's shift-invert factor while a mode derivative may need it.
    factor: ShiftInvert | None = None


class _Scan(NamedTuple):
    """One ARPACK call's eigenpairs, their backward errors and layer shares."""

    lam: np.ndarray
    vecs: np.ndarray
    res: np.ndarray
    share: np.ndarray
    #: Which eigenpairs are candidates (propagating, converged, off the layer).
    ok: np.ndarray

    def pick(self, i: int) -> _Pick:
        """Eigenpair ``i``."""
        return _Pick(self.lam[i], self.vecs[:, i], self.res[i], self.share[i])


class _Pick(NamedTuple):
    """One picked eigenpair."""

    lam: Any
    vec: np.ndarray
    res: Any
    share: Any


@dataclasses.dataclass(frozen=True)
class _DerivativeRow:
    r"""A reduced row's mode, mode derivative and the residuals that guard them.

    Attributes:
        derivative: The row's `ModeDerivative`, on the derivative basis'
            coordinates.
        eta: Residual of the mode at the sweep's eigenvalue, on the scale
            of the sweep's own row (`_Reduced.derivative_row`).
        eta1: Residual of the mode derivative equation, on the scale of the
            sweep's own row's right-hand side (`_Reduced.derivative_row`).
        y: The mode's coordinates on the derivative basis.
    """

    derivative: ModeDerivative
    eta: float
    eta1: float
    y: np.ndarray


class _Greedy:
    """State of one port mode sweep over a sorted grid.

    Tracking is certified link by link: a frequency's modes are trusted only
    when chosen by overlap with accurate modes at the previous grid
    frequency. An exact solve whose reference was further away (the top of
    the band at the start, or a frequency after a failing stretch) is
    re-checked once its neighbour below passes, and re-selected if another
    of its candidates continues a mode better.

    Per mode ``k`` (row of ``lam``, ``residual``, ``floor``; entry of
    ``coords`` and of each ``pick``); ``solved`` is shared by the modes.

    The modes ``k`` in ``dispersive_modes`` carry dispersion: each exact solve kept
    for them gives their mode derivative (``derivatives[j][k]``, ``None`` for
    the other modes) while its factor is alive. With ``dispersive_modes`` empty no
    factor outlives its solve and nothing more is computed. After the sweep
    `dispersion_pass` gives the reduced rows theirs.
    """

    def __init__(
        self,
        space: ModeSpace,
        f: np.ndarray,
        modes: list[int],
        tolerance: float,
        ports: list[WavePort],
        *,
        dispersive_modes: Sequence[int] = (),
    ) -> None:
        self.space = space
        self.f = f
        self.modes = list(modes)
        self.ports = list(ports)
        self.tolerance = tolerance
        self.dispersive_modes = tuple(dispersive_modes)
        self.derivatives: dict[int, list[ModeDerivative | None]] = {}
        # reduced rows' dispersion by mode, and the derivatives-only exact
        # solves the guard asked for (their mode derivatives, by (k, row))
        self._reduced_rows: dict[int, np.ndarray] = {}
        self.fallbacks: dict[tuple[int, int], ModeDerivative] = {}
        n, n_modes = len(f), len(self.modes)
        #: Whether any port keeps a mode below cutoff: the scan then also admits
        #: evanescent candidates, ranked after the propagating ones.
        self.evanescent = any(p.allow_evanescent for p in self.ports)
        self.k0 = np.empty(n)
        self.k_max = np.empty(n)
        self.shift = np.empty(n)
        for i, fi in enumerate(f):
            self.k0[i], self.k_max[i], self.shift[i] = _wavenumbers(space.section, fi)
        self.reduced = _Reduced(space)
        self.bent = getattr(space.section, "bend", None) is not None
        # a bent section's closed straight predictor, built on first use
        self._closed: PortModeSpace | None = None
        #: Each mode's pencil vectors in the reduced basis, set by `normalized`.
        self.rows: list[np.ndarray] = [np.zeros((n, 0)) for _ in self.modes]
        self.lam = np.zeros((n_modes, n), complex)
        self.coords: list[list[np.ndarray]] = [[np.zeros(0)] * n for _ in self.modes]
        self.residual = np.full((n_modes, n), np.inf)
        self.solved = np.zeros(n, bool)
        self.floor = np.full((n_modes, n), tolerance)
        # exact solves: candidates and the positions (in shift-inverted order) kept
        self.candidates: dict[int, _Candidates] = {}
        self.pick: dict[int, list[int]] = {}
        # the sweep's size, for its record: its samples' exact solves and the
        # derivatives-only ones of `dispersion_pass`
        self.exact_solves = 0
        self.fallback_solves = 0
        self.rounds = 0

    # --- driver ---------------------------------------------------------------

    def run(self) -> None:
        # the row counts the output frequencies that pass, so it can go down
        # after a re-selection
        with _progress.task(_progress.PORT_MODE_SWEEP, total=len(self.f)) as row:
            self._run(row)

    def _run(self, row: _progress.ProgressRow) -> None:
        n = len(self.f)
        self.exact_solves += 1
        first = self._solve(0)
        if self._lacking(first, 0):
            raise self._cutoff_error(first)
        for k in range(len(self.modes)):
            self._check_degenerate(first, k)
        self._warn_clearance(first)
        self._keep(0, first, [mode - 1 for mode in self.modes])
        del first  # the kept copy dropped its factor; this one would hold it
        if n <= _EXACT_ONLY:
            for j in range(1, n):
                self._exact(j, j - 1, certain=True)
        else:
            self._exact(n - 1, 0, certain=False)
        for _ in range(4 * n + 8):
            passed = self._tabulate()
            if passed is None:  # a sample was re-selected
                self._rebuild()
                continue
            row.update(completed=int(passed.sum()), exact_solves=self.exact_solves)
            if passed.all():
                for k in range(len(self.modes)):
                    self._warn_crossing(k)
                return
            # the worst mode at each frequency decides where to enrich
            excess = (self.residual / self.floor).max(axis=0)
            j = int(np.argmax(np.where(passed, -np.inf, excess)))
            if passed[j - 1]:
                self._exact(j, j - 1, certain=True)
            else:
                good = np.flatnonzero(passed)
                self._exact(j, int(good[np.argmin(np.abs(good - j))]), certain=False)
        raise self._unsettled_error()  # pragma: no cover - a safeguard

    def log(self) -> None:
        """Log the sweep's record (`gfem._linalg.report.port_mode_sweep`).

        After `dispersion_pass`, so that it counts the fallback solves.
        """
        report.port_mode_sweep(
            ports=list(dict.fromkeys(p.name for p in self.ports)),
            frequencies=len(self.f),
            exact_solves=self.exact_solves,
            rounds=self.rounds,
            fallback_solves=self.fallback_solves,
        )

    # --- exact solves -----------------------------------------------------------

    def candidates_at(self, j: int) -> _Candidates:
        """The exact solve's candidates at grid index ``j``, in rank order."""
        return self._solve(j)

    def _solve(self, j: int, tracked: np.ndarray | None = None) -> _Candidates:
        """The exact solve at ``j``.

        On a bent section the closed straight mode predicts the bent one
        (`_predicted`), unless ``tracked`` gives the modes' eigenvalue
        estimates at ``j`` (one per mode, NaN for a mode not wanted): the
        solve then shifts at the lowest wanted mode's estimate.
        """
        space, mode = self.space, max(self.modes)
        bent = self.bent
        covers = None
        if bent and tracked is None:
            predicted = self._predicted(j)
            if predicted is not None:
                return predicted
        elif bent:
            assert tracked is not None  # noqa: S101 - narrowing
            wanted = [
                k
                for k in np.argsort(self.modes, kind="stable")
                if tracked[k] == tracked[k]
            ]
            if wanted:
                self.shift[j] = tracked[wanted[0]].real
                covers = tracked[wanted]
        a, b = space.pencil.at(self.k0[j])
        n = a.shape[0]
        factor = ShiftInvert.factor(a, b, self.shift[j])
        first = max(6, 2 * mode + 4)
        # PML modes crowd the shift: widen until enough candidates survive
        layered = space.section.layer is not None
        wants = itertools.count() if layered else range(2)
        for step in wants:
            want = first * (2**step if layered else 4**step)
            scan = self._scan(
                space, j, pencil=(a, b), factor=factor, want=min(want, n - 2)
            )
            lam, vecs, res, share, ok = scan
            # Order: largest Re of the shift-inverted eigenvalue
            ranked = self._ranked(lam, ok, self.shift[j], j)
            if (len(ranked) >= mode and _covers(lam[ranked], covers)) or want >= n - 2:
                break
        if covers is not None:
            # a shift at the tracked mode is no rank: order by decreasing Re beta
            # as a shift above every mode does
            ranked = ranked[np.argsort(-np.sqrt(-lam[ranked] + 0j).real, kind="stable")]
        return _Candidates(
            lam=lam[ranked],
            vecs=vecs[:, ranked],
            residual=res[ranked],
            share=share[ranked],
            factor=factor if self.dispersive_modes else None,
        )

    def _ranked(
        self, lam: np.ndarray, ok: np.ndarray, shift: float, j: int
    ) -> np.ndarray:
        r"""The admitted candidates' indices in rank order.

        Largest :math:`\mathrm{Re}\,1/(\lambda - \sigma)` first. With a port that
        keeps a mode below cutoff the evanescent candidates come after the
        propagating ones, by ascending :math:`\mathrm{Re}\,\lambda`, so the
        propagating ranks are those of a sweep without it.
        """
        rank = (1 / (lam - shift)).real
        ranked = np.flatnonzero(ok)[np.argsort(-rank[ok], kind="stable")]
        if not self.evanescent:
            return ranked
        beta = np.sqrt(-lam[ranked] + 0j)
        propagating = _propagates(beta, self.k_max[j])
        tail = ranked[~propagating]
        tail = tail[np.argsort(lam[tail].real, kind="stable")]
        return np.concatenate([ranked[propagating], tail])

    def _n_propagating(self, cand: _Candidates, j: int) -> int:
        """How many of ``cand`` (the solve at ``j``) propagate: they come first."""
        if not self.evanescent:
            return len(cand.lam)
        return int(_propagates(np.sqrt(-cand.lam + 0j), self.k_max[j]).sum())

    def _lacking(self, cand: _Candidates, j: int) -> bool:
        """Whether a mode has no candidate at ``j``.

        A port that keeps modes below cutoff needs the candidate to exist; any
        other needs it to propagate.
        """
        n_prop = self._n_propagating(cand, j)
        return any(
            mode > (len(cand.lam) if port.allow_evanescent else n_prop)
            for mode, port in zip(self.modes, self.ports, strict=False)
        ) or max(self.modes) > len(cand.lam)

    def _scan(
        self,
        space: ModeSpace,
        j: int,
        *,
        pencil: tuple[Any, Any],
        factor: ShiftInvert,
        want: int,
        k_max: float | None = None,
    ) -> _Scan:
        """The ``want`` eigenpairs nearest ``factor``'s shift, and which are candidates.

        A candidate propagates, is converged and holds at most `_PML_SHARE`
        of its energy in the layer. ``k_max`` (the sweep's at ``j``) bounds the
        propagation constants.
        """
        a, b = pencil
        lam, vecs = factor.eigs(want)
        beta = np.sqrt(-lam + 0j)
        res = np.array(
            [backward_error(a, b, lam[i], vecs[:, i]) for i in range(len(lam))]
        )
        share = space.layer_share(vecs, lam)
        k_max = self.k_max[j] if k_max is None else k_max
        # a port that keeps a mode below cutoff admits every converged mode off
        # the null space (|beta| ~ 0), as the reduced model does
        admitted = (
            np.abs(beta) > _EVANESCENT * k_max
            if self.evanescent
            else _propagates(beta, k_max)
        )
        ok = (
            admitted
            & (beta.real <= k_max * (1 + 1e-6))
            & (res < _UNCONVERGED)
            & (share <= _PML_SHARE)
        )
        return _Scan(lam, vecs, res, share, ok)

    # --- bent sections: the closed straight mode ------------------------------------

    def _closed_space(self) -> PortModeSpace:
        """The bent section with no bend and no layer: its closed straight guide."""
        if self._closed is None:
            section = self.space.section
            assert isinstance(section, PortSection)  # noqa: S101 - bent ports only
            straight = dataclasses.replace(section, stretch=None, layer=None, bend=None)
            self._closed = PortModeSpace(straight, self.space.order)
        return self._closed

    def _predicted(self, j: int) -> _Candidates | None:
        r"""The bent modes continued from the closed straight modes, at ``j``.

        Mode :math:`m` of a bent port is the bent continuation of the closed
        straight guide's mode :math:`m`: the candidate of the bent pencil
        that overlaps it most. One shift-invert factor at the straight
        fundamental's eigenvalue serves every requested mode; a mode that
        fails the acceptance rule (`_pick`) gets a factor of its own, and, if
        still unclear, is continued in :math:`1/R` (`_continued`). The
        candidates come in rank order, with mode :math:`m` at position
        :math:`m - 1`; ``None`` when the straight guide has fewer modes than
        requested (the usual search then reports the cutoff).
        """
        closed = self._closed_space()
        straight = _Greedy(closed, self.f[j : j + 1], self.modes, 1.0, [])
        pred = straight.candidates_at(0)
        if len(pred.lam) < max(self.modes):
            return None
        space = self.space
        a, b = space.pencil.at(self.k0[j])
        mass = closed.pencil.at(self.k0[j])[1]
        first = max(6, 2 * max(self.modes) + 4)
        kmin = int(np.argmin(self.modes))
        shared = pred.lam[self.modes[kmin] - 1].real
        factor = ShiftInvert.factor(a, b, shared)
        n = a.shape[0]
        want = min(first, n - 2)
        while True:
            scan = self._scan(space, j, pencil=(a, b), factor=factor, want=want)
            # the picks and the candidates between them need this many
            if scan.ok.sum() >= max(self.modes) or want >= n - 2:
                break
            want = min(2 * want, n - 2)
        chosen = [_pick(mass, pred.vecs[:, mode - 1], scan) for mode in self.modes]
        # two modes on one candidate are neither clear
        picks: list[_Pick | None] = [
            None if p is None or chosen.count(p) > 1 else scan.pick(p) for p in chosen
        ]
        scans, shifts = [scan], [shared]
        for k, mode in enumerate(self.modes):
            if picks[k] is not None:
                continue
            x = pred.vecs[:, mode - 1]
            lam_s = pred.lam[mode - 1]
            if lam_s.real != shared:
                own = ShiftInvert.factor(a, b, lam_s.real)
                alone = self._scan(
                    space, j, pencil=(a, b), factor=own, want=min(first, n - 2)
                )
                p = _pick(mass, x, alone)
                if p is not None:
                    picks[k] = alone.pick(p)
                    scans.append(alone)
                    shifts.append(lam_s.real)
                    if k == kmin:
                        factor, shared = own, lam_s.real
                    continue
            last, picks[k], used, own = self._continued(j, lam_s, x, mass, first)
            scans.append(last)
            shifts.append(used)
            if k == kmin:
                factor, shared = own, used
        self.shift[j] = shared
        return self._assemble_candidates(j, picks, scans, shifts, factor)

    def _assemble_candidates(
        self,
        j: int,
        picks: list[_Pick | None],
        scans: list[_Scan],
        shifts: list[float],
        factor: ShiftInvert,
    ) -> _Candidates:
        """The picked bent modes at their mode slots, other candidates between."""
        entries: list[_Pick] = [p for p in picks if p is not None]
        slots: dict[int, _Pick] = {
            mode - 1: p
            for mode, p in zip(self.modes, picks, strict=True)
            if p is not None
        }
        fill: list[_Pick] = []
        for scan, shift in zip(scans, shifts, strict=True):
            for i in self._ranked(scan.lam, scan.ok, shift, j):
                lam = scan.lam[i]
                if any(
                    abs(lam - e.lam) <= _SAME_MODE * abs(e.lam) for e in entries + fill
                ):
                    continue
                fill.append(scan.pick(i))
        ordered: list[_Pick] = []
        spare = iter(fill)
        for position in range(max(max(slots) + 1, len(slots) + len(fill))):
            if position in slots:
                ordered.append(slots[position])
            elif (extra := next(spare, None)) is not None:
                ordered.append(extra)
        return _Candidates(
            lam=np.array([e.lam for e in ordered]),
            vecs=np.column_stack([e.vec for e in ordered]),
            residual=np.array([e.res for e in ordered]),
            share=np.array([e.share for e in ordered]),
            factor=factor if self.dispersive_modes else None,
        )

    def _continued(
        self,
        j: int,
        lam_s: complex,
        x_s: np.ndarray,
        mass: Any,
        first: int,
    ) -> tuple[_Scan, _Pick, float, ShiftInvert]:
        r"""A bent mode continued from the closed straight mode in :math:`u = 1/R`.

        :math:`K = 2, 4, \dots` equal steps from :math:`u = 0`; each step
        shifts at the linear extrapolation of the previous two eigenvalues
        and picks the candidate that overlaps the previous step's mode most,
        by the acceptance rule of `_pick`. The first :math:`K` whose steps all
        pass gives the mode; the last step is the section's own radius.

        Returns:
            The last step's scan, the picked mode, and the shift and factor of
            the scan.

        Raises:
            PortModeError: No step count up to `_BEND_STEPS` is accepted.
        """
        section = self.space.section
        assert isinstance(section, PortSection)  # noqa: S101 - bent ports only
        assert section.bend is not None  # noqa: S101
        bend = section.bend
        spaces: dict[Fraction, ModeSpace] = {Fraction(1): self.space}
        steps = 2
        while steps <= _BEND_STEPS:
            x_prev, lams, us = x_s, [lam_s], [0.0]
            last: _Pick | None = None
            for step in range(1, steps + 1):
                frac = Fraction(step, steps)
                if frac not in spaces:
                    stepped = dataclasses.replace(
                        section,
                        bend=dataclasses.replace(bend, radius=bend.radius / frac),
                    )
                    spaces[frac] = PortModeSpace(stepped, self.space.order)
                space = spaces[frac]
                guess = lams[-1]
                if len(lams) > 1:
                    guess = guess + (guess - lams[-2]) / (us[-1] - us[-2]) * (
                        float(frac) - us[-1]
                    )
                a, b = space.pencil.at(self.k0[j])
                k_max = _wavenumbers(space.section, self.f[j])[1]
                shift = complex(guess).real
                factor = ShiftInvert.factor(a, b, shift)
                scan = self._scan(
                    space,
                    j,
                    pencil=(a, b),
                    factor=factor,
                    want=min(first, a.shape[0] - 2),
                    k_max=k_max,
                )
                p = _pick(mass, x_prev, scan)
                if p is None:
                    break
                x_prev = scan.vecs[:, p]
                lams.append(scan.lam[p])
                us.append(float(frac))
                last = scan.pick(p)
            else:
                assert last is not None  # noqa: S101 - the loop ran
                return scan, last, shift, factor
            steps *= 2
        raise PortModeError(
            f"{self._who(0)}: the bent mode could not be followed from the "
            f"straight guide at {hertz(self.f[j])}: its overlap with the "
            f"previous step's mode stayed unclear down to steps of "
            f"1/(R {_BEND_STEPS}). Use a larger radius, or solve the guide "
            "straight."
        )

    def _warn_clearance(self, first: _Candidates) -> None:
        """Warn for each picked mode with a layer energy share above `_CLEARANCE`."""
        if first.share is None:
            return
        if getattr(self.space.section, "bend", None) is not None:
            self._warn_caustic(first)
            return
        for k, mode in enumerate(self.modes):
            share = float(first.share[mode - 1])
            if share <= _CLEARANCE:
                continue
            warnings.warn(
                f"{self._who(k)}: mode {mode} holds {share:.1%} of its electric "
                f"energy in the perfectly matched layer at {hertz(self.f[0])}, so "
                "the layer perturbs the mode itself (its propagation constant "
                "and power): the clearance between the guide and the layer is "
                "too small. Move the layer's inner face away from the guide; "
                "the perturbation falls about as exp(-2 gamma_t c) with the "
                "clearance c and the mode's transverse decay rate gamma_t.",
                LayerClearanceWarning,
                stacklevel=_mpi.outside_gfem(),
            )

    def _warn_caustic(self, first: _Candidates) -> None:
        r"""Warn for each picked mode whose outer layer starts before its caustic.

        The mode radiates beyond :math:`x_c = R\,(\mathrm{Re}\,n_\mathrm{eff}
        / n_\mathrm{cl} - 1)`, the distance from the reference arc where the
        local wavenumber :math:`k_0 n_\mathrm{cl}(R + x)/R` falls to
        :math:`\mathrm{Re}\,\beta`; a layer that starts before it sits on the
        guided field.
        """
        section = self.space.section
        assert isinstance(section, PortSection)  # noqa: S101 - checked by the caller
        bend = section.bend
        assert bend is not None  # noqa: S101 - checked by the caller
        sides = outer_sides(section)
        if not sides or section.layer is None:
            return
        n_cl = math.sqrt(float(section.eps_nn[section.layer].real.max()))
        outward = np.asarray(bend.outward) @ section.frame[:2]
        for k, mode in enumerate(self.modes):
            beta = np.sqrt(-first.lam[mode - 1] + 0j)
            x_c = bend.radius * (beta.real / self.k0[0] / n_cl - 1)
            for side in sides:
                c = outward[side.axis]
                if abs(abs(c) - 1) > 1e-6:
                    continue
                t = (side.face / section.scale - section.origin[side.axis]) / c
                face = (t - bend.offset) * section.scale
                if face >= x_c:
                    continue
                warnings.warn(
                    f"{self._who(k)}: the perfectly matched layer on the outer "
                    f"side starts {face:.4g} m from the reference arc, before "
                    f"the caustic x_c = {x_c:.4g} m of mode {mode} at "
                    f"{hertz(self.f[0])}: the layer sits on the guided field and "
                    "perturbs the mode itself. Move the layer's inner face "
                    "beyond x_c.",
                    LayerClearanceWarning,
                    stacklevel=_mpi.outside_gfem(),
                )

    def _keep(self, j: int, cand: _Candidates, picks: list[int]) -> None:
        """Make candidate ``picks[k]`` of the exact solve at ``j`` mode ``k`` there.

        The modes that carry dispersion get their mode derivative, on the
        solve's factor or, for a sample re-picked after its factor was
        dropped, on a new one; then the factor is dropped (one per sample
        would hold hundreds of MiB each on a large section).
        """
        new = j not in self.candidates
        if self.dispersive_modes:
            self._derive(j, cand, picks)
            cand = dataclasses.replace(cand, factor=None)
        self.candidates[j] = cand
        self.pick[j] = list(picks)
        self.solved[j] = True
        self.lam[:, j] = cand.lam[picks]
        self.residual[:, j] = cand.residual[picks]
        if new:
            self.reduced.add(cand.vecs[:, picks])

    def _derive(self, j: int, cand: _Candidates, picks: list[int]) -> None:
        """The mode derivatives at ``j`` of the modes that carry dispersion.

        A mode whose pick is unchanged keeps its derivative.
        """
        new = self.derivatives.get(j, [None] * len(self.modes))
        todo = [
            k
            for k in self.dispersive_modes
            if new[k] is None or self.pick[j][k] != picks[k]
        ]
        if not todo:
            return
        factor = self._factor_of(j, cand)
        new = list(new)
        for k in todo:
            new[k] = self._mode_derivative(j, cand, picks[k], factor)
        self.derivatives[j] = new

    def _factor_of(self, j: int, cand: _Candidates) -> ShiftInvert:
        """The exact solve at ``j``'s shift-invert factor, refactored if dropped."""
        if cand.factor is not None:
            return cand.factor
        a, b = self.space.pencil.at(self.k0[j])
        return ShiftInvert.factor(a, b, self.shift[j])

    def _mode_derivative(
        self, j: int, cand: _Candidates, p: int, factor: ShiftInvert
    ) -> ModeDerivative:
        """The mode derivative of candidate ``p`` of the exact solve at ``j``."""
        pencil = self.space.pencil
        rtol = max(_DERIVATIVE_RTOL, float(cand.residual[p]))
        return mode_derivative(
            pencil.a,
            pencil.b,
            pencil.taylor(self.k0[j]),
            complex(cand.lam[p]),
            cand.vecs[:, p],
            solve=gmres_on(factor, rtol),
            k0=float(self.k0[j]),
        )

    def dispersion_rows(self, k: int) -> np.ndarray:
        r"""Mode ``k``'s :math:`n_g`, :math:`\beta_2`, :math:`\beta_3` per row.

        ``(3, n)``, SI (s²/m, s³/m). Each exact solve's mode derivative gives
        its row; a reduced row is NaN until `dispersion_pass` fills it.
        """
        out = np.full(
            (len(_DISPERSION_FIELDS), len(self.f)), complex(math.nan, math.nan)
        )
        reduced = self._reduced_rows.get(k)
        if reduced is not None:
            out[:] = reduced
        scale = self.space.section.scale
        for (mode, j), derivative in self.fallbacks.items():
            if mode == k:
                out[:, j] = derivative.dispersion(scale)
        for j, derivatives in self.derivatives.items():
            derivative = derivatives[k]
            if derivative is not None:
                out[:, j] = derivative.dispersion(scale)
        if self._opted(k):  # no group index or dispersion below cutoff
            out[:, ~_propagates(np.sqrt(-self.lam[k] + 0j), self.k_max)] = complex(
                math.nan, math.nan
            )
        return out

    def dispersion_pass(self) -> None:
        r"""Give every reduced row its dispersion, after `run`.

        The derivative basis is the sweep's reduced basis, unchanged and
        first, then the real and imaginary parts of every sample's mode
        derivative. A reduced row's mode and its derivative are the vectors
        of that basis that minimize their residuals at the sweep's eigenvalue,
        the row keeping its mode across a fallback when that gives the smaller
        derivative residual (`_Reduced.derivative_row`), so adding vectors
        never raises them. A row passes when both are at
        most its floor (`_guard_floor`) and its mode still overlaps the
        sweep's own row; the worst failing row gets an exact solve for its
        mode derivative alone, whose mode and derivative join the basis (the
        row's :math:`\gamma`, field and ``solved`` stay the sweep's). Never
        touches the sweep's own basis, eigenvalues or fields.
        """
        rows = np.flatnonzero(~self.solved)
        self._reduced_rows = {
            k: np.full(
                (len(_DISPERSION_FIELDS), len(self.f)), complex(math.nan, math.nan)
            )
            for k in self.dispersive_modes
        }
        if not self.dispersive_modes or not len(rows):
            return
        floor = self._guard_floor()
        scale = self.space.section.scale
        taylor = {int(i): self.space.pencil.taylor(self.k0[i]) for i in rows}
        extra: list[np.ndarray] = []
        modes: dict[tuple[int, int], np.ndarray] = {}
        while True:
            basis = self._derivative_basis(extra)
            ratio: dict[tuple[int, int], float] = {}
            for k in self.dispersive_modes:
                for i in map(int, rows):
                    if (k, i) in self.fallbacks:
                        continue
                    if self._opted(k) and not _propagates(
                        np.sqrt(-self.lam[k, i] + 0j), self.k_max[i]
                    ):
                        continue  # no dispersion below cutoff: NaN
                    ref = self.coords_at(i, k)
                    row = basis.derivative_row(
                        taylor[i],
                        self.lam[k, i],
                        self.k0[i],
                        ref,
                        prior=modes.get((k, i)),
                    )
                    modes[k, i] = row.y
                    self._reduced_rows[k][:, i] = row.derivative.dispersion(scale)
                    ratio[k, i] = _excess(row, basis.overlap(ref, row.y), floor[k, i])
            if not ratio:
                return
            (k, i), worst = max(ratio.items(), key=lambda item: item[1])
            if worst <= 1:
                return
            derivative = self._fallback(i, k)
            self.fallbacks[k, i] = derivative
            extra += [derivative.x, derivative.x1]

    def _guard_floor(self) -> np.ndarray:
        """The residual each mode's reduced derivative at each row must reach.

        ``(n_modes, n)``: the row's own floor from the sweep.
        """
        return self.floor

    def _derivative_basis(self, extra: Sequence[np.ndarray]) -> _Reduced:
        """The sweep's basis then the samples' mode derivatives and ``extra``."""
        basis = _Reduced.on(self.space, self.reduced.v.copy())
        vectors = [
            d.x1
            for j in sorted(self.derivatives)
            for d in self.derivatives[j]
            if d is not None
        ]
        if vectors or extra:
            basis.add(np.column_stack([*vectors, *extra]))
        return basis

    def _fallback(self, i: int, k: int) -> ModeDerivative:
        """Mode ``k``'s mode derivative at row ``i`` from an exact solve.

        The solve is not a sweep sample: it is counted apart from the exact
        solves, and its eigenvalue and field are discarded.
        """
        self.fallback_solves += 1
        if self.bent:  # shift at the mode's estimate at the row
            tracked = np.full(len(self.modes), np.nan + 0j)
            tracked[k] = self.lam[k, i]
            cand = self._solve(i, tracked)
        else:
            cand = self._solve(i)
        x_ref = self.reduced.lift(self.coords_at(i, k))
        p = int(np.argmax(_overlaps(self.space.mass, x_ref, cand.vecs)))
        return self._mode_derivative(i, cand, p, self._factor_of(i, cand))

    def _exact(self, j: int, ref: int, *, certain: bool) -> None:
        """Exact solve at ``j``, keeping the candidates closest to the modes at ``ref``.

        Args:
            j: Grid index.
            ref: Grid index of the reference modes.
            certain: The reference is accurate modes at the grid neighbour,
                so an ambiguous choice is an error.
        """
        self.exact_solves += 1
        if self.bent:  # shift at the modes' eigenvalues at ``ref``, scaled as k0^2
            tracked = self.lam[:, ref] * (self.k0[j] / self.k0[ref]) ** 2
            cand = self._solve(j, tracked)
        else:
            cand = self._solve(j)
        if len(cand.lam) < len(self.modes):
            # too few candidates for distinct modes
            raise self._evanescent_error(j, self._lost(cand, ref, certain=certain))
        self._keep(j, cand, self._closest(j, cand, ref, certain=certain))

    def _lost(self, cand: _Candidates, ref: int, *, certain: bool) -> list[int]:
        """The modes left without a candidate when ``cand`` has too few.

        Each candidate goes to the mode at ``ref`` it overlaps most, the
        largest overlaps first, as the tracker matches them. Against an
        uncertain reference with every mode still at its rank, the
        highest-ranked modes are the ones lost.
        """
        n = len(self.modes)
        ranks = [mode - 1 for mode in self.modes]
        if not certain and all(pick == ranks for pick in self.pick.values()):
            by_rank = np.argsort(self.modes, kind="stable")
            return sorted(int(k) for k in by_rank[len(cand.lam) :])
        overlap = np.array(
            [
                _overlaps(
                    self.space.mass,
                    self.reduced.lift(self.coords_at(ref, k)),
                    cand.vecs,
                )
                for k in range(n)
            ]
        ).reshape(n, len(cand.lam))
        lost = set(range(n))
        for _ in range(len(cand.lam)):
            k, c = np.unravel_index(int(np.argmax(overlap)), overlap.shape)
            lost.discard(int(k))
            overlap[k, :] = -np.inf
            overlap[:, c] = -np.inf
        return sorted(lost)

    def _closest(
        self, j: int, cand: _Candidates, ref: int, *, certain: bool
    ) -> list[int]:
        """Each mode's candidate at ``j``: the closest to that mode at ``ref``.

        With an uncertain reference two modes may provisionally share a
        candidate; the certain re-check in `_tabulate` raises if they still do.
        """
        picks = []
        for k, mode in enumerate(self.modes):
            x_ref = self.reduced.lift(self.coords_at(ref, k))
            overlap = _overlaps(self.space.mass, x_ref, cand.vecs)
            best = np.argsort(-overlap, kind="stable")
            if (
                certain
                and len(best) > 1
                and overlap[best[1]] > overlap[best[0]] - _AMBIGUOUS
            ):
                raise PortModeError(
                    f"{self._who(k)}: two modes overlap the tracked mode "
                    f"{mode} almost equally at {hertz(self.f[j])} (overlaps "
                    f"{overlap[best[0]]:.3f} and {overlap[best[1]]:.3f}): a degenerate "
                    "pair or an unresolved crossing. Break the cross-section's "
                    "symmetry, or make the unwanted mode non-propagating (for "
                    "example with WavePort(pec_edges=...))."
                )
            picks.append(int(best[0]))
        if certain:
            for k, other in itertools.combinations(range(len(picks)), 2):
                if picks[k] == picks[other]:
                    raise PortModeError(
                        f"WavePorts {self.ports[k].name!r} and "
                        f"{self.ports[other].name!r} on surface "
                        f"{self.ports[k].surface!r} follow the same mode at "
                        f"{hertz(self.f[j])} (an unresolved crossing between "
                        "two requested modes). Narrow the band or break the "
                        "cross-section's symmetry."
                    )
        return picks

    def _rebuild(self) -> None:
        """A fresh basis from the exact modes kept now."""
        self.reduced = _Reduced(self.space)
        for j in sorted(self.candidates):
            self.reduced.add(self.candidates[j].vecs[:, self.pick[j]])

    def _warn_crossing(self, k: int) -> None:
        if self._opted(k):  # a kept mode changes rank where it crosses cutoff
            return
        mode = self.modes[k]
        for j in sorted(self.pick):
            if self.pick[j][k] != mode - 1:
                warnings.warn(
                    f"WavePort {self.ports[k].name!r}: the tracked mode {mode} is "
                    f"mode {self.pick[j][k] + 1} in rank order at "
                    f"{hertz(self.f[j])} (a mode crossing inside the band). "
                    "gfem keeps following the physical mode; a solver that ranks "
                    "modes at each frequency would switch modes there, so its "
                    "S-parameters differ beyond the crossing.",
                    PortModeTrackingWarning,
                    stacklevel=_mpi.outside_gfem(),
                )
                return

    def coords_at(self, j: int, k: int) -> np.ndarray:
        """Reduced coordinates of mode ``k`` tabulated at ``j`` (current basis)."""
        if self.solved[j]:
            return self.reduced.project(self.candidates[j].vecs[:, self.pick[j][k]])
        y = self.coords[k][j]
        return np.concatenate([y, np.zeros(self.reduced.size - len(y))])

    # --- reduced solves -----------------------------------------------------------

    def _tabulate(self) -> np.ndarray | None:
        """Walk up the grid, filling unsolved frequencies from the reduced model.

        Each frequency tracks the modes at the previous one; it passes when
        every mode passes.

        Returns:
            Which frequencies pass the guard, or ``None`` when an exact sample
            was re-selected (the basis must be rebuilt).
        """
        self.rounds += 1
        n, n_modes = len(self.f), len(self.modes)
        solved = np.flatnonzero(self.solved)
        passed = np.zeros(n, bool)
        for i in range(n):
            nearest = solved[np.argmin(np.abs(self.f[solved] - self.f[i]))]
            self.floor[:, i] = np.maximum(
                self.tolerance, _FLOOR * self.residual[:, nearest]
            )
            if self.solved[i]:
                if i > 0 and passed[i - 1]:
                    cand = self.candidates[i]
                    picks = self._closest(i, cand, i - 1, certain=True)
                    if picks != self.pick[i]:
                        self._keep(i, cand, picks)
                        return None
                for k in range(n_modes):
                    self.coords[k][i] = self.coords_at(i, k)
            else:
                refs = [self.coords_at(i - 1, k) for k in range(n_modes)]
                pairs = self.reduced.solve(self.k0[i], self.k_max[i], refs)
                for k, (lam, y, eta) in enumerate(pairs):
                    self.lam[k, i], self.coords[k][i], self.residual[k, i] = lam, y, eta
            passed[i] = (self.residual[:, i] <= self.floor[:, i]).all()
        beta = np.sqrt(-self.lam + 0j)
        gone = passed & ~self.solved & ~_propagates(beta, self.k_max)
        for k in range(n_modes):
            if self._opted(k):
                gone[k] = False
        if gone.any():
            i = int(np.flatnonzero(gone.any(axis=0))[0])
            raise self._evanescent_error(i, [int(np.flatnonzero(gone[:, i])[0])])
        return passed

    # --- normalization ----------------------------------------------------------

    def normalized(
        self,
        rules: list[tuple[np.ndarray, np.ndarray] | None],
        arrows: list[tuple[np.ndarray, np.ndarray] | None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The fixed basis, then per mode 1 W coordinates, gamma, voltages, powers.

        The space gives the field basis on the reduced one, the power and the
        phase gauge (`gfem._engine.port_mode.PortModeSpace.reduce`); a line
        port's mode carries 1 W/m.

        Args:
            rules: Each mode's voltage path rule, or ``None`` without a path.
            arrows: Each mode's polarity rule (`_polarity_rule`), or ``None``
                without a polarity.

        Raises:
            PortModeError: A mode's field at its polarity point is too small
                at some frequency to fix its sign.

        Returns:
            ``basis (ndof, M)``, ``coords (n_modes, n_f, M)``, ``gamma
            (n_modes, n_f)`` (1/m), ``voltage (n_modes, n_f)`` (V) and the
            complex ``power (n_modes, n_f)`` (W, of modulus 1).
        """
        space = self.space
        fields = space.reduce(self.reduced.v)
        w = fields.w
        mass = w.T @ (space.mass @ w)
        basis = np.zeros((space.space.ndof, w.shape[1]))
        basis[space.free] = w

        n, n_modes = len(self.f), len(self.modes)
        coords = np.zeros((n_modes, n, w.shape[1]), complex)
        voltage = np.full((n_modes, n), complex(math.nan, math.nan))
        phase = np.ones((n_modes, n), complex)
        gamma = _branch_gamma(self.lam)
        omega = 2 * math.pi * self.f
        for k, (rule, arrow) in enumerate(zip(rules, arrows, strict=True)):
            volts = (
                None
                if rule is None
                else _voltage_functional(planar(space), basis, rule)
            )
            y = np.array([self.coords_at(i, k) for i in range(n)])
            self.rows[k] = y
            polar = None
            if arrow is not None:
                polar, peak = _polarity_functionals(space, basis, arrow)
                for i in range(n):
                    self._check_polarity(
                        k, i, fields.coords(y[i], gamma[k, i]), polar, peak
                    )
            rows = normalize_rows(
                fields, mass, y, gamma[k], omega, volts=volts, polarity=polar
            )
            coords[k], voltage[k], phase[k] = rows.z, rows.voltage, rows.phase
        return basis, coords, gamma / space.section.scale, voltage, phase

    def _check_polarity(
        self, k: int, i: int, z: np.ndarray, at: np.ndarray, peak: np.ndarray
    ) -> None:
        """Mode ``k``'s field at its polarity point along the arrow fixes a sign.

        The reading's ratio to the port's peak does not depend on the scale or
        the phase of ``z`` (the coordinates at ``f[i]``).

        Raises:
            PortModeError: It is below `_POLARITY_GUARD` of the port's peak.
        """
        a = complex(at @ z)
        top = float(np.linalg.norm(np.einsum("k,kpc->pc", z, peak), axis=1).max())
        ratio = abs(a) / top if top > 0 else 0.0
        if ratio < _POLARITY_GUARD:
            # functionals are made from the port's polarity only
            point, direction = cast("Polarity", self.ports[k].polarity)
            raise PortModeError(
                f"{self._who(k)}: its polarity reads a field of {ratio:.1e} of the "
                f"port's peak at the point {_numbers(point)} m along "
                f"{_numbers(direction)} at {hertz(self.f[i])}, too small to fix "
                "the mode's sign (the point sits near a node of the field, or the "
                "direction is across the field). Move the point towards the "
                "field's maximum, or turn the direction along the field."
            )

    # --- errors -------------------------------------------------------------------

    def _band(self) -> str:
        return hertz_band(self.f[0], self.f[-1])

    def _who(self, k: int) -> str:
        """Mode ``k``'s wave port, with its surface when that is shared."""
        port = self.ports[k]
        others = [p.name for p in self.ports if p is not port]
        if not others:
            return f"WavePort {port.name!r}"
        shared = ", ".join(repr(name) for name in others)
        return (
            f"WavePort {port.name!r} on surface {port.surface!r} (shared with {shared})"
        )

    def _opted(self, k: int) -> bool:
        """Whether mode ``k``'s port keeps its mode below cutoff."""
        return k < len(self.ports) and self.ports[k].allow_evanescent

    def _remedy(self, first: str) -> str:
        """``first``, then the evanescent opt-in and LumpedPort remedies.

        A shared port may also be dropped.
        """
        keep = (
            "keep the mode below cutoff with `allow_evanescent=True`, e.g. the "
            "kept modes of a mode cascade (`WavePort.kept_modes`)"
        )
        if len(self.ports) == 1:
            return f"{first}, {keep}, or use a LumpedPort."
        return f"{first}, {keep}, use a LumpedPort, or drop that wave port."

    def _unsettled_error(self) -> PortModeError:  # pragma: no cover - a safeguard
        if len(self.modes) == 1:
            which = f"mode continues mode {self.modes[0]}"
        else:
            *rest, last = (str(mode) for mode in self.modes)
            which = f"modes continue modes {', '.join(rest)} and {last}"
        return PortModeError(
            f"{wave_port_label(self.ports)}: the port mode sweep could not settle "
            f"which {which} over {self._band()}. Narrow the band or use a "
            "LumpedPort."
        )

    def _cutoff_error(self, first: _Candidates) -> PortModeError:
        """A mode is evanescent at the lowest frequency: locate its cutoff.

        The lowest-ranked failing mode is the one reported.
        """
        n_first = self._n_propagating(first, 0)
        k = min(
            (
                k
                for k, mode in enumerate(self.modes)
                if mode > (n_first if not self._opted(k) else len(first.lam))
            ),
            key=lambda k: self.modes[k],
        )
        n, mode, who = len(self.f), self.modes[k], self._who(k)
        if self._opted(k):
            return PortModeError(
                f"{who}: the cross-section yields only {len(first.lam)} mode(s) "
                f"(propagating and evanescent) at {hertz(self.f[0])}, fewer "
                f"than the {mode} kept: refine the port line or keep fewer "
                "modes."
            )
        remedy = self._remedy("Start the band above the cutoff, pick a lower mode")
        top = self._solve(n - 1) if n > 1 else first
        n_top = self._n_propagating(top, n - 1)
        if n == 1 or n_top < mode:
            return PortModeError(
                f"{who}: mode {mode} does not propagate in the band "
                f"{self._band()} (the cross-section carries {n_first} "
                f"propagating mode(s) at {hertz(self.f[0])} and "
                f"{n_top} at {hertz(self.f[-1])}): the band is below "
                f"the mode's cutoff. {remedy}"
            )
        lo, hi, cand = 0, n - 1, top
        while hi - lo > 1:
            mid = (lo + hi) // 2
            c = self._solve(mid)
            if self._n_propagating(c, mid) >= mode:
                hi, cand = mid, c
            else:
                lo = mid
        cutoff = self._cutoff_estimate(hi, cand, top, mode)
        return PortModeError(
            f"{who}: mode {mode} has its cutoff inside the band "
            f"{self._band()}, at about {hertz(cutoff, 4)} (it is evanescent at "
            f"{hertz(self.f[lo])} and propagates at {hertz(self.f[hi])}). "
            f"{remedy}"
        )

    def _cutoff_estimate(
        self, hi: int, cand: _Candidates, top: _Candidates, mode: int
    ) -> float:
        """Cutoff frequency from ``beta^2`` linear in ``k0^2`` (bracketed)."""
        f_lo, f_hi = self.f[hi - 1], self.f[hi]
        n = len(self.f)
        if hi == n - 1:
            return 0.5 * (f_lo + f_hi)
        b1 = -cand.lam[mode - 1].real
        b2 = -top.lam[mode - 1].real
        k1, k2 = self.k0[hi] ** 2, self.k0[n - 1] ** 2
        slope = (b2 - b1) / (k2 - k1)
        if slope <= 0:
            return 0.5 * (f_lo + f_hi)
        kc2 = max(k1 - b1 / slope, 0.0)
        fc = math.sqrt(kc2) / self.k0[hi] * self.f[hi]
        return min(max(fc, f_lo), f_hi)

    def _evanescent_error(self, j: int, lost: list[int]) -> PortModeError:
        """Modes ``lost`` (indices into ``modes``) stop propagating at ``j``."""
        if len(lost) == 1:
            (k,) = lost
            remedy = self._remedy("Restrict the band to where the mode propagates")
            return PortModeError(
                f"{self._who(k)}: the tracked mode {self.modes[k]} stops "
                f"propagating at {hertz(self.f[j])} inside the band "
                f"{self._band()} (a cutoff in the band). {remedy}"
            )
        *rest, last = (str(self.modes[k]) for k in lost)
        label = wave_port_label([self.ports[k] for k in lost], lone_surface=True)
        remedy = self._remedy("Restrict the band to where the modes propagate")
        return PortModeError(
            f"{label}: the tracked modes {', '.join(rest)} and {last} stop "
            f"propagating at {hertz(self.f[j])} inside the band "
            f"{self._band()} (a cutoff in the band). {remedy}"
        )

    def _check_degenerate(self, first: _Candidates, k: int) -> None:
        """Mode ``k`` against every candidate at the lowest frequency."""
        beta = np.sqrt(-first.lam + 0j)
        pos = self.modes[k] - 1
        close = np.abs(beta - beta[pos]) <= _DEGENERATE * abs(beta[pos])
        close[pos] = False
        if not close.any():
            return
        partners = [i for i, mode in enumerate(self.modes) if close[mode - 1]]
        if partners:
            port, other = self.ports[k], self.ports[partners[0]]
            raise PortModeError(
                f"WavePorts {port.name!r} and {other.name!r} on surface "
                f"{port.surface!r} request modes {self.modes[k]} and "
                f"{self.modes[partners[0]]}, which are degenerate at "
                f"{hertz(self.f[0])} (the same propagation constant), so the "
                "ports' fields are not defined. " + _DEGENERATE_REMEDY
            )
        raise PortModeError(
            f"{self._who(k)}: mode {self.modes[k]} is degenerate at "
            f"{hertz(self.f[0])} (another mode has the same propagation "
            "constant), so the port's field is not defined. " + _DEGENERATE_REMEDY
        )


class _Reduced:
    """Rayleigh-Ritz model of the affine pencil on a growing real basis."""

    def __init__(self, space: ModeSpace) -> None:
        self.space = space
        self.v = np.zeros((len(space.free), 0))
        self._pieces = [
            (is_b, k, m)
            for is_b, mats in ((False, space.pencil.a), (True, space.pencil.b))
            for k, m in enumerate(mats)
            if m.count_nonzero()
        ]

    @property
    def size(self) -> int:
        return self.v.shape[1]

    def add(self, xs: np.ndarray) -> None:
        """Add the real and imaginary parts of each column of ``xs``, in order.

        Gram-Schmidt twice per part.
        """
        for x in xs.T:
            scale = np.linalg.norm(x)
            for part in (x.real, x.imag):
                w = part.copy()
                for _ in range(2):
                    w -= self.v @ (self.v.T @ w)
                norm = np.linalg.norm(w)
                if norm > _NEW_DIRECTION * scale:
                    self.v = np.column_stack([self.v, w / norm])
        self._project()

    @classmethod
    def on(cls, space: ModeSpace, v: np.ndarray) -> _Reduced:
        """The model on a sweep's fixed basis ``v`` (real, orthonormal columns)."""
        out = cls(space)
        out.v = np.asarray(v, float)
        out._project()
        return out

    def _project(self) -> None:
        products = [m @ self.v for _, _, m in self._pieces]
        self._r = np.linalg.qr(np.concatenate(products, axis=1), mode="r")
        self._red = [self.v.T @ p for p in products]
        self._mass = self.v.T @ (self.space.mass @ self.v)
        self.margins: list[tuple[float, float]] = []

    @property
    def mass(self) -> np.ndarray:
        """The reduced L2 mass, the default inner product of `solve`."""
        return self._mass

    def project(self, x: np.ndarray) -> np.ndarray:
        return self.v.T @ x

    def overlap(self, ref: np.ndarray, y: np.ndarray) -> float:
        """``|<ref, y>| / (|ref| |y|)`` in the mass inner product.

        ``ref`` holds the coordinates on a prefix of this basis.
        """
        ref = np.concatenate([ref, np.zeros(self.size - len(ref))])
        num = abs(ref.conj() @ self._mass @ y)
        den = np.sqrt(
            (ref.conj() @ self._mass @ ref).real * (y.conj() @ self._mass @ y).real
        )
        return float(num / den)

    def derivative_row(
        self,
        coefficients: np.ndarray,
        lam: complex,
        k0: float,
        ref: np.ndarray,
        *,
        prior: np.ndarray | None = None,
    ) -> _DerivativeRow:
        r"""The mode and mode derivative of the pencil at the pieces' ``coefficients``.

        Both are taken at the sweep's ``lam``, written :math:`\lambda_s`, and
        :math:`G = A_0 - \lambda_sB_0`. The mode :math:`y` is the right
        singular vector of the smallest singular value of :math:`G` on the
        basis (a refined Ritz vector, through the QR factor of the pieces'
        products). Its eigenvalue :math:`\lambda` is its Rayleigh quotient and
        :math:`\lambda'` follows from Hellmann-Feynman (`slope`). Then
        :math:`y'`, with :math:`y^TB_0y' = 0`, minimizes :math:`|Gy' - r|`,
        where :math:`r = -(A_1 - \lambda'B_0 - \lambda_sB_1)y`
        (`right_hand_side`).

        The two residuals the guard reads are those minima over fixed scales,
        both from the sweep's own row :math:`x_s` (``ref``):
        :math:`\eta = |Gy| / s` with :math:`|y| = 1` and
        :math:`s = (|A_0x_s| + |\lambda_s||B_0x_s|)/|x_s|`, and
        :math:`\eta' = |Gy' - r| / s'` with :math:`s'` the uncancelled size of
        :math:`x_s`'s own right-hand side per unit :math:`|x_s|`
        (`gfem._engine.mode_derivative.residual`). Not :math:`|r|`: where the
        mode does not change with :math:`k_0` (a homogeneous fill,
        :math:`y' = 0`), :math:`r` is rounding of its terms, and a residual
        relative to it would fail every row. Over the sweep's basis alone
        :math:`\eta` is at most the sweep's own backward error of the row,
        and on a larger basis it can only fall. So can :math:`\eta'` for the
        same :math:`y`, but a sharper :math:`y` moves :math:`r`: given the
        mode ``prior`` of a basis this one extends, the row keeps that mode,
        with :math:`y'` minimized anew, whenever the refined vector's
        :math:`\eta'` would be the larger. Neither residual then rises as the
        basis grows.

        Args:
            coefficients: :math:`c_{jn}` of the pieces, shape (J, 4)
                (`gfem._engine.port_mode.Pencil.taylor`).
            lam: The sweep's eigenvalue of the row.
            k0: The wavenumber of the row, per local unit.
            ref: The sweep's coordinates of the row, on a prefix of this
                basis.
            prior: The row's mode on a prefix of this basis, if any.
        """
        m = self.size
        pieces = self._pieces

        def orders(blocks: Sequence[np.ndarray], is_b: bool) -> list[np.ndarray]:
            chosen = [p for p, (b, _, _) in enumerate(pieces) if b == is_b]
            rows = coefficients[[pieces[p][1] for p in chosen]]
            return piece_series([blocks[p] for p in chosen], rows)

        # on the QR factor for norms of full-space products, projected for
        # bilinear forms
        rs = [self._r[:, p * m : (p + 1) * m] for p in range(len(pieces))]
        ra, rb = orders(rs, False), orders(rs, True)
        ar, br = orders(self._red, False), orders(self._red, True)

        norm = np.linalg.norm
        g = ra[0] - lam * rb[0]
        ref = np.concatenate([ref, np.zeros(m - len(ref))])
        scale = (norm(ra[0] @ ref) + abs(lam) * norm(rb[0] @ ref)) / norm(ref)
        ref1 = slope(ref, ar[1] @ ref, br[0] @ ref, br[1] @ ref, lam)
        _, size = right_hand_side(ra[1] @ ref, rb[0] @ ref, rb[1] @ ref, lam, ref1)
        size /= norm(ref)

        def row(y: np.ndarray) -> _DerivativeRow:
            lam_y = complex((y @ ar[0] @ y) / (y @ br[0] @ y))
            lam1 = slope(y, ar[1] @ y, br[0] @ y, br[1] @ y, lam_y)
            rhs, _ = right_hand_side(ra[1] @ y, rb[0] @ y, rb[1] @ y, lam, lam1)
            # y' orthogonal to B_0 y in the bilinear form: a basis of its
            # complement
            q, _ = np.linalg.qr(
                np.column_stack([(br[0] @ y).conj(), np.eye(m)[:, : m - 1]])
            )
            free = q[:, 1:]
            z, *_ = np.linalg.lstsq(g @ free, rhs, rcond=None)
            y1 = free @ z
            eta1 = float(norm(g @ y1 - rhs) / size)
            series = rayleigh_series(ar, br, y, y1)
            derivative = ModeDerivative(
                k0=k0,
                lam=lam_y,
                x=y,
                x1=y1,
                coefficients=coefficients,
                lam_taylor=np.array([lam_y, lam1, series[2], series[3]], complex),
                residual=eta1,
            )
            return _DerivativeRow(derivative, norm(g @ y) / scale, eta1, y)

        _, _, vh = np.linalg.svd(g, full_matrices=False)
        out = row(vh[-1].conj())
        if prior is not None:
            kept = row(np.concatenate([prior, np.zeros(m - len(prior))]))
            if kept.eta1 < out.eta1:
                return kept
        return out

    def lift(self, y: np.ndarray) -> np.ndarray:
        return self.v @ y

    def matrices(self, theta: Sequence[complex]) -> tuple[np.ndarray, np.ndarray]:
        """The reduced ``A`` and ``B`` with the scalars ``theta`` of the pencil's slots.

        With the slots' k0-derivatives, ``dA/dk0`` and ``dB/dk0``.
        """
        n = self.size
        ar = np.zeros((n, n), complex)
        br = np.zeros((n, n), complex)
        for (is_b, k, _), red in zip(self._pieces, self._red, strict=True):
            if is_b:
                br += theta[k] * red
            else:
                ar += theta[k] * red
        return ar, br

    def solve(
        self,
        k0: float,
        k_max: float,
        refs: list[np.ndarray],
        *,
        mass: np.ndarray | None = None,
    ) -> list[tuple[complex, np.ndarray, float]]:
        """The reduced pair that tracks each of ``refs``, and its backward error.

        A backward error is infinite when no pair qualifies, two overlap its
        reference almost equally, or two references pick the same pair.
        Pairs are told apart by their overlap in the reduced matrix ``mass``
        (the reduced L2 mass by default); afterwards ``margins`` holds, per
        reference, its two largest overlaps (the second is 0 with one pair).
        """
        mass = self._mass if mass is None else mass
        theta = self.space.pencil.theta(k0)
        n = self.size
        ar, br = self.matrices(theta)
        lam, vecs = scipy.linalg.eig(ar, br)
        beta = np.sqrt(-lam + 0j)
        ok = np.isfinite(lam) & (beta.real <= k_max * (1 + 1e-6))
        # a pair at gamma ~ 0 is the formulation's null space, not a mode; on
        # the basis of a mode whose shape does not change with frequency (a
        # homogeneous cross-section) it overlaps the mode within `_AMBIGUOUS`
        ok &= np.abs(beta) > _EVANESCENT * k_max
        self.margins = [(0.0, 0.0) for _ in refs]
        if not ok.any():
            return [(complex(math.nan), np.zeros(n), math.inf) for _ in refs]
        idx = np.flatnonzero(ok)
        y = vecs[:, idx]
        norms = np.einsum("ik,ij,jk->k", y.conj(), mass, y).real
        picks, clear = [], []
        for r, ref in enumerate(refs):
            num = np.abs(ref.conj() @ mass @ y)
            overlap = num / np.sqrt((ref.conj() @ mass @ ref).real * norms)
            best = np.argsort(-overlap, kind="stable")
            i = int(best[0])
            second = float(overlap[best[1]]) if len(best) > 1 else 0.0
            self.margins[r] = (float(overlap[i]), second)
            picks.append(i)
            clear.append(len(best) == 1 or overlap[best[1]] <= overlap[i] - _AMBIGUOUS)
        out = []
        for i, ok_i in zip(picks, clear, strict=True):
            lam_i, y_i = complex(lam[idx[i]]), y[:, i]
            if ok_i and picks.count(i) == 1:
                out.append((lam_i, y_i, self.backward_error(theta, lam_i, y_i)))
            else:  # ambiguous, or two modes on one pair
                out.append((lam_i, y_i, math.inf))
        return out

    def backward_error(
        self, theta: tuple[complex, ...], lam: complex, y: np.ndarray
    ) -> float:
        """``|A x - lam B x| / (|A x| + |lam| |B x|)`` of ``x = V y``, exactly."""
        n = self.size
        ax = np.zeros(self._r.shape[0], complex)
        bx = np.zeros(self._r.shape[0], complex)
        for p, (is_b, k, _) in enumerate(self._pieces):
            block = self._r[:, p * n : (p + 1) * n] @ (theta[k] * y)
            if is_b:
                bx += block
            else:
                ax += block
        den = np.linalg.norm(ax) + abs(lam) * np.linalg.norm(bx)
        return float(np.linalg.norm(ax - lam * bx) / den) if den > 0 else math.inf


def _excess(row: _DerivativeRow, overlap: float, floor: float) -> float:
    """How far a reduced derivative row is past its guard (above 1 fails).

    The larger residual over ``floor``, infinite when the row's mode no longer
    overlaps the sweep's.
    """
    if overlap < 1 - _AMBIGUOUS:
        return math.inf
    error = max(row.eta, row.eta1)
    return error / floor if floor > 0 else math.inf


def propagating_modes(space: ModeSpace, frequency: float, mode: int) -> int:
    """How many of the ``mode`` lowest-ranked modes propagate at ``frequency``.

    One exact solve, with the sweep's candidate rules: at least ``mode`` means
    mode ``mode`` in rank order is above its cutoff.
    """
    sweep = _Greedy(space, np.array([float(frequency)]), [mode], 1.0, [])
    return min(len(sweep.candidates_at(0).lam), mode)


def _branch_gamma(lam: Any) -> Any:
    r"""The propagation constant :math:`\gamma` with :math:`\gamma^2 = \lambda`.

    The root :math:`i\sqrt{-\lambda}` is kept where
    :math:`\mathrm{Re}(\gamma e^{-i\pi/4}) > 0` and negated elsewhere: the
    half-plane of :math:`e^{i\pi/4}\sqrt{-i\gamma^2}`. It gives
    :math:`\mathrm{Im}\,\gamma > 0` above cutoff, :math:`\mathrm{Re}\,\gamma > 0`
    below it (a decaying mode), and the passive root of a lossy mode. Every
    propagating mode, lossless or lossy, keeps the root :math:`i\sqrt{-\lambda}`
    bit for bit.
    """
    gamma = 1j * np.sqrt(-np.asarray(lam) + 0j)
    # [()]: a scalar stays a scalar
    return np.where((gamma * np.exp(-0.25j * np.pi)).real <= 0, -gamma, gamma)[()]


def _propagates(beta: np.ndarray, k_max: float | np.ndarray) -> np.ndarray:
    """Which propagation constants are of propagating modes (see `_EVANESCENT`)."""
    return (beta.real > _EVANESCENT * k_max) & (beta.real > np.abs(beta.imag))


def _overlaps(mass, x_ref: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """``|<x_ref, v>| / (|x_ref| |v|)`` in the mass inner product, per column.

    A column of negative norm in an indefinite ``mass`` (a pencil's ``B``) has
    no overlap: 0.
    """
    mv = mass @ vecs
    num = np.abs(x_ref.conj() @ mv)
    with np.errstate(invalid="ignore"):
        den = np.sqrt(
            (x_ref.conj() @ (mass @ x_ref)).real * np.sum(vecs.conj() * mv, 0).real
        )
    return np.nan_to_num(num / den, nan=0.0, posinf=0.0)


def _pick(mass, x_ref: np.ndarray, scan: _Scan) -> int | None:
    """The candidate of ``scan`` that continues ``x_ref``, if the choice is clear.

    The largest overlap in ``mass``, when at least `_BEND_OVERLAP` and ahead of
    the next candidate's by `_BEND_MARGIN`.
    """
    candidates = np.flatnonzero(scan.ok)
    if not len(candidates):
        return None
    overlap = _overlaps(mass, x_ref, scan.vecs[:, candidates])
    order = np.argsort(-overlap, kind="stable")
    best = overlap[order[0]]
    runner = overlap[order[1]] if len(order) > 1 else 0.0
    if best < _BEND_OVERLAP or best - runner < _BEND_MARGIN:
        return None
    return int(candidates[order[0]])


def _covers(lam: np.ndarray, estimates: np.ndarray | None) -> bool:
    """Whether each tracked estimate has a candidate within half its gap.

    The gap is the distance to the nearest other estimate; one estimate, or
    none, is always covered.
    """
    if estimates is None or len(estimates) < 2:
        return True
    for i, e in enumerate(estimates):
        gap = np.min(np.abs(np.delete(estimates, i) - e))
        if not len(lam) or np.min(np.abs(lam - e)) >= gap / 2:
            return False
    return True


def _wavenumbers(
    section: PortSection | LineSection, frequency: float
) -> tuple[float, float, float]:
    """``k0``, the largest propagating wavenumber and the shift, per local unit.

    The section's laws are read at ``frequency``.

    The largest wavenumber is the largest bulk one, times `_SLOW_WAVE` when
    `Conductor` or `Impedance` edges can slow the mode down.
    """
    k0_si = 2 * math.pi * frequency / scipy.constants.c
    k0 = k0_si * section.scale
    n2_shift, n2_max = section.index_squared(
        section.max_conductivity * _Z0 / k0_si, frequency
    )
    k_max = math.sqrt(n2_max) * k0
    if len(section.impedance_terms):
        k_max *= _SLOW_WAVE
    if isinstance(section, PortSection) and section.bend is not None:
        # a bent mode's wavenumber per unit length of the reference arc is at
        # most the bulk one times the largest h of the physical section
        k_max *= section.bend_h_range(physical=True)[1]
    return k0, k_max, -_TARGET * n2_shift * k0**2


def planar(space: ModeSpace) -> PortModeSpace:
    """``space`` as a planar port's: a voltage or current path needs one.

    Raises:
        TypeError: ``space`` is a line port's (a 2D model refuses its paths).
    """
    if not isinstance(space, PortModeSpace):
        raise TypeError(
            f"WavePort {space.section.port.name!r} is a line port: its mode has no "
            "voltage or current path."
        )
    return space


def _voltage_functional(
    space: PortModeSpace, basis: np.ndarray, rule: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    r""":math:`V = \int E \cdot dl` along the voltage path for each basis column."""
    pts, wts = rule
    e = space.evaluate(basis.T, pts)
    return np.einsum("kpc,pc->k", e, wts)


def _numbers(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:.6g}" for v in values) + ")"


def _polarity_rule(
    space: ModeSpace,
    name: str,
    polarity: Polarity,
) -> tuple[np.ndarray, np.ndarray]:
    """A polarity's point on the section and its arrow's weights.

    Returns:
        The point's local coordinates (``(1, 2)`` plane coordinates, or
        ``(1,)`` along a line) and the weights of the field components the
        arrow reads (the in-plane components :math:`(E_u, E_v)`, or the sign of
        the solved field normal to a 2D plane).

    Raises:
        ValueError: The point is off the port's plane or line, off its
            cross-section, or inside a perfectly matched layer, or a 3D
            arrow is normal to the port plane.
    """
    point, direction = polarity
    section = space.section
    local, dist = section.to_local(np.asarray([point], float))
    where = f"WavePort {name!r}: the polarity point {_numbers(point)} m"
    if abs(float(dist[0])) > 1e-5 * section.diameter:
        line = "line" if isinstance(section, LineSection) else "plane"
        raise ValueError(
            f"{where} is off the port {line}; give a point (m) on the port surface."
        )
    inside = (
        space.mesh(float(local[0]))
        if isinstance(section, LineSection)
        else space.mesh(float(local[0, 0]), float(local[0, 1]))
    )
    if inside.nr < 0:
        raise ValueError(
            f"{where} is off the port's cross-section (beyond its edge, or in a "
            "hole such as a conductor); give a point where the mode has a field."
        )
    if isinstance(section, LineSection):
        return local, np.array([direction[2]])
    regions = _layer_regions(section, local)
    if regions:
        raise ValueError(
            f"{where} lies in the perfectly matched layer ({regions}), where the "
            "field is not physical. Give a point in the physical part of the port "
            "face."
        )
    weights = section.frame[:2] @ np.asarray(direction, float)
    if np.linalg.norm(weights) < _NORMAL_ARROW:
        raise ValueError(
            f"WavePort {name!r}: the polarity direction {_numbers(direction)} is "
            "normal to the port plane, so it reads no field of the mode (the "
            "arrow reads the in-plane E field). Give a direction in the port "
            "plane, along the mode's field at the point."
        )
    return local, weights


def _polarity_functionals(
    space: ModeSpace, basis: np.ndarray, arrow: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The arrow's reading of each basis column, and the columns' peak fields.

    Returns:
        ``(M,)``: the field at the point along the arrow per column, and
        ``(M, n_nodes, c)``: the field the arrow reads (``c = 2`` in-plane
        components, or ``c = 1``) at the section's nodes, for its peak.
    """
    local, weights = arrow
    nodes = space.section.points
    if isinstance(space, PortModeSpace):
        at = space.evaluate(basis.T, local)[:, 0, :2] @ weights
        return at, space.evaluate(basis.T, nodes)[:, :, :2]
    at = space.evaluate(basis.T, local)[:, 0] * weights[0]
    return at, space.evaluate(basis.T, nodes)[:, :, None]


def _polarity_row(space: ModeSpace, arrow: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The arrow's reading as a vector on the free DOFs (`_polarity_functionals`'s).

    The field at the point is linear in the DOF vector and reads no
    material, so a material's derivative reaches it only through the mode.
    """
    local, weights = arrow
    dofs, columns = space.point_columns(local)
    reads = (
        np.einsum("ck,c->k", columns[0, :2], weights)
        if isinstance(space, PortModeSpace)
        else columns[0] * weights[0]
    )
    row = np.zeros(space.space.ndof, complex)
    np.add.at(row, dofs[0], reads)
    return row[space.free]


def _voltage_row(
    space: PortModeSpace, rule: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    r""":math:`V = \int E \cdot dl` as a vector on the free DOFs.

    The row of `_voltage_functional`.
    """
    pts, wts = rule
    dofs, weights = space.point_columns(pts)
    row = np.zeros(space.space.ndof, complex)
    np.add.at(row, dofs, np.einsum("pc,pcl->pl", wts, weights))
    return row[space.free]


def _voltage_rule(
    section: PortSection,
    path: np.ndarray,
    name: str,
    what: str = "voltage_path",
    subject: str = "WavePort",
) -> tuple[np.ndarray, np.ndarray]:
    """Plane points and weights (local direction x length, m) of a path.

    ``name`` is the wave port the path belongs to, ``what`` the path's name
    and ``subject`` what reads it, for messages.

    Raises:
        ValueError: The path leaves the port plane, has points outside the
            port section or passes through a perfectly matched layer.
    """
    path = np.asarray(path, float)
    _local, dist = section.to_local(path)
    if (np.abs(dist) > 1e-5 * section.diameter).any():
        raise ValueError(
            f"WavePort {name!r}: {what} leaves the port plane; "
            "give points (m) on the port surface."
        )
    x, w = _GAUSS
    pts, wts = [], []
    for p, q in itertools.pairwise(path):
        d_local = section.frame @ (q - p)
        for k in range(_PIECES):
            s = (k + (x + 1) / 2) / _PIECES
            pts.append(section.to_local(p + np.outer(s, q - p))[0])
            wts.append(np.outer(w / (2 * _PIECES), d_local))
    pts = np.concatenate(pts)
    _refuse_outside_path(section, pts, name, what, subject)
    _refuse_layer_path(section, pts, name, what, subject)
    return pts, np.concatenate(wts)


#: Barycentric slack of a path point at the rim of the port section.
_RIM_TOL = 1e-6


def _refuse_outside_path(
    section: PortSection, local: np.ndarray, name: str, what: str, subject: str
) -> None:
    """Raise if a path has points outside the port section's triangles.

    The path is a fixed set of points that the field is read at: off the
    section the mode has no field, and a design that moves the face under a
    path would integrate a segment that is no longer there.
    """
    pts = section.points
    tri = section.triangles
    a = pts[tri[:, 0]]
    e1, e2 = pts[tri[:, 1]] - a, pts[tri[:, 2]] - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    step = max(1, 2_000_000 // max(len(tri), 1))
    outside = []
    for start in range(0, len(local), step):
        d = local[start : start + step, None, :] - a[None]
        l1 = (d[..., 0] * e2[:, 1] - d[..., 1] * e2[:, 0]) / det
        l2 = (e1[:, 0] * d[..., 1] - e1[:, 1] * d[..., 0]) / det
        inside = (l1 >= -_RIM_TOL) & (l2 >= -_RIM_TOL) & (1 - l1 - l2 >= -_RIM_TOL)
        outside.extend(start + np.flatnonzero(~inside.any(axis=1)))
    if not outside:
        return
    first = section.to_global(local[outside[:1]])[0]
    raise ValueError(
        f"{subject}: {what} on port {name!r} has {len(outside)} of its sample "
        f"points outside the port section, the first at ({first[0]:.6g}, "
        f"{first[1]:.6g}, {first[2]:.6g}) m. The mode has no field there: keep the "
        "path on the port face, between the conductors it joins."
    )


def _refuse_layer_path(
    section: PortSection, local: np.ndarray, name: str, what: str, subject: str
) -> None:
    """Raise if a path's points lie beyond a face of the layer's interior box.

    The field there is the stretched one, so a line integral along real
    coordinates is not a voltage or a current.
    """
    names = _layer_regions(section, local)
    if names:
        raise ValueError(
            f"{subject}: {what} on port {name!r} passes through the perfectly "
            f"matched layer ({names}), where the field is not physical. Keep the "
            "path in the physical part of the port face."
        )


def _layer_regions(section: PortSection, local: np.ndarray) -> str:
    """The layer regions beyond whose interior faces plane points lie, or ``""``."""
    stretch = section.stretch
    if stretch is None:
        return ""
    points = section.to_global(local)
    tol = PLANE_RTOL * section.diameter * section.scale
    regions = [
        r
        for side in stretch.layers.sides
        if (side.sign * (points[:, side.axis] - side.face) > tol).any()
        for r in side.regions
    ]
    return ", ".join(map(repr, dict.fromkeys(regions)))


def path_voltage(
    space: PortModeSpace,
    basis: np.ndarray,
    coords: np.ndarray,
    path: np.ndarray,
    name: str,
    *,
    subject: str = "WavePort",
) -> np.ndarray:
    r""":math:`V = \int E \cdot dl` along ``path`` (m) of each field.

    The field of row ``i`` is ``basis @ coords[i]``.

    ``name`` is the wave port the field belongs to and ``subject`` what
    reads it, for messages.
    """
    rule = _voltage_rule(space.section, path, name, subject=subject)
    return coords @ _voltage_functional(space, basis, rule)


def path_current(
    space: PortModeSpace,
    basis: np.ndarray,
    coords: np.ndarray,
    *,
    gamma: np.ndarray,
    frequency: np.ndarray,
    path: np.ndarray,
    name: str,
    subject: str = "WavePort",
) -> np.ndarray:
    r"""``I = int H . dl`` along ``path`` (m) of each field ``basis @ coords[i]``.

    :math:`H_t = \frac{i}{\omega \mu_0} \nu (\nabla E_n + \gamma E_t)
    \times n` (`PortModeSpace.magnetic_parts`), at the propagation constant
    ``gamma[i]`` (1/m) and ``frequency[i]`` (Hz). ``name`` is the wave port
    the field belongs to and ``subject`` what reads it, for messages.
    """
    rule = _voltage_rule(space.section, path, name, "current_path", subject)
    pts, wts = rule
    hn, ht = space.magnetic_parts(basis.T, pts)
    ln = np.einsum("kpc,pc->k", hn, wts[:, :2])
    lt = np.einsum("kpc,pc->k", ht, wts[:, :2])
    s = space.section.scale
    omega = 2 * math.pi * np.asarray(frequency, float)
    g = np.asarray(gamma) * s  # per local unit
    return 1j / (omega * scipy.constants.mu_0 * s) * (coords @ ln + g * (coords @ lt))
