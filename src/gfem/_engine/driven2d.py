r"""Driven rule of a 2D mesh: the scalar field normal to the plane, in H1.

A 2D model solves one polarization at normal incidence, the field
:math:`u = E_z` (``"Ez"``) or :math:`u = H_z` (``"Hz"``) of a structure
uniform along :math:`z`:

.. math::

    \int w\,\nabla u\cdot\nabla v - k_0^2 \int m\,u\,v
    + (\text{boundary and port terms}) = (\text{port loads}),

with :math:`w = \mu_r^{-1}`, :math:`m = \varepsilon_r` for Ez and
:math:`w = \varepsilon_r^{-1}`, :math:`m = \mu_r` for Hz, lengths in the mesh
file's unit and :math:`k_0` per file unit. The operator is affine in
frequency as the 3D one is (`gfem._engine.terms`), and these are its pieces:

- ``stiffness``, :math:`\theta = 1`: :math:`\int w\,\nabla u\cdot\nabla v`.
  In Hz a region whose permittivity depends on the frequency (a conductivity
  or a permittivity law) leaves it for its own piece ``stiffness <region>``,
  :math:`\theta = 1/\varepsilon_r(\omega)` with
  :math:`\varepsilon_r(\omega) = \varepsilon_r(1 - i\tan\delta)
  - i\sigma/(\omega\varepsilon_0)` (plus the law's susceptibility), on the
  region's indicator;
- ``mass``, :math:`\theta = -k_0^2`: :math:`\int m\,u\,v`, with Ez's
  permittivity law and blend pieces and its ``conductivity`` piece
  (:math:`\theta = ik_0`, :math:`\int \eta_0\sigma\,u\,v` per file unit) as
  in 3D;
- ``absorbing``, :math:`\theta = ik_0`: :math:`\int Y\,u\,v` on the absorbing
  lines, :math:`Y = \sqrt{\varepsilon_r/\mu_r}` (Ez) or
  :math:`\sqrt{\mu_r/\varepsilon_r}` (Hz) of the element beside each segment
  (real parts), and ``absorbing order 2``, :math:`\theta = 1/(2ik_0)`:
  :math:`\int w(\varepsilon_r\mu_r)^{-1/2}\,\partial_t u\,\partial_t v` on the
  second-order ones. A segment beside a permittivity law takes its law's own
  pieces, with :math:`\operatorname{Re}\varepsilon_r(\omega)` in :math:`\theta`;
- each line port's Robin term :math:`\gamma_0\int w\,u\,v`
  (`gfem._engine.wave_ports.line_robin_pieces`).

A perfect conductor is the Dirichlet condition :math:`u = 0` in Ez and the
natural one in Hz; a perfect magnetic conductor, the default of every
exterior line, the reverse. So in Hz every exterior edge of the solve domain
is fixed but those of a `PEC`, `Absorbing` or line port line, and in Ez the
`PEC` lines are. A natural condition on an interior line would do nothing to a
continuous field, so a `PEC` line inside the domain raises in Hz and a `PMC`
line in Ez.

The solution is the field in V/m (Ez) or A/m (Hz) of a port fed 1 W/m. S is
read, as on a 3D model, from the electric field across each port, along the
in-plane part of :math:`(1, 1, 1)` as a 3D port's mode is phased: in Ez that
field is :math:`E_z`, in Hz the tangential :math:`E_t`, whose reflected wave
has the opposite sign of its :math:`H_z`, so a line port's Hz S is minus its
:math:`H_z` S, signed by the port's orientation (`hz_signs`). Each port's S
equals that of the slab it is the cross-section of.
"""

from __future__ import annotations

import cmath
import functools
import hashlib
import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem import _mpi, _timing
from gfem._adjoint import adjoint_tolerance
from gfem._boundaries import PEC, PMC, Absorbing, WavePort
from gfem._dim2 import refuse_anisotropic
from gfem._domain import h1_unknowns
from gfem._engine import dispersion, points, quadrature, terms
from gfem._engine.cache import default_retain_memory, discretization
from gfem._engine.coefficients import (
    coefficient,
    element_values,
    surface_coefficient,
    surface_p0,
)
from gfem._engine.domain import boundary_region, h1_space, space_key
from gfem._engine.driven import (
    DrivenOperator,
    DrivenRule,
    PortDesign,
    _check_low_frequency,
    _check_skipped_port_modes,
)
from gfem._engine.driven_sweep import SweepSplit
from gfem._engine.driven_system import DrivenSystemBuilder
from gfem._engine.fields import dof_shape
from gfem._engine.gauge import GaugeChoice
from gfem._engine.port_mode import admittance
from gfem._engine.sensitivity import Derivative
from gfem._engine.sweep_outputs import SweepOutputs
from gfem._engine.system_operator import FixedPieces
from gfem._linalg import ResolvedLinearSolver
from gfem._linalg.pieces import KeptParts
from gfem._line_section import line_section
from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import _POLARIZATIONS, Driven, Polarization
from gfem.results import DrivenResult, Field, NetworkParameters, SweepInfo

__all__ = ["Driven2DRule", "driven2d_operator", "hz_signs"]

#: A 2D solve's gauge choice: there is no gradient null space to gauge.
_UNGAUGED = GaugeChoice(on=False, reason="a 2D model's scalar field")


class Driven2DRule(DrivenRule):
    """Host-side forward of a `Driven` problem on a 2D `Model`.

    The uniform sweep of `DrivenRule` on the scalar operator of the
    problem's polarization (`driven2d_operator`), without MPI ranks; its
    frequencies may be split over frequency workers.
    """

    def __init__(
        self,
        model: Model,
        problem: Driven,
        options: SolverOptions,
        *,
        comm: Any = None,
        order_frequency: Sequence[float] | None = None,
        gauge: GaugeChoice | None = None,
    ) -> None:
        """Check that the problem fits the 2D model.

        ``comm`` is always ``None`` (refused at the solve entry);
        ``order_frequency`` and ``gauge`` are accepted for a frequency worker
        and ignored.

        Raises:
            ValueError: The problem does not fit the model: no ports, an
                unknown excited port, or a natural condition on an interior
                line.
            NotImplementedError: An Hz material field whose permittivity
                depends on the frequency.
        """
        del order_frequency, gauge
        if not isinstance(problem, Driven):
            raise TypeError(f"Expected gfem.Driven, got {problem!r}.")
        polarization = problem.polarization
        if polarization is None or polarization not in _POLARIZATIONS:
            # the solve entry checks it first
            raise ValueError(f"Unknown polarization {polarization!r}.")
        self.problem = problem
        self.options = options
        self.comm = comm
        self.polarization: Polarization = polarization
        self.swept = False
        self.wave_vector = None
        self.swept_spaces = False
        self.orders: dict = {}
        #: No perfectly matched layer: a 2D model refuses one.
        self.stretch = self.pml = None
        refuse_anisotropic(model)
        _check_lines(model, self.polarization)
        _check_materials(model, self.polarization)
        #: The layer grids of two constant ends, whose pixels Hz differentiates.
        self._constant_grids = frozenset(
            r
            for r, f in points.point_blends(model).items()
            if f.blend_ends is not None and _constant_ends(f.blend_ends)
        )
        self.linear = _resolve(model, problem, options)
        self.gauge = _UNGAUGED
        # a lossless pole raises, a law outside its fitted range warns
        dispersion.check_frequencies(
            model, problem.frequency, reader="driven solve", comm=comm
        )
        if not model.ports:
            raise ValueError(
                "The model has no ports, so a driven problem has nothing to "
                "excite. Add a gfem.WavePort on a straight exterior line."
            )
        names = model.port_names
        self._init_columns(names, problem)
        self.order_frequency = tuple(problem.frequency)
        self.kept = tuple(problem.frequency.index(f) for f in problem.kept_frequencies)
        self.adaptive = False
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
        self.wave_ports = tuple(p.name for p in model.ports if isinstance(p, WavePort))
        #: The wave ports by name, for the fields a port mode result echoes.
        self._wave_port_specs = {
            p.name: p for p in model.ports if isinstance(p, WavePort)
        }
        self.lumped_loads = ()
        self.modes = {p.name: p.mode for p in model.ports if isinstance(p, WavePort)}
        self.offsets = tuple(
            (p.name, p.offset) for p in model.ports if isinstance(p, WavePort)
        )
        self.port_design = PortDesign.of(model, sheets=False)
        self.nodes_unsupported = _NO_GRADIENTS
        self._mesh = model.mesh
        self._excluded = model._domain.excluded  # noqa: SLF001
        #: Per port, the sign its Hz S row and column take (`hz_signs`); ones
        #: in Ez.
        self.signs = hz_signs(model, self.polarization)
        self.systems = DrivenSystemBuilder(
            frequency=problem.frequency,
            columns=tuple(names.index(n) for n in self.excite),
            excite=tuple(self.excite),
            port_mode_tolerance=options.port_mode_tolerance,
            floquet_order_tolerance=options.floquet_order_tolerance,
            orders=self.orders,
            order_frequency=self.order_frequency,
            wave_vector=self.wave_vector,
            problem_wave_vector=problem.wave_vector,
            wave_vector_frequency=problem.wave_vector_frequency,
            swept_spaces=self.swept_spaces,
            comm=comm,
            linear=self.linear,
            adjoint_tolerance=adjoint_tolerance(options),
            retain_memory=options.retain_memory,
            gauge=self.gauge.on,
            polarization=self.polarization,
        )
        #: No fixed-angle sweep in 2D.
        self.children = None
        #: Frequency workers' blocks, or ``None``.
        self.split = SweepSplit.of(self)
        _check_low_frequency(model, problem.frequency, None, stacklevel=4)
        _check_skipped_port_modes(model, stacklevel=4)

    def _init_columns(self, names: tuple[str, ...], problem: Driven) -> None:
        """The excited ports, and the source attributes the shared code reads.

        A 2D `Driven` refuses sources when it is built, so the source
        attributes are empty and the solve's columns are the excited ports.

        Raises:
            ValueError: ``problem.excite`` names a port the model lacks.
        """
        self.port_names = names
        self.excite = names if problem.excite is None else problem.excite
        for name in self.excite:
            if name not in names:
                raise ValueError(
                    f"Driven.excite names port {name!r}, but the model's ports are "
                    f"{', '.join(map(repr, names))}."
                )
        self.sources = ()
        self.incident = ()
        self.dipoles = ()
        self.reciprocal = False
        self.emitting = ()
        self.columns = tuple(self.excite)
        self.background = None
        self.layers = ()
        self.truncation_designs = {}

    def operator(self, model: Model) -> DrivenOperator:
        """The model's scalar driven operator; the last model's is kept.

        Raises:
            NotImplementedError: An Hz material field with a nonzero
                conductivity (`_check_materials`, on the concrete values).
        """
        _check_materials(model, self.polarization)
        last = getattr(self, "_last_operator", None)
        if last is not None and last[0] is model:
            return last[1]
        with _timing.phase("discretize") as span:
            op = driven2d_operator(model, self.options.order, self.polarization)
            span.set(ndof=int(op.space.ndof))
        self._last_operator = (model, op)
        return op

    def _field_shape(self, model: Model) -> tuple[int, int]:
        """``(width, ndof)`` of a kept field: H1 of the solve order."""
        return dof_shape(
            model.mesh, self.regions, self._excluded, "h1", self.options.order
        )

    def unsupported_input(self, kind: tuple) -> str | None:
        """Why a read input cannot be differentiated here, or ``None``.

        A `MaterialField`'s values are differentiated, as on a 3D model, those
        of a region beside a line port included (through the port's mode); in
        Hz but a law blend's density, which the forward refuses (a layer
        grid's pixels between two constant ends are differentiated). An Hz
        field's conductivity is differentiated at its only admitted value,
        zero.
        Every other input (node coordinates, law parameters, boundary and
        port values) is not.
        """
        if kind[0] != "material":
            return _NO_GRADIENTS
        if (
            self.polarization == "Hz"
            and kind[2] == "blend"
            and kind[1] not in self._constant_grids
        ):
            return _HZ_FREQUENCY_DEPENDENT
        return super().unsupported_input(kind)

    def result(self, outputs: dict) -> DrivenResult:
        """The public result from the forward outputs."""
        return self._uniform_result(SweepOutputs.read(outputs))

    def _uniform_result(self, out: SweepOutputs) -> DrivenResult:
        """`result` of the uniform sweep over the grid indices ``out.rows``.

        S and the kept fields take each port's sign (`signs`): S as
        ``-signs_i S_ij signs_j`` in Hz, a field of port j times its sign.
        """
        grid = self.problem.frequency
        frequencies = tuple(grid[i] for i in out.rows)
        kept = [
            (grid[n], port) for n in self.kept if n in out.rows for port in self.excite
        ]
        hz = self.polarization == "Hz"
        signs = np.asarray(self.signs, float)
        columns = [self.port_names.index(p) for p in self.excite]
        s = jnp.asarray(out.s)
        if hz:
            s = -s * signs[None, :, None] * signs[None, None, columns]
        fields = []
        coefficients: Any = out.e_field
        if coefficients is not None:
            for k, (f, port) in enumerate(kept):
                sign = signs[self.port_names.index(port)]
                fields.append(
                    Field(
                        coefficients=coefficients[:, k] * sign,
                        mesh=self._mesh,
                        quantity=self.polarization,
                        unit="A/m" if hz else "V/m",
                        space="h1",
                        order=self.options.order,
                        support=self.regions,
                        omega=jnp.asarray(2 * math.pi * f),
                        _excluded=self._excluded,
                        _comm=_mpi.static_comm(self.comm),
                    )
                )
        return DrivenResult(
            s=NetworkParameters(
                values=s,
                z_ref=jnp.asarray(out.z_ref),
                frequency=frequencies,
                ports=self.port_names,
                excited=tuple(self.excite),
            ),
            e_fields=() if hz else tuple(fields),
            h_fields=tuple(fields) if hz else (),
            polarization=self.problem.polarization,
            kept=tuple(kept),
            sweep=SweepInfo.uniform(
                frequencies,
                iterations=out.iterations,
                preconditioner_reused=out.reused,
            ),
            port_modes=self._port_modes(out),
            offsets=self.offsets,
        )


#: Why a 2D model's other inputs cannot be differentiated in this version.
_NO_GRADIENTS = (
    "gradients of a 2D model are available in this version with respect to "
    "material field values only"
)
#: Why an Hz solve cannot be differentiated by a frequency-dependent field.
_HZ_FREQUENCY_DEPENDENT = (
    "an Hz solve of a 2D model takes no material field blended with a "
    "permittivity law; solve the Ez polarization"
)


def _resolve(
    model: Model, problem: Driven, options: SolverOptions
) -> ResolvedLinearSolver:
    """The linear solver, with ``"auto"`` sized by the H1 unknowns.

    The solve entry refuses BDDC on a 2D mesh, so ``"auto"`` resolves to the
    direct solver, and states its estimate from the H1 unknown count.
    """
    inputs: dict[str, Any] = {}
    if options.linear_solver == "auto" and options.device == "cpu":
        inputs = {
            "unknowns": h1_unknowns(model.mesh, model._domain, options.order),  # noqa: SLF001
            "in_flight": min(options.frequency_workers, len(problem.frequency)),
            "budget": default_retain_memory(),
        }
    return ResolvedLinearSolver.resolve(
        options, default="direct", problem="driven", **inputs
    )


# --- checks ------------------------------------------------------------------------


def _check_lines(model: Model, polarization: Polarization) -> None:
    """Raise for a natural condition on an interior line, or an Hz port's end.

    Raises:
        ValueError: A `PEC` line inside the solve domain in Hz, or a `PMC`
            line in Ez; in Hz, a wave port's ``pec_edges`` line on which the
            field is fixed.
    """
    domain = model._domain  # noqa: SLF001
    natural, what = (PEC, "PEC") if polarization == "Hz" else (PMC, "PMC")
    for b in model.boundaries:
        if not isinstance(b, natural) or not domain.interior(b.surface):
            continue
        remedy = (
            "Cut the mesh along the strip (two lines, one per side) or mesh the "
            "metal as an area of its own and exclude it (excluded=), naming its "
            "edge PEC."
            if polarization == "Hz"
            else "Cut the mesh along the line, or name an exterior line PMC."
        )
        raise ValueError(
            f"{what}({b.surface!r}) lies inside the solve domain, and in "
            f"{polarization} it is a natural condition, which does nothing on an "
            f"interior line: the line would be transparent. {remedy}"
        )
    if polarization == "Hz":
        _check_pec_edges(model)


def _check_pec_edges(model: Model) -> None:
    """Raise for an Hz port's ``pec_edges`` line on which the field is fixed.

    In Hz a PEC end of the line mode is natural, so the mode is free there,
    while the 2D field is fixed on every line but a `PEC`, `Absorbing` or
    wave port line: the coupling would drop the mode's end value.
    """
    natural = {
        b.surface for b in model.boundaries if isinstance(b, (PEC, Absorbing))
    } | {p.surface for p in model.ports if isinstance(p, WavePort)}
    for port in model.ports:
        if not isinstance(port, WavePort):
            continue
        for line in port.pec_edges:
            if line in natural:
                continue
            raise ValueError(
                f"WavePort {port.name!r} names {line!r} in pec_edges, but in Hz "
                f"the 2D field is fixed on {line!r} (a PMC line, the default) "
                "while the port mode is free at a PEC end, so the two would "
                f"disagree. Add gfem.PEC({line!r}) to the boundaries, or leave "
                "the line out of pec_edges."
            )


def _check_materials(model: Model, polarization: Polarization) -> None:
    """Raise for an Hz material field whose permittivity depends on the frequency.

    In Hz the permittivity enters the stiffness as its inverse, one piece per
    frequency-dependent group of elements: a field's per-element conductivity
    or law blend is no such family.

    Raises:
        NotImplementedError: An Hz `MaterialField` with a conductivity or a
            permittivity-law end.
    """
    if polarization != "Hz":
        return
    for region, material in model.materials.items():
        if not isinstance(material, MaterialField):
            continue
        conductive = False
        if material.conductivity is not None:
            try:
                conductive = bool(np.any(np.asarray(material.conductivity) != 0))
            except jax.errors.TracerArrayConversionError:
                # traced at the solve entry: the host forward checks the values
                conductive = False
        ends = material.blend_ends
        if ends is not None and ends.pixels is not None:
            # a layer grid's blend of two constant ends is applied per point
            conductive = any(
                n == "conductivity" and (a or b) for n, a, b in ends.others
            )
            law = not _constant_ends(ends)
        else:
            law = ends is not None
        if conductive or law:
            what = "a conductivity" if conductive else "a permittivity-law end"
            raise NotImplementedError(
                f"The material field of region {region!r} has {what}, so its "
                "permittivity depends on the frequency element by element, which "
                'an Hz solve (Driven(polarization="Hz")) of a 2D model does not '
                "take. Give the region a gfem.Material, or solve the Ez "
                "polarization."
            )


def _constant_ends(ends: Any) -> bool:
    """Whether neither end of a blend is a permittivity law."""
    return not any(isinstance(e, PermittivityLaw) for e in (ends.void, ends.solid))


# --- the operator ------------------------------------------------------------------


def hz_signs(model: Model, polarization: Polarization) -> np.ndarray:
    r"""Per port, the sign that phases its Hz mode as a 3D port's mode is phased.

    A 3D port's mode is phased by :math:`\int t\cdot(\hat n_\text{out}\times
    H)`, ``t`` the in-plane part of :math:`(1, 1, 1)`. On the slab of a line
    port with inward normal :math:`\hat n` that is
    :math:`(n_x - n_y)\int H_z`, so the slab's mode is the line port's (whose
    :math:`\int w H_z` is positive) times :math:`\operatorname{sign}(n_x -
    n_y)`. Ones in Ez, where both phasings read :math:`E_z`. A port whose
    normal has :math:`n_x = n_y`, :math:`\pm(1, 1)/\sqrt{2}`, gets +1: there the 3D
    phasing integral is zero and fixes no sign, so the slab's mode may differ
    from the 2D port's by its sign.
    """
    out = np.ones(len(model.ports))
    if polarization != "Hz":
        return out
    for i, port in enumerate(model.ports):
        if isinstance(port, WavePort):
            n = line_section(model, port, geometry=True).frame[1]
            out[i] = -1.0 if n[1] > n[0] else 1.0
    return out


def driven2d_operator(
    model: Model, order: int, polarization: Polarization
) -> DrivenOperator:
    """The scalar driven operator of a 2D ``model`` (see the module docstring)."""
    disc = discretization(model.mesh, (), None)
    built = disc.built
    dirichlet, fixed_edges = _fixed(model, polarization)
    # the edges fixed besides the named lines are part of the space's key
    edges = hashlib.blake2b(fixed_edges.tobytes(), digest_size=16).hexdigest()
    key = space_key(model, "h1", order, fixed=(polarization, *dirichlet, edges))

    def build() -> ngs.FESpace:
        space = h1_space(model, built, order, dirichlet, complex_valued=True)
        _fix_edges(space, built, fixed_edges)
        return space

    space = disc.space(key, build)
    pieces, latent = scalar_pieces(model, built, order, polarization)
    mesh = model.mesh
    inside = np.zeros(len(mesh.volume_elements), bool)
    for region, material in model.materials.items():
        if isinstance(material, MaterialField):
            inside[mesh.elements(region)] = True
    fixed = None
    if inside.any() and not inside.all():
        no_faces = np.zeros(len(mesh.surface_elements), bool)
        fixed = FixedPieces(
            disc.space((*key, "fixed pieces"), KeptParts),
            ~inside,
            lambda: scalar_pieces(
                model, built, order, polarization, elements=inside, faces=no_faces
            )[0],
        )
    return DrivenOperator(
        built,
        space,
        pieces,
        (),
        order,
        fixed=fixed,
        topology=mesh._topology,  # noqa: SLF001
        space_key=key,
        latent=latent,
    )


def _fixed(
    model: Model, polarization: Polarization
) -> tuple[tuple[str, ...], np.ndarray]:
    """The Dirichlet lines by name, and the exterior edges fixed besides them.

    Ez fixes the `PEC` lines. Hz fixes the `PMC` lines and every exterior edge
    of the solve domain that no `PEC`, `Absorbing` or port line covers: the
    default PMC, on named and unnamed edges alike. Edges as ``(n, 2)`` gfem
    node pairs, sorted.
    """
    names = PEC if polarization == "Ez" else PMC
    dirichlet = tuple(
        sorted({b.surface for b in model.boundaries if isinstance(b, names)})
    )
    if polarization == "Ez":
        return dirichlet, np.zeros((0, 2), np.int64)
    mesh = model.mesh
    topology = mesh._topology  # noqa: SLF001
    active = model._domain.active  # noqa: SLF001
    cells = topology.volume_elements[active]
    edges = np.sort(cells[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, count = np.unique(edges, axis=0, return_counts=True)
    exterior = edges[count == 1]
    natural = {
        b.surface for b in model.boundaries if isinstance(b, (PEC, Absorbing))
    } | {p.surface for p in model.ports if isinstance(p, WavePort)}
    tags = [mesh.region(s).tag for s in natural]
    covered = np.sort(topology.surface_elements[np.isin(topology.surface_tags, tags)])
    keep = ~_rows_in(exterior, covered)
    return dirichlet, np.ascontiguousarray(exterior[keep], np.int64)


def _rows_in(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Whether each row of ``rows`` is a row of ``table`` (both ``(n, 2)`` ints)."""
    if not len(table) or not len(rows):
        return np.zeros(len(rows), bool)
    width = int(max(rows.max(), table.max())) + 1
    return np.isin(rows[:, 0] * width + rows[:, 1], table[:, 0] * width + table[:, 1])


def _fix_edges(space: ngs.FESpace, built: Any, edges: np.ndarray) -> None:
    """Take the DOFs of the gfem node pairs ``edges`` out of ``space.FreeDofs()``.

    Their vertices' and the edges' own DOFs: the essential condition on
    edges with no line of their own in the mesh.
    """
    if not len(edges):
        return
    vertex = np.empty(len(built.node_order), np.int64)
    vertex[built.node_order] = np.arange(len(built.node_order))
    wanted = {(int(a), int(b)) for a, b in np.sort(vertex[edges], axis=1)}
    free = space.FreeDofs()
    for edge in built.mesh.edges:
        a, b = sorted(v.nr for v in edge.vertices)
        if (a, b) not in wanted:
            continue
        dofs = [
            *space.GetDofNrs(edge),
            *space.GetDofNrs(ngs.NodeId(ngs.VERTEX, a)),
            *space.GetDofNrs(ngs.NodeId(ngs.VERTEX, b)),
        ]
        for d in dofs:
            if d >= 0:
                free.Clear(int(d))


def scalar_pieces(
    model: Model,
    built: Any,
    order: int,
    polarization: Polarization,
    *,
    elements: np.ndarray | None = None,
    faces: np.ndarray | None = None,
) -> tuple[list[terms.Piece], tuple[terms.Piece, ...]]:
    """The pieces of the scalar operator (see the module docstring), and the latent.

    ``elements`` and ``faces`` restrict them to those area triangles and line
    segments (boolean masks, ``None`` for all), as `gfem._engine.terms.pieces`.
    """
    scale = built.scale
    cells = terms._bits(elements, built.cell_order, "triangle")  # noqa: SLF001
    dx = quadrature.dx(order, definedonelements=cells)
    eps = dispersion.high_frequency_eps(model)
    mu = element_values(model, "mu_r")
    # a layer grid's elements take their coefficients at the points only
    grids = np.zeros(len(mu), bool)
    for region in points.point_blends(model):
        grids[model.mesh.elements(region)] = True
    out: list[terms.Piece] = []
    latent: tuple[terms.Piece, ...] = ()
    if polarization == "Ez":
        nu = np.where(grids, 0.0, 1.0 / mu)
        nu_cf = coefficient(built, nu)
        eps_cf = coefficient(built, eps)
        out += [
            terms.Piece(
                "stiffness",
                terms._constant,  # noqa: SLF001
                lambda u, v: (nu_cf * ngs.grad(u)) * ngs.grad(v) * dx,
                kind="gradient",
                coefficient=nu_cf,
                values=nu,
                derivatives={"mu_r": Derivative(lambda: -(mu**-2.0), holomorphic=True)},
            ),
            terms.Piece(
                "mass",
                lambda w: -((w * scale / const.c) ** 2),
                lambda u, v: (eps_cf * u) * v * dx,
                kind="value",
                coefficient=eps_cf,
                values=eps,
                derivatives=terms.mass_derivatives(model),
            ),
        ]
        out += terms.susceptibility_pieces(model, built, order, elements=elements)
        out += terms.blend_pieces(model, built, order, elements=elements)
        conductivity = terms.conductivity_piece(model, built, order, elements=elements)
        if np.any(conductivity.values):
            out.append(conductivity)
        else:
            latent = (conductivity,)
    else:
        varying = _varying_regions(model)
        steady = np.where(grids, 0.0, 1.0)
        for mask, _ in varying.values():
            steady[mask] = 0.0
        inverse = np.where(steady > 0, 1.0 / np.where(steady > 0, eps, 1.0), 0.0)
        inv_cf = coefficient(built, inverse)
        mu = np.where(grids, 0.0, mu)
        mu_cf = coefficient(built, mu)
        out += [
            terms.Piece(
                "stiffness",
                terms._constant,  # noqa: SLF001
                lambda u, v: (inv_cf * ngs.grad(u)) * ngs.grad(v) * dx,
                kind="gradient",
                coefficient=inv_cf,
                values=inverse,
                derivatives=_inverse_derivatives(model, steady > 0),
            ),
            terms.Piece(
                "mass",
                lambda w: -((w * scale / const.c) ** 2),
                lambda u, v: (mu_cf * u) * v * dx,
                kind="value",
                coefficient=mu_cf,
                values=mu,
                derivatives={"mu_r": Derivative(1.0, holomorphic=True)},
            ),
        ]
        for region, (mask, material) in varying.items():
            out.append(_stiffness_piece(built, region, mask, material, dx))
        latent = (_conductivity_stiffness(model, built, steady > 0, dx),)
    out += _point_pieces(model, built, order, polarization, elements=elements)
    out += _absorbing_pieces(model, built, order, polarization, faces=faces)
    return out, latent


def _point_pieces(
    model: Model,
    built: Any,
    order: int,
    polarization: Polarization,
    *,
    elements: np.ndarray | None,
) -> list[terms.Piece]:
    r"""The pieces of each layer grid's region, on its quadrature points.

    The blend at the points' densities :math:`\rho_q` (`gfem._engine.points`):
    in Ez the stiffness :math:`\mu_r^{-1}`, the mass :math:`\varepsilon_r`
    (with a law end, its eps_inf limit and one piece per susceptibility part,
    as `gfem._engine.terms.blend_pieces` per element) and a nonzero
    conductivity; in Hz, whose ends are constant, the stiffness
    :math:`\varepsilon_r^{-1}` and the mass :math:`\mu_r`. Each carries
    :math:`dc/d\rho` per point as its ``"blend"`` derivative.
    """
    scale = built.scale
    out: list[terms.Piece] = []

    def k2(w: complex) -> complex:
        return (w * scale / const.c) ** 2

    def mass_theta(w: complex) -> complex:
        return -k2(w)

    for region, field in points.point_blends(model).items():
        pm = points.point_map(model, region, order)
        if elements is not None and not elements[pm.elements].any():
            continue
        ends = field.blend_ends
        if ends is None:  # pragma: no cover - point_blends holds blends only
            continue
        rho = pm.matrix @ np.asarray(field.blend, dtype=float).ravel()
        mu, d_mu = points.other(ends, "mu_r", rho, 1.0)

        piece = functools.partial(
            _point_piece, model=model, order=order, pm=pm, region=region
        )

        if polarization == "Hz":
            eps, d_eps = points.constant_permittivity(ends, rho)
            out += [
                piece(
                    "stiffness",
                    terms._constant,  # noqa: SLF001
                    1.0 / eps,
                    -d_eps / eps**2,
                    kind="gradient",
                ),
                piece("mass", mass_theta, mu, d_mu),
            ]
            continue
        out.append(
            piece(
                "stiffness",
                terms._constant,  # noqa: SLF001
                1.0 / mu,
                -d_mu / mu**2,
                kind="gradient",
            )
        )
        law = any(isinstance(e, PermittivityLaw) for e in (ends.void, ends.solid))
        if not law:
            eps, d_eps = points.constant_permittivity(ends, rho)
            out.append(piece("mass", mass_theta, eps, d_eps))
        else:
            br = dispersion.BlendRegion(region, ends, np.ones(len(rho), bool), rho)
            eps, d_eps = dispersion.blend_mass(br)
            out.append(piece("mass", mass_theta, eps, d_eps))
            for part in dispersion.blend_parts(br):

                def theta(w: complex, part: Any = part) -> complex:
                    return -k2(w) * part.chi(w)

                def d_theta(w: complex, part: Any = part) -> complex:
                    return -k2(w) * (2 / w * part.chi(w) + part.d_chi(w))

                out.append(
                    piece(
                        f"blend {part.name}",
                        theta,
                        part.c,
                        part.dc,
                        frequency_class="nonpolynomial",
                        d_theta=d_theta,
                        probe=terms._LAW_PROBE,  # noqa: SLF001
                    )
                )
        sigma, d_sigma = points.other(ends, "conductivity", rho, 0.0)
        if np.any(sigma) or np.any(d_sigma):
            out.append(
                piece(
                    "conductivity",
                    lambda w: 1j * terms.wavenumber(w, scale),
                    terms.ETA0 * scale * np.real(sigma),
                    terms.ETA0 * scale * np.real(d_sigma),
                )
            )
    return out


def _inverse_derivatives(model: Model, steady: np.ndarray) -> dict[str, Derivative]:
    r"""The derivatives of the Hz stiffness coefficient :math:`1/\varepsilon`.

    With :math:`\varepsilon = \varepsilon_r(1 - i\tan\delta)` on the
    ``steady`` elements (zero on the others, which have pieces of their own):
    :math:`-(1 - i\tan\delta)/\varepsilon^2` by ``eps_r`` and
    :math:`i\varepsilon_r/\varepsilon^2` by ``tan_delta``.
    """

    def by(factor: np.ndarray) -> np.ndarray:
        eps = dispersion.high_frequency_eps(model)
        return np.where(steady, factor / np.where(steady, eps, 1.0) ** 2, 0.0)

    def loss() -> np.ndarray:
        return 1.0 - 1j * np.real(element_values(model, "tan_delta"))

    return {
        "eps_r": Derivative(lambda: by(-loss()), holomorphic=True),
        "tan_delta": Derivative(
            lambda: by(1j * element_values(model, "eps_r", law=1.0)),
            holomorphic=False,
        ),
    }


def _conductivity_stiffness(
    model: Model, built: Any, steady: np.ndarray, dx: Any
) -> terms.Piece:
    r"""The latent Hz piece of a material field's conductivity.

    An Hz field's conductivity is zero (`_check_materials`), but its
    derivative is not: :math:`\varepsilon - i\sigma/(\omega\varepsilon_0)`
    moves the stiffness coefficient by
    :math:`(i/(\omega\varepsilon_0))\,\varepsilon^{-2}` per unit
    :math:`\sigma`. So the piece has
    :math:`\theta = i/(\omega\varepsilon_0)`, a zero coefficient, and the
    derivative :math:`\varepsilon^{-2}` on the ``steady`` elements.
    """
    values = np.zeros(len(steady))
    cf = coefficient(built, values)

    def by_sigma() -> np.ndarray:
        eps = dispersion.high_frequency_eps(model)
        return np.where(steady, 1.0 / np.where(steady, eps, 1.0) ** 2, 0.0)

    return terms.Piece(
        "stiffness conductivity",
        lambda w: 1j / (w * const.epsilon_0),
        lambda u, v: (cf * ngs.grad(u)) * ngs.grad(v) * dx,
        kind="gradient",
        coefficient=cf,
        values=values,
        frequency_class="nonpolynomial",
        d_theta=lambda w: -1j / (w**2 * const.epsilon_0),
        derivatives={"conductivity": Derivative(by_sigma, holomorphic=False)},
    )


def _varying_regions(model: Model) -> dict[str, tuple[np.ndarray, Material]]:
    """The `Material` regions whose permittivity depends on the frequency.

    Those with a conductivity or a permittivity law, each with its active
    elements.
    """
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    out = {}
    for region, material in model.materials.items():
        if not isinstance(material, Material):
            continue
        law = isinstance(material.eps_r, PermittivityLaw)
        if not law and not material.conductivity:
            continue
        mask = np.zeros(len(mesh.volume_elements), bool)
        mask[mesh.elements(region)] = True
        mask &= active
        if mask.any():
            out[region] = (mask, material)
    return out


def _stiffness_piece(
    built: Any, region: str, mask: np.ndarray, material: Material, dx: Any
) -> terms.Piece:
    r"""An Hz region's stiffness piece.

    :math:`\int\nabla u\cdot\nabla v` on the region, with
    :math:`\theta = 1/\varepsilon_r(\omega)`.
    """
    law = material.eps_r if isinstance(material.eps_r, PermittivityLaw) else None
    # an isotropic material (anisotropy is refused on a 2D mesh): one value each
    values: Any = material
    sigma = float(values.conductivity or 0.0)
    if law is None:
        tan = float(values.tan_delta or 0.0)
        steady = complex(values.eps_r) * (1 - 1j * tan)
    else:
        steady = complex(float(law.eps_inf))

    def eps(w: complex) -> complex:
        value = steady - 1j * sigma / (w * const.epsilon_0)
        if law is not None:
            value += dispersion.susceptibility(law, w)
        return value

    def d_eps(w: complex) -> complex:
        slope = 1j * sigma / (w**2 * const.epsilon_0)
        if law is not None:
            slope += dispersion.d_susceptibility(law, w)
        return slope

    values = mask.astype(float)
    cf = coefficient(built, values)
    return terms.Piece(
        f"stiffness {region}",
        lambda w: 1 / eps(w),
        lambda u, v: (cf * ngs.grad(u)) * ngs.grad(v) * dx,
        kind="gradient",
        coefficient=cf,
        values=values,
        frequency_class="nonpolynomial",
        d_theta=lambda w: -d_eps(w) / eps(w) ** 2,
        probe=terms._LAW_PROBE,  # noqa: SLF001
    )


def _point_piece(
    name: str,
    theta: Any,
    values: np.ndarray,
    slope: np.ndarray,
    *,
    model: Model,
    order: int,
    pm: points.PointMap,
    region: str,
    kind: str = "value",
    **kw: Any,
) -> terms.Piece:
    """A piece of ``region``'s layer grid: ``values`` and their slope per point."""
    cf = points.function(model, order, pm, values)
    dx = quadrature.dx(order, definedonelements=pm.cells)
    if kind == "gradient":
        form = lambda u, v: (cf * ngs.grad(u)) * ngs.grad(v) * dx  # noqa: E731
    else:
        form = lambda u, v: (cf * u) * v * dx  # noqa: E731
    return terms.Piece(
        f"points {name} {region}",
        theta,
        form,
        kind="gradient" if kind == "gradient" else "value",
        coefficient=cf,
        derivatives={"blend": Derivative(lambda: slope, holomorphic=True)},
        points=region,
        **kw,
    )


def _absorbing_pieces(
    model: Model,
    built: Any,
    order: int,
    polarization: Polarization,
    *,
    faces: np.ndarray | None,
) -> list[terms.Piece]:
    r"""The absorbing lines' pieces, order 1 and 2, with their law pieces.

    With :math:`e`, :math:`m` the real permittivity and permeability of the
    element beside each segment: order 1 :math:`\sqrt{e/m}` (Ez) or
    :math:`\sqrt{m/e}` (Hz), order 2 :math:`1/(m\sqrt{em})` (Ez) or
    :math:`1/(e\sqrt{em})` (Hz). Beside a law :math:`e` is
    :math:`\operatorname{Re}\varepsilon_r(\omega)`, in θ.
    """
    absorbing = [b for b in model.boundaries if isinstance(b, Absorbing)]
    if not absorbing:
        return []
    terms._refuse_absorbing_blends(model)  # noqa: SLF001
    scale = built.scale
    segments = terms._bits(faces, built.facet_order, "line segment")  # noqa: SLF001
    surfaces, owner = terms.absorbing_faces(model, order=1)
    second = tuple(b.surface for b in absorbing if b.order == 2)
    tan = np.real(element_values(model, "tan_delta"))
    e = np.real(element_values(model, "eps_r", law=1.0) * (1.0 - 1j * tan))
    m = np.real(element_values(model, "mu_r"))
    ez = polarization == "Ez"
    first = admittance(e, m) if ez else admittance(m, e)
    curl = 1 / (m * np.sqrt(e * m)) if ez else 1 / (e * np.sqrt(e * m))
    ok = owner >= 0
    y1 = np.where(ok, first[np.maximum(owner, 0)], 0.0)
    y2 = np.where(ok, curl[np.maximum(owner, 0)], 0.0)
    laws = terms.law_faces(model, surfaces, owner)
    # a law's segments take its own pieces, which no material field reads
    on_law = np.zeros(len(owner), bool)
    for _, lf in laws:
        y1[lf] = 0.0
        y2[lf] = 0.0
        on_law |= lf

    def owned(lines: tuple[str, ...]) -> np.ndarray:
        return terms._owned(model, lines, owner, faces)  # noqa: SLF001

    region1 = boundary_region(built, surfaces)
    ds1 = quadrature.ds(order, definedon=region1, definedonelements=segments)
    y1_cf = surface_coefficient(built, y1)

    def k(w: complex) -> complex:
        return w * scale / const.c

    out = [
        terms.Piece(
            "absorbing",
            lambda w: 1j * k(w),
            lambda u, v: y1_cf * u * v * ds1,
            kind="trace",
            coefficient=y1_cf,
            region=region1,
            derivatives=terms.absorbing_derivatives(model, order=1, swapped=not ez),
            owner=np.where(on_law, -1, owned(surfaces)),
        )
    ]
    if second:
        region2 = boundary_region(built, second)
        ds2 = quadrature.ds(order, definedon=region2, definedonelements=segments)
        y2_cf = surface_p0(built, y2)
        out.append(
            terms.Piece(
                "absorbing order 2",
                lambda w: 1 / (2j * k(w)),
                lambda u, v: y2_cf * u.Trace().Deriv() * v.Trace().Deriv() * ds2,
                kind="tangential gradient",
                coefficient=y2_cf,
                region=region2,
                frequency_class="nonpolynomial",
                d_theta=lambda w: -1 / (2j * k(w) * w),
                derivatives=terms.absorbing_derivatives(model, order=2, swapped=not ez),
                owner=np.where(on_law, -1, owned(second)),
            )
        )
    tags = model.mesh._topology.surface_tags  # noqa: SLF001
    on_second = np.isin(tags, [model.mesh.region(s).tag for s in second])
    for lr, lf in laws:
        out += _law_absorbing(
            lr,
            lf,
            on_second & lf,
            m[np.maximum(owner, 0)],
            ez=ez,
            k=k,
            built=built,
            forms=(ds1, ds2 if second else None),
        )
    return out


def _law_absorbing(
    lr: Any,
    lf: np.ndarray,
    on: np.ndarray,
    m: np.ndarray,
    *,
    ez: bool,
    k: Any,
    built: Any,
    forms: tuple[Any, Any],
) -> list[terms.Piece]:
    r"""A permittivity law's absorbing pieces.

    :math:`e = \operatorname{Re}\varepsilon(\omega)` is in θ.

    Order 1 :math:`\theta = ik_0 e^{\pm 1/2}` on :math:`m^{\mp 1/2}`, order 2
    :math:`\theta = e^{-1/2}/(2ik_0)` on :math:`m^{-3/2}` (Ez) or
    :math:`e^{-3/2}/(2ik_0)` on :math:`m^{-1/2}` (Hz); ``d_theta`` along real ω.
    """
    ds1, ds2 = forms
    law = lr.law

    def e(w: complex) -> float:
        return float(law.eps_inf) + dispersion.susceptibility(law, w).real

    def de(w: complex) -> float:
        return dispersion.d_susceptibility(law, w).real

    p1 = 0.5 if ez else -0.5
    p2 = -0.5 if ez else -1.5

    def theta1(w: complex) -> complex:
        return 1j * k(w) * cmath.exp(p1 * cmath.log(e(w)))

    def d_theta1(w: complex) -> complex:
        return theta1(w) * (1 / w + p1 * de(w) / e(w))

    def theta2(w: complex) -> complex:
        return cmath.exp(p2 * cmath.log(e(w))) / (2j * k(w))

    def d_theta2(w: complex) -> complex:
        return theta2(w) * (-1 / w + p2 * de(w) / e(w))

    y1_cf = surface_coefficient(built, np.where(lf, m ** (-p1), 0.0))
    out = [
        terms.Piece(
            f"absorbing {lr.region}",
            theta1,
            lambda u, v: y1_cf * u * v * ds1,
            kind="trace",
            coefficient=y1_cf,
            frequency_class="nonpolynomial",
            d_theta=d_theta1,
            probe=terms._LAW_PROBE,  # noqa: SLF001
        )
    ]
    if ds2 is not None and on.any():
        y2_cf = surface_p0(built, np.where(on, m ** (-1.5 if ez else -0.5), 0.0))
        out.append(
            terms.Piece(
                f"absorbing order 2 {lr.region}",
                theta2,
                lambda u, v: y2_cf * u.Trace().Deriv() * v.Trace().Deriv() * ds2,
                kind="tangential gradient",
                coefficient=y2_cf,
                frequency_class="nonpolynomial",
                d_theta=d_theta2,
                probe=terms._LAW_PROBE,  # noqa: SLF001
            )
        )
    return out
