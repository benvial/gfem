"""Mesh refinement: error estimates of a solved result, and the refinement loop.

`refine_mesh` solves, estimates the error, marks the tetrahedra holding most
of it (Dörfler marking), bisects them and solves again, until the error
estimate meets a tolerance or a cap stops it. `error_estimate`
gives one estimate of a result without refining (a measurement).
Estimators are value objects named by what they estimate: the
`FieldEstimator` estimates the field's error in the energy norm by flux
recovery; the `QuantityEstimator` names one quantity of interest.

`refine_mesh` logs one INFO record per refinement cycle on the logger
``gfem.mesh_refinement`` (silent by default), once the cycle's outcome is
settled: before the refined mesh is solved, or where the loop stops on that
cycle. The record's attributes are the cycle's `RefinementCycle` numbers
(``tetrahedra``, ``dofs``, ``estimate``, ``value``, ``marked``; not
``modes``), with ``cycle`` (0 is the input mesh), ``tolerance`` and
``stop`` (the `MeshRefinement.stop` reason on the cycle the loop stops on,
else ``None``). A mesh whose solve failed was never solved, so it has no
record; the not-converged warning names that stop.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import types
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem import _progress
from gfem._dim2 import refuse_entry
from gfem._dual_supply import (
    DualNeed,
    DualSupply,
    RiddenDuals,
)
from gfem._pytree import readonly
from gfem._refine import Refinement, bisect, transfer, transfer_design
from gfem._text import hertz
from gfem.errors import (
    DegenerateModeError,
    EigenmodeConvergenceError,
    LinearSolverConvergenceError,
    MeshQualityWarning,
    MeshRefinementNotConvergedWarning,
)
from gfem.mesh import Mesh
from gfem.model import (
    Model,
)
from gfem.options import SolverOptions
from gfem.problems import Driven, Eigenmode, Electrostatic
from gfem.results import EigenmodeResult, ElectrostaticResult
from gfem.shape import _check_quality
from gfem.solve import _solve

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gfem._engine.estimate import RecoveryCache
    from gfem._engine.gauge import GaugeChoice

__all__ = [
    "ErrorEstimate",
    "FieldEstimator",
    "MeshRefinement",
    "QuantityEstimator",
    "RefinementCycle",
    "error_estimate",
    "refine_mesh",
]

#: One INFO record per refinement cycle, once its outcome is settled (see
#: `_cycle_settled`). Silent by default: Python's root level is WARNING.
LOGGER = logging.getLogger("gfem.mesh_refinement")


@dataclasses.dataclass(frozen=True)
class FieldEstimator:
    r"""Estimates the error of a result's fields in the energy norm.

    The estimate is relative to the field's energy norm: 1e-2 means the
    field is about 1 % off its value on the same geometry with an infinitely
    fine mesh. Each tetrahedron's indicator is the distance between the
    solved electric flux and a smoother flux recovered from it (flux
    recovery), so no extra solve is needed. It is an estimate, not a bound;
    it does not include the faceting of curved surfaces or the error left by
    the linear solve.

    With loss, the flux is the total current
    :math:`\varepsilon_\mathrm{eff} E`, :math:`\varepsilon_\mathrm{eff} =
    \varepsilon_r (1 - i \tan\delta) - i \sigma / (\omega \varepsilon_0)`,
    and the norm is :math:`\|\nabla \times E\|^2_{\mu^{-1}} + |k|^2 \int
    |\varepsilon_\mathrm{eff}| |E|^2`: the conduction current counts, as it
    does in the field. Weighting the flux error by
    :math:`\mathrm{Re}(\varepsilon_\mathrm{eff})^{-1}` instead overestimates;
    recovering :math:`\varepsilon_r E`, as
    without loss, underestimates without bound under conduction loss.

    A dispersive material (a `gfem.PermittivityLaw`) enters at its permittivity
    at each solved frequency, lossless or not,
    :math:`\mathrm{Re}\,\varepsilon < 0` included. Where it is exactly zero (a
    lossless Drude term at its plasma frequency, without conductivity) the flux
    has no weight, and `refine_mesh` and `error_estimate` raise `ValueError`.

    `refine_mesh` and `error_estimate` raise
    `gfem.MeshRefinementUnsupportedError` at order 1.
    """


@dataclasses.dataclass(frozen=True)
class QuantityEstimator:
    r"""Estimates the error of one quantity of a solved result.

    Estimates how far one quantity of a solved result (a capacitance entry,
    an S entry whose source port is excited, one mode's ``omega`` or its
    Q, the mean ``omega`` of a degenerate cluster, or an objective of the
    result) is from its value on the same geometry with an infinitely fine
    mesh. `ErrorEstimate.kind` names which, and each kind has its own
    measured effectivity range (verification page).
    The estimate uses the fields already solved, with no extra solve, except
    for the adjoint duals below. It does not include
    the faceting of curved surfaces or the error left by the linear solve.

    The estimate is a magnitude in the units of ``tolerance``: relative to
    the capacitance entry, to ``omega`` or to Q, absolute for an S entry. It has
    no sign or phase, and gfem never adds it to the quantity. It is an
    estimate, not a bound. A diagonal capacitance entry never lies below
    its value on the meshed geometry: :math:`C_h \ge C`; an off-diagonal entry
    may lie on either side.

    Where it is pessimistic (an objective at its target, a reflection near a
    reflection zero, several modes at their target) refinement stops late,
    never early. A wave port's own propagation constant is solved on the
    port's 2D mesh and its error is not estimated; it enters S through
    de-embedding offsets, so keep offsets shorter than the leads you model.
    Below a guide's cutoff a mesh coarser than the field's decay length
    underestimates both S and the field. An S entry whose read port is a wave
    port with a hybrid mode in a lossy cross-section, and an objective of a
    field measurement, an electrostatic energy, a potential mean or a
    participation, also need an adjoint dual: outside `refine_mesh`, and on
    the first mesh of an objective, `error_estimate` factorizes the system
    once more with the direct solver, which on a large model may not fit in
    memory (``max_cycles=0`` estimates through `refine_mesh` instead). A
    participation's dual needs an isolated mode, the direct solver on the CPU
    and ``eigen_tolerance`` at most 1e-10.

    Give exactly one keyword, named after the result attribute that holds
    the quantity and indexed the same way.

    Example:
        Refine for S21, then read its estimated error::

            est = gfem.QuantityEstimator(s=("p2", "p1"))
            ref = gfem.refine_mesh(model, gfem.Driven([5e9]), estimator=est)
            ref.result.s["p2", "p1"], ref.estimate.values

    Attributes:
        capacitance: Two terminal names ``(i, j)``, the entry
            ``ElectrostaticResult.capacitance[i, j]``; the same name twice
            for a diagonal entry.
        s: ``(to, from)`` port names, the entry ``result.s[to, from]``; the
            source port ``from`` must be excited (`Driven.excite`), ``to``
            may be any port.
        omega: An index into ``EigenmodeResult.omega`` on the first mesh;
            on each refined mesh the estimate follows the mode whose omega
            is nearest the previous mesh's, and `refine_mesh` raises
            `gfem.DegenerateModeError` when that mode is not clearly the
            same one. Or a tuple of indices naming whole degenerate clusters
            (``result.cluster``), stored sorted, whose mean omega is the
            quantity, estimated relative to its modulus: on each refined
            mesh the estimate follows as many modes, those nearest the
            previous mean, and raises `gfem.DegenerateModeError` when the
            mean moved further than the previous estimate allows, or another
            mode is less than twice as far from it as the farthest of them.
            Ask for at least one mode more than the tuple names.
        q: An index into ``EigenmodeResult.q``, followed as ``omega``; the
            estimate is relative to Q. Q's relative error is :math:`\cot(\arg\omega)`
            times the imaginary part of omega's relative error, estimated on
            the same terms, with no extra solve. A lossless mode (infinite Q)
            raises `ValueError`.
        objective: ``objective(result, model)``, a real 0-d JAX value: the
            same function you differentiate. gfem traces it to find what it
            reads (``ErrorEstimate.kind``) and raises
            `gfem.MeshRefinementUnsupportedError` for anything it does not
            estimate (``feature`` names it): a field read other than through
            `gfem.power_flow` or `gfem.region_mean`, a point value, a port
            mode output, a mixture of kinds, a branch on a value it reads
            (max, min, sort, where) or a custom derivative of its own. In
            this version an objective of S is estimated (S and its networks:
            ``db``, ``y()``, ``z()``, ``renormalize``, ``with_offsets``),
            and of an electrostatic result: of its capacitance (``mutual``
            and ``inverse`` too), of its region energies, or of potential
            means (`gfem.region_mean` of ``result.potential``), and of a
            driven result's fields: the power absorbed in a side without
            ports and with only volume losses of isotropic materials without
            a permittivity law (``-gfem.power_flow``), or the means of
            `gfem.region_mean` over named regions; and an objective of an
            eigenmode result's ``omega`` (``frequency``, ``q``) and
            ``bulk_participation``, each mode it reads followed as a mode of
            its own; ``q_port``, ``participation``, ``sheet_participation``,
            ``principal_participation`` and the mode fields raise. A
            participation needs its mode-field dual: an isolated mode, the
            direct solver on the CPU and ``eigen_tolerance`` at most 1e-10
            (in `refine_mesh` it rides on the eigen solves from the second
            mesh on; `error_estimate` factorizes once more per mode). The estimate
            is absolute, in the objective's units, and `refine_mesh` needs
            its ``tolerance`` given. A read row whose port is not excited
            uses that port's adjoint dual, as for
            ``s``, and a read energy, potential mean, absorbed power or
            field mean its own dual; in
            `refine_mesh` each is solved once more on the first mesh and
            rides on the solves after it.

    Raises:
        TypeError: Not exactly one keyword is given, or a value has the
            wrong type (a tuple for ``q``: the Q of a cluster is not
            estimated; an ``objective`` that is not callable).
        ValueError: ``omega`` or ``q`` is negative, or a tuple ``omega``
            names fewer than two modes or a mode twice.

    `refine_mesh` and `error_estimate` raise
    `gfem.MeshRefinementUnsupportedError` for entries with no verified
    estimate. An S entry and every objective of a driven result raise it
    before any solve on a model with an `gfem.Impedance` or `gfem.Absorbing`
    surface (``feature`` ``"impedance or absorbing surface"``), with a
    lumped port whose face borders materials of different ``mu_r``
    (``"lumped port spanning a permeability contrast"``), or with a
    permittivity law at :math:`-20 < \mathrm{Re}\,\varepsilon < 0` at a solve
    frequency (``"plasmonic permittivity"``); refine those with
    `FieldEstimator`. The ``omega`` of a mode bound to a permittivity law
    region where :math:`\mathrm{Re}\,\varepsilon < 0` at the mode raises it
    when it is estimated. Both estimators also raise it at order 1. For a lossy
    mode the ``omega`` estimate covers the complex ``omega``; ``q`` estimates
    Q, and raises on a
    mode of a model with a permittivity law (``feature`` ``"dispersive
    eigenmode"``), a second-order absorbing boundary, a complex ``mu_r`` or
    loss inside an anisotropic material (``"lossy eigenmode"``). Every
    quantity of a mode of a model with a `gfem.WavePort` (``omega``, ``q`` and
    objectives) raises it before any solve (``"wave port eigenmode"``): refine
    such a mode with `FieldEstimator`. An S entry
    ``(to, from)`` whose port ``to`` is not excited needs that port's adjoint
    dual: `refine_mesh` adds it to each of its solves, one more right-hand
    side with its options; `error_estimate` on a finished result factorizes
    once more per frequency, with the direct solver at the result's order,
    so estimate a large model through `refine_mesh` (``max_cycles=0``).
    """

    capacitance: tuple[str, str] | None = None
    s: tuple[str, str] | None = None
    omega: int | tuple[int, ...] | None = None
    q: int | None = None
    objective: Callable[[Any, Model], Any] | None = None

    def __post_init__(self) -> None:
        """Check that exactly one quantity is named, with valid keys."""
        given = [
            name
            for name in ("capacitance", "s", "omega", "q", "objective")
            if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise TypeError(
                "QuantityEstimator takes exactly one of capacitance=(t1, t2), "
                "s=(to, from), omega=index, q=index or objective=J; got "
                f"{given or 'none'}."
            )
        if self.objective is not None and not callable(self.objective):
            raise TypeError(
                "QuantityEstimator(objective=...) takes a function "
                f"objective(result, model), got {self.objective!r}."
            )
        for name in ("capacitance", "s"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, str)
                or len(tuple(value)) != 2
                or not all(isinstance(k, str) for k in value)
            ):
                raise TypeError(
                    f"QuantityEstimator({name}=...) takes two names, got {value!r}."
                )
            object.__setattr__(self, name, tuple(value))
        for name in ("omega", "q"):
            value = getattr(self, name)
            if value is None:
                continue
            if name == "q" and isinstance(value, (tuple, list)):
                raise TypeError(
                    f"QuantityEstimator(q=...) takes one mode index, got {value!r}: "
                    "the Q of a cluster is not estimated."
                )
            if isinstance(value, (tuple, list)):
                object.__setattr__(self, name, _cluster_members(value))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"QuantityEstimator({name}=...) takes a mode index, got {value!r}."
                )
            if value < 0:
                raise ValueError(
                    f"QuantityEstimator({name}=...) takes an index >= 0, got {value}."
                )
            object.__setattr__(self, name, int(value))

    @property
    def _mode_keyword(self) -> str | None:
        """``"omega"`` or ``"q"`` when the quantity is of one mode, else ``None``."""
        if self.omega is not None:
            return "omega"
        return "q" if self.q is not None else None


def _cluster_members(value: tuple | list) -> tuple[int, ...]:
    """The sorted modes of ``omega=(i, j, ...)``: two or more, each once."""
    for each in value:
        if isinstance(each, bool) or not isinstance(each, (int, np.integer)):
            raise TypeError(
                "QuantityEstimator(omega=(...)) takes mode indices, got "
                f"{each!r} in {value!r}."
            )
        if each < 0:
            raise ValueError(
                f"QuantityEstimator(omega=(...)) takes indices >= 0, got {each} "
                f"in {value!r}."
            )
    members = tuple(sorted(int(each) for each in value))
    if len(set(members)) != len(members):
        raise ValueError(
            f"QuantityEstimator(omega={tuple(value)!r}) names a mode twice: name "
            "each mode of the cluster once."
        )
    if len(members) < 2:
        raise ValueError(
            f"QuantityEstimator(omega={tuple(value)!r}) names one mode: a cluster "
            "mean needs at least two modes; for one mode, give omega=i."
        )
    return members


Estimator = FieldEstimator | QuantityEstimator

#: What an `ErrorEstimate` can be of (its ``kind``): a closed set.
#: ``"field"`` is the `FieldEstimator`'s; ``"capacitance"``, ``"s"`` and
#: ``"omega"`` a `QuantityEstimator`'s entries; ``"q"`` and ``"cluster mean"``
#: a mode's Q and a degenerate cluster's mean omega; every ``"objective of
#: ..."`` an objective's, by what it reads. Each kind has its own verified
#: effectivity range (``EFFECTIVITY_BY_KIND`` in the estimate tests), which
#: the docs generate per kind. Private: not in ``gfem.__all__``.
ESTIMATE_KINDS = (
    "field",
    "capacitance",
    "s",
    "omega",
    "q",
    "cluster mean",
    "objective of s",
    "objective of s, lossless rewrite",
    "objective of omega",
    "objective of capacitance",
    "objective of electrostatic energy",
    "objective of potential means",
    "objective of absorbed power",
    "objective of field means",
    "objective of participation",
)

#: The kinds the estimators emit, in `ESTIMATE_KINDS`'s order. A kind joined
#: when its estimate landed; every kind has now, and the docs build requires
#: the two to be equal, each kind with its verification cases.
EMITTED_KINDS = ESTIMATE_KINDS


@dataclasses.dataclass(frozen=True, eq=False)
class ErrorEstimate:
    """An error estimator's estimate for one solved result.

    Attributes:
        estimator: The estimator that made it.
        kind: What was estimated: ``"field"`` for a `FieldEstimator`;
            ``"capacitance"``, ``"s"``, ``"omega"`` or ``"q"`` for a
            `QuantityEstimator`, after its keyword, ``"cluster mean"``
            for a tuple ``omega``; for an objective, what it reads
            (``"objective of s"``), or ``"objective of s, lossless
            rewrite"`` on a lossless model with lumped ports only, whose
            estimate also uses the power each port's column must carry
            (tighter). Each kind has its own measured
            effectivity range (verification page).
        values: Estimated size of the error per frequency (a driven result),
            per mode (an eigenmode result under a `FieldEstimator`), or one
            entry, in the units of ``tolerance``. For a `FieldEstimator`
            this is relative to the field's energy norm. For a
            `QuantityEstimator` this is the quantity's error: relative to
            the capacitance entry, to ``omega``, to a cluster's mean
            ``omega`` or to Q, absolute for an S entry.
            It is never negative and carries no sign or phase. It is an
            estimate, not a bound.
        indicators: Each tetrahedron's share of the estimated error, shape
            ``(n_tetrahedra,)``, in the order of the model's
            ``mesh.tetrahedra``, combined over frequencies, terminals, ports
            and modes; zero outside the solve domain. For a single solve
            ``sqrt(sum(indicators**2))`` is ``value`` for a
            `FieldEstimator`. For a `QuantityEstimator`, ``indicators**2``
            is each tetrahedron's term of the quantity's error, in its
            units; ``sum(indicators**2)`` is ``value`` for a capacitance,
            and may lie above or below it for an S entry or ``omega``,
            whose ``value`` keeps the curl and mass terms from cancelling.
    """

    estimator: Estimator
    kind: str
    values: tuple[float, ...]
    indicators: np.ndarray
    #: An objective's value on the result it estimated (`RefinementCycle.value`).
    _quantity: float | None = dataclasses.field(default=None, repr=False)
    #: The solved duals an objective's estimate reads: the ports whose
    #: S-functional adjoints, the keys of its field or electrostatic duals,
    #: the (mode, region) participations whose mode-field duals.
    _duals: DualNeed = dataclasses.field(default=DualNeed(), repr=False)
    #: An objective's first-order sensitivity, sum_a |ct_a| s_a (its floor).
    _sensitivity: float = dataclasses.field(default=0.0, repr=False)
    #: The modes an objective reads, by index on its result: each is followed.
    _modes: tuple[int, ...] = dataclasses.field(default=(), repr=False)
    #: A lossless rewrite's computed defect term lam . D, signed:
    #: ``value - abs(_defect)`` is the rewritten objective's own estimate.
    _defect: float = dataclasses.field(default=0.0, repr=False)

    @property
    def value(self) -> float:
        """The largest of `values`: what `refine_mesh` compares to ``tolerance``.

        NaN when any of `values` is NaN.
        """
        values = np.asarray(self.values, float)
        return float(np.nan) if np.isnan(values).any() else float(values.max())


@dataclasses.dataclass(frozen=True)
class RefinementCycle:
    """The numbers of one solved mesh of a `MeshRefinement`.

    With a tuple of estimators, ``estimate`` and ``value`` are tuples, one
    entry per estimator in the given order.

    Attributes:
        tetrahedra: The mesh's tetrahedron count.
        dofs: The field's coefficient count (one solve's unknowns).
        estimate: The error estimate's ``value`` on this mesh.
        value: The quantity of a `QuantityEstimator` on this mesh: the
            capacitance entry in farads, a tuple of the S entry per
            frequency, the followed mode's complex ``omega`` in rad/s, its
            Q, the followed cluster's mean complex ``omega``, or an
            objective's value; ``None`` for a `FieldEstimator`.
        marked: How many tetrahedra Dörfler marking picked on this mesh; 0
            when the loop stopped on it before marking.
        modes: For an eigenmode quantity, each mode index the estimators
            name on the first mesh, mapped to the index of the mode followed
            to this mesh (read-only); empty otherwise. A cluster's members,
            in order, map to the modes followed as the cluster, in order.
    """

    tetrahedra: int
    dofs: int
    estimate: float | tuple[float, ...]
    value: Any
    marked: int
    modes: Mapping[int, int] = dataclasses.field(
        default_factory=lambda: types.MappingProxyType({}), hash=False
    )


@dataclasses.dataclass(frozen=True, eq=False)
class MeshRefinement:
    """What `refine_mesh` returns: the last solved model, its result and history.

    Not a JAX pytree and not differentiable: differentiate `gfem.solve` on
    ``model`` instead.

    Attributes:
        model: The last solved model (its mesh refined, its design carried).
        result: That model's solve result.
        estimate: That result's `ErrorEstimate`; with a tuple of estimators,
            a tuple of them in the estimators' order.
        cycles: One `RefinementCycle` per solved mesh; ``cycles[0]`` is the
            input mesh.
        stop: Why the loop stopped: ``"tolerance"``, ``"max_cycles"``,
            ``"max_tetrahedra"``, ``"solve failed"`` or ``"non-finite
            estimate"`` (an estimate is NaN or infinite: it counts as above
            its tolerance, and the loop warns).
        converged: Whether ``estimate.value`` is at or below the tolerance;
            with a tuple of estimators, whether every one is at or below its
            own.
        tetrahedron_parents: For each tetrahedron of ``model.mesh``, the row
            of the input mesh's ``tetrahedra`` it lies in.
        triangle_parents: For each triangle of ``model.mesh``, the row of the
            input mesh's ``triangles`` it lies in.
    """

    model: Model
    result: Any
    estimate: ErrorEstimate | tuple[ErrorEstimate, ...]
    cycles: tuple[RefinementCycle, ...]
    stop: str
    converged: bool
    tetrahedron_parents: np.ndarray
    triangle_parents: np.ndarray
    _input_mesh: Mesh = dataclasses.field(repr=False)

    def transfer(self, values: Any, region: str) -> Any:
        """Carry per-element values of a region from the input mesh to ``model``.

        Each element of the refined mesh takes the value of the input element
        it lies in. Plain indexing: a JAX array stays one, and `jax.grad`
        passes through it.

        Args:
            values: Shape ``(n,)`` or ``(n, k)``, one value per element of
                ``region`` on the input mesh, in ``mesh.elements(region)``
                order.
            region: A volume (tetrahedra) or surface (triangles) region name.

        Returns:
            One value per element of ``region`` on ``model.mesh``, in
            ``model.mesh.elements(region)`` order.

        Raises:
            RegionNotFoundError: ``region`` is not a region of the mesh.
            ValueError: ``values`` does not hold one value per element.
        """
        return transfer(
            values,
            region,
            coarse=self._input_mesh,
            fine=self.model.mesh,
            refinement=Refinement(self.tetrahedron_parents, self.triangle_parents),
        )


#: The lowest ratio of an estimate to the error over the verified cases; mode
#: following bounds a mode's shift between two meshes with it.
_ESTIMATE_FLOOR = 0.9


def error_estimate(result: Any, estimator: Estimator, *, model: Model) -> ErrorEstimate:
    r"""Estimate the error of a solved result, without refining.

    No solve is needed, except for an S entry, or an objective reading a
    row of S, whose port ``to`` is not excited: its adjoint dual factorizes
    the system once more per frequency, with the direct solver at the
    result's order (the result does not record its solve options). An
    objective reading an electrostatic energy or a potential mean
    factorizes once more the same way, for all its duals at once, and one
    reading a driven field through `gfem.power_flow` or `gfem.region_mean`
    once more per frequency it reads. On a
    large model, estimate through `refine_mesh` with ``max_cycles=0``,
    whose adjoint rides on its own solve with your options (from its second
    mesh on, for an objective).

    Example:
        The field error of a capacitance solve::

            result = gfem.solve(model, gfem.Electrostatic(terminals=["t1"]))
            est = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
            est.value, est.indicators  # relative energy error, per tetrahedron

    Args:
        result: A result of `gfem.solve` on ``model``: electrostatic,
            eigenmode (every mode is estimated), or a uniform driven sweep
            solved with ``keep_fields=True`` (every kept field).
        estimator: `FieldEstimator()`, or a `QuantityEstimator` naming an
            entry of ``result`` (its ``omega`` index is a mode of
            ``result``).
        model: The model the result was solved on (its materials weight the
            fields).

    Returns:
        The estimate, with NumPy values.

    Raises:
        TypeError: An argument has the wrong type (a tuple of estimators
            too: call it once per quantity), or a `QuantityEstimator` names a
            quantity of another result type.
        KeyError: A `QuantityEstimator` names no terminal or port of the
            result.
        ValueError: The result was not solved on ``model``'s mesh, or is a
            driven result without the fields of every frequency (solve with
            ``keep_fields=True``); a `QuantityEstimator` names a source port
            that was not excited, a mode the result does not hold, the Q of a
            lossless mode, or a cluster with no other mode in the result; a
            permittivity law is zero at a driven frequency.
        MeshRefinementUnsupportedError: The estimator or the result has no
            estimate in this version of gfem (``feature`` names what): an
            adaptive sweep, a periodic model, a driven result with sources
            (``feature`` ``"incident field"`` or ``"point dipole"``), a driven
            or eigenmode result on a model with a `gfem.PML`, another result
            type, a result
            solved at order 1, a quantity entry with no verified estimate (an S
            entry read at a lumped port with no resistance, whose row is
            undefined; an S entry or a driven objective beside an impedance or
            absorbing surface, a lumped port across a permeability contrast or
            a permittivity law at :math:`-20 < \mathrm{Re}\,\varepsilon < 0`; a
            plasmon-bound mode; see `QuantityEstimator`), or an objective
            reading what it may not.
        DegenerateModeError: A `QuantityEstimator` names a mode of a
            degenerate cluster, or one within the default
            `SolverOptions.cluster_rtol` of another mode (the message names
            the cluster); a cluster ``omega=(i, ...)`` holds part of a
            degenerate cluster, or another mode lies less than twice as far
            from its mean as its farthest member.
        NotImplementedError: ``model`` holds a metal-dielectric blend
            (`MaterialField.interpolate` with a permittivity law), which is
            supported on the driven uniform sweep only.
        DistributedUnsupportedError: The result was solved across ranks.
        NonDifferentiableError: A model or result leaf is a JAX tracer, or
            the call is inside `jax.jit`.
    """
    if not isinstance(model, Model):
        raise TypeError(f"model must be a gfem.Model, got {type(model).__name__}.")
    refuse_entry(model.mesh, "error_estimate")
    if isinstance(estimator, (tuple, list)):
        raise TypeError(
            "error_estimate takes one estimator, got a "
            f"{type(estimator).__name__}: call it once per quantity."
        )
    from gfem._estimate_admission import admit_estimate

    admit_estimate(result, estimator, model)
    mode: int | tuple[int, ...] | None = _estimator_mode(estimator)
    if mode is not None:
        # `refine_mesh`'s rule, at the default tolerance: the result does not
        # record the options it was solved with
        _track_mode(result, mode, None, SolverOptions().cluster_rtol, None)
    elif (start := _track_start(estimator)) is not None:
        mode = _track_cluster(result, start, None, None)
    from gfem._estimate_kinds import estimate

    return estimate(model, result, estimator, threads=None, mode=mode)


def refine_mesh(
    model: Model,
    problem: Electrostatic | Driven | Eigenmode,
    *,
    estimator: Estimator | tuple[QuantityEstimator, ...] = FieldEstimator(),  # noqa: B008 - frozen value object
    tolerance: float | tuple[float, ...] | None = None,
    error_fraction: float = 0.5,
    max_cycles: int = 10,
    max_tetrahedra: int | None = None,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> MeshRefinement:
    r"""Refine the mesh where the error is, until the error estimate meets a tolerance.

    Each refinement cycle solves the problem, estimates the error, stops if
    the estimate is at or below ``tolerance``, marks the fewest tetrahedra
    holding ``error_fraction`` of the squared error (Dörfler marking),
    bisects them and their neighbours until the mesh conforms (new nodes at
    edge midpoints, physical groups and imported facets kept), and carries
    the model's per-element design (`MaterialField` values, sheet fills) to
    the children. Cycle 0 solves the input mesh.

    With a tuple of estimators, several quantities refine one mesh: every
    one is estimated on every mesh, the marking sums each quantity's
    ``indicators**2`` divided by its tolerance over the quantities still
    above it, and the loop stops when every one meets its tolerance. Modes
    named by several estimators are followed once each.

    A driven problem is solved with ``keep_fields=True`` whatever it asks:
    the estimate reads every frequency's field of every excited port, and
    ``result`` keeps them. Its low-frequency floor is checked once, on the
    returned mesh, not on every cycle.

    Example:
        Refine a capacitance solve to a 1 % field error::

            ref = gfem.refine_mesh(model, gfem.Electrostatic(terminals=["t1"]))
            ref.model, ref.result, ref.estimate.value, ref.stop

        Refine for S21 and S11 at once, each to its own tolerance::

            estimators = (
                gfem.QuantityEstimator(s=("p2", "p1")),
                gfem.QuantityEstimator(s=("p1", "p1")),
            )
            ref = gfem.refine_mesh(
                model, gfem.Driven([5e9]), estimator=estimators, tolerance=(1e-4, 1e-3)
            )
            s21_estimate, s11_estimate = ref.estimate

    Args:
        model: The device, with concrete values.
        problem: An `Electrostatic` problem, a `Driven` problem on a uniform
            sweep (lumped or wave ports), or an `Eigenmode` problem. The
            estimate of a driven run is the worst frequency's, of an
            eigenmode run the worst mode's.
        estimator: `FieldEstimator()`, a `QuantityEstimator` naming a
            quantity of ``problem``, or a tuple of distinct
            `QuantityEstimator` objects; the record then holds tuples, one
            entry per estimator in this order.
        tolerance: The error estimate to reach, one float for every
            estimator or a tuple with one per estimator; ``None`` is 1e-2,
            except for an objective, whose tolerance is in its own units and
            must be given (at least 10 x the solve tolerance times the
            objective's first-order sensitivity, checked on the first
            estimate).
            Relative to the field's energy
            norm for a `FieldEstimator`; for a `QuantityEstimator` relative
            to the capacitance entry, to ``omega``, to a cluster's mean
            ``omega`` or to Q, absolute for an S entry. At least 10 times
            ``options.tolerance`` (``options.eigen_tolerance`` for an
            `Eigenmode`; for Q times :math:`\cot(\arg\omega)`, about 2Q, checked on
            the first solve), since the estimate does not include the linear
            solve's error. Each tolerance is checked on its own.
        error_fraction: Dörfler's fraction, in (0, 1]: the share of the summed
            squared error indicators the marked tetrahedra hold.
        max_cycles: How many times the mesh may be refined; 0 solves and
            estimates only.
        max_tetrahedra: A cap on the refined mesh's tetrahedra: a bisected
            mesh above it is discarded and the loop stops on the last solved
            one. ``None`` sets no cap; each cycle grows the mesh, by about
            1.6 times on the first cycle and 1.25 times on later ones.
        options: The numerics of every solve; ``None`` is ``SolverOptions()``.
        comm: As for `gfem.solve`; mesh refinement runs on one process.

    Returns:
        The last solved model, its result and estimate (a tuple with a tuple
        of estimators), and one `RefinementCycle` per solved mesh.

    Warns:
        MeshRefinementNotConvergedWarning: The loop stopped on a cap or a
            failed solve with an estimate above its tolerance; it names each
            such quantity.
        LowFrequencyWarning: A frequency of a driven problem is below the
            returned mesh's low-frequency floor.
        MeshQualityWarning: The input mesh's nodes have moved and an element
            fell far below its imported quality.

    Raises:
        TypeError: An argument has the wrong type, a `QuantityEstimator`
            names a quantity of another problem type, or a tuple of
            estimators holds a `FieldEstimator`.
        KeyError: A `QuantityEstimator` names no terminal or port of the
            problem.
        ValueError: A parameter is out of range; a tuple of estimators is
            empty or names a quantity twice, or the tolerances do not match
            the estimators one to one, or a cluster shares a mode with
            another estimator; a tolerance is below 10
            times the solve tolerance (for Q, times :math:`\cot(\arg\omega)`); a
            `QuantityEstimator` names a source port that is not excited, a
            mode index at or above ``n_modes``, a cluster of at least
            ``n_modes`` modes (one more is needed to tell it apart), or the Q
            of a lossless mode; a permittivity law is zero at a driven
            frequency.
        MeshRefinementUnsupportedError: The problem or the estimator has no
            mesh refinement (``feature`` names what): an adaptive sweep, a port
            mode problem, a periodic model or a Bloch wave vector, a driven
            problem with sources (``feature`` ``"incident field"`` or
            ``"point dipole"``), a driven or eigenmode problem on a model with
            a `gfem.PML` (before any solve),
            ``options.order=1`` (the ``"fast"`` preset), a quantity entry with
            no verified estimate (an S entry read at a lumped port with no
            resistance, whose row is undefined; an S entry or a driven
            objective beside an impedance or absorbing surface, a lumped port
            across a permeability contrast or a permittivity law at
            :math:`-20 < \mathrm{Re}\,\varepsilon < 0`, all before any solve; a
            plasmon-bound mode on the first estimate, after the first solve;
            see `QuantityEstimator`), an objective reading what it may not (on
            the first estimate), or ``frequency_workers`` above 1 for an
            estimate that solves an adjoint dual (an unexcited ``to`` or an
            objective's dual), which rides on the
            forward solve.
        DegenerateModeError: The mode a `QuantityEstimator` follows is not
            alone: another mode lies within ``options.cluster_rtol`` of it,
            or shares its ``result.cluster`` label (the message names the
            cluster, to estimate its mean with ``omega=(i, j, ...)``).
            Or, on a refined mesh, the nearest mode is not clearly the
            followed one: a second mode is less than twice as far from the
            previous frequency, or the nearest moved further than the
            previous estimate allows (the mode probably left the solved
            modes: ask for more modes or start from a finer mesh). A
            cluster ``omega=(i, ...)`` raises it when it holds part of a
            labelled cluster, when another mode lies less than twice as far
            from its mean (on the first mesh) or from its previous mean as
            its farthest member, or when its mean moved further than the
            previous estimate allows. Or two modes named on the first mesh
            reached one mode.
        NotImplementedError: ``model`` holds a metal-dielectric blend
            (`MaterialField.interpolate` with a permittivity law), which is
            supported on the driven uniform sweep only.
        DistributedUnsupportedError: Launched on more than one rank.
        MeshQualityError: The input mesh's nodes have moved and an element is
            inverted.
        NonDifferentiableError: A model or problem leaf is a JAX tracer, or
            the call is inside `jax.jit`. Under `jax.grad` or `jax.vmap`
            with concrete inputs it runs; refine outside the transformed
            function, then differentiate `gfem.solve` on the refined model.
        LinearSolverConvergenceError, EigenmodeConvergenceError: The solve
            on the input mesh failed. On a later cycle the loop stops instead
            (``stop="solve failed"``).
    """
    if not isinstance(model, Model):
        raise TypeError(f"refine_mesh needs a gfem.Model, got {type(model).__name__}.")
    refuse_entry(model.mesh, "refine_mesh")
    if options is None:
        options = SolverOptions()
    elif not isinstance(options, SolverOptions):
        raise TypeError(
            f"options must be a gfem.SolverOptions, got {type(options).__name__}."
        )
    from gfem._estimate_admission import admit_refinement

    estimators, tolerances = admit_refinement(
        model,
        problem,
        estimator,
        tolerance,
        options,
        comm=comm,
        error_fraction=error_fraction,
        max_cycles=max_cycles,
        max_tetrahedra=max_tetrahedra,
    )
    # a moved input mesh is checked here, so a warning names the caller's line;
    # the loop's solves and bisections skip the check
    _check_quality(model.mesh, model._domain.active, stacklevel=2)  # noqa: SLF001
    _free_consumed(options)
    return _loop(
        model,
        problem,
        estimators=estimators,
        tolerances=tuple(float(t) for t in tolerances),
        joint=isinstance(estimator, tuple),
        error_fraction=float(error_fraction),
        max_cycles=int(max_cycles),
        max_tetrahedra=max_tetrahedra,
        options=options,
        comm=comm,
    )


def _describe(estimator: Estimator) -> str:
    """``QuantityEstimator(s=('p2', 'p1'))``: an estimator by its one keyword."""
    if isinstance(estimator, FieldEstimator):
        return "FieldEstimator()"
    if estimator.objective is not None:
        name = getattr(
            estimator.objective, "__name__", type(estimator.objective).__name__
        )
        return f"QuantityEstimator(objective={name})"
    keyword = next(
        name
        for name in ("capacitance", "s", "omega", "q")
        if getattr(estimator, name) is not None
    )
    return f"QuantityEstimator({keyword}={getattr(estimator, keyword)!r})"


def _deferred_solve(
    model: Model,
    problem: Any,
    *,
    options: SolverOptions,
    comm: Any,
    adjoint_loads: Any,
) -> tuple[Any, dict[Any, Any] | None, GaugeChoice | None]:
    """`_solve` with the driven low-frequency check deferred.

    Also returns the gauge choice of the solve's driven rule (``None`` for
    another problem): each pass resolves it afresh, and the deferred check
    on the returned mesh reads the last pass's.
    """
    from gfem._engine import driven

    with driven.low_frequency_deferred() as deferred:
        result, adjoints = _solve(
            model, problem, options=options, comm=comm, adjoint_loads=adjoint_loads
        )
    return result, adjoints, deferred.gauge


def _loop(  # noqa: PLR0915 - one refinement loop, each cycle settled in order
    model: Model,
    problem: Any,
    *,
    estimators: tuple[Estimator, ...],
    tolerances: tuple[float, ...],
    joint: bool,
    error_fraction: float,
    max_cycles: int,
    max_tetrahedra: int | None,
    options: SolverOptions,
    comm: Any,
) -> MeshRefinement:
    """Solve, estimate, stop, mark, bisect, cap, transfer; see `refine_mesh`.

    Every estimator is estimated on every mesh; marking sums the
    tolerance-normalized weights of those still above their tolerance
    (`_marking`), and the loop stops when none is. ``joint``
    (a tuple was given) makes the records tuples. Each solved mesh gets one
    `RecoveryCache`, shared by its estimates and dropped with the mesh.
    """
    from gfem._engine import driven
    from gfem._estimate_admission import admit_first_estimates

    if isinstance(problem, Driven):
        problem = dataclasses.replace(problem, keep_fields=True)
    tet_parents = np.arange(len(model.mesh.tetrahedra))
    tri_parents = np.arange(len(model.mesh.triangles))
    duals = RiddenDuals(estimators, model, problem)
    tracks = _Tracks(estimators)
    # as the cycles report it: the record's and the row's
    reported = _shaped(list(tolerances), joint)
    # opened before the first mesh is solved, whose phases show in it; the
    # number of refinement cycles is not known in advance
    with _progress.task(_progress.MESH_REFINEMENT) as row:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MeshQualityWarning)  # `refine_mesh` warned
            result, adjoints, gauge = _deferred_solve(
                model,
                problem,
                options=options,
                comm=comm,
                adjoint_loads=duals.loads(tracks),
            )
        current = model
        cycles: list[RefinementCycle] = []
        failure: Exception | None = None
        while True:
            cache = _recovery_cache()  # this mesh's, dropped before the next solve
            estimates, modes = _estimates(
                current,
                result,
                estimators,
                tracks,
                options,
                duals=duals.supply(adjoints),
                cache=cache,
                first=not cycles,
            )
            admit_first_estimates(
                problem,
                result,
                (estimators, estimates, tolerances),
                modes,
                options,
                first=not cycles,
            )
            duals.add(estimates)
            above = [
                i
                for i, (estimate, tolerance) in enumerate(
                    zip(estimates, tolerances, strict=True)
                )
                if not estimate.value <= tolerance  # a NaN is above
            ]
            cycle = RefinementCycle(
                tetrahedra=len(current.mesh.tetrahedra),
                dofs=_dofs(result),
                estimate=_shaped([e.value for e in estimates], joint),
                value=_shaped(
                    [
                        _quantity(result, e, _followed(modes, e), estimate)
                        for e, estimate in zip(estimators, estimates, strict=True)
                    ],
                    joint,
                ),
                marked=0,
                modes=types.MappingProxyType(modes),
            )
            cycles.append(cycle)
            stop = _settled(estimates, above) or (
                "max_cycles" if len(cycles) > max_cycles else None
            )
            if stop is not None:
                _cycle_settled(len(cycles) - 1, cycle, reported, stop, row=row)
                break
            marked = _dorfler(_marking(estimates, tolerances, above), error_fraction)
            cycles[-1] = dataclasses.replace(cycle, marked=int(marked.sum()))
            fine, refinement = bisect(current.mesh, marked)
            if max_tetrahedra is not None and len(fine.tetrahedra) > max_tetrahedra:
                stop = "max_tetrahedra"
            # settled: the record precedes the refined mesh's solve records
            _cycle_settled(len(cycles) - 1, cycles[-1], reported, stop, row=row)
            if stop is not None:
                break
            tracks.bound(
                current, result, estimates, threads=options.threads, cache=cache
            )
            refined = transfer_design(current, fine, refinement)
            try:
                refined_result, refined_adjoints, refined_gauge = _deferred_solve(
                    refined,
                    problem,
                    options=options,
                    comm=comm,
                    adjoint_loads=duals.loads(tracks),
                )
            except (LinearSolverConvergenceError, EigenmodeConvergenceError) as err:
                stop, failure = "solve failed", err
                break
            current, result, adjoints, gauge = (
                refined,
                refined_result,
                refined_adjoints,
                refined_gauge,
            )
            tet_parents = tet_parents[refinement.tetrahedron_parents]
            tri_parents = tri_parents[refinement.triangle_parents]
    if isinstance(problem, Driven):
        driven.check_low_frequency(current, problem.frequency, gauge, stacklevel=3)
    if above:
        _warn_not_converged(
            stop,
            failure,
            [(estimators[i], estimates[i].value, tolerances[i]) for i in above],
            joint=joint,
            max_cycles=max_cycles,
            max_tetrahedra=max_tetrahedra,
        )
    return MeshRefinement(
        model=current,
        result=result,
        estimate=_shaped(list(estimates), joint),
        cycles=tuple(cycles),
        stop=stop,
        converged=not above,
        tetrahedron_parents=readonly(tet_parents),
        triangle_parents=readonly(tri_parents),
        _input_mesh=model.mesh,
    )


def _cycle_settled(
    index: int,
    cycle: RefinementCycle,
    tolerance: float | tuple[float, ...],
    stop: str | None,
    *,
    row: _progress.ProgressRow,
) -> None:
    """Refinement cycle ``index``'s outcome is settled: advance ``row``, log it.

    The record on `LOGGER` has the cycle's attributes (``modes`` aside: a
    read-only mapping does not pickle, and a queue handler across processes
    pickles the record), with ``cycle``, ``tolerance`` (a tuple for a tuple of
    estimators) and ``stop`` (the loop's stop reason on this cycle, else
    ``None``).
    """
    row.advance(
        cycle=index,
        tetrahedra=cycle.tetrahedra,
        estimate=cycle.estimate,
        tolerance=tolerance,
    )
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    LOGGER.info(
        "mesh refinement cycle %d: %d tetrahedra, %d unknowns, estimate %s "
        "(tolerance %s), marked %d%s",
        index,
        cycle.tetrahedra,
        cycle.dofs,
        _numbers(cycle.estimate),
        _numbers(tolerance),
        cycle.marked,
        "" if stop is None else f", stopped: {stop}",
        extra={
            "cycle": index,
            "tetrahedra": cycle.tetrahedra,
            "dofs": cycle.dofs,
            "estimate": cycle.estimate,
            "value": cycle.value,
            "marked": cycle.marked,
            "tolerance": tolerance,
            "stop": stop,
        },
    )


def _numbers(x: float | tuple[float, ...]) -> str:
    """``x`` to 3 significant digits; a tuple's entries in parentheses."""
    if isinstance(x, tuple):
        return "(" + ", ".join(f"{v:.3g}" for v in x) + ")"
    return f"{x:.3g}"


def _settled(estimates: tuple[ErrorEstimate, ...], above: list[int]) -> str | None:
    """Why the loop stops before marking: none above, or one not finite."""
    if not above:
        return "tolerance"
    if not all(np.isfinite(estimates[i].value) for i in above):
        return "non-finite estimate"
    return None


def _followed(
    modes: dict[int, int], estimator: Estimator
) -> int | tuple[int, ...] | None:
    """The index on this mesh of the mode ``estimator`` names, else ``None``.

    For a cluster, the indices of the modes followed as the cluster.
    """
    start = _track_start(estimator)
    if start is None:
        return None
    if len(start) > 1:
        return tuple(modes[each] for each in start)
    return modes[start[0]]


def _estimates(
    model: Model,
    result: Any,
    estimators: tuple[Estimator, ...],
    tracks: _Tracks,
    options: SolverOptions,
    *,
    duals: DualSupply,
    cache: RecoveryCache,
    first: bool,
) -> tuple[tuple[ErrorEstimate, ...], dict[int, int]]:
    """Every estimator's estimate of one mesh's ``result``, and the modes followed.

    The modes are ``tracks``' (`_Tracks.follow`), plus, on the ``first``
    mesh, those its objectives read (`_Tracks.add`). An objective sees an
    eigenmode result's followed modes in the first mesh's order
    (`_aligned`).
    """
    from gfem._estimate_kinds import estimate, mode_field_reach

    modes = tracks.follow(result, options.cluster_rtol)
    view = _aligned(result, modes)
    estimates = tuple(
        estimate(
            model,
            view if _is_objective(each) else result,
            each,
            threads=options.threads,
            mode=_followed(modes, each),
            duals=duals,
            cache=cache,
            rtol=options.cluster_rtol,
            reach=mode_field_reach(options),
        )
        for each in estimators
    )
    if first:
        modes = tracks.add(estimates, modes, result)
    return estimates, modes


def _aligned(result: Any, modes: dict[int, int]) -> Any:
    """An eigenmode ``result`` with the followed modes in the first mesh's order."""
    if not isinstance(result, EigenmodeResult) or not modes:
        return result
    from gfem import _objective

    return _objective.aligned(result, modes)


def _shaped(items: list, joint: bool) -> Any:
    """``items`` as a tuple for a tuple of estimators, else its one item."""
    return tuple(items) if joint else items[0]


def _marking(
    estimates: tuple[ErrorEstimate, ...],
    tolerances: tuple[float, ...],
    above: list[int],
) -> np.ndarray:
    """The indicators Dörfler marks on: those of the quantities above tolerance.

    Each quantity's ``indicators**2`` is in its own units; divided by its
    tolerance they add up across quantities. A lone quantity's
    indicators are marked on as they are.
    """
    if len(above) == 1:
        return estimates[above[0]].indicators
    squared = sum(
        np.asarray(estimates[i].indicators, float) ** 2 / tolerances[i] for i in above
    )
    return np.sqrt(squared)


class _Tracks:
    """The modes the estimators follow: one track per distinct start.

    A start is one mode index, or a cluster's members (``omega=(i, ...)``).
    Estimators naming the same mode on the first mesh share its track.
    On each mesh a mode's track follows the mode nearest its
    previous omega (`_track_mode`), a cluster's the as many modes nearest its
    previous mean (`_track_cluster`), each shift bounded by the omega
    estimate on the previous mesh (`bound`); two tracks landing on one mode
    raise `DegenerateModeError`.
    """

    def __init__(self, estimators: tuple[Estimator, ...]) -> None:
        starts = sorted({m for e in estimators if (m := _track_start(e)) is not None})
        #: Per start: the followed indices, their mean omega and its omega
        #: estimate on the last mesh (``None`` before the first).
        self._state: dict[
            tuple[int, ...], tuple[tuple[int, ...], complex | None, float | None]
        ] = {start: (start, None, None) for start in starts}
        self._omega = {
            _track_start(e): i
            for i, e in enumerate(estimators)
            if isinstance(e, QuantityEstimator) and e.omega is not None
        }
        self._reached: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._omegas = np.zeros(0, complex)

    def add(
        self, estimates: tuple[ErrorEstimate, ...], modes: dict[int, int], result: Any
    ) -> dict[int, int]:
        """Follow the modes the first mesh's objectives read, one track each.

        On the first mesh each such mode is itself:
        ``modes`` with them added.
        """
        read = sorted({m for e in estimates for m in e._modes})  # noqa: SLF001
        new = [m for m in read if (m,) not in self._state]
        if not new:
            return modes
        self._omegas = np.asarray(result.omega)
        for m in new:
            self._state[(m,)] = ((m,), None, None)
            self._reached[(m,)] = (m,)
        return {**modes, **{m: m for m in new}}

    def omega(self, start: int) -> complex | None:
        """The omega of mode ``start``'s track on the last mesh, if followed."""
        state = self._state.get((start,))
        return None if state is None else state[1]

    def follow(self, result: Any, rtol: float) -> dict[int, int]:
        """Each named mode's index on ``result``: index on the first mesh -> there."""
        if not isinstance(result, EigenmodeResult) or not self._state:
            return {}
        reached: dict[tuple[int, ...], tuple[int, ...]] = {}
        for start, (indices, previous, bound) in self._state.items():
            if len(start) == 1:
                reached[start] = (
                    _track_mode(result, indices[0], previous, rtol, bound),
                )
            else:
                reached[start] = _track_cluster(result, indices, previous, bound)
        for (a, ma), (b, mb) in itertools.combinations(reached.items(), 2):
            if set(ma) & set(mb):
                mode = min(set(ma) & set(mb))
                omega = complex(np.asarray(result.omega)[mode])
                raise DegenerateModeError(
                    f"Two followed modes reached one mode: the modes named "
                    f"{_named(a)} and {_named(b)} on the first mesh both "
                    f"followed mode {mode} at "
                    f"{hertz(omega.real / (2 * np.pi))} on a refined mesh. Start "
                    "from a finer mesh, on which the modes are apart, or ask "
                    "for more modes (gfem.Eigenmode(n_modes=...))."
                )
        self._reached = reached
        self._omegas = np.asarray(result.omega)
        return {
            named: index
            for start, indices in reached.items()
            for named, index in zip(start, indices, strict=True)
        }

    def bound(
        self,
        model: Model,
        result: Any,
        estimates: tuple[ErrorEstimate, ...],
        *,
        threads: int | None,
        cache: RecoveryCache,
    ) -> None:
        """Keep each track's mean omega and estimate on ``result`` for the next mesh.

        An ``omega=`` estimate of the track's mode or cluster is its bound;
        any other track computes the omega kernel of its mode on the mesh's
        cache.
        """
        if not self._state:
            return
        from gfem._engine import estimate as kernels

        for start, indices in self._reached.items():
            if start in self._omega:
                bound = estimates[self._omega[start]].value
            else:
                bound = kernels.omega(
                    model, result, indices[0], threads, cache=cache
                ).value
            mean = complex(self._omegas[list(indices)].mean())
            self._state[start] = (indices, mean, bound)


def _named(start: tuple[int, ...]) -> str:
    """A track's start in words: ``"2"`` or ``"(0, 1)"``."""
    return str(start[0]) if len(start) == 1 else str(start)


def _warn_not_converged(
    stop: str,
    failure: Exception | None,
    above: list[tuple[Estimator, float, float]],
    *,
    joint: bool,
    max_cycles: int,
    max_tetrahedra: int | None,
) -> None:
    """Warn that `refine_mesh` stopped on ``stop`` with quantities above tolerance.

    ``above`` holds each such estimator, its estimate and its tolerance; with
    a tuple of estimators (``joint``) the warning names each. The warning
    points at the caller of `refine_mesh`, three frames up.
    """
    reason = {
        "max_cycles": f"after max_cycles={max_cycles} refinements",
        "max_tetrahedra": (
            f"because the next mesh would exceed max_tetrahedra={max_tetrahedra}"
        ),
        "solve failed": f"because the solve on the next mesh failed: {failure}",
        "non-finite estimate": (
            "because an error estimate is not finite, so it has nothing to mark on"
        ),
    }[stop]
    if joint:
        what = "; ".join(
            f"{_describe(e)} at {value:.3g} above its tolerance={tolerance:g}"
            for e, value, tolerance in above
        )
        estimates = f"with the error estimate of {what}"
    else:
        ((_, value, tolerance),) = above
        estimates = f"with the error estimate {value:.3g} above tolerance={tolerance:g}"
    warnings.warn(
        f"Mesh refinement stopped {reason}, {estimates}. The record "
        "holds the last solved mesh. Raise the cap, loosen tolerance, or "
        "start from a finer mesh.",
        MeshRefinementNotConvergedWarning,
        stacklevel=4,
    )


def _free_consumed(options: SolverOptions) -> None:
    """Free the solve entries a backward has read, before the first solve.

    As a differentiated forward does: their base handles return to the plan
    pool, for the solve of the input mesh to take, within
    ``options.retain_memory``. `refine_mesh` runs on one process.
    """
    from gfem._engine.cache import ENTRIES, retain_budget

    ENTRIES.free_consumed(None, retain_budget(options.retain_memory))


#: Squared indicators within this share of the largest are one tie group
#: in Dörfler marking (`_dorfler`).
_TIE_RTOL = 1e-9


def _dorfler(indicators: np.ndarray, error_fraction: float) -> np.ndarray:
    """Dörfler marking: the fewest tetrahedra holding a share of the squared error.

    Marks the tetrahedra with the largest indicators until their squared
    indicators add up to at least ``error_fraction`` of the total. Squared
    indicators equal to `_TIE_RTOL` of the largest are ties: they are sorted
    by tetrahedron index, and the tie group at the cut is marked whole, so
    round-off never picks among the equal indicators of a symmetric mesh,
    which refines symmetrically.

    Returns:
        A boolean mask over the tetrahedra; at least one is marked.
    """
    squared = np.asarray(indicators, float) ** 2
    largest = float(squared.max(initial=0.0))
    keys = (
        np.rint(squared / (_TIE_RTOL * largest)).astype(np.int64)
        if largest > 0
        else np.zeros(len(squared), np.int64)
    )
    order = np.argsort(-keys, kind="stable")
    cumulative = np.cumsum(squared[order])
    count = int(np.searchsorted(cumulative, error_fraction * cumulative[-1])) + 1
    count = min(count, len(squared))
    if largest > 0:
        sorted_keys = keys[order]
        # the whole tie group of the last marked tetrahedron
        count = int(np.searchsorted(-sorted_keys, -sorted_keys[count - 1], "right"))
    marked = np.zeros(len(squared), bool)
    marked[order[:count]] = True
    return marked


def _is_objective(estimator: Any) -> bool:
    return isinstance(estimator, QuantityEstimator) and estimator.objective is not None


def _recovery_cache() -> RecoveryCache:
    """A new `gfem._engine.estimate.RecoveryCache`: one per solved mesh."""
    from gfem._engine import estimate as kernels

    return kernels.RecoveryCache()


def _estimator_mode(estimator: Estimator) -> int | None:
    """The mode index of a `QuantityEstimator` (``omega=i``, ``q=``), else ``None``."""
    if not isinstance(estimator, QuantityEstimator) or isinstance(
        estimator.omega, tuple
    ):
        return None
    return estimator.omega if estimator.omega is not None else estimator.q


def _track_start(estimator: Estimator) -> tuple[int, ...] | None:
    """The modes an estimator follows on the first mesh: ``(i,)`` or a cluster."""
    if isinstance(estimator, QuantityEstimator) and isinstance(estimator.omega, tuple):
        return estimator.omega
    mode = _estimator_mode(estimator)
    return None if mode is None else (mode,)


def _quantity(
    result: Any,
    estimator: Estimator,
    mode: int | tuple[int, ...] | None,
    estimate: ErrorEstimate | None = None,
) -> float | complex | tuple[complex, ...] | None:
    """The quantity a `QuantityEstimator` names, read off ``result``.

    An objective's is its value, which its ``estimate`` holds.
    """
    if not isinstance(estimator, QuantityEstimator) or estimator.objective is not None:
        # None for a FieldEstimator, whose estimate holds no quantity
        return getattr(estimate, "_quantity", None)
    if estimator.capacitance is not None:
        i, j = (result.terminals.index(t) for t in estimator.capacitance)
        return float(np.asarray(result.capacitance)[i, j])
    if estimator.s is not None:
        return tuple(complex(v) for v in np.asarray(result.s[estimator.s]))
    if estimator.q is not None:
        return float(np.asarray(result.q)[mode])
    if isinstance(mode, tuple):
        return complex(np.asarray(result.omega)[list(mode)].mean())
    return complex(np.asarray(result.omega)[mode])


def _track_mode(
    result: EigenmodeResult,
    mode: int,
    previous: complex | None,
    rtol: float,
    previous_estimate: float | None,
) -> int:
    """The mode of ``result`` a quantity follows: nearest ``previous`` omega.

    On the first result (``previous`` is ``None``) it is ``mode`` itself.
    ``previous_estimate`` is the omega estimate on the previous mesh, relative
    to ``previous``.

    Raises:
        DegenerateModeError: Another mode lies within ``rtol`` of the
            followed one, or it belongs to a degenerate cluster; the nearest
            mode is further from ``previous`` than the previous estimate
            allows, or the second nearest is less than twice as far. The
            message names the cluster's modes and points to the cluster
            keyword.
    """
    omegas = np.asarray(result.omega)
    if previous is not None:
        distances = np.abs(omegas - previous)
        mode = int(np.argmin(distances))
        _check_followed(omegas, mode, distances, previous, previous_estimate)
        reference = previous
    else:
        reference = omegas[mode]
    close = np.abs(omegas - reference) <= rtol * abs(reference)
    cluster = np.asarray(result.cluster)
    together = np.flatnonzero(close | (cluster == cluster[mode]))
    if len(together) > 1:
        others = [int(i) for i in together if i != mode]
        named = (
            f", QuantityEstimator(omega={tuple(int(i) for i in together)})"
            if previous is None
            else " on the first mesh, QuantityEstimator(omega=(i, j, ...))"
        )
        raise DegenerateModeError(
            f"QuantityEstimator follows one mode by its frequency, and mode "
            f"{mode} at {_frequency(omegas, mode)} is in a degenerate cluster "
            f"with {_modes_text(omegas, others)} (labelled together in "
            "result.cluster, or within SolverOptions.cluster_rtol="
            f"{rtol:g}): name all of them{named}, or use "
            "gfem.FieldEstimator(), which estimates every mode."
        )
    return mode


def _frequency(omegas: np.ndarray, mode: int) -> str:
    """Mode ``mode``'s frequency in words, ``"9.12 GHz"``."""
    return hertz(complex(omegas[mode]).real / (2 * np.pi))


def _modes_text(omegas: np.ndarray, modes: list[int]) -> str:
    """``"modes 0 (9.12 GHz) and 2 (9.12 GHz)"``: modes with their frequencies."""
    items = [f"{i} ({_frequency(omegas, i)})" for i in modes]
    if len(items) == 1:
        return f"mode {items[0]}"
    return f"modes {', '.join(items[:-1])} and {items[-1]}"


def _check_followed(
    omegas: np.ndarray,
    mode: int,
    distances: np.ndarray,
    previous: complex,
    previous_estimate: float | None,
) -> None:
    """Raise unless ``mode`` is clearly the mode followed from ``previous``.

    The followed mode's errors on both meshes are at most the previous one,
    and the estimate is at least `_ESTIMATE_FLOOR` of the error on every
    verified case, so the mode moves at most twice the previous estimate over
    that floor. Further away, the nearest mode is another one.
    """
    nearest = float(distances[mode])
    frequency = _frequency(omegas, mode)
    if previous_estimate is not None:
        allowed = 2 * previous_estimate * abs(previous) / _ESTIMATE_FLOOR
        if nearest > allowed:
            raise DegenerateModeError(
                "QuantityEstimator(omega=...) follows one mode by its frequency, "
                f"and the nearest mode on the refined mesh, mode {mode} at "
                f"{frequency}, moved {nearest / abs(previous):.3g} of the "
                "followed frequency, further than the previous error estimate "
                f"{previous_estimate:.3g} allows. The followed mode probably "
                "left the solved modes: ask for more modes "
                "(gfem.Eigenmode(n_modes=...)) or start from a finer mesh."
            )
    near = [int(i) for i in np.flatnonzero(distances < 2 * nearest) if int(i) != mode]
    if near:
        raise DegenerateModeError(
            "QuantityEstimator(omega=...) follows one mode by its frequency, "
            f"and mode {mode} at {frequency} is not clearly nearest the "
            f"followed frequency on the refined mesh: "
            f"{_modes_text(omegas, near)} {'is' if len(near) == 1 else 'are'} "
            "less than twice as far, a degenerate cluster with it. Name all of "
            "them on the first mesh, QuantityEstimator(omega=(i, j, ...)), "
            "start from a finer mesh, on which the modes are apart, or use "
            "gfem.FieldEstimator(), which estimates every mode."
        )


def _track_cluster(
    result: EigenmodeResult,
    members: tuple[int, ...],
    previous: complex | None,
    previous_estimate: float | None,
) -> tuple[int, ...]:
    """The modes of ``result`` a cluster mean follows.

    On the first result (``previous`` is ``None``) they are ``members``, which
    must be whole ``result.cluster`` groups. On a refined mesh they are the
    as many modes nearest ``previous``, the previous mean, whose mean moved
    at most twice the previous estimate over `_ESTIMATE_FLOOR` (the
    single-mode bound, `_check_followed`). On every mesh the nearest other
    mode must lie at least twice as far from the mean (the first mesh) or
    from ``previous`` as the farthest member (the gap test).

    Raises:
        DegenerateModeError: ``members`` holds part of a labelled cluster;
            the mean moved too far; another mode is too near.
    """
    omegas = np.asarray(result.omega)
    count = len(members)
    keyword = f"QuantityEstimator(omega={members})"
    if previous is None:
        cluster = np.asarray(result.cluster)
        for mode in members:
            group = [int(i) for i in np.flatnonzero(cluster == cluster[mode])]
            missing = [i for i in group if i not in members]
            if missing:
                others = [i for i in group if i != mode]
                raise DegenerateModeError(
                    f"{keyword}: mode {mode} at {_frequency(omegas, mode)} is in "
                    f"a degenerate cluster with {_modes_text(omegas, others)} "
                    "(result.cluster): name all of them, QuantityEstimator("
                    f"omega={tuple(sorted({*members, *group}))})."
                )
        reference = complex(omegas[list(members)].mean())
        chosen = members
        where = "on the first mesh"
    else:
        distances = np.abs(omegas - previous)
        order = np.argsort(distances, kind="stable")
        chosen = tuple(sorted(int(i) for i in order[:count]))
        mean = complex(omegas[list(chosen)].mean())
        reference = previous
        if previous_estimate is not None:
            moved = abs(mean - previous)
            allowed = 2 * previous_estimate * abs(previous) / _ESTIMATE_FLOOR
            if moved > allowed:
                raise DegenerateModeError(
                    f"{keyword} follows a cluster by its mean frequency, and "
                    f"the {count} modes nearest it on the refined mesh, "
                    f"{_modes_text(omegas, list(chosen))}, moved their mean "
                    f"{moved / abs(previous):.3g} of the followed frequency, "
                    "further than the previous error estimate "
                    f"{previous_estimate:.3g} allows. The cluster probably left "
                    "the solved modes: ask for more modes "
                    "(gfem.Eigenmode(n_modes=...)) or start from a finer mesh."
                )
        where = "on the refined mesh"
    distances = np.abs(omegas - reference)
    farthest = float(distances[list(chosen)].max())
    others = [int(i) for i in np.argsort(distances, kind="stable") if i not in chosen]
    nearest = others[0]
    if distances[nearest] < 2 * farthest or distances[nearest] == 0:
        raise DegenerateModeError(
            f"{keyword} follows a cluster by its mean frequency, and the "
            f"cluster, {_modes_text(omegas, list(chosen))} {where}, is not "
            f"clearly apart from mode {nearest} at {_frequency(omegas, nearest)}: "
            "it is less than twice as far from the mean as the farthest "
            "member. Start from a finer mesh, on which the modes are apart, "
            "name it in the cluster too, or use gfem.FieldEstimator(), which "
            "estimates every mode."
        )
    return chosen


def _dofs(result: Any) -> int:
    """A solve's unknowns: one field's coefficient count."""
    return int(_fields(result)[0].coefficients.shape[0])


def _fields(result: Any) -> list:
    """The fields of a result an estimate reads."""
    if isinstance(result, ElectrostaticResult):
        return list(result.potential.values())
    return list(result.e_fields)
