"""Estimate admission: the refusals of `refine_mesh` and `error_estimate`.

Every refusal `refine_mesh` raises before its first solve, every refusal
`error_estimate` raises before it reads a field, and the refusals of
`refine_mesh` that wait for its first estimate. Each entry point keeps its
own sequence (`admit_refinement`, `admit_estimate`), since the two raise in
different orders and word some refusals differently; the checks they share
are written once here.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any

import numpy as np

from gfem import _mpi, _pml
from gfem._boundaries import first_wave_port
from gfem._checks import is_staging, is_traced
from gfem._domain import physics_of
from gfem._dual_supply import _adjoint_ports, _excited
from gfem._floquet import floquet_ports_in
from gfem._sources import INCIDENT, POLARIZATION
from gfem._text import hertz
from gfem.errors import MeshRefinementUnsupportedError, NonDifferentiableError
from gfem.mesh import _is_model_mesh
from gfem.model import (
    Absorbing,
    Impedance,
    LumpedPort,
    Model,
    surfaces_of,
)
from gfem.options import SolverOptions
from gfem.problems import Driven, Eigenmode, Electrostatic, PortMode
from gfem.refinement import (
    ErrorEstimate,
    Estimator,
    FieldEstimator,
    QuantityEstimator,
    _describe,
    _estimator_mode,
    _fields,
    _is_objective,
    _track_start,
)
from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult


def admit_refinement(
    model: Model,
    problem: Any,
    estimator: Any,
    tolerance: Any,
    options: SolverOptions,
    *,
    comm: Any,
    error_fraction: Any,
    max_cycles: Any,
    max_tetrahedra: Any,
) -> tuple[tuple[Estimator, ...], tuple[Any, ...]]:
    """Raise `refine_mesh`'s refusals in its order, before any mesh or solve.

    Returns the estimators and one tolerance each.
    """
    estimators, tolerances = _quantities(estimator, tolerance)
    _check_problem(model, problem)
    _refuse_blends(model, "refine_mesh")
    for each in estimators:
        if isinstance(each, QuantityEstimator) and each.objective is None:
            _check_entries(each, _entries_of_problem(problem, model))
            _check_quantity_model(each, model)
        if isinstance(each, QuantityEstimator) and isinstance(problem, Eigenmode):
            _check_wave_port_mode(model, _describe(each))
    if isinstance(problem, Driven):
        from gfem._engine import dispersion

        dispersion.check_estimator_frequencies(
            model, [float(f) for f in problem.frequency]
        )
    for each in estimators:
        _check_order(options.order, each)
    if _mpi.resolve_comm(comm) is not None:
        _mpi.refuse_feature("mesh_refinement")
    _check_adjoint_workers(estimators, model, problem, options)
    for each, limit in zip(estimators, tolerances, strict=True):
        _check_parameters(
            problem,
            options,
            limit,
            error_fraction,
            max_cycles,
            floor=not _is_objective(each),
        )
    if max_tetrahedra is not None:
        _check_int(max_tetrahedra, "max_tetrahedra", 1)
    _check_untraced("refine_mesh", (model, problem))
    if isinstance(problem, Driven):
        frequencies = [float(f) for f in problem.frequency]
        _check_driven_estimable(estimators, model, frequencies)
    return estimators, tolerances


def admit_estimate(result: Any, estimator: Any, model: Model) -> None:
    """Raise `error_estimate`'s refusals in its order, before any field is read."""
    _check_estimator(estimator)
    _refuse_blends(model, "error_estimate")
    _check_untraced("error_estimate", (model, result))
    _check_estimable(result, estimator, model)


def admit_first_estimates(
    problem: Any,
    result: Any,
    quantities: tuple[
        tuple[Estimator, ...], tuple[ErrorEstimate, ...], tuple[float, ...]
    ],
    modes: dict[int, int],
    options: SolverOptions,
    *,
    first: bool,
) -> None:
    r"""On the first mesh, raise for a Q or objective floor, or for workers.

    ``quantities`` holds the estimators, their estimates and tolerances.
    Q's floor is `_check_q_floors`. An objective's is the solve tolerance
    :math:`\tau` carried through the objective's first-order sensitivity,
    :math:`10\, \tau \sum_a |\partial J / \partial y_a| \, |y_a|`,
    since the estimate does not include the solve's error. A dual that
    rides on the forward solve needs one worker.
    """
    if not first:
        return
    estimators, estimates, tolerances = quantities
    _check_q_floors(result, estimators, tolerances, modes, options)
    solve_tolerance, name = _solve_tolerance(problem, options)
    workers = 1
    if isinstance(problem, Driven):
        workers = min(options.frequency_workers, len(problem.frequency))
    for estimator, estimate, tolerance in zip(
        estimators, estimates, tolerances, strict=True
    ):
        if not _is_objective(estimator):
            continue
        sensitivity = estimate._sensitivity  # noqa: SLF001
        floor = 10 * solve_tolerance * sensitivity
        if tolerance < floor:
            raise ValueError(
                f"refine_mesh tolerance={tolerance:g} for {_describe(estimator)} "
                f"must be at least {floor:.3g}: 10 x SolverOptions.{name}="
                f"{solve_tolerance:g} x the objective's first-order sensitivity "
                f"{sensitivity:.3g} (the sum of |dJ/dy| |y| over what it reads, "
                "|y| = 1 for S). The error estimate does not include the "
                f"solve's error. Lower SolverOptions.{name} to at most "
                f"{tolerance / (10 * sensitivity):.2g}, or raise tolerance."
            )
        if workers > 1 and (estimate._duals.ports or estimate._duals.keys):  # noqa: SLF001
            raise MeshRefinementUnsupportedError(
                "frequency workers",
                f"{_describe(estimator)} solves its adjoint duals with the "
                "forward, which frequency workers do not do in this version of "
                "gfem: it needs frequency_workers=1, got "
                f"{options.frequency_workers}.",
            )


#: `refine_mesh`'s tolerance when none is given.
_DEFAULT_TOLERANCE = 1e-2


def _quantities(
    estimator: Any, tolerance: Any
) -> tuple[tuple[Estimator, ...], tuple[Any, ...]]:
    """The estimators of `refine_mesh` and one tolerance each.

    A tuple of estimators holds `QuantityEstimator`s, at least one, none
    twice; a tuple of tolerances needs a tuple of estimators of its length.
    The tolerances' values are checked by `_check_parameters`.
    """
    if isinstance(estimator, tuple):
        if not estimator:
            raise ValueError(
                "refine_mesh needs at least one estimator in a tuple of estimators."
            )
        for each in estimator:
            if isinstance(each, FieldEstimator):
                raise TypeError(
                    "A tuple of estimators takes gfem.QuantityEstimator(...)s only: "
                    "gfem.FieldEstimator() already estimates every field of the "
                    "result, in units no quantity has. Refine for it on its own."
                )
            _check_estimator(each)
        for first, second in itertools.combinations(estimator, 2):
            if first == second:
                raise ValueError(
                    f"The tuple of estimators names {_describe(first)} twice: "
                    "give each quantity once."
                )
            _check_disjoint_tracks(first, second)
        estimators: tuple[Estimator, ...] = estimator
    else:
        _check_estimator(estimator)
        estimators = (estimator,)
    if tolerance is None:
        if any(_is_objective(e) for e in estimators):
            raise TypeError(
                "An objective's estimate is in the objective's own units, so it "
                "has no default tolerance: pass refine_mesh(tolerance=...) in "
                "those units."
            )
        return estimators, (_DEFAULT_TOLERANCE,) * len(estimators)
    if not isinstance(tolerance, tuple):
        return estimators, (tolerance,) * len(estimators)
    if not isinstance(estimator, tuple):
        raise TypeError(
            "A tuple of tolerances needs a tuple of estimators, one tolerance "
            "each: give tolerance as a float."
        )
    if len(tolerance) != len(estimators):
        raise ValueError(
            f"tolerance gives {len(tolerance)} values for {len(estimators)} "
            "estimators: give one per estimator, in their order, or one float "
            "for all."
        )
    return estimators, tolerance


def _check_disjoint_tracks(first: Estimator, second: Estimator) -> None:
    """Raise when a cluster shares a mode with another estimator's mode or cluster.

    A cluster is followed as a whole, and one mode of it has no
    track of its own; two estimators of one mode share its track.
    """
    a, b = _track_start(first), _track_start(second)
    if a is None or b is None or a == b or not set(a) & set(b):
        return
    raise ValueError(
        f"{_describe(first)} and {_describe(second)} share mode "
        f"{min(set(a) & set(b))}: a cluster is followed as a whole, so a mode "
        "of it can be named in one cluster only, and not alone. Estimate the "
        "cluster, or its modes one by one where they are apart."
    )


def _check_q_floors(
    result: Any,
    estimators: tuple[Estimator, ...],
    tolerances: tuple[float, ...],
    modes: dict[int, int],
    options: SolverOptions,
) -> None:
    """`_check_q_floor` of each ``q=`` estimator, on the first result."""
    for each, tolerance in zip(estimators, tolerances, strict=True):
        if isinstance(each, QuantityEstimator) and each.q is not None:
            _check_q_floor(result, modes[each.q], tolerance, options)


def _check_lossy_q(result: EigenmodeResult, mode: int) -> None:
    """Raise for the Q of a lossless mode: infinite, with no error to estimate."""
    if complex(np.asarray(result.omega)[mode]).imag == 0:
        raise ValueError(
            f"QuantityEstimator(q={mode}): mode {mode} is lossless, so its Q is "
            "infinite and has no error to estimate; estimate "
            f"QuantityEstimator(omega={mode})."
        )


def _check_q_floor(
    result: EigenmodeResult, mode: int, tolerance: float, options: SolverOptions
) -> None:
    """Raise for a Q tolerance below the eigen solve's error carried into Q.

    Q's relative error is cot(arg omega) times omega's relative error,
    so the solve's ``eigen_tolerance`` reaches Q multiplied by it.
    """
    _check_lossy_q(result, mode)
    omega = complex(np.asarray(result.omega)[mode])
    cot = abs(omega.real / omega.imag)
    floor = 10 * options.eigen_tolerance * cot
    if tolerance < floor:
        raise ValueError(
            f"refine_mesh tolerance={tolerance:g} for QuantityEstimator(q={mode}) "
            f"must be at least 10 x SolverOptions.eigen_tolerance="
            f"{options.eigen_tolerance:g} x cot(arg omega)={cot:.4g}, that is "
            f"{floor:.3g}: Q's relative error is cot(arg omega) times omega's, "
            "and the error estimate does not include the eigen solve's error. "
            "Lower SolverOptions.eigen_tolerance to at most "
            f"{tolerance / (10 * cot):.2g}, or raise tolerance."
        )


def _refuse_blends(model: Model, where: str) -> None:
    """Raise for a law blend: the estimators do not read one."""
    from gfem._engine import dispersion

    dispersion.refuse_blends(model, where)


def _check_estimator(estimator: Any) -> None:
    if not isinstance(estimator, (FieldEstimator, QuantityEstimator)):
        raise TypeError(
            "estimator must be gfem.FieldEstimator(), gfem.QuantityEstimator(...) "
            "or a tuple of gfem.QuantityEstimator(...)s, got "
            f"{type(estimator).__name__}."
        )


def _check_problem(model: Model, problem: Any) -> None:
    """Raise for a problem mesh refinement has no path for, before any solve."""
    if isinstance(problem, PortMode):
        raise MeshRefinementUnsupportedError(
            "port mode",
            "Mesh refinement has no path for a gfem.PortMode problem: refine "
            "the device with a gfem.Driven problem on its wave ports, whose "
            "mode solve runs on the refined port faces.",
        )
    # "thermal" or "dc_conduction": the estimators are electromagnetic
    feature = physics_of(problem)
    if feature != "em":
        raise MeshRefinementUnsupportedError(
            feature,
            f"Mesh refinement has no path for a gfem.{type(problem).__name__} "
            "problem: its error estimators are electromagnetic. Refine the mesh "
            "in your mesher.",
        )
    if not isinstance(problem, (Electrostatic, Driven, Eigenmode)):
        raise TypeError(
            "refine_mesh needs a gfem.Electrostatic, gfem.Driven or "
            f"gfem.Eigenmode problem, got {type(problem).__name__}."
        )
    if isinstance(problem, Driven) and problem.sources:
        # before the layer: its remedy, an absorbing face, would meet this next
        raise _refuse_source("gfem.refine_mesh", problem.sources[0])
    if model.pml is not None and isinstance(problem, (Driven, Eigenmode)):
        # the electrostatic estimator reads the background the solve sees
        raise _pml.refuse_estimate(
            "gfem.refine_mesh", model.pml, type(problem).__name__
        )
    if isinstance(problem, Driven) and problem.sweep == "adaptive":
        raise MeshRefinementUnsupportedError(
            "adaptive sweep",
            "Mesh refinement has no path for an adaptive sweep: refine on a "
            "uniform sweep at a few frequencies of the band, then sweep "
            "adaptively on refinement.model.",
        )
    if floquet_ports_in(model.ports):
        raise MeshRefinementUnsupportedError(
            "floquet port",
            "Mesh refinement has no path for a Floquet port: refining one side "
            "of a periodic pair would break its node pairing. Refine the mesh in "
            "your mesher.",
        )
    if getattr(problem, "wave_vector", None) is not None:
        raise MeshRefinementUnsupportedError(
            "bloch",
            "Mesh refinement has no path for a Bloch wave vector: refining one "
            "side of a periodic pair would break its node pairing. Refine the "
            "mesh in your mesher.",
        )
    if model._periodic:  # noqa: SLF001
        raise MeshRefinementUnsupportedError(
            "periodic",
            "Mesh refinement has no path for a periodic model: refining one "
            "side of a periodic pair would break its node pairing. Refine the "
            "mesh in your mesher.",
        )


def _refuse_source(subject: str, source: Any) -> MeshRefinementUnsupportedError:
    """The refusal of an estimator or refinement on a problem with a source.

    The estimators' residuals read the model's materials and ports only: a
    volume or a point source would make them mark the scatterer, the polarized
    region or the dipole's position, where the field is right.
    """
    if isinstance(source, INCIDENT):
        feature = "incident field"
    elif isinstance(source, POLARIZATION):
        feature = "polarization source"
    else:
        feature = "point dipole"
    return MeshRefinementUnsupportedError(
        feature,
        f"{subject} has no path for the {feature} {source.name!r}: its error "
        f"indicators know nothing of the {feature}'s load, so they would mark "
        "the field it drives as error. Refine the mesh in your mesher.",
    )


@dataclasses.dataclass(frozen=True)
class _Entries:
    """What a quantity estimator may name in a problem or a result.

    ``kind`` is the problem class (`Electrostatic`, `Driven` or
    `Eigenmode`) the given value is, or stands for, else ``None``;
    ``given`` its type name, and ``expected`` the name each kind goes by in
    this path's messages. Only the fields of ``kind`` are read; the others
    stay empty.
    """

    kind: type | None
    given: str
    expected: dict[type, str]
    terminals: list[str] = dataclasses.field(default_factory=list)
    ports: list[str] = dataclasses.field(default_factory=list)
    excited: list[str] = dataclasses.field(default_factory=list)
    n_modes: int = 0


def _entries_of_problem(problem: Any, model: Model) -> _Entries:
    """The entries ``problem`` solves for, its ports read off ``model``."""
    kinds = (Electrostatic, Driven, Eigenmode)
    entries = _Entries(
        kind=next((k for k in kinds if isinstance(problem, k)), None),
        given=type(problem).__name__,
        expected={k: k.__name__ for k in kinds},
    )
    if entries.kind is Electrostatic:
        terminals = [name for name, _ in problem.terminals]
        return dataclasses.replace(entries, terminals=terminals)
    if entries.kind is Driven:
        ports = [p.name for p in model.ports]
        excited = ports if problem.excite is None else list(problem.excite)
        return dataclasses.replace(entries, ports=ports, excited=excited)
    if entries.kind is Eigenmode:
        return dataclasses.replace(entries, n_modes=problem.n_modes)
    return entries


def _entries_of_result(result: Any) -> _Entries:
    """The entries ``result`` holds."""
    kinds = {
        ElectrostaticResult: Electrostatic,
        DrivenResult: Driven,
        EigenmodeResult: Eigenmode,
    }
    entries = _Entries(
        kind=next((k for r, k in kinds.items() if isinstance(result, r)), None),
        given=type(result).__name__,
        expected={k: r.__name__ for r, k in kinds.items()},
    )
    if entries.kind is Electrostatic:
        return dataclasses.replace(entries, terminals=list(result.terminals))
    if entries.kind is Driven:
        ports, excited = list(result.s.ports), list(result.s.excited)
        return dataclasses.replace(entries, ports=ports, excited=excited)
    if entries.kind is Eigenmode:
        n_modes = len(np.asarray(result.omega))
        return dataclasses.replace(entries, n_modes=n_modes)
    return entries


def _check_entries(estimator: QuantityEstimator, entries: _Entries) -> None:
    """Raise unless the quantity names one of ``entries``."""
    if estimator.capacitance is not None:
        _check_kind("capacitance", entries, Electrostatic)
        _check_terminals(estimator.capacitance, entries.terminals)
    elif estimator.s is not None:
        _check_kind("s", entries, Driven)
        _check_ports(estimator.s, entries.ports, entries.excited)
    else:
        keyword = estimator._mode_keyword or "omega"  # noqa: SLF001
        _check_kind(keyword, entries, Eigenmode)
        _check_modes(estimator, entries.n_modes, keyword)


def _check_kind(keyword: str, entries: _Entries, kind: type) -> None:
    if entries.kind is not kind:
        raise TypeError(
            f"QuantityEstimator({keyword}=...) needs a "
            f"gfem.{entries.expected[kind]}, got {entries.given}: name a quantity "
            "of this problem "
            "(capacitance= for Electrostatic, s= for Driven, omega= or q= for "
            "Eigenmode)."
        )


def _check_terminals(names: tuple[str, str], terminals: list[str]) -> None:
    for name in names:
        if name not in terminals:
            raise KeyError(
                f"No terminal {name!r}; the terminals are "
                f"{', '.join(map(repr, terminals))}."
            )


def _check_ports(names: tuple[str, str], ports: list[str], excited: list[str]) -> None:
    """Raise unless both ports exist and the source port (the column) is excited.

    The read port ``to`` may be any port with a resistance
    (`_check_defined_row`, with the model): when it is not excited, its
    S-functional adjoint is solved (`_adjoint_ports`).
    """
    for name in names:
        if name not in ports:
            raise KeyError(
                f"No port {name!r}; the ports are {', '.join(map(repr, ports))}."
            )
    source = names[1]
    if source not in excited:
        raise ValueError(
            f"QuantityEstimator(s=...) reads the field of its source port, and "
            f"port {source!r} is not excited: add it to Driven.excite."
        )


def _check_modes(estimator: QuantityEstimator, n_modes: int, keyword: str) -> None:
    """Raise unless the estimator's modes, and a cluster's guard mode, are solved.

    A cluster's gap test (`_track_cluster`) needs one mode more than the
    cluster holds.
    """
    members = _track_start(estimator) or ()
    if len(members) < 2:
        _check_mode(_estimator_mode(estimator), n_modes, keyword)
        return
    _check_mode(max(members), n_modes, keyword)
    if n_modes <= len(members):
        raise ValueError(
            f"QuantityEstimator(omega={members}) follows a cluster of "
            f"{len(members)} modes, told apart from the nearest other mode, "
            f"and n_modes={n_modes} leaves none: ask for one more mode: "
            f"gfem.Eigenmode(n_modes={len(members) + 1})."
        )


def _check_mode(mode: int | None, n_modes: int, keyword: str = "omega") -> None:
    if mode is not None and mode >= n_modes:
        raise ValueError(
            f"QuantityEstimator({keyword}={mode}) names a mode index at or above "
            f"n_modes={n_modes}: ask for more modes, or name one of 0 to "
            f"{n_modes - 1}."
        )


def _check_quantity_model(estimator: QuantityEstimator, model: Model) -> None:
    """Raise for a quantity entry the verification tier has no case for.

    ``q`` keeps the mode refusals lifted for ``omega``
    (`_check_lossy_mode`); the omega estimate checks its mode's Re eps when
    it estimates (`gfem._estimate_kinds._check_mode_eps`).
    """
    if estimator.q is not None:
        _check_lossy_mode(model, "q")
    if estimator.s is not None:
        _check_defined_row(estimator.s[0], model)


def _undefined_rows(model: Model) -> tuple[str, ...]:
    """The lumped ports with no resistance: their rows of S are undefined (NaN)."""
    from gfem._engine.port_network import resistance

    return tuple(
        p.name
        for p in model.ports
        if isinstance(p, LumpedPort) and math.isnan(resistance(p))
    )


def _check_defined_row(to: str, model: Model) -> None:
    """Raise when the read port ``to`` has no resistance (an undefined S row)."""
    if to in _undefined_rows(model):
        raise MeshRefinementUnsupportedError(
            "undefined S row",
            f"QuantityEstimator(s=({to!r}, ...)) reads the row of port {to!r}, "
            "which has no resistance: its row of S is undefined (NaN), so "
            "there is no error to estimate. Give the port a resistance, or "
            "read another row.",
        )


def _check_adjoint_workers(
    estimators: tuple[Estimator, ...],
    model: Model,
    problem: Any,
    options: SolverOptions,
) -> None:
    """Refuse frequency workers for an estimate that solves a dual.

    Its adjoint loads (`_loads`) ride on the forward's
    factorization, which a frequency worker does not hand back:
    frequency workers take wave ports but spread no adjoint
    solve.
    """
    if not isinstance(problem, Driven):
        return
    workers = min(options.frequency_workers, len(problem.frequency))
    excited = _excited(problem, model)
    solving = [e for e in estimators if _adjoint_ports(e, model, excited)]
    if workers > 1 and solving:
        raise MeshRefinementUnsupportedError(
            "frequency workers",
            f"{_describe(solving[0])} solves its "
            "adjoint dual with the forward, which frequency workers do not do "
            "in this version of gfem: it needs frequency_workers=1, got "
            f"{options.frequency_workers}.",
        )


def _lossy_elements(model: Model) -> np.ndarray:
    """Whether each tetrahedron is lossy: tan_delta, conductivity, complex eps, mu.

    A law region is lossy when a term of its law carries loss.
    """
    from gfem._engine import dispersion
    from gfem._engine.coefficients import element_tensors, lossy_eps

    laws = dispersion.law_regions(model)
    out = np.zeros(len(model.mesh.tetrahedra), bool)
    for name in ("tan_delta", "conductivity", "eps_r", "mu_r"):
        if name == "eps_r" and laws:
            # eps_r (1 - i tan_delta) is complex exactly where eps_r is or
            # tan_delta (counted above) is nonzero; a law region holds a real
            # placeholder (no traced value read), its loss is the law's
            values = np.asarray(lossy_eps(model, law=1.0))
        else:
            values = np.asarray(element_tensors(model, name))
        lossy = values.imag if name in ("eps_r", "mu_r") else values
        lossy = np.abs(lossy) > 0
        out |= lossy.reshape(len(out), -1).any(axis=1)
    for lr in laws:
        if dispersion.has_loss(lr.law):
            out |= lr.mask
    return out


def _check_lossy_mode(model: Model, keyword: str) -> None:
    """Raise for a mode term an estimate built on the mode has no case for.

    The terms lifted for omega alone: a permittivity law
    ("dispersive eigenmode"), and a second-order absorbing boundary, a
    complex mu_r or loss inside an anisotropic material ("lossy
    eigenmode"). Q and the objectives of a mode keep refusing them;
    ``keyword`` names the refused estimator's keyword.
    """
    from gfem._engine import dispersion
    from gfem._engine.coefficients import element_tensors

    laws = dispersion.law_regions(model)
    if laws:
        raise MeshRefinementUnsupportedError(
            "dispersive eigenmode",
            f"QuantityEstimator({keyword}=...) has no verified estimate for a mode "
            f"of a model with a permittivity law (region {laws[0].region!r}): "
            "use gfem.FieldEstimator().",
        )
    what = None
    if any(isinstance(b, Absorbing) and b.order == 2 for b in model.boundaries):
        what = "a second-order absorbing boundary"
    elif np.any(np.asarray(element_tensors(model, "mu_r")).imag):
        what = "a complex mu_r"
    elif (region := _lossy_anisotropic_region(model)) is not None:
        what = f"loss inside the anisotropic material of {region!r}"
    if what is not None:
        raise MeshRefinementUnsupportedError(
            "lossy eigenmode",
            f"QuantityEstimator({keyword}=...) has no verified estimate for a mode "
            f"with {what}: use gfem.FieldEstimator().",
        )


def _check_wave_port_mode(model: Model, what: str) -> None:
    """Raise for a quantity estimate of a mode of a model with a wave port.

    No estimate of an eigenfrequency carrying a port term (``omega``, ``q``
    or an objective of the mode) is verified; ``what`` names the estimator.
    """
    port = first_wave_port(model.ports)
    if port is not None:
        raise MeshRefinementUnsupportedError(
            "wave port eigenmode",
            f"{what} has no verified estimate for a mode with WavePort "
            f"{port.name!r}: its port term is not in the error identity. Use "
            "gfem.FieldEstimator().",
        )


def _lossy_anisotropic_region(model: Model) -> str | None:
    """The first anisotropic material region holding loss, if any."""
    from gfem._engine.coefficients import anisotropic_regions

    lossy = _lossy_elements(model)
    for region in anisotropic_regions(model, tuple(model.materials)):
        if lossy[model.mesh.elements(region)].any():
            return region
    return None


#: The plasmonic window of a law region's Re eps, open at both ends.
_PLASMONIC_WINDOW = (-20.0, 0.0)


def _check_driven_estimable(
    estimators: tuple[Estimator, ...], model: Model, frequencies: list[float]
) -> None:
    """Raise for a driven quantity estimate the identity has no term for.

    The S estimate's and every objective's on a driven result (objectives
    of S and driven field objectives alike).
    """
    for estimator in estimators:
        if isinstance(estimator, QuantityEstimator) and estimator.s is not None:
            _check_driven_quantity(model, frequencies, "QuantityEstimator(s=...)")
        elif _is_objective(estimator):
            _check_driven_quantity(model, frequencies, _describe(estimator))


def _check_driven_quantity(model: Model, frequencies: list[float], what: str) -> None:
    """Raise where a driven quantity estimate was measured underestimating.

    One check for the S estimate, objectives of S and driven field
    objectives, before any solve. It runs after
    `dispersion.check_estimator_frequencies` has gone over every frequency,
    whose zero-weight `ValueError` comes first. The field estimate is not
    refused here: it is unclaimed (recipe Limits).

    Args:
        model: The driven model.
        frequencies: The solve frequencies, Hz.
        what: The estimate, as the message names it (for example
            ``"QuantityEstimator(s=...)"``).

    Raises:
        MeshRefinementUnsupportedError: ``feature`` is ``"impedance or
            absorbing surface"`` (any `Impedance` or `Absorbing` surface),
            ``"lumped port spanning a permeability contrast"`` (the
            tetrahedra beside a `LumpedPort`'s surface differ in mu_r,
            tensors compared whole) or ``"plasmonic permittivity"`` (a law
            region with Re eps in `_PLASMONIC_WINDOW` at a frequency).
    """
    remedy = (
        "Refine with gfem.FieldEstimator() and check the quantity by a finer solve."
    )
    surface = next(
        (b.surface for b in model.boundaries if isinstance(b, (Impedance, Absorbing))),
        None,
    )
    if surface is not None:
        raise MeshRefinementUnsupportedError(
            "impedance or absorbing surface",
            f"{what} has no verified estimate on a driven model with an "
            f"impedance or absorbing surface ({surface!r}): the current on such "
            "a surface carries a share of the error that the estimate has no "
            f"term for. {remedy}",
        )
    port = _lumped_port_on_mu_contrast(model)
    if port is not None:
        raise MeshRefinementUnsupportedError(
            "lumped port spanning a permeability contrast",
            f"{what} has no verified estimate where the face of lumped port "
            f"{port!r} borders materials of different mu_r: the estimate "
            "misses the error at the edge where the permeability interface "
            f"meets the port's face. {remedy}",
        )
    plasmonic = _plasmonic_law(model, frequencies)
    if plasmonic is not None:
        region, f, eps = plasmonic
        low, high = _PLASMONIC_WINDOW
        raise MeshRefinementUnsupportedError(
            "plasmonic permittivity",
            f"{what} has no verified estimate where a permittivity law has Re "
            f"eps between {low:g} and {high:g}: region {region!r} has Re eps = "
            f"{eps:.3g} at {hertz(f)}. There, metal edges and lumped port edges "
            f"on the metal make the solution itself converge poorly. {remedy}",
        )


def _lumped_port_on_mu_contrast(model: Model) -> str | None:
    """The first lumped port whose surface borders tetrahedra of different mu_r."""
    from gfem._engine.coefficients import element_tensors

    lumped = [p for p in model.ports if isinstance(p, LumpedPort)]
    if not lumped:
        return None
    mesh = model.mesh
    mu = np.asarray(element_tensors(model, "mu_r"))
    mu = mu.reshape(len(mu), -1)
    active = model._domain.active  # noqa: SLF001
    neighbours = np.asarray(mesh.triangle_neighbours)
    for port in lumped:
        for surface in surfaces_of(port):
            beside = neighbours[mesh.elements(surface)].ravel()
            beside = beside[beside >= 0]
            values = mu[beside[active[beside]]]
            if len(values) and np.any(values != values[0]):
                return port.name
    return None


def _plasmonic_law(
    model: Model, frequencies: list[float]
) -> tuple[str, float, float] | None:
    """The first (region, frequency, Re eps) of a law in `_PLASMONIC_WINDOW`.

    Re eps is the kernel's (`dispersion.eps_at`): eps_inf plus the law's
    susceptibility; a conductivity adds to Im eps only.
    """
    from gfem._engine import dispersion

    low, high = _PLASMONIC_WINDOW
    regions = [lr for lr in dispersion.law_regions(model) if not is_traced(lr.law)]
    for f in frequencies:
        for lr in regions:
            eps = complex(lr.law.eps_inf) + dispersion.susceptibility(
                lr.law, 2 * math.pi * f
            )
            if low < eps.real < high:
                return lr.region, f, eps.real
    return None


def _touching(model: Model, surface: str) -> np.ndarray:
    """Whether each tetrahedron has a node on ``surface``."""
    mesh = model.mesh
    nodes = np.zeros(len(mesh.nodes), bool)
    nodes[np.asarray(mesh.triangles)[mesh.elements(surface)].ravel()] = True
    return nodes[np.asarray(mesh.tetrahedra)].any(axis=1)


def _check_estimable(result: Any, estimator: Estimator, model: Model) -> None:
    if isinstance(result, DrivenResult) and result.sources:
        source = next(iter(result.sources.values()))
        raise _refuse_source(f"gfem.{type(estimator).__name__}", source)
    if isinstance(estimator, QuantityEstimator) and estimator.objective is None:
        _check_entries(estimator, _entries_of_result(result))
    if isinstance(estimator, QuantityEstimator) and isinstance(result, EigenmodeResult):
        _check_wave_port_mode(model, _describe(estimator))
    if isinstance(estimator, QuantityEstimator) and estimator.objective is None:
        _check_quantity_model(estimator, model)
    if not isinstance(result, (ElectrostaticResult, DrivenResult, EigenmodeResult)):
        raise _unsupported_result(result, estimator)
    if any(field._comm is not None for field in _fields(result)):  # noqa: SLF001
        _mpi.refuse_feature("error_estimate")
    if model.pml is not None and isinstance(result, (DrivenResult, EigenmodeResult)):
        # the electrostatic estimator reads the background the solve sees
        raise _pml.refuse_estimate(f"gfem.{type(estimator).__name__}", model.pml)
    if floquet_ports_in(model.ports):
        raise MeshRefinementUnsupportedError(
            "floquet port",
            f"gfem.{type(estimator).__name__} has no path for a Floquet port: its "
            "flux recovery does not pair the periodic surfaces.",
        )
    if model._periodic:  # noqa: SLF001
        raise MeshRefinementUnsupportedError(
            "periodic",
            f"gfem.{type(estimator).__name__} has no path for a periodic model: "
            "its flux recovery does not pair the periodic surfaces.",
        )
    if isinstance(result, DrivenResult):
        _check_kept(result, estimator)
        from gfem._engine import dispersion

        frequencies = [float(f) for f in result.s.frequency]
        dispersion.check_estimator_frequencies(model, frequencies)
        _check_driven_estimable((estimator,), model, frequencies)
    _check_order(_fields(result)[0].order, estimator)
    mesh = _fields(result)[0].mesh
    if not _is_model_mesh(mesh, model.mesh):
        raise ValueError(
            "error_estimate needs the model the result was solved on: the "
            "result's fields live on another mesh, or on other node positions."
        )


def _check_order(order: int, estimator: Estimator) -> None:
    """Raise at order 1, where no estimate is verified."""
    if order == 1:
        raise MeshRefinementUnsupportedError(
            "order 1",
            f"gfem.{type(estimator).__name__} has no verified estimate at order 1: "
            "there, the frequency of a thin layer meshed with one cell across "
            "its thickness moved away from the exact one as the mesh was "
            "refined, and the estimate missed it. Solve at order 2 or 3 "
            "(gfem.SolverOptions(order=2)).",
        )


def _check_kept(result: DrivenResult, estimator: Estimator) -> None:
    """Raise unless a driven result is a uniform sweep with every field kept."""
    if result.sweep is not None and result.sweep.kind == "adaptive":
        raise MeshRefinementUnsupportedError(
            "adaptive sweep",
            f"gfem.{type(estimator).__name__} has no path for an adaptive sweep: "
            "estimate a uniform sweep at a few frequencies of the band.",
        )
    kept = {f for f, _ in result.kept}
    missing = [f for f in result.s.frequency if f not in kept]
    if missing:
        raise ValueError(
            "error_estimate reads the electric fields of every frequency, and "
            f"this result kept none at {', '.join(f'{f:g}' for f in missing)} Hz: "
            "solve with gfem.Driven(..., keep_fields=True)."
        )


def _unsupported_result(result: Any, estimator: Estimator) -> Exception:
    kind = type(result).__name__
    if not kind.endswith("Result"):
        return TypeError(f"error_estimate needs a result of gfem.solve, got {kind}.")
    return MeshRefinementUnsupportedError(
        kind.removesuffix("Result").lower(),
        f"gfem.{type(estimator).__name__} has no path for a gfem.{kind}: it "
        "estimates electrostatic, driven and eigenmode results.",
    )


def _check_untraced(name: str, values: Any) -> None:
    """Raise `NonDifferentiableError` on a traced leaf or a staging trace."""
    if is_traced(values) or is_staging():
        raise NonDifferentiableError(
            f"gfem.{name} cannot run inside jax.jit, or on values computed "
            "inside jax.grad, jax.jit or jax.vmap: mesh refinement builds a new "
            "mesh, which has no derivative. Refine with concrete values outside "
            "the transformed function, then differentiate gfem.solve on the "
            "refined model."
        )


def _check_parameters(
    problem: Any,
    options: SolverOptions,
    tolerance: Any,
    error_fraction: Any,
    max_cycles: Any,
    *,
    floor: bool = True,
) -> None:
    """Check the loop's parameters; ``floor``: the solve tolerance's floor too.

    An objective's floor is its sensitivity's (`admit_first_estimates`),
    checked on the first estimate.
    """
    for value, name in ((tolerance, "tolerance"), (error_fraction, "error_fraction")):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a number, got {value!r}.")
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(f"tolerance must be a positive number, got {tolerance!r}.")
    if not 0 < error_fraction <= 1:
        raise ValueError(f"error_fraction must be in (0, 1], got {error_fraction!r}.")
    _check_int(max_cycles, "max_cycles", 0)
    if not floor:
        return
    solve_tolerance, floor_name = _solve_tolerance(problem, options)
    if tolerance < 10 * solve_tolerance:
        raise ValueError(
            f"refine_mesh tolerance={tolerance:g} must be at least 10 x "
            f"SolverOptions.{floor_name}={solve_tolerance:g}: the error estimate "
            "does not include the linear solve's error. Lower "
            f"SolverOptions.{floor_name}, or raise tolerance."
        )


def _solve_tolerance(problem: Any, options: SolverOptions) -> tuple[float, str]:
    """The solve's tolerance the estimate does not include, and its option name."""
    if isinstance(problem, Eigenmode):
        return options.eigen_tolerance, "eigen_tolerance"
    return options.tolerance, "tolerance"


def _check_int(value: Any, name: str, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {value!r}.")
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}.")
