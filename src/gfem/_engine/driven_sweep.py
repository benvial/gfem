"""`SweepSplit`: a driven sweep split into blocks of its grid, run and joined.

Two driven sweeps split their grid: frequency workers (contiguous blocks,
each swept by a frequency worker's own rule in a spawned process) and a
fixed-angle sweep (one block per frequency, each solved by a
single-frequency rule of its own in this process). Both share what the
split owns: the partition, the parent's port mode sweep over the whole
grid and each block's rows of it, the join in frequency order, the
frequencies whose linear solve did not converge and the one error that
reports them, and the cache entry that holds the solutions only. How a
block runs is a `BlockRunner`: `FrequencyWorkers` or `FixedAngleChildren`.

An adaptive sweep splits nothing: what its forward keeps is a
`ReducedKept`.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
import sys
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

import numpy as np

from gfem import _progress, _timing
from gfem._engine import workers
from gfem._engine.cache import Entry
from gfem._engine.runtime import task_manager
from gfem._engine.solve_request import SolveRequest
from gfem._engine.sweep_outputs import SweepOutputs
from gfem._engine.wave_ports import SweptSurface, port_mode_sweeps
from gfem._linalg import report
from gfem._sources import IncidentField, PolarizationSource, qualified
from gfem._sweep.port_mode import rows_of
from gfem.errors import LinearSolverConvergenceError
from gfem.problems import Driven

if TYPE_CHECKING:
    from gfem._engine.driven import DrivenRule
    from gfem._engine.gauge import GaugeChoice
    from gfem._engine.sensitivity import FieldPair, Integrated
    from gfem._pml import Stretch
    from gfem.model import Model
    from gfem.options import SolverOptions
    from gfem.problems import Polarization

__all__ = [
    "Block",
    "BlockOutcome",
    "BlockRunner",
    "FixedAngleChildren",
    "FrequencyWorkers",
    "Partial",
    "ReducedKept",
    "SweepSplit",
    "check_loadable",
    "raise_unsolved",
    "solutions_entry",
    "solve_block",
]


@dataclasses.dataclass(frozen=True)
class Block:
    """One block of a split sweep's grid.

    Attributes:
        indices: Its frequencies' indices into the grid.
        port_modes: Its rows of the parent's port mode sweep, per wave port
            surface (`SolveRequest.port_modes`); ``None`` without wave ports.
    """

    indices: Sequence[int] | np.ndarray
    port_modes: list[dict[str, np.ndarray]] | None


@dataclasses.dataclass(frozen=True)
class BlockOutcome:
    """What one block's forward gave.

    Attributes:
        indices: The block's grid indices.
        outputs: Its outputs at the frequencies it solved (rows index the
            block), without the wave ports' modes; ``None`` when an
            in-process frequency did not converge.
        solution: Its solutions (``keep``), else ``None``.
        miss: ``(grid index, the linear solver's message)`` where its
            linear solve did not converge, else ``None``.
        cause: The linear solver's error of an in-process miss (a frequency
            worker's does not pickle).
    """

    indices: Sequence[int] | np.ndarray
    outputs: SweepOutputs | None
    solution: np.ndarray | None
    miss: tuple[int, str] | None
    cause: BaseException | None


class BlockRunner(Protocol):
    """Runs a split sweep's blocks: one outcome per block it ran, in order."""

    def run(
        self,
        model: Model,
        blocks: Sequence[Block],
        *,
        keep: bool,
        request: SolveRequest,
    ) -> list[BlockOutcome]:
        """Each block's forward of ``model``, under the parent's ``request``."""
        ...


@dataclasses.dataclass(frozen=True, eq=False)
class FrequencyWorkers:
    """Blocks run at the same time, each in a frequency worker (`workers.run`).

    Each block is swept by `gfem._engine.driven.DrivenRule.forward` in its
    worker (`solve_block`), with its share of ``threads``; a block whose
    linear solve does not converge stops there and returns the rows it
    solved before.

    Attributes:
        problem: The parent's problem.
        options: The parent's options.
        linear_kind: The parent's resolved linear solver kind.
        kept: Indices of the frequencies whose fields are kept.
        excite: The excited port names.
        gauge: The parent's gauge choice.
    """

    problem: Driven
    options: SolverOptions
    linear_kind: str
    kept: tuple[int, ...]
    excite: tuple[str, ...]
    gauge: GaugeChoice

    def run(
        self,
        model: Model,
        blocks: Sequence[Block],
        *,
        keep: bool,
        request: SolveRequest,  # noqa: ARG002 - a worker's request is its own
    ) -> list[BlockOutcome]:
        """Each block's forward in its own worker; the records replayed here."""
        frequencies = self.problem.frequency
        indices = [block.indices for block in blocks]
        threads = workers.worker_threads(self.options.threads, len(indices))
        # a worker's block cannot re-decide "auto"; its bases die
        # with it, and would each take half the machine
        options = dataclasses.replace(
            self.options,
            threads=threads,
            frequency_workers=1,
            linear_solver=self.linear_kind,
            retain_memory=0,
        )
        problems = [
            Driven(
                [frequencies[i] for i in block],
                excite=tuple(self.excite),
                keep_fields=[frequencies[i] for i in block if i in self.kept],
                wave_vector=self.problem.wave_vector,
                wave_vector_frequency=self.problem.wave_vector_frequency,
                polarization=self.problem.polarization,
                sources=self.problem.sources,
                source_fields=self.problem.source_fields,
            )
            for block in indices
        ]
        parts = workers.model_parts(model)
        with _progress.task(_progress.UNIFORM_SWEEP, total=len(frequencies)) as row:
            results = workers.run(
                solve_block,
                [(*parts, problem, options, keep) for problem in problems],
                [_describe(problem.frequency) for problem in problems],
                threads=threads,
                row=row,
                keywords=[
                    {
                        "port_modes": block.port_modes,
                        "order_frequency": tuple(frequencies),
                        "log": report.LOGGER.isEnabledFor(logging.INFO)
                        or _timing.LOGGER.isEnabledFor(logging.INFO),
                        "gauge": self.gauge,
                    }
                    for block in blocks
                ],
            )
        for r in results:
            report.replay(r[3])
        # each block stopped at its first miss: its solved rows lead it
        return [
            BlockOutcome(
                indices=block,
                outputs=r[0],
                solution=r[1],
                miss=None if r[2] is None else (int(block[len(r[0].rows)]), r[2]),
                cause=None,
            )
            for block, r in zip(indices, results, strict=True)
        ]


class FixedAngleChildren:
    """A fixed-angle sweep's single-frequency rules, and the blocks they run.

    Each frequency n of the parent's grid is solved by a rule of its own
    (`child`), built once: Bloch space, expansion, pieces, factorization (or
    BDDC preconditioner) and port network at k(f). It also keeps the port
    mode rows its forward handed each frequency, which the backward
    rebuilds that frequency's system on.
    """

    def __init__(self, rule: DrivenRule) -> None:
        """The children of ``rule``, a fixed-angle sweep; none built yet.

        Holds ``rule``'s settings, not ``rule``: the parent owns its children,
        and no cycle keeps either alive.
        """
        self.parent = _Parent(
            problem=rule.problem,
            options=rule.options,
            linear_kind=rule.linear.kind,
            excite=tuple(rule.excite),
            kept=rule.kept,
            comm=rule.comm,
            order_frequency=rule.order_frequency,
            gauge=rule.gauge,
            physics=rule.physics,
        )
        self.rules: dict[int, DrivenRule] = {}
        self.handed: dict[int, list[dict[str, np.ndarray]] | None] = {}

    def child(self, model: Model, n: int) -> DrivenRule:
        """The single-frequency rule of a fixed-angle sweep's frequency ``n``.

        Its problem is this one at that frequency alone, so its operator,
        Floquet couplings, port network and transposed system are at k(f);
        its order table is over the parent's grid (the whole
        sweep's), its linear solver the parent's resolved kind. Built once per
        frequency, without the warnings the parent gave for the whole sweep.
        """
        from gfem._engine.driven import FORWARD_WARNINGS, DrivenRule

        parent = self.parent
        rule = self.rules.get(n)
        if rule is not None:
            return rule
        # the child below drops the sources: a polarization source drives one
        # frequency, so a sweep (several frequencies) never holds one
        assert not any(
            isinstance(s, PolarizationSource) for s in parent.problem.sources
        )
        f = parent.problem.frequency[n]
        problem = Driven(
            [f],
            excite=tuple(parent.excite),
            keep_fields=n in parent.kept,
            wave_vector=parent.problem.wave_vector,
            wave_vector_frequency=parent.problem.wave_vector_frequency,
        )
        options = dataclasses.replace(
            parent.options, linear_solver=parent.linear_kind, frequency_workers=1
        )
        with warnings.catch_warnings():
            for category in FORWARD_WARNINGS:
                warnings.simplefilter("ignore", category)
            rule = DrivenRule(
                model,
                problem,
                options,
                comm=parent.comm,
                order_frequency=parent.order_frequency,
                gauge=parent.gauge,
            )
        self.rules[n] = rule
        return rule

    def run(
        self,
        model: Model,
        blocks: Sequence[Block],
        *,
        keep: bool,
        request: SolveRequest,
    ) -> list[BlockOutcome]:
        """Each frequency's forward by its rule, until the first miss.

        Each frequency's rule solves the request's adjoint loads into its
        ``adjoints``; its handles are freed before the next frequency.
        """
        outcomes: list[BlockOutcome] = []
        with _progress.task(
            _progress.UNIFORM_SWEEP, total=sum(len(b.indices) for b in blocks)
        ):
            for block in blocks:
                (n,) = block.indices
                rule = self.child(model, n)
                rows_n = block.port_modes
                # the backward rebuilds frequency n's system on the same rows
                self.handed[n] = rows_n
                child = dataclasses.replace(
                    request,
                    keeps_partial=True,
                    port_modes=rows_n,
                    adjoints=request.adjoints,
                )
                try:
                    outputs, entry = rule.forward(model, keep=keep, request=child)
                except LinearSolverConvergenceError as err:
                    partial = err._partial  # noqa: SLF001
                    miss = (n, str(err) if partial is None else partial.reason)
                    outcomes.append(BlockOutcome(block.indices, None, None, miss, err))
                    break
                finally:
                    rule.forget_operator()
                out = SweepOutputs.read(outputs).without_port_modes()
                solution = None
                if entry is not None:
                    solution = entry.fields["solution"]
                    entry.free()
                outcomes.append(BlockOutcome(block.indices, out, solution, None, None))
        return outcomes

    def backward(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None,
    ) -> list[FieldPair | Integrated]:
        """The backward of a fixed-angle sweep: each frequency on its own ±k(f).

        Each frequency's single-frequency rule (`child`) builds its
        system at k(f) again, on the port mode rows the forward handed it,
        and solves its adjoint loads on the transposed
        system at -k(f), per frequency; its handles are
        freed before the next frequency.
        """
        from gfem._engine.driven import _Kept

        parent = self.parent
        solution = entry.fields["solution"]
        width = len(parent.excite)
        ct_s = cotangents.get("s")
        ct_e = cotangents.get("e_field")
        pairs: list[FieldPair | Integrated] = []
        frequencies = parent.problem.frequency
        with _progress.task(
            _progress.UNIFORM_SWEEP_ADJOINT, total=len(frequencies)
        ) as row:
            for n in range(len(frequencies)):
                rule = self.child(model, n)
                part: dict[str, np.ndarray] = {}
                if ct_s is not None:
                    part["s"] = np.asarray(ct_s)[n : n + 1]
                if ct_e is not None and n in parent.kept:
                    k = parent.kept.index(n)
                    part["e_field"] = np.asarray(ct_e)[:, k * width : (k + 1) * width]
                if not part:
                    # no adjoint to solve: the frequency is done
                    row.advance(frequency=frequencies[n])
                    continue
                sub = Entry(
                    physics=parent.physics,
                    fields={"solution": solution[:, n * width : (n + 1) * width]},
                )
                try:
                    pairs.extend(
                        rule.backward(
                            model,
                            sub,
                            part,
                            wanted=wanted,
                            port_modes=self.handed.get(n),
                        )
                    )
                finally:
                    if isinstance(sub.handle, _Kept):
                        sub.handle.free()
                    rule.forget_operator()
        return pairs


@dataclasses.dataclass(frozen=True, eq=False)
class _Parent:
    """The settings of a fixed-angle sweep's rule its children are built from."""

    problem: Driven
    options: SolverOptions
    linear_kind: str
    excite: tuple[str, ...]
    kept: tuple[int, ...]
    comm: Any
    order_frequency: tuple[float, ...]
    gauge: GaugeChoice
    physics: str


@dataclasses.dataclass(frozen=True, eq=False)
class SweepSplit:
    """A driven sweep whose grid splits into blocks, each run by ``runner``.

    Chosen once per rule (`of`). The blocks join in frequency order, so
    every output stays frequency-major. The entry holds the solutions only:
    a backward pass builds its port network and system operator again.

    With wave ports, each surface's port mode sweep runs in the parent, once
    over the whole grid and before any block: a block's own sweep would
    track other modes. Each block takes its rows of the swept arrays; the
    port modes of the result and a `PortModeError` are the parent's. A block
    handed rows already (a fixed-angle sweep in a frequency worker) takes its
    rows of those.

    Attributes:
        frequency: The grid, Hz.
        wave_ports: The wave ports' names, in model port order.
        port_names: The model's port names.
        excite: The excited port names.
        physics: The physics the entry is kept for.
        options: The parent's options.
        blocks: Each block's grid indices.
        runner: What runs the blocks.
        polarization: A 2D problem's polarization, else ``None``.
        stretch: The parent's perfectly matched layer stretch, which the
            port sections of the whole-grid sweep take; ``None`` without a
            layer.
        n_sources: The source columns after the excited ports'.
        n_dipoles: The point dipoles among them.
    """

    frequency: tuple[float, ...]
    wave_ports: tuple[str, ...]
    port_names: tuple[str, ...]
    excite: tuple[str, ...]
    physics: str
    options: SolverOptions
    blocks: tuple[Sequence[int] | np.ndarray, ...]
    runner: BlockRunner
    polarization: Polarization | None = None
    stretch: Stretch | None = None
    n_sources: int = 0
    n_dipoles: int = 0

    @classmethod
    def of(cls, rule: DrivenRule) -> SweepSplit | None:
        """``rule``'s split: frequency workers, else a fixed-angle sweep's.

        ``None`` for an adaptive sweep, and for a uniform sweep that solves
        its frequencies in one pass. Frequency workers take ``rule``'s
        frequencies in ``frequency_workers`` contiguous blocks
        (`numpy.array_split`, without the empty ones); a fixed-angle sweep
        takes one block per frequency.
        """
        if rule.adaptive:
            return None
        frequencies = rule.problem.frequency
        runner: BlockRunner
        if min(rule.options.frequency_workers, len(frequencies)) > 1:
            indices = np.array_split(
                np.arange(len(frequencies)), rule.options.frequency_workers
            )
            blocks = tuple(block for block in indices if len(block))
            check_loadable(rule.problem.sources)
            runner = FrequencyWorkers(
                problem=rule.problem,
                options=rule.options,
                linear_kind=rule.linear.kind,
                kept=rule.kept,
                excite=tuple(rule.excite),
                gauge=rule.gauge,
            )
        elif rule.children is not None:
            blocks = tuple([n] for n in range(len(frequencies)))
            runner = rule.children
        else:
            return None
        return cls(
            frequency=tuple(frequencies),
            wave_ports=rule.wave_ports,
            port_names=tuple(rule.port_names),
            excite=tuple(rule.excite),
            physics=rule.physics,
            options=rule.options,
            blocks=blocks,
            runner=runner,
            polarization=rule.problem.polarization,
            stretch=rule.stretch,
            n_sources=len(rule.sources),
            n_dipoles=len(rule.emitting),
        )

    def forward(
        self, model: Model, *, keep: bool, request: SolveRequest
    ) -> tuple[dict[str, object], Entry | None]:
        """`DrivenRule.forward` over the blocks, joined.

        Raises:
            LinearSolverConvergenceError: A block's linear solve did not
                converge; once every block that ran is done, one error reports
                every block's miss (`raise_unsolved`).
        """
        handed = request.port_modes
        swept = [] if handed is not None else self._port_modes(model)
        arrays = handed if handed is not None else [s.arrays for s in swept]
        blocks = [
            Block(indices, [rows_of(a, indices) for a in arrays] if arrays else None)
            for indices in self.blocks
        ]
        outcomes = self.runner.run(model, blocks, keep=keep, request=request)
        sweeps = {
            port.name: sweep
            for s in swept
            for port, sweep in zip(s.ports, s.sweeps, strict=True)
        }
        joined = [(o.indices, o.outputs) for o in outcomes if o.outputs is not None]
        if joined:
            out = SweepOutputs.join(
                joined, sweeps=[sweeps[name] for name in self.wave_ports if sweeps]
            )
        else:
            out = SweepOutputs.uniform(
                (),
                s=np.zeros((0, len(self.port_names), len(self.excite)), complex),
                z_ref=np.zeros((0, len(self.port_names))),
                iterations=[],
                reused=[],
                e_field=None,
                waves=[],
                port_waves=np.zeros((0, len(self.port_names), self.n_sources), complex)
                if self.n_sources
                else None,
                emitted_power=np.zeros((0, self.n_dipoles)) if self.n_dipoles else None,
            )
        misses = [o.miss for o in outcomes if o.miss is not None]
        if misses:
            cause = next((o.cause for o in outcomes if o.cause is not None), None)
            raise_unsolved(
                self.frequency,
                out,
                misses,
                cause,
                keeps_partial=request.keeps_partial,
            )
        entry = None
        if keep:
            entry = solutions_entry(
                self.physics,
                [o.solution for o in outcomes if o.solution is not None],
            )
        return out.outputs(), entry

    def _port_modes(self, model: Model) -> list[SweptSurface]:
        """Each wave port surface's port mode sweep over the whole grid.

        Run once by the parent, which hands each block its rows. Empty without
        wave ports.
        """
        if not self.wave_ports:
            return []
        frequencies = self.frequency
        with task_manager(self.options.threads):
            swept = port_mode_sweeps(
                model,
                order=self.options.order,
                frequency=frequencies,
                tolerance=self.options.port_mode_tolerance,
                polarization=self.polarization,
                stretch=self.stretch,
            )
        return swept


@dataclasses.dataclass(frozen=True)
class Partial:
    """What a uniform sweep solved before a linear solve did not converge.

    Carried by its error as ``_partial`` (`DrivenRule.partial_result`).
    """

    #: The forward outputs at the solved frequencies (its rows) only.
    outputs: SweepOutputs
    #: The linear solver's message where the (first) linear solve did not
    #: converge.
    reason: str


def raise_unsolved(
    frequencies: Sequence[float],
    outputs: SweepOutputs,
    misses: list[tuple[int, str]],
    cause: BaseException | None = None,
    *,
    keeps_partial: bool,
) -> NoReturn:
    """Raise a uniform sweep's `LinearSolverConvergenceError`.

    With ``keeps_partial`` (`SolveRequest.keeps_partial`), the error keeps
    ``outputs`` for `DrivenRule.partial_result`.

    Args:
        frequencies: The sweep's grid, Hz.
        outputs: The forward outputs at the solved frequencies (its
            rows).
        misses: ``(grid index, the linear solver's message)`` of each
            frequency where a block's linear solve did not converge
            (one per frequency worker that stopped).
        cause: The linear solver's error, when there is one.
        keeps_partial: Whether the error keeps the outputs.
    """
    solved = outputs.rows
    first, reason = min(misses)
    where = "there"
    if len(misses) > 1:
        stops = " and ".join(f"{frequencies[i]:g} Hz" for i, _ in sorted(misses))
        where = f"at {stops} (one frequency worker each)"
    message = (
        f"At {frequencies[first]:g} Hz, {reason} The uniform sweep stopped "
        f"{where}, with {len(solved)} of {len(frequencies)} frequencies solved."
    )
    err = LinearSolverConvergenceError(
        message,
        frequency=frequencies[first],
        unsolved=tuple(f for i, f in enumerate(frequencies) if i not in solved),
    )
    if keeps_partial:
        err._partial = Partial(outputs, reason)  # noqa: SLF001
    raise err from cause


def solutions_entry(physics: str, solutions: list[np.ndarray]) -> Entry:
    """The cache entry of a split sweep: its solutions only, frequency-major."""
    return Entry(
        physics=physics,
        handle=None,
        fields={"solution": np.concatenate(solutions, axis=1)},
    )


@dataclasses.dataclass(frozen=True)
class ReducedKept:
    """What an adaptive sweep's forward keeps for its frozen-basis backward.

    Its fields are the cache entry's, by name; the entry holds no handle.

    Attributes:
        basis: The reduced basis :math:`V`.
        matrices: The reduced matrix at each grid frequency.
        functionals: The reduced S functionals at each grid frequency.
        deembedding: The port network's de-embedding factors.
        coordinates: The kept fields' reduced coordinates.
    """

    basis: np.ndarray
    matrices: np.ndarray
    functionals: np.ndarray
    deembedding: np.ndarray
    coordinates: np.ndarray

    def entry(self, physics: str) -> Entry:
        """The cache entry that keeps these."""
        return Entry(
            physics=physics,
            fields={
                "basis": self.basis,
                "matrices": self.matrices,
                "functionals": self.functionals,
                "deembedding": self.deembedding,
                "coordinates": self.coordinates,
            },
        )

    @classmethod
    def of(cls, entry: Entry) -> ReducedKept:
        """What ``entry``, an adaptive sweep's, keeps."""
        fields = entry.fields
        return cls(
            basis=fields["basis"],
            matrices=fields["matrices"],
            functionals=fields["functionals"],
            deembedding=fields["deembedding"],
            coordinates=fields["coordinates"],
        )


def solve_block(
    aux: Any,
    children: Any,
    problem: Driven,
    options: SolverOptions,
    keep: bool,
    *,
    port_modes: list[dict[str, np.ndarray]] | None = None,
    order_frequency: Sequence[float] | None = None,
    log: bool = False,
    gauge: GaugeChoice | None = None,
) -> tuple[SweepOutputs, np.ndarray | None, str | None, list[logging.LogRecord]]:
    """A frequency worker's `DrivenRule.forward` of its block.

    ``aux`` and ``children`` are the model's parts (`gfem._engine.workers`),
    ``port_modes`` the block's rows of the parent's port mode sweeps, per
    wave port surface (`SolveRequest.port_modes`), ``order_frequency``
    the whole sweep's grid, over which the block computes its Floquet order
    table (`DrivenRule.order_frequency`), ``gauge`` the parent rule's gauge
    choice (`DrivenRule.gauge`).
    Returns the block's outputs (rows: indices into the block; without the
    wave ports' modes, which are the parent's), its
    solutions (``keep``; else ``None``), ``None`` and, with ``log``, its
    ``gfem.solver`` records for the parent to log (`report.replay`). A
    block whose linear solve does not converge at some frequency stops
    there and returns the rows it solved, no solutions, and the linear
    solver's message, plain data that pickles (the parent raises once for
    every block).
    """
    from gfem._engine.driven import FORWARD_WARNINGS, rule_class

    model = workers.model_from_parts(aux, children)
    with warnings.catch_warnings():
        # the parent's rule already warned for the whole sweep
        for category in FORWARD_WARNINGS:
            warnings.simplefilter("ignore", category)
        rule = rule_class(model)(
            model, problem, options, order_frequency=order_frequency, gauge=gauge
        )
        request = SolveRequest(keeps_partial=True, port_modes=port_modes)
        with (
            report.collected(log) as records,
            _timing.scope(_timing.problem_name(problem)),
        ):
            try:
                outputs, entry = rule.forward(model, keep=keep, request=request)
            except LinearSolverConvergenceError as err:
                partial = err._partial  # noqa: SLF001
                if partial is None:  # pragma: no cover - a uniform sweep sets it
                    raise
                out = partial.outputs.without_port_modes()
                return out, None, partial.reason, records
    solution = None
    if entry is not None:
        solution = entry.fields["solution"]
        entry.free()
    return SweepOutputs.read(outputs).without_port_modes(), solution, None, records


def check_loadable(sources: Sequence[Any]) -> None:
    """Raise for a user incident function a spawned frequency worker cannot load.

    A worker unpickles its problem in a new process: a function pickles by
    reference to its module, so a lambda or a closure does not pickle at all,
    and a function of an interactive ``__main__`` (a notebook, a REPL,
    ``python -c``), which has no file, pickles here but kills the worker
    that imports it. Checked before any worker starts, since a dead worker
    cannot say why.

    Raises:
        ValueError: Naming the source.
    """
    for source in sources:
        if not isinstance(source, IncidentField):
            continue
        function = source.function
        why = None
        try:
            pickle.dumps(function)
        except Exception as err:  # noqa: BLE001 - any pickling failure says the same
            why = f"it does not pickle ({type(err).__name__}: {err})"
        else:
            main = sys.modules.get("__main__")
            if getattr(function, "__module__", None) == "__main__" and not hasattr(
                main, "__file__"
            ):
                why = (
                    "it is defined in an interactive __main__ (a notebook, a REPL "
                    "or python -c), which a worker cannot import"
                )
        if why is not None:
            raise ValueError(
                f"The IncidentField {source.name!r} (function "
                f"{qualified(function)}) cannot be sent to a frequency worker: "
                f"{why}. "
                "Define the function at module level in a script or an "
                "importable module, or use frequency_workers=1."
            )


def _describe(frequencies: tuple[float, ...]) -> str:
    """A block of frequencies, for errors."""
    first, last = frequencies[0], frequencies[-1]
    if first == last:
        return f"{first:g} Hz"
    return f"{first:g} to {last:g} Hz"
