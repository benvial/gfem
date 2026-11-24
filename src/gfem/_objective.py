"""Objective classification: what an objective ``J(result, model)`` reads.

`classify` traces the user's objective on a solved result and returns its
estimate kind, value, read entries with their cotangents, and the curvature
of the objective over those entries (the 2x2 spectral norms of its Hessian's
(Re, Im) blocks, for the remainder term). Two levels:

1. **Kind, refusals and mixtures** come from structural dependence on whole
   result leaves and on recorded measurement outputs: the objective's jaxpr,
   pruned by dead-code elimination. Each leaf of the result is an entry leaf
   (`ENTRY_KINDS`: its elements are the objective's scalar inputs), a refused
   leaf (`REFUSED_LEAVES`, or a field's coefficients: read through a
   recognized measurement only), or a constant.
2. **Read entries** are the entries of the live leaves whose NaN reaches the
   objective's value, or where its cotangent is nonzero.

Cotangents are the derivative of the objective's value, gfem's contract
``dJ/dx - i dJ/dy`` per complex entry, with ``stop_gradient`` evaluated as the
identity: the user's gradient choice does not change the quantity's error.

A recognized measurement reads a field only under the `Recorder`: a pass of
`jax.eval_shape` discovers which outputs the objective reads, and the jaxpr
takes those outputs as inputs, so no host kernel runs while tracing. Each
measured output is evaluated once, eagerly, for its value. A new kind plugs
in by an entry of `ENTRY_KINDS` (a result leaf) or of `MEASUREMENT_KINDS` (a
measurement, through `Recorder.output`), and an estimate kernel in
`gfem._estimate_kinds`.

The classifier relies on JAX internals (``dce_jaxpr``, ``eval_jaxpr``) at the
pinned JAX version; its unit tests assert a known classification table, so a
JAX upgrade that changes them fails loudly.
"""

from __future__ import annotations

import collections
import contextvars
import dataclasses
import os
import pathlib
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jcore
from jax._src.interpreters import partial_eval as pe

from gfem._checks import is_staging, is_traced
from gfem.errors import MeshRefinementUnsupportedError, NonDifferentiableError

__all__ = [
    "ENTRY_KINDS",
    "MEASUREMENT_KINDS",
    "Objective",
    "Read",
    "Recorder",
    "aligned",
    "classify",
    "recording",
]

#: The result leaves an objective may read, by result type and leaf path
#: (`jax.tree_util.keystr`, matched as a prefix): their estimate kind and
#: whether the tolerance floor scales an entry absolutely (1) or by its value.
ENTRY_KINDS: dict[tuple[str, str], tuple[str, bool]] = {
    ("DrivenResult", ".s.values"): ("objective of s", True),
    ("ElectrostaticResult", ".capacitance"): ("objective of capacitance", False),
    ("ElectrostaticResult", ".energy["): ("objective of electrostatic energy", False),
    ("EigenmodeResult", ".omega"): ("objective of omega", False),
    ("EigenmodeResult", ".bulk_participation["): ("objective of participation", False),
}

#: The leaves an objective may not read, by path prefix: the feature its
#: `MeshRefinementUnsupportedError` names.
REFUSED_LEAVES: dict[str, str] = {
    ".q_port[": "q_port",
    ".participation[": "participation",
    ".principal_participation[": "principal_participation",
    ".sheet_participation[": "sheet_participation",
    ".error": "eigen error",
    ".requested_power_share[": "requested power share",
    ".port_modes[": "port mode output",
    ".diffraction_orders[": "diffraction order output",
}

#: A recognized measurement's estimate kind, by (measurement, field kind).
MEASUREMENT_KINDS: dict[tuple[str, str], str] = {
    ("power_flow", "driven"): "objective of absorbed power",
    ("region_mean", "driven"): "objective of field means",
    ("region_mean", "potential"): "objective of potential means",
}

#: The one admitted mixture of kinds, and the kind it takes.
_MIXTURE = (
    frozenset({"objective of omega", "objective of participation"}),
    "objective of participation",
)

#: Primitives that branch on their operands: an objective applying one to a
#: read entry is not smooth there. ``select_n`` and
#: ``cond`` count when their predicate depends on a read entry.
_BRANCHING = frozenset(
    {
        "max",
        "min",
        "reduce_max",
        "reduce_min",
        "argmax",
        "argmin",
        "sort",
        "clamp",
        "cummax",
        "cummin",
        "top_k",
    }
)

_FIELD_KINDS = {
    "DrivenResult": "driven",
    "ElectrostaticResult": "potential",
    "EigenmodeResult": "mode",
}

_TRACE_ADVICE = (
    "gfem traces the objective with JAX to find what it reads: write it with "
    "jax.numpy, with no Python branch (if, bool(), float()) or NumPy conversion "
    "of a value of the result."
)

_REMEDY = "Estimate the quantities it reads one by one, or use gfem.FieldEstimator()."


@dataclasses.dataclass(frozen=True)
class Read:
    """One scalar an objective reads: a result entry or a measurement output.

    Attributes:
        leaf: The result leaf's path (``".s.values"``), or the measurement
            output's key (`Recorder.output`).
        index: The entry's index in the leaf, ``()`` for a 0-d leaf; a
            measurement output's component.
        value: The entry's value.
        cotangent: The objective's cotangent on it, ``dJ/dx - i dJ/dy`` for a
            complex entry, ``dJ/dx`` for a real one.
        is_complex: Whether the entry is complex.
        source: What it reads: the leaf's path without its key
            (``"s.values"``, ``"energy"``), or the measurement
            (``"power_flow"``, ``"region_mean"``).
        at: The key in the leaf's path (an energy's or a participation's
            region), or in the measured field's label (a kept field's index,
            a terminal); ``None`` for a path with no key.
        regions: A measurement's side (`gfem.power_flow`) or averaged regions
            (`gfem.region_mean`); ``()`` for a result entry.
    """

    leaf: str
    index: tuple[int, ...]
    value: complex
    cotangent: complex
    is_complex: bool
    source: str = ""
    at: int | str | None = None
    regions: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class Objective:
    """An objective classified on one result (`classify`).

    Attributes:
        kind: Its estimate kind (``gfem.refinement.ESTIMATE_KINDS``).
        value: The objective's value on the result.
        reads: Each scalar it reads, in a fixed order.
        curvature: ``(n, n)``, ``||H_ab||_2`` of the (Re, Im) blocks of the
            objective's Hessian over the read scalars ``a``, ``b`` (a real
            scalar has a one-row block).
        diagonal: Per read scalar, its own (Re, Im) block ``H_aa`` of the
            Hessian (2 x 2 for a complex entry, 1 x 1 for a real one): the
            blocks a rewrite of the objective shifts.
        floor_scales: Per read, its scale in the tolerance floor: 1 for an
            absolute entry (S), its magnitude otherwise.
    """

    kind: str
    value: float
    reads: tuple[Read, ...]
    curvature: np.ndarray
    floor_scales: tuple[float, ...]
    diagonal: tuple[np.ndarray, ...] = ()

    @property
    def sensitivity(self) -> float:
        """``sum_a |ct_a| s_a``: the objective's first-order sensitivity scale."""
        return float(
            sum(
                abs(r.cotangent) * s
                for r, s in zip(self.reads, self.floor_scales, strict=True)
            )
        )


# --- the recorder -----------------------------------------------------------------

_RECORDER: contextvars.ContextVar[Recorder | None] = contextvars.ContextVar(
    "gfem_objective_recorder", default=None
)


def recording() -> Recorder | None:
    """The `Recorder` of the objective being classified, else ``None``."""
    return _RECORDER.get()


class Recorder:
    """What the measurements an objective calls see while it is classified.

    Under a recorder, a recognized measurement does not touch the field: it
    names the field it was given (by the identity of its coefficients,
    `label`) and returns `output`, a scalar input of the traced objective.
    Other field reads record what they did, so a refusal can name it.

    Attributes:
        model: The model the objective was called with.
        discover: Whether this is the abstract pass (`jax.eval_shape`),
            which records each output's key, or the jaxpr's, which reads them.
    """

    def __init__(
        self,
        labels: Mapping[int, str],
        field_kinds: Mapping[str, str],
        fields: Mapping[str, Any],
        model: Any,
        *,
        discover: bool,
        specs: dict[str, dict[str, Any]],
        outputs: Mapping[str, jax.Array] | None = None,
        constants: Mapping[int, Any] | None = None,
    ) -> None:
        """Hold the field labels and what the measurements record."""
        self._labels = labels
        self._field_kinds = field_kinds
        self._fields = fields
        self._specs = specs
        self._outputs = outputs or {}
        self._constants = constants or {}
        self.model = model
        self.discover = discover
        self.point_reads: set[str] = set()

    def label(self, field: Any, who: str) -> str:
        """The label of a field of the classified result.

        Raises:
            MeshRefinementUnsupportedError: ``field`` is not one of the
                result's fields (``"field coefficients"``), or is an eigen
                mode field (``"normalized mode field"``).
        """
        label = self._labels.get(id(getattr(field, "coefficients", None)))
        if label is None:
            raise MeshRefinementUnsupportedError(
                "field coefficients",
                f"The objective calls {who} on a field that is not one of the "
                "fields of its own result argument, so its estimate has no "
                "dual for it: read fields from the result the objective is "
                "given.",
            )
        if self._field_kinds[label] == "mode":
            raise MeshRefinementUnsupportedError(
                "normalized mode field",
                f"The objective calls {who} on an eigen mode field ({label}), "
                "whose normalization makes its error one no estimate covers. "
                f"{_REMEDY}",
            )
        return label

    def point(self, field: Any) -> None:
        """Record that ``field`` is read at points (`gfem.Field.at`)."""
        label = self._labels.get(id(getattr(field, "coefficients", None)))
        if label is not None:
            self.point_reads.add(label)

    def field(self, label: str) -> Any:
        """The concrete field of ``label``."""
        return self._fields[label]

    def output(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: Any,
        evaluate: Callable[[], Any],
        **spec: Any,
    ) -> jax.Array:
        """A recognized measurement's output, as the traced objective's input.

        Args:
            key: Unique per (measurement, field, side or region).
            shape: The output's shape; each element is one scalar input.
            dtype: Its dtype.
            evaluate: The measurement's concrete value, run once eagerly.
            spec: What the estimate kernel needs to know of it; must hold
                ``measurement`` and ``label`` (its field's).
        """
        names = [f"{key}#{i}" for i in range(int(np.prod(shape, dtype=int)))]
        if self.discover:
            self._specs.setdefault(
                key,
                dict(spec, names=names, shape=shape, dtype=dtype, evaluate=evaluate),
            )
            return jnp.zeros(shape, dtype)
        values = [self._outputs[n] for n in names]
        return jnp.stack(values).reshape(shape) if shape else values[0]

    def constant(self, x: Any) -> Any:
        """``x`` as the concrete value, if it is a traced leaf held constant.

        A wave port's or a Floquet port's ``gamma`` is a constant of the
        solve when it moves a reference plane
        (`gfem.DrivenResult.with_offsets`): the objective is then still one
        of S.
        """
        return self._constants.get(id(x), x)


def constant(x: Any) -> Any:
    """`Recorder.constant` of the active recorder; ``x`` itself outside one."""
    recorder = recording()
    return x if recorder is None else recorder.constant(x)


# --- splitting a result into the objective's inputs ---------------------------------


@dataclasses.dataclass
class _Slot:
    path: str
    how: str  # "entries", "array" or "const"
    leaf: Any
    kind: str | None = None  # an entry leaf's kind


class _Split:
    """A result's leaves as the objective's traced inputs (see module docstring)."""

    def __init__(self, result: Any) -> None:
        self.result = result
        rtype = type(result).__name__
        flat, self.treedef = jax.tree_util.tree_flatten_with_path(result)
        self.slots: list[_Slot] = []
        self.entries: list[tuple[str, tuple[int, ...]]] = []
        self.entry_values: list[jax.Array] = []
        self.entry_kinds: list[str] = []
        self.entry_absolute: list[bool] = []
        self.arrays: list[str] = []
        self.array_values: list[jax.Array] = []
        self.field_kinds: dict[str, str] = {}
        self.fields: dict[str, Any] = {}
        self.port_mode_leaves: list[int] = []
        for path, leaf in flat:
            p = jax.tree_util.keystr(path)
            entry = next(
                (
                    v
                    for (r, prefix), v in ENTRY_KINDS.items()
                    if r == rtype and p.startswith(prefix)
                ),
                None,
            )
            if p.endswith(".coefficients") and rtype in _FIELD_KINDS:
                label = p.removesuffix(".coefficients")
                self.field_kinds[label] = _FIELD_KINDS[rtype]
                self._array(p, leaf)
            elif entry is not None:
                a = jnp.asarray(leaf)
                self.slots.append(_Slot(p, "entries", a.shape, entry[0]))
                for index in np.ndindex(a.shape):
                    self.entries.append((p, tuple(int(i) for i in index)))
                    self.entry_values.append(a[index])
                    self.entry_kinds.append(entry[0])
                    self.entry_absolute.append(entry[1])
            elif any(p.startswith(prefix) for prefix in REFUSED_LEAVES) and _is_array(
                leaf
            ):
                if p.startswith((".port_modes[", ".diffraction_orders[")):
                    self.port_mode_leaves.append(len(self.arrays))
                self._array(p, leaf)
            else:
                self.slots.append(_Slot(p, "const", leaf))

    def _array(self, path: str, leaf: Any) -> None:
        self.slots.append(_Slot(path, "array", len(self.arrays)))
        self.arrays.append(path)
        self.array_values.append(jnp.asarray(leaf))

    def build(
        self, entries: Sequence[jax.Array], arrays: Sequence[jax.Array]
    ) -> tuple[Any, dict[int, str]]:
        """The result on traced inputs, and its fields' labels by coefficients id."""
        leaves = []
        labels: dict[int, str] = {}
        k = 0
        for slot in self.slots:
            if slot.how == "const":
                leaves.append(slot.leaf)
            elif slot.how == "array":
                a = arrays[slot.leaf]
                leaves.append(a)
                label = slot.path.removesuffix(".coefficients")
                if label in self.field_kinds:
                    labels[id(a)] = label
            else:
                n = int(np.prod(slot.leaf, dtype=int))
                part = list(entries[k : k + n])
                k += n
                leaves.append(
                    jnp.stack(part).reshape(slot.leaf) if slot.leaf else part[0]
                )
        return self.treedef.unflatten(leaves), labels


def _is_array(leaf: Any) -> bool:
    return isinstance(leaf, (np.ndarray, jax.Array)) and np.ndim(leaf) >= 0


def _concrete_fields(split: _Split) -> dict[str, Any]:
    return {label: _field_at(split.result, label) for label in split.field_kinds}


def _field_at(result: Any, label: str) -> Any:
    """The field of ``result`` at a label: ``".e_fields[2]"``, ``".potential['a']"``."""
    return getattr(result, label[1:].partition("[")[0])[_bracket(label)]


def _bracket(path: str) -> int | str | None:
    """The key in a path's brackets: ``".e_fields[2]"`` 2, ``".energy['a']"`` ``"a"``.

    ``None`` for a path with no key (``".s.values"``).
    """
    _, bracket, key = path[1:].partition("[")
    if not bracket:
        return None
    key = key.removesuffix("]")
    return int(key) if key.isdigit() else key.strip("'\"")


# --- classification ---------------------------------------------------------------


def classify(
    objective: Callable[[Any, Any], Any], result: Any, model: Any
) -> Objective:
    """Classify ``objective(result, model)``: kind, value, reads and curvature.

    Raises:
        TypeError: The objective is not traceable (Python control flow or a
            NumPy conversion on a value of the result), or does not return a
            real 0-d value.
        ValueError: The objective is not finite at the result, or reads no
            entry of it.
        NonDifferentiableError: Its value carries a tracer of an enclosing
            JAX transformation.
        MeshRefinementUnsupportedError: It reads something no estimate
            covers (``feature`` names it), or mixes two estimate kinds.
    """
    trace = _Trace(objective, result, model)
    kind = trace.kind()
    compiled = _compiled(trace.pruned)
    if not compiled.walked:
        _walk(trace.pruned, [True] * len(trace.pruned.invars))
        compiled.walked = True
    consts, xs = trace.closed.consts, trace.inputs()
    value_array, cts = compiled.value_and_cotangents(consts, xs)
    _check_untraced(value_array)
    value = float(value_array)
    if not np.isfinite(value):
        _refuse_not_finite(result, kind, value)
    # positions in the live inputs ``xs`` of the scalars: entries and outputs
    scalars = [p for p, i in enumerate(trace.live) if trace.is_scalar(i)]
    cotangents = {}
    for p in scalars:
        ct = complex(np.asarray(cts[p]))
        cotangents[p] = ct if np.isfinite(ct) else 0j
    zero = [p for p in scalars if cotangents[p] == 0]
    read = sorted(
        {
            *(p for p in scalars if cotangents[p] != 0),
            *_nan_reads(compiled, consts, xs, zero),
        }
    )
    reads, scales = [], []
    curvature, diagonal = _curvature(compiled, consts, xs, read)
    for p in read:
        leaf, index, absolute = trace.name(trace.live[p])
        x = complex(np.asarray(xs[p]))
        is_complex = bool(jnp.iscomplexobj(xs[p]))
        if leaf in trace.specs:
            spec = trace.specs[leaf]
            source, at = spec["measurement"], _bracket(spec["label"])
            regions = tuple(spec.get("side") or spec.get("regions") or ())
        else:
            source, at, regions = leaf[1:].partition("[")[0], _bracket(leaf), ()
        reads.append(
            Read(leaf, index, x, cotangents[p], is_complex, source, at, regions)
        )
        scales.append(1.0 if absolute else abs(x))
    return Objective(
        kind=kind,
        value=value,
        reads=tuple(reads),
        curvature=curvature,
        diagonal=diagonal,
        floor_scales=tuple(scales),
    )


class _Trace:
    """The objective's jaxpr on a result's split inputs, and its pruning.

    The inputs are, in order: the entries, the arrays (fields and refused
    leaves), and the recorded measurement outputs (`_Split`, `Recorder`).
    """

    def __init__(
        self, objective: Callable[[Any, Any], Any], result: Any, model: Any
    ) -> None:
        self.split = split = _Split(result)
        self.model = model
        self.objective = objective
        self.fields = _concrete_fields(split)
        self.specs: dict[str, dict[str, Any]] = {}
        self.point_reads: set[str] = set()
        entries, arrays = list(split.entry_values), list(split.array_values)
        try:
            with _tracing_errors():
                self._discovering = True
                jax.eval_shape(lambda e, a: self._run(e, a, {}), entries, arrays)
                self._discovering = False
                outputs = {
                    n: jnp.zeros((), s["dtype"])
                    for s in self.specs.values()
                    for n in s["names"]
                }
                self.closed = jax.make_jaxpr(self._run)(entries, arrays, outputs)
        except _Unsupported as err:
            raise err.error from None
        out = self.closed.out_avals
        if (
            len(out) != 1
            or out[0].shape != ()
            or not jnp.issubdtype(out[0].dtype, jnp.floating)
        ):
            got = f"{out[0].dtype} of shape {out[0].shape}" if len(out) == 1 else out
            raise TypeError(f"{_REAL_ADVICE} Got {got}.")
        self.names = list(outputs)
        self.n_entries, self.n_arrays = len(entries), len(arrays)
        self.pruned, self.used = pe.dce_jaxpr(
            self.closed.jaxpr, [True], instantiate=False
        )
        self.live = [i for i, u in enumerate(self.used) if u]

    def _run(
        self,
        entries: Sequence[jax.Array],
        arrays: Sequence[jax.Array],
        outputs: Mapping[str, jax.Array],
    ) -> Any:
        """The objective on the inputs; ``outputs`` ``{}`` discovers them."""
        split = self.split
        res, labels = split.build(entries, arrays)
        recorder = Recorder(
            labels,
            split.field_kinds,
            self.fields,
            self.model,
            discover=self._discovering,
            specs=self.specs,
            outputs=outputs,
            constants={
                id(arrays[i]): split.array_values[i] for i in split.port_mode_leaves
            },
        )
        token = _RECORDER.set(recorder)
        try:
            return self.objective(res, self.model)
        finally:
            _RECORDER.reset(token)
            self.point_reads |= recorder.point_reads

    def _is_entry(self, i: int) -> bool:
        return i < self.n_entries

    def _is_output(self, i: int) -> bool:
        return i >= self.n_entries + self.n_arrays

    def kind(self) -> str:
        """The estimate kind of the live inputs; raises for a refused one."""
        split = self.split
        _refuse_arrays(
            split,
            [
                i - self.n_entries
                for i in self.live
                if not (self._is_entry(i) or self._is_output(i))
            ],
            self.point_reads,
        )
        kinds = {split.entry_kinds[i] for i in self.live if self._is_entry(i)}
        for i in self.live:
            if self._is_output(i):
                spec = self.specs[self.name(i)[0]]
                field_kind = split.field_kinds[spec["label"]]
                kinds.add(MEASUREMENT_KINDS[spec["measurement"], field_kind])
        if not kinds:
            raise ValueError(
                "The objective reads nothing gfem solved: it is a constant of "
                "the result. Read the result's S, capacitance, energy, omega or "
                "participation, or a field through gfem.power_flow or "
                "gfem.region_mean, from the result argument the objective is "
                "given."
            )
        return _kind(kinds)

    def is_scalar(self, i: int) -> bool:
        """Whether input ``i`` is a scalar: an entry or a measurement output."""
        return self._is_entry(i) or self._is_output(i)

    def name(self, i: int) -> tuple[str, tuple[int, ...], bool]:
        """A scalar input's leaf (or output key), index, and floor absoluteness."""
        if self._is_entry(i):
            leaf, index = self.split.entries[i]
            return leaf, index, self.split.entry_absolute[i]
        key, _, component = self.names[i - self.n_entries - self.n_arrays].partition(
            "#"
        )
        return key, (int(component),), False

    def inputs(self) -> list[Any]:
        """The live inputs' concrete values; each live measurement run once.

        A measurement the pruned objective does not read is not run.
        """
        offset = self.n_entries + self.n_arrays
        wanted = {self.names[i - offset] for i in self.live if self._is_output(i)}
        measured: dict[str, jax.Array] = {}
        for spec in self.specs.values():
            if wanted.isdisjoint(spec["names"]):
                continue
            values = np.ravel(np.asarray(spec["evaluate"]()))
            for n, v in zip(spec["names"], values, strict=True):
                measured[n] = jnp.asarray(v)
        out = []
        for i in self.live:
            if self._is_entry(i):
                out.append(self.split.entry_values[i])
            elif self._is_output(i):
                out.append(measured[self.names[i - offset]])
            else:  # pragma: no cover - a live array raised in `kind`
                out.append(self.split.array_values[i - self.n_entries])
        return out


_REAL_ADVICE = (
    "An objective is one real number: return a real 0-d value, for example "
    "its .real, jnp.abs(...) or jnp.abs(...) ** 2."
)


class _Unsupported(Exception):  # noqa: N818 - carries an error out of a JAX trace
    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class _tracing_errors:  # noqa: N801 - a context manager used as a block
    """Turn a JAX tracing error raised by the objective into `TypeError`."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        tb: types.TracebackType | None,
    ) -> bool:
        if error is None:
            return False
        if isinstance(error, MeshRefinementUnsupportedError):
            raise _Unsupported(error)
        if isinstance(
            error,
            (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerBoolConversionError,
                jax.errors.TracerArrayConversionError,
                jax.errors.TracerIntegerConversionError,
            ),
        ):
            raise TypeError(_TRACE_ADVICE) from error
        return False


def _kind(kinds: set[str]) -> str:
    if len(kinds) == 1:
        return next(iter(kinds))
    if kinds == _MIXTURE[0]:
        return _MIXTURE[1]
    raise MeshRefinementUnsupportedError(
        "mixed objective",
        f"The objective reads quantities of different estimate kinds "
        f"({', '.join(sorted(kinds))}), which one estimate does not combine. "
        "Estimate each as its own objective, together in a tuple of "
        "QuantityEstimators.",
    )


def _refuse_arrays(split: _Split, live: list[int], point_reads: set[str]) -> None:
    """Raise for a live field or refused leaf."""
    for i in live:
        path = split.arrays[i]
        label = path.removesuffix(".coefficients")
        if label in split.field_kinds:
            if split.field_kinds[label] == "mode":
                raise MeshRefinementUnsupportedError(
                    "normalized mode field",
                    f"The objective reads the eigen mode field {label}, whose "
                    f"normalization makes its error one no estimate covers. {_REMEDY}",
                )
            if label in point_reads:
                raise MeshRefinementUnsupportedError(
                    "point value",
                    f"The objective reads {label} at points (Field.at), and a "
                    "point value has no estimate: read the mean "
                    "over a small region with gfem.region_mean.",
                )
            raise MeshRefinementUnsupportedError(
                "field coefficients",
                f"The objective reads the coefficients of {label} directly. "
                "Read fields through gfem.power_flow or gfem.region_mean, "
                "whose duals the estimate knows.",
            )
        feature = next(f for p, f in REFUSED_LEAVES.items() if path.startswith(p))
        raise MeshRefinementUnsupportedError(
            feature,
            f"The objective reads {path.lstrip('.')}, and {feature} has no "
            f"estimate in this version of gfem. {_REMEDY}",
        )


def _refuse_not_finite(result: Any, kind: str, value: float) -> None:
    if kind == "objective of s":
        s = result.s
        undefined = [
            p
            for i, p in enumerate(s.ports)
            if np.isnan(np.asarray(s.z_ref)[:, i]).any()
        ]
        if undefined:
            raise MeshRefinementUnsupportedError(
                "undefined S row",
                "The objective reads S of a port without a resistance "
                f"({', '.join(map(repr, undefined))}): its row of S is "
                "undefined (NaN), as it is for the gradient. Give the port a "
                "resistance, or leave its row out.",
            )
    raise ValueError(
        f"The objective is {value} at the result, not a finite number, so it "
        "has no error to estimate. (y() and z() of a wave port result are NaN: "
        "a wave port has no reference impedance.)"
    )


# --- jaxpr structure: branches and custom derivatives ------------------------------

#: Call-like primitives: their one sub-jaxpr runs on their operands, one to one.
_CALLS = frozenset({"pjit", "jit", "closed_call", "core_call", "remat", "checkpoint"})

#: JAX's own custom-derivative rules that are smooth although their primal
#: branches (a ``max`` that only stabilizes the evaluation): admitted whole.
#: Any other rule JAX defines is walked like the objective (``relu`` is a
#: branch); a rule defined outside JAX is the user's own, and raises.
_SMOOTH_JAX_RULES = frozenset({"logaddexp", "logaddexp2"})

_JAX_DIR = str(pathlib.Path(jax.__file__).parent) + os.sep

#: Higher-order primitives smooth in their operands, admitted whole: a
#: linear solve (``jnp.linalg.solve``, ``inv``; ``S.y()``, ``inverse``).
_SMOOTH_HIGHER_ORDER = frozenset({"custom_linear_solve"})


#: JAX before 0.10 holds call bodies as closed jaxprs; later JAX has no such type.
_CLOSED: tuple[type, ...] = (
    (jcore.ClosedJaxpr,) if hasattr(jcore, "ClosedJaxpr") else ()
)
_JAXPRS: tuple[type, ...] = (jcore.Jaxpr, *_CLOSED)


def _jaxprs(params: Mapping[str, Any]) -> Iterator[tuple[str, int | None, Any]]:
    """Each jaxpr held in an equation's ``params``: (key, tuple index, jaxpr)."""
    for key, value in params.items():
        if isinstance(value, _JAXPRS):
            yield key, None, value
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                if isinstance(item, _JAXPRS):
                    yield key, i, item


def _open(jaxpr: Any) -> Any:
    """The open jaxpr of ``jaxpr``, closed or not."""
    return jaxpr.jaxpr if isinstance(jaxpr, _CLOSED) else jaxpr


def _rule(eqn: Any) -> tuple[str, bool]:
    """A custom-derivative equation's function name, and whether JAX defines it."""
    info = getattr(eqn.params["call_jaxpr"], "debug_info", None)
    where = getattr(info, "func_src_info", None) or ""
    name, _, path = where.partition(" at ")
    return name, path.startswith(_JAX_DIR)


def _live(jaxpr: Any, live_out: Sequence[bool]) -> set[Any]:
    """The variables whose value reaches a live output past no ``stop_gradient``.

    A branch whose outputs reach the objective only through
    ``stop_gradient`` does not change its value (``logsumexp``'s max).
    Through an equation with sub-jaxprs, every operand of a live output is
    live.
    """
    live = {
        v
        for v, keep in zip(jaxpr.outvars, live_out, strict=True)
        if keep and not isinstance(v, jcore.Literal)
    }
    for eqn in reversed(jaxpr.eqns):
        if eqn.primitive.name == "stop_gradient":
            continue
        if any(v in live for v in eqn.outvars):
            live.update(v for v in eqn.invars if not isinstance(v, jcore.Literal))
    return live


def _refuse_branch(name: str) -> MeshRefinementUnsupportedError:
    return MeshRefinementUnsupportedError(
        "non-smooth objective",
        f"The objective branches on a value it reads ({name}: max, min, sort, "
        "where/cond on a predicate of the result, or a loop whose stop or "
        "body depends on it). At a near-tie its derivative follows one branch "
        "only, and the other's error is missed: no estimate is verified for "
        "that. Estimate each branch as its own objective.",
    )


def _walk(
    jaxpr: Any, dependent: Sequence[bool], live_out: Sequence[bool] | None = None
) -> list[bool]:
    """Forward dependence on read values through ``jaxpr``, raising on the way.

    Raises for a branch on a read value that reaches a live output
    (``live_out``, every output by default; `_live`), a user's custom
    derivative, or a higher-order primitive this walk does not know.
    Recurses into every sub-jaxpr: calls, ``cond``,
    ``scan``, ``while`` and JAX's own custom rules. On the objective's
    pruned jaxpr every input is read.
    """
    if live_out is None:
        live_out = [True] * len(jaxpr.outvars)
    live = _live(jaxpr, live_out)
    dep: dict[Any, bool] = {}

    def read(v: Any) -> bool:
        return False if isinstance(v, jcore.Literal) else dep.get(v, False)

    dep.update(zip(jaxpr.invars, dependent, strict=True))
    for eqn in jaxpr.eqns:
        ins = [read(v) for v in eqn.invars]
        reaches = [v in live for v in eqn.outvars]
        outs = _walk_eqn(eqn, ins, reaches)
        dep.update(zip(eqn.outvars, outs, strict=True))
    return [read(v) for v in jaxpr.outvars]


def _walk_eqn(eqn: Any, ins: list[bool], reaches: list[bool]) -> list[bool]:
    """One equation's output dependence (see `_walk`)."""
    name = eqn.primitive.name
    flat = [any(ins)] * len(eqn.outvars)
    if not any(ins):
        return flat
    if name.startswith(("custom_jvp_call", "custom_vjp_call")):
        return _walk_custom(eqn, ins, reaches)
    walker = _HIGHER_ORDER.get(name)
    if walker is not None:
        return walker(eqn, ins, any(reaches))
    branching = name in _BRANCHING or (name == "select_n" and ins[0])
    unknown = name not in _SMOOTH_HIGHER_ORDER and any(
        True for _ in _jaxprs(eqn.params)
    )
    if (branching or unknown) and any(reaches):
        raise _refuse_branch(name)
    return flat


def _walk_custom(eqn: Any, ins: list[bool], reaches: list[bool]) -> list[bool]:
    """A custom-derivative rule: JAX's own are walked (or admitted), a user's raises."""
    rule, in_jax = _rule(eqn)
    if not in_jax:
        raise MeshRefinementUnsupportedError(
            "custom derivative in objective",
            "The objective holds a jax.custom_jvp or jax.custom_vjp rule of "
            f"its own ({rule or 'unnamed'}) on what it reads. The estimate "
            "needs the true derivative of the objective's value, and a user "
            "rule was not verified: write the objective without it.",
        )
    if rule in _SMOOTH_JAX_RULES:
        return [any(ins)] * len(eqn.outvars)
    return _walk(eqn.params["call_jaxpr"], ins, reaches)


def _walk_call(eqn: Any, ins: list[bool], live: bool) -> list[bool]:
    """A call: its one sub-jaxpr, on its operands one to one."""
    ((_, _, sub),) = list(_jaxprs(eqn.params))
    return _walk(_open(sub), ins, [live] * len(eqn.outvars))


def _walk_cond(eqn: Any, ins: list[bool], live: bool) -> list[bool]:
    """A ``cond``: a branch when its index depends on a read value."""
    if ins[0] and live:
        raise _refuse_branch("cond")
    n_out = len(eqn.outvars)
    outs = [ins[0]] * n_out
    for branch in eqn.params["branches"]:
        got = _walk(branch, ins[1:], [live] * n_out)
        outs = [a or b for a, b in zip(outs, got, strict=True)]
    return outs


def _walk_scan(eqn: Any, ins: list[bool], live: bool) -> list[bool]:
    """A ``scan``: its body walked to a fixed point of the carry's dependence."""
    body = eqn.params["jaxpr"]
    ft_in = eqn.params.get("ft_in")
    if ft_in is None or len(ft_in) != 3:  # pragma: no cover - another JAX layout
        n_consts, n_carry = 0, 0
        ins = [any(ins)] * len(ins)
    else:
        n_consts, n_carry = len(ft_in[0]), len(ft_in[1])
    dep = list(ins)
    while True:
        outs = _walk(body, dep, [live] * len(body.outvars))
        carry = [
            a or b
            for a, b in zip(
                dep[n_consts : n_consts + n_carry], outs[:n_carry], strict=True
            )
        ]
        new = [*dep[:n_consts], *carry, *dep[n_consts + n_carry :]]
        if new == dep:
            return outs
        dep = new


def _walk_while(eqn: Any, ins: list[bool], live: bool) -> list[bool]:
    """A ``while``: a branch when its stop depends on a read value."""
    params = eqn.params
    nc, nb = params["cond_nconsts"], params["body_nconsts"]
    cond, body = params["cond_jaxpr"], params["body_jaxpr"]
    carry = list(ins[nc + nb :])
    while True:
        outs = _walk(body, [*ins[nc : nc + nb], *carry], [live] * len(carry))
        new = [a or b for a, b in zip(carry, outs, strict=True)]
        if new == carry:
            break
        carry = new
    if _walk(cond, [*ins[:nc], *carry])[0] and live:
        raise _refuse_branch("while")
    return carry


#: The higher-order primitives `_walk` follows into, by name.
_HIGHER_ORDER: dict[str, Callable[[Any, list[bool], bool], list[bool]]] = {
    **dict.fromkeys(_CALLS, _walk_call),
    "cond": _walk_cond,
    "scan": _walk_scan,
    "while": _walk_while,
}


# --- values: stop_gradient as identity, cotangents, NaN reach, curvature -----------


def _identity_stop_gradient(jaxpr: Any) -> Any:
    """``jaxpr`` with every ``stop_gradient`` (in every sub-jaxpr too) the identity.

    A structural rewrite: each ``stop_gradient`` equation is dropped and its
    output renamed to its operand.
    """
    rename: dict[Any, Any] = {}

    def atom(v: Any) -> Any:
        return v if isinstance(v, jcore.Literal) else rename.get(v, v)

    eqns = []
    for eqn in jaxpr.eqns:
        invars = [atom(v) for v in eqn.invars]
        if eqn.primitive.name == "stop_gradient":
            rename.update(zip(eqn.outvars, invars, strict=True))
            continue
        eqns.append(eqn.replace(invars=invars, params=_stripped_params(eqn.params)))
    return jaxpr.replace(eqns=eqns, outvars=[atom(v) for v in jaxpr.outvars])


def _stripped_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(params)
    for key, index, sub in _jaxprs(params):
        stripped = (
            sub.replace(jaxpr=_identity_stop_gradient(sub.jaxpr))
            if isinstance(sub, _CLOSED)
            else _identity_stop_gradient(sub)
        )
        if index is None:
            out[key] = stripped
        else:
            items = list(out[key])
            items[index] = stripped
            value = out[key]
            out[key] = (
                type(value)(*items) if hasattr(value, "_fields") else tuple(items)
            )
    return out


#: The compiled functions of recent objectives, by their pruned jaxpr's text
#: (`_compiled`): a refinement cycle, or the next estimate of one result,
#: reuses them.
_COMPILED: collections.OrderedDict[str, _Compiled] = collections.OrderedDict()
_MAX_COMPILED = 32


class _Compiled:
    """The stripped jaxpr of one pruned jaxpr, its jitted NaN reach and Hessian.

    Every function takes the jaxpr's constants and its (live) inputs as
    arguments, so one entry serves every result whose objective traces to
    the same jaxpr text: the shapes and dtypes of what it reads, and its
    operations.
    """

    def __init__(self, pruned: Any) -> None:
        self.stripped = _identity_stop_gradient(pruned)
        self.walked = False
        self._nans: dict[tuple[int, ...], Callable[..., jax.Array]] = {}
        self._hessians: dict[tuple[Any, ...], Callable[..., jax.Array]] = {}

    def _evaluate(self, consts: Sequence[Any], xs: Sequence[Any]) -> jax.Array:
        return jcore.eval_jaxpr(self.stripped, consts, *xs)[0]

    def value_and_cotangents(
        self, consts: Sequence[Any], xs: Sequence[Any]
    ) -> tuple[jax.Array, Any]:
        """The value and its cotangents, eagerly: as the objective computes them.

        Not jitted: a fused evaluation may round a zero cotangent (an
        objective at its target) to 1e-16.
        """
        value, pullback = jax.vjp(lambda *ys: self._evaluate(consts, ys), *xs)
        return value, pullback(jnp.ones((), value.dtype))

    def nan_reach(self, positions: tuple[int, ...]) -> Callable[..., jax.Array]:
        """``f(consts, xs)``: per position, whether its NaN reaches the value.

        The pruned jaxpr's own NaN propagation (`stop_gradient` is the
        identity on values, so the stripped one's is the same).
        """
        found = self._nans.get(positions)
        if found is None:

            def reach(consts: Sequence[Any], xs: Sequence[Any]) -> jax.Array:
                def one(k: jax.Array) -> jax.Array:
                    ys = list(xs)
                    for j, p in enumerate(positions):
                        ys[p] = jnp.where(k == j, jnp.full_like(ys[p], jnp.nan), ys[p])
                    return self._evaluate(consts, ys)

                return jnp.isnan(jax.vmap(one)(jnp.arange(len(positions))))

            found = self._nans[positions] = jax.jit(reach)
        return found

    def hessian(
        self, positions: tuple[int, ...], is_complex: tuple[bool, ...]
    ) -> Callable[..., jax.Array]:
        """``f(consts, xs, v)``: the Hessian of the value over the (Re, Im) ``v``."""
        key = (positions, is_complex)
        found = self._hessians.get(key)
        if found is None:
            widths = [2 if c else 1 for c in is_complex]
            starts = np.concatenate([[0], np.cumsum(widths)]).astype(int)

            def real_function(
                v: jax.Array, consts: Sequence[Any], xs: Sequence[Any]
            ) -> jax.Array:
                ys = list(xs)
                for k, p in enumerate(positions):
                    s = starts[k]
                    if is_complex[k]:
                        ys[p] = (v[s] + 1j * v[s + 1]).astype(ys[p].dtype)
                    else:
                        ys[p] = v[s].astype(ys[p].dtype)
                return jnp.real(self._evaluate(consts, ys))

            def hessian(
                consts: Sequence[Any], xs: Sequence[Any], v: jax.Array
            ) -> jax.Array:
                return jax.hessian(real_function)(v, consts, xs)

            found = self._hessians[key] = jax.jit(hessian)
        return found


def _compiled(pruned: Any) -> _Compiled:
    """The cache entry of ``pruned`` (`_COMPILED`), made on first use."""
    key = str(pruned)
    found = _COMPILED.get(key)
    if found is None:
        found = _COMPILED[key] = _Compiled(pruned)
        while len(_COMPILED) > _MAX_COMPILED:
            _COMPILED.popitem(last=False)
    else:
        _COMPILED.move_to_end(key)
    return found


def _check_untraced(value: jax.Array) -> None:
    """Raise when the objective's value carries a tracer of an enclosing transform."""
    if is_traced(value) or is_staging():
        raise NonDifferentiableError(
            "The objective's value carries a tracer of an enclosing jax.grad, "
            "jax.jit or jax.vmap: mesh refinement estimates a fixed design, "
            "which has no derivative. Estimate outside the transformed "
            "function, with concrete values."
        )


def _nan_reads(
    compiled: _Compiled,
    consts: Sequence[Any],
    xs: list[Any],
    candidates: list[int],
) -> list[int]:
    """The candidate positions (in ``xs``) whose NaN reaches the objective's value.

    Vectorized over the candidates and jitted: a loop over them is quadratic
    in the entries.
    """
    if not candidates:
        return []
    reached = np.asarray(compiled.nan_reach(tuple(candidates))(consts, xs))
    return [p for p, r in zip(candidates, reached, strict=True) if r]


def _curvature(
    compiled: _Compiled, consts: Sequence[Any], xs: list[Any], read: list[int]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """``||H_ab||_2`` over the (Re, Im) blocks of the read positions ``read``.

    Also each read's own block ``H_aa`` (`Objective.diagonal`).
    """
    if not read:
        return np.zeros((0, 0)), ()
    is_complex = tuple(bool(jnp.iscomplexobj(xs[p])) for p in read)
    widths = [2 if c else 1 for c in is_complex]
    starts = np.concatenate([[0], np.cumsum(widths)])
    v0 = []
    for k, p in enumerate(read):
        x = jnp.asarray(xs[p])
        v0 += [jnp.real(x), jnp.imag(x)] if is_complex[k] else [jnp.real(x)]
    hessian = np.asarray(
        compiled.hessian(tuple(read), is_complex)(
            consts, xs, jnp.stack(v0).astype(float)
        )
    )
    n = len(read)
    out = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            block = hessian[starts[a] : starts[a + 1], starts[b] : starts[b + 1]]
            out[a, b] = np.linalg.norm(block, 2)
    diagonal = tuple(
        np.nan_to_num(hessian[starts[a] : starts[a + 1], starts[a] : starts[a + 1]])
        for a in range(n)
    )
    return np.nan_to_num(out), diagonal


# --- the mode-aligned view ------------------------------------------------------------

#: The `EigenmodeResult` attributes with one entry per mode along axis 0.
_MODE_AXIS = ("omega", "error", "cluster", "v_g")
_MODE_MAPS = (
    "q_port",
    "participation",
    "bulk_participation",
    "principal_participation",
    "sheet_participation",
    "requested_power_share",
)


def aligned(result: Any, modes: Mapping[int, int]) -> Any:
    """An eigenmode result with its modes in the first mesh's order.

    ``modes`` maps each mode index read on the first mesh to the index of
    the mode followed from it on ``result``'s mesh: that mode moves to the
    index; the other modes keep the solver's order in the remaining slots.
    """
    n = len(np.asarray(result.omega))
    order = _alignment(n, modes)
    if order == list(range(n)):
        return result
    take = np.asarray(order)

    def permute(x: Any) -> Any:
        return None if x is None else jnp.asarray(x)[take]

    changes: dict[str, Any] = {
        name: permute(getattr(result, name)) for name in _MODE_AXIS
    }
    for name in _MODE_MAPS:
        changes[name] = {k: permute(v) for k, v in getattr(result, name).items()}
    changes["e_fields"] = (
        tuple(result.e_fields[i] for i in order) if result.e_fields else ()
    )
    return dataclasses.replace(result, **changes)


def _alignment(n: int, modes: Mapping[int, int]) -> list[int]:
    """Solver index at each aligned position."""
    order: list[int | None] = [None] * n
    for start, index in modes.items():
        order[start] = index
    rest: Iterator[int] = iter(i for i in range(n) if i not in set(modes.values()))
    return [next(rest) if i is None else i for i in order]
