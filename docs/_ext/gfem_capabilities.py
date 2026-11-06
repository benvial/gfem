"""The Capabilities page's claims, derived from the tests at docs build time.

The gradient matrix (problem type x gradient input) and the raises list are
never typed by hand: they come from the verification tests' module-level
data, imported by path without running a solve: ``CELLS``
(tests/verification/test_gradient_matrix.py), ``EXPECTED`` and
``OUT_OF_SCOPE`` (tests/verification/test_gradient_matrix_raise_table.py).
The table of what raises above one rank comes the same way: its rows are the
cases of tests/mpi_cases.py, its text gfem's own refusal messages.

A cell input with the suffix ``_axes`` (``eps_r_axes``) is the same
property per principal value of a field with fixed axes: it folds into its
property's column, whose cells then name the forms their tests cover
("isotropic and principal values").

Every claim that cannot be derived raises `CapabilitiesError`, which the
Sphinx hook turns into an `ExtensionError`: a problem key or input kind the
mapping below does not know, a grid cell with no outcome or with
conflicting ones, a missing ``OUT_OF_SCOPE`` row the grid needs, and an
``OUT_OF_SCOPE`` row off the grid with no site label (or a label with no row).

`derive` returns one plain dict (lists, dicts, strings and numbers). The
directives on docs/capabilities.md (the matrix, the raises list in the site's
words of ``RAISES_LABELS``) and docs/verification.md (the matrix's provenance
sentence) render it, and every page template
gets it as ``gfem_capabilities`` (the template context, as if set in
``html_context``) for the front page's compact matrix, which shows the rows
marked ``front`` only. Only
`setup` imports Sphinx, so the derivation runs (and is unit-tested) without
it.
"""

import dataclasses
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX_TEST = "tests/verification/test_gradient_matrix.py"
RAISE_TEST = "tests/verification/test_gradient_matrix_raise_table.py"
REFUSAL_CASES = "tests/mpi_cases.py"
CAPABILITIES_DOC = "capabilities"
VERIFICATION_DOC = "verification"  # renders the matrix's provenance sentence
FRONT_DOC = "index"  # the gfem-home layout renders the compact matrix
DISTRIBUTED_DOC = "recipes/distributed-solve"
CONTEXT_KEY = "gfem_capabilities"  # the template context name


class CapabilitiesError(Exception):
    """A Capabilities claim cannot be derived from the tests."""


# --- Mapping: test vocabulary -> site vocabulary ------------------------------


@dataclasses.dataclass(frozen=True)
class Row:
    """A row of the grid: one site problem type."""

    label: str
    detail: str  # the headline result, as the row header's second line
    recipe: str = ""  # docname of the recipe that backs the row
    front: bool = True  # also a row of the front page's compact matrix


@dataclasses.dataclass(frozen=True)
class Problem:
    """A problem key of the tests: its row, headline result leaf, port kind."""

    row: str
    headline: str  # the result leaf of EXPECTED that answers the row
    ports: str = ""  # port kind of its verification cells, if it has ports


ROWS = {
    "driven": Row(
        "Driven",
        "S-parameters, uniform and adaptive sweep",
        recipe="recipes/s-parameter-gradient",
    ),
    "eigenmode": Row(
        "Eigenmode", "frequency and Q", recipe="recipes/eigenmode-gradient"
    ),
    "electrostatic": Row(
        "Electrostatic", "capacitance", recipe="recipes/capacitance-gradient"
    ),
    "port_mode": Row("Port mode problem", "2D, wave-port modes"),
    # periodic models: one path per row, so each row lists its own solvers
    # and sweeps (Capabilities page only)
    "driven_periodic_plain": Row(
        "Driven, plain periodic",
        "S-parameters",
        recipe="recipes/periodic-cell",
        front=False,
    ),
    "driven_periodic_bloch": Row(
        "Driven, Bloch",
        "S-parameters at a Bloch wave vector",
        recipe="recipes/periodic-cell",
        front=False,
    ),
    "eigenmode_periodic_plain": Row(
        "Eigenmode, plain periodic",
        "frequency and Q",
        recipe="recipes/periodic-cell",
        front=False,
    ),
    "eigenmode_periodic_bloch": Row(
        "Eigenmode, Bloch",
        "band frequencies and Q",
        recipe="recipes/periodic-cell",
        front=False,
    ),
}
COLUMNS = {
    "eps_r": Row("Permittivity", ""),
    "conductivity": Row("Conductivity", ""),
    "fill": Row("Sheet fill", "", "recipes/sheet-design"),
    "nodes": Row("Node coordinates", "", "recipes/shape-gradient"),
}
PROBLEMS = {
    "driven": Problem("driven", "s.values", "lumped"),
    "driven_wave": Problem("driven", "s.values", "wave"),
    "driven_wave_shared": Problem("driven", "s.values", "wave"),
    "driven_sheet": Problem("driven", "s.values", "lumped"),
    "driven_adaptive": Problem("driven", "s.values", "lumped"),
    "driven_periodic_plain": Problem("driven_periodic_plain", "s.values", "lumped"),
    "driven_periodic_bloch": Problem("driven_periodic_bloch", "s.values", "lumped"),
    "eigenmode": Problem("eigenmode", "omega"),
    "eigenmode_sheet": Problem("eigenmode", "omega"),
    "eigenmode_periodic_plain": Problem("eigenmode_periodic_plain", "omega"),
    "eigenmode_periodic_bloch": Problem("eigenmode_periodic_bloch", "omega"),
    "electrostatic": Problem("electrostatic", "capacitance"),
    "port_mode": Problem("port_mode", "gamma"),
    "port_mode_sheet": Problem("port_mode", "gamma"),
}
# CELLS input names, and the material properties of EXPECTED's input kinds
# ("material", <region>, <property>); None: known, but not a grid column.
# A suffix AXES names the property per principal value (fixed axes).
AXES = "_axes"
CELL_INPUTS = {name: name for name in COLUMNS} | {
    f"{name}{AXES}": name for name in ("eps_r", "conductivity")
}
# The value forms of a material column's cells, by cell input.
FORMS = {"eps_r": "isotropic", "conductivity": "isotropic"} | {
    f"{name}{AXES}": "principal values" for name in ("eps_r", "conductivity")
}
MATERIAL_PROPERTIES = {
    "eps_r": "eps_r",
    "conductivity": "conductivity",
    "mu_r": None,
    "tan_delta": None,
    "thermal_conductivity": None,
    "density": None,
    "specific_heat": None,
}
SOLVERS = {"direct": "direct solver", "bddc_gmres": "BDDC + GMRES"}
# OUT_OF_SCOPE rows the grid needs (by their label): the row's cell raises.
GRID_LABELS = {"electrostatic x sheet fill": ("electrostatic", "fill")}
#: What each raises row asks for, in the site's words, by the test's label (the
#: match key). A row off the grid with no label here, or a label with no row,
#: fails the build.
RAISES_LABELS = {
    "driven x fill of a sheet beside a wave port": (
        "Driven S-parameters, with respect to the fill of a sheet that shares "
        "an edge with a wave port's face"
    ),
    "driven x material on a wave port face": (
        "Driven S-parameters, with respect to the material of a region with "
        "elements on a wave port's face"
    ),
    "driven x material field with axes on a wave port face": (
        "Driven S-parameters, with respect to the principal values of a "
        "`MaterialField` with axes, in a region with elements on a wave port's face"
    ),
    "material axes (orientation)": (
        "Any result, with respect to a material's principal axes"
    ),
    "values of an anisotropic Material": (
        "Any result, with respect to the values of an anisotropic `Material`"
    ),
    "nodes beside an anisotropic absorbing face": (
        "Any result, with respect to node coordinates, when an `Absorbing` "
        "surface borders an anisotropic material"
    ),
    "driven x material on a shared wave port face": (
        "Driven S-parameters, with respect to the material of a region with "
        "elements on a shared wave port surface"
    ),
    "driven x nodes of a port surface": (
        "Driven S-parameters, with respect to node coordinates on a port surface"
    ),
    "eigenmode x nodes of a port surface": (
        "Eigenmode results, with respect to node coordinates on a port surface"
    ),
    "greedy adaptive sweep under a transformation": (
        '`Driven(sweep="adaptive")` without `samples`, inside `jax.grad`, '
        "`jax.jit` or `jax.vmap`"
    ),
    "forward mode": "Forward mode: `jax.jvp` of a solve",
    "probe x node coordinates": (
        "A field value at points (`Field.at`), with respect to node coordinates"
    ),
    "driven x wave_vector": (
        "Driven S-parameters, with respect to the Bloch wave vector"
    ),
    "periodic model x node move that breaks a pair": (
        "A node move that breaks a `Periodic` pair (the two surfaces no longer match)"
    ),
    "boundary_morph x paired moving surface": (
        "`boundary_morph` with a surface of a `Periodic` pair as the moving surface"
    ),
    "Bloch driven x adaptive sweep": (
        '`Driven(sweep="adaptive")` with a Bloch `wave_vector`'
    ),
    "eigenmode x wave_vector": (
        "Eigenmode results, with respect to the Bloch wave vector"
    ),
    "Bloch eigenmode x bddc_gmres": (
        '`Eigenmode` with a Bloch `wave_vector` on `linear_solver="bddc_gmres"`'
    ),
}
STATUS_TEXT = {"verified": "FD-verified", "zero": "exact zero", "raises": "raises"}
OUTCOMES = {"gradient", "zero", "raises"}
#: How a gfem error raised in the backward pass reaches the caller.
GRADIENT = "while the gradient is computed"
WRAPPED = f"{GRADIENT}, inside a `jax.errors.JaxRuntimeError`"
#: The label of the recipes' "When a gradient raises" section, which says how
#: to catch an error raised while the gradient is computed; each such row
#: links it (`links_gradient_errors`).
GRADIENT_ERRORS = "gradient-errors"


def forms_text(forms: list[str]) -> str:
    """A column's value forms as a cell detail ("" for isotropic only)."""
    if forms == ["isotropic", "principal values"]:
        return "isotropic and principal values"
    if forms == ["principal values"]:
        return "principal values only"
    return ""


def _input_column(kind) -> str | None:
    """The grid column of an EXPECTED input kind (None: off the grid)."""
    if not isinstance(kind, tuple):
        kind = (kind,)
    if kind == ("nodes",):
        return "nodes"
    if len(kind) == 3 and kind[0] == "material" and kind[2] in MATERIAL_PROPERTIES:
        return MATERIAL_PROPERTIES[kind[2]]
    if len(kind) == 4 and kind[0] == "boundary" and kind[2:] == ("fill", "fill"):
        return "fill"
    raise CapabilitiesError(
        f"{RAISE_TEST}: input kind {kind!r} is not in the Capabilities mapping "
        "(docs/_ext/gfem_capabilities.py)"
    )


def _problem(key: str, source: str) -> Problem:
    if key not in PROBLEMS:
        raise CapabilitiesError(
            f"{source}: problem {key!r} is not in the Capabilities mapping "
            "(docs/_ext/gfem_capabilities.py): add it to PROBLEMS"
        )
    return PROBLEMS[key]


def _cell_column(cell) -> str:
    if cell.input not in CELL_INPUTS:
        raise CapabilitiesError(
            f"{MATRIX_TEST}: input {cell.input!r} of cell {cell} is not in the "
            "Capabilities mapping (docs/_ext/gfem_capabilities.py)"
        )
    return CELL_INPUTS[cell.input]


# --- Gradient matrix ----------------------------------------------------------


def gradient_matrix(cells, expected, out_of_scope) -> dict:
    """The grid, counts and raises list from the verification tests' data.

    Args:
        cells: ``CELLS``, each with ``problem``, ``input``, ``solver``,
            ``sweep`` and ``variant``.
        expected: ``EXPECTED``, ``(problem, result leaf, input kind) ->
            "gradient" | "zero" | "raises"``.
        out_of_scope: ``OUT_OF_SCOPE``, rows ``(label, call, error, match)``.

    Returns:
        ``columns``, ``rows`` (each with one cell per column), ``counts``,
        ``provenance`` and ``raises``.

    Raises:
        CapabilitiesError: when a claim cannot be derived.
    """
    verified: dict[tuple[str, str], list] = {}
    for cell in cells:
        if cell.solver not in SOLVERS:
            raise CapabilitiesError(
                f"{MATRIX_TEST}: solver {cell.solver!r} of cell {cell} is not in "
                "the Capabilities mapping (docs/_ext/gfem_capabilities.py)"
            )
        problem = _problem(cell.problem, MATRIX_TEST)
        verified.setdefault((problem.row, _cell_column(cell)), []).append(cell)

    claimed: dict[tuple[str, str], set[str]] = {}
    headlines: set[str] = set()
    for (key, leaf, kind), outcome in expected.items():
        problem = _problem(key, RAISE_TEST)
        column = _input_column(kind)
        if outcome not in OUTCOMES:
            raise CapabilitiesError(
                f"{RAISE_TEST}: outcome {outcome!r} of {(key, leaf, kind)!r} is not "
                f"one of {sorted(OUTCOMES)}"
            )
        if leaf != problem.headline:
            continue
        headlines.add(key)
        if column is not None:
            claimed.setdefault((problem.row, column), set()).add(outcome)
    missing = sorted({k for k, _, _ in expected} - headlines)
    if missing:
        leaves = ", ".join(f"{k!r} ({PROBLEMS[k].headline!r})" for k in missing)
        raise CapabilitiesError(
            f"{RAISE_TEST}: no row for the headline result leaf of problem {leaves}"
        )

    labels = {}
    for row in out_of_scope:
        label, error, match = _out_of_scope_row(row)
        labels[label] = (error, match)

    rows = []
    for row_key, row in ROWS.items():
        ports_of_row = {
            PROBLEMS[c.problem].ports
            for (r, _), cs in verified.items()
            if r == row_key
            for c in cs
        } - {""}
        out = [
            _resolve(
                row_key,
                column,
                cells=verified.get((row_key, column), []),
                claimed=claimed.get((row_key, column), set()),
                labels=labels,
                ports_of_row=ports_of_row,
            )
            for column in COLUMNS
        ]
        rows.append(
            {
                "key": row_key,
                "label": row.label,
                "detail": row.detail,
                "recipe": row.recipe,
                "front": row.front,
                "cells": out,
            }
        )

    counts = {"cells": len(cells)} | {
        s: sum(c.solver == s for c in cells) for s in SOLVERS
    }
    provenance = (
        f"{counts['cells']} cells ({counts['direct']} on the direct solver, "
        f"{counts['bddc_gmres']} repeated on BDDC + GMRES), each checked along "
        "3 random directions against fourth-order central differences"
    )
    off_grid = [label for label in labels if label not in GRID_LABELS]
    unlabelled = [label for label in off_grid if label not in RAISES_LABELS]
    stale = [label for label in RAISES_LABELS if label not in off_grid]
    if unlabelled or stale:
        raise CapabilitiesError(
            f"{RAISE_TEST}: OUT_OF_SCOPE rows with no site label {unlabelled}, "
            f"site labels with no row {stale} (RAISES_LABELS in "
            "docs/_ext/gfem_capabilities.py)"
        )
    raises = [
        {"key": label, "label": RAISES_LABELS[label], **_error(*labels[label])}
        for label in off_grid
    ]
    return {
        "columns": [
            {"key": k, "label": c.label, "recipe": c.recipe} for k, c in COLUMNS.items()
        ],
        "rows": rows,
        "counts": counts,
        "provenance": provenance,
        "raises": raises,
    }


def _resolve(row, column, *, cells, claimed, labels, ports_of_row) -> dict:
    """One grid cell: FD-verified, exact zero or raises (and only one)."""
    where = f"grid cell {ROWS[row].label} x {COLUMNS[column].label}"
    statuses = set()
    if cells:
        statuses.add("verified")
    if "gradient" in claimed and not cells:
        raise CapabilitiesError(
            f"{where}: {RAISE_TEST} expects a gradient that no cell of "
            f"{MATRIX_TEST} verifies against finite differences"
        )
    statuses |= {{"gradient": "verified"}.get(o, o) for o in claimed}
    grid_labels = [lab for lab, cell in GRID_LABELS.items() if cell == (row, column)]
    error = ""
    for label in grid_labels:
        if label in labels:
            statuses.add("raises")
            error = _error(*labels[label])["error"]
    if not statuses:
        missing = [lab for lab in grid_labels if lab not in labels]
        hint = f" (the OUT_OF_SCOPE row {missing[0]!r} is gone)" if missing else ""
        raise CapabilitiesError(f"{where}: no outcome can be derived{hint}")
    if len(statuses) > 1:
        raise CapabilitiesError(
            f"{where}: conflicting outcomes {sorted(statuses)} between "
            f"{MATRIX_TEST} and {RAISE_TEST}"
        )
    (status,) = statuses
    ports = [p for p in dict.fromkeys(PROBLEMS[c.problem].ports for c in cells) if p]
    forms = sorted({FORMS[c.input] for c in cells if c.input in FORMS})
    note = f"{' and '.join(ports)} ports" if ports and set(ports) < ports_of_row else ""
    label = STATUS_TEXT[status]
    if error:
        label += f" {error}"
    elif note:
        label += f", {note}"
    return {
        "column": column,
        "status": status,
        "text": STATUS_TEXT[status],
        "label": label,
        "note": note,
        "error": error,
        "cells": len(cells),
        "solvers": [SOLVERS[s] for s in dict.fromkeys(c.solver for c in cells)],
        "sweeps": [s for s in dict.fromkeys(c.sweep for c in cells) if s],
        "ports": ports,
        "forms": forms,
        "forms_text": forms_text(forms),
        "source": MATRIX_TEST if cells else RAISE_TEST,
    }


def _out_of_scope_row(row) -> tuple[str, type, str]:
    if not (
        isinstance(row, tuple)
        and len(row) == 4
        and isinstance(row[0], str)
        and isinstance(row[2], type)
    ):
        raise CapabilitiesError(
            f"{RAISE_TEST}: OUT_OF_SCOPE row {row!r} is not (label, call, error, match)"
        )
    return row[0], row[2], str(row[3])


def _error(error: type, match: str) -> dict:
    """The error a raises row names.

    A gfem error raised in the backward pass reaches the caller wrapped in
    JAX's runtime error (a ``ValueError`` once a jitted function has already
    succeeded, which no test of the raise table reaches), so its own name (the
    head of the test's ``match``) is the one to publish.
    """
    if error.__name__ == "JaxRuntimeError":
        head = re.match(r"\w+Error\b", match)
        if head:
            return {"error": head.group(), "when": WRAPPED}
        return {"error": error.__name__, "when": GRADIENT}
    return {"error": error.__name__, "when": ""}


def links_gradient_errors(row: dict) -> bool:
    """Whether a raises row links the "When a gradient raises" section."""
    return row["when"].startswith(GRADIENT)


# --- What raises above one rank --------------------------------------------------

#: What each refusal case asks for, in the site's words. A case with no label here,
#: or a label with no case, fails the build.
REFUSAL_LABELS = {
    "adaptive_sweep": '`Driven(sweep="adaptive")`, with or without `samples`',
    "eigenmode": "`Eigenmode`, nonlinear ones included",
    "electrostatic": "`Electrostatic`",
    "periodic": "A `Periodic` boundary on a `Driven` problem, on any number of axes",
    "bloch": "`Driven(wave_vector=...)`",
    "frequency_workers": "`frequency_workers` above 1, on more than one frequency",
    "device": '`device="cuda"`',
    "bddc_order_1": '`linear_solver="bddc_gmres"` at `order=1`',
    "bddc_order_1_auto": '`linear_solver="auto"` at `order=1`',
    "field_export": "`Field.to_meshio`, `Field.to_pyvista`",
    "load_result_fields": "`load_result` of a bundle with fields",
    "power_flow": "`power_flow`",
}


def distributed_refusals(keys, methods, guards, feature, messages) -> list[dict]:
    """One row per refusal case: what it asks for, its ``feature``, its message.

    Args:
        keys: The solves refused, ``tests/mpi_cases.py``'s ``KEYS``.
        methods: The method calls refused, its ``METHOD_KEYS``.
        guards: The refusal keys no public call reaches, its ``GUARD_KEYS``. No row.
        feature: The ``feature`` a case raises.
        messages: gfem's own refusal messages by ``feature``.

    Raises:
        CapabilitiesError: The cases, the labels and the messages differ.
    """
    cases = [*keys, *methods]
    if set(cases) != set(REFUSAL_LABELS):
        raise CapabilitiesError(
            f"{REFUSAL_CASES}: the cases {sorted(cases)} are not the labelled ones "
            f"{sorted(REFUSAL_LABELS)}"
        )
    features = {feature(case) for case in cases} | set(guards)
    if features != set(messages):
        raise CapabilitiesError(
            f"{REFUSAL_CASES}: the cases raise {sorted(features)}, and gfem has "
            f"messages for {sorted(messages)}"
        )
    return [
        {
            "label": REFUSAL_LABELS[case],
            "feature": feature(case),
            "message": messages[feature(case)],
        }
        for case in cases
    ]


# --- Loading the verification tests --------------------------------------------


def load_refusals(repo: Path) -> list[dict]:
    """The rows of `distributed_refusals`, from the cases under ``repo``."""
    from gfem import _mpi

    paths = [str(repo / "tests"), str(repo / "tools")]
    sys.path[:0] = paths
    try:
        cases = _import(repo / REFUSAL_CASES)
    finally:
        for p in paths:
            sys.path.remove(p)
    try:
        return distributed_refusals(
            cases.KEYS,
            cases.METHOD_KEYS,
            cases.GUARD_KEYS,
            cases.feature,
            _mpi._REFUSALS,  # noqa: SLF001
        )
    except AttributeError as exc:
        raise CapabilitiesError(f"the refusal cases are gone: {exc}") from exc


def load_verification(repo: Path) -> tuple[list, dict, list]:
    """``CELLS``, ``EXPECTED`` and ``OUT_OF_SCOPE``, imported by path.

    The modules import their helpers from tests/ and tools/ (as pytest's
    ``pythonpath`` does), and gfem, JAX and pytest; nothing is solved.
    """
    paths = [str(repo / "tests"), str(repo / "tools")]
    sys.path[:0] = paths
    try:
        matrix = _import(repo / MATRIX_TEST)
        table = _import(repo / RAISE_TEST)
    finally:
        for p in paths:
            sys.path.remove(p)
    try:
        return list(matrix.CELLS), dict(table.EXPECTED), list(table.OUT_OF_SCOPE)
    except AttributeError as exc:
        raise CapabilitiesError(f"verification data is gone: {exc}") from exc


def _import(path: Path):
    name = f"_gfem_capabilities_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None or not path.is_file():
        raise CapabilitiesError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CapabilitiesError(f"cannot import {path}: {exc!r}") from exc
    return module


def derive(repo: Path = REPO) -> dict:
    """Every Capabilities claim, from the tests under ``repo``.

    Returns:
        ``columns``, ``rows``, ``counts``, ``provenance`` and ``raises`` (see
        `gradient_matrix`), and ``refusals`` (see `distributed_refusals`).
    """
    return gradient_matrix(*load_verification(repo)) | {"refusals": load_refusals(repo)}


# --- Sphinx --------------------------------------------------------------------


def setup(app):
    """Register the directives and derive the claims once per build."""
    from sphinx.errors import ExtensionError

    def derive_claims(app) -> None:
        try:
            data = derive(REPO)
        except CapabilitiesError as exc:
            raise ExtensionError(
                f"Capabilities: {exc}", modname="gfem_capabilities"
            ) from exc
        app.env.gfem_capabilities = data  # the directives' source

    def add_context(app, _pagename, _templatename, context, _doctree) -> None:
        # Every page template sees the claims (the gfem-home layout renders
        # the compact matrix from them).
        context[CONTEXT_KEY] = app.env.gfem_capabilities

    def reread(_app, env, *_changes) -> list[str]:
        # These pages render the tests' data, not their own source: read them
        # again on every build so an incremental build never serves stale claims.
        pages = (CAPABILITIES_DOC, VERIFICATION_DOC, FRONT_DOC, DISTRIBUTED_DOC)
        return [d for d in pages if d in env.found_docs]

    app.add_config_value("gfem_capabilities_source_url", "", "html", str)
    app.connect("builder-inited", derive_claims)
    app.connect("env-get-outdated", reread)
    app.connect("html-page-context", add_context)
    for name, directive in _directives().items():
        app.add_directive(name, directive)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}


def _directives() -> dict:
    """The directives (defined here so the module imports without Sphinx)."""
    from docutils import nodes
    from sphinx import addnodes
    from sphinx.util.docutils import SphinxDirective

    class _Claims(SphinxDirective):
        has_content = False

        @property
        def data(self) -> dict:
            return self.env.gfem_capabilities

        def source(self, file: str, *children):
            """``children`` linked to ``file`` once a source URL is set."""
            base = self.config.gfem_capabilities_source_url
            if not base:
                return list(children)
            ref = nodes.reference("", "", refuri=f"{base}{file}", internal=False)
            ref.extend(children)
            return [ref]

        def ref(self, label: str, *children):
            """``children`` linked to the section labelled ``label``."""
            xref = addnodes.pending_xref(
                "",
                refdomain="std",
                reftype="ref",
                reftarget=label,
                refdoc=self.env.docname,
                refexplicit=True,
                refwarn=True,
            )
            xref.extend(children)
            return xref

        def doc(self, docname: str, *children):
            """``children`` linked to the page ``docname``."""
            if not docname:
                return list(children)
            xref = addnodes.pending_xref(
                "",
                refdomain="std",
                reftype="doc",
                reftarget=f"/{docname}",
                refdoc=self.env.docname,
                refexplicit=True,
                refwarn=True,
            )
            xref.extend(children)
            return [xref]

    class GradientMatrix(_Claims):
        """The full gradient matrix (its provenance is `GradientChecks`)."""

        def run(self):
            data = self.data
            table = nodes.table(classes=["gfem-matrix"])
            group = nodes.tgroup(cols=1 + len(data["columns"]))
            table += group
            group += nodes.colspec(colwidth=24, stub=1)
            group.extend(nodes.colspec(colwidth=19) for _ in data["columns"])
            head = nodes.row()
            head += _entry(nodes.Text("Problem"))
            for column in data["columns"]:
                head += _entry(*self.doc(column["recipe"], nodes.Text(column["label"])))
            group += nodes.thead("", head)
            body = nodes.tbody()
            for row in data["rows"]:
                line = nodes.row()
                line += _entry(
                    nodes.strong(
                        "", "", *self.doc(row["recipe"], nodes.Text(row["label"]))
                    ),
                    nodes.Text(" "),
                    nodes.inline(text=row["detail"], classes=["gfem-detail"]),
                )
                for cell in row["cells"]:
                    line += _entry(*self.cell(cell))
                body += line
            group += body
            return [table]

        def cell(self, cell: dict) -> list:
            mark = nodes.inline(classes=["gfem-mark", f"gfem-{cell['status']}"])
            mark += nodes.inline(classes=["gfem-glyph"])
            mark.extend(self.source(cell["source"], nodes.Text(cell["label"])))
            out = [mark]
            if cell["status"] == "verified":
                parts = [cell["forms_text"]] if cell["forms_text"] else []
                parts.append(", ".join(cell["solvers"]))
                if cell["sweeps"]:
                    parts.append(f"{' and '.join(cell['sweeps'])} sweep")
                if cell["ports"]:
                    parts.append(f"{' and '.join(cell['ports'])} ports")
                out += [
                    nodes.Text(" "),
                    nodes.inline(text=" · ".join(parts), classes=["gfem-detail"]),
                ]
            return out

    class GradientChecks(_Claims):
        """How many cells the matrix checks, and how: one generated sentence."""

        def run(self):
            text = f"{self.data['provenance']}."
            return [nodes.paragraph(text=text[0].upper() + text[1:])]

    class Raises(_Claims):
        """The out-of-scope pairs off the grid, each with its named error."""

        def run(self):
            items = nodes.bullet_list(classes=["gfem-status"])
            for row in self.data["raises"]:
                tag = nodes.inline(classes=["tag", "raises"])
                tag += nodes.Text("raises ")
                tag += nodes.literal(text=row["error"])
                if row["when"]:
                    tag.extend(_with_code(f" {row['when']}"))
                para = nodes.paragraph()
                what = nodes.inline(classes=["what"])
                what.extend(_with_code(row["label"]))
                para += what
                para += nodes.Text(" ")
                para.extend(self.source(RAISE_TEST, tag))
                if links_gradient_errors(row):  # outside the source link: no nesting
                    para += nodes.Text(" (")
                    para += self.ref(GRADIENT_ERRORS, nodes.Text("how to catch it"))
                    para += nodes.Text(")")
                items += nodes.list_item("", para)
            return [items]

    class DistributedRefusals(_Claims):
        """What raises above one rank, with its ``feature`` and gfem's message."""

        def run(self):
            table = nodes.table()
            group = nodes.tgroup(cols=3)
            table += group
            group += nodes.colspec(colwidth=30)
            group += nodes.colspec(colwidth=15)
            group += nodes.colspec(colwidth=55)
            head = nodes.row()
            for title in ("You ask for", "feature", "The error says"):
                head += _entry(nodes.Text(title))
            group += nodes.thead("", head)
            body = nodes.tbody()
            for row in self.data["refusals"]:
                line = nodes.row()
                line += _entry(*_with_code(row["label"]))
                line += _entry(nodes.literal(text=row["feature"]))
                line += _entry(nodes.Text(row["message"]))
                body += line
            group += body
            return [table]

    def _with_code(text: str) -> list:
        """``text``'s backticked parts as literals."""
        return [
            nodes.literal(text=part) if i % 2 else nodes.Text(part)
            for i, part in enumerate(text.split("`"))
            if part
        ]

    def _entry(*children):
        return nodes.entry("", nodes.paragraph("", "", *children))

    return {
        "gfem-gradient-matrix": GradientMatrix,
        "gfem-gradient-checks": GradientChecks,
        "gfem-raises": Raises,
        "gfem-distributed-refusals": DistributedRefusals,
    }
