"""The Capabilities page's claims, derived from the tests at docs build time.

The gradient matrix (problem type x gradient input), the raises list and the
agreement-with-Palace tolerances are not typed by hand (ADR 0017):

- the matrix and the raises list come from the verification tests'
  module-level data, imported by path without running a solve:
  ``CELLS`` (tests/verification/test_gradient_matrix.py), ``EXPECTED`` and
  ``OUT_OF_SCOPE`` (tests/verification/test_gradient_matrix_raise_table.py);
- the Palace tolerances are read with `ast` from an allowlist of
  reference-test constants (`AGREEMENT`), never by importing those tests.

Every claim that cannot be derived raises `CapabilitiesError`, which the
Sphinx hook turns into an `ExtensionError`: a problem key or input kind the
mapping below does not know, a grid cell with no outcome or with
conflicting ones, a missing ``OUT_OF_SCOPE`` row the grid needs, and an
allowlisted constant that is missing, not a literal, unused, or named like
a timing figure. Timing has no path to the page: the wall-time assertions
are inline expressions the allowlist cannot name.

`derive` returns one plain dict (lists, dicts, strings and numbers). The
three directives on docs/capabilities.md render it, and every page template
gets it as ``gfem_capabilities`` (the template context, as if set in
``html_context``) for the front page's compact matrix and Palace rows. Only
`setup` imports Sphinx, so the derivation runs (and is unit-tested) without
it.
"""

import ast
import dataclasses
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX_TEST = "tests/verification/test_gradient_matrix.py"
RAISE_TEST = "tests/verification/test_gradient_matrix_raise_table.py"
CAPABILITIES_DOC = "capabilities"
FRONT_DOC = "index"  # the gfem-home layout renders the compact matrix and Palace rows
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
    "driven_sheet": Problem("driven", "s.values", "lumped"),
    "driven_adaptive": Problem("driven", "s.values", "lumped"),
    "eigenmode": Problem("eigenmode", "omega"),
    "eigenmode_sheet": Problem("eigenmode", "omega"),
    "electrostatic": Problem("electrostatic", "capacitance"),
    "port_mode": Problem("port_mode", "gamma"),
    "port_mode_sheet": Problem("port_mode", "gamma"),
}
# CELLS input names, and the material properties of EXPECTED's input kinds
# ("material", <region>, <property>); None: known, but not a grid column.
CELL_INPUTS = {name: name for name in COLUMNS}
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
STATUS_TEXT = {"verified": "FD-verified", "zero": "exact zero", "raises": "raises"}
OUTCOMES = {"gradient", "zero", "raises"}


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
            f"{RAISE_TEST}: no row for the headline result leaf of problem "
            f"{leaves}"
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
    raises = [
        {"label": _display(label), **_error(error, match)}
        for label, (error, match) in labels.items()
        if label not in GRID_LABELS
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
    JAX's runtime error, so its own name (the head of the test's ``match``)
    is the one to publish.
    """
    if error.__name__ == "JaxRuntimeError":
        head = re.match(r"\w+Error\b", match)
        if head:
            return {"error": head.group(), "when": "in the backward pass"}
        return {"error": error.__name__, "when": "in the backward pass"}
    return {"error": error.__name__, "when": ""}


def _display(label: str) -> str:
    return label.replace(" x ", " \N{MULTIPLICATION SIGN} ")


# --- Agreement with Palace ------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Agreement:
    """An allowlisted row: tolerance constants of one reference test.

    ``names`` are module-level constants of ``file`` (relative to the repo),
    one per measured ``quantities`` entry; ``kind`` is ``"relative"`` or
    ``"absolute"``; ``problem`` rows (one per problem type) are also the
    front page's, the others (measurements) are Capabilities only.
    """

    file: str
    names: tuple[str, ...]
    quantities: tuple[str, ...]
    label: str
    detail: str
    kind: str
    problem: bool = True


AGREEMENT = (
    Agreement(
        "tests/reference/test_driven.py",
        ("TOLERANCE",),
        ("max |ΔS|",),
        "Driven, lumped ports",
        "800 µm coplanar line on the IHP SG13G2 stack",
        "absolute",
    ),
    Agreement(
        "tests/reference/test_eigenmode.py",
        ("PARITY_F", "PARITY_Q"),
        ("Re f", "Q"),
        "Eigenmode, port-loaded",
        "4 mm CPW resonator on sapphire",
        "relative",
    ),
    Agreement(
        "tests/reference/test_eigenmode_nonlinear.py",
        ("PARITY_F", "PARITY_Q"),
        ("Re f", "Q"),
        "Nonlinear eigenmode, conductor loss",
        "IHP half-wave resonator near 104 GHz",
        "relative",
    ),
    Agreement(
        "tests/reference/test_electrostatic.py",
        ("RTOL",),
        ("capacitance",),
        "Electrostatic",
        "two-layer plate stack, every capacitance entry",
        "relative",
    ),
    Agreement(
        "tests/reference/test_measurements.py",
        ("PARITY",),
        ("participation",),
        "Interface participation",
        "metal-air, metal-substrate and substrate-air on the CPW resonator",
        "relative",
        problem=False,
    ),
)
# No timing figure reaches the site (no speed claims before the release gate).
TIMING = re.compile(r"WALL|TIME|RSS|PER_SAMPLE|BUDGET|SPEED")


def palace_agreement(repo: Path, allowlist=AGREEMENT) -> list[dict]:
    """The agreement-with-Palace rows, read from the reference tests by AST.

    Raises:
        CapabilitiesError: when an allowlisted constant is named like a
            timing figure, is missing, is not a number literal, or is never
            read by its test.
    """
    rows = []
    for row in allowlist:
        if row.kind not in ("relative", "absolute"):
            raise CapabilitiesError(f"{row.label}: kind {row.kind!r} is unknown")
        values = _constants(repo, row.file, row.names)
        entries = [
            {"name": name, "quantity": q, "value": values[name], "text": _fmt(row, v)}
            for name, q, v in zip(
                row.names, row.quantities, values.values(), strict=True
            )
        ]
        if row.kind == "absolute":
            text = " · ".join(f"{e['quantity']} ≤ {e['text']}" for e in entries)
        elif len(entries) == 1:
            text = f"relative {entries[0]['text']}"
        else:
            text = "relative: " + " · ".join(
                f"{e['quantity']} {e['text']}" for e in entries
            )
        rows.append(
            {
                "label": row.label,
                "detail": row.detail,
                "kind": row.kind,
                "values": entries,
                "text": text,
                "file": row.file,
                "problem": row.problem,
            }
        )
    return rows


def _constants(repo: Path, file: str, names: tuple[str, ...]) -> dict:
    """Module-level number literals of a test, never importing it."""
    for name in names:
        if TIMING.search(name.upper()):
            raise CapabilitiesError(
                f"{file}: {name} looks like a timing constant; timing is never "
                "published on the site"
            )
    path = repo / file
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise CapabilitiesError(f"cannot read {path}: {exc}") from exc
    assigned = _assignments(tree)
    reads = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    out = {}
    for name in names:
        if name not in assigned:
            raise CapabilitiesError(
                f"{file}: allowlisted constant {name} is missing "
                "(docs/_ext/gfem_capabilities.py AGREEMENT)"
            )
        try:
            value = ast.literal_eval(assigned[name])  # type: ignore[arg-type]
        except (ValueError, TypeError, SyntaxError):
            value = None
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise CapabilitiesError(f"{file}: {name} is not a number literal")
        if name not in reads:
            raise CapabilitiesError(
                f"{file}: {name} is unused (only its own assignment): a dead "
                "constant is not a tolerance the test asserts"
            )
        out[name] = value
    return out


def _assignments(tree: ast.Module) -> dict[str, ast.expr | None]:
    """Module-level ``NAME = value`` and ``A, B = a, b`` assignments.

    A name whose value is not one expression of its own (``A, B = f()``)
    maps to None, which `ast.literal_eval` rejects.
    """
    assigned: dict[str, ast.expr | None] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assigned[target.id] = value
            elif isinstance(target, ast.Tuple):
                elts = value.elts if isinstance(value, ast.Tuple) else []
                ok = len(elts) == len(target.elts)
                for i, t in enumerate(target.elts):
                    if isinstance(t, ast.Name):
                        assigned[t.id] = elts[i] if ok else None
    return assigned


def _fmt(row: Agreement, value: float) -> str:
    if row.kind == "relative" and value >= 0.01:
        return f"{value * 100:g} %"
    if 0 < abs(value) < 0.01:  # 1e-3, not 0.001: tolerances read as exponents
        mantissa, exponent = f"{value:.6e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    return f"{value:g}"


# --- Loading the verification tests --------------------------------------------


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
        `gradient_matrix`) and ``palace`` (see `palace_agreement`).
    """
    return gradient_matrix(*load_verification(repo)) | {
        "palace": palace_agreement(repo)
    }


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
        # the compact matrix and the Palace rows from them).
        context[CONTEXT_KEY] = app.env.gfem_capabilities

    def reread(_app, env, *_changes) -> list[str]:
        # Both pages render the tests' data, not their own source: read them
        # again on every build so an incremental build never serves stale claims.
        return [d for d in (CAPABILITIES_DOC, FRONT_DOC) if d in env.found_docs]

    app.add_config_value("gfem_capabilities_source_url", "", "html", str)
    app.connect("builder-inited", derive_claims)
    app.connect("env-get-outdated", reread)
    app.connect("html-page-context", add_context)
    for name, directive in _directives().items():
        app.add_directive(name, directive)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}


def _directives() -> dict:
    """The three directives (defined here so the module imports without Sphinx)."""
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
        """The full gradient matrix and its provenance line."""

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
            provenance = nodes.paragraph(
                text=f"{data['provenance']}.", classes=["gfem-provenance"]
            )
            return [table, provenance]

        def cell(self, cell: dict) -> list:
            mark = nodes.inline(classes=["gfem-mark", f"gfem-{cell['status']}"])
            mark += nodes.inline(classes=["gfem-glyph"])
            mark.extend(self.source(cell["source"], nodes.Text(cell["label"])))
            out = [mark]
            if cell["status"] == "verified":
                parts = [", ".join(cell["solvers"])]
                if cell["sweeps"]:
                    parts.append(f"{' and '.join(cell['sweeps'])} sweep")
                if cell["ports"]:
                    parts.append(f"{' and '.join(cell['ports'])} ports")
                out += [
                    nodes.Text(" "),
                    nodes.inline(text=" · ".join(parts), classes=["gfem-detail"]),
                ]
            return out

    class Raises(_Claims):
        """The out-of-scope pairs off the grid, each with its named error."""

        def run(self):
            items = nodes.bullet_list(classes=["gfem-status"])
            for row in self.data["raises"]:
                tag = nodes.inline(classes=["tag", "raises"])
                tag += nodes.Text("raises ")
                tag += nodes.literal(text=row["error"])
                if row["when"]:
                    tag += nodes.Text(f" {row['when']}")
                para = nodes.paragraph()
                para += nodes.inline(text=row["label"], classes=["what"])
                para += nodes.Text(" ")
                para.extend(self.source(RAISE_TEST, tag))
                items += nodes.list_item("", para)
            return [items]

    class PalaceAgreement(_Claims):
        """One row per reference problem: the tolerance its test asserts."""

        def run(self):
            items = nodes.bullet_list(classes=["gfem-status", "gfem-palace"])
            for row in self.data["palace"]:
                what = nodes.inline(classes=["what"])
                what += nodes.strong(text=row["label"])
                what += nodes.Text(" ")
                what += nodes.inline(text=row["detail"], classes=["gfem-detail"])
                tag = nodes.inline(classes=["tag"])
                tag += nodes.Text(f"{row['text']} ")
                tag += nodes.literal(text=Path(row["file"]).name)
                para = nodes.paragraph()
                para += what
                para += nodes.Text(" ")
                para.extend(self.source(row["file"], tag))
                items += nodes.list_item("", para)
            return [items]

    def _entry(*children):
        return nodes.entry("", nodes.paragraph("", "", *children))

    return {
        "gfem-gradient-matrix": GradientMatrix,
        "gfem-raises": Raises,
        "gfem-palace-agreement": PalaceAgreement,
    }
