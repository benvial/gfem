"""The Capabilities page generator (docs/_ext/gfem_capabilities.py).

The site's gradient claims are derived from the tests at docs build time
(no hand-typed table); a claim that cannot be derived fails the build. These
tests pin the derivation against a grid reviewed by hand, and each drift
guard against a broken input. The generator is imported without Sphinx.
"""

import dataclasses
import importlib.util
from pathlib import Path

import mpi_cases
import pytest

from gfem import _mpi

REPO = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location(
    "gfem_capabilities", REPO / "docs" / "_ext" / "gfem_capabilities.py"
)
caps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(caps)

V, Z, R = "verified", "zero", "raises"

# The grid as reviewed by hand against the tests:
# rows driven, eigenmode, electrostatic, port mode, then the periodic rows
# (plain and Bloch, driven and eigenmode); columns permittivity,
# conductivity, sheet fill, node coordinates.
GRID = {
    "driven": (V, V, V, V),
    "eigenmode": (V, V, V, V),
    "electrostatic": (V, Z, R, V),
    "port_mode": (R, R, R, R),
    "driven_periodic_plain": (V, V, V, V),
    "driven_periodic_bloch": (V, V, V, V),
    "eigenmode_periodic_plain": (V, V, V, V),
    "eigenmode_periodic_bloch": (V, V, V, V),
}
FRONT = ("driven", "eigenmode", "electrostatic", "port_mode")


@pytest.fixture(scope="module")
def derived():
    return caps.derive(REPO)


def _cell(data, row, column):
    (r,) = [r for r in data["rows"] if r["key"] == row]
    (c,) = [c for c in r["cells"] if c["column"] == column]
    return c


def test_matrix_matches_the_reviewed_grid(derived):
    columns = [c["key"] for c in derived["columns"]]
    assert columns == ["eps_r", "conductivity", "fill", "nodes"]
    got = {r["key"]: tuple(c["status"] for c in r["cells"]) for r in derived["rows"]}
    assert got == GRID


def test_counts_come_from_the_cells(derived):
    counts = derived["counts"]
    per_grid_cell = [c["cells"] for r in derived["rows"] for c in r["cells"]]
    assert counts["cells"] == sum(per_grid_cell)
    assert counts["cells"] == counts["direct"] + counts["bddc_gmres"]
    assert counts["direct"] > 0
    assert counts["bddc_gmres"] > 0
    assert derived["provenance"].startswith(
        f"{counts['cells']} cells ({counts['direct']} on the direct solver, "
        f"{counts['bddc_gmres']} repeated on BDDC + GMRES)"
    )


def test_driven_sheet_fill_is_verified_on_lumped_ports_only(derived):
    fill = _cell(derived, "driven", "fill")
    assert fill["label"] == "FD-verified, lumped ports"
    assert _cell(derived, "driven", "eps_r")["ports"] == ["lumped", "wave"]
    assert _cell(derived, "driven", "eps_r")["note"] == ""
    assert _cell(derived, "eigenmode", "fill")["note"] == ""


def test_verified_cells_list_their_solvers_and_sweeps(derived):
    cell = _cell(derived, "driven", "nodes")
    assert cell["solvers"] == ["direct solver", "BDDC + GMRES"]
    assert cell["sweeps"] == ["uniform", "adaptive"]
    assert cell["source"] == "tests/verification/test_gradient_matrix.py"


def test_principal_value_cells_fold_into_their_property_column(derived):
    assert caps.CELL_INPUTS["eps_r_axes"] == "eps_r"
    assert caps.CELL_INPUTS["conductivity_axes"] == "conductivity"
    both = {
        ("driven", "eps_r"),
        ("driven", "conductivity"),
        ("eigenmode", "eps_r"),
        ("eigenmode", "conductivity"),
        ("electrostatic", "eps_r"),
    }
    for row in derived["rows"]:
        for cell in row["cells"]:
            key = (row["key"], cell["column"])
            if key in both:
                assert cell["forms"] == ["isotropic", "principal values"], key
                assert cell["forms_text"] == "isotropic and principal values", key
            else:
                assert cell["forms_text"] == "", key
    # the label (and so the front page's compact matrix) does not change
    assert _cell(derived, "driven", "eps_r")["label"] == "FD-verified"


def test_periodic_rows_carry_their_own_solvers_and_sweeps(derived):
    # one path per row: Bloch eigenmode solves on the direct solver only,
    # a Bloch driven solve on the uniform sweep only
    rows = {r["key"]: r for r in derived["rows"]}
    assert rows["eigenmode_periodic_bloch"]["label"] == "Eigenmode, Bloch"
    for cell in rows["eigenmode_periodic_bloch"]["cells"]:
        assert cell["solvers"] == ["direct solver"]
    assert _cell(derived, "eigenmode_periodic_plain", "eps_r")["solvers"] == [
        "direct solver",
        "BDDC + GMRES",
    ]
    assert _cell(derived, "driven_periodic_plain", "nodes")["sweeps"] == [
        "uniform",
        "adaptive",
    ]
    for cell in rows["driven_periodic_bloch"]["cells"]:
        assert cell["sweeps"] == ["uniform"]
    assert {r["recipe"] for k, r in rows.items() if k not in FRONT} == {
        "recipes/periodic-cell"
    }


def test_only_the_four_problem_types_are_front_page_rows(derived):
    front = [r["key"] for r in derived["rows"] if r["front"]]
    assert front == list(FRONT)
    assert all(not r["front"] for r in derived["rows"] if r["key"] not in FRONT)


def test_raising_cells_name_their_error(derived):
    sheet = _cell(derived, "electrostatic", "fill")
    assert sheet["label"] == "raises NotImplementedError"
    assert _cell(derived, "port_mode", "eps_r")["label"] == "raises"
    assert _cell(derived, "electrostatic", "conductivity")["label"] == "exact zero"


def test_raises_list_is_the_out_of_scope_rows_off_the_grid(derived):
    rows = {r["key"]: r for r in derived["raises"]}
    assert len(rows) == 18
    assert "electrostatic x sheet fill" not in rows
    assert rows["probe x node coordinates"]["error"] == "NonDifferentiableError"
    assert rows["material axes (orientation)"] == {
        "key": "material axes (orientation)",
        "label": "Any result, with respect to a material's principal axes",
        "error": "NonDifferentiableError",
        "when": "",
    }
    nodes = rows["driven x nodes of a port surface"]
    assert nodes["error"] == "PortGeometryError"
    assert nodes["when"] == (
        "while the gradient is computed, inside a `jax.errors.JaxRuntimeError`"
    )


def test_a_wrapped_error_with_no_gfem_name_still_says_when_it_raises():
    from jax.errors import JaxRuntimeError

    row = caps._error(JaxRuntimeError, "no error name here")
    assert row["error"] == "JaxRuntimeError"
    assert row["when"] == "while the gradient is computed"
    assert caps.links_gradient_errors(row)
    assert caps.links_gradient_errors(caps._error(JaxRuntimeError, "PortGeometryError"))
    assert not caps.links_gradient_errors(caps._error(ValueError, "ValueError"))


def test_raises_rows_read_in_site_words_not_test_labels(derived):
    for row in derived["raises"]:
        assert row["label"] == caps.RAISES_LABELS[row["key"]]
        assert row["label"] != row["key"]
        assert " x " not in row["label"]


#: The periodic rows of the raises list (test labels), with their errors.
PERIODIC_RAISES = {
    "driven x wave_vector": "NonDifferentiableError",
    "eigenmode x wave_vector": "NonDifferentiableError",
    "periodic model x node move that breaks a pair": "ValueError",
    "boundary_morph x paired moving surface": "ValueError",
    "Bloch driven x adaptive sweep": "NotImplementedError",
    "Bloch eigenmode x bddc_gmres": "ValueError",
}


def test_periodic_raises_rows_publish_their_labels_and_errors(derived):
    rows = {r["key"]: r["error"] for r in derived["raises"]}
    assert {k: rows.get(k) for k in PERIODIC_RAISES} == PERIODIC_RAISES


# --- drift guards: each broken input fails the derivation ----------------------


@pytest.fixture(scope="module")
def data():
    return caps.load_verification(REPO)


def _derive(data, cells=None, expected=None, out_of_scope=None):
    c, e, o = data
    return caps.gradient_matrix(
        c if cells is None else cells,
        e if expected is None else expected,
        o if out_of_scope is None else out_of_scope,
    )


def test_unchanged_data_derives(data):
    assert len(_derive(data)["rows"]) == 8


def test_a_new_problem_in_the_cells_fails(data):
    cells = [*data[0], dataclasses.replace(data[0][0], problem="magnetostatic")]
    with pytest.raises(caps.CapabilitiesError, match="magnetostatic"):
        _derive(data, cells=cells)


def test_a_new_input_in_the_cells_fails(data):
    cells = [*data[0], dataclasses.replace(data[0][0], input="chirality")]
    with pytest.raises(caps.CapabilitiesError, match="chirality"):
        _derive(data, cells=cells)


def test_an_unknown_suffix_of_a_property_fails(data):
    cells = [*data[0], dataclasses.replace(data[0][0], input="eps_r_tensor")]
    with pytest.raises(caps.CapabilitiesError, match="eps_r_tensor"):
        _derive(data, cells=cells)


def test_a_new_problem_in_the_raise_table_fails(data):
    expected = data[1] | {("thermal", "temperature", ("nodes",)): "gradient"}
    with pytest.raises(caps.CapabilitiesError, match="thermal"):
        _derive(data, expected=expected)


def test_a_new_input_kind_in_the_raise_table_fails(data):
    kind = ("material", "oxide", "chirality")
    expected = data[1] | {("electrostatic", "capacitance", kind): "zero"}
    with pytest.raises(caps.CapabilitiesError, match="chirality"):
        _derive(data, expected=expected)


def test_a_problem_without_its_headline_leaf_fails(data):
    expected = {k: v for k, v in data[1].items() if k[:2] != ("port_mode", "gamma")}
    with pytest.raises(caps.CapabilitiesError, match=r"port_mode.*gamma"):
        _derive(data, expected=expected)


def test_every_problem_without_its_headline_leaf_is_named_at_once(data):
    gone = {("port_mode", "gamma"), ("electrostatic", "capacitance")}
    expected = {k: v for k, v in data[1].items() if k[:2] not in gone}
    with pytest.raises(caps.CapabilitiesError) as error:
        _derive(data, expected=expected)
    assert "'port_mode'" in str(error.value)
    assert "'electrostatic'" in str(error.value)


def test_a_bare_input_kind_reads_as_a_one_tuple(data):
    expected = {
        (key, leaf, "nodes" if kind == ("nodes",) else kind): outcome
        for (key, leaf, kind), outcome in data[1].items()
    }
    assert _derive(data, expected=expected) == _derive(data)


def test_a_gradient_no_cell_verifies_fails(data):
    cells = [
        c
        for c in data[0]
        if (c.problem, c.input.removesuffix("_axes")) != ("electrostatic", "eps_r")
    ]
    with pytest.raises(caps.CapabilitiesError, match="Electrostatic x Permittivity"):
        _derive(data, cells=cells)


def test_a_cell_with_no_outcome_fails(data):
    expected = {
        k: v
        for k, v in data[1].items()
        if k[:2] != ("electrostatic", "capacitance") or k[2][-1] != "conductivity"
    }
    with pytest.raises(caps.CapabilitiesError, match="no outcome"):
        _derive(data, expected=expected)


def test_the_missing_out_of_scope_row_the_grid_needs_fails(data):
    rows = [r for r in data[2] if r[0] != "electrostatic x sheet fill"]
    with pytest.raises(caps.CapabilitiesError, match="electrostatic x sheet fill"):
        _derive(data, out_of_scope=rows)


def test_a_verified_cell_the_raise_table_calls_zero_fails(data):
    kind = ("material", "dielectric", "eps_r")
    expected = data[1] | {("driven", "s.values", kind): "zero"}
    with pytest.raises(caps.CapabilitiesError, match="conflicting"):
        _derive(data, expected=expected)


def test_a_zero_and_a_raise_in_one_cell_fails(data):
    kind = ("boundary", 0, "fill", "fill")
    expected = data[1] | {("electrostatic", "capacitance", kind): "zero"}
    with pytest.raises(caps.CapabilitiesError, match="conflicting"):
        _derive(data, expected=expected)


def test_a_raises_row_with_no_site_label_fails(data):
    row = ("tensor x chirality", None, ValueError, "chirality")
    with pytest.raises(caps.CapabilitiesError, match="tensor x chirality"):
        _derive(data, out_of_scope=[*data[2], row])


def test_a_site_label_with_no_raises_row_fails(data):
    rows = [r for r in data[2] if r[0] != "forward mode"]
    with pytest.raises(caps.CapabilitiesError, match="forward mode"):
        _derive(data, out_of_scope=rows)


def test_the_matrix_provenance_is_its_own_sentence(derived):
    assert derived["provenance"].endswith(
        "each checked along 3 random directions against fourth-order "
        "central differences"
    )


# --- the table of what raises above one rank ---------------------------------------

_MESSAGES = {"a": "a has no path", "b": "b has no path", "guard": "guarded"}
_LABELS = {"a": "A", "a_auto": "Auto", "b": "B"}


def _refusals(monkeypatch, **kwargs):
    monkeypatch.setattr(caps, "REFUSAL_LABELS", _LABELS)
    arguments = {
        "keys": ("a", "a_auto"),
        "methods": ("b",),
        "guards": ("guard",),
        "feature": lambda key: key.removesuffix("_auto"),
        "messages": _MESSAGES,
    } | kwargs
    return caps.distributed_refusals(*arguments.values())


def test_each_refusal_row_is_its_case_and_gfems_message(derived):
    rows = derived["refusals"]
    assert [r["feature"] for r in rows] == [
        mpi_cases.feature(k) for k in (*mpi_cases.KEYS, *mpi_cases.METHOD_KEYS)
    ]
    assert all(r["message"] == _mpi._REFUSALS[r["feature"]] for r in rows)
    assert all(r["label"] for r in rows)


def test_two_cases_of_one_key_are_two_rows(monkeypatch):
    rows = _refusals(monkeypatch)
    assert [(r["label"], r["feature"]) for r in rows] == [
        ("A", "a"),
        ("Auto", "a"),
        ("B", "b"),
    ]
    assert rows[0]["message"] == "a has no path"


def test_a_guard_no_user_reaches_has_no_row(derived):
    assert "general_operator" not in {r["feature"] for r in derived["refusals"]}


def test_a_case_with_no_label_fails_the_build(monkeypatch):
    with pytest.raises(caps.CapabilitiesError, match="labelled"):
        _refusals(monkeypatch, keys=("a", "a_auto", "new"))


def test_a_refusal_with_no_case_fails_the_build(monkeypatch):
    with pytest.raises(caps.CapabilitiesError, match="messages"):
        _refusals(monkeypatch, messages=_MESSAGES | {"new": "new"})


def test_a_raise_seen_while_the_gradient_is_computed_links_the_shared_section(
    tmp_path,
):
    # The raises list renders in a site with a page holding the shared
    # "When a gradient raises" section: each row whose error arrives wrapped
    # links that section, which says how to catch it.
    build = pytest.importorskip("sphinx.cmd.build")
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(REPO / 'docs' / '_ext')!r})\n"
        "extensions = ['gfem_capabilities']\n"
        "gfem_capabilities_source_url = 'https://example.org/'\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text(
        "Raises\n======\n\n.. toctree::\n\n   errors\n\n.. gfem-raises::\n",
        encoding="utf-8",
    )
    (source / "errors.rst").write_text(
        ".. _gradient-errors:\n\nWhen a gradient raises\n======================\n",
        encoding="utf-8",
    )
    out = tmp_path / "html"
    assert build.build_main(["-q", "-b", "html", str(source), str(out)]) == 0
    items = (out / "index.html").read_text(encoding="utf-8").split("<li>")[1:]
    wrapped = [i for i in items if "JaxRuntimeError" in i]
    assert wrapped
    assert all('href="errors.html#gradient-errors"' in i for i in wrapped)
    assert all("</a> (<a" in i for i in wrapped)  # beside the source link, not in it
    assert not any("#gradient-errors" in i for i in items if i not in wrapped)
