"""The Capabilities page generator (docs/_ext/gfem_capabilities.py).

The site's gradient claims and Palace tolerances are derived from the tests
at docs build time (no hand-typed table); a claim that cannot be derived
fails the build. These tests pin the derivation against a grid reviewed by
hand, and each drift guard against a broken input. The generator is imported
without Sphinx.
"""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location(
    "gfem_capabilities", REPO / "docs" / "_ext" / "gfem_capabilities.py"
)
caps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(caps)

V, Z, R = "verified", "zero", "raises"

# The grid as reviewed by hand against the tests:
# rows driven, eigenmode, electrostatic, port mode; columns permittivity,
# conductivity, sheet fill, node coordinates.
GRID = {
    "driven": (V, V, V, V),
    "eigenmode": (V, V, V, V),
    "electrostatic": (V, Z, R, V),
    "port_mode": (R, R, R, R),
}


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


def test_raising_cells_name_their_error(derived):
    sheet = _cell(derived, "electrostatic", "fill")
    assert sheet["label"] == "raises NotImplementedError"
    assert _cell(derived, "port_mode", "eps_r")["label"] == "raises"
    assert _cell(derived, "electrostatic", "conductivity")["label"] == "exact zero"


def test_raises_list_is_the_out_of_scope_rows_off_the_grid(derived):
    rows = {r["label"]: r for r in derived["raises"]}
    assert len(rows) == 7
    assert "electrostatic \N{MULTIPLICATION SIGN} sheet fill" not in rows
    assert (
        rows["probe \N{MULTIPLICATION SIGN} node coordinates"]["error"]
        == "NonDifferentiableError"
    )
    nodes = rows["driven \N{MULTIPLICATION SIGN} nodes of a port surface"]
    assert nodes == {
        "label": "driven \N{MULTIPLICATION SIGN} nodes of a port surface",
        "error": "PortGeometryError",
        "when": "in the backward pass",
    }


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
    assert len(_derive(data)["rows"]) == 4


def test_a_new_problem_in_the_cells_fails(data):
    cells = [*data[0], dataclasses.replace(data[0][0], problem="magnetostatic")]
    with pytest.raises(caps.CapabilitiesError, match="magnetostatic"):
        _derive(data, cells=cells)


def test_a_new_input_in_the_cells_fails(data):
    cells = [*data[0], dataclasses.replace(data[0][0], input="chirality")]
    with pytest.raises(caps.CapabilitiesError, match="chirality"):
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
    cells = [c for c in data[0] if (c.problem, c.input) != ("electrostatic", "eps_r")]
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


# --- agreement with Palace ------------------------------------------------------


def test_palace_rows_quote_the_reference_tolerances(derived):
    rows = {r["label"]: r for r in derived["palace"]}
    assert rows["Driven, lumped ports"]["text"] == "max |ΔS| ≤ 1e-5"
    assert rows["Driven, lumped ports"]["kind"] == "absolute"
    assert rows["Eigenmode, port-loaded"]["text"] == "relative: Re f 1e-5 · Q 1e-3"
    nonlinear = rows["Nonlinear eigenmode, conductor loss"]
    assert [v["value"] for v in nonlinear["values"]] == [1e-5, 1e-3]
    assert rows["Electrostatic"]["text"] == "relative 1e-4"
    assert rows["Electrostatic"]["file"] == "tests/reference/test_electrostatic.py"
    assert rows["Interface participation"]["text"] == "relative 5 %"
    problems = [r["label"] for r in derived["palace"] if r["problem"]]
    assert problems == [
        "Driven, lumped ports",
        "Eigenmode, port-loaded",
        "Nonlinear eigenmode, conductor loss",
        "Electrostatic",
    ]


def test_the_driven_row_names_the_first_order_absorbing_boundary(derived):
    # Its tolerance is asserted on the abc1 cases only: at order 2 gfem and
    # Palace up to v0.18.1 use opposite signs of the curl term.
    (driven,) = [r for r in derived["palace"] if r["label"] == "Driven, lumped ports"]
    assert "first-order absorbing boundary" in driven["detail"]


def _module(tmp_path, source):
    (tmp_path / "test_x.py").write_text(source)
    return tmp_path


def _row(*names):
    return caps.Agreement("test_x.py", names, ("q",) * len(names), "X", "", "relative")


def test_a_tuple_assignment_is_read(tmp_path):
    repo = _module(tmp_path, "A, B = 1e-5, 2e-3\nassert A < B\n")
    (row,) = caps.palace_agreement(repo, [_row("A", "B")])
    assert [v["value"] for v in row["values"]] == [1e-5, 2e-3]


def test_a_missing_constant_fails(tmp_path):
    repo = _module(tmp_path, "RTOL_X = 1e-4\nassert RTOL_X\n")
    with pytest.raises(caps.CapabilitiesError, match=r"RTOL.*missing"):
        caps.palace_agreement(repo, [_row("RTOL")])


def test_a_non_literal_constant_fails(tmp_path):
    repo = _module(tmp_path, "import math\nRTOL = math.sqrt(1e-8)\nassert RTOL\n")
    with pytest.raises(caps.CapabilitiesError, match=r"RTOL.*literal"):
        caps.palace_agreement(repo, [_row("RTOL")])


def test_a_dead_constant_fails(tmp_path):
    repo = _module(tmp_path, "RTOL = 1e-4\n")
    with pytest.raises(caps.CapabilitiesError, match=r"RTOL.*unused"):
        caps.palace_agreement(repo, [_row("RTOL")])


@pytest.mark.parametrize("name", ["PALACE_PER_SAMPLE", "WALL_FACTOR", "TIME_BUDGET"])
def test_a_timing_constant_is_never_published(tmp_path, name):
    repo = _module(tmp_path, f"{name} = 1.5\nassert {name}\n")
    with pytest.raises(caps.CapabilitiesError, match="timing"):
        caps.palace_agreement(repo, [_row(name)])
