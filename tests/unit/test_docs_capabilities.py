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
    # the piezoelectric section: principal permittivities and a pitch map are
    # verified; it reads no conductivity, and a sheet field needs a slab
    "piezoelectric": (V, Z, R, V),
    "port_mode": (V, V, V, V),
    "driven_periodic_plain": (V, V, V, V),
    "driven_periodic_bloch": (V, V, V, V),
    "eigenmode_periodic_plain": (V, V, V, V),
    "eigenmode_periodic_bloch": (V, V, V, V),
    "eigenmode_participation": (V, V, V, V),
    "eigenmode_field": (V, V, V, R),
    # the energy has node terms; the heat kernel has none
    "electric_energy": (V, V, V, V),
    "dissipated_power": (V, V, V, R),
    "electromagnetic_heating": (V, V, V, R),
    # sources: the plane wave's load and the magnetic dipole's have no node
    # derivative
    "driven_incident": (V, V, V, R),
    "driven_dipole": (V, V, V, V),
    "driven_magnetic_dipole": (V, V, V, R),
    "steady_thermal": (Z, Z, Z, R),
    "transient_thermal": (Z, Z, Z, R),
    "dc_conduction": (Z, V, Z, R),
    # the held mode's solve reads a sheet's fill too: it raises
    "thermo_optic_shift": (R, R, R, R),
    "route_phase": (R, R, R, R),
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
    off_grid = [entry["cells"] for entry in derived["also"]]
    assert counts["cells"] == sum(per_grid_cell) + sum(off_grid)
    assert counts["cells"] == counts["direct"] + counts["bddc_gmres"]
    assert counts["direct"] > 0
    assert counts["bddc_gmres"] > 0
    assert derived["provenance"].startswith(
        f"{counts['cells']} cells ({counts['direct']} on the direct solver, "
        f"{counts['bddc_gmres']} repeated on BDDC + GMRES)"
    )


def test_driven_sheet_fill_is_verified_on_lumped_and_wave_ports(derived):
    # a sheet beside wave ports moves their modes too (the port pass)
    fill = _cell(derived, "driven", "fill")
    assert fill["label"] == "FD-verified"
    assert fill["ports"] == ["lumped", "wave"]
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
        ("port_mode", "eps_r"),
        ("eigenmode_participation", "eps_r"),
        ("electric_energy", "eps_r"),
    }
    for row in derived["rows"]:
        for cell in row["cells"]:
            key = (row["key"], cell["column"])
            if key == ("piezoelectric", "eps_r"):
                assert cell["forms_text"] == "principal values only", key
            elif key in both:
                assert cell["forms"] == ["isotropic", "principal values"], key
                assert cell["forms_text"] == "isotropic and principal values", key
            else:
                assert cell["forms_text"] == "", key
    # the label (and so the front page's compact matrix) does not change
    assert _cell(derived, "driven", "eps_r")["label"] == "FD-verified"


def test_permeability_loss_tangent_and_lumped_values_are_off_the_grid(derived):
    # no column of their own: one phrase per row and group under the matrix
    assert [(e["key"], e["inputs"], e["cells"]) for e in derived["also"]] == [
        ("driven", ["permeability", "loss tangent"], 6),
        ("driven", ["lumped values"], 5),
        ("driven", ["permittivity-law parameters"], 9),
        ("driven", ["density of a metal-dielectric blend"], 2),
        ("eigenmode", ["permeability", "loss tangent"], 2),
        ("eigenmode", ["lumped values"], 3),
        ("eigenmode", ["permittivity-law parameters"], 4),
        ("electrostatic", ["permittivity-law parameters"], 1),
        ("port_mode", ["permeability", "loss tangent"], 2),
        ("port_mode", ["lumped values"], 1),
        ("port_mode", ["permittivity-law parameters"], 1),
        ("driven_periodic_plain", ["permittivity-law parameters"], 1),
        ("eigenmode_participation", ["lumped values"], 6),
        ("eigenmode_field", ["lumped values"], 1),
        ("electric_energy", ["loss tangent"], 1),
        ("electric_energy", ["permittivity-law parameters"], 1),
        ("dissipated_power", ["loss tangent"], 1),
        ("dissipated_power", ["lumped values"], 1),
        ("dissipated_power", ["permittivity-law parameters"], 1),
        ("dissipated_power", ["density of a metal-dielectric blend"], 2),
        ("electromagnetic_heating", ["lumped values"], 1),
        ("electromagnetic_heating", ["thermal conductivity"], 1),
        ("electromagnetic_heating", ["incident powers and phases"], 1),
        ("steady_thermal", ["thermal conductivity"], 3),
        ("steady_thermal", ["heat-source amount and waveform scales"], 1),
        (
            "transient_thermal",
            ["thermal conductivity", "mass density", "specific heat"],
            36,
        ),
        ("transient_thermal", ["heat-source amount and waveform scales"], 24),
    ]
    assert [e["text"] for e in derived["also"]] == [
        (
            "Driven with respect to the permeability and the loss tangent "
            "(isotropic and principal values, direct solver, uniform sweep)"
        ),
        (
            "Driven with respect to the lumped values "
            "(direct solver, BDDC + GMRES, uniform and adaptive sweep)"
        ),
        (
            "Driven with respect to the permittivity-law parameters "
            "(direct solver, uniform and adaptive sweep)"
        ),
        (
            "Driven with respect to the density of a metal-dielectric blend "
            "(direct solver, uniform sweep)"
        ),
        (
            "Eigenmode with respect to the permeability and the loss tangent "
            "(direct solver)"
        ),
        "Eigenmode with respect to the lumped values (direct solver, BDDC + GMRES)",
        "Eigenmode with respect to the permittivity-law parameters (direct solver)",
        "Electrostatic with respect to the permittivity-law parameters (direct solver)",
        (
            "Port mode problem with respect to the permeability and the loss "
            "tangent (direct solver)"
        ),
        "Port mode problem with respect to the lumped values (direct solver)",
        (
            "Port mode problem with respect to the permittivity-law parameters "
            "(direct solver)"
        ),
        (
            "Driven, plain periodic with respect to the permittivity-law "
            "parameters (direct solver, uniform sweep)"
        ),
        "Eigenmode, participations with respect to the lumped values (direct solver)",
        "Eigenmode, mode field with respect to the lumped values (direct solver)",
        "Electric energy with respect to the loss tangent (direct solver)",
        (
            "Electric energy with respect to the permittivity-law parameters "
            "(direct solver)"
        ),
        (
            "Dissipated power with respect to the loss tangent "
            "(principal values only, direct solver)"
        ),
        "Dissipated power with respect to the lumped values (direct solver)",
        (
            "Dissipated power with respect to the permittivity-law parameters "
            "(direct solver)"
        ),
        (
            "Dissipated power with respect to the density of a metal-dielectric "
            "blend (direct solver)"
        ),
        "Electromagnetic heating with respect to the lumped values (direct solver)",
        (
            "Electromagnetic heating with respect to the thermal conductivity "
            "(direct solver)"
        ),
        (
            "Electromagnetic heating with respect to the incident powers and "
            "phases (direct solver)"
        ),
        (
            "Steady thermal with respect to the thermal conductivity "
            "(direct solver, BDDC + GMRES)"
        ),
        (
            "Steady thermal with respect to the heat-source amount and waveform "
            "scales (direct solver)"
        ),
        (
            "Transient thermal with respect to the thermal conductivity and the "
            "mass density and the specific heat (direct solver, BDDC + GMRES)"
        ),
        (
            "Transient thermal with respect to the heat-source amount and waveform "
            "scales (direct solver, BDDC + GMRES)"
        ),
    ]


def test_an_off_grid_cell_the_raise_table_does_not_expect_fails(data):
    # the electrostatic solve never reads mu_r: an exact zero, not a gradient
    cells = [*data[0], dataclasses.replace(data[0][0], input="mu_r")]
    with pytest.raises(caps.CapabilitiesError, match="Electrostatic x permeability"):
        _derive(data, cells=cells)


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
    assert {r["recipe"] for k, r in rows.items() if "periodic" in k} == {
        "recipes/periodic-cell"
    }


def test_floquet_cells_join_the_periodic_driven_rows(derived):
    # the Floquet keys verify the permittivity and the nodes on F2: those
    # cells list both port kinds, the others stay on the lumped-port cell G
    for row in ("driven_periodic_plain", "driven_periodic_bloch"):
        for column in ("eps_r", "nodes"):
            assert _cell(derived, row, column)["ports"] == ["lumped", "floquet"]
            assert _cell(derived, row, column)["label"] == "FD-verified"
        for column in ("conductivity", "fill"):
            assert _cell(derived, row, column)["label"] == "FD-verified, lumped ports"


def test_the_mode_field_rows_link_the_recipe_of_their_outputs(derived):
    # participations and port Q are eigenmode outputs; the field is read by
    # the measurements (probes and interface participation)
    rows = {r["key"]: r for r in derived["rows"]}
    assert rows["eigenmode_participation"]["recipe"] == "recipes/eigenmode-gradient"
    assert rows["eigenmode_field"]["recipe"] == "recipes/measurements"


def test_the_port_mode_row_links_its_recipe(derived):
    (row,) = [r for r in derived["rows"] if r["key"] == "port_mode"]
    assert row["recipe"] == "recipes/port-mode-gradient"


def test_cells_at_wave_port_faces_say_so(derived):
    for column in ("eps_r", "conductivity", "fill", "nodes"):
        assert _cell(derived, "driven", column)["faces"]
    assert not _cell(derived, "electrostatic", "eps_r")["faces"]
    assert not _cell(derived, "port_mode", "eps_r")["faces"]


def test_only_the_four_problem_types_are_front_page_rows(derived):
    front = [r["key"] for r in derived["rows"] if r["front"]]
    assert front == list(FRONT)
    assert all(not r["front"] for r in derived["rows"] if r["key"] not in FRONT)


def test_raising_cells_name_their_error(derived):
    sheet = _cell(derived, "electrostatic", "fill")
    assert sheet["label"] == "raises NotImplementedError"
    assert _cell(derived, "port_mode", "nodes")["label"] == "FD-verified"
    assert _cell(derived, "electrostatic", "conductivity")["label"] == "exact zero"


def test_raises_list_is_the_out_of_scope_rows_off_the_grid(derived):
    rows = {r["key"]: r for r in derived["raises"]}
    assert len(rows) == 96
    assert rows["design in the far-field layer"]["error"] == "ValueError"
    fit = rows["fit_rational under jax.grad"]
    assert fit["error"] == "NonDifferentiableError"
    assert fit["label"].startswith("`NetworkParameters.fit_rational`")
    assert "electrostatic x sheet fill" not in rows
    for knob in ("target_reflection", "reference_frequency"):
        layer = rows[f"PML target reflection ({knob})"]
        assert layer["error"] == "NonDifferentiableError"
        assert "perfectly matched layer" in layer["label"]
    shift = rows["thermo-optic shift x optical permittivity"]
    assert shift["error"] == "NonDifferentiableError"
    assert shift["label"].startswith("`thermo_optic_shift`")
    route = rows["route phase x window permittivity"]
    assert route["error"] == "NonDifferentiableError"
    assert route["label"].startswith("`route_phase`")
    assert rows["law parameter of a blend's end"]["error"] == "NonDifferentiableError"
    assert rows["probe x node coordinates"]["error"] == "NonDifferentiableError"
    heat = rows["heat_flow x node coordinates"]
    assert heat["error"] == "NonDifferentiableError"
    assert heat["label"].startswith("The heat through a surface")
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
    "eigenmode band group velocity": "NonDifferentiableError",
    "band_diagram inside jax.jit or jax.vmap": "NonDifferentiableError",
    "band_diagram x wave vectors or target": "NonDifferentiableError",
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
    assert len(_derive(data)["rows"]) == 22


def test_a_new_problem_in_the_cells_fails(data):
    cells = [*data[0], dataclasses.replace(data[0][0], problem="magnetostatic")]
    with pytest.raises(caps.CapabilitiesError, match="magnetostatic"):
        _derive(data, cells=cells)


def test_a_face_variant_without_wave_ports_fails(data):
    cells = [
        *data[0],
        dataclasses.replace(
            data[0][0], problem="electrostatic", input="eps_r", variant="face_lossy"
        ),
    ]
    with pytest.raises(caps.CapabilitiesError, match="face_lossy"):
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
        # every electrostatic model's (the law's too) conductivity rows
        if k[0] not in ("electrostatic", "electrostatic_law")
        or k[1] != "capacitance"
        or k[2][-1] != "conductivity"
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


def test_no_guard_key_is_left():
    assert mpi_cases.GUARD_KEYS == ()
    assert "general_operator" not in _mpi._REFUSALS


def test_the_lifted_solves_have_no_row(derived):
    features = {r["feature"] for r in derived["refusals"]}
    assert not features & {"eigenmode", "electrostatic", "general_operator"}


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


# --- the error estimates' effectivity ranges --------------------------------------


@pytest.fixture(scope="module")
def estimate_test():
    return caps._import(REPO / caps.ESTIMATE_TEST)


@pytest.fixture(scope="module")
def field_estimate_test():
    return caps._import(REPO / caps.FIELD_ESTIMATE_TEST)


@pytest.fixture(scope="module")
def objective_estimate_test():
    return caps._import(REPO / caps.OBJECTIVE_ESTIMATE_TEST)


def _cases(*modules, kind):
    """The case names of ``kind`` over ``modules``, order 3 after order 2."""
    names = []
    for module in modules:
        names += list(getattr(module, "EFFECTIVITY_BY_KIND", {}).get(kind, {}))
    for module in modules:
        order_3 = getattr(module, "ORDER_3_EFFECTIVITY_BY_KIND", {}).get(kind, {})
        names += [f"{case}, order 3" for case in order_3]
    return names


def _span(module, kind):
    ranges = list(module.EFFECTIVITY_BY_KIND[kind].values())
    ranges += list(
        getattr(module, "ORDER_3_EFFECTIVITY_BY_KIND", {}).get(kind, {}).values()
    )
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def test_each_emitted_kind_is_quoted_as_its_cases_span(
    derived, estimate_test, objective_estimate_test, field_estimate_test
):
    from gfem.refinement import EMITTED_KINDS

    modules = (estimate_test, objective_estimate_test, field_estimate_test)
    for kind in EMITTED_KINDS:
        module = next(m for m in modules if kind in m.EFFECTIVITY_BY_KIND)
        low, high = _span(module, kind)
        assert derived["effectivity"][kind] == {
            "low": f"{low:g}",
            "high": f"{high:g}",
            "cases": _cases(*modules, kind=kind),
        }


def test_the_quoted_quantity_effectivity_spans_the_entry_kinds(derived, estimate_test):
    entries = ("capacitance", "s", "omega")
    spans = [_span(estimate_test, kind) for kind in entries]
    quoted = derived["effectivity"]["quantity"]
    assert quoted["low"] == f"{min(s[0] for s in spans):g}"
    assert quoted["high"] == f"{max(s[1] for s in spans):g}"
    assert quoted["cases"] == [
        case for kind in entries for case in estimate_test.EFFECTIVITY_BY_KIND[kind]
    ]


def test_the_quoted_field_effectivity_spans_both_orders_of_the_field_test(
    derived, field_estimate_test
):
    # The field estimator is claimed at orders 2 and 3: the quoted range
    # covers both dicts, the order-3 cases named as such.
    quoted = derived["effectivity"]["field"]
    assert (quoted["low"], quoted["high"]) == ("0.9", "2")
    order_2 = list(field_estimate_test.EFFECTIVITY_BY_KIND["field"])
    order_3 = list(field_estimate_test.ORDER_3_EFFECTIVITY_BY_KIND["field"])
    assert quoted["cases"] == order_2 + [f"{c}, order 3" for c in order_3]


def test_the_loss_jump_witness_is_the_field_tests_measured_band(derived):
    # A lumped port across a loss jump: a measured band below the field
    # range, quoted so the recipe's Limits says how far below.
    assert derived["effectivity"]["lumped port across a loss jump"] == {
        "low": "0.84",
        "high": "0.9",
    }


def test_every_estimate_test_module_holds_the_three_dicts(
    estimate_test, field_estimate_test
):
    for module in (estimate_test, field_estimate_test):
        assert isinstance(module.EFFECTIVITY_BY_KIND, dict)
        assert isinstance(module.OVERESTIMATES, dict)
        assert isinstance(module.WITNESSES, dict)
        assert not hasattr(module, "EFFECTIVITY_BY_CASE")
        assert not hasattr(module, "EFFECTIVITY")


_KINDS = ("field", "s", "omega")


def test_a_kind_the_code_does_not_know_fails_the_build():
    by_kind = {"field": {"a": (0.9, 2.0)}, "flux": {"b": (0.9, 2.0)}}
    with pytest.raises(caps.CapabilitiesError, match="flux"):
        caps.effectivity_by_kind(by_kind, _KINDS, ("field",))


def test_a_kind_the_code_does_not_emit_yet_fails_the_build():
    by_kind = {"field": {"a": (0.9, 2.0)}, "s": {"b": (0.9, 2.0)}}
    with pytest.raises(caps.CapabilitiesError, match="'s'"):
        caps.effectivity_by_kind(by_kind, _KINDS, ("field",))


def test_an_emitted_kind_without_a_case_fails_the_build():
    with pytest.raises(caps.CapabilitiesError, match="'omega'"):
        caps.effectivity_by_kind(
            {"field": {"a": (0.9, 2.0)}}, _KINDS, ("field", "omega")
        )


def test_a_case_below_the_claims_floor_fails_the_build():
    with pytest.raises(caps.CapabilitiesError, match="claims floor"):
        caps.effectivity_by_kind({"field": {"a": (0.85, 2.0)}}, _KINDS, ("field",))


def test_an_unordered_case_range_fails_the_build():
    with pytest.raises(caps.CapabilitiesError, match="bad case ranges"):
        caps.effectivity_by_kind({"field": {"a": (2.0, 0.9)}}, _KINDS, ("field",))


def test_a_kind_quotes_its_cases_span():
    by_kind = {"field": {"a": (0.9, 2.0), "b": (1.0, 15.0)}}
    assert caps.effectivity_by_kind(by_kind, _KINDS, ("field",)) == {
        "field": {"low": "0.9", "high": "15", "cases": ["a", "b"]}
    }


def test_a_witness_inside_the_quoted_range_fails_the_build():
    with pytest.raises(caps.CapabilitiesError, match="below"):
        caps.witness((0.95, 1.2), {"low": "0.9", "high": "2"})


def test_a_witness_reaching_below_the_quoted_range_is_quoted():
    # The rule is the band's low end: a band straddling the floor is unclaimed.
    assert caps.witness((0.87, 0.97), {"low": "0.9", "high": "2"}) == {
        "low": "0.87",
        "high": "0.97",
    }


def test_an_unordered_witness_fails_the_build():
    with pytest.raises(caps.CapabilitiesError, match="bad witness"):
        caps.witness((0.9, 0.84), {"low": "0.9", "high": "2"})


def test_an_overestimate_inside_its_kinds_range_fails_the_build():
    with pytest.raises(caps.CapabilitiesError, match="inside"):
        caps.overestimate((1.0, 1.5), {"low": "0.9", "high": "2"})


def test_an_overestimate_below_the_claims_floor_fails_the_build():
    with pytest.raises(caps.CapabilitiesError, match="below"):
        caps.overestimate((0.5, 50.0), {"low": "0.9", "high": "2"})


def test_an_overestimate_above_its_kinds_range_is_quoted():
    assert caps.overestimate((10.0, 70.0), {"low": "0.9", "high": "20"}) == {
        "low": "10",
        "high": "70",
    }


def test_the_kind_words_cover_exactly_the_estimate_kinds():
    from gfem.refinement import ESTIMATE_KINDS

    assert set(caps.KIND_WORDS) == set(ESTIMATE_KINDS)
    with pytest.raises(caps.CapabilitiesError, match="words"):
        caps.effectivity_table({}, ESTIMATE_KINDS, (), words={"field": "the field"})


def test_the_effectivity_table_has_a_row_per_emitted_kind(derived):
    from gfem.refinement import EMITTED_KINDS

    rows = derived["effectivity_table"]
    assert [row["kind"] for row in rows] == list(EMITTED_KINDS)
    for row in rows:
        quoted = derived["effectivity"][row["kind"]]
        assert (row["low"], row["high"]) == (quoted["low"], quoted["high"])
        assert row["cases"] == len(quoted["cases"])
        assert row["words"] == caps.KIND_WORDS[row["kind"]]


def test_every_estimate_kind_is_emitted():
    # Every kind the code knows is emitted, so every one is in the table.
    from gfem.refinement import EMITTED_KINDS, ESTIMATE_KINDS

    assert EMITTED_KINDS == ESTIMATE_KINDS
    assert caps._estimate_kinds() == (ESTIMATE_KINDS, ESTIMATE_KINDS)


def test_a_kind_the_estimators_do_not_emit_fails_the_build(monkeypatch):
    from gfem import refinement

    monkeypatch.setattr(refinement, "EMITTED_KINDS", refinement.ESTIMATE_KINDS[:-1])
    with pytest.raises(caps.CapabilitiesError, match="objective of participation"):
        caps._estimate_kinds()


def test_the_effectivity_table_lists_every_estimate_kind(derived):
    from gfem.refinement import ESTIMATE_KINDS

    rows = derived["effectivity_table"]
    assert [row["kind"] for row in rows] == list(ESTIMATE_KINDS)


def test_the_port_mode_share_is_the_estimate_tests_bound(derived, estimate_test):
    # the Limits sentence quotes the bound the hybrid strip test holds.
    share = estimate_test.PORT_MODE_SHARE
    assert derived["effectivity"]["port mode share"] == {"high": f"{share:g}"}


@pytest.mark.parametrize("share", [0.0, -0.1, 1.0, 2.0, "x", None])
def test_a_port_mode_share_outside_zero_and_one_fails_the_build(share):
    with pytest.raises(caps.CapabilitiesError, match="port mode share"):
        caps.port_mode_share(share)


def _build_role_page(tmp_path, body):
    build = pytest.importorskip("sphinx.cmd.build")
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(REPO / 'docs' / '_ext')!r})\n"
        "extensions = ['gfem_capabilities']\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text(f"Range\n=====\n\n{body}\n", encoding="utf-8")
    out = tmp_path / "html"
    status = build.build_main(["-q", "-W", "-b", "html", str(source), str(out)])
    return status, out / "index.html"


def test_the_effectivity_role_renders_both_ranges(tmp_path, derived):
    status, page = _build_role_page(
        tmp_path,
        "Field :gfem-effectivity:`field low` to :gfem-effectivity:`field high`, "
        "quantity :gfem-effectivity:`quantity low` to "
        ":gfem-effectivity:`quantity high`, s :gfem-effectivity:`s high`, loss jump "
        ":gfem-effectivity:`lumped port across a loss jump low` to "
        ":gfem-effectivity:`lumped port across a loss jump high`.",
    )
    assert status == 0
    ranges = derived["effectivity"]
    field, quantity, s = ranges["field"], ranges["quantity"], ranges["s"]
    jump = ranges["lumped port across a loss jump"]
    text = page.read_text(encoding="utf-8")
    assert (
        f"Field {field['low']} to {field['high']}, quantity {quantity['low']} to "
        f"{quantity['high']}, s {s['high']}, loss jump {jump['low']} to "
        f"{jump['high']}." in text
    )


def test_the_effectivity_role_renders_the_port_mode_share(tmp_path, derived):
    status, page = _build_role_page(
        tmp_path, "Below :gfem-effectivity:`port mode share high`."
    )
    assert status == 0
    share = derived["effectivity"]["port mode share"]["high"]
    assert f"Below {share}." in page.read_text(encoding="utf-8")


def test_the_effectivity_table_directive_renders_every_emitted_kind(tmp_path):
    from gfem.refinement import EMITTED_KINDS

    status, page = _build_role_page(tmp_path, ".. gfem-effectivity-table::")
    assert status == 0
    text = page.read_text(encoding="utf-8")
    for kind in EMITTED_KINDS:
        assert caps.KIND_WORDS[kind] in text


@pytest.mark.parametrize(
    "text",
    [
        "low",
        "high",
        "field",
        "quantity middle",
        "loss low",
        "jump high",
        "loss jump low",
        "port mode share low",
    ],
)
def test_an_effectivity_role_with_no_range_name_fails_the_build(tmp_path, text):
    # Several ranges exist: a bare end is ambiguous, so it is an error.
    status, _ = _build_role_page(tmp_path, f"From :gfem-effectivity:`{text}`.")
    assert status != 0


# --- the test-link role -------------------------------------------------------------

_TEST_URL = "https://example.org/blob/main/"


def _build_test_role_page(tmp_path, body, *, markdown=False):
    build = pytest.importorskip("sphinx.cmd.build")
    source = tmp_path / "source"
    source.mkdir()
    extensions = (
        "['gfem_capabilities', 'myst_parser']"
        if markdown
        else ("['gfem_capabilities']")
    )
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(REPO / 'docs' / '_ext')!r})\n"
        f"extensions = {extensions}\n"
        f"gfem_capabilities_source_url = {_TEST_URL!r}\n",
        encoding="utf-8",
    )
    if markdown:
        (source / "index.md").write_text(f"# Tests\n\n{body}\n", encoding="utf-8")
    else:
        (source / "index.rst").write_text(f"Tests\n=====\n\n{body}\n", encoding="utf-8")
    out = tmp_path / "html"
    status = build.build_main(["-q", "-W", "-b", "html", str(source), str(out)])
    return status, out / "index.html"


def test_the_test_role_links_a_file_a_node_and_a_directory(tmp_path):
    node = (
        "tests/verification/test_gradient_matrix.py"
        "::test_gradient_matrix_matches_central_differences"
    )
    status, page = _build_test_role_page(
        tmp_path,
        ":gfem-test:`tests/verification/test_gradient_matrix.py`, "
        f":gfem-test:`{node}` and :gfem-test:`tests/mpi`.",
    )
    assert status == 0
    text = page.read_text(encoding="utf-8")
    assert f'href="{_TEST_URL}tests/verification/test_gradient_matrix.py"' in text
    assert f'href="{_TEST_URL}tests/mpi"' in text
    assert node in text
    assert text.count(f'href="{_TEST_URL}tests/verification/') == 2


def test_the_test_role_renders_inside_a_pipe_table_cell(tmp_path):
    pytest.importorskip("myst_parser")
    status, page = _build_test_role_page(
        tmp_path,
        "| Case | Test |\n| --- | --- |\n"
        "| A | {gfem-test}`tests/verification/test_gradient_matrix.py` |\n",
        markdown=True,
    )
    assert status == 0
    text = page.read_text(encoding="utf-8")
    cell = text.split("<td")[2]
    assert f'href="{_TEST_URL}tests/verification/test_gradient_matrix.py"' in cell


@pytest.mark.parametrize(
    "target",
    [
        "tests/verification/test_no_such_file.py",
        "tests/no_such_directory",
        "tests/verification/test_gradient_matrix.py::test_no_such_node",
        "tests/verification/no_such.py::test_x",
        "../outside.py",
        "",
    ],
)
def test_the_test_role_with_a_missing_target_fails_the_build(tmp_path, target):
    status, _ = _build_test_role_page(tmp_path, f"See :gfem-test:`{target}`.")
    assert status != 0
