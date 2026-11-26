r"""The "SPICE and Verilog-A export" recipe runs and prints what it shows.

docs/recipes/rational-model.md runs nothing at docs build time. This runs its
blocks in order, as a reader does, on the repository's own
docs/tutorials/data/line.msh, then reads the files it writes: the SPICE
subcircuit, solved by the test-only AC reader, gives the line's closed-form
transmission and its matched-port DC limit, and the Verilog-A module has the
same pins. The decks the page shows instantiate those files by their names
and pins. A circuit simulator itself runs only in the ``netlist`` tests.
"""

import re

import docs_pages as dp
import numpy as np
import pytest
import spice_mna

PAGE = dp.DOCS / "recipes" / "rational-model.md"
Z_LINE = 376.73 / 6


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("rational_model")


@pytest.fixture(scope="module")
def ran(workdir):
    return dp.run_page(PAGE, workdir)


def _printed(ran, section: str) -> str:
    (example,) = [example for example in ran if example.section == section]
    return example.printed


def test_the_page_downloads_the_tutorial_mesh(page):
    first = dp.examples(page)[0]
    assert f'url = "{dp.MESH_URL}"' in first.code


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_every_section_with_a_number_prints_it(ran):
    printing = {example.section for example in ran if example.printed.strip()}
    assert printing == {
        "Fit a rational model",
        "Write the netlist",
        "The step a transient must reproduce",
    }


def test_the_fit_is_passive_within_tolerance_on_the_ports_resistance(ran):
    shown, z_ref = _printed(ran, "Fit a rational model").splitlines()
    assert "passive=True" in shown
    error = float(re.search(r"error=([\d.e-]+)", shown)[1])
    assert 0 < error <= 1e-3
    assert dp.numbers(z_ref) == pytest.approx([Z_LINE, Z_LINE], rel=1e-9)


def test_the_closed_form_step_agrees_with_the_windowed_one_once_settled(ran):
    assert _printed(ran, "The step a transient must reproduce").strip() == "True"


@pytest.mark.usefixtures("ran")  # the page writes the files
def test_the_subcircuit_gives_the_lines_transmission(workdir):
    text = (workdir / "line.sp").read_text()
    frequency = np.array([0.0, 2e9, 5e9, 8e9, 15e9])
    s = spice_mna.s_parameters(text, frequency, [Z_LINE, Z_LINE])
    # the matched-line formula of a 20 mm, eps_r = 10 section; 1 at DC
    expected = dp.line_formula(frequency[1:])
    assert abs(s[1:, 1, 0]) == pytest.approx(expected, abs=5e-3)
    assert s[0] == pytest.approx(np.eye(2)[::-1], abs=2e-3)  # a through at DC
    assert s[:, 0, 1] == pytest.approx(s[:, 1, 0], abs=1e-6)  # reciprocal


@pytest.mark.usefixtures("ran")
def test_the_netlists_name_the_reference_and_the_pins(workdir):
    spice = (workdir / "line.sp").read_text()
    verilog = (workdir / "line.va").read_text()
    pins, _ = spice_mna.subcircuit(spice)
    assert pins == ["p1", "p1_ref", "p2", "p2_ref"]
    assert "module line(p1, p1_ref, p2, p2_ref);" in verilog
    for text in (spice, verilog):
        assert "62.788" in text  # the reference impedance in the header
        assert "exp(+j omega t)" in text


def _decks(page: str) -> list[str]:
    section = dp.section(page, "Write the netlist")
    return [block for block in dp.blocks(section, "text") if ".end" in block]


def test_the_spice_deck_instantiates_the_subcircuit_with_its_four_pins(page):
    deck = next(block for block in _decks(page) if ".include" in block)
    assert ".include line.sp" in deck
    (instance,) = [line for line in deck.splitlines() if line.startswith("x")]
    *_, subckt = instance.split()
    assert subckt == "line"
    assert len(instance.split()) == 1 + 4 + 1  # name, two pins per port, subckt
    # the sources it terminates on are the ports' reference resistance
    for line in deck.splitlines():
        if line.startswith("r"):
            assert float(line.split()[-1]) == pytest.approx(Z_LINE, abs=5e-3)


def test_the_verilog_a_deck_loads_the_compiled_module(page):
    deck = next(block for block in _decks(page) if "pre_osdi" in block)
    assert "pre_osdi line.osdi" in deck
    assert ".model line_va line" in deck
    (instance,) = [line for line in deck.splitlines() if line.startswith("n")]
    assert instance.split()[-1] == "line_va"
    assert len(instance.split()) == 1 + 4 + 1
    (command,) = dp.blocks(dp.section(page, "Write the netlist"), "bash")
    assert command.split() == ["openvaf-r", "line.va", "-o", "line.osdi"]


def test_the_page_links_its_terms_and_pages(page):
    for link in (
        "{term}`rational model`",
        "{term}`passivity violation`",
        "{term}`harmonic grid`",
        "{doc}`time-responses`",
        "{doc}`save-and-export`",
        "{doc}`../conventions`",
    ):
        assert link in page, link
