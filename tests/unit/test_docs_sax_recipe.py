"""The "SAX circuits with gfem blocks" recipe runs and is right.

docs/recipes/sax-circuit.md runs nothing at docs build time. This runs its
blocks in order, as a reader does, on the repository's own
docs/tutorials/data/bend2d.msh and line.msh, and compares what they print with
the page's output blocks. Then it rebuilds each example without a component:
the photonic interferometer from the bend's own solve at 1.55 um composed in
SAX, and the RF reflection from the line's S renormalized to 50 ohm and a
series capacitor written out by hand. The gradients on the page are checked
against central differences of those independent versions.
"""

import contextlib
import re
import shutil
import urllib.request

import docs_pages as dp
import numpy as np
import pytest

pytest.importorskip("sax")
pytest.importorskip("skrf")

import sax
from scipy.constants import c

import gfem

PAGE = dp.DOCS / "recipes" / "sax-circuit.md"
PHOTONIC = "A photonic circuit"
BAND = "The band and the circuit's wavelengths"
RF = "An RF circuit"
CHECK = "Check your results"
UM = 1e-6


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("sax_circuit"))


@pytest.fixture(scope="module")
def code():
    return [e.code for e in dp.examples(PAGE.read_text(encoding="utf-8"))]


def _printed(ran, section: str) -> list[str]:
    return [e.printed for e in ran if e.section == section]


@pytest.fixture(scope="module")
def model(code, tmp_path_factory):
    """The page's models, without their components: its code up to each one."""
    workdir = tmp_path_factory.mktemp("sax_circuit_model")

    def urlretrieve(url, filename):
        shutil.copy(dp.MESHES[url], workdir / filename)

    namespace: dict = {}
    first = next(block for block in code if "bend2d.msh" in block)
    second = next(block for block in code if "line.msh" in block)
    with pytest.MonkeyPatch.context() as mp, contextlib.chdir(workdir):
        mp.setattr(urllib.request, "urlretrieve", urlretrieve)
        for source in (
            first.split("wl = np.linspace(1.50")[0],
            second.split("section = gfem.SaxComponent")[0],
        ):
            exec(compile(source, str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace


def test_the_page_downloads_both_meshes(code):
    for name in ("bend2d.msh", "line.msh"):
        (block,) = [b for b in code if f'tutorials/data/{name}"' in b]
        assert f'urlretrieve(url, "{name}")' in block


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_page_names_neither_the_forbidden_products_nor_local_notes():
    text = PAGE.read_text(encoding="utf-8").lower()
    for word in ("palace", "flexcompute", "adr", ".scratch", "ticket"):
        assert not re.search(rf"\b{re.escape(word)}\b", text), word


def test_the_component_has_the_settings_of_the_model_function_and_the_wavelength(ran):
    assert _printed(ran, PHOTONIC)[0] == "['density', 'wl']\n"


def _guide(model) -> dict:
    """The page's analytic guide: the bend's port mode index at three wavelengths."""
    wavelengths = c / (np.array([1.54, 1.55, 1.56]) * UM)
    modes = gfem.solve(
        model["bend"], gfem.PortMode("p1", wavelengths, polarization="Ez")
    )
    n = np.real(np.asarray(modes.n_eff))
    return {"wl0": 1.55, "neff": n[1], "ng": n[1] - 1.55 * (n[2] - n[0]) / 0.02}


def _interferometer(bend_model, guide, dl):
    """Power at the bar output, the bend's S solved directly at 1.55 um."""
    direct = gfem.solve(
        bend_model, gfem.Driven(c / (1.55 * UM), polarization="Ez")
    ).s.to_sdict(circuit="optical", ports={"p1": "o1", "p2": "o2"})
    fixed = {pair: values[0] for pair, values in direct.items()}
    circuit, _ = sax.circuit(
        netlist={
            "instances": {
                "split": "coupler",
                "top": "bend",
                "bottom": "bend",
                "top_wg": "guide",
                "bottom_wg": "guide",
                "join": "coupler",
            },
            "connections": {
                "split,out0": "top,o1",
                "top,o2": "top_wg,in0",
                "top_wg,out0": "join,in0",
                "split,out1": "bottom,o1",
                "bottom,o2": "bottom_wg,in0",
                "bottom_wg,out0": "join,in1",
            },
            "ports": {"in": "split,in0", "bar": "join,out0", "cross": "join,out1"},
        },
        models={
            "coupler": sax.models.coupler_ideal,
            "bend": lambda: fixed,
            "guide": sax.models.straight,
        },
    )
    s = circuit(
        wl=1.55,
        top_wg={**guide, "length": 10.0},
        bottom_wg={**guide, "length": 10.0 + dl},
    )
    return s, abs(s["in", "bar"]) ** 2


def test_the_interferometer_is_the_bends_own_solve_composed_in_sax(ran, model):
    guide = _guide(model)
    printed = _printed(ran, PHOTONIC)
    n_eff, n_g = dp.numbers(printed[1])
    assert n_eff == pytest.approx(guide["neff"], abs=5e-5)
    assert n_g == pytest.approx(guide["ng"], abs=5e-5)
    _, power = _interferometer(model["bend_at"](), guide, 20.0)
    shown, shape_rows, shape_cols, g_dl = dp.numbers(printed[2])
    assert (shape_rows, shape_cols) == (60, 60)
    assert shown == pytest.approx(float(power), abs=1e-4)
    # the arms' length difference is SAX's alone: a central difference of the
    # directly composed circuit gives its derivative
    step = 1e-3
    fd = (
        _interferometer(model["bend_at"](), guide, 20.0 + step)[1]
        - _interferometer(model["bend_at"](), guide, 20.0 - step)[1]
    ) / (2 * step)
    assert g_dl == pytest.approx(float(fd), abs=1e-4)


def test_the_spectrum_and_the_power_balance_stay_physical(ran):
    shape, low, _, high = _printed(ran, BAND)[0].split()
    assert shape == "(1001,)"
    assert 0.0 <= float(low) < float(high) <= 1.0
    # the fringe is visible across the band: the output nearly extinguishes
    assert float(low) < 0.05
    _, balance = _printed(ran, CHECK)[1].split()
    assert 0.0 < float(balance) <= 1.0


def test_the_gradient_checks_agree_to_their_printed_digits(ran):
    sax_only, along = dp.numbers(_printed(ran, CHECK)[0]).reshape(2, 2)
    assert sax_only[0] == pytest.approx(sax_only[1], abs=1e-4)
    assert along[0] == pytest.approx(along[1], rel=2e-3)


def test_the_component_at_a_band_frequency_is_the_solve(ran):
    assert _printed(ran, CHECK)[1].split()[0] == "True"


def _series_capacitor(line_model, eps_r: float, capacitance: float) -> float:
    """|S11|^2 at 5 GHz of the line behind a series capacitor, on 50 ohm."""
    s = gfem.solve(line_model(eps_r=eps_r), gfem.Driven(frequency=5e9)).s
    a = np.asarray(s.renormalize(50.0).values)[0]
    z = 1.0 / (2j * np.pi * 5e9 * capacitance)  # the capacitor's impedance
    gamma = z / (z + 2 * 50.0)  # its reflection with the far port matched
    s11 = a[0, 0] + a[0, 1] * a[1, 0] * gamma / (1 - a[1, 1] * gamma)
    return float(abs(s11) ** 2)


def test_the_rf_reflection_is_the_renormalized_line_behind_a_capacitor(ran, model):
    value, g_eps, g_cap = dp.numbers(_printed(ran, RF)[0])
    by_hand = _series_capacitor(model["line_at"], 10.0, 2e-12)
    assert value == pytest.approx(by_hand, abs=1e-4)
    # the capacitance is SAX's, the permittivity gfem's: both gradients match
    # central differences of the hand-written chain
    step = 1e-3
    fd_eps = (
        _series_capacitor(model["line_at"], 10.0 + step, 2e-12)
        - _series_capacitor(model["line_at"], 10.0 - step, 2e-12)
    ) / (2 * step)
    assert g_eps == pytest.approx(fd_eps, rel=2e-3)
    h = 1e-15
    fd_cap = (
        _series_capacitor(model["line_at"], 10.0, 2e-12 + h)
        - _series_capacitor(model["line_at"], 10.0, 2e-12 - h)
    ) / (2 * h)
    assert g_cap == pytest.approx(fd_cap, rel=2e-3)


def test_a_solved_block_and_the_component_give_the_same_reflection(ran):
    (fixed,) = _printed(ran, "A solved result as a circuit block")
    value = dp.numbers(_printed(ran, RF)[0])[0]
    assert float(fixed) == pytest.approx(value, abs=1e-4)


def test_the_scikit_rf_cascade_is_the_same_reflection(ran):
    value = dp.numbers(_printed(ran, RF)[0])[0]
    assert float(_printed(ran, CHECK)[-1]) == pytest.approx(value, abs=1e-4)
