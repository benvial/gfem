"""The Brillouin gain recipe runs and prints what the page shows.

docs/recipes/brillouin-gain.md runs nothing at docs build time. This runs its
blocks in order on the suspended silicon wire at the page's mesh (built by the
repository's own docs/recipes/wire_section.py, which the page downloads) and
checks the printed numbers against what they must be: the selected modes are the
E class at the expected frequencies, the total gain of a mode is the amplitude
relation of its parts, the forward gain is the optomechanical coupling recipe's,
the gradient is the stiffness scaling law, and the limits it lists raise what it
says.
"""

import docs_pages as dp
import numpy as np
import pytest

PAGE = dp.DOCS / "recipes" / "brillouin-gain.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("brillouin_gain")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # the section builder it downloads
        return dp.run_page(PAGE, workdir)


def _example(ran, section: str, which: int = 0) -> dp.Example:
    return [example for example in ran if example.section == section][which]


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_q_is_twice_the_optical_wavenumber_of_the_wire(ran):
    (q,) = dp.numbers(_example(ran, "The phase-matched wavenumber").printed)
    # beta = 2 pi n_eff / lambda with the n_eff 1.8392 of the coupling recipe
    # (printed to 5 digits, so relative 3e-5 on n_eff)
    assert q == pytest.approx(2 * 2 * np.pi * 1.8392 / 1.552e-6, rel=1e-4)


def test_the_selected_modes_are_the_e_class_at_the_expected_frequencies(ran):
    lines = _example(ran, "The gain of every mode").printed.splitlines()
    assert lines[0] == "backward"
    # E4, E2, another class, E1 by total; frequencies are mesh independent to
    # 4 digits (the wire's own 13.890 and 15.415 GHz at q)
    frequencies = [dp.numbers(line)[0] for line in lines[1:]]
    assert frequencies[0] == pytest.approx(22.541, abs=2e-3)
    assert frequencies[1] == pytest.approx(15.415, abs=2e-3)
    assert frequencies[3] == pytest.approx(13.890, abs=2e-3)
    # sorted by total
    totals = [dp.numbers(line)[3] for line in lines[1:]]
    assert totals == sorted(totals, reverse=True)


def test_the_totals_are_the_destructive_amplitude_relation_of_the_parts(ran):
    lines = _example(ran, "The gain of every mode").printed.splitlines()[1:]
    for line, destructive in zip(lines, (True, True, False, True, True), strict=True):
        _, mb, pe, total = dp.numbers(line)[:4]
        amplitude = np.sqrt(pe) + np.sqrt(mb)
        if destructive:
            amplitude = abs(np.sqrt(pe) - np.sqrt(mb))
        # three or four printed digits on each of mb, pe and total: the
        # amplitudes carry half the relative error of the gains
        assert total == pytest.approx(amplitude**2, rel=2e-2, abs=1.0)


def test_the_published_comparison_is_the_printed_percentage(ran):
    printed = _example(ran, "The gain of every mode", 1).printed.splitlines()
    for line, published, f_pub in zip(printed, (890, 860), (13.99, 15.54), strict=True):
        f, _mb, _pe, total, text, percent = dp.numbers(line)[:6]
        assert text == published
        assert f == pytest.approx(f_pub, rel=2e-2)  # gfem's 0.7-0.8 % low
        # the printed percentage is the total over the published value
        assert percent == pytest.approx(100 * (total / published - 1), abs=0.06)
        # the gate on the verification tier is 15 % on a finer mesh; here 20 %
        assert abs(percent) < 20


def test_the_forward_gain_is_the_coupling_recipes(ran):
    printed = _example(ran, "Forward").printed
    assert printed.split()[0] == "forward"
    # the same mesh and modes as the g0 recipe's 16802 1/(W m): the printed
    # digits of both pages
    assert dp.numbers(printed)[0] == pytest.approx(16802, abs=0.5)


def test_the_gradient_is_the_stiffness_scaling_law(ran):
    # G ~ E^-1 exactly at fixed Q_m: the mode shape does not change, the 1 J/m
    # displacement ~ E^(-1/2) and |dn|^2 ~ E^-1. The gradient is exact in the
    # discrete problem; the gate is the printed five digits.
    (value,) = dp.numbers(_example(ran, "The gradient").printed)
    assert value == pytest.approx(-1.0, abs=2e-5)


def test_the_limits_raise_what_the_page_says(ran):
    printed = _example(ran, "Limits").printed
    assert printed.splitlines() == [
        "wrong q: ValueError",
        "lossless: ValueError",
        "2D model: NotImplementedError",
    ]
