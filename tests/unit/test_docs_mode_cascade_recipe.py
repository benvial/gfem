"""The "Length sweeps by mode cascade" recipe runs and is right.

docs/recipes/mode-cascade.md runs nothing at docs build time. This runs its
blocks in order, as a reader does, on the repository's own
docs/recipes/mmi_blocks.py, and compares what they print with the page's output
blocks. Then it rebuilds what the page claims without the cascade: the optimum
and two table lengths from full finite element solves of the whole coupler,
and the gradient from a central difference of the printed power.
"""

import re

import docs_pages as dp
import numpy as np
import pytest
from scipy.constants import c

import gfem

PAGE = dp.DOCS / "recipes" / "mode-cascade.md"
RECIPES = dp.DOCS / "recipes"
SWEEP = "Sweep the length"
KEPT = "Kept modes"
DIFFERENTIATE = "Differentiate in the length"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("mode_cascade")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # the page imports its geometry script
        yield dp.run_page(PAGE, workdir)


def _printed(ran, section: str) -> list[str]:
    return [e.printed for e in ran if e.section == section]


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_page_names_neither_the_forbidden_products_nor_local_notes():
    text = PAGE.read_text(encoding="utf-8").lower()
    for word in ("palace", "flexcompute", "tidy3d", "photonforge", "eme", "adr"):
        assert not re.search(rf"\b{word}\b", text), word
    for word in (".scratch", "ticket"):
        assert word not in text, word


def test_every_gfem_name_on_the_page_is_public():
    text = PAGE.read_text(encoding="utf-8")
    names = {
        name
        for name in dp._DOTTED.findall(text)
        if name.rsplit(".", 1)[-1] not in dp._FILE_SUFFIXES
    }
    assert "gfem.cascade" in names
    for name in sorted(names):
        assert name.split(".")[1] in gfem.__all__, name
        dp.resolve(name)


def _full_power(length_um: float) -> float:
    """T at the first output of the whole coupler, one full solve."""
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(RECIPES))
        import mmi_blocks

        mesh = mmi_blocks.full_device(length_um)
    model = gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(eps_r=2.0**2),
            "clad": gfem.Material(eps_r=1.444**2),
        },
        boundaries=[gfem.PEC("side")],
        ports=[
            gfem.WavePort("in1", "left"),
            gfem.WavePort("out1", "right", mode=1),
            gfem.WavePort("out2", "right", mode=2),
        ],
    )
    result = gfem.solve(
        model,
        gfem.Driven(c / 1.55e-6, polarization="Ez"),
        options=gfem.SolverOptions(order=4, linear_solver="direct", threads=1),
    )
    return float(abs(np.asarray(result.s.values)[0, 1, 0]) ** 2)


def test_the_optimum_is_what_one_full_solve_gives(ran):
    sweep = _printed(ran, SWEEP)[0].splitlines()
    best, power = dp.numbers(sweep[0])[:2]
    assert 3.0 < best < 3.5  # the paraxial 1x2 length 3 L_pi / 8 is 3.5 um
    # The cascade and the full solve differ by 6e-8 in S at N = 12 (measured);
    # the page prints T to 5 digits (rounding 5e-6) and L* to 3 digits, where
    # dT/dL = 0 makes the rounding of L* second order. 3e-5 covers both, with
    # margin 5 on the rounding.
    assert _full_power(best) == pytest.approx(power, abs=3e-5)


def test_the_table_lengths_are_what_full_solves_give(ran):
    lines = _printed(ran, SWEEP)[0].splitlines()[1:]
    rows = {float(dp.numbers(line)[0]): dp.numbers(line)[1] for line in lines}
    assert set(rows) >= {3.0, 10.0}
    # the table prints T to 4 digits (rounding 5e-5); the S difference of the
    # two routes at N = 12 is below 1e-6 (measured 6e-8 at L*, and not growing
    # with the length), so 6e-5 is the rounding plus a small margin
    for length in (3.0, 10.0):
        assert _full_power(length) == pytest.approx(rows[length], abs=6e-5)


def test_the_kept_mode_errors_fall_with_the_modes_kept(ran):
    errors = [dp.numbers(line)[-1] for line in _printed(ran, KEPT)[0].splitlines()]
    assert len(errors) == 4
    # measured 1.9e-4, 4.4e-6, 9.5e-8, 5.6e-8: dropping the evanescent modes
    # costs orders of magnitude, and the last step reaches the floor of the
    # two discretizations (mesh and block planes), so it is not strict
    assert errors[0] > 10 * errors[1] > 100 * errors[2]
    assert errors[3] <= 2 * errors[2]
    assert errors[0] > 1e-5 > errors[3]


def test_the_gradient_is_the_central_difference_of_the_power(ran):
    (line,) = _printed(ran, DIFFERENTIATE)
    gradient, difference = dp.numbers(line)[1:]
    # the page's central difference has truncation T''' h^2 / 6 with h = 1e-3
    # um and |T'''| below (2 beta)^3 / 4 of about 1e3 /um^3 (the power is a sum
    # of fringes at the 2 beta ripple, beta about 8 /um): 2e-4. It is measured
    # at 1e-5 or less here, its printed digits agree to 5e-6.
    assert gradient == pytest.approx(difference, abs=2e-4)
    assert gradient == pytest.approx(0.12934, abs=2e-5)


def test_the_check_agrees_to_its_printed_digits(ran):
    cascade, full = dp.numbers(_printed(ran, "Check your results")[0])
    assert cascade == pytest.approx(full, abs=3e-5)
