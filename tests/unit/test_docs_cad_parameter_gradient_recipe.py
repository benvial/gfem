"""The CAD parameter gradient recipe runs, converges and checks its gradient.

docs/recipes/cad-parameter-gradient.md runs nothing at docs build time. This
runs the whole page with gmsh: a strip width sized to 42 aF by Newton steps
against the closed form of a step, one remesh after the first step, and the
gradient against a central difference of remeshed solves.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "cad-parameter-gradient.md"
TARGET = 42.0  # aF
FORBIDDEN = ("Palace", "Flexcompute", "Tidy3D", "PhotonForge")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    # Inside the fixture, not at module level: collecting this file must not
    # import gmsh in jobs that have no system libraries for it. Only a missing
    # module skips; a library that fails to load is an error.
    try:
        import gmsh  # noqa: F401
    except ImportError:
        pytest.skip("gmsh is not installed")
    workdir = tmp_path_factory.mktemp("cad_parameter_gradient")
    return dp.run_page(PAGE, workdir)


def _lines(example) -> list:
    return [dp.numbers(line) for line in example.printed.splitlines()]


def _loop(ran):
    return next(
        e for e in ran if e.section == "Size the strip for 42 aF" and "for _" in e.code
    )


def test_the_loop_output_is_what_the_code_prints(ran):
    # gmsh meshes differ across platforms and versions. Over 6 distinct meshes
    # at one width the capacitance spreads 7e-6 relative (std), jax.grad 1.4e-4
    # and the next Newton width 1.2e-4 (range 3.5e-4), so 1e-3 clears them and
    # 1e-5 would not. The check block compares a remeshed difference with
    # jax.grad at 2e-2: the measured gap is 4e-3 at lc 0.15, h = 1/8 um.
    problems = dp.output_problems(ran[:2], tolerance=1e-3)
    check = dp.output_problems(ran[2:], tolerance=2e-2)
    assert not problems + check, "\n".join(problems + check)


def test_each_newton_line_is_its_closed_form(ran):
    lines = _lines(_loop(ran))
    (w0, c0, g0), (w1,), (w1b, c1, g1), (w2, _c2, _g2) = lines
    # w' = w - (C - 42) / C'; the page prints 4 decimals, whose rounding moves
    # a step by about 1e-4 / 5 * 3.5 = 7e-5 um, so 2e-4 um bounds it.
    assert w1 == pytest.approx(w0 - (c0 - TARGET) / g0, abs=2e-4)
    assert w1b == w1
    assert w2 == pytest.approx(w1b - (c1 - TARGET) / g1, abs=2e-4)


def test_the_strip_reaches_the_target(ran):
    lines = _lines(_loop(ran))
    assert lines[-1][1] == pytest.approx(TARGET, abs=1e-3)


def test_the_remesh_happens_once_after_the_first_step(ran):
    lines = _loop(ran).printed.splitlines()
    marks = [i for i, line in enumerate(lines) if line.startswith("remeshed at")]
    assert marks == [1]


def test_the_check_numbers_agree(ran):
    check = next(e for e in ran if e.section == "Check your results")
    grad, difference = dp.numbers(check.printed)
    assert grad == pytest.approx(difference, rel=2e-2)


def test_the_page_names_its_extra_and_its_faceting_note():
    text = PAGE.read_text(encoding="utf-8")
    assert "gfem[gmsh]" in text
    assert r"0.2\,(\ell/R)^2" in text
    for name in FORBIDDEN:
        assert name.lower() not in text.lower(), name
