"""The Floquet port recipe names public objects, and its closed form is right.

docs/recipes/floquet-ports.md runs nothing at docs build time: its meshes are
the reader's own. Its slab formula is checked here against the tangential-E
Airy form the Floquet verification cases use, its public names against
gfem, and the page against quoting test evidence.
"""

import math
import re

import docs_pages as dp
import numpy as np
import periodic_cases as pc
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "floquet-ports.md"

#: Every tolerance the Floquet verification cases assert.
TOLERANCES = (pc.F1_COARSE_TOL, pc.F1_S_TOL, pc.F1_CROSS_TOL, pc.BREWSTER_TOL)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def airy(page):
    """The page's ``airy`` function, run from its own block."""
    (block,) = [b for b in dp.blocks(page, "python") if b.startswith("def airy")]
    head = block.split("\n\n\nfor ")[0]  # the function, not its loop over S
    namespace = {"np": np, "c": pc.C0}
    exec(compile(head, str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace["airy"]


@pytest.mark.parametrize("pol", ["TE", "TM"])
@pytest.mark.parametrize("theta", [0.0, 30.0, 60.0])
def test_the_slab_formula_is_the_tangential_e_airy_form(airy, pol, theta):
    f, n1, n2, n3, h = pc.F1_FREQUENCY, 1.0, 2.0, 1.5, 0.3e-6
    kt = 2 * math.pi * f / pc.C0 * n1 * math.sin(math.radians(theta))
    expected = pc.airy(f, kt, [(n1**2, 0.0), (n2**2, h), (n3**2, 0.0)], pol)
    got = airy(n1, n2, n3, h, f, math.radians(theta), pol)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_the_page_quotes_no_case_id(page):
    assert not re.search(r"\*\*[CFR]\d\b", page)


@pytest.mark.parametrize("value", TOLERANCES)
def test_the_page_quotes_no_test_tolerance(page, value):
    mantissa, exponent = f"{value:.0e}".split("e")
    written = f"{mantissa}e{int(exponent)}"
    assert not re.search(rf"(?<![\d.e-]){re.escape(written)}(?!\d)", page)


def test_the_page_names_no_test_file_and_no_other_solver(page):
    assert "tests/" not in page
    assert "palace" not in page.lower()
    assert "gdsfactory" not in page.lower()


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert names
    assert names - {"conventions", "design"} <= set(gfem.__all__)


def test_the_page_states_the_option_default_it_quotes(page):
    default = gfem.SolverOptions().floquet_order_tolerance
    assert f"`SolverOptions.floquet_order_tolerance` ({default:.0e}".replace(
        "e-0", "e-"
    ) in " ".join(page.split())


@pytest.mark.parametrize(
    "text",
    [
        # reciprocity, orders relabelled
        r"\sigma_p \sigma_q S_{-q\leftarrow -p}(-\mathbf{k})",
        r"(S_{:,\mathrm{TM}} + i S_{:,\mathrm{TE}})",
        "wave_vector_frequency=f",  # the fixed-angle sweep
        "periodic=(True, False)",  # the periodic design grid
        "{py:obj}`gfem.UnrequestedOrderWarning`",
        "{doc}`../verification`",
        "{doc}`../tutorials/metagrating`",
        "{doc}`diffraction-efficiency`",
    ],
)
def test_the_page_carries(page, text):
    flat = " ".join(page.split())
    assert text in flat or text in page, text


def test_floquet_text_names_the_floquet_surface_never_a_port_face(page):
    # GLOSSARY: a Floquet port's face is its Floquet surface ("port face" is
    # the term to avoid), in the recipe, the capabilities limit and the result
    capabilities = (dp.DOCS / "capabilities.md").read_text(encoding="utf-8")
    (limit,) = [
        r
        for r in dp.section(capabilities, "Limits").splitlines()
        if r.startswith("| **Floquet")
    ]
    texts = {
        "recipe": page,
        "capabilities": limit,
        "DiffractionOrderResult": gfem.DiffractionOrderResult.__doc__ or "",
    }
    for where, text in texts.items():
        assert "port face" not in " ".join(text.split()), where
