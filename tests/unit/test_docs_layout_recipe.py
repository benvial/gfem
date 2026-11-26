"""The "GDS export of a design" recipe runs and writes its file.

docs/recipes/fabricable-layout.md runs nothing at docs build time. This runs
its blocks in order, as a reader does, on the repository's own
docs/tutorials/data/bend2d.msh with the hand-drawn arc as the design, and
compares what they print with the page's output blocks (the pattern, the width
and spacing checks, the printed variants and the one re-solve). Both GDS files
the page writes are read back by an independent reader (gdstk): the layers
and datatypes the page names, the polygons and their area. The numbers the
page quotes from the bend tutorial are checked against its committed outputs.
"""

import json
import re

import docs_pages as dp
import numpy as np
import pytest

PAGE = dp.DOCS / "recipes" / "fabricable-layout.md"
NOTEBOOK = dp.DOCS / "tutorials" / "bend.ipynb"
#: The bias of the page's printed variants on each side, nm: the example process's
#: -10 nm edge bias minus its 10 nm tolerance.
ERODED_BIAS_NM = -20.0
#: The example process's blur, nm.
SIGMA_NM = 100.0


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("fabricable_layout")
    return workdir, dp.run_page(PAGE, workdir)


@pytest.fixture(scope="module")
def ran(page):
    return page[1]


@pytest.fixture(scope="module")
def workdir(page):
    return page[0]


def _printed(ran, needle: str) -> str:
    (example,) = [e for e in ran if needle in e.code]
    return example.printed


def _prose(text: str) -> str:
    return re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)


def test_the_page_downloads_the_bend_mesh():
    first = dp.examples(PAGE.read_text(encoding="utf-8"))[0]
    assert "bend2d.msh" in first.code
    # Matched on the quoted address alone: ruff wraps the long assignment.
    assert f'"{dp.MESH_URL.replace("line.msh", "bend2d.msh")}"' in first.code


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def _gds(path):
    gdstk = pytest.importorskip("gdstk")
    library = gdstk.read_gds(str(path))
    assert library.unit == pytest.approx(1e-6)
    assert library.precision == pytest.approx(1e-9)
    (cell,) = library.cells
    layers: dict[tuple[int, int], list] = {}
    for polygon in cell.polygons:
        layers.setdefault((polygon.layer, polygon.datatype), []).append(polygon)
    return cell.name, layers


def test_the_gds_file_holds_the_pattern_and_the_guides_on_layer_1(ran, workdir):
    name, layers = _gds(workdir / "bend.gds")
    assert name == "bend"
    assert set(layers) == {(1, 0)}
    # The page prints one polygon of pattern and two guides, none with a hole.
    assert "1 polygon, 0 holes" in _printed(ran, "print(pattern)")
    assert "2 polygons, 0 holes" in _printed(ran, "print(guides)")
    assert len(layers[1, 0]) == 3
    # The ring (0.55 um wide, radius 1.5 um, a quarter turn) and two guides
    # 1 um x 0.55 um, in um^2; the ring's filtered contour keeps its area to
    # well within a percent.
    area = sum(p.area() for p in layers[1, 0])
    ring = np.pi / 4 * (1.775**2 - 1.225**2)
    assert area == pytest.approx(ring + 2 * 0.55, rel=5e-3)


def test_the_review_file_adds_the_printed_variants_on_their_datatypes(workdir):
    name, layers = _gds(workdir / "bend_review.gds")
    assert name == "bend_review"
    assert set(layers) == {(1, 0), (1, 10), (1, 11)}
    eroded = sum(p.area() for p in layers[1, 10])
    drawn = sum(p.area() for p in layers[1, 0]) - 2 * 0.55  # without the guides
    dilated = sum(p.area() for p in layers[1, 11])
    assert eroded < drawn
    # The dilated variant prints at zero bias (-10 nm + 10 nm): the drawn ring,
    # up to the blur's corner rounding where it meets the footprint.
    assert dilated == pytest.approx(drawn, rel=2e-3)


def test_the_eroded_ring_prints_the_bias_narrower(ran):
    drawn = dp.numbers(_printed(ran, "print(check)"))[1]
    eroded = dp.numbers(_printed(ran, "eroded.check("))[1]
    # The page: "each edge sits within about 0.01 sigma (1 nm) of where the
    # bias puts it", so the width within two of them.
    assert abs(eroded - (drawn + 2 * ERODED_BIAS_NM)) <= 2 * 0.015 * SIGMA_NM
    prose = _prose(PAGE.read_text(encoding="utf-8"))
    assert f"{eroded:.0f} nm wide" in prose
    assert f"against {drawn:.0f} nm drawn" in prose


def test_the_numbers_quoted_from_the_bend_tutorial_are_its_outputs():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    printed = " ".join(
        "".join(o.get("text", ""))
        for c in cells
        if c["cell_type"] == "code"
        for o in c["outputs"]
    )
    tutorial = re.search(
        r"In the bend tutorial the drawn\s+pattern of the optimized design transmits "
        r"([\d.]+) against ([\d.]+)",
        PAGE.read_text(encoding="utf-8"),
    )
    assert tutorial, "the page no longer quotes the bend tutorial"
    pattern, last = tutorial.groups()
    assert f"drawn pattern: |S21|^2 = {pattern}" in printed
    assert f"iteration 41: |S21|^2 = {last}" in printed
