"""The "Mesh preparation" recipe: its gmsh script and its gfem blocks.

docs/recipes/prepare-a-mesh.md runs nothing at docs build time, and gmsh is
not a gfem dependency. The page's gmsh blocks are kept verbatim parts of
tools/make_guide_mesh.py, whose output is committed as
tests/unit/data/guide_line.msh. The page's gfem blocks run here on that copy,
and what they print is compared with the matched-line formula and with the
output blocks the page shows. Where gmsh is installed, the script is run
again and must give the same regions.
"""

import runpy
import shutil
from pathlib import Path

import docs_pages as dp
import numpy as np
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "prepare-a-mesh.md"
SCRIPT = dp.ROOT / "tools" / "make_guide_mesh.py"
DATA = Path(__file__).parent / "data"
GUIDE_MESH = DATA / "guide_line.msh"
GMSH_SECTION = "Build and name the regions"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("prepare_a_mesh")
    shutil.copy(GUIDE_MESH, workdir / "my_line.msh")
    shutil.copy(DATA / "two_box_msh41.msh", workdir / "chip.msh")
    shutil.copy(DATA / "coax_order2.msh", workdir / "coax.msh")
    examples = dp.run_page(PAGE, workdir, skip=(GMSH_SECTION,))
    return examples, workdir


def _printed(examples, section: str) -> list[np.ndarray]:
    found = [dp.numbers(e.printed) for e in examples if e.section == section]
    assert found, f"no Python block under {section!r}"
    return found


def test_every_gmsh_block_is_part_of_the_script():
    page = PAGE.read_text(encoding="utf-8")
    shown = dp.blocks(dp.section(page, GMSH_SECTION), "python")
    assert len(shown) == 4
    script = SCRIPT.read_text(encoding="utf-8")
    for block in shown:
        assert block in script, block
    # Together the blocks are the whole script after its docstring.
    body = script.split('"""', 2)[2]
    assert "".join("".join(shown).split()) == "".join(body.split())


def test_every_output_block_is_what_the_code_prints(ran):
    examples, _ = ran
    assert not dp.output_problems(examples), "\n".join(dp.output_problems(examples))


def test_the_names_and_the_unit_are_the_scripts(ran):
    examples, _ = ran
    (read,) = [e for e in examples if e.section == "Read it and check the names"]
    assert "('feed', 'design')" in read.printed
    assert "('bottom', 'top', 'port1', 'port2')" in read.printed
    (corner,) = _printed(examples, "Read it and check the names")
    assert corner[-3:] == pytest.approx([0.04, 0.006, 0.001], abs=1e-12)


def test_the_line_transmits_what_the_formula_says(ran):
    examples, _ = ran
    (s21,) = _printed(examples, "Solve on it")
    frequencies = np.array([2e9, 5e9, 8e9])
    assert s21 == pytest.approx(dp.line_formula(frequencies), abs=1e-3)
    # The page compares with the get-started values to the fourth digit.
    assert s21 == pytest.approx([0.8306, 0.9013, 0.6052], abs=6e-5)


def test_order_3_moves_s21_by_about_1e_5_at_8_ghz(ran):
    examples, _ = ran
    (printed,) = _printed(examples, "Is it fine enough?")
    order_3, change = printed[:3], printed[3]
    (order_2,) = _printed(examples, "Solve on it")
    # The page says "about 1e-5, most at 8 GHz".
    assert 5e-6 < change < 2e-5
    assert np.argmax(np.abs(order_3 - order_2)) == 2


def test_the_mesh_file_is_written(ran):
    _, workdir = ran
    assert (workdir / "my_line.vtu").stat().st_size > 0


# -- What a bad mesh raises ---------------------------------------------------


def _msh22(elements: list[str], names: list[str]) -> str:
    """A gmsh 2.2 file with the unit tetrahedron's nodes plus a cube corner."""
    nodes = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat"]
    if names:
        lines += ["$PhysicalNames", str(len(names)), *names, "$EndPhysicalNames"]
    lines += ["$Nodes", str(len(nodes))]
    lines += [f"{i} {x} {y} {z}" for i, (x, y, z) in enumerate(nodes, start=1)]
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [f"{i} {e}" for i, e in enumerate(elements, start=1)]
    lines += ["$EndElements", ""]
    return "\n".join(lines)


TET = "4 2 {phys} 1 1 2 3 4"
BAD_MESHES = {
    "no physical groups": (_msh22(["4 0 1 2 3 4"], []), "no physical groups"),
    "hexahedra": (
        _msh22(["5 2 1 1 1 2 5 3 4 6 8 7"], ['3 1 "box"']),
        "tetrahedral meshes only",
    ),
    "a volume outside any group": (
        _msh22([TET.format(phys=1), "4 2 0 2 2 3 4 8"], ['3 1 "box"']),
        "outside any",
    ),
    "a group with no name": (
        _msh22([TET.format(phys=1), "4 2 2 2 2 3 4 8"], ['3 1 "box"']),
        "no name",
    ),
}


@pytest.mark.parametrize("case", sorted(BAD_MESHES))
def test_a_bad_mesh_raises_with_its_remedy(case, tmp_path):
    text, phrase = BAD_MESHES[case]
    path = tmp_path / "bad.msh"
    path.write_text(text)
    with pytest.raises(ValueError, match=phrase):
        gfem.Mesh.from_gmsh(path)


@pytest.mark.parametrize(
    ("fixture", "phrase"),
    [
        ("octant_order3.msh", "order 3 or more"),
        ("cube_overlap_msh41.msh", "more than one"),
    ],
)
def test_a_high_order_or_overlapping_mesh_raises(fixture, phrase):
    with pytest.raises(ValueError, match=phrase):
        gfem.Mesh.from_gmsh(DATA / fixture)


def test_the_page_lists_each_bad_mesh():
    page = PAGE.read_text(encoding="utf-8")
    listed = dp.section(page, "What a bad mesh raises")
    for words in (
        "No physical groups",
        "Order 3 and above",
        "Hexahedra",
        "outside any volume group",
        "without a name",
        "two groups",
    ):
        assert words in listed, words


def test_a_second_order_file_reads_as_a_curved_mesh_not_a_refusal():
    # The page's curved block reads coax.msh; a cube written at order 2 keeps
    # flat faces, so its mid-edge nodes sit on the chords.
    assert gfem.Mesh.from_gmsh(DATA / "coax_order2.msh").curved
    cube = gfem.Mesh.from_gmsh(DATA / "cube_order2_msh41.msh")
    assert cube.curved


def test_the_curved_block_prints_true(ran):
    examples, _ = ran
    (printed,) = [e.printed for e in examples if e.section == "Curved walls"]
    assert printed.strip() == "True"


def test_the_page_no_longer_calls_the_mesh_straight_sided_only():
    page = PAGE.read_text(encoding="utf-8")
    assert "Straight-sided tetrahedra only" not in page
    assert "second-order (curved) mesh: regenerate" not in page


# -- The script, where gmsh is installed --------------------------------------


def test_the_script_makes_the_committed_mesh_s_regions(tmp_path, monkeypatch):
    pytest.importorskip("gmsh")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    made = gfem.Mesh.from_gmsh(tmp_path / "my_line.msh", length_unit="mm")
    committed = gfem.Mesh.from_gmsh(GUIDE_MESH, length_unit="mm")
    assert made.regions == committed.regions
    assert np.ptp(made.imported_nodes, axis=0) == pytest.approx(
        np.ptp(committed.imported_nodes, axis=0)
    )
