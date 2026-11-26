"""No docs page ships a placeholder.

A number not yet measured is written between angle brackets while a change is in
flight (``⟨0.78 → 0.46⟩``). It must be measured or its sentence deleted before the
docs are built for a release.
"""

from pathlib import Path

DOCS = Path(__file__).parents[2] / "docs"
PLACEHOLDER = ("⟨", "⟩")


def test_no_docs_page_holds_a_placeholder():
    pages = sorted(DOCS.rglob("*.md")) + sorted(DOCS.rglob("*.rst"))
    assert pages
    found = [
        f"{page.relative_to(DOCS)}:{number}"
        for page in pages
        for number, line in enumerate(
            page.read_text(encoding="utf-8").splitlines(), start=1
        )
        if any(mark in line for mark in PLACEHOLDER)
    ]
    assert not found, f"placeholders left in: {found}"


def test_the_thermal_block_flags_the_density_rename_as_the_code_does():
    import warnings

    import pytest

    import gfem

    notes = (DOCS / "project" / "release-notes.md").read_text(encoding="utf-8")
    (release,) = (s for s in notes.split("\n## ") if s.startswith("0.1.0 "))
    assert "\nHeat and thermo-optics:\n" in release
    block = " ".join(
        release.split("\nHeat and thermo-optics:\n")[1].split("\n\n")[0].split()
    )
    assert block.startswith("- **Behaviour change: `density` is now `mass_density`.**")
    for said in ("`DeprecationWarning`", "`AttributeError`", "Job files"):
        assert said in block, said
    # what the block says the old name does
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        material = gfem.Material(density=2203.0)
    assert [w.category for w in caught] == [DeprecationWarning]
    assert material.mass_density == 2203.0
    with pytest.raises(AttributeError, match="mass_density"):
        _ = material.density


def test_the_unreleased_thermal_block_flags_the_joule_heating_type_error():
    import pytest

    import gfem

    notes = (DOCS / "project" / "release-notes.md").read_text(encoding="utf-8")
    (unreleased,) = (s for s in notes.split("\n## ") if s.startswith("Unreleased\n"))
    block = " ".join(unreleased.split("\nHeat and thermo-optics:\n")[1].split())
    flag = (
        "**Behaviour change: `joule_heating` of a driven result raises `TypeError`.**"
    )
    assert flag in block
    for said in ("`NotImplementedError`", "`gfem.electromagnetic_heating`"):
        assert said in block.split(flag)[1], said
    # what the block says the call does
    line = DOCS / "tutorials" / "data" / "line.msh"
    mesh = gfem.Mesh.from_gmsh(line, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    problem = gfem.Driven(frequency=5e9, excite="p1", keep_fields=True)
    driven = gfem.solve(model, problem)
    with pytest.raises(TypeError, match="electromagnetic_heating"):
        gfem.joule_heating(driven, power=1e-3)
