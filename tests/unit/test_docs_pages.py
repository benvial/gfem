"""Every user page's code is checked against gfem without solving.

The recipes run nothing at docs build time, and most are fragments on meshes
the repository does not ship, so a renamed keyword or a removed name in one of
them would reach readers unnoticed. This reads every page as a reader copies
from it: its Python blocks parse, every gfem name it writes exists, and every
gfem call binds to the callee's signature. It also keeps the hand-written
tables of contents, the get-started links and the package description in step
with the pages.
"""

import re
import tomllib

import docs_pages as dp
import pytest

PAGES = dp.user_pages()
RECIPES = sorted(page.stem for page in (dp.DOCS / "recipes").glob("*.md"))


def _id(page):
    return page.relative_to(dp.DOCS).as_posix()


# -- The checker itself, on pages written for the purpose --------------------


def _page(*python: str, prose: str = "") -> str:
    fenced = "\n\n".join(f"```python\n{block}\n```" for block in python)
    return f"# A page\n\n{prose}\n\n{fenced}\n"


def test_a_page_of_current_code_has_no_problem():
    page = _page(
        "import gfem\nfrom gfem import Material\n"
        'mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")\n'
        "air = Material(eps_r=1.0)",
        "problem = gfem.Driven(frequency=5e9)\nresult = gfem.solve(model, problem)",
        prose="Use {py:obj}`gfem.Driven` and `gfem.SolverOptions(order=3)`.",
    )
    assert dp.static_problems(page) == []


def test_a_block_that_does_not_parse_is_reported():
    (problem,) = dp.static_problems(_page("result = gfem.solve(model,"))
    assert "block 1 does not parse" in problem


def test_an_indented_block_is_read():
    page = "- A step:\n\n  ```python\n  import gfem\n  gfem.nothing_here\n  ```\n"
    (problem,) = dp.static_problems(page)
    assert "nothing_here" in problem


@pytest.mark.parametrize(
    "text",
    [
        "```python\nimport gfem\ngfem.Mesh.from_nowhere(path)\n```",
        "```python\nfrom gfem import NoSuchThing\n```",
        "```python\nimport gfem.no_such_module\n```",
        "Call `gfem.no_such_function` first.",
        "Call `gfem.no_such_function(x)` first.",
        "See {py:obj}`gfem.NoSuchThing`.",
        "See {py:obj}`the options <gfem.SolverOptions.no_such_field>`.",
        "See {py:meth}`~gfem.Mesh.from_nowhere`.",
    ],
)
def test_a_name_gfem_does_not_have_is_reported(text):
    (problem,) = dp.static_problems(f"# A page\n\n{text}\n")
    assert re.search(r"no '\w+' in gfem", problem)


def test_a_field_without_a_default_is_a_name():
    assert dp.static_problems("The {py:obj}`gfem.Driven.frequency` grid.\n") == []


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        ("gfem.Driven(frequencies=5e9)", "unexpected keyword argument 'frequencies'"),
        ("gfem.Driven()", "missing a required argument: 'frequency'"),
        ("gfem.Mesh.from_gmsh('a.msh', 'mm', 'more')", "too many positional"),
    ],
)
def test_a_call_that_does_not_bind_is_reported(call, reason):
    (problem,) = dp.static_problems(_page(f"import gfem\nproblem = {call}"))
    assert call.split("(")[0] in problem
    assert reason in problem


def test_a_call_through_an_imported_name_is_bound():
    page = _page("from gfem import Driven as Sweep\nproblem = Sweep(frequencies=5e9)")
    (problem,) = dp.static_problems(page)
    assert "frequencies" in problem


def test_a_name_the_page_rebinds_is_the_pages_own():
    page = _page(
        "from gfem import solve",
        "def solve(model, **anything):\n    return model\n\nsolve(1, anything_goes=2)",
    )
    assert dp.static_problems(page) == []


def test_inline_code_may_abbreviate_a_call_but_not_rename_a_keyword():
    assert dp.static_problems("Pass `gfem.Driven(keep_fields=True)`.\n") == []
    assert dp.static_problems("Pass `Driven(keep_fields=True)`.\n") == []
    (problem,) = dp.static_problems("Pass `Driven(keeps_fields=True)`.\n")
    assert "keeps_fields" in problem


def test_unpacked_arguments_leave_the_named_keywords_checked():
    page = _page("import gfem\nproblem = gfem.Driven(*args, frequencies=1.0, **more)")
    (problem,) = dp.static_problems(page)
    assert "frequencies" in problem


def test_a_file_name_is_not_a_gfem_name():
    assert dp.static_problems("Run `python -m gfem run gfem.json`.\n") == []


# -- Every page ---------------------------------------------------------------


def test_the_pages_are_found():
    names = {_id(page) for page in PAGES}
    assert {
        "get-started.md",
        "modes-and-ports.md",
        "capabilities.md",
        "verification.md",
    } <= names
    assert {f"recipes/{name}.md" for name in RECIPES} <= names


@pytest.mark.parametrize("page", PAGES, ids=_id)
def test_the_page_code_matches_gfem(page):
    problems = dp.static_problems(dp.page_text(page))
    assert not problems, "\n".join([f"{_id(page)}:", *problems])


# -- Tables of contents -------------------------------------------------------


def test_every_recipe_is_in_the_site_tree():
    tree = dp.toctree_entries((dp.DOCS / "index.md").read_text(encoding="utf-8"))
    assert sorted(set(RECIPES) - {e.removeprefix("recipes/") for e in tree}) == []


def test_the_modes_and_ports_page_is_in_the_site_tree():
    tree = dp.toctree_entries((dp.DOCS / "index.md").read_text(encoding="utf-8"))
    assert "modes-and-ports" in tree


def test_get_started_next_steps_pick_pages_of_the_site_tree():
    """Next steps is a short pick; the site tree, not the page, lists every page."""
    tree = set(dp.toctree_entries((dp.DOCS / "index.md").read_text(encoding="utf-8")))
    page = (dp.DOCS / "get-started.md").read_text(encoding="utf-8")
    links = dp.doc_links(dp.section(page, "Next steps"))
    assert links
    assert sorted(links - tree) == []
    assert len(links) <= 8


# -- The user glossary --------------------------------------------------------

USER_TERMS = (
    "Group index",
    "Group-velocity dispersion",
    "Dispersion parameter",
    "Third-order dispersion",
    "S-parameters",
    "Power waves",
    "Lumped port",
    "Wave port",
    "Loss tangent",
    "Element order",
    "Adjoint solve",
    "Participation ratio",
    "Sheet participation",
    "Dressed mode",
    "Dispersive shift",
    "Anharmonicity",
    "Lamb shift",
    "Loss budget",
    "Dispersive material",
    "Permittivity law",
    "Dispersion term",
    "Static permittivity",
    "Energy permittivity",
    "Index interpolation",
    "Dispersive cross-section",
    "Unresolved band",
    "Modal overlap",
    "Floquet port",
    "Floquet surface",
    "Diffraction order",
    "Diffraction efficiency",
    "Rayleigh anomaly",
    "Jones matrix",
    "Absorptance",
    "Angle sweep",
    "Conical incidence",
    "Emissivity",
    "Hemispherical emissivity",
    "Cell response",
    "Aperture field",
    "Locally periodic approximation",
    "Slab cross-section",
    "End faces",
    "Thermal domain",
    "Requested power share",
    "Trapped mode",
    "Unrequested port mode",
    "Heat sink",
    "Insulated surface",
    "Temperature rise",
    "Source-free temperature",
    "Conductance matrix",
    "Thermo-optic phase",
    "Thermal crosstalk",
    "Optical window",
    "Route",
    "Station",
    "Pass",
    "Waveform",
    "Trace",
    "Settled trace",
    "Per-length quantity",
    "Dissipated power",
    "Electromagnetic heating",
    "Coherent drive",
    "Sheet heat",
    "Heat flow",
    "Heat out",
    "Tree-cotree gauge",
    "2D model",
    "Ez polarization",
    "Line port",
    "Perfectly matched layer",
    "Target reflection",
    "Reference frequency",
    "Interior box",
    "Physical region",
    "PML mode",
    "Clearance",
    "Port mode",
    "Bound mode",
    "Leaky mode",
    "Resonant mode",
    "Quasi-normal mode",
    "Excited port",
    "Electro-optic tensor",
    "Electro-optic shift",
    "Push-pull",
    "Voltage-length product",
    "Electro-optic overlap factor",
    "Electrode clearance",
    "Log display",
    "Progress row",
    "Coupling Q",
    "Outgoing branch",
    "Incident field",
    "Scattered field",
    "Total field",
    "Background stack",
    "Incidence layer",
    "Scatterer",
    "Source",
    "Excitation",
    "Gaussian beam",
    "Waist",
    "Focal intensity",
    "Incident intensity",
    "Cross sections",
    "Port wave",
    "Point dipole",
    "Emitted power",
    "Bulk power",
    "Purcell factor",
    "Beta factor",
    "Radiative efficiency",
    "Far-field amplitude",
    "Far-field layer",
    "Exterior medium",
    "Radiation intensity",
    "Directivity",
    "Ground plane",
    "Beam coupling",
    "Incident contact",
    "Truncation layer",
    "Multipole decomposition",
    "Multipole coefficients",
    "Multipole degree",
    "Multipole origin",
    "Incident amplitude",
    "Partial scattering cross section",
    "Exact multipole moments",
    "Induced current",
    "Current moments",
    "Long-wavelength dipole",
    "Toroidal dipole",
    "Anapole",
    "Kerker condition",
    "Library material",
    "Permittivity fit",
    "Extinction coefficient",
    "Fitted range",
    "Layer pattern",
    "Fabrication model",
    "Edge bias",
    "Minimum width",
    "Minimum spacing",
    "Rational model",
    "Passivity violation",
    "Harmonic grid",
    "Time response",
    "Impedance profile",
    "SAX component",
    "Circuit port",
    "Component band",
    "Common reference impedance",
    "Internal lumped load",
    "Polarity reference",
    "Stiffness",
    "Piezoelectric tensor",
    "Clamped permittivity",
    "Mechanical quality factor",
    "Mechanical domain",
    "Interdigital transducer",
    "Meshed electrode",
    "Line electrode",
    "Clamped surface",
    "Damping frame",
    "Loss factor (of a damping frame)",
    "Admittance per length",
    "Clamped capacitance",
    "Aperture",
    "Series resistance",
    "Engineering strain",
    "Radiation conductance",
    "Synchronous frequency",
    "Node map",
    "Reciprocal lattice",
    "High-symmetry point",
    "Wave vector path",
    "Band",
    "Static band",
    "Band connection",
    "Band diagram",
    "Band gap",
    "Band group velocity",
    "Anchored band window",
    "Lost band",
)


@pytest.mark.parametrize("term", USER_TERMS)
def test_a_user_term_is_defined_and_linked_from_a_page(term):
    """A glossary entry no page links is one no reader finds."""
    glossary = (dp.DOCS / "glossary.md").read_text(encoding="utf-8")
    assert re.search(rf"^{re.escape(term)}\n  \S", glossary, re.MULTILINE), term
    targets = set()
    for page in PAGES:
        for role in re.findall(r"\{term\}`([^`]+)`", dp.page_text(page)):
            explicit = re.search(r"<([^>]+)>\s*$", role)
            # a role wrapped across lines names the term with single spaces
            target = explicit[1] if explicit else role
            targets.add(" ".join(target.split()).lower())
    assert term.lower() in targets, f"no user page links the term {term!r}"


# -- The package description and the aliases' API entries ---------------------


def test_the_package_description_is_the_readme_first_sentence():
    readme = (dp.ROOT / "README.md").read_text(encoding="utf-8")
    prose = next(
        line
        for line in readme.splitlines()
        if line.strip() and not line.lstrip().startswith(("<", "#", "!", "["))
    )
    sentence = prose.split(". ")[0].strip()
    with (dp.ROOT / "pyproject.toml").open("rb") as file:
        description = tomllib.load(file)["project"]["description"]
    assert description == sentence


@pytest.mark.parametrize("alias", ["Boundary", "Port"])
def test_a_type_alias_has_a_documentation_comment(alias):
    """The API reference shows an alias's text only from a ``#:`` comment."""
    source = (dp.ROOT / "src" / "gfem" / "_boundaries.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith(f"{alias} = "))
    assert lines[at - 1].startswith("#: "), f"no '#:' comment above {alias}"


def test_the_permittivity_law_names_sit_beside_material_in_the_api():
    """The API reference documents a name on the first public module listing it."""
    import gfem.model

    laws = {"PermittivityLaw", "Drude", "Lorentz", "Debye", "DjordjevicSarkar"}
    assert laws <= set(gfem.model.__all__)
