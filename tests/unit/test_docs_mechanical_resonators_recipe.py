"""The mechanical resonators recipe runs and prints what the page shows.

docs/recipes/mechanical-resonators.md runs nothing at docs build time. This runs
its blocks in order on the cantilever built by the repository's own
docs/recipes/resonator_meshes.py (the page downloads it), checks that the output
blocks are what the code prints, that the uniform quality factor is the
material's, that the first frequency is within 1 % of Euler-Bernoulli's and the
shape's ratio within 1 % of the beam theory's, and that a 3D body has no group
velocity.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "mechanical-resonators.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("mechanical_resonators")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _printed(ran, section: str) -> str:
    return "".join(e.printed for e in ran if e.section == section)


def test_every_output_block_is_what_the_code_prints(ran):
    # the page prints four significant digits: 1e-3 relative covers the rounding
    # (5e-5 of the smallest printed value) and the last digit's platform noise
    assert not dp.output_problems(ran, tolerance=1e-3), "\n".join(
        dp.output_problems(ran, tolerance=1e-3)
    )


def test_a_uniform_quality_factor_is_the_materials_by_the_identity(ran):
    # Q = Re w / (2 Im w) of w0 sqrt(1 + i/Qm) is Qm (1 + 1/(4 Qm^2)): 2.5e-7 above
    # Qm = 1000; the gate is 4x that, the page prints Q to 0.1
    quality = dp.numbers(_printed(ran, "A 3D body").split("Q = ")[-1])[0]
    assert quality == pytest.approx(1000.0, rel=1e-6)


def test_the_first_frequency_is_the_euler_bernoulli_one_within_one_percent(ran):
    # measured +0.38 % on the 1 um mesh (shear, rotary inertia and the clamp of a
    # beam 50 times longer than thick); the gate is 1 %
    deviation = dp.numbers(_printed(ran, "A 3D body").split("f_EB - 1 = ")[1])[0]
    assert abs(deviation) < 1.0  # percent


def test_the_deflection_shape_follows_the_beam_theory(ran):
    # the page prints 0.339 against 0.340; the mesh is coarse (two elements
    # across the thickness), the gate 1 %
    ratio, theory = dp.numbers(_printed(ran, "The tip and the root"))[:2]
    assert ratio == pytest.approx(theory, rel=1e-2)


def test_a_3d_body_has_wavenumber_zero_and_no_group_velocity(ran):
    # exact by construction: a body has no axial direction
    wavenumber, *velocities = (
        _printed(ran, "The tip and the root")
        .splitlines()[-1]
        .replace("[", " ")
        .replace("]", " ")
        .split()
    )
    assert wavenumber == "0.0"
    assert velocities == ["nan", "nan"]
