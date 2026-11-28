"""The mechanical resonators recipe runs and prints what the page shows.

docs/recipes/mechanical-resonators.md runs nothing at docs build time. This runs
its blocks in order on the cantilever built by the repository's own
docs/recipes/resonator_meshes.py (the page downloads it), checks that the output
blocks are what the code prints, that the uniform quality factor is the
material's, that the first frequency is within 1 % of Euler-Bernoulli's and the
shape's ratio within 1 % of the beam theory's, that a 3D body has no group
velocity, and that the anchored beam's Q is within the gate of the refined value
and differs from the damping frame's estimate.
"""

import docs_pages as dp
import numpy as np
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


#: The anchor Q of the page's mesh (L/h = 10, order 3, ``maxh`` 0.5 um, 12 090
#: free DOFs) and the refined value (``maxh`` 0.0625 um, 469 k free DOFs), both
#: measured on MN5 with the same idiom; the steps with ``maxh`` are 7.3, 3.7 and
#: 1.7, so the page's mesh is 12.7 below the refined value. The gate is twice
#: that, as the solver's unit test.
ANCHOR_PAGE_Q = 1647.0392
ANCHOR_REFINED_Q = 1659.7670


def test_the_anchor_q_is_within_the_gate_of_the_refined_value(ran):
    # the page prints Q to the unit, rounded: 0.5 off at most, on a gate of 25
    text = _printed(ran, "Anchor loss with a layer")
    quality = dp.numbers(text.split("PML: ")[1].split("Q = ")[1])[0]
    assert abs(quality - ANCHOR_REFINED_Q) < 2 * abs(ANCHOR_PAGE_Q - ANCHOR_REFINED_Q)
    # the converged constant against the rigid root's 1.5833: 4.8 % above it at
    # L/h = 10 (the page's mesh 4.0 %); a root that radiated nothing more would
    # sit on it, so a constant below it or 10 % above it is a wrong mode
    constant = dp.numbers(text.split("Q / (L/h)^3 = ")[1])[0]
    assert 1.583 < constant < 1.583 * 1.10


def test_the_frame_estimate_and_the_layer_differ_and_the_layer_q_is_finite(ran):
    # the frame's Q is an estimate (1577 against 1647: 4 %); equal prints would
    # mean the layer run returned the frame's mode
    text = _printed(ran, "Anchor loss with a layer")
    frame = dp.numbers(text.split("frame: ")[1].split("Q = ")[1])[0]
    layer = dp.numbers(text.split("PML: ")[1].split("Q = ")[1])[0]
    assert np.isfinite(layer)
    assert layer > 0
    assert frame != layer
