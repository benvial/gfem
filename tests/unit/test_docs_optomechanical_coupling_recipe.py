"""The optomechanical coupling recipe runs and prints what the page shows.

docs/recipes/optomechanical-coupling.md runs nothing at docs build time. This
runs its blocks in order on the suspended silicon wire at the page's mesh (built
by the repository's own docs/recipes/wire_section.py, which the page downloads),
and checks the printed numbers against what they must be: the mode that moves
along the guide does not couple, the total gain is the amplitude sum of the
parts, the overlap helper times the zero-point factor is the coupling, the
gradient is the scaling law, and the limits it lists raise what it says.

The page's blocks take about 25 s on one thread, mostly the optical solves and
the backward pass of the gradient, so the test belongs to the unit tier.
"""

import docs_pages as dp
import numpy as np
import pytest

PAGE = dp.DOCS / "recipes" / "optomechanical-coupling.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("optomechanical_coupling")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # the section builder it downloads
        return dp.run_page(PAGE, workdir)


def _example(ran, section: str, which: int = 0) -> dp.Example:
    return [example for example in ran if example.section == section][which]


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_modes_are_the_published_wires_and_one_moves_along_the_guide(ran):
    f0, f1, f2 = (
        dp.numbers(line)[1]
        for line in _example(ran, "The wire and its vibrations", 1).printed.splitlines()
    )
    # published E2 of the 315 x 283.5 nm wire, 12.561 GHz on this mesh (the
    # frequency is mesh independent to 4 digits); the next two are the shear
    # mode and the in-plane mode that is odd under a mirror
    assert f0 == pytest.approx(12.561, abs=2e-3)
    assert f1 == pytest.approx(12.670, abs=2e-3)
    assert f2 == pytest.approx(13.140, abs=2e-3)
    z_share = [
        dp.numbers(line)[-1]
        for line in _example(ran, "The wire and its vibrations", 1).printed.splitlines()
    ]
    assert z_share == [0.0, 1.0, 0.0]


def test_the_even_mode_couples_and_the_others_do_not(ran):
    lines = _example(ran, "The coupling of every mode").printed.splitlines()
    n_eff, n_g = dp.numbers(lines[0])
    assert n_eff == pytest.approx(1.8392, abs=2e-3)
    assert n_g == pytest.approx(5.665, abs=1e-2)
    g = [dp.numbers(line)[1] for line in lines[1:]]
    # the z-polarized mode is at round-off of the even mode's coupling, the
    # odd in-plane one at the mesh's asymmetry (1.4 % here, gate 5 %)
    assert g[0] == pytest.approx(1.09e4, rel=2e-2)
    assert g[1] < 1e-6 * g[0]
    assert g[2] < 5e-2 * g[0]


def test_the_two_parts_add_and_g0_is_the_coupling_over_the_root_length(ran):
    first, second = _example(ran, "The coupling of every mode", 1).printed.splitlines()
    mb, pe, total = dp.numbers(first)[:3]
    assert mb * pe > 0  # same sign: they add
    assert total == pytest.approx(mb + pe, rel=2e-3)  # three printed digits each
    _, g0_khz, length_um = dp.numbers(second)
    assert length_um == pytest.approx(62.8, abs=0.05)
    assert g0_khz * 1e3 * 2 * np.pi * np.sqrt(length_um * 1e-6) == pytest.approx(
        abs(total), rel=2e-3
    )


def test_the_gain_is_the_published_one_to_the_coarse_meshs_error(ran):
    mb, pe, total = dp.numbers(
        _example(ran, "The Brillouin gain of the guide").printed
    )[:3]
    # published 4400, 4200, 17200 1/(W m); measured -1.6, -3.0, -2.3 % on this mesh;
    # the gate is twice the largest, as the verification test's
    assert mb == pytest.approx(4400, rel=6e-2)
    assert pe == pytest.approx(4200, rel=6e-2)
    assert total == pytest.approx(17200, rel=6e-2)
    # the parts add in amplitude
    assert total == pytest.approx((np.sqrt(mb) + np.sqrt(pe)) ** 2, rel=2e-3)


def test_the_overlap_helper_times_the_zero_point_factor_is_the_coupling(ran):
    lines = _example(ran, "The overlap of any displacement").printed.splitlines()
    (ratio,) = dp.numbers(lines[1])
    assert ratio == pytest.approx(1.0, abs=1e-5)


def test_the_gradient_is_the_stiffness_scaling_law(ran):
    # g0 ~ E^(-1/4) exactly: the mode shape does not change, Omega ~ E^(1/2) and
    # the 1 J/m displacement ~ E^(-1/2). The gradient is exact in the discrete
    # problem; the gate is the printed five digits.
    (value,) = dp.numbers(_example(ran, "The gradient of the coupling").printed)
    assert value == pytest.approx(-0.25, abs=2e-5)


def test_the_limits_raise_what_the_page_says(ran):
    printed = _example(ran, "Limits").printed
    assert printed.splitlines() == [
        "length: ValueError",
        "mode: IndexError",
        "2D model: NotImplementedError",
    ]
