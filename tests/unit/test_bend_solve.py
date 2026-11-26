"""The bent port mode solve targets the closed straight guide's mode.

A bent mode is the bent continuation of the closed straight section's mode: the
candidate of the bent pencil that overlaps it most (one step), or, when that
choice is unclear, the end of a walk in $1/R$ from the straight guide. These
tests pin the picks on the leaky slab (exact literals, `bend_cases`) and the
whispering-gallery slab, the several-mode, several-frequency and straight
paths. Sections are small (3000 DOFs): the file runs in a few seconds.
"""

from __future__ import annotations

import bend_cases as bc
import numpy as np
import pytest

from gfem import _pml
from gfem._engine.port_mode import PortModeSpace
from gfem._port_section import port_section
from gfem._sweep import port_mode as sw


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    return tmp_path_factory.mktemp("bend_solve")


@pytest.fixture
def walks(monkeypatch):
    """The number of continuation walks a solve took, in a list."""
    count: list[int] = []
    original = sw._Greedy._continued

    def counted(self, *args, **kwargs):
        count.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sw._Greedy, "_continued", counted)
    return count


def _model(meshes, pol, w=None):
    name = f"slab_{pol}_{w}.msh"
    return bc.slab_model(
        bc.slab_mesh(meshes / name, pol, **({} if w is None else {"w": w}))
    )


# --- one step ----------------------------------------------------------------------


# One step lands every guided case. Overlaps of the picked and the runner-up
# candidate with the closed straight mode, measured at p = 3 on h = 0.1 um:
# (R, Ey) 5: 0.974/0.050, 3: 0.893/0.074; (R, Hy) 5: 0.949/0.024,
# 3: 0.808/0.035; Hy at R = 1.6 um 0.532/0.062 (the weakest of the set).
# The picks match the exact roots to the p = 3 discretization error on this
# mesh: measured |nu/ref - 1| = 3.7e-9 and 4.9e-9 (Ey R = 5, 3), 1.2e-9 and 4.8e-9
# (Hy), so the 1e-8 gate (as in test_bend) is 2x the largest.
@pytest.mark.parametrize("pol", ["Ey", "Hy"])
@pytest.mark.parametrize("radius", [3, 5])
def test_a_slab_bend_is_picked_in_one_step(meshes, walks, pol, radius):
    model = _model(meshes, pol)
    result = bc.solve_mode(model, bc.SLAB_F, bc.slab_bend(radius))
    ref = bc.SLAB_NU[(radius, pol)]
    assert abs(bc.nu_of(result, radius * 1e-6) / ref - 1) <= 1e-8
    assert walks == []


# No exact root at R = 1.6 um: the p = 4 value of the same mesh,
# 12.175863682554414 - 0.4167549683155181i, differs from the p = 3 pick by
# 3.4e-8 relative (measured), so the gate 7e-8 is 2x that.
def test_a_tight_slab_bend_is_still_one_step(meshes, walks):
    model = _model(meshes, "Hy")
    result = bc.solve_mode(model, bc.SLAB_F, bc.slab_bend(1.6))
    ref = 12.175863682554414 - 0.4167549683155181j
    assert abs(bc.nu_of(result, 1.6e-6) - ref) <= 7e-8 * abs(ref)
    assert walks == []


# --- the whispering-gallery mode ---------------------------------------------------


# w = 2 um, R = 2 um: n_eff = 2.549 exceeds the core index. The best overlap
# of one step is 0.433 against 0.39 for another candidate, so the rule fails
# and the walk in 1/R takes over: two steps still fail the margin (0.59 against
# 0.547 at the first), four pass (overlaps 0.74, 0.97, 0.98, 0.98). The p = 3 pick
# differs from the p = 4 reference by 3.2e-9 relative (measured); the gate is 2x.
def test_a_whispering_gallery_mode_is_found_by_the_walk(meshes, walks):
    model = _model(meshes, "Ey", w=bc.WGM_W)
    result = bc.solve_mode(model, bc.SLAB_F, bc.slab_bend(2.0))
    ref = 20.6685941754655 - 0.06335274364804511j
    assert abs(bc.nu_of(result, 2e-6) - ref) <= 6.4e-9 * abs(ref)
    assert walks == [1]


# --- several modes and frequencies -------------------------------------------------


def _sweeps(model, frequencies, modes, radius_um):
    """The bent port mode sweeps of one request, modes in one sweep."""
    bend = bc.slab_bend(radius_um)
    stretch = _pml.stretch(model, tuple(frequencies))
    section = port_section(model, model.ports[0], stretch, bend)
    space = PortModeSpace(section, 3)
    ports = [model.ports[0]] * len(modes)
    return sw.port_mode_sweep(
        space, frequencies, list(modes), tolerance=1e-8, ports=ports
    )


# The picks are converged eigenvalues of one pencil found from two shifts.
# ARPACK's tolerance $10^{-10}$ is on $\theta = 1/(\lambda-\sigma)$, so each
# $\lambda = \gamma^2$ of a well-conditioned mode is within
# $10^{-10}|\lambda-\sigma|$ of the pencil's. The shifts here sit at most at
# $|\lambda-\sigma|/|\lambda| = 0.75$ (the second mode
# of the 0.6 um slab, from the first one's shift), so the two $\gamma$ agree
# to $2 \times 0.75\times10^{-10}/2 = 7.5\times10^{-11}$ at worst; measured
# agreement is 1e-14.
RTOL = 1e-10


@pytest.mark.parametrize(
    ("w", "modes"),
    [(1.0, (1, 2)), (0.6, (1, 2))],
    ids=["both by one step", "second by the walk"],
)
def test_two_bent_modes_equal_their_single_mode_solves(meshes, w, modes):
    model = _model(meshes, "Ey", w=w)
    together = _sweeps(model, [bc.SLAB_F], modes, 5.0)
    for k, mode in enumerate(modes):
        (alone,) = _sweeps(model, [bc.SLAB_F], [mode], 5.0)
        np.testing.assert_allclose(together[k].gamma, alone.gamma, rtol=RTOL)


def test_a_bent_sweep_row_equals_its_single_frequency_solve(meshes):
    model = _model(meshes, "Ey")
    frequencies = [bc.SLAB_F, 1.03 * bc.SLAB_F]
    (sweep,) = _sweeps(model, frequencies, [1], 5.0)
    for i, f in enumerate(frequencies):
        (alone,) = _sweeps(model, [f], [1], 5.0)
        np.testing.assert_allclose(sweep.gamma[i], alone.gamma[0], rtol=RTOL)


# --- a straight section ------------------------------------------------------------


# gamma of the straight slab (Ey, p = 3, h = 0.1 um) before the bent solve was
# targeted. The straight path runs the same operations, so a platform's
# BLAS alone moves it: ARPACK's relative tolerance is 1e-10 on the shifted
# eigenvalue, the gate rtol 1e-9 is 10x that. The loss part (5e-8 of 7e6) is
# round-off noise of a lossless mode and is not compared.
def test_a_straight_section_solves_as_before(meshes):
    model = _model(meshes, "Ey")
    result = bc.solve_mode(model, bc.SLAB_F)
    gamma = complex(np.asarray(result.gamma)[0])
    assert abs(gamma.imag / 7445987.952480065 - 1) <= 1e-9
