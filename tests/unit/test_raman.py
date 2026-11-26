"""`gfem.raman_gain`: polarization weights, symmetries and refusals."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
import raman_cases as rc
import scipy.constants as sc
from slab2d import write_slab

import gfem
from gfem import raman

N = 1.5
FP = rc.F_S + 13.2e12
ORDER = gfem.SolverOptions(order=2)


def _unit(*v):
    v = np.asarray(v, float)
    return jnp.asarray(v / np.linalg.norm(v), dtype=complex)


@pytest.mark.parametrize(
    ("a", "b", "weight"),
    [
        ((1, 0, 0), (1, 0, 0), 0.0),
        ((1, 1, 0), (1, 1, 0), 1.0),
        ((1, 1, 1), (1, 1, 1), 4 / 3),
        ((1, 1, 0), (1, -1, 0), 0.0),
        ((0, 1, 0), (0, 0, 1), 1.0),
    ],
)
def test_the_cubic_weight_of_unit_fields(a, b, weight):
    # sums of a few products of numbers of order 1: round-off 1e-16, gate 1e-14
    assert float(raman._cubic_weight(_unit(*a), _unit(*b))) == pytest.approx(
        weight, abs=1e-14
    )


def test_the_isotropic_weights_of_unit_fields():
    # co-polarized: both 1; crossed: the overlaps vanish, |a|^2|b|^2 stays
    same = raman._isotropic_weights(_unit(0, 1, 0), _unit(0, 1, 0))
    cross = raman._isotropic_weights(_unit(0, 1, 0), _unit(1, 0, 0))
    assert [float(x) for x in same] == pytest.approx([1.0, 2.0], abs=1e-14)
    assert [float(x) for x in cross] == pytest.approx([0.0, 1.0], abs=1e-14)


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    spectrum = rc.flat_spectrum(6e-14, symmetry="isotropic", perpendicular=3.6e-15)
    model = rc.guide_model(
        tmp_path_factory.mktemp("g") / "g.msh",
        2.0,
        1.2,
        gfem.Material(eps_r=N**2, raman_spectrum=spectrum),
        nx=4,
        ny=3,
    )
    modes = gfem.solve(model, gfem.PortMode("p1", [FP, rc.F_S]), options=ORDER)
    return model, modes


def test_the_gain_is_antisymmetric_by_manley_rowe_and_zero_on_the_diagonal(guide):
    model, modes = guide
    result = gfem.raman_gain(model, modes)
    g = np.asarray(result.gain)
    assert result.port == "p1"
    assert [m for m, _ in result.waves] == [1, 1]
    assert g[1, 0] > 0 > g[0, 1]
    # the Stokes wave is row 1: g_ps = -(f_p / f_s) g_sp; the same products
    # summed in the same order, so round-off only (1e-15), gate 1e-13
    assert g[0, 1] == pytest.approx(-FP / rc.F_S * g[1, 0], rel=1e-13)
    assert g[0, 0] == g[1, 1] == 0.0


def test_equal_frequencies_give_zero(guide):
    model, _ = guide
    one = gfem.solve(model, gfem.PortMode("p1", FP), options=ORDER)
    g = np.asarray(gfem.raman_gain(model, [one, one]).gain)
    # the shift is exactly 0, where the odd spectrum is exactly 0
    assert (g == 0.0).all()


def test_the_result_is_linear_in_the_spectrum(guide, tmp_path):
    model, modes = guide
    doubled = rc.guide_model(
        tmp_path / "g.msh",
        2.0,
        1.2,
        gfem.Material(
            eps_r=N**2,
            raman_spectrum=rc.flat_spectrum(
                1.2e-13, symmetry="isotropic", perpendicular=7.2e-15
            ),
        ),
        nx=4,
        ny=3,
    )
    a = np.asarray(gfem.raman_gain(model, modes).gain)
    b = np.asarray(gfem.raman_gain(doubled, modes).gain)
    # a power of two: the scaling is exact up to the sum's own round-off (1e-15)
    assert b == pytest.approx(2 * a, rel=1e-13)


def test_a_model_without_a_spectrum_names_the_port(tmp_path):
    model = rc.guide_model(
        tmp_path / "g.msh", 2.0, 1.2, gfem.Material(eps_r=N**2), nx=3, ny=2
    )
    modes = gfem.solve(model, gfem.PortMode("p1", FP), options=ORDER)
    with pytest.raises(ValueError, match=r"wave port 'p1'.*raman_spectrum"):
        gfem.raman_gain(model, modes)


def test_an_hz_line_port_is_refused(tmp_path):
    path = write_slab(
        tmp_path / "s.msh", thick=0.22, clear=1.0, h_core=0.055, h_clad=0.2
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "core": gfem.Material(
                eps_r=3.476**2,
                raman_spectrum=rc.flat_spectrum(2e-10, symmetry="cubic"),
            ),
            "clad": gfem.Material(eps_r=1.444**2),
        },
        ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
    )
    hz = gfem.solve(
        model, gfem.PortMode("p1", sc.c / 1.55e-6, polarization="Hz"), options=ORDER
    )
    with pytest.raises(NotImplementedError, match="Hz line port"):
        gfem.raman_gain(model, hz)


def test_non_results_are_type_errors(guide):
    model, modes = guide
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.raman_gain(model, "mode")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.raman_gain(modes, modes)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"gfem\.PortMode"):
        gfem.raman_gain(model, dataclasses.replace(modes, e_field=None))


def test_the_repr_names_the_port_and_waves(guide):
    model, modes = guide
    assert "RamanGain(port 'p1', 2 waves" in repr(gfem.raman_gain(model, modes))
