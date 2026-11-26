"""`RamanSpectrum` and `Material.raman_spectrum`: shapes, scaling, refusals."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._solve_cache import solve_key

C = 299792458.0
F_795 = C / 795.5e-9
F_1550 = C / 1550e-9
LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=2e15, damping=0.0)],
)


def _silicon_line(peak=2e-10, symmetry="cubic"):
    return gfem.RamanSpectrum.lorentzian(
        peak, 15.6e12, 105e9, F_1550, symmetry=symmetry
    )


def test_lorentzian_peak_and_width_are_exact_grid_nodes():
    s = _silicon_line()
    f0, w = 15.6e12, 105e9
    # the peak and both half-maximum points are nodes and the Lorentzian is
    # evaluated there in closed form; the only error is the relative round-off
    # of a few operations, below 1e-14, so 1e-12 holds on every platform
    k = int(np.argmax(s.gain))
    assert abs(s.shift[k] - f0) <= 1e-12 * f0
    assert abs(s.gain[k] - 2e-10) <= 1e-12 * 2e-10
    g = lambda f: float(s.gain_at(f, F_1550))  # noqa: E731
    assert g(f0 - w / 2) == pytest.approx(1e-10, rel=1e-12)
    assert g(f0 + w / 2) == pytest.approx(1e-10, rel=1e-12)
    # the half-maximum crossings of the interpolated curve are f0 -+ w/2
    fine = np.linspace(f0 - w, f0 + w, 400001)
    above = fine[np.asarray(s.gain_at(fine, F_1550)) >= 1e-10]
    # linear interpolation between two nodes is exact at the nodes, and the
    # crossing lies between grid points of step 5e-6 w: 1e-5 w bounds it
    assert above[-1] - above[0] == pytest.approx(w, abs=1e-5 * w)


def test_lorentzian_tail_is_the_lorentzian_to_the_interpolation_error():
    s = _silicon_line()
    f = 15.6e12 + 2.2e9
    exact = 2e-10 * (52.5e9) ** 2 / (2.2e9**2 + 52.5e9**2)
    # linear interpolation between nodes a step h = w/16 apart errs by at most
    # h^2/8 |g''|, and near the centre |g''|/g = 2/a^2 with a = w/2: a relative
    # (h/a)^2/4 = (1/8)^2/4 = 3.9e-3 (measured: 3.4e-3); 5e-3 is that bound
    assert float(s.gain_at(f, F_1550)) == pytest.approx(exact, rel=5e-3)


@pytest.mark.parametrize(
    ("frequency", "expected"), [(F_795, 1.204e-13), (F_1550, 6.18e-14)]
)
def test_lin_agrawal_peak_gain_at_the_quoted_wavelengths(frequency, expected):
    # the shape's maximum is the user's peak at the reference frequency, and
    # scales with the Stokes frequency; the user gives the 795.5 nm model peak
    s = gfem.RamanSpectrum.lin_agrawal(1.204e-13, F_795)
    top = float(jnp.max(s.gain_at(s.shift, frequency)))
    # the quoted digits: 4 for 1.204e-13 (half a unit in the last place,
    # 4.2e-4 relative) and 3 for 6.18e-14 (8.1e-4)
    assert top == pytest.approx(expected, rel=9e-4)
    assert s.shift[np.argmax(s.gain)] == pytest.approx(13.04e12, abs=1e10)


def test_lin_agrawal_shape_follows_the_response_model():
    # the model's own peak: g = 2 k0 n2 f_R Im h, n2 = 2.6e-20, f_R = 0.245
    k0 = 2 * np.pi * F_795 / C
    omega = 2 * np.pi * np.linspace(0.01e12, 40e12, 4000)
    a, b, ab = 1 / 32e-15, 1 / 12.2e-15, 1 / 96e-15
    h_a = (a * a + b * b) / ((a - 1j * omega) ** 2 + b * b)
    h_b = (2 * ab) / (ab - 1j * omega) - ab**2 / (ab - 1j * omega) ** 2
    peak = (2 * k0 * 2.6e-20 * 0.245 * (0.79 * h_a + 0.21 * h_b).imag).max()
    # 4000 samples at 10 GHz of a peak 5 THz wide: the maximum is found to
    # (1e10/5e12)^2 ~ 4e-6 relative; the quoted figure has 4 digits
    assert peak == pytest.approx(1.204e-13, rel=5e-4)
    s = gfem.RamanSpectrum.lin_agrawal(peak, F_795)
    assert s.symmetry == "isotropic"
    # the cross-polarized gain at the parallel peak is 4.58 % of it
    k = int(np.argmax(s.gain))
    assert s.perpendicular_gain[k] / s.gain[k] == pytest.approx(0.0458, abs=5e-5)


def test_gain_at_scales_with_frequency_is_odd_and_zero_beyond_the_grid():
    s = gfem.RamanSpectrum(
        [0.0, 1e12, 2e12], [0.0, 1.0, 0.0], 2e14, "isotropic", [0.0, 0.5, 0.0]
    )
    # a triangle: interpolation at 0.5e12 is the mean of its ends, exactly
    assert float(s.gain_at(0.5e12, 2e14)) == pytest.approx(0.5, rel=1e-14)
    assert float(s.gain_at(1e12, 1e14)) == pytest.approx(0.5, rel=1e-14)
    assert float(s.gain_at(-1e12, 2e14)) == pytest.approx(-1.0, rel=1e-14)
    assert float(s.gain_at(3e12, 2e14)) == 0.0
    assert float(s.perpendicular_gain_at(1e12, 4e14)) == pytest.approx(1.0, rel=1e-14)
    assert float(_silicon_line().perpendicular_gain_at(15.6e12, F_1550)) == 0.0


def test_gain_may_be_traced_and_differentiates():
    def peak(scale):
        s = gfem.RamanSpectrum(
            [0.0, 1e12, 2e12], scale * jnp.array([0.0, 1.0, 0.0]), 2e14, "cubic"
        )
        return s.gain_at(1.5e12, 2e14)

    assert float(jax.grad(peak)(3.0)) == pytest.approx(0.5, rel=1e-14)
    assert float(jax.jit(peak)(3.0)) == pytest.approx(1.5, rel=1e-14)


def test_spectrum_is_a_pytree_and_hashable_by_content():
    a, b = _silicon_line(), _silicon_line()
    assert a == b
    assert hash(a) == hash(b)
    assert a != _silicon_line(3e-10)
    leaves, tree = jax.tree.flatten(a)
    assert len(leaves) == 1
    assert jax.tree.unflatten(tree, leaves) == a


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shift": [0.0, 1.0], "gain": [0.0, -1.0]},
        {"shift": [0.0, 1.0], "gain": [0.0, np.nan]},
        {"shift": [0.0, 1.0], "gain": [0.0, np.inf]},
        {"shift": [1.0, 2.0], "gain": [0.0, 1.0]},
        {"shift": [0.0, 1.0], "gain": [1.0, 1.0]},
        {"shift": [0.0, 1.0, 1.0], "gain": [0.0, 1.0, 1.0]},
        {"shift": [0.0, 1.0], "gain": [0.0, 1.0, 0.0]},
        {"shift": [0.0, 1.0], "gain": [0.0, 1.0], "frequency": 0.0},
        {"shift": [0.0, 1.0], "gain": [0.0, 1.0], "symmetry": "hexagonal"},
        {"shift": [0.0, 1.0], "gain": [0.0, 1.0], "perpendicular_gain": [0.0, 1.0]},
        {
            "shift": [0.0, 1.0],
            "gain": [0.0, 1.0],
            "symmetry": "isotropic",
            "perpendicular_gain": [0.0, -1.0],
        },
    ],
)
def test_spectrum_refusals(kwargs):
    args = {"frequency": 2e14, "symmetry": "cubic", **kwargs}
    with pytest.raises(ValueError):
        gfem.RamanSpectrum(**args)


def test_symmetry_is_required():
    with pytest.raises(TypeError):
        gfem.RamanSpectrum([0.0, 1.0], [0.0, 1.0], 2e14)  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        gfem.RamanSpectrum.lorentzian(1.0, 15e12, 1e11, 2e14)  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    "args",
    [(-1.0, 15e12, 1e11), (1.0, 15e12, 0.0), (1.0, 15e12, 40e12), (1.0, 0.0, 1e11)],
)
def test_lorentzian_refusals(args):
    with pytest.raises(ValueError):
        gfem.RamanSpectrum.lorentzian(*args, 2e14, symmetry="cubic")
    with pytest.raises(ValueError):
        gfem.RamanSpectrum.lin_agrawal(-1.0, 2e14)


def test_material_holds_the_spectrum_and_stays_a_pytree():
    s = _silicon_line()
    m = gfem.Material(
        eps_r=12.1, axes=((1, 0, 0), (0, 1, 0), (0, 0, 1)), raman_spectrum=s
    )
    assert m.raman_spectrum is s
    assert "raman_spectrum" in repr(m)
    assert "raman_spectrum" not in repr(gfem.Material(eps_r=12.1))
    leaves, tree = jax.tree.flatten(m)
    assert jax.tree.unflatten(tree, leaves) == m
    assert gfem.Material(eps_r=12.1).raman_spectrum is None
    # an isotropic spectrum beside a law needs no frame
    iso = gfem.RamanSpectrum.lin_agrawal(6e-14, F_1550)
    assert gfem.Material(eps_r=LAW, raman_spectrum=iso).raman_spectrum is iso


def test_material_refusals():
    cubic, iso = _silicon_line(), gfem.RamanSpectrum.lin_agrawal(6e-14, F_1550)
    for eps in ((11.0, 11.0, 12.0), np.diag([11.0, 11.0, 12.0])):
        for spectrum in (cubic, iso):
            with pytest.raises(ValueError, match="isotropic eps_r"):
                gfem.Material(eps_r=eps, raman_spectrum=spectrum)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="cubic raman_spectrum"):
        gfem.Material(eps_r=LAW, raman_spectrum=cubic)
    with pytest.raises(TypeError, match="RamanSpectrum"):
        gfem.Material(eps_r=2.0, raman_spectrum="silicon")  # ty: ignore[invalid-argument-type]

    def traced(scale):
        s = gfem.RamanSpectrum([0.0, 1.0], scale * jnp.array([0.0, 1.0]), 2e14, "cubic")
        return gfem.Material(eps_r=12.1, raman_spectrum=s).raman_spectrum.gain[1]

    with pytest.raises(gfem.NonDifferentiableError):
        jax.grad(traced)(1.0)


def test_interpolate_refuses_an_end_with_a_spectrum():
    with_raman = gfem.Material(eps_r=12.1, raman_spectrum=_silicon_line())
    plain = gfem.Material(eps_r=1.0)
    for void, solid in ((plain, with_raman), (with_raman, plain)):
        with pytest.raises(ValueError, match="raman_spectrum"):
            gfem.MaterialField.interpolate(jnp.array([0.2, 0.7]), void, solid)


def test_spectrum_round_trips_through_the_job_codec_and_the_key_is_unchanged(tmp_path):
    mesh = gfem.Mesh.from_gmsh(
        sm.write_msh22(
            tmp_path / "stack.msh",
            sm.plate(4.0, 3.0, (("oxide", 0.5), ("nitride", 0.25)), (1, 1, 1)),
        )
    )
    problem = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
    silicon = gfem.Material(
        eps_r=12.1,
        axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        raman_spectrum=_silicon_line(),
    )
    silica = gfem.Material(
        eps_r=2.1, raman_spectrum=gfem.RamanSpectrum.lin_agrawal(6e-14, F_1550)
    )
    model = gfem.Model(mesh, materials={"oxide": silica, "nitride": silicon})
    path = gfem.Job(model, problem).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path).model
    assert back.materials["nitride"] == silicon
    assert back.materials["oxide"] == silica
    # without a spectrum the file and the solve key carry no trace of it
    plain = gfem.Model(
        mesh, materials={"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material()}
    )
    doc = json.loads(gfem.Job(plain, problem).to_json(tmp_path / "p.json").read_text())
    assert "raman_spectrum" not in json.dumps(doc)
    explicit = gfem.Model(
        mesh,
        materials={
            "oxide": gfem.Material(eps_r=3.9, raman_spectrum=None),
            "nitride": gfem.Material(),
        },
    )
    options = gfem.SolverOptions(order=1)
    assert (
        solve_key(explicit, problem, options)[0]
        == solve_key(plain, problem, options)[0]
    )
