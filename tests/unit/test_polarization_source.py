"""The second-order polarization source: its values, its refusals and its solve.

The cell is the slab of `second_order_cases` at 10 cells/um along z (336
tetrahedra) solved at order 2, periodic, with a tensor $d_{xxx}$ alone, so the
polarization is $P_x = \\varepsilon_0 d A_x B_x$ (times 1 or 2 by the process) and
the other components vanish.

Tolerances. Values compared with the pump's `Field.at` evaluate the same
polynomial of the same coefficients by two routes (the rule-space gather and the
field's own point evaluation), so they agree to round-off in the sums of up to
30 terms of an element's degrees of freedom: measured $8\times10^{-15}$ of the
largest value, and the tests use $10^{-12}$ of it. Scalings of a pump's power and phase
and the swap of two pumps run the same operations on the same numbers and agree
to $10^{-12}$ of the largest value. Two solves of one direct factorization with
loads that differ by round-off (a mesh in um and in m, a rule of higher order
that integrates the same polynomial exactly) agree to $10^{-8}$ relative: the
condition number of the cell's matrix is below $10^{6}$, times the $10^{-16}$ of
double precision, with a margin of 100.
"""

import jax
import numpy as np
import pytest
import scipy.constants as const
from second_order_cases import (
    FREQUENCY,
    PERIODIC,
    SIDE,
    run_slab,
    slab_mesh,
    slab_model,
)

import gfem
from gfem._solve_cache import solve_key

EPS0 = const.epsilon_0
D = 20e-12
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
#: A second pump frequency (Hz) for sum and difference frequency.
F2 = 1.2 * FREQUENCY
#: Values from two routes, and exactly repeated operations, relative to the maximum.
REL_ROUTES, REL_SAME = 1e-12, 1e-12
#: Two solves of one operator with round-off-different loads, relative to the maximum.
REL_SOLVES = 1e-8
#: The emitted power against the sum of the port powers: the cell is lossless, and
#: the balance misses what the evanescent orders leak into the ports; measured
#: 7.6e-4 at order 2 on this cell, a margin of 4.
REL_BALANCE = 3e-3
PORTS = ("z_min TE", "z_min TM", "z_max TE", "z_max TM")


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return slab_mesh(tmp_path_factory.mktemp("cell") / "cell.msh", 10)


@pytest.fixture(scope="module")
def pump_model(mesh):
    return slab_model(mesh, 2.2, chi2=True)


@pytest.fixture(scope="module")
def pump(pump_model):
    """The pump solved at two frequencies, fields kept."""
    return gfem.solve(
        pump_model,
        gfem.Driven([FREQUENCY, F2], excite="z_min TM", keep_fields=True),
        options=OPTIONS,
    )


@pytest.fixture(scope="module")
def shg(pump_model, pump):
    return gfem.second_order_polarization(
        pump_model, gfem.Pump(pump, FREQUENCY, incident_power=1.0)
    )


def _pump(pump, f, power=1.0, phase=0.0):
    return gfem.Pump(pump, f, incident_power=power, incident_phase=phase)


def _px(source):
    return np.asarray(source.re)[:, 0] + 1j * np.asarray(source.im)[:, 0]


def _field_at(pump, f, source):
    """The pump's kept $E_x$ at the source's points (SI metres per um)."""
    points = np.asarray(source.points) * 1e-6
    return np.asarray(pump.e_field(f, "z_min TM").at(points))


def _generated(mesh, source, **kwargs):
    model = slab_model(mesh, 2.2, chi2=False)
    return gfem.solve(
        model,
        gfem.Driven(source.frequency, excite=(), sources=source, **kwargs),
        options=OPTIONS,
    )


def _close(a, b, rel):
    a, b = np.asarray(a), np.asarray(b)
    assert np.abs(a - b).max() <= rel * np.abs(b).max()


def test_the_source_is_a_two_leaf_pytree_and_a_driven_with_it_has_two(shg):
    assert len(jax.tree_util.tree_leaves(shg)) == 2
    plain = gfem.Driven(2 * FREQUENCY, excite="z_min TM")
    assert jax.tree_util.tree_leaves(plain) == []
    with_source = gfem.Driven(shg.frequency, excite=(), sources=shg)
    assert len(jax.tree_util.tree_leaves(with_source)) == 2


def test_second_harmonic_polarization_against_the_pump_field(shg, pump):
    e = _field_at(pump, FREQUENCY, shg)
    assert shg.process == "second-harmonic"
    assert shg.frequency == pytest.approx(2 * FREQUENCY, rel=1e-12)
    expected = EPS0 * D * e[:, 0] ** 2
    _close(_px(shg), expected, REL_ROUTES)
    # the tensor has d_xxx alone: no other component
    other = np.asarray(shg.re)[:, 1:] + 1j * np.asarray(shg.im)[:, 1:]
    assert np.abs(other).max() <= REL_SAME * np.abs(_px(shg)).max()


def test_the_points_lie_in_the_slab_only(shg):
    z = np.asarray(shg.points)[:, 2]
    assert z.min() >= 0.25
    assert z.max() <= 1.25
    assert shg.regions == ("slab",)


def _pml_model(mesh, carrying):
    """The cell with its pads as a perfectly matched layer, tensor in ``carrying``."""
    d = np.zeros((3, 6))
    d[0, 0] = D
    materials = {
        r: gfem.Material(
            eps_r=2.2**2, second_order_nonlinear_tensor=d if r in carrying else None
        )
        for r in ("slab", "pad")
    }
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=PERIODIC,
        pml=gfem.PML("pad"),
    )


def test_a_tensor_inside_the_layer_contributes_no_points(mesh, pump):
    # the pads carry the tensor too, yet the polarization lives in the slab alone
    model = _pml_model(mesh, ("slab", "pad"))
    src = gfem.second_order_polarization(model, _pump(pump, FREQUENCY))
    z = np.asarray(src.points)[:, 2]
    assert z.min() >= 0.25
    assert z.max() <= 1.25
    # the same points and values as the model whose pads carry no tensor: the
    # rule is built from the same elements by the same operations
    plain = gfem.second_order_polarization(
        _pml_model(mesh, ("slab",)), _pump(pump, FREQUENCY)
    )
    assert np.asarray(src.points).shape == np.asarray(plain.points).shape
    _close(_px(src), _px(plain), REL_SAME)


def test_a_tensor_only_inside_the_layer_is_refused(mesh, pump):
    model = _pml_model(mesh, ("pad",))
    with pytest.raises(ValueError, match="perfectly matched layer"):
        gfem.second_order_polarization(model, _pump(pump, FREQUENCY))


def test_sum_frequency_is_twice_the_product_of_the_two_pumps(pump_model, pump):
    a, b = _pump(pump, FREQUENCY), _pump(pump, F2)
    sfg = gfem.second_order_polarization(pump_model, (a, b), process="sum-frequency")
    assert sfg.frequency == pytest.approx(FREQUENCY + F2, rel=1e-12)
    e1, e2 = _field_at(pump, FREQUENCY, sfg), _field_at(pump, F2, sfg)
    _close(_px(sfg), 2 * EPS0 * D * e1[:, 0] * e2[:, 0], REL_ROUTES)
    swapped = gfem.second_order_polarization(
        pump_model, (b, a), process="sum-frequency"
    )
    _close(_px(swapped), _px(sfg), REL_SAME)


def test_difference_frequency_conjugates_the_second_pump(pump_model, pump):
    a, b = _pump(pump, F2), _pump(pump, FREQUENCY)
    dfg = gfem.second_order_polarization(
        pump_model, (a, b), process="difference-frequency"
    )
    assert dfg.frequency == pytest.approx(F2 - FREQUENCY, rel=1e-12)
    e1, e2 = _field_at(pump, F2, dfg), _field_at(pump, FREQUENCY, dfg)
    _close(_px(dfg), 2 * EPS0 * D * e1[:, 0] * np.conj(e2[:, 0]), REL_ROUTES)
    # the second pump's phase enters conjugated, the first's as it is
    turned = gfem.second_order_polarization(
        pump_model,
        (_pump(pump, F2, phase=0.3), _pump(pump, FREQUENCY, phase=0.8)),
        process="difference-frequency",
    )
    _close(_px(turned), _px(dfg) * np.exp(1j * (0.3 - 0.8)), REL_SAME)


def test_power_and_phase_scale_the_polarization(pump_model, pump, shg):
    # P is quadratic in the amplitude: power 4 doubles the amplitude, so P x4
    four = gfem.second_order_polarization(pump_model, _pump(pump, FREQUENCY, 4.0))
    _close(_px(four), 4 * _px(shg), REL_SAME)
    turned = gfem.second_order_polarization(
        pump_model, _pump(pump, FREQUENCY, 1.0, 0.7)
    )
    _close(_px(turned), _px(shg) * np.exp(1.4j), REL_SAME)


def test_the_load_does_not_depend_on_the_file_length_unit(tmp_path, shg):
    # the same cell in metres: the same P at the same points (in metres) and the
    # same generated waves
    mesh_m = slab_mesh(tmp_path / "cell_m.msh", 10, unit_um=1e-6)
    model = slab_model(mesh_m, 2.2, chi2=True)
    pump = gfem.solve(
        model,
        gfem.Driven(FREQUENCY, excite="z_min TM", keep_fields=True),
        options=OPTIONS,
    )
    source = gfem.second_order_polarization(model, _pump(pump, FREQUENCY))
    np.testing.assert_allclose(
        np.asarray(source.points) * 1e6, np.asarray(shg.points), atol=1e-9
    )
    _close(_px(source), _px(shg), REL_SOLVES)
    here = _generated(mesh_m, source)
    there = run_slab(slab_mesh(tmp_path / "cell_um.msh", 10), order=2).generated
    top_m = complex(here.port_wave("z_max TM", source.name)[0])
    top_um = complex(there.port_wave("z_max TM", "second harmonic")[0])
    assert abs(top_m - top_um) <= REL_SOLVES * abs(top_um)


def test_emitted_power_is_the_sum_of_the_port_powers(mesh, shg):
    result = _generated(mesh, shg)
    leaving = sum(abs(complex(result.port_wave(p, shg.name)[0])) ** 2 for p in PORTS)
    emitted = float(result.emitted_power[shg.name][0])
    assert emitted > 0
    assert emitted == pytest.approx(leaving, rel=REL_BALANCE)


def test_two_sources_solve_as_two_separate_solves(mesh, pump_model, pump, shg):
    double = gfem.second_order_polarization(
        pump_model, _pump(pump, FREQUENCY, 2.0), name="double"
    )
    both = gfem.solve(
        slab_model(mesh, 2.2, chi2=False),
        gfem.Driven(shg.frequency, excite=(), sources=[shg, double]),
        options=OPTIONS,
    )
    alone = _generated(mesh, double)
    first = _generated(mesh, shg)
    for port in PORTS:
        for name, ref in (("double", alone), (shg.name, first)):
            want = complex(ref.port_wave(port, name)[0])
            got = complex(both.port_wave(port, name)[0])
            scale = max(abs(complex(first.port_wave(q, shg.name)[0])) for q in PORTS)
            assert abs(got - want) <= REL_SOLVES * scale
    # the second source is 2x the amplitude squared: twice the power's field
    top = [complex(both.port_wave("z_max TM", n)[0]) for n in (shg.name, "double")]
    assert abs(top[1] - 2 * top[0]) <= REL_SOLVES * abs(top[0])


def test_a_source_of_a_higher_generated_order_gives_the_same_solve(
    mesh, pump_model, pump, shg
):
    # order 3 is a rule of degree 8, and the order-2 solve's integrand has degree 6:
    # both rules are exact for it
    higher = gfem.second_order_polarization(
        pump_model, _pump(pump, FREQUENCY), generated_order=3
    )
    assert higher.generated_order == 3
    assert higher.order > shg.order
    a, b = _generated(mesh, shg), _generated(mesh, higher)
    for port in PORTS:
        want = complex(a.port_wave(port, shg.name)[0])
        got = complex(b.port_wave(port, shg.name)[0])
        assert abs(got - want) <= REL_SOLVES * abs(
            complex(a.port_wave("z_max TM", shg.name)[0])
        )


def test_a_solve_of_a_higher_order_than_the_source_is_refused(mesh, pump_model, pump):
    low = gfem.second_order_polarization(
        pump_model, _pump(pump, FREQUENCY), generated_order=1
    )
    with pytest.raises(ValueError, match=r"generated_order=2"):
        _generated(mesh, low)


def test_the_source_survives_a_job_round_trip(tmp_path, mesh, shg):
    model = slab_model(mesh, 2.2, chi2=False)
    problem = gfem.Driven(shg.frequency, excite=(), sources=shg)
    path = gfem.Job(model, problem, OPTIONS).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path)
    (source,) = back.problem.sources
    assert isinstance(source, gfem.PolarizationSource)
    assert source.name == shg.name
    assert source.process == shg.process
    assert source.order == shg.order
    np.testing.assert_array_equal(source.re, np.asarray(shg.re))
    np.testing.assert_array_equal(source.im, np.asarray(shg.im))
    assert (
        solve_key(back.model, back.problem, back.options)[0]
        == solve_key(model, problem, OPTIONS)[0]
    )
    # a different polarization is another key
    other = gfem.PolarizationSource(
        **{
            **{f: getattr(shg, f) for f in shg.__dataclass_fields__},
            "re": 2 * np.asarray(shg.re),
        }
    )
    assert (
        solve_key(model, gfem.Driven(shg.frequency, excite=(), sources=other), OPTIONS)[
            0
        ]
        != solve_key(model, problem, OPTIONS)[0]
    )


def test_the_repr_is_short_and_names_the_process(shg):
    text = repr(shg)
    assert "second harmonic" in text
    assert "second-harmonic" in text
    assert len(text) < 200


def test_the_far_field_of_a_polarization_source_is_refused(mesh, shg):
    result = _generated(mesh, shg)
    model = slab_model(mesh, 2.2, chi2=False)
    with pytest.raises(NotImplementedError, match="PolarizationSource"):
        gfem.far_field(result, shg.name, [0.0], [0.0], model=model)


def test_the_generated_electric_field_is_kept_and_polarized_along_the_tensor(mesh, shg):
    result = _generated(mesh, shg, keep_fields=True)
    field = result.e_field(shg.frequency, shg.name)
    mid = np.array([[0.5 * SIDE, 0.5 * SIDE, 0.75]]) * 1e-6
    e = np.asarray(field.at(mid))[0]
    # the source and the cell are x-polarized and uniform along x and y: the
    # field in the slab is x-polarized, the other components a discretization
    # error of the coarse order-2 cell (measured 4e-2 of it, tolerance 1e-1)
    assert abs(e[0]) > 0
    assert np.abs(e[1:]).max() <= 1e-1 * abs(e[0])


class TestRefusals:
    def test_the_pump_needs_a_result(self):
        with pytest.raises(TypeError, match=r"DrivenResult|keep_fields"):
            gfem.Pump(object(), FREQUENCY)

    def test_the_pump_needs_its_power(self, pump):
        with pytest.raises(ValueError, match="incident_power"):
            gfem.Pump(pump, FREQUENCY)

    def test_the_power_is_not_negative(self, pump):
        with pytest.raises(ValueError, match=r">= 0"):
            gfem.Pump(pump, FREQUENCY, incident_power=-1.0)

    def test_the_column_must_exist(self, pump):
        with pytest.raises(KeyError, match="nope"):
            gfem.Pump(pump, FREQUENCY, excite="nope", incident_power=1.0)

    def test_the_frequency_must_be_kept(self, pump):
        with pytest.raises(ValueError, match="keeps no field"):
            gfem.Pump(pump, 1.1 * FREQUENCY, incident_power=1.0)

    def test_the_frequency_is_not_traced(self, pump):
        def run(f):
            return gfem.Pump(pump, f, incident_power=1.0)

        with pytest.raises(gfem.NonDifferentiableError):
            jax.jit(run)(FREQUENCY)

    def test_one_pump_takes_no_process(self, pump_model, pump):
        with pytest.raises(ValueError, match="needs two pumps"):
            gfem.second_order_polarization(
                pump_model, _pump(pump, FREQUENCY), process="sum-frequency"
            )

    def test_two_pumps_take_a_process(self, pump_model, pump):
        with pytest.raises(ValueError, match="process="):
            gfem.second_order_polarization(
                pump_model, (_pump(pump, FREQUENCY), _pump(pump, F2))
            )

    def test_two_pumps_at_one_frequency_are_not_a_sum(self, pump_model, pump):
        a = _pump(pump, FREQUENCY)
        with pytest.raises(ValueError, match="second-harmonic"):
            gfem.second_order_polarization(pump_model, (a, a), process="sum-frequency")

    def test_a_difference_takes_the_higher_pump_first(self, pump_model, pump):
        with pytest.raises(ValueError, match="higher"):
            gfem.second_order_polarization(
                pump_model,
                (_pump(pump, FREQUENCY), _pump(pump, F2)),
                process="difference-frequency",
            )

    def test_the_generated_order_is_at_least_one(self, pump_model, pump):
        with pytest.raises(ValueError, match="generated_order"):
            gfem.second_order_polarization(
                pump_model, _pump(pump, FREQUENCY), generated_order=0
            )

    def test_a_model_without_a_tensor_is_refused(self, mesh, pump):
        plain = slab_model(mesh, 2.2, chi2=False)
        with pytest.raises(ValueError, match="second_order_nonlinear_tensor"):
            gfem.second_order_polarization(plain, _pump(pump, FREQUENCY))

    def test_the_pump_must_be_a_pump(self, pump_model):
        with pytest.raises(TypeError, match=r"gfem\.Pump"):
            gfem.second_order_polarization(pump_model, "pump")

    def test_the_frequency_must_match_the_source(self, shg):
        with pytest.raises(ValueError, match="drives one frequency"):
            gfem.Driven(1.5 * shg.frequency, excite=(), sources=shg)

    def test_the_adaptive_sweep_is_refused(self, shg):
        with pytest.raises(NotImplementedError, match="adaptive"):
            gfem.Driven(shg.frequency, excite=(), sources=shg, sweep="adaptive")

    def test_reciprocity_is_refused(self, shg):
        with pytest.raises(NotImplementedError, match="source_fields"):
            gfem.Driven(shg.frequency, excite=(), sources=shg, source_fields=False)

    def test_a_dipole_with_a_bloch_wave_vector_is_still_refused(self, shg):
        dipole = gfem.ElectricDipole("d", (0.0, 0.0, 0.0), (1e-12, 0.0, 0.0))
        with pytest.raises(NotImplementedError, match="incident fields and dipoles"):
            gfem.Driven(
                shg.frequency,
                excite=(),
                sources=(shg, dipole),
                wave_vector=(1e6, 0.0, 0.0),
            )

    def test_a_plain_source_refuses_a_bloch_wave_vector(self, mesh, shg):
        with pytest.raises(ValueError, match=r"Bloch wave vector \(0.0, 0.0, 0.0\)"):
            _generated(mesh, shg, wave_vector=(2e6, 0.0, 0.0))

    def test_the_wrong_vector_is_refused_with_the_one_to_write(self, mesh, bloch):
        _, source = bloch
        with pytest.raises(ValueError, match=r"wave_vector=\(4000000.0, 0.0, 0.0\)"):
            _generated(mesh, source)
        with pytest.raises(ValueError, match="a different array"):
            _generated(mesh, source, wave_vector=(2e6, 0.0, 0.0))

    def test_a_model_with_other_pairs_is_refused(self, mesh, shg):
        one_pair = slab_model(mesh, 2.2, chi2=False).replace(
            boundaries=[PERIODIC[0]], ports=[]
        )
        with pytest.raises(ValueError, match="periodic pair"):
            gfem.solve(
                one_pair,
                gfem.Driven(shg.frequency, excite=(), sources=shg),
                options=OPTIONS,
            )

    def test_a_fixed_angle_sweep_over_several_frequencies_is_refused(self, bloch):
        _, source = bloch
        with pytest.raises(ValueError, match="fixed-angle sweep drops"):
            gfem.Driven(
                [source.frequency, 1.1 * source.frequency],
                excite=(),
                sources=source,
                wave_vector=source.wave_vector,
                wave_vector_frequency=source.frequency,
            )

    def test_another_mesh_is_refused(self, tmp_path, shg):
        other = slab_mesh(tmp_path / "other.msh", 5)
        model = slab_model(other, 2.2, chi2=False)
        with pytest.raises(ValueError, match="volume elements"):
            gfem.solve(
                model,
                gfem.Driven(shg.frequency, excite=(), sources=shg),
                options=OPTIONS,
            )

    def test_a_port_name_is_not_a_source_name(self, mesh, pump_model, pump):
        source = gfem.second_order_polarization(
            pump_model, _pump(pump, FREQUENCY), name="z_max TM"
        )
        with pytest.raises(ValueError, match="names both a port"):
            gfem.solve(
                slab_model(mesh, 2.2, chi2=False),
                gfem.Driven(source.frequency, excite=(), sources=source),
                options=OPTIONS,
            )


#: The pump's Bloch wave vector (rad/m): a phase of 0.5 rad across the cell.
KX = 2e6


@pytest.fixture(scope="module")
def bloch_pump(pump_model):
    """The pump on a Bloch cell at a fixed angle, at two frequencies."""
    return gfem.solve(
        pump_model,
        gfem.Driven(
            [FREQUENCY, F2],
            excite="z_min TM",
            wave_vector=(KX, 0.0, 0.0),
            wave_vector_frequency=FREQUENCY,
            keep_fields=True,
        ),
        options=OPTIONS,
    )


@pytest.fixture(scope="module")
def bloch(pump_model, bloch_pump):
    return bloch_pump, gfem.second_order_polarization(
        pump_model, gfem.Pump(bloch_pump, FREQUENCY, incident_power=1.0)
    )


class TestBlochVectors:
    def test_the_kept_fields_record_the_vector_of_their_frequency(self, bloch_pump):
        first = bloch_pump.e_field(FREQUENCY, "z_min TM")
        second = bloch_pump.e_field(F2, "z_min TM")
        assert first._wave_vector == pytest.approx((KX, 0.0, 0.0), rel=1e-15)
        # a fixed angle scales the vector with the frequency (a product of two
        # rounded numbers: a few eps)
        assert second._wave_vector == pytest.approx(
            (KX * F2 / FREQUENCY, 0, 0), rel=1e-14
        )
        assert len(first._lattice) == 2

    def test_a_plain_pump_has_no_vector_and_the_lattice(self, shg):
        assert shg.wave_vector is None
        assert len(shg.lattice) == 2

    def test_the_second_harmonic_doubles_the_vector(self, bloch):
        assert bloch[1].wave_vector == pytest.approx((2 * KX, 0.0, 0.0), rel=1e-15)

    def test_sum_and_difference_add_and_subtract_the_vectors(
        self, pump_model, bloch_pump
    ):
        k1, k2 = KX, KX * F2 / FREQUENCY
        a = gfem.Pump(bloch_pump, FREQUENCY, incident_power=1.0)
        b = gfem.Pump(bloch_pump, F2, incident_power=1.0)
        total = gfem.second_order_polarization(
            pump_model, (a, b), process="sum-frequency"
        )
        diff = gfem.second_order_polarization(
            pump_model, (b, a), process="difference-frequency"
        )
        assert total.wave_vector == pytest.approx((k1 + k2, 0.0, 0.0), rel=1e-14)
        assert diff.wave_vector == pytest.approx((k2 - k1, 0.0, 0.0), rel=1e-12)

    def test_a_plain_pump_beside_a_bloch_one_counts_as_the_zero_vector(
        self, pump_model, pump, bloch_pump
    ):
        a = gfem.Pump(bloch_pump, F2, incident_power=1.0)
        b = _pump(pump, FREQUENCY)
        total = gfem.second_order_polarization(
            pump_model, (a, b), process="sum-frequency"
        )
        assert total.wave_vector == pytest.approx(
            (KX * F2 / FREQUENCY, 0, 0), rel=1e-14
        )

    def test_the_vector_to_write_solves(self, mesh, bloch):
        _, source = bloch
        result = _generated(mesh, source, wave_vector=source.wave_vector)
        power = float(result.emitted_power[source.name][0])
        assert np.isfinite(power)
        assert power > 0.0

    def test_the_source_round_trips_with_its_vector(self, tmp_path, mesh, bloch):
        _, source = bloch
        model = slab_model(mesh, 2.2, chi2=False)
        problem = gfem.Driven(
            source.frequency, excite=(), sources=source, wave_vector=source.wave_vector
        )
        path = gfem.Job(model, problem, OPTIONS).to_json(tmp_path / "job.json")
        (back,) = gfem.Job.from_json(path).problem.sources
        assert back.wave_vector == source.wave_vector
        assert back.lattice == source.lattice
