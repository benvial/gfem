"""gfem.solve on a Driven problem: lumped ports on a parallel-plate line.

The line runs along x between PEC plates at z = 0 and z = h, with PMC side
walls (the default), so its TEM mode is exact: a uniform E_z. Lumped ports
on the end faces see a transmission line of impedance Z0 = eta0 h / (W
sqrt(eps_r)) and the S-parameters have a closed form (ABCD matrices), up to
the discretization's dispersion error.
"""

import cmath
import dataclasses
import math

import jax
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm
from preconditioners import count_preconditioners

import gfem
from gfem._engine import driven, terms
from gfem._engine.driven import DrivenRule
from gfem._linalg import bddc, prepare
from gfem._linalg.base import same_pattern
from gfem._linalg.bddc import BddcHandle
from gfem._linalg.direct import DirectHandle

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0
F = 5e9
R = 50.0
ETA0 = const.mu_0 * const.c


def _mesh(tmp, cells=(10, 1, 1), **kwargs):
    path = tmp / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, cells, **kwargs))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh):
    """The strip line between PEC plates, a lumped port at each end."""
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return _mesh(tmp_path_factory.mktemp("line"))


@pytest.fixture(scope="module")
def line(mesh):
    return _line(mesh)


@pytest.fixture(scope="module")
def fine_line(tmp_path_factory):
    return _line(_mesh(tmp_path_factory.mktemp("fine_line"), cells=(20, 2, 2)))


def _abcd_line(f, length_m, eps=EPS):
    beta = 2 * math.pi * f / const.c * math.sqrt(eps)
    z0 = ETA0 * H / (W * math.sqrt(eps))
    c, s = math.cos(beta * length_m), math.sin(beta * length_m)
    return np.array([[c, 1j * z0 * s], [1j * s / z0, c]])


def _shunt(y):
    return np.array([[1, 0], [y, 1]], complex)


def _s_from_abcd(m, r=R):
    (a, b), (c, d) = m
    den = a + b / r + c * r + d
    return np.array(
        [
            [(a + b / r - c * r - d) / den, 2 * (a * d - b * c) / den],
            [2 / den, (-a + b / r - c * r + d) / den],
        ]
    )


def _s(model, problem=None, **options):
    problem = problem or gfem.Driven(frequency=F)
    options.setdefault("linear_solver", "direct")
    return gfem.solve(model, problem, options=gfem.SolverOptions(**options))


# --- the problem value -----------------------------------------------------------


def test_driven_normalizes_its_frequencies_and_excitations():
    p = gfem.Driven(frequency=[1e9, 2e9], excite="p1")
    assert p.frequency == (1e9, 2e9)
    assert p.excite == ("p1",)
    assert p.sweep == "uniform"
    assert gfem.Driven(frequency=3e9).frequency == (3e9,)
    assert gfem.Driven(frequency=3e9).excite is None
    assert gfem.Driven(frequency=1e9, excite=["a", "b"]).excite == ("a", "b")


def test_driven_is_frozen_hashable_and_static_under_jax():
    p = gfem.Driven(frequency=np.array([1e9, 2e9]))
    assert p == gfem.Driven(frequency=(1e9, 2e9))
    assert hash(p) == hash(gfem.Driven(frequency=[1e9, 2e9]))
    assert jax.tree.leaves(p) == []
    with pytest.raises(AttributeError):
        p.sweep = "adaptive"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"frequency": -1.0}, ValueError, "> 0 Hz"),
        ({"frequency": []}, ValueError, "one frequency"),
        ({"frequency": 1e9, "sweep": "fast"}, ValueError, "'uniform'"),
        ({"frequency": 1e9, "excite": []}, ValueError, "at least one port"),
        ({"frequency": 1e9, "excite": ["a", "a"]}, ValueError, "twice"),
        ({"frequency": 1e9, "excite": [1]}, TypeError, "port names"),
    ],
)
def test_driven_rejects_bad_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        gfem.Driven(**kwargs)


# --- network parameters ----------------------------------------------------------


def test_network_parameters_index_by_port_name():
    values = np.arange(12).reshape(3, 2, 2) * (1 + 1j)
    net = gfem.NetworkParameters(
        values=values,
        z_ref=np.full((3, 2), 50.0),
        frequency=(1e9, 2e9, 3e9),
        ports=("a", "b"),
        excited=("a", "b"),
    )
    np.testing.assert_array_equal(net["b", "a"], values[:, 1, 0])
    np.testing.assert_allclose(net.db("a", "b"), 20 * np.log10(np.abs(values[:, 0, 1])))
    np.testing.assert_array_equal(net.frequency, [1e9, 2e9, 3e9])
    leaves = jax.tree_util.tree_flatten_with_path(net)[0]
    assert [jax.tree_util.keystr(p) for p, _ in leaves] == [".values", ".z_ref"]
    with pytest.raises(KeyError, match="'c'"):
        net["c", "a"]
    with pytest.raises(KeyError, match="not excited"):
        gfem.NetworkParameters(
            values=values[:, :, :1],
            z_ref=np.full((3, 2), 50.0),
            frequency=(1e9, 2e9, 3e9),
            ports=("a", "b"),
            excited=("a",),
        )["a", "b"]


# --- physics on the parallel-plate line ------------------------------------------


def test_two_port_line_matches_the_transmission_line(line):
    result = _s(line)
    assert isinstance(result, gfem.DrivenResult)
    s = np.asarray(result.s.values)
    assert s.shape == (1, 2, 2)
    expected = _s_from_abcd(_abcd_line(F, L * 1e-3))
    np.testing.assert_allclose(s[0], expected, atol=2e-3)
    np.testing.assert_array_equal(result.s.z_ref, [[R, R]])
    assert result.s.ports == result.s.excited == ("p1", "p2")
    np.testing.assert_array_equal(result.frequency, [F])


def test_lossless_line_is_reciprocal_and_conserves_power(line):
    # The port loads also absorb the (tiny) non-TEM part of the discrete field,
    # which the port voltage does not see: S is passive, unitary to ~1e-8.
    s = np.asarray(_s(line, gfem.Driven(frequency=[2e9, 7e9])).s.values)
    for m in s:
        np.testing.assert_allclose(m, m.T, atol=1e-12)
        np.testing.assert_allclose(m.conj().T @ m, np.eye(2), atol=1e-6)
        assert np.linalg.svd(m, compute_uv=False).max() <= 1 + 1e-12


def test_lossy_line_is_passive(mesh):
    lossy = gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=EPS, tan_delta=0.02, conductivity=0.01)
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    s = np.asarray(_s(lossy).s.values)[0]
    np.testing.assert_allclose(s, s.T, atol=1e-12)
    assert np.linalg.svd(s, compute_uv=False).max() < 1 - 1e-3


def test_per_axis_loss_tangent_acts_along_its_axis(mesh):
    # the TEM field is E_z: only the principal values along z (the second
    # axis here) load the line, as an isotropic material with those values
    # would (up to the discrete field's small E_x / E_y part)
    def model(material):
        return gfem.Model(
            mesh,
            materials={"dielectric": material},
            boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
            ports=[
                gfem.LumpedPort("p1", "port1", "+z"),
                gfem.LumpedPort("p2", "port2", "+z"),
            ],
        )

    per_axis = gfem.Material(
        eps_r=(3.0, EPS, 6.0),
        tan_delta=(0.2, 0.02, 0.1),
        axes=((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    )
    s = np.asarray(_s(model(per_axis)).s.values)
    isotropic = np.asarray(_s(model(gfem.Material(eps_r=EPS, tan_delta=0.02))).s.values)
    np.testing.assert_allclose(s, isotropic, atol=1e-5)


def test_bddc_gmres_is_the_default_and_matches_direct(line):
    problem = gfem.Driven(frequency=[3e9, 4e9])
    direct = np.asarray(_s(line, problem).s.values)
    rule = DrivenRule(line, problem, gfem.SolverOptions())
    assert rule.kind == "bddc_gmres"
    outputs, entry = rule.forward(line, keep=False)
    assert entry is None
    np.testing.assert_allclose(outputs["s"], direct, atol=1e-6)
    # GMRES at the second frequency starts from the first one's solution
    assert len(outputs["iterations"]) == 2
    assert all(len(its) == 2 for its in outputs["iterations"])


def _count_prepares(monkeypatch) -> list[int]:
    """Spy on the driven sweep's `prepare`: one entry per base handle."""
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(driven, "prepare", spy)
    return calls


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_direct_sweep_plans_once_and_matches_fresh_handles(line, monkeypatch, sweep):
    problem = gfem.Driven(frequency=(3e9, 4e9, 5e9, 6e9), sweep=sweep)
    calls = _count_prepares(monkeypatch)
    s = np.asarray(_s(line, problem).s.values)
    assert len(calls) == 1
    # a handle that cannot refactorize is prepared afresh at each frequency
    monkeypatch.delattr(DirectHandle, "refactorize")
    calls.clear()
    result = _s(line, problem)
    solved = 4 if sweep == "uniform" else len(result.sweep.sampled_frequencies)
    assert len(calls) == solved > 1
    np.testing.assert_allclose(result.s.values, s, rtol=0, atol=1e-12)


def test_a_bddc_sweep_keeps_its_preconditioner_and_matches_fresh_setups(
    line, monkeypatch
):
    problem = gfem.Driven(frequency=(3e9, 3.5e9, 4e9, 4.5e9))
    builds = count_preconditioners(monkeypatch)
    s = np.asarray(_s(line, problem, linear_solver="bddc_gmres").s.values)
    assert len(builds) == 1
    # without refactorization, BDDC is set up afresh at each frequency
    monkeypatch.delattr(BddcHandle, "refactorize")
    builds.clear()
    fresh = np.asarray(_s(line, problem, linear_solver="bddc_gmres").s.values)
    assert len(builds) == 4
    np.testing.assert_allclose(s, fresh, rtol=0, atol=1e-6)


def test_a_wide_bddc_sweep_rebuilds_its_reused_preconditioner(line, monkeypatch):
    # a preconditioner from 1 GHz costs GMRES 3x its iterations at 15 GHz
    problem = gfem.Driven(frequency=np.linspace(1e9, 29e9, 8))
    direct = np.asarray(_s(line, problem).s.values)
    builds = count_preconditioners(monkeypatch)
    s = np.asarray(_s(line, problem, linear_solver="bddc_gmres").s.values)
    assert 1 < len(builds) < 8
    # GMRES stops at 1e-6 of |P b|; a reused P measures the residual less well
    np.testing.assert_allclose(s, direct, rtol=0, atol=1e-5)


def test_a_reused_solve_at_its_cap_rebuilds_and_converges(line, monkeypatch):
    # 1.5 GHz sets the reference; 30 GHz would take ~90 iterations per port
    problem = gfem.Driven(frequency=(1e9, 1.5e9, 30e9))
    direct = np.asarray(_s(line, problem).s.values)
    builds = count_preconditioners(monkeypatch)
    s = np.asarray(_s(line, problem, linear_solver="bddc_gmres").s.values)
    # set up at 1 GHz, and again inside the 30 GHz solve (no later refactorization)
    assert len(builds) == 2
    np.testing.assert_allclose(s, direct, rtol=0, atol=1e-5)


def test_a_bddc_sweep_rebuilds_at_the_same_iterations_every_run(line):
    # the rebuild rule counts iterations, never times: runs repeat exactly
    problem = gfem.Driven(frequency=(1e9, 1.5e9, 12e9, 13e9, 20e9, 30e9))
    rule = DrivenRule(line, problem, gfem.SolverOptions())
    first, _ = rule.forward(line, keep=False)
    second, _ = rule.forward(line, keep=False)
    assert first["iterations"] == second["iterations"]
    assert max(map(max, first["iterations"])) > 45  # a capped solve, set up again


def _bddc_forward(model, problem):
    outputs, _ = DrivenRule(model, problem, gfem.SolverOptions()).forward(
        model, keep=False
    )
    return outputs


@pytest.mark.parametrize("model", ["line", "fine_line"])
def test_a_bddc_sweep_caps_the_first_warm_solve_after_a_large_jump(
    model, request, monkeypatch
):
    # BDDC set up at 1 GHz and reused at 60 GHz would need more than MAXITER
    # GMRES iterations on the fine line: 60 GHz stops at n_ref + 2 delta, sets
    # BDDC up, and goes on from the partial iterate
    model = request.getfixturevalue(model)
    problem = gfem.Driven(frequency=(1e9, 60e9))
    direct = np.asarray(_s(model, problem).s.values)
    builds = count_preconditioners(monkeypatch)
    out = _bddc_forward(model, problem)
    np.testing.assert_allclose(out["s"], direct, rtol=0, atol=1e-5)
    assert len(builds) == 2
    monkeypatch.undo()
    slack = bddc._slack(2)
    cold = max(_bddc_forward(model, gfem.Driven(frequency=60e9))["iterations"][0])
    # the capped stale part, then a fresh solve from the partial iterate, which
    # may take a few more than a cold one
    bound = max(out["iterations"][0]) + 2 * slack + cold + slack
    assert max(out["iterations"][1]) <= bound


def test_a_bddc_adaptive_sweep_sets_bddc_up_at_every_sample(fine_line, monkeypatch):
    # adaptive samples lie far apart (1 GHz, then 60 GHz): a reused
    # preconditioner would mostly hit its cap, so each sample sets BDDC up
    problem = gfem.Driven(frequency=np.linspace(1e9, 60e9, 60), sweep="adaptive")
    direct = np.asarray(_s(fine_line, problem).s.values)
    builds = count_preconditioners(monkeypatch)
    out = _bddc_forward(fine_line, problem)
    np.testing.assert_allclose(out["s"], direct, rtol=0, atol=1e-4)
    assert len(builds) == len(out["iterations"]) > 1


def test_excite_selects_the_columns_of_s(line):
    full = np.asarray(_s(line).s.values)
    one = _s(line, gfem.Driven(frequency=F, excite="p2")).s
    assert one.excited == ("p2",)
    assert np.asarray(one.values).shape == (1, 2, 1)
    np.testing.assert_allclose(one.values, full[:, :, 1:], atol=1e-12)


def test_a_multi_element_port_drives_its_elements_in_parallel(tmp_path, line):
    split = gfem.Model(
        _mesh(tmp_path, split_ports=True),
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", ("port1a", "port1b"), ("+z", "+z")),
            gfem.LumpedPort("p2", ("port2a", "port2b"), ("-z", "-z")),
        ],
    )
    s = np.asarray(_s(split).s.values)
    ref = np.asarray(_s(line).s.values)
    # port 2 is driven the other way round: its waves change sign
    flip = np.diag([1, -1])
    np.testing.assert_allclose(s[0], flip @ ref[0] @ flip, atol=1e-10)


def test_reactive_terminations_load_the_line_in_parallel(mesh):
    lh, cf = 2e-9, 0.3e-12
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", resistance=R, inductance=lh),
            gfem.LumpedPort("p2", "port2", "+z", resistance=R, capacitance=cf),
        ],
    )
    s = np.asarray(_s(model).s.values)[0]
    w = 2 * math.pi * F
    abcd = _shunt(1 / (1j * w * lh)) @ _abcd_line(F, L * 1e-3) @ _shunt(1j * w * cf)
    np.testing.assert_allclose(s, _s_from_abcd(abcd), atol=2e-3)


def test_first_order_absorbing_boundary_matches_the_tem_wave(mesh):
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("port2")],
        ports=[gfem.LumpedPort("p1", "port1", "+z", impedance=R)],
    )
    s = np.asarray(_s(model, gfem.Driven(frequency=[2e9, 6e9])).s.values)
    z0 = ETA0 * H / (W * math.sqrt(EPS))
    np.testing.assert_allclose(s[:, 0, 0], (z0 - R) / (z0 + R), atol=1e-5)


def test_good_conductor_plates_approach_pec_and_lose_power(mesh, line):
    def plates(sigma):
        return gfem.Model(
            mesh,
            materials={"dielectric": gfem.Material(eps_r=EPS)},
            boundaries=[
                gfem.Conductor("bottom", sigma, 1e-6),
                gfem.Conductor("top", sigma, 1e-6),
            ],
            ports=line.ports,
        )

    pec = np.asarray(_s(line).s.values)[0]
    copper = np.asarray(_s(plates(5.8e7)).s.values)[0]
    poor = np.asarray(_s(plates(1e5)).s.values)[0]
    np.testing.assert_allclose(copper, copper.T, atol=1e-12)
    assert abs(copper[1, 0] - pec[1, 0]) < 0.05
    assert np.linalg.svd(copper, compute_uv=False).max() < 1
    assert abs(poor[1, 0]) < abs(copper[1, 0])


def test_conductor_surface_impedance_and_exterior_thickness():
    sigma, t, f = 5.8e7, 1e-6, 10e9
    w = 2 * math.pi * f
    delta = math.sqrt(2 / (const.mu_0 * sigma * w))
    thick = terms.surface_impedance(sigma, None, w)
    assert thick == pytest.approx((1 + 1j) / (sigma * delta), rel=1e-14)
    # a film much thinner than the skin depth: each face carries half the
    # current, 2 / (sigma t) per face
    film = terms.surface_impedance(sigma, delta / 1000, w)
    assert film.real == pytest.approx(2 / (sigma * delta / 1000), rel=1e-5)
    # a film much thicker than the skin depth is the thick conductor
    assert terms.surface_impedance(sigma, 50 * delta, w) == pytest.approx(thick)
    assert terms.surface_impedance(sigma, t, w) != pytest.approx(thick)


def test_frequency_terms_are_affine_pieces():
    piece = terms.Piece("mass", lambda w: -(w**2), lambda u, v: u * v)
    assert piece.theta(2.0) == -4.0
    with pytest.raises(ValueError, match="not affine"):
        terms.Piece("bad", lambda w: np.ones(3) * w, lambda u, v: u * v)


def test_operator_pieces_sum_to_the_driven_operator(line):
    rule = DrivenRule(line, gfem.Driven(frequency=F), gfem.SolverOptions())
    op = rule.operator(line)
    names = [p.name for p in op.pieces]
    assert names[:2] == ["curl_curl", "mass"]
    assert "port p1 R" in names
    assert all(callable(p.theta) for p in op.pieces)


def _every_piece_kind(tmp):
    """A line with a piece of every kind: volume, boundary, sheet and port terms.

    A conductive region, a first- and second-order absorbing side, a one-sided
    conductor plate, an interior impedance sheet with a fill, a lumped R/L/C
    port and a wave port (its Robin piece).
    """
    eps = 1e-9 * L

    def volume(c):
        return "left" if c[0] < L / 2 else "right"

    def surface(c):
        for x, name in ((0.0, "port1"), (L, "port2"), (L / 2, "sheet")):
            if abs(c[0] - x) < eps:
                return name
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - H) < eps:
            return "top"
        return "side"

    edges = (np.linspace(0, L, 11), np.linspace(0, W, 3), np.linspace(0, H, 2))
    path = sm.write_msh22(tmp / "sheet.msh", sm.structured(edges, volume, surface))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    fill = gfem.SheetField(np.linspace(0.2, 1.0, len(mesh.elements("sheet"))))
    rlc = {"resistance": R, "inductance": 1e-9, "capacitance": 1e-13}
    return gfem.Model(
        mesh,
        materials={
            "left": gfem.Material(eps_r=EPS),
            "right": gfem.Material(eps_r=2.0, conductivity=1.0),
        },
        boundaries=[
            gfem.PEC("top"),
            gfem.Conductor("bottom", 1e3, 5e-6),
            gfem.Absorbing("side", order=2),
            gfem.Impedance("sheet", **rlc, fill=fill),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", **rlc),
            gfem.WavePort("p2", "port2"),
        ],
    )


def test_a_direct_handle_sums_the_pieces_into_the_assembled_operator(tmp_path):
    model = _every_piece_kind(tmp_path)
    frequencies = (4e9, 5e9)
    options = gfem.SolverOptions(linear_solver="direct")
    rule = DrivenRule(model, gfem.Driven(frequencies), options)
    ports = rule._ports(model)
    names = {p.name for p in ports.operator.pieces}
    assert {
        "conductivity",
        "absorbing",
        "absorbing order 2",
        "conductor bottom (one-sided)",
        "impedance sheet C",
        "port p1 L",
        "waveport p2",
    } <= names
    handle = None
    for f in frequencies:
        omega = 2 * math.pi * f
        operator = ports.operator_at(omega)
        if handle is None:
            handle = DirectHandle.from_operator(operator, "complex_symmetric")
        else:
            handle.refactorize(operator)
        with pytest.raises(Exception, match="assemble"):  # the form is not assembled
            _ = operator.form.mat
        form = ports.operator.form(omega)
        form.Assemble()
        assert same_pattern(handle._matrix, form.mat)
        summed = handle._matrix.AsVector().FV().NumPy()
        assembled = form.mat.AsVector().FV().NumPy()
        assert np.abs(summed - assembled).max() <= 1e-14 * np.abs(assembled).max()


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_direct_sweep_by_piece_sums_matches_assembled_operators(
    line, monkeypatch, sweep
):
    problem = gfem.Driven(frequency=(3e9, 4e9, 5e9, 6e9), sweep=sweep)
    summed = np.asarray(_s(line, problem).s.values)
    operator_at = driven._Ports.operator_at

    def assembled(ports, omega):  # the operator without its piece sum
        return dataclasses.replace(operator_at(ports, omega), affine=None)

    monkeypatch.setattr(driven._Ports, "operator_at", assembled)
    s = np.asarray(_s(line, problem).s.values)
    np.testing.assert_allclose(summed, s, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["direct", "bddc_gmres"])
@pytest.mark.parametrize("keep", [True, False])
def test_a_forward_holds_its_assembled_pieces_with_its_handles(
    line, monkeypatch, kind, keep
):
    options = gfem.SolverOptions(linear_solver=kind)
    rule = DrivenRule(line, gfem.Driven(frequency=(4e9, 5e9)), options)
    built = []
    make_ports = rule._ports

    def spy(model):
        built.append(make_ports(model))
        return built[-1]

    monkeypatch.setattr(rule, "_ports", spy)
    _, entry = rule.forward(line, keep=keep)
    (ports,) = built
    system = ports.system
    vectors = sum(t.vectors.nbytes for t in (*system.excitation, *system.outputs))
    form = ports.operator.form(1.0)
    form.Assemble()
    if not (kind == "direct" and keep):
        # BDDC assembles forms; a forward that keeps nothing drops its pieces
        assert entry is None or entry.handle.nbytes == vectors
        assert ports.nbytes == vectors
        return
    # at least the values of the curl-curl and mass pieces, complex
    assert ports.nbytes >= vectors + 2 * 16 * form.mat.nze
    assert entry.handle.nbytes == ports.nbytes
    entry.free()
    assert ports.nbytes == vectors


# --- what the driven problem refuses ---------------------------------------------


def _with(line, **changes):
    return line.replace(**changes)


@pytest.mark.parametrize(
    ("change", "problem", "error", "message"),
    [
        ({}, gfem.Driven(frequency=F, excite="p3"), ValueError, "'p3'"),
        ({"ports": []}, None, ValueError, "no ports"),
        (
            {"ports": [gfem.LumpedPort("p1", "port1", "+z", capacitance=1e-12)]},
            None,
            ValueError,
            "resistance",
        ),
        (
            {"ports": [gfem.LumpedPort("p1", "port1", "+x")]},
            None,
            gfem.PortGeometryError,
            "tangent",
        ),
    ],
)
def test_unsupported_or_inconsistent_driven_inputs_raise(
    line, change, problem, error, message
):
    model = _with(line, **change)
    with pytest.raises(error, match=message):
        _s(model, problem)


def test_result_repr_and_db(line):
    result = _s(line)
    assert "p1" in repr(result)
    assert "GHz" in repr(result)
    s21 = complex(np.asarray(result.s["p2", "p1"])[0])
    assert float(result.s.db("p2", "p1")[0]) == pytest.approx(20 * math.log10(abs(s21)))
    assert cmath.isfinite(s21)


# --- kept fields -----------------------------------------------------------------


def test_keep_fields_is_off_by_default_and_normalized():
    assert gfem.Driven(frequency=F).keep_fields is False
    assert gfem.Driven(frequency=F, keep_fields=True).keep_fields is True
    p = gfem.Driven(frequency=[1e9, 2e9, 3e9], keep_fields=[3e9, 1e9 * (1 + 1e-14)])
    assert p.keep_fields == (1e9, 3e9)
    assert gfem.Driven(frequency=[1e9, 2e9], keep_fields=2e9).keep_fields == (2e9,)
    with pytest.raises(ValueError, match=r"4e\+09.*not one of"):
        gfem.Driven(frequency=[1e9, 2e9], keep_fields=[4e9])
    with pytest.raises(TypeError, match="keep_fields"):
        gfem.Driven(frequency=1e9, keep_fields="all")


def test_fields_are_not_kept_by_default(line):
    result = _s(line)
    assert result.e_fields == ()
    with pytest.raises(ValueError, match="keep_fields"):
        result.e_field(F, excite="p1")


def _poynting_power(field):
    """Time-averaged power along +x, (1/2) Re ∫ E x H* . x dV / length, W."""
    import ngsolve as ngs

    from gfem._engine.fields import LENGTH_POWER, fe_space

    b = field.b()
    excluded = field._excluded
    parts = {}
    for f in (field, b):
        built, fes, region = fe_space(f.mesh, f.support, excluded, f.space, f.order)
        factor = built.scale ** LENGTH_POWER[f.space]
        for part, values in (("re", f.coefficients.real), ("im", f.coefficients.imag)):
            gf = ngs.GridFunction(fes)
            gf.vec.FV().NumPy()[:] = np.asarray(values) * factor
            parts[f.quantity, part] = gf
    e_re, e_im, b_re, b_im = (parts[q, p] for q in ("E", "B") for p in ("re", "im"))
    flux = ngs.Cross(e_re, b_re) + ngs.Cross(e_im, b_im)
    volume = ngs.Integrate(flux[0], built.mesh, definedon=region, order=8)
    volume *= built.scale**3  # SI volume
    return 0.5 * volume / const.mu_0 / (L * 1e-3)


def test_kept_field_is_the_peak_phasor_for_one_watt_incident(line):
    result = _s(line, gfem.Driven(frequency=F, keep_fields=True))
    e = result.e_field(F, excite="p1")
    assert isinstance(e, gfem.Field)
    assert (e.quantity, e.unit, e.space, e.order) == ("E", "V/m", "hcurl", 2)
    assert e.support == ("dielectric",)
    assert e.is_complex
    assert float(e.omega) == pytest.approx(2 * math.pi * F)
    s = np.asarray(result.s.values)[0]
    # the lossless line carries 1 - |S11|^2 of the 1 W incident to the far
    # port, where its load absorbs |S21|^2
    power = _poynting_power(e)
    assert power == pytest.approx(1 - abs(s[0, 0]) ** 2, rel=1e-6)
    assert power == pytest.approx(abs(s[1, 0]) ** 2, rel=1e-6)
    # fed from port 2, the power flows the other way
    assert _poynting_power(result.e_field(F, excite="p2")) == pytest.approx(-power)


def test_kept_field_on_a_matched_line_carries_one_watt(mesh):
    z0 = ETA0 * H / (W * math.sqrt(EPS))
    matched = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", resistance=z0),
            gfem.LumpedPort("p2", "port2", "+z", resistance=z0),
        ],
    )
    result = _s(matched, gfem.Driven(frequency=F, excite="p1", keep_fields=True))
    e = result.e_field(F)  # the only excited port
    assert _poynting_power(e) == pytest.approx(1.0, rel=1e-4)
    # peak voltage of a 1 W wave on z0: |V| = sqrt(2 z0 P), uniform E_z = V / H
    sampled = e.to_meshio().point_data
    ez = sampled["E_real"][:, 2] + 1j * sampled["E_imag"][:, 2]
    # (sampled per element: the coarse mesh's non-TEM part is ~0.5 %)
    peak = math.sqrt(2 * z0) / (H * 1e-3)
    assert np.mean(np.abs(ez)) == pytest.approx(peak, rel=1e-3)
    np.testing.assert_allclose(np.abs(ez), peak, rtol=1e-2)


def test_keep_fields_selects_frequencies_and_labels_them(line):
    problem = gfem.Driven(frequency=[2e9, 3e9, 4e9], excite="p2", keep_fields=[4e9])
    result = _s(line, problem)
    assert result.kept == ((4e9, "p2"),)
    assert len(result.e_fields) == 1
    assert float(result.e_field(4e9).omega) == pytest.approx(2 * math.pi * 4e9)
    with pytest.raises(ValueError, match=r"3e\+09.*4e\+09"):
        result.e_field(3e9)
    with pytest.raises(KeyError, match="'p1'"):
        result.e_field(4e9, excite="p1")
    both = _s(line, gfem.Driven(frequency=F, keep_fields=True))
    with pytest.raises(ValueError, match="excite="):
        both.e_field(F)


def test_kept_fields_come_out_of_jit_unchanged(line):
    problem = gfem.Driven(frequency=F, excite="p1", keep_fields=True)
    options = gfem.SolverOptions(linear_solver="direct")
    plain = gfem.solve(line, problem, options=options).e_field(F).coefficients

    @jax.jit
    def coefficients(model):
        return gfem.solve(model, problem, options=options).e_field(F).coefficients

    np.testing.assert_allclose(coefficients(line), plain, rtol=1e-12, atol=1e-12)


def test_kept_fields_round_trip_through_a_result_bundle(line, tmp_path):
    problem = gfem.Driven(frequency=F, excite="p1", keep_fields=True)
    options = gfem.SolverOptions(linear_solver="direct")
    result = gfem.Job(line, problem, options).run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    assert loaded.kept == result.kept
    e, back = result.e_field(F), loaded.e_field(F)
    np.testing.assert_array_equal(back.coefficients, e.coefficients)
    assert (back.space, back.order, back.support) == (e.space, e.order, e.support)
    assert _poynting_power(back) == pytest.approx(_poynting_power(e), rel=1e-12)


def test_driven_job_and_result_round_trip(line, tmp_path):
    problem = gfem.Driven(frequency=[2e9, 3e9], excite="p1")
    options = gfem.SolverOptions(linear_solver="direct")
    job = gfem.Job.from_json(
        gfem.Job(line, problem, options).to_json(tmp_path / "j.json")
    )
    assert job.problem == problem
    result = job.run(out=tmp_path / "results")
    loaded = gfem.load_result(tmp_path / "results")
    assert type(loaded) is gfem.DrivenResult
    assert (loaded.s.ports, loaded.s.excited) == (("p1", "p2"), ("p1",))
    assert loaded.s.frequency == (2e9, 3e9)
    np.testing.assert_array_equal(loaded.s.values, result.s.values)
    np.testing.assert_array_equal(loaded.s.z_ref, result.s.z_ref)
