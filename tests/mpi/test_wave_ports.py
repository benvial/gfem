"""Wave ports and `gfem.PortMode` through the real `gfem.solve` across ranks.

Every solve compares ranks with the same function in pytest's one process: S bit for
bit across ranks (`on_ranks`), to `S_GATE` against one process, and the Krylov
iterations exactly. `tests/mpi/test_driven_bddc.py` holds the lumped port
twin of these tests.
"""

import dataclasses
import warnings
from pathlib import Path

import bend_cases as bc
import gradient_cases as cases
import jax
import mpi_cases
import numpy as np
import pytest
from ranks import RanksFailed, on_ranks

import gfem
from gfem import _mpi
from gfem._engine.driven import DrivenRule, driven_operator
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.port_network import parallel_scale
from gfem._engine.runtime import task_manager
from gfem._engine.wave_ports import trace_dofs
from gfem._linalg import low_rank_update, prepare
from gfem._linalg.base import free_mask
from gfem._port_section import port_section
from gfem.mesh import Mesh

#: Relative to the largest entry of the one-process S: the BDDC rank gate.
S_GATE = 1e-9
WAVE_SWEEP = (12e9, 24e9)
HYBRID_SWEEP = (55e9, 65e9)
RANKS = (3, 4, 8)
DATA = Path(__file__).parents[1] / "unit" / "data"


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _largest(values) -> float:
    from mpi4py import MPI

    largest = float(np.max(np.abs(values), initial=0.0))
    return _world().allreduce(largest, op=MPI.MAX)


def _model(case, directory):
    """A case by name, so that a launched function can build it from arguments."""
    directory = Path(directory)
    match case:
        case "hybrid":
            return cases.hybrid_line(cases.hybrid_mesh(directory, cells=(4, 2, 2)))
        case "permuted hybrid":
            # fine enough that port faces lie across ranks
            mesh = cases.hybrid_mesh(directory, cells=(8, 8, 4))
            return cases.hybrid_line(mpi_cases.permuted(mesh))
        case "permuted lossless":
            return cases.wave_line(
                mpi_cases.permuted(cases.wave_mesh(directory)), "lossless"
            )
    return cases.wave_line(cases.wave_mesh(directory), case)


def _sweep(case):
    return HYBRID_SWEEP if "hybrid" in case else WAVE_SWEEP


def _s_and_iterations(case, directory):
    seen = {}
    forward = DrivenRule.forward

    def spy(self, model, *, keep, request=None):
        outputs, entry = forward(self, model, keep=keep, request=request)
        seen["iterations"] = outputs["iterations"]
        return outputs, entry

    DrivenRule.forward = spy
    try:
        result = gfem.solve(
            _model(case, directory),
            gfem.Driven(list(_sweep(case))),
            options=cases.tight("bddc_gmres", threads=1),
        )
    finally:
        DrivenRule.forward = forward
    return np.asarray(result.s.values), np.asarray(seen["iterations"])


@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize(
    "case",
    ["lossless", "lossy", "mixed", "hybrid", "permuted lossless", "permuted hybrid"],
)
def test_s_and_iterations_match_one_process(tmp_path, case, ranks):
    s_one, iterations_one = _s_and_iterations(case, tmp_path)
    (s, iterations), *_ = on_ranks(
        _s_and_iterations, case, tmp_path, ranks=ranks, limit=300
    )
    error = np.abs(s - s_one).max() / np.abs(s_one).max()
    print(  # noqa: T201
        f"{case} ranks={ranks} S error={error:.3g} iterations={iterations.tolist()} "
        f"one process={iterations_one.tolist()}"
    )
    assert error < S_GATE, f"S off by {error:.3g} relative"
    np.testing.assert_array_equal(iterations, iterations_one)


# --- what lies under the solve ------------------------------------------------------


def _network(case, directory, comm):
    """The system operator and port network of a case, with no solve."""
    model = _model(case, directory)
    rule = DrivenRule(
        model,
        gfem.Driven(list(_sweep(case))),
        cases.tight("bddc_gmres", threads=1),
        comm=comm,
    )
    with task_manager(1, comm):
        return rule.system(model)


def _reciprocity(case, directory):
    """Each wave port's functional against its excitation (`parallel_scale`)."""
    _, network = _network(case, directory, _mpi.resolve_comm())
    omega = 2 * np.pi * _sweep(case)[0]
    functional = network.functional(omega)
    rhs = network.rhs(omega)
    return [
        parallel_scale(
            functional[:, c.index],
            rhs[:, network.columns.index(c.index)],
            network.layout,
        )
        for c in network.waves
    ]


@pytest.mark.parametrize("case", ["lossless", "lossy"])
def test_a_lone_wave_port_is_reciprocal_with_the_scale_of_one_process(tmp_path, case):
    # lossy too: the biorthogonal functional is parallel to the load
    one = _reciprocity(case, tmp_path)
    scales, *_ = on_ranks(_reciprocity, case, tmp_path, ranks=4)
    assert None not in one
    np.testing.assert_allclose(scales, one, rtol=1e-12, atol=0)


def _update_residual(case, directory):
    """``apply`` of a Woodbury update gives back the load its ``solve`` was given."""
    comm = _mpi.resolve_comm()
    system, network = _network(case, directory, comm)
    with task_manager(1, comm):
        omega = 2 * np.pi * _sweep(case)[0]
        handle = low_rank_update(
            prepare(
                system.operator_at(omega),
                symmetry="complex_symmetric",
                kind="bddc_gmres",
                tolerance=1e-12,
            ),
            (term := system.corrections(omega)).left,
            term.coefficients,
            right=term.right,
        )
        layout = network.layout
        free = free_mask(system.operator.space.FreeDofs(), system.operator.space.ndof)
        rhs = network.rhs(omega)[:, 0]
        back = handle.apply(handle.solve(rhs))
        handle.free()
    return layout.norm(np.where(free, back - rhs, 0.0)) / layout.norm(rhs)


def test_a_woodbury_update_applies_what_it_solves(tmp_path):
    # the hybrid line's modes have E_n, so its correction does not vanish
    residual, *_ = on_ranks(_update_residual, "hybrid", tmp_path, ranks=4)
    assert residual < 1e-9


# --- the trace of a 3D field on a port ----------------------------------------------


#: Three points of a triangle: its centre and two more off its symmetry axes.
BARYCENTRIC = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])


def _trace_against_the_port_space(directory, order, surface):
    """A random 2D field against the 3D field holding its coefficients.

    Each is evaluated through its own mesh's point lookup inside every local port
    element, so this shares no assumption with the DOF map.
    """
    import ngsolve as ngs

    world = _world()
    model = cases.hybrid_line(
        mpi_cases.permuted(cases.hybrid_mesh(directory, cells=(8, 8, 4)))
    )
    port = next(p for p in model.ports if p.surface == surface)
    operator = driven_operator(model, order, comm=_mpi.resolve_comm())
    space, built = operator.space, operator.built
    section = port_section(model, port)
    modes = PortModeSpace(section, order)
    idx3, idx2 = trace_dofs(model, section, modes, built, space)

    nd = modes.space.components[0]
    field2 = ngs.GridFunction(nd)
    rng = np.random.default_rng(11)
    values2 = rng.normal(size=nd.ndof) + 1j * rng.normal(size=nd.ndof)
    field2.vec.FV().NumPy()[:] = values2
    field3 = ngs.GridFunction(space)
    values3 = field3.vec.FV().NumPy()
    values3[:] = 0.0
    values3[idx3] = values2[idx2]

    mesh3, mesh2 = built.mesh, modes.mesh
    elements = [e for e in mesh3.Elements(ngs.BND) if e.mat == surface]

    def worst() -> float:
        out = 0.0
        for element in elements:
            corners = np.array([list(mesh3[v].point) for v in element.vertices])
            for x in BARYCENTRIC @ corners:
                e3 = np.array(field3(mesh3(*x, VOL_or_BND=ngs.BND)))
                u, v = (x - section.origin) @ section.frame[:2].T
                e2 = np.array(field2(mesh2(u, v)))
                in_3d = e2[0] * section.frame[0] + e2[1] * section.frame[1]
                out = max(out, float(np.linalg.norm(e3 - in_3d)))
        return out

    same = worst()
    # the control, on the last rank holding a port DOF
    holders = world.allgather(len(idx3) > 0)
    flipper = max(r for r, held in enumerate(holders) if held)
    if world.rank == flipper:
        values3[idx3[0]] *= -1
    flipped = worst()
    return {
        "same": _largest([same]),
        "flipped": _largest([flipped]),
        # every 2D DOF is found by a rank that holds a face of it
        "found": np.unique(np.concatenate(world.allgather(idx2))),
        "all": np.arange(nd.ndof),
    }


@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_the_trace_is_the_port_field_and_a_planted_flip_is_seen(tmp_path, order, ranks):
    for surface in ("port1", "port2"):
        row, *_ = on_ranks(
            _trace_against_the_port_space,
            tmp_path,
            order,
            surface,
            ranks=ranks,
            limit=300,
        )
        print(  # noqa: T201
            f"order={order} ranks={ranks} {surface}: |E3 - E2| = {row['same']:.3g}, "
            f"with a flip {row['flipped']:.3g}"
        )
        assert row["same"] < 1e-11
        assert row["flipped"] > 1.0
        # edges on no port element stay out, as in one process
        assert np.isin(row["found"], row["all"]).all()
        assert len(row["found"]) > 0


def _port_vector_columns(case, directory):
    """Each surface's `vectors`, paired with the state 1: its sum over all DOFs."""
    system, network = _network(case, directory, _mpi.resolve_comm())
    ones = np.ones(system.operator.space.ndof)
    return [network.layout.pair(s.vectors, ones) for s in system.surfaces]


def test_a_port_load_sums_over_its_writers_to_the_one_process_value(tmp_path):
    one = _port_vector_columns("permuted lossless", tmp_path)
    many, *_ = on_ranks(_port_vector_columns, "permuted lossless", tmp_path, ranks=8)
    for a, b in zip(many, one, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max())


# --- what a job sees ----------------------------------------------------------------


def _below_cutoff(directory, problem_type):
    """The hybrid line's mode 2 is below cutoff at 20 GHz: the error, then a job."""
    model = _model("hybrid", directory)
    problem = {
        "driven": gfem.Driven(20e9),
        "port mode": gfem.PortMode("p1_m2", 20e9),
    }[problem_type]
    options = (
        cases.tight("bddc_gmres", threads=1)
        if problem_type == "driven"
        else gfem.SolverOptions(threads=1)
    )
    try:
        gfem.solve(model, problem, options=options)
    except gfem.PortModeError as err:
        message = str(err)
    else:
        raise AssertionError("a mode below cutoff was solved")
    if _mpi.launched_tasks() == 1:
        return message, 1
    return message, _world().allreduce(1)


@pytest.mark.parametrize("problem_type", ["driven", "port mode"])
def test_a_port_mode_error_raises_on_every_rank_and_a_collective_runs(
    tmp_path, problem_type
):
    alone, _ = _below_cutoff(tmp_path, problem_type)
    (message, ranks), *_ = on_ranks(
        _below_cutoff, tmp_path, problem_type, ranks=4, limit=120
    )
    assert message == alone
    assert ranks == 4


def _off_the_plane(directory):
    """A deterministic error of rank 0's section: a voltage path off the port."""
    model = _model("lossless", directory)
    ports = [
        dataclasses.replace(
            model.ports[0], voltage_path=((1e-3, 0.0, 0.0), (2e-3, 0.0, 0.0))
        ),
        model.ports[1],
    ]
    try:
        gfem.solve(
            model.replace(ports=ports),
            gfem.PortMode("p1", 12e9),
            options=gfem.SolverOptions(threads=1),
        )
    except ValueError as err:
        return str(err), _world().allreduce(1)
    raise AssertionError("a path off the port ran")


def test_a_voltage_path_off_the_port_raises_on_every_rank(tmp_path):
    (message, ranks), *_ = on_ranks(_off_the_plane, tmp_path, ranks=3, limit=60)
    assert "voltage_path leaves the port plane" in message
    assert ranks == 3


def _eigenmode_refused(directory):
    """An eigenmode solve with wave ports: refused before any mesh, then a job."""
    try:
        gfem.solve(
            _model("lossless", directory),
            gfem.Eigenmode(target=WAVE_SWEEP[0]),
            options=gfem.SolverOptions(threads=1),
        )
    except gfem.DistributedUnsupportedError as err:
        return err.feature, str(err), _world().allreduce(1)
    raise AssertionError("an eigenmode solve with a wave port ran")


def test_an_eigenmode_with_a_wave_port_raises_on_every_rank(tmp_path):
    (feature, message, ranks), _ = on_ranks(
        _eigenmode_refused, tmp_path, ranks=2, limit=60
    )
    assert feature == "eigenmode_wave_port"
    assert "WavePort 'p1'" in message
    assert ranks == 2


def _root_fails(directory):
    del directory

    def compute():
        raise OSError("rank 0's own failure")

    _mpi.on_root(_world(), compute, results=(ValueError,))


def test_any_other_failure_on_rank_0_ends_the_job(tmp_path):
    with pytest.raises(RanksFailed, match="rank 0's own failure"):
        on_ranks(_root_fails, tmp_path, ranks=3, limit=60)


def _crossing():
    """Mode 4 of the slab guide crosses inside this band (`test_port_mode_sweep`)."""
    return (
        _slab(),
        gfem.PortMode("p1", np.linspace(16e9, 40e9, 60), mode=4),
        gfem.SolverOptions(threads=1),
    )


def _slab():
    return gfem.Model(
        gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm"),
        materials={"slab": gfem.Material(eps_r=4.0), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def _skipped(directory):
    model = _model("hybrid", directory)
    ports = [gfem.WavePort("p1", "port1", mode=2), gfem.WavePort("p2", "port2", mode=2)]
    return (
        model.replace(ports=ports),
        gfem.Driven(55e9),
        cases.tight("bddc_gmres", threads=1),
    )


def _low_frequency(directory):
    return (
        cases.layered_line(directory),
        gfem.Driven(1.0),
        cases.tight("bddc_gmres", threads=1),
    )


def _squashed(directory):
    model = cases.layered_line(directory)
    mesh = model.mesh
    moved = Mesh(np.asarray(mesh.nodes) * np.array([1.0, 1.0, 0.01]), mesh._topology)
    return (
        model.replace(mesh=moved),
        gfem.Driven(5e9),
        cases.tight("bddc_gmres", threads=1),
    )


#: Each warning a driven or port mode solve can give, and a solve that gives it.
WARNING_SITES = {
    "PortModeTrackingWarning": lambda _: _crossing(),
    "SkippedPortModeWarning": _skipped,
    "LowFrequencyWarning": _low_frequency,
    "MeshQualityWarning": _squashed,
}


def _under_an_error_filter(name, directory):
    category = getattr(gfem, name)
    model, problem, options = WARNING_SITES[name](directory)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category)
        try:
            gfem.solve(model, problem, options=options)
        except category as err:
            return str(err), _world().allreduce(1)
    raise AssertionError(f"no {name}")


@pytest.mark.parametrize("name", sorted(WARNING_SITES))
def test_a_warning_under_an_error_filter_raises_on_every_rank(tmp_path, name):
    (message, ranks), *_ = on_ranks(
        _under_an_error_filter, name, tmp_path, ranks=4, limit=120
    )
    assert message
    assert ranks == 4


def _warnings_seen(name, directory):
    category = getattr(gfem, name)
    model, problem, options = WARNING_SITES[name](directory)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = gfem.solve(model, problem, options=options)
    return [str(w.message) for w in seen if w.category is category], result.gamma


def test_a_tracking_warning_is_shown_by_rank_0_alone(tmp_path):
    rows = on_ranks(
        _warnings_seen,
        "PortModeTrackingWarning",
        tmp_path,
        ranks=3,
        limit=120,
        replicated=False,
    )
    assert rows[0][0]
    assert "mode crossing" in rows[0][0][0]
    assert rows[1][0] == []
    assert rows[2][0] == []
    np.testing.assert_array_equal(rows[1][1], rows[0][1])


# --- gfem.PortMode ------------------------------------------------------------------


def _port_mode(case, directory):
    if case == "slab":
        model, port, frequencies = _slab(), "p1", np.linspace(10e9, 20e9, 8)
    else:
        model, port, frequencies = _model(case, directory), "p1_m2", HYBRID_SWEEP
    result = gfem.solve(
        model, gfem.PortMode(port, frequencies), options=gfem.SolverOptions(threads=1)
    )
    section = port_section(model, next(p for p in model.ports if p.name == port))
    points = section.to_global(section.points[section.triangles].mean(axis=1))
    return {
        "gamma": np.asarray(result.gamma),
        "Z_PV": np.asarray(result.Z_PV),
        "residual": np.asarray(result.residual),
        "solved": np.asarray(result.solved),
        "e_field": np.asarray(result.e_field.at(points)),
        "n_g": np.asarray(result.n_g),
        "beta_2": np.asarray(result.beta_2),
        "beta_3": np.asarray(result.beta_3),
    }


#: Rank gate of each compared output (relative to its largest entry): the
#: dispersion's higher orders amplify the assembly's rounding.
PORT_MODE_GATES = {
    "gamma": 1e-8,
    "e_field": 1e-8,
    "n_g": 1e-8,
    "beta_2": 1e-8,
    "beta_3": 1e-7,
}


@pytest.mark.parametrize("case", ["slab", "hybrid"])
@pytest.mark.parametrize("ranks", [3, 8])
def test_a_port_mode_solve_agrees_with_one_process(tmp_path, case, ranks):
    one = _port_mode(case, tmp_path)
    row, *_ = on_ranks(_port_mode, case, tmp_path, ranks=ranks, limit=180)
    for name, gate in PORT_MODE_GATES.items():
        error = np.abs(row[name] - one[name]).max() / np.abs(one[name]).max()
        print(f"{case} ranks={ranks} {name} error={error:.3g}")  # noqa: T201
        assert error < gate, name
    np.testing.assert_array_equal(row["solved"], one["solved"])


def _port_mode_under_jax():
    model = _slab()
    options = gfem.SolverOptions(threads=1)
    mesh = model.mesh
    # a point inside a triangle of the port face (m)
    point = np.asarray(mesh.nodes)[mesh.triangles[mesh.elements("port1")[0]]].mean(0)

    def solved(eps):
        n = len(mesh.elements("slab"))
        moved = model.with_material(
            "slab", gfem.MaterialField(eps_r=jax.numpy.full(n, eps))
        )
        return gfem.solve(moved, gfem.PortMode("p1", 12e9), options=options)

    def gamma(eps):
        return solved(eps).gamma

    def field(eps):
        # a field cotangent needs the 2D adjoint solve, on rank 0 alone
        result = solved(eps)
        return result.gamma.imag.sum() + jax.numpy.sum(
            jax.numpy.abs(result.e_field.at(point)) ** 2
        )

    value = np.asarray(jax.jit(gamma)(4.0))
    # the 2D adjoint runs on rank 0 alone, and the ranks' sum counts it once
    grad = float(jax.grad(lambda eps: gamma(eps).imag.sum())(4.0))
    grad_field = float(jax.grad(field)(4.0))
    return value, grad, grad_field


def test_a_port_mode_under_jax_is_as_on_one_process():
    value_one, grad_one, field_one = _port_mode_under_jax()
    (value, grad, field), *_ = on_ranks(_port_mode_under_jax, ranks=3, limit=180)
    assert abs(value - value_one).max() < 1e-8 * abs(value_one).max()
    assert grad == pytest.approx(grad_one, rel=1e-9)
    assert field == pytest.approx(field_one, rel=1e-9)


def _bent_port_mode(mesh_path):
    mesh = Mesh.from_gmsh(Path(mesh_path), length_unit="mm")
    # a lossy filling, so that bend_loss and Re gamma carry a signal to compare
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material(eps_r=1.2, tan_delta=0.02)},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "p1")],
    )
    options = gfem.SolverOptions(order=3, threads=1)

    def solved(radius):
        problem = gfem.PortMode("p1", bc.WR90_F, bend=bc.wr90_bend_m(radius))
        return gfem.solve(model, problem, options=options)

    result = solved(30e-3)
    # the 2D adjoint of the radius runs on rank 0 alone, and the ranks' sum
    # counts it once
    slope = jax.grad(lambda r: jax.numpy.real(solved(r).gamma[0]))(
        jax.numpy.asarray(30e-3)
    )
    return (
        complex(np.asarray(result.gamma)[0]),
        float(np.asarray(result.bend_loss)[0]),
        float(slope),
    )


def test_a_bent_port_mode_agrees_with_one_process(tmp_path):
    path = bc.wr90_mesh(tmp_path / "wr90.msh")
    one = np.array(_bent_port_mode(path))
    row, *_ = on_ranks(_bent_port_mode, path, ranks=2, limit=180)
    # the 2D section is solved on the same data whatever the ranks; only the
    # order of a reduction differs, a few ulps of the solve (1e-13), so 1e-10
    # holds with three orders to spare. gamma, bend_loss and d Re gamma / dR
    np.testing.assert_allclose(np.array(row), one, rtol=1e-10, atol=0)


def _named_solver(directory):
    options = gfem.SolverOptions(linear_solver="direct")
    model = _model("lossless", directory)
    return np.asarray(
        gfem.solve(model, gfem.PortMode("p1", 12e9), options=options).gamma
    )


def test_a_port_mode_naming_a_solver_ignores_it_on_every_rank(tmp_path):
    # the 2D problem runs SuperLU and SciPy whatever the options name
    one = np.asarray(
        gfem.solve(_model("lossless", tmp_path), gfem.PortMode("p1", 12e9)).gamma
    )
    gamma, *_ = on_ranks(_named_solver, tmp_path, ranks=3, limit=60)
    np.testing.assert_allclose(gamma, one, rtol=1e-8, atol=0)
