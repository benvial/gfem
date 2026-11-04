"""Kept fields through the real `gfem.solve` across ranks (ADR 0063).

A kept field is a rank's piece, and each of its methods is collective: every rank
gets the same bits of a value, checked by `on_ranks`, and they agree with the same
method in pytest's one process. A value at a tet's centroid is off every face, where
a point's tet depends on the partition, one process included.
"""

import dataclasses
from pathlib import Path

import gradient_cases as cases
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from ranks import on_ranks
from solver_misses import cannot_converge_at

import gfem
from gfem import _mpi, _petsc
from gfem._engine import fields as engine_fields
from gfem._mpi import AgreedError

#: Relative to the largest one-process value: ADR 0070's gates.
GATES = {"bddc_gmres": 1e-9, "direct": 1e-11}
SOLVERS = (
    "bddc_gmres",
    pytest.param(
        "direct",
        marks=pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack"),
    ),
)
RANKS = (3, 4, 8)
FREQUENCIES = (18e9, 27e9)
STEPS = {"eps_r": 1e-2, "conductivity": 5e-2}
#: Sizes of this line's fields, so that an objective is of order 1.
FIELD_SCALE = 1e3
B_SCALE = 1e-5
#: A fourth-order difference per direction, two directions.
SAMPLES_PER_GRADIENT = 8
MM = 1e-3
OMEGA = 2 * np.pi * 27e9
#: Relative to the largest exact value, on a column no solver touched.
EXACT_GATE = 1e-12


def _design(directory):
    line = cases.layered_line(Path(directory))
    values, _ = cases.line_design(line, seed=2)
    return line, values


def _centroids(mesh, n=16, seed=3):
    """Points well inside ``n`` tets, off every face."""
    chosen = np.random.default_rng(seed).choice(len(mesh.tetrahedra), n, replace=False)
    return np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)[chosen]].mean(axis=1)


def _solved(directory, solver):
    line, values = _design(directory)
    model = cases.with_fields(line, values)
    driven = gfem.Driven(frequency=FREQUENCIES, keep_fields=[FREQUENCIES[1]])
    return gfem.solve(model, driven, options=cases.tight(solver, threads=1))


def _assert_below_the_gate(a, b, what, solver):
    error = np.abs(np.asarray(a) - b).max() / np.abs(b).max()
    print(f"{what} {solver} error={error:.3g}")  # noqa: T201
    assert error < GATES[solver], f"{what} off by {error:.3g} relative"


def _values(directory, solver):
    result = _solved(directory, solver)
    field = result.e_field(FREQUENCIES[1], "p1")
    return np.asarray(field.at(_centroids(field.mesh)))


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("ranks", RANKS)
def test_at_matches_one_process_off_faces(tmp_path, ranks, solver):
    one = _values(tmp_path, solver)
    values, *_ = on_ranks(_values, tmp_path, solver, ranks=ranks, limit=300)
    assert np.abs(one).max() > 0
    _assert_below_the_gate(values, one, f"at ranks={ranks}", solver)


# --- a physics gradient through the field ---------------------------------------------


def _at_objective(field):
    """A real function of a field's values at points, order 1 in size."""
    values = field.at(_centroids(field.mesh))
    weights = np.arange(1.0, 1.0 + values.size).reshape(values.shape)
    return jnp.sum(weights * jnp.abs(values / FIELD_SCALE) ** 2)


def _row(row, directory, solver):
    """``(objective, design, directions, step)`` of one row."""
    line, values = _design(directory)
    if row == "moved":
        nodes = np.asarray(line.mesh.nodes).copy()
        nodes[:, 0] += 1e-4 * np.sin(np.pi * nodes[:, 0] / (4 * MM))
        line = line.replace(
            mesh=jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(line.mesh), [jnp.asarray(nodes)]
            )
        )
    rng = np.random.default_rng(11)
    name = "conductivity" if row == "conductivity" else "eps_r"
    kept = FREQUENCIES[1]

    def solved(p, **problem):
        model = cases.with_fields(line, {**values, name: p})
        driven = gfem.Driven(frequency=FREQUENCIES, keep_fields=[kept], **problem)
        return gfem.solve(model, driven, options=cases.tight(solver, threads=1))

    def objective(p):
        match row:
            case "eps_r" | "conductivity" | "moved":
                return _at_objective(solved(p).e_field(kept, "p1"))
            case "one_port":
                return _at_objective(solved(p, excite="p2").e_field(kept, "p2"))
            case "both_ports":
                result = solved(p)
                return _at_objective(result.e_field(kept, "p1")) + 0.5 * _at_objective(
                    result.e_field(kept, "p2")
                )
            case "with_s":
                result = solved(p)
                return _at_objective(result.e_field(kept, "p1")) + cases.s_functional(
                    result.s.values
                )
            case "b":
                field = solved(p).e_field(kept, "p1").b()
                values_b = field.at(_centroids(field.mesh))
                return jnp.sum(jnp.abs(values_b / B_SCALE) ** 2)
        raise ValueError(row)

    p = values[name]
    return objective, p, unit_directions(rng, p.shape, 2), STEPS[name]


def _gradient(row, directory, solver, *, differences=True):
    """The value, the gradient and every finite-difference sample of a row."""
    objective, p, directions, step = _row(row, directory, solver)
    samples = []

    def sampled(x):
        value = objective(x)
        samples.append(float(value))
        return value

    value, grad = jax.value_and_grad(objective)(p)
    grad = np.asarray(grad)
    if differences:
        assert_gradient_matches(sampled, p, directions, step=step, grad=grad)
    return np.asarray(value), grad, np.asarray(samples)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize(
    "row", ["eps_r", "conductivity", "one_port", "both_ports", "with_s", "b"]
)
def test_a_physics_gradient_matches_one_process_and_differences(
    tmp_path, row, ranks, solver
):
    _, grad_one, _ = _gradient(row, tmp_path, solver, differences=False)
    (_, grad, samples), *_ = on_ranks(
        _gradient, row, tmp_path, solver, ranks=ranks, limit=600
    )
    assert np.isfinite(grad).all()
    assert np.abs(grad_one).max() > 0
    assert len(samples) == SAMPLES_PER_GRADIENT
    _assert_below_the_gate(grad, grad_one, f"{row} ranks={ranks}", solver)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("ranks", RANKS)
def test_a_gradient_on_a_moved_mesh_matches_one_process(tmp_path, ranks, solver):
    _, grad_one, _ = _gradient("moved", tmp_path, solver, differences=False)
    (_, grad, samples), *_ = on_ranks(
        _gradient, "moved", tmp_path, solver, ranks=ranks, limit=600
    )
    assert len(samples) == SAMPLES_PER_GRADIENT
    _assert_below_the_gate(grad, grad_one, f"moved ranks={ranks}", solver)


# --- under jit and vmap ---------------------------------------------------------------


def _transformed(row, directory):
    objective, p, _, _ = _row(row, directory, "bddc_gmres")
    eager = objective(p)
    jitted = jax.jit(objective)(p)
    twice = jax.vmap(objective)(jnp.stack([p, 1.01 * p]))
    grad = jax.grad(objective)(p)
    jit_grad = jax.jit(jax.grad(objective))(p)
    return {
        "eager": np.asarray(eager),
        "jit": np.asarray(jitted),
        "vmap": np.asarray(twice),
        "grad": np.asarray(grad),
        "jit_grad": np.asarray(jit_grad),
    }


@pytest.mark.parametrize("row", ["eps_r", "b"])
@pytest.mark.parametrize("ranks", RANKS)
def test_jit_of_a_field_only_objective_runs_on_every_rank(tmp_path, ranks, row):
    got, *_ = on_ranks(_transformed, row, tmp_path, ranks=ranks, limit=300)
    assert np.abs(got["jit"] - got["eager"]) <= 1e-12 * np.abs(got["eager"])
    assert np.abs(got["vmap"][0] - got["eager"]) <= 1e-12 * np.abs(got["eager"])
    assert got["vmap"][1] != got["vmap"][0]
    error = np.abs(got["jit_grad"] - got["grad"]).max() / np.abs(got["grad"]).max()
    assert error < 1e-11


# --- the piece, the load and b ------------------------------------------------------


def _analytic(mesh, order, comm):
    """A polynomial E in the space, as a field; its curl is known without gfem.

    In the mesh's coordinates ``u``: ``(u_y, u_z, u_x)`` at order 2, whose curl is
    ``(-1, -1, -1)``; ``(u_y^2, u_z^2, u_x^2)`` at order 3, ``-2 (u_z, u_x, u_y)``.
    """
    import ngsolve as ngs

    power = order - 1
    support = ("sub", "cover")
    _, space, _ = engine_fields.fe_space(mesh, support, (), "hcurl", order, comm=comm)
    function = ngs.GridFunction(space)
    function.Set(ngs.CF((ngs.y**power, ngs.z**power, ngs.x**power)))
    width, ndof = engine_fields.dof_shape(mesh, support, (), "hcurl", order, comm=comm)
    piece = np.zeros(width)
    piece[:ndof] = function.vec.FV().NumPy()
    return gfem.Field(
        coefficients=jnp.asarray(piece, dtype=complex),
        mesh=mesh,
        quantity="E",
        unit="V/m",
        space="hcurl",
        order=order,
        support=support,
        omega=jnp.asarray(OMEGA),
        _comm=_mpi.static_comm(comm),
    )


def _exact_e(points, order):
    u = points / MM
    return np.stack([u[:, 1], u[:, 2], u[:, 0]], axis=1) ** (order - 1) / MM + 0j


def _exact_b(points, order):
    u = points / MM
    curl = (
        -np.ones_like(u)
        if order == 2
        else -2 * np.stack([u[:, 2], u[:, 0], u[:, 1]], axis=1)
    )
    return curl / MM**2 / (-1j * OMEGA)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _b_and_e(directory, order):
    mesh = cases.layered_line(Path(directory)).mesh
    field = _analytic(mesh, order, _world())
    points = _centroids(mesh)
    return (
        np.asarray(field.at(points)),
        np.asarray(field.b().at(points)),
        field.b().coefficients.shape,
    )


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("ranks", RANKS)
def test_b_with_holder_weights_is_the_exact_curl(tmp_path, ranks, order):
    (e, b, _), *_ = on_ranks(_b_and_e, tmp_path, order, ranks=ranks, limit=120)
    mesh = cases.layered_line(tmp_path).mesh
    points = _centroids(mesh)
    exact_e, exact_b = _exact_e(points, order), _exact_b(points, order)
    b_one = np.asarray(_analytic(mesh, order, None).b().at(points))
    e_error = np.abs(e - exact_e).max() / np.abs(exact_e).max()
    b_error = np.abs(b - exact_b).max() / np.abs(exact_b).max()
    b_tesla = np.abs(b - exact_b).max()
    one_error = np.abs(b_one - exact_b).max() / np.abs(exact_b).max()
    print(  # noqa: T201
        f"ranks={ranks} order={order} E {e_error:.3g} B {b_error:.3g} relative, "
        f"{b_tesla:.3g} T of {np.abs(exact_b).max():.3g}; one process {one_error:.3g}"
    )
    assert e_error < EXACT_GATE
    assert b_error < EXACT_GATE
    assert b_tesla < 1e-15


def _load_pairing(directory, order):
    """A rank sum of the transpose's load against a column, and its one value."""
    mesh = cases.layered_line(Path(directory)).mesh
    comm = _world()
    field = _analytic(mesh, order, comm)
    points = _centroids(mesh)
    rng = np.random.default_rng(13)
    g = rng.standard_normal((len(points), 3)) + 1j * rng.standard_normal(
        (len(points), 3)
    )

    def at(c):
        return dataclasses.replace(field, coefficients=c).at(points)

    values, pullback = jax.vjp(at, field.coefficients)
    (load,) = pullback(jnp.asarray(g))
    load = np.asarray(load)
    ndof = engine_fields.dof_shape(mesh, field.support, (), "hcurl", order, comm=comm)[
        1
    ]
    pair = _mpi.reduce_bcast(
        comm, np.array([np.dot(load, np.asarray(field.coefficients))])
    )[0]
    return {
        "pair": pair,
        "values": np.sum(g * np.asarray(values)),
        "exact": np.sum(g * _exact_e(points, order)),
        "padding": comm.allreduce(float(np.abs(load[ndof:]).sum())),
    }


@pytest.mark.parametrize("ranks", RANKS)
def test_the_transpose_is_a_load_counted_once(tmp_path, ranks):
    got, *_ = on_ranks(_load_pairing, tmp_path, 3, ranks=ranks, limit=120)
    assert abs(got["pair"] - got["exact"]) < EXACT_GATE * abs(got["exact"])
    assert abs(got["values"] - got["exact"]) < EXACT_GATE * abs(got["exact"])
    assert got["padding"] == 0.0


def _piece_shape(directory):
    result = _solved(directory, "bddc_gmres")
    field = result.e_field(FREQUENCIES[1], "p1")
    width, ndof = engine_fields.dof_shape(
        field.mesh, field.support, (), "hcurl", 2, comm=_world()
    )
    coefficients = np.asarray(field.coefficients)
    return {
        "shape": coefficients.shape,
        "width": width,
        "ndof": ndof,
        "padding": float(np.abs(coefficients[ndof:]).sum()),
        "repr": repr(field),
    }


def test_a_piece_has_one_shape_with_zeros_past_its_own_rows(tmp_path):
    got = on_ranks(_piece_shape, tmp_path, ranks=4, limit=120, replicated=False)
    assert {g["shape"] for g in got} == {(max(g["ndof"] for g in got),)}
    assert {g["width"] for g in got} == {max(g["ndof"] for g in got)}
    assert got[0]["ndof"] == 0
    assert all(g["padding"] == 0.0 for g in got)
    assert len({g["repr"] for g in got}) == 1


# --- what raises ----------------------------------------------------------------------


def _feature(call):
    try:
        call()
    except gfem.DistributedUnsupportedError as err:
        return err.feature
    return None


def _refused_methods(directory):
    line, values = _design(directory)
    model = cases.with_fields(line, values)
    result = _solved(directory, "bddc_gmres")
    field = result.e_field(FREQUENCIES[1], "p1")
    seen = {
        "to_meshio": _feature(field.to_meshio),
        "to_pyvista": _feature(field.to_pyvista),
        "power_flow": _feature(lambda: gfem.power_flow(field, "sub", model=model)),
    }
    return seen, _world().allreduce(_world().rank)


def test_export_and_power_flow_raise_their_keys_on_every_rank(tmp_path):
    (seen, total), *_ = on_ranks(_refused_methods, tmp_path, ranks=3, limit=120)
    assert seen == {
        "to_meshio": "field_export",
        "to_pyvista": "field_export",
        "power_flow": "power_flow",
    }
    assert total == 0 + 1 + 2


def _message(call):
    try:
        call()
    except ValueError as err:
        return str(err)
    return None


def _bad_points(directory):
    result = _solved(directory, "bddc_gmres")
    field = result.e_field(FREQUENCIES[1], "p1")
    mesh = field.mesh
    cover = np.asarray(mesh.tetrahedra)[mesh.elements("cover")[0]]
    inside_cover = np.asarray(mesh.nodes)[cover].mean(axis=0)
    only_sub = dataclasses.replace(field, support=("sub",))
    return (
        _message(lambda: field.at(np.array([1.0, 1.0, 1.0]))),
        _message(lambda: only_sub.at(inside_cover)),
        _world().allreduce(_world().rank),
        np.asarray(field.at(_centroids(mesh))),
    )


def test_a_point_with_no_value_raises_on_every_rank_and_the_job_goes_on(tmp_path):
    (outside, unsupported, total, values), *_ = on_ranks(
        _bad_points, tmp_path, ranks=4, limit=120
    )
    assert "outside the mesh" in outside
    assert "where the field is not defined" in unsupported
    assert total == 0 + 1 + 2 + 3
    assert np.isfinite(values).all()


def _points_differing_on_one_rank(directory):
    result = _solved(directory, "bddc_gmres")
    field = result.e_field(FREQUENCIES[1], "p1")
    points = _centroids(field.mesh)
    moved = points.copy()
    if _world().rank == 1:
        moved[0] = _centroids(field.mesh, seed=4)[0]
    try:
        field.at(moved)
    except AgreedError as err:
        named = "points" in str(err)
    else:
        named = False
    return named, np.asarray(field.at(points))


def test_points_differing_on_one_rank_raise_on_every_rank(tmp_path):
    (named, values), *_ = on_ranks(
        _points_differing_on_one_rank, tmp_path, ranks=3, limit=120
    )
    assert named
    assert np.isfinite(values).all()


# --- other paths to a field -----------------------------------------------------------


def _partial_field(directory):
    line, values = _design(directory)
    model = cases.with_fields(line, values)
    driven = gfem.Driven(frequency=FREQUENCIES, keep_fields=[FREQUENCIES[0]])
    options = cases.tight("bddc_gmres", threads=1)
    whole = gfem.solve(model, driven, options=options)
    points = _centroids(line.mesh)
    with pytest.MonkeyPatch.context() as patch:
        cannot_converge_at(patch, (FREQUENCIES[1],))
        try:
            gfem.solve(model, driven, options=options)
        except gfem.LinearSolverConvergenceError as err:
            partial = err.result
    return (
        np.asarray(partial.e_field(FREQUENCIES[0], "p1").at(points)),
        np.asarray(whole.e_field(FREQUENCIES[0], "p1").at(points)),
    )


def test_a_partial_result_keeps_its_fields_on_every_rank(tmp_path):
    (partial, whole), *_ = on_ranks(_partial_field, tmp_path, ranks=4, limit=300)
    _assert_below_the_gate(partial, whole, "partial", "bddc_gmres")


def _wave_values(directory):
    mesh = cases.wave_mesh(Path(directory))
    model = cases.wave_line(mesh, "lossy")
    design, _ = cases.core_design(mesh, seed=4)
    model = model.with_material("core", gfem.MaterialField(**design))
    driven = gfem.Driven(frequency=(12e9,), keep_fields=True)
    options = cases.tight("bddc_gmres", threads=1)
    field = gfem.solve(model, driven, options=options).e_field(12e9, "p1")
    return np.asarray(field.at(_centroids(mesh)))


@pytest.mark.parametrize("ranks", [3, 4])
def test_a_field_with_wave_ports_matches_one_process(tmp_path, ranks):
    one = _wave_values(tmp_path)
    values, *_ = on_ranks(_wave_values, tmp_path, ranks=ranks, limit=300)
    _assert_below_the_gate(values, one, f"wave ranks={ranks}", "bddc_gmres")
