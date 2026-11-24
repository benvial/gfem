"""``cache=`` on `gfem.solve`: the same call again reads its result back."""

import dataclasses
import inspect
import json
import logging
import warnings

import jax
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine.electrostatic import ElectrostaticRule
from gfem._solve_cache import SolveCache
from gfem.solve import _solve

LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
ELECTROSTATIC = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
OPTIONS = gfem.SolverOptions(order=1)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """A two-layer electrostatic model."""
    path = tmp_path_factory.mktemp("stack") / "stack.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)


@pytest.fixture
def forwards(monkeypatch):
    """How many electrostatic forwards ran."""
    calls = []
    forward = ElectrostaticRule.forward

    def counted(self, *args, **kwargs):
        calls.append(1)
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(ElectrostaticRule, "forward", counted)
    return calls


def _entries(root):
    return [p for p in root.iterdir() if len(p.name) == 64]


def _assert_bit_equal(a, b):
    assert type(a) is type(b)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
    ):
        assert np.asarray(x).dtype == np.asarray(y).dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_the_same_call_again_reads_the_result_back_without_solving(
    stack, forwards, tmp_path
):
    first = gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=tmp_path / "c")
    second = gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=tmp_path / "c")
    assert len(forwards) == 1
    _assert_bit_equal(second, first)
    np.testing.assert_array_equal(second.capacitance, first.capacitance)
    assert all(f.mesh is stack.mesh for f in second.potential.values())
    assert len(_entries(tmp_path / "c")) == 1


def _squashed(model):
    """The model with a top corner pushed down: poor elements, none inverted."""
    nodes = np.asarray(model.mesh.nodes).copy()
    z = nodes[:, 2]
    corner = np.flatnonzero(np.isclose(z, z.max()))[0]
    nodes[corner, 2] = 5e-7 + 0.1 * 2.5e-7
    return gfem.Model(_with_nodes(model.mesh, nodes), materials=MATERIALS)


def _with_nodes(mesh, nodes):
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [nodes])


def _solve_noting_warnings(model, cache):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        gfem.solve(model, ELECTROSTATIC, options=OPTIONS, cache=cache)
    return [(w.category, w.filename, w.lineno) for w in caught], line


def test_a_hit_shows_the_solve_s_warnings_at_the_caller_s_line(
    stack, forwards, tmp_path
):
    model = _squashed(stack)
    miss, line = _solve_noting_warnings(model, tmp_path / "c")
    hit, _ = _solve_noting_warnings(model, tmp_path / "c")
    expected = [(gfem.MeshQualityWarning, __file__, line)]
    assert miss == expected
    assert hit == expected
    assert len(forwards) == 1


def test_hits_and_misses_are_logged(stack, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="gfem.solve_cache")
    for _ in range(2):
        gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=tmp_path / "c")
    (entry,) = _entries(tmp_path / "c")
    lines = [r.getMessage() for r in caplog.records if r.name == "gfem.solve_cache"]
    assert lines == [
        f"miss {entry.name[:12]}: solved and stored ({entry})",
        f"hit {entry.name[:12]} ({entry})",
    ]


def test_a_gradient_through_a_cached_solve_stores_nothing(stack, tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="gfem.solve_cache")

    def capacitance(nodes):
        model = gfem.Model(_with_nodes(stack.mesh, nodes), materials=MATERIALS)
        result = gfem.solve(model, ELECTROSTATIC, options=OPTIONS, cache=tmp_path / "c")
        return result.capacitance[0, 0]

    assert np.isfinite(jax.grad(capacitance)(stack.mesh.nodes)).all()
    assert not (tmp_path / "c").exists() or not _entries(tmp_path / "c")
    assert "bypass: traced inputs" in caplog.messages


def test_a_solve_with_adjoint_loads_stores_nothing(stack, tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="gfem.solve_cache")
    cache = SolveCache(tmp_path / "c")
    _solve(
        stack, ELECTROSTATIC, options=OPTIONS, adjoint_loads=lambda _: {}, cache=cache
    )
    assert not (tmp_path / "c").exists()
    assert cache.report() == {"status": "bypass", "key": None}
    assert "bypass: adjoint loads (refine_mesh)" in caplog.messages


def test_a_solve_that_raises_stores_nothing(stack, tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="gfem.solve_cache")

    def fails(*_args, **_kwargs):
        raise gfem.LinearSolverConvergenceError("did not converge")

    monkeypatch.setattr(ElectrostaticRule, "forward", fails)
    with pytest.raises(gfem.LinearSolverConvergenceError):
        gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=tmp_path / "c")
    assert not _entries(tmp_path / "c")
    (line,) = caplog.messages
    assert line.endswith(
        ": nothing stored (the solve raised LinearSolverConvergenceError)"
    )


def test_a_hit_estimates_like_the_fresh_result(stack, tmp_path):
    order2 = gfem.SolverOptions(order=2)
    fresh = gfem.solve(stack, ELECTROSTATIC, options=order2, cache=tmp_path / "c")
    hit = gfem.solve(stack, ELECTROSTATIC, options=order2, cache=tmp_path / "c")
    expected = gfem.error_estimate(fresh, gfem.FieldEstimator(), model=stack)
    estimate = gfem.error_estimate(hit, gfem.FieldEstimator(), model=stack)
    np.testing.assert_array_equal(estimate.indicators, expected.indicators)


def test_a_warm_cache_still_raises_for_a_region_the_mesh_lacks(stack, tmp_path):
    gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=tmp_path / "c")
    bad = gfem.Electrostatic(terminals={"top": "top", "mid": "no_such_region"})
    with pytest.raises(gfem.RegionNotFoundError):
        gfem.solve(stack, bad, options=OPTIONS, cache=tmp_path / "c")


def test_a_hit_on_cuda_needs_no_gpu(stack, tmp_path):
    cpu = gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=tmp_path / "c")
    cuda = dataclasses.replace(OPTIONS, device="cuda")
    hit = gfem.solve(stack, ELECTROSTATIC, options=cuda, cache=tmp_path / "c")
    _assert_bit_equal(hit, cpu)


def _solve_catching(model, cache, options=OPTIONS):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        line = inspect.currentframe().f_lineno + 1
        result = gfem.solve(model, ELECTROSTATIC, options=options, cache=cache)
    found = [
        (w.filename, w.lineno, str(w.message))
        for w in caught
        if w.category is gfem.SolveCacheWarning
    ]
    return result, found, line


def test_an_entry_whose_result_no_longer_constructs_is_solved_again(
    stack, forwards, tmp_path
):
    first = gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=tmp_path / "c")
    (entry,) = _entries(tmp_path / "c")
    doc = json.loads((entry / "entry.json").read_text())
    doc["result"]["potential"] = 5  # not a mapping of fields
    (entry / "entry.json").write_text(json.dumps(doc))
    again, found, line = _solve_catching(stack, tmp_path / "c")
    ((filename, lineno, message),) = found
    assert (filename, lineno) == (__file__, line)
    assert "could not be read (TypeError" in message, message
    assert len(forwards) == 2
    _assert_bit_equal(again, first)
    (entry,) = _entries(tmp_path / "c")
    assert isinstance(
        json.loads((entry / "entry.json").read_text())["result"]["potential"], dict
    )


def test_an_input_the_key_cannot_encode_warns_once_and_solves(
    stack, forwards, tmp_path
):
    @dataclasses.dataclass(frozen=True)
    class Glass(gfem.Material):
        pass

    model = stack.with_material("oxide", Glass(eps_r=3.9))
    for _ in range(2):
        result, found, line = _solve_catching(model, tmp_path / "c")
        ((filename, lineno, message),) = found
        assert (filename, lineno) == (__file__, line)
    assert len(forwards) == 2
    assert np.isfinite(result.capacitance).all()
    assert not (tmp_path / "c").exists()  # decided before the directory
    assert message.startswith(f"cache={str(tmp_path / 'c')!r} was given, but")
    assert "cannot key on model.materials['oxide'] (" in message
    assert "Glass" in message


def test_a_directory_that_is_not_a_cache_raises_before_solving(
    stack, forwards, tmp_path
):
    (tmp_path / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="is not a gfem solve cache"):
        gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=tmp_path)
    assert not forwards


def test_cache_must_be_a_path(stack):
    with pytest.raises(TypeError, match="cache must be a directory path"):
        gfem.solve(stack, ELECTROSTATIC, options=OPTIONS, cache=True)


def _wave_port_line(tmp_path):
    """The model of a WR-90 line with a dielectric and a wave port, and its options."""
    # WR-90 from a short at z = 0 through eps_r = 4 to z = 10 mm, then air to
    # a wave port at z = 15 mm (mm)
    edges = (np.linspace(0, 22.86, 5), np.linspace(0, 10.16, 2), np.linspace(0, 15, 7))

    def surface(c):
        return "port" if abs(c[2] - 15) < 1e-9 else "walls"

    path = tmp_path / "guide.msh"

    def volume(c):
        return "dielectric" if c[2] < 10 else "air"

    sm.write_msh22(path, sm.structured(edges, volume, surface))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            "dielectric": gfem.Material(eps_r=4.0),
            "air": gfem.Material(eps_r=1.0),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port")],
    )
    return model, gfem.SolverOptions(order=2, linear_solver="direct")


def test_an_eigenmode_solve_with_a_wave_port_is_read_back(tmp_path, monkeypatch):
    from gfem._eigen.rule import EigenmodeRule

    calls = []
    forward = EigenmodeRule.forward

    def counted(self, *args, **kwargs):
        calls.append(1)
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(EigenmodeRule, "forward", counted)
    model, options = _wave_port_line(tmp_path)
    problem = gfem.Eigenmode(target=11.67e9)
    first = gfem.solve(model, problem, options=options, cache=tmp_path / "c")
    second = gfem.solve(model, problem, options=options, cache=tmp_path / "c")
    assert len(calls) == 1
    assert list(second.q_port) == ["p1"]
    _assert_bit_equal(second, first)


def test_a_cache_hit_replays_the_unrequested_port_mode_warning(tmp_path, monkeypatch):
    from gfem._eigen import share

    # every mode counts as unrequested, so the line's own mode is warned
    monkeypatch.setattr(share, "THRESHOLD", 2.0)
    model, options = _wave_port_line(tmp_path)
    problem = gfem.Eigenmode(target=11.67e9)
    with pytest.warns(gfem.UnrequestedPortModeWarning) as miss:
        first = gfem.solve(model, problem, options=options, cache=tmp_path / "c")
    with pytest.warns(gfem.UnrequestedPortModeWarning) as hit:
        second = gfem.solve(model, problem, options=options, cache=tmp_path / "c")
    assert [str(w.message) for w in hit] == [str(w.message) for w in miss]
    np.testing.assert_array_equal(
        second.requested_power_share["port"], first.requested_power_share["port"]
    )
