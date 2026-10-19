"""Engine caches, the pinned quadrature rule, and kept forward solves."""

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import cache, quadrature
from gfem._engine.cache import Entry, EntryCache, discretization
from gfem._engine.electrostatic import ElectrostaticRule


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 2)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"box": gfem.Material(eps_r=2.0)}
    )


# --- quadrature --------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pinned_rule_integrates_degree_2p_exactly(model, order):
    mesh = discretization(model.mesh).built.mesh  # unit cube in file units
    f = ngs.x ** (2 * order)
    vol = ngs.Integrate(f * quadrature.dx(order), mesh)
    top = ngs.Integrate(
        f * quadrature.ds(order, definedon=mesh.Boundaries("top")), mesh
    )
    assert vol == pytest.approx(1 / (2 * order + 1), rel=1e-13)
    assert top == pytest.approx(1 / (2 * order + 1), rel=1e-13)


def test_pinned_rule_is_the_one_used_not_a_default(model):
    # degree 2 cannot integrate x^4 exactly on two cells: the rule is pinned
    mesh = discretization(model.mesh).built.mesh
    value = ngs.Integrate(ngs.x**4 * quadrature.dx(1), mesh)
    assert abs(value - 1 / 5) > 1e-6


# --- discretization cache ----------------------------------------------------------


def test_discretization_is_shared_by_one_mesh_and_rebuilt_for_moved_nodes(model):
    first = discretization(model.mesh)
    assert discretization(model.mesh) is first
    moved = model.mesh.nodes.at[:, 2].multiply(2.0)
    other = discretization(type(model.mesh)(moved, model.mesh._topology))
    assert other is not first
    assert other.built.mesh.nv == first.built.mesh.nv


def test_spaces_are_cached_by_key(model):
    disc = discretization(model.mesh)
    calls = []

    def build():
        calls.append(1)
        return object()

    a = disc.space(("em", (), "h1", 7), build)
    assert disc.space(("em", (), "h1", 7), build) is a
    assert len(calls) == 1


# --- solve entries -----------------------------------------------------------------


class FakeHandle:
    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.freed = False

    def free(self):
        self.freed = True


def _entry(nbytes):
    return Entry(physics="em", handle=FakeHandle(nbytes))


def test_entries_are_evicted_least_recently_used_first_over_budget():
    entries = EntryCache()
    a, b, c = _entry(40), _entry(40), _entry(40)
    ka = entries.put(a, budget=100)
    kb = entries.put(b, budget=100)
    assert entries.get(ka) is a  # a is now the most recent
    kc = entries.put(c, budget=100)
    assert entries.get(kb) is None
    assert b.handle is None  # freed on eviction
    assert entries.get(ka) is a
    assert entries.get(kc) is c
    assert entries.nbytes == 80


def test_reading_an_entry_never_frees_it():
    entries = EntryCache()
    e = _entry(10)
    key = entries.put(e, budget=100)
    for _ in range(3):
        assert entries.get(key) is e
    assert not e.handle.freed


def test_zero_budget_keeps_nothing():
    entries = EntryCache()
    handle = FakeHandle(1)
    key = entries.put(Entry(physics="em", handle=handle), budget=0)
    assert entries.get(key) is None
    assert handle.freed
    assert len(entries) == 0


def test_entry_bytes_count_fields_and_handle():
    e = Entry(physics="em", handle=FakeHandle(100), fields={"u": np.zeros(10)})
    assert e.nbytes == 180
    e.free()
    assert e.nbytes == 0


def test_default_budget_is_half_the_machine():
    assert cache.default_retain_memory() > 2**28


# --- a kept forward solve --------------------------------------------------------


def test_kept_electrostatic_solve_holds_a_handle_ready_for_an_adjoint(model):
    problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
    rule = ElectrostaticRule(model, problem, gfem.SolverOptions(order=2))
    outputs, entry = rule.forward(model, keep=True)
    assert entry is not None
    assert entry.physics == "em"
    phi = entry.fields["potential"]
    handle = entry.handle
    # phi already satisfies the free-DOF equations: one more solve on the
    # same factors (what an adjoint does) corrects nothing
    correction = handle.solve(-handle.apply(phi))
    assert np.abs(correction).max() < 1e-12
    # phi^T K phi is eps_r A / d in file units (unit cube, eps_r = 2)
    assert float(phi[:, 0] @ handle.apply(phi[:, 0])) == pytest.approx(2.0, rel=1e-12)
    assert outputs["capacitance"].shape == (1, 1)
    entry.free()


def test_unkept_solve_returns_no_entry(model):
    problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
    rule = ElectrostaticRule(model, problem, gfem.SolverOptions())
    _, entry = rule.forward(model, keep=False)
    assert entry is None
    assert rule.reads == {"eps_r"}
