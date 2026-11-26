"""Bent and PML port mode results written and read back; the section key."""

from __future__ import annotations

import warnings

import bend_cases as bc
import jax
import numpy as np
import pytest

import gfem
from gfem._port_section import port_section_key

OPTIONS = gfem.SolverOptions(order=2, threads=1)


def _write_and_read(model, problem, out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gfem.Job(model, problem, OPTIONS).run(out)
    return result, gfem.load_result(out)


def _section(result):
    return result.e_field._space.section


def _assert_same(result, again, points):
    # the codec writes and reads the same float bits: equality, not a tolerance
    np.testing.assert_array_equal(np.asarray(again.gamma), np.asarray(result.gamma))
    np.testing.assert_array_equal(
        np.asarray(again.e_field.at(points)), np.asarray(result.e_field.at(points))
    )


def test_a_bent_result_is_written_and_read_back(tmp_path):
    model = bc.wr90_model(bc.wr90_mesh(tmp_path / "wr90.msh"))
    bend = bc.wr90_bend(30)
    problem = gfem.PortMode("p1", (bc.WR90_F,), bend=bend)
    result, again = _write_and_read(model, problem, tmp_path / "out")
    assert again.bend.radius == bend.radius
    a, b = _section(again).bend, _section(result).bend
    assert (a.radius, a.outward, a.offset) == (b.radius, b.outward, b.offset)
    np.testing.assert_array_equal(
        np.asarray(again.bend_loss), np.asarray(result.bend_loss)
    )
    points = np.array([[5e-3, 5e-3, 0.0], [15e-3, 5e-3, 0.0]])
    _assert_same(result, again, points)
    # the comma separates the radius from the band that follows
    assert "bend radius 0.03 m, " in repr(result)
    assert "bend radius 0.03 m, " in repr(again)


def test_a_straight_result_has_no_bend_in_its_section(tmp_path):
    model = bc.wr90_model(bc.wr90_mesh(tmp_path / "wr90.msh"))
    problem = gfem.PortMode("p1", (bc.WR90_F,))
    result, again = _write_and_read(model, problem, tmp_path / "out")
    assert _section(again).bend is None
    assert '"bend"' not in (tmp_path / "out" / "result.json").read_text()
    points = np.array([[5e-3, 5e-3, 0.0]])
    _assert_same(result, again, points)


@pytest.mark.parametrize("bent", [False, True])
def test_a_section_with_a_pml_is_written_and_read_back(tmp_path, bent):
    model = bc.slab_model(bc.slab_mesh(tmp_path / "slab.msh", "Ey", 0.4))
    problem = gfem.PortMode("p1", (bc.SLAB_F,), bend=bc.slab_bend(5) if bent else None)
    result, again = _write_and_read(model, problem, tmp_path / "out")
    stretch = _section(again).stretch
    assert stretch is not None
    assert stretch.key == _section(result).stretch.key
    points = np.array([[0.0, 0.05, 0.0], [1.0, 0.05, 0.0]])
    _assert_same(result, again, points)


def test_section_keys_of_radii_one_part_in_1e9_apart_differ(tmp_path):
    model = bc.wr90_model(bc.wr90_mesh(tmp_path / "wr90.msh"))
    port = model.ports[0]
    r = 0.03
    # a relative 1e-9 is 3e-11 m: the repr's 6 digits print the same radius
    near = bc.wr90_bend_m(r * (1 + 1e-9))
    assert repr(bc.wr90_bend_m(r)) == repr(near)
    keys = {
        port_section_key(model, port, None, b) for b in (None, bc.wr90_bend_m(r), near)
    }
    assert len(keys) == 3


def test_a_traced_bend_radius_has_no_section_key(tmp_path):
    model = bc.wr90_model(bc.wr90_mesh(tmp_path / "wr90.msh"))

    def key(r):
        return port_section_key(model, model.ports[0], None, bc.wr90_bend_m(r))

    with pytest.raises(TypeError, match="traced"):
        jax.jit(key)(0.03)
