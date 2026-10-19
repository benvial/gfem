"""`SolverOptions`: every numerical knob, frozen, with presets."""

import dataclasses

import pytest

import gfem


def test_defaults():
    o = gfem.SolverOptions()
    assert o.order == 2
    assert o.linear_solver == "auto"
    assert o.tolerance == 1e-6
    assert o.device == "cpu"
    assert o.threads is None
    assert (o.adaptive_tolerance, o.adaptive_max_samples, o.adaptive_memory) == (
        1e-3,
        20,
        2,
    )
    assert o.port_mode_tolerance == 1e-6
    assert o.retain_memory is None  # half of the physical memory, at solve time
    assert o.adjoint_tolerance is None  # the forward tolerance
    assert o.cluster_rtol == 1e-6
    assert o.max_refine_iterations == 50


def test_options_are_frozen_and_keyword_only():
    o = gfem.SolverOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.order = 3  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        gfem.SolverOptions(3)  # ty: ignore[too-many-positional-arguments]
    hash(o)


def test_presets():
    assert gfem.SolverOptions.preset("default") == gfem.SolverOptions()
    fast = gfem.SolverOptions.preset("fast")
    assert fast.order == 1
    accurate = gfem.SolverOptions.preset("accurate")
    assert (accurate.order, accurate.tolerance, accurate.adaptive_tolerance) == (
        3,
        1e-10,
        1e-5,
    )
    with pytest.raises(ValueError, match="'fast'"):
        gfem.SolverOptions.preset("quick")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("kwargs", "word"),
    [
        ({"order": 0}, "order"),
        ({"order": 1.5}, "order"),
        ({"linear_solver": "umfpack"}, "linear_solver"),
        ({"tolerance": 0.0}, "tolerance"),
        ({"tolerance": 1.0}, "tolerance"),
        ({"threads": 0}, "threads"),
        ({"device": "gpu"}, "device"),
        ({"device": "cuda:x", "linear_solver": "direct"}, "device"),
        ({"adaptive_max_samples": 1}, "adaptive_max_samples"),
        ({"adaptive_memory": 0}, "adaptive_memory"),
        ({"retain_memory": -1}, "retain_memory"),
        ({"adjoint_tolerance": -1e-6}, "adjoint_tolerance"),
        ({"cluster_rtol": -1.0}, "cluster_rtol"),
        ({"max_refine_iterations": 0}, "max_refine_iterations"),
        ({"port_mode_tolerance": 0.0}, "port_mode_tolerance"),
    ],
)
def test_invalid_values_name_the_option(kwargs, word):
    with pytest.raises(ValueError, match=word):
        gfem.SolverOptions(**kwargs)


def test_cuda_needs_the_direct_solver():
    assert (
        gfem.SolverOptions(device="cuda:1", linear_solver="direct").device == "cuda:1"
    )
    with pytest.raises(ValueError, match="linear_solver='direct'"):
        gfem.SolverOptions(device="cuda")


def test_adaptive_tolerance_must_sit_above_the_solve_tolerance():
    with pytest.raises(ValueError, match="10"):
        gfem.SolverOptions(tolerance=1e-4, adaptive_tolerance=1e-4)


def test_retain_memory_zero_means_always_recompute():
    assert gfem.SolverOptions(retain_memory=0).retain_memory == 0
