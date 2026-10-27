"""`SolverOptions`: every numerical knob, frozen, with presets."""

import dataclasses
import pickle

import numpy as np
import pytest
from petsc_recipe import SOLVER_OPTIONS

import gfem
from gfem.options import split_petsc_options


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
    # Palace's backward error (Eigenmode.Tol), tight enough to bound f and Q
    assert o.eigen_tolerance == 1e-10
    assert o.cluster_rtol == 1e-6
    assert o.max_refine_iterations == 50
    assert o.frequency_workers == 1  # one process: the sweep as it always ran
    assert o.eigen_solver == "auto"  # ARPACK: "auto" never picks SLEPc
    assert o.petsc_options == ()


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


def _petsc(options):
    return {"linear_solver": "petsc", "petsc_options": options}


def _slepc(options):
    return {"eigen_solver": "slepc", "petsc_options": options}


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
        ({"eigen_tolerance": 0.0}, "eigen_tolerance"),
        ({"max_refine_iterations": 0}, "max_refine_iterations"),
        ({"port_mode_tolerance": 0.0}, "port_mode_tolerance"),
        ({"frequency_workers": 0}, "frequency_workers"),
        ({"frequency_workers": 2.0}, "frequency_workers"),
        ({"eigen_solver": "lapack"}, "eigen_solver"),
        (_petsc({"-ksp_type": "gmres"}), "petsc_options"),
        (_petsc({"ksp_type": ["gmres"]}), "petsc_options"),
        (_petsc({"ksp_atol": float("nan")}), "petsc_options"),
        (_petsc({"ksp_rtol": 1e-8}), "tolerance"),
        (_petsc({"ksp_initial_guess_nonzero": True}), "initial guess"),
        ({"petsc_options": {"ksp_type": "gmres"}}, "petsc_options"),
        ({"device": "cuda", "linear_solver": "petsc"}, "linear_solver"),
        (_slepc({"eps_largest_real": None}), "target"),
        (_slepc({"eps_gen_hermitian": None}), "non-Hermitian"),
        (_slepc({"eps_conv_abs": None}), "eigen_tolerance"),
        (_slepc({"st_ksp_type": "gmres"}), "linear_solver"),
        (_slepc({"eps_type": "lapack"}), "eps_type"),
        (_slepc({"ksp_type": "gmres"}), "linear_solver='petsc'"),
        (_petsc({"eps_ncv": 30}), "eigen_solver='slepc'"),
        (_slepc({"eps_ncv": "x"}), r"\['eps_ncv'\] must be an integer >= 1"),
        (_slepc({"eps_ncv": "30"}), r"\['eps_ncv'\] must be an integer >= 1"),
        (_slepc({"eps_ncv": 30.0}), r"\['eps_ncv'\] must be an integer >= 1"),
        (_slepc({"eps_ncv": 0}), r"\['eps_ncv'\] must be an integer >= 1"),
        (_slepc({"eps_ncv": True}), r"\['eps_ncv'\] must be an integer >= 1"),
        (_slepc({"eps_ncv": None}), r"\['eps_ncv'\] must be an integer >= 1"),
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


def test_frequency_workers_stay_off_a_gpu():
    # k processes would share one GPU's memory
    with pytest.raises(ValueError, match="frequency_workers=1"):
        gfem.SolverOptions(device="cuda", linear_solver="direct", frequency_workers=2)
    assert gfem.SolverOptions(frequency_workers=2).frequency_workers == 2


def test_adaptive_tolerance_must_sit_above_the_solve_tolerance():
    with pytest.raises(ValueError, match="10"):
        gfem.SolverOptions(tolerance=1e-4, adaptive_tolerance=1e-4)


def test_retain_memory_zero_means_always_recompute():
    assert gfem.SolverOptions(retain_memory=0).retain_memory == 0


def test_petsc_options_are_sorted_pairs():
    a = gfem.SolverOptions(linear_solver="petsc", petsc_options={"a": 1, "b": None})
    b = gfem.SolverOptions(linear_solver="petsc", petsc_options={"b": None, "a": 1})
    assert a == b
    assert hash(a) == hash(b)
    assert a.petsc_options == (("a", 1), ("b", None))
    assert dataclasses.replace(a, order=1).petsc_options == a.petsc_options
    assert pickle.loads(pickle.dumps(a)) == a  # noqa: S301 - our own bytes


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["ksp_type", "gmres"], "must map PETSc option names to values"),
        ((("ksp_type", "gmres"), ("ksp_type", "cg")), "names 'ksp_type' twice"),
        ({"ksp_type": np.float64(1.0)}, "got float64"),
        ({"ksp-type": "gmres"}, "without the leading '-'"),
    ],
)
def test_petsc_options_are_checked_strictly(options, message):
    with pytest.raises(ValueError, match=message):
        gfem.SolverOptions(linear_solver="petsc", petsc_options=options)


def test_petsc_options_split_by_namespace():
    o = gfem.SolverOptions(
        linear_solver="petsc",
        eigen_solver="slepc",
        petsc_options={"ksp_type": "gmres", "eps_ncv": 24, "bv_type": "mat"},
    )
    assert split_petsc_options(o.petsc_options) == (
        (("ksp_type", "gmres"),),
        (("bv_type", "mat"), ("eps_ncv", 24)),
    )


def test_slepc_takes_the_krylov_schur_or_arnoldi_eps_type():
    for eps_type in ("krylovschur", "arnoldi"):
        o = gfem.SolverOptions(
            eigen_solver="slepc", petsc_options={"eps_type": eps_type}
        )
        assert o.petsc_options == (("eps_type", eps_type),)


@pytest.mark.parametrize("snippet", list(SOLVER_OPTIONS))
def test_the_petsc_recipes_options_are_accepted(snippet):
    # every SolverOptions that docs/recipes/petsc-slepc.md builds passes the
    # namespace and reserved-option rules, without PETSc installed
    kwargs = SOLVER_OPTIONS[snippet]
    options = gfem.SolverOptions(**kwargs)
    assert dict(options.petsc_options) == kwargs.get("petsc_options", {})
