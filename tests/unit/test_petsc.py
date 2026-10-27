"""The missing-solver contract: a PETSc or SLEPc solve raises, never falls back.

``linear_solver="petsc"`` and ``eigen_solver="slepc"`` need complex
petsc4py / slepc4py, which gfem does not install. These tests block those
modules or replace them by fakes, so they hold on any machine, with or
without PETSc.
"""

import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem import _linalg, _petsc
from gfem._eigen import rule as eigen_rule
from gfem._engine import driven, eigen, electrostatic, quadrature, workers
from gfem._engine.mesh import to_ngsolve

F1 = 7.49e9  # the line's first half-wave resonance, Hz


def _clear():
    _petsc.require_petsc.cache_clear()
    _petsc.require_slepc.cache_clear()


def fake(name, *, version="3.26.0", scalar=np.complex128, error=None):
    """A petsc4py or slepc4py module that records its ``init`` calls."""
    module = types.ModuleType(name)
    module.__version__ = version
    module.calls = []

    def init(args):
        if error is not None:
            raise error
        module.calls.append(args)

    module.init = init
    attribute = {"petsc4py": "PETSc", "slepc4py": "SLEPc"}[name]
    library = types.ModuleType(f"{name}.{attribute}")
    library.ScalarType = scalar
    setattr(module, attribute, library)
    return module


@pytest.fixture
def stack(monkeypatch):
    """Set what ``import petsc4py`` / ``slepc4py`` give: a module, or None to block."""

    def set_(petsc=None, slepc=None):
        monkeypatch.setitem(sys.modules, "petsc4py", petsc)
        monkeypatch.setitem(sys.modules, "slepc4py", slepc)
        _clear()

    set_()
    yield set_
    _clear()


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"box": gfem.Material(eps_r=1.0)}
    )


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _case(name, plate, line):
    """The model and problem of a problem type."""
    if name == "electrostatic":
        return plate, gfem.Electrostatic(terminals={"t": "top"}, ground="bottom")
    if name == "driven":
        return line, gfem.Driven((3e9, 4e9))
    return line, gfem.Eigenmode(target=1.6 * F1, n_modes=1)


PROBLEMS = ("electrostatic", "driven", "eigenmode")


def test_solver_unavailable_error_is_a_runtime_error():
    assert issubclass(gfem.SolverUnavailableError, RuntimeError)
    assert gfem.SolverUnavailableError is gfem.errors.SolverUnavailableError


@pytest.mark.usefixtures("stack")
@pytest.mark.parametrize("problem", PROBLEMS)
def test_a_petsc_solve_without_petsc4py_names_it_and_the_recipe(plate, line, problem):
    model, p = _case(problem, plate, line)
    options = gfem.SolverOptions(order=1, linear_solver="petsc")
    with pytest.raises(gfem.SolverUnavailableError) as err:
        gfem.solve(model, p, options=options)
    message = str(err.value)
    assert message.startswith("linear_solver='petsc' cannot run: petsc4py is not ")
    assert "'PETSc and SLEPc' recipe" in message
    assert "linear_solver='direct'" in message


def _eigen(line, **options):
    problem = gfem.Eigenmode(target=1.6 * F1, n_modes=1)
    gfem.solve(line, problem, options=gfem.SolverOptions(order=1, **options))


def test_a_slepc_solve_without_slepc4py_names_it(stack, line):
    stack(petsc=fake("petsc4py"))
    with pytest.raises(gfem.SolverUnavailableError) as err:
        _eigen(line, eigen_solver="slepc")
    message = str(err.value)
    assert message.startswith("eigen_solver='slepc' cannot run: slepc4py is not ")
    assert "Build SLEPc and slepc4py on a complex PETSc" in message
    assert "eigen_solver='arpack'" in message


@pytest.mark.usefixtures("stack")
def test_a_slepc_solve_without_petsc4py_names_petsc4py(line):
    with pytest.raises(gfem.SolverUnavailableError) as err:
        _eigen(line, eigen_solver="slepc")
    message = str(err.value)
    assert message.startswith("eigen_solver='slepc' cannot run: petsc4py is not ")
    assert "eigen_solver='arpack'" in message


def _electrostatic(plate):
    problem = gfem.Electrostatic(terminals={"t": "top"}, ground="bottom")
    options = gfem.SolverOptions(order=1, linear_solver="petsc")
    gfem.solve(plate, problem, options=options)


def test_a_real_scalar_petsc_is_refused(stack, plate):
    stack(petsc=fake("petsc4py", scalar=np.float64))
    match = r"real scalars \(PETSc.ScalarType is float64\).*scalar-type=complex"
    with pytest.raises(gfem.SolverUnavailableError, match=match):
        _electrostatic(plate)


@pytest.mark.parametrize("package", ["petsc4py", "slepc4py"])
def test_a_version_below_the_floor_is_refused(stack, line, package):
    old = {package: fake(package, version="3.25.1")}
    stack(petsc=old.get("petsc4py", fake("petsc4py")), slepc=old.get("slepc4py"))
    match = rf"{package} 3\.25\.1 is older than gfem supports \(3\.26\)"
    with pytest.raises(gfem.SolverUnavailableError, match=match):
        _eigen(line, eigen_solver="slepc")


def test_a_broken_import_is_reported_with_its_text(stack, plate):
    error = ImportError("libmumps.so.5.9: cannot open shared object file")
    stack(petsc=fake("petsc4py", error=error))
    match = "petsc4py cannot be imported: ImportError: libmumps.so.5.9: cannot open"
    with pytest.raises(gfem.SolverUnavailableError, match=match):
        _electrostatic(plate)


def _spy(monkeypatch, owner, name, calls):
    real = getattr(owner, name)

    def spy(*args, **kwargs):
        calls.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)


@pytest.mark.usefixtures("stack")
@pytest.mark.parametrize("problem", PROBLEMS)
def test_the_solver_is_checked_before_any_assembly(plate, line, problem, monkeypatch):
    calls = []
    for module in (_linalg, driven, electrostatic, eigen):
        _spy(monkeypatch, module, "prepare", calls)
    _spy(monkeypatch, driven.DrivenRule, "_ports", calls)
    model, p = _case(problem, plate, line)
    options = gfem.SolverOptions(order=1, linear_solver="petsc")
    with pytest.raises(gfem.SolverUnavailableError):
        gfem.solve(model, p, options=options)
    assert calls == []


@pytest.mark.usefixtures("stack")
def test_frequency_workers_fail_in_the_parent(line, monkeypatch):
    calls = []
    _spy(monkeypatch, workers, "run", calls)
    options = gfem.SolverOptions(order=1, linear_solver="petsc", frequency_workers=2)
    with pytest.raises(gfem.SolverUnavailableError, match="petsc4py"):
        gfem.solve(line, gfem.Driven((3e9, 4e9)), options=options)
    assert calls == []


@pytest.mark.usefixtures("stack")
def test_prepare_checks_petsc_again_before_assembly(plate):
    import ngsolve as ngs

    built = to_ngsolve(plate.mesh)
    space = ngs.H1(built.mesh, order=1, dirichlet="bottom|top")
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += ngs.grad(u) * ngs.grad(v) * quadrature.dx(1)
    op = _linalg.Operator(form, space.FreeDofs())
    with pytest.raises(gfem.SolverUnavailableError, match="linear_solver='petsc'"):
        _linalg.prepare(op, symmetry="hermitian", kind="petsc")
    with pytest.raises(TypeError, match="assemble"):
        _ = op.form.mat  # never assembled


def test_petsc_is_initialised_once_without_arguments(stack):
    petsc, slepc = fake("petsc4py"), fake("slepc4py")
    stack(petsc=petsc, slepc=slepc)
    assert _petsc.require_petsc() is petsc.PETSc
    assert _petsc.require_slepc() is slepc.SLEPc
    inits = len(petsc.calls), len(slepc.calls)
    assert _petsc.require_petsc() is petsc.PETSc
    assert _petsc.require_slepc() is slepc.SLEPc
    assert (len(petsc.calls), len(slepc.calls)) == inits  # probed once
    # never the script's sys.argv: PETSc would read the user's options
    assert all(args == [] for args in petsc.calls + slepc.calls)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_auto_never_picks_petsc_or_slepc_even_when_installed(
    stack, plate, line, problem, monkeypatch
):
    petsc, slepc = fake("petsc4py"), fake("slepc4py")
    stack(petsc=petsc, slepc=slepc)
    seeds = []
    _spy(monkeypatch, eigen_rule, "shift_invert", seeds)
    model, p = _case(problem, plate, line)
    gfem.solve(model, p, options=gfem.SolverOptions(order=1))
    assert petsc.calls == slepc.calls == []
    assert bool(seeds) == (problem == "eigenmode")  # the ARPACK seed solve


# --- the build script users and CI run: tools/petsc_wheelhouse.sh -------------------

WHEELHOUSE_SCRIPT = Path(__file__).parents[2] / "tools" / "petsc_wheelhouse.sh"


def _script_lines():
    """The script's lines that are not comments."""
    lines = WHEELHOUSE_SCRIPT.read_text().splitlines()
    return [line for line in lines if not line.lstrip().startswith("#")]


def _pin(name):
    (version,) = [
        m.group(1)
        for line in _script_lines()
        if (m := re.fullmatch(rf"{name}=(\d+\.\d+\.\d+)", line.strip()))
    ]
    return tuple(int(part) for part in version.split("."))


def test_the_version_floor_is_the_minor_the_build_script_pins():
    # bumping the script to a new PETSc minor raises the floor in the same commit
    assert _pin("PETSC_VERSION")[:2] == _pin("SLEPC_VERSION")[:2] == _petsc.MIN_VERSION


def test_the_build_script_downloads_only_mumps_metis_and_openblas():
    # The licence control (no non-free or non-commercial dependency): the
    # stack holds what these downloads build and nothing else, the set the
    # licence page names. No ParMETIS, PT-Scotch or ScaLAPACK, which an MPI
    # build would pull in.
    code = "\n".join(_script_lines())
    options = re.findall(r"--download-([\w-]+)", code)
    assert {option.split("-")[0] for option in options} == {
        "mumps",
        "metis",
        "openblas",
    }
    assert "--with-mpi=0" in code


# --- where PETSc and SLEPc are installed (skipped elsewhere, by the markers) ------


@pytest.mark.petsc
def test_the_installed_petsc_is_complex_and_recent():
    PETSc = _petsc.require_petsc()
    assert np.dtype(PETSc.ScalarType).kind == "c"
    assert PETSc.Sys.getVersion()[:2] >= _petsc.MIN_VERSION


@pytest.mark.slepc
def test_the_installed_slepc_is_recent():
    SLEPc = _petsc.require_slepc()
    assert SLEPc.Sys.getVersion()[:2] >= _petsc.MIN_VERSION


def test_unwrap_gives_the_exception_a_petsc_callback_raised():
    original = KeyboardInterrupt()
    err = SystemError("<cyfunction EPS.solve> returned a result with an exception set")
    err.__cause__ = original
    assert _petsc.unwrap(err) is original
    plain = ValueError("x")
    assert _petsc.unwrap(plain) is plain


@pytest.mark.petsc
def test_each_petsc_object_gets_its_own_prefix():
    a, b = _petsc.next_prefix(), _petsc.next_prefix()
    assert a != b
    assert re.fullmatch(r"gfem\d+_", a)


def _gfem_keys(PETSc):
    return [k for k in PETSc.Options().getAll() if k.startswith("gfem")]


@pytest.mark.petsc
def test_scoped_options_reach_the_prefixed_object_only_inside_the_scope():
    PETSc = _petsc.require_petsc()
    prefix = _petsc.next_prefix()
    pairs = (
        ("ksp_atol", 1e-12),
        ("ksp_initial_guess_knoll", False),
        ("ksp_max_it", 7),
        ("ksp_type", "gmres"),
        ("ksp_view", None),
    )
    PETSc.Options().setValue("ksp_type", "cg")  # a user's own, unprefixed
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    try:
        ksp.setOptionsPrefix(prefix)
        with _petsc.scoped_options(prefix, pairs):
            ksp.setFromOptions()
            options = PETSc.Options()
            assert options.getAll()[f"{prefix}ksp_initial_guess_knoll"] == "false"
            assert options.hasName(f"{prefix}ksp_view")
        assert ksp.getType() == "gmres"
        _, atol, _, max_it = ksp.getTolerances()
        assert (atol, max_it) == (1e-12, 7)
        assert ksp.getInitialGuessKnoll() is False
        assert _gfem_keys(PETSc) == []
    finally:
        ksp.destroy()
        PETSc.Options().delValue("ksp_type")


@pytest.mark.petsc
def test_scoped_options_are_removed_when_the_scope_raises():
    PETSc = _petsc.require_petsc()
    prefix = _petsc.next_prefix()
    inside = []

    def fail():
        with _petsc.scoped_options(prefix, (("a", 1),)):
            inside.extend(_gfem_keys(PETSc))
            raise RuntimeError

    with pytest.raises(RuntimeError):
        fail()
    assert inside == [f"{prefix}a"]
    assert _gfem_keys(PETSc) == []


@pytest.mark.petsc
def test_error_text_keeps_petscs_reason_without_its_traceback():
    PETSc = _petsc.require_petsc()
    prefix = _petsc.next_prefix()
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    try:
        ksp.setOptionsPrefix(prefix)
        with (
            pytest.raises(PETSc.Error) as err,
            _petsc.scoped_options(prefix, (("ksp_type", "gmress"),)),
        ):
            ksp.setFromOptions()
    finally:
        ksp.destroy()
    text = _petsc.error_text(err.value)
    assert "Unable to find requested KSP type gmress" in text
    assert text.endswith("(PETSc error 86)")
    assert "() at " not in text
    assert "[0]" not in text
