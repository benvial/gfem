import ast
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import gfem

#: The only packages that may import NGSolve.
ENGINE_PACKAGES = ("_engine", "_linalg")


def _run_fresh(code: str, jax_enable_x64: str | None = None) -> str:
    """Run `code` in a fresh interpreter, so JAX config set by other tests is absent."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    if jax_enable_x64 is not None:
        env["JAX_ENABLE_X64"] = jax_enable_x64
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("jax_enable_x64", [None, "0", "false"])
def test_importing_gfem_enables_jax_double_precision(jax_enable_x64):
    out = _run_fresh(
        "import gfem, jax.numpy as jnp;"
        "print(jnp.asarray(1.0).dtype, jnp.asarray(1j).dtype)",
        jax_enable_x64,
    )
    assert out == "float64 complex128"


def test_double_precision_holds_when_jax_was_imported_first():
    out = _run_fresh(
        "import jax, jax.numpy as jnp; jnp.zeros(1);"
        "import gfem; print(jnp.asarray(1.0).dtype)"
    )
    assert out == "float64"


def test_importing_gfem_never_imports_petsc():
    out = _run_fresh(
        "import gfem, sys; print([m for m in ('petsc4py', 'slepc4py', 'mpi4py') "
        "if m in sys.modules])"
    )
    assert out == "[]"


def test_importing_gfem_never_imports_sax():
    # scikit-rf too: the S-parameter export imports it only when called.
    out = _run_fresh(
        "import gfem, sys; print([m for m in ('sax', 'klujax', 'skrf') "
        "if m in sys.modules])"
    )
    assert out == "[]"


def test_a_one_process_mesh_never_imports_mpi4py():
    path = Path(__file__).parent / "data" / "two_box_msh22.msh"
    out = _run_fresh(
        "import gfem, sys; from gfem._engine.cache import discretization;"
        f"discretization(gfem.Mesh.from_gmsh({str(path)!r}));"
        "print('mpi4py' in sys.modules)"
    )
    assert out == "False"


def test_petsc_is_imported_only_by_gfem_petsc():
    root = Path(gfem.__file__).parent
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if p.name != "_petsc.py" and {"petsc4py", "slepc4py"} & _imports(p)
    ]
    assert offenders == []


def test_mpi4py_is_imported_only_by_gfem_mpi():
    root = Path(gfem.__file__).parent
    assert "mpi4py" in _imports(root / "_mpi.py")
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if p.name != "_mpi.py" and "mpi4py" in _imports(p)
    ]
    assert offenders == []


def test_version_comes_from_installed_metadata():
    import gfem

    assert gfem.__version__ == version("gfem")


def _imports(path: Path) -> set[str]:
    """The top-level package of every import statement in ``path``."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module.split(".")[0])
    return out


def _dynamic_imports(path: Path) -> set[str]:
    """The top-level package of every literal ``import_module`` or ``__import__``."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = node.args[0]
        if (
            name in {"import_module", "__import__"}
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
        ):
            out.add(first.value.split(".")[0])
    return out


def test_sax_is_never_imported_by_gfem():
    # A SAX component returns a plain dictionary: gfem needs no SAX, not even
    # for type hints, so a user without it can still call one.
    root = Path(gfem.__file__).parent
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if {"sax", "klujax"} & (_imports(p) | _dynamic_imports(p))
    ]
    assert offenders == []


def test_ngsolve_is_imported_only_by_the_engine():
    root = Path(gfem.__file__).parent
    modules = [
        p
        for p in root.rglob("*.py")
        if p.relative_to(root).parts[0] not in ENGINE_PACKAGES
    ]
    assert any(p.parent.name == "_eigen" for p in modules)
    offenders = [str(p.relative_to(root)) for p in modules if "ngsolve" in _imports(p)]
    assert offenders == []


def test_gmsh_is_imported_only_by_the_cad_module():
    # gmsh is an optional extra: one lazy import site, so a missing or broken
    # package gets one error message and `import gfem` never needs it.
    root = Path(gfem.__file__).parent
    cad = root / "_cad.py"
    assert cad.exists()
    assert "gmsh" in _dynamic_imports(cad)
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if p != cad and "gmsh" in (_imports(p) | _dynamic_imports(p))
    ]
    assert offenders == []
    assert "gmsh" not in _imports(cad)


def test_importing_gfem_and_its_shape_module_leaves_gmsh_out():
    out = _run_fresh("import sys, gfem, gfem.shape; print('gmsh' in sys.modules)")
    assert out == "False"
