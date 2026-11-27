"""The lazy gmsh import of the CAD layer."""

import importlib
import sys

import pytest

from gfem import _cad


def test_a_missing_gmsh_says_which_extra_to_install(monkeypatch):
    # A None entry in sys.modules makes `import gmsh` raise ImportError.
    monkeypatch.setitem(sys.modules, "gmsh", None)
    with pytest.raises(ImportError) as info:
        _cad.import_gmsh()
    assert str(info.value) == (
        "gfem.shape.cad_morph needs gmsh, an optional dependency: "
        "pip install 'gfem[gmsh]'"
    )
    assert isinstance(info.value.__cause__, ImportError)


def test_a_gmsh_library_that_fails_to_load_names_the_apt_packages(monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("libGLU.so.1: cannot open")

    monkeypatch.setattr(_cad.importlib, "import_module", broken)
    with pytest.raises(OSError) as info:
        _cad.import_gmsh()
    message = str(info.value)
    assert message == (
        "gmsh is installed but its library failed to load "
        "(libGLU.so.1: cannot open). The PyPI gmsh wheel needs system "
        "OpenGL/X11 libraries; on Debian/Ubuntu: apt-get install libglu1-mesa "
        "libxcursor1 libxft2 libxinerama1 libgl1 libgomp1."
    )
    assert not isinstance(info.value, ImportError)
    assert isinstance(info.value.__cause__, OSError)


@pytest.fixture(scope="module")
def gmsh():
    # Inside a fixture, not at module level: collecting this file must not
    # import gmsh in jobs that have no system libraries for it.
    pytest.importorskip("gmsh")
    return _cad.import_gmsh()


def test_import_gmsh_returns_the_module_and_an_occ_box_meshes(gmsh):
    assert gmsh is importlib.import_module("gmsh")
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("smoke_box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.model.mesh.generate(3)
        tags, _, _ = gmsh.model.mesh.getNodes()
        assert len(tags) >= 8
        types, _, _ = gmsh.model.mesh.getElements(3)
        assert 4 in types  # linear tetrahedra
    finally:
        gmsh.finalize()
