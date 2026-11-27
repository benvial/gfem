"""The one place gfem imports gmsh.

gmsh is the optional extra ``gfem[gmsh]``: importing gfem never needs it, and
this module imports it only when a CAD feature asks, so a missing package or a
broken system library fails with the remedy instead of at ``import gfem``.
"""

import importlib
from types import ModuleType

#: Debian/Ubuntu packages the PyPI gmsh wheel loads at import (OpenGL and X11).
_APT_PACKAGES = "libglu1-mesa libxcursor1 libxft2 libxinerama1 libgl1 libgomp1"


def import_gmsh() -> ModuleType:
    """Import and return gmsh, with an actionable error when it cannot load.

    A missing package raises :class:`ImportError` (a test may skip on it); a
    package whose shared library fails to load raises :class:`OSError`, a
    different type so a runner without the system libraries fails loudly
    instead of skipping.
    """
    try:
        return importlib.import_module("gmsh")
    except ImportError as exc:
        raise ImportError(
            "gfem.shape.cad_morph needs gmsh, an optional dependency: "
            "pip install 'gfem[gmsh]'"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"gmsh is installed but its library failed to load ({exc}). "
            "The PyPI gmsh wheel needs system OpenGL/X11 libraries; on "
            f"Debian/Ubuntu: apt-get install {_APT_PACKAGES}."
        ) from exc
