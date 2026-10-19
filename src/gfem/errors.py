"""gfem's exception and warning classes."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


class RegionNotFoundError(LookupError):
    """A region name that the mesh does not define.

    Regions are gmsh physical groups, addressed by their name. The message
    suggests the closest names the mesh does define.
    """

    def __init__(self, name: str, known: Iterable[str], kind: str = "region") -> None:
        """Build the message from the missing name and the known ones.

        Args:
            name: The name that was asked for.
            known: The names the mesh defines.
            kind: What was looked up ("region", "volume" or "surface").
        """
        known = sorted(known)
        close = difflib.get_close_matches(name, known, n=3)
        hint = (
            f" Did you mean {', '.join(repr(c) for c in close)}?"
            if close
            else f" Known {kind}s: {', '.join(repr(k) for k in known)}."
        )
        super().__init__(
            f"No {kind} named {name!r} in the mesh (gmsh physical-group names).{hint}"
        )
        self.name = name


class NonDifferentiableError(TypeError):
    """A gradient was asked of something gfem does not differentiate.

    The message names the quantity, the reason and the remedy. To use such a
    value in an objective without its derivative, wrap it in
    `jax.lax.stop_gradient`.
    """


class PortGeometryError(ValueError):
    """A port whose geometry the solver cannot use.

    For example a wave port surface that is not planar. The message names the
    port and the remedy.
    """


class PortModeError(RuntimeError):
    """The port mode problem has no usable mode.

    For example the requested mode is below cutoff (evanescent) at some
    frequency of the band. The message names the port, the frequency and the
    remedy.
    """
