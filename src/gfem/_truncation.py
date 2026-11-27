"""The truncation layer of a model: the physical elements touching its truncation.

A model radiating into free space is truncated by a perfectly matched layer
(its inner face) or by `gfem.Absorbing` faces. The elements of the solve
domain outside the layer with a vertex on either are the **truncation
layer**. Three readers share it, so that they agree element for element:

- the default side of the far field and of the cross sections (the
  physical elements with no vertex on the truncation), whose outside
  element layer is the truncation layer;
- `gfem.boundary_morph`, which holds every node of the truncation layer, so
  that a measurement read there sees the geometry as declared.

Topology only: the node coordinates are not read.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from gfem import _pml
from gfem._boundaries import Absorbing, surfaces_of

if TYPE_CHECKING:
    from gfem.model import Model

__all__ = ["Truncation", "truncation"]


@dataclasses.dataclass(frozen=True, eq=False)
class Truncation:
    """Where a model is truncated, per node and per tetrahedron.

    Attributes:
        marked: One boolean per node: a vertex of a perfectly matched layer
            element of the solve domain, or of an `gfem.Absorbing` face.
        layer: One boolean per tetrahedron: a physical element of the solve
            domain (outside the perfectly matched layer) with a marked vertex,
            the truncation layer.
        side: One boolean per tetrahedron: a physical element of the solve
            domain with no marked vertex, the default side whose outside
            element layer is the truncation layer.
    """

    marked: np.ndarray
    layer: np.ndarray
    side: np.ndarray


def truncation(model: Model) -> Truncation:
    """The truncation of ``model`` (see `Truncation`)."""
    mesh = model.mesh
    tets = np.asarray(mesh.tetrahedra)
    active = np.asarray(model._domain.active)  # noqa: SLF001
    pml = _pml.layer_tetrahedra(mesh, model.pml)
    pml = np.zeros(len(tets), bool) if pml is None else np.asarray(pml)
    marked = np.zeros(len(mesh.imported_nodes), bool)
    marked[np.unique(tets[pml & active])] = True
    triangles = np.asarray(mesh.triangles)
    for item in model.boundaries:
        if isinstance(item, Absorbing):
            for surface in surfaces_of(item):
                marked[np.unique(triangles[mesh.elements(surface)])] = True
    physical = active & ~pml
    touching = marked[tets].any(axis=1)
    return Truncation(
        marked=marked, layer=physical & touching, side=physical & ~touching
    )
