"""Material coefficients as element-wise constant (P0) NGSolve coefficients.

Every material value, from a `Material` (constant per region) or a
`MaterialField` (one value per element), becomes one value per tetrahedron,
so the forward forms and the per-element sensitivity forms read the same
coefficient.
"""

from __future__ import annotations

import ngsolve as ngs
import numpy as np

from gfem._engine.mesh import EngineMesh
from gfem._materials import (
    PROPERTIES,
    Material,
    lossy_permittivity,
    principal_tensor,
)
from gfem.model import Model

__all__ = [
    "coefficient",
    "element_tensors",
    "element_values",
    "lossy_eps",
    "p0",
    "permittivity",
    "surface_p0",
]

_DEFAULTS = {p.name: p.default for p in PROPERTIES}


def element_values(model: Model, name: str) -> np.ndarray:
    """Property ``name`` per tetrahedron (gfem element order), isotropic part.

    A tensor `Material` value gives the mean of its principal values here;
    use `element_tensors` or `lossy_eps` for the full tensor. Tetrahedra
    without a material (excluded volumes) get the property's default.
    """
    mesh = model.mesh
    default = _DEFAULTS[name]
    out = np.full(len(mesh.tetrahedra), np.nan if default is None else default)
    for region, material in model.materials.items():
        value = getattr(material, name)
        if value is None:
            continue
        if isinstance(material, Material):
            value = float(np.mean(value))
        value = np.asarray(value)
        if np.iscomplexobj(value) and not np.iscomplexobj(out):
            out = out.astype(complex)
        out[mesh.elements(region)] = value
    return out


def p0(built: EngineMesh, values: np.ndarray) -> ngs.GridFunction:
    """An element-wise constant field with ``values`` per gfem tetrahedron."""
    is_complex = np.iscomplexobj(values)
    gf = ngs.GridFunction(ngs.L2(built.mesh, order=0, complex=is_complex))
    gf.vec.FV().NumPy()[:] = np.asarray(values)[built.tetrahedron_order]
    return gf


def _tensor(material: Material, name: str = "eps_r") -> np.ndarray:
    return principal_tensor(np.asarray(getattr(material, name), float), material.axes)


def permittivity(model: Model, built: EngineMesh) -> ngs.CoefficientFunction:
    """Relative permittivity, real part, per element: a scalar or 3x3 tensor.

    Only the real part is taken: the electrostatic problem has no loss.
    """
    scalar = element_values(model, "eps_r").real
    anisotropic = {
        r: m
        for r, m in model.materials.items()
        if isinstance(m, Material) and not m.isotropic
    }
    if not anisotropic:
        return p0(built, scalar)
    mesh = model.mesh
    tensor = scalar[:, None, None] * np.eye(3)
    for region, material in anisotropic.items():
        tensor[mesh.elements(region)] = _tensor(material)
    parts = [p0(built, tensor[:, i, j]) for i in range(3) for j in range(3)]
    return ngs.CF(tuple(parts), dims=(3, 3))


def element_tensors(model: Model, name: str) -> np.ndarray:
    """Property ``name`` per tetrahedron, ``(n,)`` or ``(n, 3, 3)`` if anisotropic.

    The ``(n, 3, 3)`` form appears only when some `Material` gives three
    principal values of ``name``; isotropic elements then hold multiples of
    the identity.
    """
    scalar = element_values(model, name)
    anisotropic = {
        r: m
        for r, m in model.materials.items()
        if isinstance(m, Material) and isinstance(getattr(m, name), tuple)
    }
    if not anisotropic:
        return scalar
    mesh = model.mesh
    tensor = scalar[:, None, None] * np.eye(3)
    for region, material in anisotropic.items():
        tensor[mesh.elements(region)] = _tensor(material, name)
    return tensor


def lossy_eps(model: Model) -> np.ndarray:
    """``eps_r (1 - i tan_delta)`` per tetrahedron, ``(n,)`` or ``(n, 3, 3)``.

    A `Material` with per-axis `eps_r` or `tan_delta` gives the tensor of its
    principal values ``eps_r[k] (1 - i tan_delta[k])`` along its axes (the
    ``(n, 3, 3)`` form then holds multiples of the identity elsewhere);
    `MaterialField` values are isotropic.
    """
    scalar = element_values(model, "eps_r") * (
        1.0 - 1j * element_values(model, "tan_delta").real
    )
    anisotropic = {
        r: m
        for r, m in model.materials.items()
        if isinstance(m, Material) and not m.isotropic
    }
    if not anisotropic:
        return scalar
    tensor = scalar[:, None, None] * np.eye(3)
    for region, material in anisotropic.items():
        tensor[model.mesh.elements(region)] = lossy_permittivity(material)
    return tensor


def coefficient(built: EngineMesh, values: np.ndarray) -> ngs.CoefficientFunction:
    """Per-tetrahedron values, ``(n,)`` or ``(n, 3, 3)``, as an NGSolve coefficient."""
    if values.ndim == 1:
        return p0(built, values)
    parts = [p0(built, values[:, i, j]) for i in range(3) for j in range(3)]
    return ngs.CF(tuple(parts), dims=(3, 3))


def surface_p0(built: EngineMesh, values: np.ndarray) -> ngs.GridFunction:
    """A face-wise constant field with ``values`` per gfem triangle."""
    is_complex = np.iscomplexobj(values)
    gf = ngs.GridFunction(ngs.SurfaceL2(built.mesh, order=0, complex=is_complex))
    gf.vec.FV().NumPy()[:] = np.asarray(values)[built.triangle_order]
    return gf
