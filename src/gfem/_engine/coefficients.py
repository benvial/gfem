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
    MaterialField,
    lossy_permittivity,
    principal_tensor,
)
from gfem.model import Model

__all__ = [
    "anisotropic_elements",
    "coefficient",
    "element_tensors",
    "element_values",
    "lossy_eps",
    "p0",
    "permittivity",
    "surface_coefficient",
    "surface_p0",
]

_DEFAULTS = {p.name: p.default for p in PROPERTIES}


def element_values(
    model: Model, name: str, *, per_axis: float | None = None
) -> np.ndarray:
    """Property ``name`` per tetrahedron (gfem element order), isotropic part.

    A tensor `Material` value gives the mean of its principal values here;
    use `element_tensors` or `lossy_eps` for the full tensor. Tetrahedra
    without a material (excluded volumes) get the property's default.

    A `MaterialField` with `axes` holds ``name`` per principal axis, which
    has no one value: its tetrahedra get ``per_axis``, a placeholder for a
    caller that reads them from the tensors, and raise when it is ``None``.

    Raises:
        ValueError: A field holds ``name`` per principal axis and
            ``per_axis`` is ``None``.
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
        elif material.per_axis(name):
            if per_axis is None:
                raise ValueError(
                    f"The MaterialField of region {region!r} holds {name} per "
                    "principal axis; it has no one value per element (read "
                    "element_tensors)."
                )
            value = per_axis
        value = np.asarray(value)
        if np.iscomplexobj(value) and not np.iscomplexobj(out):
            out = out.astype(complex)
        out[mesh.elements(region)] = value
    return out


def _per_axis_fields(model: Model) -> dict[str, MaterialField]:
    """The `MaterialField` regions with `axes`."""
    return {
        r: m
        for r, m in model.materials.items()
        if isinstance(m, MaterialField) and m.axes is not None
    }


def _principal_values(field: MaterialField, name: str) -> np.ndarray:
    """A field's principal values of ``name``, ``(n, 3)`` or ``(3,)``."""
    value = getattr(field, name)
    if value is None:
        return np.full(3, _DEFAULTS[name])
    return np.asarray(value)


def _field_tensors(field: MaterialField, values: np.ndarray) -> np.ndarray:
    """``sum_k values[..., k] a_k a_k^T`` on the field's axes, ``(..., 3, 3)``."""
    a = np.asarray(field.axes, float)
    return np.einsum("...k,ki,kj->...ij", values, a, a)


def _with_tensors(
    scalar: np.ndarray,
    model: Model,
    materials: dict[str, np.ndarray],
    fields: dict[str, np.ndarray],
) -> np.ndarray:
    """``scalar`` as ``(n, 3, 3)`` with the regions' tensors put in.

    ``materials`` holds one ``(3, 3)`` tensor per region, ``fields`` one
    ``(3, 3)`` or ``(n_region, 3, 3)``; their regions' ``scalar`` values
    are placeholders.
    """
    mesh = model.mesh
    dtype = np.result_type(scalar, *materials.values(), *fields.values())
    tensor = (scalar[:, None, None] * np.eye(3)).astype(dtype)
    for region, value in (materials | fields).items():
        tensor[mesh.elements(region)] = value
    return tensor


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
    scalar = element_values(model, "eps_r", per_axis=0.0).real
    anisotropic = {
        r: _tensor(m)
        for r, m in model.materials.items()
        if isinstance(m, Material) and not m.isotropic
    }
    fields = {
        r: _field_tensors(f, _principal_values(f, "eps_r").real)
        for r, f in _per_axis_fields(model).items()
    }
    if not anisotropic and not fields:
        return p0(built, scalar)
    tensor = _with_tensors(scalar, model, anisotropic, fields)
    parts = [p0(built, tensor[:, i, j]) for i in range(3) for j in range(3)]
    return ngs.CF(tuple(parts), dims=(3, 3))


def element_tensors(model: Model, name: str) -> np.ndarray:
    """Property ``name`` per tetrahedron, ``(n,)`` or ``(n, 3, 3)`` if anisotropic.

    The ``(n, 3, 3)`` form appears only when some `Material` gives three
    principal values of ``name``, or some `MaterialField` has `axes`;
    isotropic elements then hold multiples of the identity.
    """
    scalar = element_values(model, name, per_axis=0.0)
    anisotropic = {
        r: _tensor(m, name)
        for r, m in model.materials.items()
        if isinstance(m, Material) and isinstance(getattr(m, name), tuple)
    }
    fields = {
        r: _field_tensors(f, _principal_values(f, name))
        for r, f in _per_axis_fields(model).items()
    }
    if not anisotropic and not fields:
        return scalar
    return _with_tensors(scalar, model, anisotropic, fields)


def anisotropic_elements(model: Model) -> np.ndarray:
    """Whether each tetrahedron has an anisotropic real ``eps_r`` or ``mu_r``.

    Structural: the tetrahedra of a `Material` with three principal values
    of ``eps_r`` or ``mu_r``, and of a `MaterialField` with `axes`, whatever
    the values (so an optimizer moving a field's values never changes it).
    A per-axis ``tan_delta`` of a `Material` alone does not count (it moves
    only the imaginary part).
    """
    out = np.zeros(len(model.mesh.tetrahedra), bool)
    for region, material in model.materials.items():
        if isinstance(material, MaterialField):
            tensor = material.axes is not None
        else:
            tensor = isinstance(material.eps_r, tuple) or isinstance(
                material.mu_r, tuple
            )
        if tensor:
            out[model.mesh.elements(region)] = True
    return out


def lossy_eps(model: Model) -> np.ndarray:
    """``eps_r (1 - i tan_delta)`` per tetrahedron, ``(n,)`` or ``(n, 3, 3)``.

    A `Material` with per-axis `eps_r` or `tan_delta`, and a `MaterialField`
    with `axes`, give the tensor of their principal values ``eps_r[k] (1 -
    i tan_delta[k])`` along their axes (the ``(n, 3, 3)`` form then holds
    multiples of the identity elsewhere).
    """
    scalar = element_values(model, "eps_r", per_axis=0.0) * (
        1.0 - 1j * element_values(model, "tan_delta", per_axis=0.0).real
    )
    anisotropic = {
        r: lossy_permittivity(m)
        for r, m in model.materials.items()
        if isinstance(m, Material) and not m.isotropic
    }
    fields = {}
    for region, f in _per_axis_fields(model).items():
        eps = _principal_values(f, "eps_r")
        tan = _principal_values(f, "tan_delta").real
        fields[region] = _field_tensors(f, eps * (1 - 1j * tan))
    if not anisotropic and not fields:
        return scalar
    return _with_tensors(scalar, model, anisotropic, fields)


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


def surface_coefficient(
    built: EngineMesh, values: np.ndarray
) -> ngs.CoefficientFunction:
    """Per-triangle values, ``(n,)`` or ``(n, 3, 3)``, as an NGSolve coefficient."""
    if values.ndim == 1:
        return surface_p0(built, values)
    parts = [surface_p0(built, values[:, i, j]) for i in range(3) for j in range(3)]
    return ngs.CF(tuple(parts), dims=(3, 3))
