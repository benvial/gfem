"""Material coefficients as element-wise constant (P0) NGSolve coefficients.

Every material value, from a `Material` (constant per region) or a
`MaterialField` (one value per element), becomes one value per tetrahedron,
so the forward forms and the per-element sensitivity forms read the same
coefficient.

"Anisotropic" has two structural meanings here, both read from which
properties are tensors, never from values:

- `anisotropic_elements`: the real wave tensors ``eps_r`` or ``mu_r`` are
  tensors. The boundary paths (the absorbing admittance, a wave port's
  inverse-permeability tensor) branch on it, since only those two shape a
  wave's admittance; a loss-only tensor (``tan_delta`` or
  ``conductivity``) keeps the scalar boundary coefficient.
- `anisotropic_regions`: any tensor property, loss-only ones included. The
  principal participation splits these regions, since a loss tensor alone
  needs the energy along each principal axis.

A `MaterialField` with `axes` is anisotropic in both senses
(`per_axis_elements` marks its tetrahedra).
"""

from __future__ import annotations

import ngsolve as ngs
import numpy as np

from gfem._engine.mesh import EngineMesh
from gfem._materials import (
    PROPERTIES,
    Material,
    MaterialField,
    field_principal_values,
    lossy_permittivity,
    principal_tensor,
)
from gfem.model import Model

__all__ = [
    "anisotropic_elements",
    "anisotropic_regions",
    "coefficient",
    "element_axes",
    "element_tensors",
    "element_values",
    "lossy_eps",
    "p0",
    "per_axis_elements",
    "permittivity",
    "principal_permittivity",
    "principal_values",
    "surface_coefficient",
    "surface_p0",
]

_DEFAULTS = {p.name: p.default for p in PROPERTIES}


def element_values(
    model: Model, name: str, *, per_axis: float | None = None
) -> np.ndarray:
    """Property ``name`` per tetrahedron (gfem element order), one value each.

    Tetrahedra without a material (excluded volumes) get the property's
    default. An anisotropic material that holds ``name`` as principal values
    (a `Material` with three of them, or a `MaterialField` with `axes`) has
    no one value: its tetrahedra get ``per_axis``, a placeholder for a
    caller that reads them from the tensors (`element_tensors`,
    `lossy_eps`, `principal_values`), and raise when it is ``None``.

    Raises:
        ValueError: A region holds ``name`` as principal values and
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
            principal = isinstance(value, tuple)
        else:
            principal = material.has_principal_values(name)
        if principal:
            if per_axis is None:
                raise ValueError(
                    f"The material of region {region!r} holds {name} as three "
                    "principal values; it has no one value per element (read "
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


def per_axis_elements(model: Model) -> np.ndarray:
    """Whether each tetrahedron lies in a `MaterialField` with `axes`."""
    out = np.zeros(len(model.mesh.tetrahedra), bool)
    for region in _per_axis_fields(model):
        out[model.mesh.elements(region)] = True
    return out


def element_axes(model: Model) -> np.ndarray:
    """The principal axes of each tetrahedron's field, ``(n, 3, 3)``, rows a_k.

    The axes of the `MaterialField` with `axes` the tetrahedron lies in; the
    coordinate axes elsewhere.
    """
    out = np.broadcast_to(np.eye(3), (len(model.mesh.tetrahedra), 3, 3)).copy()
    for region, field in _per_axis_fields(model).items():
        out[model.mesh.elements(region)] = np.asarray(field.axes, float)
    return out


def principal_values(model: Model, name: str) -> np.ndarray:
    """Property ``name`` per tetrahedron and principal axis, ``(n, 3)``.

    The values of the `MaterialField` with `axes` each tetrahedron lies in,
    along `element_axes` (a field's default where it leaves ``name`` out).
    Elsewhere the element's one value three times, and 1 for a `Material`
    with principal values of ``name`` (`element_values`): a placeholder for
    a reader of the fields' tetrahedra only.
    """
    scalar = element_values(model, name, per_axis=1.0)
    fields = _per_axis_fields(model)
    values = {r: np.asarray(field_principal_values(f, name)) for r, f in fields.items()}
    dtype = np.result_type(scalar, *values.values())
    out = np.repeat(scalar[:, None], 3, axis=1).astype(dtype)
    for region, value in values.items():
        out[model.mesh.elements(region)] = value
    return out


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


def anisotropic_regions(model: Model, regions: tuple[str, ...]) -> tuple[str, ...]:
    """The ``regions`` with any tensor property, in order.

    Structural: a `Material` that is not `Material.isotropic` (principal
    values of ``tan_delta`` or ``conductivity`` alone count) and a
    `MaterialField` with `axes`, whatever its values. Wider than
    `anisotropic_elements`, which ignores loss-only tensors (see the module
    docstring).
    """
    out = []
    for region in regions:
        material = model.materials.get(region)
        if isinstance(material, Material):
            anisotropic = not material.isotropic
        else:
            anisotropic = material is not None and material.axes is not None
        if anisotropic:
            out.append(region)
    return tuple(out)


def principal_permittivity(
    model: Model, regions: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Real principal ``eps_r`` and principal axes per tetrahedron.

    ``(n, 3)`` values and ``(n, 3, 3)`` axes (rows ``a_k``) for the
    tetrahedra of ``regions`` (`anisotropic_regions`), in their material's
    `axes` order (the coordinate axes when ``None``); zero elsewhere, so
    ``sum_k values[:, k] a_k a_k^T`` is ``Re eps_r`` on those tetrahedra.
    """
    mesh = model.mesh
    values = np.zeros((len(mesh.tetrahedra), 3))
    axes = np.zeros((len(mesh.tetrahedra), 3, 3))
    for region in regions:
        material = model.materials[region]
        if isinstance(material, Material):
            value = np.real(np.asarray(material.eps_r, complex))
        else:
            value = np.real(np.asarray(field_principal_values(material, "eps_r")))
        members = mesh.elements(region)
        values[members] = np.broadcast_to(value, (len(members), 3))
        axes[members] = np.eye(3) if material.axes is None else material.axes
    return values, axes


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
        r: _field_tensors(f, np.asarray(field_principal_values(f, "eps_r")).real)
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
        r: _field_tensors(f, np.asarray(field_principal_values(f, name)))
        for r, f in _per_axis_fields(model).items()
    }
    if not anisotropic and not fields:
        return scalar
    return _with_tensors(scalar, model, anisotropic, fields)


def anisotropic_elements(model: Model) -> np.ndarray:
    """Whether each tetrahedron has a tensor real ``eps_r`` or ``mu_r``.

    Structural: the tetrahedra of a `Material` with three principal values
    of ``eps_r`` or ``mu_r``, and of a `MaterialField` with `axes`, whatever
    the values (so an optimizer moving a field's values never changes it).
    Principal values of a `Material`'s ``tan_delta`` or ``conductivity``
    alone do not count: they leave the real wave tensors, and so the
    boundary admittances, isotropic. `anisotropic_regions` counts them (see
    the module docstring).
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

    A `Material` with principal values of `eps_r` or `tan_delta`, and a
    `MaterialField` with `axes`, give the tensor of their principal values
    ``eps_r[k] (1 - i tan_delta[k])`` along their axes (the ``(n, 3, 3)``
    form then holds multiples of the identity elsewhere).
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
        eps = np.asarray(field_principal_values(f, "eps_r"))
        tan = np.asarray(field_principal_values(f, "tan_delta")).real
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
    if built.comm is not None:
        space, dofs = _facet_space(built, is_complex=is_complex)
        gf = ngs.GridFunction(space)
        gf.vec.FV().NumPy()[dofs] = np.asarray(values)[built.triangle_order]
        return gf
    gf = ngs.GridFunction(ngs.SurfaceL2(built.mesh, order=0, complex=is_complex))
    gf.vec.FV().NumPy()[:] = np.asarray(values)[built.triangle_order]
    return gf


def _facet_space(
    built: EngineMesh, *, is_complex: bool
) -> tuple[ngs.FESpace, np.ndarray]:
    """Surface P0 under ranks: the space, and the DOF of each local surface element.

    `ngs.SurfaceL2` deadlocks in its constructor when a surface triangle lies on
    the cut between two ranks and only one of them holds such triangles (ADR 0066),
    so it never runs above one rank. One space per discretization; collective.
    """
    found = built.surface_spaces.get(is_complex)
    if found is None:
        space = ngs.FacetFESpace(built.mesh, order=0, complex=is_complex)
        n = built.mesh.GetNE(ngs.BND)
        dofs = np.fromiter(
            (space.GetDofNrs(ngs.ElementId(ngs.BND, i))[0] for i in range(n)),
            dtype=np.int64,
            count=n,
        )
        found = built.surface_spaces[is_complex] = (space, dofs)
    return found


def surface_coefficient(
    built: EngineMesh, values: np.ndarray
) -> ngs.CoefficientFunction:
    """Per-triangle values, ``(n,)`` or ``(n, 3, 3)``, as an NGSolve coefficient."""
    if values.ndim == 1:
        return surface_p0(built, values)
    parts = [surface_p0(built, values[:, i, j]) for i in range(3) for j in range(3)]
    return ngs.CF(tuple(parts), dims=(3, 3))
