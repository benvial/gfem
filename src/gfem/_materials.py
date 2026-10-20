"""`Material` (constant per region) and `MaterialField` (one value per element)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._pytree import register_fields
from gfem.errors import NonDifferentiableError


@dataclasses.dataclass(frozen=True)
class _Property:
    """One row of the material property table.

    The table drives `Material`, `MaterialField` and `MaterialField.interpolate`:
    a new property (for example a thermal one) is a new row, not a new code path.
    """

    name: str
    unit: str
    physics: Literal["em", "thermal"]
    default: float | None
    strict: bool  # value > 0 (else >= 0)
    may_be_complex: bool = False  # in a MaterialField: lossy eps_r, mu_r
    may_be_tensor: bool = False  # in a Material: three principal values


PROPERTIES: tuple[_Property, ...] = (
    _Property(
        "eps_r", "1", "em", 1.0, strict=True, may_be_complex=True, may_be_tensor=True
    ),
    _Property(
        "mu_r", "1", "em", 1.0, strict=True, may_be_complex=True, may_be_tensor=True
    ),
    _Property("tan_delta", "1", "em", 0.0, strict=False, may_be_tensor=True),
    _Property("conductivity", "S/m", "em", 0.0, strict=False),
    _Property("thermal_conductivity", "W/(m K)", "thermal", None, strict=True),
    _Property("density", "kg/m^3", "thermal", None, strict=True),
    _Property("specific_heat", "J/(kg K)", "thermal", None, strict=True),
)
PROPERTY_NAMES = tuple(p.name for p in PROPERTIES)

_TRACED_REMEDY = (
    "To differentiate with respect to a region's material, assign "
    "gfem.MaterialField.uniform(region, material) to that region instead: its "
    "values are pytree leaves."
)


def _bound_error(prop: _Property, owner: str, value: Any) -> ValueError:
    bound = "> 0" if prop.strict else ">= 0"
    if prop.physics == "em":
        return ValueError(
            f"{owner} is not passive: {prop.name}={value} but {prop.name} must be "
            f"{bound}. A passive medium has eps_r > 0, mu_r > 0 and losses "
            "(tan_delta, conductivity) >= 0."
        )
    return ValueError(f"{owner}: {prop.name} must be {bound} {prop.unit}, got {value}.")


def _material_value(prop: _Property, value: Any, owner: str) -> Any:
    if value is None:
        if prop.default is not None:
            raise ValueError(f"{owner}: {prop.name} cannot be None.")
        return None
    if is_traced(value):
        raise NonDifferentiableError(
            f"{owner}: {prop.name} is traced by JAX, but Material values are fixed "
            f"constants. {_TRACED_REMEDY}"
        )
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise ValueError(
            f"{owner}: {prop.name} must be real; give losses with tan_delta and "
            "conductivity."
        )
    try:
        arr = arr.astype(float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{owner}: {prop.name} must be a number, got {value!r}."
        ) from None
    if arr.ndim == 1 and prop.may_be_tensor:
        if arr.shape != (3,):
            raise ValueError(
                f"{owner}: {prop.name} takes one value or three principal values "
                f"(along `axes`), got {arr.shape[0]}."
            )
    elif arr.ndim != 0:
        raise ValueError(f"{owner}: {prop.name} must be a single number.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{owner}: {prop.name} must be finite, got {value!r}.")
    if (arr <= 0).any() if prop.strict else (arr < 0).any():
        raise _bound_error(prop, owner, value)
    return float(arr) if arr.ndim == 0 else tuple(float(v) for v in arr)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Material:
    """A linear medium, constant over the region it is assigned to.

    Keyword-only. Units are SI. Under e^{+iωt} the complex permittivity is
    ε = ε0 eps_r (1 - i tan_delta) - i conductivity / ω.

    Attributes:
        eps_r: Relative permittivity, one value or three principal values
            along `axes` (symmetric anisotropy).
        mu_r: Relative permeability, one value or three principal values.
        tan_delta: Dielectric loss tangent, >= 0: one value, or three
            principal values along `axes` (an anisotropic crystal such as
            sapphire). Each principal value of the permittivity is then
            ``eps_r[k] (1 - i tan_delta[k])``.
        conductivity: The electrical conductivity in S/m, >= 0 (not the
            thermal conductivity, which is `thermal_conductivity`).
        axes: Three orthonormal vectors, the principal directions of an
            anisotropic `eps_r` / `mu_r` / `tan_delta`. ``None``: the
            coordinate axes.
        thermal_conductivity: Reserved for thermal problems, W/(m K).
            ``None`` means not given.
        density: Reserved for thermal problems, kg/m^3.
        specific_heat: Reserved for thermal problems, J/(kg K).
        name: A free-form label (for example ``"silicon"``), shown in reprs.

    Raises:
        ValueError: The medium is not passive (eps_r or mu_r <= 0, a negative
            loss), a value is not finite, or `axes` is not orthonormal.
        NonDifferentiableError: A value is traced by `jax.grad`; use
            `MaterialField.uniform` to differentiate with respect to a
            region's material.
    """

    eps_r: float | tuple[float, float, float] = 1.0
    mu_r: float | tuple[float, float, float] = 1.0
    tan_delta: float | tuple[float, float, float] = 0.0
    conductivity: float = 0.0
    axes: tuple[tuple[float, float, float], ...] | None = None
    thermal_conductivity: float | None = None
    density: float | None = None
    specific_heat: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        owner = f"Material {self.name!r}" if self.name else "Material"
        for prop in PROPERTIES:
            value = _material_value(prop, getattr(self, prop.name), owner)
            object.__setattr__(self, prop.name, value)
        if self.axes is not None:
            object.__setattr__(self, "axes", _axes(self.axes, owner))

    def __repr__(self) -> str:
        parts = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != f.default
        ]
        return f"Material({', '.join(parts)})"

    @property
    def isotropic(self) -> bool:
        """Whether `eps_r`, `mu_r` and `tan_delta` are single numbers."""
        return not any(
            isinstance(v, tuple) for v in (self.eps_r, self.mu_r, self.tan_delta)
        )


jax.tree_util.register_static(Material)


def principal_tensor(
    values: complex | tuple | np.ndarray,
    axes: tuple[tuple[float, float, float], ...] | None,
) -> np.ndarray:
    """``sum_k values[k] a_k a_k^T``: a 3x3 tensor in the global frame.

    ``values`` is one value (a multiple of the identity) or three principal
    values along the rows ``a_k`` of ``axes`` (``None``: the coordinate axes).
    """
    v = np.broadcast_to(np.asarray(values), (3,))
    a = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
    return np.einsum("k,ki,kj->ij", v, a, a)


def lossy_permittivity(material: Material) -> np.ndarray:
    """``eps_r (1 - i tan_delta)`` per principal axis, as a 3x3 complex tensor."""
    eps = np.broadcast_to(np.asarray(material.eps_r, dtype=float), (3,))
    tan = np.broadcast_to(np.asarray(material.tan_delta, dtype=float), (3,))
    return principal_tensor(eps * (1 - 1j * tan), material.axes)


def _axes(axes: Any, owner: str) -> tuple[tuple[float, float, float], ...]:
    if is_traced(axes):
        raise NonDifferentiableError(f"{owner}: axes are fixed. {_TRACED_REMEDY}")
    a = np.asarray(axes, dtype=float)
    if a.shape != (3, 3) or not np.allclose(a @ a.T, np.eye(3), atol=1e-9):
        raise ValueError(
            f"{owner}: axes must be three orthonormal 3-vectors (the principal "
            f"directions of eps_r / mu_r / tan_delta), got {axes!r}."
        )
    return tuple(tuple(float(x) for x in row) for row in a)  # ty: ignore[invalid-return-type]


# --- MaterialField --------------------------------------------------------------


def _field_value(prop: _Property, value: Any) -> jax.Array | None:
    if value is None:
        return None
    arr = jnp.asarray(value)
    if jnp.iscomplexobj(arr):
        if not prop.may_be_complex:
            raise ValueError(
                f"MaterialField: {prop.name} must be real; complex values are "
                "allowed for eps_r and mu_r only."
            )
        arr = arr.astype(jnp.complex128)
    else:
        arr = arr.astype(jnp.float64)
    if arr.ndim > 1:
        raise ValueError(
            f"MaterialField: {prop.name} has shape {arr.shape}, but a material "
            "field holds one value per element (shape (n_elements,)), or one "
            "value for the whole region (shape ())."
        )
    if not is_traced(arr):
        _check_field_values(prop, np.asarray(arr))
    return arr


def _check_field_values(prop: _Property, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"MaterialField: {prop.name} must be finite.")
    real = arr.real
    bad = (real <= 0) if prop.strict else (real < 0)
    if np.iscomplexobj(arr):
        bad = bad | (arr.imag > 0)  # e^{+iωt}: a lossy medium has Im <= 0
    if bad.any():
        first = arr.reshape(-1)[int(np.flatnonzero(bad.reshape(-1))[0])]
        err = _bound_error(prop, "MaterialField", first)
        if np.iscomplexobj(arr):
            raise ValueError(
                f"{err} Under e^{{+iωt}} a lossy {prop.name} has a negative "
                "imaginary part."
            )
        raise err


_SCHEMES: dict[str, Callable[[jax.Array, float], jax.Array]] = {
    "linear": lambda rho, _: rho,
    "simp": lambda rho, p: rho**p,
    "ramp": lambda rho, p: rho / (1 + p * (1 - rho)),
}


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class MaterialField:
    """Material properties that vary over one volume region, per element (P0).

    Keyword-only. Any `Material` property may be given; a property left out
    is ``None`` (an empty pytree subtree, so adding a property later never
    changes the tree structure of existing code). Each given property is
    either an array of one value per tetrahedron of the region, in mesh order
    (``mesh.elements(region)``), or a single value for the whole region.

    The values are pytree leaves: `jax.grad` of any objective reaches them.
    `eps_r` and `mu_r` may be complex (lossy, negative imaginary part under
    e^{+iωt}); `conductivity` is electrical, S/m.

    Build one with `MaterialField.interpolate` (a design density blended
    between two materials) or `MaterialField.uniform` (a whole region whose
    constant material should be differentiable).

    Attributes:
        region: The region the field was built for, or ``None``. When set,
            the model checks that the field is assigned to this region.
    """

    eps_r: jax.Array | None = None
    mu_r: jax.Array | None = None
    tan_delta: jax.Array | None = None
    conductivity: jax.Array | None = None
    thermal_conductivity: jax.Array | None = None
    density: jax.Array | None = None
    specific_heat: jax.Array | None = None
    region: str | None = None

    def __post_init__(self) -> None:
        sizes = set()
        for prop in PROPERTIES:
            value = _field_value(prop, getattr(self, prop.name))
            object.__setattr__(self, prop.name, value)
            if value is not None and value.ndim == 1:
                sizes.add(value.shape[0])
        if all(getattr(self, n) is None for n in PROPERTY_NAMES):
            raise ValueError(
                "MaterialField needs at least one property, e.g. "
                "MaterialField(eps_r=values)."
            )
        if len(sizes) > 1:
            raise ValueError(
                "MaterialField: every per-element property must have the same "
                f"number of values (one per element), got {sorted(sizes)}."
            )

    @property
    def n_elements(self) -> int | None:
        """Number of per-element values, or ``None`` if every value is uniform."""
        for name in PROPERTY_NAMES:
            v = getattr(self, name)
            if v is not None and v.ndim == 1:
                return int(v.shape[0])
        return None

    @property
    def properties(self) -> tuple[str, ...]:
        """Names of the properties this field gives."""
        return tuple(n for n in PROPERTY_NAMES if getattr(self, n) is not None)

    @classmethod
    def interpolate(
        cls,
        density: jax.Array,
        void: Material,
        solid: Material,
        *,
        scheme: Literal["linear", "simp", "ramp"] = "linear",
        penalty: float = 3.0,
    ) -> MaterialField:
        """Blend two isotropic materials by a density per element.

        Each property that both materials define becomes
        ``void + w(density) * (solid - void)``, with ``w`` the scheme's
        weight: ``linear`` (w = rho), ``simp`` (w = rho^penalty) or ``ramp``
        (w = rho / (1 + penalty (1 - rho))). Differentiable in ``density``.

        Args:
            density: One value per element, in [0, 1] (0 = void, 1 = solid).
            void: The material at density 0.
            solid: The material at density 1.
            scheme: The interpolation weight.
            penalty: The SIMP / RAMP penalty.

        Raises:
            ValueError: Unknown scheme, or an anisotropic material.
        """
        if scheme not in _SCHEMES:
            raise ValueError(
                f"Unknown interpolation scheme {scheme!r}; use one of "
                f"{', '.join(repr(s) for s in _SCHEMES)}."
            )
        for m in (void, solid):
            if not m.isotropic:
                raise ValueError(
                    "MaterialField supports isotropic materials only; "
                    f"{m.name or 'a material'} has per-axis eps_r, mu_r or "
                    "tan_delta."
                )
        rho = jnp.asarray(density, dtype=jnp.float64)
        w = _SCHEMES[scheme](rho, penalty)
        props = {}
        for name in PROPERTY_NAMES:
            a, b = getattr(void, name), getattr(solid, name)
            if a is not None and b is not None:
                props[name] = a + w * (b - a)
        return cls(**props)

    @classmethod
    def uniform(cls, region: str, material: Material) -> MaterialField:
        """A whole region's constant material, as a differentiable field.

        Each property the material defines becomes one scalar leaf, so
        ``jax.grad`` gives the sensitivity to the region's eps_r,
        conductivity, ... directly.

        Args:
            region: The volume region the field is for.
            material: An isotropic material.

        Raises:
            ValueError: The material is anisotropic.
        """
        if not material.isotropic:
            raise ValueError(
                "MaterialField supports isotropic materials only; "
                f"{material.name or 'this material'} has per-axis eps_r, mu_r or "
                "tan_delta."
            )
        props = {
            n: jnp.asarray(getattr(material, n), dtype=jnp.float64)
            for n in PROPERTY_NAMES
            if getattr(material, n) is not None
        }
        return cls(**props, region=region)

    def __repr__(self) -> str:
        n = self.n_elements
        size = "uniform" if n is None else f"{n} elements"
        parts = []
        for name in self.properties:
            v = getattr(self, name)
            if is_traced(v):
                parts.append(f"{name}=<traced>")
                continue
            a = np.asarray(v)
            lo, hi = a.real.min(), a.real.max()
            parts.append(
                f"{name}={lo:.4g}" if lo == hi else f"{name} in [{lo:.4g}, {hi:.4g}]"
            )
        where = f", region={self.region!r}" if self.region else ""
        return f"MaterialField({size}{where}, {', '.join(parts)})"


register_fields(MaterialField, PROPERTY_NAMES)
