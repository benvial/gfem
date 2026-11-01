"""`Material` (constant per region) and `MaterialField` (one value per element)."""

from __future__ import annotations

import dataclasses
import itertools
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
    may_be_tensor: bool = False  # in a Material: principal values or a matrix


PROPERTIES: tuple[_Property, ...] = (
    _Property(
        "eps_r", "1", "em", 1.0, strict=True, may_be_complex=True, may_be_tensor=True
    ),
    _Property(
        "mu_r", "1", "em", 1.0, strict=True, may_be_complex=True, may_be_tensor=True
    ),
    _Property("tan_delta", "1", "em", 0.0, strict=False, may_be_tensor=True),
    _Property("conductivity", "S/m", "em", 0.0, strict=False, may_be_tensor=True),
    _Property("thermal_conductivity", "W/(m K)", "thermal", None, strict=True),
    _Property("density", "kg/m^3", "thermal", None, strict=True),
    _Property("specific_heat", "J/(kg K)", "thermal", None, strict=True),
)
PROPERTY_NAMES = tuple(p.name for p in PROPERTIES)
_BY_NAME = {p.name: p for p in PROPERTIES}

_TRACED_REMEDY = (
    "To differentiate with respect to a region's material, assign "
    "gfem.MaterialField.uniform(region, material) to that region instead: its "
    "values are pytree leaves."
)


_TENSOR_TOL = 1e-9  # symmetry, diagonality and equal principal values, relative


def _bound_error(
    prop: _Property, owner: str, value: Any, axis: str | None = None
) -> ValueError:
    """The passivity error; ``axis`` names a principal axis for a tensor value."""
    bound = "> 0" if prop.strict else ">= 0"
    if prop.physics != "em":
        return ValueError(
            f"{owner}: {prop.name} must be {bound} {prop.unit}, got {value}."
        )
    if axis is None:
        what = f"{prop.name}={value} but {prop.name} must be {bound}"
    else:
        what = (
            f"{prop.name} has principal value {value} (along axis {axis}) but "
            f"every principal value of {prop.name} must be {bound}"
        )
    return ValueError(
        f"{owner} is not passive: {what}. A passive medium has eps_r > 0, "
        "mu_r > 0 and losses (tan_delta, conductivity) >= 0."
    )


def _short(x: float) -> str:
    """``x`` to 12 significant digits, so round-off does not show in messages."""
    return repr(float(f"{x:.12g}") + 0.0)


def _vector(a: np.ndarray) -> str:
    return f"({', '.join(_short(x) for x in a)})"


def _check_passive(
    prop: _Property, values: np.ndarray, owner: str, axes: np.ndarray | None
) -> None:
    """Every principal value meets the bound; the first that does not raises."""
    bad = (values <= 0) if prop.strict else (values < 0)
    if not bad.any():
        return
    k = int(np.flatnonzero(bad)[0])
    axis = str(k + 1) if axes is None else _vector(axes[k])
    raise _bound_error(prop, owner, _short(values[k]), axis)


def _check_shape(prop: _Property, arr: np.ndarray, owner: str) -> None:
    if arr.ndim == 1 and prop.may_be_tensor:
        if arr.shape != (3,):
            raise ValueError(
                f"{owner}: {prop.name} takes one value or three principal values "
                f"(along `axes`), got {arr.shape[0]}."
            )
    elif arr.ndim == 2 and prop.may_be_tensor:
        if arr.shape != (3, 3):
            raise ValueError(
                f"{owner}: {prop.name} takes one value, three principal values "
                f"(along `axes`) or a 3x3 matrix, got shape {arr.shape}."
            )
    elif arr.ndim != 0:
        raise ValueError(f"{owner}: {prop.name} must be a single number.")


def _material_value(prop: _Property, value: Any, owner: str) -> Any:
    """A checked value: a float, a 3-tuple, or a symmetric 3x3 matrix.

    A matrix is returned as a symmetrized array, its passivity unchecked:
    `_principal` diagonalizes it with the material's other matrices.
    """
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
    _check_shape(prop, arr, owner)
    if not np.isfinite(arr).all():
        raise ValueError(f"{owner}: {prop.name} must be finite, got {value!r}.")
    if arr.ndim == 2:
        if np.linalg.norm(arr - arr.T) > _TENSOR_TOL * np.linalg.norm(arr):
            raise ValueError(
                f"{owner}: {prop.name} is not symmetric; non-symmetric "
                "(gyrotropic) tensors are not supported."
            )
        return (arr + arr.T) / 2
    if arr.ndim == 0:
        if (arr <= 0) if prop.strict else (arr < 0):
            raise _bound_error(prop, owner, value)
        return float(arr)
    _check_passive(prop, arr, owner, None)
    if (arr == arr[0]).all():  # equal principal values are one value, exactly
        return float(arr[0])
    return tuple(float(v) for v in arr)


def _eigenspaces(matrices: list[np.ndarray]) -> list[np.ndarray]:
    """Orthonormal bases (3 x d) of the common eigenspaces of ``matrices``.

    Each matrix splits the eigenspaces left by the ones before it; eigenvalues
    within `_TENSOR_TOL` of the matrix's norm stay one eigenspace.
    """
    spaces = [np.eye(3)]
    for m in matrices:
        tol = _TENSOR_TOL * np.linalg.norm(m)
        split = []
        for q in spaces:
            if q.shape[1] == 1:
                split.append(q)
                continue
            w, v = np.linalg.eigh(q.T @ m @ q)
            basis = q @ v
            start = 0
            for i in range(1, len(w) + 1):
                if i == len(w) or w[i] - w[i - 1] > tol:
                    split.append(basis[:, start:i])
                    start = i
        spaces = split
    return spaces


def _ordered_axes(spaces: list[np.ndarray]) -> np.ndarray:
    """The principal axes (rows) spanning ``spaces``, in the normal form.

    Axis k is the one closest to coordinate axis k: of the assignments of
    coordinate axes to eigenspaces, the one maximizing the summed ``|a_k[k]|``
    (the first, within round-off). In an eigenspace of dimension > 1 the axes
    are the Gram-Schmidt projection of its coordinate axes. Each axis's
    dominant component is positive.
    """
    slots = [i for i, q in enumerate(spaces) for _ in range(q.shape[1])]
    best, best_score = np.eye(3), -np.inf
    for perm in itertools.permutations(range(3)):
        axes = np.zeros((3, 3))
        for i, q in enumerate(spaces):
            ks = [k for k in range(3) if slots[perm[k]] == i]
            if q.shape[1] == 1:
                axes[ks[0]] = q[:, 0]
                continue
            p = q @ q.T
            for j, k in enumerate(ks):
                v = p[:, k] - axes[ks[:j]].T @ (axes[ks[:j]] @ p[:, k])
                axes[k] = v / max(np.linalg.norm(v), 1e-300)
        if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-9):
            continue  # a coordinate axis (nearly) orthogonal to its eigenspace
        score = np.abs(np.diag(axes)).sum()
        if score > best_score + _TENSOR_TOL:
            best, best_score = axes, score
    dominant = np.abs(best).argmax(axis=1)
    return best * np.sign(best[np.arange(3), dominant])[:, None]


def _principal(
    values: dict[str, Any], axes: Any, owner: str
) -> tuple[dict[str, Any], Any]:
    """Matrix values turned into principal values along one shared `axes`."""
    matrices = {n: v for n, v in values.items() if isinstance(v, np.ndarray)}
    if not matrices:
        return values, axes
    first = next(iter(matrices))
    if axes is not None:
        raise ValueError(
            f"{owner}: {first} is a 3x3 matrix, which carries its own principal "
            "axes; leave `axes` as None."
        )
    along_axes = [n for n, v in values.items() if isinstance(v, tuple)]
    if along_axes:
        name = along_axes[0]
        raise ValueError(
            f"{owner}: {first} is a 3x3 matrix but {name} gives three "
            "principal values along `axes`, which the matrix derives itself; "
            f"give {name} as one value or a 3x3 matrix."
        )

    def off_diagonal(m: np.ndarray, a: np.ndarray) -> bool:
        d = a @ m @ a.T
        return np.abs(d - np.diag(np.diag(d))).max() > _TENSOR_TOL * np.linalg.norm(m)

    if not any(off_diagonal(m, np.eye(3)) for m in matrices.values()):
        a = None
    else:
        a = _ordered_axes(_eigenspaces(list(matrices.values())))
        for name, m in matrices.items():
            if off_diagonal(m, a):
                other = next(
                    (
                        o
                        for o, n in matrices.items()
                        if o != name
                        and np.linalg.norm(m @ n - n @ m)
                        > _TENSOR_TOL * np.linalg.norm(m) * np.linalg.norm(n)
                    ),
                    next(o for o in matrices if o != name),
                )
                pair = [p.name for p in PROPERTIES if p.name in (name, other)]
                raise ValueError(
                    f"{pair[0]} and {pair[1]} of {owner} have different principal "
                    "axes; a material's tensors share one set of principal axes."
                )
    out = dict(values)
    tensor = False
    for name, m in matrices.items():
        prop = next(p for p in PROPERTIES if p.name == name)
        v = np.diag(m) if a is None else np.einsum("ki,ij,kj->k", a, m, a)
        _check_passive(prop, v, owner, a)
        if np.ptp(v) <= _TENSOR_TOL * np.linalg.norm(m):
            # equal to 1e-9, not an approximation; the mean keeps the trace
            out[name] = float(np.mean(v))
        else:
            out[name] = tuple(float(x) for x in v)
            tensor = True
    return out, (a if tensor and a is not None else None)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Material:
    """A linear medium, constant over the region it is assigned to.

    Keyword-only. Units are SI. Under e^{+iωt} the complex permittivity is,
    per principal axis ``a_k`` (the rows of `axes`),
    ε = ε0 Σ_k [eps_r[k] (1 - i tan_delta[k]) - i conductivity[k] / (ε0 ω)]
    a_k a_kᵀ; a single value counts for all three axes. The two loss
    models, `tan_delta` and `conductivity`, add.

    Attributes:
        eps_r: Relative permittivity: one value, three principal values
            along `axes`, or a real symmetric 3x3 matrix in the global frame.
        mu_r: Relative permeability, in the same forms as `eps_r`.
        tan_delta: Dielectric loss tangent, >= 0, in the same forms as
            `eps_r` (an anisotropic crystal such as sapphire). Each principal
            value of the permittivity is then ``eps_r[k] (1 - i tan_delta[k])``.
        conductivity: The electrical conductivity in S/m, in the same forms
            as `eps_r`, each principal value >= 0 (not the thermal
            conductivity, which is `thermal_conductivity`).
        axes: Three orthonormal vectors, the principal directions shared by
            `eps_r`, `mu_r`, `tan_delta` and `conductivity`. ``None``: the
            coordinate axes. A matrix input sets them: the material stores
            the matrix as principal values along derived axes (axis k the
            one closest to coordinate axis k, its dominant component
            positive; ``None`` if every matrix is diagonal). Equal principal
            values are stored as one value.
        thermal_conductivity: Reserved for thermal problems, W/(m K).
            ``None`` means not given.
        density: Reserved for thermal problems, kg/m^3.
        specific_heat: Reserved for thermal problems, J/(kg K).
        name: A free-form label (for example ``"silicon"``), shown in reprs.

    Raises:
        ValueError: The medium is not passive (a principal value of eps_r or
            mu_r <= 0, or of a loss < 0), a value is not finite, `axes` is
            not orthonormal, a matrix is not symmetric (gyrotropic tensors are
            not supported), a matrix is given with `axes` or beside three
            principal values, or the matrices do not share principal axes.
        NonDifferentiableError: A value is traced by `jax.grad`; use
            `MaterialField.uniform` to differentiate with respect to a
            region's material.
    """

    eps_r: float | tuple[float, float, float] = 1.0
    mu_r: float | tuple[float, float, float] = 1.0
    tan_delta: float | tuple[float, float, float] = 0.0
    conductivity: float | tuple[float, float, float] = 0.0
    axes: tuple[tuple[float, float, float], ...] | None = None
    thermal_conductivity: float | None = None
    density: float | None = None
    specific_heat: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        owner = f"Material {self.name!r}" if self.name else "Material"
        values = {
            p.name: _material_value(p, getattr(self, p.name), owner) for p in PROPERTIES
        }
        values, axes = _principal(values, self.axes, owner)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if axes is not None:
            object.__setattr__(self, "axes", _axes(axes, owner))

    def __repr__(self) -> str:
        parts = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != f.default
        ]
        return f"Material({', '.join(parts)})"

    @property
    def isotropic(self) -> bool:
        """Whether every tensor property is one number.

        The tensor properties are `eps_r`, `mu_r`, `tan_delta` and
        `conductivity`; when each is one number, the material does not depend
        on the field direction.
        """
        return not any(
            isinstance(getattr(self, p.name), tuple)
            for p in PROPERTIES
            if p.may_be_tensor
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


def _axes(
    axes: Any, owner: str, remedy: str = _TRACED_REMEDY
) -> tuple[tuple[float, float, float], ...]:
    if is_traced(axes):
        raise NonDifferentiableError(f"{owner}: axes are fixed. {remedy}")
    a = np.asarray(axes, dtype=float)
    if a.shape != (3, 3) or not np.allclose(a @ a.T, np.eye(3), atol=1e-9):
        raise ValueError(
            f"{owner}: axes must be three orthonormal 3-vectors (the principal "
            f"directions of eps_r / mu_r / tan_delta / conductivity), got {axes!r}."
        )
    return tuple(tuple(float(x) for x in row) for row in a)  # ty: ignore[invalid-return-type]


# --- MaterialField --------------------------------------------------------------


def _field_value(prop: _Property, value: Any, per_axis: bool) -> jax.Array | None:
    """A checked field value; ``per_axis``: three principal values (along `axes`)."""
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
    if per_axis:
        if arr.ndim not in (1, 2) or arr.shape[-1] != 3:
            raise ValueError(
                f"MaterialField: {prop.name} has shape {arr.shape}, but a field "
                "with `axes` holds three principal values per element (shape "
                "(n_elements, 3)), or for the whole region (shape (3,))."
            )
    elif arr.ndim > 1:
        need = (
            "; principal values need `axes` (np.eye(3) for the coordinate axes)"
            if prop.may_be_tensor
            else ""
        )
        raise ValueError(
            f"MaterialField: {prop.name} has shape {arr.shape}, but a material "
            "field holds one value per element (shape (n_elements,)), or one "
            f"value for the whole region (shape ()){need}."
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


def _common_axes(
    void: Material, solid: Material
) -> tuple[tuple[tuple[float, float, float], ...], tuple[dict, dict]]:
    """The shared principal axes of two materials and their values along them.

    The axes are `_principal`'s normal form of the union of both materials'
    tensors; each tensor property's values come back as three principal
    values along them. Each axis and value is taken exactly from the
    material it matches (the eigensolver only orders them), so materials
    equal up to the order or sign of their axes give equal values.

    Raises:
        ValueError: The materials' tensors have different principal axes.
    """
    tensors = [p.name for p in PROPERTIES if p.may_be_tensor]
    own = {}  # each material's axes as an array
    matrices = []
    for m in (void, solid):
        own[id(m)] = np.eye(3) if m.axes is None else np.asarray(m.axes, float)
        for name in tensors:
            value = getattr(m, name)
            if isinstance(value, tuple):
                matrices.append(principal_tensor(value, m.axes))
    a = _ordered_axes(_eigenspaces(matrices))
    # snap each axis to the material axis it matches, signed as in the
    # normal form (an axis in a degenerate eigenspace may match none)
    for m in (void, solid):
        if m.isotropic:
            continue
        for k in range(3):
            dots = own[id(m)] @ a[k]
            j = int(np.abs(dots).argmax())
            if abs(dots[j]) > 1 - _TENSOR_TOL:
                axis = own[id(m)][j]
                a[k] = axis * np.sign(axis[np.abs(axis).argmax()])
    ends, ends_of = [], (void, solid)
    for m in ends_of:
        values: dict[str, Any] = {n: getattr(m, n) for n in PROPERTY_NAMES}
        for name in tensors:
            value = getattr(m, name)
            if not isinstance(value, tuple):
                values[name] = np.full(3, value)
                continue
            matrix = principal_tensor(value, m.axes)
            d = a @ matrix @ a.T
            scale = np.linalg.norm(matrix)
            if np.abs(d - np.diag(np.diag(d))).max() > _TENSOR_TOL * scale:
                names = [repr(x.name) if x.name else "(unnamed)" for x in ends_of]
                raise ValueError(
                    f"MaterialField.interpolate: void {names[0]} and solid "
                    f"{names[1]} have different principal axes; a material "
                    "field holds principal values along one fixed set of axes. "
                    "Rotate one material, or build the field directly."
                )
            # each principal value exactly as the material gives it
            own_values = np.asarray(value)
            nearest = np.abs(np.diag(d)[:, None] - own_values[None, :]).argmin(axis=1)
            values[name] = own_values[nearest]
        ends.append(values)
    return _axes(a, "MaterialField.interpolate"), (ends[0], ends[1])


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

    With `axes`, the field is anisotropic: each of `eps_r`, `mu_r`,
    `tan_delta` and `conductivity` holds three principal values along the
    axes, shape ``(n_elements, 3)`` per element or ``(3,)`` for the whole
    region (an isotropic value is three equal values); the thermal
    properties keep one value. The axes are fixed: the values vary, the
    principal directions do not.

    Build one with `MaterialField.interpolate` (a design density blended
    between two materials) or `MaterialField.uniform` (a whole region whose
    constant material should be differentiable).

    Attributes:
        region: The region the field was built for, or ``None``. When set,
            the model checks that the field is assigned to this region.
        axes: Three orthonormal vectors, the principal directions of the
            per-axis values, or ``None`` (an isotropic field). Static data,
            not a pytree leaf: it cannot be differentiated.

    Raises:
        ValueError: A value is not passive or finite, a tensor property's
            shape does not fit `axes` (a 2-D array needs `axes`; with `axes`
            the last dimension is 3), the per-element sizes differ, or
            `axes` is not orthonormal.
        NonDifferentiableError: `axes` is traced by JAX.
    """

    eps_r: jax.Array | None = None
    mu_r: jax.Array | None = None
    tan_delta: jax.Array | None = None
    conductivity: jax.Array | None = None
    thermal_conductivity: jax.Array | None = None
    density: jax.Array | None = None
    specific_heat: jax.Array | None = None
    region: str | None = None
    axes: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.axes is not None:
            remedy = (
                "The orientation is not a design input; differentiate with "
                "respect to the principal values instead."
            )
            object.__setattr__(self, "axes", _axes(self.axes, "MaterialField", remedy))
        sizes = set()
        for prop in PROPERTIES:
            per_axis = self.per_axis(prop.name)
            value = _field_value(prop, getattr(self, prop.name), per_axis)
            object.__setattr__(self, prop.name, value)
            if value is not None and value.ndim == 1 + per_axis:
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
            if v is not None and v.ndim == 1 + self.per_axis(name):
                return int(v.shape[0])
        return None

    def per_axis(self, name: str) -> bool:
        """Whether property ``name`` holds three principal values along `axes`."""
        return self.axes is not None and _BY_NAME[name].may_be_tensor

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
        """Blend two materials by a density per element.

        Each property that both materials define becomes
        ``void + w(density) * (solid - void)``, with ``w`` the scheme's
        weight: ``linear`` (w = rho), ``simp`` (w = rho^penalty) or ``ramp``
        (w = rho / (1 + penalty (1 - rho))). Differentiable in ``density``.

        Anisotropic materials blend per principal value, when every tensor
        of both shares one set of principal axes (an isotropic material
        shares any, so air and a crystal blend). The field then has those
        `axes` (in `Material`'s normal form: axis k the one closest to
        coordinate axis k), so axes given in another order or sign give the
        same field. Two isotropic materials give an isotropic field.

        Args:
            density: One value per element, in [0, 1] (0 = void, 1 = solid).
            void: The material at density 0.
            solid: The material at density 1.
            scheme: The interpolation weight.
            penalty: The SIMP / RAMP penalty.

        Raises:
            ValueError: Unknown scheme, or two materials whose tensors have
                different principal axes.
        """
        if scheme not in _SCHEMES:
            raise ValueError(
                f"Unknown interpolation scheme {scheme!r}; use one of "
                f"{', '.join(repr(s) for s in _SCHEMES)}."
            )
        rho = jnp.asarray(density, dtype=jnp.float64)
        w = _SCHEMES[scheme](rho, penalty)
        if void.isotropic and solid.isotropic:
            axes = None
            ends = tuple(
                {n: getattr(m, n) for n in PROPERTY_NAMES} for m in (void, solid)
            )
        else:
            axes, ends = _common_axes(void, solid)
        props = {}
        for name in PROPERTY_NAMES:
            a, b = ends[0][name], ends[1][name]
            if a is not None and b is not None:
                per_axis = axes is not None and _BY_NAME[name].may_be_tensor
                props[name] = a + (w[..., None] if per_axis else w) * (b - a)
        return cls(**props, axes=axes)

    @classmethod
    def uniform(cls, region: str, material: Material) -> MaterialField:
        """A whole region's constant material, as a differentiable field.

        Each property the material defines becomes one scalar leaf, so
        ``jax.grad`` gives the sensitivity to the region's eps_r,
        conductivity, ... directly. An anisotropic material gives a field
        on its `axes` (the coordinate axes if ``None``) whose tensor
        properties are ``(3,)`` leaves of principal values.

        Args:
            region: The volume region the field is for.
            material: The material.
        """
        axes = None
        if not material.isotropic:
            axes = material.axes or _axes(np.eye(3), "MaterialField.uniform")
        props = {}
        for n in PROPERTY_NAMES:
            value = getattr(material, n)
            if value is None:
                continue
            arr = jnp.asarray(value, dtype=jnp.float64)
            if axes is not None and _BY_NAME[n].may_be_tensor:
                arr = jnp.broadcast_to(arr, (3,))
            props[n] = arr
        return cls(**props, region=region, axes=axes)

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
        if self.axes is not None:
            parts.append(f"axes={self.axes!r}")
        return f"MaterialField({size}{where}, {', '.join(parts)})"


register_fields(MaterialField, PROPERTY_NAMES)
