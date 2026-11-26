"""`Material` (constant per region) and `MaterialField` (one value per element)."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._pytree import register_fields
from gfem._raman_spectrum import RamanSpectrum
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from gfem.design import LayerDensity


@dataclasses.dataclass(frozen=True)
class _Property:
    """One row of the material property table.

    The table drives `Material`, `MaterialField` and `MaterialField.interpolate`:
    a new property (for example a thermal one) is a new row, not a new code path.
    """

    name: str
    unit: str
    physics: Literal["em", "thermal", "mechanical"]
    default: float | None
    strict: bool  # value > 0 (else >= 0)
    bounded: bool = True  # False: any finite real value (``strict`` unread)
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
    _Property("mass_density", "kg/m^3", "thermal", None, strict=True),
    _Property("specific_heat", "J/(kg K)", "thermal", None, strict=True),
    _Property(
        "thermo_optic_coefficient", "1/K", "thermal", None, strict=False, bounded=False
    ),
    _Property("mechanical_quality_factor", "1", "mechanical", None, strict=True),
    _Property("nonlinear_index", "m^2/W", "em", None, strict=False, bounded=False),
    _Property("two_photon_absorption", "m/W", "em", None, strict=False),
)
PROPERTY_NAMES = tuple(p.name for p in PROPERTIES)
_BY_NAME = {p.name: p for p in PROPERTIES}

# the Kerr properties: read by the nonlinear coefficients of a port mode only
_NONLINEAR = ("nonlinear_index", "two_photon_absorption")

_TRACED_REMEDY = (
    "To differentiate with respect to a region's material, assign "
    "gfem.MaterialField.uniform(region, material) to that region instead: "
    "jax.grad reaches its values (an anisotropic material gives principal values "
    "along its axes)."
)


_TENSOR_TOL = 1e-9  # symmetry, diagonality and equal principal values, relative

# The electro-optic tensor's contracted (Voigt) rows: xx, yy, zz, yz, xz, xy.
_VOIGT_I = np.array([0, 1, 2, 1, 0, 0])
_VOIGT_J = np.array([0, 1, 2, 2, 2, 1])
# row of the contracted tensor holding r_ij (and r_ji)
_VOIGT_ROW = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
_EO_FORM = (
    "the contracted 6x3 matrix r_Ik in m/V (rows xx, yy, zz, yz, xz, xy; "
    "columns x, y, z of the field)"
)
# m/V: above the largest known coefficients (about 1.3e-9), far below a value
# typed in pm/V
_EO_MAX = 1e-8


def _eo_shape_error(owner: str, shape: tuple, field: bool) -> ValueError:
    want = (
        "shape (6, 3) for the whole region or (n_elements, 6, 3) per element"
        if field
        else "shape (6, 3)"
    )
    full = (
        "; contract a full r_ijk to its rows ij = xx, yy, zz, yz, xz, xy"
        if tuple(shape[-3:]) == (3, 3, 3)
        else ""
    )
    return ValueError(
        f"{owner}: electro_optic_tensor is {_EO_FORM}, {want}, got shape "
        f"{tuple(shape)}{full}."
    )


def _check_eo_values(arr: np.ndarray, owner: str) -> None:
    """Raise unless a concrete electro-optic tensor is finite and in m/V."""
    if not np.isfinite(arr).all():
        raise ValueError(f"{owner}: electro_optic_tensor must be finite.")
    big = float(np.abs(arr).max(initial=0.0))
    if big > _EO_MAX:
        raise ValueError(
            f"{owner}: electro_optic_tensor has an entry of {_short(big)} m/V, "
            f"above {_EO_MAX:g} m/V: the tensor is in m/V, so 30.9 pm/V is "
            "30.9e-12."
        )


def _eo_real(owner: str) -> ValueError:
    return ValueError(
        f"{owner}: electro_optic_tensor must be real; electro-absorption is not "
        "modelled."
    )


def _material_eo_tensor(
    value: Any, owner: str
) -> tuple[tuple[float, float, float], ...] | None:
    """A checked constant electro-optic tensor: six rows of three floats."""
    if value is None:
        return None
    if is_traced(value):
        raise NonDifferentiableError(
            f"{owner}: electro_optic_tensor is computed inside jax.grad, jax.jit "
            f"or jax.vmap, but Material values are fixed constants. {_TRACED_REMEDY}"
        )
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise _eo_real(owner)
    try:
        arr = arr.astype(float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{owner}: electro_optic_tensor must be {_EO_FORM}, got {value!r}."
        ) from None
    if arr.shape != (6, 3):
        raise _eo_shape_error(owner, arr.shape, field=False)
    _check_eo_values(arr, owner)
    return tuple(tuple(float(x) for x in row) for row in arr)  # ty: ignore[invalid-return-type]


def _check_right_handed(
    axes: Any, owner: str, tensor: str = "electro_optic_tensor"
) -> None:
    """Raise if orthonormal ``axes`` are left-handed (the frame of ``tensor``)."""
    if np.linalg.det(np.asarray(axes, dtype=float)) < 0:
        raise ValueError(
            f"{owner}: axes are left-handed (det -1), but the {tensor} "
            "is given in the frame whose x, y, z are the rows of axes, which must "
            "be right-handed: reverse one axis (for example the last row)."
        )


# --- stiffness and piezoelectric tensor ---------------------------------------------


@dataclasses.dataclass(frozen=True)
class _SolidTensor:
    """A contracted tensor of a solid: its shape, its form and how to contract it."""

    name: str
    shape: tuple[int, int]
    full: tuple[int, ...]  # the shape of the uncontracted tensor
    form: str
    contract: str


_STIFFNESS = _SolidTensor(
    "stiffness",
    (6, 6),
    (3, 3, 3, 3),
    "the contracted 6x6 matrix c_IJ in Pa (rows and columns xx, yy, zz, yz, xz, "
    "xy, engineering shear strain)",
    "contract a full c_ijkl to rows and columns ij = xx, yy, zz, yz, xz, xy, with "
    "no factors of 2",
)
_PIEZOELECTRIC = _SolidTensor(
    "piezoelectric_tensor",
    (3, 6),
    (3, 3, 3),
    "the contracted 3x6 matrix e_kJ of stress constants in C/m^2 (rows x, y, z "
    "of the field; columns xx, yy, zz, yz, xz, xy)",
    "contract a full e_kij to columns ij = xx, yy, zz, yz, xz, xy",
)
_SOLID_TENSORS = (_STIFFNESS, _PIEZOELECTRIC)
# the photoelastic tensor is checked and carried like the solid's, but belongs to
# the optical material, so it is not one of the acoustic `_SOLID_TENSORS`
_PHOTOELASTIC = _SolidTensor(
    "photoelastic_tensor",
    (6, 6),
    (3, 3, 3, 3),
    "the contracted 6x6 matrix p_IJ at constant electric field, dimensionless "
    "(rows xx, yy, zz, yz, xz, xy of the inverse permittivity change, columns "
    "xx, yy, zz, yz, xz, xy of the engineering strain)",
    "contract a full p_ijkl to rows and columns ij = xx, yy, zz, yz, xz, xy, with "
    "no factors of 2",
)
# the second-order nonlinear tensor d is optical too: contracted like the
# piezoelectric tensor (3x6, no factor 2 on the shear columns)
_SECOND_ORDER = _SolidTensor(
    "second_order_nonlinear_tensor",
    (3, 6),
    (3, 3, 3),
    "the contracted 3x6 matrix d_il = d_ijk in m/V, d = chi2 / 2 (rows x, y, z "
    "of the polarization; columns xx, yy, zz, yz, xz, xy of the field pair)",
    "contract a full d_ijk to columns jk = xx, yy, zz, yz, xz, xy, with no "
    "factors of 2",
)
_SOLID_BY_NAME = {t.name: t for t in (*_SOLID_TENSORS, _PHOTOELASTIC, _SECOND_ORDER)}
#: A material field's tensors outside the property table, as its leaves.
_FIELD_TENSORS = (
    "electro_optic_tensor",
    "photoelastic_tensor",
    "second_order_nonlinear_tensor",
    "stiffness",
    "piezoelectric_tensor",
)
# dimensionless: above every tabulated photoelastic constant (below 1 in every
# solid), far below a value typed in percent
_PHOTOELASTIC_MAX = 10.0
# m/V: above the largest known d (about 1e-9), far below a value typed in pm/V
_SECOND_ORDER_MAX = 1e-8
# Pa: below the largest constant of every solid (even a rubber-like c11 is
# near 1e9 Pa), far above a value typed in GPa
_STIFFNESS_MIN = 1e5
# C/m^2: below every stress constant (quartz 0.17), far above the strain
# constants d in C/N (1e-12 to 1e-9)
_PIEZOELECTRIC_MIN = 1e-6


def _solid_shape_error(
    tensor: _SolidTensor, owner: str, shape: tuple, field: bool
) -> ValueError:
    n = tensor.shape
    want = (
        f"shape {n} for the whole region or (n_elements, {n[0]}, {n[1]}) per element"
        if field
        else f"shape {n}"
    )
    full = tuple(shape[-len(tensor.full) :]) == tensor.full
    return ValueError(
        f"{owner}: {tensor.name} is {tensor.form}, {want}, got shape "
        f"{tuple(shape)}{f'; {tensor.contract}' if full else ''}."
    )


def _solid_real(tensor: _SolidTensor, owner: str) -> ValueError:
    remedy = (
        "; give mechanical loss as a mechanical_quality_factor Q_m, which makes "
        "the stiffness c (1 + i / Q_m)"
        if tensor is _STIFFNESS
        else ""
    )
    if tensor is _PHOTOELASTIC:
        remedy = "; no photoelastic loss is modelled"
    if tensor is _SECOND_ORDER:
        remedy = "; no nonlinear loss is modelled"
    return ValueError(f"{owner}: {tensor.name} must be real{remedy}.")


def _check_solid_values(tensor: _SolidTensor, arr: np.ndarray, owner: str) -> None:
    """Raise unless a concrete tensor, ``(..., *shape)``, is finite and in SI units.

    A stiffness is also symmetric and positive definite (a positive strain
    energy), element by element.
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{owner}: {tensor.name} must be finite.")
    big = np.abs(arr).max(axis=(-2, -1))
    if tensor is _PHOTOELASTIC:
        if big.max(initial=0.0) > _PHOTOELASTIC_MAX:
            raise ValueError(
                f"{owner}: photoelastic_tensor has an entry of "
                f"{_short(big.max())}, above {_PHOTOELASTIC_MAX:g}: p is "
                "dimensionless and below 1 in every solid, so 0.12 is not 12 "
                "(percent)."
            )
        return
    if tensor is _SECOND_ORDER:
        if big.max(initial=0.0) > _SECOND_ORDER_MAX:
            raise ValueError(
                f"{owner}: second_order_nonlinear_tensor has an entry of "
                f"{_short(big.max())} m/V, above {_SECOND_ORDER_MAX:g} m/V: d is "
                "in m/V, so 25 pm/V is 25e-12."
            )
        return
    if tensor is _PIEZOELECTRIC:
        small = (big > 0) & (big < _PIEZOELECTRIC_MIN)
        if small.any():
            raise ValueError(
                f"{owner}: piezoelectric_tensor has a largest entry of "
                f"{_short(big[small].flat[0])} C/m^2, below {_PIEZOELECTRIC_MIN:g}: "
                "e is in C/m^2; these look like the strain constants d in C/N, "
                "and e = d c^E."
            )
        return
    soft = big < _STIFFNESS_MIN
    if soft.any():
        raise ValueError(
            f"{owner}: stiffness has a largest entry of {_short(big[soft].flat[0])} "
            f"Pa, below {_STIFFNESS_MIN:g} Pa: the stiffness is in Pa, so 203 GPa "
            "is 203e9."
        )
    skew = np.linalg.norm(arr - np.swapaxes(arr, -2, -1), axis=(-2, -1))
    if (skew > _TENSOR_TOL * np.linalg.norm(arr, axis=(-2, -1))).any():
        raise ValueError(
            f"{owner}: stiffness is not symmetric: c_IJ = c_JI is the major "
            "symmetry c_ijkl = c_klij of an elastic solid."
        )
    low = np.linalg.eigvalsh((arr + np.swapaxes(arr, -2, -1)) / 2)[..., 0]
    if (low <= 0).any():
        raise ValueError(
            f"{owner}: stiffness is not positive definite (smallest eigenvalue "
            f"{_short(low[low <= 0].flat[0])} Pa): the strain energy of a solid "
            "is positive."
        )


def _material_solid_tensor(
    tensor: _SolidTensor, value: Any, owner: str
) -> tuple[tuple[float, ...], ...] | None:
    """A checked constant stiffness or piezoelectric tensor, as rows of floats."""
    if value is None:
        return None
    if is_traced(value):
        raise NonDifferentiableError(
            f"{owner}: {tensor.name} is computed inside jax.grad, jax.jit or "
            f"jax.vmap, but Material values are fixed constants. {_TRACED_REMEDY}"
        )
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise _solid_real(tensor, owner)
    try:
        arr = arr.astype(float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{owner}: {tensor.name} must be {tensor.form}, got {value!r}."
        ) from None
    if arr.shape != tensor.shape:
        raise _solid_shape_error(tensor, owner, arr.shape, field=False)
    _check_solid_values(tensor, arr, owner)
    return tuple(tuple(float(x) for x in row) for row in arr)


def _check_solid(
    owner: str,
    *,
    stiffness: Any,
    piezoelectric: Any,
    quality_factor: Any,
    electro_optic: Any,
    photoelastic: Any = None,
    second_order: Any = None,
) -> None:
    """Raise for a solid's values that do not go together.

    A piezoelectric medium is a solid; a quality factor damps a stiffness;
    a material holding acoustic constants reads its eps_r as the clamped
    static permittivity, which cannot also be the optical permittivity an
    electro-optic tensor reads.
    """
    if electro_optic is not None and (
        stiffness is not None or piezoelectric is not None
    ):
        given = "stiffness" if stiffness is not None else "piezoelectric_tensor"
        raise ValueError(
            f"{owner}: a {given} sits beside an electro_optic_tensor, but the two "
            "read different eps_r: the clamped static permittivity of an "
            "acoustic problem, the optical permittivity at the wavelength of an "
            "electro-optic one. Give the crystal two Materials, an acoustic one "
            "(eps_r, stiffness, piezoelectric_tensor, mass_density) and an "
            "optical one (eps_r, electro_optic_tensor), one for each model."
        )
    if photoelastic is not None and (
        stiffness is not None or piezoelectric is not None
    ):
        given = "stiffness" if stiffness is not None else "piezoelectric_tensor"
        raise ValueError(
            f"{owner}: a {given} sits beside a photoelastic_tensor, but the two "
            "read different eps_r: the clamped static permittivity of an "
            "acoustic problem, the optical permittivity at the wavelength of a "
            "photoelastic one. Give the crystal two Materials, an acoustic one "
            "(eps_r, stiffness, piezoelectric_tensor, mass_density) and an "
            "optical one (eps_r, photoelastic_tensor, electro_optic_tensor), one "
            "for each model."
        )
    if second_order is not None and (
        stiffness is not None or piezoelectric is not None
    ):
        given = "stiffness" if stiffness is not None else "piezoelectric_tensor"
        raise ValueError(
            f"{owner}: a {given} sits beside a second_order_nonlinear_tensor, but "
            "the two read different eps_r: the clamped static permittivity of an "
            "acoustic problem, the optical permittivity at the wavelength of a "
            "nonlinear one. Give the crystal two Materials, an acoustic one "
            "(eps_r, stiffness, piezoelectric_tensor, mass_density) and an "
            "optical one (eps_r, second_order_nonlinear_tensor), one for each "
            "model."
        )
    if piezoelectric is not None and stiffness is None:
        raise ValueError(
            f"{owner}: a piezoelectric_tensor needs a stiffness: a piezoelectric "
            "medium is a solid. Give its stiffness c^E in Pa."
        )
    if quality_factor is not None and stiffness is None:
        raise ValueError(
            f"{owner}: a mechanical_quality_factor needs a stiffness, which it "
            "damps as c (1 + i / Q_m)."
        )


def _piezoelectric_without_eps_message(owner: str) -> str:
    r"""Why a piezoelectric tensor needs the crystal's clamped permittivity.

    The coupling scales as :math:`e^2 / (\varepsilon c)`: an eps_r left at 1
    makes it 29 to 44 times too strong in lithium niobate, with no other
    sign of the omission.
    """
    return (
        f"{owner}: a piezoelectric_tensor needs the crystal's clamped "
        "(constant-strain) static permittivity as eps_r, e.g. eps_r=(44, 44, 29) "
        "with axes for lithium niobate; without it eps_r is 1 and the coupling "
        "e^2 / (eps c) is many times too strong."
    )


# the Lamé constants' patterns: lambda on the normal block, mu with the
# factor 2 of engineering shear on the diagonal
_LAMBDA_PATTERN = np.pad(np.ones((3, 3)), ((0, 3), (0, 3)))
_MU_PATTERN = np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])


def isotropic_stiffness(young_modulus: Any, poisson_ratio: Any) -> Any:
    r"""The 6x6 stiffness of an isotropic solid, for `Material.stiffness`.

    With the Lamé constants :math:`\lambda = E\nu / ((1 + \nu)(1 - 2\nu))`
    and :math:`\mu = E / (2(1 + \nu))`, the upper-left 3x3 block is
    :math:`\lambda` plus :math:`2\mu` on its diagonal and the shear
    diagonal is :math:`\mu`, in the Voigt order xx, yy, zz, yz, xz, xy with
    engineering shear strain. The matrix is the same in every frame, so the
    material takes any ``axes``, or none.

    Plain array arithmetic: JAX inputs give a JAX array, so `jax.grad` with
    respect to :math:`E` and :math:`\nu` flows through a
    `MaterialField.stiffness` built from it.

    Args:
        young_modulus: Young's modulus :math:`E` in Pa, > 0 (70e9 for
            aluminium).
        poisson_ratio: Poisson's ratio :math:`\nu`, in (-1, 1/2).

    Returns:
        The stiffness :math:`c_{IJ}` in Pa, shape ``(6, 6)``: a NumPy array,
        or a JAX array for JAX inputs.

    Raises:
        ValueError: :math:`E \le 0`, or :math:`\nu` outside (-1, 1/2), where
            the stiffness is not positive definite; checked on concrete
            values only.
    """
    jax_input = any(isinstance(v, jax.Array) for v in (young_modulus, poisson_ratio))
    xp = jnp if jax_input else np
    e = xp.asarray(young_modulus, dtype=float)
    nu = xp.asarray(poisson_ratio, dtype=float)
    if not is_traced(e) and not float(e) > 0:
        raise ValueError(
            f"isotropic_stiffness: young_modulus must be > 0 Pa, got {float(e)!r}."
        )
    if not is_traced(nu) and not -1 < float(nu) < 0.5:
        raise ValueError(
            "isotropic_stiffness: poisson_ratio must lie in (-1, 1/2), where the "
            f"stiffness is positive definite, got {float(nu)!r}."
        )
    lame = e * nu / ((1 + nu) * (1 - 2 * nu))
    shear = e / (2 * (1 + nu))
    return lame * _LAMBDA_PATTERN + shear * _MU_PATTERN


def _photoelastic_law_message(owner: str) -> str:
    """Why a photoelastic tensor cannot sit beside a permittivity law."""
    return (
        f"{owner}: a photoelastic_tensor needs a constant eps_r, but eps_r is a "
        "PermittivityLaw, which takes no axes, so the frame of the tensor would "
        "be lost. Give the optical permittivity at the wavelength as a constant "
        "eps_r, with axes."
    )


def _solid_law_message(owner: str, tensor: str) -> str:
    """Why a stiffness or piezoelectric tensor cannot sit beside a permittivity law."""
    return (
        f"{owner}: a {tensor} is given in the frame of axes, but eps_r is a "
        "PermittivityLaw, which takes no axes. Give the static permittivity a "
        "piezoelectric problem reads as a constant eps_r."
    )


def electro_optic_in_frame(r: Any, source: Any, target: Any = None) -> Any:
    r"""A contracted electro-optic tensor moved from one frame to another.

    With :math:`a_l` the rows of ``source`` and :math:`b_p` those of
    ``target``, :math:`r^t_{pqs} = \sum_{lmn} Q_{pl} Q_{qm} Q_{sn}
    r^s_{lmn}`, :math:`Q = b a^T`; ``None`` is the global frame. The
    tensor is ``(..., 6, 3)``, a NumPy array (the result is one) or a JAX
    array (traced values stay traced).
    """
    xp = np if isinstance(r, np.ndarray) else jnp
    a = np.eye(3) if source is None else np.asarray(source, dtype=float)
    b = np.eye(3) if target is None else np.asarray(target, dtype=float)
    q = b @ a.T
    full = r[..., _VOIGT_ROW, :]
    out = xp.einsum("pl,qm,sn,...lmn->...pqs", q, q, q, full)
    return out[..., _VOIGT_I, _VOIGT_J, :]


def stiffness_in_frame(c: Any, source: Any, target: Any = None) -> Any:
    r"""A contracted stiffness moved from one frame to another.

    With :math:`Q = b a^T` as in `electro_optic_in_frame`,
    :math:`c^t_{pqrs} = \sum_{ijkl} Q_{pi} Q_{qj} Q_{rk} Q_{sl}
    c^s_{ijkl}`: the 6x6 matrix is expanded to :math:`c_{ijkl}`, rotated
    and contracted again, which is :math:`M c M^T` with the Bond matrix
    :math:`M`. The stiffness is ``(..., 6, 6)``, a NumPy or a JAX array.
    """
    xp = np if isinstance(c, np.ndarray) else jnp
    q = _frame_rotation(source, target)
    full = c[..., _VOIGT_ROW, :][..., _VOIGT_ROW]
    out = xp.einsum("pi,qj,rk,sl,...ijkl->...pqrs", q, q, q, q, full)
    return out[..., _VOIGT_I, _VOIGT_J, :, :][..., _VOIGT_I, _VOIGT_J]


def piezoelectric_in_frame(e: Any, source: Any, target: Any = None) -> Any:
    r"""A contracted piezoelectric tensor moved from one frame to another.

    :math:`e^t_{pqr} = \sum_{kij} Q_{pk} Q_{qi} Q_{rj} e^s_{kij}`, with
    :math:`Q` as in `stiffness_in_frame`: :math:`Q e M^T`. The tensor is
    ``(..., 3, 6)``, a NumPy or a JAX array.
    """
    xp = np if isinstance(e, np.ndarray) else jnp
    q = _frame_rotation(source, target)
    out = xp.einsum("pk,qi,rj,...kij->...pqr", q, q, q, e[..., _VOIGT_ROW])
    return out[..., _VOIGT_I, _VOIGT_J]


def _frame_rotation(source: Any, target: Any) -> np.ndarray:
    """:math:`Q = b a^T` of the rows of ``source`` and ``target`` (``None``: global)."""
    a = np.eye(3) if source is None else np.asarray(source, dtype=float)
    b = np.eye(3) if target is None else np.asarray(target, dtype=float)
    return b @ a.T


def _bound_error(
    prop: _Property, owner: str, value: Any, axis: str | None = None
) -> ValueError:
    """The passivity error; ``axis`` names a principal axis for a tensor value."""
    bound = "> 0" if prop.strict else ">= 0"
    if prop.physics != "em" or prop.name in _NONLINEAR:
        unit = "" if prop.unit == "1" else f" {prop.unit}"
        return ValueError(f"{owner}: {prop.name} must be {bound}{unit}, got {value}.")
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
    if not prop.bounded:
        return
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


def _refuse_law_value(prop: _Property, value: Any, owner: str) -> None:
    """Raise if ``value``, a constant property's value, is a law or a term."""
    if not _is_law(value):
        return
    if prop.name != "eps_r":
        raise ValueError(
            f"{owner}: {prop.name} cannot be a permittivity law or term; "
            "only eps_r can depend on frequency."
        )
    if isinstance(value, _DispersionTerm):
        raise TypeError(
            f"{owner}: eps_r is a bare {type(value).__name__} term, a "
            "susceptibility rather than a permittivity. Give "
            f"eps_r=gfem.PermittivityLaw(eps_inf=..., terms=[{value!r}])."
        )
    raise ValueError(f"{owner}: {_ANISOTROPIC_LAW}.")


def _material_value(prop: _Property, value: Any, owner: str) -> Any:
    """A checked value: a float, a 3-tuple, or a symmetric 3x3 matrix.

    A matrix is returned as a symmetrized array, its passivity unchecked:
    `_principal` diagonalizes it with the material's other matrices.
    """
    if value is None:
        if prop.default is not None:
            raise ValueError(f"{owner}: {prop.name} cannot be None.")
        return None
    _refuse_law_value(prop, value, owner)
    if is_traced(value):
        raise NonDifferentiableError(
            f"{owner}: {prop.name} is computed inside jax.grad, jax.jit or jax.vmap, "
            f"but Material values are fixed constants. {_TRACED_REMEDY}"
        )
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        remedy = {
            "eps_r": "; give dielectric loss with tan_delta or conductivity, or a "
            "complex value through gfem.MaterialField",
            "mu_r": "; for magnetic loss give a complex mu_r through "
            "gfem.MaterialField",
        }.get(prop.name, "")
        raise ValueError(f"{owner}: {prop.name} must be real{remedy}.")
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
        if prop.bounded and ((arr <= 0) if prop.strict else (arr < 0)):
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


def _off_diagonal(matrix: np.ndarray, axes: np.ndarray) -> bool:
    """Whether ``matrix`` is not diagonal along the rows of ``axes`` (`_TENSOR_TOL`)."""
    d = axes @ matrix @ axes.T
    scale = _TENSOR_TOL * np.linalg.norm(matrix)
    return bool(np.abs(d - np.diag(np.diag(d))).max() > scale)


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

    if not any(_off_diagonal(m, np.eye(3)) for m in matrices.values()):
        a = None
    else:
        a = _ordered_axes(_eigenspaces(list(matrices.values())))
        for name, m in matrices.items():
            if _off_diagonal(m, a):
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
        prop = _BY_NAME[name]
        v = np.diag(m) if a is None else np.einsum("ki,ij,kj->k", a, m, a)
        _check_passive(prop, v, owner, a)
        if np.ptp(v) <= _TENSOR_TOL * np.linalg.norm(m):
            # equal to 1e-9, not an approximation; the mean keeps the trace
            out[name] = float(np.mean(v))
        else:
            out[name] = tuple(float(x) for x in v)
            tensor = True
    return out, (a if tensor and a is not None else None)


# --- permittivity laws -----------------------------------------------------------

_GAIN = (
    " A term with a negative strength or damping is a gain; with every "
    "strength and damping >= 0 each term, and so the law, is passive "
    "(Im eps <= 0 at every real frequency under e^{+iωt}, see "
    "gfem.conventions)."
)


def _law_number(owner: str, name: str, value: Any, bound: str, unit: str) -> Any:
    """A checked law parameter: a float, or ``value`` itself when traced.

    ``bound`` is ``">"`` (> 0) or ``">="`` (>= 0). A traced value skips the
    check; the host re-checks the concrete law before assembly.
    """
    if is_traced(value):
        return value
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise ValueError(f"{owner}: {name} must be real, got {value!r}.")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{owner}: {name} must be a number, got {value!r}.")
    arr = arr.astype(float)
    if arr.ndim != 0:
        raise ValueError(f"{owner}: {name} must be a single number, got {value!r}.")
    if not np.isfinite(arr):
        raise ValueError(f"{owner}: {name} must be finite, got {value!r}.")
    x = float(arr)
    if (x <= 0) if bound == ">" else (x < 0):
        shown = f"{_short(x)}{' ' + unit if unit else ''}"
        what = f"{name}={shown} but {name} must be {bound} 0"
        if name in ("delta_eps", "damping", "plasma_frequency"):
            raise ValueError(f"{owner} is not passive: {what}.{_GAIN}")
        raise ValueError(f"{owner}: {what}.")
    return x


def _fitted_range(value: Any) -> tuple[float, float] | None:
    r"""A checked fitted range: ``None``, or two floats in Hz.

    The two are :math:`(f_\mathrm{low}, f_\mathrm{high})` with
    :math:`0 < f_\mathrm{low} < f_\mathrm{high}`.

    The range is static pytree data, so a traced value or an array (whose
    entries could be traced, and which does not hash) is refused.
    """
    if value is None:
        return None
    message = f"frequency_range takes two numbers in Hz, got {value!r}."
    if is_traced(value) or isinstance(value, (np.ndarray, jax.Array)):
        raise TypeError(message)
    try:
        pair = tuple(value)
    except TypeError:
        raise TypeError(message) from None
    if len(pair) != 2:
        raise TypeError(message)
    numbers = []
    for x in pair:
        if is_traced(x) or isinstance(x, bool) or np.ndim(x) != 0:
            raise TypeError(message)
        kind = np.asarray(x).dtype.kind
        if kind not in "iuf":
            raise TypeError(message)
        numbers.append(float(x))
    lo, hi = numbers
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError(
            f"PermittivityLaw: frequency_range=({lo!r}, {hi!r}) must be two "
            "finite frequencies in Hz with 0 < low < high."
        )
    return lo, hi


class _DispersionTerm:
    """A dispersion term: a susceptibility added to a law's eps_inf.

    Each subclass is a frozen keyword-only dataclass whose fields are all
    numbers (the pytree leaves), and ``_bounds`` gives each field's bound and
    unit.
    """

    _bounds: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        owner = f"{type(self).__name__} term"
        for name, bound, unit in self._bounds:
            value = _law_number(owner, name, getattr(self, name), bound, unit)
            object.__setattr__(self, name, value)

    def __call__(self, frequency: Any) -> jax.Array:
        """This term's contribution to the relative permittivity.

        Args:
            frequency: Frequency in Hz: a number or an array (the result
                broadcasts), real or complex (the analytic continuation; under
                e^{+iωt} a decaying mode has Im f > 0).

        Returns:
            The complex susceptibility, complex128, the shape of ``frequency``.
        """
        return self._value(jnp.asarray(frequency).astype(jnp.complex128))

    def _value(self, f: jax.Array) -> jax.Array:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drude(_DispersionTerm):
    r"""A free-electron (Drude) term of a `PermittivityLaw`.

    Under :math:`e^{+i\omega t}` it adds
    :math:`-\omega_p^2 / (\omega (\omega - i\gamma))`, with
    :math:`\omega = 2\pi f`, :math:`\omega_p = 2\pi \cdot`
    ``plasma_frequency`` and :math:`\gamma = 2\pi \cdot` ``damping``; under
    :math:`e^{-i\omega t}` (most optics papers) the same term is its complex
    conjugate, with :math:`+i\gamma`. See `gfem.conventions` (Dispersive
    permittivity).

    It has no static permittivity: it diverges as :math:`f \to 0`. A fit in
    eV converts as :math:`f = E / h` (:math:`\hbar\omega_p` = 9.03 eV is
    2.183e15 Hz).

    Attributes:
        plasma_frequency: The plasma frequency, Hz, > 0.
        damping: The collision frequency :math:`\gamma / 2\pi`, Hz, >= 0.

    Raises:
        ValueError: A parameter is not a finite real number, or out of its
            bound.
    """

    plasma_frequency: float
    damping: float

    _bounds = (("plasma_frequency", ">", "Hz"), ("damping", ">=", "Hz"))

    def __call__(self, frequency: Any) -> jax.Array:
        """This term's contribution at ``frequency`` (Hz); see `_DispersionTerm`.

        Raises:
            ValueError: A frequency is 0: a Drude term has no static permittivity.
        """
        if not is_traced(frequency) and np.any(np.asarray(frequency) == 0):
            raise ValueError(
                "A Drude term has no static permittivity: it diverges as the "
                "frequency goes to 0 (a free-electron metal screens a static "
                "field). Evaluate it at a frequency > 0."
            )
        return super().__call__(frequency)

    def _value(self, f: jax.Array) -> jax.Array:
        fp, g = self.plasma_frequency, self.damping
        return -(fp**2) / (f * (f - 1j * g))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Lorentz(_DispersionTerm):
    r"""A Lorentz term of a `PermittivityLaw`: a damped bound-charge resonance.

    Under :math:`e^{+i\omega t}` it adds
    :math:`\Delta\varepsilon\, \omega_0^2 /
    (\omega_0^2 - \omega^2 + i\gamma\omega)`, with :math:`\omega = 2\pi f`,
    :math:`\omega_0 = 2\pi \cdot`
    ``resonance_frequency`` and :math:`\gamma = 2\pi \cdot` ``damping``;
    under :math:`e^{-i\omega t}` it is the complex conjugate
    (:math:`-i\gamma\omega`). See `gfem.conventions` (Dispersive
    permittivity). Its static value is ``delta_eps``. A Sellmeier term
    :math:`B \lambda^2 / (\lambda^2 - C)` is a lossless Lorentz term with
    ``delta_eps = B`` and ``resonance_frequency = c / sqrt(C)``.

    Attributes:
        delta_eps: The strength, > 0 (dimensionless).
        resonance_frequency: The resonance frequency, Hz, > 0.
        damping: The damping :math:`\gamma / 2\pi`, Hz, >= 0 (0: lossless).

    Raises:
        ValueError: A parameter is not a finite real number, or out of its
            bound (a negative strength is a gain).
    """

    delta_eps: float
    resonance_frequency: float
    damping: float

    _bounds = (
        ("delta_eps", ">", ""),
        ("resonance_frequency", ">", "Hz"),
        ("damping", ">=", "Hz"),
    )

    def _value(self, f: jax.Array) -> jax.Array:
        f0 = self.resonance_frequency
        return self.delta_eps * f0**2 / (f0**2 - f**2 + 1j * self.damping * f)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Debye(_DispersionTerm):
    r"""A relaxation (Debye) term of a `PermittivityLaw`.

    Under :math:`e^{+i\omega t}` it adds
    :math:`\Delta\varepsilon / (1 + i f / f_r)`; under :math:`e^{-i\omega t}`
    it is the complex conjugate. A relaxation time :math:`\tau` converts as
    :math:`f_r = 1 / (2\pi\tau)`. See `gfem.conventions` (Dispersive
    permittivity).
    Its static value is ``delta_eps``.

    Attributes:
        delta_eps: The strength (static minus high-frequency permittivity),
            > 0.
        relaxation_frequency: The relaxation frequency :math:`f_r`, Hz, > 0.

    Raises:
        ValueError: A parameter is not a finite real number, or out of its
            bound (a negative strength is a gain).
    """

    delta_eps: float
    relaxation_frequency: float

    _bounds = (("delta_eps", ">", ""), ("relaxation_frequency", ">", "Hz"))

    def _value(self, f: jax.Array) -> jax.Array:
        return self.delta_eps / (1 + 1j * f / self.relaxation_frequency)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DjordjevicSarkar(_DispersionTerm):
    r"""A wideband (Djordjevic-Sarkar) term of a `PermittivityLaw`.

    Under :math:`e^{+i\omega t}` it adds
    :math:`\frac{\Delta\varepsilon}{\ln(f_2 / f_1)} \ln\frac{f_2 + i f}{f_1 + i f}`,
    the principal log of the ratio (a continuum of Debye relaxations between
    the corners, with a nearly constant loss tangent between them); under
    :math:`e^{-i\omega t}` it is the
    complex conjugate. The corners in Hz are exact: the form depends on
    frequency ratios only. Its static value is ``delta_eps``. RF datasheets
    give a permittivity and a loss tangent at one frequency: use
    `PermittivityLaw.djordjevic_sarkar` to fit this term to them. See
    `gfem.conventions` (Dispersive permittivity).

    Attributes:
        delta_eps: The strength, > 0.
        low_frequency: The lower corner :math:`f_1`, Hz, > 0.
        high_frequency: The upper corner :math:`f_2`, Hz, :math:`> f_1`.

    Raises:
        ValueError: A parameter is not a finite real number, or out of its
            bound (a negative strength is a gain), or f1 >= f2.
    """

    delta_eps: float
    low_frequency: float
    high_frequency: float

    _bounds = (
        ("delta_eps", ">", ""),
        ("low_frequency", ">", "Hz"),
        ("high_frequency", ">", "Hz"),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        f1, f2 = self.low_frequency, self.high_frequency
        if not is_traced((f1, f2)) and f1 >= f2:
            raise ValueError(
                f"DjordjevicSarkar term: low_frequency={_short(f1)} Hz must be "
                f"below high_frequency={_short(f2)} Hz."
            )

    def _value(self, f: jax.Array) -> jax.Array:
        f1, f2 = self.low_frequency, self.high_frequency
        return (
            self.delta_eps / jnp.log(f2 / f1) * jnp.log((f2 + 1j * f) / (f1 + 1j * f))
        )


_TERMS = (Drude, Lorentz, Debye, DjordjevicSarkar)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PermittivityLaw:
    r"""A relative permittivity that depends on frequency: eps_inf plus terms.

    The value of `Material`'s ``eps_r`` for a dispersive material:

    .. math::

        \varepsilon_r(f) = \varepsilon_\infty + \sum \mathrm{term}(f)

    with :math:`f` in Hz, under :math:`e^{+i\omega t}`, so a lossy law has
    :math:`\mathrm{Im}\,\varepsilon_r < 0`. Every term is passive alone
    (strengths and damping >= 0), so the law is passive at every real
    frequency. The forms, and how to convert a fit written under
    :math:`e^{-i\omega t}`, are in `gfem.conventions` (Dispersive
    permittivity).

    A law is callable: ``law(f)`` is the complex relative permittivity at
    ``f`` (Hz, broadcast, complex accepted as the analytic continuation),
    computed in `jax.numpy`; ``law(0)`` is the static permittivity. A law
    is a JAX pytree whose leaves are its numbers, so `jax.grad` through a
    model reaches each parameter, in its own units (per Hz for a
    frequency).

    Example:
        Gold (a Drude fit) and an FR-4 datasheet value::

            gold = gfem.Material(
                eps_r=gfem.PermittivityLaw(
                    terms=[gfem.Drude(plasma_frequency=2.183e15, damping=1.28e13)]
                ),
                name="gold",
            )
            fr4 = gfem.Material(
                eps_r=gfem.PermittivityLaw.djordjevic_sarkar(4.3, 0.02, frequency=1e9)
            )

    Attributes:
        eps_inf: The high-frequency permittivity, > 0.
        terms: The dispersion terms (`Drude`, `Lorentz`, `Debye`,
            `DjordjevicSarkar`), at least one, stored as a tuple.
        frequency_range: The law's fitted range ``(low, high)`` in Hz: the
            frequencies over which its parameters were matched to data or
            to a source's formula, or ``None`` (the default) for a law
            with no such range. Outside it the law stays passive but its
            accuracy is unknown, so a solve that reads it there warns
            `gfem.FittedRangeWarning` and runs on; ``law(f)`` itself stays
            silent. `dataclasses.replace` keeps the range: once a law has
            been edited, or checked outside its range, clear it with
            ``dataclasses.replace(law, frequency_range=None)``. It is
            static data, not a JAX leaf.

    Raises:
        ValueError: ``eps_inf`` is not a finite real number > 0, or
            ``terms`` is empty (a constant permittivity is ``eps_r=<number>``),
            or ``frequency_range`` is not finite with
            :math:`0 < f_\mathrm{low} < f_\mathrm{high}`.
        TypeError: An entry of ``terms`` is not a dispersion term, or
            ``frequency_range`` is not two numbers (an array or a value
            under a JAX transformation is refused: the range is static).
    """

    eps_inf: float = 1.0
    terms: tuple[Drude | Lorentz | Debye | DjordjevicSarkar, ...]
    frequency_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        eps_inf = _law_number("PermittivityLaw", "eps_inf", self.eps_inf, ">", "")
        object.__setattr__(self, "eps_inf", eps_inf)
        try:
            terms = tuple(self.terms)
        except TypeError:
            raise TypeError(
                "PermittivityLaw: terms takes a list of terms, e.g. "
                f"terms=[{self.terms!r}]."
            ) from None
        if not terms:
            raise ValueError(
                "PermittivityLaw needs at least one term; a permittivity that "
                "does not depend on frequency is eps_r=<number>."
            )
        for term in terms:
            if not isinstance(term, _TERMS):
                raise TypeError(
                    "PermittivityLaw: each term must be a gfem.Drude, Lorentz, "
                    f"Debye or DjordjevicSarkar, got {term!r}."
                )
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "frequency_range", _fitted_range(self.frequency_range))

    def __call__(self, frequency: Any) -> jax.Array:
        """The complex relative permittivity at ``frequency``.

        Args:
            frequency: Frequency in Hz: a number or an array (the result
                broadcasts), real or complex (the analytic continuation).

        Returns:
            ``eps_inf + Σ term(frequency)``, complex128.

        Raises:
            ValueError: A frequency is 0 and the law has a `Drude` term
                (no static permittivity).
        """
        f = jnp.asarray(frequency)
        return self.eps_inf + sum((t(f) for t in self.terms), jnp.zeros(f.shape))

    @classmethod
    def djordjevic_sarkar(
        cls,
        eps_r: float,
        tan_delta: float,
        *,
        frequency: float,
        low_frequency: float = 1e3,
        high_frequency: float = 1e12,
    ) -> PermittivityLaw:
        """A wideband law fitted to a datasheet permittivity and loss tangent.

        Returns ``eps_inf`` plus one `DjordjevicSarkar` term such that the
        law equals ``eps_r (1 - i tan_delta)`` at ``frequency``. The corner
        defaults (1 kHz, 1 THz) are those of KiCad and scikit-rf.

        Args:
            eps_r: The datasheet relative permittivity, > 0.
            tan_delta: The datasheet loss tangent, > 0.
            frequency: Where the datasheet gives them, Hz.
            low_frequency: The lower corner, Hz.
            high_frequency: The upper corner, Hz.

        Raises:
            ValueError: A value is out of range, or the fitted eps_inf is
                <= 0 (the loss is too large for the permittivity between
                these corners).
        """
        owner = "PermittivityLaw.djordjevic_sarkar"
        eps = _law_number(owner, "eps_r", eps_r, ">", "")
        tan = _law_number(owner, "tan_delta", tan_delta, ">", "")
        f = _law_number(owner, "frequency", frequency, ">", "Hz")
        unit = complex(
            DjordjevicSarkar(
                delta_eps=1.0,
                low_frequency=low_frequency,
                high_frequency=high_frequency,
            )(f)
        )
        delta = eps * tan / -unit.imag
        eps_inf = eps - delta * unit.real
        if eps_inf <= 0:
            raise ValueError(
                f"{owner}: eps_r={_short(eps)} with tan_delta={_short(tan)} at "
                f"{_short(f)} Hz needs eps_inf={_short(eps_inf)}, but eps_inf "
                "must be > 0; check the datasheet values or widen the corners."
            )
        return cls(
            eps_inf=eps_inf,
            terms=(
                DjordjevicSarkar(
                    delta_eps=delta,
                    low_frequency=low_frequency,
                    high_frequency=high_frequency,
                ),
            ),
        )


for _cls in _TERMS:
    register_fields(_cls, tuple(f.name for f in dataclasses.fields(_cls)))
register_fields(PermittivityLaw, ("eps_inf", "terms"))

#: How close to an edge of a fitted range, relative, a frequency still counts
#: as inside: it absorbs the rounding of a frequency computed as c / wavelength.
#: There is no other margin, because each edge is the source's own.
FITTED_EDGE = 1e-9


def inside_fitted_range(frequency_range: tuple[float, float], frequency: float) -> bool:
    """Whether ``frequency`` (Hz) lies in a law's fitted range, edges included.

    A frequency within `FITTED_EDGE` of an edge, relative, counts as inside.
    """
    lo, hi = frequency_range
    return lo * (1 - FITTED_EDGE) <= frequency <= hi * (1 + FITTED_EDGE)


_ANISOTROPIC_LAW = (
    "anisotropic dispersive materials are not supported: a permittivity law "
    "is isotropic, beside one scalar mu_r and conductivity and no axes"
)


def _renamed_density[T: type](cls: T) -> T:
    """Map the deprecated keyword ``density=`` to ``mass_density=``.

    The keyword warns for one release; the attribute ``.density`` is gone
    (reading it raises `AttributeError` naming `mass_density`), because a
    read alias would keep the clash with `MaterialField.interpolate`'s
    design density.
    """
    init = cls.__init__
    name = cls.__name__

    @functools.wraps(init)
    def init_with_alias(self: Any, *args: Any, **kwargs: Any) -> None:
        if "density" in kwargs:
            if "mass_density" in kwargs:
                raise TypeError(
                    f"{name}: give mass_density only; density= is its deprecated name."
                )
            warnings.warn(
                f"{name}(density=...) is deprecated: the mass density is now "
                "mass_density=..., in kg/m^3 (density= will be removed).",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["mass_density"] = kwargs.pop("density")
        init(self, *args, **kwargs)

    cls.__init__ = init_with_alias  # ty: ignore[invalid-assignment]
    return cls


def _no_density_attribute(self: Any) -> Any:
    raise AttributeError(
        f"{type(self).__name__} has no attribute 'density': the mass density "
        "is 'mass_density' (kg/m^3)."
    )


def _is_law(value: Any) -> bool:
    """Whether ``value`` is a law or a term, or a sequence holding one."""
    if isinstance(value, (PermittivityLaw, _DispersionTerm)):
        return True
    return isinstance(value, (tuple, list)) and any(_is_law(v) for v in value)


def _law_material_values(material: Material, owner: str) -> dict[str, Any]:
    """The checked values of a `Material` whose ``eps_r`` is a law."""
    if material.axes is not None:
        raise ValueError(f"{owner}: {_ANISOTROPIC_LAW}; leave axes as None.")
    values: dict[str, Any] = {"eps_r": material.eps_r}
    for p in PROPERTIES[1:]:
        value = _material_value(p, getattr(material, p.name), owner)
        if p.may_be_tensor and isinstance(value, (tuple, np.ndarray)):
            raise ValueError(f"{owner}: {_ANISOTROPIC_LAW}; {p.name} is a tensor.")
        values[p.name] = value
    if values["tan_delta"] != 0:
        raise ValueError(
            f"{owner}: tan_delta must be 0 beside a permittivity law: a constant "
            "loss tangent is not causal over a wide band. Use "
            "gfem.PermittivityLaw.djordjevic_sarkar(eps_r=, tan_delta=, "
            "frequency=) for a datasheet loss tangent, or add a Debye or "
            "Lorentz damping term."
        )
    return values


@_renamed_density
@dataclasses.dataclass(frozen=True, kw_only=True)
class Material:
    r"""A linear medium, constant over the region it is assigned to.

    Keyword-only. Units are SI. Under :math:`e^{+i\omega t}` the complex
    permittivity is, per principal axis :math:`a_k` (the rows of `axes`),

    .. math::

        \varepsilon = \varepsilon_0 \sum_k \left[\varepsilon_{r,k}
            (1 - i \tan\delta_k) - i \frac{\sigma_k}{\varepsilon_0 \omega}\right]
            a_k a_k^T

    with :math:`\varepsilon_{r,k}`, :math:`\tan\delta_k` and
    :math:`\sigma_k` the k-th values of `eps_r`, `tan_delta` and
    `conductivity`; a single value counts for all three axes. The two loss
    models, `tan_delta` and `conductivity`, add.

    A dispersive material's `eps_r` is a `PermittivityLaw`: then
    :math:`\varepsilon = \varepsilon_0 [\varepsilon_r(f) - i \sigma /
    (\varepsilon_0 \omega)]`, isotropic, with `tan_delta`
    0 (the law carries the dielectric loss), one `mu_r` and one
    `conductivity`, and no `axes`.

    Anisotropic values are real symmetric tensors that share one set of
    principal axes; non-symmetric (gyrotropic) tensors are not supported. A
    wave port's cross-section accepts an anisotropic material only with a
    principal axis along the port normal (see `WavePort`).

    Attributes:
        eps_r: Relative permittivity: one value, three principal values
            along `axes`, a real symmetric 3x3 matrix in the global frame, or
            a `PermittivityLaw` (a dispersive material).
        mu_r: Relative permeability, in the same forms as `eps_r`.
        tan_delta: Dielectric loss tangent, >= 0, in the same forms as
            `eps_r` (an anisotropic material such as sapphire). Each principal
            value of the permittivity is then ``eps_r[k] (1 - i tan_delta[k])``.
        conductivity: The electrical conductivity in S/m, in the same forms
            as `eps_r`, each principal value >= 0 (not the thermal
            conductivity, which is `thermal_conductivity`).
        axes: Three orthonormal vectors, as rows: the principal directions
            shared by `eps_r`, `mu_r`, `tan_delta` and `conductivity`.
            ``None``: the coordinate axes. At a wave port one of them must
            lie along the port normal (see `WavePort`). With a matrix
            input, leave ``None``: the material stores the matrix as
            principal values along derived axes (axis k the one closest to
            coordinate axis k, its dominant component positive; ``None`` if
            every matrix is diagonal). Equal principal values are stored as
            one value.
        thermal_conductivity: The thermal conductivity in W/(m K), > 0,
            read by thermal problems. ``None`` means not given.
        mass_density: The mass density in kg/m^3, > 0, read by transient
            thermal and piezoelectric problems (``density=`` is its
            deprecated name, kept for one release with a
            `DeprecationWarning`). ``None`` means not given.
        specific_heat: The specific heat capacity in J/(kg K), > 0, read
            by transient thermal problems. ``None`` means not given.
        thermo_optic_coefficient: dn/dT, the change of the refractive index
            with temperature, in 1/K: one real number of any sign.
            ``None`` means the index does not change with temperature.
        nonlinear_index: The Kerr index :math:`n_2`, the change of the
            refractive index per unit intensity
            (:math:`\delta n = n_2 I`), in m^2/W: one real number of any
            sign, one scalar: in a birefringent crystal (three `eps_r`
            values along `axes`) it is the change :math:`n_2 I` of the index
            of light polarized along each principal axis. Read by
            `gfem.kerr_nonlinearity` and
            `gfem.kerr_coupling` only, never by a solve. ``None`` means no
            Kerr effect.
        two_photon_absorption: The two-photon absorption coefficient
            :math:`\beta_\mathrm{TPA}`, the intensity-dependent loss
            :math:`\delta\alpha = \beta_\mathrm{TPA} I`, in m/W: one real
            number, :math:`\ge 0`. Read by `gfem.kerr_nonlinearity` and
            `gfem.kerr_coupling` only. ``None`` means none.
        nonlinear_anisotropy: The anisotropy :math:`\rho =
            3\chi_{1122}/\chi_{1111}` of a cubic crystal's Kerr effect (1.27
            for silicon at 1.2 to 2.4 um), one real number, the same for
            `nonlinear_index` and `two_photon_absorption`, which then refer to
            light polarized along a crystal axis (a row of `axes`, ``[100]``)
            rather than to ``[110]``: :math:`n_2^{[100]} = 2
            n_2^{[110]}/(1+\rho)`. A field along the unit vector :math:`u`
            of the crystal frame sees
            :math:`\rho + (1-\rho)\sum_s u_s^4` times that value. Read by
            `gfem.kerr_nonlinearity` and `gfem.kerr_coupling` only. ``None``
            means isotropic (:math:`\rho = 1`, whatever `axes`).
        raman_spectrum: The bulk Raman gain, a `gfem.RamanSpectrum` (its
            ``symmetry`` is ``"cubic"`` for silicon, on this material's
            `axes`, or ``"isotropic"``). Read by `gfem.raman_gain` only,
            never by a solve. ``None`` means no Raman gain.
        electro_optic_tensor: The linear electro-optic (Pockels) tensor
            :math:`r_{Ik}` in m/V, contracted to a 6x3 matrix: rows
            :math:`I` in the order xx, yy, zz, yz, xz, xy, columns
            :math:`k` the field components x, y, z, so that
            :math:`\Delta(\varepsilon_r^{-1})_{ij} = r_{ijk} E_k`. It is
            given in the frame whose x, y, z are the rows of `axes`, in
            order and with sign (the global frame when `axes` is ``None``):
            an x-cut and a z-cut film differ by their `axes`. Enter the
            clamped (constant-strain) coefficients for thin films and for
            drives above the acoustic resonances. With it, every tensor
            property is one value or three principal values (a matrix
            raises), `axes` are right-handed, and `eps_r` is a constant.
            It leaves `isotropic` unchanged. ``None``: no electro-optic
            effect.
        stiffness: The stiffness :math:`c^E_{IJ}` of a solid at constant
            electric field, in Pa, contracted to a symmetric positive
            definite 6x6 matrix: rows and columns :math:`I, J` in the order
            xx, yy, zz, yz, xz, xy, with engineering shear strain, so that
            :math:`c_{IJ} = c_{ijkl}` with no factors of 2, as datasheets
            print it. A piezoelectric problem adds the piezoelectric
            stiffening itself: enter :math:`c^E`, not :math:`c^D`. Given in
            the frame of `axes`, as `electro_optic_tensor` is, with the same
            rules beside it (a reflection changes :math:`c_{14}` of a trigonal
            crystal). `isotropic_stiffness` builds an isotropic solid's.
            Read by piezoelectric problems. ``None``: not a solid.
        piezoelectric_tensor: The piezoelectric stress constants
            :math:`e_{kJ}` in C/m^2 (not the strain constants :math:`d` in
            C/N; :math:`e = d c^E`), contracted to a 3x6 matrix: rows
            :math:`k` the field components x, y, z, columns :math:`J` in the
            Voigt order of `stiffness`, so that :math:`T = c^E S - e^T E`
            and :math:`D = e S + \varepsilon^S E`. Given in the frame of
            `axes`, beside a `stiffness`; `eps_r` is then the clamped
            (constant-strain) static permittivity :math:`\varepsilon^S`.
            ``None``: not piezoelectric.
        mechanical_quality_factor: The mechanical quality factor
            :math:`Q_m`, > 0, beside a `stiffness`: the solid's loss, which
            makes the stiffness :math:`c (1 + i / Q_m)` under
            :math:`e^{+i\omega t}`. ``None``: lossless.
        photoelastic_tensor: The photoelastic tensor :math:`p^E_{IJ}`,
            dimensionless, contracted to a real 6x6 matrix: rows :math:`I` in
            the Voigt order xx, yy, zz, yz, xz, xy of the inverse relative
            permittivity (tensor components), columns :math:`J` in the same
            order of the engineering strain, so that
            :math:`\Delta(\varepsilon_r^{-1})_I = \sum_J p_{IJ} S_J` and
            :math:`p_{IJ} = p_{ijkl}` with no factors of 2, as Nye and the
            compiled tables print it. It is not symmetric in general
            (:math:`p_{14} \neq p_{41}` in a 3m crystal). Given in the frame of
            `axes` like the `stiffness`, rotated as one, on the optical
            `Material` of a crystal (`eps_r` the optical permittivity) beside
            its `electro_optic_tensor`; the acoustic `Material` of the same
            crystal holds the `stiffness`. Enter the constant-field
            coefficients :math:`p^E`: the electro-optic part of an acoustic
            field is counted through the clamped `electro_optic_tensor`.
            ``None``: no photoelastic effect.
        second_order_nonlinear_tensor: The second-order nonlinear tensor
            :math:`d_{il}` in m/V, with :math:`d = \chi^{(2)}/2` (the
            convention of the data sheets and of Boyd, not :math:`\chi^{(2)}`
            itself), contracted to a real 3x6 matrix: rows :math:`i` the
            components x, y, z of the polarization, columns :math:`l` in the
            Voigt order xx, yy, zz, yz, xz, xy of the field pair, so that
            :math:`d_{il} = d_{ijk}` with no factor of 2 on the shear
            columns. Given in the frame whose x, y, z are the rows of `axes`,
            in order and with sign (the global frame if ``None``), rotated as
            the `piezoelectric_tensor` is; with it `axes` are right-handed
            (an odd-rank tensor changes sign under a reflection) and `eps_r`
            is the crystal's optical permittivity. A `PermittivityLaw` is
            accepted (the tensor is then in the global frame, since a law
            takes no `axes`). Read by `gfem.three_wave_coupling` and
            `gfem.second_harmonic` only, never by a solve. ``None``: no
            second-order nonlinearity.
        name: A free-form label (for example ``"silicon"``), shown in reprs.

    Raises:
        ValueError: The medium is not passive (a principal value of eps_r or
            mu_r <= 0, or of a loss < 0), a value is not finite, `axes` is
            not orthonormal, a matrix is not symmetric (gyrotropic tensors are
            not supported), a matrix is given with `axes` or beside three
            principal values, or the matrices do not share principal axes;
            beside a `PermittivityLaw`, a nonzero `tan_delta`, `axes`, or a
            tensor `mu_r` or `conductivity`; a law or term as `mu_r` (only
            eps_r depends on frequency); a thermal property <= 0; a
            thermal property, `thermo_optic_coefficient` or `nonlinear_index`
            that is not one real number; a `two_photon_absorption` < 0; an
            `electro_optic_tensor` that is not a real, finite 6x3 matrix, has
            an entry above 1e-8 m/V (a value in pm/V), or sits beside a
            matrix property, left-handed `axes`, a `PermittivityLaw` or an
            `eps_r` of 1 (the crystal's optical permittivity left out); a
            `nonlinear_anisotropy` that is not one finite real number,
            sits beside an anisotropic `eps_r` (cubic crystals are optically
            isotropic) or a `PermittivityLaw` (no `axes`), or has neither a `nonlinear_index` nor a
            `two_photon_absorption`; a `raman_spectrum` that holds a
            traced gain (`NonDifferentiableError`), sits beside an
            anisotropic `eps_r`, or is cubic beside a `PermittivityLaw`; a
            `stiffness` that is not a real,
            finite, symmetric positive definite 6x6 matrix, or whose largest
            entry is below 1e5 Pa (a value in GPa); a `piezoelectric_tensor`
            that is not a real, finite 3x6 matrix, or whose nonzero largest
            entry is below 1e-6 C/m^2 (strain constants d); a
            `piezoelectric_tensor` or a `mechanical_quality_factor` without
            a `stiffness`, or a `mechanical_quality_factor` <= 0; a
            `photoelastic_tensor` that is not a real, finite 6x6 matrix or
            has an entry above 10 in magnitude, or sits beside a `stiffness`
            or `piezoelectric_tensor`, a matrix property, left-handed `axes`
            or a `PermittivityLaw`; either
            tensor beside a matrix property, left-handed `axes`, a
            `PermittivityLaw` or an `electro_optic_tensor` (give the crystal
            an acoustic and an optical `Material`); a `piezoelectric_tensor`
            with an `eps_r` of 1 (the clamped permittivity left out); a
            `second_order_nonlinear_tensor` that is not a real, finite 3x6
            matrix (a 3x3x3 input is refused: contract it), has an entry
            above 1e-8 m/V (a value in pm/V), or sits beside a `stiffness`
            or `piezoelectric_tensor`, a matrix property, left-handed `axes`
            or an `eps_r` of 1.
        TypeError: A `raman_spectrum` that is not a `gfem.RamanSpectrum`;
            `eps_r` is a bare dispersion term such as `Drude`; wrap
            it in a `PermittivityLaw`; ``density=`` beside ``mass_density=``.
        NonDifferentiableError: A constant value is computed inside
            `jax.grad`, `jax.jit` or `jax.vmap` from their arguments; pass a
            fixed value, or use `MaterialField.uniform` to differentiate with
            respect to a region's material. A law's parameters are
            differentiable: the law is a pytree.
    """

    eps_r: float | tuple[float, float, float] | PermittivityLaw = 1.0
    mu_r: float | tuple[float, float, float] = 1.0
    tan_delta: float | tuple[float, float, float] = 0.0
    conductivity: float | tuple[float, float, float] = 0.0
    axes: tuple[tuple[float, float, float], ...] | None = None
    thermal_conductivity: float | None = None
    mass_density: float | None = None
    specific_heat: float | None = None
    thermo_optic_coefficient: float | None = None
    nonlinear_index: float | None = None
    two_photon_absorption: float | None = None
    nonlinear_anisotropy: float | None = None
    electro_optic_tensor: tuple[tuple[float, float, float], ...] | None = None
    stiffness: tuple[tuple[float, ...], ...] | None = None
    piezoelectric_tensor: tuple[tuple[float, ...], ...] | None = None
    mechanical_quality_factor: float | None = None
    photoelastic_tensor: tuple[tuple[float, ...], ...] | None = None
    second_order_nonlinear_tensor: tuple[tuple[float, ...], ...] | None = None
    raman_spectrum: RamanSpectrum | None = None
    name: str | None = None

    density = property(_no_density_attribute)

    def __post_init__(self) -> None:
        owner = f"Material {self.name!r}" if self.name else "Material"
        tensor = _material_eo_tensor(self.electro_optic_tensor, owner)
        object.__setattr__(self, "electro_optic_tensor", tensor)
        stiffness = _material_solid_tensor(_STIFFNESS, self.stiffness, owner)
        object.__setattr__(self, "stiffness", stiffness)
        piezoelectric = _material_solid_tensor(
            _PIEZOELECTRIC, self.piezoelectric_tensor, owner
        )
        object.__setattr__(self, "piezoelectric_tensor", piezoelectric)
        photoelastic = _material_solid_tensor(
            _PHOTOELASTIC, self.photoelastic_tensor, owner
        )
        object.__setattr__(self, "photoelastic_tensor", photoelastic)
        second_order = _material_solid_tensor(
            _SECOND_ORDER, self.second_order_nonlinear_tensor, owner
        )
        object.__setattr__(self, "second_order_nonlinear_tensor", second_order)
        _check_solid(
            owner,
            stiffness=stiffness,
            piezoelectric=piezoelectric,
            quality_factor=self.mechanical_quality_factor,
            electro_optic=tensor,
            photoelastic=photoelastic,
            second_order=second_order,
        )
        # a piezoelectric tensor comes with a stiffness, which names the frame
        solid = None if stiffness is None else "stiffness"
        if isinstance(self.eps_r, PermittivityLaw):
            if tensor is not None:
                raise ValueError(_eo_law_message(owner))
            if photoelastic is not None:
                raise ValueError(_photoelastic_law_message(owner))
            if solid is not None:
                raise ValueError(_solid_law_message(owner, solid))
            for name, value in _law_material_values(self, owner).items():
                object.__setattr__(self, name, value)
            _check_raman(self, owner)
            _check_anisotropy(self, owner)
            return
        values = {
            p.name: _material_value(p, getattr(self, p.name), owner) for p in PROPERTIES
        }
        crystal = (
            "electro_optic_tensor"
            if tensor is not None
            else "photoelastic_tensor"
            if photoelastic is not None
            else "second_order_nonlinear_tensor"
            if second_order is not None
            else solid
        )
        if crystal is not None:
            _check_crystal_frame(values, self.axes, owner, crystal)
        vacuum = np.all(np.asarray(values["eps_r"]) == 1.0)
        if tensor is not None and vacuum:
            raise ValueError(_eo_without_eps_message(owner))
        if photoelastic is not None and vacuum:
            raise ValueError(_pe_without_eps_message(owner))
        if second_order is not None and vacuum:
            raise ValueError(_second_order_without_eps_message(owner))
        if piezoelectric is not None and vacuum:
            raise ValueError(_piezoelectric_without_eps_message(owner))
        values, axes = _principal(values, self.axes, owner)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if axes is not None:
            object.__setattr__(self, "axes", _axes(axes, owner))
        _check_raman(self, owner)
        _check_anisotropy(self, owner)

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


def _check_anisotropy(material: Material, owner: str) -> None:
    """Raise if a Kerr anisotropy cannot sit beside the rest of the material."""
    rho = material.nonlinear_anisotropy
    if rho is None:
        return
    if is_traced(rho):
        raise NonDifferentiableError(
            f"{owner}: nonlinear_anisotropy is computed inside jax.grad, "
            f"jax.jit or jax.vmap, but Material values are fixed constants. "
            f"{_TRACED_REMEDY}"
        )
    arr = np.asarray(rho)
    if arr.shape != () or np.iscomplexobj(arr) or arr.dtype.kind not in "iuf":
        raise ValueError(
            f"{owner}: nonlinear_anisotropy must be one real number "
            f"(rho = 3 chi_1122 / chi_1111), got {rho!r}."
        )
    if not np.isfinite(arr):
        raise ValueError(f"{owner}: nonlinear_anisotropy must be finite, got {rho!r}.")
    object.__setattr__(material, "nonlinear_anisotropy", float(arr))
    if material.nonlinear_index is None and material.two_photon_absorption is None:
        raise ValueError(
            f"{owner}: nonlinear_anisotropy scales the Kerr effect, but the "
            "material has no nonlinear_index and no two_photon_absorption. "
            "Give n2 (or beta_TPA) along a crystal axis."
        )
    if isinstance(material.eps_r, PermittivityLaw):
        raise ValueError(
            f"{owner}: nonlinear_anisotropy is given in the crystal frame of axes, "
            "but eps_r is a PermittivityLaw, which takes no axes, so the frame "
            "would be lost. Give the optical permittivity at the wavelength as a "
            "constant eps_r, with axes."
        )
    if not material.isotropic:
        raise ValueError(
            f"{owner}: nonlinear_anisotropy is the cubic form, for a crystal "
            "that is optically isotropic, but this material has an anisotropic "
            "tensor property. Give eps_r one value (with axes for the crystal "
            "frame)."
        )


def _check_raman(material: Material, owner: str) -> None:
    """Raise if a `RamanSpectrum` cannot sit beside the rest of the material."""
    spectrum = material.raman_spectrum
    if spectrum is None:
        return
    if not isinstance(spectrum, RamanSpectrum):
        raise TypeError(
            f"{owner}: raman_spectrum must be a gfem.RamanSpectrum, got "
            f"{type(spectrum).__name__}."
        )
    if is_traced(spectrum):
        raise NonDifferentiableError(
            f"{owner}: raman_spectrum holds a traced gain, but Material values "
            f"are fixed constants. {_TRACED_REMEDY}"
        )
    if isinstance(material.eps_r, PermittivityLaw):
        if spectrum.symmetry == "cubic":
            raise ValueError(
                f"{owner}: a cubic raman_spectrum needs a constant eps_r with "
                "axes, but eps_r is a PermittivityLaw, which takes no axes, "
                "so the crystal frame would be lost. Give the optical "
                "permittivity at the wavelength as a constant eps_r, with axes."
            )
    elif isinstance(material.eps_r, tuple):
        raise ValueError(  # noqa: TRY004
            f"{owner}: a raman_spectrum needs an isotropic eps_r (one value): "
            "the Raman selection rules are the spectrum's symmetry, not the "
            "birefringence of the index."
        )


_MATERIAL_FIELDS = tuple(f.name for f in dataclasses.fields(Material))
_MATERIAL_STATICS = tuple(n for n in _MATERIAL_FIELDS if n != "eps_r")


def _eo_law_message(owner: str) -> str:
    """Why an electro-optic tensor cannot sit beside a permittivity law."""
    return (
        f"{owner}: an electro_optic_tensor needs a constant eps_r, but "
        "eps_r is a PermittivityLaw, which takes no axes, so the frame "
        "of the tensor would be lost. Give the optical permittivity "
        "at the wavelength as a constant eps_r, with axes."
    )


def _eo_without_eps_message(owner: str) -> str:
    """Why an electro-optic tensor needs the crystal's optical permittivity.

    The shift scales as :math:`n^4`: an eps_r left at 1 gives a shift many
    times too small, with no other sign of the omission.
    """
    return (
        f"{owner}: an electro_optic_tensor needs the crystal's optical eps_r "
        "at the wavelength, e.g. eps_r=(n_o**2, n_o**2, n_e**2) with axes; "
        "without it eps_r is 1 and the index shift is n^4 too small."
    )


def _pe_without_eps_message(owner: str) -> str:
    """Why a photoelastic tensor needs the crystal's optical permittivity.

    The shift scales as :math:`n^4`: an eps_r left at 1 gives a shift many
    times too small, with no other sign of the omission.
    """
    return (
        f"{owner}: a photoelastic_tensor needs the crystal's optical eps_r "
        "at the wavelength, e.g. eps_r=(n_o**2, n_o**2, n_e**2) with axes; "
        "without it eps_r is 1 and the index shift is n^4 too small."
    )


def _second_order_without_eps_message(owner: str) -> str:
    """Why a second-order nonlinear tensor needs the crystal's optical permittivity."""
    return (
        f"{owner}: a second_order_nonlinear_tensor needs the crystal's optical "
        "eps_r at the wavelength, e.g. eps_r=(n_o**2, n_o**2, n_e**2) with axes; "
        "without it eps_r is 1, the guide has no crystal and the overlap no "
        "mode."
    )


def _check_crystal_frame(
    values: dict[str, Any], axes: Any, owner: str, tensor: str
) -> None:
    """Raise unless a material's tensors fix the frame of its crystal ``tensor``.

    The order and sign of the axes matter: a matrix input derives axes of
    neither, a reflection flips the sign of every electro-optic or
    piezoelectric coefficient and of the stiffness entries with one index
    along the reflected axis (:math:`c_{14}` of a trigonal crystal), and an axis swap
    that a permittivity with two equal principal values does not see changes
    the stiffness.
    """
    matrices = [name for name, v in values.items() if isinstance(v, np.ndarray)]
    if matrices:
        article = "an" if tensor[0] in "aeiou" else "a"
        raise ValueError(
            f"{owner}: {matrices[0]} is a 3x3 matrix, but beside {article} "
            f"{tensor} every tensor property takes one value or "
            "three principal values along explicit `axes`: a matrix's "
            "derived axes fix neither the order nor the sign of the frame "
            "the tensor is given in."
        )
    if axes is not None:
        _check_right_handed(_axes(axes, owner), owner, tensor)


def _flatten_material_with_keys(m: Material) -> tuple[tuple, tuple]:
    """A law material's only child is its law; a constant material has none.

    A constant material's every field is static aux data, as when `Material`
    was registered static, so existing models keep their leaves.
    """
    if isinstance(m.eps_r, PermittivityLaw):
        kids = ((jax.tree_util.GetAttrKey("eps_r"), m.eps_r),)
        return kids, (True, *(getattr(m, n) for n in _MATERIAL_STATICS))
    return (), (False, *(getattr(m, n) for n in _MATERIAL_FIELDS))


def _flatten_material(m: Material) -> tuple[tuple, tuple]:
    kids, aux = _flatten_material_with_keys(m)
    return tuple(v for _, v in kids), aux


def _unflatten_material(aux: tuple, kids: Any) -> Material:
    m = object.__new__(Material)
    has_law, *statics = aux
    names = _MATERIAL_STATICS if has_law else _MATERIAL_FIELDS
    for n, v in zip(names, statics, strict=True):
        object.__setattr__(m, n, v)
    if has_law:
        (law,) = kids
        object.__setattr__(m, "eps_r", law)
    return m


jax.tree_util.register_pytree_with_keys(
    Material, _flatten_material_with_keys, _unflatten_material, _flatten_material
)


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


def _field_value(prop: _Property, value: Any, principal: bool) -> jax.Array | None:
    """A checked field value; ``principal``: three principal values (along `axes`)."""
    if value is None:
        return None
    if _is_law(value):
        raise TypeError(
            f"MaterialField: {prop.name} cannot be a PermittivityLaw or term; a "
            "law is per region: assign gfem.Material(eps_r=<law>) to the region."
        )
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
    if principal:
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
    if not prop.bounded:
        return
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


def _field_eo_tensor(value: Any) -> jax.Array | None:
    """A checked field electro-optic tensor, ``(6, 3)`` or ``(n_elements, 6, 3)``."""
    if value is None:
        return None
    arr = jnp.asarray(value)
    if jnp.iscomplexobj(arr):
        raise _eo_real("MaterialField")
    arr = arr.astype(jnp.float64)
    if arr.ndim not in (2, 3) or arr.shape[-2:] != (6, 3):
        raise _eo_shape_error("MaterialField", arr.shape, field=True)
    if not is_traced(arr):
        _check_eo_values(np.asarray(arr), "MaterialField")
    return arr


def _field_solid_tensor(tensor: _SolidTensor, value: Any) -> jax.Array | None:
    """A checked field stiffness or piezoelectric tensor, one or one per element."""
    if value is None:
        return None
    arr = jnp.asarray(value)
    if jnp.iscomplexobj(arr):
        raise _solid_real(tensor, "MaterialField")
    arr = arr.astype(jnp.float64)
    if arr.ndim not in (2, 3) or arr.shape[-2:] != tensor.shape:
        raise _solid_shape_error(tensor, "MaterialField", arr.shape, field=True)
    if not is_traced(arr):
        _check_solid_values(tensor, np.asarray(arr), "MaterialField")
    return arr


def _check_blend_values(rho: np.ndarray) -> None:
    """Raise unless a blend's concrete density is finite and in [0, 1].

    Outside [0, 1] a weight end's coefficient 1 - w or w turns negative,
    which makes a lossy susceptibility gain, and SIMP of a negative density
    is NaN: no passivity check of a blended eps_r catches it, since a law
    blend blends no eps_r.
    """
    if not np.isfinite(rho).all():
        raise ValueError("MaterialField: blend must be finite.")
    bad = (rho < 0) | (rho > 1)
    if bad.any():
        first = float(rho.reshape(-1)[int(np.flatnonzero(bad.reshape(-1))[0])])
        raise ValueError(f"MaterialField: blend is a density in [0, 1], got {first:g}.")


def check_field_values(materials: Mapping[str, Any]) -> None:
    """Check the concrete values of every `MaterialField`, region by region.

    A field built under a JAX transformation holds tracers, which its
    constructor cannot check; the host calls this at solve time, on the
    concrete values, before any assembly.

    Args:
        materials: A model's materials by region, with concrete values.

    Raises:
        ValueError: A value the constructor refuses (not finite, not
            passive, or a blend density outside [0, 1]), with the region
            named; the same for a permittivity law's parameters.
    """
    for region, material in materials.items():
        if isinstance(material, Material) and isinstance(
            material.eps_r, PermittivityLaw
        ):
            _check_law(region, material.eps_r)
        if not isinstance(material, MaterialField):
            continue
        for name in (*PROPERTY_NAMES, *_FIELD_TENSORS, "blend"):
            value = getattr(material, name)
            if value is None:
                continue
            try:
                if name == "blend":
                    _check_blend_values(np.asarray(value))
                elif name == "electro_optic_tensor":
                    _check_eo_values(np.asarray(value), "MaterialField")
                elif name in _SOLID_BY_NAME:
                    _check_solid_values(
                        _SOLID_BY_NAME[name], np.asarray(value), "MaterialField"
                    )
                else:
                    _check_field_values(_BY_NAME[name], np.asarray(value))
            except ValueError as err:
                raise ValueError(
                    f"{err} (region {region!r}; the value was computed under a "
                    "JAX transformation, so it is checked at solve time. Keep "
                    "the design in range, e.g. a density in [0, 1].)"
                ) from None


def _check_law(region: str, law: PermittivityLaw) -> None:
    """Rerun a law's construction checks on its concrete values.

    A law built under a JAX transformation skipped them: each term and the
    law are rebuilt with `dataclasses.replace`, and the constructor's error
    is raised with the region named.
    """
    try:
        terms = [dataclasses.replace(t) for t in law.terms]
        dataclasses.replace(law, terms=terms)
    except ValueError as err:
        raise ValueError(
            f"{err} (region {region!r}; the permittivity law was computed under "
            "a JAX transformation, so it is checked at solve time. Keep its "
            "parameters in range, e.g. through a positive parametrization.)"
        ) from None


_SCHEMES: dict[str, Callable[[jax.Array, float], jax.Array]] = {
    "linear": lambda rho, _: rho,
    "simp": lambda rho, p: rho**p,
    "ramp": lambda rho, p: rho / (1 + p * (1 - rho)),
}
#: The interpolation schemes `MaterialField.interpolate` takes: the weights,
#: and the refractive-index blend of the permittivity.
_ALL_SCHEMES = (*_SCHEMES, "index")
#: dw / d rho of each weight, on the host (a law blend's pieces).
_SLOPES: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "linear": lambda rho, _: np.ones_like(rho),
    "simp": lambda rho, p: p * rho ** (p - 1),
    "ramp": lambda rho, p: (1 + p) / (1 + p * (1 - rho)) ** 2,
}


@dataclasses.dataclass(frozen=True)
class PixelGeometry:
    """Where a layer grid's pixels lie: what a solve needs to sample them.

    Attributes:
        region: The volume region the grid covers.
        origin: The grid's lower-left corner ``(x, y)`` (m).
        pitch: The pixel side (m).
        shape: ``(ny, nx)``.
        periodic: Per axis ``(x, y)``, whether the grid wraps around a seam.
    """

    region: str
    origin: tuple[float, float]
    pitch: float
    shape: tuple[int, int]
    periodic: tuple[bool, bool] = (False, False)

    def __post_init__(self) -> None:
        # plain Python values: hashable static data, exact in JSON
        object.__setattr__(self, "origin", tuple(float(x) for x in self.origin))
        object.__setattr__(self, "pitch", float(self.pitch))
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "periodic", tuple(bool(p) for p in self.periodic))


@dataclasses.dataclass(frozen=True)
class Blend:
    r"""The fixed half of a blend held by its density.

    A `MaterialField` built by `MaterialField.interpolate` from a material
    whose ``eps_r`` is a `PermittivityLaw` holds its density as the leaf
    `MaterialField.blend` and this record as static data: the permittivity
    of each end and the scheme. The driven operator reads it as the mass's
    eps_inf limit plus one susceptibility piece per end and, for
    ``"index"``, a cross piece (`gfem._engine.terms.blend_pieces`).

    A field built from a layer density of a 2D mesh
    (`gfem.design.LayerDensity`) takes this form whatever its ends: the
    leaf is then the pixel array, `pixels` locates it, and `others` holds
    the other properties both ends define. A solve applies the scheme at
    each of its quadrature points.

    Attributes:
        void: The permittivity at density 0: a law, or a constant end's
            complex ``eps_r (1 - i tan_delta)``.
        solid: The permittivity at density 1, the same way.
        scheme: ``"index"``, ``"linear"``, ``"simp"`` or ``"ramp"``.
        penalty: The SIMP / RAMP penalty.
        others: For a layer density, ``(name, void, solid)`` of each other
            property both ends define (``mu_r``, ``conductivity``, ...),
            blended by the scheme's weight (:math:`w = \rho` for
            ``"index"``); ``()`` otherwise.
        pixels: For a layer density, where its pixels lie; ``None`` for a
            density per element.
    """

    void: PermittivityLaw | complex
    solid: PermittivityLaw | complex
    scheme: str
    penalty: float
    others: tuple[tuple[str, Any, Any], ...] = ()
    pixels: PixelGeometry | None = None

    def other(self, name: str, rho: np.ndarray) -> tuple[Any, Any] | None:
        """Property ``name`` of `others` at densities ``rho``, and its slope.

        ``None`` when the ends do not both define it.
        """
        for n, a, b in self.others:
            if n == name:
                return a + self.weight(rho) * (b - a), self.slope(rho) * (b - a)
        return None

    def weight(self, rho: np.ndarray) -> np.ndarray:
        """``w(rho)`` of a weight scheme (``rho`` for ``"index"``)."""
        rho = np.asarray(rho, dtype=float)
        if self.scheme == "index":
            return rho
        return np.asarray(_SCHEMES[self.scheme](rho, self.penalty), dtype=float)  # ty: ignore[invalid-argument-type]

    def slope(self, rho: np.ndarray) -> np.ndarray:
        """``dw / d rho`` of `weight`."""
        rho = np.asarray(rho, dtype=float)
        if self.scheme == "index":
            return np.ones_like(rho)
        return np.asarray(_SLOPES[self.scheme](rho, self.penalty), dtype=float)

    @property
    def drude(self) -> bool:
        """Whether an end's law has a `Drude` term (no static permittivity)."""
        return any(
            isinstance(t, Drude)
            for end in (self.void, self.solid)
            if isinstance(end, PermittivityLaw)
            for t in end.terms
        )


def _blend_end(material: Material, which: str) -> PermittivityLaw | complex:
    """One end of a blend: its law, or its constant lossy permittivity.

    Raises:
        ValueError: The end is anisotropic.
        NonDifferentiableError: The end's law is computed inside a JAX
            transformation (its parameters are fixed).
    """
    eps_r = material.eps_r
    if isinstance(eps_r, PermittivityLaw):
        if is_traced(eps_r):
            raise NonDifferentiableError(
                f"MaterialField.interpolate: the permittivity law of the {which} "
                "end is computed inside jax.grad, jax.jit or jax.vmap, but the "
                "laws of a blend's ends are fixed: the blend differentiates its "
                "density only. To differentiate a law's parameters, "
                "differentiate the law on a `Material` region "
                "(model.with_material(region, gfem.Material(eps_r=law)))."
            )
        return eps_r
    if not material.isotropic:
        raise ValueError(
            f"MaterialField.interpolate: the {which} end is anisotropic, but the "
            "index scheme and a permittivity-law blend take isotropic ends only."
        )
    return complex(float(eps_r) * (1 - 1j * float(material.tan_delta)))  # ty: ignore[invalid-argument-type]


def _scalar(value: Any) -> float | complex:
    """A material's isotropic property value as a Python number."""
    value = complex(value)
    return value.real if value.imag == 0 else value


def refractive_index(eps: complex) -> complex:
    """``n - i kappa = sqrt(eps)`` on the branch with ``kappa >= 0``."""
    n = complex(np.sqrt(complex(eps)))
    return -n if n.imag > 0 else n


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
            if _off_diagonal(matrix, a):
                names = [repr(x.name) if x.name else "(unnamed)" for x in ends_of]
                raise ValueError(
                    f"MaterialField.interpolate: void {names[0]} and solid "
                    f"{names[1]} have different principal axes; a material "
                    "field holds principal values along one fixed set of axes. "
                    "Rotate one material, or build the field directly."
                )
            # each principal value exactly as the material gives it
            own_values = np.asarray(value)
            along = np.diag(a @ matrix @ a.T)
            nearest = np.abs(along[:, None] - own_values[None, :]).argmin(axis=1)
            values[name] = own_values[nearest]
        ends.append(values)
    return _axes(a, "MaterialField.interpolate"), (ends[0], ends[1])


def _has_eo(*materials: Material) -> bool:
    return any(m.electro_optic_tensor is not None for m in materials)


def _has_photoelastic(*materials: Material) -> bool:
    return any(m.photoelastic_tensor is not None for m in materials)


def _has_second_order(*materials: Material) -> bool:
    return any(m.second_order_nonlinear_tensor is not None for m in materials)


def _has_crystal(*materials: Material) -> bool:
    """Whether a material has a tensor given in the frame of its axes."""
    return (
        _has_eo(*materials)
        or _has_photoelastic(*materials)
        or _has_second_order(*materials)
        or any(m.stiffness is not None for m in materials)
    )


def _blend_solid(void: Material, solid: Material, axes: Any, w: Any) -> dict:
    r"""A solid's stiffness, piezoelectric tensor and :math:`Q_m`, blended by ``w``.

    The tensors blend linearly in the frame of ``axes``: the piezoelectric
    tensor with an end without one counting as zero, the stiffness only
    between two solids (zero stiffness inside the mechanical domain is
    singular). The loss :math:`1/Q_m` blends linearly, between two ends
    that both give a quality factor. Empty when neither end is a solid.

    Raises:
        ValueError: One end only is a solid, or one solid only has a
            quality factor.
    """
    out: dict[str, Any] = {}
    ends = {"void": void, "solid": solid}
    if all(m.stiffness is None for m in ends.values()):
        return out
    for which, m in ends.items():
        if m.stiffness is None:
            raise ValueError(
                f"MaterialField.interpolate: the {which} end has no stiffness, but "
                "the other end is a solid: a blend of a solid takes both ends' "
                "stiffness, since zero stiffness inside the mechanical domain is "
                "singular. Give both ends a stiffness."
            )
    c = [
        stiffness_in_frame(np.asarray(m.stiffness), m.axes, axes) for m in (void, solid)
    ]
    out["stiffness"] = c[0] + jnp.asarray(w)[..., None, None] * (c[1] - c[0])
    if any(m.piezoelectric_tensor is not None for m in (void, solid)):
        e = [
            np.zeros((3, 6))
            if m.piezoelectric_tensor is None
            else piezoelectric_in_frame(
                np.asarray(m.piezoelectric_tensor), m.axes, axes
            )
            for m in (void, solid)
        ]
        out["piezoelectric_tensor"] = e[0] + jnp.asarray(w)[..., None, None] * (
            e[1] - e[0]
        )
    q = [m.mechanical_quality_factor for m in (void, solid)]
    if q[0] is None and q[1] is None:
        return out
    for which, m in ends.items():
        if m.mechanical_quality_factor is None:
            raise ValueError(
                "MaterialField.interpolate: the other end has a "
                f"mechanical_quality_factor but the {which} end has none: the "
                "blend is linear in the loss 1 / Q_m, so give both ends one (a "
                "nearly lossless end takes a large Q_m)."
            )
    loss = [1 / float(x) for x in q if x is not None]  # both given
    out["mechanical_quality_factor"] = 1 / (loss[0] + w * (loss[1] - loss[0]))
    return out


def _check_kerr_ends(void: Material, solid: Material) -> None:
    """Raise if one end only gives a Kerr property: the blend is linear in it.

    A Kerr anisotropy is the crystal's, not a number that blends.
    """
    for end in (void, solid):
        if end.nonlinear_anisotropy is not None:
            raise ValueError(
                "MaterialField.interpolate: a nonlinear_anisotropy is a "
                "property of the crystal, not a number that blends; give "
                "the material on a Material region instead."
            )
    for name in _NONLINEAR:
        given = [getattr(m, name) is not None for m in (void, solid)]
        if any(given) and not all(given):
            which = "void" if not given[0] else "solid"
            raise ValueError(
                f"MaterialField.interpolate: the other end has a {name} but the "
                f"{which} end has none: the blend is linear in it, so give both "
                "ends one (0 for a material without the effect)."
            )


def _blend_eo(void: Material, solid: Material, axes: Any, w: Any) -> Any:
    """The electro-optic tensor blended by weight ``w`` in the frame of ``axes``.

    An end without a tensor counts as zero; ``None`` when neither has one.
    """
    if not _has_eo(void, solid):
        return None
    a, b = (
        np.zeros((6, 3))
        if m.electro_optic_tensor is None
        else electro_optic_in_frame(np.asarray(m.electro_optic_tensor), m.axes, axes)
        for m in (void, solid)
    )
    return a + jnp.asarray(w)[..., None, None] * (b - a)


def _blend_photoelastic(void: Material, solid: Material, axes: Any, w: Any) -> Any:
    """The photoelastic tensor blended by weight ``w`` in the frame of ``axes``.

    An end without a tensor counts as zero; ``None`` when neither has one.
    """
    if not _has_photoelastic(void, solid):
        return None
    a, b = (
        np.zeros((6, 6))
        if m.photoelastic_tensor is None
        else stiffness_in_frame(np.asarray(m.photoelastic_tensor), m.axes, axes)
        for m in (void, solid)
    )
    return a + jnp.asarray(w)[..., None, None] * (b - a)


def _blend_second_order(void: Material, solid: Material, axes: Any, w: Any) -> Any:
    """The second-order nonlinear tensor blended by weight ``w`` in ``axes``.

    An end without a tensor counts as zero; ``None`` when neither has one.
    """
    if not _has_second_order(void, solid):
        return None
    a, b = (
        np.zeros((3, 6))
        if m.second_order_nonlinear_tensor is None
        else piezoelectric_in_frame(
            np.asarray(m.second_order_nonlinear_tensor), m.axes, axes
        )
        for m in (void, solid)
    )
    return a + jnp.asarray(w)[..., None, None] * (b - a)


@_renamed_density
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class MaterialField:
    r"""Material properties that vary over one volume region, per element (P0).

    Keyword-only. Any `Material` property may be given; a property left out
    is ``None`` (adding a property later never changes the structure that
    `jax.tree` functions see in existing code). Each given property is
    either an array of one value per element of the region, in mesh order
    (``mesh.elements(region)``), or a single value for the whole region.

    `jax.grad` of any objective reaches the values.
    `eps_r` and `mu_r` may be complex (lossy, negative imaginary part under
    :math:`e^{+i\omega t}`); `conductivity` is electrical, S/m. The thermal properties
    (`thermal_conductivity`, `mass_density`, `specific_heat`) and
    `thermo_optic_coefficient` (:math:`dn/dT`, 1/K, any sign) are real, in the units
    of `Material`; ``density=`` is the deprecated name of `mass_density`.
    `nonlinear_index` (:math:`n_2`, m^2/W, any sign) and `two_photon_absorption`
    (:math:`\beta_\mathrm{TPA}`, m/W, :math:`\ge 0`) are real too; only
    `gfem.kerr_nonlinearity` and `gfem.kerr_coupling` read them.

    With `axes`, the field is an anisotropic material field: each of
    `eps_r`, `mu_r`, `tan_delta` and `conductivity` holds three principal
    values along the principal axes, shape ``(n_elements, 3)`` per element
    or ``(3,)`` for the whole region (an isotropic value is three equal
    values); the thermal properties keep one value. The axes are fixed: the
    values vary, the principal axes do not. `jax.grad` gives one derivative
    per principal value, in the shape of the values; the axes have none.

    `electro_optic_tensor` is the contracted 6x3 tensor :math:`r_{Ik}` of
    `Material` (m/V, rows xx, yy, zz, yz, xz, xy), shape ``(6, 3)`` for the
    whole region or ``(n_elements, 6, 3)`` per element, in the frame whose
    x, y, z are the rows of `axes` (the global frame if ``None``); with it,
    `axes` must be right-handed. `jax.grad` reaches its entries.

    `stiffness` (:math:`c^E`, Pa) and `piezoelectric_tensor` (:math:`e`,
    C/m^2) are the contracted matrices of `Material`, shape ``(6, 6)`` and
    ``(3, 6)`` for the whole region or ``(n_elements, 6, 6)`` and
    ``(n_elements, 3, 6)`` per element, in the frame of `axes` as the
    electro-optic tensor is; `mechanical_quality_factor` (:math:`Q_m`) is one
    real value per element or for the region. `jax.grad` reaches their
    entries.

    `photoelastic_tensor` is the contracted 6x6 matrix :math:`p^E_{IJ}` of
    `Material` (dimensionless), shape ``(6, 6)`` for the whole region or
    ``(n_elements, 6, 6)`` per element, in the frame of `axes` as the
    stiffness is; `jax.grad` reaches its entries.

    `second_order_nonlinear_tensor` is the contracted 3x6 matrix :math:`d_{il}`
    of `Material` (m/V), shape ``(3, 6)`` for the whole region or
    ``(n_elements, 3, 6)`` per element, in the frame of `axes` as the
    piezoelectric tensor is; `jax.grad` reaches its entries.

    Build one with `MaterialField.interpolate` (a design density blended
    between two materials) or `MaterialField.uniform` (a whole region whose
    constant material should be differentiable).

    Attributes:
        region: The region the field was built for, or ``None``. When set,
            the model checks that the field is assigned to this region.
        axes: Three orthonormal vectors, the principal axes of the
            principal values, or ``None`` (an isotropic field). Fixed: it
            cannot be differentiated.
        blend: For a blend with a permittivity-law end
            (`MaterialField.interpolate`), its density per element, the
            design leaf that `jax.grad` reaches; for a layer density of a 2D
            model, its pixels ``(ny, nx)``; ``None`` otherwise. Its
            permittivity is then `blend_ends` at that density (`eps_r` and
            `tan_delta` are ``None``).
        blend_ends: The fixed ends and scheme of that blend (`Blend`), or
            ``None``.

    Raises:
        ValueError: A value is not passive or finite, a tensor property's
            shape does not fit `axes` (a 2-D array needs `axes`; with `axes`
            the last dimension is 3), the per-element sizes differ, or
            `axes` is not orthonormal; an `electro_optic_tensor` that is not
            real and finite, of shape ``(6, 3)`` or ``(n_elements, 6, 3)``,
            has an entry above 1e-8 m/V (a value in pm/V), or comes with
            left-handed `axes`, without `eps_r` or in a blend; a
            `stiffness` or `piezoelectric_tensor` that fails the checks of
            `Material` (per element), or comes with left-handed `axes`, in a
            blend or beside an `electro_optic_tensor`; a
            `piezoelectric_tensor` without `stiffness` or `eps_r`; a
            `mechanical_quality_factor` without `stiffness`; a
            `photoelastic_tensor` that fails the checks of `Material` (per
            element), or comes with left-handed `axes`, in a blend or beside a
            `stiffness` or `piezoelectric_tensor`; a
            `second_order_nonlinear_tensor` that fails the checks of
            `Material` (per element), or comes with left-handed `axes`,
            without `eps_r` (outside a blend) or beside a `stiffness` or
            `piezoelectric_tensor`.
        NonDifferentiableError: `axes` is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap` from their arguments; pass a fixed value.
    """

    eps_r: jax.Array | None = None
    mu_r: jax.Array | None = None
    tan_delta: jax.Array | None = None
    conductivity: jax.Array | None = None
    thermal_conductivity: jax.Array | None = None
    mass_density: jax.Array | None = None
    specific_heat: jax.Array | None = None
    thermo_optic_coefficient: jax.Array | None = None
    nonlinear_index: jax.Array | None = None
    two_photon_absorption: jax.Array | None = None
    electro_optic_tensor: jax.Array | None = None
    stiffness: jax.Array | None = None
    piezoelectric_tensor: jax.Array | None = None
    mechanical_quality_factor: jax.Array | None = None
    photoelastic_tensor: jax.Array | None = None
    second_order_nonlinear_tensor: jax.Array | None = None
    region: str | None = None
    axes: tuple[tuple[float, float, float], ...] | None = None
    blend: jax.Array | None = None
    blend_ends: Blend | None = None

    density = property(_no_density_attribute)

    def __post_init__(self) -> None:
        if self.axes is not None:
            remedy = (
                "The orientation is not a design input; differentiate with "
                "respect to the principal values instead."
            )
            object.__setattr__(self, "axes", _axes(self.axes, "MaterialField", remedy))
        sizes = set()
        for prop in PROPERTIES:
            principal = self.has_principal_values(prop.name)
            value = _field_value(prop, getattr(self, prop.name), principal)
            object.__setattr__(self, prop.name, value)
            if value is not None and value.ndim == 1 + principal:
                sizes.add(value.shape[0])
        tensor = _field_eo_tensor(self.electro_optic_tensor)
        object.__setattr__(self, "electro_optic_tensor", tensor)
        if tensor is not None:
            if self.blend is not None:
                raise ValueError(_eo_law_message("MaterialField"))
            if self.eps_r is None:
                raise ValueError(_eo_without_eps_message("MaterialField"))
            if tensor.ndim == 3:
                sizes.add(tensor.shape[0])
            if self.axes is not None:
                _check_right_handed(self.axes, "MaterialField")
        sizes |= self._check_solid()
        sizes |= self._check_photoelastic()
        sizes |= self._check_second_order()
        size = self._check_blend()
        if size is not None:
            sizes.add(size)
        if self.blend is None and not self.properties:
            raise ValueError(
                "MaterialField needs at least one property, e.g. "
                "MaterialField(eps_r=values)."
            )
        if len(sizes) > 1:
            raise ValueError(
                "MaterialField: every per-element property must have the same "
                f"number of values (one per element), got {sorted(sizes)}."
            )

    def _check_solid(self) -> set[int]:
        """Check the stiffness and piezoelectric tensor; their per-element sizes."""
        sizes = set()
        for tensor in _SOLID_TENSORS:
            value = _field_solid_tensor(tensor, getattr(self, tensor.name))
            object.__setattr__(self, tensor.name, value)
            if value is not None and value.ndim == 3:
                sizes.add(value.shape[0])
        _check_solid(
            "MaterialField",
            stiffness=self.stiffness,
            piezoelectric=self.piezoelectric_tensor,
            quality_factor=self.mechanical_quality_factor,
            electro_optic=self.electro_optic_tensor,
        )
        if self.stiffness is None:
            return sizes
        if self.blend is not None:
            raise ValueError(
                "MaterialField: a stiffness is given in the frame of axes, but a "
                "blend holds a permittivity law or a layer grid, which take no "
                "axes; build the field per element."
            )
        if self.piezoelectric_tensor is not None and self.eps_r is None:
            raise ValueError(_piezoelectric_without_eps_message("MaterialField"))
        if self.axes is not None:
            _check_right_handed(self.axes, "MaterialField", "stiffness")
        return sizes

    def _check_photoelastic(self) -> set[int]:
        """Check the photoelastic tensor; its per-element size."""
        value = _field_solid_tensor(_PHOTOELASTIC, self.photoelastic_tensor)
        object.__setattr__(self, "photoelastic_tensor", value)
        if value is None:
            return set()
        _check_solid(
            "MaterialField",
            stiffness=self.stiffness,
            piezoelectric=self.piezoelectric_tensor,
            quality_factor=None,
            electro_optic=None,
            photoelastic=value,
        )
        if self.blend is not None:
            raise ValueError(_photoelastic_law_message("MaterialField"))
        if self.axes is not None:
            _check_right_handed(self.axes, "MaterialField", "photoelastic_tensor")
        if self.eps_r is None:
            raise ValueError(_pe_without_eps_message("MaterialField"))
        return {value.shape[0]} if value.ndim == 3 else set()

    def _check_second_order(self) -> set[int]:
        """Check the second-order nonlinear tensor; its per-element size."""
        value = _field_solid_tensor(_SECOND_ORDER, self.second_order_nonlinear_tensor)
        object.__setattr__(self, "second_order_nonlinear_tensor", value)
        if value is None:
            return set()
        _check_solid(
            "MaterialField",
            stiffness=self.stiffness,
            piezoelectric=self.piezoelectric_tensor,
            quality_factor=None,
            electro_optic=None,
            second_order=value,
        )
        if self.axes is not None:
            _check_right_handed(
                self.axes, "MaterialField", "second_order_nonlinear_tensor"
            )
        if self.eps_r is None and self.blend is None:
            raise ValueError(_second_order_without_eps_message("MaterialField"))
        return {value.shape[0]} if value.ndim == 3 else set()

    def _check_blend(self) -> int | None:
        """Check `blend` and `blend_ends`; the size of a per-element `blend`."""
        if (self.blend is None) != (self.blend_ends is None):
            raise ValueError(
                "MaterialField: blend and blend_ends go together; build a blend "
                "with MaterialField.interpolate."
            )
        if self.blend is None:
            return None
        if self.eps_r is not None or self.tan_delta is not None or self.axes:
            raise ValueError(
                "MaterialField: a blend holds its permittivity in blend_ends; "
                "eps_r, tan_delta and axes must be None."
            )
        rho = jnp.asarray(self.blend)
        grid = getattr(self.blend_ends, "pixels", None)
        if grid is not None:
            if jnp.iscomplexobj(rho) or rho.shape != grid.shape:
                raise ValueError(
                    "MaterialField: the blend of a layer grid is a real pixel "
                    f"array of shape {grid.shape} (ny, nx), got shape {rho.shape}."
                )
        elif jnp.iscomplexobj(rho) or rho.ndim > 1:
            raise ValueError(
                "MaterialField: blend is a real density, one value per element "
                f"or one for the region, got shape {rho.shape}."
            )
        rho = rho.astype(jnp.float64)
        if not is_traced(rho):
            _check_blend_values(np.asarray(rho))
        object.__setattr__(self, "blend", rho)
        return int(rho.shape[0]) if rho.ndim == 1 else None

    @property
    def n_elements(self) -> int | None:
        """Number of per-element values, or ``None`` if every value is uniform."""
        if self.blend is not None and self.blend.ndim == 1:
            return int(self.blend.shape[0])
        for name in PROPERTY_NAMES:
            v = getattr(self, name)
            if v is not None and v.ndim == 1 + self.has_principal_values(name):
                return int(v.shape[0])
        for name in _FIELD_TENSORS:
            r = getattr(self, name)
            if r is not None and r.ndim == 3:
                return int(r.shape[0])
        return None

    def has_principal_values(self, name: str) -> bool:
        """Whether property ``name`` holds three principal values along `axes`."""
        return self.axes is not None and _BY_NAME[name].may_be_tensor

    @property
    def properties(self) -> tuple[str, ...]:
        """Names of the properties this field gives."""
        names = (*PROPERTY_NAMES, *_FIELD_TENSORS)
        return tuple(n for n in names if getattr(self, n) is not None)

    @classmethod
    def interpolate(
        cls,
        density: jax.Array | LayerDensity,
        void: Material,
        solid: Material,
        *,
        scheme: Literal["linear", "simp", "ramp", "index"] = "linear",
        penalty: float = 3.0,
    ) -> MaterialField:
        r"""Blend two materials by a density per element.

        Each property that both materials define becomes
        ``void + w(density) * (solid - void)``, with ``w`` the scheme's
        weight: ``linear`` (:math:`w = \rho`), ``simp``
        (:math:`w = \rho^p`, :math:`p` the ``penalty``) or ``ramp``
        (:math:`w = \rho / (1 + p (1 - \rho))`). Differentiable in
        ``density``.

        ``index`` blends the refractive index instead: :math:`n - i\kappa` is
        linear in :math:`\rho`, so
        :math:`\varepsilon = ((1 - \rho) N_\mathrm{void} + \rho N_\mathrm{solid})^2`
        with :math:`N = \sqrt{\varepsilon_r (1 - i \tan\delta)}` on the branch
        with :math:`\kappa \ge 0` (Christiansen et al., CMAME 343, 2019); the
        other properties blend with :math:`w = \rho`. Use it to blend a metal
        and a dielectric: a linear blend of :math:`\varepsilon` crosses
        :math:`\mathrm{Re}\,\varepsilon = -2 \varepsilon_\mathrm{host}`, where
        an intermediate element is
        a small plasmonic resonator, almost lossless, and the optimizer finds
        artificial resonances there. Two constant ends give a complex
        `eps_r` per element.

        Either end's ``eps_r`` may be a `PermittivityLaw` (a dispersive
        material, a Drude metal for example), under every scheme. The field
        then holds the density as its leaf `blend` and the two ends as fixed
        data (`blend_ends`); its permittivity follows the scheme at every
        frequency. Such a blend is solved by the driven problem on the
        uniform sweep only; the adaptive sweep, the eigenmode and port mode
        problems and an `Absorbing` or wave port face on the region raise,
        and so does the electrostatic problem (a `Drude` end has no static
        permittivity). The ends' law parameters are not design inputs.

        Warning:
            Under ``"index"`` an intermediate density between a dielectric
            and a metal is a strong absorber: :math:`\kappa` is linear in
            :math:`\rho`, so at :math:`\rho = 0.5` next to gold :math:`\kappa`
            is about 5. A design of uniform :math:`\rho = 0.5` transmits
            almost nothing and its gradient vanishes; start a
            metal-dielectric optimization from a binary design (or one close
            to it), not from 0.5 everywhere.

        Anisotropic materials blend per principal value, when every tensor
        of both shares one set of principal axes (an isotropic material
        shares any, so air and an anisotropic material blend). The field
        then has those `axes` (in `Material`'s normal form: axis k the one
        closest to coordinate axis k), so axes given in another order or
        sign give the same field. Two isotropic materials give an isotropic
        field (an anisotropic material field otherwise). The ``index``
        scheme and a law end take isotropic ends only.

        On a 2D model, ``density`` is a layer grid's `gfem.design.LayerDensity`
        (``grid.to_elements(rho)``): the field keeps the pixels as its leaf
        `blend`, with `blend_ends` holding the ends, the scheme and the grid's
        geometry, and a solve applies the scheme at each of its quadrature
        points. Its ends are isotropic.

        The `electro_optic_tensor` blends linearly by the scheme's weight
        (:math:`w = \rho` for ``"index"``), an end without one counting as
        zero, each end's tensor moved to the field's frame: its `axes`, made
        right-handed, or the global frame for an isotropic field. A layer
        density and a law end refuse it.

        The `second_order_nonlinear_tensor` blends the same way, an end
        without one counting as zero; a law end accepts it (in the global
        frame), a layer density refuses it.

        A `stiffness` blends linearly between two solids, and a
        `piezoelectric_tensor` with an end without one counting as zero,
        both in the field's frame as the electro-optic tensor is, and so does
        the `photoelastic_tensor`, an end without one counting as zero. The
        `mechanical_quality_factor` blends through its loss: :math:`1/Q_m`
        is linear in the weight, between two ends that both give one. A
        layer density refuses them.

        Args:
            density: One value per element, in [0, 1] (0 = void, 1 = solid),
                or a `gfem.design.LayerDensity` on a 2D model.
            void: The material at density 0.
            solid: The material at density 1.
            scheme: The interpolation: a weight, or ``"index"``.
            penalty: The SIMP / RAMP penalty.

        Raises:
            ValueError: Unknown scheme, two materials whose tensors have
                different principal axes, or an anisotropic end under
                ``"index"`` or beside a law; an electro-optic end under a
                layer density or beside a law; one end only a solid (with a
                `stiffness`) or with a `mechanical_quality_factor`; a solid
                end under a layer density; one end only with a
                `nonlinear_index` or a `two_photon_absorption`; an end with a
                `raman_spectrum` or a `nonlinear_anisotropy`.
            NonDifferentiableError: An end's law is computed inside
                `jax.grad`, `jax.jit` or `jax.vmap`: differentiate a law on a
                `Material` region instead.
        """
        _check_kerr_ends(void, solid)
        for end in (void, solid):
            if end.raman_spectrum is not None:
                raise ValueError(
                    "MaterialField.interpolate: a raman_spectrum is a "
                    "tabulated curve, not a number that blends; give the "
                    "spectrum on a Material region instead."
                )
        if scheme not in _ALL_SCHEMES:
            raise ValueError(
                f"Unknown interpolation scheme {scheme!r}; use one of "
                f"{', '.join(repr(s) for s in _ALL_SCHEMES)}."
            )
        # imported here: gfem.design imports this module
        from gfem.design import LayerDensity

        if isinstance(density, LayerDensity):
            return cls._interpolate_pixels(density, void, solid, scheme, penalty)
        rho = jnp.asarray(density, dtype=jnp.float64)
        law = any(isinstance(m.eps_r, PermittivityLaw) for m in (void, solid))
        if law and _has_eo(void, solid):
            raise ValueError(_eo_law_message("MaterialField.interpolate"))
        if law and _has_photoelastic(void, solid):
            raise ValueError(_photoelastic_law_message("MaterialField.interpolate"))
        if law or scheme == "index":
            return cls._interpolate_eps(rho, void, solid, scheme, penalty, law=law)
        w = _SCHEMES[scheme](rho, penalty)
        if void.isotropic and solid.isotropic:
            axes = None
            ends = tuple(
                {n: getattr(m, n) for n in PROPERTY_NAMES} for m in (void, solid)
            )
        else:
            axes, ends = _common_axes(void, solid)
            if _has_crystal(void, solid) and np.linalg.det(np.asarray(axes)) < 0:
                # the tensors' frame is right-handed; the sign of a principal
                # axis leaves the principal values unchanged
                flip = np.array([[1.0], [1.0], [-1.0]])
                axes = _axes(np.asarray(axes) * flip, "MaterialField.interpolate")
        props = _blend_solid(void, solid, axes, w)
        for name in PROPERTY_NAMES:
            a, b = ends[0][name], ends[1][name]
            if name in props or a is None or b is None:
                continue
            principal = axes is not None and _BY_NAME[name].may_be_tensor
            props[name] = a + (w[..., None] if principal else w) * (b - a)
        tensor = _blend_eo(void, solid, axes, w)
        photoelastic = _blend_photoelastic(void, solid, axes, w)
        return cls(
            **props,
            electro_optic_tensor=tensor,
            photoelastic_tensor=photoelastic,
            second_order_nonlinear_tensor=_blend_second_order(void, solid, axes, w),
            axes=axes,
        )

    @classmethod
    def _interpolate_eps(
        cls,
        rho: jax.Array,
        void: Material,
        solid: Material,
        scheme: str,
        penalty: float,
        *,
        law: bool,
    ) -> MaterialField:
        """`interpolate` under ``"index"`` or with a law end.

        The permittivity (with its loss tangent) blends by the scheme; every
        other property both ends define by the weight (rho for ``"index"``).
        """
        ends = (_blend_end(void, "void"), _blend_end(solid, "solid"))
        w = rho if scheme == "index" else _SCHEMES[scheme](rho, penalty)
        props: dict[str, Any] = _blend_solid(void, solid, None, w)
        for name in PROPERTY_NAMES:
            if name in ("eps_r", "tan_delta") or name in props:
                continue
            a, b = getattr(void, name), getattr(solid, name)
            if a is not None and b is not None:
                props[name] = a + w * (b - a)
        props["electro_optic_tensor"] = _blend_eo(void, solid, None, w)
        props["photoelastic_tensor"] = _blend_photoelastic(void, solid, None, w)
        props["second_order_nonlinear_tensor"] = _blend_second_order(
            void, solid, None, w
        )
        if law:
            blend = Blend(ends[0], ends[1], scheme, float(penalty))
            return cls(**props, blend=rho, blend_ends=blend)
        na, nb = (
            refractive_index(e) for e in ends if not isinstance(e, PermittivityLaw)
        )
        return cls(**props, eps_r=((1 - rho) * na + rho * nb) ** 2)

    @classmethod
    def _interpolate_pixels(
        cls, density: Any, void: Material, solid: Material, scheme: str, penalty: float
    ) -> MaterialField:
        """`interpolate` of a `gfem.design.LayerDensity`: the pixels in the blend.

        Every scheme, constant ends included, is applied at each point a solve
        samples, so the ends, the other properties both define and the grid's
        geometry are the blend's static data.

        Raises:
            ValueError: An end is anisotropic.
        """
        for which, m in (("void", void), ("solid", solid)):
            if not m.isotropic:
                raise ValueError(
                    f"MaterialField.interpolate: the {which} end is anisotropic, "
                    "but a layer density of a 2D model takes isotropic ends only."
                )
            tensor = next(
                (n for n in _FIELD_TENSORS if getattr(m, n) is not None), None
            )
            if tensor is not None:
                article = "an" if tensor[0] in "aeiou" else "a"
                raise ValueError(
                    f"MaterialField.interpolate: the {which} end has {article} "
                    f"{tensor}, but a layer density of a 2D model "
                    "does not blend it; give the region a Material or a "
                    "MaterialField per element."
                )
        ends = (_blend_end(void, "void"), _blend_end(solid, "solid"))
        others = []
        for name in PROPERTY_NAMES:
            if name in ("eps_r", "tan_delta"):
                continue
            a, b = getattr(void, name), getattr(solid, name)
            if a is not None and b is not None:
                others.append((name, _scalar(a), _scalar(b)))
        blend = Blend(
            ends[0],
            ends[1],
            scheme,
            float(penalty),
            others=tuple(others),
            pixels=density.grid.geometry,
        )
        rho = jnp.asarray(density.pixels, dtype=jnp.float64)
        return cls(blend=rho, blend_ends=blend, region=density.grid.region)

    @classmethod
    def uniform(cls, region: str, material: Material) -> MaterialField:
        """A whole region's constant material, as a differentiable field.

        Each property the material defines becomes one scalar array, so
        ``jax.grad`` gives the sensitivity to the region's eps_r,
        conductivity, ... directly. An anisotropic material gives a field
        on its `axes` (the coordinate axes if ``None``) whose tensor
        properties are ``(3,)`` arrays of principal values, and so does a
        material with an `electro_optic_tensor` or a `stiffness` and `axes`,
        whose frame the field keeps; each such tensor is one array, ``(6, 3)``,
        ``(6, 6)`` or ``(3, 6)``.

        Args:
            region: The volume region the field is for.
            material: The material.

        Raises:
            ValueError: The material has a `nonlinear_anisotropy` or a
                `raman_spectrum`, which a field does not carry.
        """
        for dropped in ("nonlinear_anisotropy", "raman_spectrum"):
            if getattr(material, dropped) is not None:
                raise ValueError(
                    f"MaterialField.uniform: the material has a {dropped}, which a "
                    "MaterialField does not carry, so it would be dropped. Assign "
                    "the Material itself to the region."
                )
        if isinstance(material.eps_r, PermittivityLaw):
            raise TypeError(
                "MaterialField.uniform: the material's eps_r is a PermittivityLaw. "
                "Assign the Material itself: jax.grad reaches the law's "
                "parameters directly (the law is a pytree)."
            )
        axes = None
        if not material.isotropic or (_has_crystal(material) and material.axes):
            axes = material.axes or _axes(np.eye(3), "MaterialField.uniform")
        props: dict[str, Any] = {}
        for n in PROPERTY_NAMES:
            value = getattr(material, n)
            if value is None:
                continue
            arr = jnp.asarray(value, dtype=jnp.float64)
            if axes is not None and _BY_NAME[n].may_be_tensor:
                arr = jnp.broadcast_to(arr, (3,))
            props[n] = arr
        for n in _FIELD_TENSORS:
            tensor = getattr(material, n)
            if tensor is not None:
                props[n] = jnp.asarray(tensor, dtype=jnp.float64)
        return cls(**props, region=region, axes=axes)

    def __repr__(self) -> str:
        n = self.n_elements
        size = "uniform" if n is None else f"{n} elements"
        if self.blend_ends is not None and self.blend_ends.pixels is not None:
            size = f"layer grid {self.blend_ends.pixels.shape}"
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
        if self.blend is not None and self.blend_ends is not None:
            ends = self.blend_ends
            rho = "<traced>" if is_traced(self.blend) else self.blend
            if not isinstance(rho, str):
                a = np.asarray(rho)
                lo, hi = a.min(), a.max()
                rho = f"{lo:.4g}" if lo == hi else f"in [{lo:.4g}, {hi:.4g}]"
            parts.append(f"blend {ends.scheme!r} density {rho}")
        where = f", region={self.region!r}" if self.region else ""
        if self.axes is not None:
            parts.append(f"axes={self.axes!r}")
        return f"MaterialField({size}{where}, {', '.join(parts)})"


register_fields(MaterialField, (*PROPERTY_NAMES, *_FIELD_TENSORS, "blend"))


def field_principal_values(
    field: MaterialField, name: str, at: np.ndarray | None = None
) -> Any:
    """The principal values of property ``name`` of an anisotropic material field.

    The leaf as held, ``(n_elements, 3)`` or ``(3,)`` (a JAX array stays
    one, so a traced value stays traced), or the property's default three
    times when the field leaves ``name`` out. With ``at`` (element
    positions in the region, any shape), the values of those elements,
    ``(*at.shape, 3)``.
    """
    value = getattr(field, name)
    if value is None:
        value = np.full(3, _BY_NAME[name].default)
    if at is None:
        return value
    if value.ndim == 2:
        return value[at]
    xp = np if isinstance(value, np.ndarray) else jnp
    return xp.broadcast_to(value, (*np.shape(at), 3))
