"""`SheetField`: a per-face fill scaling a sheet boundary's admittance."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced, static_float
from gfem._pytree import register_fields

#: The default contrast ``r`` of the exponential fill scheme.
DEFAULT_CONTRAST = 1e10


def _fill_values(value: Any) -> jax.Array:
    arr = jnp.asarray(value)
    if jnp.iscomplexobj(arr):
        raise ValueError(
            "SheetField: fill must be real; it scales the sheet's admittance, "
            "which the Conductor or Impedance gives."
        )
    arr = arr.astype(jnp.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"SheetField: fill has shape {arr.shape}, but a sheet field holds "
            "one value per face of its surface, shape (n_faces,), in the order "
            "of mesh.elements(surface)."
        )
    if not is_traced(arr):
        check_fill(np.asarray(arr))
    return arr


def check_fill(values: np.ndarray) -> None:
    """Raise unless every fill value is finite and >= 0 (a passive sheet).

    Raises:
        ValueError: A value is negative or not finite.
    """
    if not np.isfinite(values).all():
        raise ValueError("SheetField: fill must be finite.")
    if (values < 0).any():
        first = int(np.flatnonzero(values < 0)[0])
        raise ValueError(
            f"SheetField: fill must be >= 0 (a passive sheet), but face {first} "
            f"has fill {values[first]:.6g}. Fill 0 is no sheet and 1 the solid "
            "sheet; interpolate a density in [0, 1] with SheetField.interpolate."
        )


@dataclasses.dataclass(frozen=True, eq=False)
class SheetField:
    """A per-face fill of a sheet: the design of a patterned metal surface.

    Attach it to a `Conductor` or `Impedance` on an interior surface (both
    sides in the solve domain) with ``fill=``: the boundary's admittance on
    face f becomes ``fill[f]`` times the solid sheet's, ``Y_f(ω) = w_f
    Y_solid(ω)``. Fill 0 is no sheet (the surface is plain dielectric
    continuity), 1 the solid sheet; values above 1 are legal (more sheet
    than the solid), negative ones are not (the sheet would be active).

    The fill is a pytree leaf: `jax.grad` of a driven or eigenmode objective
    reaches it. Build it from a design density with `interpolate`.

    Attributes:
        fill: One real value per face of the surface, in
            ``mesh.elements(surface)`` order, shape ``(n_faces,)``.
        surface: The surface the field was built for, or ``None``. When
            set, the model checks that the boundary carrying the field is on
            this surface.

    Raises:
        ValueError: ``fill`` is complex, not one-dimensional, not finite or
            negative.
    """

    fill: jax.Array
    surface: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill", _fill_values(self.fill))
        if self.surface is not None and not isinstance(self.surface, str):
            raise TypeError(
                f"SheetField: surface must be a surface name, got {self.surface!r}."
            )

    @property
    def n_faces(self) -> int:
        """Number of faces (the length of `fill`)."""
        return int(self.fill.shape[0])

    @classmethod
    def interpolate(
        cls,
        density: jax.Array,
        *,
        scheme: Literal["exponential", "linear"] = "exponential",
        contrast: float = DEFAULT_CONTRAST,
        surface: str | None = None,
    ) -> SheetField:
        """A fill from a design density per face, 0 = void, 1 = solid.

        ``exponential`` (the default) maps ``rho`` to ``w = (r^rho - 1) / (r
        - 1)``, ``r = contrast``: log-scale in the sheet admittance (``w ≈
        r^(rho - 1)`` away from 0) with an exact void (``w(0) = 0``) and
        solid (``w(1) = 1``). A sheet affects the fields over a window of
        admittances, so a linear map would put the whole void-to-metal
        transition in the first or last fraction of a percent of ``rho``;
        the exponential one centres it near ``rho_c ≈ 1 - log10(eta |Y|) /
        log10(r)``, with ``eta |Y|`` the solid sheet's admittance relative to
        the wave admittance around it (about 1e2 to 1e6 for metal films,
        so ``rho_c`` between about 0.4 and 0.8 with the default contrast).
        ``linear`` is ``w = rho``, for a density that already is the
        admittance fraction. Grey resistive sheets absorb, so pair the
        design with `gfem.design.project` and a rising ``beta``.
        Differentiable in ``density``.

        Args:
            density: One value per face, in [0, 1].
            scheme: ``"exponential"`` or ``"linear"``.
            contrast: The exponential scheme's ``r``, > 1.
            surface: The surface the field is for (see the class).

        Raises:
            ValueError: Unknown scheme, or ``contrast <= 1``.
        """
        rho = jnp.asarray(density, dtype=jnp.float64)
        if scheme == "linear":
            return cls(fill=rho, surface=surface)
        if scheme != "exponential":
            raise ValueError(
                f"Unknown sheet interpolation scheme {scheme!r}; use "
                "'exponential' or 'linear'."
            )
        r = static_float(contrast, "SheetField contrast", "The contrast is fixed.")
        if not r > 1:
            raise ValueError(f"SheetField: contrast must be > 1, got {r}.")
        log_r = math.log(r)
        return cls(fill=jnp.expm1(rho * log_r) / math.expm1(log_r), surface=surface)

    def __repr__(self) -> str:
        where = f", surface={self.surface!r}" if self.surface else ""
        if is_traced(self.fill):
            return f"SheetField({self.n_faces} faces{where}, fill=<traced>)"
        a = np.asarray(self.fill)
        lo, hi = (float(a.min()), float(a.max())) if a.size else (0.0, 0.0)
        span = f"fill={lo:.4g}" if lo == hi else f"fill in [{lo:.4g}, {hi:.4g}]"
        return f"SheetField({self.n_faces} faces{where}, {span})"


register_fields(SheetField, ("fill",))
