r"""The Kerr nonlinearity of a port mode and the coupling between its waves.

A material's ``nonlinear_index`` (:math:`n_2`, m²/W) and
``two_photon_absorption`` (:math:`\beta_\mathrm{TPA}`, m/W) act on the mode
of a wave port that `gfem.solve` found with `gfem.PortMode`. Two functions
read that result and the model's materials, on every row of its frequency
grid (reduced rows included), so a curve over frequency costs no extra solve:

- `kerr_nonlinearity`: the nonlinear coefficient :math:`\gamma_\mathrm{NL}`
  of each row, the number a pulse-propagation tool takes;
- `kerr_coupling`: the tensor :math:`\gamma_{abcd}` between the waves (the
  rows) of one or several results: self- and cross-phase modulation and
  four-wave mixing.

Both integrate the mode at 1 W on the port's cross-section. They run on
lossy and leaky sections, where they hold while :math:`\alpha \ll \beta`.
Under `jax.grad` they read the traced mode field, so they differentiate with
respect to the section's materials (not the node coordinates).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem._mode_overlap import (
    _COLUMNS,
    _cached_columns,
    _columns,
    _elements,
    _per_element,
    _vector,  # noqa: F401  (the tests read it from here)
    _Waves,  # noqa: F401  (the tests build one)
    _waves,
)
from gfem._pml import layer_tetrahedra
from gfem._pytree import register_fields
from gfem.results import PortModeResult

if TYPE_CHECKING:
    from gfem._engine.kerr_columns import Columns as _Columns
    from gfem.model import Model

__all__ = ["KerrCoupling", "KerrNonlinearity", "kerr_coupling", "kerr_nonlinearity"]

_C = scipy.constants.c
_EPS0 = scipy.constants.epsilon_0
_PROPERTIES = ("nonlinear_index", "two_photon_absorption")


@dataclasses.dataclass(frozen=True, eq=False)
class KerrNonlinearity:
    r"""The nonlinear coefficient of a port mode over frequency.

    :math:`\gamma_\mathrm{NL} = \overline{\gamma_K}` with, for the mode
    :math:`E` at 1 W, :math:`k_0 = \omega/c` and :math:`n^2 =
    \mathrm{Re}\,\varepsilon_r` of each element,

    .. math::

        \gamma_K = \frac{k_0 c^2 \varepsilon_0^2}{12} \int_S n^2\,\tilde n_2
        \left[2 |E|^4 + |E\cdot E|^2\right] \mathrm{d}A,
        \qquad \tilde n_2 = n_2 - \frac{i \beta_\mathrm{TPA}}{2 k_0},

    over all three components of :math:`E` (the unconjugated product
    :math:`E\cdot E` too). The power-dependent changes of the propagation
    constant are therefore :math:`\delta\beta = \mathrm{Re}\,\gamma_\mathrm{NL}\,P`
    and :math:`\delta\alpha = \mathrm{Im}\,\gamma_\mathrm{NL}\,P \ge 0`, the
    convention of the generalized nonlinear Schrodinger equation.

    A material with a `Material.nonlinear_anisotropy` :math:`\rho` replaces
    the bracket by :math:`\rho \left[2 |E|^4 + |E\cdot E|^2\right] +
    3(1-\rho) \sum_s |a_s\cdot E|^4`, with :math:`a_s` the rows of its
    `axes` (the cube axes): the bracket at :math:`\rho = 1`, whatever the axes.

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: ``(n_freq,)`` Hz.
        gamma_nl: :math:`\gamma_\mathrm{NL}` at each frequency, ``(n_freq,)``
            complex, 1/(W m); 1/W on a line port of a 2D mesh (the field is
            then 1 W/m and the integral runs over the line). The real part
            is the Kerr effect, the imaginary part the two-photon loss.
    """

    port: str
    mode: int
    frequency: np.ndarray
    gamma_nl: jax.Array

    def __repr__(self) -> str:
        n_f = np.shape(self.frequency)[-1]
        head = (
            f"KerrNonlinearity(port {self.port!r}, mode {self.mode}, {n_f} frequencies"
        )
        try:
            g = np.asarray(self.gamma_nl)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return f"{head}, gamma_nl Re {_range(g.real)}, Im {_range(g.imag)} 1/(W m))"


register_fields(KerrNonlinearity, ("gamma_nl",))


@dataclasses.dataclass(frozen=True, eq=False)
class KerrCoupling:
    r"""The Kerr coupling tensor between the waves of one port.

    A wave is one row of a `gfem.PortModeResult`: a mode at a frequency. In
    the equation of wave :math:`a`'s amplitude (:math:`|A|^2` a power,
    :math:`e^{-i\omega t}`, the convention of pulse-propagation tools),

    .. math::

        \frac{\mathrm{d}A_a}{\mathrm{d}z} \ni i \sum_{bcd}
        \gamma_{abcd}\, A_b A_c \overline{A_d},

    summed over every index tuple with no degeneracy factors, so the factor 2
    of cross-phase modulation and of four-wave mixing comes out of the sum.
    :math:`\gamma_{abcd} = \overline{T_{abcd}}`,

    .. math::

        T_{abcd} = \frac{\omega_a \varepsilon_0}{16} \int_S \chi_a\,
        S(E_b, E_c, \overline{E_a}, \overline{E_d})\,\mathrm{d}A,
        \quad
        \chi_a = \tfrac43 n_a^2 c \varepsilon_0
        \left(n_2 - \frac{i \beta_\mathrm{TPA} c}{2 \omega_a}\right),

    :math:`S(a,b,c,d) = (a\cdot b)(c\cdot d) + (a\cdot c)(b\cdot d) +
    (a\cdot d)(b\cdot c)` with unconjugated dot products, :math:`E` the modes
    at 1 W and :math:`n_a^2 = \mathrm{Re}\,\varepsilon_r` at wave :math:`a`'s
    frequency. :math:`\gamma_{aaaa}` is `KerrNonlinearity`'s
    :math:`\gamma_\mathrm{NL}`. Every tuple is computed; an equation uses
    only those that conserve energy (:math:`\omega_a + \omega_b = \omega_c +
    \omega_d` up to ordering).

    A material with a `Material.nonlinear_anisotropy` :math:`\rho` replaces
    :math:`S` by :math:`S_\rho = \rho S + 3(1-\rho)\sum_s (a_s\cdot
    u)(a_s\cdot v)(a_s\cdot w)(a_s\cdot z)` over the rows :math:`a_s` of its
    `axes` (the cube axes), which is :math:`S` at :math:`\rho = 1`.

    Attributes:
        port: The wave port's name.
        waves: ``(mode, frequency)`` of each wave, in order (Hz).
        gamma_nl: :math:`\gamma_{abcd}`, ``(K, K, K, K)`` complex, 1/(W m);
            1/W on a line port of a 2D mesh.
    """

    port: str
    waves: tuple[tuple[int, float], ...]
    gamma_nl: jax.Array

    def __repr__(self) -> str:
        head = f"KerrCoupling(port {self.port!r}, {len(self.waves)} waves"
        try:
            g = np.asarray(self.gamma_nl)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return f"{head}, |gamma_nl| up to {np.abs(g).max():.4g} 1/(W m))"


register_fields(KerrCoupling, ("gamma_nl",))


def _range(a: np.ndarray) -> str:
    lo, hi = float(a.min()), float(a.max())
    return f"{lo:.4g}" if lo == hi else f"{lo:.4g}-{hi:.4g}"


def _kerr_columns(model: Model, section: Any, port: str, space: Any) -> _Columns:
    """The columns of the Kerr overlaps: this module's cache and builder."""
    return _cached_columns(model, section, port, space, store=_COLUMNS, build=_columns)


# --- materials -------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Kerr:
    """The Kerr material values of the section's elements, in the columns' order.

    ``n2`` and ``beta`` ``(n_el,)`` (0 where an element has none), ``eps``
    ``(n_wave, n_el)`` the real part of the permittivity at each wave's
    frequency, ``keep`` ``(n_el,)`` False on a perfectly matched layer's
    triangles. ``rho`` ``(n_el,)`` and ``axes`` ``(n_el, 3, 3)`` (the crystal
    axes as rows) are the Kerr anisotropy of each element and its frame, both
    ``None`` when no element carries a `Material.nonlinear_anisotropy`: the
    isotropic path then runs unchanged.
    """

    n2: jax.Array
    beta: jax.Array
    eps: jax.Array
    keep: np.ndarray
    rho: jax.Array | None = None
    axes: np.ndarray | None = None


def _anisotropic(material: Material | MaterialField) -> bool:
    # a Material's axes with one permittivity only orient its crystal (the Kerr
    # anisotropy); a field's axes mean a tensor, as everywhere in the engine
    if isinstance(material, Material):
        return not material.isotropic
    return material.axes is not None


def _crystal_components(e: jax.Array, axes: jax.Array) -> jax.Array:
    r""":math:`a_s\cdot E` of every point: ``(..., n_c)`` fields, ``(..., 3)`` result.

    ``axes`` broadcasts against ``e`` as ``(..., 3, 3)`` rows. The field of an
    Ez line port has one component, along :math:`\hat z`.
    """
    if e.shape[-1] == 1:
        return e * axes[..., :, 2]
    return jnp.einsum("...c,...sc->...s", e, axes)


def _region_permittivity(
    material: Material | MaterialField,
    pos: np.ndarray,
    frequency: np.ndarray,
    *,
    port: str,
    region: str,
    owner: str,
) -> jax.Array:
    r""":math:`\mathrm{Re}\,\varepsilon_r` of a region's elements at each frequency."""
    if isinstance(material, Material):
        eps = material.eps_r
        if isinstance(eps, PermittivityLaw):
            return jnp.real(eps(jnp.asarray(frequency)))[:, None] * jnp.ones(len(pos))
        return jnp.full((len(frequency), len(pos)), float(np.real(eps)))
    if material.eps_r is None:
        raise NotImplementedError(
            f"{owner}: region {region!r} of wave port {port!r}'s section is a "
            "density blend with a permittivity-law end or a layer grid and carries "
            "a Kerr property: its permittivity is not a per-element value. Give "
            "the elements beside the port a region of their own with a constant "
            "gfem.Material."
        )
    eps = jnp.real(jnp.asarray(material.eps_r))
    return jnp.broadcast_to(_per_element(eps, pos), (len(frequency), len(pos)))


def _kerr_materials(
    model: Model, section: Any, frequency: np.ndarray, port: str, owner: str
) -> _Kerr:
    """Read the Kerr properties of the elements of ``section`` by owner element.

    Raises:
        ValueError: No element outside a perfectly matched layer carries a
            Kerr property.
        NotImplementedError: A region carrying one is anisotropic, a density
            blend without a per-element permittivity, or a layer-grid design
            blending a Kerr property.
    """
    mesh = model.mesh
    cells, tags = _elements(model, section)
    names = mesh._region_names()  # noqa: SLF001
    n_el = len(cells)
    n2 = jnp.zeros(n_el)
    beta = jnp.zeros(n_el)
    eps = jnp.zeros((len(frequency), n_el))
    carries = np.zeros(n_el, bool)
    rho = np.ones(n_el)
    frames = np.broadcast_to(np.eye(3), (n_el, 3, 3)).copy()
    anisotropy = False
    for tag in np.unique(tags):
        region = names[int(tag)]
        material = model.materials[region]
        given = {n: getattr(material, n) for n in _PROPERTIES}
        ends = getattr(material, "blend_ends", None)
        if ends is not None and ends.pixels is not None:
            named = [n for n, _, _ in ends.others if n in _PROPERTIES]
            if named:
                raise NotImplementedError(
                    f"{owner}: region {region!r} of wave port {port!r}'s section is "
                    f"a layer-grid design that blends {named[0]!r}: a Kerr "
                    "property is not a per-element value there in this version. "
                    "Give the elements beside the port a region of their own with "
                    "a constant gfem.Material."
                )
        if all(v is None for v in given.values()):
            continue
        if _anisotropic(material):
            raise NotImplementedError(
                f"{owner}: region {region!r} of wave port {port!r}'s section is "
                "anisotropic and carries a Kerr property: the refractive index "
                "squared and the isotropic nonlinearity are not defined on a "
                "permittivity tensor in this version. Give the region an isotropic "
                "eps_r."
            )
        rows = np.flatnonzero(tags == tag)
        pos = np.searchsorted(mesh.elements(region), cells[rows])
        carries[rows] = True
        if isinstance(material, Material) and material.nonlinear_anisotropy is not None:
            anisotropy = True
            rho[rows] = material.nonlinear_anisotropy
            if material.axes is not None:
                frames[rows] = np.asarray(material.axes)
        if given["nonlinear_index"] is not None:
            n2 = n2.at[rows].set(_per_element(given["nonlinear_index"], pos))
        if given["two_photon_absorption"] is not None:
            beta = beta.at[rows].set(_per_element(given["two_photon_absorption"], pos))
        eps = eps.at[:, rows].set(
            _region_permittivity(
                material, pos, frequency, port=port, region=region, owner=owner
            )
        )
    # from the mesh, as the section of a traced field has no layer to read
    layer = layer_tetrahedra(mesh, model.pml) if len(section.normal) == 3 else None
    keep = np.ones(n_el, bool) if layer is None else ~layer[cells]
    if not (carries & keep).any():
        regions = ", ".join(repr(names[int(t)]) for t in np.unique(tags))
        raise ValueError(
            f"{owner}: no element of wave port {port!r}'s section outside a "
            "perfectly matched layer (whose triangles are skipped) has a "
            "nonlinear_index or a two_photon_absorption (its regions are "
            f"{regions}), so the nonlinearity is zero. Set one on the core's "
            "material."
        )
    if not anisotropy:
        return _Kerr(n2, beta, eps, keep)
    return _Kerr(n2, beta, eps, keep, jnp.asarray(rho), frames)


# --- public functions ------------------------------------------------------------


def kerr_nonlinearity(model: Model, mode: PortModeResult) -> KerrNonlinearity:
    r"""The nonlinear coefficient :math:`\gamma_\mathrm{NL}` of a port mode.

    Reads the electric field of ``mode`` (the 1 W mode of a `gfem.PortMode`
    solve) on every row of its frequency grid, reduced rows included, and the
    Kerr properties of ``model``'s materials. :math:`n^2 =
    \mathrm{Re}\,\varepsilon_r` follows each region's permittivity (or law) at
    the row's frequency; ``nonlinear_index`` and ``two_photon_absorption``
    are constants over the grid, given at its centre, and an element without
    one counts as zero. See `KerrNonlinearity` for the formula and the sign
    convention: the real part is :math:`\delta\beta/P`, the imaginary part
    :math:`\delta\alpha/P \ge 0` is the two-photon loss.

    The integral runs over the cross-section outside any perfectly matched
    layer (its triangles are skipped), with a quadrature rule exact for the
    mode's polynomial degree. A lossy or leaky section is computed with the
    mode normalized to 1 W; the formula holds while :math:`\alpha \ll \beta`.
    An Ez line port of a 2D mesh gives the slab's value, in 1/W. For a
    single-material core, :math:`A_\mathrm{eff} = k_0 n_2 / \mathrm{Re}\,
    \gamma_\mathrm{NL}`.

    ``nonlinear_anisotropy`` :math:`\rho` (a cubic crystal such as silicon,
    with its cube axes in ``axes``) makes the coefficient depend on the field's
    direction in the crystal; ``nonlinear_index`` and ``two_photon_absorption``
    then refer to light polarized along a cube axis. See `KerrNonlinearity`.

    Exactly linear in ``nonlinear_index`` and ``two_photon_absorption``;
    under `jax.grad` the traced mode field makes it differentiable with
    respect to the section's ``eps_r``, laws, losses and density blends (one 2D
    adjoint per row), not to the node coordinates (width is co-designed by
    finite differences over a remesh). The inputs of the solve can still be
    differentiated through the exact :math:`\gamma` of the same rows.

    Args:
        model: The model the mode was solved on, with the Kerr properties.
        mode: The result of ``gfem.solve(model, gfem.PortMode(...))``,
            concrete or traced.

    Returns:
        The coefficient on every row of ``mode``'s grid.

    Raises:
        ValueError: No element outside a perfectly matched layer carries a
            Kerr property.
        NotImplementedError: A region carrying one has an anisotropic
            permittivity, or is a density blend without a per-element
            permittivity; an Hz line port.
        TypeError: ``mode`` is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`.
    """
    owner = "gfem.kerr_nonlinearity"
    if not isinstance(mode, PortModeResult):
        raise TypeError(
            f"{owner} takes one gfem.PortModeResult, got {type(mode).__name__}: "
            "solve gfem.solve(model, gfem.PortMode(...)) and pass that result "
            "(gfem.kerr_coupling takes several)."
        )
    waves = _waves(model, mode, owner, _kerr_materials, _kerr_columns)
    weights = waves.weights
    kerr = waves.materials
    rho = None if kerr.rho is None else kerr.rho[:, None]
    frames = None if kerr.axes is None else jnp.asarray(kerr.axes)[:, None]

    @jax.checkpoint
    def integral(vector: jax.Array) -> jax.Array:
        e = waves.row(vector)
        re, im = jnp.real(e), jnp.imag(e)
        e2 = jnp.sum(re * re + im * im, axis=-1)
        ee = jnp.sum(e * e, axis=-1)
        quartic = 2 * e2 * e2 + jnp.real(ee) ** 2 + jnp.imag(ee) ** 2
        if rho is not None and frames is not None:
            # S_rho: the cubic part sums |a_s . E|^4 over the crystal axes
            p = _crystal_components(e, frames)
            cubic = jnp.sum(jnp.real(p * jnp.conj(p)) ** 2, axis=-1)
            quartic = rho * quartic + 3 * (1 - rho) * cubic
        return jnp.sum(weights * quartic, axis=-1)

    # one row at a time: the point values of a row are (n_el, n_q, 3) complex
    per_element = jax.lax.map(integral, waves.vectors)  # (n_wave, n_el)
    k0 = jnp.asarray(2 * np.pi * waves.frequency / _C)[:, None]
    # conjugate of n2 - i beta / (2 k0): the e^{-i omega t} convention
    n2 = kerr.n2[None] + 1j * kerr.beta[None] / (2 * k0)
    gamma = (k0[:, 0] * _C**2 * _EPS0**2 / 12) * jnp.sum(
        kerr.eps * n2 * per_element, axis=-1
    )
    return KerrNonlinearity(
        port=waves.port,
        mode=int(mode.mode),
        frequency=np.asarray(waves.frequency),
        gamma_nl=gamma,
    )


def kerr_coupling(
    model: Model, modes: PortModeResult | Sequence[PortModeResult]
) -> KerrCoupling:
    r"""The Kerr coupling tensor :math:`\gamma_{abcd}` between the waves of a port.

    Every row of every result in ``modes`` is one wave, in order: another mode
    number, another frequency, or both. Entry :math:`abcd` is the coefficient
    of :math:`A_b A_c \overline{A_d}` in wave :math:`a`'s amplitude equation,
    summed over all index tuples with no degeneracy factors; its special cases
    are self-phase (:math:`aaaa`), cross-phase and four-wave mixing. See
    `KerrCoupling` for the formula and the sign convention. The diagonal
    :math:`\gamma_{aaaa}` equals `kerr_nonlinearity`.

    Every tuple is computed, energy-conserving or not; an equation uses
    only those with :math:`\omega_a + \omega_b = \omega_c + \omega_d`. The
    phase mismatch of a four-wave mixing process, :math:`\Delta\beta = 2\beta_p
    - \beta_s - \beta_i`, is one line on the results' exact ``gamma``.

    All waves must share one port, element order and cross-section (solve them
    with one `gfem.SolverOptions` on one model). The cost is
    :math:`K^4` per quadrature point, so pass the few waves of a process, not
    a whole frequency grid (that is `kerr_nonlinearity`'s diagonal).
    Everything else is as `kerr_nonlinearity`: material values, the layer,
    lossy sections, and the gradient through the mode fields and materials.

    Args:
        model: The model the modes were solved on, with the Kerr properties.
        modes: One `gfem.PortModeResult` or a sequence of them.

    Returns:
        The ``(K, K, K, K)`` tensor for the :math:`K` waves.

    Raises:
        ValueError: The results are of different ports, element orders or
            section geometries; no element outside a perfectly matched layer
            carries a Kerr property.
        NotImplementedError: As `kerr_nonlinearity`.
        TypeError: An entry is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`.
    """
    waves = _waves(model, modes, "gfem.kerr_coupling", _kerr_materials, _kerr_columns)
    e = waves.field()
    n_wave, n_el, n_q, n_c = e.shape
    points = e.reshape(n_wave, n_el * n_q, n_c)
    omega = jnp.asarray(2 * np.pi * waves.frequency)
    kerr = waves.materials
    chi = (
        (4 / 3)
        * kerr.eps
        * _C
        * _EPS0
        * (kerr.n2[None] - 1j * kerr.beta[None] * _C / (2 * omega[:, None]))
    )
    x = jnp.repeat(chi, n_q, axis=1) * waves.weights.reshape(-1)[None]
    if kerr.rho is not None:
        rho = jnp.repeat(kerr.rho, n_q)[None]
        x_cubic = 3 * (1 - rho) * x
        x = rho * x
    p = jnp.einsum("bnx,cnx->bcn", points, points)
    q = jnp.einsum("bnx,cnx->bcn", points, jnp.conj(points))
    t = (
        jnp.einsum("an,bcn,adn->abcd", x, p, jnp.conj(p))
        + jnp.einsum("an,ban,cdn->abcd", x, q, q)
        + jnp.einsum("an,bdn,can->abcd", x, q, q)
    )
    if kerr.rho is not None:
        frames = jnp.repeat(jnp.asarray(kerr.axes), n_q, axis=0)
        r = _crystal_components(points, frames[None])  # (n_wave, n, 3)
        t = t + jnp.einsum(
            "an,bns,cns,ans,dns->abcd", x_cubic, r, r, jnp.conj(r), jnp.conj(r)
        )
    gamma = jnp.conj(omega[:, None, None, None] * _EPS0 / 16 * t)
    return KerrCoupling(
        port=waves.port,
        waves=tuple(
            (m, float(f)) for m, f in zip(waves.modes, waves.frequency, strict=True)
        ),
        gamma_nl=gamma,
    )
