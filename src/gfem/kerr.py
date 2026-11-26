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
  four-wave mixing;
- `free_carriers`: the confinement and the pair generation of the regions
  where two-photon absorption makes carriers.

The first two integrate the mode at 1 W on the port's cross-section. They run on
lossy and leaky sections, where they hold while :math:`\alpha \ll \beta`.
Under `jax.grad` they read the traced mode field, so they differentiate with
respect to the section's materials and to the node coordinates of a planar
port (the section's elements move with them, the mode's DOFs held).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._checks import is_traced
from gfem._materials import PROPERTIES, Material, MaterialField, PermittivityLaw
from gfem._mode_overlap import (
    _COLUMNS,
    _cached_columns,
    _columns,
    _elements,
    _per_element,
    _vector,  # noqa: F401  (the tests read it from here)
    _Waves,  # noqa: F401  (the tests build one)
    _waves,
    refuse_bend,
)
from gfem._pml import layer_tetrahedra
from gfem._pytree import register_fields
from gfem.results import PortModeResult

if TYPE_CHECKING:
    from gfem._engine.kerr_columns import Columns as _Columns
    from gfem.model import Model

__all__ = [
    "FreeCarriers",
    "KerrCoupling",
    "KerrNonlinearity",
    "free_carriers",
    "kerr_coupling",
    "kerr_nonlinearity",
]

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
    ``axes`` (the cube axes): the bracket at :math:`\rho = 1`, whatever the axes.

    A region whose `Material.eps_r` has three principal values along ``axes``
    (a birefringent crystal) replaces :math:`n^2` and the products
    :math:`u\cdot v` by :math:`N = \sum_k n_k a_k a_k^\mathsf{T}`,
    :math:`n_k = \sqrt{\mathrm{Re}\,\varepsilon_{r,k}}`: the bracket becomes
    :math:`2 (E^\mathsf{H} N E)^2 + |E^\mathsf{T} N E|^2`, which is
    :math:`n^2` times the isotropic one for :math:`N = n I`. The scalar
    :math:`n_2` is then the index change :math:`n_2 I` of light polarized
    along each principal axis.

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
    ``axes`` (the cube axes), which is :math:`S` at :math:`\rho = 1`. A region
    with a birefringent `Material.eps_r` replaces :math:`n_a^2 S` by
    :math:`S_N`, every :math:`u\cdot v` of :math:`S` becoming
    :math:`u^\mathsf{T} N_a v` with :math:`N_a = \sum_k n_k a_k a_k^\mathsf{T}`
    at wave :math:`a`'s frequency.

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


@dataclasses.dataclass(frozen=True, eq=False)
class FreeCarriers:
    r"""The free-carrier figures of a port mode over frequency.

    Free carriers are made by two-photon absorption in the regions :math:`R`
    and spread uniformly over them; :math:`N` is their density (pairs per
    m³). For the mode :math:`E` at 1 W, :math:`n = \sqrt{\mathrm{Re}\,
    \varepsilon_r}` of each element and :math:`k_0 = \omega/c`:

    .. math::

        \Gamma_R = \frac{c\varepsilon_0}{2} \int_R n\,|E|^2\,\mathrm{d}A,
        \qquad
        \theta = \frac{\mathrm{Im}\,\gamma_{\mathrm{NL},R}}{\hbar\omega A_R},

    with :math:`\mathrm{Im}\,\gamma_{\mathrm{NL},R} = \frac{c^2\varepsilon_0^2}{24}
    \int_R n^2 \beta_\mathrm{TPA} [2|E|^4 + |E\cdot E|^2]\,\mathrm{d}A`, the
    imaginary part of `KerrNonlinearity`'s integral restricted to :math:`R`
    (so with its :math:`\rho` and :math:`N` forms). A uniform index change
    :math:`\Delta\tilde n = \Delta n_\mathrm{FC} - i\sigma N/(2k_0)` in
    :math:`R` gives :math:`\delta n_\mathrm{eff} = \Gamma_R\Delta\tilde n`: a
    propagation-constant change :math:`\delta\beta = k_0 \Gamma_R
    \Delta n_\mathrm{FC}` and a power loss :math:`2\delta\alpha = \Gamma_R
    \sigma N`. :math:`\Gamma_R` can exceed 1; :math:`\sum_R n_R \Gamma_R = n_g`
    for nondispersive constant materials. In a birefringent region
    :math:`n|E|^2` reads :math:`E^\mathsf{H} N E`. The carrier density follows
    :math:`\mathrm{d}N/\mathrm{d}t = \theta P^2 - N/\tau` for the power
    :math:`P` in the guide, one pair per two photons absorbed.

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: ``(n_freq,)`` Hz.
        regions: The names of the carrier regions :math:`R`.
        overlap: :math:`\Gamma_R` at each frequency, ``(n_freq,)``, dimensionless.
        area: :math:`A_R`, the cross-section of :math:`R` outside any perfectly
            matched layer, m² (m on a line port).
        generation: :math:`\theta` at each frequency, ``(n_freq,)``,
            1/(W² m³ s); 1/(W² m s) on a line port of a 2D mesh, where the
            power per length is in W/m.
    """

    port: str
    mode: int
    frequency: np.ndarray
    regions: tuple[str, ...]
    overlap: jax.Array
    area: float | jax.Array
    generation: jax.Array

    def __repr__(self) -> str:
        n_f = np.shape(self.frequency)[-1]
        head = (
            f"FreeCarriers(port {self.port!r}, mode {self.mode}, {n_f} frequencies, "
            f"regions {self.regions}"
        )
        try:
            g = np.asarray(self.overlap)
            t = np.asarray(self.generation)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return f"{head}, overlap {_range(g)}, generation {_range(t)} 1/(W^2 m^3 s))"


register_fields(FreeCarriers, ("overlap", "area", "generation"))


def _range(a: np.ndarray) -> str:
    lo, hi = float(a.min()), float(a.max())
    return f"{lo:.4g}" if lo == hi else f"{lo:.4g}-{hi:.4g}"


def _kerr_columns(model: Model, section: Any, port: str, space: Any) -> _Columns:
    """The columns of the Kerr overlaps: this module's cache and builder."""
    return _cached_columns(model, section, port, space, store=_COLUMNS, build=_columns)


# --- materials -------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Kerr:
    r"""The Kerr material values of the section's elements, in the columns' order.

    ``n2`` and ``beta`` ``(n_el,)`` (0 where an element has none), ``eps``
    ``(n_wave, n_el)`` the real part of the permittivity at each wave's
    frequency, ``keep`` ``(n_el,)`` False on a perfectly matched layer's
    triangles. ``rho`` ``(n_el,)`` and ``axes`` ``(n_el, 3, 3)`` (the crystal
    axes as rows) are the Kerr anisotropy of each element and its frame, both
    ``None`` when no element carries a `Material.nonlinear_anisotropy`: the
    isotropic path then runs unchanged. ``index`` ``(n_wave, n_el, 3, 3)`` is
    the real symmetric matrix :math:`N = \sum_k n_k a_k a_k^\mathsf{T}` of each
    element at each wave's frequency, ``None`` unless a region has a
    birefringent ``eps_r``; it replaces ``eps`` (and excludes ``rho``).
    ``member`` ``(n_el,)`` marks the elements of the carrier regions of
    `free_carriers`, ``None`` for the Kerr functions.
    """

    n2: jax.Array
    beta: jax.Array
    eps: jax.Array
    keep: np.ndarray
    rho: jax.Array | None = None
    axes: np.ndarray | None = None
    index: jax.Array | None = None
    member: np.ndarray | None = None


def _birefringent(material: Material | MaterialField) -> bool:
    # a Material's axes with one permittivity only orient its crystal (the Kerr
    # anisotropy); three principal values of eps_r make it birefringent. A
    # field's axes mean a tensor, as everywhere in the engine, and are refused
    return isinstance(material, Material) and isinstance(material.eps_r, tuple)


def _refuse_tensor(
    material: Material | MaterialField, region: str, port: str, owner: str
) -> None:
    """Refuse a tensor the overlap cannot read: only a `Material`'s ``eps_r`` may be.

    Raises:
        NotImplementedError: ``material`` is an anisotropic
            `gfem.MaterialField`, or a `gfem.Material` with a tensor property
            other than ``eps_r``.
    """
    if isinstance(material, MaterialField) and material.axes is not None:
        raise NotImplementedError(
            f"{owner}: region {region!r} of wave port {port!r}'s section is "
            "an anisotropic gfem.MaterialField and carries a Kerr property: "
            "the index matrix is read from a constant gfem.Material only. "
            "Give the region a gfem.Material with a three-value eps_r and "
            "axes, or an isotropic eps_r."
        )
    if not isinstance(material, Material):
        return
    for p in PROPERTIES:
        if (
            p.may_be_tensor
            and p.name != "eps_r"
            and isinstance(getattr(material, p.name), tuple)
        ):
            raise NotImplementedError(
                f"{owner}: region {region!r} of wave port {port!r}'s section is "
                f"anisotropic in {p.name} and is read by the overlap: only eps_r "
                "may be a tensor there (read through the index matrix) in this "
                f"version. Give the region a scalar {p.name}."
            )


def _index_matrix(n: jax.Array, axes: np.ndarray) -> jax.Array:
    r""":math:`N = \sum_k n_k a_k a_k^\mathsf{T}` from ``(..., 3)`` indices and rows."""
    return jnp.einsum("...k,...ki,...kj->...ij", n, axes, axes)


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


def _assemble(
    kerr: _Kerr,
    rho: np.ndarray | None,
    frames: np.ndarray,
    principal: np.ndarray,
    birefringent: np.ndarray,
    *,
    owner: str,
    port: str,
) -> _Kerr:
    """Add the anisotropy (``rho``) or the index matrix of the birefringent elements."""
    if not birefringent.any():
        if rho is None:
            return kerr
        return dataclasses.replace(kerr, rho=jnp.asarray(rho), axes=frames)
    if rho is not None:
        raise NotImplementedError(
            f"{owner}: wave port {port!r}'s section has a region with a "
            "nonlinear_anisotropy and another with a birefringent eps_r: the "
            "two forms are not combined in this version. Give the section "
            "one or the other."
        )
    # N = n I on an isotropic element, so that n^2 S is the isotropic formula
    n = jnp.where(
        birefringent[None, :, None],
        jnp.asarray(principal)[None],
        jnp.sqrt(kerr.eps)[..., None],
    )
    return dataclasses.replace(kerr, index=_index_matrix(n, frames))


def _no_carrier_message(owner: str, port: str, names: Any, tags: np.ndarray) -> str:
    regions = ", ".join(repr(names[int(t)]) for t in np.unique(tags))
    return (
        f"{owner}: no element of wave port {port!r}'s section outside a "
        "perfectly matched layer (whose triangles are skipped) has a "
        "nonlinear_index or a two_photon_absorption (its regions are "
        f"{regions}), so the nonlinearity is zero. Set one on the core's "
        "material."
    )


def _kerr_materials(
    model: Model,
    section: Any,
    frequency: np.ndarray,
    port: str,
    owner: str,
    *,
    also: Sequence[str] = (),
) -> _Kerr:
    """Read the Kerr properties of the elements of ``section`` by owner element.

    The permittivity of the regions named in ``also`` is read whether they
    carry a Kerr property or not (the free carriers' overlap needs it).

    Raises:
        ValueError: No element outside a perfectly matched layer carries a
            Kerr property.
        NotImplementedError: A region carrying one is an anisotropic
            `gfem.MaterialField`, a `gfem.Material` with a tensor property
            other than ``eps_r``, a density blend without a per-element
            permittivity, or a layer-grid design blending a Kerr property;
            a birefringent region beside a `Material.nonlinear_anisotropy`.
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
    principal = np.zeros((n_el, 3))
    birefringent = np.zeros(n_el, bool)
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
        if region not in also and all(v is None for v in given.values()):
            continue
        _refuse_tensor(material, region, port, owner)
        rows = np.flatnonzero(tags == tag)
        pos = np.searchsorted(mesh.elements(region), cells[rows])
        carries[rows] = any(v is not None for v in given.values())
        if isinstance(material, Material):
            if material.axes is not None:
                frames[rows] = np.asarray(material.axes)
            if material.nonlinear_anisotropy is not None:
                anisotropy = True
                rho[rows] = material.nonlinear_anisotropy
        if given["nonlinear_index"] is not None:
            n2 = n2.at[rows].set(_per_element(given["nonlinear_index"], pos))
        if given["two_photon_absorption"] is not None:
            beta = beta.at[rows].set(_per_element(given["two_photon_absorption"], pos))
        if _birefringent(material):
            values = np.sqrt(np.real(np.asarray(material.eps_r, dtype=complex)))
            birefringent[rows] = True
            principal[rows] = values
            # the index matrix replaces this value, which only keeps the array full
            eps = eps.at[:, rows].set(float(np.mean(values**2)))
            continue
        eps = eps.at[:, rows].set(
            _region_permittivity(
                material, pos, frequency, port=port, region=region, owner=owner
            )
        )
    # from the mesh, as the section of a traced field has no layer to read
    layer = layer_tetrahedra(mesh, model.pml) if len(section.normal) == 3 else None
    keep = np.ones(n_el, bool) if layer is None else ~layer[cells]
    if not (carries & keep).any():
        raise ValueError(_no_carrier_message(owner, port, names, tags))
    return _assemble(
        _Kerr(n2, beta, eps, keep),
        rho if anisotropy else None,
        frames,
        principal,
        birefringent,
        owner=owner,
        port=port,
    )


def _isotropic_tensor(
    points: jax.Array, kerr: _Kerr, x: jax.Array, n_q: int
) -> jax.Array:
    r""":math:`T_{abcd}` with :math:`S` or :math:`S_\rho`; ``x`` is :math:`\chi_a`."""
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
    return t


def _birefringent_tensor(
    points: jax.Array, index: jax.Array, x: jax.Array
) -> jax.Array:
    r""":math:`T_{abcd}` with :math:`S_N`, :math:`u\cdot v` as :math:`u^\mathsf{T}N_av`.

    ``index`` ``(n_wave, n, 3, 3)`` is wave :math:`a`'s :math:`N_a` at each point.
    """
    if points.shape[-1] == 1:  # an Ez line port: the zz entry of N
        index = index[..., 2:, 2:]
    ne = jnp.einsum("anxy,cny->acnx", index, points)  # N_a E_c
    # P[a, b, c] = E_b . N_a E_c and R[a, b, c] = E_b . N_a conj(E_c), N real
    p = jnp.einsum("bnx,acnx->abcn", points, ne)
    r = jnp.einsum("bnx,acnx->abcn", points, jnp.conj(ne))
    p_aa = jnp.einsum("aadn->adn", p)  # E_a . N_a E_d
    r_a = jnp.einsum("aban->abn", r)  # E_b . N_a conj(E_a)
    return (
        jnp.einsum("an,abcn,adn->abcd", x, p, jnp.conj(p_aa))
        + jnp.einsum("an,abn,acdn->abcd", x, r_a, r)
        + jnp.einsum("an,abdn,acn->abcd", x, r, r_a)
    )


def _refuse_bent_anisotropy(waves: Any, owner: str) -> None:
    """Refuse a bent mode in an anisotropic Kerr material.

    A crystal's axes are fixed in the global frame, so the material is not
    invariant about the bend's axis and the metric does not carry over.
    """
    kerr = waves.materials
    if getattr(waves, "bend", None) is not None and (
        kerr.rho is not None or kerr.index is not None
    ):
        raise NotImplementedError(
            f"{owner}: a bent port mode in a material with nonlinear_anisotropy "
            "or a birefringent index is not supported: the crystal's axes do "
            "not turn with the bend."
        )


def _quartic_overlaps(waves: Any) -> tuple[jax.Array, jax.Array]:
    r"""The Kerr integrand of each element: ``(n_wave, n_el)`` each.

    Returns :math:`\int_e [2|E|^4 + |E\cdot E|^2]\,\mathrm{d}A` (or its
    :math:`\rho` and :math:`N` forms) and the factor :math:`n^2` that goes with it
    (ones where the index matrix holds it already).
    """
    weights = waves.weights
    kerr = waves.materials
    rho = None if kerr.rho is None else kerr.rho[:, None]
    frames = None if kerr.axes is None else jnp.asarray(kerr.axes)[:, None]

    index = kerr.index

    @jax.checkpoint
    def integral(vector: jax.Array, matrix: jax.Array | None) -> jax.Array:
        e = waves.row(vector)
        re, im = jnp.real(e), jnp.imag(e)
        if matrix is not None:
            # 2 (E^H N E)^2 + |E^T N E|^2 with N real symmetric, n^2 S for N = n I
            if e.shape[-1] == 1:  # an Ez line port: the zz entry of N
                matrix = matrix[:, 2:, 2:]
            ne = jnp.einsum("eij,eqj->eqi", matrix, e)
            hh = jnp.sum(re * jnp.real(ne) + im * jnp.imag(ne), axis=-1)
            tt = jnp.sum(e * ne, axis=-1)
            quartic = 2 * hh * hh + jnp.real(tt) ** 2 + jnp.imag(tt) ** 2
            return jnp.sum(weights * quartic, axis=-1)
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
    if index is None:
        per_element = jax.lax.map(lambda v: integral(v, None), waves.vectors)
    else:
        per_element = jax.lax.map(lambda a: integral(*a), (waves.vectors, index))
    # the index matrix holds n^2 already
    eps = kerr.eps if index is None else jnp.ones_like(kerr.eps)
    return per_element, eps


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

    On a bent mode (`gfem.Bend`) the integrand is that of the physical field,
    :math:`E_s = e_n/h`, weighted by the metric factor :math:`h` (the physical
    volume element), per unit length of the reference arc; it also
    differentiates with respect to the bend's radius.

    Exactly linear in ``nonlinear_index`` and ``two_photon_absorption``;
    under `jax.grad` the traced mode field makes it differentiable with
    respect to the section's ``eps_r``, laws, losses and density blends (one 2D
    adjoint per row) and to the node coordinates of a planar port: the section's
    elements move, the mode's field with them (its own shape adjoint), and the
    overlap is read on the moved triangles. A `gfem.MaterialField`'s values
    belong to their elements and move with them, in the overlap as in the mode
    solve: the node derivative holds each element's material, not the material
    at a fixed point. A line port of a 2D model has no node derivative. For a
    node derivative, solve ``mode`` on ``model`` inside the differentiated
    function. A mode's phase is free: under node gradients an objective that
    depends on it raises `gfem.NonDifferentiableError` in the backward pass,
    which reaches the caller as ``jax.errors.JaxRuntimeError`` carrying its
    message.

    Args:
        model: The model the mode was solved on, with the Kerr properties.
        mode: The result of ``gfem.solve(model, gfem.PortMode(...))``,
            concrete or traced.

    Returns:
        The coefficient on every row of ``mode``'s grid.

    Raises:
        ValueError: No element outside a perfectly matched layer carries a
            Kerr property.
        NotImplementedError: A region carrying one is an anisotropic
            `gfem.MaterialField`, a `gfem.Material` with a tensor property
            other than ``eps_r``, or a density blend without a per-element
            permittivity; a section with both a birefringent region and a
            `Material.nonlinear_anisotropy`; an Hz line port; a bent mode
            (`gfem.Bend`) in a birefringent or anisotropic Kerr material.
        TypeError: ``mode`` is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`
            on a line port of a 2D model; those of ``model`` are traced and
            those of a mode's solve are not, or the reverse.
        TypeError: The node coordinates are traced by `jax.jit`.
    """
    owner = "gfem.kerr_nonlinearity"
    if not isinstance(mode, PortModeResult):
        raise TypeError(
            f"{owner} takes one gfem.PortModeResult, got {type(mode).__name__}: "
            "solve gfem.solve(model, gfem.PortMode(...)) and pass that result "
            "(gfem.kerr_coupling takes several)."
        )
    waves = _waves(model, mode, owner, _kerr_materials, _kerr_columns, shape=True)
    _refuse_bent_anisotropy(waves, owner)
    per_element, eps = _quartic_overlaps(waves)
    kerr = waves.materials
    k0 = jnp.asarray(2 * np.pi * waves.frequency / _C)[:, None]
    # conjugate of n2 - i beta / (2 k0): the e^{-i omega t} convention
    n2 = kerr.n2[None] + 1j * kerr.beta[None] / (2 * k0)
    gamma = (k0[:, 0] * _C**2 * _EPS0**2 / 12) * jnp.sum(
        eps * n2 * per_element, axis=-1
    )
    return KerrNonlinearity(
        port=waves.port,
        mode=int(mode.mode),
        frequency=np.asarray(waves.frequency),
        gamma_nl=gamma,
    )


def _linear_overlaps(waves: Any) -> jax.Array:
    r""":math:`\int_e E^\mathsf{H} N E\,\mathrm{d}A` per element, ``(n_wave, n_el)``.

    With the index matrix :math:`N` of a birefringent region; :math:`|E|^2`
    otherwise, which the caller multiplies by :math:`n`.
    """
    weights = waves.weights
    index = waves.materials.index

    @jax.checkpoint
    def integral(vector: jax.Array, matrix: jax.Array | None) -> jax.Array:
        e = waves.row(vector)
        if matrix is None:
            density = jnp.sum(jnp.real(e * jnp.conj(e)), axis=-1)
        else:
            if e.shape[-1] == 1:  # an Ez line port: the zz entry of N
                matrix = matrix[:, 2:, 2:]
            ne = jnp.einsum("eij,eqj->eqi", matrix, e)
            density = jnp.sum(jnp.real(jnp.conj(e) * ne), axis=-1)
        return jnp.sum(weights * density, axis=-1)

    if index is None:
        return jax.lax.map(lambda v: integral(v, None), waves.vectors)
    return jax.lax.map(lambda a: integral(*a), (waves.vectors, index))


def _carrier_regions(
    model: Model, section: Any, regions: Any, owner: str, port: str
) -> tuple[tuple[str, ...], np.ndarray]:
    """The carrier regions and the mask of their elements outside a layer."""
    mesh = model.mesh
    cells, tags = _elements(model, section)
    names = mesh._region_names()  # noqa: SLF001
    present = [names[int(t)] for t in np.unique(tags)]
    tag_of = {name: tag for tag, name in names.items()}
    layer = layer_tetrahedra(mesh, model.pml) if len(section.normal) == 3 else None
    keep = np.ones(len(cells), bool) if layer is None else ~layer[cells]

    def tpa(region: str) -> bool:
        return (
            getattr(model.materials[region], "two_photon_absorption", None) is not None
        )

    def outside(region: str) -> bool:
        return bool(keep[tags == tag_of[region]].any())

    if regions is None:
        chosen = tuple(r for r in present if tpa(r) and outside(r))
        if not chosen:
            raise ValueError(
                f"{owner}: no region of wave port {port!r}'s section outside a "
                "perfectly matched layer has a two_photon_absorption (its regions "
                f"are {present}), so no carriers are generated. Set one on the "
                "core's material, or name the regions."
            )
    else:
        chosen = (regions,) if isinstance(regions, str) else tuple(regions)
        if not chosen or not all(isinstance(r, str) for r in chosen):
            raise TypeError(
                f"{owner}: regions is a region name or a sequence of names, got "
                f"{regions!r}."
            )
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"{owner}: the regions {list(chosen)} repeat a name.")
        missing = [r for r in chosen if r not in present]
        if missing:
            raise ValueError(
                f"{owner}: region {missing[0]!r} is not in wave port {port!r}'s "
                f"section, whose regions are {present}."
            )
        inside = [r for r in chosen if not outside(r)]
        if inside:
            raise ValueError(
                f"{owner}: region {inside[0]!r} lies wholly in a perfectly matched "
                "layer, whose triangles are skipped."
            )
        if not any(tpa(r) for r in chosen):
            raise ValueError(
                f"{owner}: the regions {list(chosen)} have no two_photon_absorption, "
                "so no carriers are generated and the generation is zero. Set one "
                "on the material of the region the carriers are made in."
            )
    mask = np.isin(tags, [tag_of[r] for r in chosen]) & keep
    return chosen, mask


def free_carriers(
    model: Model, mode: PortModeResult, *, regions: str | Sequence[str] | None = None
) -> FreeCarriers:
    r"""The free-carrier overlap and generation of a port mode.

    Reads the electric field of ``mode`` (the 1 W mode of a `gfem.PortMode`
    solve) on every row of its frequency grid, reduced rows included, and the
    ``two_photon_absorption`` of ``model``'s materials, as `kerr_nonlinearity`
    does: the same quadrature, the same layer and lossy-section behaviour. See
    `FreeCarriers` for the formulas and units.

    The carrier law is yours: gfem returns the modal overlap
    :math:`\Gamma_R` that turns a carrier density into a loss and an index
    change, and the generation coefficient :math:`\theta`; the cross-section
    :math:`\sigma`, the dispersion coefficient and the lifetime
    :math:`\tau` belong to the pulse tool (:math:`N = \theta\tau P^2` is the
    continuous-wave density). Differentiable through the materials, the traced
    mode field and the node coordinates of a planar port (as
    `kerr_nonlinearity`); exactly linear in
    ``two_photon_absorption``.

    Args:
        model: The model the mode was solved on.
        mode: The result of ``gfem.solve(model, gfem.PortMode(...))``,
            concrete or traced.
        regions: The name or names of the regions the carriers spread
            uniformly over (the silicon core). Defaults to every region of
            the section outside a perfectly matched layer that carries a
            ``two_photon_absorption``.

    Returns:
        The figures on every row of ``mode``'s grid.

    Raises:
        ValueError: A named region is not in the section or lies wholly in a
            perfectly matched layer; the regions carry no
            ``two_photon_absorption``; no region does, by default.
        NotImplementedError: A region is an anisotropic `gfem.MaterialField`,
            a `gfem.Material` with a tensor property other than ``eps_r``, or
            a density blend without a per-element permittivity; an Hz line
            port; a bent mode (`gfem.Bend`).
        TypeError: ``mode`` is not a `gfem.PortModeResult`, or has no electric
            field; ``regions`` is not names.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`
            on a line port of a 2D model; those of ``model`` are traced and
            those of a mode's solve are not, or the reverse.
        TypeError: The node coordinates are traced by `jax.jit`.
    """
    owner = "gfem.free_carriers"
    refuse_bend(owner, mode)
    if not isinstance(mode, PortModeResult):
        raise TypeError(
            f"{owner} takes one gfem.PortModeResult, got {type(mode).__name__}: "
            "solve gfem.solve(model, gfem.PortMode(...)) and pass that result."
        )
    chosen: list[tuple[str, ...]] = []

    def read(
        model: Model, section: Any, frequency: np.ndarray, port: str, owner: str
    ) -> _Kerr:
        names, member = _carrier_regions(model, section, regions, owner, port)
        chosen.append(names)
        kerr = _kerr_materials(model, section, frequency, port, owner, also=names)
        if not is_traced(kerr.beta) and not np.any(np.asarray(kerr.beta)[member]):
            raise ValueError(
                f"{owner}: the regions {list(names)} of wave port {port!r} have a "
                "two_photon_absorption of zero, so the generation is zero."
            )
        return dataclasses.replace(kerr, member=member)

    waves = _waves(model, mode, owner, read, _kerr_columns, shape=True)
    kerr = waves.materials
    member = jnp.asarray(kerr.member)
    quartic, eps = _quartic_overlaps(waves)
    linear = _linear_overlaps(waves)
    # the index matrix holds n already; |E|^2 is weighted by n = sqrt(Re eps)
    n = jnp.sqrt(eps) if kerr.index is None else jnp.ones_like(eps)
    overlap = (_C * _EPS0 / 2) * jnp.sum(member * n * linear, axis=-1)
    if waves.moved is None:
        area = float(np.sum(np.asarray(waves.weights)[kerr.member]))
    else:  # the moving section's area, traced
        area = jnp.sum(waves.weights[jnp.asarray(kerr.member)])
    # Im of gamma_NL's integral over R: k0 c^2 eps0^2/12 * beta/(2 k0)
    absorbed = (_C**2 * _EPS0**2 / 24) * jnp.sum(
        member * eps * kerr.beta[None] * quartic, axis=-1
    )
    energy = scipy.constants.hbar * 2 * np.pi * jnp.asarray(waves.frequency) * area
    return FreeCarriers(
        port=waves.port,
        mode=int(mode.mode),
        frequency=np.asarray(waves.frequency),
        regions=chosen[0],
        overlap=overlap,
        area=area,
        generation=absorbed / energy,
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
    with one `gfem.SolverOptions` on one model); on a bent port, one
    `gfem.Bend`. The cost is
    :math:`K^4` per quadrature point, so pass the few waves of a process, not
    a whole frequency grid (that is `kerr_nonlinearity`'s diagonal).
    Everything else is as `kerr_nonlinearity`: material values, the layer,
    lossy sections, and the gradient through the mode fields, materials and node
    coordinates (a modulus such as :math:`|\gamma_{abcd}|^2` is free of the rows'
    phases; under node gradients an entry whose phases do not cancel is not,
    and raises ``jax.errors.JaxRuntimeError`` carrying the
    `gfem.NonDifferentiableError` message from the backward pass).

    Args:
        model: The model the modes were solved on, with the Kerr properties.
        modes: One `gfem.PortModeResult` or a sequence of them.

    Returns:
        The ``(K, K, K, K)`` tensor for the :math:`K` waves.

    Raises:
        ValueError: The results are of different ports, element orders,
            section geometries or bends (a straight and a bent mode); no
            element outside a perfectly matched layer carries a Kerr property.
        NotImplementedError: As `kerr_nonlinearity`.
        TypeError: An entry is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`
            on a line port of a 2D model; those of ``model`` are traced and
            those of a mode's solve are not, or the reverse.
        TypeError: The node coordinates are traced by `jax.jit`.
    """
    waves = _waves(
        model,
        modes,
        "gfem.kerr_coupling",
        _kerr_materials,
        _kerr_columns,
        shape=True,
    )
    _refuse_bent_anisotropy(waves, "gfem.kerr_coupling")
    e = waves.field()
    n_wave, n_el, n_q, n_c = e.shape
    points = e.reshape(n_wave, n_el * n_q, n_c)
    omega = jnp.asarray(2 * np.pi * waves.frequency)
    kerr = waves.materials
    index = kerr.index
    # the index matrix holds n^2 already
    eps = kerr.eps if index is None else jnp.ones_like(kerr.eps)
    chi = (
        (4 / 3)
        * eps
        * _C
        * _EPS0
        * (kerr.n2[None] - 1j * kerr.beta[None] * _C / (2 * omega[:, None]))
    )
    x = jnp.repeat(chi, n_q, axis=1) * waves.weights.reshape(-1)[None]
    if index is not None:
        t = _birefringent_tensor(points, jnp.repeat(index, n_q, axis=1), x)
    else:
        t = _isotropic_tensor(points, kerr, x, n_q)
    gamma = jnp.conj(omega[:, None, None, None] * _EPS0 / 16 * t)
    return KerrCoupling(
        port=waves.port,
        waves=tuple(
            (m, float(f)) for m, f in zip(waves.modes, waves.frequency, strict=True)
        ),
        gamma_nl=gamma,
    )
