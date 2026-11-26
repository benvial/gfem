r"""The Raman gain between the waves of a port.

A material's ``raman_spectrum`` (a `gfem.RamanSpectrum`: the bulk gain in
m/W against the frequency shift) acts on the modes of a wave port that
`gfem.solve` found with `gfem.PortMode`. `raman_gain` reads those results and
the model's materials and gives, between every two waves (rows of the
results), the modal gain of the lower one in the power of the higher one:
the Raman gain of a pump and a Stokes wave, and the loss of the pump, each a
number per watt and metre. A curve over the Stokes frequency costs no extra
solve: it is a column of the matrix.

The integral runs on the port's cross-section at 1 W per mode, as
`gfem.kerr_coupling`'s; the same refusals apply, and the same lossy and
leaky sections are computed while :math:`\alpha \ll \beta`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._materials import Material, PermittivityLaw
from gfem._mode_overlap import _elements, _waves
from gfem._pml import layer_tetrahedra
from gfem._pytree import register_fields
from gfem.results import PortModeResult

if TYPE_CHECKING:
    from gfem.model import Model

__all__ = ["RamanGain", "raman_gain"]

_C = scipy.constants.c
_EPS0 = scipy.constants.epsilon_0
#: the pair of crystal axes each Raman tensor R^x, R^y, R^z couples
_PAIRS = ((1, 2), (0, 2), (0, 1))


@dataclasses.dataclass(frozen=True, eq=False)
class RamanGain:
    r"""The modal Raman gain between the waves of one port.

    A wave is one row of a `gfem.PortModeResult`: a mode at a frequency. With
    :math:`E` the modes at 1 W, :math:`n_a^2 = \mathrm{Re}\,\varepsilon_r` at
    wave :math:`a`'s frequency and :math:`\Omega = \omega_b - \omega_a`,

    .. math::

        g_{ab} = \frac{(c\varepsilon_0)^2}{4} \int_S n_a n_b\,
        G(\Omega; \omega_a)\, W(E_a, E_b)\,\mathrm{d}A,
        \qquad
        \frac{\mathrm{d}P_a}{\mathrm{d}z} \ni g_{ab} P_b P_a,

    with :math:`G(\Omega; \omega_a) = \mathrm{sign}(\Omega)\,g(|\Omega|)\,
    \omega_a/\omega_\mathrm{ref}` the material's `gfem.RamanSpectrum`
    (`RamanSpectrum.gain_at`) and the polarization weight :math:`W` of its
    symmetry class, summed in each element of the section:

    - ``"cubic"``: :math:`W = \sum_{v}|E_a^\dagger R^v E_b|^2` with
      :math:`R^x = \hat y\hat z + \hat z\hat y` and its cyclic permutations,
      the fields taken in the crystal frame of the region's
      `Material.axes` (the identity without);
    - ``"isotropic"``: :math:`W\,G = g_\parallel |E_a^\dagger E_b|^2 +
      g_\perp\left(|E_a|^2 |E_b|^2 + |E_a\cdot E_b|^2 - 2 |E_a^\dagger
      E_b|^2\right)`.

    So :math:`g_{ab} > 0` for wave :math:`a` below wave :math:`b`, and the
    matrix obeys the Manley-Rowe relation :math:`g_{ba} = -(\omega_b/\omega_a)
    g_{ab}` exactly; equal frequencies give 0. The result does not depend on
    the phase of any mode.

    Attributes:
        port: The wave port's name.
        waves: ``(mode, frequency)`` of each wave, in order (Hz).
        gain: :math:`g_{ab}`, ``(K, K)`` real, 1/(W m); 1/W on a line port of
            a 2D mesh (the field is then 1 W/m).
    """

    port: str
    waves: tuple[tuple[int, float], ...]
    gain: jax.Array

    def __repr__(self) -> str:
        head = f"RamanGain(port {self.port!r}, {len(self.waves)} waves"
        try:
            g = np.asarray(self.gain)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return f"{head}, |gain| up to {np.abs(g).max():.4g} 1/(W m))"


register_fields(RamanGain, ("gain",))


@dataclasses.dataclass(frozen=True, eq=False)
class _Raman:
    """The Raman material values of the section's elements, in the columns' order.

    ``eps`` ``(n_wave, n_el)`` the real part of the permittivity at each
    wave's frequency (1 where an element has no spectrum), ``index``
    ``(n_el,)`` one more than the spectrum of the element (0 for none),
    ``parallel`` and ``perpendicular`` ``(n_spectra + 1, n_wave, n_wave)``
    the gains :math:`G` of each spectrum between the waves (row 0 is zero),
    ``cubic`` ``(n_spectra + 1,)`` its class, ``axes`` ``(n_el, 3, 3)`` the
    crystal frame of each element, ``keep`` ``(n_el,)`` False on a perfectly
    matched layer's triangles.
    """

    eps: jax.Array
    index: np.ndarray
    parallel: jax.Array
    perpendicular: jax.Array
    cubic: np.ndarray
    axes: np.ndarray
    keep: np.ndarray


def _permittivity(material: Material, frequency: np.ndarray) -> jax.Array:
    r""":math:`\mathrm{Re}\,\varepsilon_r` of a region at each frequency."""
    eps = material.eps_r
    if isinstance(eps, PermittivityLaw):
        return jnp.real(eps(jnp.asarray(frequency)))
    return jnp.real(jnp.asarray(eps)) * jnp.ones(len(frequency))


def _raman_materials(
    model: Model, section: Any, frequency: np.ndarray, port: str, owner: str
) -> _Raman:
    """Read the Raman spectra of the elements of ``section`` by owner element.

    Raises:
        ValueError: No element outside a perfectly matched layer carries a
            spectrum.
        NotImplementedError: A region carrying one is anisotropic.
    """
    mesh = model.mesh
    cells, tags = _elements(model, section)
    names = mesh._region_names()  # noqa: SLF001
    n_el = len(cells)
    n_wave = len(frequency)
    eps = jnp.ones((n_wave, n_el))
    index = np.zeros(n_el, int)
    axes = np.broadcast_to(np.eye(3), (n_el, 3, 3)).copy()
    parallel, perpendicular, cubic = [jnp.zeros((n_wave, n_wave))], [], [False]
    perpendicular.append(jnp.zeros((n_wave, n_wave)))
    shift = frequency[None, :] - frequency[:, None]
    for tag in np.unique(tags):
        region = names[int(tag)]
        material = model.materials[region]
        # a MaterialField carries no spectrum
        spectrum = material.raman_spectrum if isinstance(material, Material) else None
        if spectrum is None or not isinstance(material, Material):
            continue
        if not material.isotropic:
            raise NotImplementedError(
                f"{owner}: region {region!r} of wave port {port!r}'s section is "
                "anisotropic and carries a raman_spectrum: the refractive index "
                "squared is not defined on a permittivity tensor in this "
                "version. Give the region an isotropic eps_r."
            )
        rows = np.flatnonzero(tags == tag)
        parallel.append(spectrum.gain_at(shift, frequency[:, None]))
        perpendicular.append(spectrum.perpendicular_gain_at(shift, frequency[:, None]))
        cubic.append(spectrum.symmetry == "cubic")
        index[rows] = len(cubic) - 1
        if material.axes is not None:
            axes[rows] = np.asarray(material.axes, dtype=float)
        eps = eps.at[:, rows].set(_permittivity(material, frequency)[:, None])
    layer = layer_tetrahedra(mesh, model.pml) if len(section.normal) == 3 else None
    keep = np.ones(n_el, bool) if layer is None else ~layer[cells]
    if not ((index > 0) & keep).any():
        regions = ", ".join(repr(names[int(t)]) for t in np.unique(tags))
        raise ValueError(
            f"{owner}: no element of wave port {port!r}'s section outside a "
            "perfectly matched layer (whose triangles are skipped) has a "
            f"raman_spectrum (its regions are {regions}), so the gain is zero. "
            "Set one on the core's material."
        )
    return _Raman(
        eps,
        index,
        jnp.stack(parallel),
        jnp.stack(perpendicular),
        np.asarray(cubic),
        axes,
        keep,
    )


def _cubic_weight(a: jax.Array, b: jax.Array) -> jax.Array:
    r""":math:`\sum_v |a^\dagger R^v b|^2` of crystal-frame fields ``(..., 3)``."""
    total = jnp.zeros(())
    for j, k in _PAIRS:
        s = jnp.conj(a[..., j]) * b[..., k] + jnp.conj(a[..., k]) * b[..., j]
        total = total + jnp.abs(s) ** 2
    return total


def _isotropic_weights(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""The two weights of an isotropic material, fields ``(..., 3)``.

    :math:`|a^\dagger b|^2` and :math:`|a|^2|b|^2 + |a\cdot b|^2`.
    """
    overlap = jnp.abs(jnp.sum(jnp.conj(a) * b, axis=-1)) ** 2
    dot = jnp.abs(jnp.sum(a * b, axis=-1)) ** 2
    norm = jnp.sum(jnp.abs(a) ** 2, axis=-1) * jnp.sum(jnp.abs(b) ** 2, axis=-1)
    return overlap, norm + dot


def raman_gain(
    model: Model, modes: PortModeResult | Sequence[PortModeResult]
) -> RamanGain:
    r"""The Raman gain :math:`g_{ab}` between the waves of a port.

    Every row of every result in ``modes`` is one wave, in order: another mode
    number, another frequency, or both. Entry :math:`ab` is the power gain of
    wave :math:`a` per watt in wave :math:`b`: positive when :math:`a` lies
    below :math:`b` in frequency (a Stokes wave of the pump :math:`b`),
    negative when above (the pump's loss, :math:`-\omega_b/\omega_a` times the
    transposed entry), 0 on the diagonal. See `RamanGain` for the formula and
    the polarization weights. A pump and a Stokes result on a frequency grid
    give the gain spectrum of the guide as one column.

    The gain is that of the continuous wave, with the mode of each wave at its
    own frequency; the spectrum's gain is scaled by :math:`\omega_a /
    \omega_\mathrm{ref}`, and the index change between the reference and the
    wave is neglected. ``nonlinear_index`` is not touched: it is what was
    measured, with the Kerr symmetry, and the gain is separate from it.

    All waves must share one port, element order and cross-section (solve
    them with one `gfem.SolverOptions` on one model). The integral runs over
    the cross-section outside any perfectly matched layer, with a quadrature
    rule exact for the mode's polynomial degree. Elements without a
    ``raman_spectrum`` do not contribute; a spectrum of class ``"cubic"``
    reads the region's `Material.axes`, the identity without.

    Exactly linear in the spectrum's gain (so there is no gradient with
    respect to it: the spectrum is a fixed value of the material). Under
    `jax.grad` the traced mode field makes it differentiable with respect to
    the section's ``eps_r`` (through the solve and the index) and to the node
    coordinates of a planar port (as `kerr_nonlinearity`).

    Args:
        model: The model the modes were solved on, with a ``raman_spectrum``.
        modes: One `gfem.PortModeResult` or a sequence of them.

    Returns:
        The ``(K, K)`` matrix for the :math:`K` waves.

    Raises:
        ValueError: The results are of different ports, element orders or
            section geometries; no element outside a perfectly matched layer
            carries a ``raman_spectrum``.
        NotImplementedError: A region carrying a spectrum is anisotropic; an
            Hz line port.
        TypeError: An entry is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`
            on a line port of a 2D model; those of ``model`` are traced and
            those of a mode's solve are not, or the reverse.
        TypeError: The node coordinates are traced by `jax.jit`.
    """
    waves = _waves(model, modes, "gfem.raman_gain", _raman_materials, shape=True)
    raman = waves.materials
    has = raman.index > 0
    n = jnp.sqrt(jnp.where(jnp.asarray(has)[None], raman.eps, 1.0))
    index = jnp.asarray(raman.index)
    parallel = jnp.moveaxis(raman.parallel[index], 0, -1)  # (K, K, n_el)
    perpendicular = jnp.moveaxis(raman.perpendicular[index], 0, -1)
    cubic = jnp.asarray(raman.cubic[raman.index])
    w = waves.weights
    e = waves.field()  # (K, n_el, n_q, 3), or (..., 1) on a line port
    if e.shape[-1] == 1:
        # an Ez line port: the one component is the out-of-plane z
        e = jnp.concatenate([jnp.zeros_like(e), jnp.zeros_like(e), e], axis=-1)

    def integral(value: jax.Array) -> jax.Array:
        return jnp.einsum("abeq,eq->abe", value, w)

    terms = jnp.zeros_like(parallel)
    if (~raman.cubic[raman.index] & has).any():
        parts, both = (integral(x) for x in _isotropic_weights(e[:, None], e[None]))
        terms += parallel * parts + perpendicular * (both - 2 * parts)
    if (raman.cubic[raman.index] & has).any():
        crystal = jnp.einsum("eij,aeqj->aeqi", jnp.asarray(raman.axes), e)
        total = integral(_cubic_weight(crystal[:, None], crystal[None]))
        terms = jnp.where(cubic[None, None], parallel * total, terms)
    pair = n[:, None] * n[None]
    gain = (_C * _EPS0) ** 2 / 4 * jnp.sum(pair * terms, axis=-1)
    return RamanGain(
        port=waves.port,
        waves=tuple(
            (m, float(f)) for m, f in zip(waves.modes, waves.frequency, strict=True)
        ),
        gain=gain,
    )
