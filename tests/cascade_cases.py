r"""Blocks for the mode cascade tests.

The synthetic part builds blocks without any FEM: a `gfem.CascadeResult` (the
same kind of block `gfem.cascade` takes from a driven solve) made of an S
matrix, the propagation constants of its section ports and their polarity
arrows. A Fresnel junction between two guides of propagation constants
:math:`\gamma_1`, :math:`\gamma_2` has :math:`S = \begin{pmatrix} r & t \\ t &
-r \end{pmatrix}`, :math:`r = (\gamma_1 - \gamma_2)/(\gamma_1 + \gamma_2)`,
:math:`t^2 = 1 - r^2`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

import gfem
from gfem.results import NetworkParameters, PortModeResult

C0 = 299792458.0
#: Two frequencies, Hz.
FREQUENCY = (1.0e14, 1.1e14)
#: The +z polarity arrow of a line port: a point and a direction.
UP = ((0.0, 0.0), (0.0, 0.0, 1.0))
DOWN = ((0.0, 0.0), (0.0, 0.0, -1.0))


def k0(frequency: Sequence[float] = FREQUENCY) -> np.ndarray:
    """The free-space wave number at each frequency, 1/m."""
    return 2 * np.pi * np.asarray(frequency) / C0


def synthetic_block(
    ports: Sequence[str],
    values,
    *,
    gamma: Mapping[str, object] | None = None,
    polarity: Mapping[str, object] | None = None,
    frequency: Sequence[float] = FREQUENCY,
    z_ref: float = 50.0,
    polarization: str | None = "Ez",
    offsets: tuple[tuple[str, float], ...] = (),
    **mode_fields,
) -> gfem.CascadeResult:
    """A block of ``ports`` with S ``values`` (``(n_freq, n, n)``) and no FEM.

    ``gamma`` gives a port mode its propagation constant (a scalar or one
    value per frequency); ``polarity`` its arrow (default: ``UP`` on every
    port with a ``gamma``). ``mode_fields`` set other `PortModeResult` fields
    of every such port (``pml``, ``Z_PV``, ``e_field``).
    """
    n_f = len(frequency)
    values = jnp.asarray(values)
    ports = tuple(ports)
    modes = {}
    for name, g in (gamma or {}).items():
        rank = int(name.split("@")[1]) if "@" in name else 1
        arrow = (polarity or {}).get(name, UP) if polarity is not None else UP
        fields = {"Z_PV": jnp.full(n_f, jnp.nan), "polarity": arrow, **mode_fields}
        modes[name] = PortModeResult(
            port=name,
            mode=rank,
            frequency=jnp.asarray(frequency),
            gamma=jnp.broadcast_to(jnp.asarray(g, dtype=complex), (n_f,)),
            residual=jnp.zeros(n_f),
            solved=jnp.ones(n_f, dtype=bool),
            **fields,
        )
    s = NetworkParameters(
        values=values,
        z_ref=jnp.full((n_f, len(ports)), z_ref),
        frequency=tuple(frequency),
        ports=ports,
        excited=ports,
    )
    return gfem.CascadeResult(
        s=s, port_modes=modes, polarization=polarization, offsets=offsets
    )


def section_ports(section: str, n: int) -> list[str]:
    """The names ``section@1 .. section@n``."""
    return [f"{section}@{m}" for m in range(1, n + 1)]


def evanescent(n: int, frequency: Sequence[float] = FREQUENCY) -> np.ndarray:
    """:math:`\\gamma` of a mode far below cutoff, :math:`\\alpha = (2 + n) k_0`."""
    return (2.0 + n) * k0(frequency)


def fresnel_pair(
    gamma_outer, gamma_slab, *, section: str = "s", n_pad: int = 1
) -> tuple[gfem.CascadeResult, gfem.CascadeResult]:
    r"""Two Fresnel junctions bounding a slab: ports (a, s@1) and (s@1, b).

    The slab's mode is ``s@1`` (propagation constant ``gamma_slab``); the
    section keeps ``n_pad`` evanescent modes beyond it that couple to
    nothing (their rows of S are zero), so that the last kept mode cannot
    propagate. The first block is the outer guide into the slab,
    :math:`r = (\gamma_1 - \gamma_2)/(\gamma_1+\gamma_2)`; the second is the
    slab into the outer guide.
    """
    g1, g2 = (
        np.broadcast_to(gamma_outer, (len(FREQUENCY),)),
        np.broadcast_to(gamma_slab, (len(FREQUENCY),)),
    )
    r = (g1 - g2) / (g1 + g2)
    t = np.sqrt(1 - r**2 + 0j)
    n = 1 + n_pad
    names = section_ports(section, n)
    gamma = {names[0]: g2} | {names[m]: evanescent(m + 1) for m in range(1, n)}
    first = np.zeros((len(FREQUENCY), 1 + n, 1 + n), complex)
    second = first.copy()
    # block one: ports (a, s@1, ...); block two: (s@1, ..., b)
    first[:, 0, 0], first[:, 0, 1], first[:, 1, 0], first[:, 1, 1] = r, t, t, -r
    second[:, 0, 0], second[:, 0, n], second[:, n, 0], second[:, n, n] = -r, t, t, r
    a = synthetic_block(["a", *names], first, gamma=gamma)
    b = synthetic_block([*names, "b"], second, gamma=gamma)
    return a, b


def airy(gamma_outer, gamma_slab, length) -> tuple[np.ndarray, np.ndarray]:
    r"""Reflection and transmission of a slab of length :math:`\ell` in the outer guide.

    :math:`\rho = r(1 - p)/(1 - r^2 p)`, :math:`\tau = (1 - r^2)\,e^{-\gamma_2
    \ell}/(1 - r^2 p)` with :math:`p = e^{-2\gamma_2 \ell}`.
    """
    g1, g2 = np.asarray(gamma_outer), np.asarray(gamma_slab)
    r = (g1 - g2) / (g1 + g2)
    p = np.exp(-2 * g2 * length)
    return r * (1 - p) / (1 - r**2 * p), (1 - r**2) * np.exp(-g2 * length) / (
        1 - r**2 * p
    )


def random_block(
    rng: np.random.Generator,
    ports: Sequence[str],
    *,
    kind: str = "passive",
    dead: Sequence[str] = (),
    gamma: Mapping[str, object] | None = None,
    **kwargs,
) -> gfem.CascadeResult:
    """A reciprocal block of random S: symmetric, passive or unitary.

    ``passive`` scales a complex symmetric matrix to spectral norm 0.9;
    ``unitary`` is :math:`Q Q^\\mathsf{T}` of a random unitary :math:`Q`, over
    the ports that are not ``dead`` (the rows and columns of a dead port are
    zero, a mode that couples to nothing).
    """
    n_f, n = len(FREQUENCY), len(ports)
    live = [i for i, p in enumerate(ports) if p not in dead]
    out = np.zeros((n_f, n, n), complex)
    for f in range(n_f):
        m = len(live)
        z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        if kind == "passive":
            sym = (z + z.T) / 2
            sub = 0.9 * sym / np.linalg.norm(sym, 2)
        else:
            q, _ = np.linalg.qr(z)
            sub = q @ q.T
        out[f][np.ix_(live, live)] = sub
    return synthetic_block(ports, out, gamma=gamma, **kwargs)
