r"""The junction coupling of two port modes of one section.

`gfem.junction_coupling` pairs two modes with the reciprocity pairing of the
port mode problem. The pairing is bilinear in the two DOF vectors for a fixed
:math:`\nu_{tt}` and :math:`\gamma`, so the host builds, per mode, two sparse
matrices on the section's free DOFs, with that mode's own :math:`\nu_{tt}` (the
bend and the layer are inside it):

.. math::

    \langle s, b\rangle = c\, x_s^T (G_b + \gamma_b T_b)\, x_b, \quad
    G = \int \mathbf{v}_t \cdot \nu_{tt} \nabla e_n, \quad
    T = \int \mathbf{v}_t \cdot \nu_{tt}\, \mathbf{e}_t,

with :math:`c = -\tfrac{i}{2}\,s/(\omega\mu_0)` and :math:`\gamma` per local
unit. The value and its derivative in the vectors, the propagation constants
and the radii are one host call each, under one ``custom_vjp``: the vectors and
the propagation constants carry the dependence on the materials (through the
solves that made them), the radii the explicit dependence of :math:`\nu_{tt}`
on :math:`R`, by the derivative forms of the bend (the coefficient's
derivative by :math:`u = 1/R`, as the mode's backward does).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp

from gfem import _pml
from gfem._boundaries import lead_port, wave_port_surfaces
from gfem._checks import is_traced
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.runtime import task_manager
from gfem._port_section import PortSection, port_section
from gfem.errors import NonDifferentiableError
from gfem.problems import Bend

if TYPE_CHECKING:
    from gfem.model import Model
    from gfem.results import PortModeResult

__all__ = ["junction_tau2"]

_NODES = (
    "junction_coupling reads the modes on the section while gfem differentiates "
    "the node coordinates: the derivative of a coupling on a moving section is "
    "not available in this version. Differentiate with respect to the radius or "
    "the materials, or wrap the node coordinates in jax.lax.stop_gradient."
)
_MU = (
    "junction_coupling reads the inverse permeability of each mode's section "
    "directly, and gfem does not differentiate it in this version: {path} is "
    "traced. Differentiate with respect to the permittivity, the radius or the "
    "other materials, or wrap it in jax.lax.stop_gradient."
)
#: Leaves of the model that enter :math:`\nu_{tt}`.
_NU_LEAVES = ("mu_r", "axes")


@dataclasses.dataclass(frozen=True, eq=False)
class _Mode:
    """What the host rebuilds a mode's space from (static)."""

    bend: Bend | None
    order: int


@dataclasses.dataclass(frozen=True, eq=False)
class _Spec:
    """The static description of a junction: port, frequencies and the two modes."""

    port: str
    frequency: np.ndarray
    modes: tuple[_Mode, _Mode]
    treedef: Any
    fixed: tuple[Any, ...]
    traced: tuple[int, ...]


def _vector(field: Any) -> jax.Array:
    """The full DOF vector of each frequency, ``(n_freq, n_dof)``."""
    if field._vector is not None:  # noqa: SLF001
        return jnp.asarray(field._vector)  # noqa: SLF001
    _, basis, coords = field._data()  # noqa: SLF001
    return jnp.asarray(np.asarray(coords) @ np.asarray(basis).T)


def _model(spec: _Spec, leaves: tuple[Any, ...]) -> Model:
    values = list(spec.fixed)
    for i, leaf in zip(spec.traced, leaves, strict=True):
        values[i] = np.asarray(leaf)
    return jax.tree.unflatten(spec.treedef, values)


@dataclasses.dataclass(eq=False)
class _Pairing:
    r"""The matrices of one mode's pairing on the free DOFs.

    ``g`` and ``t`` are the integrals of the test tangential field times
    :math:`\nu_{tt}` and the trial gradient of :math:`e_n`, respectively
    the trial tangential field; ``dg`` and ``dt`` their derivatives by the
    radius (metres), ``None`` for a straight mode.
    """

    g: sp.csr_matrix
    t: sp.csr_matrix
    dg: sp.csr_matrix | None
    dt: sp.csr_matrix | None
    scale: float
    free: np.ndarray
    ndof: int

    def at(self, gamma: complex, omega: float) -> sp.csr_matrix:
        r""":math:`P = c\,(G + \gamma T)`, for ``gamma`` in 1/m."""
        c = complex(-0.5j * self.scale / (omega * scipy.constants.mu_0))
        return c * (self.g + complex(gamma) * self.scale * self.t)

    def d_gamma(self, omega: float) -> sp.csr_matrix:
        r""":math:`\partial P/\partial\gamma` with ``gamma`` in 1/m."""
        c = complex(-0.5j * self.scale / (omega * scipy.constants.mu_0))
        return c * self.scale * self.t

    def d_radius(self, gamma: complex, omega: float) -> sp.csr_matrix | None:
        r""":math:`\partial P/\partial R` at fixed ``gamma``, ``None`` straight."""
        if self.dg is None or self.dt is None:
            return None
        c = complex(-0.5j * self.scale / (omega * scipy.constants.mu_0))
        return c * (self.dg + complex(gamma) * self.scale * self.dt)


def _pairing(model: Model, spec: _Spec, mode: _Mode, radius: float) -> _Pairing:
    """The pairing matrices of ``mode`` solved at ``radius`` (if bent)."""
    named = next(p for p in model.ports if p.name == spec.port)
    surface = wave_port_surfaces(model.ports)[str(named.surface)]
    port = lead_port(surface)
    stretch = _pml.stretch(model, spec.frequency)
    bend = (
        None
        if mode.bend is None
        else Bend(radius, outward=mode.bend.outward, at=mode.bend.at)
    )
    section = port_section(model, port, stretch, bend)
    assert isinstance(section, PortSection)  # noqa: S101 - a planar port
    with task_manager(1, None):
        space = PortModeSpace(section, mode.order)
        (et, en), (vt, _) = space.space.TnT()
        nu = space._coef["nu_tt"]  # noqa: SLF001
        dx = space._dx()  # noqa: SLF001
        g = space._matrix([vt * (nu * ngs.grad(en)) * dx])  # noqa: SLF001
        t = space._matrix([vt * (nu * et) * dx])  # noqa: SLF001
        dg = dt = None
        if bend is not None:
            u = space.bend_u
            assert u is not None  # noqa: S101
            assert section.bend is not None  # noqa: S101
            dnu = nu.Diff(u)
            # u = scale / R in m: d u / d R = -u^2 / scale
            chain = -(section.bend.inverse_radius**2) * section.scale
            dg = chain * space._matrix([vt * (dnu * ngs.grad(en)) * dx])  # noqa: SLF001
            dt = chain * space._matrix([vt * (dnu * et) * dx])  # noqa: SLF001
    return _Pairing(g, t, dg, dt, float(section.scale), space.free, space.space.ndof)


def _matrices(spec: _Spec, leaves: tuple[Any, ...], radii: np.ndarray):
    model = _model(spec, leaves)
    a = _pairing(model, spec, spec.modes[0], float(radii[0]))
    b = _pairing(model, spec, spec.modes[1], float(radii[1]))
    if not np.array_equal(a.free, b.free):
        msg = "junction_coupling: the two modes' spaces have different free DOFs."
        raise ValueError(msg)
    return a, b


def _forward(spec: _Spec, leaves, freq, radii, xa, xb, ga, gb) -> np.ndarray:  # noqa: PLR0917
    spec = dataclasses.replace(spec, frequency=np.atleast_1d(np.asarray(freq, float)))
    a, b = _matrices(spec, tuple(leaves), np.asarray(radii))
    omega = 2 * np.pi * spec.frequency
    out = np.empty(len(omega), complex)
    for i, w in enumerate(omega):
        pa, pb = a.at(ga[i], w), b.at(gb[i], w)
        sa, sb = xa[i][a.free], xb[i][a.free]
        cross = 0.5 * (sa @ (pb @ sb) + sb @ (pa @ sa))
        out[i] = cross**2 / ((sa @ (pa @ sa)) * (sb @ (pb @ sb)))
    return out


def _shift(d, left, right, *, tau, cross, norm):
    r"""The derivative of :math:`\tau^2` for a change ``d`` of one mode's matrix.

    :math:`\mathrm{d}\tau^2/\tau^2 = 2\,\mathrm{d}N/N - \mathrm{d}(xx)/xx`, with
    :math:`N = \tfrac12(\ell^T B r + r^T A \ell)` where only the matrix of
    the mode ``right`` changes: :math:`\mathrm{d}N = \tfrac12\,\ell^T\mathrm{d}
    P\,r` and :math:`\mathrm{d}(xx) = r^T \mathrm{d}P\,r` (``norm`` is its
    :math:`xx`, ``cross`` is :math:`N`).
    """
    return tau * ((left @ (d @ right)) / cross - (right @ (d @ right)) / norm)


def _backward(spec: _Spec, leaves, freq, radii, xa, xb, ga, gb, ct):  # noqa: PLR0917
    spec = dataclasses.replace(spec, frequency=np.atleast_1d(np.asarray(freq, float)))
    a, b = _matrices(spec, tuple(leaves), np.asarray(radii))
    omega = 2 * np.pi * spec.frequency
    n = len(omega)
    d_xa, d_xb = np.zeros_like(xa), np.zeros_like(xb)
    d_ga, d_gb = np.zeros(n, complex), np.zeros(n, complex)
    d_r = np.zeros(2)
    for i, w in enumerate(omega):
        pa, pb = a.at(ga[i], w), b.at(gb[i], w)
        sa, sb = xa[i][a.free], xb[i][a.free]
        bxb, axa = pb @ sb, pa @ sa
        cross = 0.5 * (sa @ bxb + sb @ axa)
        ss, bb = sa @ axa, sb @ bxb
        tau = cross**2 / (ss * bb)
        d_xa[i][a.free] = (
            ct[i] * tau * ((bxb + pa.T @ sb) / cross - (axa + pa.T @ sa) / ss)
        )
        d_xb[i][a.free] = (
            ct[i] * tau * ((pb.T @ sa + axa) / cross - (bxb + pb.T @ sb) / bb)
        )

        d_ga[i] = ct[i] * _shift(a.d_gamma(w), sb, sa, tau=tau, cross=cross, norm=ss)
        d_gb[i] = ct[i] * _shift(b.d_gamma(w), sa, sb, tau=tau, cross=cross, norm=bb)
        if (da := a.d_radius(ga[i], w)) is not None:
            d_r[0] += (ct[i] * _shift(da, sb, sa, tau=tau, cross=cross, norm=ss)).real
        if (db := b.d_radius(gb[i], w)) is not None:
            d_r[1] += (ct[i] * _shift(db, sa, sb, tau=tau, cross=cross, norm=bb)).real
    return d_r, d_xa, d_xb, d_ga, d_gb


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _tau2(spec: _Spec, leaves, freq, radii, xa, xb, ga, gb):  # noqa: PLR0917
    shape = jax.ShapeDtypeStruct(np.shape(freq), jnp.complex128)
    return jax.pure_callback(
        lambda *args: _forward(spec, *args),
        shape,
        leaves,
        freq,
        radii,
        xa,
        xb,
        ga,
        gb,
        vmap_method="sequential",
    )


def _tau2_fwd(spec, leaves, freq, radii, xa, xb, ga, gb):  # noqa: PLR0917
    return _tau2(spec, leaves, freq, radii, xa, xb, ga, gb), (
        leaves,
        freq,
        radii,
        xa,
        xb,
        ga,
        gb,
    )


def _tau2_bwd(spec, residual, ct):
    leaves, freq, radii, xa, xb, ga, gb = residual
    shapes = (
        jax.ShapeDtypeStruct(np.shape(radii), jnp.float64),
        jax.ShapeDtypeStruct(np.shape(xa), xa.dtype),
        jax.ShapeDtypeStruct(np.shape(xb), xb.dtype),
        jax.ShapeDtypeStruct(np.shape(ga), ga.dtype),
        jax.ShapeDtypeStruct(np.shape(gb), gb.dtype),
    )
    d_r, d_xa, d_xb, d_ga, d_gb = jax.pure_callback(
        lambda *args: _backward(spec, *args),
        shapes,
        leaves,
        freq,
        radii,
        xa,
        xb,
        ga,
        gb,
        ct,
        vmap_method="sequential",
    )
    return (
        jax.tree.map(jnp.zeros_like, leaves),
        jnp.zeros_like(freq),
        d_r,
        d_xa,
        d_xb,
        d_ga,
        d_gb,
    )


_tau2.defvjp(_tau2_fwd, _tau2_bwd)


def _check_modes(a: PortModeResult, b: PortModeResult) -> tuple[Any, Any]:
    """The two modes' spaces; raise ``ValueError`` naming what differs."""
    from gfem.results import PortModeResult

    for name, r in (("a", a), ("b", b)):
        if not isinstance(r, PortModeResult):
            raise TypeError(
                f"junction_coupling needs gfem.PortModeResult modes, got "
                f"{type(r).__name__} for {name}: solve them with "
                "gfem.solve(model, gfem.PortMode(...))."
            )
        if r.h_field is not None:
            raise NotImplementedError(
                f"junction_coupling: wave port {r.port!r} is an Hz line port; "
                "the coupling is for the modes of a planar port."
            )
        if r.e_field is None:
            raise TypeError(
                f"junction_coupling: the result of wave port {r.port!r} carries "
                "no electric field: solve gfem.solve(model, gfem.PortMode(...)) "
                "and pass that result."
            )
    spaces = []
    for r in (a, b):
        try:
            spaces.append(r.e_field._space)  # noqa: SLF001  # ty: ignore[unresolved-attribute]
        except NonDifferentiableError as err:
            raise NonDifferentiableError(_NODES) from err
    sa, sb = spaces
    if len(sa.section.normal) != 3:
        raise NotImplementedError(
            "junction_coupling: a line port's mode has no junction; the modes "
            "must be of a planar port."
        )
    if a.port != b.port:
        raise ValueError(
            f"junction_coupling: the modes must be of one port; got {a.port!r} "
            f"and {b.port!r}."
        )
    if sa.order != sb.order:
        raise ValueError(
            "junction_coupling: the modes must share one element order; got "
            f"{sa.order} and {sb.order}. Solve them with the same SolverOptions."
        )
    if not all(
        np.array_equal(getattr(sa.section, n), getattr(sb.section, n))
        for n in ("points", "triangles")
    ):
        raise ValueError(
            "junction_coupling: the modes' sections differ (points or triangles); "
            "solve them on the same model."
        )
    fa, fb = (jnp.atleast_1d(jnp.asarray(r.frequency)) for r in (a, b))
    # a frequency staged by `jax.jit` is a constant of the call: compared
    # when concrete, else the counts decide
    if fa.shape != fb.shape or (
        not (is_traced(fa) or is_traced(fb)) and not np.array_equal(fa, fb)
    ):
        raise ValueError(
            "junction_coupling: the modes must share one frequency grid; got "
            f"{fa.shape[0]} and {fb.shape[0]} frequencies "
            f"({'different values' if fa.shape == fb.shape else 'different counts'})."
        )
    return sa, sb


def junction_tau2(model: Model, a: PortModeResult, b: PortModeResult) -> jax.Array:
    r""":math:`\tau^2` of two modes of one section; see `gfem.junction_coupling`."""
    from gfem._mode_overlap import _check_model
    from gfem.model import Model

    if not isinstance(model, Model):
        raise TypeError(
            f"junction_coupling needs a gfem.Model, got {type(model).__name__}."
        )
    sa, sb = _check_modes(a, b)
    if is_traced(model.mesh.nodes):
        raise NonDifferentiableError(_NODES)
    _check_model(model, sa.section, a.port, "junction_coupling")
    paths, treedef = jax.tree.flatten_with_path(model)
    leaves = [leaf for _, leaf in paths]
    traced = tuple(i for i, leaf in enumerate(leaves) if is_traced(leaf))
    for i in traced:
        name = jax.tree_util.keystr(paths[i][0])
        if any(key in name for key in _NU_LEAVES):
            raise NonDifferentiableError(_MU.format(path=name))
    spec = _Spec(
        a.port,
        np.zeros(0),
        (_Mode(_static(a.bend), sa.order), _Mode(_static(b.bend), sb.order)),
        treedef,
        tuple(None if i in traced else leaves[i] for i in range(len(leaves))),
        traced,
    )
    radii = jnp.stack(
        [
            jnp.asarray(1.0 if r.bend is None else r.bend.radius, jnp.float64)
            for r in (a, b)
        ]
    )
    return _tau2(
        spec,
        tuple(leaves[i] for i in traced),
        jnp.atleast_1d(jnp.asarray(a.frequency, jnp.float64)),
        radii,
        _vector(a.e_field),
        _vector(b.e_field),
        jnp.atleast_1d(jnp.asarray(a.gamma)),
        jnp.atleast_1d(jnp.asarray(b.gamma)),
    )


def _static(bend: Bend | None) -> Bend | None:
    """The bend with a placeholder radius: the host sets the value."""
    if bend is None:
        return None
    return Bend(1.0, outward=bend.outward, at=bend.at)
