r"""The port pass of a driven backward: the gradient through the wave ports' modes.

A wave port enters the driven system through the modes of its 2D section: the
Robin term :math:`\gamma_0 M`, the rank-:math:`2n` modal correction
:math:`W`, the incident load :math:`2 s h/\kappa`, the S functional
:math:`\kappa R^{-1} H^\mathsf{T}/s`, the de-embedding :math:`e^{\gamma\ell}`
and the outputs :math:`\gamma` and :math:`Z_{PV}`. With the 3D adjoint
columns :math:`\lambda_j` and solutions :math:`x_j` held, the gradient of a
design the modes read is the held part (the 3D pairing, unchanged) plus the
derivative of

.. math::

    \Psi = \operatorname{Re}\Big[\sum_{ij} C_{ij} d_i s_j (f_i\cdot u_j - \delta_{ij})
    + \sum_k a_{c(k)}\cdot b_k - \sum_j \big(\gamma_0\,a_j^\mathsf{T} M u_j
    + a_j^\mathsf{T} W u_j\big) + \sum_k \bar\gamma_k\gamma_k + \bar z_k Z_k\Big]

by the modes, where :math:`a_j = T^\mathsf{T}\lambda_j` and :math:`u_j =
T^\mathsf{T}x_j` are the traces on the section's Nédélec DOFs (`FaceTrace`),
:math:`C` the cotangent of S, :math:`d_i` and :math:`s_j` its de-embedding
factors and :math:`\bar\gamma`, :math:`\bar z` the cotangents of the
outputs. `pullback_row` is that derivative written by hand against the
section's coupling forms. It gives, per mode, the cotangent of the field
coordinates, of :math:`\gamma`, of the voltage and of the power phase; the
normalization's pullback (`gfem._engine.port_mode_normalization`) turns them
into one pair :math:`(\bar x, \bar\lambda)` per mode, the deflated adjoint
(`gfem._engine.port_mode_adjoint`) into the vector :math:`q`, and the section
kernel (`gfem._engine.port_mode_sensitivity`) contracts :math:`q` with the
pencil's forms.

S, the port waves, the kept fields, :math:`\gamma`, :math:`Z_{PV}` and
``z_ref`` do not depend on the phase of a mode, and a driven row reads its
mode at its own frequency, so the quadrant gauge and the overlap chain
pull back to zero and the rows are independent: each row is normalized against
its own stored coordinates, which fixes nothing but the reference of the phase.
The power normalization matters only through a voltage path's pseudo-wave.

`PortPass` is the seam in `DrivenRule.backward`: `PortPass.collect` stores, per
active surface and frequency, the traces and the row's cotangents (one
collective sum over ranks), and `PortPass.finish` runs the 2D work once, on
rank 0, over every row.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.constants
import scipy.sparse as sp

from gfem import _mpi
from gfem._engine.line_mode import LineModeSpace
from gfem._engine.line_mode_sensitivity import line_integrated
from gfem._engine.line_port_pass import LineFaceForms, line_face
from gfem._engine.port_mode import FullReduction, PortModeSpace
from gfem._engine.port_mode_adjoint import (
    pencil_cotangent,
    polish,
)
from gfem._engine.port_mode_normalization import normalize_rows, normalize_vjp
from gfem._engine.port_mode_sensitivity import (
    FormPair,
    SectionCotangents,
    SectionKernel,
    law_cotangents,
    section_inputs,
)
from gfem._engine.port_mode_shape import ShapeForm, chain_to_nodes, voltage_eulerian
from gfem._engine.port_rows import Mode, ModeValues, Row, RowInputs, sample_factor
from gfem._engine.sensitivity import Integrated
from gfem._engine.wave_ports import PortSurfaceCoupling
from gfem._sweep.port_mode import (
    _branch_gamma,
    _voltage_row,
    _voltage_rule,
    _wavenumbers,
)
from gfem.errors import PortModeError

if TYPE_CHECKING:
    from gfem._engine.port_network import PortNetwork
    from gfem.model import Model

__all__ = [
    "FaceForms",
    "PortPass",
    "PortResult",
    "pullback_row",
]

MU0 = scipy.constants.mu_0
#: The tag of the Robin term's `FormPair`, which the node report keeps apart,
#: and of everything else in the port pass's node cotangent.
ROBIN = "robin"
REST = "rest"
#: A ``Z_PV`` whose imaginary part is below this fraction of its modulus is
#: cast to real in ``z_ref`` (`WavePortCoupling.z_ref`).
_REAL_Z = 1e-12


# --- the dense pullback ---------------------------------------------------------


@dataclasses.dataclass(eq=False)
class FaceForms:
    r"""The constants of one wave port surface's pullback.

    Attributes:
        coupling: ``(n_t, n_free)``, :math:`K_c = M_T + M_G`: the modes'
            :math:`n\times H` on the face is :math:`K_c x` for the pencil
            vector :math:`x = (\gamma E_t, E_n)`.
        robin: ``(n_t, n_free)`` :math:`M_T`: the Robin mass of the
            tangential field (zero on the normal columns).
        select: ``(n_t, n_free)`` the embedding of the tangential free
            coordinates into the Nédélec DOFs (a 0-1 matrix).
        tangential: ``(n_free,)`` bool, the free DOFs that are Nédélec.
        scale: Metres per file unit.
        lead: The position of the surface's lead port (its :math:`\gamma_0`
            is the Robin term's) among its modes.
        excited: Per mode, its column among the solve's, or ``-1``.
        free_t: The Nédélec DOF of each tangential free DOF.
    """

    coupling: sp.csr_matrix
    robin: sp.csr_matrix
    select: sp.csr_matrix
    tangential: np.ndarray
    scale: float
    lead: int
    excited: Sequence[int]
    free_t: np.ndarray

    @classmethod
    def of(
        cls,
        space: PortModeSpace,
        *,
        lead: int,
        excited: Sequence[int],
    ) -> FaceForms:
        """The forms of ``space``, whose lead port is ``lead``-th of its modes."""
        g_form, t_form = space.coupling_forms
        tangential = space.free < space.n_t
        positions = np.flatnonzero(tangential)
        free_t = space.free[tangential]
        select = sp.csr_matrix(
            (np.ones(len(positions)), (free_t, positions)),
            shape=(space.n_t, len(space.free)),
        )
        return cls(
            coupling=sp.csr_matrix(t_form + g_form),
            robin=sp.csr_matrix(t_form),
            select=select,
            tangential=tangential,
            scale=space.section.scale,
            lead=lead,
            excited=list(excited),
            free_t=free_t,
        )

    def embed(self, trace: np.ndarray) -> np.ndarray:
        """Traces ``(n_t, width)`` as vectors on the free DOFs (tangential part)."""
        out = np.zeros((len(self.tangential), trace.shape[1]), trace.dtype)
        out[self.tangential] = trace[self.free_t]
        return out


@dataclasses.dataclass(eq=False)
class RowBars:
    r"""The cotangents `pullback_row` returns.

    Attributes:
        field: ``(n_free, nk)`` of the normalized fields.
        gamma: ``(nk,)`` of the propagation constants (all that reads
            :math:`\gamma` itself).
        voltage: ``(nk,)`` of the voltages.
        phase: ``(nk,)`` of the power phases.
        pairs: The cotangents of the coupling forms' :math:`\nu` (the
            modes' :math:`n\times H`, the Robin mass).
    """

    field: np.ndarray
    gamma: np.ndarray
    voltage: np.ndarray
    phase: np.ndarray
    pairs: list[FormPair]


def pullback_row(  # noqa: PLR0915
    forms: FaceForms, modes: ModeValues, row: RowInputs, *, explicit: bool = True
) -> RowBars:
    r"""The derivative of :math:`\Psi` by one surface's modes at one frequency.

    The convention is :math:`\mathrm{d}\Psi = \operatorname{Re}\sum
    \bar v\,\mathrm{d}v` (plain, not conjugated, cotangents). With
    :math:`x_k = (\gamma_k E_{t,k}, E_{n,k})`: :math:`h_k = K_c x_k`,
    :math:`e_k` the tangential trace of the field, :math:`s_k = M_T E_{t,k}`,
    :math:`R = H^\mathsf{T}E`, :math:`G = S^\mathsf{T}E`, the dual
    :math:`D = R^{-1}H^\mathsf{T}`,
    :math:`\kappa_k = \sqrt{p R_{kk}}` (``p`` the pairing factor) or, with a
    path, :math:`(V/|V|)\sqrt{\operatorname{Re}\,\mathrm{phase}}`, the S
    functional :math:`f_k = \kappa_k D_k/s`, the incident load :math:`b_k =
    2 s h_k/\kappa_k` and the modal correction :math:`W = P
    \operatorname{blockdiag}(R_\mathrm{sym}^{-1}, -(\gamma_0G)^{-1})P^\mathsf{T}`
    with :math:`P = [H, \gamma_0 S]`.

    Args:
        forms: The surface's constants.
        modes: Its modes at the frequency.
        row: The frequency's traces and cotangents.
        explicit: Also return the :math:`\nu` cotangents of the coupling and
            Robin forms (`RowBars.pairs`).
    """
    tang = forms.tangential
    field, gam = modes.field, modes.gamma
    n_modes = field.shape[1]
    scale = forms.scale
    a, u, weights = row.adjoint, row.solution, row.weights
    lead = forms.lead
    gam0 = gam[lead]
    pairing = -0.5j * scale / (row.omega * MU0)
    path = ~np.isnan(modes.voltage)

    x = np.where(tang[:, None], field * gam[None, :], field)
    h = forms.coupling @ x
    e = forms.select @ field
    s = forms.robin @ field
    r = h.T @ e
    r_inv = np.linalg.inv(r)
    r_sym = 0.5 * (r + r.T)
    r_sym_inv = np.linalg.inv(r_sym)
    g = s.T @ e
    g_s = gam0 * g
    g_s_inv = np.linalg.inv(g_s)
    dual = r_inv @ h.T
    kappa = np.empty(n_modes, complex)
    for k in range(n_modes):
        if path[k]:
            v = modes.voltage[k]
            kappa[k] = v / abs(v) * np.sqrt(modes.phase[k].real)
        else:
            kappa[k] = np.sqrt(pairing * r[k, k] + 0j)

    # the S functional: f_k = kappa_k D_k / s, so that S reads f . u
    f_bar = weights @ u.T  # (nk, n_t)
    kappa_bar = np.einsum("kt,kt->k", dual, f_bar) / scale
    dual_bar = kappa[:, None] * f_bar / scale

    # the incident load b_k = 2 s h_k / kappa_k, dotted with the adjoint column
    h_bar = np.zeros_like(h)
    for k, col in enumerate(forms.excited):
        if col < 0:
            continue
        b_bar = a[:, col]
        h_bar[:, k] += 2 * scale * b_bar / kappa[k]
        kappa_bar[k] += -2 * scale / kappa[k] ** 2 * (b_bar @ h[:, k])

    # D = R^-1 H^T
    r_bar = -r_inv.T @ dual_bar @ dual.T
    h_bar += dual_bar.T @ r_inv

    # the Robin scalar gamma_0 sum_j a_j^T M u_j, and the modal correction
    u_free = forms.embed(u)
    robin = np.sum(a * (forms.robin @ u_free))
    gam0_bar = -robin
    p_mat = np.concatenate([h, gam0 * s], axis=1)
    pa, pu = p_mat.T @ a, p_mat.T @ u
    n1 = pa[:n_modes] @ pu[:n_modes].T
    n2 = pa[n_modes:] @ pu[n_modes:].T
    m1, m2 = -r_sym_inv, g_s_inv
    r_sym_bar = r_sym_inv.T @ n1 @ r_sym_inv.T
    g_s_bar = -g_s_inv.T @ n2 @ g_s_inv.T
    r_bar += 0.5 * (r_sym_bar + r_sym_bar.T)
    pa_bar = np.concatenate([m1 @ pu[:n_modes], m2 @ pu[n_modes:]])
    pu_bar = np.concatenate([m1.T @ pa[:n_modes], m2.T @ pa[n_modes:]])
    p_bar = a @ pa_bar.T + u @ pu_bar.T
    h_bar += p_bar[:, :n_modes]
    s_pre_bar = p_bar[:, n_modes:]  # of gamma_0 S
    s_bar = gam0 * s_pre_bar
    gam0_bar += np.sum(s_pre_bar * s)
    g_bar = gam0 * g_s_bar
    gam0_bar += np.sum(g_s_bar * g)

    # kappa: the pairing (no path), or the pseudo-wave and the impedance (a path)
    voltage_bar = np.zeros(n_modes, complex)
    phase_bar = np.zeros(n_modes, complex)
    for k in range(n_modes):
        if path[k]:
            v, p = modes.voltage[k], modes.phase[k]
            root = math.sqrt(p.real)
            unit = v / abs(v)
            c = kappa_bar[k] * root
            voltage_bar[k] += c / abs(v) - (c * v).real * np.conj(v) / abs(v) ** 3
            phase_bar[k] += (kappa_bar[k] * unit).real / (2 * root)
            z_bar = row.impedance[k]
            voltage_bar[k] += (z_bar * p).real * np.conj(v)
            phase_bar[k] += z_bar * abs(v) ** 2 / 2
        else:
            r_bar[k, k] += kappa_bar[k] * pairing / (2 * kappa[k])

    # R = H^T E and G = S^T E
    h_bar += e @ r_bar.T
    e_bar = h @ r_bar
    s_bar += e @ g_bar.T
    e_bar += s @ g_bar

    # back to the fields and the propagation constants
    x_bar = forms.coupling.T @ h_bar
    field_bar = np.where(tang[:, None], x_bar * gam[None, :], x_bar)
    field_bar += forms.select.T @ e_bar
    field_bar += forms.robin.T @ s_bar
    gamma_bar = np.sum(np.where(tang[:, None], x_bar * field, 0.0), axis=0)
    gamma_bar = gamma_bar + row.gamma
    gamma_bar[lead] += gam0_bar
    pairs: list[FormPair] = []
    if explicit:
        for k in range(n_modes):
            pairs += [
                FormPair("coupling_t", h_bar[:, k], x[:, k], 1.0),
                FormPair("coupling_g", h_bar[:, k], x[:, k], 1.0),
                FormPair("coupling_t", s_bar[:, k], field[:, k], 1.0),
            ]
        pairs += [
            FormPair("coupling_t", a[:, j], u_free[:, j], -gam0, tag=ROBIN)
            for j in range(a.shape[1])
        ]
    return RowBars(field_bar, gamma_bar, voltage_bar, phase_bar, pairs)


# --- the full-space normalization ----------------------------------------------


# --- the pass ---------------------------------------------------------------------


@dataclasses.dataclass(eq=False)
class PortResult:
    """What the port pass adds to the held gradient.

    Attributes:
        integrated: Per owner tetrahedron, the material cotangents of every
            active surface (rank 0 only).
        explicit: The cotangents by input kind that no 3D pass produces
            (permittivity laws, sheet fills, `Impedance` values), rank 0 only.
        nodes: The face nodes' cotangent, ``(n_nodes, 3)`` per metre, rank 0
            only; ``None`` when the node coordinates are not wanted.
        report: The same, by the part of the port pass it comes from:
            ``"robin"`` (the Robin term on the traced fields) and ``"rest"``
            (the modes, the normalization and the coupling forms).
    """

    integrated: list[Integrated]
    explicit: dict[tuple, Any]
    nodes: np.ndarray | None = None
    report: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(eq=False)
class _Surface:
    """One active wave port surface (a line, on a 2D model)."""

    coupling: PortSurfaceCoupling
    #: Per mode of the surface, its position among the model's wave ports.
    wave: list[int]
    #: Per mode, the model's port index.
    index: list[int]


@dataclasses.dataclass(eq=False)
class _Face(_Surface):
    """One active planar wave port surface."""

    space: PortModeSpace
    forms: FaceForms


@dataclasses.dataclass(eq=False)
class _Line(_Surface):
    """One active line port surface."""

    space: LineModeSpace
    forms: LineFaceForms


class PortPass:
    r"""The 2D port pass of one driven backward (see the module docstring).

    Args:
        rule: The `gfem._engine.driven.DrivenRule`.
        model: The design state.
        system: The kept system operator (its port surfaces).
        network: The kept port network.
        wanted: The wanted input kinds (``None``: every kind).
    """

    def __init__(
        self,
        rule: Any,
        model: Model,
        system: Any,
        network: PortNetwork,
        wanted: frozenset | None,
    ) -> None:
        self.rule, self.model, self.network = rule, model, network
        self.comm = rule.comm
        self.wanted = wanted
        #: Whether the node coordinates are wanted: every face's nodes are
        #: inputs, unless the rule holds the faces (a Bloch wave vector).
        self.wants_nodes = (
            wanted is None or ("nodes",) in wanted
        ) and rule.wave_faces_held is None
        self.width = len(rule.excite)
        self.columns = list(network.columns)
        self.waves = {c.index: pos for pos, c in enumerate(network.waves)}
        self.faces: list[_Face] = []
        self.lines: list[_Line] = []
        for surface in system.surfaces:
            if not isinstance(surface, PortSurfaceCoupling):
                continue
            space = surface.ports[0].sweep.space
            names = [c.port.name for c in surface.ports]
            planar = isinstance(space, PortModeSpace)
            if not (
                rule.port_design.reads(wanted, names) or (self.wants_nodes and planar)
            ):
                continue
            lead = [c.port for c in surface.ports].index(surface.lead.port)
            index = [c.index for c in surface.ports]
            excited = [
                self.columns.index(i) if i in self.columns else -1 for i in index
            ]
            if isinstance(space, LineModeSpace):
                self.lines.append(
                    _Line(
                        surface,
                        [self.waves[i] for i in index],
                        index,
                        space,
                        LineFaceForms.of(space, lead=lead, excited=excited),
                    )
                )
                continue
            if not isinstance(space, PortModeSpace):  # pragma: no cover
                continue
            self.faces.append(
                _Face(
                    surface,
                    [self.waves[i] for i in index],
                    index,
                    space,
                    FaceForms.of(space, lead=lead, excited=excited),
                )
            )
        self.rows: dict[int, Row] = {}
        self.wants_mu = wanted is None or any(
            k[0] == "material" and k[2] == "mu_r" for k in wanted
        )

    @property
    def active(self) -> bool:
        """Whether any surface needs the pass."""
        return bool(self.faces or self.lines)

    # --- collecting ---

    def collect(
        self,
        n: int,
        adjoint: np.ndarray,
        x: np.ndarray,
        cotangents: Mapping[str, np.ndarray],
    ) -> None:
        r"""Store frequency ``n``'s traces and cotangents.

        Collective: every rank calls it for every frequency. ``adjoint`` and
        ``x`` are the full adjoint and solution columns (excited ports, then
        sources) with the undefined columns NaN.
        """
        omega = 2 * math.pi * self.rule.problem.frequency[n]
        safe = np.where(np.isfinite(adjoint), adjoint, 0.0)

        def traces(trace: Any, n_t: int) -> tuple[np.ndarray, np.ndarray]:
            return trace.gather(safe, n_t), trace.gather(x, n_t)

        self._store(
            n,
            omega,
            adjoint.shape[1],
            cotangents,
            outgoing=lambda: self._outgoing(omega, x),
            traces=traces,
            live=bool(adjoint.any()),
        )

    def collect_reduced(
        self,
        n: int,
        basis: np.ndarray,
        adjoint: np.ndarray | None,
        *,
        coordinates: np.ndarray,
        functional: np.ndarray,
        cotangents: Mapping[str, np.ndarray],
    ) -> None:
        r"""Store frequency ``n``'s traces from a frozen reduced basis.

        The adaptive backward's `collect`: the adjoint and the solution live
        on the basis :math:`V`, as :math:`\lambda_r` and :math:`y`, so their
        traces are :math:`(T^\mathsf{T}V)\lambda_r` and :math:`(T^\mathsf{T}V)y`,
        read from the basis rows on the face, and the outgoing waves are the
        reduced S functionals on :math:`y`; no full-size vector is formed.

        Args:
            n: The frequency's index.
            basis: ``(N, n)`` the reduced basis.
            adjoint: ``(n, width)`` the reduced adjoint (NaN on an undefined
                column), or ``None`` for zero.
            coordinates: ``(n, width)`` the reduced solutions.
            functional: ``(n_ports, n)`` the reduced S functionals.
            cotangents: The outputs' cotangents.
        """
        omega = 2 * math.pi * self.rule.problem.frequency[n]
        if adjoint is None:  # no reduced solve: the adjoint is zero
            adjoint = np.zeros(coordinates.shape, complex)
        safe = np.where(np.isfinite(adjoint), adjoint, 0.0)

        def outgoing() -> np.ndarray:
            f = functional @ coordinates
            for pos, i in enumerate(self.columns):
                f[i, pos] -= 1.0
            return f

        def traces(trace: Any, n_t: int) -> tuple[np.ndarray, np.ndarray]:
            rows = basis[trace.idx3] / trace.writers[trace.idx3, None]
            a = np.zeros((n_t, safe.shape[1]), complex)
            u = np.zeros((n_t, coordinates.shape[1]), complex)
            a[trace.idx2], u[trace.idx2] = rows @ safe, rows @ coordinates
            return a, u

        self._store(
            n,
            omega,
            adjoint.shape[1],
            cotangents,
            outgoing=outgoing,
            traces=traces,
            live=bool(adjoint.any()),
        )

    def _store(
        self,
        n: int,
        omega: float,
        n_columns: int,
        cotangents: Mapping[str, np.ndarray],
        *,
        outgoing: Any,
        traces: Any,
        live: bool,
    ) -> None:
        """Keep frequency ``n``'s row: ``traces(trace, n_t)`` of each surface."""
        network, rule = self.network, self.rule
        n_ports = len(rule.port_names)
        ct, ct_gamma, ct_z = self._cotangents(n, n_columns, cotangents)
        live = live or bool(ct.any() or ct_gamma.any() or ct_z.any())
        if not _mpi.agree_any(self.comm, flag=live):
            return
        d = network.deembedding[n]
        s = np.ones(n_columns, complex)
        s[: self.width] = d[self.columns]
        # the cotangent of each port's de-embedding factor d_i = exp(gamma_i l_i)
        bar_d = np.zeros(n_ports, complex)
        if ct.any():
            f = outgoing()
            bar_d += np.sum(ct * s[None, :] * f, axis=1)
            for pos, i in enumerate(self.columns):
                bar_d[i] += np.sum(ct[:, pos] * d * f[:, pos])
        row_gamma = np.zeros(n_ports, complex)
        for c in network.waves:
            outside = ct_gamma[self.waves[c.index]]
            outside += bar_d[c.index] * d[c.index] * c.port.offset
            row_gamma[c.index] = outside / c.scale
        stored = [self._trace(face, traces, face.space.n_t) for face in self.faces]
        line_traces = [
            self._trace(line, traces, line.space.space.ndof) for line in self.lines
        ]
        weights = ct * d[:, None] * s[None, :]
        self.rows[n] = Row(omega, stored, weights, row_gamma, ct_z, line_traces)

    def _trace(
        self, surface: _Surface, traces: Any, n_dof: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The adjoint and solution traces of ``surface``, on every rank."""
        trace = surface.coupling.trace
        if trace is None:  # pragma: no cover - built with its maps
            raise ValueError("The port surface keeps no face trace maps.")
        a, u = traces(trace, n_dof)
        return _mpi.reduce_bcast(self.comm, a), _mpi.reduce_bcast(self.comm, u)

    def _cotangents(
        self, n: int, n_columns: int, cotangents: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Frequency ``n``'s cotangents on S, the modes and the impedances.

        Returns ``ct`` on S and the port waves ``(n_ports, n_columns)``,
        ``ct_gamma`` on the modes' ``gamma`` ``(n_wave,)`` and ``ct_z`` on
        ``Z_PV``, also from ``z_ref``, ``(n_ports,)``.
        """
        network = self.network
        ct = np.zeros((len(self.rule.port_names), n_columns), complex)
        for name, part in (
            ("s", slice(0, self.width)),
            ("port_waves", slice(self.width, None)),
        ):
            if cotangents.get(name) is not None:
                ct[:, part] = np.asarray(cotangents[name])[n]
        ct_gamma = np.zeros(len(self.waves), complex)
        if cotangents.get("port_gamma") is not None:
            ct_gamma += np.asarray(cotangents["port_gamma"])[n]
        ct_z = np.zeros(len(self.rule.port_names), complex)
        ct_zref = cotangents.get("z_ref")
        ct_zpv = cotangents.get("port_z_pv")
        for c in network.waves:
            # a NaN impedance (no path) is a constant: its cotangent is a zero
            if c.port.voltage_path is None:
                continue
            if ct_zpv is not None:
                ct_z[c.index] += np.asarray(ct_zpv)[n][self.waves[c.index]]
            if ct_zref is not None:
                ct_z[c.index] += self._z_ref_cotangent(
                    c, n, np.asarray(ct_zref)[n][c.index]
                )
        return ct, ct_gamma, ct_z

    def _outgoing(self, omega: float, x: np.ndarray) -> np.ndarray:
        """The waves of S before de-embedding, ``(n_ports, n_columns)``.

        Each port's S functional on the solutions, less the incident wave of
        an excited column.
        """
        network, width = self.network, self.width
        f = np.zeros((len(self.rule.port_names), x.shape[1]), complex)
        f[:, :width] = network.outgoing(omega, x[:, :width])
        for pos, i in enumerate(self.columns):
            f[i, pos] -= 1.0
        if x.shape[1] > width:
            f[:, width:] = network.outgoing(omega, x[:, width:])
        return f

    def _z_ref_cotangent(self, c: Any, n: int, ct: complex) -> complex:
        """The cotangent of ``Z_PV`` from one ``z_ref`` entry's.

        ``z_ref`` is ``Z_PV``, or its real part where it is real (a real
        reference array, or an imaginary part within rounding of zero).
        """
        z = c.sweep.z_pv[n]
        real = not np.iscomplexobj(self.network.z_ref) or (
            abs(z.imag) <= _REAL_Z * abs(z)
        )
        return complex(ct.real) if real else complex(ct)

    # --- finishing ---

    def finish(self) -> PortResult:
        r"""Run the 2D work over every stored row, on rank 0.

        Collective: it runs inside `gfem._mpi.on_root`, so a `PortModeError`
        of the adjoint is raised on every rank, and only rank 0 returns terms.
        """
        mine = _mpi.rank(self.comm) == 0
        arrays = _mpi.on_root(
            self.comm, self._compute, results=(PortModeError,), broadcast=False
        )
        if not mine:
            return PortResult([], {})
        return _unpack(arrays)

    def _compute(self) -> dict[str, np.ndarray]:
        """All the 2D work: the result's arrays by name (`_pack`)."""
        integrated_all: list[Integrated] = []
        explicit: dict[tuple, Any] = {}
        n_tets = len(self.model.mesh.volume_elements)
        mesh = self.model.mesh
        report = {tag: np.zeros((len(mesh.imported_nodes), 3)) for tag in (ROBIN, REST)}
        for position, face in enumerate(self.faces):
            total, laws, points = self._face(position, face)
            section = face.space.section
            for tag, cot in points.items():
                report[tag] += chain_to_nodes(
                    section, mesh.triangles, cot, len(mesh.imported_nodes)
                )
            item, kinds = section_inputs(self.model, section, total, n_tets)
            integrated_all.append(item)
            for k, v in [*laws.items(), *kinds.items()]:
                explicit[k] = explicit.get(k, 0.0) + v
        for position, line in enumerate(self.lines):
            cot = line_face(
                line.coupling,
                line.space,
                line.forms,
                index=line.index,
                rows=self.rows,
                position=position,
                who=lambda k, line=line: _who(line, k),
            )
            integrated_all.append(line_integrated(line.space.section, cot, n_tets))
        return _pack(integrated_all, explicit, report if self.wants_nodes else None)

    def _face(
        self, position: int, face: _Face
    ) -> tuple[SectionCotangents, dict[tuple, float], dict[str, np.ndarray]]:
        """The section cotangents of one surface, summed over rows and modes.

        Also the cotangent of the section's points (``n_points, 2``), by part
        (`ROBIN`, `REST`), when the node coordinates are wanted.
        """
        space, section = face.space, face.space.section
        total = SectionCotangents.zeros(section)
        laws: dict[tuple, float] = {}
        shapes = (
            {ROBIN: ShapeForm(space), REST: ShapeForm(space)}
            if self.wants_nodes
            else {}
        )
        eulerian = np.zeros((len(section.points), 2))
        fields = FullReduction(space)
        context = _Context(
            face,
            fields,
            space.mass,
            [self._volts(space, c) for c in face.coupling.ports],
            [self._rule(space, c) for c in face.coupling.ports],
        )
        sweeps = [c.sweep for c in face.coupling.ports]
        solved = np.flatnonzero(sweeps[0].solved)
        freq = sweeps[0].frequency
        kernel = SectionKernel(space, 1.0)
        factors: dict[int, Any] = {}
        for n in sorted(self.rows):
            row = self.rows[n]
            sample = int(solved[np.argmin(np.abs(freq[solved] - freq[n]))])
            if sample not in factors:
                factors.clear()
                factors[sample] = sample_factor(space, freq[sample])
            k0 = _wavenumbers(section, freq[n])[0]
            pencil = space.pencil.at(k0)
            modes, state = self._modes(context, n, row, pencil, factors[sample])
            a_tr, u_tr = row.traces[position]
            inputs = RowInputs(
                omega=row.omega,
                adjoint=a_tr,
                solution=u_tr,
                weights=row.weights[face.index],
                gamma=row.gamma[face.index],
                impedance=row.impedance[face.index],
            )
            bars = pullback_row(
                face.forms, modes, inputs, explicit=self.wants_mu or self.wants_nodes
            )
            pairs = list(bars.pairs)
            material_pairs = list(pairs) if self.wants_mu else []
            triples = []
            for k, st in enumerate(state):
                nv = self._normalization(context, k, st, bars, row.omega)
                pairs += nv.pairs
                material_pairs = material_pairs + nv.pairs
                lam_bar = (bars.gamma[k] + nv.gamma[0]) / (2 * st.gamma)
                q = pencil_cotangent(
                    *pencil,
                    st.lam,
                    st.x,
                    lam_bar=lam_bar,
                    x_bar=nv.y[0],
                    factor=factors[sample],
                    who=_who(face, k),
                )
                triples.append((q, st.x, st.lam))
                if shapes:
                    shapes[REST].add_pencil(q, st.x, st.lam, space.pencil.theta(k0))
                    rule = context.rules[k]
                    if rule is not None and bars.voltage[k] != 0:
                        # the path's points stay put while the mesh moves
                        eulerian += voltage_eulerian(
                            space,
                            space.full(modes.field[:, k]),
                            *rule,
                            bars.voltage[k],
                        )
            if shapes:
                for pair in pairs:
                    shapes[ROBIN if pair.tag == ROBIN else REST].add_pair(pair)
            kernel_row = kernel.at(k0)
            cot = kernel_row.cotangents(triples)
            if material_pairs:
                cot.add(kernel_row.explicit(material_pairs))
            total.add(cot)
            if len(section.laws):
                for key, value in law_cotangents(section, cot, row.omega).items():
                    laws[key] = laws.get(key, 0.0) + value
        points = {tag: form.result() for tag, form in shapes.items()}
        if points:
            points[REST] = points[REST] + eulerian
        return total, laws, points

    def _rule(
        self, space: PortModeSpace, c: Any
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The voltage path's plane points and weights of a port, if it has one."""
        path = c.port.voltage_path
        if path is None:
            return None
        return _voltage_rule(space.section, np.asarray(path, float), c.port.name)

    def _volts(self, space: PortModeSpace, c: Any) -> np.ndarray | None:
        """The voltage functional of a mode on the full field coordinates."""
        rule = self._rule(space, c)
        return None if rule is None else _voltage_row(space, rule)

    def _normalization(
        self, context: _Context, k: int, state: Mode, bars: RowBars, omega: float
    ) -> Any:
        """The pullback of mode ``k``'s normalization: its raw pair's cotangents."""
        field_bar = bars.field[:, k]
        return normalize_vjp(
            context.fields,
            context.mass,
            state.norm,
            state.x[None, :],
            np.array([state.gamma]),
            np.array([omega]),
            z_bar=field_bar[None, :],
            voltage_bar=np.array([bars.voltage[k]]),
            phase_bar=np.array([bars.phase[k]]),
            volts=context.volts[k],
            reference=state.reference,
        )

    def _modes(
        self, context: _Context, n: int, row: Row, pencil: Any, factor: Any
    ) -> tuple[ModeValues, list[Mode]]:
        """The surface's modes at frequency ``n``, polished where reduced.

        Each mode's stored pair is a solve of the reduced model unless the
        frequency is an exact sample; a Newton step brings it to the exact
        pair (`polish`), and its normalization is evaluated against the stored
        field, which fixes the phase only.
        """
        face = context.face
        space, forms = face.space, face.forms
        tang = forms.tangential
        scale = space.section.scale
        sweeps = [c.sweep for c in face.coupling.ports]
        n_modes = len(sweeps)
        field = np.zeros((len(space.free), n_modes), complex)
        gamma = np.zeros(n_modes, complex)
        voltage = np.full(n_modes, complex(math.nan, math.nan))
        phase = np.ones(n_modes, complex)
        state = []
        for k, sw in enumerate(sweeps):
            stored = sw.basis[space.free] @ sw.coords[n]
            gam = sw.gamma[n] * scale
            x = np.where(tang, gam * stored, stored)
            lam = gam**2
            if not sw.solved[n]:
                lam, x = polish(
                    pencil[0], pencil[1], lam, x, factor=factor, who=_who(face, k)
                )
                gam = _branch_gamma(lam)
            reference = stored
            norm = normalize_rows(
                context.fields,
                context.mass,
                x[None, :],
                np.array([gam]),
                np.array([row.omega]),
                volts=context.volts[k],
                reference=reference,
            )
            field[:, k] = norm.z[0]
            gamma[k] = gam
            voltage[k] = norm.voltage[0]
            phase[k] = norm.phase[0]
            state.append(Mode(x, lam, gam, norm, reference))
        return ModeValues(field, gamma, voltage, phase), state


@dataclasses.dataclass(eq=False)
class _Context:
    """What the rows of one surface share: its reduction, mass and voltages."""

    face: _Face
    fields: FullReduction
    mass: sp.csr_matrix
    volts: list[np.ndarray | None]
    rules: list[tuple[np.ndarray, np.ndarray] | None]


def _who(face: _Surface, k: int) -> str:
    """The label of surface ``face``'s ``k``-th wave port, for messages."""
    return f"WavePort {face.coupling.ports[k].port.name!r}"


def _pack(
    items: list[Integrated],
    explicit: dict[tuple, Any],
    report: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """The pass's terms as named arrays (what `gfem._mpi.on_root` sends)."""
    out: dict[str, np.ndarray] = {"n": np.array(len(items))}
    for tag, value in (report or {}).items():
        out[f"r/{tag}"] = np.asarray(value)
    for i, item in enumerate(items):
        for name, v in item.values.items():
            out[f"v{i}/{name}"] = np.asarray(v)
        for name, v in item.moments.items():
            out[f"m{i}/{name}"] = np.asarray(v)
    for j, (kind, value) in enumerate(explicit.items()):
        out[f"k{j}/{kind!r}"] = np.asarray(value)
    return out


def _unpack(arrays: Mapping[str, np.ndarray]) -> PortResult:
    """`_pack`'s inverse."""
    import ast

    items = []
    for i in range(int(arrays["n"])):
        values = {
            k.split("/", 1)[1]: v for k, v in arrays.items() if k.startswith(f"v{i}/")
        }
        moments = {
            k.split("/", 1)[1]: v for k, v in arrays.items() if k.startswith(f"m{i}/")
        }
        items.append(Integrated(values=values, moments=moments))
    explicit = {}
    for k, v in arrays.items():
        if k[0] == "k" and "/" in k:
            kind = ast.literal_eval(k.split("/", 1)[1])
            explicit[kind] = v if v.ndim else v.item()
    report = {k[2:]: v for k, v in arrays.items() if k.startswith("r/")}
    nodes = np.sum(list(report.values()), axis=0) if report else None
    return PortResult(items, explicit, nodes, report)
