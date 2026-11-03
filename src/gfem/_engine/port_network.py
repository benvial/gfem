"""The port network of a driven solve: S forward and S adjoint in one place.

Every port enters S twice, both affine in ω: through its incident wave's
right-hand side (one column per excited port) and through its S
functional (one row of S per port: ``1 / sqrt(2 R)`` times the port
voltage for a lumped port, the projection on the conjugated mode for a
wave port, `gfem._engine.wave_ports`). A `PortNetwork` holds them with the
reference impedances ``z_ref`` and the wave ports' de-embedding, and
answers what S is and what an S cotangent asks of the solver: S from the
outgoing waves (`PortNetwork.scattering`), and the split of a cotangent on
S into adjoint columns given by reciprocity, adjoint loads and undefined
columns (`split_cotangent`).

Reciprocity: with ``A`` complex symmetric and port i's S functional
``f_i`` parallel to its excitation ``b_i`` (every lumped port; a wave port
whose mode is real up to one phase), ``A^-1 f_i`` is port i's own solution
scaled, so its row needs no adjoint solve. With a Bloch wave vector
``A(k)^T = A(-k)``: port i's adjoint is its own solution at -k, which a
sweep at k does not have, so the network takes no shortcut there
(`PortNetwork.reciprocal`) and every row with a cotangent is an adjoint
load. A port without a resistance
has no reference impedance: its S row and ``z_ref`` are NaN, and a
cotangent on that row makes the gradient NaN like the value (ADR 0011).

The network only consumes the wave port couplings; the system operator
(`gfem._engine.system_operator`) solves with the same couplings.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from gfem import _mpi
from gfem._boundaries import LumpedPort, WavePort
from gfem._engine import terms
from gfem._engine.wave_ports import PortSurfaceCoupling, WavePortCoupling
from gfem._linalg.layout import ONE_PROCESS, DofLayout
from gfem._sweep.adaptive import AffineTerm
from gfem.model import Model

if TYPE_CHECKING:
    from gfem._engine.driven import DrivenOperator

__all__ = [
    "PARALLEL_RTOL",
    "CotangentSplit",
    "PortNetwork",
    "parallel_scale",
    "resistance",
    "split_cotangent",
]

#: A port's S functional counts as parallel to its excitation (so its
#: adjoint is the port's own solution scaled) to this relative residual.
PARALLEL_RTOL = 1e-10


def resistance(port: LumpedPort | WavePort) -> float:
    """A lumped port's resistance in ohm, NaN without one (an undefined S row)."""
    if not isinstance(port, LumpedPort):  # pragma: no cover - refused earlier
        return math.nan
    r = port.rlc[0]
    return math.nan if r is None else r


def parallel_scale(
    functional: np.ndarray, excitation: np.ndarray, layout: DofLayout = ONE_PROCESS
) -> complex | None:
    """``c`` with ``functional = c excitation`` to `PARALLEL_RTOL`, else ``None``.

    True for every lumped port and for a wave port whose mode is real up to
    one phase (lossless); a lossy wave port's S functional projects on the
    conjugated mode, which is not parallel to its excitation. Both are load
    columns, so across ranks each product is the layout's agreed one.
    """
    norm = np.real(layout.inner(excitation, excitation))
    if norm == 0:  # pragma: no cover - an excited port always has a load
        return None
    c = layout.inner(excitation, functional) / norm
    residual = layout.norm(functional - c * excitation)
    if residual > PARALLEL_RTOL * layout.norm(functional):
        return None
    return complex(c)


def _reciprocal_scale(
    functional: np.ndarray,
    i: int,
    columns: Sequence[int],
    rhs: np.ndarray,
    layout: DofLayout = ONE_PROCESS,
) -> complex | None:
    """``c`` if port ``i`` is excited, its S functional ``c`` times its excitation.

    Its adjoint is then ``c`` times its own solution (reciprocity). ``None``
    for a port that is not excited or not parallel (`parallel_scale`).

    Args:
        functional: ``(N,)``, port i's S functional.
        i: The port's index.
        columns: The excited ports' indices.
        rhs: ``(N, width)``, the excitations, one column per excited port.
    """
    if i not in columns:
        return None
    return parallel_scale(functional, rhs[:, columns.index(i)], layout)


@dataclasses.dataclass(frozen=True, eq=False)
class CotangentSplit:
    """An S cotangent at one frequency, split for the adjoint.

    Columns are the excited ports.

    Attributes:
        free: ``(N, width)``: the adjoint columns given by reciprocity, no
            solve (zeros without it).
        loads: ``(N, width)``: the right-hand sides of the adjoint solve,
            always finite.
        undefined: ``(width,)`` bool: the columns whose cotangent meets an S
            row of a port without a reference impedance (NaN gradient).
    """

    free: np.ndarray
    loads: np.ndarray
    undefined: np.ndarray


def split_cotangent(
    ct: np.ndarray,
    d: np.ndarray,
    columns: Sequence[int],
    functional: np.ndarray,
    *,
    own: tuple[np.ndarray, np.ndarray] | None = None,
    layout: DofLayout = ONE_PROCESS,
) -> CotangentSplit:
    """Split the cotangent ``ct`` on S at one frequency (see `CotangentSplit`).

    Row i adds ``functional[:, i] ⊗ ct_i`` to the loads, or, with
    reciprocity, ``c x_m ⊗ ct_i`` to the free columns when port i is the
    excited column m and its S functional is ``c`` times its excitation
    (`parallel_scale`). A row without a cotangent is skipped (even where its
    functional is undefined); an undefined row with one marks its nonzero
    columns undefined and adds nothing.

    Args:
        ct: ``(n_ports, width)``, the cotangent on S (de-embedded).
        d: ``(n_ports,)``, the de-embedding at this frequency: ``ct`` is
            scaled by ``d_i d_{columns[j]}``.
        columns: The excited ports' indices.
        functional: ``(N, n_ports)``, the S functionals (full or reduced).
        own: ``(rhs, x)``, the excitations and solutions ``(N, width)``
            each; turns reciprocity on.
    """
    columns = list(columns)
    ct = np.asarray(ct) * d[:, None] * d[columns][None, :]
    shape = (functional.shape[0], ct.shape[1])
    free = np.zeros(shape, complex)
    loads = np.zeros(shape, complex)
    undefined = np.zeros(ct.shape[1], bool)
    for i in map(int, np.flatnonzero((ct != 0).any(axis=1))):
        f_i = functional[:, i]
        # S row of a port without R; a rank without port DOFs sees no NaN
        if not _mpi.agree_all(layout.comm, flag=bool(np.isfinite(f_i).all())):
            undefined |= ct[i] != 0
            continue
        if own is not None:
            rhs, x = own
            scale = _reciprocal_scale(f_i, i, columns, rhs, layout)
            if scale is not None:  # reciprocity: no solve
                free += np.outer(scale * x[:, columns.index(i)], ct[i])
                continue
        loads += np.outer(f_i, ct[i])
    return CotangentSplit(free, loads, undefined)


@dataclasses.dataclass(eq=False)
class PortNetwork:
    """Every port's incident wave and S functional in one driven solve.

    Attributes:
        excitation: The right-hand sides, one column per excited port.
        outputs: The S functionals, one row per port.
        waves: The wave ports' couplings, in model port order.
        columns: The excited ports' indices.
        z_ref: ``(n_freq, n_ports)`` reference impedances, ohm.
        reciprocal: Whether the operator is complex symmetric (no Bloch
            wave vector), so that `pullback` may take S rows from the
            solutions by reciprocity.
        layout: The DOF layout of the excitation and output vectors, load columns
            under ranks; S is their pairing with state columns.
        deembedding: ``(n_freq, n_ports)``: S_ij is scaled by ``d_i d_j``
            (1 for lumped; a wave port's ``exp(gamma offset)`` moves its
            reference plane).
    """

    excitation: list[AffineTerm]
    outputs: list[AffineTerm]
    waves: list[WavePortCoupling]
    columns: list[int]
    z_ref: np.ndarray
    reciprocal: bool = True
    layout: DofLayout = ONE_PROCESS
    deembedding: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        n_ports = self.z_ref.shape[1]
        d = np.ones((len(self.z_ref), n_ports), complex)
        for c in self.waves:
            d[:, c.index] = [c.deembedding(w) for w in c.omegas]
        self.deembedding = d

    @classmethod
    def build(
        cls,
        model: Model,
        operator: DrivenOperator,
        surfaces: Sequence[PortSurfaceCoupling],
        *,
        columns: Sequence[int],
        n_freq: int,
    ) -> PortNetwork:
        """The network of ``model``'s ports on ``operator``'s space.

        Args:
            model: The model.
            operator: The driven operator (lumped ports and space).
            surfaces: The wave port surfaces' couplings: one excitation
                term (if any of its ports is excited) and one output term
                each.
            columns: The excited ports' indices.
            n_freq: The output grid's size.
        """
        names = model.port_names
        columns = list(columns)
        n_ports, width = len(names), len(columns)
        excitation: list[AffineTerm] = []
        outputs: list[AffineTerm] = []
        lumped = [names.index(data.port.name) for data in operator.ports]
        if lumped:
            vectors = [
                terms.port_vectors(operator.space, operator.built, operator.order, data)
                for data in operator.ports
            ]
            r = np.array([resistance(model.ports[i]) for i in lumped])
            fed = [(k, columns.index(i)) for k, i in enumerate(lumped) if i in columns]
            if fed:
                b = np.stack([vectors[k][0] for k, _ in fed], axis=1)
                select = np.zeros((len(fed), width))
                for a, (_, pos) in enumerate(fed):
                    select[a, pos] = 1.0

                def lumped_rhs(w: float, select=select) -> np.ndarray:
                    return 1j * operator.wavenumber(w) * select

                excitation.append(AffineTerm(b, lumped_rhs))
            rows = np.zeros((n_ports, len(lumped)))
            rows[lumped, range(len(lumped))] = 1.0 / np.sqrt(2 * r)
            voltage = np.stack([v[1] for v in vectors], axis=1)
            outputs.append(AffineTerm(voltage, lambda _, rows=rows: rows))
        for surface in surfaces:
            m2 = surface.vectors.shape[1]
            fed = [
                (c, columns.index(c.index)) for c in surface.ports if c.index in columns
            ]
            if fed:

                def wave_rhs(w: float, fed=fed, m2=m2) -> np.ndarray:
                    out = np.zeros((m2, width), complex)
                    for c, pos in fed:
                        out[:, pos] = c.excitation(w)
                    return out

                excitation.append(AffineTerm(surface.vectors, wave_rhs))

            def wave_rows(w: float, ports=surface.ports, m2=m2) -> np.ndarray:
                out = np.zeros((n_ports, m2), complex)
                for c in ports:
                    out[c.index] = c.functional(w)
                return out

            outputs.append(AffineTerm(np.conj(surface.vectors), wave_rows))
        waves = sorted((c for s in surfaces for c in s.ports), key=lambda c: c.index)
        z_ref = np.zeros((n_freq, n_ports))
        for i in lumped:
            z_ref[:, i] = resistance(model.ports[i])
        for c in waves:
            z_ref[:, c.index] = c.z_ref
        return cls(
            excitation,
            outputs,
            list(waves),
            columns,
            z_ref,
            reciprocal=operator.symmetric,
            layout=operator.layout or ONE_PROCESS,
        )

    def rhs(self, omega: float) -> np.ndarray:
        """The right-hand sides at ω, ``(N, width)``."""
        return sum(
            t.vectors @ np.asarray(t.coefficients(omega)) for t in self.excitation
        )

    def outgoing(self, omega: float, x: np.ndarray) -> np.ndarray:
        """``(n_ports, width)``: each port's outgoing wave in the solutions ``x``."""
        pair = self.layout.pair
        return self._rows(omega, (pair(t.vectors, x) for t in self.outputs))

    def reduced_functional(
        self, omega: float, projected: Sequence[np.ndarray]
    ) -> np.ndarray:
        """``(n_ports, n)``: the S functionals on a basis V.

        ``projected`` holds ``L_tᵀ V`` per output term (``ReducedModel.outputs``).
        """
        return self._rows(omega, projected)

    def _rows(self, omega: float, projected: Iterable[np.ndarray]) -> np.ndarray:
        """``Σ_t coefficients_t(ω) @ projected_t`` over the output terms."""
        return sum(
            np.asarray(t.coefficients(omega)) @ r
            for r, t in zip(projected, self.outputs, strict=True)
        )

    def functional(self, omega: float) -> np.ndarray:
        """``(N, n_ports)``: port i's outgoing wave is ``functional[:, i] @ x``.

        A wave port's projects on its conjugated mode (`WavePortCoupling`).
        """
        return sum(
            t.vectors @ np.asarray(t.coefficients(omega)).T for t in self.outputs
        )

    def scattering(self, waves: np.ndarray) -> np.ndarray:
        """S from the outgoing waves ``(n_freq, n_ports, width)``.

        Subtracts the incident wave at each excited port, then de-embeds
        each wave port by ``exp(gamma offset)`` on its row and column.
        """
        s = np.array(waves, dtype=complex)
        s[:, self.columns, range(len(self.columns))] -= 1.0
        d = self.deembedding
        return s * d[:, :, None] * d[:, None, self.columns]

    def adjoint_loads(self, omega: float) -> np.ndarray:
        """``(N, q)``: the S functionals whose adjoints the solutions miss.

        A wave port's S functional, unless the port is excited and its
        functional parallel to its excitation (lossless: the adjoint is the
        port's own solution). An adaptive sweep adds their solutions to its
        basis at every sample (`gfem._sweep.adaptive`, ADR 0010), so the
        frozen-basis S gradient of a lossy wave port is exact at the samples.
        """
        functional = self.functional(omega)
        excited = any(c.index in self.columns for c in self.waves)
        rhs = self.rhs(omega) if excited else None
        loads = []
        for c in self.waves:
            f = functional[:, c.index]
            if (
                rhs is not None
                and _reciprocal_scale(f, c.index, self.columns, rhs, self.layout)
                is not None
            ):
                continue
            loads.append(f)
        if not loads:
            return np.zeros((len(functional), 0), complex)
        return np.stack(loads, axis=1)

    def pullback(
        self, n: int, omega: float, ct: np.ndarray, x: np.ndarray
    ) -> CotangentSplit:
        """Split the S cotangent ``ct`` at frequency ``n``, with reciprocity.

        ``x`` ``(N, width)`` holds the excited ports' solutions at ω. Without
        `reciprocal` (a Bloch wave vector) every row is an adjoint load.
        """
        return split_cotangent(
            ct,
            self.deembedding[n],
            self.columns,
            self.functional(omega),
            own=(self.rhs(omega), x) if self.reciprocal else None,
            layout=self.layout,
        )

    @property
    def nbytes(self) -> int:
        """Host memory held by the excitation and output vectors, in bytes.

        The system operator counts ``W``'s (`SystemOperator.nbytes`).
        """
        return sum(int(t.vectors.nbytes) for t in (*self.excitation, *self.outputs))
