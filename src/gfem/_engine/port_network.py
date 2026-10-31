"""The port network of a driven solve: S forward and S adjoint in one place.

Every port enters S through two affine terms over ω: its incident wave's
excitation (one right-hand side column per excited port) and its S
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
scaled, so its row needs no adjoint solve. A port without a resistance
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

from gfem._boundaries import LumpedPort, WavePort
from gfem._engine import terms
from gfem._engine.wave_ports import WavePortCoupling
from gfem._sweep.adaptive import AffineTerm
from gfem.model import Model

if TYPE_CHECKING:
    from gfem._engine.driven import DrivenOperator

__all__ = [
    "PARALLEL_RTOL",
    "CotangentSplit",
    "PortNetwork",
    "parallel",
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


def parallel(functional: np.ndarray, excitation: np.ndarray) -> complex | None:
    """``c`` with ``functional = c excitation`` to `PARALLEL_RTOL`, else ``None``.

    True for every lumped port and for a wave port whose mode is real up to
    one phase (lossless); a lossy wave port's S functional projects on the
    conjugated mode, which is not parallel to its excitation.
    """
    norm = np.vdot(excitation, excitation).real
    if norm == 0:  # pragma: no cover - an excited port always has a load
        return None
    c = np.vdot(excitation, functional) / norm
    residual = np.linalg.norm(functional - c * excitation)
    if residual > PARALLEL_RTOL * np.linalg.norm(functional):
        return None
    return complex(c)


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
) -> CotangentSplit:
    """Split the cotangent ``ct`` on S at one frequency (see `CotangentSplit`).

    Row i adds ``functional[:, i] ⊗ ct_i`` to the loads, or, with
    reciprocity, ``c x_m ⊗ ct_i`` to the free columns when port i is the
    excited column m and its S functional is ``c`` times its excitation
    (`parallel`). A row without a cotangent is skipped (even where its
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
        if not np.isfinite(f_i).all():  # S row of a port without R
            undefined |= ct[i] != 0
            continue
        if own is not None and i in columns:
            rhs, x = own
            m = columns.index(i)
            scale = parallel(f_i, rhs[:, m])
            if scale is not None:  # reciprocity: no solve
                free += np.outer(scale * x[:, m], ct[i])
                continue
        loads += np.outer(f_i, ct[i])
    return CotangentSplit(free, loads, undefined)


@dataclasses.dataclass(eq=False)
class PortNetwork:
    """Every port's incident wave and S functional in one driven solve.

    Attributes:
        excitation: The right-hand side terms, one column per excited port.
        outputs: The S functionals' terms, one row per port.
        waves: The wave ports' couplings.
        columns: The excited ports' indices.
        z_ref: ``(n_freq, n_ports)`` reference impedances, ohm.
        deembedding: ``(n_freq, n_ports)``: S_ij is scaled by ``d_i d_j``
            (1 for lumped; a wave port's ``exp(gamma offset)`` moves its
            reference plane).
    """

    excitation: list[AffineTerm]
    outputs: list[AffineTerm]
    waves: list[WavePortCoupling]
    columns: list[int]
    z_ref: np.ndarray
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
        waves: Sequence[WavePortCoupling],
        *,
        columns: Sequence[int],
        n_freq: int,
    ) -> PortNetwork:
        """The network of ``model``'s ports on ``operator``'s space.

        Args:
            model: The model.
            operator: The driven operator (lumped ports and space).
            waves: The wave ports' couplings.
            columns: The excited ports' indices.
            n_freq: The output grid's size.
        """
        op = operator
        names = model.port_names
        columns = list(columns)
        n_ports, width = len(names), len(columns)
        excitation: list[AffineTerm] = []
        outputs: list[AffineTerm] = []
        lumped = [names.index(data.port.name) for data in op.ports]
        if lumped:
            vectors = [
                terms.port_vectors(op.space, op.built, op.order, data)
                for data in op.ports
            ]
            r = np.array([resistance(model.ports[i]) for i in lumped])
            fed = [(k, columns.index(i)) for k, i in enumerate(lumped) if i in columns]
            if fed:
                b = np.stack([vectors[k][0] for k, _ in fed], axis=1)
                select = np.zeros((len(fed), width))
                for a, (_, pos) in enumerate(fed):
                    select[a, pos] = 1.0

                def lumped_rhs(w: float, select=select) -> np.ndarray:
                    return 1j * op.wavenumber(w) * select

                excitation.append(AffineTerm(b, lumped_rhs))
            rows = np.zeros((n_ports, len(lumped)))
            rows[lumped, range(len(lumped))] = 1.0 / np.sqrt(2 * r)
            voltage = np.stack([v[1] for v in vectors], axis=1)
            outputs.append(AffineTerm(voltage, lambda _, rows=rows: rows))
        for c in waves:
            m2 = c.vectors.shape[1]
            if c.index in columns:
                pos = columns.index(c.index)

                def wave_rhs(w: float, c=c, pos=pos, m2=m2) -> np.ndarray:
                    out = np.zeros((m2, width), complex)
                    out[:, pos] = c.excitation(w)
                    return out

                excitation.append(AffineTerm(c.vectors, wave_rhs))

            def wave_row(w: float, c=c, m2=m2) -> np.ndarray:
                out = np.zeros((n_ports, m2), complex)
                out[c.index] = c.functional(w)
                return out

            outputs.append(AffineTerm(np.conj(c.vectors), wave_row))
        z_ref = np.zeros((n_freq, n_ports))
        for i in lumped:
            z_ref[:, i] = resistance(model.ports[i])
        for c in waves:
            z_ref[:, c.index] = c.z_ref
        return cls(excitation, outputs, list(waves), columns, z_ref)

    def rhs(self, omega: float) -> np.ndarray:
        """The right-hand sides at ω, ``(N, width)``."""
        return sum(
            t.vectors @ np.asarray(t.coefficients(omega)) for t in self.excitation
        )

    def outgoing(self, omega: float, x: np.ndarray) -> np.ndarray:
        """``(n_ports, width)``: each port's outgoing wave in the solutions ``x``."""
        return self._rows(omega, (t.vectors.T @ x for t in self.outputs))

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
        rhs = None
        loads = []
        for c in self.waves:
            f = functional[:, c.index]
            if c.index in self.columns:
                if rhs is None:
                    rhs = self.rhs(omega)
                if parallel(f, rhs[:, self.columns.index(c.index)]) is not None:
                    continue
            loads.append(f)
        if not loads:
            return np.zeros((len(functional), 0), complex)
        return np.stack(loads, axis=1)

    def pullback(
        self, n: int, omega: float, ct: np.ndarray, x: np.ndarray
    ) -> CotangentSplit:
        """Split the S cotangent ``ct`` at frequency ``n``, with reciprocity.

        ``x`` ``(N, width)`` holds the excited ports' solutions at ω.
        """
        return split_cotangent(
            ct,
            self.deembedding[n],
            self.columns,
            self.functional(omega),
            own=(self.rhs(omega), x),
        )

    @property
    def nbytes(self) -> int:
        """Host memory held by the excitation and output vectors, in bytes.

        The system operator counts ``W``'s (`SystemOperator.nbytes`).
        """
        terms_ = [*self.excitation, *self.outputs]
        return sum(int(t.vectors.nbytes) for t in terms_)
