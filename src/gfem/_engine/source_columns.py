"""The source columns of a driven solve: right-hand sides beside the excited ports.

Each source (`gfem.Driven(sources=...)`) is one more column of the block
solved on the one factorization per frequency, after the excited ports, in
the problem's order. Each kind of source has one builder here
(`COLUMN_KINDS`): given the forward's concrete model and system operator
and its sources of that kind, it returns their loads as a function of ω,
``(N, n)`` on the solve space. `SourceColumns` places every kind's
columns in problem order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem._sources import DIPOLES, INCIDENT, POLARIZATION, Source
from gfem._sweep.adaptive import AffineTerm

if TYPE_CHECKING:
    from gfem._engine.dipoles import PointLoads
    from gfem._engine.system_operator import SystemOperator
    from gfem._engine.volume_loads import VolumeLoads
    from gfem._pml import Stretch
    from gfem.model import Model

__all__ = ["COLUMN_KINDS", "ColumnContext", "SourceColumns", "real_up_to_phase"]


@dataclasses.dataclass(frozen=True, eq=False)
class ColumnContext:
    """What a builder of source columns reads of one forward.

    Attributes:
        model: The model, with concrete leaves.
        system: The system operator of the solve (its operator's space is
            the solve space).
        order: The Nédélec order.
        stretch: The perfectly matched layer's stretch, or ``None``.
        frequencies: The frequencies the forward solves, Hz.
        field_layout: The kept fields' probe layout
            (`gfem._engine.probe.Layout`: support, exclusions, space,
            order), where a point source is located.
        comm: The communicator of a distributed solve, or ``None``.
    """

    model: Model
    system: SystemOperator
    order: int
    stretch: Stretch | None
    frequencies: tuple[float, ...]
    field_layout: Any = None
    comm: Any = None


#: ``context, sources -> (omega -> (N, len(sources)) loads)``.
ColumnBuilder = Callable[[ColumnContext, Sequence[Any]], Callable[[float], np.ndarray]]


def _incident(
    context: ColumnContext, sources: Sequence[Any]
) -> Callable[[float], np.ndarray]:
    from gfem._engine.incident import incident_columns

    return incident_columns(
        context.model,
        context.system,
        sources,
        order=context.order,
        stretch=context.stretch,
        comm=context.comm,
    )


def _dipoles(context: ColumnContext, sources: Sequence[Any]) -> PointLoads:
    from gfem._engine.dipoles import point_loads

    op = context.system.operator
    return point_loads(
        context.model,
        sources,
        context.field_layout,
        op.space,
        context.frequencies,
        dof_layout=op.layout,
        comm=context.comm,
    )


def _polarization(context: ColumnContext, sources: Sequence[Any]) -> VolumeLoads:
    from gfem._engine.volume_loads import volume_loads

    if context.comm is not None:
        raise NotImplementedError(
            "A PolarizationSource on MPI ranks: its load is assembled from one "
            "process's rule points. Solve it on one process."
        )
    return volume_loads(context.system.operator, sources, order=context.order)


#: Per kind of source (its classes), the builder of its columns.
COLUMN_KINDS: tuple[tuple[tuple[type, ...], ColumnBuilder], ...] = (
    (INCIDENT, _incident),
    (DIPOLES, _dipoles),
    (POLARIZATION, _polarization),
)


def real_up_to_phase(moment: np.ndarray) -> bool:
    r"""Whether a moment is a real vector times one phase: :math:`\bar m \parallel m`.

    Then a dipole's emitted power has the dipole's own solution as its
    adjoint.
    """
    m = np.asarray(moment, complex)
    size = float(np.vdot(m, m).real)
    return abs(np.dot(m, m)) >= (1 - 1e-12) * size


@dataclasses.dataclass(eq=False)
class SourceColumns:
    """The source columns of one forward, in the problem's source order.

    Attributes:
        names: The sources' names, in column order.
        ndof: The solve space's size.
        kinds: Per kind, the column positions it fills and its loads.
    """

    names: tuple[str, ...]
    ndof: int
    kinds: list[tuple[list[int], Callable[[float], np.ndarray]]]

    @classmethod
    def build(cls, sources: Sequence[Source], context: ColumnContext) -> SourceColumns:
        """Every kind's builder on its sources.

        Raises:
            NotImplementedError: A source of a kind without a builder.
        """
        kinds = []
        placed: set[int] = set()
        for classes, builder in COLUMN_KINDS:
            positions = [k for k, s in enumerate(sources) if isinstance(s, classes)]
            if positions:
                loads = builder(context, [sources[k] for k in positions])
                kinds.append((positions, loads))
                placed.update(positions)
        for k, source in enumerate(sources):
            if k not in placed:
                raise NotImplementedError(
                    f"The {type(source).__name__} {source.name!r} cannot be solved "
                    "in this gfem version."
                )
        ndof = int(context.system.operator.space.ndof)
        return cls(tuple(s.name for s in sources), ndof, kinds)

    def rhs(self, omega: float) -> np.ndarray:
        """The loads at ω, ``(N, n_sources)``."""
        out = np.zeros((self.ndof, len(self.names)), complex)
        for positions, loads in self.kinds:
            out[:, positions] = loads(omega)
        return out

    @property
    def dipoles(self) -> tuple[list[int], PointLoads] | None:
        """The point dipoles' column positions and loads, or ``None``."""
        from gfem._engine.dipoles import PointLoads

        for positions, loads in self.kinds:
            if isinstance(loads, PointLoads):
                return positions, loads
        return None

    def port_waves(self, omega: float, z: np.ndarray) -> np.ndarray:
        """``(n_ports, n_sources)``: the sources' port waves from the ports' solutions.

        ``z`` is ``(N, n_ports)`` (`PortSolutions.solutions`); a dipole's
        wave is one point overlap with it (`PointLoads.port_waves`). Only
        point dipoles are read this way.
        """
        out = np.zeros((z.shape[1], len(self.names)), complex)
        found = self.dipoles
        if found is not None:
            positions, loads = found
            out[:, positions] = loads.port_waves(omega, z)
        return out

    def reduced_port_waves(
        self, omega: float, projected: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        r"""`port_waves` from the ports' reduced solutions ``(n, n_ports)``.

        ``projected`` is :math:`B^\mathsf{T} V`, ``(n_dipoles, n)``, with
        :math:`B` = `PointLoads.vectors` and :math:`V` the real basis.
        """
        out = np.zeros((coordinates.shape[1], len(self.names)), complex)
        found = self.dipoles
        if found is not None:
            positions, loads = found
            out[:, positions] = loads.reduced_port_waves(omega, projected, coordinates)
        return out

    def port_wave_loads(self, omega: float, weights: np.ndarray) -> np.ndarray:
        """``(N, r)``: the dipoles' loads summed with each row of ``weights``.

        ``weights`` ``(r, n_sources)``, the cotangent rows of `port_waves`
        de-embedded: the adjoint loads of their backward
        (`PointLoads.weighted`).
        """
        found = self.dipoles
        if found is None:
            return np.zeros((self.ndof, len(weights)), complex)
        positions, loads = found
        return loads.port_wave_loads(omega, np.asarray(weights)[:, positions])

    def emitted_power(self, omega: float, x: np.ndarray) -> np.ndarray:
        """``(n_emitters,)`` W from the source columns' solutions ``(N, n_sources)``.

        The emitters are the point dipoles and the polarization sources, in
        the problem's source order.
        """
        found = []
        for positions, loads in self.kinds:
            emitted = getattr(loads, "emitted_power", None)
            if emitted is not None:
                power = emitted(omega, x[:, positions])
                found.extend(zip(positions, power, strict=True))
        return np.array([p for _, p in sorted(found)], float)

    def power_loads(self, omega: float, cotangent: np.ndarray) -> np.ndarray:
        """``(N, n_sources)``: the cotangent on ``x`` of one on the emitted power.

        Zero in the other sources' columns (`PointLoads.power_loads`).
        """
        out = np.zeros((self.ndof, len(self.names)), complex)
        found = self.dipoles
        if found is not None:
            positions, loads = found
            out[:, positions] = loads.power_loads(omega, cotangent)
        return out

    def adjoint_loads(self, omega: float) -> np.ndarray:
        """``(N, q)``: the emitted powers' adjoint loads the solutions miss.

        A dipole whose moment is real up to one phase is its own adjoint (its
        power's load is parallel to its own); a circular one's is the
        conjugate load, which an adaptive sweep solves at each sample.
        """
        found = self.dipoles
        if found is None:
            return np.zeros((self.ndof, 0), complex)
        _, loads = found
        complex_ = [k for k, m in enumerate(loads.moments) if not real_up_to_phase(m)]
        return np.conj(loads.rhs(omega)[:, complex_])

    def reduced_power(
        self, omega: float, projected: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        r"""``(n_dipoles,)`` W from the sources' reduced coordinates ``(n, n_sources)``.

        ``projected`` is :math:`V^\mathsf{T} B` (dense), ``(n, n_dipoles)``,
        with :math:`B` = `PointLoads.vectors` and :math:`V` the real basis.
        """
        found = self.dipoles
        if found is None:
            return np.zeros(0)
        positions, loads = found
        y = coordinates[:, positions]
        pairs = np.conj(loads.coefficients(omega)) * np.sum(np.conj(projected) * y, 0)
        return -np.imag(pairs) * loads.power_scale(omega)

    def reduced_power_loads(
        self, omega: float, projected: np.ndarray, cotangent: np.ndarray
    ) -> np.ndarray:
        """``(n, n_sources)``: `power_loads` on a real basis, ``projected`` given."""
        out = np.zeros((projected.shape[0], len(self.names)), complex)
        found = self.dipoles
        if found is not None:
            positions, loads = found
            scale = np.asarray(cotangent, float) * loads.power_scale(omega)
            c = np.conj(loads.coefficients(omega)) * scale
            out[:, positions] = 1j * np.conj(projected) * c[None, :]
        return out

    def excitation(
        self, port_terms: Sequence[AffineTerm], n_excited: int
    ) -> list[AffineTerm]:
        """An adaptive sweep's excitation terms of the whole block.

        The ports' terms widened by zero source columns, then the dipoles'
        (a fixed vector times a scalar of ω, `PointLoads.coefficients`).

        Raises:
            NotImplementedError: A source whose load is not affine in ω (an
                incident field; `gfem.Driven` refuses it first).
        """
        width = n_excited + len(self.names)

        def widen(term: AffineTerm) -> AffineTerm:
            def coefficients(w: float, term: AffineTerm = term) -> np.ndarray:
                c = np.asarray(term.coefficients(w))
                out = np.zeros((c.shape[0], width), complex)
                out[:, :n_excited] = c
                return out

            return AffineTerm(term.vectors, coefficients, term.conjugate)

        out = [widen(t) for t in port_terms]
        found = self.dipoles
        affine = 0 if found is None else len(found[0])
        if affine != len(self.names):  # pragma: no cover - refused at Driven
            raise NotImplementedError("only point dipoles are affine in frequency")
        if found is not None:
            positions, loads = found
            block = [n_excited + k for k in positions]

            def dipole_coefficients(w: float) -> np.ndarray:
                c = np.zeros((len(block), width), complex)
                c[range(len(block)), block] = loads.coefficients(w)
                return c

            out.append(AffineTerm(loads.vectors, dipole_coefficients))
        return out
