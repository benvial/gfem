"""Results of `gfem.solve`: frozen pytrees of arrays with static labels."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from gfem._pytree import register_fields

__all__ = ["ElectrostaticResult"]


@dataclasses.dataclass(frozen=True, eq=False)
class ElectrostaticResult:
    """The result of an `Electrostatic` problem.

    Excitation ``i`` puts terminal ``i`` at 1 V and every other terminal
    and the ground at 0 V.

    Attributes:
        capacitance: The Maxwell capacitance matrix in farads, shape
            ``(n_terminals, n_terminals)``, rows and columns in `terminals`
            order: ``capacitance[i, j]`` is the charge on terminal ``i`` in
            excitation ``j``. The diagonal is positive and the off-diagonal
            entries are negative or zero (Palace's ``terminal-C.csv``
            convention).
        terminals: The terminal names (static labels).
        energy: Electric energy in joules stored in each volume region of
            the solve domain, by region name: an ``(n_terminals,)`` array,
            entry ``i`` for excitation ``i`` (Palace's ``domain-E.csv`` at
            1 V). Excluded volumes are left out. Summed over the regions,
            excitation ``i`` stores ``capacitance[i, i] / 2``.
    """

    capacitance: jax.Array
    terminals: tuple[str, ...]
    energy: dict[str, jax.Array] = dataclasses.field(default_factory=dict)

    @property
    def mutual(self) -> jax.Array:
        """The mutual capacitance matrix in farads (Palace ``terminal-Cm.csv``).

        Off the diagonal, ``mutual[i, j] = -capacitance[i, j]``, the
        capacitance between terminals ``i`` and ``j``; on it, the row sum
        of `capacitance`, the capacitance of terminal ``i`` to ground.
        """
        c = self.capacitance
        return jnp.diag(c.sum(axis=1)) - (c - jnp.diag(jnp.diag(c)))

    @property
    def inverse(self) -> jax.Array:
        """The inverse of `capacitance` in 1/F (Palace ``terminal-Cinv.csv``).

        Entry ``[i, j]`` is the voltage of terminal ``i`` per coulomb on
        terminal ``j``, the others carrying no charge. Without a ground the
        rows of `capacitance` sum to zero and it has no inverse.
        """
        return jnp.linalg.inv(self.capacitance)

    def index(self, terminal: str) -> int:
        """Row / column of ``terminal`` in `capacitance`.

        Raises:
            KeyError: ``terminal`` is not a terminal of this result.
        """
        try:
            return self.terminals.index(terminal)
        except ValueError:
            raise KeyError(
                f"No terminal {terminal!r}; the terminals are "
                f"{', '.join(repr(t) for t in self.terminals)}."
            ) from None

    def __repr__(self) -> str:
        try:
            c = np.asarray(self.capacitance) * 1e15
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"ElectrostaticResult(terminals={list(self.terminals)})"
        width = max(len(t) for t in self.terminals)
        rows = [
            f"  {t:>{width}}: " + "  ".join(f"{v:10.4g}" for v in row)
            for t, row in zip(self.terminals, c, strict=True)
        ]
        return (
            f"ElectrostaticResult(terminals={list(self.terminals)}, "
            "capacitance in fF:\n" + "\n".join(rows) + ")"
        )


register_fields(ElectrostaticResult, ("capacitance", "energy"))
