"""Results of `gfem.solve`: frozen pytrees of arrays with static labels."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np

from gfem._pytree import register_fields

__all__ = ["ElectrostaticResult"]


@dataclasses.dataclass(frozen=True, eq=False)
class ElectrostaticResult:
    """The result of an `Electrostatic` problem.

    Attributes:
        capacitance: The Maxwell capacitance matrix in farads, shape
            ``(n_terminals, n_terminals)``, rows and columns in `terminals`
            order: ``capacitance[i, j]`` is the charge on terminal ``i`` per
            volt on terminal ``j`` with every other terminal and the ground
            at 0 V. The diagonal is positive and the off-diagonal entries
            are negative or zero (Palace's ``terminal-C.csv`` convention).
        terminals: The terminal names (static labels).
    """

    capacitance: jax.Array
    terminals: tuple[str, ...]

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


register_fields(ElectrostaticResult, ("capacitance",))
