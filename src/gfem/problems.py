"""Problem types: what to solve on a model, as frozen values.

A problem names the physics analysis and its physical inputs (terminals,
frequencies, ...). The model holds everything about the device; the solver
options hold the numerics. ``gfem.solve(model, problem)`` dispatches on the
problem's type.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, ClassVar

import jax

__all__ = ["Electrostatic"]


def _surfaces(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(s, str) for s in value):
        return tuple(value)
    raise TypeError(
        f"{what} must be a surface name or a list of surface names, got {value!r}."
    )


@dataclasses.dataclass(frozen=True, init=False)
class Electrostatic:
    """The Maxwell capacitance matrix between named terminals.

    For each terminal in turn, the potential is 1 V on that terminal and
    0 V on every other terminal and on ground, and the charge-free Laplace
    equation div(eps grad V) = 0 is solved in the electromagnetic solve
    domain (the model's excluded volumes are left out). Surfaces that are
    neither terminal nor ground carry no surface charge (zero normal
    displacement), except that every surface of a `PEC` or `Conductor`
    boundary of the model that is not a terminal is grounded, as conductors
    are. Ports and other boundaries are charge-free.

    Only the real part of the permittivity is read: losses
    (``tan_delta``, ``conductivity``, a negative imaginary ``eps_r``) and
    ``mu_r`` do not enter the electrostatic problem.

    Args:
        terminals: Terminal name -> its surface name or names (a conductor's
            shell), in capacitance-matrix order. A list of surface names
            makes one terminal per surface, named after it.
        ground: The surface name or names held at 0 V besides the model's
            conductors. Empty: only the model's `PEC` / `Conductor`
            surfaces, or no ground at all (then each row of the matrix sums
            to zero).

    Raises:
        TypeError: A name is not a string.
        ValueError: No terminal, or a surface in two terminals or in a
            terminal and the ground.

    Example:
        ``gfem.Electrostatic(terminals={"trace": ["m1_xy", "m1_z"]},
        ground="ground")``
    """

    terminals: tuple[tuple[str, tuple[str, ...]], ...]
    ground: tuple[str, ...]
    _rule: ClassVar[str] = "gfem._engine.electrostatic:ElectrostaticRule"

    def __init__(
        self,
        terminals: Mapping[str, str | list[str] | tuple[str, ...]] | list[str],
        ground: str | list[str] | tuple[str, ...] = (),
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        if isinstance(terminals, Mapping):
            items = []
            for name, surfaces in terminals.items():
                if not isinstance(name, str):
                    raise TypeError(f"Terminal names must be strings, got {name!r}.")
                items.append((name, _surfaces(surfaces, f"terminals[{name!r}]")))
        else:
            items = [(s, (s,)) for s in _surfaces(terminals, "terminals")]
        if not items:
            raise ValueError(
                "Electrostatic needs at least one terminal, e.g. "
                "terminals={'trace': 'trace_surface'}."
            )
        ground = _surfaces(ground, "ground") if ground else ()
        owner: dict[str, str] = {}
        for name, surfaces in items:
            if len(set(surfaces)) != len(surfaces):
                raise ValueError(f"Terminal {name!r} lists a surface twice.")
            for s in surfaces:
                if s in owner:
                    raise ValueError(
                        f"Surface {s!r} is in terminals {owner[s]!r} and {name!r}; "
                        "a surface belongs to one terminal."
                    )
                owner[s] = name
        for s in ground:
            if s in owner:
                raise ValueError(
                    f"Surface {s!r} is both in terminal {owner[s]!r} and in the "
                    "ground; remove it from one of them."
                )
        object.__setattr__(self, "terminals", tuple(items))
        object.__setattr__(self, "ground", tuple(dict.fromkeys(ground)))

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """Terminal names, in capacitance-matrix order."""
        return tuple(name for name, _ in self.terminals)

    def __repr__(self) -> str:
        terms = ", ".join(f"{n!r}: {list(s)}" for n, s in self.terminals)
        ground = f", ground={list(self.ground)}" if self.ground else ""
        return f"Electrostatic(terminals={{{terms}}}{ground})"


jax.tree_util.register_static(Electrostatic)
