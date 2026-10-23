"""Affine pieces assembled once, summed per operator: the piece sum.

An operator ``A = Σ θ_k A_k`` whose pieces ``A_k`` do not change (a driven
sweep's frequencies) need not be assembled again for each set of weights
θ_k: `AssembledPieces` assembles each piece once, with θ = 1, and
`PieceSum` writes ``Σ θ_k A_k`` into a matrix of the full sparsity pattern.
That skips NGSolve's allocation of a new matrix and graph, the part of an
assembly that does not scale with threads.

The pieces with the full pattern (the volume terms) are summed as NGSolve
vectors over the matrix values; each other piece (a surface term, whose
graph couples its surface DOFs only) keeps its compact values and their
positions in the full pattern, and is added there.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem._linalg.base import host_bytes, same_pattern


class AssembledPieces:
    """The pieces ``A_k`` of a complex space, assembled on first use (θ = 1).

    Holds the full-pattern matrix of the piece with the most nonzeros (its
    pattern is the operator's), the values of the other full-pattern pieces,
    and each remaining piece's compact values with their positions in that
    pattern. `free` drops them; a later use assembles them again.
    """

    def __init__(
        self, space: ngs.FESpace, integrands: Sequence[Callable[[Any, Any], Any]]
    ) -> None:
        """Hold ``space`` and each piece's integrand ``(trial, test) -> integrals``."""
        self._space = space
        self._integrands = tuple(integrands)
        self._built = False
        self._template: ngs.BaseMatrix | None = None
        self._parts: list[tuple[Any, np.ndarray | None]] = []

    @property
    def pattern(self) -> ngs.BaseMatrix | None:
        """The full-pattern matrix every sum is shaped as, or ``None``.

        ``None`` when a piece couples DOFs that the full pattern does not:
        the pieces cannot be summed, the operator must be assembled.
        """
        if not self._built:
            self._build()
        return self._template

    def matrix(self, weights: Sequence[complex]) -> ngs.BaseMatrix | None:
        """A new matrix of `pattern` holding ``Σ θ_k A_k``; ``None`` without one."""
        template = self.pattern
        if template is None:
            return None
        matrix = template.CreateMatrix()
        self.sum_into(weights, matrix)
        return matrix

    def sum_into(self, weights: Sequence[complex], matrix: ngs.BaseMatrix) -> None:
        """Overwrite the values of ``matrix`` (of `pattern`) with ``Σ θ_k A_k``."""
        if self.pattern is None:  # pragma: no cover - callers check the pattern
            raise ValueError("These pieces have no common sparsity pattern.")
        weights = [complex(w) for w in weights]
        target = matrix.AsVector()
        full = [
            (w, v)
            for w, (v, at) in zip(weights, self._parts, strict=True)
            if at is None
        ]
        (w0, v0), *rest = full
        target.data = w0 * v0
        for w, values in rest:
            target.data += w * values
        out = target.FV().NumPy()
        for w, (values, at) in zip(weights, self._parts, strict=True):
            if at is not None:
                out[at] += w * values

    @property
    def nbytes(self) -> int:
        """Host memory of the assembled pieces, in bytes (0 before first use)."""
        if self._template is None:
            return 0
        # the template's own values are in its matrix; the other full pieces'
        # are copies of the same size
        full = sum(at is None for _, at in self._parts)
        held = host_bytes(self._template) + 16 * self._template.nze * (full - 1)
        for values, at in self._parts:
            if at is not None:
                held += int(values.nbytes) + int(at.nbytes)
        return held

    def free(self) -> None:
        """Drop the assembled pieces."""
        self._built = False
        self._template = None
        self._parts = []

    def _build(self) -> None:
        u, v = self._space.TnT()
        matrices = []
        for integrand in self._integrands:
            form = ngs.BilinearForm(self._space, symmetric=True)
            form += integrand(u, v)
            form.Assemble()
            matrices.append(form.mat)
        template = max(matrices, key=lambda m: m.nze)
        _, cols, rowptr = template.CSR()
        cols = np.asarray(cols)
        rowptr = np.asarray(rowptr).astype(np.int64)
        parts: list[tuple[Any, np.ndarray | None]] = []
        for m in matrices:
            if m is template:
                parts.append((template.AsVector(), None))
            elif same_pattern(m, template):
                # a copy, so that the piece's own matrix and graph are released
                values = ngs.BaseVector(m.nze, True)
                values.FV().NumPy()[:] = m.AsVector().FV().NumPy()
                parts.append((values, None))
            else:
                data, c, r = m.CSR()
                at = _positions(rowptr, cols, np.asarray(r), np.asarray(c))
                if at is None:
                    self._built = True
                    return
                parts.append((np.array(data), at))
        self._template, self._parts, self._built = template, parts, True


def _positions(
    rowptr: np.ndarray, cols: np.ndarray, sub_rowptr: np.ndarray, sub_cols: np.ndarray
) -> np.ndarray | None:
    """The positions of a sub-pattern's entries in the CSR value vector of a pattern.

    Reads only the pattern's rows that the sub-pattern touches. Columns are
    sorted within each row (NGSolve's graphs are). ``None`` when an entry is
    not in the pattern.
    """
    counts = np.diff(sub_rowptr.astype(np.int64))
    rows = np.flatnonzero(counts)
    starts = rowptr[rows]
    lengths = rowptr[rows + 1] - starts
    # the positions of the touched rows' entries, row after row
    first = np.cumsum(lengths) - lengths
    positions = np.repeat(starts - first, lengths) + np.arange(lengths.sum())
    width = len(rowptr)  # above every column index
    keys = np.repeat(np.arange(len(rows)), lengths) * width + cols[positions]
    wanted = np.repeat(np.arange(len(rows)), counts[rows]) * width + sub_cols
    if not len(wanted):
        return np.zeros(0, np.int64)
    found = np.searchsorted(keys, wanted)
    if found.max() >= len(keys) or not np.array_equal(keys[found], wanted):
        return None
    return positions[found]


@dataclasses.dataclass(frozen=True, eq=False)
class PieceSum:
    """One operator as a piece sum: ``Σ θ_k A_k`` of assembled pieces.

    Attributes:
        pieces: The assembled pieces ``A_k``.
        weights: θ_k, one per piece.
    """

    pieces: AssembledPieces
    weights: tuple[complex, ...]

    @property
    def pattern(self) -> ngs.BaseMatrix | None:
        """See `AssembledPieces.pattern`."""
        return self.pieces.pattern

    def matrix(self) -> ngs.BaseMatrix | None:
        """See `AssembledPieces.matrix`."""
        return self.pieces.matrix(self.weights)

    def sum_into(self, matrix: ngs.BaseMatrix) -> None:
        """See `AssembledPieces.sum_into`."""
        self.pieces.sum_into(self.weights, matrix)
