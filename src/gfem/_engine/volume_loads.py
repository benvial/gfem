r"""Polarization sources as load columns, and the fields of a pump at their points.

A `gfem.PolarizationSource` holds :math:`P` at the points of NGSolve's rule of
degree :math:`2Q` (an ``IntegrationRuleSpace`` of order :math:`Q \le 5`: NGSolve's
rule space is wrong above degree 10), element by element, on the elements
``elements`` of the mesh. Two maps of the H(curl) coefficients to those points
serve it. The maps of a rule of degree 10 on an order-3 space hold some
:math:`3\times10^4` entries per element, tens of GiB on :math:`10^5` elements,
so they are never assembled for the mesh. On a straight tetrahedron the field
of the element's basis is the covariant Piola image
:math:`E = J^{-\mathsf{T}}\hat E` of the reference field :math:`\hat E`, which
depends on the order of the element's vertices only: `_tables` assembles a
couple of elements per vertex order (`_Rule.chunks`, a few cells at a time),
takes each element's :math:`J` from its points, and keeps one table per class
(`_Tables`; a curved element keeps its own).

- the pump's field at the points, :math:`E = (M x)/(w L)`, a gather of each
  element's DOFs and one einsum (`PumpColumns`, no sparse product, so that it
  runs on traced coefficients), with :math:`w` the weight of a point
  (:math:`\omega_q |J|`, file units cubed) and :math:`L` the metres per file
  unit: the kept coefficients are on the file-unit basis. On a straight
  tetrahedron the field is the covariant Piola image
  :math:`E = J^{-\mathsf{T}}\hat E`, `PumpColumns` keeps the shared tables
  with each element's :math:`J^{-1}`;
- the generated load, :math:`b = \mu_0 L^3\omega^2\,M^\mathsf{T} P` (`VolumeLoads`),
  in file units: :math:`P` and the volume are SI, so :math:`\int P\cdot\varphi
  \,\mathrm{d}V = L^2 M^\mathsf{T} P` on the file-unit basis, times the
  :math:`\mu_0 L\omega^2` of the driven equations (`gfem._engine.dipoles`). A
  mesh in um and the same mesh in m give the same :math:`P` and the same load.

On a periodic cell the maps read the unpaired space, where the kept pump fields
live (already Bloch-phased), and the load is carried to the solve space by
:math:`\mathsf{P}^H` of the Bloch-phased expansion :math:`\mathsf{P}` (the maps
become :math:`M\overline{\mathsf{P}}`): the test functions of the phased space are
:math:`\overline{\mathsf{P}}y`, which is how a Floquet port's vector is carried
(:math:`\mathsf{P}^\mathsf{T}` would test with the opposite phase).

The load is a fixed vector times a scalar of :math:`\omega` (affine), and the
emitted power is the dipole's, :math:`-\operatorname{Im}(b^H x)/(2\omega\mu_0 L)`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants as const
import scipy.sparse as sp

from gfem._engine.fields import fe_space
from gfem._engine.runtime import serial_spaces
from gfem._linalg.base import free_mask
from gfem._sources import PolarizationSource

__all__ = ["PumpColumns", "VolumeLoads", "pump_columns", "volume_loads"]

#: Relative distance (of the points' extent) within which the rule points of
#: two builds of one mesh count as the same: both are the reference rule
#: mapped by the same vertices, equal to round-off.
_SAME_POINTS = 1e-8
#: Entries of the three maps assembled at once (about 100 MB of working memory).
_CHUNK_ENTRIES = 2**22
#: Entries of one point's map per component, the order-3 element's DOF count
#: (the chunk size only; an element with more or fewer DOFs changes the memory
#: of a chunk, not its result).
_ENTRIES_PER_POINT = 45
#: Bytes of the expanded tables of the elements `_Tables` and `PumpColumns`
#: expand at once, so that the expansion is a bounded fraction of the memory of
#: the dense weights it replaces: about 60 MB.
_EVAL_BYTES = 6 * 10**7
#: Relative distance, of the reference tables of two elements of one class and
#: of a point to its affine fit: round-off of sums of up to a few hundred terms.
_AFFINE = 1e-8
#: Elements whose points `_straight` fits at once.
_FIT_ELEMENTS = 4096
#: The largest rule order: NGSolve's rule space returns wrong weights and
#: points above degree 10 (measured from degree 12 on).
MAX_RULE_ORDER = 5


class _Rule:
    """One order's rule space on a mesh, assembled a few elements at a time.

    The rule space has ``per_cell`` points per element of the whole mesh, and
    each point's maps from the coefficients hold as many entries as the
    element has DOFs: a rule of degree 10 (216 points) on an order-3 space is
    some 3e4 entries per element, so the maps of a mesh of 1e5 elements would
    hold tens of GiB. `chunks` assembles them for a few cells at a time, and
    every caller reduces a chunk before it asks for the next.
    """

    def __init__(self, space: ngs.FESpace, order: int) -> None:
        mesh = space.mesh
        if order > MAX_RULE_ORDER:
            raise NotImplementedError(
                f"the rule space of order {order} (degree {2 * order}) is not "
                "supported: NGSolve's integration rule space is wrong above "
                f"degree {2 * MAX_RULE_ORDER}, so the rule order is at most "
                f"{MAX_RULE_ORDER}."
            )
        self.space = space
        self.order = order
        rule = ngs.IntegrationRule(ngs.ET.TET, 2 * order)
        #: ``(per_cell,)`` the weights :math:`w_q` of the reference rule.
        self.weights = np.array(list(rule.weights), float)
        #: ``(per_cell, 3)`` the points in the reference tetrahedron.
        self.reference = np.array([[*p] for p in rule.points], float)
        self.per_cell = len(rule.points)
        self._rule = rule
        with serial_spaces():
            self.irs = ngs.comp.IntegrationRuleSpace(
                mesh, order=order, complex=space.is_complex
            )
        if self.irs.ndof != int(mesh.ne) * self.per_cell:  # pragma: no cover
            raise RuntimeError(  # NGSolve contract
                f"the integration rule space holds {self.irs.ndof} points, not "
                f"{self.per_cell} per element on {int(mesh.ne)} elements."
            )

    def chunks(
        self, cells: np.ndarray, *, weights: bool = False
    ) -> Iterator[tuple[int, int, list[sp.csr_matrix], np.ndarray]]:
        """``(lo, hi, maps, w)`` for the points of successive groups of ``cells``.

        The points ``lo:hi`` are those of ``cells[lo // per_cell: hi // per_cell]``,
        element by element. ``maps`` are the three component maps
        ``(hi - lo, space.ndof)`` (entries carry the weight of each point: the
        point's test function of the rule space is its value times the weight
        :math:`w_q |J|` of the rule), and ``w`` the weights, ``(hi - lo,)``
        when ``weights`` is set (else empty).
        """
        space, irs, n_q = self.space, self.irs, self.per_cell
        ne = int(space.mesh.ne)
        step = max(1, _CHUNK_ENTRIES // (3 * n_q * _ENTRIES_PER_POINT))
        local = np.full(ne, -1, np.int64)
        v = irs.TestFunction()
        u = space.TrialFunction()
        for first in range(0, len(cells), step):
            group = cells[first : first + step]
            local[group] = np.arange(len(group))
            bits = ngs.BitArray(ne)
            bits.Clear()
            for c in group:
                bits.Set(int(c))
            dx = ngs.dx(intrules={ngs.ET.TET: self._rule}, definedonelements=bits)
            shape = (len(group) * n_q, int(space.ndof))
            maps = []
            for c in range(3):
                form = ngs.BilinearForm(space, irs)
                form += u[c] * v * dx
                maps.append(_coo(form, local, n_q, shape))
            w = np.zeros(0)
            if weights:
                mass = ngs.BilinearForm(irs)
                mass += irs.TrialFunction() * v * dx
                w = _diagonal(mass, local, n_q, shape[0])
            local[group] = -1
            yield first * n_q, first * n_q + shape[0], maps, w

    def coordinates(self, cells: np.ndarray) -> np.ndarray:
        """``(len(cells) * per_cell, 3)``: the points of ``cells``, file units."""
        rows = _rows(cells, self.per_cell)
        mapped = self.space.mesh.MapToAllElements({ngs.ET.TET: self._rule}, ngs.VOL)
        out = np.empty((len(rows), 3))
        for j, coordinate in enumerate((ngs.x, ngs.y, ngs.z)):
            out[:, j] = np.asarray(coordinate(mapped)).reshape(-1)[rows]
        return out


def _diagonal(form: Any, local: np.ndarray, per_cell: int, n: int) -> np.ndarray:
    """The diagonal of a pointwise form (the points' weights) on the cells ``local``."""
    form.Assemble()
    rows, cols, vals = form.mat.COO()
    rows, cols = np.asarray(rows), np.asarray(cols)
    mine = local[rows // per_cell]
    keep = (mine >= 0) & (rows == cols)
    out = np.zeros(n)
    out[(mine * per_cell + rows % per_cell)[keep]] = np.real(np.asarray(vals)[keep])
    return out


def _coo(
    form: Any, local: np.ndarray, per_cell: int, shape: tuple[int, int]
) -> sp.csr_matrix:
    """The assembled form as a matrix with the rows of the cells ``local`` only."""
    form.Assemble()
    rows, cols, vals = form.mat.COO()
    rows = np.asarray(rows)
    mine = local[rows // per_cell]
    keep = mine >= 0
    return sp.csr_matrix(
        (
            np.asarray(vals)[keep],
            ((mine * per_cell + rows % per_cell)[keep], np.asarray(cols)[keep]),
        ),
        shape=shape,
    )


def _cells(built: Any, elements: np.ndarray) -> np.ndarray:
    """The NGSolve cells of the gfem volume elements ``elements``."""
    cell_of = np.empty(len(built.cell_order), np.int64)
    cell_of[np.asarray(built.cell_order)] = np.arange(len(built.cell_order))
    return cell_of[np.asarray(elements)]


def _rows(cells: np.ndarray, per_cell: int) -> np.ndarray:
    """The rule-space DOFs of ``cells``, element by element."""
    return (cells[:, None] * per_cell + np.arange(per_cell)).ravel()


@dataclasses.dataclass(frozen=True, eq=False)
class _Tables:
    r"""The basis of a space at the rule points of some cells: shared tables.

    Attributes:
        dofs: ``(n_el, k)`` the DOFs each element's points read (padding 0).
        reference: ``(n_tables, n_q, 3, k)`` real: the reference field
            :math:`\hat E` of the basis at the points of each class (zero on
            the padding); a curved element's own table is its field itself.
        classes: ``(n_el,)`` the table of each element.
        jinv: ``(n_el, 3, 3)`` the inverse Jacobian :math:`J^{-1}` of each
            element (the identity where the element keeps its own table).
        weights: ``(n_el, n_q)`` the weights :math:`w_q |J|` of the points.
        points: ``(n_el * n_q, 3)`` the points' coordinates, file units.
    """

    dofs: np.ndarray
    reference: np.ndarray
    classes: np.ndarray
    jinv: np.ndarray
    weights: np.ndarray
    points: np.ndarray

    def _parts(self) -> Iterator[slice]:
        """Groups of elements whose expanded tables fit `_EVAL_BYTES`."""
        n_q, _, k = self.reference.shape[1:]
        step = max(1, _EVAL_BYTES // (8 * n_q * 3 * k))
        for lo in range(0, len(self.classes), step):
            yield slice(lo, lo + step)

    def _basis(self, part: slice) -> np.ndarray:
        r"""``(m, n_q, 3, k)`` the basis field :math:`J^{-\mathsf{T}}\hat E`."""
        table = self.reference[self.classes[part]]
        return np.einsum("eac,eqak->eqck", self.jinv[part], table)

    def load(self, p: np.ndarray, ndof: int) -> np.ndarray:
        r"""``(ndof,)`` complex: :math:`M^\mathsf{T} p`, ``p`` ``(n_el * n_q, 3)``."""
        p = np.asarray(p).reshape(len(self.classes), -1, 3)
        out = np.zeros(ndof, complex)
        for part in self._parts():
            local = np.einsum(
                "eq,eqc,eqck->ek", self.weights[part], p[part], self._basis(part)
            )
            dofs = self.dofs[part].ravel()
            out += np.bincount(dofs, np.real(local).ravel(), minlength=ndof)
            out += 1j * np.bincount(dofs, np.imag(local).ravel(), minlength=ndof)
        return out

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        r"""``(n_el * n_q, 3)`` complex: the weighted field :math:`M x`."""
        out = np.zeros((len(self.classes), self.weights.shape[1], 3), complex)
        for part in self._parts():
            out[part] = np.einsum(
                "eq,eqck,ek->eqc",
                self.weights[part],
                self._basis(part),
                x[self.dofs[part]],
            )
        return out.reshape(-1, 3)


@dataclasses.dataclass(frozen=True, eq=False)
class PumpColumns:
    r"""A pump field's values at the rule points, as a gather and an einsum.

    Attributes:
        dofs: ``(n_el, k)`` the DOFs each element's points read (padding 0).
        reference: ``(n_tables, n_q, 3, k)`` real: the reference field
            :math:`\hat E` at the points of the tables' classes, on the
            file-unit basis (zero on the padding).
        classes: ``(n_el,)`` the table of each element.
        jinv: ``(n_el, 3, 3)`` the inverse Jacobian :math:`J^{-1}` of each
            element (the identity where the element keeps its own table).
        points: ``(n_el * n_q, 3)`` the points' coordinates, file units.
        scale: The metres per file unit.
    """

    dofs: jax.Array
    reference: jax.Array
    classes: np.ndarray
    jinv: jax.Array
    points: np.ndarray
    scale: float

    def values(self, coefficients: Any) -> jax.Array:
        """``(n_el, n_q, 3)``: the field at the points, V/m for a field in V/m."""
        x = jnp.asarray(coefficients)[self.dofs]
        n_el = len(self.classes)
        n_q, _, k = self.reference.shape[1:]
        step = max(1, _EVAL_BYTES // (8 * n_q * 3 * k))
        out = []
        for lo in range(0, n_el, step):
            part = slice(lo, lo + step)
            out.append(
                _expand(self.reference, self.classes[part], self.jinv[part], x[part])
            )
        return jnp.concatenate(out) / self.scale


@jax.checkpoint
def _expand(
    reference: jax.Array, classes: np.ndarray, jinv: jax.Array, x: jax.Array
) -> jax.Array:
    """The field at the points of a group of elements: ``(m, n_q, 3)``.

    Checkpointed: the gathered tables are recomputed in the backward pass, so
    the residuals of a gradient are the shared tables and the inputs, not the
    expansion of every element (the memory of the dense weights).
    """
    e_ref = jnp.einsum("eqak,ek->eqa", reference[classes], x)
    return jnp.einsum("eac,eqa->eqc", jinv, e_ref)


def _local_dofs(space: ngs.FESpace, cells: np.ndarray) -> np.ndarray:
    """``(len(cells), k)`` the DOFs of each cell in the element's own order, -1 pad."""
    lists = [
        np.asarray(space.GetDofNrs(ngs.ElementId(ngs.VOL, int(c))), np.int64)
        for c in cells
    ]
    out = np.full((len(cells), max(1, *(len(d) for d in lists))), -1, np.int64)
    for i, d in enumerate(lists):
        out[i, : len(d)] = d
    return out


def _dense_weights(
    maps: list[sp.csr_matrix],
    w: np.ndarray,
    dofs: np.ndarray,
    n_q: int,
    ndof: int,
) -> np.ndarray:
    """``(n_el, n_q, 3, k)``: the maps of a group of cells on its elements' DOFs."""
    n_el, k = dofs.shape
    valid = dofs >= 0
    keys = (np.arange(n_el)[:, None] * ndof + dofs)[valid]
    slots = np.broadcast_to(np.arange(k), dofs.shape)[valid]
    order = np.argsort(keys)
    keys, slots = keys[order], slots[order]
    weights = np.zeros((n_el, n_q, 3, k))
    for c, matrix in enumerate(maps):
        coo = matrix.tocoo()
        el = coo.row // n_q
        key = el * ndof + coo.col
        at = np.searchsorted(keys, key)
        if len(key) and (at.max() >= len(keys) or (keys[at] != key).any()):
            raise RuntimeError(  # pragma: no cover - the DOFs are the element's
                "a rule point reads a DOF outside its element's DOFs."
            )
        np.add.at(weights, (el, coo.row % n_q, c, slots[at]), np.real(coo.data))
    # the maps hold the point's weight in each entry; the field is the value
    return weights / np.where(w > 0, w, 1.0).reshape(n_el, n_q, 1, 1)


def _straight(
    points: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""``(jac, straight)``: the Jacobians :math:`J` of straight elements, and which.

    ``points`` is ``(n_el, n_q, 3)``, the image of ``reference`` ``(n_q, 3)``
    under :math:`x = x_0 + J\hat x` for a straight tetrahedron. ``jac`` is
    ``(n_el, 3, 3)`` (the identity for the others) and ``straight`` ``(n_el,)``
    marks the elements whose points lie on their affine fit to within
    ``_AFFINE`` of their :math:`J` and whose :math:`J` is not singular.
    """
    basis = np.concatenate([np.ones((len(reference), 1)), reference], axis=1)
    pinv = np.linalg.pinv(basis)
    jac = np.broadcast_to(np.eye(3), (len(points), 3, 3)).copy()
    straight = np.zeros(len(points), bool)
    for lo in range(0, len(points), _FIT_ELEMENTS):
        xyz = points[lo : lo + _FIT_ELEMENTS]
        fit = np.einsum("jq,eqc->ejc", pinv, xyz)
        cand = fit[:, 1:, :].transpose(0, 2, 1)
        scale = np.abs(cand).max(axis=(1, 2))
        residual = np.abs(xyz - np.einsum("qj,ejc->eqc", basis, fit)).max(axis=(1, 2))
        det = np.linalg.det(cand)
        ok = (
            (scale > 0)
            & (residual <= _AFFINE * scale)
            & (np.abs(det) > _AFFINE * scale**3)
        )
        straight[lo : lo + _FIT_ELEMENTS] = ok
        jac[lo : lo + _FIT_ELEMENTS][ok] = cand[ok]
    return jac, straight


def _affine(points: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    r"""``(n_el, 3, 3)`` the Jacobians :math:`J` of straight elements, or ``None``."""
    jac, straight = _straight(points, reference)
    return jac if straight.all() else None


def _classes(mesh: ngs.Mesh, cells: np.ndarray, dofs: np.ndarray) -> np.ndarray:
    """``(n_el,)`` the class of each cell: its vertex order and its unused DOFs.

    The reference field of an element's basis depends on how the global
    numbers order its vertices (the orientation of its edges and faces), and
    on which of its DOFs are used.
    """
    numbers = np.array(
        [[v.nr for v in e.vertices] for e in mesh.Elements(ngs.VOL)], np.int64
    )[cells]
    key = np.concatenate([np.argsort(numbers, axis=1), dofs < 0], axis=1)
    return np.unique(key, axis=0, return_inverse=True)[1].reshape(-1)


def _tables(rule: _Rule, cells: np.ndarray) -> _Tables:
    """The `_Tables` of ``cells`` on ``rule``'s space, a few elements assembled each.

    A straight element is assembled only as a representative of its class (the
    first and the last of each; the two must agree to round-off), a curved
    element is assembled alone.
    """
    space, n_q, ndof = rule.space, rule.per_cell, int(rule.space.ndof)
    n_el = len(cells)
    points = rule.coordinates(cells)
    jac, straight = _straight(points.reshape(n_el, n_q, 3), rule.reference)
    dofs = _local_dofs(space, cells)
    tables: list[np.ndarray] = []
    classes = np.full(n_el, -1, np.int64)
    jinv = np.broadcast_to(np.eye(3), (n_el, 3, 3)).copy()
    weights = np.empty((n_el, n_q))
    shared = np.flatnonzero(straight)
    if len(shared):
        jinv[shared] = np.linalg.inv(jac[shared])
        weights[shared] = rule.weights * np.abs(np.linalg.det(jac[shared]))[:, None]
        label = _classes(space.mesh, cells[shared], dofs[shared])
        last = np.zeros(label.max() + 1, np.int64)
        last[label] = np.arange(len(shared))
        first = np.zeros_like(last)
        first[label[::-1]] = np.arange(len(shared))[::-1]
        picked = np.unique(np.concatenate([first, last]))
        at = shared[picked]
        table = np.empty((len(picked), n_q, 3, dofs.shape[1]))
        for lo, hi, maps, w in rule.chunks(cells[at], weights=True):
            dense = _dense_weights(maps, w, dofs[at[lo // n_q : hi // n_q]], n_q, ndof)
            table[lo // n_q : hi // n_q] = np.einsum(
                "eca,eqck->eqak", jac[at[lo // n_q : hi // n_q]], dense
            )
        row = {int(i): j for j, i in enumerate(picked)}
        top = float(np.abs(table).max()) or 1.0
        for c in range(len(first)):
            a, b = table[row[int(first[c])]], table[row[int(last[c])]]
            # the tables of one class agree to round-off
            if np.abs(a - b).max() > _AFFINE * top:  # pragma: no cover
                raise RuntimeError(
                    "two elements of one reference table differ by more than "
                    f"round-off ({np.abs(a - b).max():.2e})."
                )
            tables.append(a)
        classes[shared] = label
    own = np.flatnonzero(~straight)
    for lo, hi, maps, w in rule.chunks(cells[own], weights=True):
        e = own[lo // n_q : hi // n_q]
        dense = _dense_weights(maps, w, dofs[e], n_q, ndof)
        classes[e] = len(tables) + np.arange(len(e))
        tables.extend(dense)  # a curved element's table is its field
        weights[e] = w.reshape(len(e), n_q)
    k = dofs.shape[1]
    reference = np.stack(tables) if tables else np.zeros((0, n_q, 3, k))
    return _Tables(np.maximum(dofs, 0), reference, classes, jinv, weights, points)


def pump_columns(field: Any, elements: np.ndarray, order: int) -> PumpColumns:
    """The columns of a kept field at the points of rule order ``order``.

    ``elements`` are gfem volume elements, in increasing order.
    """
    built, space, _ = fe_space(
        field.mesh,
        field.support,
        field._excluded,  # noqa: SLF001
        "hcurl",
        field.order,
    )
    tables = _tables(_Rule(space, order), _cells(built, elements))
    return PumpColumns(
        jnp.asarray(tables.dofs),
        jnp.asarray(tables.reference),
        tables.classes,
        jnp.asarray(tables.jinv),
        tables.points,
        float(built.scale),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class VolumeLoads:
    r"""The polarization sources' load columns on one operator's space.

    Attributes:
        names: The sources' names, in column order.
        vectors: ``(N, n)`` complex dense: :math:`M^\mathsf{T} P` of each
            source on the solve space, zero on fixed DOFs; its load at
            :math:`\omega` is ``coefficients(omega)[k] * vectors[:, k]``.
        scale: The metres per file unit.
        blocks: Per source, its map :math:`M` to the points on the solve space
            (fixed DOFs' columns zeroed) as the shared tables of `_Blocks`
            (the maps of all points are too large to keep): the adjoint of
            the load, `cotangent`, the only reader.
    """

    names: tuple[str, ...]
    vectors: np.ndarray
    scale: float
    blocks: tuple[_Blocks, ...] = ()

    def coefficients(self, omega: float) -> np.ndarray:
        r"""``(n,)``: :math:`\mu_0 L^3\omega^2`."""
        return np.full(len(self.names), const.mu_0 * self.scale**3 * omega**2)

    def rhs(self, omega: float) -> np.ndarray:
        r"""The loads at :math:`\omega`, ``(N, n)``."""
        return self.vectors * self.coefficients(omega)[None, :]

    def __call__(self, omega: float) -> np.ndarray:
        """`rhs`: the loads of a source kind as `SourceColumns` reads them."""
        return self.rhs(omega)

    def cotangent(self, omega: float, adjoint: np.ndarray) -> list[np.ndarray]:
        r"""Per source, the complex cotangent :math:`\mu_0 L^3\omega^2 M_c\lambda`.

        ``adjoint`` is ``(N, n)``, the adjoint :math:`\lambda_k` of source k's
        column; each term is ``(n_points, 3)``, whose real part is the
        cotangent of :math:`\mathrm{Re}\,P` and whose negated imaginary part
        is that of :math:`\mathrm{Im}\,P`.
        """
        factor = self.coefficients(omega)
        out = []
        for k, blocks in enumerate(self.blocks):
            lam = np.asarray(adjoint)[:, k]
            out.append(factor[k] * blocks.evaluate(lam))
        return out

    def power_scale(self, omega: float) -> float:
        r"""Emitted power per :math:`-\operatorname{Im}(b^H x)`, in W."""
        return 1.0 / (2 * omega * const.mu_0 * self.scale)

    def emitted_power(self, omega: float, x: np.ndarray) -> np.ndarray:
        r"""``(n,)`` W: :math:`-\operatorname{Im}(b_k^H x_k) / (2\omega\mu_0 L)`.

        ``x`` ``(N, n)`` holds the sources' solutions, column k source k's.
        """
        pairs = np.sum(np.conj(self.rhs(omega)) * x, axis=0)
        return -np.imag(pairs) * self.power_scale(omega)


def volume_loads(
    operator: Any,
    sources: Sequence[PolarizationSource],
    *,
    order: int,
) -> VolumeLoads:
    """The sources' `VolumeLoads` on ``operator``'s solve space.

    The maps of the points are not kept, only the shared tables they are made
    of (`_Blocks`): the loads and their adjoint (`VolumeLoads.cotangent`) are
    gathers and einsums over the elements, a few at a time.

    Raises:
        ValueError: A source's points are not the points of this mesh's rule
            (built on another mesh or at other node positions), or the solve
            order is above the source's ``generated_order``.
        NotImplementedError: A source's rule order is above the largest
            NGSolve's rule space supports.
    """
    space = operator.space
    built = operator.built
    scale = float(built.scale)
    free = free_mask(space.FreeDofs(), space.ndof)
    columns = np.zeros((int(space.ndof), len(sources)), complex)
    # On a periodic cell the points read the unpaired space, and the load is
    # carried to the solve space by P^H (the maps by conj(P) on the right): the
    # test functions of the Bloch-phased space are conj(P) y, not P y.
    expansion = operator.expansion
    unpaired = space if expansion is None else expansion.space
    restrict = sp.csr_matrix(sp.diags(free.astype(float)))
    if expansion is not None:
        restrict = sp.csr_matrix(expansion.matrix.conj()) @ restrict
    blocks: list[_Blocks] = []
    rules: dict[int, _Rule] = {}
    for k, source in enumerate(sources):
        if order > source.generated_order:
            raise ValueError(
                f"The PolarizationSource {source.name!r} integrates its load "
                f"exactly for solve orders up to {source.generated_order}, but "
                f"the solve has order {order}: build it with "
                f"gfem.second_order_polarization(..., generated_order={order})."
            )
        cells = _cells(built, source.elements)
        if source.order not in rules:
            rules[source.order] = _Rule(unpaired, source.order)
        mine = _Blocks(_tables(rules[source.order], cells), restrict)
        _same_points(source, mine.tables.points)
        p = np.asarray(source.re) + 1j * np.asarray(source.im)
        # a traced value was not checked when the source was built
        if not np.isfinite(p).all():
            raise ValueError(
                f"The PolarizationSource {source.name!r} holds a non-finite "
                "polarization: check the pump's powers, phases and the tensor."
            )
        columns[:, k] = mine.load(p)
        blocks.append(mine)
    columns[~free] = 0.0
    return VolumeLoads(tuple(s.name for s in sources), columns, scale, tuple(blocks))


@dataclasses.dataclass(frozen=True, eq=False)
class _Blocks:
    r"""One source's map :math:`M` to its points on the solve space.

    ``restrict`` carries the solve space to the space the tables read:
    :math:`M` of the solve space is ``tables`` after ``restrict``.
    """

    tables: _Tables
    restrict: sp.csr_matrix

    def load(self, p: np.ndarray) -> np.ndarray:
        r"""``(N,)``: :math:`(M R)^\mathsf{T} p` for ``p`` ``(n_points, 3)``."""
        return self.restrict.T @ self.tables.load(p, self.restrict.shape[0])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        r"""``(n_points, 3)`` complex: :math:`M R x` for ``x`` ``(N,)``."""
        return self.tables.evaluate(self.restrict @ x)


def _same_points(source: PolarizationSource, here: np.ndarray) -> None:
    """Raise unless the rule's points are the source's."""
    extent = float(np.ptp(source.points, axis=0).max()) or 1.0
    if here.shape != source.points.shape or (
        np.abs(here - source.points).max() > _SAME_POINTS * extent
    ):
        raise ValueError(
            f"The PolarizationSource {source.name!r} was built on other points "
            "than this solve's rule: solve it on a model of the mesh (at the "
            "node positions) of the pump results it was built from."
        )
