r"""Polarization sources as load columns, and the fields of a pump at their points.

A `gfem.PolarizationSource` holds :math:`P` at the points of NGSolve's rule of
degree :math:`2Q` (an ``IntegrationRuleSpace`` of order :math:`Q`), element by
element, on the elements ``elements`` of the mesh. Two maps of the H(curl)
coefficients to those points serve it, both assembled a few elements at a
time (`_Rule.chunks`): the maps of a rule of degree 10 on an order-3 space
hold some :math:`3\times10^4` entries per element, tens of GiB on
:math:`10^5` elements.

- the pump's field at the points, :math:`E = (M x)/(w L)`, a gather of each
  element's DOFs and one einsum (`PumpColumns`, no sparse product, so that it
  runs on traced coefficients), with :math:`w` the weight of a point
  (:math:`\omega_q |J|`, file units cubed) and :math:`L` the metres per file
  unit: the kept coefficients are on the file-unit basis. On a straight
  tetrahedron the field is the covariant Piola image
  :math:`E = J^{-\mathsf{T}}\hat E` of the reference field :math:`\hat E` of
  the element's basis, which depends on the order of its vertices only: a
  handful of reference tables serve every element, and `PumpColumns` keeps
  them with each element's :math:`J^{-1}` (curved elements keep their own
  table);
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
import hashlib
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
#: Elements whose reference tables `PumpColumns.values` expands at once, so that
#: the expansion is a bounded fraction of the memory of the dense weights it
#: replaces: about 60 MB.
_EVAL_BYTES = 6 * 10**7
#: Relative distance, of the reference tables of two elements of one class and
#: of a point to its affine fit: round-off of sums of up to a few hundred terms.
_AFFINE = 1e-8
#: Decimals a reference table, relative to its largest entry, is rounded to
#: before it is compared with the others (the tables of one class agree to
#: round-off).
_DECIMALS = 8


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
        self.space = space
        self.order = order
        rule = ngs.IntegrationRule(ngs.ET.TET, 2 * order)
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
        out = np.empty((len(rows), 3))
        gf = ngs.GridFunction(
            ngs.comp.IntegrationRuleSpace(self.space.mesh, order=self.order)
        )
        for j, coordinate in enumerate((ngs.x, ngs.y, ngs.z)):
            gf.Set(coordinate)
            out[:, j] = np.array(gf.vec.FV().NumPy())[rows]
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
            table = self.reference[self.classes[part]]
            e_ref = jnp.einsum("eqak,ek->eqa", table, x[part])
            out.append(jnp.einsum("eac,eqa->eqc", self.jinv[part], e_ref))
        return jnp.concatenate(out) / self.scale


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


def _affine(points: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    r"""``(n_el, 3, 3)`` the Jacobians :math:`J` of straight elements, or ``None``.

    ``points`` is ``(n_el, n_q, 3)``, the image of ``reference`` ``(n_q, 3)``
    under :math:`x = x_0 + J\hat x` for a straight tetrahedron.
    """
    basis = np.concatenate([np.ones((len(reference), 1)), reference], axis=1)
    fit = np.einsum("jq,eqc->ejc", np.linalg.pinv(basis), points)
    jac = fit[:, 1:, :].transpose(0, 2, 1)
    scale = float(np.abs(jac).max())
    residual = np.abs(points - np.einsum("qj,ejc->eqc", basis, fit)).max()
    if scale == 0 or residual > _AFFINE * scale:
        return None
    det = np.linalg.det(jac)
    if (np.abs(det) <= _AFFINE * scale**3).any():
        return None
    return jac


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
    cells = _cells(built, elements)
    rule = _Rule(space, order)
    n_q, ndof = rule.per_cell, int(space.ndof)
    points = rule.coordinates(cells)
    tables: dict[bytes, int] = {}
    reference: list[np.ndarray] = []
    classes, dof_blocks, jinv_blocks = [], [], []
    for lo, hi, maps, w in rule.chunks(cells, weights=True):
        group = cells[lo // n_q : hi // n_q]
        dofs = _local_dofs(space, group)
        weights = _dense_weights(maps, w, dofs, n_q, ndof)
        xyz = points[lo:hi].reshape(len(group), n_q, 3)
        jac = _affine(xyz, rule.reference)
        if jac is None:  # curved: every element is its own table
            table, jinv = weights, np.broadcast_to(np.eye(3), (len(group), 3, 3))
        else:
            table = np.einsum("eca,eqck->eqak", jac, weights)
            jinv = np.linalg.inv(jac)
        ids = np.empty(len(group), np.int64)
        for i, t in enumerate(table):
            top = float(np.abs(t).max()) or 1.0
            key = hashlib.blake2b(
                np.ascontiguousarray(np.round(t / top, _DECIMALS) + 0.0).tobytes()
                + repr(t.shape).encode(),
                digest_size=16,
            ).digest()
            if key not in tables:
                tables[key] = len(reference)
                reference.append(t)
            ids[i] = tables[key]
        # the tables of one class agree to round-off: a class is one table
        mismatch = max(
            (np.abs(reference[j] - t).max() for j, t in zip(ids, table, strict=True)),
            default=0.0,
        )
        if mismatch > _AFFINE * float(np.abs(table).max()):
            raise RuntimeError(  # pragma: no cover - two tables, one rounding
                "two elements of one reference table differ by more than "
                f"round-off ({mismatch:.2e})."
            )
        classes.append(ids)
        dof_blocks.append(dofs)
        jinv_blocks.append(jinv)
    k = max(d.shape[1] for d in dof_blocks)
    dofs = np.concatenate(
        [
            np.pad(d, ((0, 0), (0, k - d.shape[1])), constant_values=-1)
            for d in dof_blocks
        ]
    )
    stack = np.stack(
        [np.pad(t, ((0, 0),) * 2 + ((0, k - t.shape[2]),)) for t in reference]
    )
    return PumpColumns(
        jnp.asarray(np.maximum(dofs, 0)),
        jnp.asarray(stack),
        np.concatenate(classes),
        jnp.asarray(np.concatenate(jinv_blocks)),
        points,
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
        blocks: Per source, its three component maps :math:`M_c`, ``(n_points,
            N)`` sparse with the fixed DOFs' columns zeroed, as the groups of
            points `_Blocks` assembles on demand (the maps of all points are
            too large to keep): the adjoint of the load, `cotangent`, the
            only reader.
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
            value = np.zeros((blocks.n_points, 3), complex)
            for lo, hi, maps in blocks.chunks():
                value[lo:hi] = np.stack([m @ lam for m in maps], axis=1)
            out.append(factor[k] * value)
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

    The maps of the points are not kept: the adjoint of the loads
    (`VolumeLoads.cotangent`) assembles them again, a pass as long as the one
    that builds the loads.

    Raises:
        ValueError: A source's points are not the points of this mesh's rule
            (built on another mesh or at other node positions), or the solve
            order is above the source's ``generated_order``.
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
    restrict = sp.diags(free.astype(float))
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
        rule = rules[source.order]
        _same_points(source, rule.coordinates(cells))
        p = np.asarray(source.re) + 1j * np.asarray(source.im)
        # a traced value was not checked when the source was built
        if not np.isfinite(p).all():
            raise ValueError(
                f"The PolarizationSource {source.name!r} holds a non-finite "
                "polarization: check the pump's powers, phases and the tensor."
            )
        mine = _Blocks(rule, cells, restrict)
        for lo, hi, maps in mine.chunks():
            for c, matrix in enumerate(maps):
                columns[:, k] += matrix.T @ p[lo:hi, c]
        blocks.append(mine)
    columns[~free] = 0.0
    return VolumeLoads(tuple(s.name for s in sources), columns, scale, tuple(blocks))


@dataclasses.dataclass(frozen=True, eq=False)
class _Blocks:
    """One source's component maps on the solve space, a group of cells at a time."""

    rule: _Rule
    cells: np.ndarray
    restrict: sp.spmatrix

    @property
    def n_points(self) -> int:
        return len(self.cells) * self.rule.per_cell

    def chunks(self) -> Iterator[tuple[int, int, list[sp.csr_matrix]]]:
        """``(lo, hi, maps)``: the maps of the points ``lo:hi``, ``(hi - lo, N)``."""
        for lo, hi, maps, _ in self.rule.chunks(self.cells):
            yield lo, hi, [(m @ self.restrict).tocsr() for m in maps]


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
