r"""Polarization sources as load columns, and the fields of a pump at their points.

A `gfem.PolarizationSource` holds :math:`P` at the points of NGSolve's rule of
degree :math:`2Q` (an ``IntegrationRuleSpace`` of order :math:`Q`), element by
element, on the elements ``elements`` of the mesh. Two sparse maps of the
H(curl) coefficients to those points serve it:

- the pump's field at the points, :math:`E = (M x)/(w L)`, a gather of each
  element's DOFs and one einsum (`PumpColumns`, no sparse product, so that it
  runs on traced coefficients), with :math:`w` the weight of a point
  (:math:`\omega_q |J|`, file units cubed) and :math:`L` the metres per file
  unit: the kept coefficients are on the file-unit basis;
- the generated load, :math:`b = \mu_0 L^3\omega^2\,M^\mathsf{T} P` (`VolumeLoads`),
  in file units: :math:`P` and the volume are SI, so :math:`\int P\cdot\varphi
  \,\mathrm{d}V = L^2 M^\mathsf{T} P` on the file-unit basis, times the
  :math:`\mu_0 L\omega^2` of the driven equations (`gfem._engine.dipoles`). A
  mesh in um and the same mesh in m give the same :math:`P` and the same load.

The load is a fixed vector times a scalar of :math:`\omega` (affine), and the
emitted power is the dipole's, :math:`-\operatorname{Im}(b^H x)/(2\omega\mu_0 L)`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
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


@dataclasses.dataclass(frozen=True, eq=False)
class _Rule:
    """One order's rule points: the component maps, the weights, the coordinates."""

    maps: list[sp.csr_matrix]
    weights: np.ndarray
    coordinates: np.ndarray
    per_cell: int


def _coo(form: Any, shape: tuple[int, int]) -> sp.csr_matrix:
    form.Assemble()
    rows, cols, vals = form.mat.COO()
    return sp.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=shape
    )


def _rule(space: ngs.FESpace, order: int, cells: np.ndarray) -> _Rule:
    """The maps of ``space``'s coefficients to the points of the cells ``cells``.

    ``cells`` are NGSolve volume elements; the maps and the weights are zero
    elsewhere. The map of component ``c`` has the weight of each point in its
    entries (the point's test function of the rule space is the point value
    times the weight :math:`w_q |J|` of the rule).
    """
    mesh = space.mesh
    rule = ngs.IntegrationRule(ngs.ET.TET, 2 * order)
    with serial_spaces():
        irs = ngs.comp.IntegrationRuleSpace(mesh, order=order, complex=space.is_complex)
    per_cell = len(rule.points)
    if irs.ndof != int(mesh.ne) * per_cell:  # pragma: no cover - NGSolve contract
        raise RuntimeError(
            f"the integration rule space holds {irs.ndof} points, not {per_cell} "
            f"per element on {int(mesh.ne)} elements."
        )
    bits = ngs.BitArray(int(mesh.ne))
    bits.Clear()
    for c in cells:
        bits.Set(int(c))
    dx = ngs.dx(intrules={ngs.ET.TET: rule}, definedonelements=bits)
    v = irs.TestFunction()
    u = space.TrialFunction()
    maps = []
    for c in range(3):
        form = ngs.BilinearForm(space, irs)
        form += u[c] * v * dx
        maps.append(_coo(form, (irs.ndof, space.ndof)))
    mass = ngs.BilinearForm(irs)
    mass += irs.TrialFunction() * v * dx
    w = _coo(mass, (irs.ndof, irs.ndof)).diagonal()
    xyz = []
    for coordinate in (ngs.x, ngs.y, ngs.z):
        gf = ngs.GridFunction(ngs.comp.IntegrationRuleSpace(mesh, order=order))
        gf.Set(coordinate)
        xyz.append(np.array(gf.vec.FV().NumPy()))
    return _Rule(maps, np.real(w), np.stack(xyz, axis=1), per_cell)


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
    """A pump field's values at the rule points, as a gather and an einsum.

    Attributes:
        dofs: ``(n_el, k)`` the DOFs each element's points read (padding 0).
        weights: ``(n_el, n_q, 3, k)`` real: component ``c`` at point ``q`` of
            element ``e`` is ``weights[e, q, c] @ x[dofs[e]]``, on the
            file-unit basis (zero on the padding).
        points: ``(n_el * n_q, 3)`` the points' coordinates, file units.
        scale: The metres per file unit.
    """

    dofs: jax.Array
    weights: jax.Array
    points: np.ndarray
    scale: float

    def values(self, coefficients: Any) -> jax.Array:
        """``(n_el, n_q, 3)``: the field at the points, V/m for a field in V/m."""
        x = jnp.asarray(coefficients)[self.dofs]
        return jnp.einsum("eqck,ek->eqc", self.weights, x) / self.scale


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
    rule = _rule(space, order, cells)
    rows = _rows(cells, rule.per_cell)
    n_el, n_q = len(cells), rule.per_cell
    ndof = int(space.ndof)
    parts = []
    for c, matrix in enumerate(rule.maps):
        coo = matrix[rows].tocoo()
        parts.append((c, coo.row // n_q, coo.row % n_q, coo.col, coo.data, coo.row))
    keys = np.unique(
        np.concatenate(
            [el.astype(np.int64) * ndof + col for _, el, _, col, _, _ in parts]
        )
        if parts
        else np.zeros(0, np.int64)
    )
    owner, col_u = keys // ndof, keys % ndof
    first = np.searchsorted(owner, np.arange(n_el))
    slot = np.arange(len(keys)) - first[owner]
    k = int(slot.max()) + 1 if len(keys) else 1
    dofs = np.zeros((n_el, k), np.int64)
    dofs[owner, slot] = col_u
    weights = np.zeros((n_el, n_q, 3, k))
    w = rule.weights[rows]
    for c, el, q, col, data, row in parts:
        at = np.searchsorted(keys, el.astype(np.int64) * ndof + col)
        np.add.at(weights, (el, q, c, slot[at]), np.real(data) / w[row])
    return PumpColumns(
        jnp.asarray(dofs),
        jnp.asarray(weights),
        rule.coordinates[rows],
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
    """

    names: tuple[str, ...]
    vectors: np.ndarray
    scale: float

    def coefficients(self, omega: float) -> np.ndarray:
        r"""``(n,)``: :math:`\mu_0 L^3\omega^2`."""
        return np.full(len(self.names), const.mu_0 * self.scale**3 * omega**2)

    def rhs(self, omega: float) -> np.ndarray:
        """The loads at ω, ``(N, n)``."""
        return self.vectors * self.coefficients(omega)[None, :]

    def __call__(self, omega: float) -> np.ndarray:
        """`rhs`: the loads of a source kind as `SourceColumns` reads them."""
        return self.rhs(omega)

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
    operator: Any, sources: Sequence[PolarizationSource], *, order: int
) -> VolumeLoads:
    """The sources' `VolumeLoads` on ``operator``'s solve space.

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
    cache: dict[tuple[int, bytes], tuple[_Rule, np.ndarray]] = {}
    for k, source in enumerate(sources):
        if order > source.generated_order:
            raise ValueError(
                f"The PolarizationSource {source.name!r} integrates its load "
                f"exactly for solve orders up to {source.generated_order}, but "
                f"the solve has order {order}: build it with "
                f"gfem.second_order_polarization(..., generated_order={order})."
            )
        cells = _cells(built, source.elements)
        key = (source.order, cells.tobytes())
        if key not in cache:
            cache[key] = (_rule(space, source.order, cells), cells)
        rule, _ = cache[key]
        rows = _rows(cells, rule.per_cell)
        _same_points(source, rule.coordinates[rows])
        p = np.asarray(source.re) + 1j * np.asarray(source.im)
        for c, matrix in enumerate(rule.maps):
            columns[:, k] += matrix[rows].T @ p[:, c]
    columns[~free] = 0.0
    return VolumeLoads(tuple(s.name for s in sources), columns, scale)


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
