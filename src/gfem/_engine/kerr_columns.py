r"""The quadrature columns of a port mode's field, for the Kerr overlaps.

NGSolve evaluates the mode at the points of a degree-:math:`4p` rule per
element of the port's section as a sparse map of the full DOF vector;
`gfem.kerr` forms the quartic integrand in JAX on those columns.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._engine.runtime import serial_spaces

__all__ = ["Columns", "columns"]


@dataclasses.dataclass(frozen=True, eq=False)
class Columns:
    """The mode field at the points of a degree-4p rule, per element of the section.

    ``dofs (n_el, k)`` are the DOF numbers of each element's tangential
    (Nedelec) part, ``weights (n_el, n_q, n_c, k)`` real, so that component
    ``c`` at point ``q`` of element ``e`` is ``weights[e, q, c] @ x[dofs[e]]``
    (``n_c`` = 2 for ``(E_u, E_v)`` of a planar port, 1 for the scalar of a
    line port). A planar port's normal component ``E_n`` lives on the other
    DOFs of the element: ``normal_dofs (n_el, k_n)`` and ``normal_weights
    (n_el, n_q, k_n)``, ``None`` on a line. Both are jax arrays, put on the
    device once. ``measure (n_el, n_q)`` is the weight of each point in m² (m
    on a line). Elements are in the order of the section's triangles (segments).
    ``points (n_el, n_q, 2)`` are the points' plane coordinates in the section's
    local units, read by a bent mode's metric factor: ``None`` on a line.
    """

    dofs: jax.Array
    weights: jax.Array
    normal_dofs: jax.Array | None
    normal_weights: jax.Array | None
    measure: np.ndarray
    points: np.ndarray | None = None


def _coo(form: Any, shape: tuple[int, int]) -> sp.csr_matrix:
    form.Assemble()
    rows, cols, vals = form.mat.COO()
    return sp.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=shape
    )


def _point_maps(
    space: Any, raised: int
) -> tuple[list[sp.csr_matrix], np.ndarray, np.ndarray]:
    """The component maps of the DOF vector to the rule's points, and their weights.

    Returns the sparse ``(n_points, n_dof)`` map of each component, the weight
    of each point (local units squared) and ``point_dofs (n_el, n_q)``, the
    point numbers of each element. ``raised`` adds to the order of the
    integration rule space (a test's check that the rule is exact).
    """
    planar = len(space.section.normal) == 3
    q = 2 * space.order + raised
    mesh = space.mesh
    with serial_spaces():
        irs = ngs.comp.IntegrationRuleSpace(mesh, order=q, complex=True)
    cell = ngs.ET.TRIG if planar else ngs.ET.SEGM
    rule = ngs.IntegrationRule(cell, 2 * q)
    n_el, n_q = int(mesh.ne), len(rule.points)
    if irs.ndof != n_el * n_q:  # pragma: no cover - guards the NGSolve contract
        raise RuntimeError(
            f"the integration rule space holds {irs.ndof} points, not {n_q} per "
            f"element on {n_el} elements."
        )
    dx = ngs.dx(intrules={cell: rule})
    v = irs.TestFunction()
    if planar:
        et, en = space.space.TrialFunction()
        comps = (et[0], et[1], en)
    else:
        comps = (space.space.TrialFunction(),)
    mats = []
    for comp in comps:
        form = ngs.BilinearForm(space.space, irs)
        form += comp * v * dx
        mats.append(_coo(form, (irs.ndof, space.space.ndof)))
    mass = ngs.BilinearForm(irs)
    mass += irs.TrialFunction() * v * dx
    w = _coo(mass, (irs.ndof, irs.ndof)).diagonal()
    if np.abs(w.imag).max() > 1e-14 * np.abs(w.real).max():  # pragma: no cover
        raise RuntimeError("the quadrature weights of the mode space are complex.")
    point_dofs = np.array(
        [list(irs.GetDofNrs(ngs.ElementId(ngs.VOL, e))) for e in range(n_el)], int
    )
    return mats, w.real, point_dofs


def _plane_points(space: Any, point_dofs: np.ndarray, raised: int) -> np.ndarray:
    """The plane coordinates of the rule's points, ``(n_el, n_q, 2)``.

    The two coordinates are P1 fields mapped to the points like the mode's
    components, so a point lies where the rule puts it whatever the element's
    vertex order.
    """
    q = 2 * space.order + raised
    mesh = space.mesh
    h1 = ngs.H1(mesh, order=1)
    with serial_spaces():
        irs = ngs.comp.IntegrationRuleSpace(mesh, order=q, complex=True)
    cell = ngs.ET.TRIG
    dx = ngs.dx(intrules={cell: ngs.IntegrationRule(cell, 2 * q)})
    form = ngs.BilinearForm(h1, irs)
    form += h1.TrialFunction() * irs.TestFunction() * dx
    mass = ngs.BilinearForm(irs)
    mass += irs.TrialFunction() * irs.TestFunction() * dx
    w = _coo(mass, (irs.ndof, irs.ndof)).diagonal()
    at_vertices = np.array([v.point for v in mesh.vertices])[:, :2]
    nodal = _coo(form, (irs.ndof, h1.ndof)) @ at_vertices / w[:, None]
    return nodal.real[point_dofs]


def columns(space: Any, raised: int = 0) -> Columns:
    r"""Build the columns of ``space`` (a port mode space), cached on it.

    NGSolve evaluates the field at the points of an integration rule as a
    sparse map of the full DOF vector: an ``IntegrationRuleSpace`` holds one
    DOF per point, a mixed bilinear form per component maps the field into
    it, and the space's own mass (a diagonal, :math:`w_q |J|`) divides the
    weights out. The rule of degree :math:`4p` integrates the quartic
    integrand exactly for fields of degree :math:`p` and a material constant
    per element on straight elements. The rule is named on the form: a line
    mesh's ``GetIntegrationRules()`` has no segment key and the default rule
    gives wrong point values.
    """
    cache = space.__dict__.setdefault("_kerr_columns", {})
    if raised in cache:
        return cache[raised]
    section = space.section
    planar = len(section.normal) == 3
    mats, w, point_dofs = _point_maps(space, raised)
    n_el, n_q = point_dofs.shape
    n_dof = space.space.ndof
    owner = [
        list(space.space.GetDofNrs(ngs.ElementId(ngs.VOL, e))) for e in range(n_el)
    ]
    k = max(len(d) for d in owner)
    dofs = np.full((n_el, k), -1, np.int64)
    for e, d in enumerate(owner):
        dofs[e, : len(d)] = d
    row_el = np.empty(len(w), int)
    row_q = np.empty(len(w), int)
    row_el[point_dofs] = np.arange(n_el)[:, None]
    row_q[point_dofs] = np.arange(n_q)[None, :]
    # slot of a (element, DOF) pair among the element's DOFs
    keys = (np.arange(n_el)[:, None] * n_dof + dofs)[dofs >= 0]
    slots = np.broadcast_to(np.arange(k), dofs.shape)[dofs >= 0]
    by_key = np.argsort(keys)
    sorted_keys, sorted_slots = keys[by_key], slots[by_key]
    weights = np.zeros((n_el, n_q, len(mats), k))
    for c, mat in enumerate(mats):
        coo = mat.tocoo()
        el, pt = row_el[coo.row], row_q[coo.row]
        key = el * n_dof + coo.col
        at = np.minimum(np.searchsorted(sorted_keys, key), len(sorted_keys) - 1)
        hit = sorted_keys[at] == key
        value = coo.data / w[coo.row]
        if not hit.all() and np.abs(value[~hit]).max() > 0:  # pragma: no cover
            raise RuntimeError(
                "a quadrature point couples to a DOF of another element."
            )
        value = value[hit]
        if np.abs(value.imag).max(initial=0.0) > 1e-14 * np.abs(value.real).max(
            initial=0.0
        ):  # pragma: no cover - guards the NGSolve contract
            raise RuntimeError("the field's quadrature columns are complex.")
        np.add.at(weights, (el[hit], pt[hit], c, sorted_slots[at[hit]]), value.real)
    measure = np.zeros((n_el, n_q))
    measure[row_el, row_q] = w
    measure *= section.scale**2 if planar else section.scale
    points = _plane_points(space, point_dofs, raised) if planar else None
    dofs = np.maximum(dofs, 0)
    if not planar:
        # NGSolve elements to section segments
        order = np.empty(n_el, int)
        order[np.asarray(space.element_segment)] = np.arange(n_el)
        dofs, weights, measure = dofs[order], weights[order], measure[order]
    if planar:
        # the tangential and the normal component read disjoint slots of the
        # element: keep each its own, without the zeros of the other
        on_t = (weights[:, :, :2] != 0).any(axis=(0, 1, 2))
        on_n = (weights[:, :, 2] != 0).any(axis=(0, 1))
        cols = Columns(
            jnp.asarray(dofs[:, on_t]),
            jnp.asarray(weights[:, :, :2][..., on_t]),
            jnp.asarray(dofs[:, on_n]),
            jnp.asarray(weights[:, :, 2][..., on_n]),
            measure,
            points,
        )
    else:
        cols = Columns(jnp.asarray(dofs), jnp.asarray(weights), None, None, measure)
    cache[raised] = cols
    return cache[raised]
