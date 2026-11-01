"""NumPy assembly of the boundary morph's mesh extension.

The extension maps prescribed displacements of the moving nodes to every
node, by solving a stiffened elasticity (or component-wise Laplace) problem
on the imported P1 mesh. Unknowns are restricted to the directions each node
may move in: free nodes move in 3D, nodes on a sliding surface move within
its plane (or along the line where two planes meet), held nodes not at all.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

#: Poisson ratio of the elastic extension (Lamé mu = 1, lambda = 1.5).
POISSON = 0.3
_MU = 1.0
_LAMBDA = 2 * _MU * POISSON / (1 - 2 * POISSON)

#: Relative singular-value threshold for distinct sliding-surface normals.
_RANK_TOL = 1e-8


def p1_gradients(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of the four P1 hat functions per tet, and the tet volumes.

    Returns:
        ``(g, vol)``: ``g[e, i]`` is the gradient of local hat function ``i``
        on tet ``e`` (shape ``(n_tet, 4, 3)``), ``vol`` the unsigned volume.
    """
    edges = nodes[tets[:, 1:]] - nodes[tets[:, :1]]  # (n_tet, 3, 3), rows = edges
    inv = np.linalg.inv(edges)  # columns are the gradients of hats 1..3
    g = np.empty((len(tets), 4, 3))
    g[:, 1:, :] = np.transpose(inv, (0, 2, 1))
    g[:, 0, :] = -g[:, 1:, :].sum(axis=1)
    return g, np.abs(np.linalg.det(edges)) / 6


def stiffness(nodes: np.ndarray, tets: np.ndarray, kind: str) -> sp.csr_array:
    """The stiffened extension operator on all ``3 * n_nodes`` coordinates.

    Each element's modulus scales as ``V_min / V_e`` (Jacobian-based
    stiffening, exponent 1), so small elements, which sit next to the
    geometry's features, resist deformation most.

    Args:
        nodes: Imported node coordinates.
        tets: Tetrahedra.
        kind: ``"elastic"`` (linear elasticity) or ``"laplace"``
            (component-wise harmonic extension).
    """
    g, vol = p1_gradients(nodes, tets)
    weight = vol.min()  # = (V_min / V_e) * V_e
    gg = np.einsum("eid,ejd->eij", g, g)
    eye = np.eye(3)
    if kind == "laplace":
        ke = np.einsum("eij,ab->eiajb", gg, eye)
    else:
        ke = (
            _MU * np.einsum("eij,ab->eiajb", gg, eye)
            + _MU * np.einsum("eib,eja->eiajb", g, g)
            + _LAMBDA * np.einsum("eia,ejb->eiajb", g, g)
        )
    ke = (weight * ke).reshape(len(tets), 12, 12)
    dofs = (3 * tets[:, :, None] + np.arange(3)).reshape(len(tets), 12)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    n = 3 * len(nodes)
    return sp.csr_array((ke.ravel(), (rows, cols)), shape=(n, n))


def unit_normals(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Unit normal of each triangle (right-hand rule on its node order)."""
    a, b, c = (nodes[tris[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def free_directions(
    n_nodes: int,
    slide_tris: np.ndarray,
    slide_normals: np.ndarray,
    constrained: np.ndarray,
    ties: np.ndarray | None = None,
) -> sp.csr_array:
    """The map from free coordinates to node displacements, ``(3 n_nodes, k)``.

    Nodes flagged in ``constrained`` (held or prescribed) get no free
    coordinate. A node on sliding triangles keeps the orthogonal complement of
    their normals: a plane, a line, or nothing. Every other node moves freely.

    ``ties`` (`gfem._periodic.tie_classes`, the representative of each
    node) makes each tie class move as one: it is constrained if any member
    is, its sliding constraint stacks the normals of every member's sliding
    triangles, and its columns are written into every member's rows.
    """
    ties = np.arange(n_nodes) if ties is None else np.asarray(ties)
    class_constrained = np.zeros(n_nodes, dtype=bool)
    np.logical_or.at(class_constrained, ties, constrained)
    # the representatives of the free classes
    free = np.flatnonzero(~class_constrained & (ties == np.arange(n_nodes)))
    on_slide = np.zeros(n_nodes, dtype=bool)
    on_slide[ties[slide_tris.ravel()]] = True
    # Interior classes: three unit columns each.
    interior = free[~on_slide[free]]
    # Sliding classes: group the incident triangle normals per class.
    corner_node = ties[slide_tris.ravel()]
    corner_normal = np.repeat(slide_normals, 3, axis=0)
    order = np.argsort(corner_node, kind="stable")
    corner_node, corner_normal = corner_node[order], corner_normal[order]
    starts = np.searchsorted(corner_node, np.arange(n_nodes + 1))
    basis: dict[int, np.ndarray] = {}
    for v in free[on_slide[free]]:
        normals = corner_normal[starts[v] : starts[v + 1]]
        _, s, vt = np.linalg.svd(normals)
        rank = int((s > _RANK_TOL * s[0]).sum())
        if rank < 3:
            basis[int(v)] = vt[rank:].T  # (3, 3 - rank), orthonormal
    # the members of each class, by representative
    by_class = np.argsort(ties, kind="stable")
    first = np.searchsorted(ties[by_class], np.arange(n_nodes + 1))
    base = np.full(n_nodes, -1)
    base[interior] = 3 * np.arange(len(interior))
    members = np.flatnonzero(base[ties] >= 0)
    rows = [(3 * members[:, None] + np.arange(3)).ravel()]
    cols = [(base[ties[members]][:, None] + np.arange(3)).ravel()]
    vals = [np.ones(3 * len(members))]
    k = 3 * len(interior)
    for v, b in basis.items():
        for m in by_class[first[v] : first[v + 1]]:
            rows.append(np.repeat(3 * m + np.arange(3), b.shape[1]))
            cols.append(np.tile(k + np.arange(b.shape[1]), 3))
            vals.append(b.ravel())
        k += b.shape[1]
    return sp.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n_nodes, k),
    )


def prescribed(
    n_nodes: int,
    nodes: np.ndarray,
    normals: np.ndarray,
    ties: np.ndarray | None = None,
) -> sp.csr_array:
    """The map from normal displacements to node displacements, ``(3 n, p)``.

    With ``ties`` (the representative of each node, ``nodes`` being
    representatives) parameter p moves every member of its node's tie class
    by ``normals[p]``.
    """
    p = len(nodes)
    if ties is None:
        members, column = np.asarray(nodes), np.arange(p)
    else:
        param = np.full(n_nodes, -1)
        param[nodes] = np.arange(p)
        members = np.flatnonzero(param[ties] >= 0)
        column = param[ties[members]]
    rows = (3 * members[:, None] + np.arange(3)).ravel()
    cols = np.repeat(column, 3)
    values = normals[column].ravel()
    return sp.csr_array((values, (rows, cols)), shape=(3 * n_nodes, p))
