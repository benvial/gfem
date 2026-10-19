"""Engine kernels behind `gfem.Field`: its NGSolve space, sampling, curl.

A field's coefficients are DOF values of a compressed NGSolve space on the
mesh in its file length unit, ``definedon`` the support minus the excluded
volume entities. That numbering depends only on the mesh topology, the
support, the space and the order, so the space can be rebuilt from the
field alone. The engine value of a field times ``scale ** LENGTH_POWER``
(``scale`` in metres per file unit) is the SI value: H(curl) values are
covariant (1/length), H(div) values contravariant (1/area).
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import ngsolve as ngs
import numpy as np

from gfem._engine.cache import discretization
from gfem._engine.coefficients import p0
from gfem._engine.mesh import EngineMesh
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field

__all__ = ["LENGTH_POWER", "curl_matrix", "fe_space", "sample", "support_region"]

LENGTH_POWER = {"h1": 0, "l2": 0, "hcurl": -1, "hdiv": -2}


def support_region(
    built: EngineMesh,
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
) -> ngs.Region:
    """The NGSolve volume region of ``support`` minus ``excluded`` entities."""
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    wanted, out = set(support), set(excluded)
    bits = ngs.BitArray(len(built.domains))
    bits.Clear()
    for k, (tag, entity) in enumerate(built.domains):
        if names[tag] in wanted and (tag, entity) not in out:
            bits.Set(k)
    return ngs.Region(built.mesh, ngs.VOL, bits)


def _build(region: ngs.Region, space: str, order: int) -> ngs.FESpace:
    mesh = region.mesh
    if space == "h1":
        fes = ngs.H1(mesh, order=order, definedon=region)
    elif space == "hcurl":
        fes = ngs.HCurl(mesh, order=order, type1=True, definedon=region)
    elif space == "hdiv":
        fes = ngs.HDiv(mesh, order=order, definedon=region)
    elif space == "l2":
        fes = ngs.L2(mesh, order=order, definedon=region)
    else:  # pragma: no cover - Field validates the space
        raise ValueError(f"Unknown space {space!r}.")
    return ngs.Compress(fes)


def fe_space(
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
    space: str,
    order: int,
) -> tuple[EngineMesh, ngs.FESpace, ngs.Region]:
    """The real NGSolve space of this layout on ``mesh`` at its node positions.

    Its DOF numbering is that of every compressed space of the same kind
    and order on the same region, real or complex, so solves and fields
    agree.
    """
    disc = discretization(mesh)
    built = disc.built
    key = (("field", tuple(support), tuple(excluded)), space, order)

    def build():
        region = support_region(built, mesh, support, excluded)
        return _build(region, space, order), region

    fes, region = disc.space(key, build)
    return built, fes, region


def _check_size(field: Field, fes: ngs.FESpace) -> None:
    if fes.ndof != field.coefficients.shape[0]:
        raise ValueError(
            f"The field has {field.coefficients.shape[0]} coefficients but its "
            f"{field.space} order-{field.order} space on {list(field.support)} has "
            f"{fes.ndof} DOFs: the field does not belong to this mesh."
        )


# --- sampling ------------------------------------------------------------------


def _lattice(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Points and sub-tetrahedra of the reference tetrahedron cut ``n`` times.

    Returns the barycentric lattice points ``(i, j, k) / n`` (reference
    coordinates, ``i + j + k <= n``) and ``n**3`` positively oriented
    sub-tetrahedra as point indices.
    """
    ijk = [
        (i, j, k)
        for k in range(n + 1)
        for j in range(n + 1 - k)
        for i in range(n + 1 - k - j)
    ]
    index = {p: m for m, p in enumerate(ijk)}

    def at(i, j, k):
        return index[i, j, k]

    tets = []
    for i, j, k in ijk:
        s = i + j + k
        if s <= n - 1:
            corner = [at(i, j, k), at(i + 1, j, k), at(i, j + 1, k), at(i, j, k + 1)]
            tets.append(corner)
        if s <= n - 2:
            # the octahedron between the corner tetrahedra, cut along one diagonal
            a, b = at(i + 1, j, k), at(i, j + 1, k + 1)
            ring = [
                at(i, j + 1, k),
                at(i + 1, j + 1, k),
                at(i + 1, j, k + 1),
                at(i, j, k + 1),
            ]
            tets += [[a, b, ring[m], ring[(m + 1) % 4]] for m in range(4)]
        if s <= n - 3:
            tets.append(
                [
                    at(i + 1, j + 1, k),
                    at(i + 1, j, k + 1),
                    at(i, j + 1, k + 1),
                    at(i + 1, j + 1, k + 1),
                ]
            )
    points = np.array(ijk, dtype=float) / n
    cells = _orient(points, np.array(tets, dtype=np.int64))
    return points, cells


def _orient(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """``cells`` with the last two vertices swapped where the volume is < 0."""
    p = points[cells]
    vol = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    out = cells.copy()
    flip = vol < 0
    out[flip, 2], out[flip, 3] = cells[flip, 3], cells[flip, 2]
    return out


def sample(
    field: Field,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The field at the vertices of its support's tetrahedra cut ``order`` times.

    Returns:
        ``(points (N, 3) in m, cells (M, 4), point_data, cell_data)``; the
        points of each tetrahedron are its own (no sharing).
    """
    mesh = field.mesh
    excluded = field._excluded  # noqa: SLF001
    built, fes, region = fe_space(
        mesh, field.support, excluded, field.space, field.order
    )
    _check_size(field, fes)
    n = max(field.order, 1)
    ref, sub = _lattice(n)
    rule = ngs.IntegrationRule([tuple(p) for p in ref], [0.0] * len(ref))
    mips = built.mesh.MapToAllElements(rule, region)
    xyz = np.asarray(ngs.CF((ngs.x, ngs.y, ngs.z))(mips)) * built.scale
    n_el = len(xyz) // len(ref)

    centre = ngs.IntegrationRule([(0.25, 0.25, 0.25)], [0.0])
    owner = p0(built, np.arange(len(mesh.tetrahedra), dtype=float))
    tet = np.rint(
        np.asarray(owner(built.mesh.MapToAllElements(centre, region)))[:, 0]
    ).astype(np.int64)

    cells = (sub[None] + len(ref) * np.arange(n_el)[:, None, None]).reshape(-1, 4)
    cells = _orient(xyz, cells)

    coefficients = np.asarray(field.coefficients)
    factor = built.scale ** LENGTH_POWER[field.space]
    parts = (
        {f"{field.quantity}_real": coefficients.real}
        | {f"{field.quantity}_imag": coefficients.imag}
        if np.iscomplexobj(coefficients)
        else {field.quantity: coefficients}
    )
    point_data = {}
    gf = ngs.GridFunction(fes)
    for name, values in parts.items():
        gf.vec.FV().NumPy()[:] = values
        sampled = np.asarray(gf(mips)) * factor
        point_data[name] = sampled[:, 0] if sampled.shape[1] == 1 else sampled
    cell_data = {
        "region": np.repeat(mesh.tetrahedron_tags[tet], len(sub)),
        "entity": np.repeat(mesh.tetrahedron_entities[tet], len(sub)),
    }
    return xyz, cells, point_data, cell_data


# --- curl ------------------------------------------------------------------------

_CURLS: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()
_MAX_CURLS = 4


def curl_matrix(field: Field) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The map from H(curl) coefficients to the H(div) coefficients of the curl.

    Exact: H(div) of order ``max(p - 1, 0)`` contains the curl of first-kind
    Nédélec of order ``p``. For affine elements the map depends only on the
    mesh topology (the Piola maps cancel), so it is built at the imported
    node positions and serves moved and traced meshes alike.

    Returns:
        ``(rows, cols, values, n_rows)`` of the sparse matrix.
    """
    mesh = field.mesh
    topology = mesh._topology  # noqa: SLF001
    key = (id(topology), field.support, field._excluded, field.order)  # noqa: SLF001
    found = _CURLS.get(key)
    if found is not None and found[0] is topology:
        _CURLS.move_to_end(key)
        return found[1]
    still = Mesh(mesh.imported_nodes, topology)
    support, excluded = field.support, field._excluded  # noqa: SLF001
    _, hcurl, _ = fe_space(still, support, excluded, "hcurl", field.order)
    _check_size(field, hcurl)
    _, hdiv, _ = fe_space(still, support, excluded, "hdiv", max(field.order - 1, 0))
    op = ngs.ConvertOperator(
        spacea=hcurl, spaceb=hdiv, trial_proxy=ngs.curl(hcurl.TrialFunction())
    )
    rows, cols, values = op.COO()
    matrix = (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(values, dtype=float),
        int(hdiv.ndof),
    )
    _CURLS[key] = (topology, matrix)
    while len(_CURLS) > _MAX_CURLS:
        _CURLS.popitem(last=False)
    return matrix
