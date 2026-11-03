"""The problems with no distributed path: one case per refusal key (ADR 0072).

`tests/unit/test_mpi.py` raises each one on a stub communicator, and
`tests/mpi/test_launch.py` raises each one on every rank of a launched job.
Nothing here solves. Every model is built from a directory, so a launched
function can build it from arguments that pickle.
"""

import dataclasses
from pathlib import Path

import gradient_cases as cases
import numpy as np
import periodic_cases as pc

import gfem
from gfem.mesh import LENGTH_UNITS, Region, _Topology, _triangle_neighbours

FREQUENCY = 5e9

#: Two terminals of `stack`, which `cases.stack` names by its layers.
TERMINALS = {"top": "top", "mid": "oxide__nitride"}


def stack(directory: Path) -> gfem.Model:
    """A small electrostatic stack with its materials set."""
    return cases.stack(directory).replace(
        materials={
            "oxide": gfem.Material(eps_r=3.9),
            "nitride": gfem.Material(eps_r=7.5),
        }
    )


def _line(directory: Path) -> gfem.Model:
    return cases.layered_line(directory)


def _cell(directory: Path) -> gfem.Model:
    return pc.c1(pc.c1_mesh(directory))


def refused(key: str, directory: Path):  # noqa: PLR0911
    """``(model, problem, options)`` of a solve refused under ``key``."""
    direct = gfem.SolverOptions(linear_solver="direct")
    match key:
        case "adaptive_sweep":
            problem = gfem.Driven([1e9, 5e9], sweep="adaptive")
            return _line(directory), problem, direct
        case "eigenmode":
            return _line(directory), gfem.Eigenmode(target=FREQUENCY, n_modes=1), direct
        case "electrostatic":
            return _line(directory), gfem.Electrostatic(terminals=TERMINALS), direct
        case "periodic":
            return _cell(directory), gfem.Driven(FREQUENCY), direct
        case "bloch":
            problem = gfem.Driven(FREQUENCY, wave_vector=pc.C1_K)
            return _cell(directory), problem, direct
        case "frequency_workers":
            options = gfem.SolverOptions(linear_solver="direct", frequency_workers=2)
            return _line(directory), gfem.Driven([1e9, 5e9]), options
        case "device":
            options = gfem.SolverOptions(linear_solver="direct", device="cuda")
            return _line(directory), gfem.Driven(FREQUENCY), options
        case "bddc_order_1":
            options = gfem.SolverOptions(order=1, linear_solver="bddc_gmres")
            return _line(directory), gfem.Driven(FREQUENCY), options
        case "bddc_order_1_auto":
            options = gfem.SolverOptions(order=1)
            return _line(directory), gfem.Driven(FREQUENCY), options
        # Not built yet: direct under ranks (slice 4).
        case "driven":
            return _line(directory), gfem.Driven(FREQUENCY), direct
    raise ValueError(key)


#: Every case of `refused`.
KEYS = (
    "adaptive_sweep",
    "eigenmode",
    "electrostatic",
    "periodic",
    "bloch",
    "frequency_workers",
    "device",
    "bddc_order_1",
    "bddc_order_1_auto",
    "driven",
)


def feature(key: str) -> str:
    """The ``feature`` a case raises."""
    return key.removesuffix("_auto")


def permuted(mesh: gfem.Mesh, seed: int = 7) -> gfem.Mesh:
    """``mesh`` with its nodes renumbered at random, so order carries no geometry."""
    topology = mesh._topology
    order = np.random.default_rng(seed).permutation(len(mesh.nodes))
    number = np.empty_like(order)
    number[order] = np.arange(len(order))
    return gfem.Mesh(
        np.asarray(mesh.nodes)[order],
        dataclasses.replace(
            topology,
            imported_nodes=np.asarray(topology.imported_nodes)[order],
            tetrahedra=number[topology.tetrahedra],
            triangles=number[topology.triangles],
            source=None,
        ),
    )


# --- the unit cube, built from arrays --------------------------------------------

_KUHN = np.array(
    [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]]
)
_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
_TAGS = {
    "air": 1,
    "diel": 2,
    "metal": 3,
    "port": 11,
    "abc": 12,
    "xmin": 13,
    "xmax": 14,
    "ymin": 15,
    "ymax": 16,
    "sheet": 17,
    "metal_surf": 18,
}


def box(n: int, seed: int = 1) -> gfem.Mesh:
    """A cube of ``n`` cells a side in um, tetrahedra shuffled as in a file.

    Volumes: air (z < 1/2), diel (z > 1/2), metal (a block inside the air). Surfaces:
    port (z = 0), abc (z = 1), the side walls, and two interior ones, sheet (z = 1/2)
    and metal_surf (around the block).
    """
    grid = np.arange(n + 1) / n
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")
    nodes = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    k, j, i = (a.ravel() for a in np.meshgrid(*[np.arange(n)] * 3, indexing="ij"))
    corners = np.stack(
        [
            (i + a) + (n + 1) * ((j + b) + (n + 1) * (k + c))
            for c in (0, 1)
            for b in (0, 1)
            for a in (0, 1)
        ],
        axis=1,
    )
    tets = corners[:, _KUHN].reshape(-1, 4)
    centres = nodes[tets].mean(axis=1)
    tags = np.where(centres[:, 2] < 0.5, _TAGS["air"], _TAGS["diel"])
    metal = (
        (centres[:, 0] > 0.25)
        & (centres[:, 0] < 0.5)
        & (centres[:, 1] > 0.25)
        & (centres[:, 1] < 0.5)
        & (centres[:, 2] < 0.25)
    )
    tags[metal] = _TAGS["metal"]
    order = np.random.default_rng(seed).permutation(len(tets))
    tets, tags = tets[order], tags[order]

    faces = np.sort(tets[:, _FACES].reshape(-1, 3), axis=1)
    unique, inverse, count = np.unique(
        faces, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    low, high = np.full(len(unique), 99), np.zeros(len(unique), dtype=int)
    np.minimum.at(low, inverse, np.repeat(tags, 4))
    np.maximum.at(high, inverse, np.repeat(tags, 4))
    c = nodes[unique].mean(axis=1)
    outer, inner = count == 1, count == 2
    surfaces = {
        "port": outer & (np.abs(c[:, 2]) < 1e-9),
        "abc": outer & (np.abs(c[:, 2] - 1) < 1e-9),
        "xmin": outer & (np.abs(c[:, 0]) < 1e-9),
        "xmax": outer & (np.abs(c[:, 0] - 1) < 1e-9),
        "ymin": outer & (np.abs(c[:, 1]) < 1e-9),
        "ymax": outer & (np.abs(c[:, 1] - 1) < 1e-9),
        "sheet": inner & (low == _TAGS["air"]) & (high == _TAGS["diel"]),
        "metal_surf": inner & (high == _TAGS["metal"]),
    }
    tris = np.concatenate([unique[m] for m in surfaces.values()])
    tri_tags = np.concatenate(
        [np.full(m.sum(), _TAGS[name]) for name, m in surfaces.items()]
    )
    regions = tuple(
        Region(name, tag, 3 if tag < 10 else 2) for name, tag in _TAGS.items()
    )
    metres = nodes * LENGTH_UNITS["um"]
    topology = _Topology(
        length_unit="um",
        imported_nodes=metres,
        regions=regions,
        tetrahedra=tets,
        tet_tags=tags,
        tet_entities=tags.copy(),
        triangles=tris,
        tri_tags=tri_tags,
        tri_entities=tri_tags.copy(),
        tri_neighbours=_triangle_neighbours(nodes, tets, tris, Path("box")),
    )
    return gfem.Mesh(metres, topology)
