"""The problems with no distributed path: one case per refusal key.

`tests/unit/test_mpi.py` raises each one on a stub communicator, and
`tests/mpi/test_launch.py` raises each one on every rank of a launched job.
Nothing here solves; `series` is the thermal solve the exemplar tests run
as one process. Every model is built from a directory, so a launched
function can build it from arguments that pickle.
"""

import dataclasses
from pathlib import Path

import gradient_cases as cases
import numpy as np
import periodic_cases as pc
import thermal_cases as tc

import gfem
from gfem.mesh import LENGTH_UNITS, Region, _Topology, _triangle_neighbours

FREQUENCY = 5e9

#: Two terminals of `stack`, which `cases.stack` names by its layers.
TERMINALS = {"top": "top", "mid": "oxide__nitride"}
#: The electrostatic problem of `stack`: two terminals, the bottom grounded.
ELECTROSTATIC = gfem.Electrostatic(terminals=TERMINALS, ground="bottom")


def stack(directory: Path) -> gfem.Model:
    """A small electrostatic stack with its materials set."""
    return cases.stack(directory).replace(
        materials={
            "oxide": gfem.Material(eps_r=3.9),
            "nitride": gfem.Material(eps_r=7.5),
        }
    )


#: test_steady_thermal's two-layer series slab, (name, thickness in um) up z.
SERIES = (("lower", 6.0), ("upper", 2.0))


def series_mesh(directory: Path) -> Path:
    """Write the series slab's 2 x 1 x 8 um brick to ``directory``; its path.

    Write it on one process: a launched function reads it with `series`.
    """
    path = directory / "series.msh"
    bottom = (lambda p: abs(p[2]) < 1e-9, "bottom")
    tc.brick(path, (2.0, 1.0), SERIES, 0.7, [bottom])
    return path


def series(path: Path) -> tuple[gfem.Model, gfem.SteadyThermal]:
    """``(model, problem)``: the series slab heated above, a thermal solve ranks refuse.

    Rank copies (``comm=MPI.COMM_SELF``) and a communicator of one rank solve it.
    """
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    materials = {
        "lower": gfem.Material(thermal_conductivity=1.4),
        "upper": gfem.Material(thermal_conductivity=0.2),
    }
    bottom = gfem.FixedTemperature("bottom", temperature=300.0)
    model = gfem.Model(mesh, materials=materials, boundaries=[bottom])
    heat = gfem.HeatSource.uniform("upper", power_density=1e12)
    return model, gfem.SteadyThermal(heat)


#: Points (m) on the series slab's top face, where its rise is largest.
SERIES_TOP = np.array([[x * 1e-6, 0.5e-6, 8e-6] for x in (0.3, 1.0, 1.7)])


def _line(directory: Path) -> gfem.Model:
    return cases.layered_line(directory)


def _cell(directory: Path) -> gfem.Model:
    return pc.c1(pc.c1_mesh(directory))


def refused(key: str, directory: Path):  # noqa: PLR0911, PLR0912
    """``(model, problem, options)`` of a solve refused under ``key``."""
    direct = gfem.SolverOptions(linear_solver="direct")
    match key:
        case "adaptive_sweep":
            problem = gfem.Driven([1e9, 5e9], sweep="adaptive")
            return _line(directory), problem, direct
        case "arpack":
            options = gfem.SolverOptions(linear_solver="direct", eigen_solver="arpack")
            problem = gfem.Eigenmode(target=FREQUENCY, n_modes=1)
            return _line(directory), problem, options
        case "thermal":
            return stack(directory), gfem.SteadyThermal(), direct
        case "thermal_transient":
            problem = gfem.TransientThermal(duration=1e-6, steps=2)
            return stack(directory), problem, direct
        case "dc_conduction":
            problem = gfem.DCConduction(terminals=TERMINALS)
            return stack(directory), problem, direct
        case "periodic":  # two pairs: one pair runs distributed
            return pc.c2(pc.c2_mesh(directory)), gfem.Driven(FREQUENCY), direct
        case "bloch":  # a Bloch driven solve runs distributed
            problem = gfem.Eigenmode(target=FREQUENCY, n_modes=1, wave_vector=pc.C1_K)
            return _cell(directory), problem, direct
        case "eigenmode_wave_port":
            model = _line(directory).replace(ports=[gfem.WavePort("w", "port1")])
            return model, gfem.Eigenmode(target=FREQUENCY, n_modes=1), direct
        case "floquet_port":
            model = pc.f1(pc.f1_mesh(directory))
            return model, gfem.Driven(pc.F1_FREQUENCY), direct
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
        case "pmg_gmres":
            options = gfem.SolverOptions(linear_solver="pmg_gmres")
            return _line(directory), gfem.Driven(FREQUENCY), options
        case "petsc_preconditioner":
            options = gfem.SolverOptions(
                linear_solver="petsc", petsc_options={"pc_type": "ilu"}
            )
            return _line(directory), gfem.Driven(FREQUENCY), options
    raise ValueError(key)


#: Every case of `refused`.
KEYS = (
    "adaptive_sweep",
    "arpack",
    "thermal",
    "thermal_transient",
    "dc_conduction",
    "periodic",
    "bloch",
    "eigenmode_wave_port",
    "floquet_port",
    "frequency_workers",
    "device",
    "bddc_order_1",
    "bddc_order_1_auto",
    "pmg_gmres",
    "petsc_preconditioner",
)


#: What a call other than `gfem.solve` refuses (a method after a solve,
#: `gfem.error_estimate`, `gfem.refine_mesh`, or the backward of an eigenmode
#: mode-field output); `tests/mpi` raises each one on every rank.
METHOD_KEYS = (
    "field_export",
    "load_result_fields",
    "region_mean",
    "element_mean",
    "electric_energy",
    "dissipated_power",
    "optical_force",
    "electromagnetic_heating",
    "error_estimate",
    "mesh_refinement",
    "mode_field_gradient",
    "cad_morph",
)
#: The refusal keys no public call reaches, so no page lists them.
GUARD_KEYS: tuple[str, ...] = ()


def feature(key: str) -> str:
    """The ``feature`` a case raises."""
    return key.removesuffix("_auto").removesuffix("_transient")


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
            volume_elements=number[topology.volume_elements],
            surface_elements=number[topology.surface_elements],
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
        volume_elements=tets,
        volume_tags=tags,
        volume_entities=tags.copy(),
        surface_elements=tris,
        surface_tags=tri_tags,
        surface_entities=tri_tags.copy(),
        surface_neighbours=_triangle_neighbours(nodes, tets, tris, Path("box")),
    )
    return gfem.Mesh(metres, topology)
