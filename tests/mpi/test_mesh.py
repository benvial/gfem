"""A rank holds its part of the NGSolve mesh and knows gfem's number of each piece.

Replicated arrays are compared bit by bit across ranks by `on_ranks`; a rank's own
piece is returned with ``replicated=False`` and summed here. Reference values come
from this one-process pytest.
"""

import gradient_cases as cases
import mpi_cases
import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm
from ranks import on_ranks

import gfem
from gfem import _mpi
from gfem._engine.cache import discretization
from gfem._engine.coefficients import (
    coefficient,
    p0,
    surface_coefficient,
    surface_p0,
)
from gfem._engine.mesh import inverse_order, to_ngsolve
from gfem.mesh import LENGTH_UNITS

TOPOLOGY_ARRAYS = (
    "imported_nodes",
    "tetrahedra",
    "tet_tags",
    "tet_entities",
    "triangles",
    "tri_tags",
    "tri_entities",
    "tri_neighbours",
)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _write_line(directory):
    """A file of `layered_line`'s shape, written with no call to `from_gmsh`."""
    mesh = sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1), layers=cases.LAYERS)
    return sm.write_msh22(directory / "line.msh", mesh)


# --- reading the file ------------------------------------------------------------


def _read_counting(path):
    import gfem.mesh as module

    real, calls = module.meshio.read, []

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    module.meshio.read = counting
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    topology = mesh._topology
    arrays = {name: getattr(topology, name) for name in TOPOLOGY_ARRAYS}
    return {
        "reads": _world().allgather(len(calls)),
        "arrays": arrays,
        "nodes": np.asarray(mesh.nodes),
        "regions": topology.regions,
        "unit": topology.length_unit,
        "source": str(topology.source),
        "writeable": any(a.flags.writeable for a in arrays.values()),
    }


def test_rank_0_reads_the_file_once_and_every_rank_holds_the_same_mesh(tmp_path):
    path = _write_line(tmp_path)
    expected = gfem.Mesh.from_gmsh(path, length_unit="mm")
    value = on_ranks(_read_counting, path, ranks=3, limit=120)[0]
    assert value["reads"] == [1, 0, 0]
    assert not value["writeable"]
    for name, array in value["arrays"].items():
        reference = getattr(expected._topology, name)
        assert array.dtype == reference.dtype
        assert np.array_equal(array, reference), name
    assert np.array_equal(value["nodes"], np.asarray(expected.nodes))
    assert value["regions"] == expected._topology.regions
    assert value["unit"] == "mm"
    assert value["source"] == str(path.resolve())


def _read_own_path_then_a_collective(directory):
    world = _world()
    try:
        gfem.Mesh.from_gmsh(directory / f"rank-{world.rank}.msh")
    except _mpi.AgreedError as err:
        return "COMM_SELF" in str(err), world.allreduce(world.rank)
    raise AssertionError("mismatched paths were accepted")


def test_ranks_reading_different_paths_raise_on_every_rank(tmp_path):
    values = on_ranks(_read_own_path_then_a_collective, tmp_path, ranks=3, limit=60)
    assert values[0] == (True, 0 + 1 + 2)


def _read_with_own_unit_then_a_collective(path):
    world = _world()
    try:
        gfem.Mesh.from_gmsh(path, length_unit="mm" if world.rank else "um")
    except _mpi.AgreedError:
        return world.allreduce(world.rank)
    raise AssertionError("mismatched units were accepted")


def test_ranks_reading_with_different_units_raise_on_every_rank(tmp_path):
    path = _write_line(tmp_path)
    assert on_ranks(_read_with_own_unit_then_a_collective, path, ranks=3)[0] == 3


def _read_expecting(path, unit, kinds):
    world = _world()
    try:
        gfem.Mesh.from_gmsh(path, length_unit=unit)
    except tuple(kinds) as err:
        return type(err).__name__, str(err), world.allreduce(world.rank)
    raise AssertionError("the read passed")


def test_an_unknown_unit_raises_on_every_rank(tmp_path):
    path = _write_line(tmp_path)
    values = on_ranks(_read_expecting, path, "furlong", [ValueError], ranks=3)
    assert values[0][0] == "ValueError"
    assert values[0][2] == 3


@pytest.mark.parametrize(
    "text", [None, "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"], ids=["missing", "empty"]
)
def test_a_file_that_cannot_be_read_raises_on_every_rank(tmp_path, text):
    path = tmp_path / "bad.msh"
    if text is not None:
        path.write_text(text)
    with pytest.raises(Exception) as serial:
        gfem.Mesh.from_gmsh(path)
    values = on_ranks(_read_expecting, path, "um", [type(serial.value)], ranks=3)
    assert values[0][0] == type(serial.value).__name__
    assert values[0][1] == str(serial.value)
    assert values[0][2] == 3


def _read_own_file_on_comm_self(directory):
    from mpi4py import MPI

    own = directory / f"rank-{MPI.COMM_WORLD.rank}"
    own.mkdir()
    mesh = gfem.Mesh.from_gmsh(_write_line(own), length_unit="mm", comm=MPI.COMM_SELF)
    return str(mesh._topology.source)


def test_comm_self_reads_each_ranks_own_file(tmp_path):
    values = on_ranks(
        _read_own_file_on_comm_self, tmp_path, ranks=3, replicated=False, limit=120
    )
    assert values == [
        str((tmp_path / f"rank-{r}" / "line.msh").resolve()) for r in range(3)
    ]


# --- a rank's part and its maps ----------------------------------------------------

#: The check's bound on integrals of per-element fields against one process.
AGREEMENT = 3e-14
RANKS = (3, 4, 8)


def _mesh(case, directory):
    """A case by name, so that a launched function can build it from arguments."""
    if case == "box":
        return mpi_cases.box(4)
    if case == "line":
        return gfem.Mesh.from_gmsh(_write_line(directory), length_unit="mm")
    raise ValueError(case)


def _part(case, directory):
    built = to_ngsolve(_mesh(case, directory), comm=_world())
    ngmesh = built.mesh.ngmesh
    return {
        "size": built.mesh.comm.size,
        "domains": built.domains,
        "node_order": built.node_order,
        "tetrahedron_order": built.tetrahedron_order,
        "triangle_order": built.triangle_order,
        "coordinates": np.array(ngmesh.Coordinates()),
        "tets": np.asarray(ngmesh._get3dElementsAsTets()).reshape(-1, 4),
        "tris": np.asarray(ngmesh._get2dElementsAsTriangles()).reshape(-1, 3),
        "materials": [e.mat for e in built.mesh.Elements(ngs.VOL)],
        "boundaries": [e.mat for e in built.mesh.Elements(ngs.BND)],
    }


def _names(mesh):
    return {(r.dim, r.tag): r.name for r in mesh.regions}


@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize("case", ["box", "line"])
def test_each_element_is_on_one_rank_and_the_maps_name_it(case, ranks, tmp_path):
    mesh = _mesh(case, tmp_path)
    parts = on_ranks(_part, case, tmp_path, ranks=ranks, replicated=False, limit=180)
    names = _names(mesh)
    assert {p["size"] for p in parts} == {ranks}
    assert all(p["domains"] == parts[0]["domains"] for p in parts)
    tets = np.concatenate([p["tetrahedron_order"] for p in parts])
    tris = np.concatenate([p["triangle_order"] for p in parts])
    assert np.array_equal(np.sort(tets), np.arange(len(mesh.tetrahedra)))
    assert np.array_equal(np.sort(tris), np.arange(len(mesh.triangles)))
    # rank 0 keeps nothing
    assert len(parts[0]["tetrahedron_order"]) == len(parts[0]["node_order"]) == 0
    nodes = np.asarray(mesh.nodes) / LENGTH_UNITS[mesh.length_unit]
    held = np.unique(np.concatenate([p["node_order"] for p in parts]))
    assert np.array_equal(held, np.arange(len(nodes)))
    for p in parts:
        order = p["node_order"]
        assert np.array_equal(p["coordinates"], nodes[order])
        # a local element holds the nodes of the gfem element the map names
        for local, rows, map_ in (
            (p["tets"], mesh.tetrahedra, p["tetrahedron_order"]),
            (p["tris"], mesh.triangles, p["triangle_order"]),
        ):
            assert np.array_equal(
                np.sort(order[local], axis=1), np.sort(rows[map_], axis=1)
            )
        assert p["materials"] == [
            names[3, int(t)] for t in mesh.tetrahedron_tags[p["tetrahedron_order"]]
        ]
        assert p["boundaries"] == [
            names[2, int(t)] for t in mesh.triangle_tags[p["triangle_order"]]
        ]


def _inverses(case, directory):
    mesh = _mesh(case, directory)
    built = to_ngsolve(mesh, comm=_world())
    return (
        built.tetrahedron_order,
        inverse_order(built.tetrahedron_order, len(mesh.tetrahedra)),
    )


def test_a_rank_inverts_its_map_at_full_length_with_minus_one_elsewhere(tmp_path):
    mesh = _mesh("box", tmp_path)
    parts = on_ranks(_inverses, "box", tmp_path, ranks=3, replicated=False)
    for order, inverse in parts:
        assert inverse.shape == (len(mesh.tetrahedra),)
        assert np.array_equal(inverse[order], np.arange(len(order)))
        assert (np.delete(inverse, order) == -1).all()


# --- element-indexed coefficients ------------------------------------------------


def _weight():
    return 1 + ngs.x + ngs.y * ngs.z


def _integrals(built, mesh):
    """Integrals of random per-element fields through the element maps."""
    rng = np.random.default_rng(5)
    per_tet = 1 + 3 * rng.random(len(mesh.tetrahedra))
    per_tri = 1 + 3 * rng.random(len(mesh.triangles))
    per_tri_c = per_tri * (1 - 2j * rng.random(len(mesh.triangles)))
    tensor_tet = 1 + rng.random((len(mesh.tetrahedra), 3, 3))
    tensor_tri = 1 + rng.random((len(mesh.triangles), 3, 3))
    weight, normal = _weight(), ngs.CF((1, 2, 3))
    m = built.mesh
    out = {
        "volume": ngs.Integrate(p0(built, per_tet) * weight, m, order=4),
        "surface": ngs.Integrate(
            surface_p0(built, per_tri) * weight, m, ngs.BND, order=4
        ),
        "complex": ngs.Integrate(
            surface_p0(built, per_tri_c) * weight, m, ngs.BND, order=4
        ),
        "tensor volume": ngs.Integrate(
            ngs.InnerProduct(coefficient(built, tensor_tet) * normal, normal) * weight,
            m,
            order=4,
        ),
        "tensor surface": ngs.Integrate(
            ngs.InnerProduct(surface_coefficient(built, tensor_tri) * normal, normal)
            * weight,
            m,
            ngs.BND,
            order=4,
        ),
    }
    for region in m.GetMaterials():
        out[f"volume {region}"] = ngs.Integrate(
            p0(built, per_tet) * weight, m, order=4, definedon=m.Materials(region)
        )
    for region in m.GetBoundaries():
        out[f"surface {region}"] = ngs.Integrate(
            surface_p0(built, per_tri) * weight,
            m,
            ngs.BND,
            order=4,
            definedon=m.Boundaries(region),
        )
    return out


def _integrals_on_ranks(case, directory):
    mesh = _mesh(case, directory)
    return _integrals(to_ngsolve(mesh, comm=_world()), mesh)


def _assert_agree(values, expected):
    assert values.keys() == expected.keys()
    for name, value in values.items():
        # an integral near 600 differs by 1e-13 in summing the same terms in another
        # order: one unit in the last place
        bound = AGREEMENT * max(1.0, abs(expected[name]))
        assert abs(value - expected[name]) <= bound, name


@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize("case", ["box", "line"])
def test_per_element_fields_integrate_as_on_one_process(case, ranks, tmp_path):
    mesh = _mesh(case, tmp_path)
    expected = _integrals(to_ngsolve(mesh), mesh)
    values = on_ranks(_integrals_on_ranks, case, tmp_path, ranks=ranks, limit=180)[0]
    _assert_agree(values, expected)


# --- a moved mesh ------------------------------------------------------------------


def _moved(mesh):
    scale = 1e-6
    nodes = np.asarray(mesh.nodes)
    shift = np.sin(np.pi * nodes[:, [1, 2, 0]] / scale) * np.sin(np.pi * nodes / scale)
    return gfem.Mesh(nodes + 0.03 * scale * shift, mesh._topology)


def _move():
    world = _world()
    mesh = mpi_cases.box(4)
    first = discretization(mesh, comm=world).built
    moved = _moved(mesh)
    second = discretization(moved, comm=world).built
    fes = [ngs.HCurl(b.mesh, order=2, complex=True) for b in (first, second)]
    return {
        "same maps": all(
            np.array_equal(getattr(first, f), getattr(second, f))
            for f in ("node_order", "tetrahedron_order", "triangle_order")
        ),
        "same local dofs": fes[0].ndof == fes[1].ndof,
        "other meshes": first.mesh is not second.mesh,
        "integrals": _integrals(second, moved),
    }


@pytest.mark.parametrize("ranks", RANKS)
def test_a_moved_mesh_keeps_its_partition(ranks):
    mesh = mpi_cases.box(4)
    moved = _moved(mesh)
    expected = _integrals(to_ngsolve(moved), moved)
    value = on_ranks(_move, ranks=ranks, limit=180, replicated=False)
    assert all(v["same maps"] for v in value)
    assert all(v["same local dofs"] for v in value)
    assert all(v["other meshes"] for v in value)
    _assert_agree(value[0]["integrals"], expected)


def _plant_another_partition(planting):
    from gfem._engine import mesh as engine

    world = _world()
    mesh = mpi_cases.box(4)
    first = to_ngsolve(mesh, comm=world)
    if world.rank == planting:
        memo = engine.partition_memo(mesh._topology)[_mpi.comm_key(world)]
        memo.node_order = memo.node_order[::-1]
    try:
        to_ngsolve(_moved(mesh), comm=world)
    except _mpi.AgreedError as err:
        return "partition" in str(err), world.allreduce(world.rank), first.mesh.ne >= 0
    raise AssertionError("a changed partition passed")


@pytest.mark.parametrize("planting", [1, 2])
def test_a_later_partition_that_differs_raises_on_every_rank(planting):
    values = on_ranks(_plant_another_partition, planting, ranks=3, replicated=False)
    assert values == [(True, 0 + 1 + 2, True)] * 3


def _plant_other_tets():
    from gfem._engine import mesh as engine

    world = _world()
    mesh = mpi_cases.box(4)
    to_ngsolve(mesh, comm=world)
    memo = engine.partition_memo(mesh._topology)[_mpi.comm_key(world)]
    if world.rank == 1:
        memo.tetrahedron_order = memo.tetrahedron_order[::-1]
    try:
        to_ngsolve(_moved(mesh), comm=world)
    except _mpi.AgreedError:
        return world.allreduce(world.rank)
    raise AssertionError("other elements on a rank passed")


def test_a_partition_with_the_same_nodes_and_other_elements_raises():
    assert on_ranks(_plant_other_tets, ranks=3)[0] == 3


# --- the cache ---------------------------------------------------------------------


def _keys():
    world = _world()
    mesh = mpi_cases.box(2)
    on_world = discretization(mesh, comm=world)
    dup = world.Dup()
    on_dup = discretization(mesh, comm=dup)
    alone = discretization(mesh)
    result = {
        "found again": discretization(mesh, comm=world) is on_world,
        "another communicator": on_dup is not on_world,
        "one process apart": alone is not on_world and alone.built.comm is None,
        "dup is the comm": on_dup.built.comm is dup,
        "alone holds all": len(alone.built.tetrahedron_order) == len(mesh.tetrahedra),
    }
    # MPI may reuse a freed handle: the next communicator is not found as the old one
    dup.Free()
    again = world.Dup()
    result["freed"] = discretization(mesh, comm=again) is not on_dup
    again.Free()
    return result


def test_the_discretization_key_holds_the_communicator():
    value = on_ranks(_keys, ranks=3, limit=120)[0]
    assert all(value.values()), value


def _on_a_part_of_the_world():
    world = _world()
    part = world.Split(world.rank % 2, world.rank)
    mesh = mpi_cases.box(3)
    built = to_ngsolve(mesh, comm=part)
    tets = part.allreduce(len(built.tetrahedron_order))
    volume = ngs.Integrate(1, built.mesh)
    return built.mesh.comm.size, tets, len(mesh.tetrahedra), round(volume, 12)


def test_a_mesh_lives_on_a_part_of_the_world():
    values = on_ranks(_on_a_part_of_the_world, ranks=6, limit=180)
    assert values[0][:2] == (3, values[0][2])
    assert values[0][3] == 1.0


def _smaller_than_the_ranks():
    world = _world()
    mesh = mpi_cases.box(1)
    built = to_ngsolve(mesh, comm=world)
    return built.mesh.ne, built.tetrahedron_order, _integrals(built, mesh)


def test_a_rank_of_the_partition_can_hold_nothing():
    mesh = mpi_cases.box(1)
    expected = _integrals(to_ngsolve(mesh), mesh)
    values = on_ranks(_smaller_than_the_ranks, ranks=8, replicated=False, limit=120)
    holding = [ne for ne, _, _ in values[1:]]
    assert 0 in holding
    tets = np.concatenate([order for _, order, _ in values])
    assert np.array_equal(np.sort(tets), np.arange(len(mesh.tetrahedra)))
    _assert_agree(values[0][2], expected)


def _periodic_under_ranks():
    world = _world()
    mesh = mpi_cases.box(2)
    try:
        to_ngsolve(mesh, periodic=(object(),), comm=world)
    except gfem.DistributedUnsupportedError as err:
        return err.feature
    raise AssertionError("periodic ran")


def test_a_periodic_mesh_does_not_reach_netgen_under_ranks():
    assert on_ranks(_periodic_under_ranks, ranks=3)[0] == "periodic"


# --- SurfaceL2 hangs on a one-sided cut ---------------------------------------------


def _one_sided_cuts(built, mesh, world):
    """Pairs ``(holder, other)`` that ticket 09's rule predicts hang `SurfaceL2`.

    A pair is a surface triangle inside the mesh, held by one rank, with the tet on
    its other side on another rank, and no such triangle the other way round.
    """
    owner = np.zeros(len(mesh.tetrahedra), dtype=np.int64)
    owner[built.tetrahedron_order] = world.rank
    world.Allreduce(_in_place(), owner)
    holder = np.zeros(len(mesh.triangles), dtype=np.int64)
    holder[built.triangle_order] = world.rank
    world.Allreduce(_in_place(), holder)
    neighbours = np.asarray(mesh.triangle_neighbours)
    inner = (neighbours >= 0).all(axis=1)
    sides = owner[neighbours[inner]]
    held = holder[inner]
    other = np.where(sides[:, 0] == held, sides[:, 1], sides[:, 0])
    pairs = {(int(a), int(b)) for a, b in zip(held, other, strict=True) if a != b}
    return sorted(p for p in pairs if p[::-1] not in pairs)


def _in_place():
    from mpi4py import MPI

    return MPI.IN_PLACE


def _surface_p0_on_a_one_sided_cut():
    world = _world()
    mesh = mpi_cases.box(2)
    built = to_ngsolve(mesh, comm=world)
    return _one_sided_cuts(built, mesh, world), _integrals(built, mesh)


def test_surface_p0_builds_on_a_one_sided_cut():
    mesh = mpi_cases.box(2)
    expected = _integrals(to_ngsolve(mesh), mesh)
    cuts, values = on_ranks(_surface_p0_on_a_one_sided_cut, ranks=8, limit=120)[0]
    # the case still is the one that hangs `SurfaceL2` on the wheel
    assert cuts
    _assert_agree(values, expected)
