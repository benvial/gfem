"""The voided form of a gsim case: its mesh with the conductor volumes removed.

    python scale_void.py CASE [OUT]    # OUT defaults to CASE_void

gfem excludes the meshed conductor volumes from the solve, and their shells carry the
`gfem.Conductor`. This removes from the mesh the volume entities `gfem.enclosed_by`
selects, built by `scale_run.Case`, so a split and a solve need not repeat the
selection, and gfem on the voided mesh with nothing excluded solves the same system. It
writes:

- ``palace.msh``: msh 2.2 ASCII, minus those tetrahedra, the triangles no remaining
  tetrahedron touches and the nodes no remaining element uses. Nodes and elements are
  renumbered from 1, physical names and tags kept, so shells become exterior boundaries.
- ``config.json``: the case's, with ``Problem.Output`` set to ``output/palace``.
- ``port_information.json``: copied.
- ``build.json``: the entities removed, the counts before and after, and ``nd2_dof``,
  the order 2 Nedelec count ``2 edges + 2 faces``, which equals gfem's on the voided
  mesh.
  `scale_run.py` takes a case whose ``build.json`` lists them as already voided.

gsim's ``metadata.json`` is not carried over: its counts would be the source's.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scale_run import Case

TET, TRI = 4, 2
TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        if line.startswith("$End"):
            name = None
        elif line.startswith("$"):
            name = line[1:]
            out[name] = []
        elif name is not None:
            out[name].append(line)
    return out


def void_mesh(src: Path, dst: Path, excluded: set[tuple[int, int]]) -> dict:
    """Write ``src`` minus the tetrahedra of the excluded entities.

    Args:
        src: An msh 2.2 ASCII mesh of triangles and tetrahedra.
        dst: The file to write.
        excluded: ``(physical tag, gmsh entity)`` of each volume entity to remove.

    Returns:
        The counts of ``build.json``: ``[before, after]`` where they change.

    Raises:
        ValueError: The mesh is in another format or has other elements.
    """
    sections = _sections(src.read_text())
    if sections["MeshFormat"][0].split()[:2] != ["2.2", "0"]:
        raise ValueError(f"{src}: expected msh 2.2 ASCII")
    elements = [line.split() for line in sections["Elements"][1:]]
    types = {int(e[1]) for e in elements}
    if not types <= {TET, TRI}:
        raise ValueError(f"{src}: element types {sorted(types)}, expected 2 and 4")

    kept_tets, tris = [], []
    removed = 0
    for e in elements:
        tags = int(e[2])
        physical, entity = int(e[3]), int(e[4])
        nodes = tuple(int(x) for x in e[3 + tags :])
        if int(e[1]) != TET:
            tris.append((e[1 : 3 + tags], nodes))
        elif (physical, entity) in excluded:
            removed += 1
        else:
            kept_tets.append((e[1 : 3 + tags], nodes))

    faces = {tuple(sorted(n[i] for i in f)) for _, n in kept_tets for f in TET_FACES}
    kept_tris = [t for t in tris if tuple(sorted(t[1])) in faces]
    used = sorted({v for _, n in kept_tets + kept_tris for v in n})
    new_id = {old: i + 1 for i, old in enumerate(used)}
    coordinates = {
        int(line.split()[0]): line.split(None, 1)[1] for line in sections["Nodes"][1:]
    }

    lines = [
        "$MeshFormat",
        *sections["MeshFormat"],
        "$EndMeshFormat",
        "$PhysicalNames",
        *sections["PhysicalNames"],
        "$EndPhysicalNames",
        "$Nodes",
        str(len(used)),
        *(f"{new_id[v]} {coordinates[v]}" for v in used),
        "$EndNodes",
        "$Elements",
        str(len(kept_tris) + len(kept_tets)),
    ]
    for k, (head, nodes) in enumerate([*kept_tris, *kept_tets], start=1):
        lines.append(" ".join([str(k), *head, *(str(new_id[v]) for v in nodes)]))
    lines.append("$EndElements")
    dst.write_text("\n".join(lines) + "\n")

    edges = {
        tuple(sorted((n[i], n[j])))
        for _, n in kept_tets
        for i in range(4)
        for j in range(i + 1, 4)
    }
    return {
        "nd2_dof": 2 * len(edges) + 2 * len(faces),
        "tetrahedra": [len(kept_tets) + removed, len(kept_tets)],
        "tetrahedra_removed": removed,
        "triangles": [len(tris), len(kept_tris)],
        "nodes": [len(coordinates), len(used)],
    }


def void_case(case: Path, out: Path) -> dict:
    """Write the voided case; return its ``build.json``."""
    out.mkdir(parents=True, exist_ok=True)
    source = Case(case)
    tags = {r.name: r.tag for r in source.mesh.regions if r.dim == 3}
    entities = source.model().excluded_entities
    excluded = {(tags[name], entity) for name, entity in entities}
    counts = void_mesh(case / "palace.msh", out / "palace.msh", excluded)

    config = json.loads((case / "config.json").read_text())
    config["Problem"]["Output"] = "output/palace"
    (out / "config.json").write_text(json.dumps(config, indent=4) + "\n")
    if (case / "port_information.json").exists():
        shutil.copy2(case / "port_information.json", out / "port_information.json")
    info = {
        "source": case.name,
        "builder": "tools/scale_void.py",
        "excluded_entities": sorted([name, entity] for name, entity in entities),
        **counts,
    }
    (out / "build.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def main() -> None:
    """Parse the case and write its voided form."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("out", type=Path, nargs="?", default=None)
    args = parser.parse_args()
    case = args.case.resolve()
    out = args.out or case.with_name(case.name + "_void")
    print(json.dumps(void_case(case, out), indent=1))


if __name__ == "__main__":
    main()
