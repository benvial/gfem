"""Write the front-page hero data, ``docs/_data/front-page.json`` (committed).

The data comes from the topology tutorial itself: this script executes
``docs/tutorials/topology.ipynb`` (through nbclient, cwd ``docs/tutorials``)
with one cell appended that calls `export` on the notebook's namespace. Every
model value is read from that namespace, never typed here: the mesh, the
layer grid, the filter, the materials, the objective ``missed_power``, the
pixel-to-field map ``material`` and the solve ``s21_of_field``, the start
design ``rho0``, the continuation schedule ``BETAS`` / ``STEPS_PER_BETA``,
the etch offset ``DELTA``, the sweep and the results. The JSON holds:

- the top face of the ``design`` volume as triangles (x, y in mm), each with
  its tetrahedron's |dJ/d eps_r| at the start design (the nominal variant at
  the first beta), normalised to its maximum;
- the eroded / nominal / dilated designs after the optimisation, sampled at
  each triangle's tetrahedron (0 air, 1 ceramic);
- the missed-power history, the final J of each variant and the power-balance
  check of the nominal design;
- the numbers the front page quotes (``params``): layer size, filter radius,
  ceramic permittivity, pass and stop frequencies, first beta, steps per
  beta and sweep range.

The notebook file is not modified. ``docs/conf.py`` inlines the JSON into the
front page; regenerate it whenever the tutorial changes::

    just front-page-data

(about 4 minutes on CPU, in the docs environment).
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs" / "tutorials" / "topology.ipynb"
OUT = ROOT / "docs" / "_data" / "front-page.json"

# Appended to the tutorial; runs in its kernel after its last cell.
EXPORT_CELL = """\
# front-page export, appended by tools/front_page_data.py (not part of the tutorial)
import sys as _sys

_sys.path.insert(0, {tools!r})
from front_page_data import export as _export

_export(globals(), {out!r})
"""


def export(ns: dict, out: str) -> None:
    """Write the hero JSON from the executed tutorial's namespace ``ns``."""
    import json

    import jax
    import numpy as np

    from gfem import design

    mesh, grid, mm, GHz = ns["mesh"], ns["grid"], ns["mm"], ns["GHz"]
    final, start_beta = ns["final"], ns["BETAS"][0]

    # |dJ/d eps_r| per element of the nominal start design
    start = design.robust(ns["blur"](ns["rho0"]), beta=start_beta, delta=ns["DELTA"])

    def j_nominal(field):
        return ns["missed_power"](ns["s21_of_field"](field))

    field0 = ns["material"](start.nominal)
    dj = np.abs(np.asarray(jax.grad(j_nominal)(field0).eps_r))

    # the top face of the design volume, each triangle with its tetrahedron
    nodes = np.asarray(mesh.nodes) / mm
    tets = mesh.tetrahedra[mesh.elements("design")]
    owner_of = {}
    for k, tet in enumerate(tets):
        for drop in range(4):
            owner_of[frozenset(np.delete(tet, drop).tolist())] = k
    tris, owner = [], []
    for face in mesh.triangles[mesh.elements("top")]:
        k = owner_of.get(frozenset(face.tolist()))
        if k is not None:
            tris.append(face.tolist())
            owner.append(k)
    owner = np.asarray(owner)
    used = sorted({n for face in tris for n in face})
    index = {n: i for i, n in enumerate(used)}
    xy = nodes[used, :2]
    sens = dj[owner] / dj[owner].max()
    extent = [
        float(xy[:, 0].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].min()),
        float(xy[:, 1].max()),
    ]
    sweep = np.asarray(ns["sweep"].frequency) / GHz

    data = {
        "source": "docs/tutorials/topology.ipynb, executed by tools/front_page_data.py",
        "triangles": len(tris),
        "tetrahedra": len(tets),
        "extent_mm": extent,
        "params": {
            "layer_mm": [extent[1] - extent[0], extent[3] - extent[2]],
            "radius_mm": ns["blur"].radius / mm,
            "eps_r": ns["CERAMIC"].eps_r,
            "pass_ghz": ns["PASS"] / GHz,
            "stop_ghz": ns["STOP"] / GHz,
            "beta0": start_beta,
            "steps_per_beta": ns["STEPS_PER_BETA"],
            "sweep_ghz": [float(sweep.min()), float(sweep.max())],
        },
        "nodes": xy.round(4).tolist(),
        "tris": [[index[n] for n in face] for face in tris],
        "sens0": sens.round(4).tolist(),
        "variants": {
            name: np.asarray(grid.to_elements(image))[owner].round(3).tolist()
            for name, image in zip(final._fields, final, strict=True)
        },
        "history": np.asarray(ns["history"]).round(5).tolist(),
        "final": dict(
            zip(final._fields, np.asarray(ns["missed"]).tolist(), strict=True)
        ),
        "balance": float(ns["balance"]),
    }
    tmp = Path(out).with_suffix(".json.tmp")  # never leave a half-written JSON
    tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    tmp.replace(out)
    print(f"{len(tris)} top triangles over {len(tets)} design tetrahedra")


def main() -> None:
    """Execute the tutorial with the export cell and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="output JSON path")
    args = parser.parse_args()
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise SystemExit(
            f"{exc}: run in the docs environment "
            "(uv sync --python 3.13 --group docs --group site)"
        ) from exc

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    cell = EXPORT_CELL.format(tools=str(Path(__file__).resolve().parent), out=str(out))
    nb.cells.append(nbformat.v4.new_code_cell(cell))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # inherited by the kernel
    client = NotebookClient(
        nb,
        timeout=1800,  # as CI's tutorials job
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()
    for output in nb.cells[-1].outputs:
        if output.get("output_type") == "stream":
            print(output["text"], end="")
    print("wrote", out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)


if __name__ == "__main__":
    main()
