"""Write the front-page hero data, ``docs/_generated/front-page.json``.

The data comes from the topology tutorial itself: this script executes
``docs/tutorials/topology.ipynb`` (through nbclient, cwd ``docs/tutorials``)
with one export cell appended, which reads the notebook's own namespace
(``mesh``, ``grid``, ``blur``, ``line``, ``AIR``, ``CERAMIC``, ``problem``,
``options``, ``final``, ``history``, ``missed``, ``balance``) and writes:

- the top face of the ``design`` volume as triangles (x, y in mm), each with
  its tetrahedron's |dJ/d eps_r| at the tutorial's start design (nominal,
  beta = 2), normalised to its maximum, where
  J = 1 - |S21(5 GHz)|^2 + |S21(8 GHz)|^2;
- the eroded / nominal / dilated designs after the optimisation (beta = 64),
  sampled at each triangle's tetrahedron (0 air, 1 ceramic);
- the missed-power history, the final J of each variant and the power-balance
  check of the nominal design.

The only line copied from the tutorial is its start design (``jax.random.key(0)``),
so the hero cannot drift from the model it links to. The notebook file is not
modified. ``docs/conf.py`` inlines the JSON into the front page.

Run from the repo root, in the docs environment (about 4 minutes on CPU)::

    uv sync --python 3.13 --group docs --group site
    uv run python tools/front_page_data.py
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs" / "tutorials" / "topology.ipynb"
OUT = ROOT / "docs" / "_generated" / "front-page.json"

# Runs in the tutorial's kernel after its last cell; {out!r} is the JSON path.
EXPORT = """\
# front-page export, appended by tools/front_page_data.py (not part of the tutorial)
import json as _json

_rho0 = 0.5 + 0.01 * jax.random.normal(jax.random.key(0), grid.shape)
_start = design.robust(blur(_rho0), beta=2.0, delta=0.15).nominal
_field0 = gfem.MaterialField.interpolate(grid.to_elements(_start), AIR, CERAMIC)


def _j_nominal(field):
    t = gfem.solve(line.with_material("design", field), problem, options=options)
    t = t.s["p2", "p1"]
    return 1 - jnp.abs(t[0]) ** 2 + jnp.abs(t[1]) ** 2


_dj = np.abs(np.asarray(jax.grad(_j_nominal)(_field0).eps_r))

# the top face of the design volume, each triangle with its tetrahedron
_nodes = np.asarray(mesh.nodes) / mm
_tets = mesh.tetrahedra[mesh.elements("design")]
_owner_of = {{}}
for _k, _t in enumerate(_tets):
    for _drop in range(4):
        _owner_of[frozenset(np.delete(_t, _drop).tolist())] = _k
_tris, _owner = [], []
for _f in mesh.triangles[mesh.elements("top")]:
    _k = _owner_of.get(frozenset(_f.tolist()))
    if _k is not None:
        _tris.append(_f.tolist())
        _owner.append(_k)
_owner = np.asarray(_owner)
_used = sorted({{n for _f in _tris for n in _f}})
_index = {{n: i for i, n in enumerate(_used)}}
_xy = _nodes[_used, :2]
_sens = _dj[_owner] / _dj[_owner].max()

_data = {{
    "source": "docs/tutorials/topology.ipynb, executed by tools/front_page_data.py",
    "triangles": len(_tris),
    "tetrahedra": len(_tets),
    "extent_mm": [
        float(_xy[:, 0].min()),
        float(_xy[:, 0].max()),
        float(_xy[:, 1].min()),
        float(_xy[:, 1].max()),
    ],
    "nodes": _xy.round(4).tolist(),
    "tris": [[_index[n] for n in _f] for _f in _tris],
    "sens0": _sens.round(4).tolist(),
    "variants": {{
        _name: np.asarray(grid.to_elements(_image))[_owner].round(3).tolist()
        for _name, _image in zip(final._fields, final, strict=True)
    }},
    "history": np.asarray(history).round(5).tolist(),
    "final": dict(zip(final._fields, np.asarray(missed).tolist(), strict=True)),
    "balance": float(balance),
}}
with open({out!r}, "w") as _fh:
    _json.dump(_data, _fh, separators=(",", ":"))
print(f"{{len(_tris)}} top triangles over {{len(_tets)}} design tetrahedra")
"""


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
    tmp = out.with_suffix(".json.tmp")  # never leave a half-written JSON
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(EXPORT.format(out=str(tmp))))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # inherited by the kernel
    client = NotebookClient(
        nb,
        timeout=1800,  # as the docs build's notebook execution
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()
    for output in nb.cells[-1].outputs:
        if output.get("output_type") == "stream":
            print(output["text"], end="")
    tmp.replace(out)
    print("wrote", out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)


if __name__ == "__main__":
    main()
