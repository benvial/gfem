# Start from a gsim simulation

[gsim](https://pypi.org/project/gsim/) lays out, meshes and configures a device
for Palace. gfem reads what it writes: the gmsh mesh `palace.msh` and the Palace
`config.json`. Build a `gfem.Model` from those two files, then solve and
differentiate it with gfem like any other model. gfem does not import gsim.

## 1. Mesh the device with gsim

```python
from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_output_dir("sim")
sim.set_geometry(cell)  # your layout cell
sim.set_stack(get_stack())
sim.set_airbox(margin_x=50, margin_y=0, z_above=100, z_below=100)
for port in ("o1", "o2"):
    sim.add_cpw_port(port, layer="topmetal2", s_width=20, gap_width=15, excited=True)
sim.set_driven(fmin=1e9, fmax=100e9)
sim.mesh(preset="default")  # writes sim/palace.msh
sim.write_config()  # writes sim/config.json
```

## 2. Read the mesh

gsim writes micrometres; gfem works in metres, so give the file's unit:

```python
import gfem

mesh = gfem.Mesh.from_gmsh("sim/palace.msh", length_unit="um")
```

The mesh's regions are gsim's physical groups, by name.

## 3. Build the model from `config.json`

The Palace configuration names every material, conductor, absorbing surface
and port by the integer tag of its physical group. Map each tag back to its
region name and build the gfem counterparts:

```python
import json

config = json.loads(open("sim/config.json").read())
unit = config["Model"]["L0"]  # 1e-6: Palace lengths are in micrometres
name = {(r.dim, r.tag): r.name for r in mesh.regions}

materials = {
    name[3, tag]: gfem.Material(
        eps_r=m["Permittivity"], tan_delta=m.get("LossTan", 0.0)
    )
    for m in config["Domains"]["Materials"]
    for tag in m["Attributes"]
}
bnd = config["Boundaries"]
conductors = [
    gfem.Conductor(name[2, tag], c["Conductivity"], c["Thickness"] * unit)
    for c in bnd["Conductivity"]
    for tag in c["Attributes"]
]
absorbing = [
    gfem.Absorbing(name[2, tag], order=bnd["Absorbing"]["Order"])
    for tag in bnd["Absorbing"]["Attributes"]
]
ports = [
    gfem.LumpedPort(
        f"p{p['Index']}",
        tuple(name[2, e["Attributes"][0]] for e in p["Elements"]),
        tuple(e["Direction"] for e in p["Elements"]),
        impedance=p["R"],
    )
    for p in sorted(bnd["LumpedPort"], key=lambda p: p["Index"])
]
```

gsim meshes the metal traces as volumes and tags them with the name of the
dielectric around them, so a trace has no region name of its own. Exclude it
from the solve domain by the surfaces that enclose it, with
`gfem.enclosed_by`:

```python
shells = [c.surface for c in conductors]
excluded = tuple(
    gfem.enclosed_by(s, s.replace("_xy", "_z")) for s in shells if s.endswith("_xy")
)
model = gfem.Model(
    mesh,
    materials=materials,
    boundaries=[*conductors, *absorbing],
    ports=ports,
    excluded=excluded,
)
```

The conductor shells carry the `gfem.Conductor` surface impedance with Palace's
thickness convention, face for face (see `gfem.conventions`), so a gsim layout
gives the same discrete problem in both codes. gfem's reference tests build
their models from gsim's Palace cases this way (`tests/reference/test_driven.py`).

## 4. Solve and differentiate

`model` is a plain gfem model:

```python
result = gfem.solve(model, gfem.Driven(frequency=[10e9, 50e9, 100e9]))
s21 = result.s["p2", "p1"]
```

From here every gfem recipe applies. [Differentiate S-parameters and driven
fields](s-parameter-gradient.md) turns a region into a `gfem.MaterialField` and
takes `jax.grad` of an objective; the [topology
tutorial](../tutorials/topology.ipynb) runs a whole optimization.
