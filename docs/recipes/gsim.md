# Start from a gsim simulation

Solve and differentiate a device that [gsim](https://pypi.org/project/gsim/)
has laid out, meshed and configured. gfem reads what gsim writes: the gmsh mesh
`device.msh` and the solver configuration `config.json`. Build a
{py:obj}`gfem.Model` from those two files, then use it like any other model.
gfem does not import gsim.

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
sim.mesh(preset="default", model_name="device")  # writes sim/device.msh
sim.write_config()  # writes sim/config.json
```

## 2. Read the mesh

gsim writes micrometres; gfem works in metres, so give the file's unit:

```python
import gfem

mesh = gfem.Mesh.from_gmsh("sim/device.msh", length_unit="um")
```

The mesh's regions are gsim's physical groups, by name.

## 3. Build the model from `config.json`

The configuration names every material, conductor, absorbing surface
and port by the integer tag of its physical group. Map each tag back to its
region name and build the gfem counterparts:

```python
import json

config = json.loads(open("sim/config.json").read())
unit = config["Model"]["L0"]  # 1e-6: lengths are in micrometres
name = {(r.dim, r.tag): r.name for r in mesh.regions}

materials = {
    name[3, tag]: gfem.Material(
        eps_r=m["Permittivity"],
        mu_r=m.get("Permeability", 1.0),
        tan_delta=m.get("LossTan", 0.0),
        conductivity=m.get("Conductivity", 0.0),
        axes=m.get("MaterialAxes"),
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

`config.json` gives each material three principal values and a `MaterialAxes`
frame; they map onto gfem's principal values and `axes`
({doc}`anisotropic-materials`).

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

The conductor shells carry the {py:obj}`gfem.Conductor` surface impedance with the
thickness convention of gsim's configuration, face for face (see
{py:obj}`gfem.conventions`).

gsim always writes a second-order {term}`absorbing boundary`, and the model
above keeps it. gfem adds the second-order term with the textbook sign (see
{py:obj}`gfem.Absorbing`), at which it reflects less than the first-order
boundary. What it still reflects shows most on gsim's tight default
{term}`air box` and near deep notches of |S11|, and less as the air box grows.
Where those notches matter, grow the air box with `sim.set_airbox` until S stops
changing. `order=1` in `gfem.Absorbing` above gives the first-order boundary;
keep the second order for your own results.

## 4. Solve and differentiate

`model` is a plain gfem model:

```python
result = gfem.solve(model, gfem.Driven(frequency=[10e9, 50e9, 100e9]))
s21 = result.s["p2", "p1"]
```

From here every gfem recipe applies. [Differentiate S-parameters and driven
fields](s-parameter-gradient.md) turns a region into a {py:obj}`gfem.MaterialField` and
takes `jax.grad` of an objective; the [topology
tutorial](../tutorials/topology.ipynb) runs a whole optimization.

gsim sets up a broadband sweep (`fmin=1e9, fmax=100e9` above). For more than
about 20 frequencies, {doc}`frequency-sweep` reads them from a few solves or
spreads them over many cores.
