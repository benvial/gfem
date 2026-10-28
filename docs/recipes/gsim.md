# Start from a gsim simulation

[gsim](https://pypi.org/project/gsim/) lays out, meshes and configures a
device. gfem reads what it writes: the gmsh mesh `palace.msh` and the solver
configuration `config.json`. Build a `gfem.Model` from those two files, then solve and
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

The conductor shells carry the `gfem.Conductor` surface impedance with the
thickness convention of gsim's configuration, face for face (see
`gfem.conventions`).

gsim always writes a second-order {term}`absorbing boundary`, and the model
above keeps it. gfem adds the second-order term with the textbook sign (see
{py:obj}`gfem.Absorbing`), at which it reflects less than the first-order
boundary. A solver that adds it with the opposite sign reflects more, so its S
differs from gfem's: visibly on gsim's tight default {term}`air box`, most near
deep notches of |S11|, and less as the air box grows. `order=1` in
`gfem.Absorbing` above gives the first-order boundary; keep the second order
for your own results.

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

## Many frequencies: the adaptive sweep

gsim sets up a broadband sweep (`fmin=1e9, fmax=100e9` above), and a
`gfem.Driven` sweep is uniform by default: every frequency is a full solve.
For more than about 20 frequencies, ask for the adaptive sweep. It solves a
few sample frequencies in full, every excited port at once, and reads S and
the kept fields at every frequency of the grid from a reduced model built on
those solutions:

```python
import numpy as np

problem = gfem.Driven(frequency=np.linspace(1e9, 100e9, 300), sweep="adaptive")
result = gfem.solve(model, problem)
s21 = result.s["p2", "p1"]  # all 300 frequencies
result.sweep.sampled_frequencies  # the frequencies solved in full, Hz
```

`result.sweep` tells how each frequency was obtained:

- `solved` marks, frequency by frequency, the ones solved in full; the rest
  come from the reduced model.
- `converged` says whether the last samples agreed with the reduced model
  to `SolverOptions.adaptive_tolerance`. A sweep that reaches
  `SolverOptions.adaptive_max_samples` first warns with
  `gfem.SweepNotConvergedWarning` and reports `converged=False`: raise the
  sample cap.

On a short grid the greedy may run out of frequencies: it then solves every
one in full, as a uniform sweep does, and reports `converged=True`.

The speed costs field accuracy. A uniform sweep's fields are as accurate as
the linear solve (`SolverOptions.tolerance`). An adaptive sweep's fields, at
the frequencies it did not solve, are accurate to about `adaptive_tolerance`
or better (relative error, `1e-3` by default), and S to about its square. Measurements
read the fields ([Measure fields after the solve](measurements.md)), so a
probe or an interface participation on an adaptive result carries that error.
Tighten it with `gfem.SolverOptions(adaptive_tolerance=1e-5)` or
`gfem.SolverOptions.preset("accurate")`, at the cost of more samples, or keep
the uniform sweep where the fields matter most.

Under `jax.grad`, `jax.jit` or `jax.vmap` an adaptive sweep needs its samples
frozen (`Driven(samples=...)`): see [Differentiate an adaptive frequency
sweep](adaptive-sweep-gradient.md).

## Many cores: frequency workers

A uniform sweep can also run several frequencies at once.
`gfem.SolverOptions(frequency_workers=k)` splits the grid into `k` contiguous
blocks and solves each block in a {term}`frequency worker`, a separate process
with `threads // k` of the solve's threads (of the logical CPUs the process
may run on when `threads` is `None`). One small solve uses many threads
poorly, so workers pay off on a many-core machine; on a few cores the gain is
small.

```python
options = gfem.SolverOptions(frequency_workers=4, threads=16)
problem = gfem.Driven(frequency=np.linspace(1e9, 100e9, 64))
result = gfem.solve(model, problem, options=options)
```

- Memory grows about `k` times: each worker holds one solve.
- S agrees with one worker's to the solver precision: with `bddc_gmres`, each
  block's first solve starts cold, so S moves within `SolverOptions.tolerance`.
- Only uniform driven sweeps use it; adaptive sweeps, eigenmode and
  electrostatic solves ignore it. GPU devices, and wave ports on more than one
  frequency, need `frequency_workers=1`.
- The workers are spawned processes: a script that sets `frequency_workers`
  guards its entry point with `if __name__ == "__main__":`.
- Gradients work as usual; the adjoint solves run in the main process.
