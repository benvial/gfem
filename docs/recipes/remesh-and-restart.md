# Remeshing a shape design

Remesh a shape design at its current geometry and restart the optimization from it. A shape design moves the nodes of one imported mesh and never changes its topology, so a long optimization can outrun the mesh. gfem does not make meshes: you regenerate the layout and mesh (here with gdsfactory and gsim) and restart. The [shape tutorial](../tutorials/shape.ipynb) runs a whole optimization on one mesh; {doc}`shape-gradient` explains the gradients.

## 1. Parametrize smoothly

`boundary_morph` has one parameter per node of the moving surface: its normal
displacement in metres. Do not optimize those directly. Write a few CAD-like
parameters `q` and map them to `t` with `morph.points` (imported node
coordinates, metres) and `morph.normals`:

```python
import gfem
import jax
import jax.numpy as jnp
import numpy as np

UM = 1e-6


def signal_wall(p):
    """The signal conductor's +y sidewall: y = 10 um, inside the metal layer."""
    return (
        (np.abs(p[:, 1] - 10 * UM) < 1e-12)
        & (p[:, 2] > 11 * UM)
        & (p[:, 2] < 14.1 * UM)
    )


morph = gfem.shape.boundary_morph(model, "topmetal2_z", where=signal_wall)


def displacements(q, morph):
    """Half the width change, as a smooth spline of x along the line."""
    x = morph.points[:, 0]
    knots = np.linspace(x.min(), x.max(), len(q))
    return jnp.interp(x, knots, q / 2)
```

Mesh surfaces are grouped by layer and orientation (`topmetal2_z` holds every
sidewall of every conductor), so `where=` picks the feature by coordinates.
Ports never move and are not parameters.

## 2. Accept steps on mesh quality

{py:obj}`gfem.shape.quality` returns the scaled Jacobian of every tetrahedron: 1
for a regular one, at most 0 for an inverted one. Check a trial step before you
take it, and backtrack when the mesh degrades:

```python
q0 = gfem.shape.quality(model.mesh)  # imported quality
# Under jit the morph skips its own check (below), so `acceptable` decides.
design_mesh = jax.jit(lambda q: morph(displacements(q, morph)))


def acceptable(mesh):
    q = gfem.shape.quality(mesh)
    return bool(jnp.all(q > 0.25 * q0))


step = 1.0
while True:
    trial = q - step * grad
    if acceptable(design_mesh(trial)):
        break
    step /= 2
    if step < 1e-3:
        raise RuntimeError("the mesh is used up: remesh and restart (below)")
```

A morph called with concrete values warns with `MeshQualityWarning` when an
element falls below a quarter of its imported quality and raises
`MeshQualityError` on an inverted one; every solve on a moved mesh runs the
same check. Inside an objective, `quality` can serve as a smoothly aggregated
constraint.

## How far one mesh goes

On a coplanar line in IHP's TopMetal2 with a 15 µm gap, pushing one sidewall of
the signal conductor into the gap with a smooth bump gives:

| Bump amplitude into the gap | What happens |
|---|---|
| up to 10.5 µm | nothing |
| 11 µm | {py:obj}`gfem.MeshQualityWarning`: an element fell below a quarter of its imported quality |
| 14 µm (93% of the gap) | {py:obj}`gfem.MeshQualityError`: an element is inverted |

Moving the wall into the metal inverts nothing down to −15 µm. Three things
extend the travel: other named surfaces (ports, `fixed=` slides) move in their
own plane; the default extension is stiffened elasticity, where small elements
resist most; and the design is smooth along the surface.

## 3. Remesh at the current geometry and restart

When only the mesh is too coarse, refining it is lighter:
{doc}`mesh-refinement`. When backtracking stalls or the warnings keep coming,
the mesh has gone as far as it can. At the current parameters `q_k`:

1. **Regenerate the layout at `q_k`** with gdsfactory, the same cell with the
   new widths and gaps.
2. **Mesh it with gsim and build a new `Model`** with the same materials,
   boundaries, ports and `excluded=`.
3. **Build a new `boundary_morph`** with the same selection and the same
   `q → t` map *relative to `q_k`*: the new mesh already has the geometry of
   `q_k`.
4. **Restart the optimizer from `q_k` with its history reset**: the objective
   jumps a little at a remesh, so L-BFGS pairs and MMA asymptotes no longer
   apply.

```python
import gdsfactory as gf
from gsim.common.stack import get_stack
from gsim.palace import DrivenSim


def build_model(q, outdir):
    """Layout, mesh and model at the CAD parameters q (gsim's CPW example)."""
    cell = cpw_layout(q)  # your gdsfactory cell, parametrized by q
    sim = DrivenSim()
    sim.set_output_dir(outdir)
    sim.set_geometry(cell)
    sim.set_stack(get_stack())
    sim.set_airbox(margin_x=50, margin_y=0, z_above=100, z_below=100)
    for port in ("o1", "o2"):
        sim.add_cpw_port(
            port, layer="topmetal2", s_width=20, gap_width=15, excited=True
        )
    sim.mesh(
        preset="default",
        refined_mesh_size=8.0,
        max_mesh_size=40.0,
        model_name="device",
    )
    mesh = gfem.Mesh.from_gmsh(f"{outdir}/device.msh", length_unit="um")
    return make_model(mesh)  # materials, boundaries, ports, excluded=


q_base = q_k
model = build_model(q_base, "remesh-01")
morph = gfem.shape.boundary_morph(model, "topmetal2_z", where=signal_wall_at(q_base))


def design_mesh(q):
    return morph(displacements(q - q_base, morph))


optimizer = make_optimizer(x0=q_base)  # fresh state: no history from the old mesh
```

- The selection may move with the geometry: a wall at `y = 10 µm` sits at
  `y = 10 µm + Δ` on the new mesh, so write `where=` as a function of `q_base`.
- Compare objective values only within one mesh.

## Next

{py:obj}`gfem.shape.boundary_morph`, {py:obj}`gfem.shape.quality`,
{doc}`shape-gradient`, {doc}`gsim`, the [shape tutorial](../tutorials/shape.ipynb).
