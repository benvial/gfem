# Remesh and restart

A shape design in gfem moves the nodes of one imported mesh; it never changes
the mesh's topology. `gfem.shape.boundary_morph` builds a fixed linear map from
normal displacements of a surface's nodes to a moved mesh, solved once on the
imported mesh. Small and moderate moves keep the mesh valid, but a long
optimization can travel further than one mesh allows. This recipe shows how to
notice that, and how to remesh at the current geometry and carry on.

gfem v0.1 does not remesh for you: gfem reads meshes, it does not make them.
You regenerate the layout and its mesh yourself (here with gdsfactory and
gsim), and restart the optimizer from where it stopped.

## How far one mesh goes

On the IHP TopMetal2 coplanar line of the reference set (`driven/r8`, a 15 µm
gap), pushing one sidewall of the signal conductor into the gap with a smooth
bump gives:

| Bump amplitude into the gap | What happens |
|---|---|
| up to 10.5 µm | nothing |
| 11 µm | `MeshQualityWarning`: an element fell below a quarter of its imported quality |
| 14 µm (93% of the gap) | `MeshQualityError`: an element is inverted |

Moving the wall into the metal inverts nothing down to −15 µm. Three things
make this travel possible, and they are worth keeping in your own setup:

- every named surface other than the moving one, the ports and `fixed=` slides
  in its own plane, so layered meshes deform along their interfaces;
- the default extension is stiffened elasticity: small elements, which sit
  next to the geometry's features, resist deformation most;
- the design is smooth along the surface (see below).

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
Ports are held automatically: a node on a port surface never moves and is not
a parameter.

## 2. Accept steps on mesh quality

`gfem.shape.quality(mesh)` returns the scaled Jacobian of every tetrahedron: 1
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

Calling the morph with concrete values also checks the moved mesh: it warns
with `MeshQualityWarning` when an element falls below a quarter of its
imported quality and raises `MeshQualityError` on an inverted element. Every
solve on a moved mesh runs the same check. Inside an objective,
`quality` can also serve as a smoothly aggregated constraint.

## 3. Remesh at the current geometry and restart

When backtracking stalls, or the warnings keep coming, the mesh has gone as
far as it can. At the current parameters `q_k`:

1. **Regenerate the layout at `q_k`** with gdsfactory, the same cell with the
   new widths and gaps.
2. **Mesh it with gsim and build a new `Model`**, with the same materials,
   boundaries, ports and `excluded=` as before.
3. **Build a new `boundary_morph`** on the new model with the same selection,
   and the same `q → t` map *relative to `q_k`*: the new mesh already has the
   geometry of `q_k`, so its displacement is zero there.
4. **Restart the optimizer from `q_k` with its history reset.** The discrete
   objective jumps a little at a remesh (a new mesh has a new discretization
   error), so L-BFGS curvature pairs and MMA asymptotes from the old mesh no
   longer apply.

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
    sim.mesh(preset="default", refined_mesh_size=8.0, max_mesh_size=40.0)
    mesh = gfem.Mesh.from_gmsh(f"{outdir}/palace.msh", length_unit="um")
    return make_model(mesh)  # materials, boundaries, ports, excluded=


q_base = q_k
model = build_model(q_base, "remesh-01")
morph = gfem.shape.boundary_morph(model, "topmetal2_z", where=signal_wall_at(q_base))


def design_mesh(q):
    return morph(displacements(q - q_base, morph))


optimizer = make_optimizer(x0=q_base)  # fresh state: no history from the old mesh
```

Two details to keep in mind:

- The selection may move with the geometry: if the wall sat at `y = 10 µm` on
  the first mesh, it sits at `y = 10 µm + Δ` on the new one. Write `where=` as
  a function of `q_base`.
- Compare objective values only within one mesh. Record the value at `q_k` on
  both meshes if you want to see the size of the jump.

An automatic remesh-and-restart hook through gsim is on the roadmap; in v0.1
it is this manual loop.
