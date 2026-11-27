# CAD parameter gradients

Differentiate a result with respect to the named dimensions of a gmsh model you
draw, and optimize them, remeshing with the same script when the mesh runs out.
A CAD morph maps those dimensions to a moved mesh, so `jax.grad` returns one
derivative per dimension. It needs the gmsh extra: `pip install 'gfem[gmsh]'`.

## Draw the capacitor as a build function

A 2 × 2 × 1 µm parallel-plate cell holds a dielectric strip
($\varepsilon_r = 3.9$) that spans $x$, is centred at $y = 1$ µm, is $w$ wide
and sits at $0.3\ \mu\mathrm{m} < z < 0.6\ \mu\mathrm{m}$. The top face is the
terminal and the bottom face the ground. What strip width gives 42 aF?

```python
import gmsh
import jax

import gfem


def build(width):
    """The cell with a strip `width` um wide; gfem meshes it."""
    occ = gmsh.model.occ
    cell = occ.addBox(0, 0, 0, 2, 2, 1)  # x, y, z, then dx, dy, dz in um
    strip = occ.addBox(0, 1 - width / 2, 0.3, 2, width, 0.3)
    _, (cell_parts, strip_parts) = occ.fragment([(3, cell)], [(3, strip)])
    occ.synchronize()
    strip_tags = [t for _, t in strip_parts]
    gmsh.model.addPhysicalGroup(3, strip_tags, name="strip")
    air = [t for _, t in cell_parts if t not in strip_tags]
    gmsh.model.addPhysicalGroup(3, air, name="air")
    e = 1e-6
    top = gmsh.model.getEntitiesInBoundingBox(-e, -e, 1 - e, 2 + e, 2 + e, 1 + e, 2)
    bottom = gmsh.model.getEntitiesInBoundingBox(-e, -e, -e, 2 + e, 2 + e, e, 2)
    gmsh.model.addPhysicalGroup(2, [t for _, t in top], name="top")
    gmsh.model.addPhysicalGroup(2, [t for _, t in bottom], name="bottom")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.15)  # um: the sizes are yours
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0375)


def make_model(mesh):
    materials = {"strip": gfem.Material(eps_r=3.9), "air": gfem.Material()}
    return gfem.Model(mesh, materials=materials)


problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
morph = gfem.shape.cad_morph(build, {"width": 0.5}, make_model)


def capacitance(p, morph):
    """C of the top plate in aF at the CAD parameters p (um)."""
    model = morph.model.replace(mesh=morph(p))
    return gfem.solve(model, problem).capacitance[0, 0] * 1e18
```

- gfem opens gmsh, calls `build(**p)`, meshes and reads the mesh. The build
  never meshes, writes, or opens or closes gmsh, and names every volume.
- The gradient is a dict keyed like `p`, per µm.
- A key left out of `p` keeps its starting value.

## Size the strip for 42 aF

A Newton step on $C(w) - 42$ can land where the mesh no longer fits the strip.
`fits` asks whether the morph reaches a width without a poor or inverted
element; if not, `morph.remesh(p)` runs the same build there.

```python
import warnings


def fits(morph, p):
    """Whether the morph's mesh reaches p without a poor or inverted element."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.MeshQualityWarning)
        try:
            morph(p)
        except (gfem.MeshQualityWarning, gfem.MeshQualityError):
            return False
    return True


def report(p, c, grad):
    print(
        f"w = {p['width']:.4f} um: C = {float(c):.4f} aF, dC/dw = {float(grad['width']):.4f} aF/um"
    )


target = 42.0  # aF
p = morph.p0
c, grad = jax.value_and_grad(capacitance)(p, morph)
report(p, c, grad)
for _ in range(2):
    p = {"width": p["width"] - float((c - target) / grad["width"])}  # a Newton step
    if not fits(morph, p):
        morph = morph.remesh(p)  # same build and make_model, meshed at the new width
        print(f"remeshed at w = {p['width']:.4f} um")
    c, grad = jax.value_and_grad(capacitance)(p, morph)
    report(p, c, grad)
```

```text
w = 0.5000 um: C = 38.5461 aF, dC/dw = 5.1358 aF/um
remeshed at w = 1.1725 um
w = 1.1725 um: C = 41.9728 aF, dC/dw = 5.0377 aF/um
w = 1.1779 um: C = 42.0000 aF, dC/dw = 5.0372 aF/um
```

- The morph is linear and fixed at its starting width. The first step left its
  reach, so the same build function remeshed where the step landed.
- After a remesh, rebuild the objective on the new morph and reset any
  optimizer history. Newton has none.
- Compare objective values only within one mesh.

## Check your results

Against remeshed differences. Each remesh is a new discretization, so $C$ jumps
by its discretization noise at every change. Use a step near a tenth of the
feature size, never a tiny one: at $h = 10^{-5}$ µm the difference is all
noise.

```python
h = 0.125  # um, about a tenth of the strip width
up, down = (morph.remesh({"width": p["width"] + s * h}) for s in (1, -1))
difference = (capacitance(up.p0, up) - capacitance(down.p0, down)) / (2 * h)
print(
    f"jax.grad {float(grad['width']):.2f} aF/um, remeshed {float(difference):.2f} aF/um"
)
```

```text
jax.grad 5.04 aF/um, remeshed 5.06 aF/um
```

Curved walls. On a curved boundary, `jax.grad` is the derivative of the faceted
mesh, whose facet count stays fixed. It falls short of remeshed differences by
about $0.2\,(\ell/R)^2$, where $\ell$ is the mesh size and $R$ the radius of
curvature: 2.5 % on a cylinder at $\ell/R = 0.375$. Pass `curved=True` to
`cad_morph` to mesh second-order elements instead ({doc}`prepare-a-mesh`): the
mid-edge nodes of a wall move on the CAD surface, so the gradient follows the
{term}`curved mesh` and the facet shortfall is gone. A build function that sets
the gmsh element order itself is refused; `remesh` keeps `curved`. Features
that raise on curved elements still do ({doc}`../capabilities`, Limits).

How gfem itself is checked: {doc}`../verification`.

## Limits

- gmsh is an optional extra; without it `cad_morph` raises `ImportError` naming
  `gfem[gmsh]`.
- One process only: it raises `DistributedUnsupportedError` above one MPI rank.
- 3D and the OCC kernel only.
- A parameter whose rebuild changes the model's topology within the velocity
  step, moves a held port, or moves nothing raises `ValueError` naming it.
- The map is linear in $p - p_0$. Far from $p_0$ the mesh drifts from the CAD
  and the quality check fires: remesh.

## Next

{py:obj}`gfem.shape.cad_morph`, {py:obj}`gfem.shape.CadMorph`,
{doc}`shape-gradient`, {doc}`remesh-and-restart`, {doc}`prepare-a-mesh`,
{doc}`capacitance-gradient`.
