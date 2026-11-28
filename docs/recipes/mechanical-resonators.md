# Resonators and anchor loss

Find the {term}`mechanical eigenmode` of a resonator that is a 3D body: a
beam, a plate or a disk held at one surface, and the {term}`anchor loss` that
sets its $Q$. {py:class}`gfem.MechanicalEigenmode` takes a tetrahedral mesh as
it takes a 2D section ({doc}`mechanical-modes`), with the same arguments and the
same result.

## A 3D body

A 3D body has no axial direction, so it takes no wavenumber and its modes are
the standing ones of the whole body. The device is a silica-like cantilever
100 um long, 10 um wide and 2 um thick, clamped at its root. A `clamped`
surface holds the displacement at zero; {download}`resonator_meshes.py
<resonator_meshes.py>` meshes the beam and names its root face `"anchor"`.

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/resonator_meshes.py"
urllib.request.urlretrieve(url, "resonator_meshes.py")
from resonator_meshes import cantilever  # noqa: E402

silica = gfem.Material(
    stiffness=gfem.isotropic_stiffness(70e9, 0.25),
    mass_density=2500.0,
    mechanical_quality_factor=1000.0,
)
model = gfem.Model(cantilever(), materials={"beam": silica})
options = gfem.SolverOptions(order=2)
problem = gfem.MechanicalEigenmode(170e3, n_modes=2, clamped=["anchor"])
res = gfem.solve(model, problem, options=options)
print(res)
```

```text
MechanicalEigenmodeResult([171.608 kHz (Q 1000), 849.567 kHz (Q 1000)])
```

The beam bends along $z$ at the first mode. A thin beam of length $L$,
thickness $t$, modulus $E$ and density $\rho$ has the Euler-Bernoulli
frequency $f = (\beta_1^2 / 2\pi)\, (t / L^2) \sqrt{E / 12\rho}$ with
$\beta_1 = 1.8751$:

```python
L, t = 100e-6, 2e-6
f_eb = 1.8751**2 / (2 * np.pi) * t / L**2 * np.sqrt(70e9 / (12 * 2500.0))
print(f"f / f_EB - 1 = {res.frequency[0] / f_eb - 1:.2%}")
print(f"Q = {res.q[0]:.1f}")
```

```text
f / f_EB - 1 = 0.38%
Q = 1000.0
```

The 0.4 % is the shear and the rotary inertia of a beam 50 times longer than
thick, and the clamp, which the beam theory leaves out. The uniform $Q_m$
gives $Q = Q_m$.

## The tip and the root

`res.displacement(0)` is a map of the fields `"x"`, `"y"` and `"z"`, read with
`.at(points)` at $(x, y, z)$ points in metres. The mode carries 1 J, so only
ratios mean something. The deflection at mid-span over the tip's is the
Euler-Bernoulli shape's, and {py:func}`gfem.strain` reads the strain in Voigt
order (xx, yy, zz, yz, xz, xy) at points of the same kind:

```python
u = res.displacement(0)
tip, mid = [(L, 5e-6, 1e-6), (L / 2, 5e-6, 1e-6)]
uz = np.abs(np.asarray(u["z"].at([tip, mid])))
print(f"|u_z| at mid-span over the tip: {uz[1] / uz[0]:.3f} (Euler-Bernoulli 0.340)")
root, middle = np.abs(
    np.asarray(gfem.strain(u, [(1e-6, 5e-6, 2e-6), (L / 2, 5e-6, 2e-6)]))
)[:, 0]
print(f"|S_xx| on the top face at the root over mid-span: {root / middle:.2f}")
```

```text
|u_z| at mid-span over the tip: 0.339 (Euler-Bernoulli 0.340)
|S_xx| on the top face at the root over mid-span: 2.92
```

The strain is largest at the root, where the beam is clamped.

A 3D body has no wavenumber, so `res.wavenumber` is 0 and
`res.group_velocity` is NaN:

```python
print(res.wavenumber, res.group_velocity)
```

```text
0.0 [nan nan]
```

## Anchor loss with a layer

A resonator radiates sound into its supports: the {term}`anchor loss`, a quality
factor the beam's material does not set. A {term}`perfectly matched layer`
around the support absorbs what leaves it, so the support can be finite and
nothing reflects back. The example is a 2D in-plane cantilever, 1 um thick and
10 um long, on a silicon-like support; `anchored_beam()` in
{download}`resonator_meshes.py <resonator_meshes.py>` meshes it, with the
regions `"beam"`, `"support"` and `"layer"`. The support is half a shear
wavelength deep and wide at the beam's frequency, and the layer around it is a
pressure wavelength thick.

A leaky mode has a complex frequency that a real shift can miss behind the
layer's own modes, so the search takes a quality factor. A damping-frame run on
the layer estimates it. Ask it for a few modes and keep the one with the
highest $Q$: on fine meshes the frame has a cloud of modes near $Q = 3.5$.

```python
from resonator_meshes import anchored_beam

silicon = gfem.Material(
    stiffness=gfem.isotropic_stiffness(199e9, 0.25), mass_density=2329.0
)
mesh = anchored_beam()
support = gfem.Model(mesh, materials=dict.fromkeys(mesh.volumes, silicon))
ratio = 10  # L / h
f_eb = 1.8751**2 / (2 * np.pi) * 1e-6 / (ratio * 1e-6) ** 2
f_eb *= np.sqrt(199e9 / (1 - 0.25**2) / (12 * 2329.0))  # plane strain
options = gfem.SolverOptions(order=3)
frame = gfem.DampingFrame(["layer"])
problem = gfem.MechanicalEigenmode(f_eb, n_modes=4, damping_frame=frame)
estimate = gfem.solve(support, problem, options=options)
best = int(np.argmax(estimate.q))
print(f"frame: {estimate.frequency[best] / 1e6:.3f} MHz, Q = {estimate.q[best]:.0f}")
```

```text
frame: 14.207 MHz, Q = 1577
```

Then the layer replaces the frame, with the estimate as the quality factor the
search expects. The solve returns the flexural mode with the quality factor of
its radiation, plus the material's:

```python
problem = gfem.MechanicalEigenmode(
    float(estimate.frequency[best]),
    pml=gfem.PML(["layer"]),
    target_quality_factor=float(estimate.q[best]),
)
leaky = gfem.solve(support, problem, options=options)
q = float(leaky.q[0])
print(f"PML: {leaky.frequency[0] / 1e6:.3f} MHz, Q = {q:.0f}")
print(f"Q / (L/h)^3 = {q / ratio**3:.3f}, rigid root 1.583")
```

```text
PML: 14.207 MHz, Q = 1647
Q / (L/h)^3 = 1.647, rigid root 1.583
```

The frame's $Q$ is a rough estimate of the layer's. A beam on a support the
same material has $Q \to 1.583\,(L/h)^3$ as $L/h \to \infty$ when the root is
rigid. At $L/h = 10$ the converged value is 1.660, 5 % above it: the root's
compliance, which shrinks as the beam lengthens. The mesh here is coarse:
halving `maxh` moves $Q$ by 0.5 %.

## Layers on a 3D body

The same `pml=` takes a tetrahedral mesh. The layer is a shell of regions
around the support, thick on all three axes, and its edges and corners stretch
two or three axes at once:

```python
pml = gfem.PML(["layer"], target_reflection=1e-8)
```

Make it about one quasi-P wavelength thick: the layer's reference wavenumber is
that of the fastest wave along each side's normal, so every slower wave
reflects less. Use order 3 or higher when layers line the sides: at order 2 the
sides can make a body mode look like a layer mode and drop it. The default
reflection $R_0 = 10^{-6}$ suits a $Q$ of a few
hundred; take $10^{-8}$ for a tight one. A 3D beam on a half-space has
$Q^{-1} \propto (w/l)(h/l)^4$ and the support is a few hundred beam lengths
across at the flexural frequency, so a mode of $Q \sim 10^6$ takes hundreds of
thousands of unknowns and a large machine. The search is the same: the
frame run, then `pml=` with `target_quality_factor`.

## Limits

- A 3D body is solved on one process with the sparse direct solver; a slab mesh
  raises, as do `comm`, a Krylov `linear_solver`, `"petsc"` and a GPU `device`.
- `wavenumber` other than 0 raises: it is the phase along $z$ of a 2D section.
- `group_velocity` is NaN, and a gradient of it raises
  `gfem.NonDifferentiableError`; `omega` and the fields differentiate.
- Order 2 on a mesh a few elements across the thickness is the economy: a
  3D mode costs a factorization of tens of thousands of unknowns.
- `pml=` is exclusive with `damping_frame=`; its regions need a constant
  `gfem.Material` and `lossless=True` raises. A mode that lives in the layer
  moves with the layer, so the result drops it and the solve may ask for more
  modes than it returns.
- A design that moves nodes keeps them inside the interior box: moving a layer
  node changes the layer's thickness and so its reflection.
- A medium whose bulk waves run backward along a layer's normal, as in strongly
  anisotropic crystals, is not absorbed well; silicon absorbs like an isotropic
  solid.

## Next

- {doc}`mechanical-modes`: the 2D sections, the group velocity and the damping
  frame.
- {doc}`optomechanical-coupling`: the coupling of these modes to light.
- {doc}`../conventions`, "Mechanical eigenmodes": the normalization.
