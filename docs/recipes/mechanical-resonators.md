# Resonators and anchor loss

Find the {term}`mechanical eigenmode` of a resonator that is a 3D body: a
beam, a plate or a disk held at one surface. {py:class}`gfem.MechanicalEigenmode`
takes a tetrahedral mesh as it takes a 2D section ({doc}`mechanical-modes`),
with the same arguments and the same result.

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

## Limits

- A 3D body is solved on one process with the sparse direct solver; a slab mesh
  raises, as do `comm`, a Krylov `linear_solver`, `"petsc"` and a GPU `device`.
- `wavenumber` other than 0 raises: it is the phase along $z$ of a 2D section.
- `group_velocity` is NaN, and a gradient of it raises
  `gfem.NonDifferentiableError`; `omega` and the fields differentiate.
- Order 2 on a mesh a few elements across the thickness is the economy: a
  3D mode costs a factorization of tens of thousands of unknowns.

## Next

- {doc}`mechanical-modes`: the 2D sections, the group velocity and the damping
  frame.
- {doc}`optomechanical-coupling`: the coupling of these modes to light.
- {doc}`../conventions`, "Mechanical eigenmodes": the normalization.
