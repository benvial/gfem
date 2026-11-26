# Find the mechanical modes of a film or a rib

Find the modes of vibration of a solid cross-section that propagate along its
axis: the {term}`mechanical eigenmode` of a suspended film (its Lamb modes) or
of a rib, at a given {term}`axial wavenumber` $q$. Each mode has a frequency, a
quality factor $Q$, a displacement and a group velocity.
{py:class}`gfem.MechanicalEigenmode` solves a 2D section of elastic solids, with
the fields varying as $e^{i\omega t - iqz}$ along $z$, the normal of the
section: a waveguide of any cross-section is one solve a wavenumber.

The device is a silica-like plate, 500 nm thick and 3 µm long, held at both
ends by a frame of the same material 3 µm long. Its faces above and below are
free. {download}`plate_section.py <plate_section.py>` meshes it with netgen; the
outer ends of the frames are the line `"end"`.

## A solid and its modes

A solid needs a `stiffness`, a `mass_density` and, for a loss, a
{term}`mechanical quality factor`; see
{doc}`piezoelectric-admittance` for the tensors and {py:obj}`gfem.isotropic_stiffness`
for an isotropic solid. The problem needs the {term}`axial wavenumber` $q$ in
rad/m, a `target` frequency in Hz, how many modes (`n_modes`) and where the
displacement is held (`clamped`). Every other line is free.

```python
import urllib.request

import numpy as np

import gfem

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/plate_section.py"
)
urllib.request.urlretrieve(url, "plate_section.py")
from plate_section import plate_section  # noqa: E402

mesh = plate_section()
glass = gfem.Material(
    stiffness=gfem.isotropic_stiffness(70e9, 0.25),
    mass_density=2500.0,
    mechanical_quality_factor=1000.0,
)
model = gfem.Model(
    mesh, materials={"film": glass, "frame_left": glass, "frame_right": glass}
)

wavenumber = 2e6  # rad/m, a wavelength of 3.1 um along z
problem = gfem.MechanicalEigenmode(1e9, n_modes=3, wavenumber=wavenumber, clamped="end")
res = gfem.solve(model, problem)
print(res)
```

```text
MechanicalEigenmodeResult([974.596 MHz (Q 1000), 1.08455 GHz (Q 1000), 1.11127 GHz (Q 1000)], wavenumber=2e+06 rad/m)
```

The modes are the `n_modes` nearest the `target`, nearest first. `res.omega` is
the complex angular frequency, `res.frequency` its real part over $2\pi$ in Hz
and `res.q` the quality factor of each mode,
$Q = \mathrm{Re}\,\omega / (2\,\mathrm{Im}\,\omega)$. A solid with a uniform
$Q_m$ has $Q = Q_m$ for every mode, as here: the plate loses nothing else.
`res.error` is the backward error of each mode, below the solver's eigenvalue
tolerance.

## The displacement at a point

`res.displacement(0)` is the displacement of the first mode: a map with the
scalar fields `"x"`, `"y"` and `"z"`, in metres, that read like any field, with
`.at(points)` taking $(x, y)$ points in metres. A mode is normalized to
1 J per metre along $z$ and its largest coefficient is real, so the numbers are
comparable between modes and runs. The first mode is a flexural one, its
motion along $y$:

```python
u = res.displacement(0)
points = [(x * 1e-6, 0.25e-6) for x in (-1, 0, 1)]  # on the mid-plane
uy = np.abs(np.asarray(u["y"].at(points)))
print(
    f"|u_y| at x = -1, 0, 1 um: {uy[0] / uy[1]:.4f}, 1, {uy[2] / uy[1]:.4f} of the centre"
)
ux = abs(complex(u["x"].at([(0.0, 0.25e-6)][0])))
print("no motion along x at the centre:", ux < 1e-9 * uy[1])
```

```text
|u_y| at x = -1, 0, 1 um: 0.8155, 1, 0.8155 of the centre
no motion along x at the centre: True
```

## Group velocity

`res.group_velocity` is $\mathrm{Re}\,\mathrm{d}\omega/\mathrm{d}q$ in m/s, read
from the mode and its left eigenvector without a second solve. It is the
derivative of the discrete band, so a difference of two solves at
$q \pm \delta q$ agrees with it:

```python
dq = 1e-3 * wavenumber
bands = [
    gfem.solve(
        model,
        gfem.MechanicalEigenmode(1e9, wavenumber=wavenumber + s * dq, clamped="end"),
    )
    for s in (-1, 1)
]
slope = (float(bands[1].omega[0].real) - float(bands[0].omega[0].real)) / (2 * dq)
print(f"v_g = {float(res.group_velocity[0]):.1f} m/s, difference {slope:.1f} m/s")
```

```text
v_g = 1797.7 m/s, difference 1797.7 m/s
```

A mode with $q > 0$ travels along $+z$ where its group velocity is positive,
and $\omega(-q) = \omega(q)$ for every section (reciprocity), with the
opposite group velocity; at $q = 0$ the group velocity is exactly 0.

## A damping frame

A mode of a plate that rings between anchors loses its energy to the anchors. A
{py:class}`gfem.DampingFrame` makes the two frame regions absorb the waves that
reach them, as it does for {py:class}`gfem.Piezoelectric` (see
{doc}`../conventions`, "Piezoelectric problems"): the stiffness there is
$c\,(1 + i(1/Q_m + \eta r^2))$ with $\eta$ the `loss_factor` and $r$ the depth
into the frame. The modes are then those of the plate that radiates into the
frames, and their $Q$ is the plate's anchor loss:

```python
frame = gfem.DampingFrame(["frame_left", "frame_right"])
problem = gfem.MechanicalEigenmode(
    1e9, n_modes=3, wavenumber=wavenumber, clamped="end", damping_frame=frame
)
lossy = gfem.solve(model, problem)
for f, quality in zip(np.asarray(lossy.frequency), np.asarray(lossy.q), strict=True):
    print(f"{f / 1e9:.4f} GHz, Q = {quality:.2f}")
```

```text
1.0099 GHz, Q = 4.46
1.1017 GHz, Q = 22.89
0.8777 GHz, Q = 4.52
```

The sign of the loss follows the convention of {doc}`../conventions`: under
$e^{+i\omega t}$ a decaying mode has $\mathrm{Im}\,\omega > 0$. A mode that
overlaps the frames little keeps a higher $Q$ (22.9 here) than one that fills
them (4.5). Modes are sorted by the distance of their frequency
to the target, so the order differs between the two solves.

## What raises

- A tetrahedral or a slab mesh: the problem is 2D (`Mesh.from_gmsh` of a file
  of triangles).
- `linear_solver="petsc"`, any solve under `comm` and a GPU `device`: one
  process and the sparse direct solver.
- A `gfem.Absorbing` line on a solid's edge: use a damping frame.
- A region of the mechanical domain without a stiffness or a density.
- A gradient of `group_velocity` or `error`, or of a mode in a cluster:
  `gfem.NonDifferentiableError`. `omega` and the fields (`displacement`,
  `potential`, and what reads them, such as `gfem.strain` or `Field.at`)
  differentiate in the materials and the node coordinates.
- Fewer than `n_modes` modes meeting `SolverOptions.eigen_tolerance`:
  {py:obj}`gfem.EigenmodeConvergenceError`.

## Next

- {doc}`optomechanical-coupling`: the coupling of these modes to the light of a
  waveguide.
- {doc}`piezoelectric-admittance`: the same solids driven by electrodes, per
  volt.
- {doc}`../conventions`, "Mechanical eigenmodes": the sign of $q$, the
  normalization and the quality factor.
- {doc}`2d-models`: what a 2D model is and what it refuses.
