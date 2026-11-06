# Solve a periodic cell

Get the band diagram of a structure that repeats along one, two or three
directions, or the S-parameters of one phased unit cell, by solving a single
cell. You mesh one cell and pair its opposite faces with {py:obj}`gfem.Periodic`.
With no wave vector the fields repeat unchanged from cell to cell
({term}`plain periodicity`). With `wave_vector=k` they pick up a
{term}`Bloch phase` from cell to cell.

## Band frequencies of a layered cell

The cell below is a line of ε = 2 and ε = 9 layers that repeats every 10 mm
along x. Its mesh needs two outer faces that match node for node under a
translation; gmsh's `setPeriodic` makes such a pair.

```python
import numpy as np

import gfem

mesh = gfem.Mesh.from_gmsh("bragg_cell.msh", length_unit="mm")
cell = gfem.Model(
    mesh,
    materials={"low": gfem.Material(eps_r=2.0), "high": gfem.Material(eps_r=9.0)},
    boundaries=[
        gfem.Periodic(("x_min", "x_max")),
        gfem.PEC("z_min"),
        gfem.PEC("z_max"),
    ],
)
(a,) = cell.periodic_translations  # m, from x_min to x_max: (0.01, 0.0, 0.0)
a = np.asarray(a)
options = gfem.SolverOptions(order=3)

bands = []
for phase in np.linspace(0.1 * np.pi, np.pi, 10):  # phase per cell, rad
    k = tuple(phase * a / (a @ a))  # rad/m
    # aim below every band: the light line of the densest layer, n = 3
    target = 0.9 * 299_792_458 * np.linalg.norm(k) / (2 * np.pi * 3)
    problem = gfem.Eigenmode(target=target, n_modes=2, wave_vector=k)
    modes = gfem.solve(cell, problem, options=options)
    bands.append(modes.frequency)  # Hz, the two lowest bands at this k
```

- `bands` holds the two lowest band frequencies, in Hz, at 10 wave vectors
  from 0.1π/a to the zone edge π/a. `modes.frequency` is Re ω / 2π and
  `modes.q` is the Q. For a lossless cell `modes.omega` is exactly real.
- The wave vector is Cartesian, in rad/m. For a phase φ per cell along a
  translation a, k = φ a / |a|², as in the loop. Under e^{+iωt} the fields
  obey `E(r + a) = exp(-i k·a) E(r)` ({py:obj}`gfem.conventions`, Bloch
  periodicity).
- Give k only along the translations. Any other component raises `ValueError`
  at the solve, and the message gives the projection to use instead.
- Only the phase per cell enters the solve, so a k outside the first Brillouin
  zone gives the folded band. Keep your k path inside the first zone.
- At k = 0 the cell also has a field at zero frequency: aim the target above
  it.
- Each mode field covers the whole cell, so `Field.at` reads it at any point.
  Under a Bloch wave vector, gfem sets a mode's overall phase from its largest
  coefficient, not from the physics: compare fields through ratios, not raw
  phases.

## Pair the faces

`gfem.Periodic` takes the names of the two faces. gfem measures the
translation between them from the mesh, so you give none:
`Model.periodic_translations` holds one per pair, in metres, in `boundaries`
order. The order of the two faces in a pair does not change the field.

- A rotated pair, an element touching both faces, more than three pairs, or
  two parallel translations raise `ValueError` when you build the model. Remesh
  with matching faces under a translation, at most one pair per direction.
- A paired face takes no other boundary condition and no port; the default PMC
  skips it. A wave port that touches a paired face raises
  `NotImplementedError`: place the port's face away from the pair.

## Drive a cell at a Bloch wave vector

A two-port line cell, ε = 4 with tan δ = 0.02, 10 mm long, with lumped ports
on two interior sheets:

```python
mesh = gfem.Mesh.from_gmsh("line_cell.msh", length_unit="mm")
line = gfem.Model(
    mesh,
    materials={"cell": gfem.Material(eps_r=4.0, tan_delta=0.02)},
    boundaries=[
        gfem.Periodic(("x_min", "x_max")),
        gfem.PEC("z_min"),
        gfem.PEC("z_max"),
    ],
    ports=[
        gfem.LumpedPort("p1", "sheet1", "+z", impedance=50.0),
        gfem.LumpedPort("p2", "sheet2", "+z", impedance=50.0),
    ],
)
k = (0.3 * np.pi / 0.01, 0.0, 0.0)  # rad/m: a phase of 0.3π per 10 mm cell
s = gfem.solve(line, gfem.Driven(5e9, wave_vector=k)).s.values[0]
```

- S is referenced to the field E, not to a periodic envelope;
  {py:obj}`gfem.conventions` gives the factor between the two. S is not
  symmetric under a wave vector: S(k)ᵀ = S(−k) ({term}`Bloch reciprocity`).
- The same k holds at every frequency of the sweep.
- Only the uniform sweep takes a wave vector. `sweep="adaptive"` with a wave
  vector raises `NotImplementedError`: use `sweep="uniform"`. Plain
  periodicity takes both sweeps.

## Differentiate a band

A periodic model differentiates like any other. Here the lowest band at the
zone edge, with respect to the permittivity of each element of the ε = 9
layer:

```python
import jax
import jax.numpy as jnp

edge = (np.pi / 0.01, 0.0, 0.0)  # rad/m, k = π/a
problem = gfem.Eigenmode(target=4e9, n_modes=2, wave_vector=edge)


def lowest_band(eps):
    m = cell.with_material("high", gfem.MaterialField(eps_r=eps))
    return gfem.solve(m, problem, options=options).frequency[0]


n = len(cell.mesh.elements("high"))
grad = jax.grad(lowest_band)(jnp.full(n, 9.0))  # Hz per unit eps_r
```

- {py:obj}`gfem.shape.boundary_morph` moves a node and its partner on the other
  face of a pair together. A paired face cannot be the moving surface
  (`ValueError`): move an interior surface instead.
- `wave_vector` is not differentiable: a value inside `jax.grad` raises
  `gfem.NonDifferentiableError`. For a group velocity, difference two band
  solves at nearby k.
- {doc}`../capabilities` lists which periodic gradients each solver supports.

## Choose a solver

| | Plain periodic | Bloch wave vector |
|---|---|---|
| `direct` | driven and eigenmode | driven and eigenmode (MKL PARDISO with the `mkl` extra, UMFPACK otherwise) |
| `bddc_gmres` | driven and eigenmode | driven only; eigenmode raises `ValueError`: use `direct` |
| `petsc` | driven | raises `ValueError`: use `direct` |
| `device="cuda"` | untested: compare with `direct` once on your cell | raises `NotImplementedError`: use `direct` on the CPU |
| eigen solver | ARPACK or SLEPc | ARPACK (SLEPc untested: use ARPACK) |

## Check your results

Four checks you can run on your own cell:

- **Transpose.** Solve the driven cell at k and at −k. `S(k).T` should equal
  `S(−k)` to the solver tolerance.
- **Zero wave vector.** `wave_vector=(0, 0, 0)` should give the same result as
  no wave vector: a cheap check of the pairing and the ports.
- **Folding.** With the same target, k and k + 2π a / |a|² should give the
  same band frequencies.
- **Lossless cell.** With no loss in the cell, `modes.omega` is real, and the
  driven cell's S with resistive lumped ports keeps SᴴS = I to the solver
  tolerance.

How gfem itself is checked on periodic cells: {doc}`../verification`.

## Limits

- No Floquet ports, so no plane-wave incidence and no diffraction orders.
  Drive the cell with lumped or wave ports inside it.
- Translations only, one to three pairs. For a rotational symmetry, mesh the
  whole structure.
- `gfem.Electrostatic` on a periodic model raises `NotImplementedError`. Mesh
  enough cells and solve without pairs.
- `gfem.interface_participation` on a paired face raises `ValueError`, since
  that face is inside the repeated structure. `gfem.power_flow` works on
  planes that cross or touch a pair.
- No k-path argument: loop over k, as in the first example.
- A gradient through an eigenmode's field raises, as on every model. Build the
  objective from `frequency` and `q`.

## Next

{py:obj}`gfem.Periodic`, {py:obj}`gfem.conventions`, {doc}`eigenmode-gradient`,
{doc}`shape-gradient`, {doc}`measurements`, {doc}`../capabilities`.
