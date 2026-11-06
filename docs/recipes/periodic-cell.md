# Solve a periodic cell

A structure that repeats along one, two or three lattice directions is solved
on one cell, with one {py:obj}`gfem.Periodic` pair per direction. With no wave
vector on the problem, the fields repeat unchanged from cell to cell
({term}`plain periodicity`). With `wave_vector=k` on the problem, they pick up
a {term}`Bloch phase` from cell to cell, which gives band diagrams and phased
unit cells. Eigenmode and driven problems take both (a driven problem with a
wave vector on the uniform sweep only), and their gradients are checked
against finite differences like those of any model ({doc}`../capabilities`).

## Pair two surfaces

The mesh needs two exterior surfaces whose meshes match node for node under a
translation; gfem does not mesh, and gmsh's `setPeriodic`, for example, makes
such a pair. Name the two surfaces in a `gfem.Periodic` among the model's
boundaries:

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
```

- `Model.periodic_translations` holds one translation per pair, in
  `boundaries` order, measured from the mesh in metres from the pair's first
  surface to its second. The order of the two surfaces does not change the
  field.
- Building the model checks each pair and raises `ValueError` when the two
  meshes do not match under a translation (a rotated pair included), when an
  element touches both surfaces, when there are more than three pairs, or
  when two translations are parallel.
- A paired surface takes no other boundary condition and no port, and the
  default PMC skips it. A wave port may not touch a paired surface either
  (`NotImplementedError`).

## Conventions

Under the time dependence e^{+iωt}, the fields of a pair of translation a
obey `E(r + a) = exp(-i k·a) E(r)`, so the phase of a Bloch wave travels
along +k. The wave vector k is Cartesian, in rad/m. For a phase φ per cell
along a, k = φ a / |a|².

- k may only have components along the translations: any other component
  raises `ValueError` at the solve, and the message gives k's projection on
  them.
- Only the phases exp(−i k·a) enter the solve, so a k outside the first
  Brillouin zone gives the folded result.
- `wave_vector=(0, 0, 0)` solves with a Bloch phase of 1, and gives the
  solution with no wave vector (to a relative 1e-10 in the unit tests).

See {py:obj}`gfem.conventions` (Bloch periodicity).

## Band frequencies of a cell

On the cell above, a layered line of ε = 2 and ε = 9 that repeats every
10 mm along x, one eigenmode problem per k gives the band diagram:

```python
options = gfem.SolverOptions(order=3)
a = np.asarray(a)
bands = []
for phase in np.linspace(0.1 * np.pi, np.pi, 10):  # Bloch phase per cell, rad
    k = tuple(phase * a / (a @ a))  # rad/m
    # below every band: the light line of the densest layer, n = 3
    target = 0.9 * 299_792_458 * np.linalg.norm(k) / (2 * np.pi * 3)
    problem = gfem.Eigenmode(target=target, n_modes=2, wave_vector=k)
    modes = gfem.solve(cell, problem, options=options)
    bands.append(modes.frequency)  # Hz, the two lowest bands at this k
```

- `modes.omega` is exactly real for a lossless cell, and complex for a lossy
  or damped one; `modes.frequency` is Re ω / 2π and `modes.q` the Q.
- At Γ (k = 0, or no wave vector) this line also has a field at zero
  frequency: aim the target above it.
- Each mode field covers the whole mesh, with values on both surfaces of
  every pair, so `Field.at` reads it at any point of the cell.
- Under a Bloch wave vector, a mode's global phase is fixed by its largest
  coefficient: compare fields through ratios, not raw phases.
- One k per `Eigenmode`, so a band diagram is a loop over k; nothing carries
  over from one k to the next.

## A driven cell at a Bloch wave vector

A two-port line cell (ε = 4, tan δ = 0.02, 10 mm long, lumped ports on two
interior sheets), driven at one Bloch wave vector:

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
plus = gfem.solve(line, gfem.Driven(5e9, wave_vector=k)).s.values[0]
minus = gfem.solve(line, gfem.Driven(5e9, wave_vector=(-k[0], 0.0, 0.0))).s.values[0]
# plus.T equals minus to solver precision
```

- S is referenced to the field E, not to a periodic envelope.
  {py:obj}`gfem.conventions` gives the factor between the two. S is not
  symmetric: S(k)ᵀ = S(−k) ({term}`Bloch reciprocity`).
- k is the same at every frequency of the sweep.
- Only `sweep="uniform"` takes a wave vector: the adaptive sweep raises
  `NotImplementedError`, naming `sweep='uniform'`. Plain periodicity takes
  both sweeps.

## Gradients

A periodic model differentiates as any other. Here the lowest band at the
zone edge, with respect to the permittivity of the layer that touches the
pair:

```python
import jax
import jax.numpy as jnp

edge = (np.pi / 0.01, 0.0, 0.0)  # rad/m, k = π/a
problem = gfem.Eigenmode(target=4e9, n_modes=2, wave_vector=edge)


def lowest_band(eps):
    m = cell.with_material("high", gfem.MaterialField(eps_r=eps))
    return gfem.solve(m, problem, options=options).frequency[0]


n = len(mesh.elements("high"))
grad = jax.grad(lowest_band)(jnp.full(n, 9.0))
```

- {py:obj}`gfem.shape.boundary_morph` moves a node and its partner on the
  other surface of a pair together, and a paired surface cannot be the moving
  surface (`ValueError`).
- `wave_vector` is not differentiable: a traced value raises
  `gfem.NonDifferentiableError`.
- The rows "Driven, plain periodic", "Driven, Bloch", "Eigenmode, plain
  periodic" and "Eigenmode, Bloch" of the gradient matrix on
  {doc}`../capabilities` list what is checked against finite differences,
  and with which solvers and sweeps.

## Solvers

A Bloch operator is general (Aᵀ ≠ A), so it is factored with LU where a
plain periodic one is factored with LDLᵀ.

| | Plain periodic | Bloch |
|---|---|---|
| `direct` | driven and eigenmode | driven and eigenmode: MKL PARDISO LU with the `mkl` extra, UMFPACK otherwise |
| `bddc_gmres` | driven and eigenmode | driven; eigenmode raises `ValueError` (use `direct`) |
| `petsc` | driven | raises `ValueError` |
| cuDSS (`device="cuda"`) | not tested | raises `NotImplementedError` |
| eigen solver | ARPACK, SLEPc | ARPACK; SLEPc is not tested |

## How the results are checked

Each case below compares a solve with a closed form, at the tolerance its
test asserts (`tests/verification/test_periodic_driven.py`,
`tests/verification/test_periodic_eigen.py`):

- **C1**: S of the Bloch two-port line above against the closed-form Bloch
  impedance matrix, within 1e-5 at ±k and 0, on the direct and BDDC + GMRES
  solvers; S21(+k) is more than 0.5 from the closed form at −k, which pins the
  sign of the phase.
- **C1, loaded**: the same line loaded by its two 50 Ω ports is a damped
  Bloch eigenmode problem; its complex resonance matches the root of
  det(Z(f, k) + R) = 0 to a relative 1e-4.
- **C2**: the 4 lowest bands of an empty parallel-plate cell periodic along x
  and y match the empty lattice to a relative 3e-4 and are exactly real; the
  lowest band's field ratio between two points equals exp(−i k·Δr) within
  1e-2.
- **C3**: the layered line's bands match the transfer-matrix roots to a
  relative 2e-6, lossless and lossy, at both edges of the gap at k = π/a, at
  π/2a and plain at Γ.
- **C4**: the lowest band of an empty cube periodic along x, y and z matches
  the light-line doublet to a relative 1e-4.

The unit tests also check that a wave vector of (0, 0, 0) gives the solution
with no wave vector, that plain periodic walls give the same S as PMC walls
on a field uniform across the pair, and that S(k)ᵀ = S(−k).

## Limits

- No Floquet ports: no plane-wave incidence and no diffraction orders.
- Translations only, one to three pairs.
- `gfem.Electrostatic` on a periodic model raises `NotImplementedError`.
- `gfem.interface_participation` on a paired surface raises `ValueError`
  (the surface is interior to the repeated structure); `gfem.power_flow`
  works on sides that cross or touch a pair.
- k is fixed over a driven sweep.
- No derivative with respect to k (group velocity), and no k-path argument.
- A cotangent on an eigenmode field raises, as on every model.

## Next

{doc}`../capabilities`, {py:obj}`gfem.Periodic`, {py:obj}`gfem.conventions`,
{doc}`eigenmode-gradient`, {doc}`shape-gradient`, {doc}`measurements`.
