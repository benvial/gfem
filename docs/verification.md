# How gfem is checked

Each claim on {doc}`capabilities` and in the recipes is backed by a test that
solves a case with a known answer: a closed form, an identity the physics
must obey, an independent method, or finite differences for a gradient. This
page lists those cases by area, in physical terms, and links the test files
on GitHub. The tolerance each check asserts is in its test, not on this page.

Most checks run on every change to gfem. Cases too large for that run before
each release, and the PETSc and SLEPc checks run twice a week; each section
says which.

## Gradients

Every FD-verified mark of the gradient matrix on {doc}`capabilities` is a set
of cells, one per problem, input, linear solver and sweep. Each cell compares
the gradient with fourth-order central differences of the same objective,
with tight solver tolerances. It then checks, at the default tolerances, that
the Taylor remainder shrinks at second order as the step shrinks.

```{gfem-gradient-checks}
```

The inputs are the per-element permittivity and conductivity (isotropic, and
as principal values on tilted axes), the fill of a sheet, and the mesh node
coordinates moved through `gfem.shape.boundary_morph`. The problems are
capacitance, driven S on lumped ports, on wave ports and between modes of a
shared port surface, driven S on periodic and Bloch cells, and
eigenfrequency and Q.

An exact zero is checked to be zero. Each "raises" mark, and each row of
"Gradients that raise", is a call that must raise the named error. Both the
matrix and that list are built from these tests when the docs are built, so
they cannot drift from them.

The same cells run on `linear_solver="petsc"` and `eigen_solver="slepc"`
where the {term}`PETSc stack` is installed (see
[PETSc and SLEPc](#verification-petsc)), so the matrix does not list PETSc.
The cells between modes of a shared port surface are among the large cases
that run before each release; the other cells run on every change.

Tests: [gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py),
[gradients that raise](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix_raise_table.py).

## Wave ports and shared port surfaces

- A matched parallel-plate line transmits its TEM mode, and its first higher
  mode, with their analytic propagation.
- An open dielectric guide between two wave ports transmits its fundamental
  mode with its slab's effective index, almost without reflection, and S is
  passive and reciprocal.
- A rectangular guide with a dielectric slab against one wall converts TE10
  into TE20. S, with both modes on each end's shared port surface, matches an
  independent mode-matching solution. It is unitary and reciprocal, and the
  power through each lead equals the sum of the requested modes' |S|².
- Before each release: TE0 and TM0 of a 500 × 220 nm silicon strip, requested
  on one port surface at each end, are each transmitted with their own phase,
  without reflection or conversion.

Tests: [shared port surface](https://github.com/benvial/gfem/blob/main/tests/verification/test_shared_port_surface.py),
[wave ports](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_wave_ports.py),
[silicon strip](https://github.com/benvial/gfem/blob/main/tests/reference/test_shared_port_surface.py).

## Periodic cells

Each case compares a periodic solve with a closed form:

- **A Bloch two-port line.** A lossy parallel-plate line periodic along x,
  with two 50 Ω lumped ports, gives the S of the closed-form Bloch impedance
  matrix at +k, −k and 0, on the direct and BDDC + GMRES solvers. S at +k is
  far from the closed form at −k, which fixes the sign of the Bloch phase.
- **The same line, loaded.** Loaded by its two ports, the line is a damped
  Bloch eigenmode problem. Its complex resonance is the root of
  det(Z(f, k) + R) = 0.
- **An empty parallel-plate cell periodic along x and y.** Its four lowest
  bands are those of the empty lattice and are real. The lowest band's field
  advances by exp(−i k·Δr) between two points.
- **A layered (Bragg) line.** Its bands are the transfer-matrix roots,
  lossless and lossy, at both edges of the gap at k = π/a, at π/2a, and at
  k = 0 with plain periodicity.
- **An empty cube periodic along x, y and z.** Its lowest band is the
  light-line doublet c|k|/2π.

Further checks: a wave vector of (0, 0, 0) gives the solution with no wave
vector, plain periodic walls give the same S as PMC walls on a field uniform
across the pair, and S(k)ᵀ = S(−k) ({term}`Bloch reciprocity`). On PETSc,
the plain line's S matches the direct solver's, SLEPc finds the Bragg line's
bands, and a Bloch operator is refused.

Tests: [driven](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_driven.py),
[eigenmode](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_eigen.py),
[identities, driven](https://github.com/benvial/gfem/blob/main/tests/unit/test_periodic_driven.py),
[identities, eigenmode](https://github.com/benvial/gfem/blob/main/tests/unit/test_periodic_eigen.py),
[PETSc](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_petsc.py).

## Anisotropic materials

- An absorbing boundary ending a biaxial, magnetic slab reflects as the
  half-space of transmission-line theory, for both polarizations and at two
  electrical slab thicknesses.
- A WR-28 guide filled with anisotropic ε and μ transmits its TE10 mode with
  the closed-form propagation constant, without reflecting. With its tensors
  turned about the guide axis, it does not reflect either.
- The port mode of that turned tensor propagates, with a propagation constant
  between those of the two guides filled with the tensor's principal values
  across the guide.
- The loss along each principal axis from {term}`principal participation`
  matches 1/Q from the complex eigenfrequency, on a material whose field lies
  off every principal axis.
- A model and its rigidly rotated copy, with the material given as rotated
  global matrices, give the same S, eigenfrequency and capacitance. A
  wave-port guide rotated to an oblique port normal gives the same S.
- A uniform anisotropic material field on tilted axes solves as the
  anisotropic material it holds.

Tests: [wave ports](https://github.com/benvial/gfem/blob/main/tests/verification/test_anisotropic_wave_ports.py),
[rotation](https://github.com/benvial/gfem/blob/main/tests/unit/test_anisotropic_rotation.py),
[absorbing boundary](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_boundaries.py),
[port mode](https://github.com/benvial/gfem/blob/main/tests/unit/test_port_mode.py),
[principal participation](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode.py),
[material field](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven.py).

## Patterned sheets

- On a parallel-plate line, a sheet of 2 µm of copper or of a 1 pH kinetic
  inductance, solid or with a void / solid pattern, takes BDDC + GMRES the
  iterations of an empty sheet, and its S matches the direct solver's.
- A copper pixel design graded across the default contrast passes the
  finite-difference check on both solvers.
- The whole design chain (pixels on a layer grid, filter, projection,
  exponential fill) differentiates end to end, and so do node coordinates
  that move the filled sheet.
- In a cavity with a resistive film of random fill, the eigenfrequency
  matches an independent iteration that refactorizes the operator at every
  step.

Tests: [sheets](https://github.com/benvial/gfem/blob/main/tests/verification/test_sheet_gradient.py).

(verification-petsc)=
## PETSc and SLEPc

- With no `petsc_options`, `linear_solver="petsc"` is an exact factorization:
  its results agree with `"direct"`.
- SLEPc's modes are refined as ARPACK's are: modes, Q, fields and gradients
  agree with ARPACK's.

These checks need the {term}`PETSc stack`. They run twice a week and on
every change to gfem's PETSc code.

Tests: [PETSc](https://github.com/benvial/gfem/blob/main/tests/unit/test_linalg_petsc.py),
[SLEPc](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode_slepc.py),
and the PETSc cells of the
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py).

## Several MPI ranks

Each thing a {term}`distributed solve` runs (see
{doc}`recipes/distributed-solve`) has a test launched on several ranks. It
compares the result with one process, on `"direct"` and on `"bddc_gmres"`
(where the iteration count must also match), and checks that every rank
holds the same result, bit for bit. Two things have no such test yet: a
lumped port terminated by an inductance or a capacitance, and
{py:obj}`gfem.characteristic_impedance`. They run the same code and do not
raise.

The launched tests on `"bddc_gmres"` run on every change. Those on
`"direct"` need the {term}`MPI PETSc stack` and run twice a week.

Tests: [launched tests](https://github.com/benvial/gfem/tree/main/tests/mpi).

### On many nodes

The runs of {doc}`project/scaling` were checked three ways:

- **Every rank count against one process.** On a 10,248,856 DOF coplanar
  line, from 1 to 32 nodes, each solver against its own one-process run: the
  first twelve rows below.
- **The two solvers against each other.** On the largest mesh both solvers
  finished, 47,277,960 DOF on 16 nodes, `bddc_gmres` against `direct`.
  `direct` did not finish at 108,570,046 DOF, so this check did not run
  there.
- **The largest mesh against the 10,248,856 DOF line**: the last row. The
  meshes differ, so this difference is an observation, not a check.

```{include} project/_scaling/agreement.md
```
