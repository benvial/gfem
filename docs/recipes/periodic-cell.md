# Periodic cells

Get the {term}`band diagram <band diagram>` of a structure that repeats along one,
two or three directions with {py:func}`gfem.band_diagram` (the
{doc}`photonic crystal tutorial <../tutorials/photonic-crystal>` works one
through), or the S-parameters of one phased unit cell, by solving a single
cell. You mesh one cell and pair its opposite faces with {py:obj}`gfem.Periodic`.
With no wave vector the fields repeat unchanged from cell to cell
({term}`plain periodicity`). With `wave_vector=k` they pick up a
{term}`Bloch phase` from cell to cell. To light the cell with a plane wave and
read its diffraction orders, see {doc}`floquet-ports`, and to sweep its
angle or read its Jones matrix, {doc}`diffraction-efficiency`.

## Band diagram of a layered cell

The cell below is a line of $\varepsilon = 2$ and $\varepsilon = 9$ layers that repeats every 10 mm
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
options = gfem.SolverOptions(order=3)

path = gfem.wave_vector_path(cell, ["Γ", "X"], n_points=11)  # 0 to π/a along x
bands = gfem.band_diagram(cell, path, target=2e9, n_bands=2, options=options)
bands.frequency  # (11, 2) Hz: band 0, the lowest, then band 1, at each wave vector
```

- `bands.frequency` is $\operatorname{Re} \omega / 2\pi$ in Hz, one row per wave vector and one
  column per {term}`band`. `bands.q` is the Q. For a lossless cell `bands.omega` is exactly real.
- The target is fixed. It should lie below half the lowest frequency of the highest
  band you ask for: the modes found are then the lowest ones, and band $n$ is the
  $n$-th lowest ({term}`anchored band window`). Otherwise `gfem.band_diagram`
  warns with `gfem.UnanchoredBandWindowWarning` and `bands.anchored` is False at those
  wave vectors.
- At Γ the cell also has a field at zero frequency. The {term}`static band` it
  stands for is reported at 0 Hz, and the other bands keep their numbers.
- The bands are in frequency order at each wave vector. `bands.connection` links
  each band to the one it continues at the previous wave vector, so a crossing
  can be followed ({term}`band connection`). A band with no partner at a
  neighbouring wave vector is a {term}`lost band`: it warns with
  `gfem.LostBandWarning` and is marked in `bands.lost`. Add wave vectors there.
- The wave vector is Cartesian, in rad/m. For a phase $\varphi$ per cell along a
  translation $\mathbf{a}$, $\mathbf{k} = \varphi \mathbf{a} / |\mathbf{a}|^2$. Under $e^{+i\omega t}$ the fields
  obey `E(r + a) = exp(-i k·a) E(r)` ({py:obj}`gfem.conventions`, Bloch
  periodicity). Only the phase per cell enters the solve, so gfem folds any
  $\mathbf{k}$ outside the first Brillouin zone into it; `gfem.wave_vector_path` builds its
  wave vectors inside the zone.
- Give $\mathbf{k}$ only along the translations. Any other component raises `ValueError`
  at the solve, and the message gives the projection to use instead.
- Each wave vector is an independent solve. One wave vector is still
  `gfem.solve(cell, gfem.Eigenmode(target, n_modes=2, wave_vector=k))`.
- `gfem.band_diagram(..., keep_fields=[i])` keeps the fields at wave vector `i`.
  `bands.e_field(i, band)` is a `Field`, and `Field.at` reads it at any point.
  Under a Bloch wave vector, gfem sets a mode's overall phase from its largest
  coefficient, not from the physics: compare fields through ratios, not raw
  phases.
- `bands.to_xarray()` gives a dataset with the dimensions `wave_vector` and `band`.

## Other lattices, points and grids

`gfem.wave_vector_path` recognises the lattice from the cell's translations and names
its {term}`high-symmetry points <high-symmetry point>`, in reduced coordinates of
the {term}`reciprocal lattice` (`cell.reciprocal_lattice`):

| Lattice | Translations | Points besides Γ |
|---|---|---|
| line | one pair | X (1/2) |
| square | two pairs, equal and orthogonal | X (1/2, 0), M (1/2, 1/2) |
| rectangular | two pairs, orthogonal and unequal | X (1/2, 0), Y (0, 1/2), S (1/2, 1/2) |
| hexagonal | two pairs, equal, at 60° or 120° | M (1/2, 0), K at the corner of the zone |
| simple cubic | three pairs, equal and orthogonal | X (1/2, 0, 0), M (1/2, 1/2, 0), R (1/2, 1/2, 1/2) |

X lies along the first `gfem.Periodic` in `boundaries` order. Any other lattice takes
`named_points`. A face-centred cubic cell with the primitive translations
$\mathbf{a}_1 = (0, a/2, a/2)$, $\mathbf{a}_2 = (a/2, 0, a/2)$, $\mathbf{a}_3 = (a/2, a/2, 0)$,
in `boundaries` order, has these points in the reduced coordinates of
Setyawan and Curtarolo (Comput. Mater. Sci. 49, 299 (2010)):

```python
fcc_points = {
    "X": (1 / 2, 0, 1 / 2),
    "L": (1 / 2, 1 / 2, 1 / 2),
    "W": (1 / 2, 1 / 4, 3 / 4),
    "K": (3 / 8, 3 / 8, 3 / 4),
}
path = gfem.wave_vector_path(
    fcc, ["Γ", "X", "W", "K", "Γ", "L"], n_points=61, named_points=fcc_points
)
```

For a grid of wave vectors, such as an equal-frequency map, build each row from
`cell.reciprocal_lattice` and sweep the grid row by row, each row as one
`band_diagram` call with its rows' wave vectors as a straight path: bands are
connected and lost bands detected between neighbouring wave vectors, which a raw
grid does not provide at the end of a row. A path with a break in it is two calls.

## Group velocity

- `bands.v_g` is the {term}`band group velocity` $\partial\omega/\partial\mathbf{k}$ in m/s, shape
  `(n_k, n_bands, 3)`, within the span of the translations. `gfem.solve` gives it at
  one wave vector as `EigenmodeResult.v_g`.
- The component along the path is
  `np.einsum("kbd,kd->kb", bands.v_g.real, tangent)`, with `tangent` the unit vector
  along each wave vector's segment: it is the slope of the plotted band.
- Light slows by the factor $n_g = c/|\mathbf{v}_g|$, written here as a ratio. gfem
  names no such quantity for a band.
- `v_g` is NaN on the static band and where bands touch with different slopes, and
  complex on a lossy cell. It is not differentiable.

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
  `NotImplementedError`: place the port's face away from the pair; for
  plane-wave incidence on an open face of the cell, use
  {py:obj}`gfem.FloquetPort`.

## Drive a cell at a Bloch wave vector

A two-port line cell, $\varepsilon = 4$ with $\tan\delta = 0.02$, 10 mm long, with lumped ports
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
  symmetric under a wave vector: $S(\mathbf{k})^\mathsf{T} = S(-\mathbf{k})$ ({term}`Bloch reciprocity`).
- By default the same $\mathbf{k}$ holds at every frequency;
  `wave_vector_frequency=f_ref` scales it with frequency, which keeps an
  incidence angle fixed ({doc}`floquet-ports`).
- Only the uniform sweep takes a wave vector. `sweep="adaptive"` with a wave
  vector raises `NotImplementedError`: use `sweep="uniform"`. Plain
  periodicity takes both sweeps.

## Differentiate a band

A periodic model differentiates like any other. Here the lowest band at the
zone edge, with respect to the permittivity of each element of the $\varepsilon = 9$
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
  `gfem.NonDifferentiableError`. The band group velocity is `v_g` (above); it is
  not differentiable, and nor is the wave vector.
- A pixel design on a cell takes {py:obj}`gfem.design.LayerGrid` with
  `periodic=(True, False)`, one bool per grid axis (x, y): along a periodic
  axis the pitch must divide the period, the filter smooths across the seam
  and the element map wraps around it.
- {doc}`../capabilities` lists which periodic gradients each solver supports.

A function of `gfem.band_diagram(...).omega`, such as a gap, differentiates the
same way: eagerly, not inside `jax.jit` or `jax.vmap`. The
{doc}`photonic crystal tutorial <../tutorials/photonic-crystal>` differentiates a
band gap with respect to the permittivity of every element. A gap edge on one member of
a degenerate pair raises `gfem.DegenerateModeError`
({doc}`eigenmode-gradient`).

## Choose a solver

| | Plain periodic | Bloch wave vector |
|---|---|---|
| `direct` | driven and eigenmode | driven and eigenmode (MKL PARDISO with the `mkl` extra, UMFPACK otherwise) |
| `bddc_gmres` | driven and eigenmode | driven only; eigenmode raises `ValueError`: use `direct` |
| `petsc` | driven | raises `ValueError`: use `direct` |
| `device="cuda"` | untested: compare with `direct` once on your cell | raises `NotImplementedError`: use `direct` on the CPU |
| several MPI ranks | one pair: driven on `direct` and `bddc_gmres`, eigenmode on `direct`; two or three pairs raise | one pair: driven on `direct` and `bddc_gmres`; eigenmode raises |
| eigen solver | ARPACK or SLEPc | ARPACK or SLEPc |
| Floquet ports | `direct` and `bddc_gmres`, driven uniform sweep only; every other entry raises | the same |

## Check your results

Six checks you can run on your own cell:

- **Transpose.** Solve the driven cell at $\mathbf{k}$ and at $-\mathbf{k}$. With lumped and wave
  ports, `S(k).T` should equal `S(−k)` to the solver tolerance. With Floquet
  ports, the orders relabel and take a sign:
  $S_{p\leftarrow q}(\mathbf{k}) = \sigma_p \sigma_q S_{-q\leftarrow -p}(-\mathbf{k})$, see {doc}`floquet-ports`.
- **Zero wave vector.** `wave_vector=(0, 0, 0)` should give the same result as
  no wave vector: a cheap check of the pairing and the ports. It is a check
  only: the zero wave vector still takes the general Bloch solve, so run plain
  periodicity with `wave_vector=None`, which is cheaper.
- **Folding.** With the same target, $\mathbf{k}$ and $\mathbf{k} + \mathbf{b}_1$, with
  $\mathbf{b}_1$ a vector of `cell.reciprocal_lattice`, should give the same band
  frequencies.
- **Degenerate bands.** At a high-symmetry point, bands that symmetry makes equal come
  back split by your mesh's asymmetry. A mesh with the cell's symmetry shrinks the
  split.
- **Band gap.** A {term}`band gap` is a range with no band at any wave vector of your path.
  Read it from the top of one band and the bottom of the next, and check that both edges
  sit on a path point; a path that misses an edge reads a gap that is too wide.
- **Lossless cell.** With no loss in the cell, `modes.omega` should be real,
  and the driven cell's S with resistive lumped ports should give `SᴴS ≈ I`:
  form the product and look at its largest departure from the identity.

How gfem itself is checked on periodic cells: {doc}`../verification`.

## Limits

- A {py:obj}`gfem.Absorbing` face reflects a wave that meets it obliquely.
  With a wave vector that has a component $\mathbf{k}_t$ along the face, the cell
  radiates at the angle $\theta$ from the face normal given by $\sin\theta = |\mathbf{k}_t| / (k_0 n)$, with $n$ the
  refractive index behind the face. The first-order boundary then reflects
  $(1 - \cos\theta) / (1 + \cos\theta)$ in amplitude: 1.7 % at 15°, 7 % at 30°, 17 % at
  45°. `order=2` lowers this about tenfold up to 30°, but only for the
  polarization with E parallel to the face and normal to $\mathbf{k}_t$; for the other
  polarization it changes nothing. A driven solve warns with
  {py:obj}`gfem.ObliqueAbsorberWarning` when some propagating wave meets the
  face with a reflection of at least 1 %; a {py:obj}`gfem.FloquetPort` on
  that face absorbs every order at any angle, and so does a
  {term}`perfectly matched layer` meshed beyond it, which also takes a lossy
  or layered substrate ({doc}`perfectly-matched-layer`).
- A Bloch eigenmode solve with `gfem.Absorbing` faces meets the same oblique
  reflection, and gives no warning: the Q of a mode that radiates at the
  angle $\theta$ includes it. Floquet ports do not take eigenmode solves; for the
  resonances and Q of a radiating cell, mesh a perfectly matched layer
  beyond the open faces instead, stretched along the non-periodic axis
  ({doc}`perfectly-matched-layer`), or sweep a driven problem with Floquet
  ports and read them off S.
- Translations only, one to three pairs. For a rotational symmetry, mesh the
  whole structure.
- Across MPI ranks, one pair only, and no Bloch eigenmode solve
  ({doc}`distributed-solve`).
- `gfem.Electrostatic` on a periodic model raises `NotImplementedError`. Mesh
  enough cells and solve without pairs.
- `gfem.interface_participation` on a paired face raises `ValueError`, since
  that face is inside the repeated structure. `gfem.power_flow` works on
  planes that cross or touch a pair.
- A gradient with respect to `mesh.nodes` is given per node, not summed over
  a node and its partner on the other surface of a pair. A move that breaks a
  pair raises `ValueError`; move nodes through
  {py:obj}`gfem.shape.boundary_morph`, which keeps the pairs.
- `gfem.band_diagram` runs on one process; under `jax.grad`, not inside `jax.jit` or
  `jax.vmap`. Port-loaded and open (PML) cells run with no claim beyond each wave
  vector's solve.
- A gradient through an eigenmode's field raises, as on every model. Build the
  objective from `frequency` and `q`.

## Next

{doc}`../tutorials/photonic-crystal`, {py:func}`gfem.band_diagram`,
{doc}`floquet-ports`, {doc}`diffraction-efficiency`,
{py:obj}`gfem.Periodic`, {py:obj}`gfem.conventions`,
{doc}`eigenmode-gradient`,
{doc}`shape-gradient`, {doc}`measurements`, {doc}`../capabilities`.
