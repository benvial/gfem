# Periodic cells

Get the {term}`band diagram <band diagram>` of a structure that repeats along
one, two or three directions with {py:func}`gfem.band_diagram` (the
{doc}`photonic crystal tutorial <../tutorials/photonic-crystal>` works one
through), or the S-parameters of one phased unit cell, by solving a single cell
with its opposite faces paired by {py:obj}`gfem.Periodic`. Without a wave vector
the fields repeat ({term}`plain periodicity`); with `wave_vector=k` they pick up
a {term}`Bloch phase`. For plane waves see {doc}`floquet-ports`.

## Band diagram of a layered cell

A line of $\varepsilon = 2$ and $\varepsilon = 9$ layers that repeats every
10 mm along x. gmsh's `setPeriodic` makes the pair of faces that match node for
node.

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

- `bands.frequency` is $\operatorname{Re} \omega / 2\pi$ in Hz, one row per wave
  vector and one column per {term}`band`; `bands.q` is the Q.
- Put the target below half the lowest frequency of the highest band you ask
  for ({term}`anchored band window`), or `gfem.UnanchoredBandWindowWarning`
  fires and `bands.anchored` is False.
- At Γ the {term}`static band` is reported at 0 Hz.
- `bands.connection` follows a band across wave vectors ({term}`band
  connection`); a {term}`lost band` warns with `gfem.LostBandWarning`: add wave
  vectors there.
- The wave vector is Cartesian, in rad/m; under $e^{+i\omega t}$ the fields obey
  `E(r + a) = exp(-i k·a) E(r)` ({py:obj}`gfem.conventions`, Bloch periodicity).
  `gfem.wave_vector_path` builds wave vectors inside the first Brillouin zone;
  a component off the translations raises `ValueError`.
- One wave vector is
  `gfem.solve(cell, gfem.Eigenmode(target, n_modes=2, wave_vector=k))`.
- `keep_fields=[i]` keeps the fields at wave vector `i`; a Bloch mode's overall
  phase is arbitrary, so compare ratios.

## Other lattices, points and grids

`gfem.wave_vector_path` recognises the lattice from the cell's translations and
names its {term}`high-symmetry points <high-symmetry point>`, in reduced
coordinates of the {term}`reciprocal lattice` (`cell.reciprocal_lattice`):

| Lattice | Translations | Points besides Γ |
|---|---|---|
| line | one pair | X (1/2) |
| square | two pairs, equal and orthogonal | X (1/2, 0), M (1/2, 1/2) |
| rectangular | two pairs, orthogonal and unequal | X (1/2, 0), Y (0, 1/2), S (1/2, 1/2) |
| hexagonal | two pairs, equal, at 60° or 120° | M (1/2, 0), K at the corner of the zone |
| simple cubic | three pairs, equal and orthogonal | X (1/2, 0, 0), M (1/2, 1/2, 0), R (1/2, 1/2, 1/2) |

X lies along the first `gfem.Periodic` in `boundaries` order. Any other lattice
takes `named_points`; a face-centred cubic cell with translations
$(0, a/2, a/2)$, $(a/2, 0, a/2)$, $(a/2, a/2, 0)$ has the points of Setyawan
and Curtarolo (Comput. Mater. Sci. 49, 299 (2010)):

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

For a grid of wave vectors, make one `band_diagram` call per row as a straight
path: connection and lost bands are detected only along a path.

## Group velocity

- `bands.v_g` is the {term}`band group velocity` $\partial\omega/\partial\mathbf{k}$
  in m/s, shape `(n_k, n_bands, 3)`; `EigenmodeResult.v_g` gives it at one wave
  vector.
- The slope of the plotted band is
  `np.einsum("kbd,kd->kb", bands.v_g.real, tangent)`.
- `v_g` is NaN on the static band and where bands touch with different slopes,
  and complex on a lossy cell. It is not differentiable.

## Pair the faces

`gfem.Periodic` takes the names of the two faces. gfem measures the translation
between them from the mesh: `Model.periodic_translations` holds one per pair, in
metres, in `boundaries` order.

- A rotated pair, an element touching both faces, more than three pairs or two
  parallel translations raise `ValueError`; remesh.
- A paired face takes no other boundary condition and no port; use
  {py:obj}`gfem.FloquetPort` for plane-wave incidence on an open face.

## Drive a cell at a Bloch wave vector

A two-port line cell, $\varepsilon = 4$ with $\tan\delta = 0.02$, 10 mm long,
with lumped ports on two interior sheets:

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

- S is referenced to the field E ({py:obj}`gfem.conventions`) and obeys
  $S(\mathbf{k})^\mathsf{T} = S(-\mathbf{k})$ ({term}`Bloch reciprocity`).
- `wave_vector_frequency=f_ref` scales $\mathbf{k}$ with frequency, keeping an
  incidence angle fixed ({doc}`floquet-ports`).
- Only the uniform sweep takes a wave vector.

## Differentiate a band

The lowest band at the zone edge, with respect to the permittivity of each
element of the $\varepsilon = 9$ layer:

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

- {py:obj}`gfem.shape.boundary_morph` moves a node and its partner together.
- `wave_vector` is not differentiable (`gfem.NonDifferentiableError`).
- A pixel design takes {py:obj}`gfem.design.LayerGrid` with
  `periodic=(True, False)`; the pitch must divide the period.
- A gap differentiates eagerly, as in the
  {doc}`photonic crystal tutorial <../tutorials/photonic-crystal>`; a gap edge on
  a degenerate pair raises `gfem.DegenerateModeError`
  ({doc}`eigenmode-gradient`). {doc}`../capabilities` lists the supported
  gradients.

## Choose a solver

| | Plain periodic | Bloch wave vector |
|---|---|---|
| `direct` | driven, eigenmode | driven, eigenmode |
| `bddc_gmres` | driven, eigenmode | driven only |
| `petsc` | driven | raises: use `direct` |
| `device="cuda"` | untested | raises: use CPU `direct` |
| MPI ranks | one pair; eigenmode on `direct` | one pair, driven only |
| Floquet ports | driven uniform sweep on `direct`, `bddc_gmres` | the same |

## Check your results

- **Transpose.** `S(k).T` equals `S(−k)` to the solver tolerance; with Floquet
  ports the orders relabel ({doc}`floquet-ports`).
- **Zero wave vector.** `wave_vector=(0, 0, 0)` equals `None` but is slower.
- **Folding.** $\mathbf{k}$ and $\mathbf{k} + \mathbf{b}_1$, with $\mathbf{b}_1$
  in `cell.reciprocal_lattice`, give the same band frequencies.
- **Band gap.** A {term}`band gap` has both edges on path points, or it reads
  too wide.
- **Lossless cell.** `modes.omega` is real, and with resistive lumped ports
  `SᴴS ≈ I`.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **Oblique absorbers.** A {py:obj}`gfem.Absorbing` face reflects oblique waves
  $(1 - \cos\theta) / (1 + \cos\theta)$ in amplitude (7 % at 30°). A driven
  solve warns with {py:obj}`gfem.ObliqueAbsorberWarning`; an eigenmode solve
  does not. Use a {py:obj}`gfem.FloquetPort` or a
  {term}`perfectly matched layer` ({doc}`perfectly-matched-layer`).
- **Translations only**, one to three pairs.
- **MPI.** One pair, no Bloch eigenmode ({doc}`distributed-solve`).
- **Unsupported.** `gfem.Electrostatic` and `gfem.interface_participation` on a
  paired face raise; move nodes only through {py:obj}`gfem.shape.boundary_morph`.
- **One process.** `gfem.band_diagram` runs on one process, under `jax.grad` but
  not `jax.jit` or `jax.vmap`; a gradient through an eigenmode's field raises,
  so use `frequency` and `q`.

## Next

{doc}`../tutorials/photonic-crystal`, {py:func}`gfem.band_diagram`,
{doc}`floquet-ports`, {doc}`diffraction-efficiency`,
{doc}`eigenmode-gradient`, {doc}`shape-gradient`, {doc}`measurements`,
{doc}`../capabilities`.
