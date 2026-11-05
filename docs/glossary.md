# Glossary

The words these docs use, with the meaning gfem gives them.

## Problems

```{glossary}
Problem type
  One of the physics analyses gfem solves: a {term}`driven <Driven problem>`,
  {term}`eigenmode <Eigenmode problem>`, electrostatic or
  {term}`port mode <Port mode problem>` problem.

Driven problem
  A frequency-domain full-wave analysis excited through {term}`ports <Port>`,
  producing network parameters and fields at given frequencies.

Adaptive sweep
  A driven frequency sweep that solves the full problem at a few
  {term}`sample frequencies <Sample frequency>` and evaluates a
  {term}`reduced model` over the requested grid.

Frequency worker
  A separate process that solves one contiguous block of a uniform
  {term}`driven <Driven problem>` sweep's frequencies with its share of the
  solve's threads (`SolverOptions.frequency_workers`); the blocks join back in
  frequency order.

Solve threads
  The CPU threads a solve runs on: `SolverOptions.threads` when given, else
  `NGS_NUM_THREADS`, else the physical cores the process may run on (its affinity
  mask), not every hardware thread of the machine. Under MPI ranks the cores are
  shared between the ranks holding that mask.

Sample frequency
  An output frequency at which an {term}`adaptive sweep` solved the full
  problem.

Reduced model
  The projection of a {term}`driven problem` onto a basis of its sampled
  solutions, used to evaluate network parameters and fields between
  {term}`sample frequencies <Sample frequency>`.

Eigenmode problem
  A source-free analysis finding resonant frequencies, quality factors and mode
  fields of a structure.

Nonlinear eigenmode problem
  An {term}`eigenmode problem` whose operator depends on frequency in a way that
  is not polynomial of degree at most 2 (conductor surface impedance,
  second-order {term}`absorbing boundary`, wave port), so eigenfrequencies are
  roots of a frequency-dependent system.

Port mode problem
  A 2D cross-section eigenmode analysis that supplies the field profile and
  propagation constant of a wave port.

Port mode sweep
  The {term}`port mode problem` solved over a driven problem's output
  frequencies before the 3D solve, giving each wave port's propagation constant
  and mode profile at every output frequency. The wave ports of a
  {term}`shared port surface` share one sweep.

Gradient verification matrix
  The set of cells (problem type × gradient input × linear solver × sweep)
  whose gradients are checked against fourth-order central differences and a
  second-order {term}`Taylor remainder <Taylor test>`. The {doc}`capabilities`
  page is generated from it.

Raise table
  For each problem type, every result leaf × model or problem input leaf with
  its expected outcome: a gradient, an exact zero, or a named error. Never a
  silent zero.
```

## Models

```{glossary}
Model
  Everything physical about a device independent of problem type: the mesh, a
  {term}`material` per {term}`region`, boundary conditions and
  {term}`ports <Port>`. One model feeds any problem type.

Region
  A named volume or surface of the mesh (a gmsh physical group), the only way
  materials, boundaries and ports refer to geometry.

Excluded region
  A volume {term}`region` present in the mesh but removed from the
  electromagnetic solve domain, its bounding surfaces acting as boundaries;
  typically a meshed conductor. Named either by region name or, when the mesh
  tags the conductor with the surrounding dielectric's name (gsim), by
  `gfem.enclosed_by(*surfaces)`, which selects every volume entity whose whole
  boundary lies on those surfaces.

Absorbing boundary
  A boundary condition on a truncating surface that approximates an open,
  non-reflecting exterior, of first or second order.

Transverse-anisotropic
  An {term}`anisotropic material` with one of its {term}`principal axes`
  along the surface normal at a port or boundary, so each tensor is
  block-diagonal in the surface frame (a 2x2 tangential block and a normal
  value). A wave port's cross-section may contain only transverse-anisotropic
  materials; tilted axes are allowed elsewhere.

Absorbing admittance
  The relative wave admittance a first-order {term}`absorbing boundary`
  applies to the tangential electric field on each face, set by the material
  behind the face and the face normal: a scalar for an isotropic material, a
  tangential tensor for an {term}`anisotropic <Anisotropic material>` one.

Air box
  The truncated exterior region between the device and the
  {term}`absorbing boundary`.

Slab model
  A {term}`model` one element thick between PEC plates, whose field is normal to
  the plates and uniform through the thickness, so it solves a 2D TM problem;
  how gfem poses top-view (effective-index) photonic and parallel-plate problems.

Mirror half model
  A {term}`model` cut on a symmetry plane of the stack and design, with a PMC or
  PEC on the plane chosen by the parity of the wanted mode; it solves half the
  volume for the same S-matrix.

TE-like mode
TM-like mode
  A guided mode of a strip waveguide whose transverse electric field lies mostly
  parallel to the wafer (TE-like) or normal to it (TM-like).

Port
  A named excitation and measurement surface, lumped or wave, whose order in the
  {term}`model` fixes the order of the network parameters.

Shared port surface
  An exterior surface that carries several wave ports, each requesting a
  different mode of one {term}`port mode problem`; together they form one
  boundary.

Requested mode
  The port mode, by rank at the lowest frequency, that a wave port excites and
  reports, with its own row and column of the network parameters. Wave ports
  that share a surface request different modes.

Skipped port mode
  A propagating mode of a surface's {term}`port mode problem`, ranked below the
  highest mode requested on that surface, that no wave port on the surface
  requests.

Mode cross-projection
  The conjugated S functional of one {term}`requested mode` applied to another
  requested mode of the same {term}`shared port surface`; zero for a lossless
  cross-section, otherwise the error it puts into S.

Terminal
  A conductor, given as one or more regions (volumes or surfaces), whose
  potential is set in an electrostatic problem; terminals index the capacitance
  matrix.

Solver options
  The numerical knobs of a solve (polynomial order, linear solver,
  {term}`eigen solver`, tolerances, {term}`device`), kept apart from the
  {term}`model` and problem type.

Device
  The hardware a solve's linear algebra runs on, the CPU or one CUDA GPU; part
  of the {term}`solver options`, never of the {term}`model`.

Eigen solver
  The iteration that finds an eigenmode problem's modes near the target,
  ARPACK or SLEPc; gfem refines and differentiates the modes the same way for
  both.

PETSc stack
  Complex PETSc and SLEPc with their Python bindings, built as
  {doc}`recipes/petsc-slepc` describes: what `linear_solver="petsc"` and
  `eigen_solver="slepc"` need. It comes in two builds, the
  {term}`no-MPI PETSc stack` and the {term}`MPI PETSc stack`.

Periodic pair
  Two exterior surfaces of a {term}`model` related by a translation, whose
  fields are identified; one to three pairs make a 1D, 2D or 3D periodic cell
  (`gfem.Periodic`).

Plain periodicity
  A {term}`periodic pair` whose fields are equal on both surfaces: a problem
  with no {term}`Bloch wave vector`.

Bloch wave vector
  The wave vector k (rad/m) set on an eigenmode or driven problem
  (`wave_vector=`) that gives each {term}`periodic pair` its
  {term}`Bloch phase`.

Bloch phase
  The factor exp(−i k·a) between the fields of a {term}`periodic pair`,
  E(r + a) = exp(−i k·a) E(r) under e^{+iωt}, with a the translation from the
  pair's first surface to its second.

Job
  A self-contained run, {term}`model` plus problem type plus
  {term}`solver options`, and the unit that is serialized.
```

## Results

```{glossary}
Field
  A solution quantity (electric field, potential) on the model's mesh, carried
  by a result with its quantity, unit, space, order and support.

Run record
  The committed outputs of a run too costly for CI (its solver outputs,
  settings, environment and the hash of the script that produced them), which a
  docs page reads instead of solving.

Partial result
  The result of a uniform sweep restricted to the frequencies it solved before
  the linear solver stopped converging, carried by the error that reports it
  (`gfem.LinearSolverConvergenceError`).

Unsolved frequency
  A frequency of a uniform sweep that has no entry in the
  {term}`partial result`: the linear solve did not converge there, or the sweep
  (or its {term}`frequency worker`'s block) stopped at an earlier frequency that
  did not converge.

Reference impedance
  The impedance per {term}`port` that S-parameters are normalized to: the port
  resistance for a lumped port, the power-voltage impedance for a wave port with
  a voltage path.

Participation ratio
  The fraction of a mode's electric or inductive energy stored in one region,
  port or interface.

Principal participation
  The fraction of a mode's electric energy stored in one
  {term}`anisotropic <Anisotropic material>` region by the field component
  along one of the region's {term}`principal axes`; the three sum to the
  region's {term}`participation ratio`.

Measurement
  A post-solve function of a result's {term}`fields <Field>` that needs extra
  physics input (a surface, a thickness, a point, a path).

Power flow
  The net time-averaged power crossing the boundary between a chosen set of
  elements and the rest of the solve domain, positive when it leaves the set.
  Measured by `gfem.power_flow` from the residual of the discrete equations, so
  exact for the solved field. Poynting flux is kept for the pointwise integral
  of ½Re(E × H*).

Bloch reciprocity
  S(k)ᵀ = S(−k): a Bloch cell's S matrix is not symmetric, but swapping two
  ports is the same as reversing the {term}`Bloch wave vector`.
```

## Materials and optimization

```{glossary}
Material
  A linear medium with constant relative permittivity, permeability, loss
  tangent and electrical conductivity (S/m), assigned to a whole
  {term}`region`; each property may be one value or, for an
  {term}`anisotropic material`, a tensor.

Anisotropic material
  A {term}`material` whose permittivity, permeability, loss tangent or
  conductivity depends on the field direction: real symmetric tensors, given
  as {term}`principal values` along the material's {term}`principal axes` or
  as symmetric 3x3 matrices, all sharing one set of principal axes.

Principal axes
  The three orthonormal directions (`axes`) shared by every tensor property
  of one {term}`anisotropic material`; along each one a property takes its
  {term}`principal value <Principal values>`.

Principal values
  The three values a tensor property takes along a material's
  {term}`principal axes`; equal principal values mean the property is
  isotropic.

Gyrotropic material
  A material with a non-symmetric (for example Hermitian) permittivity or
  permeability tensor, such as a magnetized ferrite or a magneto-optic
  crystal; it is non-reciprocal and not supported.

Material field
  A spatially varying assignment of material properties (permittivity,
  permeability, conductivity) over the mesh, possibly derived from
  {term}`design parameters`; an {term}`anisotropic material field` holds
  principal values on fixed axes.

Anisotropic material field
  A {term}`material field` whose tensor properties are
  {term}`principal values` per element along one fixed set of
  {term}`principal axes` shared by the whole field; the principal values are
  design inputs, the axes are not.

Sheet
  A zero-thickness conducting layer on an interior surface, described by its
  admittance per square.

Sheet field
  A spatially varying fill over the faces of a named interior surface, scaling a
  {term}`sheet`'s admittance from none (0) to full (1); the surface counterpart
  of a {term}`material field`.

Layer grid
  A planar pixel grid of {term}`design parameters` over a named volume's
  footprint, each value held constant through the volume's thickness and mapped
  onto the volume's elements.

Design parameters
  The vector of quantities an optimization adjusts, such as per-element
  densities or mesh node positions, upstream of the {term}`material field` or
  geometry.

Topology optimization
  Optimization where {term}`design parameters` are densities interpolated into
  a {term}`material field` on a fixed mesh.

Shape optimization
  Optimization where {term}`design parameters` move mesh node positions, so
  material regions keep sharp interfaces.

Boundary morph
  The fixed linear map from normal displacements of a
  {term}`moving surface`'s nodes to moved mesh coordinates.

Moving surface
  The selected surface whose nodes carry the shape {term}`design parameters`.

Sliding surface
  A surface whose nodes may move only within their own facet plane during a
  {term}`boundary morph`. On a {term}`periodic pair`, a node and its partner
  move together.

Mesh extension
  The elastic or harmonic solve that carries boundary motion into the mesh
  interior.

Element quality
  The scaled Jacobian of an element; zero or below means inverted.

Remesh and restart
  Regenerating the mesh at the current {term}`design parameters` and restarting
  the optimizer's history.

Objective
  A scalar function of a solution (S-parameter, eigenfrequency, Q, field
  overlap) whose gradient with respect to {term}`design parameters` is
  requested.

Taylor test
  A gradient check that the first-order Taylor remainder of an
  {term}`objective` shrinks at second order as the perturbation shrinks.

Cluster-symmetric objective
  An {term}`objective` that depends on a degenerate group of modes only through
  quantities unchanged when the modes are relabelled, such as their sum, so its
  gradient exists where individual eigenvalue gradients do not.

Non-differentiable quantity
  A result value or model input that gfem refuses to differentiate in a given
  version, raising an error instead of returning a zero gradient.
```

## Distributed runs

```{glossary}
Distributed solve
  A solve whose mesh and linear system are split across the ranks of a communicator
  with more than one rank; every rank calls it with the same values and gets the
  same result.

Rank copies
  Independent complete solves, one per launched process, each on its own single-rank
  communicator, asked for by name.

Agreed value
  A value every rank holds bitwise: a replicated input, the result of a reduction, or
  a broadcast. The only kind of value that may decide what a {term}`distributed solve`
  does next.

Agreed error
  A failure all ranks decide on together, so that it raises on every rank and the
  script may catch it; any other failure on a rank ends the job.

Field piece
  The part of a field's values one rank holds; a value shared between ranks is the
  same on every rank holding it.

Rank agreement
  A distributed result that is bitwise the same on every rank and equal, within its
  stated tolerance, to the one-process result of the same model and options.

Distributed refusal
  A solve or method call on more than one rank that raises on every rank, naming what
  has no distributed path and the remedy; it never runs on fewer ranks instead.

No-MPI PETSc stack
  The {term}`PETSc stack` with MUMPS and METIS and no MPI: it runs in one process, or
  as {term}`rank copies`, needs no MPI library, and has no part in a
  {term}`distributed solve`.

MPI PETSc stack
  The {term}`PETSc stack` built on one Open MPI, with MUMPS, ScaLAPACK, PT-Scotch and
  METIS, 64-bit indices and no ParMETIS: what a {term}`distributed solve` through
  PETSc needs. It also runs one-process solves.

Distributed coarse factorization
  BDDC's coarse factorization across ranks on PETSc MUMPS.

Rank-0 coarse factorization
  BDDC's coarse factorization gathered to rank 0, the only route without PETSc.
```
