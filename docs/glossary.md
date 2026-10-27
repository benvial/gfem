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
  and mode profile at every output frequency.

Reference problem
  A fixed geometry plus problem type, solved with both Palace and gfem and
  compared.

Agreement with Palace
  The tolerance a {term}`reference problem` asserts between gfem and Palace,
  stated per quantity as relative or absolute.

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
  Complex PETSc and SLEPc with their Python bindings, built without MPI as
  {doc}`recipes/petsc-slepc` describes: what `linear_solver="petsc"` and
  `eigen_solver="slepc"` need.

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

Reference impedance
  The impedance per {term}`port` that S-parameters are normalized to: the port
  resistance for a lumped port, the power-voltage impedance for a wave port with
  a voltage path.

Participation ratio
  The fraction of a mode's electric or inductive energy stored in one region,
  port or interface.

Measurement
  A post-solve function of a result's {term}`fields <Field>` that needs extra
  physics input (a surface, a thickness, a point, a path).
```

## Materials and optimization

```{glossary}
Material
  A linear medium with constant relative permittivity, permeability, loss
  tangent and electrical conductivity (S/m), assigned to a whole
  {term}`region`.

Material field
  A spatially varying assignment of material properties (permittivity,
  permeability, conductivity) over the mesh, possibly derived from
  {term}`design parameters`.

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
  {term}`boundary morph`.

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
