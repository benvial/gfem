# Glossary

The words these docs use, with the meaning gfem gives them.

## Problems

```{glossary}
Problem type
  One of the physics analyses gfem solves: a {term}`driven <Driven problem>`,
  {term}`eigenmode <Eigenmode problem>`, electrostatic or
  {term}`port mode <Port mode problem>` problem.

Driven problem
  A frequency-domain full-wave analysis excited through {term}`ports <Port>`
  and {term}`sources <Source>`, producing fields at given frequencies, and
  network parameters when ports are excited.

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

Port mode
  A wave travelling along a guide, found by a {term}`port mode problem`: the
  frequency is given and the propagation constant $\gamma = \alpha + i\beta$
  is found. $\alpha > 0$ when the cross-section is lossy or the mode is a
  {term}`leaky mode`. Not a {term}`resonant mode` (see {doc}`modes-and-ports`).

Bound mode
  A {term}`port mode` whose field decays away from the guide: below the light
  line of every cladding, $\beta > k_0 n$. Lossless, it has $\alpha = 0$. The
  guided mode of a silicon strip waveguide is bound although the structure is
  open.

Leaky mode
  A mode that radiates out of its guide as it travels: a strip waveguide over
  a thin buried oxide, a bend. As a {term}`port mode` it has $\alpha > 0$ with
  lossless materials, and its field grows away from the guide, so the section
  needs an absorbing edge or a {term}`perfectly matched layer`. At a fixed
  wave number the same wave is a resonant mode with
  $\mathrm{Im}\,\omega \approx v_g \alpha$.

Group index
  $n_g = c\,\mathrm{d}\beta/\mathrm{d}\omega$ of a {term}`port mode`: a
  pulse's envelope travels along the guide at $c/n_g$. It holds the
  dispersion of the materials and of the guide's shape; on a lossless
  section it is $c$ times the stored energy over the power, with the
  {term}`energy permittivity` in law regions. Complex on a lossy or leaky
  section, written $n_g' - in_g''$ like the effective index, with
  $n_g' = c\,\mathrm{d}\beta/\mathrm{d}\omega$.

Group-velocity dispersion
  $\beta_2 = \mathrm{d}^2\beta/\mathrm{d}\omega^2$ of a {term}`port mode`, in
  s²/m: how the delay per length changes with frequency. Normal where
  $\beta_2 > 0$, anomalous where $\beta_2 < 0$. Complex on a lossy or leaky
  section, with real part $\mathrm{d}^2\beta/\mathrm{d}\omega^2$. Short form
  GVD.

Dispersion parameter
  $D = -(2\pi c/\lambda^2)\,\beta_2$, with $\lambda$ the vacuum wavelength,
  in s/m² (1 ps/(nm km) is $10^{-6}$ s/m²): the change of delay per length
  per wavelength. Positive is anomalous dispersion.

Third-order dispersion
  $\beta_3 = \mathrm{d}^3\beta/\mathrm{d}\omega^3$ of a {term}`port mode`, in
  s³/m. Not the dispersion slope $\mathrm{d}D/\mathrm{d}\lambda$.

Kerr nonlinear coefficient
  $\gamma_\mathrm{NL}$ of a {term}`port mode`, in 1/(W m): a power $P$ moves
  its propagation constant by $\mathrm{Re}\,\gamma_\mathrm{NL}\,P$ and its
  field attenuation by $\mathrm{Im}\,\gamma_\mathrm{NL}\,P$ (two-photon
  loss). Short form: nonlinear coefficient.

Normalized efficiency
  $\eta_\mathrm{norm} = |\kappa|^2$ of a pump mode and its second harmonic, in
  1/(W m²): the harmonic power is $\eta_\mathrm{norm}P^2L^2$ for a pump of power
  $P$ in a lossless guide of length $L$ at perfect phase matching.

Resonant mode
  A source-free field of the whole model, found by an
  {term}`eigenmode problem`: its complex angular frequency $\omega$ is found,
  with $\mathrm{Im}\,\omega > 0$ for a decaying mode and
  $Q = |\omega| / (2\, \mathrm{Im}\,\omega)$. Real in a closed lossless
  structure.

Quasi-normal mode
  A {term}`resonant mode` of an open structure, one that loses power through a
  {term}`perfectly matched layer`, an absorbing boundary, a resistive
  {term}`lumped port` or a {term}`wave port`: complex $\omega$, and a field
  that grows away from the structure. Short form QNM. A cavity coupled to a
  guide has quasi-normal modes; the guide's own {term}`bound mode` is not one.

Coupling Q
  The Q a {term}`resonant mode` would have if one port were its only loss:
  $\mathrm{Re}\,\omega$ times the mode's stored energy over the
  time-averaged power the port takes out. For a resistive {term}`lumped port`
  that power is the one its resistance draws; for a {term}`wave port`, the
  one its requested mode carries out through the face. With other losses,
  $1/Q$ is close to the sum of the ports' $1/Q_\mathrm{coupling}$ plus the
  material and wall losses, closer as Q grows. Reported in
  `EigenmodeResult.q_port`.

Outgoing branch
  The propagation constant of a {term}`port mode` continued from real
  frequencies above cutoff to the complex frequency of a resonant mode: the
  wave leaving the model through the port. At a decaying mode
  $\mathrm{Re}\,\gamma < 0$, so the field grows along the guide away from
  the model, as a {term}`quasi-normal mode`'s does. Below cutoff it also has
  real roots that are not resonances (anti-bound roots), which an
  eigenmode solve drops with a warning.

Requested power share
  For an eigenmode solve with {term}`wave port`s, the part of the power a
  port surface absorbs from a mode that the surface's requested modes carry
  out, $\eta_s = \sum_p P_p / P_s$, with $P_p$ the power port $p$'s
  requested mode carries and $P_s$ the surface's total. Near 1 for a mode of
  a requested family; the rest of $1 - \eta_s$ leaves through a field no port
  requested, which the port absorbs only partly. Reported in
  `EigenmodeResult.requested_power_share`.

Trapped mode
  A {term}`resonant mode` held inside the model, against a {term}`wave port`,
  by a port mode family that propagates in the model but is cut off at the
  port. Its loss is the port's absorption of the evanescent tail, so its Q is
  a lower bound on the device's and grows as the port moves away. Its
  {term}`requested power share` is near 0.

Unrequested port mode
  A mode of a port's guide that no {term}`wave port` on the surface requests.
  The port absorbs it only partly, reflecting the rest, so a resonance it
  carries has a Q that is the port's, not the device's. Request the mode on
  the surface to read where its power goes.

Port mode sweep
  The {term}`port mode problem` solved over a driven problem's output
  frequencies, or over a band around an eigenmode problem's target, before
  the 3D solve, giving each wave port's propagation constant and mode profile
  at every frequency the solve needs. The wave ports of a
  {term}`shared port surface` share one sweep.

Unresolved band
  The frequencies at which a region with a {term}`permittivity law` carries a
  wavelength shorter than two of its longest element edges,
  $k_0\, \mathrm{Re} \sqrt{\varepsilon(f)}\, h_\mathrm{max} > \pi$. An eigenmode target or mode there raises
  `ValueError`: near a Lorentz resonance the modes crowd so densely that which
  ones are nearest depends on the mesh.

Modal overlap
  The number of a lossy region's cavity modes within one linewidth of a
  frequency, about $8\pi (V / \lambda_m^3) |\mathrm{Im}\,\varepsilon| / \mathrm{Re}\,\varepsilon$, with $V$ the region's volume and
  $\lambda_m$ the wavelength in it. Above about 10 the modes nearest a target cannot
  be told apart, and an {term}`eigenmode problem` there may stop with
  `gfem.EigenmodeConvergenceError`.
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
  non-reflecting exterior, of first or second order. It reflects a wave that
  meets it obliquely; on a periodic cell a driven solve warns
  (`gfem.ObliqueAbsorberWarning`) when that reflection reaches 1 %. A
  {term}`perfectly matched layer` absorbs at any angle.

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
  {term}`absorbing boundary` or the {term}`perfectly matched layer`.

Perfectly matched layer
  A meshed shell of volume regions around the {term}`air box` that absorbs
  outgoing waves at any angle (`gfem.PML`); short form PML. Each layer region
  keeps the material it continues, which solves that ignore the layer
  (electrostatic, DC conduction, thermal) see. The fields inside it are not
  physical.

Target reflection
  The reflection $R_0$ a {term}`perfectly matched layer` is designed for, at
  normal incidence and at its {term}`reference frequency`;
  `PML.target_reflection`, $10^{-6}$ by default.

Reference frequency
  The frequency at which a {term}`perfectly matched layer` meets its
  {term}`target reflection`: `PML.reference_frequency` when given, else the
  lowest frequency of the solve or the eigenmode target. Above it the layer
  reflects less.

Interior box
  The bounding box of the solve domain outside the
  {term}`perfectly matched layer`. The layer stretches beyond its faces, and
  each side's thickness is measured from them.

Physical region
  A volume {term}`region` of the solve domain outside the
  {term}`perfectly matched layer`. Energies, participations, the mode
  normalization and a port mode's power count physical regions only.

PML mode
  A mode that lives in a {term}`perfectly matched layer` rather than in the
  device, and moves when the layer changes. Eigenmode and port mode solves
  drop it.

Clearance
  The distance from a guide's core to the edge of its section: a side
  {term}`perfectly matched layer`, or the section's outer wall. It sets how
  much the edge perturbs the guided mode: about
  $10^{-2} e^{-2\gamma_t c}$ in $n_\text{eff}$ at a layer, with $\gamma_t$ the
  mode's transverse decay rate and $c$ the clearance, and far more in the
  {term}`group-velocity dispersion`.

Slab model
  A {term}`model` one element thick between PEC plates, whose field is normal to
  the plates and uniform through the thickness, so it solves the
  {term}`Ez polarization` of a {term}`2D model`'s problem on a 3D mesh (between
  natural faces (PMC) it solves the Hz polarization instead); the
  cross-check of 2D models, and the way to pose what they refuse (oblique
  incidence, periodic cells, lumped ports, sheets).

2D model
  A {term}`model` whose mesh is a plane of triangles: neither the structure
  nor the field varies along z, the normal to the plane, and a driven solve
  computes one {term}`polarization <Ez polarization>`. Its powers and fields
  are per metre along z.

Hz polarization
Ez polarization
  The field component normal to the plane of a {term}`2D model`, the unknown of
  its driven solve (`Driven(polarization=...)`): Ez has E normal to the plane
  and H in it, Hz the reverse.

Line port
  A {term}`wave port` on a straight exterior line of a {term}`2D model`, fed by
  the mode of that line's cross-section; it carries 1 W/m.

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
  {term}`model` fixes the order of the network parameters. It has three roles:
  a boundary in every problem (it absorbs), a source when excited, and a row of
  the S-parameters always.

Excited port
  A {term}`port` that a {term}`driven problem` feeds, in turn, with 1 W of
  incident power, giving one column of the S-parameters (`Driven(excite=...)`,
  every port by default). A port that is not excited still terminates the
  device and still has its row of S.

Lumped port
  A {term}`port` on a surface that spans from one conductor to another, or on
  several such surfaces driven together: a resistor, with an optional
  inductance and capacitance in parallel, driven along a stated direction. Its
  resistance is its {term}`reference impedance`.

Lumped value
  The resistance, inductance or capacitance of a {term}`lumped port` or of an
  `Impedance` {term}`sheet`. Those that enter only the operator (a port's
  inductance and capacitance, a sheet's three values) are design inputs; a
  lumped port's resistance is not, because it is also its
  {term}`reference impedance`.

Wave port
  A {term}`port` on an exterior surface, fed by one mode of the surface's
  cross-section, which a {term}`port mode problem` supplies. It absorbs that
  mode without reflection.

Polarity reference
  A point on a {term}`wave port` and a direction along which the port mode's
  electric field (on a line port, the field normal to the plane) is real and
  positive at every frequency (`WavePort(polarity=...)`). It fixes the sign of
  the port's row and column of S without defining a voltage or an impedance.

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

Floquet port
  A {term}`port` on a {term}`Floquet surface` that is one
  {term}`diffraction order` in one polarization (TE or TM), with its own row
  and column of the S-parameters (`gfem.FloquetPort`).

Floquet surface
  An open face of a periodic cell, closed by
  {term}`Floquet ports <Floquet port>`: a planar exterior face whose rim lies on the seams of two
  {term}`periodic pairs <Periodic pair>` spanning it, with one homogeneous,
  lossless, isotropic medium behind it. Its ports together absorb every
  diffraction order at any angle.

Diffraction order
  One of the plane waves a periodic cell sends into the medium behind a face:
  order $(m, n)$ has the in-plane wave vector $\boldsymbol{\kappa} = \mathbf{k}_t + m \mathbf{b}_1 + n \mathbf{b}_2$, with $\mathbf{b}_1$, $\mathbf{b}_2$
  the reciprocal vectors of the face's two pairs and $\mathbf{k}_t$ the
  {term}`Bloch wave vector`. It propagates when $|\boldsymbol{\kappa}|$ is below the medium's
  wavenumber, and decays away from the face otherwise.

Diffraction efficiency
  The fraction of the incident power per cell that leaves in one
  {term}`diffraction order` and polarization: $|S|^2$ of its
  {term}`Floquet port`'s row.

Rayleigh anomaly
  The frequency or angle at which a {term}`diffraction order` grazes its face
  ($|\boldsymbol{\kappa}|$ equal to the medium's wavenumber), changing from evanescent to
  propagating. Efficiencies change steeply there.

Jones matrix
  The $2\times2$ block of a cell's S between one outgoing and one incoming
  {term}`diffraction order`, with rows and columns in (TM, TE). Its entries
  are power-wave ratios, so $|J_{ij}|^2$ is a {term}`diffraction efficiency`.
  At normal incidence it is the usual Jones matrix in the cell's own axes.

Absorptance
  The fraction of the incident power a cell dissipates: one minus the
  reflected, transmitted and port-coupled fractions. It is defined when every
  propagating {term}`diffraction order` has a {term}`Floquet port`.

Angle sweep
  A list of incidence directions on one {term}`Floquet surface`, each solved
  over the same frequencies with the same ports, so that its results stack
  over direction and frequency (`gfem.diffraction.angle_sweep`).

Conical incidence
  Oblique incidence whose plane is not perpendicular to a one-dimensional
  grating's lines, so the {term}`diffraction orders <Diffraction order>`
  leave on a cone and TE and TM couple.

Emissivity
  The fraction of the blackbody radiance a surface emits toward a direction,
  in one polarization, at one frequency. By Kirchhoff's law it is the
  {term}`absorptance` of the wave arriving from that direction.

Hemispherical emissivity
  The $\cos\theta$-weighted mean of the {term}`emissivity` over every
  direction above a surface, $\frac1\pi\int e\cos\theta\,d\Omega$.

Cell response
  The S of a periodic cell's {term}`Floquet ports <Floquet port>` sampled
  against one parameter of its meta-atom (a width, a radius) over frequency,
  with the derivative of S with respect to the parameter, interpolated between
  the samples (`gfem.diffraction.cell_response`).

Aperture field
  The tangential electric field on a metasurface's outgoing reference plane in
  the {term}`locally periodic approximation`: one sample per cell, at its
  centre, from the cell's {term}`Jones matrix` at its parameter
  (`gfem.diffraction.aperture_field`).

Locally periodic approximation
  Treating each cell of a slowly varying surface as if it sat in an infinite
  array of itself, so that it radiates one plane wave and the coupling to its
  neighbours is neglected. The error grows with the phase step between
  neighbouring cells.

Dispersive cross-section
  A wave port's cross-section that contains a region with a
  {term}`permittivity law`. It is supported where $\mathrm{Re}\,\varepsilon(f) > 0$ throughout, at
  every frequency of the sweep; a metal in it raises.

Terminal
  A conductor, given as one or more regions (volumes or surfaces), whose
  potential is set in an electrostatic problem; terminals index the capacitance
  matrix.

Solver options
  The numerical knobs of a solve ({term}`element order`, linear solver,
  {term}`eigen solver`, tolerances, {term}`device`), kept apart from the
  {term}`model` and problem type.

Element order
  The polynomial order of the finite elements, `SolverOptions.order`, 2 by
  default and at most 12 (the order of the Nédélec elements that carry the
  electric field). A higher order is more accurate on the same mesh and costs
  more time and memory.

Device
  The hardware a solve's linear algebra runs on, the CPU or one CUDA GPU; part
  of the {term}`solver options`, never of the {term}`model`.

P-multigrid
  The preconditioner of `linear_solver="pmg_gmres"`. It solves the
  {term}`direct set` exactly and smooths the remaining higher-order unknowns
  patch by patch around each mesh vertex, under GMRES. Its iteration count
  barely grows with a model's size in wavelengths.

Direct set
  The unknowns the {term}`p-multigrid` solves exactly: the lowest-order
  (Whitney) unknown of every edge, and every unknown of a cell in a
  {term}`perfectly matched layer` or of a {term}`poor cell`. Its share of the
  unknowns, logged at each build, sets how close the solver's memory comes to
  the direct solver's.

Poor cell
  A tetrahedron whose longest edge exceeds ten times its shortest altitude,
  $L_\mathrm{max} > 10\, h_\mathrm{min}$ with $h_\mathrm{min} = 3V / S_\mathrm{max}$
  ($V$ its volume, $S_\mathrm{max}$ its largest face area), as in a thin film
  meshed one cell across. The {term}`p-multigrid` puts its unknowns on the
  {term}`direct set`, because layers of such cells stall its smoothing.

Tree-cotree gauge
  A change of basis for a driven solve's unknowns that writes the gradients
  $\nabla \phi$, on which the curl-curl term vanishes, as node potentials along
  a spanning tree of the mesh and one floating potential per conductor, so
  that a direct solve keeps its digits below the low-frequency floor
  ($k_0 h_\mathrm{min} < 10^{-6}$). On by default for a direct solve on the
  CPU in one process (`SolverOptions.gauge`); the wave ports' mode problem is
  not gauged.

Eigen solver
  The iteration that finds an eigenmode problem's modes near the target,
  ARPACK or SLEPc (SLEPc only in a {term}`distributed solve`); gfem refines
  and differentiates the modes the same way for both.

PETSc stack
  Complex PETSc and SLEPc with their Python bindings, built as
  {doc}`recipes/petsc-slepc` describes: what `linear_solver="petsc"` and
  `eigen_solver="slepc"` need. It comes in two builds, the
  {term}`no-MPI PETSc stack` and the {term}`MPI PETSc stack`.

No-MPI PETSc stack
  The {term}`PETSc stack` for one process: what `linear_solver="petsc"` and
  `eigen_solver="slepc"` need on a workstation or as {term}`rank copies`. It
  needs no MPI library and cannot run a {term}`distributed solve`.

MPI PETSc stack
  The {term}`PETSc stack` built against your Open MPI: what a
  {term}`distributed solve` with `"direct"` or `"petsc"` needs. It runs
  one-process solves too, and only under the Open MPI it was built on.

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
  The factor $e^{-i \mathbf{k} \cdot \mathbf{a}}$ between the fields of a {term}`periodic pair`,
  $\mathbf{E}(\mathbf{r} + \mathbf{a}) = e^{-i \mathbf{k} \cdot \mathbf{a}} \mathbf{E}(\mathbf{r})$ under $e^{+i\omega t}$, with $\mathbf{a}$ the translation from the
  pair's first surface to its second.

Reciprocal lattice
  The vectors $\mathbf{b}_i$ with $\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi\delta_{ij}$,
  inside the span of the cell's translations $\mathbf{a}_j$ (`Model.reciprocal_lattice`).

High-symmetry point
  A named {term}`Bloch wave vector` of the first Brillouin zone (Γ, X, M, K,
  R…), given in reduced coordinates of the {term}`reciprocal lattice`.

Wave vector path
  The Bloch wave vectors a {term}`band diagram` samples: straight segments
  between {term}`high-symmetry points <high-symmetry point>`, evenly spaced by
  path length, with every high-symmetry point among them
  (`gfem.wave_vector_path`).

Band
  The $n$-th lowest resonant frequency (by $\mathrm{Re}\,\omega$) of a periodic cell as
  a function of the {term}`Bloch wave vector`, counting the bands that reach
  zero frequency at Γ. It is continuous in $\mathbf{k}$, with a kink where two bands
  cross. Bands are numbered from 0.

Static band
  A {term}`band` whose frequency goes to zero at Γ, where the cell's static
  fields stand in for it. A band diagram reports it at zero frequency there.

Band connection
  The link from a {term}`band` at one wave vector of a {term}`wave vector path`
  to the band at the previous wave vector that holds the same mode, found by
  field overlap. Through a crossing it joins band $n$ to band $n+1$; inside a
  degenerate or near-degenerate group it falls back to frequency order.

Band diagram
  The {term}`bands <band>` of a periodic cell along a {term}`wave vector path`
  (`gfem.band_diagram`).

Band gap
  A frequency range in which a periodic cell has no {term}`band`, in the
  polarization considered, at any Bloch wave vector; complete when no
  polarization has one. Its width over its centre frequency,
  $\Delta\omega/\omega_m$, does not depend on the lattice's scale.

Band group velocity
  The gradient $\partial\omega/\partial\mathbf{k}$ of a {term}`band`, within the
  span of the periodic translations: the exact derivative of the discrete
  band, complex on lossy cells, and undefined (NaN) where bands touch with
  different slopes. It is not a port mode's group index.

Anchored band window
  The modes found at a wave vector when the target lies below half the highest
  of them: they are then the lowest {term}`bands <band>`, and their frequency
  order is the band number.

Lost band
  A {term}`band` found at one wave vector of a {term}`band diagram` that no mode
  continues at a neighbouring wave vector: the path is too coarse there, or the
  solve missed a mode.

Job
  A self-contained run, {term}`model` plus problem type plus
  {term}`solver options`, and the unit that is serialized.
```

## Results

```{glossary}
Field
  A solution quantity (electric field, potential) on the model's mesh, carried
  by a result with its quantity, unit, space, order and support.

Partial result
  The result of a uniform sweep restricted to the frequencies it solved before
  the linear solver stopped converging, carried by the error that reports it
  (`gfem.LinearSolverConvergenceError`).

Result bundle
  The files a {term}`job`'s run writes for its result, `result.json` and
  `result.npz`, read back by `gfem.load_result` as the same result the
  in-process solve returns. It exists only for a run that completed.

Partial bundle
  The files a {term}`job`'s run writes in place of its result files when a
  uniform sweep stopped at an {term}`unsolved frequency`: `partial.json` and
  `partial.npz`, holding the {term}`partial result`, the unsolved frequencies
  and the error's message. Read it by name,
  `gfem.load_result("DIR/partial.json")`; `gfem.load_result("DIR")` reads only
  a complete result.

Solve cache
  A directory named by `cache=` on `gfem.solve` and `Job.run` (`--cache` on
  `python -m gfem run`) that keeps finished results, filed under the mesh,
  materials, problem and solver options they were solved for, so that the
  same call again reads its result back instead of solving. Off unless a call
  names one; delete the directory to clear it. What gfem keeps in memory
  between solves is not part of it, and `gfem.release_memory` leaves it alone.

Unsolved frequency
  A frequency of a uniform sweep that has no entry in the
  {term}`partial result`: the linear solve did not converge there, or the sweep
  (or its {term}`frequency worker`'s block) stopped at an earlier frequency that
  did not converge.

S-parameters
  The network parameters of a {term}`driven problem`: the entry from port j to
  port i is the wave leaving port i for a unit wave sent into port j, every
  other port terminated in its {term}`reference impedance`. gfem's are
  {term}`pseudo-waves`, one matrix per frequency.

Power waves
  The incident and outgoing waves $a$ and $b$ of each {term}`port` (Kurokawa),
  defined from its peak voltage and current and its
  {term}`reference impedance` so that $|a|^2$ is the time-averaged incident
  power in W; $b = S a$. On a real reference they equal the
  {term}`pseudo-waves`.

Pseudo-waves
  The waves gfem's S relates (Marks and Williams):
  $a = \sqrt{\mathrm{Re}\, z} / (2|z|)\,(V + zI)$ and
  $b = \sqrt{\mathrm{Re}\, z} / (2|z|)\,(V - zI)$ for a
  {term}`reference impedance` $z$. On a real $z$ they are the
  {term}`power waves`; on a line's complex characteristic impedance the line
  is matched to its own mode.

Reference impedance
  The impedance per {term}`port` that S-parameters are normalized to: the port
  resistance for a lumped port, the power-voltage impedance for a wave port with
  a voltage path, and for a {term}`Floquet port` the wave impedance of its
  order in the medium behind the face ($\eta / \cos\theta$ for TE, $\eta \cos\theta$ for TM; none
  where the order does not propagate).

Participation ratio
  The fraction of a mode's electric or inductive energy stored in one region,
  port, sheet or interface. The participation ratio of a lumped port with an
  inductance is the junction participation when that port stands for a
  Josephson junction.

Principal participation
  The fraction of a mode's electric energy stored in one
  {term}`anisotropic <Anisotropic material>` region by the field component
  along one of the region's {term}`principal axes`; the three sum to the
  region's {term}`participation ratio`.

Sheet participation
  The {term}`participation ratio` of an inductive {term}`sheet`: the fraction
  of a mode's energy stored in the sheet's inductance. For a lossless mode on
  a superconducting film it equals the kinetic inductance fraction.

Dressed mode
  A mode of the structure after the junction's cosine potential is added to
  the solved linear modes: its transition frequency, {term}`Lamb shift`,
  {term}`anharmonicity` and {term}`dispersive shifts <Dispersive shift>`.

Dispersive shift
  The change of one {term}`dressed mode`'s transition frequency per photon in
  another, a signed angular frequency; the full shift, not half of it.

Anharmonicity
  The second transition frequency of a {term}`dressed mode` minus its first, a
  signed angular frequency; negative for a transmon.

Lamb shift
  A {term}`dressed mode`'s transition frequency minus the frequency of the
  solved linear mode.

Loss budget
  The split of a mode's $1/Q$ into the contributions of its loss channels: the
  loss in the model (materials, walls, sheets, resistive ports, absorbing
  boundaries), which the eigenfrequency already carries, plus each channel
  outside the model, which adds its {term}`participation ratio` times its
  loss tangent.

Measurement
  A post-solve function of a result's {term}`fields <Field>` that needs extra
  physics input (a surface, a thickness, a point, a path).

Power flow
  The net time-averaged power crossing the boundary between a chosen set of
  elements and the rest of the solve domain, positive when it leaves the set.
  Measured by `gfem.power_flow` from the residual of the discrete equations, so
  exact for the solved field. Poynting flux is kept for the pointwise integral
  of $\tfrac{1}{2}\mathrm{Re}(\mathbf{E} \times \mathbf{H}^*)$.

Bloch reciprocity
  $S(\mathbf{k})^\mathsf{T} = S(-\mathbf{k})$: a Bloch cell's S matrix is not symmetric, but swapping two
  ports is the same as reversing the {term}`Bloch wave vector`. With
  {term}`Floquet ports <Floquet port>` the orders relabel and take a sign:
  $S_{p\leftarrow q}(\mathbf{k}) = \sigma_p \sigma_q S_{-q\leftarrow -p}(-\mathbf{k})$, with $-p$ the port of the opposite order
  and $\sigma = -1$ for an order with a nonzero in-plane wave vector, $+1$ otherwise.
```

## Circuits

```{glossary}
SAX component
  A driven solve wrapped as a model of a SAX circuit
  ({py:obj}`gfem.SaxComponent`): its design settings and the circuit's
  wavelength or frequency in, the S-parameters between its circuit ports out,
  photonic or RF.

Circuit port
  The name under which a SAX circuit reaches a port of a {term}`SAX component`.
  A port left out of the component's circuit ports still terminates the
  device, as in a solve.

Component band
  The frequencies of a {term}`SAX component`'s driven problem: it answers any
  wavelength or frequency between the lowest and the highest, exactly at them
  and interpolated between them, and refuses one outside.

Common reference impedance
  The one real impedance `z0`, 50 Ω by default, to which every RF model of a
  SAX circuit, gfem's and SAX's, refers its S-parameters; set once for the
  whole circuit.

Internal lumped load
  A {term}`lumped port` with only an inductance or a capacitance, left out of
  an RF {term}`SAX component`'s circuit ports: it stays inside the device as
  a load, and its value remains a design setting.
```

## Materials and optimization

```{glossary}
Material
  A linear medium with relative permittivity, permeability, loss tangent and
  electrical conductivity (S/m), assigned to a whole {term}`region`; each
  property may be one value or, for an {term}`anisotropic material`, a tensor,
  and the permittivity may follow a {term}`permittivity law`.

Loss tangent
  The ratio $\tan\delta$ of the imaginary to the real part of a {term}`material`'s
  permittivity, $\varepsilon = \varepsilon' (1 - i \tan\delta)$ under $e^{+i\omega t}$: the dielectric loss
  (`Material(tan_delta=...)`). A region holding the share $p$ of a mode's
  electric energy adds $p \tan\delta$ to the mode's $1/Q$.

Dispersive material
  A {term}`material` whose permittivity depends on frequency through a
  {term}`permittivity law`.

Permittivity law
  A relative permittivity that depends on frequency: a high-frequency
  permittivity $\varepsilon_\infty$ plus a sum of {term}`dispersion terms <dispersion term>`
  (`gfem.PermittivityLaw`), given as a `Material`'s `eps_r`.

Dispersion term
  One Drude, Lorentz, Debye or Djordjevic-Sarkar contribution to a
  {term}`permittivity law`, passive by its own parameter bounds.

Static permittivity
  The zero-frequency value of a {term}`permittivity law`: $\varepsilon_\infty$ plus the
  strengths of its Lorentz, Debye and Djordjevic-Sarkar terms. A Drude term
  has none, because a Drude metal conducts at zero frequency. The
  electrostatic problem reads it. For an electro-optic crystal it
  differs from the optical `eps_r` and goes in a separate electrostatic model.

Energy permittivity
  The weight $\mathrm{Re}[\varepsilon + (\omega/2)\, d\varepsilon/d\omega]$ of $|E|^2$ in a mode's energy at its frequency,
  for a region with a {term}`permittivity law`: $\mathrm{Re}\,\varepsilon_r$ for a constant
  material, and $\varepsilon_\infty$ for a lossless Drude law.

Library material
  A {term}`material` gfem ships ready-made in `gfem.materials`, built from
  the parameters of one cited published source and accurate over its
  {term}`fitted range`. Lithium niobate is built at one frequency on your
  principal axes, and its static permittivity is a separate library
  material.

Permittivity fit
  A {term}`permittivity law` fitted to a table of one material's measured
  values, with its largest relative errors in permittivity, refractive index
  and {term}`extinction coefficient` over the table
  (`gfem.fit_permittivity`).

Extinction coefficient
  The imaginary part $k \ge 0$ of a tabulated refractive index $n + ik$. It
  measures absorption and does not depend on the time convention; under
  $e^{+i\omega t}$ the permittivity is $\varepsilon = (n - ik)^2$. Not the
  extinction cross section, which is a power.

Fitted range
  The frequencies over which a {term}`permittivity law`'s parameters were
  matched to data or to a source's formula (`PermittivityLaw.frequency_range`).
  Outside it the law stays passive but its accuracy is unknown, and a solve
  that reads it there warns (`gfem.FittedRangeWarning`).

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

Index interpolation
  A density blend in which the complex refractive index $n - i\kappa$ is linear in
  the density $\rho$, $\varepsilon = ((1 - \rho) N_a + \rho N_b)^2$ (`scheme="index"`); the blend to
  use between a metal and a dielectric.

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

Layer pattern
  The polygons where a {term}`layer grid` design is solid: the part of the
  grid's footprint where the filtered density reaches the projection threshold,
  each polygon an outer boundary with its holes. It is what a GDS file holds.

Fabrication model
  A Gaussian blur of the drawn design followed by a threshold set by the
  {term}`edge bias`. The drawn design is the projected density an optimization
  ends with, whose {term}`layer pattern` goes to GDS; the printed design is the
  model's prediction of the wafer, and it stays differentiable.

Edge bias
  The outward shift, in metres, of a printed design's straight edges against the
  drawn ones; positive grows the solid. A line's width changes by twice the edge
  bias.

Tolerance report
  The linearized spread of a device's named quantities under Gaussian variations
  of named parameters about their nominal values, each tolerance $\pm t$ read as
  three standard deviations: each parameter's effect and share of the variance, the
  worst case and the {term}`linearized yield` against spec windows
  ({py:obj}`gfem.tolerance_report`).

Linearized yield
  The probability under the linear model $f_0 + J\delta$ that a quantity, or every
  quantity together, lies in its spec window: exact for one spec, a seeded Monte
  Carlo on the linear model for several.

Minimum width
  The smallest distance a foundry allows across solid between facing edges of a
  {term}`layer pattern`, measured perpendicular to an edge.

Minimum spacing
  The smallest distance a foundry allows across void between facing edges of a
  {term}`layer pattern`, of the same or different polygons.

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

Node map
  A function of {term}`design parameters`, written by you in JAX, that returns a
  mesh's node coordinates with its topology unchanged, applied with
  {py:meth}`gfem.Mesh.with_nodes`. Where a {term}`boundary morph` moves a
  surface and lets the rest follow, a node map says where every node goes.

Sliding surface
  A surface whose nodes may move only within their own facet plane during a
  {term}`boundary morph`. On a {term}`periodic pair`, a node and its partner
  move together.

Sliding port
  A {term}`wave port` named in `boundary_morph(sliding_ports=...)`: its face is
  a {term}`sliding surface`, so a shape design reaches the port, while the port
  keeps its plane and its mode follows the moved section. Other ports are held.

Mesh extension
  The elastic or harmonic solve that carries boundary motion into the mesh
  interior.

Element quality
  The scaled Jacobian of an element; zero or below means inverted.

Remesh and restart
  Regenerating the mesh at the current {term}`design parameters` and restarting
  the optimizer's history.

Adjoint solve
  The extra linear solve, on the system a forward solve already assembled,
  that gives the gradient of an {term}`objective` with respect to all
  {term}`design parameters` at once. `jax.grad` runs it; you never write it.

Objective
  A scalar function of a solution (S-parameter, eigenfrequency, Q, field
  overlap) whose gradient with respect to {term}`design parameters` is
  requested.

Cluster-symmetric objective
  An {term}`objective` that depends on a degenerate group of modes only through
  quantities unchanged when the modes are relabelled, such as their sum, so its
  gradient exists where individual eigenvalue gradients do not.

Non-differentiable quantity
  A result value or model input that gfem refuses to differentiate in a given
  version, raising an error instead of returning a zero gradient.
```

## Mesh refinement

```{glossary}
Mesh refinement
  Repeated {term}`refinement cycles <Refinement cycle>` that subdivide the
  tetrahedra where an {term}`error estimator` reports the most error, until
  its {term}`error estimate` meets a tolerance or a cap stops them
  (`gfem.refine_mesh`). It gives a new model whose mesh keeps the physical
  groups and the imported facets.

Refinement cycle
  One pass of {term}`mesh refinement`: solve, estimate the error, mark the
  tetrahedra holding most of it, split them at the midpoint of their longest
  edge (and their neighbours, so no hanging node remains), and carry the
  design to the new tetrahedra. Cycle 0 is the solve on the input mesh.

Error estimator
  A rule that computes, from a result's fields, each tetrahedron's share of
  the error and one {term}`error estimate`: the {term}`field estimator` or
  the {term}`quantity estimator`.

Field estimator
  The {term}`error estimator` of the field's error in the energy norm,
  relative to the field's energy norm (`gfem.FieldEstimator`). It compares
  the solved electric flux with a smoother flux recovered from it
  ({term}`flux recovery`), so it needs no extra solve.

Flux recovery
  Projecting the solved electric flux (`eps E`, or `eps grad V` in
  electrostatics) and the curl of `E` onto smoother fields, one polynomial
  degree below the solve's order. On the solve domain's natural surfaces,
  the recovered flux's normal component and the recovered curl's
  tangential component are held at zero. In electrostatics these are every
  exterior surface not held at a potential, ports and impedance surfaces
  included; in wave problems, the PMC surfaces and the exterior surfaces
  given no condition. The difference between the solved and the recovered
  fields, weighted by the field's energy, is the field's estimated error.

Quantity estimator
  The {term}`error estimator` of one quantity: a capacitance entry, an S
  entry, one mode's `omega` or Q, the mean `omega` of a degenerate cluster,
  or an {term}`objective` of the result (`gfem.QuantityEstimator`). It
  weights each tetrahedron's field error by its effect on that quantity,
  using fields already solved, except for an S entry or objective whose
  dual is not a solved field, which takes one {term}`adjoint dual` each.

Objective estimate
  The {term}`error estimate` of an optimization {term}`objective`, a
  function of one result, in the objective's units
  (`gfem.QuantityEstimator(objective=...)`).

Estimate kind
  What an {term}`error estimate` is of: the field, an entry, Q, a
  {term}`cluster mean`, or one kind of objective (`ErrorEstimate.kind`).
  Each kind has its own measured range ({doc}`verification`).

Cluster mean
  The mean complex `omega` of the modes of a degenerate cluster
  (`gfem.QuantityEstimator(omega=(i, j))`). It does not depend on which
  basis of the cluster the solver returns, so it can be followed and
  estimated where each member cannot.

Region mean
  A field's or a potential's mean over named volume regions
  (`gfem.region_mean`): the quantity an objective reads in place of a point
  value, which has no estimate.

Lossless rewrite
  On a lossless model with lumped ports, estimating an objective of S
  together with the power each port's column must carry, which makes its
  estimate tighter (`ErrorEstimate.kind` names it).

Adjoint dual
  The field solved with a port's S read-out, or an objective's
  measurement, as the source instead of the excitation, at the same order.
  The {term}`quantity estimator` weights field errors with it when no
  solved field can stand in: an S entry whose read port is not excited, or
  is a wave port with a hybrid mode in a lossy cross-section, and an
  objective of a field measurement, of electrostatic energy or a potential
  mean, or of a participation. Otherwise the dual is a port's own field,
  scaled, and needs no solve.

Error estimate
  The number an {term}`error estimator` gives for a result, compared with
  the tolerance of {term}`mesh refinement`. It has no sign or phase and is
  an estimate, not a bound. For a {term}`quantity estimator` it is relative
  to a capacitance entry, `omega`, a cluster's mean `omega` or Q; absolute
  for an S entry; and in the objective's units for an objective. gfem
  never adds it to the quantity. On gfem's verification cases, against
  errors on the mesh's own facets, the field estimator's estimate was
  between {gfem-effectivity}`field low` and {gfem-effectivity}`field high`
  times the true error (orders 2 and 3) and the quantity estimator's on
  entries between {gfem-effectivity}`quantity low` and
  {gfem-effectivity}`quantity high` times (order 2); each other
  {term}`estimate kind` has its own range ({doc}`verification`). Neither
  sees the faceting of curved surfaces.
```

## Free space

```{glossary}
Incident field
  The field a {term}`source` would have in the {term}`background stack` alone,
  with no {term}`scatterer`: a plane wave, a Gaussian beam or a field given by a
  function (`gfem.PlaneWave`, `gfem.GaussianBeam`, `gfem.IncidentField`).

Scattered field
  The {term}`total field` minus the {term}`incident field`; it is outgoing at
  the truncation (a {term}`perfectly matched layer` or absorbing faces).

Total field
  The incident plus the scattered field: the field a source column keeps, and
  the one every measurement reads by default.

Background stack
  The planar layers normal to z, each a constant isotropic material, whose
  exact solution is the {term}`incident field` (`Model(background=...)`,
  `gfem.Background`), optionally ending below on a PEC ground plane; a
  homogeneous background is a one-layer stack.

Incidence layer
  The layer of the {term}`background stack` a plane wave comes from: the
  bottom one for $\theta < \pi/2$, the top one for $\theta > \pi/2$. It
  must be lossless; the wave's amplitude and phase are given there.

Scatterer
  Where the model differs from the {term}`background stack` at the same
  height: its elements of another material, and its sheets, conductors and
  lumped ports.

Source
  A {term}`driven problem`'s excitation that is not a {term}`port`: an
  {term}`incident field` or a {term}`point dipole` (`Driven(sources=...)`). It
  adds a column to the solve (a point dipole adds none under
  `Driven(source_fields=False)`, which reads its {term}`port waves <Port wave>`
  by reciprocity) and is never part of the device.

Excitation
  An {term}`excited port` or a {term}`source`: one column of a driven solve (or,
  for a dipole under `Driven(source_fields=False)`, a name with port waves and
  no column), whose name labels its kept fields and its results.

Port wave
  The outgoing wave at a {term}`port` in a {term}`source`'s column, in √W
  (`DrivenResult.port_wave`); its squared modulus is the power the source sends
  into the port. Under `Driven(source_fields=False)` it is read by reciprocity
  from the port's own solution, without solving the source: the same value.

Point dipole
  An electric or magnetic point current at one position
  (`gfem.ElectricDipole`, `gfem.MagneticDipole`).

Emitted power
  The time-averaged power a {term}`point dipole` gives the field
  (`DrivenResult.emitted_power`): what it radiates plus what nearby lossy
  materials absorb.

Cross sections
  The powers a {term}`scatterer` removes from an {term}`incident field`
  (extinction), absorbs (absorption) and sends out in the
  {term}`scattered field` (scattering), each divided by the
  {term}`incident intensity`, in m² (`gfem.cross_sections`); extinction is
  absorption plus scattering.

Optical force
  The time-averaged force and torque of a solved electric field on a body
  (`gfem.optical_force`): the derivative of the discrete time-averaged
  Lagrangian under a rigid motion of the body's nodes, the volume form of the
  Maxwell stress tensor on its {term}`stress shell`. A driven field gives it
  per incident power, an eigenmode at 1 J of stored energy.

Stress shell
  The solve-domain elements outside a body that share a vertex with it, where
  `gfem.optical_force` reads the Maxwell stress. It must be one homogeneous,
  lossless medium; another shell would change the force by discretization
  error only.

Incident intensity
  The power per unit area that {term}`cross sections` are divided by: a plane
  wave's $|E_0|^2/(2\eta)$, a Gaussian beam's {term}`focal intensity`, or the
  value given for an incident field from a function.

Gaussian beam
  A focused beam carrying 1 W through any plane across its axis, given by
  its {term}`waist`, focus and direction (`gfem.GaussianBeam`): an exact
  solution of Maxwell's equations in a homogeneous lossless background whose
  paraxial limit is the fundamental Gaussian beam.

Waist
  The $1/e$ field radius $w_0$ of a {term}`Gaussian beam` at its focus, in m,
  the same at every frequency.

Focal intensity
  $I_0 = |E(r_f)|^2/(2\eta)$ at the focus $r_f$ of a Gaussian beam carrying
  1 W, with $\eta$ the wave impedance of the background
  (`gfem.GaussianBeam.focal_intensity`).

Incident contact
  The largest {term}`incident field` where a {term}`scatterer` meets a
  {term}`perfectly matched layer` or an absorbing face, or on a wave port face,
  relative to its peak in the model: it must stay below $10^{-3}$, since the
  incident field is dropped there.

Truncation layer
  The physical elements that touch a {term}`perfectly matched layer` or an
  absorbing face: where `gfem.cross_sections` reads its power balance, so it
  must hold only the {term}`background stack`.

Bulk power
  The {term}`emitted power` of the same {term}`point dipole` in the unbounded,
  lossless material at its position: the closed form the {term}`Purcell
  factor` divides by.

Purcell factor
  The {term}`emitted power` of a {term}`point dipole` over its {term}`bulk
  power` (`gfem.purcell_factor`): how much the structure around it speeds up its
  emission.

Beta factor
  The share of a {term}`point dipole`'s {term}`emitted power` that goes into one
  guided mode: the squared modulus of its {term}`port wave` at that mode's port,
  over the emitted power.

Radiative efficiency
  The share of a {term}`point dipole`'s {term}`emitted power` that leaves the
  model (through ports, a {term}`perfectly matched layer` or absorbing faces)
  rather than being absorbed by lossy materials.

Exterior medium
  The constant, isotropic, lossless material a field radiates into, the
  model's background: its wavenumber and wave impedance scale the
  {term}`far-field amplitude`.

Far-field amplitude
  The field far from every source with its outgoing spherical factor
  $e^{-ikr}/r$ removed, in V, per direction (`gfem.far_field`, `FarField`).

Far-field layer
  The elements just outside the side `gfem.far_field` reads, all in the
  {term}`exterior medium`; by default the physical elements that touch a
  {term}`perfectly matched layer` or an absorbing face.

Radiation intensity
  The power radiated per unit solid angle, $|F|^2/(2\eta)$ in W/sr.

Directivity
  $4\pi$ times the {term}`radiation intensity` over the radiated power. The
  gain divides by the accepted power instead, the realized gain by the
  incident power, and the radiation efficiency is the radiated over the
  accepted power (IEEE 145).

Ground plane
  A flat PEC or PMC exterior face that the {term}`far-field layer` meets: the
  far field of the half-space above it includes the image of every source.

Beam coupling
  The amplitude, in √W, that a driven field sends into an incoming beam (a
  fiber's), with the beam scaled to 1 W through the side it is read on
  (`gfem.beam_coupling`); by reciprocity, for a port's field, the
  {term}`port wave` the port receives when that beam lights the device. Its
  squared modulus is the coupled power on a real reference.
```

## Multipoles

```{glossary}
Multipole decomposition
  The multipoles of the field a {term}`scatterer` radiates under one
  {term}`incident field`, in a homogeneous lossless background, each with its
  share of the scattering cross section (`gfem.multipoles`). It is always of
  the whole scatterer, never of a mirror-reduced part.

Multipole coefficients
  The dimensionless electric $\alpha_{lm}$ and magnetic $\beta_{lm}$
  amplitudes of the scattered field's outgoing vector spherical waves about
  the {term}`multipole origin`, per {term}`incident amplitude`, normalized so
  that $\sigma_\text{sca} = k^{-2}\sum(|\alpha_{lm}|^2 + |\beta_{lm}|^2)$
  (`Multipoles.alpha`, `Multipoles.beta`). A sphere's Mie coefficients
  $a_l$ and $b_l$ are the same numbers up to a factor.

Multipole degree
  The index $l$ of a multipole: 1 is a dipole, 2 a quadrupole, 3 an octupole;
  $m$ is its azimuthal index, from $-l$ to $l$. It is not the element order of
  the solve.

Multipole origin
  The point the multipoles are taken about (`origin=`). The
  {term}`partial scattering cross sections <Partial scattering cross section>`
  depend on it; their sum does not.

Incident amplitude
  The complex field the {term}`multipole coefficients <Multipole
  coefficients>` are given per: a plane wave's amplitude, a Gaussian beam's
  focal field, $\sqrt{2\eta I}$ for an {py:obj}`gfem.IncidentField`;
  $|E_\text{inc}|^2/(2\eta)$ is the {term}`incident intensity`
  (`Multipoles.incident_amplitude`).

Partial scattering cross section
  The power one electric or magnetic multipole of one {term}`degree <Multipole
  degree>` (all $m$) radiates, over the {term}`incident intensity`, in m²
  (`Multipoles.electric_scattering`, `Multipoles.magnetic_scattering`). Summed
  over all degrees they give the scattering cross section.

Exact multipole moments
  The Cartesian electric and magnetic dipoles and quadrupoles $p$, $m$,
  $Q^e$, $Q^m$ that radiate exactly the degree-1 and degree-2 parts of the
  scattered field, at any particle size (`Multipoles.electric_dipole` and
  its siblings).

Induced current
  $J = +i\omega\varepsilon_0(\varepsilon_r - \varepsilon_b)E$ in the
  {term}`scatterer` under $e^{+i\omega t}$, with $E$ the {term}`total field`:
  the source of the {term}`current moments <Current moments>`.

Current moments
  The {term}`long-wavelength dipole` $p_0$ and the {term}`toroidal dipole` $T$
  of the {term}`induced current` (`gfem.current_moments`), as opposed to the
  {term}`exact multipole moments <Exact multipole moments>`. They are not
  defined for surface currents or a permeability contrast.

Long-wavelength dipole
  $p_0 = \frac{1}{i\omega}\int J\,dV$, the electric dipole of the
  {term}`induced current` without its $k^2$ corrections. It equals the exact
  $p$ only for particles small against the wavelength.

Toroidal dipole
  $T$, the $k^2$ correction that turns the {term}`long-wavelength dipole` $p_0$
  into the exact electric dipole: $p \approx p_0 - ikT$ under $e^{+i\omega t}$.
  It is not a separate radiating family and has no cross section of its own.

Anapole
  A scatterer whose exact electric dipole vanishes, $p = 0$, while $p_0$ and
  $T$ do not. Its long-wavelength approximation $p_0 = ikT$ is only a rough
  guide, 7-9 % off for a sphere of high index.

Kerker condition
  The first Kerker condition: equal electric and magnetic dipoles
  ($\alpha_{1,\pm1}$ against $\beta_{1,\pm1}$; $p_x = m_y/v$ for a wave
  travelling along $z$, $v$ the speed of light in the background), under which
  the dipoles scatter nothing backward. The quadrupoles still do.
```

## Thermal

```{glossary}
Slab cross-section
  A 2D cross-section read as a mesh one element thick along z
  ({py:meth}`gfem.Mesh.from_gmsh_cross_section`), with insulated
  {term}`end faces`, so nothing varies along the extrusion. Its extensive
  quantities are stated per metre of extrusion ({term}`per-length quantity`).

End faces
  The two surfaces `"front"` and `"back"` of a {term}`slab cross-section`. They
  carry the wave port of a mode solve and the terminals of a DC conduction
  solve along the guide.

Thermal domain
  The volumes a thermal problem solves on: every volume of the mesh except
  those excluded from thermal problems (`excluded={"thermal": [...]}`).
  Volumes excluded only from electromagnetic problems stay in it.

Modelled depth
  The depth of substrate under the stack that the mesh includes, with the
  {term}`heat sink` at its bottom: a choice of the model, not the wafer's
  thickness.

Heat sink
  A surface that removes heat: a {py:obj}`gfem.FixedTemperature` or a
  {py:obj}`gfem.Convection` boundary. Every connected piece of the
  {term}`thermal domain` needs one in a steady solve.

Insulated surface
  An exterior surface of the {term}`thermal domain` that no thermal boundary
  names: no heat crosses it (`Model.default_insulated` lists the named ones).

Temperature
  Absolute temperature, in kelvin: the unknown of a thermal problem.

Temperature rise
  The temperature minus the {term}`source-free temperature`, in kelvin: the
  part of the temperature linear in the heat sources. When every boundary
  shares one temperature, it is the temperature above that one.

Source-free temperature
  The steady temperature a thermal model reaches with every heat source off,
  set by its boundary temperatures, ambients and heat fluxes alone; where a
  transient starts by default.

Thermo-optic coefficient
  The rate of change of a material's refractive index with temperature,
  dn/dT in 1/K, constant, about the index of its `eps_r`. It may be negative.

DC conduction
  The steady current through the conductors of a model between named
  terminals ({py:obj}`gfem.DCConduction`), giving their
  {term}`conductance matrix` and the heat they dissipate.

Conductance matrix
  The currents into a DC conduction solve's terminals per volt on each, in
  siemens; its inverse for one terminal and a ground is the resistance.

Joule heating
  The heat a current dissipates in a conductor; {py:obj}`gfem.joule_heating`
  turns a DC conduction result and a drive (voltage, current or power) into a
  heat source with no new solve. For a driven result's loss, see
  {term}`electromagnetic heating`.

Dissipated power
  The time-averaged electromagnetic power turned into heat inside the model:
  the loss of its materials and of its {py:obj}`gfem.Conductor` and
  {py:obj}`gfem.Impedance` sheets ({py:obj}`gfem.dissipated_power`). Power
  that leaves through ports, absorbing boundaries or a perfectly matched
  layer, and power a port's load takes, is not part of it.

Electromagnetic heating
  The heat source a driven result's kept fields at one frequency give under
  a drive, the incident power of each port and their relative phases: the
  {term}`dissipated power` of the superposed field
  ({py:obj}`gfem.electromagnetic_heating`), with no new solve.

Coherent drive
  Several ports fed at once at one frequency with fixed relative phases, so
  their fields add and the heat depends on the phases. Drives that are not
  coherent, other tones or unrelated sources, are separate heat sources that
  add.

Sheet heat
  The part of an electromagnetic heat source that a conductor or impedance
  sheet dissipates, given per face in W/m² and entering the thermal problem
  at that face, where the face's thermal boundary decides where it goes.

Heat flow
  The net heat per unit time leaving a set of elements, or the
  {term}`thermal domain` through named surfaces, positive out, in W (W/m on
  a cross-section; {py:obj}`gfem.heat_flow`). Read from the discrete heat
  balance of the solved temperature, so the flows through every boundary
  surface add up to the source power.

Heat out
  The heat that has left a side or a surface since the start of a transient
  run, in J (J/m on a cross-section; {py:meth}`gfem.Trace.heat_out`),
  including the heat the elements store in its balance. Its step difference
  over dt is the mean {term}`heat flow` over the step.

Heater terminal
  A surface where a DC conduction solve holds a potential: a pad, a contact
  face, or an {term}`end face <End faces>` of a slab cross-section.

Thermo-optic shift
  The change of a guided mode's effective index, and of its phase per length,
  caused by a {term}`temperature rise`, to first order in the rise
  ({py:obj}`gfem.thermo_optic_shift`).

Thermo-optic phase
  The phase a temperature rise adds to guided light, $k_0\, \mathrm{Re}\, \Delta n_\mathrm{eff}$ per metre: a
  delay, positive for a positive {term}`thermo-optic coefficient`. A
  transmission's argument moves by minus it.

Thermal crosstalk
  The {term}`thermo-optic phase` a waveguide, or a {term}`pass`, that is not
  under the heater gains, as a signed fraction of the heated one's phase at
  the same drive.

Optical window
  A model of the optical neighbourhood of one waveguide core, on which a
  thermo-optic or {term}`electro-optic shift`, or a route phase, solves the
  mode: the static (thermal or electrostatic) model with the other regions
  excluded from electromagnetic solves, or a mesh of its own in the same
  coordinates. For an electro-optic shift it reaches the electrodes' inner
  edges.

Route
  The reference line of a waveguide core in a 3D model, a polyline in metres,
  along which an {term}`optical window` is swept ({py:obj}`gfem.route_phase`).

Station
  A point on a {term}`route`, evenly spaced in arc length, where the optical
  window is placed and the rise is read.

Pass
  One straight stretch of a waveguide route under or beside a heater, whose
  phase is read on its own.

Route phase
  The {term}`thermo-optic phase` a {term}`route` gains from a 3D temperature
  rise, to first order, summed over its {term}`stations <Station>`.

Waveform
  The piecewise-constant scale of a transient problem's heat sources, as
  (start time, scale) pairs on its {term}`time grid`; off before the first.

Time grid
  The uniform times $t_n = n\, dt$, $dt = \mathrm{duration} / \mathrm{steps}$, on which a transient
  thermal solve steps, records its traces and keeps fields.

Trace
  A linear functional of the temperature (a point value, a region's mean, a
  thermo-optic phase) recorded at every time of a transient solve's
  {term}`time grid` ({py:obj}`gfem.Trace`).

Settled trace
  A trace whose change still to come after its last sample is below 1 % of its
  swing; the rise time and the time constant are meaningful only on one, and
  warn ({py:obj}`gfem.UnsettledTraceWarning`) on any other.

Rise time
  The 10–90 % time of a step response, each part of the trace normalised to
  its first and last samples ({py:func}`gfem.rise_time`).

Time constant
  The time a step response takes to reach 1 − 1/e = 63.2 % of its swing, with
  the same normalisation as the rise time ({py:func}`gfem.time_constant`).

Per-length quantity
  On a {term}`slab cross-section`, an extensive quantity (a power, a
  resistance) per metre of the extrusion: the only form gfem takes for
  extensive inputs on a cross-section (`power_per_length=`).

Mass density
  A material's mass per volume, in kg/m³ (`mass_density`), read by transient
  thermal problems.
```

## Electro-optics

```{glossary}
Electro-optic tensor
  The tensor $r$ of a crystal's linear (Pockels) electro-optic effect,
  $\Delta(1/n^2)_{ij} = \sum_k r_{ijk} E_k$, given contracted as a 6 × 3 array
  $r_{Ik}$ in m/V (`electro_optic_tensor`): rows $xx, yy, zz, yz, xz, xy$,
  columns $x, y, z$, in the frame of the material's right-handed `axes`. A
  static field $E_\mathrm{s}$ changes the permittivity by
  $\Delta\varepsilon_r = -\varepsilon_r (r \cdot E_\mathrm{s}) \varepsilon_r$.

Electro-optic shift
  The change of a guided mode's effective index, and of its phase per length,
  caused by a static field through the crystals'
  {term}`electro-optic tensors <electro-optic tensor>`, to first order in the
  field and per volt on one terminal ({py:obj}`gfem.electro_optic_shift`).

Push-pull
  A modulator drive in which one electrode pattern puts opposite static
  fields across two arms, so their
  {term}`electro-optic shifts <electro-optic shift>` have opposite signs and
  the arms' phase difference is the difference of the two.

Photoelastic tensor
  The tensor $p$ of a crystal's photoelastic effect, contracted as a 6 × 6
  array $p_{IJ}$ (`photoelastic_tensor`, dimensionless, at constant electric
  field): rows and columns $xx, yy, zz, yz, xz, xy$ in the frame of the
  material's right-handed `axes`, with the {term}`engineering strain`, so that
  $\Delta(\varepsilon_r^{-1})_I = p_{IJ} S_J$. It belongs to the optical
  crystal, not to one with a stiffness.

Acousto-optic shift
  The change of a guided mode's effective index caused by an acoustic wave,
  through the {term}`photoelastic tensor` (the strain) and the
  {term}`electro-optic tensor` (the piezoelectric field), to first order and
  per volt on one terminal, as a complex phasor at each acoustic frequency
  ({py:obj}`gfem.acousto_optic_shift`).

Voltage-length product
  $V_\pi L = \lambda / (2\,|\partial n_\mathrm{eff}/\partial V|)$, the drive
  times the electrode length that turns the phase by $\pi$: per arm, or for a
  {term}`push-pull` pair from the difference of the two arms' shifts.

Electro-optic overlap factor
  $\Gamma$, the ratio of an {term}`electro-optic shift` to the bulk shift
  $\tfrac12 n^3 r V / g$ of a uniform field across an electrode gap $g$: the
  share of the ideal shift the field and the mode deliver together. Derived
  from the shift, never an input.

Electrode clearance
  The lateral distance from a waveguide's edge to the nearest electrode's
  inner edge, $(g - w)/2$ for a ridge of width $w$ centred in a gap $g$. Its
  lower bound comes from the mode's decay outside the ridge,
  $\ln(1/\delta) / (2\kappa)$ with $\kappa = k_0 \sqrt{n_\mathrm{eff}^2 -
  n_\mathrm{slab}^2}$, so that the mode's intensity at the metal is $\delta$
  of its value at the ridge's edge.
```

## Piezoelectric acoustics

```{glossary}
Stiffness
  A solid's elastic stiffness at constant electric field, $c^E$: a symmetric
  positive-definite 6 × 6 matrix in Pa in {term}`Voigt order <engineering strain>`
  (`stiffness`), in the frame of the material's `axes`. The piezoelectric
  stiffening is added by the solve. {py:obj}`gfem.isotropic_stiffness` builds an
  isotropic solid's from its Young's modulus and Poisson's ratio.

Piezoelectric tensor
  The stress constants $e$ of a crystal, a 3 × 6 matrix in C/m²
  (`piezoelectric_tensor`): rows the field components, columns in Voigt order,
  in the frame of the material's `axes`. The stress and the electric
  displacement are $T = c^E S - e^\mathsf{T} E$ and $D = e S + \varepsilon^S E$.

Clamped permittivity
  The permittivity at constant strain, $\varepsilon^S$: in gfem the `eps_r` of a
  material in a {py:obj}`gfem.Piezoelectric` problem.

Mechanical quality factor
  The loss of a solid as a complex stiffness $c (1 + i/Q_m)$ under
  $e^{+i\omega t}$ (`mechanical_quality_factor`). It lumps the material's, the
  scattering and the electrodes' losses, which the model does not compute.

Mechanical domain
  The volumes a {py:obj}`gfem.Piezoelectric` problem solves the displacement on:
  every volume not excluded with `excluded={"mechanical": [...]}`. Each needs a
  {term}`stiffness` and a `mass_density`; air and a handle's potential-only
  regions are left out of it.

Mechanical eigenmode
  A source-free solution of linear elasticity on a 2D section, varying as
  $e^{-iqz}$ along the section's normal: a complex angular frequency whose
  imaginary part is the decay from mechanical loss and damping frames, and a
  displacement normalized to 1 J/m ({py:class}`gfem.MechanicalEigenmode`). A
  material with a piezoelectric tensor couples the potential to it.

Axial wavenumber
  The real propagation constant $q$ in rad/m along $z$ of a
  {term}`mechanical eigenmode`; 0 for a mode uniform along $z$.

Interdigital transducer
  Interleaved comb electrodes on a piezoelectric surface whose alternating
  fingers launch and detect acoustic waves. In a cross-section model it is the
  whole finger set, with its reflectors if it has any.

Meshed electrode
  A metal region that carries displacement, stiffness and mass, with its
  potential fixed on the whole region. It is named as a terminal or as ground by
  its area, and loads the crystal under it.

Line electrode
  A zero-thickness electrode: a potential fixed on a named line, with no mass.
  It is named as a terminal or as ground by its line.

Clamped surface
  A line of a piezoelectric section where the displacement is held at zero
  (`clamped=`). Every other boundary is traction-free.

Damping frame
  An absorbing edge region where the stiffness becomes
  $c (1 + i(1/Q_m + \eta r^2))$, with $r$ the normalised depth into the region
  and $\eta$ its {term}`loss factor <Loss factor (of a damping frame)>`
  ({py:obj}`gfem.DampingFrame`). It absorbs the acoustic waves that leave a
  piezoelectric section, as a {term}`perfectly matched layer` does the
  electromagnetic ones.

Loss factor (of a damping frame)
  The loss $\eta$ that a {term}`damping frame` adds to the stiffness at its outer
  edge. The default, 1.2, absorbs well from about four wavelengths of frame.

Admittance per length
  The current into a terminal per volt on another, per metre along $z$:
  $Y'_{ij}(f)$ in S/m, the result's `admittance_per_length`. A device of
  {term}`aperture` $W$ has $Y = W Y'$.

Clamped capacitance
  The capacitance of the electrodes with the {term}`clamped permittivity` and the
  solid held still, $C_0'$ per metre along $z$
  (`clamped_capacitance_per_length`). It is not the low-frequency limit of
  $Y'/i\omega$, the free capacitance, which is larger by $1 + K^2$.

Aperture
  The length along $z$ over which an
  {term}`interdigital transducer <Interdigital transducer>`'s fingers overlap. It
  multiplies a per-length admittance into the device's.

Series resistance
  The resistance of an electrode's fingers and bus in series with its terminal,
  added when the {term}`admittance per length` becomes network parameters
  (`network_parameters(series_resistance=...)`).

Engineering strain
  The six strain components in Voigt order $xx, yy, zz, yz, xz, xy$, with the
  shear entries twice the tensor components: $S_{xz} = \partial_x u_z$ for a
  field that does not vary along $z$ ({py:obj}`gfem.strain`).

Radiation conductance
  The real part of a transducer's {term}`admittance per length`,
  $G' = \mathrm{Re}\,Y'$. When its waves leave through absorbing edges,
  $\tfrac12 G' |V|^2$ is the power radiated and dissipated per metre of aperture.

Synchronous frequency
  The frequency at which the acoustic wavelength equals a transducer's period,
  $f_0 = v / (2p)$ for fingers alternating at pitch $p$ and a wave of speed $v$:
  where the {term}`radiation conductance` peaks.
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
```

## Logs

```{glossary}
Log display
  What `gfem.log.enable` turns on: one line per gfem log record on stderr
  (clock, level, the logger's short name, the rank under MPI, then the
  record's message), with a {term}`progress row` for each long loop where
  the console can draw it. Off by default, and off again after
  `gfem.log.disable`.

Progress row
  One line of the {term}`log display` that follows a running loop (a uniform,
  adaptive or port mode sweep, a mode search, a mesh refinement) or a phase
  running outside any loop, and disappears when it ends. It is never a log
  record and is never written to a file.
```

## Broadband models

```{glossary}
Rational model
  S over a band written as a constant matrix plus pole-residue terms,
  $S(f) = D + \sum_n R_n/(2\pi i f - p_n)$, whose poles every element of S
  shares, on one real reference impedance per port
  ({py:obj}`gfem.RationalModel`). Fitted to S by vector fitting, made
  passive by default, and written as a SPICE subcircuit or a Verilog-A
  module.

Passivity violation
  A frequency interval where the largest singular value of a
  {term}`rational model`'s S exceeds 1: there the model would give back more
  power than it receives. The fit removes them by default.

Harmonic grid
  Frequencies $f_k = k\,\Delta f$ for $k = 1, \dots, N$, optionally with
  0 Hz, on which a low-pass {term}`time response` is computed. Its record
  lasts $1/\Delta f$, and $f_N$ sets the rise time of an edge.

Time response
  The impulse or step response of S, from its windowed inverse Fourier
  transform: on a {term}`harmonic grid` (low pass), or over an evenly spaced
  band as a magnitude (band pass). The reflected step at a port is its TDR,
  the transmitted step its TDT ({py:obj}`gfem.TimeResponse`). Not a
  {term}`trace`, which is a thermal solve's record over time.

Impedance profile
  $Z(t) = Z_0 (1 + \rho(t))/(1 - \rho(t))$, read from the reflected step
  $\rho$ of a port with real reference impedance $Z_0$: the impedance along
  the line, in round-trip time. Exact up to the first round trip past the
  first discontinuity; later, multiple reflections mix in.
```
