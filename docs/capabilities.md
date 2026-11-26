# Capabilities

What gfem solves, which results it differentiates, and what it refuses, at
v0.1.

## What you can solve

- **Problems**: {py:obj}`gfem.Driven` (the response of a model to its ports
  and sources over a frequency grid, uniform or adaptive sweep), {py:obj}`gfem.Eigenmode` (resonant
  modes nearest a target frequency, with their Q; lumped and wave ports, a
  wave port on one process with a target above its mode's cutoff; with a
  permittivity law, outside the law's
  {term}`unresolved band`), {py:obj}`gfem.Electrostatic` (the
  Maxwell capacitance matrix between named terminals) and {py:obj}`gfem.PortMode`
  (the mode of a wave port's cross-section, with its group index and
  dispersion at every frequency, {doc}`recipes/waveguide-dispersion`; a
  permittivity law in it where $\mathrm{Re}\,\varepsilon > 0$). Eigenmode and uniform driven problems take a
  Bloch wave vector (`wave_vector=`, rad/m) on a periodic model; a uniform
  driven sweep may keep the incidence angle fixed (`wave_vector_frequency`).
  {py:func}`gfem.band_diagram` solves an eigenmode problem along a
  {term}`wave vector path` and returns the bands, their connections and their
  {term}`band group velocity` ({doc}`recipes/periodic-cell`).
- **2D models**: a plane of triangles read with {py:meth}`gfem.Mesh.from_gmsh`,
  where neither the structure nor the field varies along z. A uniform driven
  sweep solves one polarization, `Driven(polarization="Ez")` or `"Hz"`, with
  {term}`line ports <Line port>`, PEC, PMC and first- or second-order absorbing
  lines, materials (permittivity laws included), material fields and
  layer-grid designs, with their gradients. Powers and fields are per metre
  along z, and S equals the slab's ({doc}`recipes/2d-models`). Piezoelectric
  problems also solve on 2D meshes (below).
- **Ports**: {py:obj}`gfem.LumpedPort` on one or several surfaces driven together,
  and {py:obj}`gfem.WavePort` on an exterior surface, fed by its port mode;
  several wave ports on one surface each request a different mode of it,
  each its own row and column of S ({doc}`recipes/shared-port-surface`);
  wave ports also load an eigenmode solve and report each one's
  {term}`coupling Q` (`q_port`);
  {py:obj}`gfem.FloquetPort` on the open face of a periodic cell: one
  diffraction order in one polarization (TE or TM) of a plane wave, each its
  own row and column of S, $|S|^2$ the diffraction efficiency
  ({doc}`recipes/floquet-ports`).
- **Boundaries**: {py:obj}`gfem.PEC`, {py:obj}`gfem.PMC` (symmetry planes), {py:obj}`gfem.Absorbing`
  (first or second order), {py:obj}`gfem.Conductor` (a finite-conductivity metal, by
  its surface impedance), {py:obj}`gfem.Impedance` (per-square R, L and C),
  and {py:obj}`gfem.Periodic` (a periodic pair of surfaces, with plain
  periodicity or a Bloch wave vector set on the problem;
  {doc}`recipes/periodic-cell`).
- **Open space**: {py:obj}`gfem.PML` on meshed layer regions around the air
  box: absorbs outgoing waves at any angle, for driven and eigenmode solves,
  wave ports and periodic cells ({doc}`recipes/perfectly-matched-layer`).
- **Free space**: sources beside the ports, each a column of the same
  factorization (`Driven(sources=...)`): incident fields on a
  {term}`background stack` (`Model(background=...)`, one material or a
  planar {py:obj}`gfem.Background` of layers, optionally over a ground
  plane), as {py:obj}`gfem.PlaneWave` (TE, TM, circular or Jones, oblique,
  past the critical angle too), {py:obj}`gfem.GaussianBeam` (an exact beam
  carrying 1 W) or {py:obj}`gfem.IncidentField` (a field you give as a
  function); and point dipoles, {py:obj}`gfem.ElectricDipole` and
  {py:obj}`gfem.MagneticDipole`. With `Driven(source_fields=False)` a
  dipole's port waves are read from the ports' own solutions by reciprocity,
  one solve per port for any number of positions. Results: the total, scattered and
  incident fields, {py:obj}`gfem.cross_sections`, `emitted_power`,
  {py:obj}`gfem.purcell_factor`, port waves, the far field with the IEEE
  145 antenna quantities and the radar cross section
  ({py:obj}`gfem.far_field`), and {py:obj}`gfem.beam_coupling` into a fiber's
  beam ({doc}`recipes/incident-fields`, {doc}`recipes/cross-sections`,
  {doc}`recipes/point-dipoles`, {doc}`recipes/far-field`).
- **Multipoles**: the electric and magnetic multipoles a scatterer radiates
  under an incident field, their partial scattering cross sections and the
  exact dipoles and quadrupoles ({py:obj}`gfem.multipoles`); the
  long-wavelength and toroidal dipoles of the induced current
  ({py:obj}`gfem.current_moments`) ({doc}`recipes/multipoles`).
- **Periodic surfaces** (`gfem.diffraction`): diffraction efficiency per order
  and polarization, power balance (reflectance, transmittance, absorptance),
  Jones, circular and Mueller matrices of a cell, angle sweeps (conical
  incidence included), emissivity by Kirchhoff's law (directional and
  hemispherical), a cell's response sampled against one meta-atom parameter
  with its derivative, the effective permittivity and permeability of a slab
  retrieved from its S ({doc}`recipes/effective-medium`), and aperture fields
  of large metasurfaces in the {term}`locally periodic approximation` with
  angular-spectrum propagation, differentiable except in the angle
  ({doc}`recipes/diffraction-efficiency`, {doc}`recipes/emissivity`,
  {doc}`recipes/metasurface-from-cells`).
- **Materials and designs**: {py:obj}`gfem.Material` per region (isotropic, or
  anisotropic: real symmetric tensors on shared principal axes,
  {doc}`recipes/anisotropic-materials`), {py:obj}`gfem.MaterialField` per
  element (isotropic, or principal values on fixed axes), and {py:obj}`gfem.SheetField` for the fill of a patterned metal sheet
  ({doc}`recipes/sheet-design`). A `Material`'s permittivity may follow a
  {py:obj}`gfem.PermittivityLaw` of Drude, Lorentz, Debye and
  Djordjevic-Sarkar terms, metals included
  ({doc}`recipes/dispersive-materials`). {py:mod}`gfem.materials` builds
  silicon, silica, silicon nitride, gold, copper, aluminium and lithium
  niobate from cited sources, and {py:obj}`gfem.fit_permittivity` fits a
  law to a measured table ({doc}`recipes/fit-measured-data`).
- **Fabricable layer patterns**: a layer-grid design's pattern as polygons
  ({py:obj}`gfem.design.LayerPattern`), checked against a minimum width and
  spacing and written to GDS ({py:obj}`gfem.design.write_gds`); a fabrication
  model (blur, edge bias) and length-scale constraints inside the gradient
  ({doc}`recipes/fabricable-layout`, {doc}`recipes/robust-optimization`).
- **Tolerance report**: {py:obj}`gfem.tolerance_report` (the spread of named
  quantities under named fabrication tolerances, each parameter's effect and
  variance share, the worst case, the linearized yield against spec windows and a
  linearity check, from one adjoint pass per quantity) ({doc}`recipes/tolerance-report`).
- **Measurements**: {py:obj}`gfem.characteristic_impedance`,
  {py:obj}`gfem.interface_participation`, {py:obj}`gfem.power_flow`,
  {py:obj}`gfem.dissipated_power` (a driven field's loss per region and
  metal surface), {py:obj}`gfem.optical_force` (the force and torque on a
  body, differentiable; {doc}`recipes/optical-force`) and
  {py:obj}`gfem.electric_energy` ({doc}`recipes/measurements`); the
  free-space measurements above.
- **Circuits**: {py:obj}`gfem.SaxComponent` puts a driven solve in a SAX
  circuit as a photonic block (wavelength in µm) or an RF block (frequency in
  Hz, on one reference impedance for the circuit), answering any wavelength or
  frequency in its band; `NetworkParameters.to_sdict` and `.to_skrf` hand a
  solved S to SAX or scikit-rf ({doc}`recipes/sax-circuit`).
- **Quantum metrics**: {py:obj}`gfem.quantum.dressed_modes` (the dressed
  frequencies, Lamb shifts, anharmonicities and dispersive shifts of one
  junction on the solved modes) and `EigenmodeResult.sheet_participation`
  (the kinetic inductance fraction of a film), with their gradients
  ({doc}`recipes/quantum-metrics`). A mode's loss splits into a $1/Q$ budget
  and $T_1$ limits ({doc}`recipes/loss-budget`).
- **Thermal problems**: {py:obj}`gfem.SteadyThermal` (the temperature under
  heat sources), {py:obj}`gfem.TransientThermal` (the temperature over time
  under switched sources, with traces recorded at every step) and
  {py:obj}`gfem.DCConduction` (the current through conductors between
  terminals: their conductance and Joule heat), on 3D meshes and on 2D
  cross-sections read with {py:meth}`gfem.Mesh.from_gmsh_cross_section`.
  Boundaries {py:obj}`gfem.FixedTemperature`, {py:obj}`gfem.Convection` and
  {py:obj}`gfem.HeatFlux`; a surface no thermal boundary names is insulated.
  Materials take `thermal_conductivity`, `mass_density`, `specific_heat`,
  an electrical `conductivity`, a `thermo_optic_coefficient`, a
  `nonlinear_index` and a `two_photon_absorption`, and
  `excluded=` leaves regions out of one physics at a time
  ({doc}`recipes/heater-temperature`).
- **Heat and phase**: {py:obj}`gfem.joule_heating`,
  {py:obj}`gfem.electromagnetic_heating` (the heat of a driven result's
  loss, in volumes and on metal sheets, for one port or several fed
  coherently; {doc}`recipes/electromagnetic-heating`) and
  {py:meth}`gfem.HeatSource.uniform` (heat sources);
  {py:obj}`gfem.heat_flow` and {py:obj}`gfem.heat_flow_per_length` (the heat
  leaving a steady temperature through surfaces or a set of elements) and
  {py:meth}`gfem.Trace.heat_out` (the heat that has left since the start of a
  transient run);
  {py:obj}`gfem.thermo_optic_shift` (a guided mode's phase per length under
  a rise, on a cross-section) and {py:obj}`gfem.route_phase` (the phase
  along routes through a 3D rise) ({doc}`recipes/thermo-optic-phase`);
  {py:obj}`gfem.rise_time` and {py:obj}`gfem.time_constant` of a step
  response ({doc}`recipes/thermal-step-response`); {py:obj}`gfem.element_mean`.
- **Electro-optic shift**: {py:obj}`gfem.electro_optic_shift` (a guided
  mode's index and phase per volt from an electrostatic potential, on a
  cross-section, through any 6 × 3 electro-optic tensor; $V_\pi L$ per arm
  and push-pull) and `ElectrostaticResult.capacitance_per_length`. Materials
  take an `electro_optic_tensor` on explicit `axes`
  ({doc}`recipes/electro-optic-phase`).
- **Acousto-optic shift**: {py:obj}`gfem.acousto_optic_shift` (a guided mode's
  index and phase per volt on a transducer's terminal, as a complex phasor at
  every kept frequency of a piezoelectric result, through a `photoelastic_tensor`
  and an `electro_optic_tensor`; photoelastic and electro-optic parts, $V_\pi L$
  per arm and push-pull; differentiable in the tensors, the acoustic inputs and
  the windows' `offset`) ({doc}`recipes/acousto-optic-shift`).
- **Kerr nonlinearity**: {py:obj}`gfem.kerr_nonlinearity` (a port mode's
  $\gamma_\mathrm{NL}$ and two-photon loss at every row of its grid) and
  {py:obj}`gfem.kerr_coupling` (the tensor between its waves, for self- and
  cross-phase modulation and four-wave mixing), from the materials'
  `nonlinear_index` and `two_photon_absorption`
  ({doc}`recipes/waveguide-nonlinearity`).
- **Second-order nonlinearity**: {py:obj}`gfem.second_harmonic` (a pump mode
  and its harmonic: $\kappa$, $\Delta\beta$, poling period, undepleted
  efficiency of a poled guide) and {py:obj}`gfem.three_wave_coupling` (the
  tensor between the waves of a port, for sum- and difference-frequency
  generation), from a material's `second_order_nonlinear_tensor`
  ({doc}`recipes/second-harmonic-generation`).
- **Photon pairs**: {py:obj}`gfem.spdc` and {py:obj}`gfem.sfwm` (spontaneous
  down-conversion and four-wave mixing of port modes: pair rate, spectral
  brightness, joint spectrum of a pulse, heralded purity, differentiable)
  ({doc}`recipes/photon-pair-sources`).
- **Broadband models and time responses**:
  `NetworkParameters.fit_rational` (a passive {term}`rational model` of S,
  written as a SPICE subcircuit or a Verilog-A module) and
  `NetworkParameters.step_response` / `impulse_response` (TDR, TDT and the
  {term}`impedance profile`, differentiable; the fit is not)
  ({doc}`recipes/rational-model`, {doc}`recipes/time-responses`).
- **Piezoelectric problems**: {py:obj}`gfem.Piezoelectric` (the admittance per
  length between electrodes of a 2D section of piezoelectric, elastic and
  dielectric regions over a list of frequencies, the clamped capacitance, and
  the displacement and potential fields; any stiffness and piezoelectric tensor
  on explicit `axes`, mechanical loss through a quality factor, graded
  {term}`damping frames <damping frame>` as absorbing edges, meshed or line
  electrodes), `PiezoelectricResult.network_parameters` (S-parameters and
  Touchstone for an aperture and a series resistance) and
  {py:obj}`gfem.strain` ({doc}`recipes/piezoelectric-admittance`).
- **Mechanical eigenmodes**: {py:class}`gfem.MechanicalEigenmode` (the complex
  frequencies, quality factors, displacements normalized to 1 J/m and group
  velocities of the elastic modes of a 2D section nearest a target, at an
  {term}`axial wavenumber`: Lamb, shear horizontal and Rayleigh waves, ribs;
  any stiffness, the piezoelectric coupling with grounded electrodes,
  mechanical loss through a quality factor and
  {term}`damping frames <damping frame>`, clamped lines or components;
  `omega` and the mode fields differentiate in the material constants and node
  coordinates)
  ({doc}`recipes/mechanical-modes`).

## Solvers

- **`linear_solver="direct"`**: pick it for a model whose factorization fits
  in one machine's memory, for eigenmode problems, and for the repeated solves
  of an optimization loop, where a design can solve on the previous design's
  factorization ({doc}`recipes/s-parameter-gradient`). It factorizes once per frequency
  and reuses the factorization for every port. The default (`"auto"`) picks it
  for a driven problem on one machine while its factorization fits in half the
  memory. Install the `mkl` extra for its fastest factorization.
- **`linear_solver="pmg_gmres"`**: pick it for a driven model several
  wavelengths across that the direct solver cannot hold, such as a photonic
  device in dielectric cladding, with absorbing faces or a perfectly matched
  layer, or a metal circuit on a fine mesh. Its iterations stay near 10 as the
  model grows, it needs about a quarter of the direct solver's memory at order 2,
  and it is faster than the direct solver from a few million unknowns on.
  Driven problems at order 2 or 3 on one CPU process, with gradients, and
  without Bloch, periodic or Floquet models. `"auto"` picks it where the direct
  solver does not fit and it does ({doc}`recipes/large-driven-model`).
- **`linear_solver="bddc_gmres"`**: pick it across MPI ranks, for periodic and
  Bloch models, and for electrostatic problems, when the direct factorization
  does not fit; `"auto"` picks it for a driven problem that neither the
  direct solver nor `"pmg_gmres"` takes, and for electrostatic problems. A model with a perfectly matched layer refuses it:
  use `"pmg_gmres"`.
- **`linear_solver="petsc"`** and **`eigen_solver="slepc"`**: pick them for
  MUMPS and its options, your own PETSc Krylov method and preconditioner, or
  SLEPc's own controls. With no options, `"petsc"` gives the same results as
  `"direct"`, and SLEPc the same modes and Q as ARPACK, the default eigen
  solver on one process (across MPI ranks the default is SLEPc). You build
  them from source ({doc}`recipes/petsc-slepc`).
- **`device="cuda"`**: the direct solver on one NVIDIA GPU through cuDSS,
  experimental ({doc}`recipes/gpu-direct-solver`).
- **Thermal and DC conduction problems** solve on `"direct"` by default and
  accept `"bddc_gmres"` and `"petsc"`. One factorization serves a whole
  transient run. They do not run on the GPU solver or across MPI ranks: both
  raise.
- **Across MPI ranks**: driven uniform sweeps, electrostatic and eigenmode
  solves, and periodic cells on one axis run split across the ranks of one
  job, with their gradients (except gradients through a mode's field), on
  Open MPI and Linux x86_64. Everything else raises
  {py:obj}`gfem.DistributedUnsupportedError` there; the full table is in
  {doc}`recipes/distributed-solve`.

How large a model each solver has finished, and in what time:
{doc}`project/scaling`.

## What you can differentiate

```{gfem-gradient-matrix}
```

- **FD-verified**: supported; the gradient is checked against finite
  differences.
- **Exact zero**: the result does not depend on this input; `jax.grad`
  returns zero, correctly.
- **Raises**: asking for this gradient raises
  {py:obj}`gfem.NonDifferentiableError` (or the error named) instead of
  returning a wrong value.

The thermal rows differentiate the temperature, the traces, the heat flow,
the conductance and the phase with respect to per-element thermal and
electrical conductivity, the thermo-optic coefficient, the boundary values (a
fixed temperature, a convection's coefficient and ambient, a heat flux), a
transient's mass density and specific heat, the heat sources and the drive
({doc}`recipes/thermal-gradient`). The heat rows differentiate the dissipated
power and the temperature it heats with respect to the driven solve's
materials, sheets and drive ({doc}`recipes/electromagnetic-heating`).

The electro-optic shift differentiates with respect to its tensor, the
static permittivity of every element, the drive and the electrostatic
mesh's nodes (an electrode gap through a shape morph), each checked against
finite differences; with respect to the optical model's permittivity or
nodes it raises, because the shift holds the mode constant (a port mode on its own
differentiates) ({doc}`recipes/electro-optic-gradient`).

A SAX component's S differentiates as the driven problem's, with respect to
every input its model function builds from its settings, so one `jax.grad` of
a SAX circuit reaches gfem's inputs and SAX's analytic models together. With
respect to the circuit's wavelength or frequency it is the derivative of the
band's interpolation ({doc}`recipes/sax-circuit`).
The piezoelectric problem differentiates its admittance per length, its
clamped capacitance and readings of its fields with respect to the stiffness,
piezoelectric tensor, permittivity, mass density and mechanical quality
factor of every element, and the mesh's nodes through a node map of your own
({py:meth}`gfem.Mesh.with_nodes`), each checked against finite differences;
its frequencies, terminals, clamped lines and damping frame raise
({doc}`recipes/piezoelectric-gradient`).

How each mark is checked: {doc}`verification`.

### Gradients that raise

Off the matrix, asking for these gradients raises the named error instead of
returning a wrong value:

```{gfem-raises}
```

## Limits

- **Low frequency**: a direct driven solve on the CPU in one process keeps its
  digits below the low-frequency floor, $k_0 h_\mathrm{min} = 10^{-6}$
  ($f h_\mathrm{min} \approx$ 48 Hz·m, with $h_\mathrm{min}$ the shortest
  edge: 48 MHz with 1 µm elements, 5 GHz with 10 nm elements), through the
  {term}`tree-cotree gauge`, on by default (`SolverOptions.gauge`). The wave
  ports' mode problem is not gauged and keeps its own floor: below it, a
  model with wave ports warns with {py:obj}`gfem.LowFrequencyWarning`, and
  lumped ports keep their digits. Other driven solves (`"bddc_gmres"`,
  `"pmg_gmres"`, `"petsc"`, the GPU, several MPI ranks, periodic and Floquet models, or
  `gauge="none"`) are ungauged: below the floor their fields, gradients and,
  with an exact factorization, the capacitive part of S lose digits, and they
  warn. For capacitance at low frequency, solve {py:obj}`gfem.Electrostatic`;
  the details are under {py:obj}`gfem.Driven`. An eigenmode solve whose target
  or lowest mode is below the floor warns too: its frequencies, fields and
  participations lose the same digits, and the reported `error` does not
  show it.
- **Gyrotropic materials**: non-symmetric (non-reciprocal) tensors, such as a
  magnetized ferrite, are not supported. Neither is loss whose principal axes
  differ from the permittivity's. Both raise `ValueError`.
- **Anisotropic materials at wave ports**: a port's cross-section may contain
  anisotropic materials when every tensor has a principal axis along the port
  normal. Elsewhere in the model the principal axes may point in any
  direction: a model with tilted principal axes and its rigidly rotated copy
  give the same S, eigenfrequency and capacitance.
- **Second-order absorbing boundaries** raise next to an anisotropic
  material; use order 1, which is exact at normal incidence for any
  anisotropic material.
- **Periodic models**: a Bloch wave vector needs the direct solver for
  eigenmode problems, and the direct or BDDC + GMRES solver for driven
  problems. PETSc and cuDSS refuse it, as does an adaptive sweep.
  Electrostatic problems refuse any periodic pair, and a wave port may not
  touch one. An absorbing boundary reflects waves that meet it obliquely,
  and a driven solve warns ({py:obj}`gfem.ObliqueAbsorberWarning`); a
  Floquet port absorbs them, or a perfectly matched layer below the cell.
  A band diagram runs on one process, under `jax.grad` but not inside
  `jax.jit` or `jax.vmap`; design gradients reach its band frequencies, not
  `v_g` or the wave vectors. Details: {doc}`recipes/periodic-cell`.
- **Floquet ports**: the half-space behind the face is homogeneous,
  isotropic and lossless, and the face needs two periodic pairs. Only the
  uniform driven sweep on the direct or BDDC + GMRES solver; every other
  path raises. Only orders that propagate over the whole sweep can be
  excited. No gradient with respect to the incidence, the half-space or the
  nodes of a Floquet surface. In the gradient matrix, a model with Floquet ports takes the
  rows of the periodic driven problems (plain and Bloch), with S read from
  its Floquet ports. Details: {doc}`recipes/floquet-ports`.
- **Periodic surfaces**: no gradient with respect to the incidence angle; a
  power balance needs a port for every propagating order and polarization; a
  cell response has one parameter at one incidence; aperture fields assume
  normal incidence, a rectangular lattice and cells that do not diffract, and
  propagate scalar tangential components at fixed distances. Details:
  {doc}`recipes/diffraction-efficiency`, {doc}`recipes/metasurface-from-cells`.
- **Perfectly matched layers**: no error estimates or mesh refinement for
  driven and eigenmode problems (electrostatic refinement runs). Box shells
  around the interior box only, one thickness per side. Each layer region's
  background is a constant `Material`, isotropic or on axis-aligned
  principal axes. No `Absorbing` face, `Impedance` or `Conductor` sheet in or
  on the layer. Lumped ports and Floquet surfaces keep clear of it, and a
  wave port may cut a side layer but not one stretched along its normal
  (driven solves only). No
  stretch along a periodic axis. No `"bddc_gmres"`; `"pmg_gmres"` solves the layer exactly
  inside its iterations. The fields inside the
  layer are not physical; energies, participations and port power count the
  physical regions. Details: {doc}`recipes/perfectly-matched-layer`.
- **Free space**: 3D only. An incident field needs a perfectly matched
  layer or absorbing faces, scatterers clear of them where it is not
  negligible, exterior walls it satisfies and ports outside it; it is
  refused on periodic models, on the adaptive sweep and by mesh
  refinement. The {term}`incidence layer` and a Gaussian beam's medium are
  lossless; a beam travels in one material, not a stack. A stack's layers
  are constant, isotropic materials whose planes no tetrahedron crosses. A
  dipole sits inside one lossless, isotropic material (not a
  `MaterialField`), off interfaces, boundary faces, ports and the layer;
  node gradients take electric dipoles only. `source_fields=False` takes
  dipoles only and needs a port with a defined wave; it gives no emitted
  power, Purcell factor, kept field or far field of them, and its adaptive
  gradients are exact at the samples and to the sweep's field error between,
  with one full-size adjoint per port at each frequency carrying a cotangent.
  Cross sections need a lossless background and a
  {term}`truncation layer` of background only; the far field needs a
  {term}`far-field layer` in one lossless, isotropic exterior medium, and
  no layered exterior. A result or job holding an `IncidentField` cannot
  be saved, and the solve cache passes it by. Gradients flow to material
  fields; node coordinates are not differentiable under an incident field.
- **Multipoles**: incident-field columns, 3D, a homogeneous lossless
  background and the whole scatterer (no ground plane, mirror half or slab).
  The induced-current moments take volume currents only (no perfect
  conductor, sheet, lumped port or magnetic contrast) and are not
  differentiable; the multipoles are not differentiable by node coordinates.
  Details: {doc}`recipes/multipoles`.
  Details: {doc}`recipes/incident-fields`, {doc}`recipes/point-dipoles`,
  {doc}`recipes/far-field`.
- **Unrequested port modes**: a propagating mode that no wave port on its
  surface requests is partly reflected at the port, not converted; request
  it to see where its power goes ({py:obj}`gfem.SkippedPortModeWarning`
  flags a mode left out below a requested one). An eigenmode solve flags a
  mode such a family carries, and a mode trapped against the port, with
  {py:obj}`gfem.UnrequestedPortModeWarning`; its Q is the port's, not the
  device's.
- **Eigenmodes with wave ports**: one process, no Bloch wave vector, no port
  face cutting a perfectly matched layer, no quantity estimates
  (field-estimator refinement runs), and no absorbing edge beside a
  permittivity law in a port's cross-section. A target below a requested
  port mode's cutoff raises {py:obj}`gfem.PortModeError`, and a root below
  a cutoff that does not decay in time is dropped with
  {py:obj}`gfem.AntiBoundRootWarning`. The solve lists the resonances of a
  strip around `target` (Q above about 2); a resonance of lower Q is out of
  reach, even when it lies nearest `target`: use a driven sweep. Under
  `jax.grad`, a design touching
  a wave port face raises {py:obj}`gfem.PortGeometryError`; a driven solve
  differentiates it. The wave ports' $\sum_p 1/q_p$ matches $1/Q$ of an otherwise
  lossless model to $O(1/Q^2)$. Details: {doc}`modes-and-ports`.
- **Lossy port lines**: wave-port S reads each mode by the biorthogonal
  projection, so a lossy port line is matched to its own mode and the modes
  of a lossy shared port surface separate. With a `voltage_path` its
  reference impedance is the mode's complex $Z_{PV}$ (pseudo-waves); give
  every wave port a `voltage_path`, or none, for a symmetric S. With ports of
  one kind, $S$ is symmetric to round-off when the port faces are meshed
  alike; otherwise it is symmetric to the discretization error of the two
  ports' modes, since $S_{12}/S_{21}$ is the ratio of the two discrete modes'
  normalizations ($\kappa^2/N$ each), about $10^{-5}$ on a coarse lossy
  microstrip and independent of the device.
- **Designs at wave ports**: a design may reach a wave port face: the
  materials of its elements (laws and material fields included), a sheet's
  fill or an `Impedance` on its edge, and its nodes
  (`gfem.shape.boundary_morph(..., sliding_ports=...)`), on every driven path
  except a Bloch wave vector and a fixed-angle sweep. The port's mode follows
  the design, so S stays in the pseudo-waves of the moved mode: renormalize S
  where a port meets a fixed external line. A {py:obj}`gfem.PortMode`
  result's $\gamma$, `n_eff`, $Z_{PV}$ and field differentiate the same way.
  An eigenmode solve raises {py:obj}`gfem.PortGeometryError` for such a
  design. A voltage path is a fixed set of points: a design that moves a
  conductor at its end, or an interface along it, gives $Z_{PV}$ a kink.
  Details: {doc}`recipes/port-mode-gradient`.

- **Group index and dispersion**: on `PortMode` results only (`None` on a
  driven solve's port modes), and not differentiable: under `jax.grad` they
  raise {py:obj}`gfem.NonDifferentiableError`, where the same result's
  $\gamma$ and `n_eff` differentiate. Their error is the mesh's and the
  section's truncation: $\beta_2$ needs far more cladding around a guide
  than $n_\mathrm{eff}$, so converge it in the clearance
  ({doc}`recipes/waveguide-dispersion`).

- **2D models**: normal incidence and one polarization only. Eigenmode,
  electrostatic and thermal problems, lumped and Floquet ports, periodic
  pairs, conductor and impedance surfaces, sheet designs, anisotropic
  materials, PMLs, the adaptive sweep, the iterative and GPU solvers, several
  processes, measurements other than `Field.at`, mesh refinement and node
  gradients raise `NotImplementedError` and name the alternative, most often
  the {term}`slab model`; a piezoelectric problem takes anisotropic materials,
  and node gradients through a node map of your own
  ({doc}`recipes/piezoelectric-admittance`). In Hz a design blend takes
  constant permittivities only. Details: {doc}`recipes/2d-models`.
- **Dispersive materials**: a permittivity law is isotropic, and
  `mu_r` does not disperse; give an anisotropic or magnetic material constant
  values per band. A metal in a wave port's cross-section raises: keep it off
  the port faces or use lumped ports. A blend with a law end solves on the
  driven uniform sweep only. An eigenmode target or mode in a law region's
  {term}`unresolved band` raises; move the target or refine that region.
  A library or fitted law is accurate within its {term}`fitted range`; a
  solve outside it warns. Details: {doc}`recipes/dispersive-materials`.
- **Mode-field gradients**: participations, port Q and the mode field
  differentiate for isolated modes of non-periodic models, on the direct
  solver, on the CPU, in one process ({doc}`recipes/eigenmode-gradient`).
  Elsewhere only the eigenfrequencies differentiate.
- **Quantum metrics**: one junction per model; the junction's cosine on the
  solved linear modes, in the transmon regime. They are checked against
  identities, finite differences and published outputs, not against a
  measured device.
- **Thermal problems**: linear heat conduction with constant, isotropic
  properties: no temperature-dependent conductivity, no radiation, no
  thermal contact resistance, and the heat does not change the current or
  the electromagnetic loss. Electromagnetic heat comes from electric and
  sheet loss only: no magnetic loss, no eigenmode, port mode or 2D model. A
  transient runs on a uniform time grid with sources switched at grid
  times. A cross-section is a slice of an infinitely long device. The
  thermo-optic shift and the route phase are first order in the rise, with
  the mode held constant, and refuse a lossy optical section. Mesh
  refinement and error estimates do not run on thermal problems, and the
  rise time and time constant have no gradient. Details:
  {doc}`recipes/heater-temperature`, {doc}`recipes/electromagnetic-heating`,
  {doc}`recipes/thermo-optic-phase`, {doc}`recipes/thermal-step-response`,
  {doc}`recipes/thermal-gradient`.
- **Electro-optic shift**: a cross-section at DC, first order in the static
  field with the mode held constant. It refuses a lossy optical section, a 2D
  model and a potential split across ranks. The optical model does not see
  the electrodes, so their absorption is absent. No gradient goes through the
  shift's mode: the ridge's shape or width takes a difference over remeshes. Details:
  {doc}`recipes/electro-optic-phase`, {doc}`recipes/electro-optic-gradient`.
- **Piezoelectric problems**: 2D sections only (a tetrahedral or slab mesh
  raises); one process and the direct solver (sparse Cholesky); no periodic
  cells, no PML (damping frames instead), and `gfem.Absorbing` on a solid's edge
  raises; dielectric loss is not read; frequencies, terminals, clamped lines and
  damping frames are not differentiable; node gradients come through a node map
  of your own (no 2D boundary morph). Details:
  {doc}`recipes/piezoelectric-admittance`.
- **Mechanical eigenmodes**: 2D sections only; one process and the direct
  solver; `gfem.Absorbing` on a solid's edge raises (use a damping frame);
  gradients of `omega` and the fields only (`group_velocity` and `error` raise
  `gfem.NonDifferentiableError`, and so does a mode in a cluster). Details:
  {doc}`recipes/mechanical-modes`.
- **Layer patterns**: width and spacing only (no minimum area), and a pre-check, not
  foundry sign-off; one flat GDS cell; layer grids only (no free-form 3D
  designs); length-scale constraints validated up to a rule of $1.25R$ and a
  pitch of $R/3$, for a filter radius $R$.
- **SAX circuits**: a component answers only inside its band, interpolated
  between its solved frequencies; every exposed wave port needs a polarity
  reference, or for RF a voltage path; RF components need a 3D model and one
  reference impedance for the whole circuit; build the circuit outside
  `jax.jit`; no forward-mode derivatives. Details:
  {doc}`recipes/sax-circuit`.

## Platforms

Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows x86_64, Python >= 3.12.
The cuDSS GPU solver installs on Linux x86_64 and Windows x86_64. Importing
gfem turns on JAX double precision (`jax_enable_x64`), which every gfem solve
requires. See {doc}`get-started`.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain
GPL-2.0+ SuiteSparse code; see {doc}`project/licence`.
