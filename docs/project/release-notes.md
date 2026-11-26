# Release notes

## Unreleased

Optics:

- **Kerr nonlinearity of a port mode.** Materials take `nonlinear_index`
  ($n_2$) and `two_photon_absorption`; `gfem.kerr_nonlinearity` gives a port
  mode's $\gamma_\mathrm{NL}$ on every row of its grid and `gfem.kerr_coupling`
  the tensor between its waves ({doc}`../recipes/waveguide-nonlinearity`).
- **Second-harmonic generation of a port mode.** Materials take a
  `second_order_nonlinear_tensor` ($d$, m/V); `gfem.second_harmonic` gives a
  pump mode's coupling to its harmonic, the phase mismatch, the poling period and
  the undepleted efficiency of a poled guide, and `gfem.three_wave_coupling`
  the tensor between the waves of a port
  ({doc}`../recipes/second-harmonic-generation`).
- **Photon-pair sources.** `gfem.spdc` and `gfem.sfwm` give the pair rate,
  the spectral brightness and the joint spectrum, with its purity, of a
  waveguide pumped by a continuous wave or a pulse
  ({doc}`../recipes/photon-pair-sources`).
- **Optomechanical coupling.** `gfem.optomechanical_coupling(model, modes, port=,
  frequency=)` gives the vacuum coupling $\tilde g_0$ of the modes of a
  `gfem.MechanicalEigenmode` solve to a wave port's mode, solved once for all
  of them, split into a moving-boundary and a photoelastic part, with
  `g0(length)`; `gfem.optomechanical_overlap` gives the change of the effective
  index per unit amplitude of any displacement. Both differentiate in the
  mechanical solve's inputs and the `photoelastic_tensor`
  ({doc}`../recipes/optomechanical-coupling`).

Measurements:

- **Optical force and torque.** `gfem.optical_force` gives the time-averaged
  force and torque of a solved 3D electric field, driven or eigenmode, on a
  body: a volume region, several, or a predicate on element centroids, in N and
  N m, about the body's centroid or an `origin`. An eigenmode is at 1 J, so its
  force is $-(1\,\mathrm{J}/\omega)\,\mathrm{d}\omega/\mathrm{d}\xi$. The force is
  differentiable. The result is a `gfem.OpticalForce`
  ({doc}`../recipes/optical-force`).

Design:

- **Tolerance report.** `gfem.tolerance_report` differentiates a forward model
  at its nominal design and returns a `gfem.ToleranceReport`: the spread of named
  quantities under named tolerances, each parameter's effect and variance share,
  the worst case and the linearized yield against spec windows. A
  `BoundaryMorph` called with a scalar moves its surface by that edge bias
  ({doc}`../recipes/tolerance-report`).

Jobs and files:

- **A refused value names both versions.** A job or result bundle written by
  another gfem version, whose value this version refuses, now says which
  versions are involved and suggests upgrading.

Periodic cells:

- **Effective medium of a slab.** `gfem.diffraction.effective_medium` retrieves
  the diagonal permittivity and permeability of a slab of a periodic cell from
  its reflection and transmission at normal incidence, differentiable under
  `jax.grad` ({doc}`../recipes/effective-medium`).
- **Band diagrams.** `gfem.wave_vector_path` gives the Bloch wave vectors along
  straight segments between high-symmetry points: named points for the line,
  square, rectangular, hexagonal and simple cubic lattices, and
  `named_points` for any other. `gfem.band_diagram` solves them and returns a
  `gfem.BandDiagram`: bands in frequency order, the static band at Γ, band
  connections, `v_g`, `lost` and `anchored` (with `gfem.LostBandWarning` and
  `gfem.UnanchoredBandWindowWarning`), fields at chosen wave vectors and
  `to_xarray`. The band frequencies differentiate under `jax.grad`; a band
  diagram refuses `jax.jit` and `jax.vmap`. `Model.reciprocal_lattice` gives
  the reciprocal vectors ({doc}`../tutorials/photonic-crystal`,
  {doc}`../recipes/periodic-cell`).
- **Band group velocity.** `EigenmodeResult.v_g` gives
  $\partial\omega/\partial\mathbf{k}$ of a mode under a Bloch wave vector, in
  m/s. It is not differentiable, and `to_xarray` carries it.
- **Faster Bloch solves.** After a mesh's second wave vector, a Bloch eigenmode
  solve builds its operators from a per-mesh template instead of assembling
  them again.

Piezoelectric:

- **Mechanical eigenmodes.** `gfem.MechanicalEigenmode` finds the elastic modes
  of a 2D section nearest a target frequency at an axial wavenumber $q$, as
  `gfem.MechanicalEigenmodeResult`: complex `omega`, `frequency`, the quality
  factor `q`, the displacement of each mode normalized to 1 J/m and the
  `group_velocity`. Mechanical loss and damping frames are those of
  `gfem.Piezoelectric`; a piezoelectric material couples the potential, with
  `ground` electrodes. `omega` and the mode fields differentiate in the material
  constants and the node coordinates ({doc}`../recipes/mechanical-modes`).
- **Acousto-optic shift.** Materials take a `photoelastic_tensor`, and
  `gfem.acousto_optic_shift(model, acoustic, port=, frequency=)` gives the index
  change per volt of a wave port's mode at every kept frequency of a
  `gfem.PiezoelectricResult`, from one mode solve, as a complex phasor with its
  photoelastic and electro-optic parts and `v_pi_l`. It differentiates under
  `jax.grad` in the tensors, the acoustic inputs and `offset`
  ({doc}`../tutorials/acousto-optic`, {doc}`../recipes/acousto-optic-shift`).

Eigenmode solves:

- **Low-Q resonances are found whatever `n_modes`.** With wave ports, an
  eigenmode solve lists the modes of a strip around `target` (Q above about
  2), nearest in $\mathrm{Re}\,f$ first, and the modes it returns no longer
  depend on `n_modes` beyond its length. A resonance of Q 3.7 behind
  high-Q ones of an unrequested port mode family is found with one mode
  asked for. A strip that holds more modes than the search can list raises
  `gfem.EigenmodeConvergenceError`; a resonance below Q about 2 is for a
  `gfem.Driven` sweep ({doc}`../modes-and-ports`).
- **`requested_power_share` and `gfem.UnrequestedPortModeWarning`.**
  `EigenmodeResult.requested_power_share` gives, per wave port surface and
  mode, the share of the power the surface absorbs that its requested modes
  carry: near 1 for a mode of a requested family, 1e-3 or less for a mode an
  unrequested family carries or one trapped against the port. A share below
  one half warns, once per surface, with the mode to request or the port to
  move. It is exported by `to_xarray` and is not differentiable
  ({doc}`../modes-and-ports`).
- **A refinement leaving a port's reach loses its seed, not the solve.**
  A mode whose refinement runs where the port cannot follow its mode, or
  where two port modes cannot be told apart, is dropped from the search
  instead of ending the solve, and a port mode is followed in a way that no
  longer depends on the mesh unit.

Thermal:

- **Thermal boundary values differentiate.** A fixed temperature, a
  convection's coefficient and ambient, and a heat flux may be traced by
  `jax.grad` in steady and transient solves, through the temperature, the
  traces, the heat flow and the heat-out traces, instead of raising
  ({doc}`../recipes/thermal-gradient`).
- **Electromagnetic heating.** `gfem.electromagnetic_heating(result,
  model=, frequency=, incident_power=)` turns the kept fields of a driven
  result at one frequency into a `gfem.HeatSource`, with no new solve: the
  loss of the lossy materials (loss tangent, conductivity, complex or
  anisotropic permittivity, a permittivity law at that frequency) and of
  `Conductor` and `Impedance` sheets. Several ports are fed coherently, each
  with its incident power and its phase at the reference plane
  (`incident_phase=`); on a cross-section slab the drive is
  `incident_power_per_length=`. A perfectly matched layer, absorbing faces,
  ports and lumped port loads give no heat
  ({doc}`../recipes/electromagnetic-heating`).
- **Thermal and DC conduction designs reuse plans and factors.** An
  optimization that solves one `gfem.SteadyThermal`, `gfem.TransientThermal` or
  `gfem.DCConduction` problem per design on `linear_solver="direct"` plans the
  factorization once and refactorizes it with each design's values, as driven
  solves do. A design that leaves the operator unchanged (a heat-source
  amount, the drive) solves on the held factors, and a gradient with respect to
  the conductivity and the boundary temperatures factorizes the operator once
  per design instead of twice. `retain_memory=0` turns it off.
- **Heat of a law blend.** `gfem.dissipated_power` and
  `gfem.electromagnetic_heating` accept a metal-dielectric blend with a
  permittivity law, and their gradients flow to the blend's density.
- **Dissipated power.** `gfem.dissipated_power(field, model=)` gives a
  driven field's loss per volume region and per sheet, in W per 1 W
  incident. With real port references it sums to
  $1 - \sum_j |S_{j1}|^2$ ({doc}`../recipes/measurements`).
- **Sheet heat.** A heat source may carry heat on faces: a sheet's loss
  enters the thermal problem at the sheet, and the face's thermal boundary
  decides where it goes (an insulated face sends it all into the model).
- **Heat flow.** `gfem.heat_flow(steady, side, model=)` and
  `gfem.heat_flow_per_length` give the heat leaving named surfaces, volume
  regions or the elements a predicate selects, in W (W/m on a cross-section),
  positive out. It is read from the discrete heat balance, so the flows
  through every boundary surface add up to the source power to round-off
  ({doc}`../recipes/heater-temperature`). The heater tutorial now closes
  its heat balance with it.
- **Heat out.** `gfem.Trace.heat_out(side)` and
  `gfem.Trace.heat_out_per_length` record the heat that has left a side
  since the start of a transient run, in J (J/m), closing the energy balance
  with the stored heat at every time under every time scheme
  ({doc}`../recipes/thermal-step-response`).
- **Gradients.** The dissipated power, the temperature its heat gives, the
  heat flow and the heat out differentiate with respect to the lossy
  materials, a sheet's fill and an `Impedance`'s resistance, the incident
  powers and phases, the thermal conductivity, and (heat out) the mass
  density and specific heat ({doc}`../capabilities`).
- **Heat outside the thermal domain names its watts.** A heat source with
  heat on a region outside the thermal domain raises as before, and the
  message now gives the region and its watts. Sheet heat on a face with no
  thermal domain on either side raises the same way, naming the surface.
- **Behaviour change: `joule_heating` of a driven result raises
  `TypeError`.** `gfem.joule_heating(driven_result, ...)` raised
  `NotImplementedError`; it now raises `TypeError`, naming
  `gfem.electromagnetic_heating`. Code that caught `NotImplementedError`
  there should catch `TypeError`, or call `electromagnetic_heating`.

Electro-optics:

- **The electro-optic shift.** `gfem.electro_optic_shift` gives the change of
  a guided mode's effective index, and of its phase per length, per volt on
  one terminal of an electrostatic solve, through each crystal's
  electro-optic tensor, on a cross-section. It is first order in the static
  field, with the mode solved once and held constant, and reads the static
  field at the optical model's elements, so the optical model may be a
  window on a mesh of its own. It returns a `gfem.ElectroOpticShift`:
  `n_eff`, `delta_n_eff_per_volt`, `phase_per_length_per_volt` and `v_pi_l`,
  the voltage-length product of one arm. `jax.grad` reaches the tensor, the
  static permittivity, the drive and the electrostatic mesh's nodes (an
  electrode gap through a shape morph); the optical model's inputs raise
  `gfem.NonDifferentiableError`.

- **Electro-optic tensors.** `Material(electro_optic_tensor=..., axes=...)`
  and `MaterialField.electro_optic_tensor` take the contracted tensor $r_{Ik}$,
  a 6 × 3 array in m/V, rows $xx, yy, zz, yz, xz, xy$ in the frame of the
  material's `axes`, which must be right-handed. The static permittivity
  goes in a separate electrostatic model's `eps_r`.

- **Capacitance per length.** `ElectrostaticResult.capacitance_per_length`
  gives the Maxwell capacitance matrix per metre of a cross-section, in F/m,
  and `to_xarray` carries it. Behaviour change: the repr of an electrostatic
  result on a cross-section now also prints the capacitance per length in
  pF/m.

- **Documentation.** A tutorial, "An electro-optic modulator: x-cut thin-film
  lithium niobate, push-pull", computes a push-pull modulator's $V_\pi L$,
  shows where the field does its work, and co-designs the electrode gap and
  the ridge width under an electrode clearance. Two recipes ("Find the phase a
  voltage puts on a waveguide", "Differentiate an electro-optic phase"), an
  "Electro-optic shift" section of the conventions and an "Electro-optics"
  section of the glossary.

Piezoelectric acoustics:

- **Piezoelectric problems.** `gfem.Piezoelectric` solves linear
  piezoelectricity on a 2D section for the admittance per length between
  electrodes over a list of frequencies. It returns a
  `gfem.PiezoelectricResult`: `admittance_per_length` (S/m), the
  `clamped_capacitance_per_length` (F/m) and, with `keep_fields`, the kept
  displacement (`displacement(f)`, a map of x, y and z) and potential. The
  electrodes are meshed fingers or lines, the handle's base is a `clamped=`
  line, and `gfem.DampingFrame` names the regions that absorb the waves leaving
  the section ({doc}`../recipes/piezoelectric-admittance`).
- **Solid materials.** `Material` and `MaterialField` take `stiffness` (the
  6 × 6 $c^E$ in Pa), `piezoelectric_tensor` (the 3 × 6 $e$ in C/m²) and
  `mechanical_quality_factor`, in the frame of `axes`; `gfem.isotropic_stiffness`
  builds an isotropic solid's stiffness. `excluded={"mechanical": [...]}` leaves
  regions, such as air, out of the mechanical domain.
- **Network parameters and strain.** `PiezoelectricResult.network_parameters`
  turns the admittance into S-parameters for an aperture, a series resistance and
  a reference impedance, with Touchstone output, and `gfem.strain` reads the
  engineering strain of a kept displacement at points.
- **Node maps.** `Mesh.with_nodes(nodes)` returns a mesh with its nodes at new
  coordinates, on any mesh: write a JAX function of your design parameters that
  returns the coordinates, apply it, and differentiate through it.
- **Gradients.** The admittance, the clamped capacitance and readings of the
  fields differentiate with respect to the stiffness, piezoelectric tensor,
  permittivity, mass density and mechanical quality factor of every element, and
  through a node map; the frequencies, terminals, clamped lines and damping frame
  raise `gfem.NonDifferentiableError`.
- **Behaviour change: anisotropic materials on a 2D model.** A 2D model no longer
  raises an anisotropic material when it is built; a 2D driven or port-mode
  solve still refuses it, now at the solve. `MissingMaterialPropertyError` names
  the missing properties of any problem, not only thermal ones.
- **Behaviour change: a region excluded from every physics.** The check that a
  region cannot carry a material when it is excluded from every physics, and
  that a model cannot exclude every volume, now counts the `"mechanical"` key
  too. A model that excludes every volume from the electromagnetic, thermal and
  DC conduction domains only now raises when it is solved, not when the `Model`
  is built.
- **Documentation.** A tutorial, "An acoustic transducer: shear plate waves in
  thin-film lithium niobate", the recipes "Find a transducer's admittance and
  reflection" and "Differentiate a transducer's admittance", a "Piezoelectric problems" section of the conventions, a
  "Piezoelectric acoustics" section of the glossary and a section of the
  verification page.

Port modes:

- **Group index and dispersion.** Every `gfem.PortMode` solve now gives its
  mode's group index, group-velocity dispersion and third-order dispersion
  at every frequency of its grid, the rows of the reduced port mode model
  included: `PortModeResult.n_g`, `beta_2` (s²/m), `beta_3` (s³/m), and the
  dispersion parameter `PortModeResult.D` (s/m²). They are exact frequency
  derivatives of the propagation constant, through permittivity laws,
  conductors, lossy and leaky sections and perfectly matched layers. They
  are complex like `n_eff`, with the derivatives of $\beta$ as their real
  parts. `to_xarray` carries them. They are not differentiable:
  `jax.grad` through them raises `gfem.NonDifferentiableError`, while
  `gamma` and `n_eff` of the same result differentiate. A driven solve's
  `port_modes` carry `None`.

- **Behaviour changes.** A `PortMode` solve takes about 15-30 % longer.
  Its result's repr adds the range of the group index, and the result has
  three more pytree leaves. Its repr no longer raises inside `jax.jit`. A
  saved `PortMode` result now holds the three arrays, and gfem 0.2.0 does
  not load it; older files load with them `None`.

- **Documentation.** A recipe, "Find the group index and dispersion of a
  waveguide": a silicon wire with Sellmeier laws, its dispersion over a
  band, the widths where it vanishes, and the cladding it needs. A "Group
  index and dispersion" section in "Modes and ports" and in the
  conventions, and four glossary terms.

Linear solvers:

- **Fewer stale iterations across a frequency sweep.** A reused BDDC or
  p-multigrid preconditioner is now set up again before a step an octave from
  the frequency it was set up at, and when a reused step costs as much as the
  mean per step since the setup, instead of after one slow step.

- **A p-multigrid linear solver.** `SolverOptions(linear_solver="pmg_gmres")`
  solves driven problems with GMRES on a two-level p-multigrid
  preconditioner: the lowest-order unknowns, and every unknown of a cell in a
  perfectly matched layer or of a poor cell, are solved exactly, and the rest
  is smoothed around each mesh vertex. Its iteration count stays near 10 as a
  model grows in wavelengths. It needs about a quarter of the direct solver's
  memory at order 2, and it is faster from a few million unknowns on. It
  solves models with a perfectly matched layer, which `"bddc_gmres"` refuses.
  Order 2 or 3, 3D, one CPU process, with gradients and reuse across a
  sweep's frequencies; Bloch, periodic and Floquet models, MPI ranks, a GPU,
  2D meshes and non-driven problems raise, each naming its alternative
  ({doc}`../recipes/large-driven-model`).

- **`"auto"` picks it.** Behaviour change: for a driven problem on one CPU
  process whose direct factorization does not fit in half the memory,
  `"auto"` now picks `"pmg_gmres"` when its own estimate fits the memory the
  process may use, before `"bddc_gmres"`; with a perfectly matched layer,
  before the direct solver over its budget. Models that ran on the direct
  solver still do. The choice record on `gfem.solver` gives the new estimate,
  and each build of the solver logs its direct-set share, PML share and
  poor-cell count.

- **Messages.** The refusal of `"bddc_gmres"` with a perfectly matched layer
  now names `"pmg_gmres"`. A job that names `"pmg_gmres"` does not load in
  gfem 0.2.0 or earlier.

- **Documentation.** A recipe, "Solve a driven model too large for the direct
  solver"; the glossary terms p-multigrid, direct set and poor cell; a
  verification section; a one-node section of the performance data.

Free space:

- **Incident fields.** `Driven(sources=...)` takes sources beside the ports,
  each one more column on the same factorization. `gfem.PlaneWave` (TE, TM,
  circular or a Jones pair, at any angle), `gfem.GaussianBeam` (an exact beam
  carrying 1 W) and `gfem.IncidentField` (a field you give as a function)
  light a model whose `Model(background=...)` is one material or a
  `gfem.Background`: planar layers along z, optionally over a ground plane,
  lit from either side and past the critical angle. gfem solves for the
  total field outside the perfectly matched layer and the scattered field in
  it; `result.e_field(f, source, part=...)` reads the total, scattered or
  incident field ({doc}`../recipes/incident-fields`).
- **Cross sections.** `gfem.cross_sections` gives extinction, absorption and
  scattering from one exact power balance, and `gfem.power_flow` on an
  incident field's total field is exact too: minus it is one region's
  absorption ({doc}`../recipes/cross-sections`).
- **Point dipoles.** `gfem.ElectricDipole` and `gfem.MagneticDipole`, with
  `result.emitted_power`, `gfem.purcell_factor`, the port waves they send
  into each port (beta factors) and the split of their power between ports,
  the layer and lossy materials ({doc}`../recipes/point-dipoles`). They also
  run on the adaptive sweep.
- **Port waves of many dipoles by reciprocity.** `Driven(source_fields=False)`
  reads each point dipole's port waves from the ports' own solutions, one
  point overlap per dipole instead of one solve, so a coupling map over many
  positions costs one solve per port, on uniform and adaptive sweeps, with
  gradients. The dipoles' loads are stored sparse. Their emitted power, Purcell factor, kept fields and far fields
  are not computed, and those readers raise `ValueError`. A job with
  `source_fields=False` does not load in gfem 0.2.0 or earlier
  ({doc}`../recipes/point-dipoles`).
- **Many dipoles on an adaptive sweep.** The adaptive sweep no longer builds a
  matrix that grows with the square of its snapshot columns, and its reduced
  basis grows without copying, so many dipoles fit in memory. Samples and
  results are unchanged, to rounding.
- **Far fields and beam coupling.** `gfem.far_field` returns a
  `gfem.FarField` with the IEEE 145 antenna quantities (intensity,
  directivity, gain, realized gain, efficiency, circular components, axial
  ratio) and the radar cross section, over ground planes too.
  `gfem.beam_coupling` gives the amplitude a field sends into a fiber's
  beam ({doc}`../recipes/far-field`).
- **Gradients.** Every new output differentiates with respect to material
  fields; `power_flow`, `far_field`, `beam_coupling` and a dipole's emitted
  power also with respect to node coordinates. `gfem.shape.boundary_morph`
  now also holds the physical elements touching a perfectly matched layer
  or an absorbing face, where these measurements read.
- **Across ranks and workers.** Sources run on frequency workers and across
  MPI ranks; `power_flow`, `cross_sections`, `far_field` and
  `beam_coupling` are collective calls there. The solve cache files plane
  waves, beams and dipoles, and passes an `IncidentField` by with a
  `gfem.SolveCacheWarning`; a result or job holding one cannot be saved.
- **Documentation.** Two tutorials, a dipole emitting near a silicon or gold
  nanosphere and a grating coupler shaped for a fiber; four recipes
  (incident fields, cross sections, point dipoles, far fields); four
  sections of the conventions and a "Free space" section of the glossary.
- **Multipoles.** `gfem.multipoles(result, source, model=)` decomposes the
  field a scatterer radiates under a plane wave, a Gaussian beam or an
  incident-field function into electric and magnetic multipoles about an
  origin you choose. It gives power-normalized coefficients, the partial
  scattering cross section of each multipole up to a degree cap (they add up
  to the scattering cross section), and the exact electric and magnetic
  dipoles and quadrupoles in SI units, at any particle size.
  `gfem.current_moments` gives the long-wavelength electric dipole and the
  toroidal dipole of the induced current, to read anapoles. The multipoles
  differentiate with respect to material designs, so the magnetic dipole can
  be maximized ({doc}`../recipes/multipoles`, {doc}`../tutorials/meta-atom`).

Material data:

- **Library materials.** `gfem.materials` builds eight materials from one
  cited published source each: `silicon`, `silica`, `silicon_nitride`,
  `gold`, `copper`, `aluminium`, `lithium_niobate` (at one frequency, on
  your axes, with its clamped electro-optic tensor) and
  `lithium_niobate_static`. Each returns a plain `gfem.Material` whose
  `name` carries its source, such as `"silica (Malitson 1965)"`; the
  dielectrics also give a constant at one frequency with `frequency=`.

- **Fit a measured table.** `gfem.fit_permittivity` fits a passive law of a
  Drude term and Lorentz terms to tabulated $n$ and $k$, or to a complex
  permittivity, and returns a `gfem.PermittivityFit`: the law and its
  largest relative errors in $\varepsilon$, $n$ and $k$. It warns
  `gfem.PermittivityFitWarning` when it stops above its tolerance, and
  refuses gain, repeated frequencies and NaN with the remedy.

- **Fitted range.** `PermittivityLaw.frequency_range` holds the frequencies
  a library or fitted law was matched over. A driven, eigenmode, port mode
  or electrostatic solve that reads the law outside it warns
  `gfem.FittedRangeWarning` once and runs on. A law without a range writes
  the same job file as before; a file with one needs this version of gfem
  to read it.

- **Documentation.** A recipe, "Fit a measured n, k table to a permittivity
  law"; a "Take a law from cited data" section of the dispersive materials
  recipe; the extinction coefficient in the conventions; four glossary
  entries; and a "Material data" section of the licence page. The modulator
  tutorial takes lithium niobate's clamped constants from `gfem.materials`:
  its push-pull $V_\pi L$ moves from 1.61 to 1.62 V·cm.

Periodic surfaces:

- **Cell outputs.** `gfem.diffraction.efficiencies` gives the efficiency of
  each order and polarization of a Floquet-port solve (linear, circular or
  unpolarized); `power_balance` splits the incident power into reflectance,
  transmittance, absorptance and the power coupled to unreported orders; and
  `jones` returns the Jones matrix between two faces with its circular and
  Mueller forms.
- **`floquet_ports`.** `gfem.diffraction.floquet_ports(model, surface,
  frequency)` declares every propagating order of a face in both
  polarizations, at a given wave vector if the incidence is oblique.
- **Angle sweeps.** `gfem.diffraction.angle_sweep` solves a periodic cell at
  a grid of polar and azimuth angles, conical incidence included, and gives
  the helpers above for every member; one warning covers the orders that
  carry power without a port, and a rerun with the same cache resumes where
  an earlier one stopped.
- **Emissivity.** `gfem.diffraction.emissivity` reads the directional
  emissivity of a cell from the absorptance of the wave arriving from the
  same direction (Kirchhoff's law), with the azimuth label that law needs;
  `hemisphere` builds the quadrature over directions and
  `hemispherical_emissivity` averages with it.
- **Cell responses.** `gfem.diffraction.cell_response` solves a cell over one
  meta-atom parameter, on one mesh morphed from sample to sample, and stores
  its S with the derivative in the parameter; the response is evaluated at
  any parameter by Hermite interpolation, differentiable in the parameter
  and in the stored arrays, and saved to JSON.
- **Aperture fields and propagation.** `gfem.diffraction.aperture_field`
  places a cell of the response at every position of a large metasurface in
  the locally periodic approximation, and `propagate` carries the field to
  any distance with the angular spectrum, so a focal spot or a deflected
  beam is a differentiable function of the parameter map.
- **Metalens tutorial.** "A two-colour metalens" designs silicon nitride
  ridges that focus green and red light on one line, from a cell response
  to a gradient-optimized lens, and checks the locally periodic result
  against a solved supercell.
- **Behaviour change: diffraction-order arrays are NumPy constants.** The
  fields of `DiffractionOrderResult` (the order's wave vectors, angles and
  whether it propagates) are NumPy arrays, no longer JAX arrays, and stay
  concrete under `jax.jit`, so a traced function can branch on which orders
  propagate. Code that called a JAX method on one of them gets the NumPy
  equivalent.
- **Behaviour change: the unrequested-order warning is per polarization.**
  `UnrequestedOrderWarning` now fires for each propagating order and
  polarization without a Floquet port, not for each order: an order declared
  in TE alone warns about its TM part, and the message names the missing
  polarizations. Declare both, or filter the warning.

Design:

- **The electric energy differentiates.** `gfem.electric_energy` may be
  traced by `jax.grad`, `jax.jit` and `jax.vmap`, instead of raising, in the
  field's coefficients (a driven field or an eigenmode's), the permittivity
  of a region without a law (a scalar, per element or per axis), a
  permittivity law's parameters, a mode field's frequency and the node
  coordinates. The field of an adaptive sweep is read but raises
  `gfem.NonDifferentiableError`; solve with `sweep="uniform"`
  ({doc}`../recipes/measurements`).

- **Minimum width and spacing in the gradient.**
  `gfem.design.length_scale_constraints(blur, min_width=..., min_spacing=...)`
  turns a layer grid's minimum line width and minimum spacing into two
  differentiable constraints on the filtered density, each at most 0 when the
  drawn design's pattern keeps its rule, for an optimizer's inequality
  constraints or a penalty. `gfem.design.radius_for_length_scale` gives the filter radius
  whose robust triple holds a minimum length.

- **Layer patterns.** `LayerGrid.pattern(rho, eta=0.5)` draws a layer-grid
  design as polygons (`gfem.design.LayerPattern` of `gfem.design.Polygon`):
  the region where the filtered density reaches the projection threshold,
  the $\beta \to \infty$ design, with sub-pixel edges, closed along the
  footprint and cut at a periodic seam. `LayerPattern.from_polygons` builds
  and validates one from your own polygons (feed guides, hand-drawn designs),
  and `LayerGrid.area_fraction(pattern)` rasterizes one exactly back onto the
  pixels, to re-solve a pattern or pin fixed neighbours. contourpy is now a
  dependency.

- **GDS export.** `gfem.design.write_gds(path, {(layer, datatype): pattern})`
  writes layer patterns to a GDSII file for a layout tool or fab: one flat
  cell, several layers, datatypes (e.g. robust variants for review) and
  patterns per layer, vertices snapped to a settable database grid (default
  1 nm), holes as keyhole cuts, large polygons split to stay within what
  strict readers accept. The same arguments give the same bytes.

- **Minimum width and spacing checks.** `LayerPattern.check(min_width=...,
  min_spacing=...)` measures a layer pattern against a foundry's minimum width
  and spacing by its facing polygon edges, perpendicular to an edge, and
  returns a `gfem.design.PatternCheck`: the narrowest width and spacing (the
  margin, also when it passes), `passed`, and each `gfem.design.Violation`
  (kind, distance, location and markers to plot), narrowest first. Edges on
  the footprint are not measured; periodic patterns are measured across the
  seam. It is a pre-check, not foundry sign-off.

- **A fabrication model in the gradient.**
  `gfem.design.FabricationModel(grid, sigma=..., bias=...)` predicts the
  printed design of a drawn (projected) design: a Gaussian blur of standard
  deviation `sigma` (m) and a threshold that moves a straight edge outwards
  by the edge bias (m), differentiable in the design and in the bias.
  `fab.robust(drawn, beta=..., tolerance=...)` gives the eroded, nominal and
  dilated designs at the bias and $\pm$ a tolerance in metres, as the
  `Robust` tuple of `gfem.design.robust`, and `fab.pattern(drawn)` the
  printed layer pattern.

- **Documentation.** The bend tutorial now ends at its drawn pattern: checked
  against an example width and spacing rule, written to a GDS file and solved
  again, within 0.02 of the optimizer's last $|S_{21}|^2$; its numbers after
  the optimization are now those of the pattern. A new recipe, "Export a
  design's layer pattern to GDS", and the robust optimization recipe, now "Optimize a
  design for fabrication: etch, blur and minimum features", cover the
  fabrication model, minimum width and spacing constraints, fixed neighbours
  and acceptance checks; the glossary gains five terms.

Broadband models:

- **Time responses.** `NetworkParameters.step_response()` and
  `impulse_response()` turn S on a harmonic grid ($f_k = k\,\Delta f$) into
  a `gfem.TimeResponse`, as a network analyzer's low-pass mode does: the
  reflected step (TDR), `TimeResponse.impedance(port)` for the impedance
  profile, and the transmitted step (TDT). `impulse_response(mode="band_pass")`
  takes any evenly spaced band, for guides with a cutoff. Both differentiate
  with respect to S ({doc}`../recipes/time-responses`).
- **Rational models.** `NetworkParameters.fit_rational()` fits S with poles
  shared by every element, on one real reference impedance per port, and
  makes the model passive by default: a `gfem.RationalModel`, evaluated with
  `network(frequency)` and stepped exactly with `step_response(time)`. New
  warnings `gfem.RationalFitNotConvergedWarning` and
  `gfem.NonPassiveDataWarning`, and `gfem.PassivityEnforcementError`
  ({doc}`../recipes/rational-model`).
- **Netlists.** `RationalModel.to_spice()` and `to_verilog_a()` write the
  model as a SPICE subcircuit and a Verilog-A module, with two pins per
  port, checked in ngspice and VACASK.
- **Documentation.** Two recipes, "See S in the time domain" and "Export S
  as a SPICE or Verilog-A model"; a "Time responses" section of the
  conventions and a "Broadband models" section of the glossary.

Circuits:

- **gfem solves as SAX components.** `gfem.SaxComponent(model, problem,
  circuit=..., ports=...)` turns a function from design settings to a
  `gfem.Model`, and a `gfem.Driven` band, into a model of a SAX circuit, so one
  `jax.grad` of the circuit reaches gfem's inputs and SAX's analytic models
  together. A photonic component takes the wavelength `wl` in µm and returns
  the complex conjugate of S, SAX's photonic phase convention; an RF component
  takes the frequency `f` in Hz and the reference impedance `z0` (50 Ω by
  default) and returns S renormalized to it. The component solves its whole
  band once per design and interpolates any wavelength or frequency inside it;
  instances with equal settings share one solve
  ({doc}`../recipes/sax-circuit`).
- **S for SAX and scikit-rf.** `NetworkParameters.to_sdict(circuit=...,
  ports=..., z0=...)` gives a solved S as a SAX dictionary, every port pair
  included, differentiable; `NetworkParameters.to_skrf(ports=..., name=...)`
  gives a scikit-rf network of pseudo-waves on the ports' own reference
  impedances.
- **Polarity reference.** `WavePort(polarity=(point, direction))` makes the
  port mode's field at a point real and positive at every frequency, so the
  sign of the port's row and column of S no longer depends on the mesh or, in
  3D, on the model's orientation. A SAX component requires it on every wave
  port it exposes that has no `voltage_path`.
- **New extras.** `gfem[sax]` (SAX 0.16.14 or later) and `gfem[skrf]`
  (scikit-rf 1.8 or later). gfem imports neither: `SaxComponent` and
  `to_sdict` return plain dictionaries.
- **Documentation.** A recipe, "Differentiate a SAX circuit with gfem blocks",
  a "SAX circuits" section of the conventions and of the verification page, a
  "Circuits" section of the glossary, and a scikit-rf section in "Save and
  export results".

Eigenmode solves:

- **Rerun eigenmode results on lossy models with a target far below the
  modes.** On a model whose loss varies in space (a lossy region beside a
  lossless one), with the target well below the nearest mode, `gfem.Eigenmode`
  could drop that mode without a word and return the next ones. It now
  returns the modes nearest the target; such results from earlier versions
  should be rerun.

Meshes:

- Fixed: a cross-section meshed periodically in gmsh (`setPeriodic`) pairs
  under `gfem.Periodic` whatever its node numbering, on any lattice.
  `gfem.Mesh.from_gmsh_cross_section` now splits the prisms by node
  coordinates, not node numbers; values computed on a cross-section move by
  up to about 4e-4 relative (the tutorials' checked values).

Wave ports:

- **Removed `gfem.PortModeProjectionWarning`.** It has not been emitted since
  0.1.0, when wave-port S became exact on lossy port lines. Delete any
  warning filter that names it.
- **Designs at wave ports.** A design may reach a wave port face: S, port
  waves, kept fields, `port_modes` and `z_ref` differentiate with respect to
  the materials of the face's elements (laws and material fields included),
  a sheet's fill and an `Impedance` on its edge, and its node coordinates,
  lone or on a shared port surface, on every driven sweep and solver and
  across ranks (not under a Bloch wave vector or on a fixed-angle sweep). Line
  ports of 2D models differentiate their materials the same way
  ({doc}`../recipes/port-mode-gradient`).
- **Port modes differentiate.** A `gfem.PortMode` result's `gamma`,
  `n_eff`, `Z_PV` and field (`e_field.at`) differentiate with respect to
  the same inputs; `residual`, and the group index and dispersion (`n_g`,
  `beta_2`, `beta_3`, `D`), raise `gfem.NonDifferentiableError`.
- **Sliding ports.** `gfem.shape.boundary_morph(..., sliding_ports=...)`
  lets the faces of the named wave ports slide in their plane, so a shape
  design reaches the port.
- **Polarity reference with gradients.** A port mode's field outputs
  differentiate exactly when its `WavePort` has a `polarity` reference: the
  gradient follows the phase the reference fixes.
- **Voltage path.** A `WavePort` whose `voltage_path` has a point outside the
  port section now raises `ValueError` in the solve.

## 0.2.0 (2026-11-23)

- **Native 2D models.** A gmsh file of triangles at one height reads as a
  {term}`2D model`; `gfem.Driven(polarization="Ez")` or `"Hz"` solves it with
  line ports, PEC, PMC and absorbing edges, and adjoint gradients, including
  layer-grid designs. A 2D model solves a far smaller system than a slab of
  the same plane. The bend, metal-bend and topology tutorials now run in 2D
  ({doc}`../recipes/2d-models`).
- **Perfectly matched layers.** {py:obj}`gfem.PML` stretches a meshed shell of
  layer regions around the air box at a reference frequency, in driven,
  eigenmode and port mode solves, with gradients through the layer. It
  absorbs waves that meet the boundary obliquely and gives the radiation Q of
  a resonator ({doc}`../recipes/perfectly-matched-layer`). The new warnings
  `gfem.ThinLayerWarning` and `gfem.LayerClearanceWarning` flag a layer too
  thin for its reference wavelength, and a port mode or eigenmode that keeps
  over about 1% of its electric energy in the layer. A 2D model refuses a
  layer.
- **JAX 0.8.3 or later, NumPy 2.2 or later.** gfem now installs next to
  packages that cap JAX below 0.10 (0.1.0 required JAX 0.12). NumPy 2.2 is
  now required directly: on NumPy 2.0, `Job.to_json` and the solve cache
  wrote a stray `allow_pickle` array into their `.npz` files. CI runs the
  unit tests on the oldest supported versions.
- **BDDC's coarse problem across ranks.** With `bddc_gmres` on the
  {term}`MPI PETSc stack`, MUMPS factors BDDC's coarse problem on a share of
  the ranks. gfem picks the share and the ordering from the stack, the coarse
  problem's size and the number of ranks and nodes:
  - On a stack built by the current script, PT-Scotch orders it on every
    fourth rank, at least 30 000 coarse DOFs each, placed so that every node
    holding one holds as many: 7 per node at 28 ranks per node. The ordering
    cap below keeps its fill flat as those ranks grow, and every set-up takes
    the same route.
  - On a stack you built with ParMETIS, ParMETIS orders it on the same ranks.
  - On a stack built by an earlier script, without the cap, PT-Scotch orders it
    on at most one rank per 100 000 coarse DOFs and at most 56 ranks, where its
    fill stays low. Sequential METIS orders it on every fourth rank, at least
    30 000 coarse DOFs each, when that gives more ranks than PT-Scotch may take
    and the set-up will serve 6 right-hand sides or more, or repeats a mesh
    already set up in this process (the designs of a loop, a sweep's rebuilds).
    Its ordering and symbolic factorization cost more than PT-Scotch's, and
    only many right-hand sides or keeping them repays it.

  The same script on the same ranks and stack always takes the same route. It
  is not an option: $S$ stays that of the coarse problem factored on one rank,
  to round-off. A singular coarse problem now raises at once; only a MUMPS
  workspace error is retried, with more workspace, which a PT-Scotch
  factorization starts with twice as much of.
- **The MPI stack caps PT-Scotch's parallel ordering: rebuild it to get the
  cap.** The MUMPS that
  [`tools/petsc_wheelhouse.sh`](https://github.com/benvial/gfem/blob/main/tools/petsc_wheelhouse.sh)
  builds now orders with PT-Scotch on at most 32 processes, where every node
  holds as many processes. Without the cap, MUMPS orders on the largest power of
  two of its processes, and the fill grows with them. On another layout, such as
  `direct` with unequal ranks per node, MUMPS still orders on all of them. The
  variable `MUMPS_PARAORD_MAX` sets another cap for a run, of 2 or more (`1`
  counts as `2`), or `0` for none. A stack built before this change has no cap: rebuild it with the
  current script, as {doc}`../recipes/petsc-slepc` says. Until then gfem keeps
  the earlier coarse route above; where a rebuilt stack would factor the
  coarse problem on more ranks, it says how many, once per process, at INFO
  level on the `"gfem.solver"` logger.
- **ParMETIS, on your own MPI stack.** `tools/petsc_wheelhouse.sh --mpi
  --with-parmetis` adds ParMETIS to the MPI stack, and gfem then orders BDDC's
  coarse problem with it. ParMETIS is free only for educational and research
  use by non-profit institutions and US government agencies, and for
  evaluation by others; it may not be redistributed. The script prints these
  terms before it builds. A stack with ParMETIS is for its builder alone: do
  not share or upload its wheels. gfem's own stacks never hold it.
- **Repeated solves keep BDDC's coarse ordering.** Under ranks, gfem keeps a
  coarse factorization when its set-up is freed, and a later set-up on the same
  mesh in the same process refactorizes it with new values instead of
  ordering it and factorizing it symbolically again. A kept factorization holds its factors in memory, so
  under ranks `retain_memory` now also bounds the coarse factorizations kept
  between solves: `retain_memory=0` keeps none past a solve, and
  `gfem.release_memory()` frees them.
- **Documentation.** A page on {doc}`modes and ports <../modes-and-ports>` tells a port mode from a
  resonant mode and states the three roles of a port; a {doc}`Benchmarks <benchmarks>` page
  reports the November 2026 comparison on MareNostrum 5.

## 0.1.0 (2026-11-20)

First release on PyPI (`pip install gfem`): driven problems (uniform and
adaptive sweeps), eigenmode problems (including nonlinear ones), electrostatic,
port mode and thermal problems, with lumped, wave and Floquet ports, on one
process or across MPI ranks. Gradients with respect to permittivity,
conductivity, sheet fill and mesh node coordinates are checked against finite
differences.

The 2026-11-01 preview was tagged 0.1.0 in the repository and never published
on PyPI; the tag now names this release. The preview stays at commit
[`18235aa`](https://github.com/benvial/gfem/commit/18235aa). Changes since the
preview:

- **Solve cache.** `gfem.solve(..., cache="solves")`, `Job.run(cache=...)` and
  `python -m gfem run ... --cache solves` keep each finished result in the
  named directory, filed under the mesh, materials, problem, solver options
  and gfem version, and read it back when the same call comes again: the
  same result type and values, its warnings shown again. Gradients and
  solves that raise are solved and store nothing. Off unless a call names a
  directory; the logger `"gfem.solve_cache"` reports each read and store
  (save and export recipe, "Solve once, read it back").

- **Error estimates of loaded results.** `gfem.error_estimate` accepts a
  result read with `gfem.load_result` and the model it was solved on: the
  mesh is compared by its elements, regions and node positions, not as the
  same object. Earlier versions raised `ValueError` for any loaded result.

- **Behaviour change: direct driven solves keep their digits below the
  low-frequency floor.** A new option, `SolverOptions.gauge`, solves a driven
  problem through the tree-cotree gauge: the gradients $\nabla \phi$, on
  which the curl-curl term vanishes, become node potentials and one floating
  potential per conductor. It is on by default (`"auto"`) wherever the linear
  solver resolves to `"direct"` on the CPU in one process, for a model with
  no periodic pair, Bloch wave vector or Floquet port, in uniform and adaptive
  sweeps and their gradients. Below the floor
  ($k_0 h_\mathrm{min} < 10^{-6}$) such a solve now keeps S, the fields and
  their gradients to round-off: on a
  parallel-plate line at 10 kHz, S is within $10^{-13}$ of its closed form.
  `gauge="tree_cotree"` asks for the gauge and raises where it does not
  apply; `gauge="none"` restores the previous path bit for bit.
  Eigenmode, electrostatic, DC conduction, thermal and port mode solves
  ignore the option.
- **Above the floor, S with `"direct"` changes by $10^{-13}$ to
  $10^{-10}$**, towards the exact answer. Set `gauge="none"` to reproduce
  results from earlier versions exactly. A new mesh topology pays a setup of
  about one factorization, reused across shape steps, and each
  refactorization costs about 10 % more.
- **`LowFrequencyWarning` fires less often.** A gauged solve does not warn
  below the floor, unless the model has wave ports: their mode problem is not
  gauged and keeps its own floor, and the warning then says to use lumped
  ports at those frequencies. An ungauged solve warns as before, and the
  message now ends with why it is ungauged and, where one exists, the setting
  that gauges it. For capacitance, solve `gfem.Electrostatic`.

- **Phase timings on the `"gfem.timing"` logger.** Silent by default; when
  enabled, every driven and eigenmode solve logs one record per phase
  (discretize, assemble, setup, solve) on each process, with the number of
  unknowns, the solver's bytes and the Krylov iterations. See "Time each
  phase" in
  {doc}`../recipes/solve-a-device`.

- **Behaviour change: exact S on lossy wave ports.** A wave port's S now reads
  the field's modal amplitudes by the biorthogonal projection on its modes,
  and S relates pseudo-waves (Marks and Williams). With a `voltage_path`, a
  wave port's `z_ref` is its mode's complex $Z_{PV}$, so a uniform lossy line
  is matched to its own mode ($S_{11} = 0$, $S_{21} = e^{-\gamma L}$), and
  `s.y()`, `s.z()` and `s.renormalize()` give the line's exact network. Earlier
  versions projected on the conjugated mode, which reflected a lossy line's own
  mode at about $\alpha/\beta$ and mixed the modes of a lossy shared port
  surface; S of lossy wave ports from earlier versions should be recomputed.
  Lossless wave ports and lumped ports are unchanged. `z_ref` is now complex,
  `gfem.conventions.WAVES` is `"pseudo"`, and `PortModeProjectionWarning` is no
  longer emitted (the class stays for warning filters). Give every wave port a
  `voltage_path`, or none: on a lossy line S is symmetric only between ports of
  one kind.

- **Complex characteristic impedances.** `PortModeResult.Z_PV` is now the
  complex power-voltage impedance $|V|^2 / (2 P^*)$, with $P$ the mode's
  complex power. Earlier versions returned its modulus $|V|^2 / (2 |P|)$,
  which differs on a lossy line (conductor loss at low frequency makes $P$
  strongly reactive). `gfem.characteristic_impedance` takes a new
  `current_path` and returns the power-current impedance $Z_{PI} = 2P / |I|^2$
  with it, or the voltage-current impedance $Z_{VI} = V / I$ with both paths.
  A wave port's reference impedance `z_ref`, and so its S-parameters, are
  unchanged: `z_ref` stays the real modulus.

- **Rerun eigenmode results on models with lossy walls.** `gfem.Eigenmode` now
  returns the modes nearest the target on a model with a `Conductor`, `Impedance`
  or `Absorbing` wall. Its starting vectors were wrong on such walls, so a solve
  could skip modes without a word, or return a different set from run to run.
  Eigenmode results on models with `Conductor`, `Impedance` or `Absorbing` walls
  from earlier versions may have skipped modes and should be rerun.
  A symmetric model with such walls could also return only one mode of a
  degenerate pair, depending on the number of threads; the second mode is now
  recovered when the solve comes close to it.

Thermal:

- **Behaviour change: `density` is now `mass_density`.** The material
  property read by transient thermal problems is named `mass_density`, in
  kg/m³, on `gfem.Material` and `gfem.MaterialField`. The keyword
  `density=` still works for one release: it warns with a
  `DeprecationWarning` and sets `mass_density`. Reading `.density` now
  raises `AttributeError`, naming `mass_density`. Job files that store
  `density` read as `mass_density`.
- **Problems.** `gfem.SteadyThermal` (the temperature under heat sources),
  `gfem.TransientThermal` (the temperature over time under switched sources,
  on a uniform time grid, recording traces at every step) and
  `gfem.DCConduction` (the current through conductors between terminals:
  conductance, potentials and Joule heat). Results: `SteadyThermalResult`,
  `TransientThermalResult` and `DCConductionResult`, with absolute
  temperatures in K and the rise above the source-free temperature.
- **Sources.** `gfem.HeatSource.uniform` spreads a power over regions;
  `gfem.joule_heating` turns a DC conduction result and a voltage, current
  or power into heat, with no new solve.
- **Boundaries.** `gfem.FixedTemperature`, `gfem.Convection` and
  `gfem.HeatFlux`; a surface no thermal boundary names is insulated
  (`Model.default_insulated`).
- **Materials.** `thermal_conductivity`, `mass_density` and `specific_heat`
  are read, and `thermo_optic_coefficient` ($dn/dT$, 1/K) is new. A thermal
  solve missing a property raises `gfem.errors.MissingMaterialPropertyError`,
  listing every region and property at once.
- **One model, several physics.** `excluded=` takes a mapping from physics
  (`"em"`, `"thermal"`, `"dc_conduction"`) to regions; a list keeps its
  meaning, electromagnetic solves only. `Model.excluded_from(physics)`
  reads it back.
- **Cross-sections.** `gfem.Mesh.from_gmsh_cross_section` reads a 2D gmsh
  section as a slab one element thick; on it every extensive input and
  output is per metre (`power_per_length`, `resistance_per_length`).
- **Measurements.** `gfem.thermo_optic_shift` (the phase per length a rise
  adds to a guided mode, to first order), `gfem.route_phase` (the phase
  along routes through a 3D rise), `Model.with_temperature_rise`,
  `gfem.Trace`, `gfem.rise_time`, `gfem.time_constant` (with
  `gfem.UnsettledTraceWarning` on a trace that has not settled) and
  `gfem.element_mean`.
- **Gradients.** `jax.grad` through DC conduction, Joule heating, steady and
  transient temperatures and the thermo-optic phase, with respect to
  per-element thermal and electrical conductivity, mass density, specific
  heat and thermo-optic coefficient, the heat sources, the drive and the
  waveform's scales. Node coordinates, the optical model's permittivity
  under the shift, and the rise time and time constant raise.
- **Solvers.** `"direct"` by default, `"bddc_gmres"` and `"petsc"` accepted;
  `SolverOptions.time_scheme` (`"tr_bdf2"`, the default, or `"theta"` with
  `SolverOptions.theta`). The GPU solver, MPI ranks, mesh refinement and
  error estimates raise on thermal problems.
- Job files carry the thermal problems, sources, boundaries and results, and
  an `excluded=` mapping; an older gfem refuses such a file by name.
- New pages: the tutorials {doc}`../tutorials/heater` and
  {doc}`../tutorials/heater3d`, and the recipes
  {doc}`../recipes/heater-temperature`, {doc}`../recipes/thermo-optic-phase`,
  {doc}`../recipes/thermal-step-response` and
  {doc}`../recipes/thermal-gradient`. The conventions page gains a
  "Temperature and thermo-optic phase" section, {doc}`../verification` a
  "Thermal" section, and the glossary the thermal terms.

Floquet ports:

- **Floquet ports.** `gfem.FloquetPort` lights a periodic cell with a plane
  wave and reads its diffraction orders. Each port is one order in one
  polarization (TE or TM) on an open face of the cell, with its own row and
  column of S; $|S|^2$ is the order's diffraction efficiency, and the row of an
  order that does not propagate reads exactly 0. The ports of a face absorb
  every order at any angle. See {doc}`../recipes/floquet-ports`.
- **Incidence.** `Model.bloch_wave_vector` turns incidence angles into a
  wave vector, `Model.propagating_orders` lists the orders that propagate
  behind a face, and `DrivenResult.diffraction_orders` gives each port's
  order, angles and propagation constant (`gfem.DiffractionOrderResult`).
  `gfem.Driven(..., wave_vector_frequency=)` keeps the incidence angle fixed
  across a sweep.
- **Gradients.** Design gradients (materials and nodes inside the cell)
  through Floquet S, on the direct and BDDC + GMRES solvers, with plain
  periodicity and under a Bloch wave vector. A gradient with respect to the
  angle, `wave_vector_frequency`, the half-space's material, a Floquet
  face's nodes or a port's `offset` raises.
- **Which problems.** The uniform driven sweep, frequency workers included.
  The adaptive sweep, `linear_solver="petsc"`, the GPU solver, MPI ranks,
  mesh refinement, error and quantity estimates, eigenmode solves and the
  port mode problem raise on a model with Floquet ports, as do a lossy,
  dispersive, anisotropic or inhomogeneous half-space and a face on one
  periodic pair with side walls.
- **New warnings.**
  - `gfem.ObliqueAbsorberWarning`, a change of behaviour: a driven solve of
    a periodic model now warns when a propagating wave meets a
    `gfem.Absorbing` face with a reflection of at least 1 %. Existing
    scripts with oblique absorbing faces in periodic cells now warn; filter
    the warning by class to silence it.
  - `gfem.UnrequestedOrderWarning`: a propagating order has no port, so its
    power is missing from S.
  - `gfem.FloquetOrderCapWarning`: the face mesh is too coarse to carry the
    evanescent orders the structure excites.
- **Changed message.** A `gfem.WavePort` touching a periodic pair now points
  to `gfem.FloquetPort`.
- **Periodic design grids.** `design.LayerGrid(..., periodic=(True, False))`
  filters across the seam of a periodic cell and wraps its element map.
- Job files carry Floquet ports and `wave_vector_frequency`; older files read
  unchanged, and an older gfem refuses a newer file by name.
- New pages: the tutorial {doc}`../tutorials/metagrating` and the recipe
  {doc}`../recipes/floquet-ports`. The conventions page gains a "Floquet
  ports" section, {doc}`../verification` a "Floquet ports" section, and the
  glossary the Floquet terms.

Dispersive materials:

- **Permittivity laws.** A `gfem.Material`'s `eps_r` may now be a
  `gfem.PermittivityLaw`: a high-frequency permittivity plus `gfem.Drude`,
  `gfem.Lorentz`, `gfem.Debye` and `gfem.DjordjevicSarkar` terms, with every
  frequency in Hz. `PermittivityLaw.djordjevic_sarkar` fits a datasheet
  permittivity and loss tangent. A law is callable, `law(f)`, and each
  term is passive alone; a negative strength raises. Metals ($\mathrm{Re}\,\varepsilon < 0$) are
  allowed. See {doc}`../recipes/dispersive-materials`.
- **Which problems.** Driven uniform and adaptive sweeps (every linear
  solver, frequency workers, MPI ranks, plain and Bloch periodic cells),
  eigenmode solves outside a law's {term}`unresolved band`, the port mode
  problem where $\mathrm{Re}\,\varepsilon > 0$ in the cross-section, and the electrostatic problem
  through the {term}`static permittivity`. A law region behind an `Absorbing`
  face needs $\mathrm{Re}\,\varepsilon > 0$ on a driven sweep and raises on an eigenmode solve. A
  Drude law raises in the electrostatic problem, and a metal in a port
  cross-section raises before any solve.
- **Gradients.** The law's numbers differentiate driven S and kept fields
  (uniform and adaptive sweeps, wave ports, plain periodic cells, behind
  absorbing faces), eigenfrequency and Q, and capacitance. Every existing
  input differentiates beside a law. By a law's numbers, a mode's
  participations and field, `gfem.electric_energy`, the port mode problem
  and a Bloch cell raise `gfem.NonDifferentiableError`.
- **Metal-dielectric topology optimization.**
  `MaterialField.interpolate(..., scheme="index")` blends the refractive
  index, the blend to use between a metal and a dielectric, and every scheme
  takes a law at either end. A blend with a law end solves on the driven
  uniform sweep only. Start such an optimization from a binary design: a
  density of 0.5 absorbs. New tutorial: {doc}`../tutorials/metal-bend`.
- **The adaptive sweep** solves both grid neighbours of a Lorentz resonance
  inside the band first, so an in-band resonance costs a few more samples.
- **Mode energy and measurements.** In a law region, the mode energy weighs
  $|E|^2$ with the {term}`energy permittivity`, and `gfem.electric_energy` with
  $\mathrm{Re}\, d(\omega\varepsilon)/d\omega$; both raise where that weight is not positive.
  `gfem.interface_participation` raises when a selected side lies in a law
  region. Constant-material results are unchanged.
- Job files and saved results carry laws; older files read unchanged.
- New pages: the recipe {doc}`../recipes/dispersive-materials` and the
  tutorial {doc}`../tutorials/metal-bend`. The conventions page gains the
  law forms, {doc}`../verification` a "Dispersive materials" section, and
  the glossary the dispersive terms.

Quantum metrics and mode-field gradients:

- **Participation ratios and port Q differentiate.** `participation`,
  `q_port`, `bulk_participation`, `principal_participation` and
  `sheet_participation` of an eigenmode result take gradients with respect to
  materials, sheet fill, node coordinates and lumped values, for an isolated
  mode (a relative frequency gap of at least 1e-4 to every other mode found).
  Each mode an objective reads this way costs one extra linear solve on the
  eigensolve's factorization. `principal_participation` raises with respect to
  a lumped value. See {doc}`../recipes/eigenmode-gradient`.
- **Mode fields differentiate.** `e_field`, `Field.at` and
  `gfem.interface_participation` of an eigenmode result take gradients with
  respect to materials, sheet fill and lumped values; with respect to node
  coordinates they raise `gfem.NonDifferentiableError`.
- **Lumped values are mode-field inputs.** A `LumpedPort`'s inductance and
  capacitance and an `Impedance`'s values reach the participations and the
  mode fields, so a junction's `L_J` and a film's kinetic inductance are
  design inputs.
- A mode-field gradient raises `gfem.NonDifferentiableError` on periodic and
  Bloch models, with `linear_solver="bddc_gmres"` or `"petsc"`, with
  `device="cuda"`, through `result.error`, and when `eigen_tolerance` is above
  1e-10. A member of a degenerate cluster, a cluster cut by `n_modes` or a
  nearer pair raises `gfem.DegenerateModeError`, and a moved port node
  `gfem.PortGeometryError`. A gradient through an infinite `q_port` raises
  `gfem.NonDifferentiableError`: differentiate `1 / q_port`. The frequencies
  differentiate as before in every case.
- **`gfem.quantum`.** `gfem.quantum.dressed_modes(omega, participation,
  inductance=L_J)` puts one junction's cosine on the solved modes and returns
  `gfem.quantum.DressedModes`: dressed frequencies, Lamb shifts,
  anharmonicities and dispersive shifts, in rad/s, differentiable to first
  order. `method="first_order"` gives the perturbative closed forms. A level
  whose label is ambiguous raises the new `gfem.ModeLabelError`. A reverse-mode
  gradient through `DressedModes.overlap` raises
  `gfem.NonDifferentiableError`; its forward-mode tangent is NaN. The
  conventions, with conversions to other sources', are in `gfem.conventions`.
  See {doc}`../recipes/quantum-metrics`.
- **`EigenmodeResult.sheet_participation`**: the share of each mode's energy
  in the inductance of each `Impedance` sheet with one, the kinetic inductance
  fraction of a superconducting film for a lossless mode.
- **Behaviour change: modes are polished.** On the direct solver, each mode
  that can take a mode-field gradient now gets a few Newton steps after the
  eigensolve. Participations, `q_port`, fields and interface participation
  of such modes move by about 1e-6 to 1e-4 relative, and the default
  `eigen_tolerance` is enough for participations and their gradients. A mode
  whose polishing does not converge is left unpolished and takes no
  mode-field gradient: a gradient through its field raises
  `gfem.DegenerateModeError` naming it. A plain solve never raises because of
  polishing.
- **Behaviour change: the mode energy counts capacitive sheets.** Each mode's
  1 J of electric energy now includes $\tfrac{1}{2} \int C_s |E_t|^2\, dS$ on every `Impedance`
  sheet with a capacitance. On models with such a sheet, field amplitudes,
  participations and `q_port` change; models without one are unchanged.
- **Behaviour change: `interface_participation` adds the two faces of a
  film.** For `"MA"` and `"MS"`, each selected side of a surface carries its
  own layer and the two layers' energies add. On a film with the same medium
  on both faces the value rises from near zero to the sum of the two faces;
  on a film between substrate and vacuum it is unchanged. `"default"` and
  `"SA"` still take the mean field of the two sides.
- New pages: the tutorial {doc}`../tutorials/transmon`, and the recipes
  {doc}`../recipes/quantum-metrics` and {doc}`../recipes/loss-budget` (a
  mode's $1/Q$ from its loss channels, and their $T_1$ limits). The capabilities
  matrix gains the rows "Eigenmode, participations" and "Eigenmode, mode
  field", and the glossary the quantum terms.

Eigenmode solves:

- **Behaviour change: the eigen solver's first pass stops after 100
  restarts.** Where the modes near the target overlap within their linewidth,
  a solve could run for hours. It now raises
  `gfem.EigenmodeConvergenceError` after 100 restarts, naming the lossy
  region with the most modes within one linewidth of the target (its
  {term}`modal overlap`) and, for a permittivity law, its term nearest the
  target. Move the target where the modes are separated, or find the
  resonances from a `gfem.Driven` sweep. With `eigen_solver="slepc"`,
  `eps_max_it` in `SolverOptions.petsc_options` overrides the bound.
- When a pair nearer the target than the farthest mode found stops within 1e-6 of
  an eigenpair without converging, `gfem.Eigenmode` now raises
  `gfem.EigenmodeConvergenceError` instead of returning a set of modes without it.
- A heavily inductive `Impedance` wall can now return its sheet modes near the
  target, where cavity modes farther away were returned before.
- `gfem.Eigenmode` warns with `gfem.LowFrequencyWarning` when its target or its
  lowest mode is below the low-frequency floor ($k_0 h_\mathrm{min} = 10^{-6}$). The reported
  `error` does not show the digits lost there.
- A linear solve that does not converge inside an eigenmode solve now names
  `SolverOptions.eigen_tolerance` as the remedy, with the inner tolerance in use.
- With `eigen_solver="slepc"`, a shift solve that fails while seeding no longer
  leaks SLEPc's eigen solver; the solve's own error is raised as before. Bloch
  eigenmode problems with `eigen_solver="slepc"` are supported and tested.
- `gfem.interface_participation` decided whether a surface belongs to a periodic
  pair from a cache kept per mesh. A wall of a non-periodic model was refused once
  a `gfem.Periodic` model had been built on the same mesh, and a paired surface was
  measured without a word after it had been measured on another model or on a
  result read with `gfem.load_result`. It now reads the pairs the result records,
  in the new `EigenmodeResult.periodic_surfaces`, which is saved with the result.
  Results saved by earlier versions read as having no pair.

Optimization and gradients:

- **Optimization loops redo less work per design.** Within one process, a driven
  solve of a new design with the same mesh, ports and frequencies takes up the
  wave ports' couplings instead of rebuilding them. With `linear_solver="direct"`
  on the CPU in one process, it also takes up the last design's solver plan, the
  assembled parts that do not depend on the design and, while `retain_memory`
  has room, one factorization per frequency, with or without the `mkl` extra. A
  model above the reuse floor may solve the next design at a frequency on that
  frequency's earlier factorization, with GMRES to a twentieth of `tolerance`,
  while the counted costs (factor size, factorization work, GMRES iterations)
  say that is cheaper than factorizing again; small models factorize every
  design. See {doc}`../recipes/s-parameter-gradient`.
  Measured on MN5 at 8 threads, 9f004a7 against 28a2bf4, the gain is small
  or absent where a reused step costs about what a factorization costs: a
  3D bend of 381k unknowns took 8.93 s per step before and 9.13 s after,
  reusing on 19 of 41 steps instead of 9, and a 4-port coplanar line of
  1.32M unknowns 14.6 s against 13.8 s per gradient step. On a 2.49M-unknown
  3D bend at 16 threads under the default `retain_memory`, 28a2bf4 reused on
  10 of 21 steps, and its solver time was 1.10x shorter than a replay of
  9f004a7's history that factorizes again at every step (1625 s against
  1783 s).
  - **Behaviour change.** `retain_memory` now bounds everything gfem keeps between
    solves, these kept parts and couplings included; 0 keeps nothing and
    factorizes every design. A direct solve on an earlier design's factorization
    is accurate to `tolerance`, not exact.
  - **Reuse decisions repeat for a script.** gfem decides what to reuse from
    counts, never from times, so a script run again with the same thread count
    and an explicit `retain_memory` reuses at the same designs. S and gradients
    depend on the design history only within `tolerance`. A finite-difference
    check across designs must call `gfem.release_memory()` (or set
    `retain_memory=0`) before each evaluation: a solve on an earlier design's
    factorization or from its solution differs from a fresh one within
    `tolerance`, and that difference does not cancel between the evaluations.
- **Behaviour change: `linear_solver="auto"` picks the direct solver for driven
  problems that fit.** On the CPU in one process, a driven problem now solves
  with `"direct"` when its factorization, estimated at about 48 KiB per unknown
  and per frequency worker, fits in half the memory the process may use, and
  with `"bddc_gmres"` above. It was `"bddc_gmres"` at every size. Measured on
  MN5 (8 threads, at 9f004a7), direct took 1.2-48x less time per optimization
  step than `"bddc_gmres"` from 24k to 2.49M unknowns, at 1.3-2.8x the memory.
  Without the `mkl` extra the choice is the same and the direct solver is
  sparse Cholesky; on a two-port lumped-port line, `"bddc_gmres"` can then be
  faster. Electrostatic problems keep `"bddc_gmres"`, eigenmode problems and
  GPUs `"direct"`, and solves across MPI ranks `"bddc_gmres"`. Where the default
  changes, S moves by the iterative solve's error. An explicit `linear_solver="direct"`
  lifts the memory guard.
- **Each frequency keeps its own factorization across designs.** A uniform
  sweep, and an adaptive sweep with frozen samples, keeps one direct
  factorization per frequency (per sample) while `retain_memory` has room, so
  every frequency, not only the first, can solve the next design on its own
  earlier factors, and a gradient's adjoint solves factorize nothing again:
  one factorization per frequency and design instead of two for all but one
  frequency. A frequency gets its own factorization the second time a sweep
  meets it, a gradient's backward counting with its forward: a single solve
  or gradient, or a sweep over new frequencies (a greedy adaptive sweep),
  keeps one, and a design loop keeps one per frequency from its second design. Frequency workers keep none between solves.
  Measured on MN5 at 8 threads, 9f004a7 against 28a2bf4, a 571k-unknown 3D
  bend at three frequencies with `retain_memory` at 128 GiB reused 23 of 33
  frequency solves over 11 steps instead of none, but a step was only 1.06x
  faster (46.1 s against 49.0 s), at 2.4x the peak memory (25.4 GiB against
  10.8 GiB).
- **Sparse Cholesky joins factor reuse.** Without the `mkl` extra, the direct
  solver's sparse Cholesky reuses a design's factorization for the next design
  under the same rule as MKL PARDISO.
- **`bddc_gmres` starts each design from the previous solution.** In one
  process, a uniform sweep's first frequency starts its GMRES iterations from
  the previous design's solution there, and the adjoint solves of a
  multi-frequency gradient no longer set BDDC up again on a preconditioner
  kept from another frequency. These initial guesses count in
  `retain_memory` and are the first thing dropped over it.
  Measured on MN5 at 8 threads, 9f004a7 against 28a2bf4, lumped-port coplanar
  lines of 338k and 1.32M unknowns with 2 or 4 ports took 1.3-1.5x less time
  per optimization step at one frequency (GMRES iterations per column from
  43-46 to 27-28), and a one-port gradient over three frequencies on the 1.32M line
  set BDDC up once per step instead of 4 times (4.05 s against 15.05 s of
  setup, 1.16x per step). A 3D bend of 1.02M unknowns gained 2 %.
- **Electrostatic energy and potential gradients prepare once per step.** A
  differentiated electrostatic solve keeps its prepared operator, within
  `retain_memory`, and the backward of an `energy` or `potential` objective
  solves on it instead of preparing again. Capacitance gradients are
  unchanged.
- **Behaviour change: the default `retain_memory` keeps a large model's last
  factorization.** Under `retain_memory=None`, the last factorization stays
  held, even over half the memory, while the process's peak memory leaves a
  tenth of the memory free, so a model whose factorization takes more than half
  the memory can still reuse it. The decision then follows the measured peak:
  on a machine shared with other memory-heavy processes, or to make the
  decisions repeat exactly, set `retain_memory` explicitly, which stays a hard
  bound. Measured on MN5 in a 64 GB slot at 8 threads, a 2.49M-unknown 3D bend
  under `linear_solver="direct"` reused its factorization on steps 2-5 at
  28a2bf4, at 57-81 s per step against 160-170 s for a step that factorizes,
  with a peak memory of 51.9 GiB; 9f004a7 reused on none of its steps there
  (160 s per step).
- **See what each solve did.** `SweepInfo.iterations` (Krylov iterations of
  each excited port, per frequency) and `SweepInfo.preconditioner_reused` (per
  frequency) are recorded on driven results solved outside JAX
  transformations; they are `None` under them and on results saved by earlier
  versions. The logger `"gfem.solver"` (silent by default) records every driven
  and electrostatic solve step, under `jax.grad` too: frequency, direction,
  solver, what it did (`factorized`, `refactorized`, `reused`), and why a
  design factorized again. Resolving `linear_solver="auto"` logs its choice
  there.
- **A wave port's couplings are held once.** The port state kept between
  solves is half its former size; S, values and gradients are unchanged.
- **Lumped values are design inputs.** A `LumpedPort`'s `inductance` and
  `capacitance`, and an `Impedance`'s `resistance`, `inductance` and
  `capacitance`, now differentiate: driven S-parameters on uniform sweeps and on
  adaptive sweeps with frozen samples (plain and Bloch periodic models included),
  kept fields, and eigenfrequencies. Electrostatic results get exact zeros, and the
  port mode problem raises.
  - A model with an `Impedance`, or with a lumped port's L or C, gains scalar
    leaves in `jax.tree.leaves(model)` and in a model's cotangent. Models with
    neither keep their leaf count. Every `Impedance` adds leaves, resistance-only
    ones included.
  - An `Impedance` whose surface shares an edge with a wave port face raises
    `gfem.PortGeometryError` when its values are differentiated.
  - A lumped port's resistance stays fixed, because it is the port's reference
    impedance: a traced one raises `gfem.NonDifferentiableError`, and the message
    says which values are differentiable.
  - A lumped value built under `jax.grad`, `jax.jit` or `jax.vmap` is checked when
    the solve runs: a non-finite or non-positive one raises `ValueError`.
- **`gfem.power_flow` is differentiable** through its field: under `jax.grad`,
  `jax.jit` and `jax.vmap`, the power absorbed, radiated or transmitted through a
  surface is an objective, on lumped-port, wave-port and periodic models. A
  material field or sheet fill in the element layer just outside the side (move
  the surface one element away), the node coordinates, the field's `omega`, and a
  field of an adaptive sweep (use `sweep="uniform"`) raise
  `gfem.NonDifferentiableError` when differentiated; every other model input gets
  an exact zero from the measurement. A side given by a predicate needs fixed node
  coordinates: under a `jax.jit` over the model, name the side's regions instead.
- **The electrostatic energy is differentiable.** `result.energy` of an
  electrostatic solve differentiates with respect to `eps_r` (isotropic and
  principal values) and the node coordinates, on both linear solvers, for one
  adjoint solve for all terminals together. No electrostatic output raises under
  `jax.grad` any more.
- **Kept fields with wave ports under a transformation.** `Driven(keep_fields=...)`
  on a model with wave ports now runs under `jax.jit`, `jax.grad` and `jax.vmap`,
  and the kept fields are differentiable there. The wave ports' mode fields are
  `None` inside a transformation.
- Fixed: the gradient with respect to a `MaterialField` conductivity was exactly
  zero where the conductivity is zero everywhere (driven and eigenmode); it is now
  the one-sided derivative.
- Fixed: in an adaptive sweep with frozen samples, the gradient of an S entry
  whose port is not excited was off at the samples, by up to about 1e-2. Each
  sample now also solves one right-hand side per lumped port left out of
  `excite`. Adaptive results with `excite=` a strict subset of the lumped ports
  move by less than the sweep tolerance, and `excite=` no longer saves solves in an
  adaptive sweep.
- An objective reading the S row of a lumped port without a resistance gets a NaN
  gradient in an adaptive sweep, as in a uniform one, where it raised an NGSolve
  exception.
- **Behaviour change.** `MaterialField` values built under `jax.grad`, `jax.jit` or
  `jax.vmap` are checked when the solve runs. A non-finite or non-passive value
  raises the constructor's error with the region named, where it gave NaN, a
  solver convergence error or a finite result.
- `gfem.design.project` returns values in [0, 1] exactly; round-off could leave a
  few ulp outside, which `SheetField` refused. The gradient at densities 0 and 1
  is unchanged.
- The {doc}`../capabilities` page's gradient checks gain the permeability and the
  loss tangent (driven, isotropic and principal values; eigenmode) and the
  electrostatic region energies.
- **Fixed: periodic gradients of a `MaterialField` with `axes`.** Recompute them if
  your periodic model's fields touch a first-order `Absorbing` surface. There, the
  gradients with respect to `eps_r` and `mu_r` principal values were wrong: off by 0.5 %
  to 10 % in the cases measured, more for eigenmodes. They now agree with finite
  differences. Periodic models without such a surface got the right gradients and now
  get them faster. Non-periodic models were not affected.

Wave ports, S-parameters and measurements:

- **`Field.at` reads the tetrahedron that holds the point.** A point near a face, within
  1e-4 of an element's size, could get the neighbouring element's value, off by the
  field's full jump there, and the value could change with the rank count. It now gets
  the tetrahedron it lies in, on any number of ranks. Within 1e-9 of an element's size
  from a face, a point counts as on the face. A point on or just above an excluded
  conductor no longer raises a false "lies in volume region"; a point just inside one,
  which used to read the air's value, now raises. See {doc}`../recipes/measurements`.
- A wave port whose port line is lossy now warns with
  `gfem.PortModeProjectionWarning` when $\alpha/\beta$ exceeds $10^{-3}$, lone ports included: its
  reflection entries of S are off by about $\alpha/\beta$, and transmission is right to the
  square of it. A shared surface still warns once, for its cross-projection when
  that is above the floor. The warning is no longer hidden by an unsolved
  frequency of a partial sweep, and a mode exactly at cutoff no longer gives a
  spurious warning.
- `characteristic_impedance` works on any eager driven result with wave ports,
  with or without `keep_fields`. Inside a transformation it raises `ValueError`
  pointing to `gfem.PortMode`.
- A uniform sweep with wave ports runs with `frequency_workers > 1`. The main
  process solves the port modes once over the whole grid before the workers
  start. S and the reference impedances match the one-worker sweep, bit for bit on
  one thread and to rounding otherwise.
- `gfem.PortMode` ignores `SolverOptions.device`, `linear_solver` and
  `eigen_solver` instead of raising, so the options of a driven solve can be
  reused. Its degenerate-mode `PortModeError` now suggests a mirror half model (a
  symmetry plane through the guide's axis) before a `pec_edges` wall.
- `DrivenResult.with_offsets({port: d})` moves wave-port reference planes after
  the solve, from the stored propagation constants. Offsets are absolute, and the
  result equals a solve with `WavePort(offset=d)`.
- `gfem.electric_energy(field, model=model)` gives the time-averaged electric
  energy, in J, that a kept driven field or a mode field stores in each volume
  region. Dielectric loss in a region with one loss tangent absorbs
  `2 * omega * tan_delta * W_e`. It is forward only. The measurements recipe adds
  that the power of several ports fed at once is `power_flow` of the superposed
  kept fields.
- `NetworkParameters.to_touchstone` writes rows in increasing frequency, as the
  format requires, whatever the order of the result. `to_touchstone(path, z0=R)`
  writes a network in which no port has a reference impedance, such as wave ports
  without `voltage_path`: S is written unchanged with R as a nominal label, and a
  comment line says so. Without `z0` it raises with that remedy.
- **Behaviour change.** `Driven.frequency` and `PortMode.frequency` refuse a
  frequency given twice (`ValueError`). A repeat was solved in full twice, crashed
  `keep_fields=True` and wrote an invalid Touchstone file. Drop repeats with
  `numpy.unique(frequency)`. The order stays free, and results keep the order
  given.
- A `LumpedPort` direction accepts an axis without a sign (`"x"`, `"Z"`): it means
  the positive axis.
- `Material`'s message for a complex `eps_r` or `mu_r` names the remedy for each:
  `tan_delta` or `conductivity` for dielectric loss, a complex value through
  `gfem.MaterialField` for either.
- Frequencies in error messages, warnings and result reprs use Hz, kHz, MHz, GHz
  or THz, so a 193 THz cutoff no longer reads "193414 GHz"; text between 1 GHz and
  1 THz is unchanged. The low-frequency warnings of driven and eigenmode solves and
  the mode-following errors of `refine_mesh` use the same units, and a
  one-frequency band in port mode messages reads "60 GHz", not "60-60 GHz".
- `power_flow`'s wrong-quantity error now reads "needs an electric field".

Mesh refinement:

- **Adaptive mesh refinement.** `gfem.refine_mesh` solves, estimates the error in
  each tetrahedron, subdivides where most of it is and solves again, until the
  {term}`error estimate` meets a tolerance. It returns the refined model, its
  result and the history of the run. See {doc}`../recipes/mesh-refinement`.
- `gfem.refine_mesh` frees the solve entries a gradient has already read before
  its first solve, so the last optimizer step's factorization no longer stays in
  memory through the refinement cycles.
- With `QuantityEstimator(omega=...)`, `gfem.refine_mesh` raises
  `gfem.DegenerateModeError` when the followed mode is not clearly the nearest on a
  refined mesh (a second mode less than twice as far), or when the nearest mode
  moved further than the previous error estimate allows: ask for more modes or
  start from a finer mesh.
- `gfem.error_estimate` raises `gfem.DistributedUnsupportedError` (feature
  `error_estimate`) for a result solved across ranks, instead of a NumPy broadcast
  error or a wrong value. `refine_mesh` across ranks raises with feature
  `mesh_refinement`.
- **Refine for your objective.** `gfem.QuantityEstimator(objective=J)` estimates
  the error of `J(result, model)`, the function you differentiate, and
  `refine_mesh` refines for it with `tolerance` in J's units. J may read S,
  capacitance and electrostatic energy, frequencies, Q, bulk participations,
  and fields through `gfem.power_flow` (absorbed power) and the new
  `gfem.region_mean`. Anything else raises `gfem.MeshRefinementUnsupportedError`
  by name. See {doc}`../recipes/mesh-refinement`.
- **Mean of a field over a region.** `gfem.region_mean(field, region)` is the
  mean of a field or a potential over named volume regions, differentiable,
  the same quantity on every refined mesh. See {doc}`../recipes/measurements`.
- **Q, degenerate modes and several quantities.** `QuantityEstimator(q=i)`
  estimates a mode's Q. `omega=(i, j)` estimates the mean frequency of a
  degenerate cluster. `refine_mesh(estimator=(e1, e2), tolerance=(t1, t2))`
  refines until every quantity meets its tolerance. `ErrorEstimate.kind` says
  what was estimated, and `RefinementCycle.modes` which mode was followed.
- **An S entry's read port no longer needs to be excited**; its adjoint is
  solved with the forward.
- **Now estimated:** frequencies of modes in permittivity laws where $\mathrm{Re}\,\varepsilon > 0$,
  behind second-order absorbing boundaries, with complex `mu_r`, and with loss
  in anisotropic materials; S and the field with loss in anisotropic materials.
  Complex `mu_r` is now weighed exactly, so its S and field estimates are
  tighter.
- **Behaviour changes (now raise `gfem.MeshRefinementUnsupportedError`):**
  - S estimates and objectives on a driven model with an `Impedance` or
    `Absorbing` surface (they underestimated);
  - S estimates and objectives on a model with a lumped port spanning a
    permeability contrast;
  - S estimates and objectives on a permittivity law with $-20 < \mathrm{Re}\,\varepsilon < 0$ at
    a solve frequency;
  - an `omega` estimate of a mode bound to a region where $\mathrm{Re}\,\varepsilon < 0$.

  Refine those models with `gfem.FieldEstimator()`.
- `refine_mesh`'s `tolerance` default is now `None`: 1e-2, except for an
  objective, which needs its own.
- {doc}`../verification` lists each estimate kind's measured range.

Solvers, memory and ranks:

- **Wrong S from a PETSc stack built by the preview's script: rebuild it.** A stack built by
  the preview's [`tools/petsc_wheelhouse.sh`](https://github.com/benvial/gfem/blob/main/tools/petsc_wheelhouse.sh)
  can return wrong S, with no error, on a large direct solve with
  `linear_solver="petsc"`. A 10 M DOF mesh solved in one process stayed below the size
  where this happens. Rebuild the stack with the current script, as
  {doc}`../recipes/petsc-slepc` says. The cause was an integer overflow in an OpenBLAS
  routine that MUMPS calls; the script now builds MUMPS without it.
  - **Behaviour change.** gfem refuses a stack built without this fix, at any size, with
    `gfem.SolverUnavailableError`. `linear_solver="bddc_gmres"` above one rank still
    runs on such a stack, without using it for its coarse problem.
- **Behaviour change: default thread count.** With `threads=None`, NGSolve now runs on
  `NGS_NUM_THREADS`, else on the physical cores the process may run on, instead of every
  hardware thread of the machine. Without an affinity mask (macOS, Windows) the count is
  the logical CPUs. Your own `ngs.SetNumThreads` no longer carries into such a solve:
  set `SolverOptions.threads` instead. Inside your own `ngs.TaskManager()`, a solve
  runs on that TaskManager's threads, whatever `threads` says, on one process and
  under ranks; gfem no longer calls `SetNumThreads` there, so NGSolve's "can't
  change number of threads" line is gone. Under ranks, `threads=None` shares each
  physical core between the ranks whose affinity mask holds it, so partly
  overlapping masks no longer oversubscribe; identical and disjoint masks give
  the same counts as before.
- **Solves across MPI ranks.** `mpirun -n N python script.py` or
  `srun --mpi=pmix` now runs a uniform driven sweep, an electrostatic solve, an
  eigenmode solve and periodic models on one axis (Bloch driven included), and
  their gradients, across the ranks, with `comm=` on `gfem.solve`, `Job.run`,
  `Mesh.from_gmsh`, `Job.from_json` and `load_result`. An eigenmode solve there
  uses SLEPc; gradients through a mode's field, two or three periodic pairs,
  Bloch eigenmode, thermal and DC conduction problems and some options raise
  {py:obj}`gfem.DistributedUnsupportedError`. See {doc}`../recipes/distributed-solve`.
  - The `mpi` extra is mpi4py; `import gfem` needs neither it nor MPI. Only Open MPI
    is accepted above one rank.
  - [`tools/petsc_wheelhouse.sh`](https://github.com/benvial/gfem/blob/main/tools/petsc_wheelhouse.sh)
    gains `--mpi`, `--sources` and `--fetch-sources`, for the {term}`MPI PETSc stack`,
    which sits beside the unchanged no-MPI stack.
  - **Behaviour change.** A launched solve used to run N complete one-process solves
    with no warning. It now runs distributed or raises, and a launched install
    without mpi4py raises `gfem.SolverUnavailableError`.
    `comm=MPI.COMM_SELF` asks for the copies by name.
- **`gfem.release_memory()`** frees everything gfem keeps between solves in a
  process: solve entries, pooled solver plans and factorizations, and cached
  discretizations. Results are unchanged; a released gradient is recomputed.
- A band sweep no longer grows memory with every wave vector: a discretization
  keeps the spaces of at most 8 Bloch phase sets.
- Freeing a direct solver handle from another thread during a solve now always
  gives "This solver handle was freed." or waits for the solve, never a native
  exception or an answer from a freed handle. `refactorize` on a direct, PETSc or
  cuDSS handle with another free-DOF set raises `ValueError` instead of solving on
  the old free DOFs. The `SolverOptions.threads` docs say that a process runs one
  solve at a time.
- **Behaviour change.** With `linear_solver="petsc"`, `ksp_type preonly` (the
  default) is refused with a `pc_type` other than `lu` or `cholesky`, and with a
  `pc_factor_shift_type` other than `none`: these returned a wrong S with no error.
  Use a Krylov `ksp_type` such as `gmres` with any other preconditioner. A PETSc
  Krylov solve that ends on its iteration count without measuring a residual
  (`ksp_convergence_test skip`) raises `ValueError`. `petsc_options["ksp_atol"]` is
  reserved, like `ksp_rtol`: gfem stops on the relative residual set from
  `tolerance`. The `SolverOptions.tolerance` docstring no longer says S is accurate
  to the tolerance.
- Above one rank, `linear_solver="petsc"` with `pc_type` `ilu` or `icc`, or `lu`
  or `cholesky` on a package other than MUMPS, raises
  `gfem.DistributedUnsupportedError` (feature `petsc_preconditioner`) before any
  mesh is read. The `power_flow` and `electric_energy` refusals across ranks name
  the route that works: `Job.run(out=)`, then `load_result` on one process.
- `linear_solver="auto"` (the default) runs the direct solver on a GPU:
  `SolverOptions(device="cuda")` and `python -m gfem run --device cuda` on a
  default job are accepted. `"bddc_gmres"` and `"petsc"` on a GPU stay refused,
  with a message naming both accepted values. An `"auto"` job written for a GPU and
  run with `--device cpu` runs the problem type's CPU default.
- `SolverOptions.order` above 12 raises `ValueError`; higher orders failed inside
  NGSolve or hung.
- New warning classes `gfem.SweepRankDeficientWarning` (an adaptive sweep near the
  accuracy of its full solves) and `gfem.UnsampledPixelsWarning` (`LayerGrid`
  pixels that no centroid samples). Both subclass `UserWarning`; their messages
  are unchanged.

Jobs and files:

- A uniform sweep that stops at an unsolved frequency under `Job.run(out)` or
  `python -m gfem run` now leaves a {term}`partial bundle`, `partial.json` and
  `partial.npz`, never `result.json`. The error carries a note saying where it is;
  the command line prints it as a second stderr line `gfem: note: ...` and still
  exits with 1. Read it with `gfem.load_result("DIR/partial.json")`: its `run`
  entry lists `unsolved` and `error`. A complete run removes a stale partial
  bundle.
- **Behaviour change.** An unknown key in a job or result file is refused by name,
  with a `ValueError` giving the file, the class, the keys and both gfem versions.
  `Job` and `Mesh` dropped such keys without a word, and other classes raised a
  bare `TypeError`.

Documentation:

- New user pages: {doc}`../recipes/solve-a-device`, {doc}`../recipes/prepare-a-mesh`,
  {doc}`../recipes/save-and-export` and {doc}`../conventions`. Every number they
  print comes from a solve and is checked against a hand formula or a file read
  back. The glossary gains S-parameters, power waves, lumped port, wave port, loss
  tangent, element order and adjoint solve.
- Every user page's code is checked against the API without solving: names
  resolve and call keywords bind.
- The {doc}`../recipes/gsim` recipe reads gsim's one-face lumped ports
  (microstrip), which raised `KeyError: 'Elements'`.
- The periodic-cell recipe and `gfem.Absorbing` state the reflection of an
  absorbing boundary under oblique incidence in a Bloch cell, $(1 - \cos\theta)/(1 + \cos\theta)$
  at order 1, and that order 2 lowers it for one polarization only. Bloch S is now
  checked against its reference to 5e-6, and periodic pairing and Bloch bands are
  tested on an unstructured gmsh mesh.
- The PETSc recipe states the stack's build time (about 22 minutes, 26 with
  `--mpi`) and what a Krylov method on the MUMPS factorization does below the
  low-frequency floor.
- The get-started page links the 3D bend tutorial, the recipes page lists mesh
  refinement, and the package description matches the README.

Anisotropic materials and shared port surfaces:

- **Anisotropic materials.** Driven, eigenmode, electrostatic and port mode
  solves, wave ports, first-order absorbing boundaries, `MaterialField` and
  its gradients, `EigenmodeResult.principal_participation`,
  `interface_participation` and job files now take an anisotropic
  `Material`. The cases listed below as unsupported raise. See
  {doc}`../recipes/anisotropic-materials`.
  - `eps_r`, `mu_r`, `tan_delta` and now `conductivity` each take one value,
    three principal values along `axes`, or a real symmetric 3×3 matrix in
    the global frame, stored as principal values plus `axes`. Driven and
    eigenmode solves read a conductivity given as principal values along its
    principal axes.
  - Wave ports accept anisotropic `mu_r`, and anisotropic materials in a
    {term}`transverse-anisotropic` cross-section: every tensor has a
    principal axis along the port normal, at any orientation in the port
    plane.
  - First-order absorbing boundaries use the material's exact admittance
    (see the behaviour change below).
  - A `MaterialField` with `axes` holds three principal values along fixed axes:
    shape `(n_elements, 3)` per element, or `(3,)` for the whole region.
    `MaterialField.uniform` takes an anisotropic `Material`.
    `MaterialField.interpolate` blends two materials per principal value when
    their tensors share principal axes (air and an anisotropic material, for
    example); materials with different axes raise. Driven, eigenmode,
    electrostatic and port mode solves read the field's tensor on each element.
    `jax.grad` gives one derivative per principal value, checked against finite
    differences. The axes are fixed: the orientation is not a design input.
  - `EigenmodeResult.principal_participation` splits each anisotropic
    region's participation along its principal axes. It gives the exact
    per-region loss for `tan_delta` and `conductivity` given as principal
    values.
  - Errors: passivity errors name the failing principal value and its axis.
    The wave-port error names the property (`eps_r`, `mu_r`, `conductivity`)
    with no principal axis along the port normal, and gives the workarounds.
    Non-symmetric (gyrotropic) tensors and tensors with different principal
    axes raise `ValueError` at `Material`.

  Job files and saved results from the preview read unchanged.
- **Behaviour change: equal principal values are one value.**
  `Material(eps_r=(2, 2, 2))` is now stored as `eps_r=2.0`: it equals
  `Material(eps_r=2)`, `isotropic` is `True`, and its repr changes. Results are
  unchanged. Old job files holding such a tuple read back as the scalar.
- **Behaviour change: absorbing boundaries next to anisotropic materials.**
  - S, fields and Q change for a model whose first-order `Absorbing` surface
    touches an anisotropic material. Each such face now uses that material's
    normal-incidence admittance. The mean of the tensor diagonals, used before,
    reflected up to about 13 % at normal incidence next to a uniaxial dielectric.
    Faces next to isotropic materials are bit for bit unchanged.
  - The port mode problem's absorbing edges change the same way: each edge
    next to an anisotropic material takes its exact admittance for the two
    polarizations.
  - A second-order surface touching an anisotropic material now raises
    `NotImplementedError`.
  - Node-coordinate gradients of a model whose absorbing face has an
    anisotropic material behind it now raise `NonDifferentiableError`; the preview
    differentiated the approximate coefficient.
- **Behaviour change: port modes of rotated transverse tensors.** A port
  cross-section holding a tensor rotated about the port normal no longer drops
  its propagating modes (or raises `PortModeError`). The effective-index bound
  now uses the tensor's largest principal value, not its largest diagonal
  entry. Models without such a tensor are unchanged.
- **Behaviour change: `interface_participation`'s vacuum test.** A side is
  vacuum only when no wave in it is slower than in vacuum (largest principal
  `eps_r` × largest `mu_r` equal to 1). A side such as `eps_r = (1, 1, 4)` was
  taken as vacuum, and `"MA"`/`"SA"` read its normal field with $\varepsilon_0$ (a 4×
  error); it no longer is. Isotropic sides are unchanged.
- **Several modes on one wave port surface.** Several
  {py:obj}`gfem.WavePort`s on one surface, one `mode` each, form a
  {term}`shared port surface`. Each mode is its own port, with its own row
  and column of S, so S holds the coupling between modes (for example TE0 to
  TM0 conversion in a silicon strip). Gradients and adaptive sweeps work as
  for one mode per port. {py:obj}`gfem.PortMode` on such a port gives the mode
  the driven solve uses. See {doc}`../recipes/shared-port-surface`.
- **New warnings.** {py:obj}`gfem.SkippedPortModeWarning`: a surface leaves
  out a port mode ranked below a requested one (a lone `mode=2`, or modes 1
  and 3), which the port then partly reflects. A lone `mode=1` never warns.
  `gfem.PortModeProjectionWarning`: on a shared port surface with
  loss in the cross-section, one port's S reads part of another port's mode
  above 1e-3.
- **New errors at `gfem.Model`.** Two wave ports on one surface that request
  the same `mode`, or give different `pec_edges`, raise `ValueError`. A lumped
  port or a boundary condition beside a wave port on one surface still does.
- **Behaviour change for a lone wave port.** A port mode whose reaction or
  Gram matrix on the port face is singular now raises
  {py:obj}`gfem.PortModeError` before the 3D solve, where the preview went on
  without warning. Lone-port S is otherwise unchanged.
- **Behaviour change: lossy modes below cutoff.** A requested port mode whose
  attenuation is at least its phase constant ($\mathrm{Re}\,\beta \le |\mathrm{Im}\,\beta|$, for example an
  RC-like line at low frequency) now counts as below cutoff, where the preview
  accepted it as a propagating mode. The solve raises the port mode sweep's
  cutoff {py:obj}`gfem.PortModeError`, naming the port (and its surface when
  shared), lone wave ports included.
- **`gfem.Eigenmode`'s message** for a wave port on a shared port surface
  names the surface and its wave ports and points to a `gfem.Driven` sweep.

Periodic cells: `gfem.Periodic` pairs two surfaces of a model related by a
translation (one to three pairs). Eigenmode and uniform driven problems take a
Bloch wave vector, `wave_vector=(kx, ky, kz)` in rad/m, with
$\mathbf{E}(\mathbf{r} + \mathbf{a}) = e^{-i \mathbf{k} \cdot \mathbf{a}} \mathbf{E}(\mathbf{r})$; without one the fields are plain periodic. See
{doc}`../recipes/periodic-cell`.

- **Gradients.** Permittivity, conductivity, sheet fill and node coordinates
  differentiate on periodic models, plain and Bloch, and are checked against
  finite differences (four new rows on {doc}`../capabilities`). A boundary
  morph moves the two nodes of a pair together. `wave_vector` is not
  differentiable.
- **Solvers under a Bloch wave vector.** Eigenmode problems need the direct
  solver. PETSc, cuDSS and the adaptive sweep refuse a wave vector. The direct
  solver factors the problem with MKL PARDISO (with the `mkl` extra) or
  UMFPACK, and BDDC factors its coarse problem likewise. Models without a
  periodic pair are unchanged.
- **Results.** Fields of a periodic model hold values on both surfaces of
  every pair. `gfem.power_flow` works on sides that cross a pair.
  `gfem.interface_participation` refuses a paired surface.
- **Job files** carry `Periodic` and `wave_vector`; older job files read
  unchanged.

## Preview (2026-11-01)

Tagged 0.1.0 in the repository until the PyPI release; now commit
[`18235aa`](https://github.com/benvial/gfem/commit/18235aa). Driven problems (uniform and adaptive sweeps), eigenmode problems
(including nonlinear ones), electrostatic and port mode problems, with lumped
and wave ports. Gradients with respect to permittivity, conductivity, sheet
fill and mesh node coordinates are checked against finite differences.

Changes since the release candidate of 2026-10-21:

The direct and `bddc_gmres` solvers now keep getting faster past 16 CPU
threads on large problems. Results change only to the solver's precision.

- **`bddc_gmres` runs its GMRES on all its threads.** A driven or eigenmode solve
  with `bddc_gmres` now uses `SolverOptions.threads` threads for the step where
  NGSolve's GMRES used one. Iteration counts and S are unchanged, to rounding.
  While such a solve runs, other SciPy BLAS calls in the same process run on one
  thread; SciPy's thread count is restored when the solve ends.
- **With the `mkl` extra, BDDC's setup is faster.** BDDC factors its coarse
  problem with MKL PARDISO instead of NGSolve's sparse Cholesky. Without the
  extra, nothing changes. Results with and without the extra agree to the
  `tolerance` of the solve, not bit for bit.
- **On 16 or more threads, a BDDC setup that serves many solves uses sparse
  Cholesky.** Once one setup serves enough solves, sparse Cholesky is faster
  overall than MKL PARDISO there, and uses less memory. This applies when one
  BDDC setup expects 16 or more right-hand sides on 16 or more threads, or 6 or
  more on 56 or more threads. A uniform driven sweep counts its excited ports
  times its frequencies; a single-frequency solve or an {term}`adaptive sweep`
  counts its excited ports only, and an electrostatic solve its terminals. A
  {term}`frequency worker` counts its own share of the threads and its own
  frequencies. Below 16 threads, or without the `mkl` extra, nothing changes.
  The choice depends on these counts only, never on timings, so a run repeats
  exactly.
- **MKL PARDISO frees its cores as soon as each call returns.** Every PARDISO
  call gfem makes runs with Intel OpenMP's block time at 0, so the rest of the
  solve gets the cores back. The setting is restored after each call, so a
  `KMP_BLOCKTIME` you set applies everywhere else.
- **A reused PARDISO factorization keeps its thread count.** With the `mkl`
  extra, a factorization reused under a different thread count, or outside the
  `TaskManager` it was made in, used to solve on one thread: about 5 times
  slower on a 10-million-unknown problem. A gradient that reused a cached
  adjoint factorization under `threads=None` could hit this. It now solves on
  the threads it was factored with. A factorization made outside any
  `TaskManager`, such as `gfem.design.helmholtz_filter`'s, likewise keeps
  solving on one thread. Results do not change.
- **Single-frequency direct solves start factorizing sooner.**
  `linear_solver="direct"` no longer copies the assembled matrix when every
  unknown is free. It builds the index it needs to refactorize with new values
  only when a second frequency needs it.
