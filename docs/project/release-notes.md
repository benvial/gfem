# Release notes

## Unreleased

Get started:

- **Physics pages.** The front page opens on a card per field of physics, each
  leading to a page of its tutorials and recipes ({doc}`../index`).

Integrated photonics:

- **Bent port modes.** `gfem.PortMode(..., bend=gfem.Bend(radius, outward=..., at=...))` gives a guide's loss per 90° and `gfem.junction_coupling` its coupling to a straight guide ({doc}`../recipes/waveguide-bends`).
- **Behaviour change: a `PortMode` solve costs more.** It takes about 15 to 30 %
  longer and its result holds three more arrays, so gfem 0.2.0 cannot load a saved
  one. Upgrade before you load it ({doc}`../recipes/waveguide-dispersion`).
- **Behaviour change: `gfem.PortModeProjectionWarning` is removed.** It has not been
  emitted since 0.1.0; delete any warning filter that names it.
- **Behaviour change: a bad voltage path raises.** A `WavePort` whose
  `voltage_path` has a point outside the port section now raises `ValueError` in
  the solve. Move the point onto the port.
- **Designs at wave ports.** S, port waves and kept fields differentiate with
  respect to the materials, sheets and nodes of a wave port's face, on every
  driven sweep and solver ({doc}`../recipes/port-mode-gradient`).
- **Port modes differentiate.** `gamma`, `n_eff`, `Z_PV` and the field of a
  `gfem.PortMode` result take gradients; `residual` and the dispersion terms
  raise `gfem.NonDifferentiableError` ({doc}`../recipes/port-mode-gradient`).
- **Sliding ports.** `gfem.shape.boundary_morph(..., sliding_ports=...)` lets the
  named wave ports' faces slide in their plane, so a shape design reaches the
  port ({doc}`../recipes/port-mode-gradient`).
- **Polarity reference.** `WavePort(polarity=(point, direction))` fixes the sign of
  a port's row and column of S, and port-mode field gradients follow it
  ({doc}`../recipes/sax-circuit`).
- **Group index and dispersion.** Every `gfem.PortMode` solve gives `n_g`,
  `beta_2`, `beta_3` and `D` at each frequency ({doc}`../recipes/waveguide-dispersion`).
- **SAX components.** `gfem.SaxComponent` turns a `gfem.Model` function and a
  `gfem.Driven` band into a SAX circuit model, so one `jax.grad` reaches gfem and
  SAX together ({doc}`../recipes/sax-circuit`).
- **S for SAX and scikit-rf.** `NetworkParameters.to_sdict` and `to_skrf` give a
  solved S as a SAX dictionary and a scikit-rf network; the extras are `gfem[sax]`
  and `gfem[skrf]` ({doc}`../recipes/sax-circuit`).

Nonlinear and quantum optics:

- **Kerr nonlinearity of a port mode.** `gfem.kerr_nonlinearity` and
  `gfem.kerr_coupling` give a port mode's nonlinear coefficient and the coupling
  tensor between its waves ({doc}`../recipes/waveguide-nonlinearity`).
- **Raman gain, crystal Kerr and free carriers.** `gfem.raman_gain` and
  `gfem.free_carriers` give a port mode's Raman gain and free-carrier figures;
  materials take a `raman_spectrum` and a `nonlinear_anisotropy`
  ({doc}`../recipes/raman-gain`).
- **Second-harmonic generation.** `gfem.second_harmonic` and
  `gfem.three_wave_coupling` give a pump mode's coupling to its harmonic, the phase
  mismatch and the poled-guide efficiency ({doc}`../recipes/second-harmonic-generation`).
- **Photon-pair sources.** `gfem.spdc` and `gfem.sfwm` give the pair rate, the
  brightness and the joint spectrum of a pumped waveguide
  ({doc}`../recipes/photon-pair-sources`).

Electro-optics:

- **The electro-optic shift.** `gfem.electro_optic_shift` gives the effective-index
  and phase change per volt on a terminal, through the crystal's electro-optic
  tensor, differentiable in the tensor, the drive and the electrostatic mesh
  ({doc}`../recipes/electro-optic-phase`).
- **Electro-optic tensors.** `Material(electro_optic_tensor=...)` and
  `MaterialField.electro_optic_tensor` take the contracted tensor in the frame of
  the material's `axes` ({doc}`../recipes/electro-optic-phase`).
- **Capacitance per length.** `ElectrostaticResult.capacitance_per_length` gives
  the Maxwell capacitance matrix of a cross-section, and the result's repr now
  prints it ({doc}`../recipes/electro-optic-phase`).
- **Modulator tutorial.** A push-pull thin-film lithium niobate modulator
  computes its voltage-length product and co-designs the electrode gap and the ridge width
  ({doc}`../tutorials/modulator`).

Heat and thermo-optics:

- **Behaviour change: `joule_heating` of a driven result raises `TypeError`.**
  `gfem.joule_heating(driven_result, ...)` raised `NotImplementedError` and now
  raises `TypeError` naming `gfem.electromagnetic_heating`. Catch `TypeError`, or
  call that function ({doc}`../recipes/electromagnetic-heating`).
- **Electromagnetic heating.** `gfem.electromagnetic_heating` turns the kept fields
  of a driven result into a `gfem.HeatSource` with no new solve
  ({doc}`../recipes/electromagnetic-heating`).
- **Dissipated power.** `gfem.dissipated_power` gives a driven field's loss per
  region and per sheet ({doc}`../recipes/measurements`).
- **Heat flow.** `gfem.heat_flow` and `gfem.heat_flow_per_length` give the heat
  leaving named surfaces or regions ({doc}`../recipes/heater-temperature`).
- **Heat out.** `gfem.Trace.heat_out` records the heat that has left a side during
  a transient run ({doc}`../recipes/thermal-step-response`).
- **Sheet heat.** A heat source may carry heat on faces; the face's thermal
  boundary decides where it goes ({doc}`../recipes/electromagnetic-heating`).
- **Thermal boundary values differentiate.** A fixed temperature, a convection
  and a heat flux may be traced by `jax.grad` in steady and transient solves
  ({doc}`../recipes/thermal-gradient`).
- **Gradients of heat.** The dissipated power, the temperature, the heat flow and
  the heat out differentiate in the lossy materials, sheets and powers
  ({doc}`../recipes/thermal-gradient`).
- **Heat of a law blend.** `gfem.dissipated_power` and
  `gfem.electromagnetic_heating` accept a metal-dielectric blend with a
  permittivity law ({doc}`../recipes/electromagnetic-heating`).
- **Plans and factors reused across designs.** Thermal and DC conduction designs on
  `linear_solver="direct"` refactorize one plan instead of rebuilding it;
  `retain_memory=0` turns it off ({doc}`../recipes/thermal-gradient`).
- **Heat outside the thermal domain.** The error for such a source now names the
  region and its watts ({doc}`../recipes/electromagnetic-heating`).

Mechanics and acoustics:

- **Behaviour change: anisotropic materials on a 2D model.** A 2D model no longer
  raises an anisotropic material when it is built, and a 2D solve refuses it at the
  solve. Catch the error there ({doc}`../recipes/piezoelectric-admittance`).
- **Behaviour change: a region excluded from every physics.** The check now counts
  the `"mechanical"` key, so a model that excludes every volume from the other
  physics raises when solved, not when built ({doc}`../recipes/piezoelectric-admittance`).
- **Piezoelectric problems.** `gfem.Piezoelectric` solves a 2D section's admittance
  per length between electrodes over a list of frequencies
  ({doc}`../recipes/piezoelectric-admittance`).
- **Solid materials.** `Material` takes `stiffness`, `piezoelectric_tensor` and
  `mechanical_quality_factor`, and `excluded={"mechanical": [...]}` leaves regions
  out ({doc}`../recipes/piezoelectric-admittance`).
- **Network parameters and strain.** `PiezoelectricResult.network_parameters` gives
  S with Touchstone output, and `gfem.strain` reads the strain of a kept
  displacement ({doc}`../recipes/piezoelectric-admittance`).
- **Piezoelectric gradients.** The admittance and the clamped capacitance
  differentiate in the material constants and through a node map
  ({doc}`../recipes/piezoelectric-gradient`).
- **Node maps.** `Mesh.with_nodes` returns a mesh with its nodes at new
  coordinates, to differentiate through ({doc}`../recipes/piezoelectric-gradient`).
- **Mechanical eigenmodes.** `gfem.MechanicalEigenmode` finds the elastic modes of a
  2D section near a target, differentiable in the material constants
  ({doc}`../recipes/mechanical-modes`).
- **Acousto-optic shift.** `gfem.acousto_optic_shift` gives a wave port mode's index
  change per volt at each kept frequency, differentiable
  ({doc}`../recipes/acousto-optic-shift`).
- **Optomechanical coupling.** `gfem.optomechanical_coupling` gives the vacuum
  coupling of mechanical modes to a wave port's mode, and
  `gfem.optomechanical_overlap` the index change of any displacement, both
  differentiable ({doc}`../recipes/optomechanical-coupling`).
- **Brillouin gain.** `gfem.brillouin_gain` gives the forward and backward gain of
  a guide's mechanical modes, with its parts ({doc}`../recipes/brillouin-gain`).

Free-space scattering:

- **Incident fields.** `gfem.PlaneWave`, `gfem.GaussianBeam` and
  `gfem.IncidentField` light a model over a layered `gfem.Background`
  ({doc}`../recipes/incident-fields`).
- **Cross sections.** `gfem.cross_sections` gives extinction, absorption and
  scattering from one power balance ({doc}`../recipes/cross-sections`).
- **Point dipoles.** `gfem.ElectricDipole` and `gfem.MagneticDipole` give the
  emitted power, the Purcell factor and the port waves ({doc}`../recipes/point-dipoles`).
- **Port waves of many dipoles.** `Driven(source_fields=False)` reads the port
  waves of every dipole from one solve per port ({doc}`../recipes/point-dipoles`).
- **Many dipoles on an adaptive sweep.** The sweep needs far less memory, with the
  same results to rounding.
- **Far fields and beam coupling.** `gfem.far_field` and `gfem.beam_coupling` give
  the antenna quantities and the coupling into a fiber's beam
  ({doc}`../recipes/far-field`).
- **Multipoles.** `gfem.multipoles` decomposes a scatterer's radiated field into
  electric and magnetic multipoles, and `gfem.current_moments` gives the dipole and
  toroidal moments ({doc}`../recipes/multipoles`).
- **Gradients of free-space outputs.** Every new output differentiates in the
  material fields, and several in the node coordinates ({doc}`../recipes/cross-sections`).
- **Sources across ranks and workers.** Sources run on frequency workers and across
  MPI ranks ({doc}`../recipes/incident-fields`).
- **Optical force and torque.** `gfem.optical_force` gives the force and torque of
  a solved 3D field on a body, differentiable ({doc}`../recipes/optical-force`).

Periodic surfaces:

- **Behaviour change: diffraction-order arrays are NumPy.** The fields of
  `DiffractionOrderResult` are NumPy arrays, not JAX arrays. Call the NumPy
  equivalent of any JAX method you used on them ({doc}`../recipes/floquet-ports`).
- **Behaviour change: the unrequested-order warning is per polarization.**
  `UnrequestedOrderWarning` now fires for each polarization without a Floquet port.
  Declare both polarizations, or filter the warning ({doc}`../recipes/floquet-ports`).
- **Cell outputs.** `gfem.diffraction.efficiencies`, `power_balance` and `jones`
  give the orders' efficiencies, the power split and the Jones matrix
  ({doc}`../recipes/diffraction-efficiency`).
- **`floquet_ports`.** `gfem.diffraction.floquet_ports` declares every propagating
  order of a face ({doc}`../recipes/floquet-ports`).
- **Angle sweeps.** `gfem.diffraction.angle_sweep` solves a cell over polar and
  azimuth angles and resumes from a cache ({doc}`../recipes/diffraction-efficiency`).
- **Emissivity.** `gfem.diffraction.emissivity` and `hemispherical_emissivity` read
  the directional and hemispherical emissivity of a cell ({doc}`../recipes/emissivity`).
- **Cell responses.** `gfem.diffraction.cell_response` solves a cell over one
  parameter and interpolates it, differentiable ({doc}`../recipes/metasurface-from-cells`).
- **Aperture fields and propagation.** `gfem.diffraction.aperture_field` and
  `propagate` turn a parameter map into a focal spot or deflected beam
  ({doc}`../recipes/metasurface-from-cells`).
- **Metalens tutorial.** Silicon nitride ridges that focus green and red light on
  one line ({doc}`../tutorials/metalens`).
- **Effective medium of a slab.** `gfem.diffraction.effective_medium` retrieves a
  slab's permittivity and permeability, differentiable ({doc}`../recipes/effective-medium`).
- **Band diagrams.** `gfem.wave_vector_path` and `gfem.band_diagram` give the bands
  of a periodic cell along a path, differentiable ({doc}`../tutorials/photonic-crystal`).
- **Band group velocity.** `EigenmodeResult.v_g` gives a mode's group velocity under
  a Bloch wave vector ({doc}`../recipes/periodic-cell`).
- **Faster Bloch solves.** A Bloch eigenmode solve after a mesh's second wave
  vector no longer assembles its operators again.

RF and microwave:

- **Time responses.** `NetworkParameters.step_response` and `impulse_response` turn
  S into a `gfem.TimeResponse`, differentiable ({doc}`../recipes/time-responses`).
- **Rational models.** `NetworkParameters.fit_rational` fits a passive
  `gfem.RationalModel`, and `to_spice` and `to_verilog_a` write it as a netlist
  ({doc}`../recipes/rational-model`).

Resonators and qubits:

- **Behaviour change: eigenmode results with a far target.** On a model whose loss
  varies in space, `gfem.Eigenmode` could drop the mode nearest a far target
  without a word. Rerun such results from earlier versions.
- **Low-Q resonances are found.** With wave ports, an eigenmode solve lists every
  mode of a strip around `target`, whatever `n_modes`
  ({doc}`../modes-and-ports`).
- **Unrequested port modes.** `EigenmodeResult.requested_power_share` and
  `gfem.UnrequestedPortModeWarning` flag a mode that an unrequested port family
  carries ({doc}`../modes-and-ports`).
- **Refinement leaving a port's reach.** Such a mode is dropped from the search
  instead of ending the solve.

Meshes, materials and boundaries:

- **Behaviour change: periodic cross-sections pair by coordinates.** A cross-section
  meshed periodically in gmsh pairs under `gfem.Periodic` on any node numbering, and
  its values move by up to 4e-4 relative. Rerun your cross-section results
  ({doc}`../recipes/2d-models`).
- **Library materials.** `gfem.materials` builds eight materials from cited sources
  ({doc}`../recipes/dispersive-materials`).
- **Fit a measured table.** `gfem.fit_permittivity` fits a passive law to a table of
  refractive index and extinction and returns a `gfem.PermittivityFit` ({doc}`../recipes/fit-measured-data`).
- **Fitted range.** A law holds the frequencies it was fitted over, and a solve that
  reads it outside them warns with `gfem.FittedRangeWarning`. A law without a range
  writes the same job file as before ({doc}`../recipes/fit-measured-data`).

Design workflow:

- **Tolerance report.** `gfem.tolerance_report` gives the spread, the worst case and
  the linearized yield of named quantities under named tolerances
  ({doc}`../recipes/tolerance-report`).
- **Minimum width and spacing.** `gfem.design.length_scale_constraints` gives two
  differentiable constraints for an optimizer ({doc}`../recipes/robust-optimization`).
- **A fabrication model.** `gfem.design.FabricationModel` predicts the printed
  design of a drawn one, differentiable in the design and the bias
  ({doc}`../recipes/robust-optimization`).
- **Layer patterns.** `LayerGrid.pattern` draws a layer-grid design as polygons;
  contourpy is now a dependency ({doc}`../recipes/fabricable-layout`).
- **GDS export.** `gfem.design.write_gds` writes layer patterns to a GDSII file
  ({doc}`../recipes/fabricable-layout`).
- **Pattern checks.** `LayerPattern.check` measures a pattern against a minimum
  width and spacing ({doc}`../recipes/fabricable-layout`).

Solvers and scale-out:

- **Behaviour change: `"auto"` may pick `pmg_gmres`.** A driven problem whose direct
  factorization does not fit now takes `"pmg_gmres"` before `"bddc_gmres"`. Models
  that ran on the direct solver still do ({doc}`../recipes/large-driven-model`).
- **Fewer stale iterations in a sweep.** A reused BDDC or p-multigrid preconditioner
  is set up again sooner as the frequency moves ({doc}`../recipes/large-driven-model`).
- **A p-multigrid solver.** `linear_solver="pmg_gmres"` solves large driven models
  in a quarter of the direct solver's memory ({doc}`../recipes/large-driven-model`).

Results and files:

- **Dissipated power and electric energy.** `gfem.electric_energy` may be traced by
  `jax.grad`, `jax.jit` and `jax.vmap` ({doc}`../recipes/measurements`).
- **A refused value names both versions.** A job or result written by another
  gfem version now says which versions are involved.

## 0.2.0 (2026-11-23)

- **Behaviour change: kept coarse factorizations count in `retain_memory`.** Under
  ranks, `retain_memory=0` keeps none past a solve and `gfem.release_memory()`
  frees them ({doc}`../recipes/distributed-solve`).
- **Native 2D models.** `gfem.Driven(polarization="Ez")` or `"Hz"` solves a 2D model
  with line ports and adjoint gradients ({doc}`../recipes/2d-models`).
- **Perfectly matched layers.** `gfem.PML` absorbs oblique waves in driven,
  eigenmode and port mode solves, with gradients
  ({doc}`../recipes/perfectly-matched-layer`).
- **JAX 0.8.3 or later, NumPy 2.2 or later.** gfem installs next to packages that
  cap JAX below 0.10 ({doc}`../get-started`).
- **BDDC's coarse problem across ranks.** With `bddc_gmres` on the MPI PETSc stack,
  gfem factors BDDC's coarse problem on a share of the ranks, the same share on
  every run; S is unchanged to round-off ({doc}`../recipes/distributed-solve`).
- **Capped PT-Scotch ordering.** Rebuilt MPI stacks order the coarse problem on at
  most 32 processes; rebuild an older stack to get it ({doc}`../recipes/petsc-slepc`).
- **ParMETIS on your own stack.** `tools/petsc_wheelhouse.sh --mpi --with-parmetis`
  adds ParMETIS for its builder's use alone ({doc}`../recipes/petsc-slepc`).
- **Modes and ports page.** It tells a port mode from a resonant mode
  ({doc}`../modes-and-ports`).
- **Benchmarks page.** The November 2026 comparison on MareNostrum 5
  ({doc}`benchmarks`).

## 0.1.0 (2026-11-20)

First release on PyPI (`pip install gfem`). The 2026-11-01 preview was tagged
0.1.0 in the repository and never published; the tag now names this release.

Get started:

- **Phase timings.** The `"gfem.timing"` logger records each phase of every driven
  and eigenmode solve, silent by default ({doc}`../recipes/solve-a-device`).

Integrated photonics:

- **Behaviour change: exact S on lossy wave ports.** S of lossy wave ports now uses
  the biorthogonal projection and `z_ref` is complex. Recompute earlier S, and give
  every wave port a `voltage_path` or none ({doc}`../modes-and-ports`).
- **Behaviour change: complex characteristic impedances.** `PortModeResult.Z_PV` is
  now complex, where it was a modulus, and `gfem.characteristic_impedance` takes a
  `current_path`. Take the modulus if you need the old value ({doc}`../modes-and-ports`).
- **Behaviour change: wave-port checks at the solve.** A singular port mode, or a
  lossy mode whose attenuation reaches its phase constant, now raises
  `gfem.PortModeError`. Request a propagating mode ({doc}`../recipes/shared-port-surface`).
- **Behaviour change: skipped port modes warn.** A surface that leaves out a port
  mode ranked below a requested one warns with `gfem.SkippedPortModeWarning`.
  Request the missing mode ({doc}`../recipes/shared-port-surface`).
- **Several modes on one surface.** Wave ports on one surface, one `mode` each,
  form a shared port surface with S holding the coupling between modes
  ({doc}`../recipes/shared-port-surface`).
- **Wave ports with workers.** A uniform sweep with wave ports runs with
  `frequency_workers > 1`.
- **Offsets after the solve.** `DrivenResult.with_offsets` moves wave-port reference
  planes after the solve.
- **Port mode options.** `gfem.PortMode` ignores `SolverOptions.device`,
  `linear_solver` and `eigen_solver` so a driven solve's options can be reused.
- **Metal bend tutorial.** Metal-dielectric topology optimization with
  `MaterialField.interpolate(..., scheme="index")` ({doc}`../tutorials/metal-bend`).

Heat and thermo-optics:

- **Behaviour change: `density` is now `mass_density`.** `density=` still sets it with a `DeprecationWarning`, reading `.density` raises `AttributeError`, and Job files that store `density` read unchanged. Rename it in your code ({doc}`../recipes/thermal-step-response`).
- **Thermal problems.** `gfem.SteadyThermal`, `gfem.TransientThermal` and
  `gfem.DCConduction` solve temperature and current, with heat sources and thermal
  boundaries ({doc}`../recipes/heater-temperature`).
- **Heat sources and boundaries.** `gfem.HeatSource`, `gfem.joule_heating`,
  `gfem.FixedTemperature`, `gfem.Convection` and `gfem.HeatFlux` set the thermal
  problem ({doc}`../recipes/heater-temperature`).
- **Thermo-optic phase.** `gfem.thermo_optic_shift` and `gfem.route_phase` give the
  phase a temperature rise adds to a guided mode ({doc}`../recipes/thermo-optic-phase`).
- **Time traces.** `gfem.Trace`, `gfem.rise_time` and `gfem.time_constant` read a
  transient run, with `gfem.UnsettledTraceWarning` on an unsettled trace
  ({doc}`../recipes/thermal-step-response`).
- **Thermal gradients.** `jax.grad` runs through DC conduction, Joule heating and
  the temperature, in the conductivities, sources and drive
  ({doc}`../recipes/thermal-gradient`).
- **Time schemes.** `SolverOptions.time_scheme` picks `"tr_bdf2"` or `"theta"`
  ({doc}`../recipes/thermal-step-response`).

Periodic surfaces:

- **Behaviour change: oblique absorbing faces warn.** A driven solve of a periodic
  model warns with `gfem.ObliqueAbsorberWarning` when a wave meets a `gfem.Absorbing`
  face with a reflection of 1 % or more. Filter the class to silence it
  ({doc}`../recipes/floquet-ports`).
- **Behaviour change: rerun periodic gradients with `axes`.** Gradients of a
  periodic model's `MaterialField` with `axes` next to a first-order `Absorbing`
  surface were off by 0.5 to 10 %. Recompute them ({doc}`../recipes/periodic-cell`).
- **Floquet ports.** `gfem.FloquetPort` lights a periodic cell with a plane wave
  and reads its diffraction orders ({doc}`../recipes/floquet-ports`).
- **Incidence.** `Model.bloch_wave_vector`, `Model.propagating_orders` and
  `DrivenResult.diffraction_orders` set and read the incidence
  ({doc}`../recipes/floquet-ports`).
- **Floquet gradients.** Design gradients run through Floquet S on the direct and
  BDDC solvers ({doc}`../recipes/floquet-ports`).
- **Periodic design grids.** `design.LayerGrid(..., periodic=(True, False))`
  filters across a periodic seam ({doc}`../tutorials/metagrating`).
- **Periodic cells.** `gfem.Periodic` pairs two surfaces, and eigenmode and uniform
  driven problems take a Bloch `wave_vector`, with gradients
  ({doc}`../recipes/periodic-cell`).

RF and microwave:

- **Behaviour change: `retain_memory` bounds everything kept.** Kept parts and
  couplings count in it, `0` keeps nothing, and a solve on an earlier design's
  factorization is accurate to `tolerance`. Call `gfem.release_memory()` before
  each finite-difference evaluation ({doc}`../recipes/s-parameter-gradient`).
- **Behaviour change: `"auto"` picks the direct solver.** A driven problem that
  fits now solves with `"direct"`, where it was `"bddc_gmres"`, so S moves by the
  iterative solve's error. Set `linear_solver` to keep the old route
  ({doc}`../recipes/s-parameter-gradient`).
- **Behaviour change: the default `retain_memory` keeps a large model's last
  factorization.** Set `retain_memory` explicitly on a shared machine, or to make
  reuse repeat exactly ({doc}`../recipes/s-parameter-gradient`).
- **Behaviour change: lumped values are design inputs.** A model with an
  `Impedance`, or a lumped port's L or C, gains pytree leaves. A traced lumped
  resistance raises `gfem.NonDifferentiableError`
  ({doc}`../recipes/s-parameter-gradient`).
- **Behaviour change: adaptive sweeps with `excite=`.** An S entry whose port is
  not excited had a gradient off by up to 1e-2. Each sample now solves every
  lumped port, so `excite=` saves nothing. Recompute those gradients
  ({doc}`../recipes/adaptive-sweep-gradient`).
- **Behaviour change: `MaterialField` values are checked at the solve.** A
  non-finite or non-passive value built under `jax.grad` now raises with the
  region named ({doc}`../recipes/s-parameter-gradient`).
- **Behaviour change: frequencies given twice.** `Driven.frequency` and
  `PortMode.frequency` raise `ValueError` for a repeat. Drop repeats with
  `numpy.unique` ({doc}`../recipes/frequency-sweep`).
- **Reuse across designs.** A design loop on `"direct"` reuses the wave-port
  couplings, the solver plan and each frequency's factorization
  ({doc}`../recipes/s-parameter-gradient`).
- **Warm starts.** `bddc_gmres` starts each design from the previous solution
  ({doc}`../recipes/s-parameter-gradient`).
- **What each solve did.** `SweepInfo.iterations` and `SweepInfo.preconditioner_reused`
  record each solve, and the `"gfem.solver"` logger says what it did
  ({doc}`../recipes/watch-a-solve`).
- **Electrostatic gradients.** `result.energy` differentiates in `eps_r` and the
  nodes, and its backward solve prepares once per step ({doc}`../recipes/capacitance-gradient`).
- **Kept fields under a transformation.** `Driven(keep_fields=...)` with wave ports
  runs under `jax.jit`, `jax.grad` and `jax.vmap` ({doc}`../recipes/port-mode-gradient`).
- **Power flow differentiates.** `gfem.power_flow` takes gradients through its
  field ({doc}`../recipes/measurements`).

Resonators and qubits:

- **Behaviour change: rerun eigenmode results on lossy walls.** `gfem.Eigenmode`
  could skip modes on a model with a `Conductor`, `Impedance` or `Absorbing`
  wall. Rerun such results from earlier versions ({doc}`../recipes/eigenmode-gradient`).
- **Behaviour change: eigen solves stop after 100 restarts.** Where modes near the
  target overlap within their linewidth, the solve raises
  `gfem.EigenmodeConvergenceError` after 100 restarts. Move the target, or sweep
  with `gfem.Driven` ({doc}`../recipes/eigenmode-gradient`).
- **Behaviour change: unconverged pairs raise.** A pair nearer the target than the
  farthest mode found that stops within 1e-6 of an eigenpair now raises
  `gfem.EigenmodeConvergenceError` ({doc}`../recipes/eigenmode-gradient`).
- **Behaviour change: modes are polished.** On the direct solver, participations,
  `q_port` and fields move by about 1e-6 to 1e-4 relative; a mode that does not
  polish raises `gfem.DegenerateModeError` when differentiated
  ({doc}`../recipes/eigenmode-gradient`).
- **Behaviour change: the mode energy counts capacitive sheets.** On models with
  an `Impedance` sheet that has a capacitance, field amplitudes, participations
  and `q_port` change ({doc}`../recipes/quantum-metrics`).
- **Behaviour change: `interface_participation` adds the two faces of a film.** On
  a film with the same medium on both faces the value rises to the sum of the
  faces ({doc}`../recipes/quantum-metrics`).
- **Behaviour change: periodic pairs of a saved result.**
  {py:obj}`gfem.interface_participation` reads `EigenmodeResult.periodic_surfaces`,
  so results saved earlier read as having no pair. Solve again to record it.
- **Participation gradients.** `participation`, `q_port` and the mode fields
  differentiate in materials, sheets and lumped values ({doc}`../recipes/eigenmode-gradient`).
- **Quantum metrics.** `gfem.quantum.dressed_modes` puts a junction's cosine on the
  solved modes, differentiable ({doc}`../recipes/quantum-metrics`).
- **Sheet participation.** `EigenmodeResult.sheet_participation` gives each mode's
  energy share in an `Impedance` sheet's inductance ({doc}`../recipes/quantum-metrics`).
- **Loss budget.** A mode's loss splits into its channels and their
  lifetime limits ({doc}`../recipes/loss-budget`).
- **Low-frequency warning.** `gfem.Eigenmode` warns with `gfem.LowFrequencyWarning`
  below the low-frequency floor ({doc}`../recipes/eigenmode-gradient`).
- **Transmon tutorial.** A transmon's modes, participations and anharmonicity
  ({doc}`../tutorials/transmon`).

Meshes, materials and boundaries:

- **Behaviour change: equal principal values are one value.**
  `Material(eps_r=(2, 2, 2))` is stored as `eps_r=2.0`, and its repr changes.
  Results are unchanged ({doc}`../recipes/anisotropic-materials`).
- **Behaviour change: absorbing faces next to anisotropic materials.** Each face
  now uses the material's exact normal-incidence admittance, so S, fields and Q
  change, and node gradients raise `NonDifferentiableError`. Rerun those models
  ({doc}`../recipes/anisotropic-materials`).
- **Behaviour change: rotated transverse tensors.** A port cross-section with a
  tensor rotated about the normal no longer drops its modes. Rerun it
  ({doc}`../recipes/anisotropic-materials`).
- **Behaviour change: `interface_participation`'s vacuum test.** A side with
  `eps_r = (1, 1, 4)` is no longer taken as vacuum. Rerun such results
  ({doc}`../recipes/anisotropic-materials`).
- **Behaviour change: refinement refuses what it underestimated.** S estimates on
  an `Impedance` or `Absorbing` surface, a lumped port across a permeability
  contrast or a law with a small negative real part raise
  `gfem.MeshRefinementUnsupportedError`. Use `gfem.FieldEstimator()`
  ({doc}`../recipes/mesh-refinement`).
- **Behaviour change: `refine_mesh` tolerance defaults to `None`.** It is 1e-2
  except for an objective, which needs its own ({doc}`../recipes/mesh-refinement`).
- **Anisotropic materials.** Driven, eigenmode, electrostatic and port mode solves,
  wave ports, `MaterialField` and its gradients take an anisotropic `Material`
  ({doc}`../recipes/anisotropic-materials`).
- **Adaptive mesh refinement.** `gfem.refine_mesh` subdivides where the error
  estimate is largest until it meets a tolerance ({doc}`../recipes/mesh-refinement`).
- **Refine for your objective.** `gfem.QuantityEstimator(objective=J)` refines for
  the error of the function you differentiate ({doc}`../recipes/mesh-refinement`).
- **Q and several quantities.** `QuantityEstimator(q=i)`, `omega=(i, j)` and
  tuples of estimators refine for a mode's Q, a cluster and several quantities
  ({doc}`../recipes/mesh-refinement`).
- **Refusals across ranks and degenerate modes.** `refine_mesh` and
  `gfem.error_estimate` raise `gfem.DistributedUnsupportedError` across ranks, and
  `gfem.DegenerateModeError` when the followed mode is unclear
  ({doc}`../recipes/mesh-refinement`).
- **Loaded results.** `gfem.error_estimate` accepts a result read with
  `gfem.load_result` ({doc}`../recipes/mesh-refinement`).
- **Permittivity laws.** `gfem.PermittivityLaw` with `gfem.Drude`, `gfem.Lorentz`,
  `gfem.Debye` and `gfem.DjordjevicSarkar` terms gives a material a frequency
  dependence, with gradients ({doc}`../recipes/dispersive-materials`).
- **Cross-section meshes.** `gfem.Mesh.from_gmsh_cross_section` reads a 2D gmsh
  section as a slab one element thick, with per-metre inputs and outputs
  ({doc}`../recipes/2d-models`).
- **Excluded per physics.** `excluded=` takes a mapping from physics to regions
  ({doc}`../recipes/heater-temperature`).

Design workflow:

- **Behaviour change: gradient of a zero conductivity.** The gradient with respect
  to a `MaterialField` conductivity that is zero everywhere was exactly zero and is
  now the one-sided derivative. Recompute it.
- **Behaviour change: `design.project` stays in [0, 1].** Round-off could leave a
  few ulp outside, which `SheetField` refused ({doc}`../recipes/robust-optimization`).

Solvers and scale-out:

- **Behaviour change: rebuild a PETSc stack made by the preview's script.** It could
  return wrong S with no error on a large direct solve. gfem now raises
  `gfem.SolverUnavailableError` on such a stack; rebuild it ({doc}`../recipes/petsc-slepc`).
- **Behaviour change: default thread count.** With `threads=None`, NGSolve runs on
  `NGS_NUM_THREADS`, else on the physical cores the process may use. Set
  `SolverOptions.threads` instead of `ngs.SetNumThreads`.
- **Behaviour change: PETSc settings that gave a wrong S raise.** `preonly` with a
  preconditioner other than `lu` or `cholesky`, or a skipped convergence test,
  raises `ValueError`. Use a Krylov `ksp_type` ({doc}`../recipes/petsc-slepc`).
- **Behaviour change: a launched solve is distributed or raises.** It used to run N
  complete one-process solves; `comm=MPI.COMM_SELF` asks for the copies by name
  ({doc}`../recipes/distributed-solve`).
- **Behaviour change: `order` above 12 raises.** Higher orders failed or hung.
- **Behaviour change: the gauge for low frequencies.** Direct driven solves use the
  tree-cotree gauge, which keeps S to round-off below the low-frequency floor and
  moves it in the tenth digit or later above. Set `gauge="none"` for the old results ({doc}`../recipes/large-driven-model`).
- **Solves across MPI ranks.** `mpirun -n N python script.py` runs uniform driven
  sweeps, electrostatic and eigenmode solves and periodic models across the ranks
  ({doc}`../recipes/distributed-solve`).
- **GPU direct solver.** `linear_solver="auto"` runs the direct solver with
  `device="cuda"` ({doc}`../recipes/gpu-direct-solver`).
- **Freeing memory.** `gfem.release_memory()` frees everything gfem keeps between
  solves.
- **New warnings.** `gfem.SweepRankDeficientWarning` and
  `gfem.UnsampledPixelsWarning` flag a near-singular adaptive sweep and unsampled
  pixels.
- **Fewer low-frequency warnings.** `LowFrequencyWarning` no longer fires on a
  gauged solve below the floor, and its message names why a solve is ungauged.

Results and files:

- **Behaviour change: unknown file keys are refused.** An unknown key in a job or
  result file raises `ValueError` naming the file, the class and both versions.
  Remove the key ({doc}`../recipes/save-and-export`).
- **Behaviour change: `Field.at` reads the right tetrahedron.** A point near a face
  could get the neighbour's value. Recompute readings taken within 1e-4 of an
  element's size from a face ({doc}`../recipes/measurements`).
- **Solve cache.** `gfem.solve(..., cache="solves")` and `--cache solves` keep each
  finished result and read it back ({doc}`../recipes/save-and-export`).
- **Partial bundles.** A uniform sweep that stops at an unsolved frequency leaves
  `partial.json` and `partial.npz` ({doc}`../recipes/save-and-export`).
- **Touchstone output.** `NetworkParameters.to_touchstone` writes rows in increasing
  frequency, and `z0=` labels a network with no reference impedance
  ({doc}`../recipes/save-and-export`).
- **Electric energy and region means.** `gfem.electric_energy` and
  `gfem.region_mean` read a field's energy and mean over regions
  ({doc}`../recipes/measurements`).
- **gsim ports.** The gsim recipe reads one-face lumped ports, which raised
  `KeyError` ({doc}`../recipes/gsim`).
- **Clearer messages.** Error messages name the remedy and use Hz, kHz, MHz, GHz or
  THz.

## Preview (2026-11-01)

Tagged 0.1.0 in the repository until the PyPI release; now commit
[`18235aa`](https://github.com/benvial/gfem/commit/18235aa). Driven problems (uniform
and adaptive sweeps), eigenmode problems (including nonlinear ones), electrostatic
and port mode problems, with lumped and wave ports. Gradients with respect to
permittivity, conductivity, sheet fill and mesh node coordinates are checked
against finite differences.

Threaded `direct` and `bddc_gmres` solves got faster past 16 threads; results change
only to the solver's precision.
