# Capabilities

What gfem solves, which results it differentiates, and what it refuses.

## What you can solve

"Matrix" in a Gradient cell points to the
{ref}`generated matrix <what-you-can-differentiate>`; "Yes" and "Partial" are
claims of the linked page. In the feature tables, "3D only" is "No" where the
feature also runs in 2D.

### Problems

| Problem | Result | 2D | 3D | Gradient | Page |
|---|---|---|---|---|---|
| {py:obj}`gfem.Driven` | S-parameters and fields over a uniform or adaptive sweep | Uniform sweep, one polarization | Yes | Matrix | {doc}`recipes/frequency-sweep` |
| {py:obj}`gfem.Eigenmode` | Resonant frequencies and Q nearest a target | No | Yes | Matrix | {doc}`recipes/eigenmode-gradient` |
| {py:obj}`gfem.Electrostatic` | Maxwell capacitance matrix | No | Yes | Matrix | {doc}`recipes/capacitance-gradient` |
| {py:obj}`gfem.PortMode` | $n_\mathrm{eff}$, $Z_{PV}$, group index, dispersion | Cross-section | n/a | Matrix; group index raises | {doc}`recipes/waveguide-dispersion` |
| {py:func}`gfem.band_diagram` | Bands and {term}`band group velocity` | No | Yes | Band frequencies only | {doc}`recipes/periodic-cell` |
| {py:obj}`gfem.SteadyThermal` | Temperature under heat sources | Cross-section | Yes | Matrix | {doc}`recipes/heater-temperature` |
| {py:obj}`gfem.TransientThermal` | Temperature traces over time | Cross-section | Yes | Matrix | {doc}`recipes/thermal-step-response` |
| {py:obj}`gfem.DCConduction` | Conductance and Joule heat | Cross-section | Yes | Matrix | {doc}`recipes/heater-temperature` |
| {py:obj}`gfem.Piezoelectric` | Admittance per length, clamped capacitance | Yes | No | Matrix | {doc}`recipes/piezoelectric-admittance` |
| {py:class}`gfem.MechanicalEigenmode` | Complex frequency, Q, group velocity at an {term}`axial wavenumber` | Yes | No | `omega` and fields | {doc}`recipes/mechanical-modes` |

### Integrated photonics

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Several modes on one wave-port surface | {py:obj}`gfem.WavePort` | Yes | Matrix, at port faces too | {doc}`recipes/shared-port-surface` |
| Circuits | {py:obj}`gfem.SaxComponent`, `to_sdict`, `to_skrf` | No | Yes, not forward mode | {doc}`recipes/sax-circuit` |

### Nonlinear and quantum optics

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Kerr nonlinearity | {py:obj}`gfem.kerr_nonlinearity`, {py:obj}`gfem.kerr_coupling` | No | Partial: materials, planar-port nodes | {doc}`recipes/waveguide-nonlinearity` |
| Crystal Kerr effect | `nonlinear_anisotropy` of a {py:obj}`gfem.Material` | No | Partial: materials, planar-port nodes | {doc}`recipes/crystal-kerr` |
| Raman gain | {py:obj}`gfem.raman_gain`, {py:obj}`gfem.RamanSpectrum` | No | Partial: materials, planar-port nodes | {doc}`recipes/raman-gain` |
| Free carriers | {py:obj}`gfem.free_carriers` | No | Partial: materials, planar-port nodes | {doc}`recipes/free-carriers` |
| Second-harmonic and three-wave coupling | {py:obj}`gfem.second_harmonic`, {py:obj}`gfem.three_wave_coupling` | No | Partial: materials, not nodes | {doc}`recipes/second-harmonic-generation` |
| Photon pairs | {py:obj}`gfem.spdc`, {py:obj}`gfem.sfwm` | No | Partial: materials, not nodes | {doc}`recipes/photon-pair-sources` |

### Electro-optics

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Electro-optic shift, $V_\pi L$ | {py:obj}`gfem.electro_optic_shift` | No | Partial: not the optical model | {doc}`recipes/electro-optic-phase` |

### Heat and thermo-optics

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Thermal boundaries | {py:obj}`gfem.FixedTemperature`, {py:obj}`gfem.Convection`, {py:obj}`gfem.HeatFlux` | No | Matrix | {doc}`recipes/heater-temperature` |
| Heat sources and heat flow | {py:obj}`gfem.joule_heating`, {py:meth}`gfem.HeatSource.uniform`, {py:obj}`gfem.heat_flow` | No | Matrix | {doc}`recipes/heater-temperature` |
| Electromagnetic heating | {py:obj}`gfem.electromagnetic_heating` | Yes | Matrix | {doc}`recipes/electromagnetic-heating` |
| Thermo-optic phase | {py:obj}`gfem.thermo_optic_shift`, {py:obj}`gfem.route_phase` | No | Matrix | {doc}`recipes/thermo-optic-phase` |
| Rise time, time constant | {py:obj}`gfem.rise_time`, {py:obj}`gfem.time_constant` | No | No | {doc}`recipes/thermal-step-response` |

### Mechanics and acoustics

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Acousto-optic shift | {py:obj}`gfem.acousto_optic_shift` | No | Partial: not the optical model | {doc}`recipes/acousto-optic-shift` |
| Optomechanical $g_0$ | {py:obj}`gfem.optomechanical_coupling`, {py:obj}`gfem.optomechanical_overlap` | No | Partial: optical mode held | {doc}`recipes/optomechanical-coupling` |
| Brillouin gain | {py:obj}`gfem.brillouin_gain` | No | Partial: optical mode held | {doc}`recipes/brillouin-gain` |

### Free-space scattering

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Incident fields | {py:obj}`gfem.PlaneWave`, {py:obj}`gfem.GaussianBeam`, {py:obj}`gfem.IncidentField` | Yes | Matrix; nodes raise | {doc}`recipes/incident-fields` |
| Point dipoles | {py:obj}`gfem.ElectricDipole`, {py:obj}`gfem.MagneticDipole` | Yes | Matrix | {doc}`recipes/point-dipoles` |
| Cross sections | {py:obj}`gfem.cross_sections`, {py:obj}`gfem.purcell_factor` | Yes | Partial: material fields | {doc}`recipes/cross-sections` |
| Far field | {py:obj}`gfem.far_field`, {py:obj}`gfem.beam_coupling` | Yes | Yes | {doc}`recipes/far-field` |
| Multipoles | {py:obj}`gfem.multipoles`, {py:obj}`gfem.current_moments` | Yes | Partial: multipoles, not nodes | {doc}`recipes/multipoles` |
| Optical force | {py:obj}`gfem.optical_force` | Yes | Yes | {doc}`recipes/optical-force` |

### Periodic surfaces

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Bloch wave vector, fixed-angle sweep | `wave_vector=`, `wave_vector_frequency` | Yes | Matrix | {doc}`recipes/periodic-cell` |
| Floquet ports | {py:obj}`gfem.FloquetPort` | Yes | Matrix (periodic rows) | {doc}`recipes/floquet-ports` |
| Diffraction efficiency, Jones and Mueller matrices | `gfem.diffraction` | Yes | Partial: not the angle | {doc}`recipes/diffraction-efficiency` |
| Effective medium | `gfem.diffraction` | Yes | Yes | {doc}`recipes/effective-medium` |
| Emissivity | `gfem.diffraction` | Yes | Partial: not the angles | {doc}`recipes/emissivity` |
| Metasurface aperture fields | `gfem.diffraction` | Yes | Partial: not the angle | {doc}`recipes/metasurface-from-cells` |

### RF and microwave

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Lumped and line ports | {py:obj}`gfem.LumpedPort` | No | Matrix; resistance raises | {doc}`recipes/s-parameter-gradient` |
| Rational model, SPICE and Verilog-A | `NetworkParameters.fit_rational` | n/a | No | {doc}`recipes/rational-model` |
| Time responses, impedance profile | `step_response`, `impulse_response` | n/a | Yes | {doc}`recipes/time-responses` |

### Resonators and qubits

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Dressed modes, kinetic inductance | {py:obj}`gfem.quantum.dressed_modes`, `sheet_participation` | Yes | Yes | {doc}`recipes/quantum-metrics` |
| $1/Q$ budget, $T_1$ limits | {py:obj}`gfem.interface_participation`, {py:obj}`gfem.dissipated_power` | Yes | Partial: isolated modes | {doc}`recipes/loss-budget` |

### Meshes, materials and boundaries

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Boundaries | {py:obj}`gfem.PEC`, {py:obj}`gfem.PMC`, {py:obj}`gfem.Absorbing`, {py:obj}`gfem.Conductor`, {py:obj}`gfem.Impedance`, {py:obj}`gfem.Periodic` | No | Matrix (impedance values) | {doc}`recipes/periodic-cell` |
| Materials, material and sheet fields | {py:obj}`gfem.Material`, {py:obj}`gfem.MaterialField`, {py:obj}`gfem.SheetField` | No | Matrix | {doc}`recipes/sheet-design` |
| Anisotropic materials | {py:obj}`gfem.Material` | No | Matrix (principal values) | {doc}`recipes/anisotropic-materials` |
| Dispersive materials | {py:obj}`gfem.PermittivityLaw` | No | Matrix (law parameters) | {doc}`recipes/dispersive-materials` |
| Material library and fit | {py:mod}`gfem.materials`, {py:obj}`gfem.fit_permittivity` | n/a | n/a | {doc}`recipes/fit-measured-data` |
| Perfectly matched layer | {py:obj}`gfem.PML` | Yes | Yes, not layer parameters | {doc}`recipes/perfectly-matched-layer` |

### Design workflow

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Layer patterns to GDS | {py:obj}`gfem.design.LayerPattern`, {py:obj}`gfem.design.write_gds` | n/a | No; call it on the final design | {doc}`recipes/fabricable-layout` |
| Tolerance report | {py:obj}`gfem.tolerance_report` | No | n/a | {doc}`recipes/tolerance-report` |

### Results and files

| Feature | Calls | 3D only | Gradient | Page |
|---|---|---|---|---|
| Measurements | {py:obj}`gfem.power_flow`, {py:obj}`gfem.dissipated_power`, {py:obj}`gfem.electric_energy` | No | Matrix (energy, loss rows) | {doc}`recipes/measurements` |

## Solvers

| Option | Pick it for | Problems | Gradients | Page |
|---|---|---|---|---|
| `linear_solver="direct"` | A factorization that fits in memory; optimization loops | All | Yes | {doc}`recipes/s-parameter-gradient` |
| `linear_solver="pmg_gmres"` | Driven models several wavelengths across | Driven, order 2-3, one CPU process, not periodic | Yes | {doc}`recipes/large-driven-model` |
| `linear_solver="bddc_gmres"` | MPI ranks, periodic and Bloch models, electrostatics | Driven, electrostatic, thermal; no PML | Matrix | {doc}`recipes/distributed-solve` |
| `linear_solver="petsc"`, `eigen_solver="slepc"` | MUMPS, your own PETSc or SLEPc settings | Driven, eigenmode, thermal | n/a | {doc}`recipes/petsc-slepc` |
| `device="cuda"` | One NVIDIA GPU, experimental | Driven, direct | n/a | {doc}`recipes/gpu-direct-solver` |
| Several MPI ranks | Uniform driven sweeps, electrostatic and eigenmode solves, one-axis periodic cells | Others raise {py:obj}`gfem.DistributedUnsupportedError` | Except mode fields | {doc}`recipes/distributed-solve` |

`"auto"` picks the direct solver while its factorization fits in half the
memory, then `"pmg_gmres"`, then `"bddc_gmres"`. Thermal and DC conduction
problems solve on `"direct"` and accept `"bddc_gmres"` and `"petsc"`; the GPU
solver and several ranks raise there. Install the `mkl` extra for the fastest
factorization. How large a model each solver has finished is in
{doc}`project/scaling`.

(what-you-can-differentiate)=

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

How each mark is checked: {doc}`verification`.

### Gradients that raise

Off the matrix, asking for these gradients raises the named error instead of
returning a wrong value:

```{gfem-raises}
```

## Limits

| Feature | Limit | Details |
|---|---|---|
| **Low frequency** | Below $k_0 h_\mathrm{min} = 10^{-6}$, ungauged solves lose digits and warn; for capacitance, solve {py:obj}`gfem.Electrostatic` | {py:obj}`gfem.Driven` |
| **Gyrotropic materials** | Non-symmetric tensors, such as magnetized ferrite, raise `ValueError` | {doc}`recipes/anisotropic-materials` |
| **Anisotropic materials** | At wave ports, a principal axis must lie along the port normal; second-order absorbing boundaries raise next to them | {doc}`recipes/anisotropic-materials` |
| **Periodic models** | Bloch needs the direct solver (eigenmode) or direct or BDDC (driven); no adaptive sweep; band diagram eager, one process | {doc}`recipes/periodic-cell` |
| **Floquet ports** | Uniform driven sweep, direct or BDDC; homogeneous lossless half-space; no incidence gradient | {doc}`recipes/floquet-ports` |
| **Periodic surfaces** | No gradient in the angle; power balance needs a port per propagating order; aperture fields at normal incidence | {doc}`recipes/diffraction-efficiency` |
| **Perfectly matched layers** | Box shells only; no error estimates; refuses `"bddc_gmres"` | {doc}`recipes/perfectly-matched-layer` |
| **Free space** | 3D only; needs a perfectly matched layer or absorbing faces; refused on periodic models and the adaptive sweep | {doc}`recipes/incident-fields` |
| **Multipoles** | Incident-field columns, homogeneous lossless background; not differentiable by nodes | {doc}`recipes/multipoles` |
| **Unrequested port modes** | A propagating mode no port requests is partly reflected, not converted | {doc}`modes-and-ports` |
| **Eigenmodes with wave ports** | One process, no Bloch vector; a target below a port mode's cutoff raises; Q above about 2 only | {doc}`modes-and-ports` |
| **Lossy port lines** | Give every wave port a `voltage_path`, or none, for a symmetric S | {doc}`recipes/shared-port-surface` |
| **Designs at wave ports** | Renormalize S where a port meets a fixed line; eigenmode solves raise {py:obj}`gfem.PortGeometryError` | {doc}`recipes/port-mode-gradient` |
| **Group index and dispersion** | `PortMode` results only; not differentiable | {doc}`recipes/waveguide-dispersion` |
| **2D models** | Normal incidence, one polarization; uniform driven sweep and piezo/mechanics only | {doc}`recipes/2d-models` |
| **Dispersive materials** | Isotropic laws only; a target in the {term}`unresolved band` raises; outside the {term}`fitted range`, a solve warns | {doc}`recipes/dispersive-materials` |
| **Mode-field gradients** | Isolated modes, direct solver, CPU, one process, non-periodic models | {doc}`recipes/eigenmode-gradient` |
| **Quantum metrics** | One junction, transmon regime; checked against identities, not a measured device | {doc}`recipes/quantum-metrics` |
| **Thermal problems** | Linear, constant isotropic properties; no radiation or contact resistance; thermo-optic shift is first order | {doc}`recipes/heater-temperature` |
| **Electro-optic shift** | Lossless section at DC, mode held constant; no 2D model | {doc}`recipes/electro-optic-phase` |
| **Piezoelectric problems** | 2D sections, one process, direct solver; no PML; damping frames instead | {doc}`recipes/piezoelectric-admittance` |
| **Mechanical eigenmodes** | 2D sections, one process, direct solver; `group_velocity` raises under `jax.grad` | {doc}`recipes/mechanical-modes` |
| **Layer patterns** | Width and spacing only; a pre-check, not foundry sign-off; layer grids only | {doc}`recipes/fabricable-layout` |
| **SAX circuits** | Answers inside its band; build outside `jax.jit`; no forward mode | {doc}`recipes/sax-circuit` |

## Platforms

| Platform | Python | GPU solver (cuDSS) |
|---|---|---|
| Linux x86_64 (glibc ≥ 2.28) | ≥ 3.12 | Yes |
| macOS arm64 | ≥ 3.12 | No |
| Windows x86_64 | ≥ 3.12 | Yes |

Importing gfem turns on JAX double precision (`jax_enable_x64`), which every
solve needs. See {doc}`get-started`.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain
GPL-2.0+ SuiteSparse code; see {doc}`project/licence`.
