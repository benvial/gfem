# Verification

Each claim on {doc}`capabilities` and in the recipes is backed by a test that
solves a case with a known answer. The answer is a closed form, an identity
the physics must obey, an independent method, or finite differences for a
gradient. Each table lists the cases of one area in physical terms and links
the test files on GitHub; the tolerance each check asserts is in its test.

Runs says when a case is run: Every change (each change to gfem), Before
release (large cases), Twice a week (cases that need the {term}`PETSc stack`
or {term}`MPI PETSc stack`) or Once (a single run, not repeated). An
observation is a comparison with published or measured data that no test
asserts.

## Gradients

Every FD-verified mark of the gradient matrix on {doc}`capabilities` is a set
of cells, one per problem, input, linear solver and sweep. Each cell compares
the gradient with fourth-order central differences of the same objective,
then checks that the Taylor remainder shrinks at second order with the step.
An exact zero is checked to be zero, and each "raises" mark is a call that
must raise the named error. The cells include the parameters of a
permittivity law. The matrix and its list of raises are built from these
tests when the docs are built, so they cannot drift from them.

```{gfem-gradient-checks}
```

The same cells run on `linear_solver="petsc"` and `eigen_solver="slepc"`
(see [PETSc and SLEPc](#verification-petsc)), so the matrix does not list
PETSc.

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Cells of the gradient matrix, one per problem and input | Fourth-order central differences; second-order Taylor remainder | {gfem-test}`tests/verification/test_gradient_matrix.py` | Every change |
| Cells between modes of a shared port surface | Fourth-order central differences | {gfem-test}`tests/verification/test_gradient_matrix.py` | Before release |
| Calls that must raise, and exact zeros | The named error; zero | {gfem-test}`tests/verification/test_gradient_matrix_raise_table.py` | Every change |
| Driven optimization reusing an earlier design's factorization | Factorizing every design | {gfem-test}`tests/verification/test_factor_reuse_gradient.py` | Every change |
| Thermal and DC conduction designs reusing plans and factors | Fresh solves, bit for bit at one solve thread | {gfem-test}`tests/unit/test_thermal_plan_pool.py` | Every change |
| PETSc and SLEPc cells of the matrix | Finite differences | {gfem-test}`tests/verification/test_gradient_matrix.py` | Twice a week |

## Wave ports and shared port surfaces

Modes of a guide section used as ports, and their gradients
({doc}`recipes/shared-port-surface`, {doc}`recipes/port-mode-gradient`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Matched parallel-plate line, TEM mode and first higher mode | Analytic propagation | {gfem-test}`tests/unit/test_driven_wave_ports.py` | Every change |
| Open dielectric guide between two wave ports | Slab effective index; passive and reciprocal S | {gfem-test}`tests/unit/test_driven_wave_ports.py` | Every change |
| Rectangular guide with a dielectric slab, TE10 to TE20 on a shared port surface | Independent mode matching; unitary and reciprocal S; power balance | {gfem-test}`tests/verification/test_shared_port_surface.py` | Every change |
| Silicon strip, TE0 and TM0 requested on one port surface | Transmitted with their own phase, no reflection or conversion | {gfem-test}`tests/reference/test_shared_port_surface.py` | Before release |
| Port mode $\gamma$ and $Z_{PV}$ gradients: lossy guide, parallel-plate TEM, stretched face, sheet | Closed forms of the discrete mode, e.g. $\partial\gamma/\partial\varepsilon_r = \gamma/(2\varepsilon_r)$ | {gfem-test}`tests/verification/test_port_mode_gradient.py` | Every change |
| Slab-loaded guide, derivative of the transverse-resonance root | Closed form, as element order rises | {gfem-test}`tests/verification/test_port_mode_gradient.py` | Every change |
| Designs on the port faces: every material input, sheet, node moves, lossy and lossless, shared surface, absorbing wall, PML-cut face | Finite differences | {gfem-test}`tests/verification/test_port_mode_gradient.py` | Every change |
| Stretched guide faces: volume part and port-mode part | Differences with the other part frozen; exact solves | {gfem-test}`tests/verification/test_shape_gradient.py` | Every change |
| Recipe's port mode gradient | Finite differences | {gfem-test}`tests/unit/test_docs_port_mode_gradient_recipe.py` | Every change |

## Group index and dispersion

A port mode's group index $n_g$, $\beta_2$, $\beta_3$ and the dispersion
parameter $D$ ({doc}`recipes/waveguide-dispersion`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Guides filled with air, dielectric, lossless or lossy Lorentz or Drude; parallel-plate lines in both polarizations | The filled guide's dispersion relation with the mesh's cutoff | {gfem-test}`tests/unit/test_port_mode_dispersion.py` | Every change |
| TE0 and TM0 of a silicon slab in oxide with Sellmeier laws | Values to 40 digits from the dispersion relation, each order closer | {gfem-test}`tests/verification/test_port_mode_dispersion.py` | Every change |
| Empty rectangular guide TE10; guide with a slab against one wall | Closed form; stored energy over power | {gfem-test}`tests/verification/test_port_mode_convergence.py` | Every change |
| Copper-wall guides, lossy plates, absorbing end, PML-cut strip | Central differences of exact solves | {gfem-test}`tests/verification/test_port_mode_convergence.py` | Every change |
| Reduced-model sweep rows: slab, hybrid section, coplanar line | Exact solves at the same frequencies | {gfem-test}`tests/verification/test_port_mode_sweep.py` | Every change |
| Port mode solve against driven solve on one grid | Same propagation constants, bit for bit | {gfem-test}`tests/verification/test_port_mode_sweep.py` | Every change |
| $D$, saving, `to_xarray`, `jax.jit`, `jax.grad` | $D$ from $\beta_2$; the same values; the documented raise | {gfem-test}`tests/unit/test_port_mode_dispersion.py` | Every change |
| Dispersion on two ranks | One process | {gfem-test}`tests/mpi/test_wave_ports.py` | Twice a week |

Observations, not checks: {doc}`recipes/waveguide-dispersion` compares a
silicon wire's group index with a measured one and its $\beta_2$ with a
published simulation. Not checked: a fabricated guide's measured dispersion;
modes at an exact degeneracy (refused).

## SAX circuits

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Reference plane moved inside a slab | Plain slab followed by SAX's straight guide with the port mode's index | {gfem-test}`tests/unit/test_sax_circuits.py` | Every change |
| RF block followed by a series capacitor, at 50 Ω and 75 Ω | scikit-rf's cascade of the two networks | {gfem-test}`tests/unit/test_sax_circuits.py` | Every change |
| Gradient of a circuit in one `jax.grad`, also under `jax.jit` | Fourth-order central differences | {gfem-test}`tests/unit/test_sax_circuits.py` | Every change |
| 3D E-plane bend rotated and mirrored; interferometer of a bend and its quarter turn | Same $S_{21}$; all power to the cross output | {gfem-test}`tests/verification/test_circuit_polarity.py` | Every change |
| Photonic and RF conversions of S | Conjugate for photonic; renormalized S for RF ({doc}`phasors-and-power-waves`) | {doc}`phasors-and-power-waves` | Every change |
| Recipe's circuits | Their printed values | {gfem-test}`tests/unit/test_docs_sax_recipe.py` | Every change |

## 2D models

{term}`2D models <2D model>` posed as a line port or a {term}`slab <Slab
model>` ({doc}`recipes/2d-models`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Loaded line in Ez and Hz, both wall kinds | Closed-form three-part transmission line | {gfem-test}`tests/verification/test_2d_models.py` | Every change |
| Guide modes on a line port at three wavelengths, Ez and Hz | Symmetric slab guide's closed form | {gfem-test}`tests/verification/test_2d_line_port_modes.py` | Every change |
| Bend with a uniform design square | The same plane posed as a slab; $\sqrt h$ field scaling | {gfem-test}`tests/verification/test_2d_models.py` | Every change |
| Gradients by material fields | Central differences; second-order Taylor remainder | {gfem-test}`tests/unit/test_2d_gradients.py` | Every change |
| Gradients by layer-grid pixels at orders 2 and 3 | Central differences | {gfem-test}`tests/unit/test_2d_layer_gradients.py` | Every change |
| Design reaching a line port, Ez and Hz, lossless and lossy | Finite differences; closed form for a uniform line | {gfem-test}`tests/verification/test_line_port_gradient.py` | Every change |
| Metal bend through gold's law and the index blend | Finite differences on the 2D model and the slab | {gfem-test}`tests/verification/test_metal_bend.py` | Every change |
| 45° gold mirror, tutorial mesh at order 3 | The 2D model on finer cells, frozen; same gradient direction as the slab | {gfem-test}`tests/reference/test_metal_bend_fine.py` | Before release |

## Kerr nonlinearity

A port mode's $\gamma_\mathrm{NL}$ and its coupling tensor
({doc}`recipes/waveguide-nonlinearity`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| TE10 of a filled rectangular guide | $\gamma_\mathrm{NL} = \tfrac32\,(k_0 n_2/ab)\,(n/n_\mathrm{eff})^2$ | {gfem-test}`tests/verification/test_kerr.py` | Every change |
| TE0 of a silicon slab in oxide on a line port | Value to 30 digits; error falls with element order | {gfem-test}`tests/verification/test_kerr.py` | Every change |
| Every row of a 41-row grid, reduced rows included | Frequency and effective-index scaling | {gfem-test}`tests/verification/test_kerr.py` | Every change |
| Two-photon absorption | $\mathrm{Im}/\mathrm{Re} = \beta_\mathrm{TPA}/(2k_0 n_2)$ | {gfem-test}`tests/verification/test_kerr.py` | Every change |
| Coupling tensor: symmetry, energy identity, equal entries, cross-phase | Identities to rounding; $\tfrac49 n_m/n_n$ of self-phase | {gfem-test}`tests/unit/test_kerr.py` | Every change |
| Silicon wire at three wavelengths | Real tensor in the real-transverse-field gauge | {gfem-test}`tests/verification/test_kerr.py` | Every change |
| Gradients in $\varepsilon_r$, $n_2$, $\beta_\mathrm{TPA}$; under `jax.grad` | Fourth-order central differences; eager value | {gfem-test}`tests/verification/test_kerr.py` | Every change |
| Quadrature columns; PML triangles carrying $n_2$ | Field at the points; section area; no change | {gfem-test}`tests/unit/test_kerr.py` | Every change |
| Each refusal | Raises its named error | {gfem-test}`tests/unit/test_kerr.py` | Every change |
| Recipe's printed value | Its source | {gfem-test}`tests/unit/test_docs_waveguide_nonlinearity_recipe.py` | Every change |

Not checked: a measured nonlinear coefficient, and any literature value (the
silicon and nitride constants are inputs). The nonlinear-optics theory is on
{doc}`nonlinear-and-quantum-optics`.

## Second-order nonlinearity

The overlap $\kappa$, $\eta_\mathrm{norm}$, the phase mismatch and the poling
period ({doc}`recipes/second-harmonic-generation`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| TE10 at $\omega$ to TE10 at $2\omega$ in a filled rectangular guide | Closed-form $\eta_\mathrm{norm}$ with $A_\mathrm{eff} = (9\pi^2/128)\,ab$; discrete indices | {gfem-test}`tests/verification/test_three_wave.py` | Every change |
| TE mode of a filled slab on a line port | Same form per unit height; error falls with element order | {gfem-test}`tests/verification/test_three_wave.py` | Every change |
| Pump TE10 squared into TE$m$0 at $2\omega$ | Overlap $-4a/(\pi m(m^2-4))$; TE20 uncoupled | {gfem-test}`tests/verification/test_three_wave.py` | Every change |
| Gauge on a lossless section | $\mathrm{Im}\,\kappa$ vanishes | {gfem-test}`tests/unit/test_three_wave.py` | Every change |
| Tensor between waves; `efficiency` | Symmetric; sinc$^2$, lossy and quasi-phase-matched forms | {gfem-test}`tests/unit/test_three_wave.py` | Every change |
| Gradient in $d$ | Central finite difference | {gfem-test}`tests/verification/test_three_wave.py` | Every change |
| Each refusal | Raises its named error | {gfem-test}`tests/unit/test_three_wave.py` | Every change |
| Recipe's printed figures | Its source | {gfem-test}`tests/unit/test_docs_second_harmonic_recipe.py` | Every change |

Observations, not checks: {doc}`recipes/second-harmonic-generation` prints a
thin-film lithium niobate ridge beside the ideal theory it follows. Not
checked: a measured conversion efficiency, and any literature value of $d$.

## Photon pairs

The pair rate, the joint spectrum of a pulse and its purity
({doc}`recipes/photon-pair-sources`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Rate of a nondegenerate pair with linear mismatch; degenerate pair growing as $L^{3/2}$ | Closed forms from $\int\mathrm{sinc}^2$ | {gfem-test}`tests/verification/test_photon_pairs.py` | Every change |
| Filled rectangular guide, down-conversion and four-wave mixing rates | Closed-form overlaps and dispersion; element order | {gfem-test}`tests/verification/test_photon_pairs.py` | Every change |
| Density of pairs at degeneracy and matching | $\lvert G\rvert^2\eta_\mathrm{norm}PL^2$ of `second_harmonic` | {gfem-test}`tests/unit/test_photon_pairs_math.py` | Every change |
| Purity of a Gaussian pump | $\sqrt{1 - A_{12}^2/(A_{11}A_{22})}$; singular value decomposition | {gfem-test}`tests/unit/test_photon_pairs_math.py` | Every change |
| Distinguishable photons; lossy coincidence law | Twice the pairs; second-harmonic law with the pair's attenuations | {gfem-test}`tests/unit/test_photon_pairs.py` | Every change |
| Rate, pairs per pulse, purity gradients | Central finite difference | {gfem-test}`tests/verification/test_photon_pairs.py` | Every change |
| Refusals and warnings | Raise or warn by name | {gfem-test}`tests/unit/test_photon_pairs.py` | Every change |
| Recipe's printed figures | Its source | {gfem-test}`tests/unit/test_docs_photon_pair_recipe.py` | Every change |

Observations, not checks: {doc}`recipes/photon-pair-sources` prints a lithium
niobate ridge and a silicon wire beside the published numbers they follow.
Not checked: a measured pair rate or brightness.

## Electro-optics

The electro-optic shift ({doc}`recipes/electro-optic-phase`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Lithium niobate film between electrodes, $r_{33}$ and $r_{22}$ | Slab's closed-form shift | {gfem-test}`tests/verification/test_electro_optic.py` | Every change |
| Ridge (diagonal part) and L-shaped core ($r_{51}$ alone) at $\pm$ drive | Mode re-solved on the perturbed permittivity | {gfem-test}`tests/verification/test_electro_optic.py` | Every change |
| x-cut section and the same turned into z-cut | Same shift | {gfem-test}`tests/verification/test_electro_optic.py` | Every change |
| Linearity; uniform static field; equal diagonal entries | Exact; thermo-optic shift of the matching coefficient | {gfem-test}`tests/unit/test_electro_optic.py` | Every change |
| Optical model on a second mesh of the same section | Same shift | {gfem-test}`tests/verification/test_electro_optic.py` | Every change |
| Parallel plates on a section | $\varepsilon_0 \varepsilon_r H / d$ per metre | {gfem-test}`tests/unit/test_electro_optic.py` | Every change |
| Gradients in the tensor and static permittivity | Fourth-order central differences | {gfem-test}`tests/unit/test_electro_optic.py` | Every change |
| Gradient in the static mesh nodes: gap morph, optical window | Fourth-order central differences | {gfem-test}`tests/verification/test_electro_optic_gradient.py` | Every change |
| Electro-optic phase of an arm: $e^{-i\beta L}$ and the push-pull $V_\pi L$ | Closed form ({doc}`multiphysics-couplings`) | {doc}`multiphysics-couplings` | Every change |

Observations, not checks: the modulator of Wang et al. (2018)
({doc}`tutorials/modulator`) and an optical window against the whole
section ({doc}`recipes/electro-optic-phase`).

## Thermal

Thermal and DC conduction, electromagnetic heat and heat flow
({doc}`recipes/heater-temperature`, {doc}`recipes/electromagnetic-heating`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Layered slabs in series and under convection; bar; two-conductor slab | Hand temperatures, $R = L/(\sigma A)$, hand formulas | {gfem-test}`tests/verification/test_thermal_closed_forms.py` | Every change |
| Strip heater on a substrate; slab's slowest mode; step response; annular bend | $\ln\coth$ profile; time constant; closed-form resistance | {gfem-test}`tests/verification/test_thermal_closed_forms.py` | Every change |
| Source on and off; doubled power; ten time constants | Two superposed steps; twice the rise; steady temperature | {gfem-test}`tests/unit/test_transient_thermal.py` | Every change |
| Steady thermal solves and their balance | Closed forms of the discrete problem | {gfem-test}`tests/unit/test_steady_thermal.py` | Every change |
| Thermo-optic shift, optical window, 3D route | Mode re-solved on the heated permittivity; phase per length times length | {gfem-test}`tests/verification/test_thermo_optic.py` | Every change |
| Phase of a route through a 3D rise | Section's phase per length times its length | {gfem-test}`tests/verification/test_route_phase.py` | Every change |
| Heater section extruded in 3D layers | Slab's resistance per length and core rise per watt per metre | {gfem-test}`tests/verification/test_thermal_extrusion.py` | Every change |
| DC conduction | Closed forms; terminal power equals dissipation | {gfem-test}`tests/verification/test_dc_conduction.py` | Every change |
| TiN heater of femwell, meshed by gsim | gsim's $P_\pi$, resistance, rise time; femwell's published 22.6 mW | {gfem-test}`tests/verification/test_thermal_parity.py` | Every change |
| Dissipated power of a driven field in each region | Power through the region's surface; losses sum to $1 - \sum_j \lvert S_{j1}\rvert^2$ | {gfem-test}`tests/unit/test_electromagnetic_heating.py` | Every change |
| Heat sources from a driven field | Superposed fields; PML and ports dissipate nothing | {gfem-test}`tests/unit/test_electromagnetic_heat_source.py` | Every change |
| Lossy line sections, metal wall, 2D cross-section slab | Transmission-line closed forms; 2D model's loss | {gfem-test}`tests/verification/test_electromagnetic_heat_closed_forms.py` | Every change |
| Gold bend's absorbed power | Power through its surface | {gfem-test}`tests/verification/test_metal_bend.py` | Every change |
| Sheet heat on a face | Hand fluxes; all to the sink, split by conductances | {gfem-test}`tests/unit/test_face_heat.py` | Every change |
| Heat out of every boundary surface | Source power; hand fluxes; series solution | {gfem-test}`tests/unit/test_heat_flow.py` | Every change |
| Transient heat out of the sinks | Heat put in minus heat stored, every time scheme | {gfem-test}`tests/unit/test_heat_out_trace.py` | Every change |
| Thermal and heat gradients through DC conduction, Joule heating and the driven chain | Fourth-order central differences; also on BDDC + GMRES | {gfem-test}`tests/verification/test_gradient_matrix.py` | Every change |

Observations, not checks: the TiN heater of Jacques et al. (2019)
({doc}`tutorials/heater`) and the 3D spiral heater on three meshes
({doc}`tutorials/heater3d`).

## Piezoelectric problems

({doc}`recipes/piezoelectric-admittance`, {doc}`recipes/piezoelectric-gradient`)

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Hexagonal plate between electrodes, thickness-extensional and thickness-shear, with and without loss | Closed-form admittance; shear displacement and strain (Voigt order, per-volt scale) | {gfem-test}`tests/verification/test_piezoelectric.py` | Every change |
| Capacitance a finger pair adds to a long array | Closed form of a periodic array | {gfem-test}`tests/verification/test_piezoelectric.py` | Every change |
| Reflection from a damping frame of four wavelengths or more | Design rule, within a factor of two ({py:class}`gfem.DampingFrame`) | {gfem-test}`tests/unit/test_piezoelectric.py` | Every change |
| Power taken equals power dissipated; no tensor; no loss; reciprocal, passive, rows sum to zero; rotated section | Identities of the discrete problem | {gfem-test}`tests/unit/test_piezoelectric_network.py` | Every change |
| Rotated stiffness and piezoelectric tensor of lithium niobate | Bulk wave speeds along the rotated axis | {gfem-test}`tests/unit/test_piezoelectric_material.py` | Every change |
| Gradients in every material property and in nodes (pitch, stretch, finger thickness, local maps); strain and displacement readings | Fourth-order central differences | {gfem-test}`tests/unit/test_piezoelectric_gradients.py` | Every change |

Observations, not checks: the transducer of the {doc}`tutorial
<tutorials/transducer>` at full size, against a measured device.

## Mechanical eigenmodes

Elastic modes of a 2D section ({doc}`recipes/mechanical-modes`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Anti-plane standing wave, shear horizontal modes of a strip, Lamb modes S0 and A0, Rayleigh wave of a deep block | Closed forms and Rayleigh-Lamb roots; error falls at least 100 times per halved edge | {gfem-test}`tests/verification/test_mechanical_eigenmode.py` | Every change |
| Uniform $Q_m$; lossless modes; $\omega(-q) = \omega(q)$; normalization to 1 J/m; group velocity | $\sqrt{1 + i/Q_m}$ scaling; identities; central difference of the band | {gfem-test}`tests/unit/test_mechanical_eigenmode.py` | Every change |
| Thickness-extensional plate of a 6mm crystal, open and grounded faces | Stiffened half-wave; root of $\tan X / X = 1/k_t^2$; pole of the admittance | {gfem-test}`tests/verification/test_mechanical_eigenmode_piezo.py` | Every change |
| Gradients of `omega` in stiffness, density, $Q_m$, piezoelectric tensor, permittivity, nodes | Fourth-order central differences | {gfem-test}`tests/unit/test_mechanical_eigenmode_gradients.py` | Every change |

(verification-acousto-optic)=
## Acousto-optic shift

{py:obj}`gfem.acousto_optic_shift` ({doc}`recipes/acousto-optic-shift`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Slab in uniform strain | Weighted sum of its permittivity change | {gfem-test}`tests/unit/test_acousto_optic.py` | Every change |
| Separate mode solve; device with crystal and sections turned | Same shift | {gfem-test}`tests/unit/test_acousto_optic.py` | Every change |
| Static potential as the field | {py:obj}`gfem.electro_optic_shift` | {gfem-test}`tests/unit/test_acousto_optic.py` | Every change |
| Photoelastic material | Tensor values | {gfem-test}`tests/unit/test_photoelastic_material.py` | Every change |
| Gradients in the photoelastic and electro-optic tensors, offset, node map, material | Fourth-order central differences | {gfem-test}`tests/unit/test_acousto_optic.py` | Every change |

Observations, not checks: the tutorial's transducer at full size against a
measured $V_\pi L$ ({doc}`tutorials/acousto-optic`).

(verification-optomechanics)=
## Optomechanical coupling

{py:obj}`gfem.optomechanical_coupling` and
{py:obj}`gfem.optomechanical_overlap`
({doc}`recipes/optomechanical-coupling`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Uniform dilation of a slab section, both polarizations | Closed-form derivative of the effective index; fivefold per halved edge | {gfem-test}`tests/unit/test_optomechanical_coupling.py` | Every change |
| Dilation of the whole section; linearity; moving boundary against node-map gradient; cladding-only displacement | $n_g - n_\mathrm{eff}$; identities | {gfem-test}`tests/unit/test_optomechanics.py` | Every change |
| Elliptic rod meshed separately for the mechanics | Same mesh cut from the optical one; gap falls with the square of the edge | {gfem-test}`tests/unit/test_optomechanical_coupling.py` | Every change |
| Gradients of mode fields and of the coupling | Fourth-order central differences | {gfem-test}`tests/unit/test_optomechanics.py` | Every change |
| Suspended silicon wire at 1.552 µm, Brillouin gain and frequency | Qiu et al., Opt. Express 21, 31402 (2013); Rakich et al., Phys. Rev. X 2, 011008 (2012) | {gfem-test}`tests/verification/test_optomechanical_wire.py` | Every change |

(verification-brillouin)=
## Brillouin gain

{py:obj}`gfem.brillouin_gain`, forward and backward, the peak gain
$G_B = Q_m\omega|\delta n_{ps}|^2/c^2$ of a mechanical mode on a guided mode
({doc}`recipes/brillouin-gain`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| TM slab film dilation, phase matched at $q = 2k_0 n_\mathrm{exact}$: moving boundary, photoelastic diagonal and shear parts | Closed form; error of each part falls at least 2.5-fold (5-fold for the moving boundary and diagonal part) per halved edge; shear part $O(h^2)$ | {gfem-test}`tests/unit/test_brillouin_overlap.py` | Every change |
| TE slab, $E_z = 0$ | Its forward closed form | {gfem-test}`tests/unit/test_brillouin_overlap.py` | Every change |
| Forward gain, gains of the overlap parts, real part of the pair weights, wavenumber outside the phase-matching window | $4Q_m\lvert\tilde g_0\rvert^2/(\hbar\omega v_g^2\,\mathrm{Re}\,\Omega)$ of the coupling to $10^{-13}$; $Q_m\omega\lvert\delta n\rvert^2/c^2$; forward weights; raises at both edges | {gfem-test}`tests/unit/test_brillouin.py` | Every change |
| Isotropic single material | $\mathrm{d}\ln G_B/\mathrm{d}\ln E = -1$ and $\mathrm{d}\ln G_B/\mathrm{d}\ln\rho = 0$ to $10^{-10}$ | {gfem-test}`tests/unit/test_brillouin.py` | Every change |
| Gradients in the photoelastic tensor, quality factor and mechanical nodes | Two-sample central difference, exact for the quadratic tensor ($10^{-12}$); fourth-order central differences of relative step $10^{-3}$ to $10^{-6}$ | {gfem-test}`tests/unit/test_brillouin.py` | Every change |
| Suspended silicon wire, 315 by 283.5 nm at 1.552 µm, backward modes E1 to E4 at $q = 2\beta$: frequencies, and the photoelastic, moving-boundary and total gains of E1, E2, E4 | Qiu et al., Opt. Express 21, 31402 (2013), Fig. 3(b); frequencies within 2 %, parts 10 %, totals 15 %; the moving boundary of E2 stays 7 % under the published bar on every mesh | {gfem-test}`tests/verification/test_brillouin_wire.py` | Every change |

Observation, not a check: the moving boundary of E2 is 7 % under the published
bar on every mesh and for either wavenumber. The figure read explains 3 % of
it; the rest is not attributed.

## Free space

Incident fields, dipoles and what is measured from them, at element order 3
({doc}`recipes/cross-sections`, {doc}`recipes/point-dipoles`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Lossy, lossless and conducting spheres, TE and TM at 30° | Mie series; no absorption if lossless; half on a symmetry plane | {gfem-test}`tests/verification/test_free_space_mie.py` | Every change |
| Same spheres on a refined mesh, bistatic radar cross section | Mie series | {gfem-test}`tests/verification/test_free_space_mie.py` | Before release |
| Wide Gaussian beam over its focal intensity | Plane wave's cross sections | {gfem-test}`tests/verification/test_free_space_mie.py` | Every change |
| Forward far-field amplitude | Optical theorem | {gfem-test}`tests/verification/test_far_field_mie.py` | Every change |
| Particle on a film over glass, lit past the critical angle | Extinction = absorption + scattering; side independence | {gfem-test}`tests/verification/test_layered.py` | Every change |
| Stack field: Fresnel, Airy, ground plane, normal flux | Closed forms; no scattered field without a particle | {gfem-test}`tests/unit/test_transfer_matrix.py` | Every change |
| Flat-interface reflection ({doc}`plane-waves-and-floquet-orders`) | $(n_1 - n_2)/(n_1 + n_2)$ in the tangential-E convention | {doc}`plane-waves-and-floquet-orders` | Every change |
| Gaussian beam | Maxwell's equations by finite differences; plane-wave superposition; 1 W through any plane | {gfem-test}`tests/unit/test_gaussian_beam.py` | Every change |
| Electric and magnetic dipoles in a homogeneous medium | Bulk emitted power; Purcell factor 1 | {gfem-test}`tests/verification/test_dipoles.py` | Every change |
| Dipoles above a perfect conductor and a magnetic wall; dipole in a two-mode guide | Image theory; modal closed form; powers add to the emitted power | {gfem-test}`tests/verification/test_dipoles.py` | Every change |
| Port waves of dipoles with `source_fields=False`, uniform and adaptive sweeps | Solved source fields; gradients | {gfem-test}`tests/verification/test_reciprocal_port_waves.py` | Every change |
| Hertzian dipole far field; crossed dipoles in quadrature | Closed-form pattern, directivity 1.5; circular polarization | {gfem-test}`tests/verification/test_far_field.py` | Every change |
| Fiber over a grating coupler; dipole into a Gaussian beam | Reciprocity | {gfem-test}`tests/verification/test_beam_coupling.py` | Every change |
| Dipole above a glass half-space | Sommerfeld reflection integrals | {gfem-test}`tests/reference/test_layered_dipole.py` | Before release |
| Purcell and beta factors, cross sections, port waves, radar cross section, directivity, beam coupling | Finite differences | {gfem-test}`tests/verification/test_free_space_gradient.py` | Every change |
| Power through a surface, far field, beam coupling by node coordinates | Finite differences | {gfem-test}`tests/verification/test_far_field_gradient.py` | Every change |
| Each refusal of a source or measurement | Raises its named error | {gfem-test}`tests/verification/test_free_space_gradient.py` | Every change |
| Plane wave, dipoles and every measurement on three ranks | One process | {gfem-test}`tests/mpi/test_free_space.py` | Twice a week |

## Optical forces

The force and torque of a solved field on a body ({doc}`recipes/optical-force`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Eigenmode at 1 J, translation and rotation | $-(1\,\mathrm{J}/\omega)\,\mathrm{d}\omega/\mathrm{d}\xi$ through a node map | {gfem-test}`tests/verification/test_optical_force.py` | Every change |
| TE101 mode of a perfectly conducting cavity on its end wall | $1\,\mathrm{J}/[a(1 + (a/d)^2)]$ | {gfem-test}`tests/verification/test_optical_force.py` | Every change |
| Lossless and lossy spheres under a plane wave; lossless cube | Mie series $Q_\mathrm{ext} - g\,Q_\mathrm{sca}$; pushed along the wave | {gfem-test}`tests/verification/test_optical_force.py` | Every change |
| Effective index through a node map moving one of two silicon slabs | Dispersion relation of two slabs, gaps of 100 and 300 nm | {gfem-test}`tests/verification/test_optical_force.py` | Every change |
| Each unsupported body, field or surrounding | Raises its named error | {gfem-test}`tests/unit/test_optical_force.py` | Every change |

## Multipoles

The multipole decomposition of a scatterer's field ({doc}`recipes/multipoles`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Dielectric, conducting and magnetic spheres under a plane wave | Mie series coefficients; partial cross sections sum to the total | {gfem-test}`tests/verification/test_multipoles.py` | Every change |
| Refined Kerker and anapole spheres | Mie series; toroidal correction $p \approx p_0 - ikT$ | {gfem-test}`tests/verification/test_multipoles.py` | Before release |
| Wide Gaussian beam; point dipole read back; moved and empty scatterer | Plane-wave multipoles; own moment; plane-wave phase | {gfem-test}`tests/unit/test_multipoles.py` | Every change |
| Gradients for a dipole, a quadrupole and a fraction | Finite differences | {gfem-test}`tests/verification/test_multipoles_gradient.py` | Every change |
| Coefficients, current moments and gradient on three ranks | One process | {gfem-test}`tests/mpi/test_multipoles.py` | Twice a week |
| Multipole formulas ({doc}`plane-waves-and-floquet-orders`) | Mie series | {doc}`plane-waves-and-floquet-orders` | Every change |

## Periodic cells

Bloch cells, bands and driven periodic solves ({doc}`recipes/periodic-cell`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Lossy parallel-plate line, periodic along x, two 50 Ω ports, $+k$, $-k$, 0 | Closed-form Bloch impedance matrix; direct and BDDC + GMRES | {gfem-test}`tests/verification/test_periodic_driven.py` | Every change |
| The same line loaded by its ports | Root of $\det(Z(f,k) + R) = 0$ | {gfem-test}`tests/verification/test_periodic_eigen.py` | Every change |
| Empty parallel-plate cell periodic along x and y; empty cube | Empty-lattice bands; light-line doublet | {gfem-test}`tests/verification/test_periodic_eigen.py` | Every change |
| Layered (Bragg) line, lossless and lossy | Transfer-matrix roots at the gap edges | {gfem-test}`tests/verification/test_periodic_eigen.py` | Every change |
| Empty square lattice along Γ-X-M-Γ; alumina rods | Empty-lattice values; converged 2D finite-element and plane-wave solutions | {gfem-test}`tests/verification/test_bands.py` | Every change |
| Zero wave vector; periodic against PMC walls; $S(k)^T = S(-k)$ ({term}`Bloch reciprocity`) | Plain solutions; identities | {gfem-test}`tests/unit/test_periodic_driven.py` | Every change |
| The same identities for eigenmode cells; Bloch operator refused | Plain solutions; named error | {gfem-test}`tests/unit/test_periodic_eigen.py` | Every change |
| Plain line on PETSc; SLEPc bands of the Bragg line | Direct solver; transfer-matrix roots | {gfem-test}`tests/verification/test_periodic_petsc.py` | Twice a week |

## Floquet ports

Two cells with Floquet ports on both open faces, against references that do
not use gfem ({doc}`recipes/floquet-ports`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Dielectric slab on a substrate, TE and TM, normal and conical incidence, three frequencies | Tangential-E transfer-matrix (Airy) form; efficiencies add to 1 | {gfem-test}`tests/verification/test_floquet_driven.py` | Every change |
| Slab between air at Brewster's angle | Zero TM reflection | {gfem-test}`tests/verification/test_floquet_driven.py` | Every change |
| Lamellar grating, every order, lossless and lossy, four incidences | Rigorous coupled-wave solution; RCWA absorption; power in the substrate | {gfem-test}`tests/verification/test_floquet_driven.py` | Every change |
| Open faces moved away with offsets grown by the same distance | Same S | {gfem-test}`tests/verification/test_floquet_driven.py` | Every change |
| Flat glass-air interface; cell closed by a PEC wall | $(n-1)/(n+1)$ in both polarizations ({doc}`plane-waves-and-floquet-orders`); total reflection | {gfem-test}`tests/unit/test_floquet_driven.py` | Every change |
| Grating S, zero wave vector, fixed-angle sweep, Rayleigh anomaly, handedness, BDDC + GMRES | Floquet {term}`Bloch reciprocity`; plain S; solves at each wave vector; continuity | {gfem-test}`tests/unit/test_floquet_driven.py` | Every change |
| Order labels and power-wave signs | Orders of each cell | {gfem-test}`tests/unit/test_floquet_orders.py` | Every change |

Gradients of the efficiencies are in the gradient matrix on
{doc}`capabilities`.

## Periodic surfaces

Cell outputs, angle sweeps, emissivity, cell responses and aperture fields
({doc}`recipes/diffraction-efficiency`, {doc}`recipes/emissivity`,
{doc}`recipes/metasurface-from-cells`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Jones matrices of slab cells at normal, conical and Brewster incidence | Airy entries; efficiencies as squared moduli | {gfem-test}`tests/verification/test_diffraction_cell_outputs.py` | Every change |
| Wave plate with a Fabry-Perot mismatch: Jones, circular, Mueller forms; Pancharatnam-Berry phase | Per-axis Airy form rotated by the axis angle; Stokes measurements; $\pm 2\psi$ | {gfem-test}`tests/verification/test_diffraction_cell_outputs.py` | Every change |
| PEC mirror, empty cell, power balance | $\mathrm{diag}(1,1,-1,-1)$; identity; sums to 1 | {gfem-test}`tests/unit/test_diffraction_cell_outputs.py` | Every change |
| Angle sweeps of the slab and the grating | Airy Jones matrix; RCWA at normal, 20° and conical incidence | {gfem-test}`tests/verification/test_angle_sweep.py` | Every change |
| Lossy slab on a PEC wall; slab between air and glass; hemispherical rule; Kirchhoff's law | $1 - \lvert r\rvert^2$; $1 - \lvert r\rvert^2 - \lvert t\rvert^2$; exact integral; absorptance from the opposite azimuth | {gfem-test}`tests/verification/test_emissivity.py` | Every change |
| Tabulated response; ridge cell with Hermite interpolation | Analytic function to rounding; held-out widths, order near four | {gfem-test}`tests/verification/test_cell_response.py` | Every change |
| Plane-wave propagation, lenses, blazed deflector | $e^{-ik_z d}$; stationary-phase and Rayleigh-Sommerfeld; grating equation | {gfem-test}`tests/unit/test_aperture.py` | Every change |
| Locally periodic approximation of a varied ridge cell | Supercell solve; four identical ridges give the single cell | {gfem-test}`tests/verification/test_lpa_supercell.py` | Every change |
| Figures of merit, emissivity, cell response, aperture field gradients, also under `jax.jit` | High-order central differences | {gfem-test}`tests/verification/test_diffraction_gradient.py` | Every change |

Gradients with respect to the incidence angle raise ({doc}`capabilities`).

## Patterned sheets

Copper and kinetic-inductance sheets ({doc}`recipes/sheet-design`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Sheets on a parallel-plate line, solid or patterned | BDDC + GMRES iterations of an empty sheet; direct S | {gfem-test}`tests/verification/test_sheet_gradient.py` | Every change |
| Copper pixel design, and the whole chain from pixels to exponential fill | Finite differences on both solvers | {gfem-test}`tests/verification/test_sheet_gradient.py` | Every change |
| Cavity with a resistive film of random fill | Independent iteration refactorizing every step | {gfem-test}`tests/verification/test_sheet_gradient.py` | Every change |

## Low frequency

The {term}`tree-cotree gauge` on a parallel-plate line between two 50 Ω lumped
ports, whose S has a closed form far below the mesh's low-frequency floor
(about 95 kHz).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Gauged direct solve at 10 kHz, 1 MHz and 100 MHz | ABCD closed form of the line | {gfem-test}`tests/unit/test_gauge_solve.py` | Every change |
| Same solve with `gauge="none"` at 10 kHz | Off from the closed form (the gauge is needed) | {gfem-test}`tests/unit/test_gauge_solve.py` | Every change |
| Adjoint gradient of $S_{11}$ in the permittivity at 10 kHz | Derivative of the closed form | {gfem-test}`tests/unit/test_gauge_solve.py` | Every change |
| Adaptive sweep through the gauge | Closed form below the floor; uniform gauged sweep above | {gfem-test}`tests/unit/test_gauge_adaptive.py` | Every change |
| Gauge basis | Tree-cotree construction | {gfem-test}`tests/unit/test_gauge_basis.py` | Every change |

Wave ports give the same S gauged and ungauged above the floor; below it their
mode problem is not gauged and the solve warns.

## Broadband models and time responses

({doc}`recipes/time-responses`, {doc}`recipes/rational-model`)

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Lossless 75 Ω line behind a 50 Ω port, ended by 30 Ω: step, impulse, windows, renormalization | Closed-form S and reflections | {gfem-test}`tests/unit/test_time_responses.py` | Every change |
| Exact rational S from known poles; lossless lines several wraps long; passivity enforcement | Same poles; tolerance; passive truth | {gfem-test}`tests/unit/test_rational_fit.py` | Every change |
| Closed-form step of a model | Inverse Laplace transform; windowed step of its own S | {gfem-test}`tests/unit/test_rational_step.py` | Every change |
| SPICE and Verilog-A netlists; AC solver of the cards | Model's S on every platform | {gfem-test}`tests/unit/test_netlist_export.py` | Every change |
| ngspice and VACASK runs of the netlists; transient | Model's S; closed-form ramp response | {gfem-test}`tests/unit/test_netlist_simulators.py` | Every change |
| Lossy parallel-plate line on wave ports | Complex $Z_{PV}$ reference; passive model from non-passive data | {gfem-test}`tests/unit/test_broadband_wave_ports.py` | Every change |
| Parallel-plate line of {doc}`get-started`, wave-port line 10 wraps long, dielectric insert above cutoff | Reflectometer profile; exact TEM line; mode matching | {gfem-test}`tests/verification/test_broadband_fem.py` | Before release |

## Eigenmodes with wave ports

Eigenmode problems with wave ports, against exact roots or a driven sweep
of the same model ({doc}`recipes/solve-a-device`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Rectangular guide shorted at one end, layers across the section, a wave port at the other | Roots of one transmission-line equation in complex frequency; tenfold per element order 2 to 4 | {gfem-test}`tests/verification/test_eigenmode_wave_ports.py` | Every change |
| Resonance above cutoff; port at each end; TE10 and TE20 of a shared surface; lossy anisotropic section; coaxial line | Closed-form roots | {gfem-test}`tests/verification/test_eigenmode_wave_ports.py` | Every change |
| Coupling $Q$ of one and two ports | $\sum_p 1/q_p = 1/Q$; exact powers out of each end | {gfem-test}`tests/verification/test_eigenmode_wave_ports.py` | Every change |
| Low-$Q$ resonance among high-$Q$ ones, each `n_modes` | Same root every time | {gfem-test}`tests/verification/test_eigenmode_wave_ports.py` | Every change |
| Requested power share, trapped modes, partly filled guide | Shares near one on exact resonances, near zero when trapped; warning | {gfem-test}`tests/unit/test_eigenmode_wave_ports.py` | Every change |
| Pole fitted to a driven sweep through the same ports | Eigenfrequency | {gfem-test}`tests/verification/test_eigenmode_wave_ports.py` | Every change |
| Line open at one end with a wave port at the other | Roots of $Z_\mathrm{in}(f) + Z_0 = 0$; `q` and `q_port` | {gfem-test}`tests/verification/test_eigenmode_wave_ports.py` | Every change |
| Eigenfrequency and port coupling $Q$ gradients | Finite differences | {gfem-test}`tests/unit/test_eigenmode_wave_ports.py` | Every change |
| Target below cutoff; unsupported combinations; neighbour's port reach | Named error; warning; other modes kept | {gfem-test}`tests/unit/test_eigenmode_wave_port_refusals.py` | Every change |
| Recipe's line | Its printed values | {gfem-test}`tests/unit/test_docs_solve_recipe.py` | Every change |

## Quantum metrics

Participations, dressed modes and interface layers
({doc}`recipes/quantum-metrics`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Junction participation, with and without port capacitance, beside a sheet | $-2\,\mathrm{d}\ln\omega/\mathrm{d}\ln L_J$ from two solves | {gfem-test}`tests/unit/test_eigenmode.py` | Every change |
| Port with an inductance and capacitance at the end of a line | Closed-form frequency of the loaded line | {gfem-test}`tests/unit/test_eigenmode.py` | Every change |
| Dressed modes as the junction phase shrinks; more Fock levels; transmon below its resonator | First-order closed forms; unchanged; negative shifts | {gfem-test}`tests/unit/test_quantum.py` | Every change |
| One and two modes of the pyEPR tutorials | Their stored outputs | {gfem-test}`tests/unit/test_quantum.py` | Every change |
| Gradient of the dressed modes | Fourth-order differences; direct differentiation of the eigenproblem | {gfem-test}`tests/unit/test_quantum.py` | Every change |
| Dispersive shift through a solve, small two-mode model | Finite differences | {gfem-test}`tests/verification/test_transmon_chi.py` | Every change |
| Dispersive-shift cells of the gradient matrix | Finite differences | {gfem-test}`tests/verification/test_gradient_matrix.py` | Every change |
| Dispersive shift of the tutorial's transmon along its pad gap | Finite differences | {gfem-test}`tests/verification/test_transmon_chi.py` | Before release |
| `"MA"` interface layer on a film in vacuum | Sum of its two faces read one at a time | {gfem-test}`tests/verification/test_interface_participation.py` | Every change |

Not checked: agreement with a measured device, and convergence of a sheet's or
an interface's participation at a film edge.

## Perfectly matched layers

Layers backed by a wall, read at a port against
$|R| = R_0^{\,n f \cos\theta / f_\text{ref}}$, at order 3
({doc}`recipes/perfectly-matched-layer`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| TEM line ended by a layer, one and two frequencies, pinned reference, background index | Target reflection; $R_0^{\,n}$ | {gfem-test}`tests/verification/test_pml.py` | Every change |
| TE10 guide ended by a layer, near cutoff | Reflection at $\cos\theta = \beta/k_0$ | {gfem-test}`tests/verification/test_pml.py` | Every change |
| Periodic cell over a layer at a Floquet port, TE and TM, three incidences | The formula; no cross-polarized reflection | {gfem-test}`tests/verification/test_pml.py` | Every change |
| Leaky dielectric guide radiating into a layer, plain and Bloch | Root of its dispersion relation; modes stay put | {gfem-test}`tests/verification/test_pml.py` | Every change |
| Strip guide through side layers between two wave ports | Own mode's propagation; symmetric passive S; wide closed section | {gfem-test}`tests/verification/test_pml.py` | Every change |
| Power absorbed; no layer unchanged; electrostatic and DC background; cache keys; port section cut by layers | Port balance; bitwise equal; identities | {gfem-test}`tests/unit/test_pml.py` | Every change |
| Each refusal; solve across ranks | Named error; one process | {gfem-test}`tests/unit/test_pml.py` | Every change |
| Gradients beside, inside and on a layer | Finite differences | {gfem-test}`tests/verification/test_pml_gradient.py` | Every change |

## Anisotropic materials

({doc}`recipes/anisotropic-materials`)

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Absorbing boundary ending a biaxial magnetic slab, both polarizations | Half-space of transmission-line theory | {gfem-test}`tests/unit/test_driven_boundaries.py` | Every change |
| WR-28 guide with anisotropic $\varepsilon$ and $\mu$, tensors turned about the axis | Closed-form propagation constant; no reflection | {gfem-test}`tests/verification/test_anisotropic_wave_ports.py` | Every change |
| Port mode of the turned tensor | Between the two principal-value guides | {gfem-test}`tests/unit/test_port_mode.py` | Every change |
| Loss along each principal axis ({term}`principal participation`) | $1/Q$ from the complex eigenfrequency | {gfem-test}`tests/unit/test_eigenmode.py` | Every change |
| Rigidly rotated model and material; oblique port normal | Same S, eigenfrequency, capacitance | {gfem-test}`tests/unit/test_anisotropic_rotation.py` | Every change |
| Uniform anisotropic material field on tilted axes | The anisotropic material it holds | {gfem-test}`tests/unit/test_driven.py` | Every change |

## Dispersive materials

Regions whose permittivity follows a {term}`permittivity law`
({doc}`recipes/dispersive-materials`).

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Law region, every term kind, beside absorbing boundaries | Constant material at the law's value per frequency | {gfem-test}`tests/unit/test_driven_dispersive.py` | Every change |
| Electrostatic problem with a law | {term}`Static permittivity <static permittivity>` | {gfem-test}`tests/unit/test_electrostatic_dispersive.py` | Every change |
| Box filled with a Lorentz or Drude medium | Closed-form polariton and plasma branches | {gfem-test}`tests/unit/test_eigenmode_dispersive.py` | Every change |
| Mode energy beside a dispersive region | Eigenproblem normalization; $-2\,\mathrm{d}\ln\omega/\mathrm{d}\ln L$ | {gfem-test}`tests/unit/test_mode_energy_dispersive.py` | Every change |
| Debye or Lorentz-filled guide behind wave ports | Closed-form propagation constant | {gfem-test}`tests/verification/test_dispersive_wave_ports.py` | Every change |
| Adaptive sweep across a resonance, Djordjevic-Sarkar substrate | Uniform sweep | {gfem-test}`tests/verification/test_dispersive_adaptive.py` | Every change |
| Material data: library laws | Their source's data or model; passive far beyond the {term}`fitted range` | {gfem-test}`tests/unit/test_library_materials.py`, {gfem-test}`tests/verification/test_library_sources.py` | Every change |
| Material data: fit to samples and to gold and aluminium tables | Recovered law; target error | {gfem-test}`tests/unit/test_fit_permittivity.py`, {gfem-test}`tests/verification/test_fit_permittivity_data.py` | Every change |
| Solve outside a law's fitted range | Warns once per problem type | {gfem-test}`tests/unit/test_fitted_range.py` | Every change |
| Law parameters on S, eigenfrequency, $Q$, capacitance | Finite differences | {gfem-test}`tests/unit/test_driven_law_gradient.py` | Every change |
| Density blend under index and linear interpolation | Finite differences | {gfem-test}`tests/unit/test_index_blend.py` | Every change |
| Gradient of the {doc}`tutorials/metal-bend` through gold | Finite differences | {gfem-test}`tests/verification/test_metal_bend.py` | Every change |
| Law and blend cells of the gradient matrix; their raises | Finite differences; named error | {gfem-test}`tests/verification/test_gradient_matrix.py`, {gfem-test}`tests/verification/test_gradient_matrix_raise_table.py` | Every change |
| Drude metal block in the field estimator | Effectivity range ([below](#verification-mesh-refinement)) | {gfem-test}`tests/verification/test_field_estimate.py` | Every change |
| S and its gradient by a law on two ranks; `"petsc"` cells | One process; `"direct"` | {gfem-test}`tests/mpi/test_dispersive.py` | Twice a week |
| Sign of a law's loss under $e^{+i\omega t}$ | Executed check on the page | {doc}`phasors-and-power-waves` | Every change |

Not checked: port modes of plasmonic guides (refused), absorbed power in a
dispersive region, agreement with measured metal data, eigenmodes inside a
crowded resonance band, convergence of an unresolved skin depth, library
values against sources other than their own, and the fit's repeatability
across platforms.

## Layer patterns and GDS

({doc}`recipes/fabricable-layout`)

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Disk, bar, area fractions | Circle, width, area and perimeter | {gfem-test}`tests/unit/test_pattern.py` | Every change |
| Width and spacing check on a dumbbell bridge, bars, notch, annulus, disk | True distances; independent layout checker's projection metric | {gfem-test}`tests/unit/test_feature_check.py` | Every change |
| GDS file read back | gdstk: units, cell, layers, datatypes, polygons, area per layer | {gfem-test}`tests/unit/test_gds.py` | Every change |
| Fabrication model moving a straight edge | Edge bias; second-order gradient checks | {gfem-test}`tests/unit/test_fabrication_model.py` | Every change |
| Bar or slot just under and over the rule | Constraint breaks and holds | {gfem-test}`tests/unit/test_length_scale.py` | Every change |
| 2D bend from arc, area fractions, $\beta = 32$ projection; three printed designs | Same pattern on finer pixels; finite difference | {gfem-test}`tests/verification/test_layer_patterns.py` | Every change |
| Toy design with and without constraints | Narrowest width and spacing against the rule | {gfem-test}`tests/verification/test_layer_patterns.py` | Every change |

(verification-mesh-refinement)=
## Mesh refinement error estimates

The {term}`error estimate` of {doc}`recipes/mesh-refinement` is checked by its
effectivity: the estimated error over the error against a known answer, which
must stay within a range set in the test for every mesh of every case. Each
kind of estimate (`ErrorEstimate.kind`) has its own range, generated from its
cases. Every range is read from the tests when the docs are built.

```{gfem-effectivity-table}
```

Over the capacitance, S and resonance cases, the quantity estimate was between
{gfem-effectivity}`quantity low` and {gfem-effectivity}`quantity high` times
the true error; the higher ratios came from fields with no sharp metal edge
and from layered cavities. The field estimate was between
{gfem-effectivity}`field low` and {gfem-effectivity}`field high` times the
true error. Some objectives are estimated far more pessimistically near their
target; the recipe's Limits quote those ratios.

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Square coaxial line with singular inner corners: capacitance over a refinement run | Constant extrapolated from fine uniform meshes; graded mesh | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| Circular coaxial line, faceted: capacitance on meshes of 1 to 8 times the base density | Higher-order solve on the same facets; below the true error against the exact circle | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| Stepped parallel-plate line: $S_{21}$ with lumped and wave ports | Constants from fine uniform and graded meshes | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| Parallel-plate line between 50 and 30 Ω ports, 3 to 25 GHz | Closed form | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| L-shaped cavity: first resonance over a refinement run | Published high-accuracy value | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| Rectangular cavity with metal walls: four modes | Closed form | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| One feature each: screened bars, interior metal sheet, anisotropic and magnetic fills, shared wave port surfaces, de-embedded ports, lossy fills and walls, resistive resonator | Closed forms; higher-order solves | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| Lossy waveguides: uniform, half filled by a lossy slab with hybrid ports ({term}`adjoint dual`), widening slab, with and without de-embedding | Higher-order solve; closed form for a dielectric strip | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| S through finite-conductivity plates, a deep metal ($\mathrm{Re}\,\varepsilon \approx -30$), permeability and loss contrasts, lossy magnetic and anisotropic fills | Closed forms; higher-order solves | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| Resonances with $\mathrm{Re}\,\varepsilon > 0$ laws, second-order absorbing walls, lossy magnetic fills | Closed forms; higher-order solves | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| $Q$, mean frequency of a degenerate cluster, absorbed power, field means, participations | Exact values; higher-order solves | {gfem-test}`tests/verification/test_objective_estimate.py` | Every change |
| Objectives of S (matched and nearly matched lines, stepped line, shared and lossy wave ports, deep metal); lossless rewrite with lumped ports | Closed forms; higher-order solves | {gfem-test}`tests/verification/test_objective_estimate.py` | Every change |
| Objectives of capacitance, energy, potential means, frequency and $Q$ | Exact values; higher-order solves | {gfem-test}`tests/verification/test_objective_estimate.py` | Every change |
| Field estimate, energy norm: coaxial lines, parallel-plate line, cavities, at orders 2 and 3 | Field's error against closed forms or higher-order solves | {gfem-test}`tests/verification/test_field_estimate.py` | Every change |
| Field estimate with loss: lossy fill, Drude block ($\mathrm{Re}\,\varepsilon < 0$), layered lossy cavity, lossy walls, absorbing walls | Closed forms; higher-order solves | {gfem-test}`tests/verification/test_field_estimate.py` | Every change |
| Line in two layers with lumped ports spanning a loss jump | Measured band: {gfem-effectivity}`lumped port across a loss jump low` to {gfem-effectivity}`lumped port across a loss jump high` × true error, not claimed | {gfem-test}`tests/verification/test_field_estimate.py` | Every change |
| Impedance walls carrying the field's current | Measured band: {gfem-effectivity}`impedance surface low` to {gfem-effectivity}`impedance surface high` | {gfem-test}`tests/verification/test_field_estimate.py` | Every change |
| Lumped port spanning a permeability contrast | Measured band: {gfem-effectivity}`lumped port across a permeability contrast low` to {gfem-effectivity}`lumped port across a permeability contrast high` | {gfem-test}`tests/verification/test_field_estimate.py` | Every change |
| Metal wedge with a law at $\mathrm{Re}\,\varepsilon = -1.5$ | Measured band: {gfem-effectivity}`plasmonic window low` to {gfem-effectivity}`plasmonic window high` | {gfem-test}`tests/verification/test_field_estimate.py` | Every change |
| S estimate: lumped port across a permittivity interface, cells stretched 10:1 | Measured band: {gfem-effectivity}`stretched cells at a lumped port across a permittivity contrast low` to {gfem-effectivity}`stretched cells at a lumped port across a permittivity contrast high` × S error | {gfem-test}`tests/verification/test_quantity_estimate.py` | Every change |
| Stepped line with no known field error; singular benchmark cases | Steady ratio to the $S_{21}$ error; smaller error than uniform splits at equal unknowns | {gfem-test}`tests/verification/test_field_estimate.py` | Every change |

Some references are extrapolated from finer meshes, with their uncertainty
below a fifth of every error checked.

## The p-multigrid solver

`linear_solver="pmg_gmres"` ({doc}`recipes/large-driven-model`) against the
direct solver, on models that need each part of its {term}`direct set`.

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Strip line at orders 2 and 3, each factorization of the direct set; two-frequency sweep; adjoint; `"auto"` at 50 MHz | Direct solver; fresh builds | {gfem-test}`tests/verification/test_pmg.py` | Every change |
| Strip guide cut by side layers | Direct solver; iterations without the layer left to the smoother | {gfem-test}`tests/verification/test_pmg.py` | Every change |
| Silicon guide with a 1 nm film on its faces, one cell across | Iterations of the guide without the film ({term}`poor cells <poor cell>` on the direct set) | {gfem-test}`tests/verification/test_pmg.py` | Every change |
| Silicon guide one and half a wavelength across, orders 2 and 3 | Direct solver | {gfem-test}`tests/verification/test_pmg.py` | Every change |
| Guide two and four wavelengths across | Same iterations; direct solver at two; reciprocity and passivity at four | {gfem-test}`tests/verification/test_pmg.py` | Before release |
| Coplanar metal lines at their reference frequencies | Reference S; fewer iterations than `"bddc_gmres"` | {gfem-test}`tests/reference/test_driven.py` | Before release |
| S gradients on lumped and wave ports, initial guess, beside a layer | Finite differences | {gfem-test}`tests/verification/test_pmg.py` | Every change |
| Memory estimate and `"auto"`; each refusal | The direct set the solver builds; named error | {gfem-test}`tests/unit/test_auto_solver.py` | Every change |
| Unit checks of the solver | Closed-form operators | {gfem-test}`tests/unit/test_pmg.py` | Every change |
| Guides of 2.9 and 6.6 million unknowns, half model with a layer of 2.1 million, bend of 17.2 million | Direct solver; its outgoing waves from an earlier run | {doc}`project/scaling` | Once |

(verification-petsc)=
## PETSc and SLEPc

| Case | Compared with | Test | Runs |
|---|---|---|---|
| `linear_solver="petsc"` with no `petsc_options` | `"direct"` | {gfem-test}`tests/unit/test_linalg_petsc.py` | Twice a week |
| SLEPc modes, $Q$, fields and gradients | ARPACK | {gfem-test}`tests/unit/test_eigenmode_slepc.py` | Twice a week |
| PETSc cells of the gradient matrix | Finite differences | {gfem-test}`tests/verification/test_gradient_matrix.py` | Twice a week |

These checks need the {term}`PETSc stack`; they also run on every change to
gfem's PETSc code.

## Several MPI ranks

A {term}`distributed solve` ({doc}`recipes/distributed-solve`): each solve has
a test launched on several ranks and compared with one process, with every
rank holding the same result, bit for bit.

| Case | Compared with | Test | Runs |
|---|---|---|---|
| Driven solves on `"bddc_gmres"`, same iteration count | One process | {gfem-test}`tests/mpi` | Every change |
| Driven solves on `"direct"`; every eigenmode solve on SLEPc | One process; frequencies, $Q$, fields | {gfem-test}`tests/mpi` | Twice a week |
| Periodic driven cells, pairs kept and swapped | One process | {gfem-test}`tests/mpi` | Twice a week |
| 10,248,856 DOF coplanar line, 1 to 32 nodes | Each solver's own one-process run; same iterations | {doc}`project/scaling` | Once |
| 47,277,960 DOF on 16 nodes | `bddc_gmres` against `direct` | {doc}`project/scaling` | Once |

Not tested on several ranks yet, running the same code without raising: a
lumped port terminated by an inductance or capacitance,
{py:obj}`gfem.characteristic_impedance`, `"petsc"` on an electrostatic,
eigenmode or periodic solve, and a periodic eigenmode solve on
`"bddc_gmres"`.

