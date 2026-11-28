# Conventions

Which sign, unit and normalization is behind a number gfem returns? The full
statement is the {py:obj}`gfem.conventions` module, the one place where each
convention is written out. This page gives one entry per convention, naming the
constant that records it; each entry ends with the theory page that derives it
and checks it on a solve. The device of those checks is the parallel-plate line
of {doc}`get-started`.

| Constant | Value | What it fixes |
| --- | --- | --- |
| `gfem.conventions.TIME_DEPENDENCE` | `exp(+iωt)` | the time dependence of every complex phasor |
| `gfem.conventions.PHASOR` | `peak` | a phasor's modulus is the amplitude of the oscillation, not its RMS value |
| `gfem.conventions.WAVES` | `pseudo` | S relates pseudo-waves (power waves on a real reference) |
| `gfem.conventions.INCIDENT_POWER` | 1 W | the incident power behind every driven field |
| `gfem.conventions.MODE_ENERGY` | 1 J | the electric energy every mode field stores |
| `gfem.conventions.BLOCH_PHASE` | `E(r + a) = exp(-i k·a) E(r)` | the phase across a periodic pair |
| `gfem.conventions.INTERIOR_SHEET_ADMITTANCE` | 0.5 | a thin `Conductor` film on an interior surface, in units of $\sigma t$ |
| `gfem.conventions.EXTERIOR_THICKNESS_FACTOR` | 2 | how often a `Conductor`'s thickness counts on an exterior surface |

## Units

Everything you pass to gfem and everything it returns is in SI units: metres,
hertz, ohms, farads, watts, V/m. The one exception is the mesh file, whose
length unit you state once when you read it; gfem converts the coordinates.
Temperatures are in kelvin.

| Quantity | Unit |
| --- | --- |
| temperature, temperature rise, `ambient` | K |
| `thermal_conductivity` | W/m/K |
| `mass_density` | kg/m³ |
| `specific_heat` | J/kg/K |
| `Convection` `coefficient` | W/m²/K |
| `HeatFlux` `flux` | W/m² |
| electrical `conductivity` | S/m |
| `thermo_optic_coefficient` | 1/K |
| heat (`SteadyThermalResult.heat`), `power_density` | W/m³ |
| time, `duration` | s |
| `heat_flow`, `dissipated_power`, `incident_power` | W |
| `Trace.heat_out` | J |
| on a cross-section: `power_per_length`, `resistance_per_length`, `capacitance_per_length` | W/m, Ω/m, F/m |
| on a cross-section: `heat_flow_per_length`, `incident_power_per_length` | W/m |
| on a cross-section: `Trace.heat_out_per_length` | J/m |
| `electro_optic_tensor` | m/V |
| `delta_n_eff_per_volt`, `phase_per_length_per_volt`, `v_pi_l` | 1/V, rad/(V m), V m |
| a port mode's `n_g`, `beta_2`, `beta_3`, `D` | 1, s²/m, s³/m, s/m² |

Derivation and a check on a solve: {ref}`Units <theory-phasors-and-power-waves-units>`.

## Time dependence

The physical field is $\mathrm{Re}(E\, e^{+i\omega t})$, the electrical engineering
convention ({py:data}`~gfem.conventions.TIME_DEPENDENCE`): write $j$ for $i$ and a circuit textbook's
formulas apply. A wave along $+x$ is $e^{-i\beta x}$, so the phase of a
transmission falls with distance and with frequency. A code that uses
$e^{-i\omega t}$ gives the complex conjugate of every S entry and every field.
Derivation and a check on a solve: {ref}`Time dependence <theory-phasors-and-power-waves-time-dependence>`.

## S-parameters and power

S relates {term}`pseudo-waves` ({py:data}`~gfem.conventions.WAVES`), normalized to each port's
{term}`reference impedance`; on a real reference they are the
{term}`power waves`, and $|a|^2$ is a power in W. Phasors are peak values
({py:data}`~gfem.conventions.PHASOR`), so a time-averaged power carries a factor $\tfrac{1}{2}$. A kept
driven field is the field for {py:data}`~gfem.conventions.INCIDENT_POWER`, 1 W, sent into the excited
port with every other port matched.
Derivation and a check on a solve: {ref}`S-parameters and power <theory-phasors-and-power-waves-s-parameters-and-power>`.

## SAX circuits

A solved S is a block of a [SAX](https://github.com/flaport/sax) circuit, and
you state its kind, `circuit="optical"` or `"rf"`: the wrong kind still
composes, with wrong interference. A photonic component hands SAX
$\overline{S}$, because SAX's photonic models are $e^{-i\omega t}$, with `wl` in
µm. An RF component hands S unchanged, renormalized to one real `z0` (50 Ω
by default), with `f` in Hz. Keys are `(input, output)`, so `sd["o1", "o2"]`
is $S_{21}$. The sign of a wave port's row and column follows its
{term}`polarity reference`, not the mesh. Recipe: {doc}`recipes/sax-circuit`.
Derivation and a check on a solve: {ref}`SAX circuits <theory-phasors-and-power-waves-sax-circuits>`.

## Sign of loss

Under $e^{+i\omega t}$ a lossy permittivity has a negative imaginary part,
$\varepsilon = \varepsilon' (1 - i \tan\delta)$, and a mode that decays has
$\mathrm{Im}\,\omega > 0$. Its quality factor is
$Q = |\omega| / (2 |\mathrm{Im}\,\omega|)$, with the {term}`loss tangent`
$\tan\delta$.
Derivation and a check on a solve: {ref}`Sign of loss <theory-phasors-and-power-waves-sign-of-loss>`.

## Dispersive materials

A {term}`permittivity law` follows the same sign of loss. Its parameters are
ordinary frequencies in hertz ($\omega = 2\pi f$), and with every strength
positive each term has $\mathrm{Im}\,\varepsilon \le 0$ at real frequencies.
Optics papers mostly fit under $e^{-i\omega t}$: keep the parameter values and
read $-i\gamma$ as $+i\gamma$. `law(0)` is the {term}`static permittivity` of the
electrostatic problem, and a mode's energy weighs a law region with the
{term}`energy permittivity`. A measured table gives $n$ and the
{term}`extinction coefficient` $k \ge 0$, so $\varepsilon = (n - ik)^2$, as
{py:obj}`gfem.fit_permittivity` takes them. Writing a law:
{doc}`recipes/dispersive-materials`.
Derivation and a check on a solve: {ref}`Dispersive materials <theory-phasors-and-power-waves-dispersive-materials>`.

## Eigenmode fields

A mode has no source to set its amplitude, so gfem scales every mode field to
store {py:data}`~gfem.conventions.MODE_ENERGY`, 1 J, of electric energy: the volume term
$\tfrac{1}{2} \mathrm{Re} \int \mathbf{D}^* \cdot \mathbf{E}\, dV$ plus the
capacitive terms of lumped ports and {py:obj}`gfem.Impedance` sheets. For a
lossless mode that is its total energy, and participation ratios are shares
of it.
Derivation and a check on a solve: {ref}`Eigenmode fields <theory-phasors-and-power-waves-eigenmode-fields>`.

## Periodic cells

Across a periodic pair of translation $\mathbf{a}$, a Bloch wave vector
$\mathbf{k}$ sets {py:data}`~gfem.conventions.BLOCH_PHASE`: `E(r + a) = exp(-i k·a) E(r)`, so the phase
travels along $+\mathbf{k}$ ({doc}`recipes/periodic-cell`). Wave vectors are
Cartesian, in rad/m. The named points of a {py:func}`gfem.wave_vector_path` are
in reduced coordinates of `Model.reciprocal_lattice`,
$\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi\delta_{ij}$. The band group velocity
$\nabla_{\mathbf{k}}\omega$ points the way energy travels, with no sign flip.
Bands are numbered from 0, the lowest.

## Floquet ports

A {py:obj}`gfem.FloquetPort` is one diffraction order in one polarization of
a plane wave on the open face of a periodic cell
({doc}`recipes/floquet-ports`). The face's frame has $\hat{\mathbf{e}}_1$ along the
first translation and $\hat{\mathbf{e}}_3$ along $\mathbf{a}_1 \times \mathbf{a}_2$.
TE is $\mathbf{E}$ normal to the plane of incidence and TM is $\mathbf{E}$ in it.
$|S|^2$ is the {term}`diffraction efficiency`. A flat interface reflects
$(n_1 - n_2) / (n_1 + n_2)$, the opposite sign of the Fresnel $r_p$ of many
optics texts. Right-handed circular light (IEEE) from the top face is
$(S[:, \mathrm{TM}] + i\, S[:, \mathrm{TE}]) / \sqrt{2}$.
Derivation and a check on a solve: {ref}`Floquet ports <theory-plane-waves-and-floquet-orders-floquet-ports>`.

## Aperture fields

`gfem.diffraction.aperture_field` samples a metasurface once per cell. The
cells of an axis of $n_1$ cells sit at $x_j = (j - \tfrac{n_1-1}{2})\,a_1$,
measured from the aperture centre along $\hat{\mathbf{e}}_1$ with the pitch $a_1$,
and likewise along $\hat{\mathbf{e}}_2$. The values are power waves relative to
the incident plane wave, with components along $\hat{\mathbf{e}}_1$ (TM) and
$\hat{\mathbf{e}}_2$ (TE).
Derivation and a check on a solve: {ref}`Aperture fields <theory-plane-waves-and-floquet-orders-aperture-fields>`.

## Temperature and thermo-optic phase

Temperatures are absolute, in kelvin, and a temperature at or below 0 K raises.
The rise is the temperature minus the {term}`source-free temperature`, the one
the boundaries alone set. {py:obj}`gfem.heat_flow` is positive out. On a
cross-section every extensive quantity is per metre of guide. A material's
`thermo_optic_coefficient` $dn/dT$ acts on the rise, about the index of its
`eps_r`; no reference temperature is stored. The phase a positive rise adds is
a delay, so a transmission's argument moves by minus it. Recipes:
{doc}`recipes/heater-temperature`, {doc}`recipes/thermal-step-response` and
{doc}`recipes/thermo-optic-phase`.
Derivation and a check on a solve: {ref}`Temperature and thermo-optic phase <theory-multiphysics-couplings-temperature-and-thermo-optic-phase>`.

## Electro-optic shift

A crystal's `electro_optic_tensor` is the contracted $r_{Ik}$ in m/V, 6 × 3:
rows $xx, yy, zz, yz, xz, xy$, columns $x, y, z$, in the frame of the material's
`axes` (a right-handed set). A static field $E_\mathrm{s} = -\nabla V$, per volt
of one terminal, changes the permittivity by
$\Delta\varepsilon_r = -\varepsilon_r (r \cdot E_\mathrm{s}) \varepsilon_r$, and the
phase per volt is a delay, like the thermo-optic one. Solve each arm's mode on
an optical window that reaches the electrodes' inner edges.
{doc}`recipes/electro-optic-phase`.
Derivation and a check on a solve: {ref}`Electro-optic shift <theory-multiphysics-couplings-electro-optic-shift>`.

## Photoelastic tensor

A crystal's `photoelastic_tensor` is the contracted $p^E_{IJ}$, a
dimensionless 6 × 6 array in rows and columns $xx, yy, zz, yz, xz, xy$, in the
frame of the material's `axes`, with the engineering shear strain:
$\Delta(\varepsilon_r^{-1})_I = \sum_J p_{IJ} S_J$. It is the tensor at constant
electric field. The shift is a phasor of the drive at $e^{+i\Omega t}$
({doc}`recipes/acousto-optic-shift`).
Derivation and a check on a solve: {ref}`Photoelastic tensor <theory-multiphysics-couplings-photoelastic-tensor>`.

## Optomechanical coupling

{py:obj}`gfem.optomechanical_coupling` gives $g_0$ in rad/s (and $\tilde g_0$,
per root length, in rad s$^{-1}$ m$^{1/2}$): divide by $2\pi$ for hertz. The sign
is that of $-\partial\omega_c/\partial\alpha$, positive when the displacement
raises the effective index, and follows the mechanical mode's phase (largest
displacement coefficient real and positive), so read $|g_0|$.
{doc}`recipes/optomechanical-coupling`.
Derivation and a check on a solve: {ref}`Optomechanical coupling <theory-multiphysics-couplings-optomechanical-coupling>`.

## Brillouin gain

{py:obj}`gfem.brillouin_gain` gives the peak gain
$G_\mathrm{B} = Q_m\omega\,|\delta n_{ps}|^2/c^2$ in 1/(W m), with
$\pm\mathrm{d}P_s/\mathrm{d}z = G_\mathrm{B}P_pP_s$ (the sign along the Stokes
direction). The pump has 1 W, the mechanical mode 1 J/m, and $\omega$ is the
pump's. $Q_m = \mathrm{Re}\,\Omega/(2\,\mathrm{Im}\,\Omega)$ and the
linewidth, full width at half maximum, is $2\,\mathrm{Im}\,\Omega$. The phonon
travels with the pump, $q = 2\beta\,(\hat n\cdot\hat z)$, with $\hat n$ the
port's normal; the shift $-\Omega n_g/c$ of the Stokes wave is dropped, and a
$q$ within $10^{-2}$ of $2\beta$ (or of 0, forward) counts as phase matched.
The gains of the two parts are not additive: their amplitudes add with sign.
{doc}`recipes/brillouin-gain` has the whole calculation.

## Piezoelectric problems

A {py:class}`gfem.Piezoelectric` problem is on a 2D section, $\partial_z = 0$,
with three displacement components. `stiffness` is $c^E$ in Pa (6 × 6),
`piezoelectric_tensor` is $e$ in C/m² (3 × 6, stress constants) and `eps_r` is
$\varepsilon^S / \varepsilon_0$, all in the frame of the material's `axes` (a
right-handed set). Strain is in Voigt order $xx, yy, zz, yz, xz, xy$ with
engineering shear. Loss is a complex stiffness $c^E (1 + i/Q_m)$: its imaginary
part is positive, the opposite of a permittivity's. Totals are per metre along
$z$, and `admittance_per_length` is indexed `[frequency, i, j]`, response first.
Derivation and a check on a solve: {ref}`Piezoelectric problems <theory-multiphysics-couplings-piezoelectric-problems>`.

## Mechanical eigenmodes

A {py:class}`gfem.MechanicalEigenmode` mode has the form
$u(x, y)\,e^{i\omega t - iqz}$, with $q$ the {term}`axial wavenumber` in rad/m: a
mode with $q > 0$ travels along $+z$ where its group velocity is positive. A 3D
body has $u(x, y, z)\,e^{i\omega t}$, wavenumber 0 and a NaN group velocity. The
loss is the piezoelectric problem's, so a decaying mode has
$\mathrm{Im}\,\omega > 0$ and $Q = \mathrm{Re}\,\omega / (2\,\mathrm{Im}\,\omega)$.
Each mode carries 1 J per metre along $z$ (1 J for a 3D body), with its largest
coefficient real and positive. With `pml=`, $Q$ is the total quality factor,
the radiation into the layer plus $Q_m$.
Derivation and a check on a solve: {ref}`Mechanical eigenmodes <theory-multiphysics-couplings-mechanical-eigenmodes>`.

## Group index and dispersion

A port mode's group index and dispersion are frequency derivatives of
$\gamma / i$, with $\omega = 2\pi f$ and $\lambda = c/f$:
$n_g = \frac{c}{i}\frac{\mathrm{d}\gamma}{\mathrm{d}\omega}$,
$\beta_k = \frac{1}{i}\frac{\mathrm{d}^k\gamma}{\mathrm{d}\omega^k}$ and
$D = -\frac{2\pi c}{\lambda^2}\,\beta_2$. For a lossless mode they are real and
the same under either time dependence; with loss they are complex, and a code
that uses $e^{-i\omega t}$ reports their conjugates. $\beta_2 > 0$, so $D < 0$,
is normal dispersion. Papers quote other units: 1 ps/(nm km) is
$10^{-6}$ s/m², 1 ps²/km is $10^{-27}$ s²/m and 1 ps³/km is $10^{-39}$ s³/m. What
they mean: {doc}`modes-and-ports`; the calculation:
{doc}`recipes/waveguide-dispersion`.

On a bent port mode, $\gamma$ and these derivatives are per unit length of the
reference arc of radius $R$, at the guide's centre line unless `at` says
otherwise; `outward` points from the bend's axis through the guide. The loss
per 90° is $(10\pi/\ln 10)\,R\,\mathrm{Re}\,\gamma$ in dB, with the sign of
loss above. {doc}`modes-and-ports`.

## Kerr nonlinearity

`nonlinear_index` $n_2$ (m²/W) and `two_photon_absorption`
$\beta_\mathrm{TPA}$ (m/W) combine in $\tilde n_2 = n_2 - i\beta_\mathrm{TPA}/(2k_0)$.
$\gamma_\mathrm{NL}$ is reported in the convention of pulse-propagation tools,
$e^{-i\omega t}$, so $\mathrm{Im}\,\gamma_\mathrm{NL} \ge 0$ is the loss of two-photon
absorption and its sign does not follow `n_eff`'s $n' - in''$. It is in
1/(W m) on a 3D section and 1/W on a line port of a 2D model.
{doc}`recipes/waveguide-nonlinearity`.
Derivation and a check on a solve: {ref}`Kerr nonlinearity <theory-nonlinear-and-quantum-optics-kerr-nonlinearity>`.
On a bent port mode the integral runs over the physical field, with the
volume element weighted by $h = 1 + \xi/R$, and is per unit length of the
reference arc. {doc}`recipes/waveguide-bends`.

## Raman gain

`gfem.raman_gain` gives the real modal gain $g_{ab}$ in 1/(W m) between waves
$a$ and $b$ of one port, at 1 W each: $\mathrm{d}P_a/\mathrm{d}z \ni g_{ab}P_bP_a$,
positive when $a$ lies below $b$ in frequency. It does not depend on the phasor
convention, and $g_{ba} = -(\omega_b/\omega_a)\,g_{ab}$ (Manley-Rowe).
{doc}`recipes/raman-gain`.

## Free carriers

`gfem.free_carriers` gives the confinement $\Gamma_R$ of a mode of 1 W in the
carrier regions and the pair generation $\theta$ of
$\mathrm{d}N/\mathrm{d}t = \theta P^2 - N/\tau$, in 1/(W² m³ s). A density $N$
changes $n_\mathrm{eff}$ by $\Gamma_R(\Delta n_\mathrm{FC} - i\sigma N/(2k_0))$, in
the $e^{-i\omega t}$ convention of $\gamma_\mathrm{NL}$.
{doc}`recipes/free-carriers`.

## Second-order nonlinearity

`second_order_nonlinear_tensor` is the contracted $d_{il} = d_{ijk}$ in m/V, with
$d = \chi^{(2)}/2$: rows $x, y, z$, columns $xx, yy, zz, yz, xz, xy$ with no
factor 2 on the shear ones, in the frame of the material's `axes`. A domain of
reversed polarization carries $-d$. The coupling $\kappa_{abc}$ is reported in
the convention $e^{-i\omega t}$, the complex conjugate of the integral in gfem's
phasors, in 1/(W$^{1/2}$ m), and $\eta_\mathrm{norm} = |\kappa|^2$ in 1/(W m²).
{doc}`recipes/second-harmonic-generation`. The polarization at the generated
frequency drives a solve as the current $J = +i\omega_3 P$ in gfem's
$e^{+i\omega t}$ phasors ({doc}`recipes/metasurface-second-harmonic`).
Derivation and a check on a solve: {ref}`Second-order nonlinearity <theory-nonlinear-and-quantum-optics-second-order-nonlinearity>`.

## Photon pairs

The pump amplitude is in the published $e^{-i\omega t}$ convention. A density $n$
is in pairs per second and hertz, and `gfem.JointSpectrum.jsa` is $f = 2\pi F$ in
1/Hz, so $\sum|f|^2\,\delta f_s\,\delta f_i$ is the number of pairs per pulse. When
one mode family carries both photons the count is $\tfrac{1}{2}\iint|F|^2$ over the
whole plane. The bands of a source act as ideal filters. Recipe:
{doc}`recipes/photon-pair-sources`.
Derivation and a check on a solve: {ref}`Photon pairs <theory-nonlinear-and-quantum-optics-photon-pairs>`.

## Thin metals

A {py:obj}`gfem.Conductor` with a thickness describes one face of a film. On
a surface inside the mesh a film much thinner than its skin depth is
therefore a sheet of admittance {py:data}`~gfem.conventions.INTERIOR_SHEET_ADMITTANCE` times $\sigma t$, half
the film's conductance: give `thickness=2 * t` for the whole film. On the
outer surface of the mesh the thickness already counts
{py:data}`~gfem.conventions.EXTERIOR_THICKNESS_FACTOR` times, and the film is $\sigma t$.

## 2D models

On a {term}`2D model` the fields do not vary along z, and every extensive
quantity is per metre along z. The rows that change:

| Quantity | On a 3D model | On a 2D model |
| --- | --- | --- |
| power fed into an excited port (`INCIDENT_POWER`) | 1 W | 1 W/m |
| kept driven field | $\mathbf{E}$ in V/m | $E_z$ in V/m (Ez) or $H_z$ in A/m (Hz), for 1 W/m |
| S-parameters | dimensionless | dimensionless, and the same |
| `Field.at` | (m, 3) points, (m, 3) values | (m, 2) points, (m,) values |
| kept piezoelectric fields | | $u_x, u_y, u_z$ in m and $\varphi$ in V, per volt on the terminal |

S is a ratio of powers, so a {term}`slab <Slab model>` of thickness $h$ between
plates and the 2D model of its plane give the same S; their fields relate by
$E^{\mathrm{2D}} = \sqrt{h}\,E^{\mathrm{slab}}$.

## Incident fields

A {py:obj}`gfem.PlaneWave` or {py:obj}`gfem.GaussianBeam` is given by its
propagation direction in spherical angles from $+z$, in rad:
$\hat{\mathbf{k}} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$. A
wave with $\theta < \pi/2$ travels up, one with $\theta > \pi/2$ down, so a
wave arriving 30° off the normal from above is `theta=np.pi - np.radians(30)`.

| Polarization | Field direction | Also called |
| --- | --- | --- |
| `"TM"` | $\hat{\boldsymbol{\theta}} = (\cos\theta\cos\phi, \cos\theta\sin\phi, -\sin\theta)$ | p |
| `"TE"` | $\hat{\boldsymbol{\phi}} = (-\sin\phi, \cos\phi, 0)$ | s |
| `"RCP"` | $(\hat{\boldsymbol{\theta}} - i\hat{\boldsymbol{\phi}})/\sqrt{2}$ | right-handed (IEEE) |
| `"LCP"` | $(\hat{\boldsymbol{\theta}} + i\hat{\boldsymbol{\phi}})/\sqrt{2}$ | left-handed (IEEE) |
| $(a_\mathrm{TM}, a_\mathrm{TE})$ | $(a_\mathrm{TM}\hat{\boldsymbol{\theta}} + a_\mathrm{TE}\hat{\boldsymbol{\phi}})/\lVert a \rVert$ | Jones pair |

$(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}}, \hat{\mathbf{k}})$ is
right-handed. TE is the TE reference of a Floquet port in every direction; TM
points along the Floquet TM reference for a wave going up and against it for
a wave going down. Under `TIME_DEPENDENCE`, the real field of RCP turns from
$\hat{\boldsymbol{\theta}}$ to $\hat{\boldsymbol{\phi}}$ with the thumb along
$\hat{\mathbf{k}}$.
So for a wave going down the Floquet circular unit vector
$(\hat{\mathbf{E}}_\mathrm{TM} + i\hat{\mathbf{E}}_\mathrm{TE})/\sqrt2$ of the RCP column is
$-(\hat{\boldsymbol\theta} - i\hat{\boldsymbol\phi})/\sqrt2$: the same right-handed
wave as `"RCP"`, with the opposite phase. A plane wave's `amplitude` is a peak
value in V/m (`PHASOR`).
Derivation and a check on a solve: {ref}`Incident fields <theory-plane-waves-and-floquet-orders-incident-fields>`.

## Point dipoles

A {py:obj}`gfem.ElectricDipole` of moment $p$ (C·m) at $\mathbf{r}_0$ is the
current $\mathbf{J} = i\omega \mathbf{p}\,\delta(\mathbf{r} - \mathbf{r}_0)$;
a {py:obj}`gfem.MagneticDipole` of moment $m$ (A·m²) the magnetic current
$\mathbf{M} = i\omega\mu_0 \mathbf{m}\,\delta(\mathbf{r} - \mathbf{r}_0)$.
Both moments are peak phasors, the same at every frequency, and an antenna's
current moment $Il$ (A·m) is $p = Il/(i\omega)$. The
{term}`emitted power <Emitted power>`
$P = -\tfrac{\omega}{2}\operatorname{Im}(\mathbf{p}^*\cdot\mathbf{E}(\mathbf{r}_0))$
is positive for a dipole that radiates; the Purcell factor divides it by the
power of the same dipole in the bulk material at $\mathbf{r}_0$.

## Far fields

{py:obj}`gfem.far_field` gives the {term}`far-field amplitude <Far-field amplitude>`
$\mathbf{F}(\hat{\mathbf{r}}) = \lim_{r\to\infty} r\,e^{+ikr}\mathbf{E}(r\hat{\mathbf{r}})$
in V: under `TIME_DEPENDENCE` an outgoing wave is $e^{-ikr}/r$, with the phase
referred to the model origin and $k$, $\eta$ those of the
{term}`exterior medium <Exterior medium>`. The direction takes the angles of an
incident field's $\hat{\mathbf{k}}$; `e_right` and `e_left` are
$(F_\theta \pm iF_\phi)/\sqrt{2}$. The radiation intensity is
$U = |\mathbf{F}|^2/(2\eta)$ in W/sr (`PHASOR`), and directivity, gain and
realized gain divide $4\pi U$ by the radiated, accepted and incident power
(IEEE 145); a port column is fed `INCIDENT_POWER`.

## Multipoles

`gfem.Multipoles` expands the scattered field about an origin in outgoing
vector spherical waves, with Condon-Shortley $Y_{lm}$, Jackson's $X_{lm}$ and
$h^{(2)}_l$ (outgoing under `TIME_DEPENDENCE`). The coefficients are
dimensionless and power-normalized: the partial scattering cross section of the
electric (magnetic) multipole of degree $l$ is $k^{-2}\sum_m|\alpha_{lm}|^2$
($k^{-2}\sum_m|\beta_{lm}|^2$), in m². They are stored $l$-major, $m$ ascending,
flat index $j = l^2 + l + m - 1$.
Derivation and a check on a solve: {ref}`Multipoles <theory-plane-waves-and-floquet-orders-multipoles>`.

## Optical force

{py:obj}`gfem.optical_force` is the time-averaged force and torque on a body,
from peak phasors: the derivative of $L = \tfrac14\int(\varepsilon|E|^2 -
|B|^2/\mu)\,dV$ of the solved field under a rigid motion of the body. A driven
field is per solved excitation, so the force in N scales with the incident
power; an eigenmode field is at 1 J, and its force is
$-(1\,\mathrm{J}/\omega)\,\mathrm{d}\omega/\mathrm{d}\xi$. The torque is about the
body's volume centroid unless `origin=` says otherwise, in N m.

## Port waves and beam coupling

A source column's `port_waves` and {py:obj}`gfem.beam_coupling` are
amplitudes in √W, like S for a port fed `INCIDENT_POWER`. On a real reference
$|b|^2$ is the power a source sends into the port and $|c|^2$ the power a
port's field couples into a beam; with a voltage path or on a lossy guide both
are pseudo-waves, like S (`WAVES`). The {term}`beam coupling <Beam coupling>`
$c$ refers to the beam scaled to 1 W, with the beam's own phase.

## Time responses

`NetworkParameters.step_response` and `impulse_response` turn S into
{term}`time responses <time response>` under the same `TIME_DEPENDENCE`, so
positive time is later: $h(t) = \int S(f)\, e^{+2\pi i f t}\, df$, the plain
inverse transform, with no conjugation. On a {term}`harmonic grid`
$f_k = k\,\Delta f$ the record lasts $1/\Delta f$, negative times included. An
isolated reflection $\Gamma$ peaks at $\Gamma$, and the step ends at $S(0)$. Data
from a code that writes $e^{-i\omega t}$ are the complex conjugate of gfem's:
conjugate them first. Recipe: {doc}`recipes/time-responses`.
Derivation and a check on a solve: {ref}`Time responses <theory-phasors-and-power-waves-time-responses>`.
