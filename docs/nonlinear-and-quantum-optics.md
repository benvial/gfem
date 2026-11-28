# Nonlinear and quantum formulas

Which sign, unit and normalization does a nonlinear coefficient, a three-wave coupling or a photon-pair figure carry, and what are their closed forms?

The nonlinear coefficients are reported in the convention of pulse-propagation
and coupled-mode tools, $e^{-i\omega t}$, which is the complex conjugate of
gfem's own phasors ({doc}`conventions`). The rules are one entry each there;
this page gives the closed forms. The modes they integrate are port modes
({doc}`modes-and-ports`).

(theory-nonlinear-and-quantum-optics-kerr-nonlinearity)=
## Kerr nonlinearity

A material's `nonlinear_index` $n_2$ (m²/W) and `two_photon_absorption`
$\beta_\mathrm{TPA}$ (m/W) make its permittivity depend on the intensity,
$n = n_0 + n_2 I$ and $\alpha_I = \beta_\mathrm{TPA} I$ in the intensity
$I$. They combine in a complex index
$\tilde n_2 = n_2 - i\beta_\mathrm{TPA}/(2k_0)$, with $k_0 = \omega/c$. The
nonlinear coefficient of a port mode $E$ (all three components, normalized
to 1 W; $n^2 = \mathrm{Re}\,\varepsilon_r$ of each element at the frequency
of the row) is

$$
\gamma_\mathrm{NL} = \frac{k_0 c^2 \varepsilon_0^2}{12} \int_S n^2\,\tilde n_2
\left[2|E|^4 + |E\cdot E|^2\right] \mathrm{d}A,
$$

with $E\cdot E$ the product without conjugation. $\gamma_\mathrm{NL}$ is
reported in the convention of pulse-propagation tools, $e^{-i\omega t}$: the
envelope $A$ of power $P = |A|^2$ obeys $\mathrm{d}A/\mathrm{d}z =
i\gamma_\mathrm{NL}|A|^2 A$, the propagation constant moves by
$\delta\beta = \mathrm{Re}\,\gamma_\mathrm{NL}\,P$ and the field attenuation
by $\delta\alpha = \mathrm{Im}\,\gamma_\mathrm{NL}\,P \ge 0$, the loss of
two-photon absorption: $\mathrm{d}P/\mathrm{d}z = -2\,\mathrm{Im}\,
\gamma_\mathrm{NL}\,P^2$. It is the complex conjugate of the same integral in
gfem's own $e^{+i\omega t}$ phasors, so the sign of its imaginary part does
not follow `n_eff`'s $n' - in''$.

Between several waves $a, b, c, d$ (rows of port mode results) the
tensor $\gamma_{abcd}$ is the coefficient of $A_b A_c \overline{A_d}$ in the
equation of $A_a$, summed over every index tuple with no degeneracy factor,
so the factor 2 of cross-phase modulation and of four-wave mixing comes out
of the sum; $\gamma_{aaaa} = \gamma_\mathrm{NL}$ of the same row.

Units: $\gamma_\mathrm{NL}$ is in 1/(W m) on a 3D section. On a line port of
a 2D model the field is for 1 W/m and the integral runs along the line, so it
is in 1/W. Only the $E_z$ polarization of a line port has it. The
isotropic scalar $n_2$ is real and does not disperse over the grid; the
formula holds while the attenuation is small against the phase constant.
The {py:obj}`gfem.conventions` module, "Kerr nonlinearity", gives the same
formulas, and {doc}`recipes/waveguide-nonlinearity` the calculation.

A microring of radius $R$ made of the straight guide takes the group index
$n_g$ and $\beta_2$ of the straight section. A one-dimensional estimate of the
bent guide with constant indices bounds that approximation (an estimate, not a
result of gfem, which has no bent port mode):

| Guide | $R$ | $\Delta n_g / n_g$ | $\beta_2$ straight → bent (ps²/m) |
| --- | --- | --- | --- |
| Si 500 × 220 nm | 10 µm | $+1.4 \times 10^{-3}$ | −0.118 → −0.104 |
| Si 500 × 220 nm | 5 µm | $+5.4 \times 10^{-3}$ | −0.118 → −0.063 |
| SiN 1.6 µm | 100 µm | $+4.5 \times 10^{-4}$ | −0.062 → −0.059 |
| SiN 1.6 µm | 23 µm | $+8 \times 10^{-3}$ | −0.062 → −0.007 |

The free spectral range $D_1$ of the straight guide is good to about
$10^{-3}$; $D_2$ is good to 5 to 15 % for a nitride ring of 100 µm or a silicon
ring of 10 µm and useless for a ring of 1 THz free spectral range. The bend
also hybridizes modes, which a straight solve cannot see. The Kerr response
here is the instantaneous electronic one with two-photon absorption: the
delayed Raman response and the free carriers of two-photon absorption are in
{doc}`recipes/raman-gain` and {doc}`recipes/free-carriers`.

The {term}`clearance` $c$ from the core to a side
{term}`perfectly matched layer` perturbs $n_\text{eff}$ by about
$10^{-2} e^{-2\gamma_t c}$, with $\gamma_t$ the mode's transverse decay rate,
and the {term}`group-velocity dispersion` far more.

(theory-nonlinear-and-quantum-optics-second-order-nonlinearity)=
## Second-order nonlinearity

A material's `second_order_nonlinear_tensor` is the contracted tensor
$d_{il} = d_{ijk}$ (m/V), with $d = \chi^{(2)}/2$ (the convention of data
sheets, Boyd and Roberts: half the susceptibility). Rows are the polarization
$x, y, z$, columns the field pair $xx, yy, zz, yz, xz, xy$ with no factor 2 on
the shear ones, in the frame of the material's `axes`. For a field $F$,
$\sum_{jk} d_{ijk}F_jF_k = \sum_l d_{il}V_l$ with $V = (F_x^2, F_y^2, F_z^2,
2F_yF_z, 2F_xF_z, 2F_xF_y)$. A domain of reversed polarization carries $-d$.
The tensor is real, constant over the grid, and read by no linear solve.

The coupling between the port modes $a, b, c$ of three waves with
$\omega_a = \omega_b + \omega_c$ (rows of results, each normalized to 1 W) is
$\kappa_{abc} = \overline{T_{abc}}$ with

$$
T_{abc} = \frac{\omega_a\varepsilon_0}{4} \int_S \sum_{ijk} d_{ijk}\,
\overline{E_{a,i}}\,E_{b,j}\,E_{c,k}\,\mathrm{d}A,
$$

all three components of each mode. It is reported in the convention of
coupled-mode tools, $e^{-i\omega t}$, the complex conjugate of the integral in
gfem's own phasors: the amplitude $A$ of a wave, with $|A|^2$ its power, obeys
$\mathrm{d}A_a/\mathrm{d}z \ni i\sum_{bc}\kappa_{abc}A_bA_c\,
e^{-i\Delta\beta_{abc}z}$, $\Delta\beta_{abc} = \beta_a - \beta_b - \beta_c$,
summed over every ordered pair with no degeneracy factor, so the factor 2 of
sum-frequency generation against second-harmonic generation comes out of the
sum. $\kappa_{abc}$ is symmetric in $b, c$ and has units
1/(W$^{1/2}$ m); on an Ez line port of a 2D model (1 W/m, integral along the
line) W$^{-1/2}$ m$^{-1/2}$.

For the second harmonic of a pump at $\omega$, $\kappa = \kappa_{hpp}$ with
$h$ the mode at $2\omega$, and the undepleted harmonic power is

$$
P_{2\omega}(L) = \eta_\mathrm{norm}\,P_\omega^2L^2\,
\mathrm{sinc}^2\!\left(\frac{\Delta\beta L}{2}\right),\qquad
\eta_\mathrm{norm} = |\kappa|^2,
$$

$\Delta\beta = \beta_{2\omega} - 2\beta_\omega$ in 1/m and
$\eta_\mathrm{norm}$ in 1/(W m²) (1/(W m) on a line port;
1 W$^{-1}$m$^{-2}$ is $10^{-2}$ %/(W cm²)). Quasi-phase matching with a
square-wave poling of period $\Lambda$, duty cycle $D$ and order $m$ multiplies
$\kappa$ by $G_m = 2\sin(\pi mD)/(\pi m)$ and replaces $\Delta\beta$ by
$|\Delta\beta| - 2\pi m/\Lambda$, matched at $\Lambda = 2\pi m/|\Delta\beta|$.
With the loss of the modes, $\alpha = \mathrm{Re}\,\gamma$ and
$\Delta\alpha = \alpha_{2\omega} - 2\alpha_\omega$,

$$
\frac{P_{2\omega}}{P_\omega} = |G|^2|\kappa|^2P_\omega\,
e^{-(\alpha_{2\omega} + 2\alpha_\omega)L}\,
\frac{\sinh^2(\Delta\alpha L/2) + \sin^2(\delta L/2)}
{(\Delta\alpha/2)^2 + (\delta/2)^2},
$$

with $\delta$ the residual mismatch. For difference-frequency generation and
parametric amplification of a pump $a$ with a signal $b$ and an idler $c$, the
amplitude of $b$ obeys $\mathrm{d}A_b/\mathrm{d}z \ni i\,(2\omega_b/\omega_a)\,
\overline{\kappa_{abc}}\,A_a\overline{A_c}\,e^{i\Delta\beta z}$ (and $b
\leftrightarrow c$): the rule conserves the photon flux when $d$ is symmetric
in its three indices (Kleinman symmetry), which holds when one $d$ serves
every process. The sum-frequency power of two waves in the undepleted
regime is $4|\kappa_{312}|^2P_1P_2L^2\,\mathrm{sinc}^2(\Delta\beta L/2)$.
The formulas hold while $\alpha \ll \beta$ and for a pump that is not
depleted. The {py:obj}`gfem.conventions` module, "Second-order nonlinearity",
gives the same formulas, and {doc}`recipes/second-harmonic-generation` the
calculation.

The tensor depends on the material. Lithium niobate (class 3m) has the four
numbers $d_{33}$, $d_{31} = d_{15}$ and $d_{22}$; gallium arsenide
($\bar 4 3 m$) the single $d_{14} = d_{25} = d_{36}$; aluminium and gallium
nitride (6mm) $d_{15} = d_{24}$, $d_{31} = d_{32}$ and $d_{33}$; potassium
titanyl phosphate (mm2) $d_{15}, d_{24}, d_{31}, d_{32}, d_{33}$. Miller's rule,
$d(\omega_3; \omega_1, \omega_2) \propto \chi^{(1)}(\omega_1)
\chi^{(1)}(\omega_2)\chi^{(1)}(\omega_3)$ with $\chi^{(1)} = n^2 - 1$ along
the polarizations involved, scales a value measured at one wavelength to
another, and is empirical. Quoted values differ between sources and
wavelengths, so read the source of a value before relying on it. A square-wave
poling at duty cycle $D$ keeps $|G_1|^2 = 4/\pi^2$ at $D = 1/2$, and the
third order has $|G_3|^2 = 4/(9\pi^2)$ there, a ninth of the overlap at a
period three times longer.

### Generated field of a full-field pump

Off a guide there is no mode to overlap, so the pump's field $F = \sum_p a_p E_p$
is read at the quadrature points ($a_p = \sqrt{P_p}\,e^{i\phi_p}$, $E_p$ the
kept field of column $p$) and contracted with $d$ in the global frame:

| Process | $\omega_3$ | $P$ |
|---|---|---|
| Second harmonic | $2\omega$ | $\varepsilon_0\,d\!:\!FF$ |
| Sum frequency | $\omega_1 + \omega_2$ | $2\varepsilon_0\,d\!:\!F_1F_2$ |
| Difference frequency | $\omega_1 - \omega_2$ | $2\varepsilon_0\,d\!:\!F_1\overline{F_2}$ |

$P$ lives only where the tensor is nonzero and outside a PML. A solve at
$\omega_3$ takes it as the current $J = +i\omega_3P$, so its load is the dipole
load integrated, and the waves at the ports are read as for any source. The
pump is the solution at its own frequency alone: undepleted, with no
back-action. On a Bloch cell the polarization carries the sum of the pump's wave vectors,
$2k_\parallel$ for a second harmonic, $k_1 + k_2$ and $k_1 - k_2$ for sum and
difference frequency: the generated solve takes `wave_vector=source.wave_vector`
in `gfem.Driven`, and gfem raises, naming the vector to write, when it differs
by anything but a reciprocal lattice vector. Declare its ports with
`gfem.diffraction.floquet_ports(model, surface, source.frequency,
wave_vector=...)`, since the orders that propagate depend on the incidence.

(theory-nonlinear-and-quantum-optics-photon-pairs)=
## Photon pairs

A pair source is described in the low-gain regime, at most a pair per mode and
pulse. The one quantization step: the amplitude of a wave, with $|A|^2$ a
power, becomes $\tilde A(\omega) \to \sqrt{\hbar\omega/2\pi}\,b(\omega)$ with
$[b(\omega), b^\dagger(\omega')] = \delta(\omega - \omega')$, so the classical
coupled-mode equations of {py:obj}`gfem.three_wave_coupling` and
{py:obj}`gfem.kerr_coupling` are the equations of the operators and a
stationary $\langle b^\dagger(\omega)b(\omega')\rangle = n(\omega)\,
\delta(\omega - \omega')$ is a flux $\int n\,\mathrm{d}\omega/2\pi$: $n$ is
in pairs per second and hertz, and is also the occupation of a
spectral-temporal mode.

The two-photon amplitude of a pump $\tilde A_p$ (the published
$e^{-i\omega t}$ convention, $\omega = 2\pi f$) is, for spontaneous
parametric down-conversion into a signal $s$ and an idler $i$ with
$\omega_p = \omega_s + \omega_i$,

$$
F(\omega_s, \omega_i) = g_{si}\,G_m\,\tilde A_p(\omega_s + \omega_i)\,L\,
e^{i\delta L/2}\,\mathrm{sinc}\frac{\delta L}{2},\qquad
g_{si} = \frac{2\sqrt{\omega_s\omega_i}}{\omega_p}\,\overline{\kappa_{psi}},
$$

with $\kappa_{psi}$ the entry of the coupling of
{doc}`recipes/second-harmonic-generation`, $\Delta\beta = \beta_p - \beta_s -
\beta_i$, and, poled with period $\Lambda$, duty cycle $D$ and order $m$,
$\delta = \Delta\beta - \mathrm{sign}(\Delta\beta_0)\,2\pi m/\Lambda$,
$|G_m| = 2|\sin(\pi mD)|/(\pi m)$ (unpoled: $G = 1$, $\delta = \Delta\beta$). The
coupling is the difference-frequency rule of that page and needs a $d$ with
Kleinman symmetry. For spontaneous four-wave mixing of a degenerate pump with
$2\omega_p = \omega_s + \omega_i$,

$$
F(\omega_s, \omega_i) = \Gamma_{si}\,[\tilde A_p \ast \tilde A_p](\omega_s +
\omega_i)\,L\,e^{i\delta L/2}\,\mathrm{sinc}\frac{\delta L}{2},\qquad
\Gamma_{si} = \tfrac12\Big(\sqrt{\tfrac{\omega_i}{\omega_s}}\,\gamma_{sppi} +
\sqrt{\tfrac{\omega_s}{\omega_i}}\,\gamma_{ipps}\Big),
$$

$\Delta\beta = 2\beta_p - \beta_s - \beta_i$, with $\beta_p$ at
$(\omega_s + \omega_i)/2$, $\gamma_{abcd}$ the entries of
{doc}`recipes/waveguide-nonlinearity`, and the pump's envelope replaced by the
spectrum of $A_p(t)^2$. The symmetrized $\Gamma_{si}$ is what the commutators
of the operators need: the stored tensor takes $n^2$ at the output wave, so
$\gamma_{sppi}/\omega_s$ and $\gamma_{ipps}/\omega_i$ differ by the
dispersion of the index. For a continuous pump of power $P$ the density of
signal photons and the rate in the signal band are

$$
n_s(\omega_s) = g_{si}^2|G_m|^2PL^2\,\mathrm{sinc}^2\frac{\delta L}{2}
\quad\Big(|\Gamma_{si}|^2P^2L^2\,\mathrm{sinc}^2\frac{\delta L}{2}\Big),\qquad
R = \int n_s\,\mathrm{d}f_s,
$$

with the idler at $\omega_p - \omega_s$ ($2\omega_p - \omega_s$) and, for
four-wave mixing, the Kerr shift $2\,\mathrm{Re}(\gamma_{pppp} - \gamma_{sspp} -
\gamma_{iipp})P$ added to $\Delta\beta$. At degeneracy in one mode family
$g_{si}^2 = \eta_\mathrm{norm}$ of {doc}`recipes/second-harmonic-generation`:
the pair density is the stimulated conversion of a seed of one photon per
second and hertz, with no free factor. A pulse of energy $U = 2\pi\int|\tilde
A_p|^2\mathrm{d}\omega$ gives $N = \iint|F|^2\,\mathrm{d}\omega_s\mathrm{d}\omega_i$
pairs; `gfem.JointSpectrum.jsa` is $f = 2\pi F$ in 1/Hz, so $\sum|f|^2\,\delta
f_s\,\delta f_i$ is the number of pairs per pulse.

When one mode family carries both photons (the same result as signal and
idler) the amplitude is symmetric under the exchange of the frequencies and a
pair is two photons of one field: the count is $\tfrac12\iint|F|^2$ over the
whole plane, which is the half-plane $\omega_s > \omega_i$, and a rate is
halved in the same way; the coupling is not. The bands of a source are the
frequency grid and act as ideal filters: every figure counts the pairs
inside them. Under loss $\alpha = \mathrm{Re}\,\gamma$ of each arm,
$L\,e^{i\delta L/2}\mathrm{sinc}(\delta L/2)$ becomes the coherent
$\int_0^L e^{-\alpha_pz - (\alpha_s + \alpha_i)(L - z) + i\delta z}\,\mathrm{d}z$
($2\alpha_p$ for four-wave mixing): coincidences only. The heralded purity of
one photon is $\sum_k\lambda_k^2$ over the Schmidt weights of the amplitude,
computed as $\mathrm{Tr}[(MM^\dagger)^2]/\mathrm{Tr}[MM^\dagger]^2$ with $M$
the amplitude on the grid, and the Schmidt number is its inverse. The
formulas hold in the low-gain regime, for a lossless or weakly lossy
section ($\alpha \ll \beta$) and a pump that is not depleted. The
{py:obj}`gfem.conventions` module, "Photon pairs", gives the same formulas, and
{doc}`recipes/photon-pair-sources` the calculation.

## Validity and limits

The formulas hold while the attenuation is small against the phase constant
($\alpha \ll \beta$) and for a pump that is not depleted. The photon-pair
formulas hold in the low-gain regime, at most a pair per mode and pulse.
