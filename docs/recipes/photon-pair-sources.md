# Photon-pair sources

Find how many pairs of photons a waveguide emits when it is pumped, in which
frequencies, and how pure each photon is once its partner is detected. Two
processes make pairs: spontaneous parametric down-conversion (SPDC) in a
guide with a second-order nonlinearity, where a pump photon splits in two, and
spontaneous four-wave mixing (SFWM) in a guide with a third-order one, where
two pump photons turn into a pair. `gfem.spdc` and `gfem.sfwm` read three
{py:obj}`gfem.PortModeResult` (the pump, the signal and the idler), and the
model's materials: they need no solve beyond the modes' own. They return a
{py:obj}`gfem.PhotonPairSource`, whose methods give the pair rate and the
spectral brightness of a continuous pump, and the
{term}`joint spectral amplitude` of a pulse with its pairs per pulse and its
{term}`heralded purity`.

The first guide is a periodically poled x-cut thin-film lithium niobate ridge,
after Zhao et al., Phys. Rev. Lett. 124, 163603 (2020): a 300 nm film etched
50 nm deep into a ridge 1.24 µm wide, on 1.8 µm of oxide, poled over 5 mm
with a first-order period of 2.8 µm, the quasi-TE mode at 785 nm splitting
into two quasi-TE photons near 1570 nm through $d_{33}$. The film, the etch,
the width and the poling length are the paper's; the sidewalls are vertical
here (the paper's are 75°), the crystal is the undoped one of the
second-harmonic recipe ({doc}`second-harmonic-generation`) and the poling
is ideal, so the numbers below are those of this guide, not of the paper's
device. The second is the silicon strip of {doc}`waveguide-nonlinearity`.
Units are SI ({doc}`../conventions`, "Photon pairs"): rates in pairs/s, a
spectral density in pairs/(s Hz), a joint spectral amplitude in 1/Hz.

## The crystal and the ridge

The nonlinear tensor, the oxide and the ridge are those of
{doc}`second-harmonic-generation`, with the paper's width, film and etch, and
its $d_{33}$ of 27 pm/V (the two other numbers are that page's):

```python
import dataclasses
import urllib.request
import warnings

import jax
import numpy as np
import scipy.constants

import gfem

c = scipy.constants.c  # m/s


def sellmeier(strengths, wavelengths_um):
    """n^2 = 1 + sum of B lambda^2 / (lambda^2 - lambda_i^2), lambda_i in um."""
    terms = [
        gfem.Lorentz(delta_eps=b, resonance_frequency=c / (li * 1e-6), damping=0.0)
        for b, li in zip(strengths, wavelengths_um, strict=True)
    ]
    return gfem.PermittivityLaw(eps_inf=1.0, terms=terms)


# Malitson (1965): 0.21 to 3.71 um
oxide = gfem.Material(
    eps_r=sellmeier((0.6961663, 0.4079426, 0.8974794), (0.0684043, 0.1162414, 9.896161))
)

d33, d31, d22 = 27e-12, 4.6e-12, 2.2e-12  # m/V: Zhao et al. 2020, Wang et al. 2018
d_ln = (  # rows x, y, z; columns xx, yy, zz, yz, xz, xy
    (0, 0, 0, 0, d31, -d22),
    (-d22, d22, 0, d31, 0, 0),
    (d31, d31, d33, 0, 0, 0),
)
X_CUT = ((0, 1, 0), (0, 0, 1), (1, 0, 0))  # crystal x, y, z in the section's frame


def lithium_niobate(frequency):
    """Zelmon's indices at the frequency, with the d tensor, on x-cut axes."""
    crystal = gfem.materials.lithium_niobate(frequency=frequency, axes=X_CUT)
    return dataclasses.replace(crystal, second_order_nonlinear_tensor=d_ln)


url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/ridge_section.py"
)
urllib.request.urlretrieve(url, "ridge_section.py")
from ridge_section import ridge_section  # noqa: E402

mesh = ridge_section(1.24e-6, film=0.3e-6, etch=0.05e-6, margin=1.8e-6)
options = gfem.SolverOptions(order=3)


def guide(frequency):
    """The ridge with the crystal's indices at ``frequency``, wave port on its section."""
    return gfem.Model(
        mesh,
        materials={
            "film": lithium_niobate(frequency),
            "box": oxide,
            "air": gfem.Material(eps_r=1.0),
        },
        ports=[gfem.WavePort("wg", "front")],
    )
```

The materials are made at the frequency of each wave, so the signal and the
pump each have a model of the same mesh. Only the permittivity differs between
them; the model passed to `gfem.spdc` supplies the tensor.

## The modes of the pump and of the pair

The pair is degenerate near 1570 nm. Which mode of the pump at 785 nm couples
to the quasi-TE fundamental at 1570 nm is read from the overlap: rank the
pump's modes and compute each one's second-harmonic efficiency with the signal
mode, as the second-harmonic page does. That efficiency is also the classical
figure of merit of the pair source (see "Check your results"):

```python
f0 = c / 1.57e-6  # Hz, the band's centre
signal_model, pump_model = guide(f0), guide(2 * f0)
seed = gfem.solve(signal_model, gfem.PortMode("wg", np.array([f0])), options=options)
shg = {}
for rank in (1, 2):
    problem = gfem.PortMode("wg", np.array([2 * f0]), mode=rank)
    candidate = gfem.solve(pump_model, problem, options=options)
    shg[rank] = gfem.second_harmonic(signal_model, seed, candidate)
    print(
        f"pump mode {rank}: n_eff {candidate.n_eff[0].real:.4f}, "
        f"eta_norm {shg[rank].eta_norm[0] * 1e-2:.3g} %/(W cm^2)"
    )
```

```text
pump mode 1: n_eff 1.9815, eta_norm 1.06e+04 %/(W cm^2)
pump mode 2: n_eff 1.9529, eta_norm 0.000147 %/(W cm^2)
```

The fundamental is the coupled one. The dispersion of each arm enters the
joint spectrum, so each is solved on a band, not at one frequency: nine
frequencies over $\pm 14$ THz around the centre (about $\pm 120$ nm) for the
signal, and nine over twice that around $2f_0$ for the pump, which must cover
the sums of the signal and idler frequencies. Between the frequencies a
sweep solves exactly, a port mode comes from a reduced model
({doc}`port-mode-gradient`):

```python
half = 14e12  # Hz
signal_frequency = f0 + np.linspace(-half, half, 9)
signal = gfem.solve(
    signal_model, gfem.PortMode("wg", signal_frequency), options=options
)
pump = gfem.solve(
    pump_model, gfem.PortMode("wg", 2 * signal_frequency), options=options
)
print(signal)
print(pump)
```

```text
PortModeResult(port 'wg', mode 1, 9 frequencies 177-205 THz (5 exact solves), n_eff 1.6684-1.7282, n_g 2.0776-2.13, no voltage_path)
PortModeResult(port 'wg', mode 1, 9 frequencies 353.9-409.9 THz (5 exact solves), n_eff 1.9602-1.9999, n_g 2.2502-2.2514, no voltage_path)
```

- Each arm's rows are fitted by a polynomial of at most degree 4 in the
  frequency, and the fit is evaluated, never extrapolated, inside the rows'
  span. Four rows is the least; more rows over a narrower band follow the
  dispersion better.
- The three results must be of one port and one mode each. For degenerate
  photons in one mode family, the same result is the signal and the idler.

## The source

```python
source = gfem.spdc(signal_model, pump, signal, signal, points=601)
print(source)
period = source.poling_period
print(f"first-order poling period {period * 1e6:.4f} um (Zhao et al.: 2.8 um)")
```

```text
PhotonPairSource(spdc, port 'wg', modes 1 -> 1 + 1, identical, pump 353.9-409.9 THz, signal 176.95-204.95 THz, idler 176.95-204.95 THz, |coupling| 1030 1/(sqrt(W) m), poling period 2.784 um)
first-order poling period 2.7840 um (Zhao et al.: 2.8 um)
```

- `signal is idler` means identical photons: one mode family carries both, and
  every figure counts a pair once.
- The coupling $\kappa$ is one overlap of the pump, signal and idler at the
  centres of the bands, taken from `gfem.three_wave_coupling`; the
  dispersion of the guide over the bands comes from the fits. The bands are the
  grid of every figure, with `points` samples each, and act as ideal filters:
  every figure counts the pairs inside them.
- `poling_period` is $2\pi/|\Delta\beta_0|$ at the bands' centres. It sets the
  first order; a period of order $m$ is $m$ times it. The poling is an
  argument of the figures, not of the source, so a design sweep costs no
  overlap.

## The rate and the spectral brightness of a continuous pump

A pump of 1 mW in the 5 mm guide, poled at that period. The brightness is the
density of signal photons in pairs/(s Hz) on the signal grid, the idler being
fixed by energy conservation; the rate is its integral, in the band:

```python
length, power = 5e-3, 1e-3  # m, W
rate = source.rate(length, power, period=period)
brightness = source.spectral_brightness(length, power, period=period)
peak = float(brightness.max())
print(f"rate in the bands {rate:.4g} pairs/s at {power * 1e3:g} mW")
print(f"peak density {peak:.4g} pairs/(s Hz)")
per_ghz = peak * 1e9 / (power * 1e3)
per_nm = peak * c / 1.57e-6**2 * 1e-9 / (power * 1e3)
print(f"brightness {per_ghz:.4g} pairs/s/GHz/mW = {per_nm:.4g} pairs/s/nm/mW")
```

```text
rate in the bands 4.06e+10 pairs/s at 1 mW
peak density 0.01075 pairs/(s Hz)
brightness 1.075e+07 pairs/s/GHz/mW = 1.307e+09 pairs/s/nm/mW
```

The density is also the mean number of pairs per spectral-temporal mode, so
it is the low-gain check: above 0.1 the first-order result is off by more than
3 % and `gfem.HighGainWarning` says so. A longer guide narrows the matched
band, and a poling period that is off detunes it:

```python
for mm in (3.5, 7.0, 14.0):
    r = source.rate(mm * 1e-3, power, period=period)
    print(f"L = {mm:4.1f} mm: rate {r:.4g} pairs/s")
for error in (-1e-3, 0.0, 1e-3):
    r = source.rate(length, power, period=period * (1 + error))
    print(f"period {error * 100:+.1f} %: rate {r / rate:.3f} of the matched rate")
```

```text
L =  3.5 mm: rate 2.374e+10 pairs/s
L =  7.0 mm: rate 6.731e+10 pairs/s
L = 14.0 mm: rate 1.905e+11 pairs/s
period -0.1 %: rate 0.017 of the matched rate
period +0.0 %: rate 1.000 of the matched rate
period +0.1 %: rate 0.561 of the matched rate
```

- The two photons are in one mode family and at the same frequency, so the
  matching is quadratic in the detuning,
  $\Delta\beta(\Omega) = \Delta\beta_0 - \beta_2\Omega^2$, and the rate grows
  as $L^{3/2}$ at $\Delta\beta_0 = 0$ and not as $L$ (a nondegenerate pair,
  with different group velocities, grows as $L$): doubling the length
  multiplies the rate by $2^{3/2} = 2.83$ while the band is wider than the
  matching.
- A period 0.1 % off detunes the matching by $\delta L/2$ of several radians
  over 5 mm, and the two signs are not alike. With $\beta_2 > 0$ the
  mismatch $\Delta\beta_0 - 2\pi/\Lambda - \beta_2\Omega^2$ crosses zero
  for a period longer than the matched one, at two frequencies $\pm\Omega$,
  and the rate stays near half; for a shorter period it never does, and the
  rate collapses.
- The rate is linear in the power at this gain; the spectral brightness per
  milliwatt is what a measurement quotes.

## The joint spectrum of a pulse

A Gaussian pump pulse of energy $U$ and intensity FWHM $T$ makes the pairs of
a joint spectral amplitude $f(f_s, f_i)$, the pump's envelope times the
phase-matching function, scaled so that $\sum|f|^2\,\delta f_s\,\delta f_i$ is
the number of pairs per pulse. This guide's matching is long and narrow in
the sum of the frequencies, because the pump and the signal differ in group
velocity (the pump's $n_g$ is 2.25, the signal's 2.11), and broad in their
difference, so a joint spectrum needs a grid fine in the sum: 601 points over
28 THz is 0.047 THz, against a matching whose first zeros lie 0.4 THz from its
peak. In a 5 mm guide a low-gain pulse is a femtojoule, not a picojoule:

```python
pulse = {"pulse_energy": 1e-15, "pulse_duration": 1e-12}  # J, s
spectrum = source.joint_spectrum(length, period=period, **pulse)
print(spectrum)
density = np.abs(np.asarray(spectrum.jsa)) ** 2
step = spectrum.signal_frequency[1] - spectrum.signal_frequency[0]
marginal = density.sum(axis=1)  # the signal photon's spectrum
width = step * np.count_nonzero(marginal >= marginal.max() / 2)
print(f"signal photon: {width * 1e-12:.2f} THz FWHM, {1e-12 * 2 * half:g} THz band")
print(f"pairs per pulse {float(spectrum.pairs_per_pulse):.4g}")
print(f"pairs per second at 100 MHz: {100e6 * float(spectrum.pairs_per_pulse):.4g}")
```

```text
JointSpectrum(601 x 601 grid, 0.0324 pairs/pulse, purity 0.03338, Schmidt number 29.96)
signal photon: 9.33 THz FWHM, 28 THz band
pairs per pulse 0.0324
pairs per second at 100 MHz: 3.24e+06
```

- The pulse's energy is not the power: 1 fJ at 100 MHz is 100 nW. The pairs
  per pulse of down-conversion grow as $U$, and its mean rate as the mean
  power.
- The photons are as wide as the matching allows, a sizable part of the band,
  against a pump whose bandwidth is 0.44 THz: the pair is strongly correlated
  in frequency, so its photons are far from pure. A narrower band is a filter
  that raises the purity at the cost of the pairs.

The heralded purity of one photon of a pair is $\sum_k\lambda_k^2$ over the
Schmidt weights $\lambda_k$ of the joint spectrum, its Schmidt number
$K = 1/\sum_k\lambda_k^2$ the number of modes a photon is spread over. It
changes with the pulse. A finer grid than the 601 points above keeps a long
pulse, whose spectrum is narrow, resolved (1001 points is 0.028 THz):

```python
fine = gfem.spdc(signal_model, pump, signal, signal, points=1001)
print("T (ps)   pairs/pulse   purity   Schmidt number")
for ps in (1.0, 1.5, 2.0, 3.0):
    js = fine.joint_spectrum(
        length, period=period, pulse_energy=1e-15, pulse_duration=ps * 1e-12
    )
    print(
        f"{ps:5.1f}  {float(js.pairs_per_pulse):12.4g}  "
        f"{float(js.purity):7.4f}  {float(js.schmidt_number):7.2f}"
    )
```

```text
T (ps)   pairs/pulse   purity   Schmidt number
  1.0        0.0324   0.0334    29.96
  1.5       0.03587   0.0321    31.15
  2.0       0.03762   0.0285    35.11
  3.0       0.03915   0.0218    45.95
```

- The purity is low, a Schmidt number of 30 to 45, and has a maximum near
  1 ps. There the pump's bandwidth, 0.44 THz, is of the order of the width of the
  matching in the sum of the frequencies, about 0.4 THz (set by the
  pump-signal group-velocity mismatch). A longer pulse is narrower than the
  matching: the joint spectrum is a thinner line along the difference of the
  frequencies, with more Schmidt modes, and the purity falls. A shorter one
  is wider than the matching, which then sets the sum alone.
- The purity is computed without a singular value decomposition, from
  $\mathrm{Tr}[(MM^\dagger)^2]/\mathrm{Tr}[MM^\dagger]^2$ with $M$ the amplitude
  on the uniform grid, so its gradient is finite where Schmidt weights are
  equal.

## Plot the joint spectrum

The amplitude, its two grids and the pairs it counts are the attributes of
`gfem.JointSpectrum`: the signal frequency down, the idler across.

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots()
image = ax.pcolormesh(
    spectrum.idler_frequency * 1e-12,
    spectrum.signal_frequency * 1e-12,
    np.abs(np.asarray(spectrum.jsa)) ** 2,
    shading="auto",
)
ax.set_xlabel("idler frequency (THz)")
ax.set_ylabel("signal frequency (THz)")
fig.colorbar(image, label="$|f|^2$ (1/Hz$^2$)")
fig.savefig("joint_spectrum.png", dpi=150)
```

The pairs lie along the line of constant sum of the frequencies, with the
pump's width across it, and run to the band's edges along it, where the
matching's sinc lobes fall.

## The gradient of the purity

The purity and the rate differentiate in every argument of the methods and, through the
fits and the overlaps, in the section's materials and in the rows' $\gamma$:

```python
def purity(duration):
    js = fine.joint_spectrum(
        length, period=period, pulse_energy=1e-15, pulse_duration=duration
    )
    return js.purity


duration = 2e-12
slope = jax.grad(purity)(duration)
h = 1e-3 * duration
central = (purity(duration + h) - purity(duration - h)) / (2 * h)
print(f"d purity / d duration {float(slope) * 1e-12:.5g} per ps")
print(f"central difference    {float(central) * 1e-12:.5g} per ps")
```

```text
d purity / d duration -0.0076168 per ps
central difference    -0.0076168 per ps
```

The same call differentiates in the pulse's energy, the guide's length, the
poling period and the duty cycle, and `jax.grad` of a function that builds the
source from a `gfem.solve` gives the sensitivity to the materials and, through the
modes' propagation constants and fields, to the section's permittivity
({doc}`port-mode-gradient`, {doc}`eigenmode-gradient`). The node coordinates
are not differentiated: a change of width is a remesh and a finite difference.

## Four-wave mixing in a silicon wire

The silicon strip of {doc}`waveguide-nonlinearity`, 500 × 220 nm in oxide,
pumped at 1.55 µm, makes pairs through $\gamma_\mathrm{NL}$: the signal above
the pump and the idler below it, with $2f_p = f_s + f_i$. The bands
leave a guard of 1 THz around the pump, where a real filter must cut it, and
run to 13 THz from it:

```python
silicon = gfem.Material(
    eps_r=sellmeier(
        (10.6684293, 0.0030434748, 1.54133408), (0.301516485, 1.13475115, 1104.0)
    )
)
# Dinu, Quochi and Garcia, Appl. Phys. Lett. 82, 2954 (2003): silicon at 1.54 um
silicon_kerr = dataclasses.replace(
    silicon, nonlinear_index=4e-18, two_photon_absorption=8e-12
)
url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/strip_section.py"
)
urllib.request.urlretrieve(url, "strip_section.py")
from strip_section import strip_section  # noqa: E402

wire = gfem.Model(
    strip_section(500e-9, height=220e-9, clearance=1e-6),
    materials={"core": silicon_kerr, "cladding": oxide},
    ports=[gfem.WavePort("wg", "front")],
)
f_pump = c / 1.55e-6
guard, reach = 1e12, 13e12  # Hz
signal_band = f_pump + np.linspace(guard, reach, 7)
idler_band = f_pump - np.linspace(reach, guard, 7)
pump_band = f_pump + np.linspace(-7e12, 7e12, 7)  # the half-sums of the bands
solved = [
    gfem.solve(wire, gfem.PortMode("wg", band), options=options)
    for band in (pump_band, signal_band, idler_band)
]
wire_source = gfem.sfwm(wire, *solved, points=401)
print(wire_source)
```

```text
PhotonPairSource(sfwm, port 'wg', modes 1 -> 1 + 1, pump 186.41-200.41 THz, signal 194.41-206.41 THz, idler 180.41-192.41 THz, |coupling| 188.6 1/(W m))
```

The pump is the degenerate pump of the process and its band need only cover
the half-sums of the signal and idler frequencies. The two coupling
entries are $\gamma_{sppi}$ and $\gamma_{ipps}$ of `gfem.kerr_coupling`, and the
source uses their symmetrized mean for the pair, since the stored tensor
takes $n^2$ at the output wave and the two differ by the dispersion of
the index.

```python
length_w, power_w = 1e-2, 1e-3  # m, W
with warnings.catch_warnings():
    # the guard band cuts the pairs on purpose: a filtered source is the question
    warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
    density = wire_source.spectral_brightness(length_w, power_w)
    rate_w = wire_source.rate(length_w, power_w)
    ten = wire_source.rate(length_w, 10 * power_w)
peak_w = float(density.max())
per_nm_w = peak_w * f_pump**2 / c * 1e-9 / (power_w * 1e3) ** 2  # Hz/nm/mW^2
print(f"peak density {peak_w:.4g} pairs/(s Hz), rate in the bands {rate_w:.4g} pairs/s")
print(f"brightness {per_nm_w:.4g} pairs/s/nm/mW^2")
print(f"rate at 10 mW over 1 mW: {ten / rate_w:.2f} (quadratic: 100)")
```

```text
peak density 3.294e-06 pairs/(s Hz), rate in the bands 7.907e+06 pairs/s
brightness 4.111e+05 pairs/s/nm/mW^2
rate at 10 mW over 1 mW: 100.88 (quadratic: 100)
```

- A pair is born from two pump photons, so the density and the rate go as the
  square of the power, and a pump above about 0.17 W puts more than 0.1 pair in
  a mode: `gfem.HighGainWarning` fires there. The continuous-wave figures include the
  Kerr shift of the pump and the cross-phase of the signal and the idler by it
  in the mismatch.
- Two-photon absorption of the pump is in the complex coupling and not in a
  loss of the pump along the guide, and neither Raman scattering nor free
  carriers are modelled. At a few milliwatts in a short guide they are small;
  they are not in a long, strongly pumped one.
- The density is largest next to the guard, close to the pump, where $\Delta\beta$ is
  smallest, and falls off over a few terahertz: this guide's dispersion is
  anomalous, so the Kerr shift of a strong pump moves the matching away from
  the pump, to larger detunings.

A pulse gives a joint spectrum as well, here cut by the guard band, on purpose:

```python
with warnings.catch_warnings():
    # the guard band cuts the pairs: a filtered source is the question here
    warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
    wire_pulse = wire_source.joint_spectrum(
        length_w, pulse_energy=5e-14, pulse_duration=0.5e-12
    )
print(wire_pulse)
```

```text
JointSpectrum(401 x 401 grid, 0.0241 pairs/pulse, purity 0.5201, Schmidt number 1.923)
```

## A ring: an estimate

A ring of this guide makes pairs when the pump, the signal and the idler sit on
resonances. gfem does not solve a ring (it has no bent mode), but the
straight guide's coupling, group index and loss give an estimate with the
coupled-mode theory of a ring with a bus, to be read as one. Take a ring of
radius 10 µm, a loaded quality factor $Q = 10^5$ for the pump, the signal and
the idler, critically coupled (so the coupler's decay rate $\kappa_\mathrm{ex}$
is half the loaded rate $\kappa$), and a pump of 0.1 mW in the bus. The
pair coupling $\Gamma_{si}$ is the one the source uses, the free spectral
range is $v_g/L_\mathrm{ring}$, and the rates are the energy decay rates in
rad/s:

```python
omega_s, omega_i = 2 * np.pi * signal_band.mean(), 2 * np.pi * idler_band.mean()
s, i = wire_source.coupling  # gamma_sppi, gamma_ipps, 1/(W m)
gamma_si = abs(0.5 * (np.sqrt(omega_i / omega_s) * s + np.sqrt(omega_s / omega_i) * i))
n_g = float(solved[0].n_g[3].real)
v_g = c / n_g
ring = 2 * np.pi * 10e-6
fsr = v_g / ring  # Hz


def ring_rate(q_loaded, p_bus=0.1e-3):
    """Pairs/s into the bus of the critically coupled ring."""
    kappa = 2 * np.pi * f_pump / q_loaded  # rad/s, loaded
    kappa_ex = kappa / 2
    p_ring = 4 * kappa_ex * fsr / kappa**2 * p_bus  # W, circulating
    g = gamma_si * v_g * p_ring  # rad/s
    occupation = 4 * g**2 / (kappa * 2 * kappa)  # pairs per mode, low gain
    rate = 4 * g**2 / (2 * kappa) * (kappa_ex / kappa) ** 2
    return rate, p_ring / p_bus, occupation


rate_r, enhancement, occupation = ring_rate(1e5)
print(f"FSR {fsr * 1e-9:.0f} GHz, |Gamma| {gamma_si:.1f} /(W m)")
print(f"circulating power {enhancement:.0f} x the bus's")
print(f"pairs into the bus {rate_r:.4g} /s, occupation {occupation:.2g} per mode")
print(f"Q doubled: factor {ring_rate(2e5)[0] / rate_r:.2f} (Q cubed: 8)")
```

```text
FSR 1137 GHz, |Gamma| 181.8 /(W m)
circulating power 187 x the bus's
pairs into the bus 2.432e+06 /s, occupation 0.0008 per mode
Q doubled: factor 8.00 (Q cubed: 8)
```

- The rate goes as $Q^3/L_\mathrm{ring}^2$ at a fixed power: a better
  quality factor concentrates more power and a narrower resonance, and a
  smaller ring has a larger free spectral range and a larger circulating power per
  round trip.
- What is an estimate: the group index of the straight guide (good to a few
  $10^{-3}$) and the assumption that the signal and the idler are on
  resonances (the dispersion of the ring detunes them by $D_2$ and the
  resonances' width must contain that), the pump on resonance with no thermal
  or Kerr shift, and every $\kappa$, which is the coupler's and the
  ring's loss and yours to give: gfem solves neither. The circulating power
  of the example, tens of milliwatts, makes two-photon absorption and free
  carriers matter in a silicon ring, and they are not in the estimate.
- A pulsed ring has a joint spectrum made of Lorentzian resonances: it is not
  in this page.

## Compared with published sources

Published on-chip brightnesses scatter by factors of 4 to 20 around the ideal
formulas, from the poling quality, the loss and coupling calibration and the
filters, so none of these is a check of the numbers above, only an
observation beside them:

| Source | Device | Published | gfem, this page |
| --- | --- | --- | --- |
| Zhao et al., Phys. Rev. Lett. 124, 163603 (2020) | x-cut TFLN ridge, 5 mm, type 0, 785 to 1570 nm | $4.6\times10^5$ pairs/s/GHz/mW, poling period 2.8 µm | brightness and period printed below, with the paper's geometry and an ideal poling |
| Javid et al., Phys. Rev. Lett. 127, 183601 (2021) | dispersion-engineered x-cut TFLN, 5 mm, type 0 | 13 GHz/mW total | the rate of this page's guide, whose geometry is not that one |
| Shi et al., Light Sci. Appl. 13, 282 (2024), table | silicon and nitride wires, 8 to 11 mm, SFWM | $1.9\times10^4$ to $1.9\times10^5$ Hz/nm/mW$^2$ | the silicon wire's brightness at 1 cm |

```python
print(
    f"TFLN: gfem {per_ghz:.3g} against 4.6e5 pairs/s/GHz/mW, ratio {per_ghz / 4.6e5:.1f}"
)
print(f"TFLN: total {rate / (power * 1e3) * 1e-9:.3g} GHz/mW in the bands (Javid: 13)")
print(
    f"Si: gfem {per_nm_w:.3g} Hz/nm/mW^2, ratio {per_nm_w / 1.9e5:.1f} to {per_nm_w / 1.9e4:.1f}"
)
```

```text
TFLN: gfem 1.07e+07 against 4.6e5 pairs/s/GHz/mW, ratio 23.4
TFLN: total 40.6 GHz/mW in the bands (Javid: 13)
Si: gfem 4.11e+05 Hz/nm/mW^2, ratio 2.2 to 21.6
```

- The poling period agrees with the paper's 2.8 µm within a percent: the
  period is a property of the modes' indices, which the solve gives. The
  brightness is 23 times the measurement. An ideal poling of the full film
  depth, no loss and a filter at the peak are in the figure; the paper's poling
  depth is 245 nm of the 300 nm film, its duty cycle is $55.2 \pm 3.0$ %, and
  the calibration of its collection and its filters enter its number. The gap
  is at the upper end of the scatter of published numbers; it is not a
  measure of gfem's accuracy.
- The silicon wire's brightness is 2 to 20 times the published range, the
  same kind of gap: a lossless guide, an ideal coupling and a filter at the
  guard are in the ideal figure, and the measured guides are not. The
  paper's silicon and nitride guides are not this wire, and its table was
  read in the main text only.

## Check your results

- **The grid.** The bands are the grid, so a figure that depends on its
  resolution converges with `points`. The rate is a trapezoidal sum, and the
  joint spectrum's lobes need about eleven points: `gfem.UnresolvedJointSpectrumWarning`
  says when they have fewer, or when the bands cut the amplitude:

  ```python
  for points in (301, 601, 1201):
      s = gfem.spdc(signal_model, pump, signal, signal, points=points)
      print(f"{points} points: rate {float(s.rate(length, power, period=period)):.6g}")
  ```

  ```text
  301 points: rate 4.05987e+10
  601 points: rate 4.05987e+10
  1201 points: rate 4.05987e+10
  ```

- **The fit.** A fit that leaves a residual phase above 0.1 rad over the
  guide's length raises. The residual per arm, times the length, is what the
  check reads, in radians, for the pump, the signal and the idler:

  ```python
  print(np.asarray(source.fit_residual) * length)
  ```

  ```text
  [3.72633599e-04 3.68914427e-05 3.68914427e-05]
  ```

- **The overlap over the band.** The coupling is taken at the bands' centres
  and held over them. Recompute it with the signal at the top of the band and
  the idler at the bottom, with the pump between, and read the difference:

  ```python
  options_edge = gfem.SolverOptions(order=3)
  edge = [
      gfem.solve(m, gfem.PortMode("wg", np.array([f])), options=options_edge)
      for m, f in (
          (pump_model, 2 * f0),
          (signal_model, f0 + half),
          (signal_model, f0 - half),
      )
  ]
  tensor = gfem.three_wave_coupling(signal_model, edge)
  change = abs(tensor.kappa[0, 1, 2]) / abs(source.coupling) - 1
  print(f"|kappa| at the band edges against the centre: {change * 100:+.2f} %")
  ```

  ```text
  |kappa| at the band edges against the centre: -0.83 %
  ```

  A few tenths of a percent: the variation of the overlap over a band of
  10 % in frequency is below the other uncertainties.
- **The port mode tolerance.** The default `port_mode_tolerance` of $10^{-6}$ is
  the backward error of each mode between the exact solves. What matters for
  a pair source is the phase $\Delta\beta L$ it makes, which should be far
  below 0.1 rad. Re-solve the bands at a tolerance $10^3$ looser and compare
  the propagation constants of the same rows:

  ```python
  loose = gfem.SolverOptions(order=3, port_mode_tolerance=1e-3)
  signal_loose = gfem.solve(
      signal_model, gfem.PortMode("wg", signal_frequency), options=loose
  )
  pump_loose = gfem.solve(
      pump_model, gfem.PortMode("wg", 2 * signal_frequency), options=loose
  )
  for name, a, b in (("signal", signal, signal_loose), ("pump", pump, pump_loose)):
      error = np.max(np.abs(a.gamma.imag - b.gamma.imag))
      print(f"{name}: beta changes by {error:.2g} 1/m, {error * length:.2g} rad")
  ```

  ```text
  signal: beta changes by 0.01 1/m, 5.2e-05 rad
  pump: beta changes by 0.0035 1/m, 1.7e-05 rad
  ```

  Even $10^{-3}$ leaves the phase at the $10^{-5}$ rad level, four orders of
  magnitude below the limit, and the default is $10^3$ tighter: against a solve at $10^{-12}$ its propagation
  constants differ by $10^{-8}$ 1/m, the round-off of the eigenproblem. The
  recipe keeps the default: a tolerance of $10^{-9}$ costs exact solves and
  gains nothing here.
- **The classical figure of merit.** At degeneracy and perfect matching, the
  density of pairs at the band's centre is the second-harmonic efficiency
  of the same overlap, $|G_1|^2\eta_\mathrm{norm}PL^2$ with $|G_1|^2 =
  4/\pi^2$ for a duty cycle of one half: spontaneous emission is stimulated emission
  with a seed of one photon per second and hertz.

  ```python
  centre = float(brightness[len(brightness) // 2])
  classical = 4 / np.pi**2 * float(shg[1].eta_norm[0]) * power * length**2
  print(f"density at the centre {centre:.6g}, classical {classical:.6g}")
  ```

  ```text
  density at the centre 0.0107483, classical 0.0107483
  ```

- **Closed forms.** The pair rate, the density and the purity are checked
  against the integrals of a sinc and against Gaussian formulas, and the
  rate against a quadrature on the filled guide, in {doc}`../verification`,
  "Photon pairs".

## Limits

- The low-gain regime: at most a pair per mode. Above it the pair number is a
  sinh, not a square, and the state is squeezed, which this page does not
  give; `gfem.HighGainWarning` marks the limit.
- Coincidences only. A loss of the arms is in the pair amplitude (the
  coherent sum over where the pair is born), but singles, heralding
  efficiency, the mixedness loss makes and a detector's are not.
- The pump is a continuous wave or a transform-limited Gaussian pulse, with
  no chirp, no other shape and no second pump. The pulsed four-wave mixing
  amplitude omits the self- and cross-phase modulation, valid while
  $\mathrm{Re}\,\gamma_\mathrm{NL}P_\mathrm{peak}L \ll 1$.
- The coupling is one overlap at the bands' centres and a polynomial fit of
  the dispersion, valid over a band of a few percent of the carrier; the
  check above shows the overlap's change. A band whose dispersion is not
  smooth, or that crosses a mode crossing, needs a narrower band.
- A line port (a 2D model) is refused: a pair rate of a slab is per
  transverse wavenumber, not a figure of a guide. The Hz polarization and a
  layered density carrying the nonlinearity are refused as in
  {doc}`second-harmonic-generation`.
- The second-order pair coupling rests on a $d$ with Kleinman symmetry, not
  checked. The poling is analytic: a square wave of one period over the
  guide, full depth.
- Not modelled: Raman scattering, free carriers, two-photon absorption
  beyond the complex coupling, the thermal and photorefractive effects, and
  a pump that is depleted.
- The ring is an estimate and no solve of a ring.
- Under `jax.grad` the figures differentiate with respect to the section's
  materials and the rows' $\gamma$, not the node coordinates.

## Next

- {doc}`second-harmonic-generation`: the classical efficiency of the same
  overlap, and the tensor of lithium niobate.
- {doc}`waveguide-nonlinearity`: the Kerr coefficient and the couplings of the
  silicon strip.
- {doc}`waveguide-dispersion`: the group index and the dispersion of a mode.
- {doc}`../conventions`: the formulas and the normalization.
- {py:obj}`gfem.spdc`, {py:obj}`gfem.sfwm`, {py:obj}`gfem.PhotonPairSource`,
  {py:obj}`gfem.JointSpectrum`.
