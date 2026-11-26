# Photon-pair sources

Find how many pairs of photons a waveguide emits when it is pumped, in which
frequencies, and how pure each photon is once its partner is detected.
`gfem.spdc` (down-conversion, second order) and `gfem.sfwm` (four-wave mixing,
third order) return a {py:obj}`gfem.PhotonPairSource`.

The guides are a poled lithium niobate ridge after Zhao et al., Phys. Rev.
Lett. 124, 163603 (2020), and the silicon strip of {doc}`waveguide-nonlinearity`.
Units are SI ({doc}`../conventions`).

## The crystal and the ridge

The tensor, oxide and ridge are those of {doc}`second-harmonic-generation`,
with the paper's geometry and $d_{33}$ = 27 pm/V:

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

## The modes of the pump and of the pair

The pair is degenerate near 1570 nm. Rank the pump's modes at 785 nm by their
second-harmonic efficiency with the signal mode:

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

The fundamental couples. Each arm is solved on a band, for the dispersion:

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

- Each arm's rows are fitted by a polynomial of degree at most 4; for
  degenerate photons one result is signal and idler.

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

- `signal is idler` means identical photons, counted once. The bands are the
  grid of every figure and act as ideal filters.
- `poling_period` is the first order; the poling is an argument of the
  figures, so a design sweep costs no overlap.

## The rate and the spectral brightness of a continuous pump

A pump of 1 mW in the 5 mm guide, poled at that period. The brightness is the
density of signal photons in pairs/(s Hz); the rate is its integral:

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

The density is also the mean number of pairs per mode: above 0.1
`gfem.HighGainWarning` says the result is off by more than 3 %. A longer guide
narrows the matched band, a period that is off detunes it:

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

- The matching is quadratic in the detuning, so the rate grows as $L^{3/2}$ at
  $\Delta\beta_0 = 0$ (2.83 for a doubled length). A period 0.1 % off detunes
  it by several radians: longer keeps half the rate, shorter collapses it.

## The joint spectrum of a pulse

A Gaussian pulse of energy $U$ and FWHM $T$ makes a joint spectral amplitude
$f(f_s, f_i)$, with $\sum|f|^2\,\delta f_s\,\delta f_i$ pairs per pulse; the
grid must be fine in the frequency sum. A low-gain pulse is a femtojoule:

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

- The photons are as wide as the matching allows, strongly correlated and far
  from pure; a narrower band is a filter that trades pairs for purity.

The heralded purity is $\sum_k\lambda_k^2$ over the Schmidt weights, the Schmidt
number $K$ its inverse:

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

- The purity is low ($K$ of 30 to 45) with a maximum near 1 ps, where the
  pump's bandwidth matches the matching's width in the sum.

## Plot the joint spectrum

The signal frequency runs down, the idler across, in `gfem.JointSpectrum`:

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

## The gradient of the purity

The purity and the rate differentiate in every argument and, through the fits
and overlaps, in the materials and the rows' $\gamma$, not the node
coordinates ({doc}`port-mode-gradient`):

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

## Four-wave mixing in a silicon wire

The strip, pumped at 1.55 µm, makes pairs through $\gamma_\mathrm{NL}$ with
$2f_p = f_s + f_i$ and a 1 THz guard around the pump:

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

The source uses the symmetrized mean of $\gamma_{sppi}$ and $\gamma_{ipps}$ of
`gfem.kerr_coupling`:

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

- The rate goes as the square of the power, and above about 0.17 W
  `gfem.HighGainWarning` fires; the Kerr shift of the pump is included.

A pulse gives a joint spectrum as well, here cut by the guard band:

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

gfem has no bent mode: the straight guide's coupling and group index give a
coupled-mode estimate for a ring of radius 10 µm, loaded $Q = 10^5$ for all
three waves, critically coupled, pumped by 0.1 mW:

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

- The rate goes as $Q^3/L_\mathrm{ring}^2$; every $\kappa$ is yours to give,
  and two-photon absorption and free carriers are not in it.

## Compared with published sources

Published brightnesses scatter by factors of 4 to 20 around the ideal formulas:
these are observations, not checks.

| Source | Published | gfem |
| --- | --- | --- |
| Zhao et al. 2020 | $4.6\times10^5$ pairs/s/GHz/mW, period 2.8 µm | below |
| Javid et al., Phys. Rev. Lett. 127, 183601 (2021) | 13 GHz/mW total | the rate, other geometry |
| Shi et al., Light Sci. Appl. 13, 282 (2024) | $1.9\times10^4$ to $1.9\times10^5$ Hz/nm/mW$^2$ | silicon wire, 1 cm |

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

- The period agrees within a percent; the brightness is 23 times the
  measurement, which has partial poling depth, a duty cycle of $55.2 \pm 3.0$ %
  and losses. The silicon wire is 2 to 20 times the range.

## Check your results

- **The grid.** A figure converges with `points`
  (`gfem.UnresolvedJointSpectrumWarning` warns below eleven points per lobe):

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

- **The fit.** A residual phase above 0.1 rad raises; the residual in radians
  for pump, signal and idler:

  ```python
  print(np.asarray(source.fit_residual) * length)
  ```

  ```text
  [3.72633599e-04 3.68914427e-05 3.68914427e-05]
  ```

- **The overlap over the band.** Recompute the coupling at the band's edges:

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

  A few tenths of a percent, below the other uncertainties.
- **The port mode tolerance.** $\Delta\beta L$ must stay far below 0.1 rad
  with the bands re-solved $10^3$ looser:

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

  It moves at the $10^{-5}$ rad level; the default is $10^3$ tighter.
- **The classical figure of merit.** At degeneracy and perfect matching the
  density at the centre is $|G_1|^2\eta_\mathrm{norm}PL^2$, the
  second-harmonic efficiency of the same overlap:

  ```python
  centre = float(brightness[len(brightness) // 2])
  classical = 4 / np.pi**2 * float(shg[1].eta_norm[0]) * power * length**2
  print(f"density at the centre {centre:.6g}, classical {classical:.6g}")
  ```

  ```text
  density at the centre 0.0107483, classical 0.0107483
  ```

- **Closed forms.** The rate, density and purity are checked in
  {doc}`../verification`.

## Limits

- **Low gain.** At most a pair per mode; beyond it the state is squeezed
  (`gfem.HighGainWarning`). Coincidences only: singles and heralding efficiency
  are not given.
- **Pump and band.** A continuous wave or transform-limited Gaussian pulse; one
  overlap and a polynomial fit, valid over a few percent of the carrier.
- **Not modelled.** Line ports, the Hz polarization, Raman, free carriers,
  thermal effects and a depleted pump.

## Next

{doc}`second-harmonic-generation`, {doc}`waveguide-nonlinearity`,
{doc}`waveguide-dispersion`, {doc}`../conventions`,
{py:obj}`gfem.spdc`, {py:obj}`gfem.sfwm`.
