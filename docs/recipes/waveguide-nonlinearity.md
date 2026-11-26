# Kerr nonlinearity

Find how a guided mode changes the phase and the loss of its own light, and
how its waves exchange power: the {term}`Kerr nonlinear coefficient`
$\gamma_\mathrm{NL}$ of a waveguide mode, its two-photon loss, the coupling
between the waves of a four-wave mixing process, and the parameters of a
microring's Lugiato-Lefever equation. `gfem.kerr_nonlinearity` reads a
{py:obj}`gfem.PortModeResult` and the model's materials: it needs no solve
beyond the mode's own, at every frequency of its grid.

The guide is the silicon strip of {doc}`waveguide-dispersion` near 1.55 µm;
units are SI ({doc}`../conventions`).

## Silicon and oxide, with their nonlinearity

Two more numbers on a material, `nonlinear_index` ($n_2$, m²/W) and
`two_photon_absorption` ($\beta_\mathrm{TPA}$, m/W), join the Sellmeier laws
of the dispersion recipe:

```python
import dataclasses
import urllib.request

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


# Salzberg and Villa (1957), three-term fit by Tatian (1984): 1.36 to 11 um
silicon = gfem.Material(
    eps_r=sellmeier(
        (10.6684293, 0.0030434748, 1.54133408), (0.301516485, 1.13475115, 1104.0)
    )
)
# Malitson (1965): 0.21 to 3.71 um
oxide = gfem.Material(
    eps_r=sellmeier((0.6961663, 0.4079426, 0.8974794), (0.0684043, 0.1162414, 9.896161))
)
# Dinu, Quochi and Garcia, Appl. Phys. Lett. 82, 2954 (2003): silicon at 1.54 um
silicon_kerr = dataclasses.replace(
    silicon, nonlinear_index=4e-18, two_photon_absorption=8e-12
)
print(silicon_kerr.nonlinear_index, silicon_kerr.two_photon_absorption)
```

```text
4e-18 8e-12
```

- Both are real, isotropic constants of the material over the grid (a graded
  `gfem.MaterialField` carries them per element); an element without them
  counts as zero, so the oxide adds nothing until you give it one.
- The values are a choice: $n_2$ of silicon varies by a factor of four between
  850 and 2200 nm (Bristow et al., Appl. Phys. Lett. 90, 191104 (2007)); read
  a value's source and wavelength first.

## The nonlinear coefficient over a band

{download}`strip_section.py <strip_section.py>` meshes the strip with netgen;
its end face `"front"` carries the wave port. Solve the mode on a grid, then
read $\gamma_\mathrm{NL}$ on every row of it:

```python
url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/strip_section.py"
)
urllib.request.urlretrieve(url, "strip_section.py")
from strip_section import strip_section  # noqa: E402

options = gfem.SolverOptions(order=3)


def strip(width, clearance=1e-6, core=silicon_kerr, cladding=oxide):
    """The strip, 220 nm thick, as a model with a wave port on its section."""
    return gfem.Model(
        strip_section(width, height=220e-9, clearance=clearance),
        materials={"core": core, "cladding": cladding},
        ports=[gfem.WavePort("wg", "front")],
    )


model = strip(500e-9)
wavelength = np.linspace(1.45e-6, 1.65e-6, 21)  # m
mode = gfem.solve(model, gfem.PortMode("wg", c / wavelength), options=options)
kerr = gfem.kerr_nonlinearity(model, mode)
print(kerr)
gamma = kerr.gamma_nl  # 1/(W m), complex
k0 = 2 * np.pi / wavelength
a_eff = k0 * silicon_kerr.nonlinear_index / gamma.real  # m^2
for i in (0, 5, 10, 15, 20):
    print(
        f"{wavelength[i] * 1e9:.0f} nm: gamma {gamma[i].real:.1f} /(W m), "
        f"A_eff {a_eff[i] * 1e12:.4f} um^2"
    )
```

```text
KerrNonlinearity(port 'wg', mode 1, 21 frequencies, gamma_nl Re 153.6-202.5, Im 40.33-46.74 1/(W m))
1450 nm: gamma 202.5 /(W m), A_eff 0.0856 um^2
1500 nm: gamma 189.5 /(W m), A_eff 0.0884 um^2
1550 nm: gamma 177.0 /(W m), A_eff 0.0916 um^2
1600 nm: gamma 165.1 /(W m), A_eff 0.0951 um^2
1650 nm: gamma 153.6 /(W m), A_eff 0.0992 um^2
```

- `gamma_nl` is the complex coefficient of
  $\mathrm{d}A/\mathrm{d}z = i\,\gamma_\mathrm{NL}|A|^2 A$, $|A|^2$ in W: a
  power $P$ shifts the propagation constant by
  $\mathrm{Re}\,\gamma_\mathrm{NL}\,P$.
- The integral runs over all three components of the mode, so it holds for a
  silicon wire, where the weak-guidance $\gamma = k_0 n_2 / A_\mathrm{eff}$ is
  wrong by tens of percent.

## Two-photon absorption and the figure of merit

The imaginary part is two-photon absorption: $\gamma_\mathrm{TPA} =
\mathrm{Im}\,\gamma_\mathrm{NL} \ge 0$ is the field attenuation per watt, so
$\mathrm{d}P/\mathrm{d}z = -2\gamma_\mathrm{TPA}P^2$. The ratio of the two
parts is the material's figure of merit, independent of the mode:

```python
i = 10  # 1.55 um
gamma_tpa = gamma[i].imag
fom = gamma[i].real / (4 * np.pi * gamma_tpa)
n2, beta_tpa = silicon_kerr.nonlinear_index, silicon_kerr.two_photon_absorption
print(f"gamma_TPA {gamma_tpa:.2f} /(W m)")
print(
    f"figure of merit n2/(lambda beta_TPA): {fom:.3f}, material {n2 / (wavelength[i] * beta_tpa):.3f}"
)
for power in (1e-3, 1e-2, 1e-1):
    loss_db_cm = 10 * np.log10(np.e) * 2 * gamma_tpa * power * 1e-2
    print(f"{power * 1e3:g} mW: two-photon loss {loss_db_cm:.4f} dB/cm")
```

```text
gamma_TPA 43.67 /(W m)
figure of merit n2/(lambda beta_TPA): 0.323, material 0.323
1 mW: two-photon loss 0.0038 dB/cm
10 mW: two-photon loss 0.0379 dB/cm
100 mW: two-photon loss 0.3793 dB/cm
```

- $\mathrm{Re}\,\gamma_\mathrm{NL}/(4\pi\gamma_\mathrm{TPA}) =
  n_2/(\lambda\beta_\mathrm{TPA})$ when the core carries both: 0.32 for the
  page's silicon, well below 1, so its two-photon loss is a large part of its
  Kerr effect.
- The loss grows as the square of the power: 0.38 dB/cm at 100 mW.

## Three waves: phase mismatch and couplings

`gfem.kerr_coupling` takes the three rows of a `PortMode` result (a pump and
sidebands at $\omega_\mathrm{p} \pm \Omega$) and returns the tensor
$\gamma_{abcd}$ between them; the exact phase mismatch is $\Delta\beta = 2\beta_\mathrm{p} - \beta_\mathrm{s} -
\beta_\mathrm{i}$:

```python
f_p = c / 1.55e-6
shift = 5e12  # Omega / 2 pi, Hz
freqs = np.array([f_p - shift, f_p, f_p + shift])  # signal, pump, idler
trio = gfem.solve(model, gfem.PortMode("wg", freqs), options=options)
s, p, idl = 0, 1, 2
beta = trio.gamma.imag
d_beta = 2 * beta[p] - beta[s] - beta[idl]
omega = 2 * np.pi * shift
taylor = -trio.beta_2[p].real * omega**2
print(f"Delta beta exact {d_beta:.1f} rad/m, -beta_2 Omega^2 {taylor:.1f} rad/m")
```

```text
Delta beta exact 605.1 rad/m, -beta_2 Omega^2 610.4 rad/m
```

The two differ by the fourth-order term $-\beta_4\Omega^4/12$, so at a shift of
5 THz use the exact value. Then the couplings:

```python
tensor = gfem.kerr_coupling(model, trio).gamma_nl  # (3, 3, 3, 3), 1/(W m)
spm = tensor[p, p, p, p].real
xpm = (tensor[s, s, p, p] + tensor[s, p, s, p]).real
fwm = tensor[s, p, p, idl].real
print(f"pump SPM {spm:.1f}, signal XPM by the pump {xpm:.1f}, FWM {fwm:.1f} /(W m)")
print(f"Kerr shift matches the mismatch at {d_beta / (2 * spm):.2f} W")
print(
    f"SPM equals gamma_NL of the same row: {tensor[p, p, p, p]:.2f} vs {gamma[10]:.2f}"
)
```

```text
pump SPM 177.0, signal XPM by the pump 339.2, FWM 171.9 /(W m)
Kerr shift matches the mismatch at 1.71 W
SPM equals gamma_NL of the same row: 177.04+43.67j vs 177.04+43.67j
```

- Cross-phase modulation of the signal by the pump is
  `[s, s, p, p] + [s, p, s, p]` (1.92 times the self-phase here); four-wave
  mixing into the signal is `[s, p, p, i]` ({doc}`../nonlinear-and-quantum-optics`).
- The Kerr shift of a pump of power $P_\mathrm{p}$ matches the phases when
  $\Delta\beta = 2\,\mathrm{Re}\,\gamma_\mathrm{NL} P_\mathrm{p}$: the power that
  does it here is `d_beta / (2 * spm)`, possible when $\beta_2 < 0$.
- The cost is $K^4$ per quadrature point: pass the few waves of a process.

## Parameters of a microring

A ring of radius $R$ of this guide has resonances spaced by
$D_1 = c/(n_g R)$, $D_2 = -(c/n_g)\,D_1^2\,\beta_2$ and a Kerr shift per photon
$g_0 = \hbar\omega_0\,c^2\,\mathrm{Re}\,\gamma_\mathrm{NL}/(n_g^2 L)$, $L = 2\pi R$:
the numbers of the Lugiato-Lefever equation. A ring critically coupled at an
intrinsic quality factor $Q_\mathrm{i}$ has $\kappa = 2\omega_0/Q_\mathrm{i}$ and a
parametric threshold $P_\mathrm{th} = \kappa^2\hbar\omega_0/(8\eta g_0)$ with
$\eta = 1/2$ (take $Q_\mathrm{i}$ from a measurement or a coupler model):

```python
radius = 10e-6  # m
length = 2 * np.pi * radius
n_g, beta_2 = mode.n_g[10].real, mode.beta_2[10].real
omega0 = 2 * np.pi * f_p
d1 = c / (n_g * radius)
d2 = -(c / n_g) * d1**2 * beta_2
g0 = scipy.constants.hbar * omega0 * c**2 * gamma[10].real / (n_g**2 * length)
q_i = 1e5  # intrinsic quality factor: an assumption, not a result
kappa = 2 * omega0 / q_i
p_th = kappa**2 * scipy.constants.hbar * omega0 / (8 * 0.5 * g0)
print(f"FSR {d1 / (2 * np.pi) / 1e9:.1f} GHz, D2/2pi {d2 / (2 * np.pi) / 1e6:.2f} MHz")
print(
    f"g0/2pi {g0 / (2 * np.pi):.0f} Hz, threshold {p_th * 1e3:.2f} mW at Q_i = {q_i:.0e}"
)
```

```text
FSR 1137.1 GHz, D2/2pi 359.02 MHz
g0/2pi 293 Hz, threshold 10.27 mW at Q_i = 1e+05
```

A straight guide approximates the ring's mode: $D_1$ is good to about
$10^{-3}$ and $D_2$ to 5 to 15 % at a 10 µm silicon ring, and not at all for a
ring of 1 THz free spectral range. gfem has no bent port mode; the estimates
are on {doc}`../nonlinear-and-quantum-optics`.

## Check your results

- **The cladding's clearance.** The mode's tail sits in the oxide, so the
  coefficient converges with the clearance:

  ```python
  for clearance in (0.5e-6, 1.0e-6, 1.5e-6):
      m = strip(500e-9, clearance=clearance)
      r = gfem.solve(m, gfem.PortMode("wg", f_p), options=options)
      g = gfem.kerr_nonlinearity(m, r).gamma_nl[0]
      print(f"clearance {clearance * 1e6:.1f} um: gamma {g.real:.2f} /(W m)")
  ```

  ```text
  clearance 0.5 um: gamma 176.81 /(W m)
clearance 1.0 um: gamma 177.04 /(W m)
clearance 1.5 um: gamma 177.04 /(W m)
  ```

  At 0.5 µm the coefficient is 0.13 % low; 1.0 µm has settled. It needs less
  clearance than $\beta_2$ does ({doc}`waveguide-dispersion`).
- **The linearity in $n_2$.** `kerr_nonlinearity` reads the model's
  materials, not the solve: changing them re-reads the same mode.

  ```python
  doubled = dataclasses.replace(silicon_kerr, nonlinear_index=2 * n2)
  twice = gfem.kerr_nonlinearity(strip(500e-9, core=doubled), mode)
  print(f"{twice.gamma_nl[10].real / gamma[10].real - 1:.6f}")
  ```

  ```text
  1.000000
  ```

  Doubling $n_2$ doubles the real part; the imaginary part, from
  $\beta_\mathrm{TPA}$, is unchanged.
- **A finer mesh.** The page's mesh gives 177.040 + 43.674i /(W m) at 1.55 µm;
  `gfem.SolverOptions(order=4)` gives 177.056 + 43.678i, a move of $10^{-4}$.
- **Closed forms.** The integral is checked on a filled rectangular guide and a
  slab ({doc}`../verification`).

## Limits

- **One scalar per material.** A single $n_2$ and $\beta_\mathrm{TPA}$, no
  tensor $\chi^{(3)}$ (silicon's $n_2$ depends on the crystal direction): an
  anisotropic or `axes`-rotated region carrying either is refused.
- **What is left out.** Free-carrier absorption and dispersion, Raman (the
  delayed response of silica, the 15.6 THz line of silicon) and heating; a
  pulse simulation takes $\gamma_\mathrm{NL}$ from here and adds them.
- **Loss.** Lossy and leaky sections run, valid while the attenuation is far
  below the phase constant.
- **2D.** A line port gives the slab's value in 1/W; the Hz polarization and a
  `Blend.pixels` density carrying a Kerr property are refused.
- **Gradients.** $\gamma_\mathrm{NL}$ differentiates with respect to the
  materials and to the node coordinates of a planar port (a width, a height).
  Line ports, the three-wave and photon-pair overlaps and `jax.jit` over the
  nodes do not; an objective that depends on a mode's phase raises.

## Next

{doc}`waveguide-dispersion`, {doc}`../conventions`,
{doc}`port-mode-gradient`, {py:obj}`gfem.kerr_nonlinearity`,
{py:obj}`gfem.kerr_coupling`.
