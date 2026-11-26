# Free carriers in silicon

Find what a port mode gives a carrier rate equation: how many electron-hole
pairs two-photon absorption makes per watt squared, and how much a unit density
of pairs changes the mode's loss and index. `gfem.free_carriers` reads the
{py:obj}`gfem.PortModeResult` of a wave port and the model's materials, at no
cost beyond the mode's own solve. The carrier law and the lifetime stay with
the pulse tool that integrates the rate equation.

The guide is the silicon strip of {doc}`waveguide-nonlinearity`, 500 × 220 nm
in oxide. Units are SI: $N$ counts pairs per m³ ($N_e = N_h = N$).

## The figures of a mode

The carriers are made by two-photon absorption and spread uniformly over the
regions you name (the core), which needs a `two_photon_absorption`. Read the
figures on every row of the mode's grid:

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


# Salzberg and Villa (1957), three-term fit by Tatian (1984); Malitson (1965)
silicon = gfem.Material(
    eps_r=sellmeier(
        (10.6684293, 0.0030434748, 1.54133408), (0.301516485, 1.13475115, 1104.0)
    ),
    # Dinu, Quochi and Garcia, Appl. Phys. Lett. 82, 2954 (2003): silicon at 1.54 um
    two_photon_absorption=8e-12,
)
oxide = gfem.Material(
    eps_r=sellmeier((0.6961663, 0.4079426, 0.8974794), (0.0684043, 0.1162414, 9.896161))
)

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/strip_section.py"
)
urllib.request.urlretrieve(url, "strip_section.py")
from strip_section import strip_section  # noqa: E402

model = gfem.Model(
    strip_section(500e-9, height=220e-9),
    materials={"core": silicon, "cladding": oxide},
    ports=[gfem.WavePort("wg", "front")],
)
wavelength = np.array([1.45e-6, 1.55e-6, 1.65e-6])  # m
mode = gfem.solve(
    model, gfem.PortMode("wg", c / wavelength), options=gfem.SolverOptions(order=3)
)
fc = gfem.free_carriers(model, mode, regions="core")
print(fc)
print(f"core area {fc.area * 1e12:.4f} um^2")
for i, wl in enumerate(wavelength):
    print(
        f"{wl * 1e9:.0f} nm: overlap {fc.overlap[i]:.3f}, "
        f"generation {fc.generation[i]:.3e} /(W^2 m^3 s)"
    )
```

```text
FreeCarriers(port 'wg', mode 1, 3 frequencies, regions ('core',), overlap 1.052-1.075, generation 3.046e+33-3.102e+33 1/(W^2 m^3 s))
core area 0.1100 um^2
1450 nm: overlap 1.075, generation 3.102e+33 /(W^2 m^3 s)
1550 nm: overlap 1.067, generation 3.098e+33 /(W^2 m^3 s)
1650 nm: overlap 1.052, generation 3.046e+33 /(W^2 m^3 s)
```

- `overlap` is the modal confinement $\Gamma_R$ of the carriers:
  $\Gamma_R = \frac{c\varepsilon_0}{2}\int_R n\,|E|^2\,\mathrm{d}A$ for the
  1 W mode, over all three field components. A uniform index change
  $\Delta\tilde n = \Delta n_\mathrm{FC} - i\sigma N/(2k_0)$ in $R$ changes the
  effective index by $\Gamma_R\Delta\tilde n$: the propagation constant by
  $k_0\Gamma_R\Delta n_\mathrm{FC}$ (negative in silicon) and the power loss
  per metre by $\Gamma_R\sigma N$. It can exceed 1, because a strongly
  guided mode has $n_g > n$.
- `generation` is $\theta$ in $\mathrm{d}N/\mathrm{d}t = \theta P^2 - N/\tau$
  with $P$ the power in the guide. It is
  $\mathrm{Im}\,\gamma_{\mathrm{NL},R}/(\hbar\omega A_R)$: the imaginary part of
  `kerr_nonlinearity`'s coefficient from the core, one pair per two photons
  absorbed, divided by the core area `area`. It is exactly linear in
  `two_photon_absorption`.
- Without `regions`, the carriers fill every region with a
  `two_photon_absorption`. Regions with none are refused: no generation.

## What the user brings

gfem ships no cross-section, dispersion coefficient or lifetime: published
values scatter and the lifetime belongs to the device. Common choices at
1.55 µm, with where each was checked:

| Quantity | Value | Source | Status |
| --- | --- | --- | --- |
| FCA cross-section $\sigma$ | $1.45 \times 10^{-21}$ m² | Lin, Painter and Agrawal, Opt. Express 15, 16604 (2007), eq. 20 | verified in the full text |
| Index change, linear form $\Delta n = -k_c N$ | $k_c = 1.35 \times 10^{-27}$ m³, $\mu = 7.5$ | Yin and Agrawal, Opt. Lett. 32, 2031 (2007) | verified |
| Index change, Lin's $\varsigma$ | $-5.3 \times 10^{-27}$ m³, $\mu \approx 30$ | Lin et al. 2007, eq. 20 | verified |
| Soref-Bennett laws | $\Delta n = -(8.8 \times 10^{-22} N_e + 8.5 \times 10^{-18} N_h^{0.8})$, $\Delta\alpha = 8.5 \times 10^{-18} N_e + 6.0 \times 10^{-18} N_h$ ($N$ in cm⁻³, $\alpha$ in cm⁻¹) | as quoted in Lin et al. 2007, eqs. 18-19, from Soref and Bennett, Proc. SPIE 704 (1987) | the coefficients come through Lin's citation; the SPIE paper was not read. The JQE 23, 123 (1987) paper gives fitted slopes only |
| Nedeljkovic-Soref-Mashanovich fits | $\Delta n = -5.4 \times 10^{-22} N_e^{1.011} - 1.53 \times 10^{-18} N_h^{0.838}$, $\Delta\alpha = 8.88 \times 10^{-21} N_e^{1.167} + 5.84 \times 10^{-20} N_h^{1.109}$ | IEEE Photonics J. 3, 1171 (2011), read in two citing papers | verified at 1550 nm through the citations; the primary is paywalled |
| 1310 nm fits of either | | | unverified |
| Lifetime $\tau$ | sub-nanosecond to tens of ns; about 1 ns in Yin and Agrawal | Lin et al. 2007; Yin and Agrawal 2007 | verified; depends on the waveguide geometry |

- Both $\sigma$ and the dispersion scale as $\lambda^2$ (a Drude
  response): $\sigma(\lambda) = \sigma_r(\lambda/\lambda_r)^2$, Lin et al. 2007,
  eq. 20. Near 1.55 µm the absorption changes by under 14 % over 100 nm.
- The linear dispersion coefficients differ by a factor of about 4, and the
  Soref and Nedeljkovic laws are not linear in $N$: pick the law that fits
  your densities.

## Handoff to a pulse equation

With $A$ the field amplitude, $|A|^2 = P$, the carriers act on the
propagation as

$$
\partial_z A \ni -\frac{\Gamma_R\,\sigma}{2}\,(1 + i\mu)\,N A,
\qquad \mu = \frac{2 k_0 k_c}{\sigma},
$$

in the $e^{-i\omega t}$ convention of Yin and Agrawal, with
$\mathrm{d}N/\mathrm{d}t = \theta P^2 - N/\tau$. `gamma_nl` from
`kerr_nonlinearity` supplies the Kerr and two-photon terms of the same
equation. For a continuous wave the density settles at $N = \theta\tau P^2$,
and the modal absorption of the carriers is $\Gamma_R\sigma\theta\tau P^2$:

```python
sigma = 1.45e-21 * (wavelength / 1.55e-6) ** 2  # m^2, scaled as lambda^2
tau, power = 1e-9, 0.1  # s, W
density = fc.generation * tau * power**2  # pairs per m^3
loss = fc.overlap * sigma * density  # 1/m, power
for i, wl in enumerate(wavelength):
    print(
        f"{wl * 1e9:.0f} nm: N {density[i]:.2e} /m^3, "
        f"free-carrier loss {4.343e-2 * loss[i]:.3f} dB/cm"
    )
```

```text
1450 nm: N 3.10e+22 /m^3, free-carrier loss 1.837 dB/cm
1550 nm: N 3.10e+22 /m^3, free-carrier loss 2.081 dB/cm
1650 nm: N 3.05e+22 /m^3, free-carrier loss 2.286 dB/cm
```

## Limits

- Carriers are uniform over the regions you name. Carriers that stay where
  they are made (pulses shorter than the diffusion time across the core) need
  a different overlap, which is not computed.
- Generation is by degenerate two-photon absorption of one mode. A pump and
  a probe at different frequencies are not combined.
- On lossy and leaky sections the figures hold while the loss is small
  against the propagation constant.
- Anisotropic regions, a density blend without a per-element permittivity and
  the hybrid Hz line port are refused.
- The overlaps differentiate through the materials and the mode field, not
  through the node coordinates.

## Next

- {doc}`waveguide-nonlinearity`: the Kerr coefficient and the two-photon term of the same guide.
- {doc}`raman-gain`: the other gain figure read from a port mode.
- {py:obj}`gfem.free_carriers`, {py:obj}`gfem.FreeCarriers`.
