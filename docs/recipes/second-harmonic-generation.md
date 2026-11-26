# Second-harmonic generation

Find how efficiently a guided mode doubles its frequency in a crystal with a
second-order nonlinearity: the normalized efficiency, the phase mismatch, the
poling period that cancels it, and the harmonic power along a poled guide, over
a band of pump wavelengths. `gfem.second_harmonic` reads the pump's and the
harmonic's {py:obj}`gfem.PortModeResult` and the model's materials.

The guide is the x-cut thin-film lithium niobate ridge of Wang et al., Optica
5, 1438 (2018): 600 nm film, 300 nm etch, 1.4 µm wide, quasi-TE at 1.55 µm into
quasi-TE at 775 nm. Units are SI ({doc}`../conventions`).

## Lithium niobate and its $d$ tensor

The tensor $d_{il}$ goes on the material beside its permittivity, made at the
pump's and at the harmonic's frequency; `dataclasses.replace` adds it to
`gfem.materials.lithium_niobate`. The x-cut film has its optic axis ($z$)
along the chip's horizontal axis:

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


# Malitson (1965): 0.21 to 3.71 um
oxide = gfem.Material(
    eps_r=sellmeier((0.6961663, 0.4079426, 0.8974794), (0.0684043, 0.1162414, 9.896161))
)

d33, d31, d22 = 25e-12, 4.6e-12, 2.2e-12  # m/V, Wang et al. 2018
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


film = lithium_niobate(c / 1.55e-6)
print(np.sqrt(film.eps_r), film.second_order_nonlinear_tensor[2][:3])
```

```text
[2.21111101 2.21111101 2.13755965] (4.6e-12, 4.6e-12, 2.5e-11)
```

Lithium niobate has four independent numbers ($d_{15} = d_{31}$ by Kleinman
symmetry), read in Wang's text after Shoji et al. 1997; $d$ varies with
wavelength, composition and measurement.

| Coefficient | Value (pm/V) |
| --- | --- |
| $d_{33}$ | 25 |
| $d_{31}$ | 4.6 |
| $d_{22}$ | 2.2 |

- The principal indices are $(n_o, n_o, n_e)$ along $x, y, z$: the quasi-TE
  mode, polarized across the ridge, sees $n_e$ and $d_{33}$.
- A tensor above $10^{-8}$ m/V (pm/V typed as a number), a traced value in a
  `Material` (use `gfem.MaterialField`) and left-handed axes are refused. A
  domain of reversed polarization carries $-d$; the poling is a parameter of
  `efficiency`, not of the material.

### Miller's rule

$d$ is not constant with the wavelength: it scales as the product of
$\chi^{(1)} = n^2 - 1$ at the three frequencies. For $d_{33}$ from the 25.2
pm/V of 1064 nm to this pump:

```python
def chi(frequency):
    """n_e^2 - 1 of lithium niobate."""
    return gfem.materials.lithium_niobate(frequency=frequency, axes=X_CUT).eps_r[2] - 1


def miller(wavelength):
    """chi(w)^2 chi(2w) of SHG at a pump wavelength, m."""
    return chi(c / wavelength) ** 2 * chi(2 * c / wavelength)


factor = miller(1.55e-6) / miller(1.064e-6)
print(f"Miller factor {factor:.4f}: d33 {25.2 * factor:.1f} pm/V at 1550 nm")
```

```text
Miller factor 0.8996: d33 22.7 pm/V at 1550 nm
```

Wang's 25 pm/V is about 10 % above the rule at 1550 nm; the choice is yours.

## The guide's modes

{download}`ridge_section.py <ridge_section.py>` meshes the ridge with netgen;
its end face `"front"` carries a wave port. The pump and the harmonic each get a
model of the same mesh, with the crystal's indices at their frequency:

```python
url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/ridge_section.py"
)
urllib.request.urlretrieve(url, "ridge_section.py")
from ridge_section import ridge_section  # noqa: E402

mesh = ridge_section(1.4e-6)
options = gfem.SolverOptions(order=3)


def guide(frequency, film=None):
    """The ridge with the crystal's indices at ``frequency``, wave port on its section."""
    return gfem.Model(
        mesh,
        materials={
            "film": film or lithium_niobate(frequency),
            "box": oxide,
            "air": gfem.Material(eps_r=1.0),
        },
        ports=[gfem.WavePort("wg", "front")],
    )


f0 = c / 1.55e-6
pump_model, harmonic_model = guide(f0), guide(2 * f0)
pump = gfem.solve(pump_model, gfem.PortMode("wg", np.array([f0])), options=options)
print(pump)
```

```text
PortModeResult(port 'wg', mode 1, 1 frequencies 193.4 THz (1 exact solves), n_eff 1.9063-1.9063, n_g 2.2395-2.2395, no voltage_path)
```

The harmonic's fundamental is not the coupled mode: the quasi-TM mode at 775 nm
lies above the quasi-TE one, its field across the ridge's other axis, where
$d_{33}$ does not act on the pump's field. Rank the harmonic's modes and read
each one's overlap with the pump:

```python
harmonics = {}
for rank in (1, 2, 3):
    problem = gfem.PortMode("wg", np.array([2 * f0]), mode=rank)
    harmonics[rank] = gfem.solve(harmonic_model, problem, options=options)
    sh = gfem.second_harmonic(pump_model, pump, harmonics[rank])
    print(
        f"harmonic mode {rank}: n_eff {harmonics[rank].n_eff[0].real:.4f}, "
        f"eta_norm {sh.eta_norm[0] * 1e-2:.3g} %/(W cm^2)"
    )
```

```text
harmonic mode 1: n_eff 2.1634, eta_norm 0.0112 %/(W cm^2)
harmonic mode 2: n_eff 2.1228, eta_norm 20.8 %/(W cm^2)
harmonic mode 3: n_eff 2.1010, eta_norm 1.1e+04 %/(W cm^2)
```

- A mode nearly orthogonal to the pump's square gives an efficiency a million
  times lower, as the first one here. The coupled mode is the quasi-TE
  fundamental, the third at 775 nm.
- `second_harmonic` pairs row $i$ of the pump with row $i$ of the harmonic,
  whose frequency must be twice the pump's to $10^{-9}$ or it raises
  `ValueError`.

## The figures of the pair

```python
harmonic = harmonics[3]
shg = gfem.second_harmonic(pump_model, pump, harmonic)
print(shg)
beta_p, beta_h = pump.gamma[0].imag, harmonic.gamma[0].imag
print(f"Delta beta {shg.delta_beta[0]:.4g} 1/m = beta_2w - 2 beta_w")
print(f"|kappa| {abs(shg.kappa[0]):.4g} 1/(W^1/2 m)")
print(f"first-order poling period {shg.poling_period[0] * 1e6:.4f} um")
```

```text
SecondHarmonic(port 'wg', modes 1 -> 3, 1 frequencies, eta_norm 1.103e+04 %/(W cm^2), poling period 3.98 um)
Delta beta 1.579e+06 1/m = beta_2w - 2 beta_w
|kappa| 1050 1/(W^1/2 m)
first-order poling period 3.9797 um
```

- $\eta_\mathrm{norm} = |\kappa|^2$ gives $P_{2\omega} =
  \eta_\mathrm{norm}P^2L^2$ with no mismatch and no loss: a figure of the mode
  pair and the crystal, not a measured efficiency.
- Wang et al. quote 4500 %/(W cm²) and a period of 4 µm; this guide, with an
  assumed etch depth, gives the same order and a period within a few percent,
  not the same device. A third-order period is three times longer, at a ninth
  of the overlap.

## The harmonic along a poled guide

`efficiency` is the harmonic power over the pump power at the input, for the
undepleted pump of {doc}`../nonlinear-and-quantum-optics`. Poled with period
$\Lambda$, duty cycle $D$ and order $m$ it has the factor $|G_m|^2$ and the
residual mismatch $|\Delta\beta| - 2\pi m/\Lambda$:

```python
length, power = 4e-3, 1e-3  # m, W (Wang et al.: 4 mm)
period = shg.poling_period[0]
coherence = np.pi / shg.delta_beta[0]  # the best length of an unpoled guide
print(f"coherence length {coherence * 1e6:.3f} um")
for label, kwargs in (
    ("unpoled, one coherence length", {}),
    ("poled, D = 1/2", {"period": period}),
    ("poled, D = 0.39", {"period": period, "duty_cycle": 0.39}),
    ("period 1 % long", {"period": 1.01 * period}),
):
    long = coherence if not kwargs else length
    ratio = shg.efficiency(long, power, **kwargs)[0]
    print(f"{label}: {100 * ratio:.4g} % of {power * 1e3:g} mW")
```

```text
coherence length 1.990 um
unpoled, one coherence length: 1.77e-07 % of 1 mW
poled, D = 1/2: 0.7154 % of 1 mW
poled, D = 0.39: 0.6333 % of 1 mW
period 1 % long: 1.692e-05 % of 1 mW
```

- The unpoled harmonic grows over a coherence length $\pi/\Delta\beta$ of two
  micrometres, then falls back, six orders of magnitude below the poled
  guide's, whose factor is $|G_1|^2 = 4/\pi^2$ at a duty cycle of one half.
- A period 1 % off loses most of the efficiency over 4 mm: its relative error
  must stay of order $10^{-4}$.
- `efficiency` includes the modes' loss and differentiates in every argument.

## Efficiency over a band

Each wavelength has its own models, since the crystal's index changes. Solve
five wavelengths within 2 nm of 1.55 µm and read the efficiency of the guide
poled at the period of the band's centre:

```python
wavelength = np.linspace(1.548e-6, 1.552e-6, 5)
rows = []
for lam in wavelength:
    f = c / lam
    model = guide(f)
    p = gfem.solve(model, gfem.PortMode("wg", np.array([f])), options=options)
    h_model = guide(2 * f)
    h = gfem.solve(
        h_model, gfem.PortMode("wg", np.array([2 * f]), mode=3), options=options
    )
    rows.append(gfem.second_harmonic(model, p, h))
delta_beta = np.array([float(r.delta_beta[0]) for r in rows])
ratio = np.array([float(r.efficiency(length, power, period=period)[0]) for r in rows])
for lam, db, r in zip(wavelength, delta_beta, ratio, strict=True):
    print(f"{lam * 1e9:.1f} nm: Delta beta {db:.4g} 1/m, efficiency {100 * r:.3g} %")
```

```text
1548.0 nm: Delta beta 1.579e+06 1/m, efficiency 0.542 %
1549.0 nm: Delta beta 1.579e+06 1/m, efficiency 0.669 %
1550.0 nm: Delta beta 1.579e+06 1/m, efficiency 0.715 %
1551.0 nm: Delta beta 1.579e+06 1/m, efficiency 0.667 %
1552.0 nm: Delta beta 1.578e+06 1/m, efficiency 0.541 %
```

- The group-velocity mismatch, $\mathrm{d}\Delta\beta/\mathrm{d}\omega =
  (n_{g,2\omega} - n_{g,\omega})/c$, sets the bandwidth: a longer guide is
  narrower. The harmonic's mode is rank 3 here: check it when the geometry
  changes.

## Sum and difference frequency, and parametric processes

`gfem.three_wave_coupling` gives the whole tensor $\kappa_{abc}$ between the
waves, each one row of a result:

```python
tensor = gfem.three_wave_coupling(pump_model, [pump, harmonic])
print(tensor)
print(f"|kappa[h, p, p]| {abs(tensor.kappa[1, 0, 0]):.4g} 1/(sqrt(W) m)")
print(f"|second_harmonic kappa| {abs(shg.kappa[0]):.4g}")
swapped = tensor.kappa.transpose(0, 2, 1)
print(f"symmetric in its two amplitudes: {bool(np.allclose(tensor.kappa, swapped))}")
```

```text
ThreeWaveCoupling(port 'wg', 2 waves, |kappa| up to 1178 1/(sqrt(W) m))
|kappa[h, p, p]| 1050 1/(sqrt(W) m)
|second_harmonic kappa| 1050
symmetric in its two amplitudes: True
```

- $\kappa_{abc}$ is the coefficient of $A_bA_c$ in the equation of wave $a$ for
  $f_a = f_b + f_c$; other entries are overlaps of waves that are not coupled.
  The sum-frequency power is four times that of a second harmonic of the same
  overlap ({doc}`../nonlinear-and-quantum-optics`).
- The results may come from different models of one mesh.

## Check your results

- **The overlap is linear in $d$.** Doubling the tensor quadruples the
  efficiency:

  ```python
  film2 = dataclasses.replace(
      lithium_niobate(f0),
      second_order_nonlinear_tensor=tuple(tuple(2 * v for v in row) for row in d_ln),
  )
  doubled = guide(f0, film2)
  twice = gfem.second_harmonic(doubled, pump, harmonic)
  print(f"{abs(twice.eta_norm[0]) / abs(shg.eta_norm[0]):.6f}")
  ```

  ```text
  4.000000
  ```

- **A region without the tensor is not silent.** It raises `ValueError`
  naming the regions.
- **A finer mesh and a higher order.** `gfem.SolverOptions(order=4)` changes
  the normalized efficiency by about $3 \times 10^{-4}$, as for the Kerr
  coefficient ({doc}`waveguide-nonlinearity`); closed forms are in
  {doc}`../verification`.

## Limits

- **Undepleted pump.** A depleted pump needs the coupled equations, which
  `kappa` and `delta_beta` feed.
- **One real $d$ per material,** constant over the grid, with analytic square
  poling at full depth: no chirp, apodization or domain-width errors.
- **Refused.** Hz line ports and a `Blend.pixels` density carrying the tensor.
- **Gradients** reach materials (including $d$ through `gfem.MaterialField`),
  not node coordinates: remesh and difference.

## Next

{doc}`waveguide-nonlinearity`, {doc}`waveguide-dispersion`,
{doc}`electro-optic-phase`, {doc}`../conventions`,
{py:obj}`gfem.second_harmonic`, {py:obj}`gfem.three_wave_coupling`.
