# Second-harmonic generation

Find how efficiently a guided mode doubles its frequency in a crystal with a
second-order nonlinearity: the normalized efficiency, the phase mismatch, the
poling period that cancels it, and the harmonic power along a poled guide, over
a band of pump wavelengths. `gfem.second_harmonic` reads two
{py:obj}`gfem.PortModeResult` (the pump and the harmonic) and the model's
materials: it needs no solve beyond the modes' own. The same overlap, between
any three waves of a port, is `gfem.three_wave_coupling`, for sum- and
difference-frequency generation and for parametric processes.

The guide is an x-cut thin-film lithium niobate ridge on oxide, in air, after
Wang et al., Optica 5, 1438 (2018): a 600 nm film etched 300 nm deep into a
ridge 1.4 µm wide at its top, the quasi-TE mode at 1.55 µm doubled into the
quasi-TE mode at 775 nm through $d_{33}$. The film thickness and the width are
the paper's; the etch depth and the vertical sidewalls are assumptions (not in
the text read), so the numbers below are those of this guide, not of the
paper's device. Units are SI ({doc}`../conventions`, "Second-order
nonlinearity"): $d$ in m/V, $\kappa$ in 1/(W$^{1/2}$ m), the normalized
efficiency $\eta_\mathrm{norm}$ in 1/(W m²), which the printed
forms give in the usual %/(W cm²) (1 W$^{-1}$m$^{-2}$ is
$10^{-2}$ %/(W cm²)).

## Lithium niobate and its $d$ tensor

The nonlinear coefficient is the contracted tensor $d_{il}$ of a material,
$d = \chi^{(2)}/2$ in m/V, in the frame of its `axes`: three rows
(the polarization $x, y, z$) by six columns (the field pair $xx, yy, zz, yz,
xz, xy$, with no factor 2 on the shear columns). Lithium niobate (class 3m)
has four independent numbers, with $d_{15} = d_{31}$ by Kleinman symmetry.

| Coefficient | Value (pm/V) | Source | Status |
| --- | --- | --- | --- |
| $d_{33}$ | 25 | Wang et al. 2018, after Shoji et al. 1997 | read in Wang's text |
| $d_{31}$ | 4.6 | the same | read in Wang's text |
| $d_{22}$ | 2.2 | the same | read in Wang's text |
| $d_{33}$, $d_{31}$ at 1064 nm | 25.2, 4.6 | Shoji et al., JOSA B 14, 2268 (1997) | not read: a summary of the abstract |
| $d_{33}$, $d_{31}$ at 1313 nm | 19.5, 3.2 | Shoji et al. | not read: the same |

The first three are the ones the page uses. The values are a choice: $d$
depends on the wavelength, the composition and the measurement, and the two
sources above do not agree on how it falls with the wavelength (Miller's rule,
below, gives a gentler fall than the 1313 nm value). Other crystals in the same
form: gallium arsenide ($\bar 4 3 m$) has the single value
$d_{14} = d_{25} = d_{36}$; aluminium and gallium nitride (6mm) have
$d_{15} = d_{24}$, $d_{31} = d_{32}$ and $d_{33}$; potassium titanyl phosphate
(mm2) has $d_{15}, d_{24}, d_{31}, d_{32}, d_{33}$. Read the source of a value,
and its wavelength, before you rely on it.

The tensor goes on the material beside its permittivity, which is the
crystal's optical one at the wavelength (so a material is made at each of the
pump's and the harmonic's frequencies). `gfem.materials.lithium_niobate` has the
index of the crystal and not the nonlinearity; `dataclasses.replace` adds it.
The x-cut film has its optic axis ($z$) along the chip's horizontal axis, in
the section's plane:

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

- The principal indices are $(n_o, n_o, n_e)$ along the crystal's $x, y, z$: the
  film's horizontal direction is $z$, so the quasi-TE mode, polarized across
  the ridge, sees $n_e$ and $d_{33}$.
- The tensor is checked when the material is built: a number above
  $10^{-8}$ m/V is refused (a value in pm/V typed as a number), as is a
  $(3, 3, 3)$ array, a traced value in a `Material` (use
  `gfem.MaterialField`) and left-handed axes. The permittivity must be given
  too, and a `PermittivityLaw` is allowed.
- A domain of reversed polarization (poling) carries $-d$: all four numbers
  change sign, $d_{22}$ too. The poling is a parameter of `efficiency`, below,
  and not of the material.

### Miller's rule

$d$ is not constant with the wavelength. Miller's rule says that
$d(\omega_3; \omega_1, \omega_2) \propto \chi^{(1)}(\omega_1)\chi^{(1)}(\omega_2)
\chi^{(1)}(\omega_3)$, with $\chi^{(1)} = n^2 - 1$ along the polarizations
involved. For $d_{33}$, with the extraordinary index, from the 25.2 pm/V of
1064 nm to the pump of this page:

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

Wang's 25 pm/V, used on this page, is therefore about 10 % above what the rule
gives at 1550 nm. The rule is an empirical one, and which value is right is a
choice of yours.

## The guide's modes

{download}`ridge_section.py <ridge_section.py>` meshes the ridge with netgen;
its end face `"front"` carries a wave port. A model has one $d$ and one
permittivity per element, so the pump and the harmonic each get a model of the
same mesh, with the crystal's indices at their frequency. Solve the pump's
fundamental mode:

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

The harmonic's fundamental is the mode of highest index, but not the one that
is coupled: the quasi-TM mode at 775 nm is above the quasi-TE one in this
film, and its field is across the ridge's other axis, where $d_{33}$ does not
act on the pump's field. Rank the harmonic's modes and read each one's overlap
with the pump:

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

- A mode is picked by rank only if you know which it is. The overlap says: a
  mode whose field is nearly orthogonal to the pump's squared gives an
  efficiency a million times lower, as the first one here. The coupled mode
  is the quasi-TE fundamental, the third of the guide at 775 nm.
- `second_harmonic` takes the pump and the harmonic as two results of the same
  number of rows and pairs row $i$ of one with row $i$ of the other: the
  frequency of the harmonic must be twice the pump's to $10^{-9}$ (relative),
  or it raises `ValueError`.

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

- $\eta_\mathrm{norm} = |\kappa|^2$ is the efficiency of a guide of length
  $L$ at pump power $P$ with no mismatch and no loss:
  $P_{2\omega} = \eta_\mathrm{norm}P^2L^2$. It is a figure of the mode pair and of
  the crystal. It is not a measured efficiency: the wire's loss, the poling's
  quality and its duty cycle are not in it.
- Wang et al. quote 4500 %/(W cm²) as the ideal theory of their device with
  25 and 4.6 pm/V, and their poling period is 4 µm. This guide, with an
  assumed etch depth and vertical sidewalls, gives the value above and a period
  within a few percent of theirs: the same order, not the same device. The
  paper's number is an observation to read against, not a target of this page.
- The period is that of the first order. A third order with a period three
  times longer is easier to pole, at a ninth of the overlap
  ($|G_3|^2 = 4/(9\pi^2)$ at a duty cycle of one half).

## The harmonic along a poled guide

`efficiency` is the harmonic power over the pump power at the guide's input,
the undepleted pump of the closed forms ({doc}`../conventions`, "Second-order
nonlinearity"). Unpoled, the harmonic is a sinc$^2$ of the mismatch; poled
with period $\Lambda$, duty cycle $D$ and order $m$ it has the factor
$|G_m|^2 = (2\sin(\pi m D)/(\pi m))^2$ and the residual mismatch
$|\Delta\beta| - 2\pi m/\Lambda$:

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

- The unpoled guide is far from phase matching: the harmonic grows over a
  coherence length $\pi/\Delta\beta$ of two micrometres, then falls back, and
  never exceeds the value in the first row, six orders of magnitude below the
  poled guide's.
- Quasi-phase matching at the right period puts the power into the harmonic
  along the whole length, at the factor $|G_1|^2 = 4/\pi^2$ of a square-wave
  poling at a duty cycle of one half. A duty cycle of 0.39 loses a little of
  it: $|G_1|^2 = (2\sin(0.39\pi)/\pi)^2$.
- A period 1 % off the matching one loses most of the efficiency over 4 mm:
  the residual mismatch must stay below $\delta L/2 \ll 1$, which over this
  length is a relative error of the period of order $10^{-4}$.
- `efficiency` is the closed form of the undepleted pump with the loss of the
  two modes ($\mathrm{Re}\,\gamma$ of each row) and no other. It is valid
  while the harmonic is a small fraction of the pump, and the attenuation is far
  below the phase constant. It differentiates in every argument, so
  `jax.grad` gives the sensitivity of the efficiency to the period, the length or
  the duty cycle, and the overlap's gradient in $d$ flows through the same
  expression.

## Efficiency over a band

The index of the crystal changes with the wavelength, so each wavelength of a
sweep has its own models: the pump's and the harmonic's group velocities
differ, and the matching moves. Solve five wavelengths within 2 nm of 1.55 µm, each
with its materials, and read the efficiency of the guide poled at the period
of the band's centre:

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

- At each wavelength the poled guide is matched where
  $|\Delta\beta| = 2\pi/\Lambda$; away from it the sinc$^2$ of the residual mismatch
  falls. The sweep's $\Delta\beta$ changes by the group-velocity mismatch of the
  pump and the harmonic, $\mathrm{d}\Delta\beta/\mathrm{d}\omega = (n_{g,2\omega}
  - n_{g,\omega})/c$, which is what sets the bandwidth of the process and why a
  longer guide is narrower.
- The harmonic's mode is picked by rank 3 at every wavelength, as the scan
  showed at the centre: check it when the geometry changes.

## Sum and difference frequency, and parametric processes

`gfem.three_wave_coupling` gives the whole tensor $\kappa_{abc}$ between the
waves (the rows of results), not only the pair of the second harmonic. A wave
is one row, so a result with several rows is several waves:

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

- $\kappa_{abc}$ is the coefficient of $A_bA_c$ in the equation of wave $a$,
  for waves with $f_a = f_b + f_c$, summed over the ordered pairs: entries
  with $f_a \neq f_b + f_c$ are overlaps of waves that are not coupled and an
  equation does not use them.
- For second-harmonic generation $b = c$ is the pump. For sum-frequency
  generation $\omega_3 = \omega_1 + \omega_2$ the amplitude of the sum wave is
  driven with a factor 2, $2\kappa_{312}A_1A_2$, which the sum over the two
  orders $(b, c) = (1, 2), (2, 1)$ gives with no degeneracy factor of its own,
  and the sum-frequency power is
  $P_3 = 4|\kappa_{312}|^2P_1P_2L^2\,\mathrm{sinc}^2(\Delta\beta L/2)$,
  four times that of a second harmonic of the same overlap.
- For difference-frequency generation and parametric amplification, a pump
  $a$ drives the signal $b$ and the idler $c$ with
  $\mathrm{d}A_b/\mathrm{d}z = i(2\omega_b/\omega_a)\overline{\kappa_{abc}}\,
  A_a\overline{A_c}\,e^{i\Delta\beta z}$ and its mirror. That rule, which conserves
  the photon flux, assumes a $d$ symmetric in its three indices (Kleinman
  symmetry): the tensors above are. The gain of an undepleted pump is
  $g^2 = (4\omega_b\omega_c/\omega_a^2)|\kappa_{abc}|^2P_a - (\Delta\beta/2)^2$.
  The conventions page has the derivation.
- The pairs of results may come from different models of one mesh (the
  pump's, the harmonic's and the idler's materials), and the model passed gives
  the tensor, at the elements' own orientations.

## Check your results

- **The overlap is linear in $d$.** Doubling the tensor doubles $\kappa$ and
  quadruples the efficiency; the modes are those of the same permittivity:

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

- **A region without the tensor is not silent.** A model whose regions carry
  no `second_order_nonlinear_tensor` raises `ValueError` naming them, instead
  of returning a coupling of zero.
- **A finer mesh and a higher order.** Raising
  `gfem.SolverOptions(order=...)` to 4 changes the normalized efficiency of the
  pair by $3 \times 10^{-4}$ relative, and refining the mesh sizes of
  `ridge_section` by a quarter by about as much: the overlap converges with the mode fields, as the Kerr
  coefficient does ({doc}`waveguide-nonlinearity`).
- **Closed forms.** The overlap is checked against the filled guide and the
  slab, which have closed forms, with order refinement ({doc}`../verification`,
  "Second-order nonlinearity").

## Limits

- The undepleted pump: the harmonic is a small fraction of the pump along the
  whole guide. A pump that is depleted (an efficiency of tens of percent, a
  gain) needs the coupled equations, which `kappa` and `delta_beta` feed.
- One $d$ per material, real and constant over the grid: no dispersion of
  $d$ across a band (Miller's rule is yours to apply), no complex $d$ near
  an absorption, and no $\chi^{(2)}$ of surfaces or interfaces.
- The poling is analytic, a square wave of period $\Lambda$ and duty cycle $D$
  over the whole guide, full depth: no partial depth of the domain, no
  chirp or apodization, no domain-width errors.
- The overlap of the guided modes only: a guide with a radiating or a
  leaky mode, or a harmonic in a mode that is the pump's cutoff neighbour, needs
  the modes as solved. A lossy section is read as it is, valid while the
  attenuation is far below the phase constant.
- Hz line ports and a layered density (`Blend.pixels`) carrying the tensor are
  refused. Under `jax.grad` the overlap differentiates with respect to the
  section's materials, including $d$ through `gfem.MaterialField`, not the node
  coordinates: the shape's effect on $\Delta\beta$ comes from the results'
  `gamma`, and on the overlap, from a remesh and finite differences.
- The full-field solve of the harmonic with the nonlinear polarization as a
  source, a depleted pump and nonlinear metasurfaces are not in this page.

## Next

- {doc}`waveguide-nonlinearity`: the Kerr coefficient of the same kind of
  mode, and the couplings of four-wave mixing.
- {doc}`waveguide-dispersion`: the group index and the dispersion of a mode.
- {doc}`../conventions`: the formulas and the sign convention.
- {doc}`electro-optic-phase`: the crystal's linear electro-optic effect.
- {py:obj}`gfem.second_harmonic`, {py:obj}`gfem.three_wave_coupling`,
  {py:obj}`gfem.SecondHarmonic`, {py:obj}`gfem.ThreeWaveCoupling`.
