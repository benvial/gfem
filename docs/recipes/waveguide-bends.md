# Waveguide bends

Find the loss of a bent waveguide and what a bend does to a ring made of it:
the {term}`bend loss` per 90°, its change with the radius from one
`jax.grad`, the {term}`junction coupling` to a straight guide, and the
group index, dispersion and Kerr shift of a microring. A silicon strip
500 × 220 nm in oxide at 1.55 µm loses 0.07 dB per 90° at a radius of 1 µm, a
quarter of that at 1.25 µm and a quarter again at 1.5 µm; the straight-guide
estimates of a ring are good to $10^{-3}$ in the free spectral range and 5 to
20 % in $D_2$.

A {py:obj}`gfem.PortMode` with a {py:obj}`gfem.Bend` solves the {term}`bent port mode`
of the guide exactly ({doc}`../modes-and-ports`). Units are SI ({doc}`../conventions`).

## A bent strip

The cross-section is the strip of {doc}`waveguide-dispersion` with an oxide
cladding on its inner side, and, on its outer side, oxide and a layer that
absorbs what the bend radiates: a {py:obj}`gfem.PML` region on the +x side.
{download}`strip_section.py <strip_section.py>` meshes it: `outer` and `inner`
are the oxide beyond the core on the two sides and `layer` the thickness of
the layer. The axis of the bend lies on the −x side, a distance $R$ behind the
{term}`reference radius` arc through the guide's centre, so the inner side
must stay within $R$ of it: with 0.25 µm of oxide and a 1 µm radius, $h$ stays
above 0.5.

```python
import dataclasses
import urllib.request

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


# Salzberg and Villa (1957), three-term fit by Tatian (1984): 1.36 to 11 um
silicon = gfem.Material(
    eps_r=sellmeier(
        (10.6684293, 0.0030434748, 1.54133408), (0.301516485, 1.13475115, 1104.0)
    ),
    nonlinear_index=4e-18,  # Dinu et al. (2003), at 1.54 um
    two_photon_absorption=8e-12,
)
# Malitson (1965): 0.21 to 3.71 um
oxide = gfem.Material(
    eps_r=sellmeier((0.6961663, 0.4079426, 0.8974794), (0.0684043, 0.1162414, 9.896161))
)
f0 = c / 1.55e-6  # Hz
n_ox = float(np.sqrt(oxide.eps_r(f0).real))

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/strip_section.py"
)
urllib.request.urlretrieve(url, "strip_section.py")
from strip_section import strip_section  # noqa: E402

options = gfem.SolverOptions(order=3)


def strip(outer, inner=0.25e-6):
    """The 500 x 220 nm strip: oxide `outer` and `inner` beyond the core, then a layer."""
    return gfem.Model(
        strip_section(500e-9, height=220e-9, outer=outer, inner=inner, layer=1e-6),
        materials={
            "core": silicon,
            "cladding": oxide,
            "pml": gfem.Material(eps_r=n_ox**2),  # the layer continues the oxide
        },
        ports=[gfem.WavePort("wg", "front")],
        pml=gfem.PML("pml", target_reflection=1e-10, reference_frequency=f0),
    )


def bend(radius):
    """A bend about an axis behind the guide's centre line, which is at x = 0."""
    return gfem.Bend(radius, outward="+x", at=(0, 0, 0))
```

The outer layer must start past the {term}`caustic` $x_c$, where the mode turns
from evanescent to radiating, by about a wavelength in the oxide; gfem warns
when it does not.

```python
model = strip(outer=1.75e-6)
bent = gfem.solve(model, gfem.PortMode("wg", f0, bend=bend(1e-6)), options=options)
print(bent)
n_eff = bent.n_eff[0]
x_c = n_eff.real / n_ox - 1  # um: R times this, R = 1 um
print(f"n_eff {n_eff.real:.4f}, loss {bent.bend_loss[0]:.2e} dB per 90 deg")
print(f"caustic at x = {x_c:.2f} um, the layer starts at 2.00 um")
```

```text
PortModeResult(port 'wg', mode 1, 1 frequencies, bend radius 1e-06 m, 193.4 THz (1 exact solves), n_eff 2.4859-2.4859, n_g 4.2414-4.2414, no voltage_path)
n_eff 2.4859, loss 6.67e-02 dB per 90 deg
caustic at x = 0.72 um, the layer starts at 2.00 um
```

The caustic sits 0.72 µm from the guide's centre, the layer 1.28 µm beyond it.

## Loss against radius

The loss falls steeply with the radius. `jax.value_and_grad` differentiates
`bend_loss` in $R$ through the eigenproblem; the radius enters in µm so that
the gradient is in dB per µm:

```python
def bend_loss_um(radius, model=model):
    """The loss per 90 degrees in dB at a radius in um, differentiable in it."""
    problem = gfem.PortMode("wg", f0, bend=bend(radius * 1e-6))
    return gfem.solve(model, problem, options=options).bend_loss[0]


loss_100 = bent.bend_loss[0]  # from the solve above
loss_125 = bend_loss_um(1.25)
loss_150, slope = jax.value_and_grad(bend_loss_um)(1.5)
for radius, value in ((1.0, loss_100), (1.25, loss_125), (1.5, loss_150)):
    print(f"R = {radius:.2f} um: {value:.2e} dB per 90 deg")
print(f"d(loss)/dR at 1.5 um: {slope:.2e} dB/um, {slope / loss_150:.1f} per um")
```

```text
R = 1.00 um: 6.67e-02 dB per 90 deg
R = 1.25 um: 1.57e-02 dB per 90 deg
R = 1.50 um: 3.87e-03 dB per 90 deg
d(loss)/dR at 1.5 um: -2.17e-02 dB/um, -5.6 per um
```

The loss at 1.5 µm falls by $e^{5.6}$ per µm, a factor 4 for each 0.25 µm, as
the table says. The gradient is exact, not a difference of solves, and the
same call differentiates with respect to the materials ({doc}`port-mode-gradient`).

## Junction with a straight guide

A bend meets a straight guide at a plane where the two modes differ. The
{term}`junction coupling` $\tau^2$ of the straight and the bent mode is the
power the junction passes, reflection neglected; $-10\log_{10}|\tau^2|$ is
its loss in dB. A 90° turn has two junctions and one bend:

```python
straight = gfem.solve(model, gfem.PortMode("wg", f0), options=options)
tau2 = gfem.junction_coupling(model, straight, bent)[0]
junction = -10 * np.log10(abs(tau2))  # dB, one junction
total = 2 * junction + bent.bend_loss[0]
print(f"tau^2 = {abs(tau2):.5f}, junction loss {junction:.4f} dB")
print(f"2 junctions + one bend: {total:.4f} dB for the 90 deg turn")
```

```text
tau^2 = 0.95814, junction loss 0.1857 dB
2 junctions + one bend: 0.4382 dB for the 90 deg turn
```

The two junctions lose 0.37 dB, five times what the bend radiates: at this
radius a bend is a problem of mode matching. A measured bend loss includes
its junctions.

## A microring

A ring of radius $R$ has resonances spaced by $D_1 = c/(n_g R)$ and
$D_2 = -(c/n_g)\,D_1^2\,\beta_2$, and a {term}`Kerr shift per photon`
$g_0 = \hbar\omega_0 c^2\,\mathrm{Re}\,\gamma_\mathrm{NL}/(n_g^2 L)$,
$L = 2\pi R$ ({doc}`waveguide-nonlinearity`). Its radiation limit is
$Q_\mathrm{rad} = k_0\,\mathrm{Re}\,n_g/(2\,\mathrm{Re}\,\gamma)$. The
bent $n_g$ and $\beta_2$ replace the straight guide's. One section serves
both rings: its layer starts at 8 µm, past the caustic at about 0.70 $R$, and
1 µm of oxide lies inside, as the dispersion needs
({doc}`waveguide-dispersion`):

```python
hbar = scipy.constants.hbar


def ring_numbers(n_g, beta_2, gamma_nl, radius):
    """D_1/2pi (GHz), D_2/2pi (MHz) and g_0/2pi (Hz) of a ring of this radius."""
    d1 = c / (n_g * radius)
    d2 = -(c / n_g) * d1**2 * beta_2
    g0 = hbar * 2 * np.pi * f0 * c**2 * gamma_nl / (n_g**2 * 2 * np.pi * radius)
    return d1 / (2 * np.pi) / 1e9, d2 / (2 * np.pi) / 1e6, g0 / (2 * np.pi)


ring = strip(outer=7.75e-6, inner=1e-6)  # past the caustic of both rings
reference = gfem.solve(ring, gfem.PortMode("wg", f0), options=options)
kerr_reference = gfem.kerr_nonlinearity(ring, reference)
for radius in (5e-6, 10e-6):
    problem = gfem.PortMode("wg", f0, bend=bend(radius))
    mode = gfem.solve(ring, problem, options=options)
    kerr = gfem.kerr_nonlinearity(ring, mode)
    for name, m, k in (("bent", mode, kerr), ("straight", reference, kerr_reference)):
        d1, d2, g0 = ring_numbers(
            m.n_g[0].real, m.beta_2[0].real, k.gamma_nl[0].real, radius
        )
        print(
            f"R = {radius * 1e6:.0f} um, {name}: n_g {m.n_g[0].real:.4f}, "
            f"beta_2 {m.beta_2[0].real * 1e24:.3f} ps^2/m, D1/2pi {d1:.1f} GHz, "
            f"D2/2pi {d2:.1f} MHz, g0/2pi {g0:.1f} Hz"
        )
    x_c = radius * (mode.n_eff[0].real / n_ox - 1)  # the caustic
    k0 = 2 * np.pi * f0 / c
    q_rad = k0 * mode.n_g[0].real / (2 * abs(mode.gamma[0].real))
    print(
        f"caustic at {x_c * 1e6:.2f} um; radiation Q above 1e12: {bool(q_rad > 1e12)}"
    )
```

```text
R = 5 um, bent: n_g 4.1988, beta_2 -0.495 ps^2/m, D1/2pi 2272.7 GHz, D2/2pi 1147.9 MHz, g0/2pi 585.4 Hz
R = 5 um, straight: n_g 4.1959, beta_2 -0.619 ps^2/m, D1/2pi 2274.3 GHz, D2/2pi 1436.3 MHz, g0/2pi 586.8 Hz
caustic at 3.48 um; radiation Q above 1e12: True
R = 10 um, bent: n_g 4.1966, beta_2 -0.588 ps^2/m, D1/2pi 1136.9 GHz, D2/2pi 341.1 MHz, g0/2pi 293.2 Hz
R = 10 um, straight: n_g 4.1959, beta_2 -0.619 ps^2/m, D1/2pi 1137.1 GHz, D2/2pi 359.1 MHz, g0/2pi 293.4 Hz
caustic at 6.95 um; radiation Q above 1e12: True
```

The free spectral range moves by less than $10^{-3}$ and the Kerr shift per
photon by less than $3 \times 10^{-3}$; the bend lowers $|\beta_2|$ and so $D_2$, by 5 % at
10 µm and 20 % at 5 µm. Both rings radiate nothing a simulation can see: their
$Q_\mathrm{rad}$ is far above every other loss, and the loss read from the
solve is rounding.

## Check your results

- **The layer's place.** Move the layer 0.5 µm out and solve again; keep the
  radii whose loss moves by a few percent. The three radii above do (here by
  under 2 %), and so does 2 µm. The caustic moves out with the radius, and
  the layer with it.

  ```python
  moved = strip(outer=2.25e-6)  # the layer 0.5 um further out
  for radius, before in ((1.0, loss_100), (1.5, loss_150)):
      after = bend_loss_um(radius, moved)
      print(f"R = {radius:.2f} um: {after:.2e} dB ({after / before - 1:+.1%})")
  finer = gfem.solve(
      model,
      gfem.PortMode("wg", f0, bend=bend(1e-6)),
      options=gfem.SolverOptions(order=4),
  )
  print(f"order 4: {finer.bend_loss[0]:.2e} dB, order 3: {loss_100:.2e} dB")
  ```

  ```text
  R = 1.00 um: 6.56e-02 dB (-1.7%)
  R = 1.50 um: 3.86e-03 dB (-0.3%)
  order 4: 6.55e-02 dB, order 3: 6.67e-02 dB
  ```

  The element order moves the loss by 2 %: a bend of 1 µm is resolved by the
  mesh of the page, not converged to the digit.
- **The reference.** Moving `at` along the guide leaves $\nu = -i\gamma R$
  unchanged: only its coordinate along `outward` counts.
- **A perfect conductor outside.** An outer wall of perfect conductor and no
  layer gives zero radiation loss; any other outer edge without a layer warns.
- **The layer's reflection.** Set `target_reflection` to $10^{-10}$ or below
  for a bend of low loss, as here.

## Limits

- **Materials.** Rotation-invariant about the bend axis: isotropic, or
  uniaxial with its axis along it. There is no mirror across the guide's
  mid-plane normal to the bend plane.
- **Overlaps.** Kerr, with {py:obj}`gfem.kerr_nonlinearity` and
  {py:obj}`gfem.kerr_coupling`, is the only bend-aware overlap.
  `three_wave_coupling`, `second_harmonic`, `spdc`, `sfwm`, `raman_gain` and
  `free_carriers` refuse a bent mode, and so does Kerr with
  `nonlinear_anisotropy` or a birefringent index; `kerr_coupling` refuses to
  mix a straight and a bent mode.
- **Junctions.** {py:obj}`gfem.junction_coupling` refuses line ports and a
  traced `mu_r` or axes. $|\tau^2|$ can exceed 1 slightly where impedances
  differ.
- **Gradients.** $R$, the materials and the section's shape differentiate
  $\gamma$, `bend_loss` and `junction_coupling`. `n_g`, `beta_k` and `D` on
  a bent mode, and the node coordinates in a bent overlap, raise
  {py:obj}`gfem.NonDifferentiableError`.
- **No bent port in a driven solve.** A bend is a port-mode problem; a 3D
  model holds the guide as a bent geometry ({doc}`../tutorials/bend3d`).

## Next

{doc}`../modes-and-ports`, {doc}`waveguide-nonlinearity`,
{doc}`waveguide-dispersion`, {doc}`port-mode-gradient`,
{py:obj}`gfem.Bend`, {py:obj}`gfem.junction_coupling`.
