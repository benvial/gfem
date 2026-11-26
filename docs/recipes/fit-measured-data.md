# Permittivity fit to n, k data

Turn a measured table of refractive index $n$ and
{term}`extinction coefficient` $k$ into a {term}`permittivity law` that a
frequency sweep can read, and see how far the law is from the table.
{py:obj}`gfem.fit_permittivity` fits a Drude term and Lorentz terms with every
strength and damping at 0 or above, so the law is passive. It holds over the
table's frequencies, its {term}`fitted range`. For a published material, take
it from {py:mod}`gfem.materials` ({doc}`dispersive-materials`).

## The table and its convention

- **Frequencies in Hz**, in any row order; a wavelength table converts with
  $f = c / \lambda$.
- **$n$ and $k$ as printed**, $k \ge 0$ under either time convention. gfem
  works under $e^{+i\omega t}$: the law matches $\varepsilon = (n - ik)^2$, so
  $\mathrm{Im}\,\varepsilon = -2nk$ ({py:obj}`gfem.conventions`).
- **A permittivity table** goes in as `eps=` with $\mathrm{Im}\,\varepsilon \le 0$
  for loss; one written under $e^{-i\omega t}$ is the conjugate: pass
  `eps=np.conj(eps)`.
- **Nothing is clipped.** Gain rows, a negative $n$, a repeated frequency and
  NaN raise `ValueError` naming the remedy: `k=-k` for a wrong sign
  throughout, `k=np.clip(k, 0, None)` for rounding of $k = 0$, averaged or
  dropped repeats.

## Fit gold over the visible and near infrared

The table is McPeak et al. (2015), a template-stripped evaporated gold film,
cut to 0.4-1.7 µm:

```python
import urllib.request

import numpy as np
from scipy.constants import c

import gfem

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/"
    "gold_mcpeak2015.csv"
)
urllib.request.urlretrieve(url, "gold.csv")
wavelength, n, k = np.loadtxt("gold.csv", delimiter=",", unpack=True)  # µm, n, k
rows = (wavelength >= 0.4) & (wavelength <= 1.7)
frequency = c / (wavelength[rows] * 1e-6)  # Hz
fit = gfem.fit_permittivity(frequency, n=n[rows], k=k[rows], drude=True)
print(fit)

gold = gfem.Material(eps_r=fit.law, name="gold (McPeak 2015)")
```

```text
PermittivityFit(Drude + 4 Lorentz; max relative error eps 0.87 %, n 18 %, k 0.37 %; 1.76e+14-7.49e+14 Hz)
```

`fit.law` is a {py:obj}`gfem.PermittivityLaw` and `eps_error`, `n_error` and
`k_error` are its largest relative errors over the table (a
{term}`permittivity fit`). `drude=True` allows a Drude term for a metal; it
does not force one. Lorentz terms are added until the error in $\varepsilon$
is within `tolerance` (1 % by default).

## Give the law to a model

A fitted law goes wherever a law goes, as a region's `eps_r`. Here gold
electrodes beside a nitride guide in oxide, swept inside the fitted range:

```python
from gfem import materials

mesh = gfem.Mesh.from_gmsh("electrodes.msh", length_unit="um")
model = gfem.Model(
    mesh,
    materials={
        "electrode": gold,
        "core": materials.silicon_nitride(),
        "cladding": materials.silica(),
    },
    boundaries=[gfem.Absorbing("outer")],
    ports=[gfem.WavePort("in", "port1"), gfem.WavePort("out", "port2")],
)
problem = gfem.Driven(frequency=c / np.linspace(1.6e-6, 1.5e-6, 21))
```

Keep the gold off the port faces and the absorbing boundary
({doc}`dispersive-materials`).

## How good the fit is

- **$\varepsilon$** is what a solve reads; 0.87 % is within the tolerance.
- **$k$** checks the loss where the material absorbs.
- **$n$**, for a metal, is the error of its absorption: $n$ is small where $k$
  is large. Near gold's index minimum, about 0.7 µm, a small error in
  $\varepsilon$ is a large one in $n$ (about twenty times here) and more terms
  do not reduce it. It is reported, never warned. Check it where it is worst:

```python
index = np.sqrt(np.asarray(fit.law(frequency)))  # n - ik
worst = np.argmax(np.abs(index.real - n[rows]) / n[rows])
print(
    f"at {wavelength[rows][worst]:.2f} µm: n {n[rows][worst]:.4f}, law {index.real[worst]:.4f}"
)
print(
    f"at {wavelength[rows][worst]:.2f} µm: k {k[rows][worst]:.3f}, law {-index.imag[worst]:.3f}"
)
```

```text
at 0.70 µm: n 0.0986, law 0.1163
at 0.70 µm: k 4.305, law 4.302
```

There the law's $\mathrm{Im}\,\varepsilon$ is about 18 % larger than the
table's, so a device whose loss comes from the gold near 0.7 µm loses that much
more. Fit a narrower range around the wavelengths you solve at when it matters.

A fit that stops above its tolerance returns its best law and warns
`gfem.PermittivityFitWarning`:

- **More terms**: raise `max_lorentz_terms` (capped so free parameters stay at
  most half the points).
- **A metal**: pass `drude=True` when $\mathrm{Re}\,\varepsilon < 0$.
- **A narrower range**: fit only the rows you solve at.
- **A looser tolerance**: set `tolerance` to the table's own scatter.

A transparent table (leave out `k`) gives a lossless law. At an absorption
edge a sum of damped terms has loss below the edge and the fit warns on $k$:
fit only where the material absorbs.

## Outside the fitted range

`fit.law.frequency_range` holds the table's lowest and highest frequency. A
solve that reads the law outside it warns `gfem.FittedRangeWarning` once,
naming the region, and runs on. Once you have checked the law there against
other data, clear the range:

```python
import dataclasses

checked = dataclasses.replace(fit.law, frequency_range=None)
```

A law with `frequency_range=None` never warns, as one written by hand.

## Limits

- **Drude and Lorentz terms only.** For a board's datasheet value use
  {py:obj}`PermittivityLaw.djordjevic_sarkar <gfem.PermittivityLaw.djordjevic_sarkar>`.
- **Isotropic.** For a crystal, give constant principal values at one frequency.
- **Not differentiable.** An input traced by `jax.grad` or `jax.jit` raises
  `gfem.NonDifferentiableError`: fit outside the transformation, then the law's
  numbers differentiate.
- **Seconds to a minute**, growing with points and terms; the same table gives
  the same law on one machine, in any row order.
- **Resonances in the range.** A fitted metal law holds one to four damped
  Lorentz resonances; each in a sweep costs the adaptive sweep two solves, and
  the eigenmode {term}`unresolved band` may refuse a target near one on a
  coarse mesh.
- **No narrow lines.** Every Lorentz damping is at least twice the median
  spacing of the table's frequencies: sample a narrower line more finely.

## Next

{doc}`dispersive-materials`, {py:mod}`gfem.materials`,
{py:obj}`gfem.PermittivityLaw`, {py:obj}`gfem.conventions`,
{doc}`frequency-sweep`.
