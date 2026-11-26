# Permittivity fit to n, k data

Turn a measured table of refractive index $n$ and
{term}`extinction coefficient` $k$ into a {term}`permittivity law` that a
frequency sweep can read, and see how far the law is from the table. Here
the table is a gold film measured from the near ultraviolet to the infrared,
and the law goes into a model's electrodes. {py:obj}`gfem.fit_permittivity`
fits a sum of a Drude term and Lorentz terms with every strength and damping
held at 0 or above, so the law is passive at every real frequency. It holds
over the table's frequencies, its {term}`fitted range`; outside them it stays
passive, but nothing says how close it is to the material.

For a material from a published source, take it from {py:mod}`gfem.materials`
instead ({doc}`dispersive-materials`).

## The table and its convention

- **Frequencies in Hz.** A wavelength table converts with $f = c / \lambda$.
  The rows may come in any order.
- **$n$ and $k$ as tabulated.** $k \ge 0$ for an absorbing medium, under
  either time convention, so a table goes in as it is printed. gfem works
  under $e^{+i\omega t}$, where the index is $n - ik$: the law matches
  $\varepsilon = (n - ik)^2$, so $\mathrm{Im}\,\varepsilon = -2nk$
  ({py:obj}`gfem.conventions`).
- **A permittivity table.** Pass it as `eps=` in gfem's convention, with
  $\mathrm{Im}\,\varepsilon \le 0$ for loss. A table written under
  $e^{-i\omega t}$ as $\varepsilon_1 + i\varepsilon_2$ is its complex
  conjugate: pass `eps=np.conj(eps)`.
- **Nothing is clipped.** Gain rows ($k < 0$, or
  $\mathrm{Im}\,\varepsilon > 0$), a negative $n$, a repeated frequency and
  NaN raise `ValueError` before the fit, and the message names the remedy.
  A $k$ column of the wrong sign throughout is passed as `k=-k`. A few
  slightly negative rows that are rounding of $k = 0$ are passed as
  `k=np.clip(k, 0, None)`. Repeated rows are averaged or dropped.

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

The result is a {term}`permittivity fit`: `fit.law` is a
{py:obj}`gfem.PermittivityLaw`, and `eps_error`, `n_error` and `k_error` are
its largest relative errors in $\varepsilon$, $n$ and $k$ over the table.
`drude=True` lets the law conduct at zero frequency, as a metal does; it
allows a Drude term, it does not force one. Lorentz terms are added one at a
time until the error in $\varepsilon$ is within `tolerance` (1 % by default).
The fit takes a few seconds here.

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

Keep the gold off the port faces and away from the absorbing boundary: a
metal in a port's cross-section raises ({doc}`dispersive-materials`).

## How good the fit is

The three errors answer different questions:

- **$\varepsilon$** is what a solve reads. 0.87 % is within the default
  tolerance.
- **$k$** checks the loss where the material absorbs.
- **$n$**, for a metal, is the error of its absorption: $n$ is small where
  $k$ is large, and $\mathrm{Im}\,\varepsilon = -2nk$. Near gold's index
  minimum, about 0.7 µm, a small error in $\varepsilon$ is a large one in
  $n$: about twenty times the $\varepsilon$ error here, and more terms do not
  reduce it. It is reported, never warned. Check it against the table where
  it is worst:

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
table's, so a device whose loss comes from the gold near 0.7 µm loses about
that much more in the model. Fit a narrower range around the
wavelengths you solve at when that matters.

When the fit stops above its tolerance it still returns its best law and
warns `gfem.PermittivityFitWarning`, naming the remedies:

- **More terms.** Raise `max_lorentz_terms`. The cap keeps the free
  parameters at most half the number of points, so a sparse table stops at
  fewer terms.
- **A metal.** Pass `drude=True` when the table has
  $\mathrm{Re}\,\varepsilon < 0$.
- **A narrower range.** Fit only the rows you will solve at.
- **A looser tolerance.** When the table's own scatter is larger than the
  tolerance, no law follows it closer: set `tolerance` to the scatter.

Two cases do not need the loss:

- **A transparent table.** Leave out `k`. The law is then lossless, with no
  resonance inside the table, and a tighter `tolerance` costs little.
- **An absorption edge.** A sum of damped terms has loss at every
  frequency, so a table that runs from absorbing to transparent gets loss
  below the edge. The fit warns on $k$. Fit only where the material absorbs,
  or the transparent part alone without `k`.

## Outside the fitted range

`fit.law.frequency_range` holds the table's lowest and highest frequency, in
Hz. A driven sweep, an eigenmode target or mode, a port mode or an
electrostatic solve that reads the law outside it warns
`gfem.FittedRangeWarning` once per solve, naming the region and how far
outside. The solve still runs. Once you have checked the law there, against
other data or a published model, clear the range:

```python
import dataclasses

checked = dataclasses.replace(fit.law, frequency_range=None)
```

A law with `frequency_range=None` never warns, as a law written by hand.

## Limits

- **Drude and Lorentz terms only.** For a circuit board's datasheet value
  use
  {py:obj}`PermittivityLaw.djordjevic_sarkar <gfem.PermittivityLaw.djordjevic_sarkar>`.
- **Isotropic.** A law is one value for every direction. For a crystal, give
  constant principal values at one frequency.
- **Not differentiable.** The fit chooses a term count and starting
  resonances, so it has no derivative, and an input traced by `jax.grad` or
  `jax.jit` raises `gfem.NonDifferentiableError`. Fit outside the
  transformation; the law's own numbers then differentiate.
- **Seconds to a minute.** The time grows with the points and the terms;
  `max_lorentz_terms` bounds it. The same table gives the same law to the
  last bit on one machine, in any row order; on another machine it may
  differ in its last digits.
- **Resonances in the range.** A fitted metal law holds one to four damped
  Lorentz resonances inside or near its fitted range. Each one among a
  sweep's frequencies costs the adaptive sweep two solves. Each also raises
  the law's index near it, and the eigenmode {term}`unresolved band` reads
  that index: a coarse mesh in the metal may refuse a target near a
  resonance.
- **No narrow lines.** Every Lorentz damping is at least twice the median
  spacing of the table's frequencies, so noise does not turn into narrow
  resonances. A real absorption line narrower than that is smoothed: sample
  it more finely.

## Next

{doc}`dispersive-materials`, {py:mod}`gfem.materials`,
{py:obj}`gfem.PermittivityLaw`, {py:obj}`gfem.conventions`,
{doc}`frequency-sweep`.
