# Waveguide dispersion

Find how fast a pulse travels along a waveguide and how it spreads: the
{term}`group index` $n_g$, the {term}`group-velocity dispersion` $\beta_2$
and the {term}`dispersion parameter` $D$ of a guided mode, over a band and as
the guide's width changes. A {py:obj}`gfem.PortMode` solve returns `n_g`,
`beta_2`, `beta_3` and `D` beside the effective index.

The guide is a silicon strip 220 nm thick in oxide near 1.55 µm, in its
fundamental TE mode. Units are SI ({doc}`../conventions`).

## Silicon and oxide as Sellmeier laws

A Sellmeier term $B \lambda^2 / (\lambda^2 - \lambda_i^2)$ is a lossless
{py:obj}`gfem.Lorentz` term with `delta_eps` $= B$ and
`resonance_frequency` $= c / \lambda_i$, on `eps_inf` = 1 (pass the square root
of a fit's $\lambda_i^2$). A fit holds over its band only:

```python
import urllib.request

import numpy as np
import scipy.constants
import scipy.optimize

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
f0 = c / 1.55e-6  # Hz
n_si, n_ox = (float(np.sqrt(m.eps_r(f0).real)) for m in (silicon, oxide))
print(f"n at 1.55 um: silicon {n_si:.5f}, oxide {n_ox:.5f}")
```

```text
n at 1.55 um: silicon 3.47772, oxide 1.44402
```

## Over a band

{download}`strip_section.py <strip_section.py>` meshes the strip with netgen:
a `width` × `height` core in an oxide cladding that reaches `clearance`
beyond it on every side. The end face `"front"` carries the wave port.

```python
url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/strip_section.py"
)
urllib.request.urlretrieve(url, "strip_section.py")
from strip_section import strip_section  # noqa: E402

options = gfem.SolverOptions(order=3)


def strip(width, clearance=1e-6, core=silicon, cladding=oxide):
    """The strip, 220 nm thick, as a model with a wave port on its section."""
    return gfem.Model(
        strip_section(width, height=220e-9, clearance=clearance),
        materials={"core": core, "cladding": cladding},
        ports=[gfem.WavePort("wg", "front")],
    )


wavelength = np.linspace(1.45e-6, 1.65e-6, 21)  # m
mode = gfem.solve(strip(500e-9), gfem.PortMode("wg", c / wavelength), options=options)
print(mode)
for i in (0, 5, 10, 15, 20):
    print(
        f"{wavelength[i] * 1e9:.0f} nm: n_eff {mode.n_eff[i].real:.4f}, "
        f"n_g {mode.n_g[i].real:.4f}, D {mode.D[i].real * 1e6:+.1f} ps/(nm km)"
    )
beta_2 = mode.beta_2[10].real  # s^2/m, at 1.55 um
print(f"beta_2 at 1.55 um: {beta_2 * 1e27:.1f} ps^2/km = {beta_2 * 1e24:.3f} ps^2/m")
```

```text
PortModeResult(port 'wg', mode 1, 21 frequencies 181.7-206.8 THz (6 exact solves), n_eff 2.3341-2.5597, n_g 4.1816-4.2077, no voltage_path)
1450 nm: n_eff 2.5597, n_g 4.1816, D +423.1 ps/(nm km)
1500 nm: n_eff 2.5036, n_g 4.1885, D +488.3 ps/(nm km)
1550 nm: n_eff 2.4473, n_g 4.1959, D +484.9 ps/(nm km)
1600 nm: n_eff 2.3908, n_g 4.2027, D +408.0 ps/(nm km)
1650 nm: n_eff 2.3341, n_g 4.2077, D +247.3 ps/(nm km)
beta_2 at 1.55 um: -618.5 ps^2/km = -0.618 ps^2/m
```

- `n_g` is $c\,\mathrm{d}\beta/\mathrm{d}\omega$: an envelope travels at
  $v_g = c / n_g$. `beta_2` is $\mathrm{d}^2\beta/\mathrm{d}\omega^2$ in s²/m
  and `D` is $-(2\pi c / \lambda^2)\,\beta_2$ in s/m², $\lambda = c / f$ the
  vacuum wavelength of each row: $D > 0$ is anomalous dispersion, as across
  this band. `beta_3` is $\mathrm{d}^3\beta/\mathrm{d}\omega^3$ in s³/m.
- In the units of papers, `D * 1e6` is in ps/(nm km), `beta_2 * 1e27` in
  ps²/km and `beta_3 * 1e39` in ps³/km.
- The values are complex like `n_eff`: read `.real` on a lossless guide
  ({doc}`../modes-and-ports` for lossy ones). `mode.to_xarray()` holds them
  over `frequency`.
- gfem differentiates the propagation constant exactly in frequency: there is
  no step to choose. A grid of many frequencies costs a few exact solves.

## Material and waveguide dispersion

The same guide with each material's index frozen at its 1.55 µm value keeps
only the dispersion of the guide's shape:

```python
frozen = strip(
    500e-9,
    core=gfem.Material(eps_r=n_si**2),
    cladding=gfem.Material(eps_r=n_ox**2),
)
shape_only = gfem.solve(frozen, gfem.PortMode("wg", f0), options=options)
print(
    f"constant indices: n_eff {shape_only.n_eff[0].real:.4f}, "
    f"n_g {shape_only.n_g[0].real:.4f}, D {shape_only.D[0].real * 1e6:+.1f} ps/(nm km)"
)
```

```text
constant indices: n_eff 2.4473, n_g 4.0553, D +1437.1 ps/(nm km)
```

The effective index is the same, but the group index is 3 % low and $D$ three
times too large: silicon's own normal dispersion takes about 950 ps/(nm km)
off the guide's. A model with constant indices gives a wrong group index,
however fine its mesh.

## Widths of zero dispersion

$D$ changes sign with the width. Solve five widths at 1.55 µm, then find each
zero between two widths of opposite sign:

```python
def at_1550(width):
    return gfem.solve(strip(width), gfem.PortMode("wg", f0), options=options)


widths = np.array([400, 450, 600, 650, 700]) * 1e-9  # m
d = []
for width in widths:
    m = at_1550(width)
    d.append(m.D[0].real)
    print(
        f"{width * 1e9:.0f} nm: n_g {m.n_g[0].real:.4f}, D {d[-1] * 1e6:+.1f} ps/(nm km)"
    )
zeros = [
    scipy.optimize.brentq(
        lambda w: at_1550(w).D[0].real, widths[i], widths[i + 1], xtol=2e-9
    )
    for i in np.flatnonzero(np.diff(np.sign(d)))
]
print("zero dispersion at", ", ".join(f"{w * 1e9:.0f} nm" for w in zeros))
```

```text
400 nm: n_g 4.3917, D -451.9 ps/(nm km)
450 nm: n_g 4.2908, D +430.5 ps/(nm km)
600 nm: n_g 4.0556, D +122.2 ps/(nm km)
650 nm: n_g 4.0059, D -68.8 ps/(nm km)
700 nm: n_g 3.9658, D -236.3 ps/(nm km)
zero dispersion at 417 nm, 631 nm
```

- Between the two zero-dispersion widths the guide is anomalous, the regime
  where a soliton or a Kerr comb can form; outside, it is normal.
- Each call meshes and solves one width. Search another wavelength or the
  height the same way.

## Check your results

- **The cladding's clearance.** $\beta_2$ needs far more cladding than
  $n_\mathrm{eff}$ does. At the first zero-dispersion width:

  ```python
  for clearance in (0.5e-6, 1.0e-6, 1.5e-6):
      m = gfem.solve(
          strip(zeros[0], clearance=clearance), gfem.PortMode("wg", f0), options=options
      )
      print(
          f"clearance {clearance * 1e6:.1f} um: n_eff {m.n_eff[0].real:.6f}, "
          f"D {m.D[0].real * 1e6:+.1f} ps/(nm km)"
      )
  ```

  ```text
  clearance 0.5 um: n_eff 2.275563, D -152.8 ps/(nm km)
  clearance 1.0 um: n_eff 2.275254, D +9.5 ps/(nm km)
  clearance 1.5 um: n_eff 2.275251, D +9.9 ps/(nm km)
  ```

  At 0.5 µm $n_\mathrm{eff}$ is within $3 \times 10^{-4}$ but $D$ reads
  strongly normal; at 1.0 µm it has settled (the +9.5 left is the search's
  2 nm tolerance). Increase the clearance until $D$ stops moving, on the
  narrowest guide and at the longest wavelength of your sweep. A section that
  ends in a perfectly matched layer needs the layer moved out and
  `PML(reference_frequency=...)` pinned when you compare two grids.
- **$D$ against the slope of the group index.** $D = (1/c)\,
  \mathrm{d}n_g/\mathrm{d}\lambda$; the difference across the band's rows
  checks one against the other:

  ```python
  slope = np.gradient(mode.n_g.real, wavelength, edge_order=2) / c  # s/m^2
  print(f"{np.max(np.abs(slope - mode.D.real)) / np.max(np.abs(mode.D.real)):.1e}")
  ```

  ```text
  2.6e-03
  ```

  The difference is the finite difference's and shrinks as the square of the
  grid spacing.
- **A finer mesh.** At 500 nm and 1.55 µm the page's mesh gives $n_g$ =
  4.1959 and $D$ = +484.9 ps/(nm km); order 4, a halved `h_core` and
  `h_cladding`, or both move $n_g$ in the fourth digit and $D$ by under
  2 ps/(nm km), about 0.3 %.
- **Against a measurement.** The 450 nm wire's group index, 4.29 above, sits
  beside the 4.30 measured from the free spectral range of racetrack
  resonators of 450 × 220 nm silicon wires (Bogaerts et al., Laser Photonics
  Rev. 6, 47 (2012)). A computed $\beta_2$ of −0.7 ps²/m for a 500 × 220 nm
  wire (Clemmen et al., Opt. Express 17, 16558 (2009)) sits beside the −0.62
  here: agreement to 10 %, not a test.

## Limits

- **No gradients.** `n_g`, `beta_2`, `beta_3` and `D` under `jax.grad` raise
  `gfem.NonDifferentiableError` for any input, while `gamma` and `n_eff`
  differentiate ({doc}`port-mode-gradient`). Search widths with a root finder,
  as above. On a driven solve's `port_modes` they are `None`: solve
  `gfem.PortMode` on the same grid.
- **Anticrossings and cutoff.** A mode whose neighbour has the same effective
  index is refused; near an anticrossing $\beta_2$ is large and follows the
  mode the sweep tracks. Near cutoff $n_g$ grows without bound: check the
  clearance there.
- **Loss.** On a lossy or leaky section $n_g$ is complex, and its real part
  is no longer the speed of the stored energy.
- **Materials.** Only the permittivity disperses; a magnetic or anisotropic
  material keeps its constant value ({doc}`dispersive-materials`).

## Next

{doc}`../modes-and-ports`, {doc}`dispersive-materials`,
{doc}`perfectly-matched-layer`, {py:obj}`gfem.PortModeResult`,
{py:obj}`gfem.PortMode`.
