# Waveguide dispersion

Find how fast a pulse travels along a waveguide and how it spreads: the
{term}`group index` $n_g$, the {term}`group-velocity dispersion` $\beta_2$
and the {term}`dispersion parameter` $D$ of a guided mode, over a band and as
the guide's width changes. Every {py:obj}`gfem.PortMode` solve returns them
beside the effective index, at every frequency of its grid: `n_g`, `beta_2`,
`beta_3` and `D` on its {py:obj}`gfem.PortModeResult`.

The guide is a silicon strip 220 nm thick in oxide, around
$\lambda$ = 1.55 µm, in its fundamental TE mode. Silicon and oxide follow
their Sellmeier fits, so the numbers hold both the materials' dispersion and
the guide's. gfem differentiates the mode's propagation constant exactly in
frequency at each row of the grid: there is no step to choose, and the error
left is the mesh's and the cladding's truncation (see "Check your results").
Units are SI: $\beta_2$ in s²/m, $\beta_3$ in s³/m and $D$ in s/m²
({doc}`../conventions`, "Group index and dispersion").
The cladding around the core is its {term}`clearance`.

## Silicon and oxide as Sellmeier laws

A Sellmeier term $B \lambda^2 / (\lambda^2 - \lambda_i^2)$ is a lossless
{py:obj}`gfem.Lorentz` term with `delta_eps` $= B$ and
`resonance_frequency` $= c / \lambda_i$, on `eps_inf` = 1:

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

- A fit holds over its band only: below 1.36 µm the silicon law is an
  extrapolation.
- Most fits give $\lambda_i^2$ (often written $C_i$, in µm²): pass its square
  root.

## Over a band

{download}`strip_section.py <strip_section.py>` meshes the strip with netgen:
a `width` × `height` core in an oxide cladding that reaches `clearance`
beyond it on every side. The section's end face `"front"` carries the wave
port.

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

- `n_g` is $c\,\mathrm{d}\beta/\mathrm{d}\omega$: a pulse's envelope travels
  at $v_g = c / n_g$ and is delayed by $n_g / c$ per metre.
- `beta_2` is $\mathrm{d}^2\beta/\mathrm{d}\omega^2$ in s²/m, and `D` is
  $-(2\pi c / \lambda^2)\,\beta_2$ in s/m², with $\lambda = c / f$ the vacuum
  wavelength of each row. $\beta_2 < 0$, $D > 0$ is anomalous dispersion:
  this guide is anomalous across the band. `beta_3` is
  $\mathrm{d}^3\beta/\mathrm{d}\omega^3$ in s³/m.
- In the units of papers: `D * 1e6` is in ps/(nm km), `beta_2 * 1e27` in
  ps²/km (fs²/mm), `beta_3 * 1e39` in ps³/km. The dispersion slope
  $\mathrm{d}D/\mathrm{d}\lambda$ is `np.gradient(mode.D.real, wavelength)`.
- The values are complex, like `n_eff`: read `.real` on a lossless guide,
  where the imaginary part is rounding. On a lossy or leaky one the
  imaginary part is the dispersion of the loss (see {doc}`../modes-and-ports`).
- `mode.to_xarray()` holds `n_g`, `beta_2` (`"s^2/m"`), `beta_3` (`"s^3/m"`)
  and `D` (`"s/m^2"`) over `frequency`;
  `.assign_coords(wavelength=c / ds.frequency)` adds the wavelength.
- A grid of many frequencies costs a few exact solves: the rows between them
  come from the reduced port mode model, with their dispersion.

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

The effective index is the same, but without the materials' dispersion the
group index is 3 % low and $D$ three times too large: silicon's own normal
dispersion takes about 950 ps/(nm km) off the guide's anomalous dispersion.
A model with constant indices gives a wrong group index, however fine its
mesh.

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
  where a soliton or a Kerr frequency comb can form; outside, it is normal.
  The wider the guide, the closer it gets to the 220 nm slab, which is
  strongly normal.
- Each call meshes and solves one width; the search costs one solve per
  step. Search another wavelength or the height the same way.
- $D$ has no gradient with respect to the width, or to any input: the group
  index and dispersion are not differentiable, though the mode's $\gamma$ is
  ({doc}`port-mode-gradient`). Search with a root finder or an optimizer
  that does not need one.

## Check your results

- **The cladding's clearance.** $\beta_2$ needs far more cladding around the
  core than $n_\mathrm{eff}$ does. At the first zero-dispersion width:

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

  At 0.5 µm the effective index is within $3 \times 10^{-4}$, and the guide
  that has zero dispersion reads strongly normal; at 1.0 µm $D$ has settled
  (the +9.5 left is the search's 2 nm tolerance). Increase the clearance
  until $D$ stops moving, on the
  narrowest guide of your sweep and at its longest wavelength, where the mode
  reaches farthest. The same holds for a section that ends in a perfectly
  matched layer: move the layer out. Pin its
  `PML(reference_frequency=...)` when you compare two grids, since an
  unpinned layer takes the lowest frequency of each grid as its reference.
- **$D$ against the slope of the group index.** $D = (1/c)\,
  \mathrm{d}n_g/\mathrm{d}\lambda$. The difference across the band's rows
  checks one against the other:

  ```python
  slope = np.gradient(mode.n_g.real, wavelength, edge_order=2) / c  # s/m^2
  print(f"{np.max(np.abs(slope - mode.D.real)) / np.max(np.abs(mode.D.real)):.1e}")
  ```

  ```text
  2.6e-03
  ```

  The difference is the finite difference's, and shrinks as the square of
  the grid's spacing.
- **A finer mesh.** At 500 nm and 1.55 µm the page's mesh gives $n_g$ =
  4.1959 and $D$ = +484.9 ps/(nm km). Raising `gfem.SolverOptions(order=...)`
  to 4 gives 4.1960 and +486.0; halving `h_core` and `h_cladding` gives
  4.1961 and +486.3; both give 4.1961 and +486.6. $n_g$ moves in the fourth
  digit and $D$ by under 2 ps/(nm km), about 0.3 %.
- **Against a measurement.** The 450 nm wire's group index, 4.29 above, sits
  beside the 4.30 measured from the free spectral range of racetrack
  resonators of 450 × 220 nm silicon wires (Bogaerts et al., Laser Photonics
  Rev. 6, 47 (2012)). A computed $\beta_2$ of −0.7 ps²/m for a 500 × 220 nm
  wire (Clemmen et al., Opt. Express 17, 16558 (2009)) sits beside the −0.62
  here. That paper gives neither its material data nor its mesh, so read the
  10 % as agreement, not as a test.

## Limits

- No gradients: `n_g`, `beta_2`, `beta_3` and `D` under `jax.grad` raise
  `gfem.NonDifferentiableError`, with respect to any input the mode reads,
  while `gamma` and `n_eff` of the same result differentiate
  ({doc}`port-mode-gradient`). Search widths and heights with a root
  finder, as above.
- On a driven solve's `port_modes` they are `None`: solve `gfem.PortMode`
  on the same grid.
- A mode whose neighbour has the same effective index is refused, as for
  any port mode; near an anticrossing of two modes $\beta_2$ is large and
  real, and follows the mode the sweep tracks.
- Near cutoff $n_g$ grows without bound, and the mode reaches the cladding's
  edge first: check the clearance there.
- On a lossy or leaky section the group index is the complex
  $(c/i)\,\mathrm{d}\gamma/\mathrm{d}\omega$; its real part is still
  $c\,\mathrm{d}\beta/\mathrm{d}\omega$, but it is no longer the speed of the
  stored energy.
- Only the permittivity disperses: a magnetic or anisotropic material keeps
  its constant value across the band ({doc}`dispersive-materials`).

## Next

- {doc}`../modes-and-ports`: what the group index and the dispersion mean on
  lossy and leaky modes.
- {doc}`dispersive-materials`: writing a permittivity law.
- {doc}`perfectly-matched-layer`: a section that ends in a layer.
- {py:obj}`gfem.PortModeResult`, {py:obj}`gfem.PortMode`.
