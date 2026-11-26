# Kerr effect in a crystal

Find how the Kerr coefficient of a waveguide changes with the orientation of
its crystal: a cubic crystal such as silicon, whose $n_2$ depends on the
polarization direction, and a birefringent one such as lithium niobate, whose
index depends on the axis. `gfem.kerr_nonlinearity` and `gfem.kerr_coupling`
read the crystal frame from the material's `axes`, and need no solve beyond
the mode's own.

The guide is the silicon strip of {doc}`waveguide-nonlinearity`, 500 × 220 nm
in oxide, at 1.55 µm, then a lithium niobate strip. Units are SI
({doc}`../conventions`, "Kerr nonlinearity").

## Silicon: one number for the anisotropy

A cubic crystal has, below its gap, one extra number: $\rho =
3\chi_{1122}/\chi_{1111}$, `Material.nonlinear_anisotropy`. A field along the
unit vector $u$ of the crystal frame sees $\rho + (1-\rho)\sum_s u_s^4$ times
the value along a cube axis, so $n_2$ and $\beta_\mathrm{TPA}$ of the
material refer to light polarized along a cube axis, $[100]$, not to the
$[110]$ of a standard silicon-on-insulator guide. A value measured along
$[110]$ converts by $n_2^{[100]} = 2 n_2^{[110]}/(1+\rho)$.

| Material | $\rho$ | Range | Source |
| --- | --- | --- | --- |
| Silicon | 1.27 | 1.2 to 2.4 µm, Kerr and two-photon absorption alike | Zhang et al., Appl. Phys. Lett. 91, 071113 (2007) |

The table is the only value: the library has no preset, and the aluminium
gallium arsenide anisotropy is larger and not covered. A dispersive
`PermittivityLaw` takes no `axes`, so a crystal's frame cannot sit beside
one. Solve the mode on the law, then read the overlap on a second model of
the same mesh whose core is the constant permittivity at the band centre,
with `axes` and $\rho$:

```python
import dataclasses
import urllib.request

import numpy as np
import scipy.constants

import gfem
from gfem import materials

c = scipy.constants.c  # m/s
url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/strip_section.py"
)
urllib.request.urlretrieve(url, "strip_section.py")
from strip_section import strip_section  # noqa: E402

options = gfem.SolverOptions(order=3)
frequency = c / 1.55e-6
model = gfem.Model(
    strip_section(500e-9),
    materials={"core": materials.silicon(), "cladding": materials.silica()},
    ports=[gfem.WavePort("wg", "front")],
)
mode = gfem.solve(model, gfem.PortMode("wg", frequency), options=options)

rho = 1.27
n2_110, beta_110 = 4e-18, 8e-12  # along [110]
scale = 2 / (1 + rho)
eps = float(np.real(materials.silicon(frequency=frequency).eps_r))
cube = ((1, 0, 0), (0, 1, 0), (0, 0, 1))  # guide along [001]
h = np.sqrt(0.5)
diagonal = ((h, 0, -h), (0, 1, 0), (h, 0, h))  # guide along [110]


def oriented(axes):
    core = gfem.Material(
        eps_r=eps,
        nonlinear_index=scale * n2_110,
        two_photon_absorption=scale * beta_110,
        nonlinear_anisotropy=rho,
        axes=axes,
    )
    return model.with_material("core", core)


g_100 = gfem.kerr_nonlinearity(oriented(cube), mode).gamma_nl[0]
g_110 = gfem.kerr_nonlinearity(oriented(diagonal), mode).gamma_nl[0]
print(f"[100] guide {g_100.real:.1f}, [110] guide {g_110.real:.1f} /(W m)")
print(f"ratio {g_110.real / g_100.real:.4f}, uniform x field {(1 + rho) / 2:.4f}")
```

```text
[100] guide 158.4, [110] guide 174.2 /(W m)
ratio 1.0998, uniform x field 1.1350
```

- The ratio is below $(1+\rho)/2$ because the mode of a wire is hybrid: its
  vertical and longitudinal components see other factors. The factor is
  exact for a field along one direction (a filled guide, a slab).
- The `axes` rows need no handedness: the cubic form is even in each axis
  and symmetric under permuting them. At $\rho = 1$ the result does not
  depend on `axes`.
- The error of the two-model route is the squared index at the band centre:
  at most 0.24 % over 1.50 to 1.60 µm for silicon.
- `kerr_coupling` takes the same material: cross-phase of two orthogonal
  polarizations along cube axes is $2\rho/3$ of the self-phase, not
  $2/3$.

## Lithium niobate: the index matrix

In a birefringent crystal "one $n_2$" has no single refractive index. gfem
takes `nonlinear_index` as the change $n_2 I$ of the index of light
polarized along each principal axis, and replaces $n^2$ by the matrix $N =
\sum_k n_k a_k a_k^\mathsf{T}$, with $n_k = \sqrt{\mathrm{Re}\,\varepsilon_{r,k}}$
the three principal values of `eps_r`. The result is exact along every
principal axis and equals the isotropic one when the crystal is isotropic. A
single scalar index differs from it by at most
$n_o^2/n_e^2 - 1$, 7.0 % in lithium niobate at 1.55 µm. A z-cut strip, TE
and TM:

```python
ln = dataclasses.replace(
    materials.lithium_niobate(
        frequency=frequency, axes=((1, 0, 0), (0, 0, -1), (0, 1, 0))
    ),
    nonlinear_index=1.8e-19,
)
guide = gfem.Model(
    strip_section(1.2e-6, height=0.6e-6),
    materials={"core": ln, "cladding": materials.silica()},
    ports=[gfem.WavePort("wg", "front")],
)
for number, name in ((1, "TE"), (2, "TM")):
    result = gfem.solve(
        guide, gfem.PortMode("wg", frequency, mode=number), options=options
    )
    gamma = gfem.kerr_nonlinearity(guide, result).gamma_nl[0].real
    scalar = dataclasses.replace(ln, eps_r=ln.eps_r[2], axes=None)  # n_e everywhere
    one = (
        gfem.kerr_nonlinearity(guide.with_material("core", scalar), result)
        .gamma_nl[0]
        .real
    )
    print(f"{name}: gamma {gamma:.4f} /(W m), scalar n_e route {one / gamma - 1:+.4f}")
```

```text
TE: gamma 1.3247 /(W m), scalar n_e route -0.0654
TM: gamma 0.9891 /(W m), scalar n_e route -0.0034
```

- The scalar index of the axis a mode mostly sees is close to the matrix for
  TM (0.3 % here, the small ordinary share of its field) and 6.5 % low for TE.
- The index matrix works from `eps_r` and `axes` alone. A `MaterialField`
  with `axes` carrying a Kerr property is refused: give the region a
  `gfem.Material`.
- The $n_2$ data of lithium niobate scatter (9.1 to $38.6 \times
  10^{-20}$ m²/W at 1064 nm, depending on the cascaded second-order
  correction), and it is measured for light along the optic axis. The value
  here, $1.8 \times 10^{-19}$ m²/W, is unverified; it is a placeholder, not
  a recommendation.

## What $n_2$ includes

`nonlinear_index` is the $n_2$ you measured. A value measured with pulses
much longer than the Raman response includes the Raman part, and gfem gives
all of it the Kerr symmetry. A pulse tool that adds a delayed Raman term
splits $n_2$ itself.

## Check your results

- The $\rho = 1$ result is the isotropic one for any `axes`.
- A filled guide has the closed form $\gamma = \tfrac32 k_0 n_2/(ab)\,(n_y/n_\mathrm{eff})^2$
  with $E\parallel\hat y$ (see {doc}`../verification`, "Kerr nonlinearity").

## Limits

- Cubic crystals with Kleinman symmetry only: one $\rho$ for $n_2$ and
  $\beta_\mathrm{TPA}$. No $3m$ tensor.
- A birefringent region and a `nonlinear_anisotropy` region cannot share one
  port section.
- $\gamma$ is affine in $\rho$, so it differentiates with respect to it wherever
  `nonlinear_index` does.

## Next

- {doc}`waveguide-nonlinearity`: the coefficient and the coupling tensor.
- {doc}`../conventions`: the formulas.
- {py:obj}`gfem.kerr_nonlinearity`, {py:obj}`gfem.kerr_coupling`.
