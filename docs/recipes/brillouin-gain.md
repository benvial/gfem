# Find the Brillouin gain of a wire

Find the Brillouin gain $G_\mathrm{B}$, in 1/(W m), that each mechanical mode
of a suspended silicon wire gives a Stokes wave per watt of pump.
{py:obj}`gfem.brillouin_gain` reads the modes of
{py:class}`gfem.MechanicalEigenmode` against one optical mode and returns the
gain, with its moving-boundary and photoelastic parts, for the backward
interaction ($q \approx 2\beta$) and the forward one ($q = 0$).

The wire is the one of {doc}`optomechanical-coupling`: 315 nm wide and
283.5 nm tall, light at 1.552 µm, in air. The published case is Qiu et al.,
Opt. Express 21, 31402 (2013). {download}`wire_section.py <wire_section.py>`
meshes the wire twice, for the light and for the sound.

## Assumptions

- Intramodal: the Stokes wave is the pump's own mode travelling the other way
  (backward) or the same way (forward).
- The optical section is lossless, and its mode is held constant.
- The phonon travels with the pump, $q = 2\beta$, and the Stokes shift
  $\Omega n_g/c$ is dropped (about $10^{-4}$ of $q$ here).
- The wire is isotropic silicon (170 GPa, $\nu = 0.28$) with the cubic
  photoelastic tensor of the published case.
- The mechanical quality factor $Q_m$ of the material, 1000, sets the gain
  linearly.

## Conventions

The pump has 1 W and the mechanical mode 1 J/m. The gain is
$G_\mathrm{B} = Q_m\omega|\delta n_{ps}|^2/c^2$ at the pump's $\omega$, and the
parts' gains are not additive. See {doc}`../conventions`, "Brillouin gain".

## The phase-matched wavenumber

A mode solve at the pump's frequency gives $\beta$, and the phonon of the
backward interaction has $q = 2\beta$:

```python
import dataclasses
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as sc

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/wire_section.py"
urllib.request.urlretrieve(url, "wire_section.py")
from wire_section import mechanical_section, optical_section  # noqa: E402

width, height = 315e-9, 283.5e-9  # m
frequency = sc.c / 1.552e-6  # Hz

p = np.zeros((6, 6))
p[:3, :3] = 0.017
p[[0, 1, 2], [0, 1, 2]] = -0.09
p[3:, 3:] = -0.051 * np.eye(3)
optical = gfem.Model(
    optical_section(width, height),
    materials={
        "wire": gfem.Material(eps_r=3.5**2, photoelastic_tensor=p),
        "air": gfem.Material(),
    },
    boundaries=[gfem.PEC("wall")],
    ports=[gfem.WavePort("p", "front")],
)
silicon = gfem.Material(
    stiffness=gfem.isotropic_stiffness(170e9, 0.28),
    mass_density=2329.0,
    mechanical_quality_factor=1000.0,
)
mechanical = gfem.Model(mechanical_section(width, height), materials={"wire": silicon})

pm = gfem.solve(optical, gfem.PortMode("p", frequency))
q = 2 * float(pm.gamma[0].imag)  # rad/m, along the pump
print(f"q = {q:.5g} rad/m")
```

```text
q = 1.4892e+07 rad/m
```

The sign follows the port: $+$ for `"front"`, whose inward normal is $+z$ (the
pump travels along $+z$), $-$ for a port facing the other way. A $q$ of the
wrong sign, or one that is neither near 0 nor near $2\beta$, raises.

## The gain of every mode

Twenty-four modes near 18 GHz at this wavenumber, cubic elements, and one call
for all of them:

```python
modes = gfem.solve(
    mechanical,
    gfem.MechanicalEigenmode(18e9, n_modes=24, wavenumber=q),
    options=gfem.SolverOptions(order=3),
)
gain = gfem.brillouin_gain(optical, modes, port="p", frequency=frequency)
print(gain.overlap.direction)
for k in np.argsort(gain.total)[::-1][:5]:
    print(
        f"{modes.frequency[k] / 1e9:.3f} GHz: moving boundary {gain.moving_boundary[k]:.0f}, "
        f"photoelastic {gain.photoelastic[k]:.0f}, total {gain.total[k]:.0f} 1/(W m)"
    )
```

```text
backward
22.541 GHz: moving boundary 29, photoelastic 1814, total 1381 1/(W m)
15.415 GHz: moving boundary 863, photoelastic 3592, total 934 1/(W m)
27.130 GHz: moving boundary 526, photoelastic 31, total 815 1/(W m)
13.890 GHz: moving boundary 2765, photoelastic 579, total 813 1/(W m)
18.233 GHz: moving boundary 468, photoelastic 35, total 248 1/(W m)
```

Four of these are Qiu's modes, E4, E2, E1 and E3, even under both mirrors of
the section. The 27.13 GHz mode is a higher mode of the same class. The two
parts of each of Qiu's modes have opposite signs except at 27.13 GHz, where
they add. Where they oppose, the total is far below their sum: about
$(\sqrt{G_\mathrm{PE}} - \sqrt{G_\mathrm{MB}})^2$, not
$G_\mathrm{MB} + G_\mathrm{PE}$.

E1 and E2 against the published frequencies, parts and totals:

```python
for name, f_pub, g_pub in (("E1", 13.99, 890), ("E2", 15.54, 860)):
    k = int(np.argmin(abs(modes.frequency / 1e9 - f_pub)))
    mb, pe, total = gain.moving_boundary[k], gain.photoelastic[k], gain.total[k]
    print(
        f"{name}: {modes.frequency[k] / 1e9:.2f} GHz, parts {mb:.0f} and {pe:.0f}, "
        f"total {total:.0f} against {g_pub} ({total / g_pub - 1:+.1%})"
    )
```

```text
E1: 13.89 GHz, parts 2765 and 579, total 813 against 890 (-8.7%)
E2: 15.41 GHz, parts 863 and 3592, total 934 against 860 (+8.6%)
```

The published parts are 2750 and 530 for E1, 930 and 3530 for E2. The total
is a difference of amplitudes, which magnifies a part's error about five
times: on this coarse mesh the parts are within 10 % and the totals within
9 %.

## Forward

The same call on modes at $q = 0$ gives the forward gain, the one that
{doc}`optomechanical-coupling` computes by hand:

```python
forward = gfem.solve(
    mechanical,
    gfem.MechanicalEigenmode(12.5e9, n_modes=1),
    options=gfem.SolverOptions(order=3),
)
g_forward = gfem.brillouin_gain(optical, forward, port="p", frequency=frequency)
print(g_forward.overlap.direction, f"{g_forward.total[0]:.0f} 1/(W m)")
```

```text
forward 16802 1/(W m)
```

## The gradient

The gain is differentiable in every input of the mechanical solve and in the
photoelastic tensor, at a fixed $q$. A scaling law checks it. For an isotropic
wire, scaling the stiffness by a factor leaves the mode shape unchanged; the
1 J/m displacement, and so $\delta n_{ps}$, scales as $E^{-1/2}$, and $Q_m$
stays fixed, so $\mathrm{d}\ln G_\mathrm{B}/\mathrm{d}\ln E = -1$ for the E1
mode:

```python
def gain_of(young):
    wire = gfem.MaterialField(
        stiffness=gfem.isotropic_stiffness(young, 0.28),
        mass_density=2329.0,
        mechanical_quality_factor=1000.0,
    )
    solved = gfem.solve(
        gfem.Model(mechanical.mesh, materials={"wire": wire}),
        gfem.MechanicalEigenmode(13.9e9, n_modes=1, wavenumber=q),
        options=gfem.SolverOptions(order=3),
    )
    return gfem.brillouin_gain(optical, solved, port="p", frequency=frequency).total[0]


young = 170e9
d_log = jax.grad(gain_of)(young) * young / gain_of(young)
print(f"d ln G / d ln E = {float(d_log):.5f}")
```

```text
d ln G / d ln E = -1.00000
```

## Check your results

- **Parts and total.** The total is about $(\sqrt{G_\mathrm{PE}} -
  \sqrt{G_\mathrm{MB}})^2$ when the parts have opposite signs, and about the
  sum of roots squared when they do not (exact for collinear complex
  amplitudes).
- **The quality factor.** $G_\mathrm{B} \propto Q_m$: a measured $Q'$ rescales the
  gain by $Q'/Q_m$.
- **Symmetry.** A mode of a symmetry class other than the pump's couples only
  through the mesh's asymmetry, orders of magnitude below the even modes.
- **A finer mesh.** Refine both meshes near the wire: E1's photoelastic part
  moves to within 2 % of the published value, the frequencies keep their first
  four digits, and E2's moving-boundary part stays about 7 % below the
  published figure on every mesh.
- See {doc}`../verification`, "Brillouin gain", for the closed form and the
  published case.

## Limits

- Intramodal only: the pump and the Stokes wave are the same optical mode, so
  pairs of two different modes ($q = \beta_p \mp \beta_s$) raise `ValueError`.
- A lossless optical section, with $\mu_r = 1$ for the backward gain; a lossy
  one raises `NotImplementedError`, as does a 2D model.
- The optical mode is held constant, and the call does not run inside
  `jax.jit`.
- A mechanical mode with no loss ($Q_m$ unset) has no linewidth and raises
  `ValueError`.
- The peak gain only: the Lorentzian has the full width at half maximum
  $2\,\mathrm{Im}\,\Omega$ (rad/s).

```python
cases = {
    "wrong q": lambda: gfem.brillouin_gain(
        optical,
        dataclasses.replace(modes, wavenumber=-q),
        port="p",
        frequency=frequency,
    ),
    "lossless": lambda: gfem.brillouin_gain(
        optical,
        gfem.solve(
            gfem.Model(
                mechanical.mesh,
                materials={
                    "wire": gfem.Material(
                        stiffness=gfem.isotropic_stiffness(170e9, 0.28),
                        mass_density=2329.0,
                    )
                },
            ),
            gfem.MechanicalEigenmode(12.5e9, n_modes=1),
            options=gfem.SolverOptions(order=3),
        ),
        port="p",
        frequency=frequency,
    ),
    "2D model": lambda: gfem.brillouin_gain(
        mechanical, modes, port="p", frequency=frequency
    ),
}
for name, call in cases.items():
    try:
        call()
    except (ValueError, NotImplementedError) as error:
        print(f"{name}: {type(error).__name__}")
```

```text
wrong q: ValueError
lossless: ValueError
2D model: NotImplementedError
```

## Next

- {doc}`optomechanical-coupling`: the vacuum coupling $g_0$ and the overlap of
  any displacement.
- {doc}`mechanical-modes`: the vibrations of a section and their quality
  factors.
- {doc}`waveguide-dispersion`: the group index that sets the forward scale.
