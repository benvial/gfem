# Acousto-optic index change

Find how much a voltage on an interdigital transducer shifts the effective index
of a waveguide that its sound wave crosses: the {term}`acousto-optic shift` of a
guided mode per volt, as a complex phasor at each acoustic frequency, the
voltage-length product $V_\pi L$ of one arm and of a push-pull pair, and the
voltage of the wave that reaches the transducer.

## A crystal with a photoelastic tensor

A material's `photoelastic_tensor` is the contracted $p^E_{IJ}$ (6 × 6) in the
frame of its `axes` ({doc}`../conventions`, "Photoelastic tensor").

```python
import dataclasses
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as sc

import gfem

base = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/"
for name in ("transducer_section.py", "aom_section.py"):
    urllib.request.urlretrieve(base + name, name)
from aom_section import optical_window  # noqa: E402
from transducer_section import transducer_section  # noqa: E402

# the acoustic crystal: stiffness (Pa) and piezoelectric tensor (C/m^2), x-cut
c11, c12, c13, c14, c33, c44 = 203e9, 53e9, 75e9, 9e9, 245e9, 60e9
c_ln = np.array(
    [
        [c11, c12, c13, c14, 0, 0],
        [c12, c11, c13, -c14, 0, 0],
        [c13, c13, c33, 0, 0, 0],
        [c14, -c14, 0, c44, 0, 0],
        [0, 0, 0, 0, c44, c14],
        [0, 0, 0, 0, c14, (c11 - c12) / 2],
    ]
)
e15, e22, e31, e33 = 3.7, 2.5, 0.2, 1.3
e_ln = np.array(
    [
        [0, 0, 0, 0, e15, -e22],
        [-e22, e22, 0, e15, 0, 0],
        [e31, e31, e33, 0, 0, 0],
    ]
)
c_si = np.zeros((6, 6))
c_si[:3, :3] = 63.9e9
c_si += np.diag([165.7e9 - 63.9e9] * 3 + [79.6e9] * 3)
axes = ((0, 1, 0), (1, 0, 0), (0, 0, -1))
q = dict(mechanical_quality_factor=200.0)
ln = gfem.Material(
    eps_r=(44.0, 44.0, 29.0), axes=axes, stiffness=c_ln,
    piezoelectric_tensor=e_ln, mass_density=4700.0, **q,
)  # fmt: skip
oxide = gfem.Material(
    eps_r=3.9, stiffness=gfem.isotropic_stiffness(73e9, 0.17), mass_density=2203.0, **q
)
silicon = gfem.Material(stiffness=c_si, mass_density=2329.0, **q)
metal = gfem.Material(
    stiffness=gfem.isotropic_stiffness(70e9, 0.35), mass_density=2700.0, **q
)

# the optical crystal: the same axes, its index, the clamped r and p^E (Weis and Gaylord)
frequency = sc.c / 1.55e-6  # Hz
p11, p12, p13, p14, p31, p33, p41, p44 = (
    -0.026, 0.090, 0.133, -0.075, 0.179, 0.071, -0.151, 0.146,
)  # fmt: skip
p_ln = np.array(
    [
        [p11, p12, p13, p14, 0, 0],
        [p12, p11, p13, -p14, 0, 0],
        [p31, p31, p33, 0, 0, 0],
        [p41, -p41, 0, p44, 0, 0],
        [0, 0, 0, 0, p44, p41],
        [0, 0, 0, 0, p14, (p11 - p12) / 2],
    ]
)
optical_ln = dataclasses.replace(
    gfem.materials.lithium_niobate(frequency=frequency, axes=axes),
    photoelastic_tensor=p_ln,
)
optical_oxide = gfem.materials.silica(frequency=frequency)
```

{py:obj}`gfem.acousto_optic_shift` reads a kept {py:class}`gfem.Piezoelectric`
solve on the optical cross-section through the {term}`photoelastic tensor` and
the electro-optic tensor, first order, with the mode held constant. The device is
the {doc}`acousto-optic tutorial <../tutorials/acousto-optic>`'s, meshed by
{download}`transducer_section.py <../tutorials/transducer_section.py>` and
{download}`aom_section.py <../tutorials/aom_section.py>`. The optical crystal
carries the tensor; the acoustic model's material for the same crystal must
share its `axes`, which the shift cannot check.

- A material is either the optical crystal (with an `electro_optic_tensor`) or
  the acoustic one (with a `stiffness` or `piezoelectric_tensor`); giving both raises.
- The shift treats $p$ as $p^E$ and the electro-optic tensor as the clamped
  $r^S$, which {py:obj}`gfem.materials.lithium_niobate` carries: the strain
  counts through $p$ and the piezoelectric field through $r$. A free $r^T$ would
  count the strain twice.
- `p_ln` is a bulk-crystal literature set; published values differ by tens of
  percent. The shift is linear in the tensor, so another set needs no new acoustic solve.
- `axes` must be right-handed.

## The shift per volt

The acoustic section is solved once, with the fields kept. Each arm has an
{term}`optical window` in the section's own coordinates.

```python
section = transducer_section(edge_scale=2.5)
frame = gfem.DampingFrame(["ln_frame", "box_frame", "si_frame"])
model = gfem.Model(
    section.mesh,
    materials={
        "ln": ln, "ln_frame": ln, "box": oxide, "box_frame": oxide,
        "si": silicon, "si_frame": silicon, "signal": metal, "ground": metal,
        "air": gfem.Material(),
    },
    excluded={"mechanical": ["air"], "em": ["si", "si_frame"]},
)  # fmt: skip
sweep = gfem.Piezoelectric(
    [0.85e9, 0.8625e9, 0.875e9],
    terminals={"signal": "signal"},
    ground="ground",
    clamped="bottom",
    damping_frame=frame,
    keep_fields=True,
)
acoustic = gfem.solve(model, sweep)


def window(x_centre, crystal=optical_ln):
    """One arm's optical window; its mode is on the end face "front"."""
    materials = {"box": optical_oxide, "slab": crystal, "rib": crystal}
    return gfem.Model(
        optical_window(x_centre, h_film=0.12, h_outer=0.5),
        materials={**materials, "air": gfem.Material()},
        ports=[gfem.WavePort("arm", "front")],
    )


arms = [window(x) for x in section.arms[:, 0]]
a, b = (
    gfem.acousto_optic_shift(
        w, acoustic, port="arm", frequency=frequency, terminal="signal", mode=1
    )
    for w in arms
)
print(a)
print(
    f"arm b / arm a: {abs(complex(b.delta_n_eff_per_volt[1] / a.delta_n_eff_per_volt[1])):.4f}"
)
print(f"V_pi L of arm a at 0.8625 GHz: {100 * float(a.v_pi_l[1]):.3f} V cm")
```

```text
AcoustoOpticShift(port 'arm', mode 1, terminal 'signal', 1.93414e+14 Hz, n_eff 1.840038, 3 acoustic frequencies 8.5e+08 to 8.75e+08 Hz, v_pi_l 0.0774106 to 0.0942257 V m)
arm b / arm a: 0.9875
V_pi L of arm a at 0.8625 GHz: 8.496 V cm
```

- `delta_n_eff_per_volt` is the complex $\delta n$ at each `acoustic_frequency`,
  per volt on `terminal`, others at 0 V: with a drive $\mathrm{Re}(V e^{i\Omega t})$
  the index is $n_\mathrm{eff} + \mathrm{Re}(\delta n\,V e^{i\Omega t})$
  ({doc}`../conventions`). One mode solve serves every kept frequency and gives `n_eff`.
- `phase_per_length_per_volt` is $k_0\,\delta n$ in rad/(V m); `v_pi_l` is
  $\lambda / (2|\delta n|)$ in V m of one arm (`100 * v_pi_l` in V cm).
- A result without fields, a missing terminal and several terminals with
  `terminal=None` raise `ValueError`.
- `mode=1` is the TE mode of this window; check the polarization of the mode you
  pick, as in {doc}`electro-optic-phase`.

## The photoelastic and the electro-optic part

The shift is the sum of the strain through $p$ and the field through $r$:
`photoelastic_per_volt` and `electro_optic_per_volt`.

```python
pe, eo = a.photoelastic_per_volt[1], a.electro_optic_per_volt[1]
print(f"photoelastic {complex(pe):.3e}, electro-optic {complex(eo):.3e}")
print(
    f"|EO / PE| = {abs(complex(eo / pe)):.3f}, phase {np.degrees(np.angle(complex(-eo / pe))):+.1f} deg from opposed"
)
print(f"|total / PE| = {abs(complex(a.delta_n_eff_per_volt[1] / pe)):.3f}")
```

```text
photoelastic -5.481e-06+1.200e-05j, electro-optic 2.592e-06-3.353e-06j
|EO / PE| = 0.321, phase +13.2 deg from opposed
|total / PE| = 0.691
```

On this stack they have opposite signs and the electro-optic part is about a
third of the photoelastic one: a model without either is wrong by tens of percent.

## Push-pull and the incident wave

The two ribs sit an odd number of half-wavelengths apart, so the arms' shifts are
in antiphase. Driven push-pull in a Mach-Zehnder interferometer, the phase
difference is the difference of the shifts:

$$
V_\pi L = \frac{\lambda}{2\,|\delta n_a - \delta n_b|}
$$

per volt on the fingers, not of the source. For an incident wave
$V_\mathrm{inc}$ on $Z_0$ the fingers carry
$V_\mathrm{IDT} = 2Z\,V_\mathrm{inc}/(Z_L + Z_0)$, with $Z = 1/(W Y')$ at the
aperture $W$ and $Z_L = Z + R_s$. The product per volt of incident wave is the
one per volt on the fingers times $|(Z_L + Z_0)/(2Z)|$.

```python
wavelength = sc.c / frequency
v_pi_l = (
    100
    * wavelength
    / (2 * np.abs(np.asarray(a.delta_n_eff_per_volt - b.delta_n_eff_per_volt)))
)
aperture, r_s, z_0 = 400e-6, 26.7, 50.0  # m, ohm (the tutorial's), ohm
z = 1 / (aperture * np.asarray(acoustic.admittance_per_length)[:, 0, 0])  # ohm
refer = np.abs((z + r_s + z_0) / (2 * z))
for f, pp, k in zip(a.acoustic_frequency, v_pi_l, refer, strict=True):
    print(
        f"{f / 1e9:.4f} GHz: {pp:.2f} V cm per volt on the fingers, {pp * k:.2f} V cm incident"
    )
```

```text
0.8500 GHz: 3.89 V cm per volt on the fingers, 2.07 V cm incident
0.8625 GHz: 4.27 V cm per volt on the fingers, 2.28 V cm incident
0.8750 GHz: 4.74 V cm per volt on the fingers, 2.52 V cm incident
```

Arms of length $L$ (the aperture) driven with $V$ on the fingers differ in phase
by $\pi V L / (V_\pi L)$, so $V_\pi = V_\pi L / L$.

## Where the windows sit, and the gradient

`offset=(dx, dy)` in metres moves the reading of the acoustic fields, not the
mode, and is differentiable. `jax.grad` reaches the tensors (give the crystal as
a {py:class}`gfem.MaterialField`), the offset, and every input the piezoelectric
solve differentiates ({doc}`piezoelectric-gradient`). The optical model's nodes
are held constant: a gradient through them raises
{py:class}`gfem.errors.NonDifferentiableError`.

```python
def push_pull(p, slide):
    """|dn_a - dn_b| per volt at 0.8625 GHz, with the windows slid by `slide` (m)."""
    crystal = gfem.MaterialField(
        eps_r=jnp.asarray(optical_ln.eps_r),
        axes=axes,
        electro_optic_tensor=optical_ln.electro_optic_tensor,
        photoelastic_tensor=p,
    )
    dn = [
        gfem.acousto_optic_shift(
            window(x, crystal), acoustic, port="arm", frequency=frequency,
            terminal="signal", mode=1, offset=jnp.stack([slide, 0.0 * slide]),
        ).delta_n_eff_per_volt[1]
        for x in section.arms[:, 0]
    ]  # fmt: skip
    return jnp.abs(dn[0] - dn[1])


p0, s0 = jnp.asarray(p_ln), jnp.asarray(0.0)
value, (d_p, d_slide) = jax.value_and_grad(push_pull, argnums=(0, 1))(p0, s0)
print(f"|dn_a - dn_b| = {float(value):.4e} per V")
shares = np.asarray(p0 * d_p / value)  # the change per unit relative change of p_IJ
labels = ("xx", "yy", "zz", "yz", "xz", "xy")
for k in np.argsort(-np.abs(shares), axis=None)[:3]:
    i, j = divmod(int(k), 6)
    print(f"p_{labels[i]},{labels[j]}: {shares[i, j]:+.3f}")
h = 1e-12  # m: no reading point crosses an element edge in this step
fd = (push_pull(p0, s0 + h) - push_pull(p0, s0 - h)) / (2 * h)
print(
    f"d/dslide = {float(d_slide):.4e} per V and per m, central difference {float(fd):.4e}"
)
```

```text
|dn_a - dn_b| = 1.8129e-05 per V
p_yy,yz: +1.191
p_yy,xx: +0.153
p_yy,yy: +0.105
d/dslide = -3.9568e-01 per V and per m, central difference -3.9568e-01
```

- **The step of a check.** The strain jumps across element edges, so a central
  difference in `offset` whose step carries a reading point across an edge is no
  check: use a step far smaller than the elements (here 1 pm for edges of
  0.1 µm), at two steps. The same holds for a node map moving the acoustic mesh.
- Through a piezoelectric solve the gradient adds that solve's adjoint.

## Check your results

- **Antiphase.** $\arg(\delta n_b/\delta n_a) = 180°$ to a fraction of a
  degree when the ribs are an odd number of half wavelengths apart.
- **Linearity.** Doubling `photoelastic_tensor` doubles `photoelastic_per_volt`.
- **Convergence.** Refine `edge_scale` (the tutorial's 1.5 against 2.5 moves the
  shift by 2 %) and the window's `h_film` and `h_outer` (0.15 %).
- **The mode.** Compare `n_eff` with a {py:class}`gfem.PortMode` solve of the window alone.

## Limits

- Photoelastic and electro-optic terms only: moving boundaries and the
  roto-optic rotation are not counted. They vanish to first order for a wave that
  moves no boundary of the section, such as the shear wave here; for Rayleigh and
  Lamb waves this is an approximation.
- Lossless optical section: a complex `eps_r`, a `tan_delta` or a `conductivity`
  raises `NotImplementedError`, as do a 2D optical model and a sequence of
  optical frequencies.
- A point outside the acoustic mechanical domain (with a `photoelastic_tensor`)
  or outside the potential's support (with an `electro_optic_tensor`) raises
  `ValueError`, naming it. A model with neither tensor raises.
- A cross-section: the shift is uniform along the arms' length, the aperture.

## Next

- {doc}`../tutorials/acousto-optic`: the push-pull modulator over a band.
- {doc}`piezoelectric-admittance`: the admittance and the strain.
- {doc}`electro-optic-phase`: the static-field counterpart.
- {doc}`piezoelectric-gradient`: the gradients through the acoustic solve.
