# Acousto-optic index change

Find how much a voltage on an interdigital transducer shifts the effective index
of a waveguide that its sound wave crosses: the {term}`acousto-optic shift` of a
guided mode per volt, as a complex phasor at each acoustic frequency, the
voltage-length product $V_\pi L$ of one arm and of a push-pull pair, and the
voltage of the wave that reaches the transducer. A
{py:class}`gfem.Piezoelectric` solve that keeps its fields gives the strain and the
electric field of the wave; {py:obj}`gfem.acousto_optic_shift` reads them on the
optical cross-section and weighs them against one mode solve, through the crystal's
{term}`photoelastic tensor` and electro-optic tensor.

The device is the section of the {doc}`acousto-optic tutorial
<../tutorials/acousto-optic>`: x-cut lithium niobate film, two ribs, aluminium
fingers on an oxide and silicon stack, light at 1.55 µm. The tutorial's
{download}`transducer_section.py <../tutorials/transducer_section.py>` and
{download}`aom_section.py <../tutorials/aom_section.py>` mesh the acoustic section and
the optical windows with netgen; here at 2.5 times the tutorial's edge lengths and
three frequencies, which is quicker and moves the shift by 2 %. The shift is first
order in the acoustic fields: the mode is solved once and held constant, and the shift
is the sum over the section of its weights times the change of permittivity the fields
cause.

## A crystal with a photoelastic tensor

A material's `photoelastic_tensor` is the contracted $p^E_{IJ}$, a 6 × 6 array,
dimensionless, in the frame of the material's `axes`: rows and columns in the Voigt
order $xx, yy, zz, yz, xz, xy$, with the engineering shear strain, so that
$\Delta(\varepsilon_r^{-1})_I = p_{IJ} S_J$
({doc}`../conventions`, "Photoelastic tensor"). It belongs to the optical crystal,
beside its `eps_r` and its `electro_optic_tensor`; the acoustic model has its own
material for the same crystal, with the stiffness and the piezoelectric tensor, and the
two must share one `axes`, which the shift cannot check.

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

- The tensor is a property of the material, so a region of crystal carries it in every
  model it appears in. It may sit beside an `electro_optic_tensor` and not beside a
  `stiffness` or a `piezoelectric_tensor`: a material is either the optical
  crystal or the acoustic one, and giving both raises.
- The shift treats $p$ as the tensor at constant electric field, $p^E$, and the
  electro-optic tensor as the clamped $r^S$: the strain is counted through $p$ and the
  piezoelectric field through $r$. {py:obj}`gfem.materials.lithium_niobate` carries
  the clamped $r^S$. A free $r^T$ would count the strain twice.
- `p_ln` is a literature set for the bulk crystal, not the film's: published values
  differ by tens of percent. The shift is linear in the tensor, so another set rescales
  it with no new acoustic solve.
- `axes` must be right-handed: the tensors' signs follow the crystal's frame.

## The shift per volt

The acoustic section is solved once, at the frequencies of the band, with the fields
kept. Each arm has an {term}`optical window`: the film, the rib, oxide below and air
above, meshed in the acoustic section's own coordinates, so the acoustic fields are
read at the optical points with no offset. A wave port on its end face `"front"` carries
the mode.

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

- `delta_n_eff_per_volt` is the complex $\delta n$ at each acoustic frequency
  (`acoustic_frequency`), per volt on `terminal` with every other terminal and the
  ground at 0 V. With a drive $\mathrm{Re}(V e^{i\Omega t})$ the effective index is
  $n_\mathrm{eff} + \mathrm{Re}(\delta n\,V e^{i\Omega t})$ to first order, the
  phasor convention of gfem's $e^{+i\omega t}$ ({doc}`../conventions`).
- One mode solve serves every kept frequency: the acoustic fields are only read. It
  gives `n_eff`, the index without the wave.
- `phase_per_length_per_volt` is $k_0\,\delta n$ in rad/(V m).
- `v_pi_l` is $\lambda / (2|\delta n|)$ in V m of one arm at each acoustic frequency:
  `100 * v_pi_l` is in V cm.
- A result that kept no fields, a terminal the result does not have, and several
  terminals with `terminal=None` raise `ValueError`; a result that kept fields for a
  list of frequencies gives those.
- `mode=1` is the TE mode of this window. Check the polarization of the mode you pick
  from its field, as in {doc}`electro-optic-phase`.

## The photoelastic and the electro-optic part

The shift is the sum of two phasors: the strain through $p$ and the field through $r$.
`photoelastic_per_volt` leaves the electro-optic tensor out, and
`electro_optic_per_volt` the photoelastic one.

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

On this stack the two have opposite signs and the electro-optic part is about a third
of the photoelastic one, so a model without the strain, or without the field, is wrong
by tens of percent. Both parts are counted in `delta_n_eff_per_volt`.

## Push-pull and the incident wave

The two ribs sit an odd number of half-wavelengths apart, so the arms' shifts are in
antiphase. Driven push-pull in a Mach-Zehnder interferometer, the phase difference is
the difference of the shifts:

$$
V_\pi L = \frac{\lambda}{2\,|\delta n_a - \delta n_b|}
$$

per volt on the fingers. That is not the voltage of the source. For an incident wave
$V_\mathrm{inc}$ on a source impedance $Z_0$, the fingers carry
$V_\mathrm{IDT} = 2Z\,V_\mathrm{inc}/(Z_L + Z_0)$, with $Z = 1/(W Y')$ the
transducer's impedance at the aperture $W$, from `admittance_per_length`, and
$Z_L = Z + R_s$ its load with the fingers' series resistance. The product per volt of
incident wave is the one per volt on the fingers times $|(Z_L + Z_0)/(2Z)|$.

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

Arms of length $L$ driven with amplitude $V$ on the fingers have a phase difference
$\pi V L / (V_\pi L)$, so $V_\pi = V_\pi L / L$. The arms run along the fingers, so
their length is the transducer's aperture.

## Where the windows sit, and the gradient

`offset=(dx, dy)` in metres moves the optical section against the acoustic one: the
fields are read at the optical points plus `offset`. It is a move of the reading,
not of the mode, and differentiable. The shift is exactly linear in
the tensors and the acoustic fields, and `jax.grad` reaches the tensors (give the
crystal as a {py:class}`gfem.MaterialField`), the offset, and every input the
piezoelectric solve differentiates (the acoustic materials and the node maps of
{doc}`piezoelectric-gradient`), through the kept fields. The optical model's own nodes
are held constant: a `jax.grad` through them raises
{py:class}`gfem.errors.NonDifferentiableError`, and a waveguide width needs a
remesh and a finite difference.

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

- **The step of a check.** The strain is piecewise polynomial on the acoustic mesh and
  jumps across element edges, and the derivative is that of the reading at fixed
  points in their elements. A central difference in `offset` whose step carries a
  reading point across an edge adds the jump and is no check of the gradient: use
  a step far smaller than the elements (here 1 pm for edges of 0.1 µm), and read
  the difference at two steps. The same holds for a node map moving the acoustic
  mesh under fixed points.
- **The acoustic solve is the gradient's cost.** With the result held fixed, the
  gradient costs one adjoint pass over the readings and the mode solves of the
  forward call. Through a piezoelectric solve it adds that solve's adjoint.

## Check your results

- **Antiphase.** The two arms' shifts have opposite phase, $\arg(\delta n_b/\delta n_a)
  = 180°$ to a fraction of a degree, when the ribs are an odd number of half
  wavelengths apart: this follows from the geometry, not the frequency.
- **Linearity.** The shifts of several terminals add, each times its drive; the shift
  scales with the tensors. Doubling `photoelastic_tensor` doubles `photoelastic_per_volt`.
- **Convergence.** The shift reads the strain's gradient on the acoustic mesh, and the
  mode on the window's. Refine `edge_scale` (the tutorial's 1.5 against this page's 2.5
  moves the shift by 2 %) and the window's `h_film` and `h_outer` (the tutorial's
  0.08 and 0.4 µm against this page's 0.12 and 0.5 µm: 0.15 %).
- **The mode.** Compare `n_eff` with a mode solve of the window alone
  ({py:class}`gfem.PortMode`).

## Limits

- Photoelastic and electro-optic terms only. The moving boundaries and the rotation of
  the crystal by the displacement (roto-optic) are not counted. For a wave whose
  displacement moves no boundary of the optical section, such as the shear wave
  polarized along the guide here, they vanish to first order; for Rayleigh and Lamb
  waves, which move the section's boundaries, they are an approximation.
- First order in the acoustic fields, with the mode held constant, for a lossless
  optical section: a complex `eps_r`, a `tan_delta` or a `conductivity` raises
  `NotImplementedError`, and so does a 2D optical model and a sequence of optical
  frequencies.
- The strain and the field are read at the quadrature points of each tetrahedron
  behind the section's triangles. A point outside the acoustic mechanical domain, for
  an element with a `photoelastic_tensor`, or outside the potential's support, for one
  with an `electro_optic_tensor`, raises `ValueError` and names it.
- An optical element with neither tensor adds nothing, and a model with none raises.
- A cross-section: the arms' length is the aperture and the shift uniform along it.

## Next

- {doc}`../tutorials/acousto-optic`: the push-pull modulator, the shift over a band, the
  derivative in the pitch and the position of the arms.
- {doc}`piezoelectric-admittance`: the admittance, the damping frames and the strain.
- {doc}`electro-optic-phase`: the static-field counterpart, with a potential as the
  drive.
- {doc}`piezoelectric-gradient`: the gradients through the acoustic solve.
