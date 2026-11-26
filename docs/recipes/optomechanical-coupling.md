# Find how strongly sound in a wire couples to the light it guides

Find the vacuum optomechanical coupling $g_0$ of the mechanical modes of a
suspended silicon wire to the guided optical mode of the same wire: the
shift of the optical resonance per zero-point amplitude of a vibration, the
quantity that sets the Brillouin gain of the guide and the optomechanical
rate of a ring built from it. {py:class}`gfem.MechanicalEigenmode` gives the
vibrations of the wire's 2D section at zero axial wavenumber;
{py:obj}`gfem.optomechanical_coupling` reads them against one solve of the
optical mode and returns $\tilde g_0$ for every mode, split into the part from
the moving boundary and the part from the photoelastic effect. The
{term}`vacuum optomechanical coupling` per unit length is $\tilde g_0$, and for
a guide of length $L$ it is $g_0 = \tilde g_0/\sqrt{L}$.

The wire is the suspended silicon wire of the published Brillouin
measurements (Rakich et al., Phys. Rev. X 2, 011008 (2012); Qiu et al., Opt.
Express 21, 22544 (2013)): 315 nm wide and 283.5 nm tall, light at
1.552 µm, in air.
{download}`wire_section.py <wire_section.py>` meshes it with netgen, twice with
the same global $(x, y)$: the wire and an air margin of 1 µm for the light,
the wire alone for the sound.

## Assumptions

- The two modes overlap at the same cross-section, with the optical mode
  propagating along $z$ and the mechanical mode at $q = 0$: the vibration
  is uniform along the guide, which couples the optical mode to itself (an
  intramodal, forward interaction). The overlap is the first-order change of
  the optical $n_\mathrm{eff}$ at fixed frequency.
- The optical section is lossless, and the optical mode is held constant: the
  coupling is exact in the mechanical inputs and the photoelastic tensor, and
  the optical model's own inputs are not differentiable.
- The wire is a cubic crystal approximated by an isotropic solid
  (170 GPa, $\nu = 0.28$) and a cubic photoelastic tensor, as in the
  published case. A real wire's orientation moves both.

## Conventions

The mechanical modes are normalized to 1 J of stored energy per metre of
guide. The coupling is in angular units: $\tilde g_0$ in
rad s$^{-1}$ m$^{1/2}$ and $g_0$ in rad/s, so divide by $2\pi$ for hertz. The
sign of $\tilde g_0$ follows the mode's phase convention, which makes its
largest displacement coefficient real and positive; $|g_0|$ does not depend on
it and the two parts of one mode keep their relative sign. See
{doc}`../conventions`, "Optomechanical coupling".

## The wire and its vibrations

The materials: silicon for the light, with the index 3.5 and the
photoelastic constants $p_{11} = -0.09$, $p_{12} = 0.017$, $p_{44} = -0.051$
of the published case, and silicon for the sound with a mechanical quality
factor of 1000, which only enters the gain. The outer wall of the optical
section is a perfect electric conductor, far from the mode.

```python
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
modes = gfem.solve(
    mechanical,
    gfem.MechanicalEigenmode(12.5e9, n_modes=3),
    options=gfem.SolverOptions(order=3),
)
print(modes)
```

```text
MechanicalEigenmodeResult([12.5613 GHz (Q 1000), 12.67 GHz (Q 1000), 13.1399 GHz (Q 1000)])
```

The three modes nearest 12.5 GHz at $q = 0$, cubic elements. The wire is free
on every side, so every mode is a vibration of the wire alone. Which of them
couples is a matter of symmetry, so look at their motion before picking one.
The share of $|u_z|^2$ tells the shear modes (motion along the guide, which
cannot move the optical mode's section) from the in-plane ones:

```python
x, y = np.meshgrid(
    np.linspace(-0.8, 0.8, 9) * width / 2, np.linspace(-0.8, 0.8, 9) * height / 2
)
points = list(zip(x.ravel(), y.ravel(), strict=True))
for k in range(modes.n_modes):
    u = modes.displacement(k)
    power = {c: np.sum(np.abs(np.asarray(u[c].at(points))) ** 2) for c in "xyz"}
    print(
        f"mode {k}: {modes.frequency[k] / 1e9:.3f} GHz, motion along z {power['z'] / sum(power.values()):.2f}"
    )
```

```text
mode 0: 12.561 GHz, motion along z 0.00
mode 1: 12.670 GHz, motion along z 1.00
mode 2: 13.140 GHz, motion along z 0.00
```

## The coupling of every mode

One call solves the optical mode once and reads every mechanical mode against
it. The optical mode is the first one of the port, its effective and group
index come with the result:

```python
coupling = gfem.optomechanical_coupling(optical, modes, port="p", frequency=frequency)
print(
    f"n_eff {float(coupling.overlap.n_eff):.4f}, n_g {float(coupling.overlap.n_g):.4f}"
)
for k in coupling.modes:
    print(f"mode {k}: |g0 sqrt(L)| = {abs(coupling.total[k]):.4g} rad/s m^1/2")
```

```text
n_eff 1.8392, n_g 5.6650
mode 0: |g0 sqrt(L)| = 1.09e+04 rad/s m^1/2
mode 1: |g0 sqrt(L)| = 2.281e-11 rad/s m^1/2
mode 2: |g0 sqrt(L)| = 157.3 rad/s m^1/2
```

The first mode is the in-plane breathing mode of the wire, even under both
mirrors of the section, which is the one that couples. The shear mode of 12.67
GHz couples to nothing (its motion is along $z$ and the section is invariant
along $z$), and the third, in-plane but odd under a mirror, almost nothing.
Pick the modes you want by number with `mechanical_modes=` once you know their
motion; the optical solve is not repeated either way.

The two parts of the first mode, and the coupling of a ring of $L = 2\pi R$
with a radius of 10 µm treated as a cavity of that length:

```python
mb, pe = coupling.moving_boundary[0], coupling.photoelastic[0]
print(
    f"moving boundary {mb.real:.4g}, photoelastic {pe.real:.4g}, total {coupling.total[0].real:.4g} rad/s m^1/2"
)
length = 2 * np.pi * 10e-6  # m
g0 = coupling.g0(length)[0]
print(
    f"g0 / 2 pi = {abs(g0) / (2 * np.pi) / 1e3:.1f} kHz for L = {length * 1e6:.1f} um"
)
```

```text
moving boundary -5534, photoelastic -5367, total -1.09e+04 rad/s m^1/2
g0 / 2 pi = 218.9 kHz for L = 62.8 um
```

The two parts have the same sign here, so they add: the displacement of the
wire's sides and the strain inside it both raise (or both lower) the
effective index. They can have opposite signs for another wire or another mode.
The sign printed is the mode's phase convention; read $|g_0|$.

## The Brillouin gain of the guide

The forward intramodal Brillouin gain of a guide of any length follows from
$\tilde g_0$, the mechanical quality factor $Q_m$ and the optical group
velocity $v_g = c/n_g$ (Van Laer et al., Phys. Rev. A 93, 053828 (2016)):

$$
G_\mathrm{B} = \frac{4 Q_m \tilde g_0^2}{\hbar\omega\,v_g^2\,\mathrm{Re}\,\Omega},
$$

in 1/(W m), $\omega$ the optical and $\Omega$ the mechanical angular
frequency. The parts' gains follow from their own $\tilde g_0$; the total is
not their sum, since the parts add in amplitude:

```python
omega = 2 * np.pi * frequency
v_g = sc.c / coupling.overlap.n_g


def gain(g):
    """The Brillouin gain, 1/(W m), of a coupling g (rad/s m^1/2) of mode 0."""
    return (
        4 * modes.q[0] * abs(g) ** 2 / (sc.hbar * omega * v_g**2 * modes.omega[0].real)
    )


print(
    f"moving boundary {gain(mb):.0f}, photoelastic {gain(pe):.0f}, total {gain(coupling.total[0]):.0f} 1/(W m)"
)
```

```text
moving boundary 4330, photoelastic 4073, total 16802 1/(W m)
```

The published values for this wire, with the radiation-pressure and
electrostriction parts of that work, are 4400, 4200 and 17200 1/(W m). This
mesh is coarse: it is 2 % low in the total, and the gain converges from
below as the mesh is refined ({doc}`../verification`, "Optomechanical
coupling").

## The overlap of any displacement

`gfem.optomechanical_coupling` reads the displacements of a mechanical solve.
{py:obj}`gfem.optomechanical_overlap` reads any displacement
{py:obj}`gfem.FieldMap` with the components $x, y, z$ on a 2D mesh, whatever
made it: another mechanical solve, or the displacement of a driven
{py:class}`gfem.Piezoelectric` problem at $q = 0$, per volt. It gives
$\partial n_\mathrm{eff}/\partial\alpha$ for the displacement scaled by a
dimensionless $\alpha$, as the two parts, with no zero-point scale. The
coupling of the mode is that overlap times
$(\omega/n_g)\sqrt{\hbar\,\mathrm{Re}\,\Omega/4}$:

```python
overlap = gfem.optomechanical_overlap(
    optical, modes.displacement(0), port="p", frequency=frequency
)
scale = omega / overlap.n_g * np.sqrt(sc.hbar * modes.omega[0].real / 4)
print(
    f"d n_eff / d alpha: moving boundary {overlap.moving_boundary.real:.4g}, photoelastic {overlap.photoelastic.real:.4g}"
)
print(
    f"scaled by the zero-point factor, the total is the coupling: {complex(scale * overlap.total / coupling.total[0]):.6f}"
)
```

```text
d n_eff / d alpha: moving boundary -17.91, photoelastic -17.37
scaled by the zero-point factor, the total is the coupling: 1.000000-0.000000j
```

## The gradient of the coupling

The coupling is differentiable in every input of the mechanical solve: the
stiffness, density, quality factor, piezoelectric tensor, permittivity and
the node coordinates of the mechanical mesh, and in the photoelastic tensor
of a {py:obj}`gfem.MaterialField`. One `jax.grad` of the solve and the
coupling gives the derivative with the backward pass of the mode. For an
isotropic wire a scaling law checks it: with the stiffness scaled by a factor,
$\Omega \propto E^{1/2}$ and the mode shape is unchanged; normalized to 1 J/m
its displacement scales as $E^{-1/2}$, so the overlap does too and
$\tilde g_0 \propto E^{-1/2}\,\Omega^{1/2} \propto E^{-1/4}$:
$\mathrm{d}\ln\tilde g_0/\mathrm{d}\ln E = -1/4$.

```python
def coupling_of(young):
    wire = gfem.MaterialField(
        stiffness=gfem.isotropic_stiffness(young, 0.28),
        mass_density=2329.0,
        mechanical_quality_factor=1000.0,
    )
    solved = gfem.solve(
        gfem.Model(mechanical.mesh, materials={"wire": wire}),
        gfem.MechanicalEigenmode(12.5e9, n_modes=1),
        options=gfem.SolverOptions(order=3),
    )
    return jnp.real(
        gfem.optomechanical_coupling(
            optical, solved, port="p", frequency=frequency
        ).total[0]
    )


young = 170e9
d_log = jax.grad(coupling_of)(young) * young / coupling_of(young)
print(f"d ln g0 / d ln E = {float(d_log):.5f}")
```

```text
d ln g0 / d ln E = -0.25000
```

## Check your results

- **Parts and total.** The total gain is
  $(\sqrt{G_\mathrm{MB}} + \sqrt{G_\mathrm{PE}})^2$ when the two parts have
  the same sign, and $(\sqrt{G_\mathrm{MB}} - \sqrt{G_\mathrm{PE}})^2$ when
  they do not.
- **A finer mesh.** The optical edge length near the wire and the mechanical
  one set the error: halving the edge lengths near the wire takes the total gain from 2.3 % to 0.1 % of
  the published value. The mode's frequency, 12.561 GHz, does not change in
  its fourth digit.
- **Symmetry.** A mode that moves only along $z$ has $|\tilde g_0|$ at the
  round-off of the solve; one that is odd under a mirror of the section couples
  only through the mesh's own asymmetry, 1.4 % of the even mode's here.
- **The normalization.** The mode stores 1 J/m, which the formula assumes:
  `gfem.optomechanical_overlap` of a displacement of another size gives the
  overlap per unit of that displacement, not $\tilde g_0$.

## Limits

- Forward intramodal couplings at $q = 0$ only: a mechanical mode with
  $q \ne 0$, which would couple modes of different wavenumber (Brillouin
  scattering backward or intermodal), raises `ValueError`, as does a
  displacement with a wavenumber.
- A 3D slab model, the section of a wave port in a plane $z$ = const, and
  a lossless one: a 2D optical model and a lossy section (a complex `eps_r`, a
  `tan_delta` or a `conductivity`) raise `NotImplementedError`.
- The rotation of the crystal by the displacement (roto-optic) is not counted,
  and the photoelastic part reads the strain in the elements that have a
  `photoelastic_tensor`; one outside the mechanical mesh raises `ValueError`
  naming its region, and so does a model with no photoelastic element and no
  node in the displacement's support.
- The optical mode is held constant: an optical permittivity, conductivity or
  node coordinate that `jax.grad` traces raises `NonDifferentiableError`, and
  the coupling cannot run under `jax.jit`.
- `mechanical_modes` out of range raises `IndexError`; an empty selection or
  a length that is not positive raises `ValueError`.
- A clustered (degenerate) mechanical mode has no gradient; the
  mechanical solve refuses it.

```python
cases = {
    "length": lambda: coupling.g0(0.0),
    "mode": lambda: gfem.optomechanical_coupling(
        optical, modes, port="p", frequency=frequency, mechanical_modes=3
    ),
    "2D model": lambda: gfem.optomechanical_coupling(
        mechanical, modes, port="p", frequency=frequency
    ),
}
for name, call in cases.items():
    try:
        call()
    except (ValueError, IndexError, NotImplementedError) as error:
        print(f"{name}: {type(error).__name__}")
```

```text
length: ValueError
mode: IndexError
2D model: NotImplementedError
```

## Next

- {doc}`mechanical-modes`: the vibrations of a section, their frequencies,
  quality factors and the displacement.
- {doc}`acousto-optic-shift`: the photoelastic shift of a driven wave, per
  volt.
- {doc}`optical-force`: the force of the light on the wire, the reciprocal
  of the coupling.
- {doc}`waveguide-dispersion`: the group index $n_g$ that sets the scale.
