# Optomechanical coupling

Find the vacuum optomechanical coupling $g_0$ of the mechanical modes of a
suspended silicon wire to the guided optical mode of the same wire: the
shift of the optical resonance per zero-point amplitude of a vibration, the
quantity that sets the Brillouin gain of the guide and the optomechanical
rate of a ring built from it. {py:obj}`gfem.optomechanical_coupling` reads the
modes of {py:class}`gfem.MechanicalEigenmode` against one optical solve and
returns $\tilde g_0$ for every mode, split into the moving-boundary and the
photoelastic part; for a guide of length $L$, $g_0 = \tilde g_0/\sqrt{L}$.

The wire is the published one (Rakich et al., Phys. Rev. X 2, 011008 (2012); Qiu
et al., Opt. Express 21, 31402 (2013)): 315 nm by 283.5 nm, light at 1.552 µm,
meshed twice by {download}`wire_section.py <wire_section.py>`.

## Assumptions

- The modes overlap at one cross-section, the mechanical one at $q = 0$: an
  intramodal forward interaction. The {term}`optomechanical overlap` is the
  first-order change of the optical $n_\mathrm{eff}$ at fixed frequency.
- The optical section is lossless and its mode held constant. The wire is an
  isotropic solid (170 GPa, $\nu = 0.28$) with a cubic photoelastic tensor, as
  published.
- The meshes need not share nodes. The optical outer wall is a truncation and
  must clear the moving parts.

## Conventions

Modes carry 1 J per metre of guide. $\tilde g_0$ is in rad s$^{-1}$ m$^{1/2}$ and
$g_0$ in rad/s: divide by $2\pi$ for hertz. The sign of $\tilde g_0$ follows the
mode's phase (largest displacement coefficient real and positive); $|g_0|$ does
not ({doc}`../conventions`, "Optomechanical coupling").

## The wire and its vibrations

Silicon has index 3.5 and $p_{11} = -0.09$, $p_{12} = 0.017$, $p_{44} = -0.051$
for the light, and a quality factor of 1000 for the sound, which only enters the
gain.

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


Symmetry decides which mode couples: the share of $|u_z|^2$ tells the shear modes
(motion along the guide) from the in-plane ones:

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

One call solves the optical mode once, the first of the port, and reads every
mechanical mode against it:

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


The first mode, in-plane and even under both mirrors, couples; the shear mode
couples to nothing and the third, odd under a mirror, almost nothing. Pick modes
with `mechanical_modes=`.

The two parts of the first mode, and the coupling of a ring of radius 10 µm
treated as a cavity of length $L = 2\pi R$:

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


The parts add here; for another wire or mode they can oppose. Read $|g_0|$.

## The Brillouin gain of the guide

The forward gain follows from $\tilde g_0$, $Q_m$ and $v_g = c/n_g$ (Van Laer et
al., Phys. Rev. A 93, 053828 (2016)):

$$
G_\mathrm{B} = \frac{4 Q_m \tilde g_0^2}{\hbar\omega\,v_g^2\,\mathrm{Re}\,\Omega},
$$

in 1/(W m). The parts add in amplitude, not in gain:

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


The published values for this wire are 4400, 4200 and 17200 1/(W m). This mesh
is 2 % low in the total, converging from below ({doc}`../verification`,
"Optomechanical coupling").

## The overlap of any displacement

{py:obj}`gfem.optomechanical_overlap` reads any displacement field map on a 2D
mesh, such as a driven {py:class}`gfem.Piezoelectric` problem's at $q = 0$. It
gives $\partial n_\mathrm{eff}/\partial\alpha$ for the displacement scaled by
$\alpha$; the mode's coupling is that times $(\omega/n_g)\sqrt{\hbar\,\mathrm{Re}\,\Omega/4}$:

```python
overlap = gfem.optomechanical_overlap(
    optical, modes.displacement(0), port="p", frequency=frequency
)
scale = omega / overlap.n_g * np.sqrt(sc.hbar * modes.omega[0].real / 4)
print(
    f"d n_eff / d alpha: moving boundary {overlap.moving_boundary.real:.4g}, photoelastic {overlap.photoelastic.real:.4g}"
)
print(
    f"scaled by the zero-point factor, the total is the coupling: {complex(scale * overlap.total / coupling.total[0]).real:.6f}"
)
```


```text
d n_eff / d alpha: moving boundary -17.91, photoelastic -17.37
scaled by the zero-point factor, the total is the coupling: 1.000000
```


## The gradient of the coupling

The coupling is differentiable in every input of the mechanical solve and in the
photoelastic tensor of a {py:obj}`gfem.MaterialField`. For an isotropic wire,
$\Omega \propto E^{1/2}$ and the overlap $\propto E^{-1/2}$, so
$\mathrm{d}\ln\tilde g_0/\mathrm{d}\ln E = -1/4$:

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

- **Parts and total.** The total gain is $(\sqrt{G_\mathrm{MB}} \pm \sqrt{G_\mathrm{PE}})^2$,
  plus when the parts have the same sign.
- **A finer mesh.** Halving the edge lengths near the wire takes the total gain
  from 2.3 % to 0.1 % off the published value.
- **Symmetry.** A mode moving only along $z$ has $|\tilde g_0|$ at round-off; one
  odd under a mirror couples at 1.4 % of the even mode's here.
- **The normalization.** `gfem.optomechanical_overlap` of a displacement of
  another size gives the overlap per unit of that displacement, not $\tilde g_0$.

## Limits

- Forward intramodal couplings at $q = 0$ only: a mechanical mode or displacement
  with $q \ne 0$ raises `ValueError`; the backward gain is {doc}`brillouin-gain`.
- A 3D slab model and a lossless section only: a 2D optical model or a lossy
  section (a complex `eps_r`, a `tan_delta` or a `conductivity`) raises
  `NotImplementedError`.
- The roto-optic rotation is not counted. A photoelastic element outside the
  mechanical mesh raises `ValueError`, naming its region.
- The optical mode is held constant: a traced optical permittivity, conductivity
  or node coordinate raises `NonDifferentiableError`, as does a call inside
  `jax.jit` (call it outside and jit the rest).
- A displacement that moves a node across the optical outer wall raises
  `ValueError`: enlarge the optical section.
- `mechanical_modes` out of range raises `IndexError`; an empty selection or a
  length that is not positive raises `ValueError`. A clustered mechanical mode has
  no gradient.

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

- {doc}`mechanical-modes`: the vibrations of a section.
- {doc}`acousto-optic-shift`: the photoelastic shift of a driven wave.
- {doc}`optical-force`: the force of the light on the wire, the reciprocal of the coupling.
- {doc}`brillouin-gain`: the forward and backward gain of every mode.
- {doc}`waveguide-dispersion`: the group index $n_g$ that sets the scale.
