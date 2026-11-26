# Thermal emissivity

A patterned emitter at a uniform temperature radiates into the space above it
with an {term}`emissivity` that depends on direction, polarization and
frequency. By Kirchhoff's law that emissivity is the {term}`absorptance` of the
wave arriving from the same direction in the same polarization, so you get it
from a driven solve of the cell, with no thermal solve. This page computes the
emissivity per direction, its average over the hemisphere, the
Planck-weighted total, and how to differentiate them. The cell, its Floquet
ports and the angle sweep are those of {doc}`diffraction-efficiency`.

Kirchhoff's law holds when the media are reciprocal and the emitter is in
thermal equilibrium at one temperature. Every cell gfem can build is
reciprocal and passive (gyrotropic and gain media are refused when you build
the material), so the first condition is met by construction. The second is
yours to judge: a surface fed by a hotter source is not in equilibrium.

## Emissivity per direction

An incidence $(\theta, \varphi)$ gives the emissivity toward
$(\theta, \varphi + \pi)$: the wave that arrives with its in-plane wave vector
at azimuth $\varphi$ is the time reverse of the one the surface emits toward
$\varphi + \pi$. {py:func}`gfem.diffraction.emissivity` labels its rows with the
emission direction, so the label is the sweep's $\varphi + \pi$. For a cell
with no symmetry that maps $\varphi$ to $\varphi + \pi$ the two differ, so use
the label, not the sweep's `phi`.

The cell below is a lossy film on a metal-backed substrate, meshed as for
{doc}`diffraction-efficiency`, with the open face `"top"`. Declare the orders
of every incidence, solve the sweep, and read the emissivity:

```python
import numpy as np

import gfem
from gfem import diffraction as dif

c = 299_792_458.0  # m/s
mesh = gfem.Mesh.from_gmsh("emitter_cell.msh", length_unit="um")
cell = gfem.Model(
    mesh,
    materials={
        "film": gfem.Material(eps_r=4.0, tan_delta=0.1),
        "air": gfem.Material(),
    },
    boundaries=[
        gfem.Periodic(("x_min", "x_max")),
        gfem.Periodic(("y_min", "y_max")),
        gfem.PEC("bottom"),
    ],
)
f0 = c / 10.0e-6  # Hz, a wavelength of 10 um
band = np.linspace(0.8, 1.2, 9) * f0
theta, phi = np.radians([0.0, 20.0, 40.0, 60.0, 80.0]), 0.0
ks = [cell.bloch_wave_vector("top", f0, theta=t, phi=phi) for t in theta]
ports = dif.floquet_ports(cell, "top", band, wave_vector=ks, wave_vector_frequency=f0)
cell = cell.replace(ports=ports)
sweep = dif.angle_sweep(
    cell,
    band,
    surface="top",
    theta=theta,
    phi=phi,
    options=gfem.SolverOptions(retain_memory=0),
)
e = dif.emissivity(sweep, model=cell, polarization="TM")
e.values  # (n_incidence, n_freq)
e.theta, e.phi  # the emission directions: phi is the sweep's phi + pi
```

- `polarization` is `"TE"`, `"TM"`, `"RCP"`, `"LCP"` or `"unpolarized"` (the
  default: thermal emission is unpolarized, and that is what "emissivity"
  means without a qualifier). TE maps to TE, TM to TM and RCP to RCP,
  right-handed about the emission direction. A Jones pair is refused: use
  `dif.power_balance` for the absorptance of one.
- The sweep is the only input that knows the direction, so `emissivity` takes
  an `AngleSweepResult`. For one direction, sweep one angle.
- A cell with an opaque backing, as here, has no transmitted rows. On a
  lossless substrate the transmitted power is counted as not absorbed.
- The medium behind the surface sets the angles. If its index is above 1,
  $\theta$ is measured in that medium and $e$ is relative to the blackbody in
  that medium.
- The balance is complete only when every propagating order has a TE and a TM
  port at every incidence, as for `power_balance`; a missing one raises with the
  name of `emissivity`. Small negative values on a lossless region are not
  clipped: they show the mesh defect.

## Hemispherical emissivity

The {term}`hemispherical emissivity` is the $\cos\theta$-weighted mean of $e$
over the directions above the surface,

$$e_h = \frac{1}{\pi}\int_0^{2\pi}\!\!\int_0^{\pi/2}
e(\theta, \varphi)\cos\theta\sin\theta\, d\theta\, d\varphi
\approx \sum_k w_k\, e(\theta_k, \varphi_k).$$

{py:func}`gfem.diffraction.hemisphere` builds the nodes and weights: Gauss-Jacobi
nodes in $\cos\theta$ times a uniform trapezoid in azimuth, with weights that
add up to 1. Solve the sweep at the rule's nodes, in order:

```python
rule = dif.hemisphere(6, 16, rotation=4, mirror=True)  # 6 polar x 3 azimuths
ks = [
    cell.bloch_wave_vector("top", f0, theta=t, phi=p)
    for t, p in zip(rule.theta, rule.phi)
]
ports = dif.floquet_ports(cell, "top", band, wave_vector=ks, wave_vector_frequency=f0)
cell = cell.replace(ports=ports)
sweep = dif.angle_sweep(
    cell,
    band,
    surface="top",
    theta=rule.theta,
    phi=rule.phi,
    options=gfem.SolverOptions(retain_memory=0),
    cache="hemisphere",
)
e_h = dif.hemispherical_emissivity(sweep, rule, model=cell)  # (n_freq,)
```

**Symmetry folding.** `rotation=r` asserts that the cell is invariant under a
rotation by $2\pi/r$ about its normal, and `mirror=True` that the reflection
$\varphi \to -\varphi$ is a symmetry: a mirror plane that contains the lattice
vector $\hat e_1$ and the normal (with `rotation=r`, the planes at
$\varphi = j\pi/r$ follow). A cell mirrored only about the plane containing
$\hat e_2$ is folded wrongly, and silently: give it `mirror=False`, or
`rotation=2, mirror=True` if it has both mirror planes. The azimuths are folded onto the fundamental
domain and equal ones merge their weights, so a cell with four-fold rotation and
mirrors needs 3 of 16 azimuths. The symmetry is your assertion: a rule folded
by a symmetry the cell lacks is wrong by the asymmetric part. An emitter that
does not depend on the azimuth, such as an unpatterned stack, takes
`hemisphere(n_theta, 1)`. `n_phi` must be a multiple of `rotation`.

**Circular polarization.** A mirror turns right-handed light into left-handed
light, so a rule folded with `mirror=True` refuses `"RCP"` and `"LCP"`. Build it
with `mirror=False`, or use `"TE"`, `"TM"` or `"unpolarized"`. Rotation folding
keeps the handedness.

**Accuracy.** Six polar nodes are enough for a smooth emitter: a thin film, a
metal, an oxide on silicon (on such cases the error was below $10^{-5}$). The
integrand is smooth in $\cos\theta$ there, and the Gauss rule converges fast.
Fringes in angle (a slab many wavelengths thick) and
{term}`Rayleigh anomalies <Rayleigh anomaly>` inside the hemisphere converge
only algebraically: a kink error falls like the number of nodes to the power
$-1.5$, so the second rule below is a check you should not skip. Use as many
azimuths as the cell's angular structure needs; 16 is a start for a
two-dimensional pattern.

To check convergence, solve a second rule with more nodes and compare. Gauss
rules do not nest, so this is a new sweep, but `cache=` keeps the first:

```python
finer = dif.hemisphere(12, 32, rotation=4, mirror=True)
```

**Near grazing.** There is no cut-off: $\theta = \pi/2$ is never a node, and
the outermost node of a six-node rule is at 86°. Directions beyond 80° carry
about 1 % of $e_h$, so the outermost node needs ordinary accuracy only. For
wide angles refine the lateral mesh, as in {doc}`diffraction-efficiency`.

**Cost.** One sweep solves $n_\theta$ times the folded azimuths times the
number of frequencies Bloch problems, each as in
{doc}`diffraction-efficiency`: the rule above is $6 \times 3 = 18$ incidences.

## The Planck-weighted total

The total emissivity of the surface at temperature $T$ weights $e_h$ by the
blackbody spectrum,

$$\varepsilon_\mathrm{tot}(T) = \frac{\int M_\nu(T)\, e_h(\nu)\, d\nu}
{\int M_\nu(T)\, d\nu},\qquad
M_\nu(T) = \frac{2\pi h \nu^3}{c^2}\,\frac{1}{e^{h\nu/k_B T} - 1},$$

and over your own band it is a weighted sum with `jax.numpy.trapezoid`, which
differentiates:

```python
import jax.numpy as jnp

h, k_B, c = 6.62607015e-34, 1.380649e-23, 299_792_458.0  # SI


def planck_total(frequency, e_h, temperature):
    """Planck-weighted emissivity over the band `frequency` (Hz) at `temperature` (K)."""
    x = h * frequency / (k_B * temperature)
    m = 2 * jnp.pi * h * frequency**3 / c**2 / jnp.expm1(x)
    return jnp.trapezoid(m * e_h, frequency) / jnp.trapezoid(m, frequency)


total = planck_total(band, e_h, 600.0)
```

The band must cover the Planck peak, near $\nu_\mathrm{peak} \approx 5.88
\times 10^{10}\,T$ Hz/K (about $3.5 \times 10^{13}$ Hz at 600 K), and the tails
beyond it: otherwise the denominator is the band's, not the whole spectrum's,
and the result is a band-limited emissivity.

## Loads

A lumped or wave port is a separate body. `emissivity` reports the absorptance
of the cell, which excludes the power delivered to a port (`coupled`), so a
fed cell emits what it absorbs in its own materials. A matched load at the
emitter's temperature radiates too: add its `coupled` power to the emitted power
when the load is part of the emitter.

## Differentiate

The emissivity is $|S|^2$ of every incidence, so `jax.grad` of $e_h$ or
`planck_total` with respect to the materials, a design density or the interior
nodes is the weighted sum of one adjoint per incidence and frequency. The rule
and the polarization are constants, and so are the angles.

```python
import jax


def e_h_of(tan):
    n = len(cell.mesh.elements("film"))
    m = cell.with_material("film", gfem.MaterialField(tan_delta=jnp.full(n, tan)))
    sw = dif.angle_sweep(
        m,
        band,
        surface="top",
        theta=rule.theta,
        phi=rule.phi,
        options=gfem.SolverOptions(retain_memory=0),
    )
    return dif.hemispherical_emissivity(sw, rule, model=m)[0]


grad = jax.grad(e_h_of)(0.1)
```

## Check your results

- **Lossless cell.** A cell without loss emits nothing: its $e$ is the power
  defect of the mesh, and shrinks as you refine.
- **Opaque backing.** With no transmitted order, $e = 1 - R$ per direction.
- **Weights.** `rule.weight.sum()` is 1, so a constant emissivity averages to
  itself.
- **A second rule** (above), the check that matters for fringes and anomalies.
- **Unpolarized.** $e_h$ for `"unpolarized"` is the mean of the TE and TM values.

## Limits

- **Equilibrium and reciprocity.** One temperature; the law is not valid for
  an emitter driven out of equilibrium.
- **Polarization.** TE, TM, RCP, LCP and unpolarized; no Jones pair.
- **Symmetry.** Yours to assert: the rule cannot check the cell's point group.
- **Anomalies.** No anomaly-aware rule: the anomaly angles move with frequency
  inside a band, so the algebraic convergence above is to be expected.
- **Gradients.** None with respect to the angles.
- **Planck total.** Computed over the band you solved, not extrapolated.

## Next

{doc}`diffraction-efficiency`, {doc}`floquet-ports`,
{doc}`periodic-cell`, {py:obj}`gfem.conventions`,
{doc}`s-parameter-gradient`, {doc}`../capabilities`.
