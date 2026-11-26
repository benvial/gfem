# Thermal emissivity

A patterned emitter at a uniform temperature has an {term}`emissivity` that
depends on direction, polarization and frequency. By Kirchhoff's law it is the
{term}`absorptance` of the wave arriving from the same direction, so a driven
solve of the cell gives it, with no thermal solve. The cell, its Floquet ports
and the angle sweep are those of {doc}`diffraction-efficiency`.

## Emissivity per direction

An incidence $(\theta, \varphi)$ gives the emissivity toward
$(\theta, \varphi + \pi)$, the time reverse of the wave that arrives.
{py:func}`gfem.diffraction.emissivity` labels its rows with the emission
direction: use the label, not the sweep's `phi`. For a lossy film on a
metal-backed substrate, declare the orders of every incidence, solve the sweep
and read the emissivity:

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
  default, as thermal emission is). RCP is right-handed about the emission
  direction. A Jones pair is refused: use `dif.power_balance`.
- `emissivity` takes an `AngleSweepResult`, the only input that knows the
  direction; for one direction sweep one angle.
- On a lossless substrate the transmitted power counts as not absorbed.
- If the medium behind the surface has index above 1, $\theta$ is measured in it
  and $e$ is relative to the blackbody in it.
- Every propagating order needs a TE and a TM port at every incidence, as for
  `power_balance`. Small negative values on a lossless region show the mesh
  defect and are not clipped.

## Hemispherical emissivity

The {term}`hemispherical emissivity` is the $\cos\theta$-weighted mean of $e$
over the directions above the surface,

$$e_h = \frac{1}{\pi}\int_0^{2\pi}\!\!\int_0^{\pi/2}
e(\theta, \varphi)\cos\theta\sin\theta\, d\theta\, d\varphi
\approx \sum_k w_k\, e(\theta_k, \varphi_k).$$

{py:func}`gfem.diffraction.hemisphere` builds the nodes and weights: Gauss-Jacobi
nodes in $\cos\theta$ times a uniform trapezoid in azimuth. Solve the sweep at
the rule's nodes, in order:

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

**Symmetry folding.** `rotation=r` asserts invariance under a rotation by
$2\pi/r$ about the normal, and `mirror=True` a mirror plane that contains
$\hat e_1$ and the normal. A cell mirrored only about the plane containing
$\hat e_2$ is folded wrongly and silently: give it `mirror=False`, or
`rotation=2, mirror=True` if it has both planes. Four-fold rotation with mirrors
needs 3 of 16 azimuths; `n_phi` must be a multiple of `rotation`. An emitter
independent of the azimuth takes `hemisphere(n_theta, 1)`. A rule folded by a
symmetry the cell lacks is wrong by the asymmetric part.

**Circular polarization.** A mirror turns right-handed light into left-handed
light, so a rule folded with `mirror=True` refuses `"RCP"` and `"LCP"`.

**Accuracy.** Six polar nodes suffice for a smooth emitter (error below
$10^{-5}$ for a thin film, a metal, an oxide on silicon). Fringes in angle and
{term}`Rayleigh anomalies <Rayleigh anomaly>` inside the hemisphere converge only
algebraically, so check with a second rule with more nodes. Gauss rules do not
nest, so it is a new sweep, but `cache=` keeps the first:

```python
finer = dif.hemisphere(12, 32, rotation=4, mirror=True)
```

Directions beyond 80° carry about 1 % of $e_h$; $\theta = \pi/2$ is never a
node. One sweep solves $n_\theta$ times the folded azimuths times the
frequencies: the rule above is $6 \times 3 = 18$ incidences.

## The Planck-weighted total

The total emissivity at temperature $T$ weights $e_h$ by the blackbody spectrum,

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
\times 10^{10}\,T$ Hz/K (about $3.5 \times 10^{13}$ Hz at 600 K), and its tails;
otherwise the result is band-limited.

## Loads

`emissivity` reports the absorptance of the cell, which excludes the power
delivered to a port (`coupled`). A matched load at the emitter's temperature
radiates too: add its `coupled` power when the load is part of the emitter.

## Differentiate

The emissivity is $|S|^2$ of every incidence, so `jax.grad` of $e_h$ or
`planck_total` with respect to the materials, a design density or the interior
nodes is the weighted sum of one adjoint per incidence and frequency. The rule,
polarization and angles are constants.

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

- **Lossless cell.** $e$ is the mesh's power defect and shrinks on refinement.
- **Opaque backing.** With no transmitted order, $e = 1 - R$ per direction.
- **Weights.** `rule.weight.sum()` is 1.
- **A second rule** (above), the check that matters for fringes and anomalies.
- **Unpolarized.** $e_h$ is the mean of the TE and TM values.

## Limits

- **Equilibrium and reciprocity.** One temperature: Kirchhoff's law fails for an
  emitter driven out of equilibrium. Every gfem cell is reciprocal and passive.
- **Polarization.** TE, TM, RCP, LCP and unpolarized; no Jones pair.
- **Symmetry.** Yours to assert; the rule cannot check the cell's point group.
- **Anomalies.** No anomaly-aware rule: expect algebraic convergence.
- **Gradients.** None with respect to the angles.
- **Planck total.** Over the band you solved, not extrapolated.

## Next

{doc}`diffraction-efficiency`, {doc}`floquet-ports`, {doc}`periodic-cell`,
{py:obj}`gfem.conventions`, {doc}`s-parameter-gradient`, {doc}`../capabilities`.
