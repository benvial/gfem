# Diffraction efficiencies

Light a periodic cell with a plane wave and read where the power goes: the
{term}`diffraction efficiency` of each order, the reflected, transmitted and
absorbed fractions, and the {term}`Jones matrix` between two orders; then repeat
for a list of incidences ({term}`angle sweep`). The helpers live in
`gfem.diffraction` and read the S of a model with
{py:obj}`gfem.FloquetPort` ports ({doc}`floquet-ports`).

## Declare every order

Every propagating {term}`diffraction order` needs a port in both polarizations.
{py:func}`gfem.diffraction.floquet_ports` declares a
TE and a TM {term}`Floquet port` of each order of one face of a ridge grating on
a substrate:

```python
import numpy as np

import gfem
from gfem import diffraction as dif

c = 299_792_458.0  # m/s
mesh = gfem.Mesh.from_gmsh("grating_cell.msh", length_unit="um")
cell = gfem.Model(
    mesh,
    materials={
        "substrate": gfem.Material(eps_r=2.25),
        "ridge": gfem.Material(eps_r=4.0, tan_delta=0.01),
        "air": gfem.Material(),
    },
    boundaries=[
        gfem.Periodic(("x_min", "x_max")),
        gfem.Periodic(("y_min", "y_max")),
    ],
)
f = c / 1.0e-6  # Hz, a wavelength of 1 um
ports = [
    *dif.floquet_ports(cell, "top", f),
    *dif.floquet_ports(cell, "bottom", f),
]
cell = cell.replace(ports=ports)
lit = [f"{face} (0, 0) {pol}" for face in ("top", "bottom") for pol in ("TE", "TM")]
options = gfem.SolverOptions(order=2, linear_solver="direct")
res = gfem.solve(cell, gfem.Driven(f, excite=lit), options=options)
```

- Pass the whole frequency grid, or at least its highest frequency: the orders
  are those that propagate at some frequency.
- At oblique incidence pass `wave_vector=` and `wave_vector_frequency=` as for
  {py:obj}`gfem.Driven`; a list declares the union of the orders.
- `offset=` moves every port's reference plane; excite only the columns you
  read.

## Efficiency of each order

`efficiencies` reports $|S|^2$ of the incident column, or of a combination of
the TE and TM columns, for every declared order. Light enters through a face (in
its specular order by default) in a polarization: `"TE"`, `"TM"`, `"RCP"`,
`"LCP"`, `"unpolarized"` or a Jones pair $(a_\mathrm{TM}, a_\mathrm{TE})$:

```python
eff = dif.efficiencies(res, "top", polarization="TE", model=cell)
eff.rows  # (face, order, polarization) of each last-axis entry
eff.values  # (n_freq, n_rows)
eff["bottom", (0, 0), "TE"]  # one row, shape (n_freq,)
eff["bottom", (0, 0)]  # the order's total over TE and TM

circ = dif.efficiencies(res, "top", polarization="RCP", model=cell, basis="circular")
circ["bottom", (0, 0), "LCP"]  # transmitted, opposite handedness
```

- A row of an order that does not propagate reads 0.
- `"unpolarized"` is the mean of the TE and TM columns and needs both excited.
- `basis="circular"` gives an `"RCP"` and an `"LCP"` row for each order, about
  each beam's own direction of travel ({py:obj}`gfem.conventions`).
- `order=` names another incident order.

## Power balance

`power_balance` adds the rows up: the reflectance of the incident face, the
transmittance through the other, the power `coupled` into lumped and wave ports,
and the {term}`absorptance` $A = 1 - R - T - \text{coupled}$, exactly:

```python
bal = dif.power_balance(res, "top", polarization="unpolarized", model=cell)
bal.reflectance, bal.transmittance, bal.absorptance  # each (n_freq,)
```

The absorptance of a lossless cell is the mesh's power defect; in a lossy cell it
is the power dissipated. A propagating order without a port would count as
absorbed, so `power_balance` raises `ValueError` naming the missing ports;
`efficiencies` and `jones` do not.

## Jones, circular and Mueller matrices

{py:func}`gfem.diffraction.jones` returns the $2\times2$ block of S between the
order a beam leaves in and the order it enters in. Reflection names the incident
face twice, transmission the other face. Lit from the bottom, the specular
transmission leaves through the top:

```python
J = dif.jones(res, "top", "bottom", model=cell)  # outgoing, then incident
J.values  # (n_freq, 2, 2) in (TM, TE): rows outgoing, columns incident
R = dif.jones(res, "bottom", "bottom", model=cell)  # reflection of the same light
J.circular  # (n_freq, 2, 2) in (RCP, LCP)
J.mueller  # (n_freq, 4, 4), real
J.mueller[:, 0, 0]  # the efficiency for unpolarized light
```

- Both polarizations of both orders need a port, and both columns excited.
- $|J_{ij}|^2$ is an efficiency. At normal incidence TM is $\hat{\mathbf{e}}_1$
  and TE is $\hat{\mathbf{e}}_2$ on both faces, so $J$ is the Jones matrix in the
  cell's axes from either side.
- `J.circular` has rows and columns (RCP, LCP) about each beam's own direction:
  a plate that turns 45° linear light into right-handed light from the top does
  so into left-handed light from the bottom.
- `J.mueller` maps an incident Stokes vector to the outgoing one
  ({py:obj}`gfem.conventions`).

To use your own polarization, combine the columns of `J.values`: it
differentiates, and the incident polarization of `efficiencies` does not.

## Sweep the incidence angle

{py:func}`gfem.diffraction.angle_sweep` solves a list of incidences
$(\theta, \varphi)$ on one face over a band, each angle fixed across the band.
Declare the orders for every incidence first, then pass the sweep to the same
helpers, which stack the incidences on a leading axis:

```python
band = np.linspace(0.95, 1.05, 5) * f
theta, phi = np.radians([0.0, 10.0, 20.0, 30.0, 40.0]), 0.0
ks = [cell.bloch_wave_vector("top", f, theta=t, phi=phi) for t in theta]
ports = [
    *dif.floquet_ports(cell, "top", band, wave_vector=ks, wave_vector_frequency=f),
    *dif.floquet_ports(cell, "bottom", band, wave_vector=ks, wave_vector_frequency=f),
]
cell = cell.replace(ports=ports)
sweep = dif.angle_sweep(
    cell,
    band,
    surface="top",
    theta=theta,
    phi=phi,
    options=gfem.SolverOptions(retain_memory=0),
    cache="angles",
)
bal = dif.power_balance(sweep, "top", polarization="unpolarized", model=cell)
bal.absorptance  # (n_incidence, n_freq)
sweep.theta, sweep.frequency  # the labels
```

`theta` is the angle from the face's inward normal and `phi` the azimuth from
$\hat{\mathbf{e}}_1$, in radians; they broadcast to one flat list, so ravel a
grid (`np.meshgrid(...)[k].ravel()`). Equal wave vectors are solved once. A
propagating order with no port warns once with
{py:obj}`gfem.UnrequestedOrderWarning`.

**Conical incidence.** When the plane of incidence is not perpendicular to the
lines of a grating, the incidence is {term}`conical <Conical incidence>`:

- For lines along $\hat{\mathbf{e}}_2$, $\varphi = 0$ is the planar mount.
- TE and TM couple, so the cross-polarized rows are not zero.
- At $\theta = 0$ the linear TE and TM entries jump by the rotation $\varphi$;
  summed efficiencies and the unpolarized $R$, $T$, $A$ are continuous.

**Cost.** Each distinct incidence and frequency is one factorization.

- Pass `SolverOptions(retain_memory=0)` when you do not differentiate: the
  default keeps factorizations no later incidence reuses.
- To spread incidences, run processes over chunks that share one `cache=`
  directory; extending the list then solves only the new incidences.
- Accuracy falls with angle: refine the lateral mesh and check each incidence's
  power balance.

**The angle is not differentiable.** `theta` and `phi` must be fixed numbers
(`NonDifferentiableError`), since $|S|$ varies as $\sqrt{|\gamma|}$ at a
{term}`Rayleigh anomaly`. Gradients with respect to materials flow through every
incidence; difference two incidences for the angle's sensitivity.

## Check your results

- **Power balance.** For a lossless cell $|A|$ is small against $R$ and $T$ and
  shrinks with refinement; a large value usually means an order without a port.
- **Column sums.** The efficiencies of one incident column add up to $R + T$.
- **Reciprocity.** At normal incidence the Jones matrix lit from the bottom is
  the transpose of the one lit from the top: `J.values[i]` against
  `dif.jones(res, "bottom", "top", model=cell).values[i].T`.
- **A flat interface** gives the Fresnel values.

## Limits

- **Ports.** A balance needs a TE and a TM port for every propagating order of
  every Floquet face ({doc}`floquet-ports`); no port is circular.
- **Angles.** One face per sweep, the incident wave in its specular order, and
  fixed angles with no gradient.
- **Emission.** Thermal emission is {doc}`emissivity`.

## Next

{doc}`emissivity`, {doc}`floquet-ports`, {doc}`periodic-cell`,
{doc}`s-parameter-gradient`, {py:obj}`gfem.conventions`,
{doc}`../tutorials/metagrating`, {doc}`../capabilities`.
