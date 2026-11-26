# Diffraction efficiencies

Light a periodic cell with a plane wave through one of its faces and read where
the power goes: the {term}`diffraction efficiency` of each order and
polarization, the reflected, transmitted and absorbed fractions, and the
{term}`Jones matrix` between two orders, with its circular and Mueller
forms. Then repeat for a list of incidence directions
({term}`angle sweep`). The helpers live in `gfem.diffraction` and read the S
of a model with {py:obj}`gfem.FloquetPort`s; the ports themselves are
explained in {doc}`floquet-ports`.

## Declare every order

A cell diffracts into every {term}`diffraction order` that propagates behind
its faces, and a power balance is only complete if each one has a port in both
polarizations. {py:func}`gfem.diffraction.floquet_ports` declares them for one
face: a TE and a TM {term}`Floquet port` of every propagating order, named
`"top (0, 0) TE"` and so on. The cell below is a ridge grating on a substrate,
meshed with the volumes `"substrate"`, `"ridge"` and `"air"`, the open faces
`"top"` and `"bottom"` and periodic side faces:

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

- The orders are those that propagate at some frequency of `frequency`. For a
  band, pass the whole grid, or at least its highest frequency, so that an
  order that opens inside the band is declared.
- At an oblique, fixed-angle incidence pass `wave_vector=` and
  `wave_vector_frequency=` as for {py:obj}`gfem.Driven`; a list of wave
  vectors declares the union of their orders, which is what an
  angle sweep (below) needs.
- `offset=` moves every port's reference plane, as for
  {py:obj}`gfem.FloquetPort`.
- Excite only the columns you read: one incident wave is one column of S. The
  four columns above are the two polarizations lit from either face.

## Efficiency of each order

`efficiencies` reports $|S|^2$ of the incident column, or of a combination of
the TE and TM columns, for every declared order. Light enters through a face
(in its specular order by default) in a polarization: `"TE"`, `"TM"`,
`"RCP"`, `"LCP"`, `"unpolarized"` or a Jones pair $(a_\mathrm{TM},
a_\mathrm{TE})$ in the incident order's references:

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
- `"unpolarized"` is the mean of the TE and TM columns' efficiencies and needs
  both excited. A Jones pair is normalized for you.
- `basis="circular"` gives an `"RCP"` and an `"LCP"` row for each order, with
  the handedness taken about each beam's own direction of travel
  ({py:obj}`gfem.conventions`). A mirror swaps it.
- `order=` names another incident order, for a cell lit by a diffracted wave.

## Power balance

`power_balance` adds the rows up: the reflectance of the incident face, the
transmittance through the other, the power `coupled` into lumped and wave
ports, and the {term}`absorptance`
$A = 1 - R - T - \text{coupled}$, exactly:

```python
bal = dif.power_balance(res, "top", polarization="unpolarized", model=cell)
bal.reflectance, bal.transmittance, bal.absorptance  # each (n_freq,)
```

The absorptance of a lossless cell is the power defect of the mesh, so it is
also a check of the solve. In a lossy cell it is the power the cell
dissipates.

The balance is refused when it would be wrong. A propagating order without a
port absorbs its power without reporting it, so $1 - R - T$ would count it as
absorbed. `power_balance` raises `ValueError`, naming the missing
orders and polarizations, when any propagating order of either face lacks a TE
or TM port at some frequency, and the fix is
`dif.floquet_ports`. Grazing orders are not required. `efficiencies` and
`jones` need no completeness: they read the rows you declared.

## Jones, circular and Mueller matrices

{py:func}`gfem.diffraction.jones` returns the $2\times2$ block of S between
the order a beam leaves in and the order it enters in. Reflection names the
incident face twice, transmission the other face. Lit from the bottom, the
specular transmission leaves through the top:

```python
J = dif.jones(res, "top", "bottom", model=cell)  # outgoing, then incident
J.values  # (n_freq, 2, 2) in (TM, TE): rows outgoing, columns incident
R = dif.jones(res, "bottom", "bottom", model=cell)  # reflection of the same light
J.circular  # (n_freq, 2, 2) in (RCP, LCP)
J.mueller  # (n_freq, 4, 4), real
J.mueller[:, 0, 0]  # the efficiency for unpolarized light
```

- Both polarizations of both orders need a port, and both incident columns must
  be excited.
- The entries are power-wave ratios like S, so $|J_{ij}|^2$ is an efficiency.
  There is no sign flip on reflection. At normal incidence TM is
  $\hat{\mathbf{e}}_1$ and TE is $\hat{\mathbf{e}}_2$ on both faces, so $J$ is the
  Jones matrix in the cell's own axes, from either side.
- `J.circular` is $U_{h_\mathrm{out}}^\dagger J\, U_{h_\mathrm{in}}$: rows and
  columns are (RCP, LCP) about each beam's own direction. A plate that turns
  45° linear light into right-handed light from the top does so into
  left-handed light from the bottom.
- `J.mueller` maps an incident Stokes vector to the outgoing one, each in its
  own beam's frame. Its signs, and why a plate shows its linear block at the
  opposite angle from the top face, are on {py:obj}`gfem.conventions`.
- An order that is evanescent at some frequency reads 0 there.

To light the cell with light of your own polarization, combine the columns
of `J.values` with your own vector: it differentiates, and the incident
polarization of `efficiencies` does not.

## Sweep the incidence angle

{py:func}`gfem.diffraction.angle_sweep` solves a list of incidences
$(\theta, \varphi)$ on one face over a band, keeping each angle fixed across
the band. Declare the orders for every incidence first, then pass the sweep to
the same helpers, which stack the incidences on a leading axis:

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
$\hat{\mathbf{e}}_1$ toward $\hat{\mathbf{e}}_2$, in radians, in the medium behind
the face. They broadcast together to one flat list of incidences. For a grid,
ravel it (`np.meshgrid(...)[k].ravel()`); there is no outer product. Normal
incidence solves with plain periodicity, and incidences with the same wave
vector are solved once. Everything is checked before the first solve:
the angles, the excited names, and that every excited port propagates at every
incidence and frequency. A propagating order with no port warns once for the
whole grid with {py:obj}`gfem.UnrequestedOrderWarning`.

**Conical incidence.** When the plane of incidence is not perpendicular to
the lines of a grating ($\varphi \ne 0, \pi$ when $\hat{\mathbf{e}}_1$ crosses
the lines), the incidence is {term}`conical <Conical incidence>`:

- $\varphi$ is measured from $\hat{\mathbf{e}}_1$, the translation of the first
  periodic pair. For a grating whose lines run along $\hat{\mathbf{e}}_2$,
  $\varphi = 0$ is the planar mount; declaring the pairs in the other order
  makes $\varphi = 90°$ planar.
- TE and TM are relative to each order's own plane of incidence. In a conical
  mount the orders leave on a cone outside the incident plane, so TE and TM
  couple: the cross-polarized rows are not zero. In a planar mount they read
  zero.
- At $\theta = 0$ the references are $(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2)$
  whatever $\varphi$, so a $\theta$ scan at $\varphi \ne 0$ sees the linear TE
  and TM entries jump at $\theta = 0$ by the rotation $\varphi$. Efficiencies
  summed over the output polarization, and the unpolarized reflectance,
  transmittance and absorptance, are continuous.

**Cost.** Each distinct incidence and frequency is one factorization of a Bloch
problem, and nothing is shared between incidences or between the frequencies
of one. Normal incidence is the cheapest point. On a cell of 130,000 unknowns
at order 2 on two threads, a normal-incidence frequency took 9 to 13 s and an
oblique one 30 to 60 s, about 2 GB at peak for each (observations on a loaded
machine, not guarantees).

- Pass `SolverOptions(retain_memory=0)` when you do not differentiate. The
  default budget keeps the factorizations of earlier incidences that no later
  one reuses: five incidences peaked at 4.7 GB with it and 2.1 GB with
  `retain_memory=0`.
- `frequency_workers` spreads the frequencies of one incidence. To spread
  incidences, run separate processes over chunks of the list that share one
  `cache=` directory; entries are published atomically, and one call over the
  whole list then reads everything back. With `cache=`, extending the list
  solves only the new incidences, and an interrupted sweep resumes.
- Accuracy falls with angle on one mesh. Refine the lateral mesh for wide
  angles, keep the face away from the structure near grazing, and check the
  power balance of each incidence on a lossless cell.

**The angle is not differentiable.** `theta` and `phi` must be fixed numbers
(`NonDifferentiableError` otherwise): the Bloch space, the order table and the
reference impedances are built per wave vector, and $|S|$ varies as
$\sqrt{|\gamma|}$ at a {term}`Rayleigh anomaly`. A gradient with respect to the
cell's materials or interior nodes flows through every incidence. For the
sensitivity to the angle, take a difference of two incidences.

## Check your results

- **Power balance.** Lossless cell: $|A|$ is small against $R$ and $T$ at
  every incidence and frequency, and shrinks as the mesh is refined or the
  order raised. A large value after refining usually means an order without a
  port in the solve (the warning names it).
- **Column sums.** The efficiencies of every row of one incident column add up
  to $R + T$, in the linear and the circular basis alike.
- **Reciprocity.** The Jones matrix of a reciprocal cell lit from the bottom
  is the transpose of the one lit from the top at normal incidence:
  `J.values[i]` against `dif.jones(res, "bottom", "top", model=cell).values[i].T`.
  The general rule with order labels is on {doc}`floquet-ports`.
- **A flat interface** gives the Fresnel values in the same references; a
  mirror gives $-P$ in the circular basis ({py:obj}`gfem.conventions`).

## Limits

- **Ports.** A balance needs a TE and a TM port for every propagating
  order of every Floquet face at every frequency of the sweep, and the half-space
  conditions of {doc}`floquet-ports` apply.
- **Circular ports.** No port is circular: circular entries are a change of
  basis of S.
- **Angles.** One face per sweep, the incident wave in its specular
  order, and fixed angles with no gradient.
- **Absorptance.** It counts the power the cell dissipates; power delivered to
  a lumped or wave port is reported separately as `coupled`.
- **Emission.** Thermal emission is {doc}`emissivity`.

## Next

{doc}`emissivity`, {doc}`floquet-ports`, {doc}`periodic-cell`,
{doc}`s-parameter-gradient`, {py:obj}`gfem.conventions`,
{doc}`../tutorials/metagrating`, {doc}`../capabilities`.
