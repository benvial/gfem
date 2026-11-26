# Point dipoles and Purcell factor

A {term}`point dipole` ({py:obj}`gfem.ElectricDipole`,
{py:obj}`gfem.MagneticDipole`) is a current at one point: a fluorescent
molecule, a quantum dot, a small antenna. Give it to a driven problem as a
{term}`source`, and the solve returns its {term}`emitted power`, the power it
sends into each port, and its field. From them come the {term}`Purcell
factor`, the {term}`beta factor` of a guided mode and the {term}`radiative
efficiency`.

A dipole is one more column of the driven solve, beside the excited ports,
on the same factorization: several dipoles (positions, orientations) cost one
back-substitution each. When only the port waves matter,
`Driven(source_fields=False)` drops the dipoles' columns altogether (Many
positions, below). A model with dipoles needs no port: close it with a
{term}`perfectly matched layer`, absorbing faces or walls.

## Place a dipole

The layer's TEM line (0.2 µm × 0.2 µm, metal plates on its $y$ walls, a
{term}`wave port` on its open end, a layer at the far end), with an electric
dipole along $y$ in its air, solved at a 1 µm wavelength:

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/pml_line.msh"
urllib.request.urlretrieve(url, "pml_line.msh")
mesh = gfem.Mesh.from_gmsh("pml_line.msh", length_unit="um")

air = gfem.Material()
model = gfem.Model(
    mesh,
    materials={"air": air, "pml": air},
    boundaries=[gfem.PEC("plates")],
    ports=[gfem.WavePort("p1", "port")],
    pml=gfem.PML("pml"),
)
f0 = 299_792_458.0 / 1e-6  # Hz, a wavelength of 1 um
r0 = np.array([0.07, 0.13, 0.74]) * 1e-6  # m
dipole = gfem.ElectricDipole("d", position=r0, moment=(0, 1e-30, 0))  # C m
problem = gfem.Driven(f0, excite=(), sources=dipole, keep_fields=True)
result = gfem.solve(model, problem, options=gfem.SolverOptions(order=3))

power = result.emitted_power["d"][0]
purcell = gfem.purcell_factor(result, "d", model=model)[0]
guided = abs(result.port_wave("p1", "d")[0]) ** 2
print(f"P = {power:.4e} W, Purcell factor {purcell:.3f}")
print(f"into the port: {guided:.4e} W, beta = {guided / power:.4f}")
```

```text
P = 8.3534e-15 W, Purcell factor 5.968
into the port: 4.1767e-15 W, beta = 0.5000
```

- `excite=()` feeds no port: the solve has the dipole's column only, and
  every port is terminated in its reference impedance. With ports excited
  too, S keeps their columns and the dipole's come after them.
- The moment is a peak phasor in C·m, the same at every frequency. An
  antenna's current moment $Il$ (A·m) is $p = Il / (i\omega)$; a magnetic
  dipole's moment is in A·m² (`gfem.conventions`, Point dipoles).
- `result.emitted_power[name]` is $P = -\tfrac{\omega}{2}
  \operatorname{Im}\left(p^* \cdot E(r_0)\right)$ in W, one value per
  frequency: positive, and finite although the exact field at $r_0$ is not
  (gfem pairs the dipole's load with the discrete solution). Its real
  counterpart, the self-reaction, has no limit and is never reported.
- `result.port_wave(port, name)` is the {term}`port wave`, outgoing at the
  port in √W, as a column of S: $|b|^2$ is the power the dipole sends into the port. The
  {term}`beta factor` of that port's mode is $\beta = |b|^2 / P$; on a
  {term}`shared port surface` each mode has its own port and its own wave.

## The Purcell factor and the local density of states

`gfem.purcell_factor(result, name, model=model)` is the {term}`Purcell factor`
$F = P / P_0$, the
emitted power over the {term}`bulk power`: the power of the same dipole in
the unbounded material at its position,

$$
P_0^{(p)} = \frac{\mu_0 \mu_r n\, \omega^4 |p|^2}{12 \pi c}, \qquad
P_0^{(m)} = \frac{\mu_0 \varepsilon_r n\, \omega^4 |m|^2}{12 \pi c^3},
\qquad n = \sqrt{\varepsilon_r \mu_r},
$$

read from the model at each frequency (a permittivity law's value there). In
a non-magnetic medium the partial local density of states along the moment
is $\rho_p = \rho_0 F$, with $\rho_0 = \omega^2 n / (\pi^2 c^3)$; gfem has no
separate function for it.

## Where the power goes

The emitted power splits exactly, to round-off, into what the ports carry
away, what the layer (or absorbing faces) takes, and what lossy materials
absorb. {py:func}`gfem.power_flow` gives the last two as the power entering
the layer regions and the lossy regions:

```python
field = result.e_field(f0, "d")
radiated = -gfem.power_flow(field, "pml", model=model)
absorbed = 0.0  # -gfem.power_flow(field, ["lossy", ...], model=model)
print(f"port {guided:.4e} + layer {radiated:.4e} + absorbed {absorbed:.1e} W")
print(f"sum / P = {(guided + radiated + absorbed) / power:.8f}")
print(f"radiative efficiency {1 - absorbed / power:.3f}")
```

```text
port 4.1767e-15 + layer 4.1767e-15 + absorbed 0.0e+00 W
sum / P = 1.00000000
radiative efficiency 1.000
```

- The {term}`radiative efficiency` $\eta = 1 - P_\text{abs} / P$ is the share
  of the emitted power that leaves rather than heats the materials: near a
  metal it falls (quenching) while the Purcell factor grows.
- A side of `power_flow` must keep its surface one element away from the
  dipole: the element layer just outside the side may not hold it, or the
  dipole's own load would count as power crossing the surface (gfem refuses
  such a side by name). A side that holds the dipole counts its emission.

## The best orientation

The emitted power is a quadratic form of a real moment: with the fields
$E^{(j)}$ of three unit moments along $x$, $y$, $z$ at the same point and
$M_{ij} = \hat e_i \cdot E^{(j)}(r_0)$, $P(p) = -\tfrac{\omega}{2}\,
p^\mathsf{T} \operatorname{Im}(M)\, p$. The orientation that emits most is
the top eigenvector of $-\operatorname{Im}(M)$ (symmetrized), from three
columns on one factorization:

```python
size = 1e-30  # C m
axes = [
    gfem.ElectricDipole(f"d{a}", position=r0, moment=size * np.eye(3)[k])
    for k, a in enumerate("xyz")
]
three = gfem.solve(
    model,
    gfem.Driven(f0, excite=(), sources=axes, keep_fields=True),
    options=gfem.SolverOptions(order=3),
)
M = np.stack([three.e_field(f0, d.name).at(r0) for d in axes], axis=1) / size
omega = 2 * np.pi * f0
form = -omega / 2 * np.imag(M + M.T) / 2  # W per (C m)^2
values, vectors = np.linalg.eigh(form)
best = vectors[:, -1] * np.sign(vectors[np.argmax(abs(vectors[:, -1])), -1])
print(f"best orientation {np.round(best, 3) + 0.0}, P = {values[-1] * size**2:.4e} W")
```

```text
best orientation [0. 1. 0.], P = 8.3534e-15 W
```

Only the imaginary part of $M$ enters, and it converges under refinement;
its real part does not.

## Many positions: port waves by reciprocity

A coupling map asks how much of an emitter's light a guide collects, position
by position. With `source_fields=False` gfem solves no dipole. It solves each
port's own field once and reads every dipole's port wave from it by
reciprocity:

$$
b_i = -\frac{i\omega}{4}\,\mathbf{p}\cdot\mathbf{E}_i(\mathbf{r}_0),
$$

where $\mathbf{E}_i$ is the field with port $i$ fed 1 W and every other port
terminated. Ten positions along the line, reusing `model`, `f0` and `r0`:

```python
z = np.linspace(0.3, 1.2, 10) * 1e-6  # m, along the line
row = [
    gfem.ElectricDipole(f"z{i}", position=(r0[0], r0[1], zi), moment=(0, 1e-30, 0))
    for i, zi in enumerate(z)
]
coupling = gfem.solve(
    model,
    gfem.Driven(f0, excite=(), sources=row, source_fields=False),
    options=gfem.SolverOptions(order=3),
)
b = coupling.port_waves[0, 0]  # sqrt(W): port p1, one value per position
k0 = 2 * np.pi * f0 / 299_792_458.0
slope = np.polyfit(z, np.unwrap(np.angle(b)), 1)[0]
print(f"|b|^2 from {min(abs(b) ** 2):.4e} to {max(abs(b) ** 2):.4e} W")
print(f"phase slope {slope / k0:.4f} k0")
```

```text
|b|^2 from 4.1770e-15 to 4.1770e-15 W
phase slope -1.0000 k0
```

- **Cost**: one solve per port with a defined wave, plus one point overlap per
  dipole, whatever the number of positions. gfem pairs each dipole's point
  load with the port's discrete solution, so the values are the port waves of
  the dipoles' solved source fields, up to the linear solve's accuracy (about
  $10^{-9}$ relative on this line), with the same mesh error.
- **The physics of this map**: on this line $\mathbf{E}_1$ is the TEM field,
  uniform across the section with phase $e^{-ik_0 z}$. So $|b|^2$ is the same
  at every position, the power the first section's column sent into the port,
  and the phase falls by $k_0$ per metre further from the port.
- **What is not there**: the dipoles' own fields are not solved.
  `coupling.emitted_power` is `{}`, and `gfem.purcell_factor`,
  `coupling.e_field(f0, "z0")` and `gfem.far_field` raise `ValueError`
  naming the flag. The beta factor $|b|^2 / P$ needs $P$, so solve the
  positions where it matters with `source_fields=True` (the default).
- **Exciting ports**: `excite=None` costs the same solves here and adds S,
  because the port's own field is solved anyway.

## Magnetic dipoles

A {py:obj}`gfem.MagneticDipole` is the current $\nabla \times (m\,
\delta(r - r_0))$: the same outputs, and $P_0$ with $\varepsilon_r$ in place
of $\mu_r$. Its emitted power converges one order more slowly than an
electric dipole's (its load reads the curl of the basis).

## Accuracy

The emitted power is accurate to 1-3 % at about 50 000 unknowns: order 3
with elements of $\lambda/5$ in the dipole's medium (or order 2 at
$\lambda/6$) and a layer about half a wavelength thick. The error is the
discretization of the waves over the whole domain, not of the point source:
refining the mesh around the dipole alone does not help. A point on a face,
edge or vertex inside one material takes the load of one of the elements
around it, chosen deterministically; the choice changes the power far below
that error.

## Sweeps and gradients

- A dipole's load is a fixed vector times $\omega^2$, so the adaptive sweep
  ({doc}`frequency-sweep`) takes dipoles, port-free models included, and
  gives their emitted power and port waves at every grid frequency.
- `emitted_power`, `port_waves` and `purcell_factor` are differentiable with
  respect to material fields and law parameters. For a moment that is real
  up to one phase, the dipole's own solution is its adjoint: the gradient
  costs no extra solve. A circular moment, such as $(0, 1, i)/\sqrt2$, costs
  one adjoint column.
- Node coordinates ({py:func}`gfem.shape.boundary_morph`) are differentiable
  with an electric dipole: its load moves with the element that holds it.
- With `source_fields=False` an adaptive sweep's samples come from the port
  solutions alone: its basis does not grow with the number of dipoles, and
  the port waves are read from the reduced model at every grid frequency.
- The port waves of `source_fields=False` differentiate like those of the solved source fields.
  The backward solves one adjoint per port at each frequency carrying a
  cotangent, whatever the number of dipoles. On a uniform sweep the gradients
  are exact. On an adaptive sweep they are exact at the samples and accurate
  to the sweep's field error between them, at one full-size factorization per
  frequency carrying a cotangent: cheap for an objective at a few
  frequencies, about a uniform backward over the whole grid when every
  frequency counts.
- The dipole's position and moment are fixed values: gfem does not
  differentiate with respect to them.

## Limits

- **A position inside one lossless, isotropic material**: a dipole raises a
  `ValueError` on an interface between two materials, in a perfectly matched
  layer region, in an excluded volume or outside the mesh, and in a lossy
  (loss tangent, conductivity, a lossy permittivity law), anisotropic or
  {py:obj}`gfem.MaterialField` material. In a lossy medium the emitted power
  grows without bound as the mesh is refined; on an interface the field
  normal to it and the bulk power are ambiguous.
- **No boundary or sheet face**: a dipole on a PEC, PMC, absorbing,
  impedance or conductor face, on a port face, or on the boundary of the
  solve domain raises: its bulk power is undefined there.
- **Node gradients of electric dipoles only**: with a magnetic dipole,
  `jax.grad` with respect to the node coordinates raises a
  `gfem.NonDifferentiableError`.
- **A power side clear of the dipole**: `power_flow` refuses a side whose
  outside element layer holds a dipole.
- **Port waves alone under `source_fields=False`**: it takes point dipoles
  only (an incident field raises `NotImplementedError`), and the model needs a
  port with a defined wave (a wave port, or a lumped port with a resistance).
  With `excite=()` it takes no `keep_fields`. The dipoles' own fields are not
  solved, so there is no emitted power, Purcell factor, kept field or far
  field of them: those readers raise `ValueError`.
- **Point dipoles only**: there are no line or area currents.
