# Model an anisotropic material

Model a crystal whose permittivity, permeability, loss tangent or
conductivity depends on the direction of the field. Lithium niobate guides
its TE and TM modes with different indices. Sapphire loses energy
differently along and across its c-axis. gfem takes each property as a real
symmetric tensor, and the tensors of one {term}`anisotropic material` share
one set of {term}`principal axes`. Driven, eigenmode, electrostatic and port
mode solves all read them, and gradients reach their
{term}`principal values`.

## Give the tensor

An x-cut lithium-niobate film at 1.55 µm, with its extraordinary (optic) axis
along y in the chip plane: three principal values along the coordinate axes.

```python
import numpy as np

import gfem

no2, ne2 = 2.211**2, 2.138**2
ln = gfem.Material(eps_r=(no2, ne2, no2), name="LN")  # extraordinary axis along y
clad = gfem.Material(eps_r=1.444**2)
ln.eps_r  # (4.889, 4.571, 4.889), along x, y, z
```

- Per principal axis `a_k`, the complex permittivity is
  `ε0 eps_r[k] (1 - i tan_delta[k]) - i conductivity[k] / ω` under e^{+iωt}
  ({py:obj}`gfem.conventions`). The tensor is the sum of these values times
  `a_k a_kᵀ`. A single value counts for all three axes.
- Every tensor property of one material shares the same principal axes, so
  the loss is coaxial with the permittivity.
- As for an isotropic `gfem.Material`, the media are linear, local and
  reciprocal, with values constant in frequency.

## Turn the axes

When the principal axes are not the coordinate axes, give them as `axes`,
three orthonormal vectors, one row per principal value. The same film with
its extraordinary axis turned 10° in the chip plane:

```python
c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
turned = gfem.Material(eps_r=(no2, ne2, no2), axes=[(c, s, 0), (-s, c, 0), (0, 0, 1)])
```

Or give the tensor itself, a real symmetric 3x3 matrix in the global frame,
and leave `axes` as `None`. gfem stores it in the same form as above: the
principal values and the axes it derives (axis k is the one closest to
coordinate axis k, with its largest component positive). Any rotation
library will do for building the matrix:

```python
r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
tensor = gfem.Material(eps_r=r @ np.diag([no2, ne2, no2]) @ r.T)
tensor.eps_r  # ≈ (no2, ne2, no2)
tensor.axes  # ≈ ((c, s, 0), (-s, c, 0), (0, 0, 1)), as `turned`
# a diagonal matrix keeps axes=None: the same material as the three values
gfem.Material(eps_r=np.diag([no2, ne2, no2])) == gfem.Material(
    eps_r=(no2, ne2, no2)
)  # True
```

`mu_r`, `tan_delta` and `conductivity` take the same three forms, on the same
axes. A wire grid that conducts along x only:

```python
grid = gfem.Material(eps_r=2.1, conductivity=(5e4, 0.0, 0.0))
```

Equal principal values are one value: `gfem.Material(eps_r=(2, 2, 2))` is
stored as `eps_r=2.0`, equals `gfem.Material(eps_r=2)`, and is `isotropic`.

A material whose tensors cannot be put in this form raises `ValueError` when
it is built. A loss tensor whose principal axes differ from the
permittivity's is one: give the loss on the permittivity's axes. A
non-symmetric tensor (a {term}`gyrotropic material`, such as a magnetized
ferrite) is the other, and gfem has no model for it.

## At ports and boundaries

The x-cut guide fits in a model like any other material:

```python
model = gfem.Model(
    mesh,
    materials={"core": ln, "design": clad, "clad": clad},
    ports=[gfem.WavePort("in", "x0"), gfem.WavePort("out", "x1")],
    boundaries=[gfem.Absorbing("box", order=1)],
)
problem = gfem.Driven(frequency=193.4e12)
s = gfem.solve(model, problem).s
```

- **Wave ports.** A port's cross-section may hold anisotropic materials only
  when they are {term}`transverse-anisotropic`. Every tensor (`eps_r` with
  `tan_delta`, `mu_r`, `conductivity`) then has a principal axis along the
  port normal, with any orientation in the port plane. The x-cut film passes on
  guides along x or y (its axes include both). A port on a guide at 45° in
  the plane does not, and the port mode solve raises `ValueError` naming the
  region and the property. End such a port in an access guide that is
  transverse-anisotropic there, or in an isotropic one. Away from the ports
  the principal axes may point in any direction.
- **First-order absorbing boundaries.** `gfem.Absorbing(..., order=1)` next
  to an anisotropic material applies that material's
  {term}`absorbing admittance` for each face's normal. It is exact for a
  plane wave at normal incidence and, like the isotropic first order,
  reflects at oblique incidence: place it where the outgoing waves meet it
  close to normal. It reads the real parts of ε and μ, as on
  isotropic faces.
- **Second-order absorbing boundaries** raise `NotImplementedError` when an
  anisotropic material, or a material field with `axes`, touches the surface.
  Use order 1, or put an isotropic buffer layer between the anisotropic
  material and the boundary.

## Loss per principal axis

A resonator on a sapphire substrate, c-axis along z: the substrate's loss
tangent differs along and across the c-axis. Solve the resonator lossless,
then weigh assumed loss tangents along each principal axis by
{term}`principal participation`. `modes.principal_participation` holds one
`(n_modes, 3)` array per anisotropic region, in the order of the material's
`axes` (here the coordinate axes):

```python
sapphire = gfem.Material(eps_r=(9.3, 9.3, 11.5), name="sapphire")  # c-axis along z
model = gfem.Model(
    mesh,
    materials={"substrate": sapphire},
    boundaries=[gfem.PEC("metal"), gfem.PEC("box")],
)
modes = gfem.solve(model, gfem.Eigenmode(target=7.5e9, n_modes=2))

p = np.asarray(modes.principal_participation["substrate"])  # (n_modes, 3)
tan_delta = np.array([5e-8, 5e-8, 2e-8])  # assumed, along x, y, z
inverse_q_substrate = p @ tan_delta  # (n_modes,)
```

- The three columns add up to `modes.bulk_participation["substrate"]`.
- A `conductivity` given as principal values adds
  `p[:, k] * conductivity[k] / (ε0 eps_r[k] Re ω)` on each axis.
- The sum is exact for any field direction, with the losses uniform over the
  region. `bulk_participation × tan_delta` is the rule for an isotropic loss:
  it does not apply to principal values.

Isotropic regions have no entry in `principal_participation`.

## Design with an anisotropic material

In the model above, the region `"design"` (between the two guides, away
from the ports) holds the cladding. A density blended there between the
cladding and the LN film is an
{term}`anisotropic material field`: its permittivity holds three principal
values per element, along the film's principal axes.

```python
import jax
import jax.numpy as jnp


def loss(rho):
    field = gfem.MaterialField.interpolate(rho, void=clad, solid=ln)  # axes from ln
    m = model.with_material("design", field)
    return -(jnp.abs(gfem.solve(m, problem).s["out", "in"]) ** 2)


n = len(mesh.elements("design"))
grad = jax.grad(loss)(jnp.full(n, 0.5))  # (n,), one value per element
```

- **Shapes.** `field.eps_r` is `(n, 3)`, and its gradient has the same
  shape. `gfem.MaterialField.uniform("design", ln)` gives `(3,)` arrays: the
  sensitivity to each principal value of the whole region.
  `gfem.MaterialField(eps_r=values, axes=...)` with `values` of shape
  `(n, 3)` builds one directly.
- **Shared axes.** `interpolate` blends two materials per principal value
  when their tensors share principal axes. An isotropic material shares any,
  so cladding and LN blend. Two anisotropic materials on different
  principal axes raise `ValueError`: blend each with an isotropic one
  instead.
- **What differentiates.** Gradients with respect to the principal values of
  `eps_r` (driven, eigenmode and electrostatic solves) and of `conductivity`
  (driven and eigenmode solves) are checked against finite differences: see
  {doc}`../capabilities`. The axes, the material's orientation, are not a
  design input: axes differentiated by `jax.grad` raise
  `gfem.NonDifferentiableError`, so keep them
  fixed numbers.
- **Ports.** As for any design, keep the design region off the wave port
  faces ({doc}`s-parameter-gradient`).

## Check your results

- Rotate the whole model (mesh, ports, voltage paths) and the material's axes
  together: S should not change.
- A lossless device with lossless anisotropic materials should keep
  `SᴴS ≈ I`.
- A solve with the loss tangents in the material should give the same 1/Q as
  the principal participations weighted by those loss tangents, to first
  order in 1/Q.

How gfem itself is checked on anisotropic materials: {doc}`../verification`.

## Limits

- **Shape gradients** (mesh node coordinates) raise
  `gfem.NonDifferentiableError` when an absorbing face has an anisotropic
  material behind it. Put an isotropic buffer layer next to the absorbing
  face.
- **Interface participation.** The thin layer of
  `gfem.interface_participation` is isotropic. Its neighbours may be
  anisotropic, and a side counts as vacuum only when no wave in it is slower
  than in vacuum ({doc}`measurements`).
- **No dispersion and no rotation helper.** Values are constant in frequency:
  solve each band with its own values. gfem has no constructor for axes from
  Euler angles or a crystal cut: build the rotation with numpy or scipy.

## Next

{py:obj}`gfem.Material`, {py:obj}`gfem.MaterialField`,
{py:obj}`gfem.WavePort`, {py:obj}`gfem.Absorbing`,
{py:obj}`gfem.conventions`, {doc}`s-parameter-gradient`,
{doc}`eigenmode-gradient`, {doc}`measurements`.
