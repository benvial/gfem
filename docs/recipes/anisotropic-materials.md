# Anisotropic materials

Model a crystal whose permittivity, permeability, loss tangent or conductivity
depends on the direction of the field. gfem takes each property as a real
symmetric tensor, and the tensors of one {term}`anisotropic material` share one
set of {term}`principal axes`. Driven, eigenmode, electrostatic and port mode
solves read them, and gradients reach their {term}`principal values`.

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
  `ε0 eps_r[k] (1 - i tan_delta[k]) - i conductivity[k] / ω` under $e^{+i\omega t}$
  ({py:obj}`gfem.conventions`); the tensor is the sum of these times
  `a_k a_kᵀ`. A single value counts for all three axes.
- Every tensor property of one material shares the principal axes, so the loss
  is coaxial with the permittivity.
- The media are linear, local, reciprocal and constant in frequency.

## Turn the axes

When the principal axes are not the coordinate axes, give them as `axes`, three
orthonormal vectors, one row per principal value. The same film with its
extraordinary axis turned 10° in the chip plane:

```python
c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
turned = gfem.Material(eps_r=(no2, ne2, no2), axes=[(c, s, 0), (-s, c, 0), (0, 0, 1)])
```

Or give the tensor itself, a real symmetric 3x3 matrix in the global frame, and
leave `axes` as `None`: gfem derives the principal values and axes (axis k is
the closest to coordinate axis k, largest component positive):

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

`mu_r`, `tan_delta` and `conductivity` take the same forms, on the same axes. A
wire grid that conducts along x only:

```python
grid = gfem.Material(eps_r=2.1, conductivity=(5e4, 0.0, 0.0))
```

Equal principal values are one value: `gfem.Material(eps_r=(2, 2, 2))` equals
`gfem.Material(eps_r=2)` and is `isotropic`. A material whose tensors cannot be
put in this form raises `ValueError` when built: a loss tensor on other axes
than the permittivity's (give it on the permittivity's), or a non-symmetric
tensor ({term}`gyrotropic material`), which gfem does not model.

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
  when they are {term}`transverse-anisotropic`: every tensor has a principal
  axis along the port normal, with any orientation in the port plane. The x-cut
  film passes on guides along x or y; a guide at 45° raises `ValueError`
  naming the region and property. End such a port in an access guide that
  passes, or in an isotropic one.
- **First-order absorbing boundaries** next to an anisotropic material use its
  {term}`absorbing admittance` per face normal: exact at normal incidence, like
  the isotropic first order, and on the real parts of $\varepsilon$ and $\mu$.
- **Second-order absorbing boundaries** raise `NotImplementedError` against an
  anisotropic material: use order 1 or an isotropic buffer layer.
- **Perfectly matched layers** accept axes along $x$, $y$, $z$ only
  ({doc}`perfectly-matched-layer`).

## Loss per principal axis

A resonator on a sapphire substrate, c-axis along z. Solve it lossless, then
weigh assumed loss tangents along each axis by {term}`principal participation`.
`modes.principal_participation` holds one `(n_modes, 3)` array per anisotropic
region, in the order of the material's `axes`:

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
- A `conductivity` adds `p[:, k] * conductivity[k] / (ε0 eps_r[k] Re ω)` on
  each axis.
- The sum is exact for any field direction with uniform loss;
  `bulk_participation × tan_delta` does not apply to principal values.
- Isotropic regions have no entry.

## Design with an anisotropic material

The region `"design"` (away from the ports) holds the cladding. A density
blended there between the cladding and the LN film is an
{term}`anisotropic material field`: three principal values per element, along
the film's axes.

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

- **Shapes.** `field.eps_r` is `(n, 3)`, as its gradient;
  `gfem.MaterialField.uniform("design", ln)` gives `(3,)` arrays.
  `gfem.MaterialField(eps_r=values, axes=...)` with `(n, 3)` values builds one
  directly.
- **Shared axes.** `interpolate` blends per principal value when the tensors
  share axes; an isotropic material shares any. Two anisotropic materials on
  different axes raise `ValueError`: blend each with an isotropic one.
- **What differentiates.** The principal values of `eps_r` (driven, eigenmode,
  electrostatic) and of `conductivity` (driven, eigenmode); see
  {doc}`../capabilities`. The axes are not a design input: a traced one raises
  `gfem.NonDifferentiableError`.
- Keep the design region off the wave port faces ({doc}`s-parameter-gradient`).

## Check your results

- Rotate the whole model and the material's axes together: S does not change.
- A lossless device keeps `SᴴS ≈ I`.
- A solve with loss tangents gives the 1/Q of the principal participations
  weighted by them, to first order in 1/Q ({doc}`../verification`).

## Limits

- **Shape gradients** raise `gfem.NonDifferentiableError` when an absorbing
  face has an anisotropic material behind it: add an isotropic buffer layer.
- **Interface participation.** The thin layer of
  `gfem.interface_participation` is isotropic; its neighbours may be
  anisotropic ({doc}`measurements`).
- **Dispersion is isotropic only.** A permittivity law on an anisotropic
  material raises `ValueError`; solve each band with its own constant values
  ({doc}`dispersive-materials`).
- **No rotation helper.** Build the axes from Euler angles or a crystal cut with
  numpy or scipy.

## Next

{py:obj}`gfem.Material`, {py:obj}`gfem.MaterialField`,
{py:obj}`gfem.WavePort`, {py:obj}`gfem.Absorbing`,
{py:obj}`gfem.conventions`, {doc}`s-parameter-gradient`,
{doc}`eigenmode-gradient`, {doc}`measurements`.
