# Model an anisotropic material

An anisotropic material's permittivity, permeability, loss tangent and
conductivity depend on the direction of the field: lithium niobate guides its
TE and TM modes with different indices, and sapphire loses energy differently
along and across its c-axis. gfem takes each of these properties as a real symmetric tensor, and
the tensors of one {term}`anisotropic material` share one set of
{term}`principal axes`. Driven, eigenmode, electrostatic and port mode solves
read them on every element, and gradients reach their
{term}`principal values`.

## Assumptions

- The media are linear, local and reciprocal, with values constant in
  frequency, as for an isotropic `gfem.Material`.
- Every tensor property of one material shares the same principal axes, so
  the loss is coaxial with the permittivity.
- Per principal axis `a_k`, the complex permittivity is
  `ε0 eps_r[k] (1 - i tan_delta[k]) - i conductivity[k] / ω` under e^{+iωt},
  and the tensor is the sum of these values times `a_k a_kᵀ`
  (`gfem.conventions`). A single value counts for all three axes.

## Give the tensor

An x-cut lithium-niobate film at 1.55 µm, with its extraordinary (optic) axis
along y in the chip plane: three principal values along the coordinate axes.

```python
import numpy as np

import gfem

no2, ne2 = 2.211**2, 2.138**2
ln = gfem.Material(eps_r=(no2, ne2, no2), name="LN")  # extraordinary axis along y
clad = gfem.Material(eps_r=1.444**2)
```

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
it is built: a non-symmetric tensor (a {term}`gyrotropic material`, such as a
magnetized ferrite), or a loss tensor whose principal axes differ from the
permittivity's.

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
  when they are {term}`transverse-anisotropic`: every tensor (`eps_r` with
  `tan_delta`, `mu_r`, `conductivity`) has a principal axis along the port
  normal, with any orientation in the port plane. The x-cut film passes on
  guides along x or y (its axes include both). A port on a guide at 45° in
  the plane does not, and the port mode solve raises `ValueError` naming the
  region and the property. End such a port in an access guide that is
  transverse-anisotropic there, or in an isotropic one. Away from the ports
  the principal axes may point in any direction: a model with tilted
  principal axes and its rigidly rotated copy give the same S,
  eigenfrequency and capacitance (`tests/unit/test_anisotropic_rotation.py`),
  and a uniform anisotropic material field on tilted axes solves as its
  material
  (`tests/unit/test_driven.py::test_a_uniform_field_of_a_crystal_solves_like_the_crystal`).
- **First-order absorbing boundaries.** `gfem.Absorbing(..., order=1)` next
  to an anisotropic material applies that material's
  {term}`absorbing admittance` for each face's normal. It is exact for a
  plane wave at normal incidence and, like the isotropic first order,
  reflects at oblique incidence. It reads the real parts of ε and μ, as on
  isotropic faces.
- **Second-order absorbing boundaries** raise `NotImplementedError` when an
  anisotropic material touches the surface. Use order 1, or put an isotropic
  buffer layer between the anisotropic material and the boundary.

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
- A solve with the loss tangents in the material gives the same 1/Q from its
  complex eigenfrequency, to first order in 1/Q.

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

- **Shapes.** `field.eps_r` is `(n, 3)`, and its cotangent has the same
  shape. `gfem.MaterialField.uniform("design", ln)` gives `(3,)` leaves: the
  sensitivity to each principal value of the whole region.
  `gfem.MaterialField(eps_r=values, axes=...)` with `values` of shape
  `(n, 3)` builds one directly.
- **Shared axes.** `interpolate` blends two materials per principal value
  when their tensors share principal axes. An isotropic material shares any,
  so cladding and LN blend; two anisotropic materials on different
  principal axes raise
  `ValueError`.
- **What differentiates.** Gradients with respect to the principal values of
  `eps_r` (driven, eigenmode and electrostatic solves) and of `conductivity`
  (driven and eigenmode solves) are checked against finite differences: see
  {doc}`../capabilities`. The axes, the material's orientation, are not a
  design input.
- **Ports.** As for any design, keep the design region off the wave port
  faces ({doc}`s-parameter-gradient`).

## Trust

The test suite checks the anisotropic solve against closed forms and
identities. No tolerance is quoted here; each is in its test.

- An absorbing boundary ending a biaxial, magnetic slab reflects as the
  half-space of transmission-line theory, for both polarizations and at two
  electrical slab thicknesses:
  `tests/unit/test_driven_boundaries.py::test_first_order_absorbing_boundary_ends_an_anisotropic_slab_as_a_half_space`.
- A WR-28 guide filled with anisotropic ε and μ transmits its TE10 mode with
  the closed-form propagation constant, without reflecting:
  `tests/verification/test_anisotropic_wave_ports.py::test_a_guide_with_anisotropic_permeability_transmits_its_te10_mode`.
  The same guide with its tensors turned about the guide axis does not
  reflect either:
  `tests/verification/test_anisotropic_wave_ports.py::test_a_fill_turned_about_the_guide_axis_does_not_reflect`.
- The port mode of that turned tensor propagates, with its propagation
  constant between those of the two guides filled with the tensor's
  principal values across the guide (`eps_r` = 2 and 4):
  `tests/unit/test_port_mode.py::test_a_tensor_turned_about_the_guide_axis_keeps_a_propagating_mode`.
- The loss along each principal axis from `principal_participation` matches
  1/Q from the complex eigenfrequency, on an anisotropic material whose
  field lies off every principal axis:
  `tests/unit/test_eigenmode.py::test_principal_participation_gives_a_per_axis_loss_exactly`.
- A model and its rigidly rotated copy, with the material given as rotated
  global matrices, give the same S, eigenfrequency and capacitance
  (`tests/unit/test_anisotropic_rotation.py`); a wave-port guide rotated to an
  oblique port normal gives the same S
  (`tests/verification/test_anisotropic_wave_ports.py`).
- The principal-value gradients are cells of the gradient verification
  matrix on {doc}`../capabilities`.

Two checks on your own model:

- Rotate the whole model (mesh, ports, voltage paths) and the material's axes
  together: S must not change.
- A lossless device with lossless anisotropic materials keeps `SᴴS ≈ I`.

## Limits

- **Real symmetric tensors on shared axes only.** Gyrotropic
  (non-symmetric) tensors, and loss whose principal axes differ from the
  permittivity's, raise `ValueError` ({py:obj}`gfem.Material`).
- **Wave ports need transverse-anisotropic cross-sections**: every tensor in
  the cross-section has a principal axis along the port normal. Otherwise the
  port mode solve raises `ValueError`; rotate the material or the port, or
  end the port in a transverse-anisotropic or isotropic access guide
  ({py:obj}`gfem.WavePort`).
- **First-order absorbing boundaries are exact at normal incidence only**, as
  for isotropic materials, and read the real parts of ε and μ
  ({py:obj}`gfem.Absorbing`).
- **Second-order absorbing boundaries** raise `NotImplementedError` next to
  an anisotropic material, or a material field with `axes`. Use order 1 or an
  isotropic buffer layer.
- **The orientation is not a design input.** Principal values differentiate;
  the axes raise `gfem.NonDifferentiableError` if traced. `interpolate` needs
  the two materials' tensors to share principal axes
  ({py:obj}`gfem.MaterialField`).
- **Shape gradients** (mesh node coordinates) raise
  `gfem.NonDifferentiableError` when an absorbing face has an anisotropic
  material behind it.
- **Interface participation.** The thin layer of
  `gfem.interface_participation` is isotropic; its neighbours may be
  anisotropic, and a side counts as vacuum only when no wave in it is slower
  than in vacuum ({doc}`measurements`).
- **No dispersion and no rotation helper.** Values are constant in frequency.
  gfem has no constructor for axes from Euler angles or a crystal cut: build
  the rotation with numpy or scipy.

## Next

- {doc}`s-parameter-gradient`, {doc}`eigenmode-gradient` and
  {doc}`measurements`.
- API: {py:obj}`gfem.Material`, {py:obj}`gfem.MaterialField`,
  {py:obj}`gfem.WavePort`, {py:obj}`gfem.Absorbing` and
  `gfem.conventions`.
