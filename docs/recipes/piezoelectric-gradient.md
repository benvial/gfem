# Transducer admittance gradients

Find how the radiation conductance of a transducer depends on what sets it:
the stiffness, piezoelectric tensor, permittivity, density and quality factor
of its materials, the pitch and the thickness of its fingers, and the strain
it puts at a point. One `jax.grad` through {py:obj}`gfem.solve` gives each
exactly, for one adjoint solve of the {py:class}`gfem.Piezoelectric`
problem, with no remesh. Material values go in a {py:obj}`gfem.MaterialField`.
Geometry goes through a {term}`node map` of your own, a JAX function of your design
parameters that returns the mesh's node coordinates, which
{py:meth}`gfem.Mesh.with_nodes` puts on the mesh.

The device is the section of {doc}`piezoelectric-admittance` and of the
{doc}`transducer tutorial <../tutorials/transducer>`: aluminium fingers on a
film of x-cut lithium niobate over oxide and a silicon handle, built by the
tutorial's {download}`transducer_section.py <../tutorials/transducer_section.py>`
with two fingers on each side of its ribs' span and coarse edges, so a gradient
takes seconds. The page uses one frequency, 0.86 GHz, and the radiation
conductance $\mathrm{Re}\,Y'$ there.

## The conductance as a function of its inputs

The materials of the page's model are those of the {doc}`admittance recipe
<piezoelectric-admittance>`; only the film and the fingers' metal are differentiable
here. {py:meth}`gfem.MaterialField.uniform` turns the film's constant material
into a field whose leaves `jax.grad` reaches: its stiffness, piezoelectric tensor,
principal permittivities on the crystal's `axes`, density and quality factor.
The fingers' stiffness comes from their Young's modulus and Poisson's ratio
through {py:obj}`gfem.isotropic_stiffness`, which is written in `jax.numpy`, so
the gradient reaches those two numbers by the chain rule.

```python
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/transducer_section.py"
urllib.request.urlretrieve(url, "transducer_section.py")
from transducer_section import transducer_section  # noqa: E402

c11, c12, c13, c14, c33, c44 = 203e9, 53e9, 75e9, 9e9, 245e9, 60e9
c_ln = np.array(
    [
        [c11, c12, c13, c14, 0, 0],
        [c12, c11, c13, -c14, 0, 0],
        [c13, c13, c33, 0, 0, 0],
        [c14, -c14, 0, c44, 0, 0],
        [0, 0, 0, 0, c44, c14],
        [0, 0, 0, 0, c14, (c11 - c12) / 2],
    ]
)
e15, e22, e31, e33 = 3.7, 2.5, 0.2, 1.3  # C/m^2
e_ln = np.array(
    [
        [0, 0, 0, 0, e15, -e22],
        [-e22, e22, 0, e15, 0, 0],
        [e31, e31, e33, 0, 0, 0],
    ]
)
c_si = np.zeros((6, 6))
c_si[:3, :3] = 63.9e9
c_si += np.diag([165.7e9 - 63.9e9] * 3 + [79.6e9] * 3)

axes = ((0, 1, 0), (1, 0, 0), (0, 0, -1))  # x-cut
q = dict(mechanical_quality_factor=200.0)
ln = gfem.Material(
    eps_r=(44.0, 44.0, 29.0),
    axes=axes,
    stiffness=c_ln,
    piezoelectric_tensor=e_ln,
    mass_density=4700.0,
    **q,
)
oxide = gfem.Material(
    eps_r=3.9, stiffness=gfem.isotropic_stiffness(73e9, 0.17), mass_density=2203.0, **q
)
silicon = gfem.Material(stiffness=c_si, mass_density=2329.0, **q)

section = transducer_section(outer=2, between=1, ribs=False, edge_scale=3.0)
frame = gfem.DampingFrame(["ln_frame", "box_frame", "si_frame"])
problem = gfem.Piezoelectric(
    0.86e9,
    terminals={"signal": "signal"},
    ground="ground",
    clamped="bottom",
    damping_frame=frame,
    keep_fields=True,
)
film = gfem.MaterialField.uniform("ln", ln)


def solved(film, young, poisson, mesh=section.mesh):
    """The result, as a function of the film's field, the metal and the mesh."""
    metal = gfem.MaterialField(
        stiffness=gfem.isotropic_stiffness(young, poisson),
        mass_density=2700.0,
        **q,
    )
    model = gfem.Model(
        mesh,
        materials={
            "ln": film, "ln_frame": ln, "box": oxide, "box_frame": oxide,
            "si": silicon, "si_frame": silicon, "signal": metal, "ground": metal,
            "air": gfem.Material(),
        },
        excluded={"mechanical": ["air"], "em": ["si", "si_frame"]},
    )  # fmt: skip
    return gfem.solve(model, problem)


def conductance(*args):
    """Re Y' (S/m) at 0.86 GHz."""
    return jnp.real(solved(*args).admittance_per_length[0, 0, 0])


value, grads = jax.value_and_grad(conductance, argnums=(0, 1, 2))(film, 70e9, 0.35)
print(f"Re Y' = {float(value):.4f} S/m")
```

```text
Re Y' = 0.1407 S/m
```

`grads` has the film's shape, one gradient per leaf, and one number each for the
metal's modulus and ratio. The change of $\mathrm{Re}\,Y'$ for a 1 % change of each
input, $x\,\partial \mathrm{Re}\,Y'/\partial x$ divided by 100 and by the value, is
the share of the conductance that input carries:

```python
film_grad, d_young, d_poisson = grads
pairs = {
    "stiffness": jnp.sum(film.stiffness * film_grad.stiffness),
    "piezoelectric tensor": jnp.sum(
        film.piezoelectric_tensor * film_grad.piezoelectric_tensor
    ),
    "permittivity": jnp.sum(film.eps_r * film_grad.eps_r),
    "density": film.mass_density * film_grad.mass_density,
    "quality factor": film.mechanical_quality_factor
    * film_grad.mechanical_quality_factor,
    "finger modulus": 70e9 * d_young,
    "finger Poisson ratio": 0.35 * d_poisson,
}
for name, elasticity in pairs.items():
    print(f"{name}: {float(elasticity) / float(value):+.3f}")
```

```text
stiffness: -1.126
piezoelectric tensor: +1.683
permittivity: -0.024
density: +0.979
quality factor: -0.006
finger modulus: +0.013
finger Poisson ratio: -0.001
```

A scale of a tensor is one direction in its space; the gradient has an entry for
every one, so it also says which entries matter: `film_grad.piezoelectric_tensor`
is a 3 × 6 array in the crystal's frame, in the order of the constants it came from.

Each gradient is checked against a central difference at a relative step small
against the curvature (see the last section). Here the density, one number:

```python
import dataclasses


def with_density(rho):
    return dataclasses.replace(film, mass_density=rho)


h = 1e-4
rho = float(film.mass_density)
fd = (
    float(conductance(with_density(rho * (1 + h)), 70e9, 0.35))
    - float(conductance(with_density(rho * (1 - h)), 70e9, 0.35))
) / (2 * h * rho)
print(
    f"d Re Y'/d rho: gradient {float(film_grad.mass_density):.6e}, difference {fd:.6e}"
)
```

```text
d Re Y'/d rho: gradient 2.931159e-05, difference 2.931159e-05
```

## The pitch, through a node map

A mesh made for one pitch serves a nearby one when its nodes move: no remesh,
and the gradient reaches the pitch through the nodes. A node map is a JAX
function of the mesh's coordinates and your design parameters. This one scales
everything between the lateral frames about the left frame's inner edge, which
carries the fingers with it, and translates the right frame rigidly. The
frames keep their width, so the damping stays what it was, and
`section.interior` is the span between them.

```python
x_left, x_right = section.interior  # m
nodes = jnp.asarray(section.mesh.nodes)


def pitch_map(nodes, sigma):
    x = nodes[:, 0]
    inside = jnp.clip(x, x_left, x_right)
    return nodes.at[:, 0].set(x - inside + x_left + sigma * (inside - x_left))


def of_sigma(sigma):
    mesh = section.mesh.with_nodes(pitch_map(nodes, sigma))
    return conductance(film, 70e9, 0.35, mesh)


g, dg = jax.value_and_grad(of_sigma)(1.0)
h = 1e-4
fd = (of_sigma(1 + h) - of_sigma(1 - h)) / (2 * h)
print(f"d Re Y'/d sigma: gradient {float(dg):.5f} S/m, difference {float(fd):.5f} S/m")
```

```text
d Re Y'/d sigma: gradient 0.15039 S/m, difference 0.15039 S/m
```

{py:meth}`gfem.Mesh.with_nodes` takes the coordinates in metres with the shape of
`mesh.nodes`, and returns a mesh with the same elements and regions. The damping
frames' graded loss follows their material, so a frame that moves rigidly keeps its
absorption; a map that stretches a frame changes it. A map that inverts or flattens
an element raises {py:obj}`gfem.MeshQualityError` when its coordinates are concrete,
and when `jax.grad` traces it, at the solve:

```python
try:
    section.mesh.with_nodes(nodes * jnp.array([-1.0, 1.0]))  # a mirror
except gfem.MeshQualityError as error:
    print(type(error).__name__)
```

```text
MeshQualityError
```

A thickness map is the same idea in $y$: on meshed fingers, scale the nodes between
the film's surface and the fingers' top, and let everything above follow. The
difference is taken at $\pm 1$ % of the thickness:

```python
t_finger = 50e-9  # the fingers' thickness, m


def thickness_map(nodes, tau):
    y = nodes[:, 1]
    inside = jnp.clip(y, 0.0, t_finger)
    return nodes.at[:, 1].set(y + tau * inside)


def of_tau(tau):
    mesh = section.mesh.with_nodes(thickness_map(nodes, tau))
    return conductance(film, 70e9, 0.35, mesh)


g_tau, dg_tau = jax.value_and_grad(of_tau)(0.0)
fd_tau = (of_tau(1e-2) - of_tau(-1e-2)) / 2e-2
print(
    f"d Re Y'/d tau: gradient {float(dg_tau):.5f} S/m, difference {float(fd_tau):.5f} S/m"
)
```

```text
d Re Y'/d tau: gradient 0.01598 S/m, difference 0.01598 S/m
```

## A reading of the field at a point

{py:meth}`gfem.Field.at` and {py:func}`gfem.strain` read the kept fields at points,
and they differentiate: each reading costs one more adjoint solve per frequency.
Through the nodes, a reading is Eulerian. The point stays where you put it, and the
field moves under it: the derivative is that of the field at a fixed place, not at
a material point that rides with the mesh. To follow a feature (a rib's centre, say)
as the pitch changes, move the point with the same node map.

```python
point = jnp.array([[section.interior[0] + 4.4e-6, -0.09e-6]])  # in the film


def strain_at_point(sigma):
    mesh = section.mesh.with_nodes(pitch_map(nodes, sigma))
    result = solved(film, 70e9, 0.35, mesh)
    return jnp.abs(gfem.strain(result.displacement(0.86e9), point)[0, 4])


s, ds = jax.value_and_grad(strain_at_point)(1.0)
print(f"|S_xz| = {float(s):.4e} per V, d/d sigma = {float(ds):.4e}")
```

```text
|S_xz| = 1.2397e-05 per V, d/d sigma = -5.2812e-05
```

## What raises

- The frequencies, the terminals, the clamped lines and the damping frame, with
  its loss factor: they are fixed values of the problem, and a traced one raises
  {py:obj}`gfem.NonDifferentiableError` when the problem is built.
- The parameters of a permittivity law, for the same reason.
- Node coordinates on any other 2D problem: only the piezoelectric problem
  reads them in 2D.
- A map that inverts an element: {py:obj}`gfem.MeshQualityError`.
- Point readings with respect to nodes on a 3D mesh.

## The step of a check

A central difference has a truncation error that falls as the square of its step. A
relative step of $10^{-3}$ is right for a short section like this page's. Near a sharp
resonance (a longer transducer, or one between reflectors) $\partial Y'/\partial\sigma$ is
about a hundred times $\lvert Y'\rvert$, and the step needs to be about ten times
smaller; shrink it, or use fourth-order differences, until the difference stops
changing. The adjoint gradient does not depend on a step at all.

## Next

- {doc}`../tutorials/transducer`: this gradient in a design loop, over the pitch of
  the full section.
- {doc}`piezoelectric-admittance`: the problem, step by step.
- {doc}`shape-gradient`: node gradients of the other problems, through
  `gfem.shape.boundary_morph`.
- {doc}`../conventions`, "Piezoelectric problems": the units and the frame of the
  tensors.
