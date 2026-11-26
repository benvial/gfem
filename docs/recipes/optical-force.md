# Optical force and torque

Find how hard light pushes and twists a body: {py:obj}`gfem.optical_force`
gives the time-averaged {term}`optical force` and torque on a body from a
solved 3D electric field, in N and N m, whether the body floats in free space
under a plane wave, sits in a cavity or waveguide, or is held on a metal wall.
The force is differentiable, so a design can be pushed, pulled or trapped.

The body is a set of elements: a volume region, several (their union), or a
predicate on the element centroids, as for {py:obj}`gfem.power_flow`. The
force is the derivative of the discrete time-averaged Lagrangian
$L = \tfrac14\int(\varepsilon|E|^2 - |B|^2/\mu)\,dV$ of the solved field at
fixed field coefficients, under a rigid motion of the body's nodes. It is the
volume form of the Maxwell stress tensor on the {term}`stress shell`, the
elements outside the body that touch it: there is no surface to place, and
another shell would change the number by discretization error only. The
result converges with the mesh to the physical force.

Units are SI, phasors are peak values and follow $\exp(+i\omega t)$
({doc}`../conventions`).

## Force on a sphere

The sphere of {doc}`cross-sections` and {doc}`incident-fields` here lossless
($\varepsilon = 4$, radius 0.25 µm, $ka = \pi/2$), under a plane wave at 30°
from +z with its electric field along y, solved at order 3. The sphere is a
{py:obj}`gfem.MaterialField` so that the force can be differentiated in its
permittivity below:

```python
import urllib.request

import jax.numpy as jnp
import numpy as np

import gfem

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/sphere.msh"
)
urllib.request.urlretrieve(url, "sphere.msh")
mesh = gfem.Mesh.from_gmsh("sphere.msh", length_unit="um")
air = gfem.Material()
n_sphere = len(mesh.elements("sphere"))
theta = np.radians(30)
wave = gfem.PlaneWave("te", theta=theta, phi=0.0, polarization="TE")
f = 299792458.0 / 1e-6


def solve(eps):
    """The model with permittivity ``eps`` in every sphere element, and its result."""
    model = gfem.Model(
        mesh,
        materials={
            "sphere": gfem.MaterialField(eps_r=jnp.full(n_sphere, eps)),
            "air": air,
            "pml": air,
        },
        pml=gfem.PML("pml"),
        background=air,
    )
    problem = gfem.Driven(f, sources=[wave], keep_fields=True)
    options = gfem.SolverOptions(order=3, linear_solver="direct")
    result = gfem.solve(model, problem, options=options)
    return model, result


model, result = solve(4.0)
field = result.e_field(f, "te")
pressure = gfem.optical_force(field, "sphere", model=model)

# the force unit I pi a^2 / c: radiation pressure of the incident wave on the
# sphere's cross section, at the radius of the meshed volume
p = np.asarray(mesh.nodes)[mesh.tetrahedra[mesh.elements("sphere")]]
volume = (
    abs(
        np.einsum(
            "ij,ij->i",
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0],
        )
    ).sum()
    / 6
)
radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
intensity = float(gfem.cross_sections(result, "te", model=model).intensity[0])
unit = intensity * np.pi * radius**2 / 299792458.0  # N
direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
force = np.asarray(pressure.force) / unit
print(f"force unit I pi a^2 / c = {unit:.4e} N")
print(f"along the wave: Q_pr = {force @ direction:.4f}")
print(f"across it: {force - (force @ direction) * direction}")
```

```text
force unit I pi a^2 / c = 8.0776e-25 N
along the wave: Q_pr = 2.1555
across it: [-0.00467899 -0.00401752  0.00270142]
```

`pressure.force` is the force in N for the incident wave's own amplitude
(1 V/m by default): a driven field is the peak phasor for 1 W incident at its
port, or for the plane wave's `amplitude`, so the force scales with the
incident power. Divided by $I\pi a^2/c$ it is the radiation-pressure
efficiency $Q_\mathrm{pr} = Q_\mathrm{ext} - g\,Q_\mathrm{sca}$, with $g$ the
asymmetry of the scattering. It points along the wave, as a lossless sphere
that scatters mostly forward should push; across the wave the force is the
mesh's asymmetry, 0.5 % of the push, since a sphere has none. Against the Mie
series at the radius of the meshed volume ($Q_\mathrm{pr} = 2.1853$) the
error is -1.4 %, a little over the cross sections' own on this mesh
({doc}`cross-sections`).

## Torque and its origin

The torque is taken about `origin`, by default the body's volume centroid.
Moving the origin adds the moment of the force, $\tau(o_2) = \tau(o_1) +
(o_1 - o_2)\times F$, exactly:

```python
origin = np.array([1e-6, 0.0, 0.0])  # m
about = gfem.optical_force(field, "sphere", model=model, origin=origin)
moment = np.cross(np.asarray(pressure.origin) - origin, pressure.force)
print(np.allclose(about.torque, pressure.torque + moment, rtol=1e-9, atol=1e-40))
```

```text
True
```

A sphere under a plane wave has no torque about its centre by symmetry; an
off-centre or asymmetric body does, from the force's moment and from the
rotation of the field's stress. The torque of a rigid rotation is about axes
through the centroid, whatever the origin.

## Eigenmodes at 1 J

An eigenmode field is normalized to 1 J of stored energy, so its force on a
body is $-(1\,\mathrm{J}/\omega)\,\mathrm{d}\omega/\mathrm{d}\xi$ with $\xi$
the translation of the body: the force is the slope of the resonance
frequency against position, which is the optical spring of a cavity. It
scales linearly with the stored energy. A mode of a metal cavity pushes on the
wall of an excluded metal region, which may slide on the cavity's other walls:

```python
cavity_mesh = gfem.Mesh.from_gmsh("cavity.msh", length_unit="mm")
cavity = gfem.Model(
    cavity_mesh,
    materials={"air": gfem.Material()},
    excluded=["slab"],
    boundaries=[gfem.PEC("wall"), gfem.PEC("front")],
)
modes = gfem.solve(cavity, gfem.Eigenmode(target=8.2e9, n_modes=1))
wall = gfem.optical_force(modes.e_field(0), "slab", model=cavity)
print(wall.force)
```

The slab spans the cavity's walls, so the components along them are NaN: the
slab can only move along x, and `wall.force[0]` is the push on its face. At
1 J the TE101 mode of a rectangular cavity of section $a \times b$ and length
$d$ pushes on the end wall $x = a$ with $1\,\mathrm{J}/[a(1 + (a/d)^2)]$. A
body that touches a wall it does not own gets NaN for the motions that
would slide it along that wall, and the force held on the wall for the others.

## Force between two waveguides

Two silicon slab waveguides side by side pull on each other through the
overlap of their evanescent fields. In the fundamental (bonding) TE
supermode, the force on a slab per unit length is
$F' = (P/c)\,\partial n_\mathrm{eff}/\partial q$ for the guided power $P$,
with $q$ the displacement of the slab: gfem differentiates the effective
index of a {py:obj}`gfem.PortMode` through a node map, so there is no field
to integrate. {download}`slab_pair_section.py <slab_pair_section.py>` meshes
the section: two slabs 220 nm wide and 3.48 in index, `gap` apart, in air,
between perfectly conducting walls along the electric field (the slabs are
as tall as the section):

```python
import jax
import scipy.constants

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/slab_pair_section.py"
urllib.request.urlretrieve(url, "slab_pair_section.py")
from slab_pair_section import slab_pair_section  # noqa: E402

c = scipy.constants.c
silicon = gfem.Material(eps_r=3.48**2)


def force_per_power(gap):
    """The force on the second slab, N/m per W: (1/c) d n_eff / d(its position)."""
    mesh = slab_pair_section(gap)
    model = gfem.Model(
        mesh,
        materials={"air": air, "si1": silicon, "si2": silicon},
        boundaries=[gfem.PEC("wall")],
        ports=[gfem.WavePort("p", "front")],
    )
    nodes = np.asarray(mesh.nodes)
    moving = np.unique(np.asarray(mesh.tetrahedra)[mesh.elements("si2")])
    away = np.zeros_like(nodes)
    away[moving, 0] = 1.0  # every node of the second slab moves along +x

    def n_eff(q):
        moved = model.replace(mesh=mesh.with_nodes(jnp.asarray(nodes) + q * away))
        mode = gfem.solve(
            moved,
            gfem.PortMode("p", c / 1.55e-6),
            options=gfem.SolverOptions(order=3, linear_solver="direct"),
        )
        return jnp.real(mode.n_eff[0])

    return float(jax.grad(n_eff)(0.0)) / c


for gap in (100e-9, 300e-9):
    f_gap = force_per_power(gap)
    print(f"gap {gap * 1e9:.0f} nm: {f_gap * 1e-3 * 1e6:.3f} pN/um per mW")
```

```text
gap 100 nm: -4.760 pN/um per mW
gap 300 nm: -0.591 pN/um per mW
```

The sign is that of $q$, the slab's displacement away from the other one: the
force is negative, an attraction, and it falls by a factor 8 as the gap
triples. A closed form of two slabs (a transfer-matrix solution of the same
structure) gives -4.760 and -0.5914 pN/µm per mW, within $10^{-5}$.

## Gradient force

The force and the torque are differentiable through the field and, for an
eigenmode, its frequency. The change of the push on the sphere with its
permittivity, the same in every element, is `jax.grad` of one component of
the force:

```python
def push(eps):
    model, result = solve(eps)
    field = result.e_field(f, "te")
    return gfem.optical_force(field, "sphere", model=model).force[2]


d_force = jax.grad(push)(4.0)
print(f"d F_z / d eps = {float(d_force) / unit:.3f} I pi a^2 / c")
```

```text
d F_z / d eps = 1.243 I pi a^2 / c
```

The gradient agrees with a finite difference of the solves. It is exact for the solved field: the force is a function of the
field coefficients, the node coordinates, the materials and, for an
eigenmode, the frequency, and `jax.grad` carries the cotangent back through
the solve. A design of the body's material (a {py:obj}`gfem.MaterialField`)
gets its gradient; the node coordinates and the materials of the stress shell
do not.

## Limits

- **A 3D electric field**: the field is the complex phasor of a
  {py:obj}`gfem.Driven` or {py:obj}`gfem.Eigenmode` solve; a magnetic field,
  a static field, a real field and a 2D model are refused by name.
- **The field's own model**: `model=` must be the model the field was solved
  with, on the same mesh.
- **A body with a shell**: the body selects at least one element and has a
  stress shell, so it is not the whole solve domain.
- **One plain medium in the shell**: the stress shell is one homogeneous,
  isotropic, lossless medium. The body itself may be anything.
- **Nothing special beside the body**: a port, an absorbing surface, a
  perfectly matched layer or a point dipole in or touching the shell, and a
  body node on a periodic pair, are refused.
- **Sheets on the body or off it**: a sheet (a {py:obj}`gfem.Conductor` or
  {py:obj}`gfem.Impedance`) lies wholly on the body, or off it; one that
  touches the body without lying on it is refused.
- **Total fields only**: of an incident-field column the total field is read,
  not its incident or scattered part.
- **One process, no refinement**: a field across MPI ranks and an objective
  classified for {py:obj}`gfem.refine_mesh` are refused.
- **No gradient through the geometry or the shell**: the gradient is exact
  through the field coefficients and, for an eigenmode, the real part of its
  frequency; the node coordinates, a {py:obj}`gfem.MaterialField` in the stress
  shell and the coefficients of an adaptive sweep raise
  {py:obj}`gfem.NonDifferentiableError` by name. A `gfem.Material` leaf cannot
  be traced: design the body with a {py:obj}`gfem.MaterialField`.
- **Typed arguments**: the arguments are checked by type, and an unknown
  region raises {py:obj}`gfem.RegionNotFoundError`.
