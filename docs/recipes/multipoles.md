# Multipole decomposition

{py:obj}`gfem.multipoles` reads the field a {term}`scatterer` radiates under an
{term}`incident field` as a sum of electric and magnetic multipoles about an
origin you choose: a {term}`multipole decomposition`. It answers the questions
a measured spectrum raises. Which multipole carries the scattering? Do the
electric and magnetic dipoles match, so that nothing goes backward? Has the
electric dipole vanished? It gives the {term}`multipole coefficients`, the
{term}`partial scattering cross section` of each multipole and the
{term}`exact multipole moments` of the dipoles and quadrupoles, from one solve
of any scatterer gfem accepts, at any size against the wavelength.
{py:obj}`gfem.current_moments` adds the {term}`current moments` of the
{term}`induced current`, to read anapoles.

The question here: a silicon-like sphere ($\varepsilon = 12.25$, radius about
0.24 µm) lit by a plane wave at the wavelength where its electric and magnetic
dipoles are equal. Which multipoles scatter, and by how much?

## The call

The sphere of {doc}`cross-sections`, lossless, at the wavelength that puts
$ka$ of its meshed volume on the first Kerker point of an index-3.5 sphere
($ka = 0.7839$, $\lambda_0 \approx 1.93$ µm). One solve at order 3 holds
every reading below:

```python
import urllib.request

import numpy as np

import gfem

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/sphere.msh"
)
urllib.request.urlretrieve(url, "sphere.msh")
mesh = gfem.Mesh.from_gmsh("sphere.msh", length_unit="um")

# the radius of the sphere of the same volume as the meshed one, m
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

c0 = 299792458.0
f = 0.783916362855 * c0 / (2 * np.pi * radius)  # Hz

air = gfem.Material()
model = gfem.Model(
    mesh,
    materials={"sphere": gfem.Material(eps_r=12.25), "air": air, "pml": air},
    pml=gfem.PML("pml"),
    background=air,
)
wave = gfem.PlaneWave("z", theta=0, polarization="TM")
problem = gfem.Driven(f, sources=(wave,), keep_fields=True)
result = gfem.solve(model, problem, options=gfem.SolverOptions(order=3))

mp = gfem.multipoles(result, "z", model=model)
print(f"wavelength {c0 / f * 1e6:.4f} um, degrees up to {mp.lmax}")
```

```text
wavelength 1.9316 um, degrees up to 7
```

The solve keeps its fields (`keep_fields=True`), as for
{py:obj}`gfem.cross_sections`, and the model's background is the one lossless
medium the scatterer sits in. The wave comes along $+z$ with $E$ along $x$;
the sphere is at the model origin, which is the default
{term}`multipole origin`. The result is a {py:obj}`gfem.Multipoles`.

## Partial cross sections

The {term}`partial scattering cross section` of the electric (E) and magnetic
(M) multipole of each {term}`degree <Multipole degree>` $l$ is
$k^{-2}\sum_m|\alpha_{lm}|^2$ and $k^{-2}\sum_m|\beta_{lm}|^2$, in m², over
the {term}`incident intensity`. `electric_scattering` and
`magnetic_scattering` hold them, one row per kept frequency and one column
per degree:

```python
um2 = 1e12
sigma_e = np.asarray(mp.electric_scattering)[0] * um2  # per degree
sigma_m = np.asarray(mp.magnetic_scattering)[0] * um2
ed, eq = sigma_e[:2]
md, mq = sigma_m[:2]
print(f"ED {ed:.4f}  MD {md:.4f}  EQ {eq:.6f}  MQ {mq:.7f} um^2")
print(f"ED / MD = {ed / md:.3f}")
```

```text
ED 0.1908  MD 0.1827  EQ 0.000207  MQ 0.0000072 um^2
ED / MD = 1.044
```

The electric and magnetic dipoles carry almost all of the scattering and are
nearly equal; the quadrupoles are a thousandth of it.

## Compare with Mie

For a sphere the decomposition is known: its coefficients are the
Bohren-Huffman coefficients $a_l$ and $b_l$, and the partial cross sections
are $2\pi(2l+1)|a_l|^2/k^2$ and $2\pi(2l+1)|b_l|^2/k^2$. Compare at the radius
of the sphere of equal volume (the faceted sphere is 2-3 % off the nominal
one):

```python
from scipy.special import spherical_jn, spherical_yn


def mie(eps, x, lmax):
    """Bohren-Huffman a_l, b_l for l = 1..lmax at size parameter x."""
    n = np.arange(1, lmax + 1)
    m = np.sqrt(np.conj(eps))  # their e^{-i omega t} index, for gfem's eps

    def psi(z):
        return z * spherical_jn(n, z)

    def dpsi(z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    h = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    xi = x * h
    dxi = h + x * (spherical_jn(n, x, True) + 1j * spherical_yn(n, x, True))
    a = (m * psi(m * x) * dpsi(x) - psi(x) * dpsi(m * x)) / (
        m * psi(m * x) * dxi - xi * dpsi(m * x)
    )
    b = (psi(m * x) * dpsi(x) - m * psi(x) * dpsi(m * x)) / (
        psi(m * x) * dxi - m * xi * dpsi(m * x)
    )
    return a, b


k = mp.wavenumber[0]
a, b = mie(12.25, k * radius, 2)
l = np.arange(1, 3)
mie_e = 2 * np.pi * (2 * l + 1) * abs(a) ** 2 / k**2 * um2
mie_m = 2 * np.pi * (2 * l + 1) * abs(b) ** 2 / k**2 * um2
gfem_e = np.asarray(mp.electric_scattering)[0, :2] * um2
gfem_m = np.asarray(mp.magnetic_scattering)[0, :2] * um2
print(f"Mie at a_vol = {radius * 1e6:.4f} um")
error_e = np.round(100 * (gfem_e / mie_e - 1), 2)
error_m = np.round(100 * (gfem_m / mie_m - 1), 2)
print(f"E, l = 1, 2: Mie {mie_e[0]:.4f} {mie_e[1]:.2e} um^2, error {error_e} %")
print(f"M, l = 1, 2: Mie {mie_m[0]:.4f} {mie_m[1]:.2e} um^2, error {error_m} %")
```

```text
Mie at a_vol = 0.2410 um
E, l = 1, 2: Mie 0.1869 1.98e-04 um^2, error [2.07 4.76] %
M, l = 1, 2: Mie 0.1869 9.95e-06 um^2, error [ -2.27 -27.3 ] %
```

The dipoles agree with the series to about 2 %, which also puts their ratio
within the mesh's error of 1. The quadrupoles carry a thousandth and a
hundred-thousandth of the scattering, so their larger relative errors cost
nothing in the sum; a degree is read to what the mesh resolves of it. The
printed errors belong to this mesh and order: gfem's own tests hold spheres
against the series at the tolerances they measured on finer ones.

## Coefficients by degree and order

`mp.alpha` and `mp.beta` hold the dimensionless coefficients, one row per
kept frequency, in flat order. `mp.index(l, m)` gives the position of one,
and `mp.l` and `mp.m` the degree and order of each position, to build masks
(they are read-only):

```python
alpha = np.asarray(mp.alpha)[0]
beta = np.asarray(mp.beta)[0]
for l_, m_ in ((1, -1), (1, 0), (1, 1)):
    j = mp.index(l_, m_)
    print(f"l={l_} m={m_:+d}: |alpha| {abs(alpha[j]):.4f}  |beta| {abs(beta[j]):.4f}")
dipole = np.asarray(mp.l) == 1
power = abs(alpha) ** 2 + abs(beta) ** 2
fraction = power[dipole].sum() / power.sum()
print(f"dipole fraction of the partial scattering cross sections: {fraction:.5f}")
```

```text
l=1 m=-1: |alpha| 1.0056  |beta| 0.9837
l=1 m=+0: |alpha| 0.0012  |beta| 0.0057
l=1 m=+1: |alpha| 1.0038  |beta| 0.9825
dipole fraction of the partial scattering cross sections: 0.99943
```

A plane wave along $z$ polarized along $x$ only drives $m = \pm1$ about the
$z$ axis; the $m = 0$ coefficients are the mesh's error. The coefficients are
per the {term}`incident amplitude` `mp.incident_amplitude`, a complex number:
a plane wave's amplitude with its phase at the model origin, not the real
$|E_0|$ of `FarField.incident_amplitude`.

## Kerker and Huygens

The exact dipoles are in SI units: `mp.electric_dipole` $p$ in C·m and
`mp.magnetic_dipole` $m$ in A·m², each of shape `(frequencies, 3)`, along
the model's $x$, $y$, $z$. The first {term}`Kerker condition` for a wave along
$z$ is $p_x = m_y/v$, with $v$ the speed of light in the background. Where it
holds the dipoles scatter nothing backward:

```python
p = np.asarray(mp.electric_dipole)[0]
m = np.asarray(mp.magnetic_dipole)[0]
print(f"p_x = {p[0]:.4e} C m, m_y / v = {m[1] / c0:.4e} C m")
print(f"(m_y / v) / p_x = {m[1] / c0 / p[0]:.4f}")

angles = np.array([0.0, np.pi])
ff = gfem.far_field(result, "z", angles, np.zeros(2), model=model)
forward, backward = np.asarray(ff.intensity)[0]
print(f"backward / forward intensity = {backward / forward:.2e}")
```

```text
p_x = 1.4971e-30-5.2638e-31j C m, m_y / v = 1.4739e-30-4.8862e-31j C m
(m_y / v) / p_x = 0.9784+0.0176j
backward / forward intensity = 2.34e-04
```

The two dipoles are equal to a few percent, and the backward intensity is
four orders of magnitude under the forward one. What is left is the
quadrupoles', which the condition does not cancel. The far field is that of
{py:obj}`gfem.far_field`: the decomposition and the pattern read the same
solved field.

## Anapoles: two calls

An {term}`anapole` is a scatterer whose exact electric dipole vanishes, $p = 0$,
because a long-wavelength electric dipole and a {term}`toroidal dipole`
cancel. {py:obj}`gfem.current_moments` reads them from the {term}`induced
current` of the same result, with the same `origin`:

```python
cm = gfem.current_moments(result, "z", model=model)
p0 = np.asarray(cm.long_wavelength_dipole)[0]
t = np.asarray(cm.toroidal_dipole)[0]
ikt = 1j * k * t
norm = np.linalg.norm
print(f"|p| {norm(p):.4e}, |p0| {norm(p0):.4e}, |ikT| {norm(ikt):.4e} C m")
print(f"|p - (p0 - ikT)| / |p| = {norm(p - (p0 - ikt)) / norm(p):.4f}")
```

```text
|p| 1.5869e-30, |p0| 1.6560e-30, |ikT| 7.0144e-32 C m
|p - (p0 - ikT)| / |p| = 0.0007
```

Two calls are needed because they read different things:
`gfem.multipoles` gives the exact $p$, which carries the whole expansion, and
`gfem.current_moments` the {term}`long-wavelength dipole` $p_0$ and the
toroidal dipole $T$ that explain it, $p \approx p_0 - ikT$ under gfem's
$e^{+i\omega t}$. The sphere here is far from an anapole, so $T$ is small; at
an anapole the two terms cancel. Read an anapole from the exact $p$, which
goes through zero. The long-wavelength statement $p_0 = ikT$ is only a rough
guide to where, a few percent off.

## Check against the cross sections

The partial cross sections add up to the scattering cross section, which
{py:obj}`gfem.cross_sections` measures another way, as a power balance on the
truncation layer:

```python
xs = gfem.cross_sections(result, "z", model=model)
mp_sca = float(mp.scattering[0]) * um2
xs_sca = float(xs.scattering[0]) * um2
print(f"multipoles {mp_sca:.4f} um^2, cross_sections {xs_sca:.4f} um^2")
print(f"gap {100 * (mp_sca / xs_sca - 1):.2f} %")
```

```text
multipoles 0.3737 um^2, cross_sections 0.3739 um^2
gap -0.06 %
```

They are two discretizations of one power, so they agree to the mesh, not to
round-off: the multipoles sum a far-field projection, the cross section a
flux of the residual.

## Origin and degree cap

The partial cross sections depend on the {term}`multipole origin`, because
the degrees are about it; their sum does not. Move it off the scatterer's
centre, or give its centre when it is not at the model origin: with `origin=`
in m, in model coordinates. The degree cap is `lmax=`: the default is
Wiscombe's count for the scatterer's size about the origin. Raise it to check
the tail:

```python
moved = gfem.multipoles(result, "z", model=model, origin=(0.1e-6, 0.0, 0.0))
more = gfem.multipoles(result, "z", model=model, lmax=10)
ed_moved = float(moved.electric_scattering[0, 0]) * um2
md_moved = float(moved.magnetic_scattering[0, 0]) * um2
print(f"origin moved 0.1 um: ED {ed_moved:.4f}, MD {md_moved:.4f} um^2")
print(
    f"sum {float(moved.scattering[0]) * um2:.4f} um^2, "
    f"{float(mp.scattering[0]) * um2:.4f} um^2 at the model origin"
)
print(
    f"lmax = 10: sum {float(more.scattering[0]) * um2:.4f} um^2, "
    f"lmax = {mp.lmax}: {float(mp.scattering[0]) * um2:.4f} um^2"
)
```

```text
origin moved 0.1 um: ED 0.1914, MD 0.1752 um^2
sum 0.3737 um^2, 0.3737 um^2 at the model origin
lmax = 10: sum 0.3737 um^2, lmax = 7: 0.3737 um^2
```

The solve's order caps what is read: at order 2 the coefficients below about
$10^{-3}$ of the norm are hidden by the discretization, so do not read the
high degrees of a coarse solve. `lmax` cannot exceed what a source inside the
far-field layer radiates; the error says so.

## Beams and incident-field functions

A {py:obj}`gfem.GaussianBeam` is decomposed with its focal field as the
{term}`incident amplitude`; a beam wide against the scatterer gives the plane
wave's coefficients. An {py:obj}`gfem.IncidentField` needs `intensity=`, in
W/m², as for {py:obj}`gfem.cross_sections`: the amplitude is then
$\sqrt{2\eta I}$, and the partial cross sections are over that intensity.
The exact moments are in SI units from the field itself, so the Cartesian
moments do not depend on it.

## As an objective

The multipoles differentiate with respect to a {py:obj}`gfem.MaterialField`
in the scatterer, inside {py:obj}`jax.grad`. Call `gfem.multipoles` outside
{py:obj}`jax.jit`. Maximize the magnetic dipole by its absolute partial
cross section in m², for instance:

```py
def objective(density):
    model = build_model(density)  # a MaterialField in the scatterer
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=2))
    mp = gfem.multipoles(result, "z", model=model)
    return -mp.magnetic_scattering[0, 0]


gradient = jax.grad(objective)(density)
```

- Use square magnitudes, a partial cross section or $|p|^2$, never the
  absolute value of a moment: at an exact anapole the gradient of $|p|$ is
  undefined.
- The absolute cross section is the objective for a strong magnetic dipole.
  The fraction $\sigma^M_1/\sigma_\text{sca}$ is the variant for purity, but
  maximizing it alone favours weak scatterers.
- Node coordinates, the origin and `intensity=` are not differentiable.
  {py:obj}`gfem.current_moments` is not differentiable at all: apply
  `jax.lax.stop_gradient` to its inputs to use it inside a function that is.

## Across ranks

On a result of a {term}`distributed solve`, `gfem.multipoles` and
`gfem.current_moments` are collective calls: call them on every rank with
the same arguments, and every rank gets the same numbers, one process's to
round-off ({doc}`distributed-solve`).

## Limits

- **Incident-field columns only**: a port or a dipole column is refused (read
  its field with {py:obj}`gfem.far_field`), and the solve must keep the
  fields with their incident part.
- **A homogeneous lossless background**: the background is one lossless
  medium, not a stack and not lossy, and the whole scatterer is meshed: a
  ground plane, a mirror half or a slab model, with a PEC or PMC wall beside
  the physical region, is refused.
- **3D, a truncation layer and the degree cap**: a 2D mesh is refused; the
  scatterer stays clear of the layer of {py:obj}`gfem.cross_sections` and of
  wave ports; `lmax` is at least 2 and at most what the far-field layer
  supports.
- **Volume currents only**: `gfem.current_moments` takes the induced current
  of a permittivity contrast, with its conduction current. A perfect
  conductor, a sheet, a lumped port or a magnetic contrast carry surface or
  magnetization currents and are refused; `gfem.multipoles` reads those
  exactly.
- **Not differentiable**: `gfem.multipoles` is not differentiable by node
  coordinates, the origin or the intensity, and `gfem.current_moments` not
  differentiable at all.

## Next

- {doc}`cross-sections`: the power balance the multipoles' sum is checked
  against.
- {doc}`far-field`: the pattern and the radar cross section of the same
  solved field.
- {doc}`incident-fields`: the sources a decomposition reads.
- `gfem.conventions`: the basis, the normalization and the maps to the moments
  and to papers written in $e^{-i\omega t}$.
