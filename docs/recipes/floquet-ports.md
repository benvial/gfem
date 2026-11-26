# Floquet ports

Light a periodic structure (a grating, a metasurface, a frequency-selective
surface) with a plane wave, and read how much power leaves in each diffraction
order. Pair a cell's side faces with {py:obj}`gfem.Periodic`
({doc}`periodic-cell`) and close each open face, a {term}`Floquet surface`,
with {py:obj}`gfem.FloquetPort` ports. A {term}`Floquet port` is one
{term}`diffraction order` in one polarization; $|S|^2$ is its
{term}`diffraction efficiency`. The
{doc}`metagrating tutorial <../tutorials/metagrating>` designs a grating this way.

## A slab under a plane wave

A glass slab ($n = 1.5$, 0.3 µm thick) in air, in a 0.4 µm cell with open faces
`"top"` and `"bottom"` 0.25 µm away. Each takes a TE and a TM port of the
specular order; `offset` moves the reference plane onto the slab:

```python
import numpy as np

import gfem

c = 299_792_458.0  # m/s
mesh = gfem.Mesh.from_gmsh("slab_cell.msh", length_unit="um")
air = gfem.Material()
pairs = [gfem.Periodic(("x_min", "x_max")), gfem.Periodic(("y_min", "y_max"))]
ports = [
    gfem.FloquetPort(f"{side} {pol}", face, polarization=pol, offset=0.25e-6)
    for side, face in (("R", "top"), ("T", "bottom"))
    for pol in ("TE", "TM")
]
slab = gfem.Model(
    mesh,
    materials={"above": air, "slab": gfem.Material(eps_r=1.5**2), "below": air},
    boundaries=pairs,
    ports=ports,
)
f = c / 1.0e-6  # Hz, a wavelength of 1 um
options = gfem.SolverOptions(order=3, linear_solver="direct")
s = gfem.solve(slab, gfem.Driven(f, excite=["R TE", "R TM"]), options=options).s
```

`s["R TE", "R TE"]` is the reflection of a TE wave and `s["T TE", "R TE"]` its
transmission. The closed form is the Airy sum of the two interfaces'
tangential-E coefficients ({py:obj}`gfem.conventions`, Floquet ports), at the
angle `theta` in the top medium:

```python
def airy(n1, n2, n3, thickness, frequency, theta, polarization):
    """Tangential-E reflection and transmission of a slab n2 between n1 (top) and n3."""
    k0 = 2 * np.pi * frequency / c
    kt = k0 * n1 * np.sin(theta)

    def impedance(n):  # the wave impedance over eta0, TE eta / cos, TM eta cos
        cos = np.sqrt(complex(1 - (kt / (k0 * n)) ** 2))
        return 1 / (n * cos) if polarization == "TE" else cos / n

    z1, z2, z3 = impedance(n1), impedance(n2), impedance(n3)
    r12, r23 = (z2 - z1) / (z2 + z1), (z3 - z2) / (z3 + z2)
    delay = np.exp(-1j * np.sqrt(complex((k0 * n2) ** 2 - kt**2)) * thickness)
    r = (r12 + r23 * delay**2) / (1 + r12 * r23 * delay**2)
    t = (1 + r12) * (1 + r23) * delay / (1 + r12 * r23 * delay**2)
    return r, t * np.sqrt(z1 / z3)  # S11, S21 as gfem defines them


for pol in ("TE", "TM"):
    r, t = airy(1.0, 1.5, 1.0, 0.3e-6, f, 0.0, pol)
    print(pol, abs(s[f"R {pol}", f"R {pol}"][0] - r))  # reflection error
    print(pol, abs(s[f"T {pol}", f"R {pol}"][0] - t))  # transmission error
```

- Both differences shrink with refinement or `order`; how gfem is checked:
  {doc}`../verification`.
- At normal incidence a flat interface reflects $(n_1 - n_2) / (n_1 + n_2)$ in
  both polarizations (tangential-E sign, opposite to the Fresnel $r_p$).

## Name the orders

Order $(m, n)$ has in-plane wave vector
$\boldsymbol{\kappa} = \mathbf{k}_t + m \mathbf{b}_1 + n \mathbf{b}_2$, with
$\mathbf{b}_i$ the reciprocal vectors of the face's pairs.
{py:meth}`gfem.Model.propagating_orders` lists the orders behind a face before
you declare any port:

```python
grating = gfem.Model(
    gfem.Mesh.from_gmsh("grating_cell.msh", length_unit="um"),
    materials={
        "substrate": gfem.Material(eps_r=2.25),
        "ridge": gfem.Material(eps_r=4.0),
        "air": air,
    },
    boundaries=pairs,
)
for face in ("top", "bottom"):
    print(face, grating.propagating_orders(face, f))  # ((0, 0), (-1, 0), ...)

ports = [
    gfem.FloquetPort(f"{side}{m:+d},{n:+d} {pol}", face, order=(m, n), polarization=pol)
    for side, face in (("R", "top"), ("T", "bottom"))
    for m, n in grating.propagating_orders(face, f)
    for pol in ("TE", "TM")
]
lit = grating.replace(ports=ports)
result = gfem.solve(lit, gfem.Driven(f, excite="R+0,+0 TE"), options=options)
incident = "R+0,+0 TE"
efficiency = {name: abs(result.s[name, incident][0]) ** 2 for name in result.s.ports}
```

- `gfem.diffraction.floquet_ports(model, face, f)` declares a TE and a TM port
  of every propagating order, named `"top (0, 0) TE"`.
- Order +1 leans along $+\mathbf{b}_1$; swapping the first pair's faces flips it.
- `result.diffraction_orders[name]` gives each port's `wave_vector`, `gamma`,
  `propagating` and angles ({py:obj}`gfem.DiffractionOrderResult`).
- Several Floquet ports share one face, as on a
  {doc}`shared port surface <shared-port-surface>`. `excite=` solves only the
  columns you need.
- A propagating order with no port is absorbed but missing from S:
  {py:obj}`gfem.UnrequestedOrderWarning` names it.
- Evanescent orders need no port: gfem keeps those reaching the face with more
  than `SolverOptions.floquet_order_tolerance` (1e-4) of their amplitude;
  {py:obj}`gfem.FloquetOrderCapWarning` says how far to move the face.

## Oblique incidence and a fixed angle across a band

The problem's `wave_vector` is the specular order's in-plane wave vector.
{py:meth}`gfem.Model.bloch_wave_vector` builds it from `theta` (from the face's
inward normal) and `phi` (from ê1 toward ê2):

```python
k = slab.bloch_wave_vector("top", f, theta=np.radians(30.0), phi=0.0)  # rad/m
tilted = gfem.Driven(f, excite=["R TE", "R TM"], wave_vector=k)
oblique = gfem.solve(slab, tilted, options=options).s
```

A fixed `wave_vector` holds at every frequency, so the angle changes. Give the
frequency at which it holds and the solve at f uses k f / f_ref, which keeps the
angle. More orders open as the frequency rises, so list them at the highest
frequency:

```python
band = np.linspace(0.9, 1.1, 5) * f
k_ref = grating.bloch_wave_vector("top", f, theta=np.radians(30.0))
k_max = np.asarray(k_ref) * band.max() / f  # rad/m, at the highest frequency
ports = [
    gfem.FloquetPort(f"{side}{m:+d},{n:+d} {pol}", face, order=(m, n), polarization=pol)
    for side, face in (("R", "top"), ("T", "bottom"))
    for m, n in grating.propagating_orders(face, band.max(), wave_vector=k_max)
    for pol in ("TE", "TM")
]
fixed = gfem.Driven(
    band, excite="R+0,+0 TE", wave_vector=k_ref, wave_vector_frequency=f
)
sweep = gfem.solve(grating.replace(ports=ports), fixed, options=options)
```

- Only the uniform sweep takes a wave vector. For several angles use
  `gfem.diffraction.angle_sweep` ({doc}`diffraction-efficiency`).

## Polarization

TE (s) is along $\hat{\mathbf{e}}_3 \times \hat{\boldsymbol{\kappa}}$, TM (p)
along $\hat{\boldsymbol{\kappa}}$. There is no circular port:
`gfem.diffraction.jones(...).circular` and `polarization="RCP"` combine the
columns ({doc}`diffraction-efficiency`). By hand, excite both ports of the
incident order; from the top face IEEE right-handed circular incidence is
$(S_{:,\mathrm{TM}} + i S_{:,\mathrm{TE}}) / \sqrt{2}$:

```python
both = gfem.Driven(f, excite=["R+0,+0 TE", "R+0,+0 TM"])
pair = gfem.solve(lit, both, options=options).s
te, tm = pair.excited.index("R+0,+0 TE"), pair.excited.index("R+0,+0 TM")
rhcp = (pair.values[:, :, tm] + 1j * pair.values[:, :, te]) / np.sqrt(2)
```

`rhcp[:, i]` is row i's response over frequency. From the bottom face the same
combination is left-handed ({py:obj}`gfem.conventions`).

## Differentiate an efficiency

S of a Floquet model differentiates with respect to the materials and nodes
inside the cell ({doc}`s-parameter-gradient`). The efficiency of the
transmitted −1 order, per element of the ridge:

```python
import jax
import jax.numpy as jnp

minus_one = "T-1,+0 TE"
problem = gfem.Driven(f, excite="R+0,+0 TE")


def efficiency_of(eps):
    model = lit.with_material("ridge", gfem.MaterialField(eps_r=eps))
    s = gfem.solve(model, problem, options=options).s
    return jnp.abs(s[minus_one, "R+0,+0 TE"][0]) ** 2


n = len(lit.mesh.elements("ridge"))
grad = jax.grad(efficiency_of)(jnp.full(n, 4.0))
```

- Build objectives from $|S|^2$, $|S|$ or $\operatorname{Re} S$; the row of an
  evanescent order is exactly 0, so take dB after optimizing.
- A pixel design takes {py:obj}`gfem.design.LayerGrid` with
  `periodic=(True, False)`, whose pitch divides the period
  ({doc}`metagrating tutorial <../tutorials/metagrating>`).
- For finite differences keep orders away from a {term}`Rayleigh anomaly`,
  where $|S|$ varies as $\sqrt{|\gamma|}$.

## A fed cell

A phased-array element: a feed inside the cell, a Floquet port on its open face,
the scan angle set by `wave_vector`. S gives the active reflection and the
efficiency radiated into each order:

```python
element = gfem.Model(
    gfem.Mesh.from_gmsh("patch_cell.msh", length_unit="mm"),
    materials={"substrate": gfem.Material(eps_r=3.0), "air": air},
    boundaries=[*pairs, gfem.PEC("ground"), gfem.PEC("patch")],
    ports=[
        gfem.LumpedPort("feed", "probe", "+z", impedance=50.0),
        gfem.FloquetPort("space TE", "top", polarization="TE"),
        gfem.FloquetPort("space TM", "top", polarization="TM"),
    ],
)
f_scan = 10e9
k_scan = element.bloch_wave_vector("top", f_scan, theta=np.radians(20.0))
scan = gfem.solve(element, gfem.Driven(f_scan, wave_vector=k_scan)).s
active_reflection = scan["feed", "feed"][0]
```

- A Floquet row's reference impedance is the order's wave impedance,
  $\eta / \cos\theta$ for TE and $\eta \cos\theta$ for TM, NaN where it does
  not propagate (`scan.z_ref`).
- `to_touchstone(path, z0=50.0)` renormalizes to one real impedance, terminating
  the Floquet channels in 50 Ω; `scan.to_xarray()` keeps S as solved.

## Check your results

- **Power balance.** For a lossless cell the efficiencies of every propagating
  order on both faces add up to 1 in each excited column;
  `gfem.diffraction.power_balance` refuses an undeclared order.
- **Reciprocity.** A reversed wave takes order $(m, n)$ to $(-m, -n)$. With
  $\sigma = -1$ for an order with $\boldsymbol{\kappa} \neq 0$ and $+1$ otherwise,
  $S_{p\leftarrow q}(\mathbf{k}) = \sigma_p \sigma_q S_{-q\leftarrow -p}(-\mathbf{k})$,
  with $-p$ the port on the same face with the opposite order; the plain
  transpose does not hold. The {doc}`metagrating tutorial
  <../tutorials/metagrating>` runs this at normal incidence.
- **Moving a face.** Moving an open face with `offset` grown by the same
  distance leaves S unchanged.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **The half-space.** The medium behind a Floquet face is homogeneous, isotropic
  and lossless, or `NotImplementedError`; put lossy layers inside the cell.
- **Two pairs.** Model a 1D grating as a thin cell paired along the lines too.
- **Which problems.** Only the uniform driven sweep, on `direct` or
  `bddc_gmres`; the adaptive sweep, PETSc, GPU, MPI, refinement and eigenmode
  solves raise.
- **Gradients.** None with respect to the angle, `wave_vector`, the half-space's
  material, a Floquet face's nodes or `offset`
  ({py:obj}`gfem.NonDifferentiableError`).
- **Ports.** A {py:obj}`gfem.WavePort` may not touch a paired face.

## Next

{doc}`diffraction-efficiency`, {doc}`emissivity`, {doc}`metasurface-from-cells`,
{doc}`../tutorials/metagrating`, {doc}`../tutorials/metalens`,
{doc}`periodic-cell`, {py:obj}`gfem.FloquetPort`, {py:obj}`gfem.conventions`,
{doc}`s-parameter-gradient`, {doc}`../capabilities`.
