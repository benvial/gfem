# Floquet ports

Light a periodic structure (a grating, a metasurface, a frequency-selective
surface, an antireflection stack) with a plane wave, and read how much power
leaves in each diffraction order. You mesh one cell, pair its side faces with
{py:obj}`gfem.Periodic` ({doc}`periodic-cell`), and close each open face (a
{term}`Floquet surface`) with ports ({py:obj}`gfem.FloquetPort`). A
{term}`Floquet port` is one {term}`diffraction order` in one polarization:
its row and column of S are that order's, and $|S|^2$ is its
{term}`diffraction efficiency`. The ports of a
face absorb every order at any angle, so the cell needs no absorber or PML.

The {doc}`metagrating tutorial <../tutorials/metagrating>` designs a grating
this way. This page covers the other tasks: oblique incidence, a fixed angle
across a band, TM and circular polarization, a fed cell, and the checks of a
Floquet model. Efficiencies grouped by order, the power balance, Jones and
Mueller matrices and angle sweeps have helpers in `gfem.diffraction`
({doc}`diffraction-efficiency`).

## A slab under a plane wave

A glass slab ($n = 1.5$, 0.3 µm thick) in air, lit from above. The mesh is one
cell, 0.4 µm × 0.4 µm, with the volumes `"above"`, `"slab"` and `"below"`, the
open faces `"top"` and `"bottom"` 0.25 µm from the slab, and the side faces
`"x_min"`, `"x_max"`, `"y_min"`, `"y_max"` meshed as translates of each other.
Each open face takes a TE and a TM port of the specular order (0, 0). An
`offset` of 0.25 µm moves each port's reference plane onto the slab surface:

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

`s["R TE", "R TE"]` is the slab's reflection of a TE wave and `s["T TE", "R TE"]`
its transmission, both referred to the slab surfaces. A slab has a closed form,
the Airy sum of the two interfaces' tangential-E coefficients
({py:obj}`gfem.conventions`, Floquet ports), here at the angle `theta` in the
top medium:

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

- Both differences should be small against |r| and |t|, and shrink as you
  refine the mesh or raise `order`. How gfem itself is checked against this
  form, at normal, oblique and Brewster incidence: {doc}`../verification`.
- At normal incidence a flat interface reflects $(n_1 - n_2) / (n_1 + n_2)$ in
  both polarizations: the tangential-E sign. The Fresnel $r_p$ of many optics
  texts has the opposite sign.
- Without `offset`, S is referred to the Floquet surfaces: $S_{11}$ picks up the
  round trip $\exp(-2i\beta d)$ through the air between face and slab.

## Name the orders

A grating diffracts into several orders. Order $(m, n)$ is the plane wave whose
in-plane wave vector is $\boldsymbol{\kappa} = \mathbf{k}_t + m \mathbf{b}_1 + n \mathbf{b}_2$, with $\mathbf{b}_1$ and $\mathbf{b}_2$ the reciprocal
vectors of the face's two pairs ($\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi \delta_{ij}$, $\mathbf{a}_1$ and $\mathbf{a}_2$ the pairs'
translations in `boundaries` order) and $\mathbf{k}_t$ the incident wave's.
{py:meth}`gfem.Model.propagating_orders` lists the orders that propagate in
the medium behind a face, before you declare any port:

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

- `gfem.diffraction.floquet_ports(model, face, f)` declares this list for one
  face: a TE and a TM port of every propagating order, named `"top (0, 0) TE"`
  and so on.
- The sign of $m$ follows $\mathbf{a}_1$: order +1 leans along $+\mathbf{b}_1$. Swapping the faces of
  the first pair flips $\mathbf{a}_1$, and the same diffracted wave becomes order −1.
- `result.diffraction_orders[name]` describes each port's order at every
  frequency: its in-plane `wave_vector` $\boldsymbol{\kappa}$, `gamma` ($i\beta$ where it propagates),
  `propagating`, and the angles `theta` from the face normal and `phi` from
  $\hat{\mathbf{e}}_1$ ({py:obj}`gfem.DiffractionOrderResult`).
- Port names are yours. Several Floquet ports share one face, each with its
  own (order, polarization), as the modes of a
  {doc}`shared port surface <shared-port-surface>` do; the face takes no
  other port or boundary.
- `excite=` solves only the columns you need: one incident wave is one
  column of S. Without it, every declared port is excited.
- A propagating order with no port on its face is still absorbed, not
  reflected, but its power is missing from S. gfem warns with
  {py:obj}`gfem.UnrequestedOrderWarning`, naming the orders: declare them to
  read them, or filter the warning by class when you do not need them.
- Evanescent orders need no port: gfem keeps those that reach the face with
  more than `SolverOptions.floquet_order_tolerance` (1e-4) of their
  amplitude. When the face mesh is too coarse to carry them, it warns with
  {py:obj}`gfem.FloquetOrderCapWarning` and names a distance to move the face
  to; move it away from the structure, or refine the face.

## Oblique incidence and a fixed angle across a band

The incident wave is set by the problem's `wave_vector`, the specular order's
in-plane wave vector ({doc}`periodic-cell`). {py:meth}`gfem.Model.bloch_wave_vector`
builds it from angles in radians: `theta` from the face's inward normal,
`phi` from ê1 toward ê2, in the medium behind the face:

```python
k = slab.bloch_wave_vector("top", f, theta=np.radians(30.0), phi=0.0)  # rad/m
tilted = gfem.Driven(f, excite=["R TE", "R TM"], wave_vector=k)
oblique = gfem.solve(slab, tilted, options=options).s
```

A fixed `wave_vector` holds at every frequency of a sweep, so the angle
changes with frequency. To keep the angle, give the frequency at which the
wave vector holds: the solve at f then uses k f / f_ref, which keeps the angle
in every port medium at once.

On the grating, more orders open as the frequency rises at a fixed angle, so
list the orders to declare at the highest frequency's wave vector:

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

- Each frequency of a fixed-angle sweep builds its own Bloch space and
  factorization, so it costs more per frequency than a sweep at one k.
- Only the uniform sweep takes a wave vector, fixed angle included.
- To solve several angles, pass them to `gfem.diffraction.angle_sweep`
  ({doc}`diffraction-efficiency`).

## Polarization

TE (s) and TM (p) are relative to each order's own plane of incidence: TE
along $\hat{\mathbf{e}}_3 \times \hat{\boldsymbol{\kappa}}$, normal to it, and TM along $\hat{\boldsymbol{\kappa}}$, in it. At normal incidence
$\hat{\boldsymbol{\kappa}}$ is $\hat{\mathbf{e}}_1$, the direction of the first pair's translation: on a cell paired
along x then y, TE is E along +y and TM is E along +x. gfem has no circular
port: circular light is a combination of the TE and TM columns of one order.
`gfem.diffraction.jones(...).circular` and `polarization="RCP"` in
`gfem.diffraction.efficiencies` do it for you ({doc}`diffraction-efficiency`);
by hand: excite both ports of the incident order, then, for incidence from the top
face (the face whose outward normal is $\hat{\mathbf{e}}_3$, incidence along $-\hat{\mathbf{e}}_3$), IEEE
right-handed circular incidence is $(S_{:,\mathrm{TM}} + i S_{:,\mathrm{TE}}) / \sqrt{2}$:

```python
both = gfem.Driven(f, excite=["R+0,+0 TE", "R+0,+0 TM"])
pair = gfem.solve(lit, both, options=options).s
te, tm = pair.excited.index("R+0,+0 TE"), pair.excited.index("R+0,+0 TM")
rhcp = (pair.values[:, :, tm] + 1j * pair.values[:, :, te]) / np.sqrt(2)
```

`rhcp[:, i]` is row i's response over frequency. From the bottom face,
incidence along +ê3, the same combination is left-handed, and `tm` minus
`1j * te` is right-handed. The convention is in
{py:obj}`gfem.conventions`.

## Differentiate an efficiency

S of a Floquet model differentiates with respect to the materials and nodes
inside the cell, as any S does ({doc}`s-parameter-gradient`). The efficiency
of the transmitted −1 order, per element of the ridge:

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

- Build objectives from $|S|^2$, $|S|$ or $\operatorname{Re} S$. The row of an order that does not
  propagate reads exactly 0, so $10 \log_{10} |S|^2$ there is $-\infty$ with a NaN
  gradient: take dB after optimizing, or use $\log_{10}(|S|^2 + \epsilon)$.
- A pixel design on a periodic cell takes
  {py:obj}`gfem.design.LayerGrid` with `periodic=(True, False)` (one bool per
  grid axis, (x, y)), whose pitch must divide the period along a periodic
  axis: the filter smooths across the seam and the element map wraps around
  it. Mesh the design region on the pixel grid, or finer: a mesh whose
  elements straddle pixel edges samples another design. The
  {doc}`metagrating tutorial <../tutorials/metagrating>` does both.
- A design that does not vary along the grating vector diffracts nothing,
  so its efficiency in a non-specular order and the gradient of it are 0:
  start from a design that already varies.
- When you check a gradient by finite differences, keep every propagating
  order away from grazing. Near a {term}`Rayleigh anomaly` $|S|$ varies as
  $\sqrt{|\gamma|}$, and a finite-difference step swamps the derivative.

## A fed cell

A phased-array element: a feed inside the cell, a Floquet port on its open
face, the scan angle set by `wave_vector`. S between the feed and the Floquet
ports gives the active reflection and the efficiency radiated into each order
at that scan angle:

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

- The reference impedance of a Floquet row is the order's wave impedance,
  $\eta / \cos\theta$ for TE and $\eta \cos\theta$ for TM, in the medium behind the face, and
  NaN where the order does not propagate (`scan.z_ref`). `scan.z()` then
  gives the oblique transmission-line model: for a sheet between two
  Floquet faces, with both reference planes on the sheet, $Z_{21}$ is its sheet
  impedance.
- A Touchstone file has one real reference impedance. S writes as it is
  only at normal incidence with every port in one medium; otherwise
  `to_touchstone(path, z0=50.0)` renormalizes first, which terminates the
  Floquet channels in 50 Ω instead of free space and changes every entry,
  the active reflection included. `scan.to_xarray()` keeps S as solved.
  A port whose order does not propagate at some frequency has no reference
  impedance there, and the file is refused: drop the port or narrow the
  grid.

## Check your results

- **Power balance.** For a lossless cell, the efficiencies of every
  propagating order on both faces, in both polarizations, add up to 1 in
  each excited column. The defect counts the orders you did not declare
  (and the warning names them), the mesh and any loss together.
  `gfem.diffraction.power_balance` adds them up, and refuses a cell where a
  propagating order or polarization has no port.
- **Reciprocity with the order labels.** A reversed wave has the opposite
  in-plane wave vector: order $(m, n)$ maps to $(-m, -n)$, and the TE and TM
  references flip sign with it. With $\sigma = -1$ for an order with $\boldsymbol{\kappa} \neq 0$ and $+1$
  for $\boldsymbol{\kappa} = 0$, $S_{p\leftarrow q}(\mathbf{k}) = \sigma_p \sigma_q S_{-q\leftarrow -p}(-\mathbf{k})$, where $-p$ is port $p$'s face
  and polarization with the opposite order. Solve at $-\mathbf{k}$ with the mirrored
  ports and compare. The plain transpose $S(\mathbf{k})^\mathsf{T} = S(-\mathbf{k})$ does not hold for
  Floquet ports. The {doc}`metagrating tutorial <../tutorials/metagrating>`
  runs this check at normal incidence.
- **Zero wave vector.** `wave_vector=(0, 0, 0)` gives the same S as `None`;
  run `None`, which is cheaper.
- **Moving a face.** Moving an open face away from the structure, with
  `offset` grown by the same distance, leaves S unchanged: the offset
  removes the phase of the extra path.

How gfem itself is checked with Floquet ports: {doc}`../verification`.

## Limits

- **The half-space.** The medium behind a Floquet face is one region,
  homogeneous, isotropic and lossless, with a constant permittivity and
  permeability; anything else raises `NotImplementedError` when you build
  the model. Put a lossy or dispersive layer inside the cell, with the port
  in a lossless medium beyond it. Its material takes no gradient
  ({py:obj}`gfem.NonDifferentiableError`).
- **Two pairs.** The face's rim must lie on the seams of two pairs whose
  translations span it. A cell with one pair and side walls raises: model a
  1D grating as a thin cell paired along the lines too, as the tutorial
  does.
- **Which problems.** Only the uniform driven sweep, on the `direct` or
  `bddc_gmres` solver, frequency workers included. The adaptive sweep,
  `linear_solver="petsc"`, the GPU solver, MPI ranks, mesh refinement,
  error and quantity estimates and eigenmode solves raise. For the
  resonances and Q of a radiating cell, sweep a driven problem and read them
  off S.
- **Exciting an order.** Only an order that propagates at every frequency of
  the sweep can be excited; otherwise `ValueError` names the ports that can.
  The row of an order that does not propagate reads exactly 0, and its
  `z_ref` is NaN.
- **Gradients.** None with respect to the angle, `wave_vector`,
  `wave_vector_frequency`, the half-space's material, the nodes of a
  Floquet face or `offset` ({py:obj}`gfem.NonDifferentiableError` or
  {py:obj}`gfem.PortGeometryError`).
- **Ports.** A {py:obj}`gfem.WavePort` still may not touch a paired face.
  No circular-polarization port: circular entries are a change of basis of S
  ({doc}`diffraction-efficiency`).

## Next

{doc}`diffraction-efficiency`, {doc}`emissivity`,
{doc}`metasurface-from-cells`, {doc}`../tutorials/metagrating`, {doc}`../tutorials/metalens`,
{doc}`periodic-cell`,
{py:obj}`gfem.FloquetPort`, {py:obj}`gfem.conventions`,
{doc}`s-parameter-gradient`, {doc}`../capabilities`.
