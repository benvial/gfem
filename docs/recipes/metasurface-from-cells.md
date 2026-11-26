# Metasurface from cell responses

A metasurface is a large array of small cells whose shape changes slowly across
the aperture: a lens, a beam deflector, a reflectarray. Solving it whole is out
of reach at a millimetre, but each cell can be solved alone as if it sat in an
infinite array of itself (the {term}`locally periodic approximation`). This
page tabulates one cell against the one number that varies across the device
(a {term}`cell response`, what the literature calls a unit-cell or phase
library), places a cell of the table at every position (an
{term}`aperture field`), propagates that field to a focal plane, and
differentiates a figure of merit with respect to the whole parameter map. The
helpers live in `gfem.diffraction`; the cell is a periodic cell with Floquet
ports as in {doc}`floquet-ports` and {doc}`diffraction-efficiency`.

The problem is a one-axis aperture, a row of ridges along $\hat{\mathbf{e}}_1$
with the lines along $\hat{\mathbf{e}}_2$, lit at normal incidence from a
substrate, in TE (the field along the lines). The ridge width $w$ is the
parameter, and a 2D aperture of pillars differs only in the width map and the
shape of the parameter array.

## Tabulate the cell

`cell_response(make_cell, parameter, ...)` solves the cell at each sample of
the parameter and keeps S together with its derivative with respect to the
parameter. `make_cell(p, dp)` returns the cell at the plain number `p`, its
mesh moved by `dp`, a JAX scalar that is 0 when the table is built and carries
the derivative. The usual way to move a ridge is one mesh and a
{py:obj}`gfem.shape.boundary_morph` of its side walls:

```python
import jax
import jax.numpy as jnp
import numpy as np

import gfem
from gfem import diffraction as dif

c = 299_792_458.0  # m/s
um = 1e-6
w0 = 0.15  # um, the width the mesh was built at
d_sub, d_air = 0.2, 0.25  # um, substrate and air thickness of the mesh
freqs = c / (np.array([0.532, 0.633]) * um)  # Hz

mesh = gfem.Mesh.from_gmsh("ridge_cell.msh", length_unit="um")
base = gfem.Model(
    mesh,
    materials={
        "substrate": gfem.materials.silica(),
        "ridge": gfem.materials.silicon_nitride(),
        "air": gfem.Material(),
    },
    boundaries=[
        gfem.Periodic(("x_min", "x_max")),
        gfem.Periodic(("y_min", "y_max")),
    ],
)
# the phase of S is read on the planes where the ports sit: one on the
# substrate's surface, one on the ridge's top
ports = [
    *dif.floquet_ports(base, "bottom", freqs, offset=d_sub * um),
    *dif.floquet_ports(base, "top", freqs, offset=d_air * um),
]
base = base.replace(ports=ports)

morph = gfem.shape.boundary_morph(base, "sidewall", grows="ridge")
width_map = 0.5 * jnp.ones(morph.n_params)  # each wall moves half the widening


def make_cell(p, dp):
    return base.replace(mesh=morph(((p - w0) + dp) * um * width_map))
```

`dp` should move the cell by one unit of `p` per unit of `dp`, so that the stored
derivative is per unit of the parameter. That is what the **width map** does:

- A ridge's two walls each move by half the widening, so the map is $\tfrac12$
  on every node of the moving surface.
- A pillar's walls are four. The displacement that widens it by one moves each
  wall plane by $\tfrac12$ outward, and a node on a vertical edge moves
  diagonally, along $(\tfrac12, \tfrac12)$ toward the corner. The morph moves a
  node along its own normal, so the map is the component of that displacement
  along the normal, $\mathbf{u}\cdot\hat{\mathbf{n}}$: $\sqrt2/2$ on a vertical
  edge and $\tfrac12$ elsewhere on the walls. A constant $\tfrac12$ would
  under-move the edges and tilt the walls.

```python
# the pillar's map, from the morph's own nodes and normals
rel = np.asarray(morph.points)[:, :2] - centre  # m, centre = the pillar's axis
on_wall = np.isclose(np.abs(rel), w0 / 2 * um, atol=1e-9)
u = 0.5 * np.sign(rel) * on_wall
width_map = jnp.asarray(np.einsum("ij,ij->i", u, np.asarray(morph.normals)[:, :2]))
```

**Check the range.** One mesh moved by a morph is smooth in the parameter, which
is why the stored derivative interpolates so well, but the morph is bounded:
the element beside a wall gets squeezed as the wall approaches its neighbour.
How far it goes depends on the cell (on one structured mesh a ridge's wall moved
1.3 elements before quality fell by a quarter, a pillar's 0.6 to 0.8), so
measure it over your range with {py:obj}`gfem.shape.quality` instead of
trusting a rule:

```python
widths = np.linspace(0.07, 0.23, 21)  # um
q0 = gfem.shape.quality(base.mesh)
for w in (widths[0], widths[-1]):
    q = gfem.shape.quality(make_cell(w, 0.0).mesh)
    print(w, float(jnp.min(q / q0)))  # gfem warns below a quarter
```

Past the range, build a mesh for each sample instead: `make_cell(p, dp)` builds
the cell at `p` (a fresh mesh, with the same topology and the same names) and
then moves it by `dp`. The table is then as smooth as the meshes are
converged: two meshes of the same cell differ by their discretization error,
which Hermite interpolation cannot remove. Converge the mesh to the accuracy
you want from the table.

`cell_response` calls `make_cell` once per sample to read the cell's signature
and once more per frequency for each solve, so $n_f + 1$ times per sample.
When meshing is the slow part, cache it per `p` (`functools.cache` on a
function that returns the mesh) and let `make_cell` wrap the cached mesh.

Then tabulate. Excite the polarization you will light the device with, in the
specular order of the incident face, and declare both polarizations of the
outgoing order (the ports above do):

```python
response = dif.cell_response(
    make_cell,
    widths,
    frequency=freqs,
    excite=["bottom (0, 0) TE"],
    options=gfem.SolverOptions(order=2, linear_solver="direct"),
    out="ridge_response.json",
)
```

- A port is named after its surface, `"bottom (0, 0) TE"` here, as in
  {doc}`diffraction-efficiency`.
- `out=` writes the response after every sample. Rerun with the same `out`,
  samples, frequencies and excited ports and the finished samples are kept, so
  an interrupted build resumes. `response.to_json(path)` writes a response you
  built another way and `dif.CellResponse.from_json(path)` reads it back; floats
  round-trip bitwise, and the arrays sit in an `.npz` file beside the JSON.
- The parameter may move the meta-atom only. A sample that changes the lattice,
  the media, the ports or the diffraction orders is refused.
- A response is one parameter at one incidence: normal incidence, plain
  periodicity. Keep the pitch small enough that only the specular order
  propagates behind the cell, or the aperture field below refuses it.

**Sample spacing.** The response interpolates S between samples by piecewise
cubic Hermite polynomials on the real and imaginary parts, with the derivative
you stored, so the interpolation error falls as the fourth power of the
spacing: halving it divides the error by about 16. With the phase of the
transmission moving 13° per interval the error in $S$ was $4\times10^{-5}$ and
with 27° per interval $6\times10^{-4}$. Space samples by the phase, not the
width: a resonant cell turns its phase quickly over a narrow range of widths,
and that range needs the dense samples. Check by solving one or two widths
between samples and comparing with `response(p)`.

**Cost.** For each sample and each frequency gfem solves the cell once, and then
runs one backward pass for each real entry of the S rows it keeps and each
excited column: with four Floquet ports (a TE and a TM port on each face), 8
passes per excited column. A table of $n_p$ samples, $n_f$ frequencies and
$n_c$ columns is therefore $n_p n_f$ forward solves and $8\,n_p n_f n_c$
backward passes, which dominate the forward solves. The solve cache is
bypassed while a table is built, because the derivative needs the factorization
of each sample.

## Evaluate and differentiate the table

`response(p)` is the {py:obj}`gfem.NetworkParameters` at the parameter value `p`,
the type of `result.s`, for every frequency of the table. It is pure JAX: it
differentiates in `p`, in the stored arrays and under `jax.jit` and `jax.vmap`,
and the table needs no solver once it exists.

```python
S = response(0.14)  # a width between two samples
S.values.shape  # (n_freq, n_ports, n_excited)
dS = jax.jacfwd(lambda p: response(p).values)(0.14)
```

The response does not extrapolate. A concrete `p` outside
`[widths[0], widths[-1]]` raises `ValueError`; a traced one, inside `jax.jit` or
`jax.vmap`, where the check cannot run, returns NaN for the out-of-range entries
and their gradients, so an objective that left the range reads NaN instead of a
silent clamp. An optimizer works with an unbounded variable, so map it onto the
sampled range, once, in the objective:

```python
lo, hi = float(widths[0]), float(widths[-1])


def width_of(x):
    return lo + (hi - lo) * jax.nn.sigmoid(x)
```

## Place the cells and propagate the field

`aperture_field(response, parameter, outgoing, incident, polarization=...)`
samples the device once per cell. At the centre of cell $j$ the field is the
cell's Jones entry at its parameter times the incident polarization and an
optional envelope: it is one plane wave per cell, as if the cell sat in an
infinite array of itself. `propagate` carries the sampled field by the
angular spectrum to a plane at a distance in the half-space behind the device:

```python
n = 1001  # cells along e1
x_cells = jnp.zeros(n)  # the design variable, one number per cell


def field_at(x, distance):
    parameter = width_of(x)
    field = dif.aperture_field(response, parameter, "top", "bottom", polarization="TE")
    return dif.propagate(field, distance)


out = field_at(x_cells, 100e-6)
out.values.shape  # (n_freq, 2, n): the components along e1 (TM) and e2 (TE)
out.intensity.shape  # (n_freq, n): relative to the incident plane wave
```

What to know about the two calls:

- **Positions and units.** A row of $n_1$ cells sits at
  $x_j = (j - \tfrac{n_1-1}{2})\,a_1$ from the aperture centre, with the pitch
  $a_1$ read from the response. With an even count the axis falls between two
  samples; an odd count puts a sample on it. The values are in power-wave
  units relative to the incident plane wave, so $|E|^2$ of a wave that leaves
  normally is the intensity relative to the incident one. The components are
  those along $\hat{\mathbf{e}}_1$ (TM) and $\hat{\mathbf{e}}_2$ (TE); at normal
  incidence they are the tangential field and each propagates on its own.
- **Axes.** `parameter` has shape `(n1,)` (uniform along $\hat{\mathbf{e}}_2$:
  deflectors, cylindrical lenses) or `(n1, n2)`, with `indexing="ij"`. At least two
  cells along each axis. The aperture assumes normal incidence, a rectangular
  lattice (the vectors at a right angle) and cells that send one plane wave
  behind them: it refuses a response whose pitch lets another order propagate
  in the outgoing half-space, naming the order and the wavelength it needs.
- **Envelope and window.** `amplitude=` multiplies the incident field of each
  cell by a complex number (Gaussian illumination, an aperture stop), with the
  shape of `parameter`. The propagated field lives on the same grid, so to
  look beyond the device, make the map larger than the device and give the extra
  cells `amplitude=0`.
- **Polarization.** `"TE"`, `"TM"`, `"RCP"`, `"LCP"` or a Jones pair, a fixed
  value that must be among the excited columns. `"unpolarized"` is refused,
  being an incoherent mean: propagate TE and TM, and average their intensities.
- **Propagation.** The angular spectrum multiplies each plane wave by
  $e^{-ik_zd}$, with $k_z = \sqrt{k^2 - q^2}$, and by a decaying exponential
  where $q > k$. The field is zero-padded to `padding` times the aperture's width
  (default 2) and the steepest components that would wrap around that window are
  dropped. Against the Rayleigh-Sommerfeld integral of the same samples, padding
  2 puts the focal spot of a lens within $1.4\times10^{-4}$ of the peak
  amplitude and the field over the whole footprint within $1.1\times10^{-3}$, from
  numerical aperture 0.1 to 0.8; padding 3 gives $4\times10^{-5}$ at the focal
  spot. The Fresnel formula is not an alternative: it is wrong by 7 to 43 % in
  amplitude at numerical aperture 0.5 to 0.8.
- **Distance.** `distance` and `padding` are fixed numbers (`NonDifferentiableError`
  otherwise). To scan an axis, call `propagate` for each distance. There is
  no gradient with respect to the distance.
- **No finer grid.** The result is on the cells' grid, one sample per pitch.
  A sample is the field at a cell centre, so the samples carry plane waves up to
  $|q| \le \pi/a_1$, and a pitch above $\lambda/2$ aliases the directions beyond
  $\arcsin(\lambda/2a_1)$. A focal spot at numerical aperture above about 0.5
  with a pitch near $\lambda/2$ is coarsely sampled: read its peak, not its
  profile.

## Objectives

An objective is a few lines of JAX over the propagated field. The only helper is
`ApertureField.intensity`, $|E_1|^2 + |E_2|^2$.

**Focal intensity.** The intensity at the focus, one number per frequency, is
the entry of `intensity` at the focal sample. The gradient with respect to the
whole map comes from one `jax.value_and_grad`:

```python
focal_length = 100e-6  # m


def focal_intensity(x):
    return field_at(x, focal_length).intensity[:, n // 2]  # (n_freq,)


def loss(x):
    return -jnp.sum(focal_intensity(x))


value, grad = jax.jit(jax.value_and_grad(loss))(x_cells)
```

**Strehl ratio.** The focal intensity of the design over that of an ideal lens
with the same amplitude, whose field you build with the `ApertureField`
constructor. A converging lens has the phase $k(\sqrt{x^2 + F^2} - F)$ for a
field that propagates as $e^{-ik_zd}$:

```python
field = field_at(x_cells, 0.0)  # grid, frequencies and index of the device
k = 2 * np.pi * field.index * field.frequency / c  # (n_freq,)
phase = k[:, None] * (np.sqrt(field.x**2 + focal_length**2) - focal_length)
t_ideal = jnp.exp(1j * phase)
ideal = dif.ApertureField(
    jnp.stack([jnp.zeros_like(t_ideal), t_ideal], axis=1),  # TE only
    field.frequency,
    field.x,
    index=field.index,
)
ideal_focus = dif.propagate(ideal, focal_length).intensity[:, n // 2]
strehl = focal_intensity(x_cells) / ideal_focus
```

A Strehl ratio near 1 says the map reproduces the lens. The ideal lens has
unit amplitude, so the ratio also counts the transmission of the cells.

**Deflection efficiency.** The fraction of the incident power that leaves in a
direction. The power of a plane wave of tangential amplitude $E_2$ (TE) is
$|E_2|^2\cos\theta$ and of $E_1$ (TM) is $|E_1|^2/\cos\theta$, with $\theta$ its
angle to the normal: the amplitudes are those of the tangential field, and the
direction changes how much of it carries power through the plane. The Fourier
transform of the aperture field splits it into plane waves, so the efficiency is
the sum of those flux factors over the directions near the target, over the
incident power, $\sum|u_j|^2$ for unit incident amplitude, which the Fourier
transform of $M$ points returns divided by $M$:

```python
def deflection_efficiency(field, theta, half_width, pad=8):
    """Power fraction leaving within sin(angle) = sin(theta) +- half_width.

    The angle is measured from the normal, positive toward +e1, for a
    one-axis aperture; the incident power is the sum of |u|^2 over the cells.
    """
    n = field.values.shape[-1]
    m = pad * n
    pitch = field.x[1] - field.x[0]
    q = 2 * np.pi * np.fft.fftfreq(m, pitch)  # rad/m, of e^{+iqx}
    k = 2 * np.pi * field.index * field.frequency / c  # (n_freq,)
    s = -q[None, :] / k[:, None]  # sin of the angle; e^{+iqx} travels toward -x
    cos = np.sqrt(np.clip(1.0 - s**2, 0.0, None))
    near = (np.abs(s - np.sin(theta)) <= half_width) & (cos > 0)
    te = cos
    tm = np.where(cos > 0, 1.0 / np.maximum(cos, 1e-12), 0.0)
    spectrum = jnp.fft.fft(field.values, n=m, axis=-1)  # (n_freq, 2, m)
    flux = jnp.abs(spectrum[:, 0]) ** 2 * tm + jnp.abs(spectrum[:, 1]) ** 2 * te
    return jnp.sum(flux * near, axis=-1) / (m * n)  # incident power: n cells
```

The result is one efficiency per frequency, and it differentiates. Choose
`half_width` as the width of the main lobe of the aperture, about
$\lambda/(N a_1)$ in $\sin\theta$ for $N$ cells. A beam that leaves at
$\theta$ carries $\cos\theta$ (TE) or $1/\cos\theta$ (TM) times the power its
tangential amplitude says, so a perfect phase ramp reads that factor, not 1,
before the loss from sampling the ramp once per cell.

For a 2D aperture, use `jnp.fft.fft2` and the in-plane wave vector
$\mathbf{q}$ of each bin; the flux factors keep their form, with $\theta$ the
angle to the normal.

## When the approximation holds

The locally periodic approximation assumes each cell radiates as if all its
neighbours were itself, and it neglects the coupling to the neighbours as they
actually are. That coupling grows with the phase step between neighbouring
cells. The table compares a full-wave solve of one supercell, with its cells
meshed as the single cell, against the aperture field of the same widths. It is
a ridge of relative permittivity 5 and height 0.6 µm on glass (2.1), pitch
0.5 µm, wavelength 1 µm, normal incidence from the glass, TE. The error is the
largest difference over the transmitted orders in the amplitude of S, and
the last column compares the strongest order's efficiency:

| widths of the cells, in µm | largest phase step between neighbours | largest amplitude error over the orders | strongest order's efficiency, solved / approximated |
|---|---|---|---|
| 0.24, 0.24, 0.24, 0.24 | 0° | round-off | 0.9201 / 0.9201 |
| 0.34, 0.32, 0.26, 0.18, 0.16, 0.18, 0.26, 0.32 | 40° | 0.014 | 0.679 / 0.684 (order 0) |
| a ramp of 8 widths, 0.06 to 0.44 | 35°, one reset of 186° | 0.051 | 0.503 / 0.483 (order +1) |
| a ramp of 4 widths, 0.06 to 0.44 | 65°, one reset of 186° | 0.092 | 0.652 / 0.570 (order +1) |
| six widths in no order | 157° | 0.116 | 0.125 / 0.221 (order 0) |
| 0.06, 0.44, 0.06, 0.44 | 186° | 0.071 | 0.0144 / 0.0072 (order 0) |

- A uniform map is exact: it is the approximation's own closed form.
- Up to neighbour steps of about 40° the amplitude of every order is within 0.01
  to 0.02. A 65° ramp with one reset is off by 0.1 (a deflected order 13 % low); at 100° and
  above the approximation fails, as it does at the $2\pi$ reset of a deflector's
  phase ramp, whose coupling is a known source of error.
- A deflector with a steady phase step of 36° per cell (ten silicon nitride
  ridges of pitch 0.3 µm covering $2\pi$ at 633 nm, whose widths follow the
  table to within 10° of phase) sends 0.870 of the power into the +1 order in
  the full-wave solve and 0.897 in the aperture field.
- The range of phase a cell covers limits a deflector: the ridge above spans
  186° as its width changes, so a ramp of $2\pi$ needs a reset or a different
  cell.
- A map optimized through the response can use steps the approximation does not
  cover: abrupt steps between neighbours, 3 to 4 % of the neighbour pairs above
  60° in a probe lens designed this way. A penalty on the transmission difference of
  neighbours is a natural term in the objective, which this page does not add.

## Memory and time

The aperture field and the propagation are plain JAX of the parameter map, so
the value and the gradient of an objective cost a few Fourier transforms. On
two threads, in complex double precision, a value and gradient after the first
(compiling) call took:

| cells | padding | time | memory |
|---|---|---|---|
| $500 \times 500$ | 2 | 0.14 s | 0.41 GiB |
| $1000 \times 1000$ | 2 | 0.63 s | 0.70 GiB |
| $2000 \times 2000$ (1 mm at a pitch of 500 nm) | 2 | 1.3 to 1.5 s | 1.9 to 2.7 GiB |
| $2000 \times 2000$ | 3 | 1.8 s | 3.4 GiB |

The memory is for one component and for two, respectively; the first call adds
the compile time (2 to 13 s). These are observations on a loaded machine. Build
the cell response once and reuse it for every design.

## Reflection and arrays

For a reflectarray, name the same face for both: `outgoing == incident`. The
cells' reflection at normal incidence is the entry used, and the field then
propagates back into the incident half-space. Everything above applies, with the
distance measured from the aperture into that half-space.

A periodic array of identical elements, a frequency-selective surface or an
array with feeds in its cells, does not need a table: it is one periodic cell
solved at the angle or the frequency you want, as in {doc}`floquet-ports` (for
the fed cell, its "A fed cell" section) and {doc}`diffraction-efficiency`.

## Check your results

- **A uniform map** gives a uniform field equal to the cell's own transmission,
  which is exact in the approximation.
- **A table against a solve.** Solve the cell at one width between samples and
  compare with `response(p)`. The error should shrink by about 16 when the
  spacing halves.
- **The approximation.** For the design you will build, solve a supercell of
  its cells with `gfem.diffraction.floquet_ports` and compare its orders with the
  Fourier transform of the aperture field, as in the table above, before
  trusting a figure of merit to the third digit.
- **The gradient.** Compare `jax.grad` of your objective with central finite
  differences along a random direction of the map.
- **The propagation.** For a lens, the intensity should peak where the lens
  focuses, and `padding=3` should leave the focal intensity within the figures
  above.

## Limits

- **One parameter.** A response has one parameter, one polarization state at the
  excited columns, and one incidence (normal). A design that varies two things
  from cell to cell, or rotates the cell, needs a response for each.
- **Normal incidence.** The aperture field assumes the light arrives normal to a
  rectangular lattice; a response sampled at an oblique incidence is refused.
- **No neighbour coupling.** See the table above. A sharp transition between
  neighbours, a reset, or a resonance that couples cells is beyond it.
- **Cells that do not diffract.** A pitch at which a non-specular order
  propagates behind the cells is refused.
- **Scalar tangential components.** The field is the two tangential components
  at fixed distances; the normal component and a polarization-resolved far
  field are not computed.
- **Observation grid.** The cell grid is the observation grid (use `amplitude=0`
  for a wider window).
- **No distance gradient.** The distance and padding are fixed.
- **Co-design.** The gradient is with respect to the parameter map and the
  envelope, not to the meta-atom's own shape or materials: a solve differentiates
  once, and the stored derivative of S would need a second derivative of the
  solve. To optimize the meta-atom itself, differentiate a solve of the cell
  ({doc}`shape-gradient`). Between samples, the derivative with respect to the
  parameter is the interpolant's, not the solve's; they agree at the samples.

## Next

{doc}`diffraction-efficiency`, {doc}`floquet-ports`, {doc}`emissivity`,
{doc}`../tutorials/metalens`, {doc}`shape-gradient`, {doc}`periodic-cell`,
{py:obj}`gfem.shape.quality`,
{py:obj}`gfem.conventions`, {doc}`../capabilities`.
