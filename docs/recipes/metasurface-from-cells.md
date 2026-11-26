# Metasurface from cell responses

Solve each cell of a metasurface alone, as if in an infinite array of itself (the
{term}`locally periodic approximation`), and tabulate it against the parameter
that varies across the device: a {term}`cell response`, what the literature calls
a unit-cell or phase library. Place a cell at each position (an
{term}`aperture field`), propagate to a focal plane and differentiate. Cells have
Floquet ports ({doc}`floquet-ports`, {doc}`diffraction-efficiency`).

## Tabulate the cell

A row of ridges lit in TE at normal incidence, the width $w$ the parameter.
`make_cell(p, dp)` returns the cell at `p` with its mesh moved by `dp`, a JAX
scalar that is 0 in the table and carries the derivative, through
{py:obj}`gfem.shape.boundary_morph` of the walls:

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

The **width map** moves the cell by one unit of `p` per unit of `dp`: $\tfrac12$
for a ridge's walls; for a pillar $\mathbf{u}\cdot\hat{\mathbf{n}}$, $\sqrt2/2$
on a vertical edge and $\tfrac12$ elsewhere.

```python
# the pillar's map, from the morph's own nodes and normals
rel = np.asarray(morph.points)[:, :2] - centre  # m, centre = the pillar's axis
on_wall = np.isclose(np.abs(rel), w0 / 2 * um, atol=1e-9)
u = 0.5 * np.sign(rel) * on_wall
width_map = jnp.asarray(np.einsum("ij,ij->i", u, np.asarray(morph.normals)[:, :2]))
```

**Check the range.** The morph squeezes the element beside a wall (a ridge's wall
moved 1.3 elements before quality fell by a quarter, a pillar's 0.6 to 0.8).
Measure it with {py:obj}`gfem.shape.quality`:

```python
widths = np.linspace(0.07, 0.23, 21)  # um
q0 = gfem.shape.quality(base.mesh)
for w in (widths[0], widths[-1]):
    q = gfem.shape.quality(make_cell(w, 0.0).mesh)
    print(w, float(jnp.min(q / q0)))  # gfem warns below a quarter
```

Past the range, build a fresh mesh per sample in `make_cell`, cached per `p`.
Then tabulate, exciting the polarization you will use in the specular order:

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

- `out=` writes the response after every sample, so rerunning resumes an
  interrupted build. `response.to_json(path)` and
  `dif.CellResponse.from_json(path)` write and read one built another way.
- The pitch must let only the specular order propagate.
- Hermite interpolation falls as the spacing to the fourth power: 13° of phase
  per interval gave $4\times10^{-5}$ in $S$, 27° gave $6\times10^{-4}$. Space
  samples by phase, not width.
- Each sample costs one solve and 8 backward passes per excited column.

## Evaluate and differentiate the table

`response(p)` is the {py:obj}`gfem.NetworkParameters` at `p`: pure JAX, so it
differentiates and runs under `jax.jit` and `jax.vmap`.

```python
S = response(0.14)  # a width between two samples
S.values.shape  # (n_freq, n_ports, n_excited)
dS = jax.jacfwd(lambda p: response(p).values)(0.14)
```

A `p` outside the range raises `ValueError` (NaN if traced). Map an unbounded
optimizer variable onto the range once, in the objective:

```python
lo, hi = float(widths[0]), float(widths[-1])


def width_of(x):
    return lo + (hi - lo) * jax.nn.sigmoid(x)
```

## Place the cells and propagate the field

`aperture_field(response, parameter, outgoing, incident, polarization=...)` gives
at each cell's centre its Jones entry times the incident polarization and an
optional envelope; `propagate` carries it by the angular spectrum behind the
device:

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

- `parameter` has shape `(n1,)` or `(n1, n2)` with `indexing="ij"`; a pitch that
  lets another order propagate is refused. Values are relative to the incident
  wave.
- `amplitude=` is a per-cell envelope; `amplitude=0` cells extend the window.
- Polarization is `"TE"`, `"TM"`, `"RCP"`, `"LCP"` or a Jones pair among the
  excited columns; `"unpolarized"` is refused: average TE and TM intensities.
- Zero-padding to `padding` times the aperture (default 2) puts a lens's focal
  spot within $1.4\times10^{-4}$ of the Rayleigh-Sommerfeld peak up to numerical
  aperture 0.8; the Fresnel formula is wrong by 7 to 43 % above 0.5.
- `distance` and `padding` are fixed (`NonDifferentiableError`).
- A pitch above $\lambda/2$ aliases; read a focus above numerical aperture 0.5
  by its peak.

## Objectives

An objective is a few lines of JAX over the propagated field; the one helper is
`ApertureField.intensity`, $|E_1|^2 + |E_2|^2$. The focal intensity is the entry
of `intensity` at the focal sample, and one `jax.value_and_grad` gives its
gradient over the whole map:

```python
focal_length = 100e-6  # m


def focal_intensity(x):
    return field_at(x, focal_length).intensity[:, n // 2]  # (n_freq,)


def loss(x):
    return -jnp.sum(focal_intensity(x))


value, grad = jax.jit(jax.value_and_grad(loss))(x_cells)
```

The Strehl ratio divides it by that of an ideal lens, built with the
`ApertureField` constructor, with phase $k(\sqrt{x^2 + F^2} - F)$:

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

For the deflection efficiency, a wave leaving at $\theta$ carries $\cos\theta$
(TE) or $1/\cos\theta$ (TM) times the power its tangential amplitude says, so a
perfect ramp reads that factor, not 1. Sum over the Fourier bins near the
target, over the incident power; for a 2D aperture use `jnp.fft.fft2`:

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

## When the approximation holds

The approximation neglects coupling to neighbours, which grows with the phase
step. A full-wave supercell against the aperture field of the same widths
(a ridge of permittivity 5, height 0.6 µm on glass, pitch 0.5 µm, 1 µm, TE):

| widths of the cells, in µm | largest phase step between neighbours | largest amplitude error over the orders | strongest order's efficiency, solved / approximated |
|---|---|---|---|
| 0.34, 0.32, 0.26, 0.18, 0.16, 0.18, 0.26, 0.32 | 40° | 0.014 | 0.679 / 0.684 |
| a ramp of 4 widths, 0.06 to 0.44 | 65°, one reset of 186° | 0.092 | 0.652 / 0.570 |

A uniform map is exact. Up to steps of about 40° every order is within 0.01 to 0.02; at 100° and above, and
at the $2\pi$ reset of a ramp, it fails. A steady 36° step per cell (ten silicon
nitride ridges, pitch 0.3 µm, at 633 nm) sends 0.870 into the +1 order in the
full-wave solve and 0.897 in the aperture field. An optimized map can use steps
beyond the approximation: penalize neighbours' transmission difference.

## Reflection and arrays

For a reflectarray name the same face for both, `outgoing == incident`. A 2000 × 2000
aperture takes about 1.5 s per value and gradient.

## Check your results

- A uniform map gives the cell's own transmission, exactly.
- One width between samples against `response(p)`: the error shrinks by about 16
  when the spacing halves.
- A supercell solved with `gfem.diffraction.floquet_ports` matches the Fourier
  transform of the aperture field.
- `jax.grad` against central differences.

## Limits

- **One parameter** at normal incidence.
- **No neighbour coupling**: resets and sharp steps are beyond the table above.
- **Co-design.** The gradient is with respect to the parameter map and envelope,
  not the meta-atom's shape ({doc}`shape-gradient`).

## Next

{doc}`diffraction-efficiency`, {doc}`floquet-ports`, {doc}`emissivity`,
{doc}`../tutorials/metalens`, {doc}`shape-gradient`, {doc}`periodic-cell`,
{py:obj}`gfem.shape.quality`, {py:obj}`gfem.conventions`, {doc}`../capabilities`.
