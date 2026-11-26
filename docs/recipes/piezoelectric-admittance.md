# Transducer admittance

Find the admittance of electrodes on a piezoelectric crystal, such as an
{term}`interdigital transducer`, per metre of aperture, over a band of
frequencies: the current into a terminal per volt, the
capacitance of the electrodes with the crystal held still, the displacement
the voltage launches, and the reflection of the finished device on a circuit.
{py:class}`gfem.Piezoelectric` solves a 2D section of the
{doc}`tutorial <../tutorials/transducer>`'s device, meshed by
{download}`transducer_section.py <../tutorials/transducer_section.py>`.

## A crystal with a stiffness and a piezoelectric tensor

A material takes the {term}`stiffness` $c^E$ (6 × 6, Pa), the
{term}`piezoelectric tensor` $e$ (3 × 6, C/m²) and the
{term}`clamped permittivity` in `eps_r`, in the frame of its `axes`
({doc}`../conventions`, "Piezoelectric problems").

```python
import urllib.request

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
c_si = np.zeros((6, 6))  # cubic: c11, c12, c44
c_si[:3, :3] = 63.9e9
c_si += np.diag([165.7e9 - 63.9e9] * 3 + [79.6e9] * 3)

# x-cut: crystal y along the section's x (the wave), crystal x up, crystal z along the fingers, reversed
axes = ((0, 1, 0), (1, 0, 0), (0, 0, -1))
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
aluminium = gfem.Material(
    stiffness=gfem.isotropic_stiffness(70e9, 0.35), mass_density=2700.0, **q
)
print(c_si[0, :3] / 1e9)
```


```text
[165.7  63.9  63.9]
```


- `axes` must be right-handed: a left-handed set flips the coupling without an error.
- A `stiffness` without a `piezoelectric_tensor` is an elastic solid. A region
  of the mechanical domain without a stiffness and a density raises
  {py:obj}`gfem.MissingMaterialPropertyError`, naming it.
- The loss is the stiffness times $1 + i/Q_m$ under $e^{+i\omega t}$, the
  opposite sign to the permittivity's. {py:obj}`gfem.isotropic_stiffness` builds
  an isotropic solid.

## Two domains: the mechanical and the electrical

The displacement lives on the {term}`mechanical domain` (every region not in
`excluded["mechanical"]`), the potential on every region not in `excluded["em"]`.
The silicon handle has three models:

- **No potential in it** (`excluded={"em": ["si"]}`), as here: the lightest,
  and the tutorial's choice.
- **A dielectric** (an `eps_r`): the tutorial's admittance moves by 2.3 %.
- **Grounded** (`ground=`): a metal plane, for a conducting handle.

```python
section = transducer_section(outer=2, between=3, edge_scale=2.0)
model = gfem.Model(
    section.mesh,
    materials={
        "ln": ln, "ln_frame": ln, "box": oxide, "box_frame": oxide,
        "si": silicon, "si_frame": silicon,
        "signal": aluminium, "ground": aluminium, "air": gfem.Material(),
    },
    excluded={"mechanical": ["air"], "em": ["si", "si_frame"]},
)  # fmt: skip
print(section.fingers, len(section.mesh.nodes))
```


```text
(3, 2) 4267
```


## Electrodes: meshed fingers or lines

A terminal names an area or a line:

- A {term}`meshed electrode` is an area of metal with stiffness and mass and
  the potential fixed on it. It loads the crystal, so it shifts and reflects
  the wave.
- A {term}`line electrode` has the potential fixed on a line and no mass: light,
  and what the closed forms assume.

A grating's stop band is 1.97 % wide with meshed 50 nm aluminium and 0.57 %
with lines: mesh real fingers whenever reflection or resonance frequency
matters. A meshed electrode must stay in the electromagnetic domain.

## Clamped lines and absorbing edges

Every edge of the mechanical domain is traction-free unless named in
`clamped=`, a list of lines where $u = 0$, each a {term}`clamped surface`.

A {py:class}`gfem.DampingFrame` names regions that absorb outgoing waves, each a {term}`damping frame` with a
{term}`loss factor <Loss factor (of a damping frame)>`: the stiffness there is
$c (1 + i(1/Q_m + \eta r^2))$ with $r$ the normalised depth. Give a frame a region
for each layer on each side, with the layer's material (`"ln_frame"` is the
same lithium niobate as `"ln"`), meshed as rectangles ending on one straight line.

A frame of thickness $L$ reflects a normal plane wave of wavelength $\lambda$
by $|R| \approx e^{-2\pi\eta (L/\lambda)/3}$, down to a floor of about
$3.4 \times 10^{-3}\,\eta\,(\lambda/L)^2$. The default $\eta = 1.2$ is best from
four wavelengths ($|R| < 10^{-3}$); three take $\eta = 2$. The helper's frames
are four wavelengths wide.

## The result

{py:obj}`gfem.solve` returns a {py:class}`gfem.PiezoelectricResult`. Its
`admittance_per_length` is the {term}`admittance per length` $Y'_{ij}(f)$ in
S/m, shaped (frequencies, terminals, terminals): the current into terminal $i$
per volt on terminal $j$, per metre along $z$. `clamped_capacitance_per_length`
is the {term}`clamped capacitance` in F/m, with $\varepsilon^S$ and no
displacement; the free capacitance is larger by $1 + K^2$. A device of
{term}`aperture` $W$ has $Y = W Y'$.

```python
frame = gfem.DampingFrame(["ln_frame", "box_frame", "si_frame"])  # loss factor 1.2
problem = gfem.Piezoelectric(
    np.linspace(0.84e9, 0.88e9, 3),
    terminals={"signal": "signal"},
    ground="ground",
    clamped="bottom",
    damping_frame=frame,
    keep_fields=True,
)
res = gfem.solve(model, problem)
y = np.asarray(res.admittance_per_length)  # (frequencies, terminals, terminals), S/m
print(res)
print(y.shape, bool((y[:, 0, 0].real > 0).all()))
```


```text
PiezoelectricResult(terminals=['signal'], 3 frequencies 840-880 MHz, clamped capacitance per length in pF/m:
  signal:      225.2)
(3, 1, 1) True
```


The real part of the diagonal, the {term}`radiation conductance` when the waves
leave through the frames, is positive; the matrix is symmetric with a positive
semidefinite real part. With several terminals, `terminals={"a": "signal_a",
"b": "signal_b"}` gives one row and column each. `res.to_xarray()` returns a
labelled `xarray.Dataset`.

## Network parameters

{py:meth}`gfem.PiezoelectricResult.network_parameters` takes the aperture, a
{term}`series resistance` $R_s$ in each terminal's lead and a reference
impedance, and returns the scattering parameters; the formulas are in
{doc}`../multiphysics-couplings`. For fingers fed from one end,
$R_s = \tfrac{R_f}{3} (1/N_S + 1/N_G)$ with $R_f = \rho W / (w t)$.

```python
aperture = 400e-6  # m
net = res.network_parameters(aperture, series_resistance=26.7)
print(np.round(np.asarray(net.db("signal", "signal")), 3))
net.to_touchstone("transducer.s1p")
```


```text
[-0.061 -0.062 -0.065]
```


`net.db("signal", "signal")` is $|S_{11}|$ in dB; `to_touchstone` writes the
file a circuit simulator reads. It differentiates in the aperture, the
resistance and the reference impedance.

## Fields and strain

`keep_fields=True` keeps the displacement and potential at every frequency and
terminal (mind the memory on a long sweep). `res.displacement(f)` is a map of
`"x"`, `"y"` and `"z"`, in m per volt; `res.potential(f)` is in V per volt. Both
read with `.at(points)`.

{py:func}`gfem.strain` reads the {term}`engineering strain` at points, per volt.
The shear wave of the section is polarized along $z$ and carries
$S_{xz} = \partial_x u_z$, the fifth entry:

```python
f = float(res.frequency[1])
s = np.asarray(gfem.strain(res.displacement(f), section.arms))  # (arms, 6), 1/V
print(s.shape)
print(f"|S_xz| at the two arms: {abs(s[0, 4]):.3e} and {abs(s[1, 4]):.3e} per volt")
print(
    f"potential in the air above: {abs(complex(res.potential(f).at([3e-5, 0.1e-6]))):.3f} V per volt"
)
```


```text
(2, 6)
|S_xz| at the two arms: 1.077e-05 and 1.077e-05 per volt
potential in the air above: 0.834 V per volt
```


## Limits

- The problem is 2D: a tetrahedral or a slab mesh raises (`Mesh.from_gmsh` of a
  file of triangles).
- `linear_solver="petsc"`, `frequency_workers` above 1 and `comm` raise: one
  process and the sparse direct solver.
- A `gfem.Absorbing` line on a solid's edge raises: use a damping frame.
- An area terminal outside the electromagnetic domain raises, as do a frame
  region name that is not in the mesh and a loss factor traced by JAX.

## Next

- {doc}`../tutorials/transducer`: this section at full size and the pitch whose
  {term}`synchronous frequency` is a chosen one.
- {doc}`piezoelectric-gradient`: the gradients of the admittance.
- {doc}`2d-models`: what a 2D model is and what it refuses.
- {doc}`../conventions`, "Piezoelectric problems": the signs and units.
