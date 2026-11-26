# Transducer admittance

Find the admittance of electrodes on a piezoelectric crystal, such as an
{term}`interdigital transducer`, per metre of aperture, over a band of
frequencies: the current into a terminal per volt, the
capacitance of the electrodes with the crystal held still, the displacement
the voltage launches, and the reflection of the finished device on a circuit.
{py:class}`gfem.Piezoelectric` solves a 2D section of piezoelectric, elastic
and dielectric regions, with the fields held uniform along $z$, so a finger
pattern is one cross-section.

The device is the transducer of the {doc}`tutorial
<../tutorials/transducer>`: aluminium fingers at a 2.2 µm pitch on a 360 nm
film of x-cut lithium niobate over oxide and a silicon handle. The tutorial's
{download}`transducer_section.py <../tutorials/transducer_section.py>` meshes
it with netgen; here with four fingers at twice the tutorial's edge lengths,
which takes a few seconds a frequency.

## A crystal with a stiffness and a piezoelectric tensor

A material's `stiffness` is the {term}`stiffness` $c^E$, a 6 × 6 matrix in Pa,
and its `piezoelectric_tensor` the {term}`piezoelectric tensor` $e$, a 3 × 6
matrix in C/m², both in Voigt order $xx, yy, zz, yz, xz, xy$ with
{term}`engineering strain`'s shear. They are the constants of
$T = c^E S - e^\mathsf{T} E$ and $D = e S + \varepsilon^S E$: the stiffness at
constant field, $c^E$ and not $c^D$, and the stress constants $e$, not the
strain constants $d$. The `eps_r` is the {term}`clamped permittivity`
$\varepsilon^S / \varepsilon_0$. All three are in the frame of the material's
`axes`, whose rows are the crystal's $x$, $y$ and $z$ in the section's
coordinates (a right-handed set), as for the electro-optic tensor
({doc}`../conventions`, "Piezoelectric problems"). The
{term}`mechanical quality factor` `mechanical_quality_factor` $Q_m$ lumps the
solid's loss.

An isotropic solid takes {py:obj}`gfem.isotropic_stiffness` from its Young's
modulus and Poisson's ratio. A cubic one is a 6 × 6 you write from $c_{11}$,
$c_{12}$ and $c_{44}$.

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

- `axes` must be right-handed: the tensors' signs follow the crystal's frame,
  and a left-handed set flips the coupling without an error.
- A material with a `stiffness` but no `piezoelectric_tensor` is an elastic
  solid; one with a `piezoelectric_tensor` needs a `stiffness`. Every region of
  the mechanical domain (next section) needs a stiffness and a density, or
  {py:obj}`gfem.solve` raises {py:obj}`gfem.MissingMaterialPropertyError` and
  names the region.
- The loss is the stiffness times $1 + i/Q_m$ under $e^{+i\omega t}$: the sign
  of the imaginary part is the opposite of the permittivity's
  ({doc}`../conventions`).

## Two domains: the mechanical and the electrical

The displacement lives on the {term}`mechanical domain`: every region not
listed in `excluded["mechanical"]`. The potential lives on the electromagnetic
domain: every region not listed in `excluded["em"]`. Air has no mechanics, so
it is left out of the first. A silicon handle that is a thick, lossy,
slightly conducting substrate is left out of the second, so it carries no
potential. Three models of the handle are possible:

- **No potential in it** (`excluded={"em": ["si"]}`), as here: the handle is
  an elastic solid. This is the lightest, and the tutorial's choice.
- **A dielectric** (leave it in the electromagnetic domain with an `eps_r`): the
  potential spreads into it. The admittance of the tutorial's transducer moves
  by 2.3 %.
- **Grounded** (name it in `ground=`): a metal plane. Use it for a conducting
  handle.

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

A terminal names a region of the mesh, an area or a line:

- A {term}`meshed electrode` is an area of metal: it carries displacement,
  stiffness and mass, with the potential fixed on the whole region. The
  fingers above are areas. They load the crystal, so they shift and reflect the
  wave.
- A {term}`line electrode` is a line on the crystal's surface with the potential
  fixed on it and no mass. It is light, and it is what the closed forms assume.

A line electrode loses the mass loading, and with it the strength of the
reflection: a grating's stop band is 1.97 % wide with meshed 50 nm aluminium
and 0.57 % with lines. Mesh real fingers whenever a transducer's reflection or
its resonance frequency matters, and use lines to check a closed form or to
keep a model light.

A meshed electrode must stay in the electromagnetic domain: the potential is
fixed on it. A line that is not named as a terminal or as ground is a free
surface; a `gfem.PEC` line that is not a terminal is grounded.

## Clamped lines and absorbing edges

Every edge of the mechanical domain is traction-free (the displacement is free)
unless named in `clamped=`, a list of lines where $u = 0$, each a {term}`clamped
surface`. The handle's base is the one clamped line of this section.

Waves that leave the section must not come back. A `gfem.Absorbing` line on a
solid's edge raises, because it is an electromagnetic boundary. The
{py:class}`gfem.DampingFrame` names the regions that absorb them instead, each
a {term}`damping frame` with a {term}`loss factor <Loss factor (of a damping
frame)>`: there the stiffness becomes
$c (1 + i(1/Q_m + \eta r^2))$ with $r$ the normalised depth into the frame, so
the wave decays and nothing returns. Give the helper's frames a region for each
layer on each side, each region keeping its layer's material in `materials=`
(`"ln_frame"` holds the same lithium niobate as `"ln"`), and meshed as
rectangles around the interior, every region on one side ending on one
straight line.

The thickness rule, for a plane wave at normal incidence on a frame of
thickness $L$ in a medium of wavelength $\lambda$: it reflects
$|R| \approx e^{-2\pi\eta (L/\lambda)/3}$, down to a floor of about
$3.4 \times 10^{-3}\,\eta\,(\lambda/L)^2$. A larger loss factor $\eta$ lowers
the decay term and raises the floor. The default $\eta = 1.2$ is the best choice
from four wavelengths of frame, where $|R| < 10^{-3}$; a frame of three
wavelengths takes $\eta = 2$. The helper's frames are 17.6 µm wide: four
wavelengths of the shear plate wave. A frame much thinner than that reflects.

## The result

{py:obj}`gfem.solve` returns a {py:class}`gfem.PiezoelectricResult`. Its
`admittance_per_length` is the {term}`admittance per length` $Y'_{ij}(f)$ in S/m, with the shape (frequencies,
terminals, terminals): the current into terminal $i$ per volt on terminal $j$, per
metre along $z$. The first index of the matrix is the response, the second the
excitation, as in S-parameters. `clamped_capacitance_per_length` is
the {term}`clamped capacitance`, in F/m: the capacitance of the electrodes with
$\varepsilon^S$ and no displacement, which is not the low-frequency limit of
$Y'/i\omega$ (the free capacitance is larger by $1 + K^2$). A device of
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
leave through the frames, is the power the terminal takes, so it is positive; the matrix is symmetric (reciprocal) with a positive semidefinite
real part. With several terminals, `terminals={"a": "signal_a", "b":
"signal_b"}` gives one row and column each, in that order; `res.index("b")`
finds one.

`res.to_xarray()` returns the admittance and the capacitance as an
`xarray.Dataset` labelled by frequency and terminal (the `xarray` extra).

## Network parameters

A circuit sees the device through its aperture and the resistance of its
fingers. {py:meth}`gfem.PiezoelectricResult.network_parameters` takes the
aperture, a {term}`series resistance` $R_s$ in each terminal's lead (one value or
one per terminal) and a reference impedance, and returns the scattering parameters.
The admittance of the aperture is $Y = W Y'$; the resistance gives
$Y_t = Y (I + R Y)^{-1}$, which needs no inverse of $Y$, and the scattering
matrix is that of $Y_t$ on the reference impedance (pseudo-waves,
{doc}`../conventions`). For one terminal on 50 Ω,
$S_{11} = (Z_L - Z_0) / (Z_L + Z_0)$ with $Z_L = 1/Y + R_s$.

The series resistance does not change the field solve, so it is an argument
here and not of the problem. How it depends on the aperture is yours: for fingers
fed from one end, $R_s = \tfrac{R_f}{3} (1/N_S + 1/N_G)$ with the finger's
resistance $R_f = \rho W / (w t)$.

```python
aperture = 400e-6  # m
net = res.network_parameters(aperture, series_resistance=26.7)
print(np.round(np.asarray(net.db("signal", "signal")), 3))
net.to_touchstone("transducer.s1p")
```

```text
[-0.061 -0.062 -0.065]
```

`net.db("signal", "signal")` is $|S_{11}|$ in dB per frequency. `to_touchstone`
writes the Touchstone file a circuit simulator reads. The function is written in
`jax.numpy`, so it differentiates in the aperture, the resistance and the
reference impedance; with several terminals it forms $Y (I + R Y)^{-1}$ for
the whole matrix.

## Fields and strain

`keep_fields=True` keeps the displacement and potential at every frequency and
terminal (a list of frequencies keeps those). Each kept frequency and terminal
holds four complex fields: mind the memory on a long sweep. `res.displacement(f)` is a map with the scalar fields `"x"`,
`"y"` and `"z"`, in m per volt on the terminal; `res.potential(f)` is the
potential in V per volt. They read like any field: `.at(points)` takes
$(x, y)$ points in metres.

{py:func}`gfem.strain` reads the {term}`engineering strain` of a displacement at
points, in Voigt order $xx, yy, zz, yz, xz, xy$, per volt. The shear wave of
the section is polarized along $z$ and carries $S_{xz} = \partial_x u_z$, the
fifth entry:

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

## What raises

- A tetrahedral or a slab mesh: the problem is 2D (`Mesh.from_gmsh` of a file
  of triangles).
- `linear_solver="petsc"` or `frequency_workers` above 1, and any solve under
  `comm`: one process and the sparse direct solver.
- A `gfem.Absorbing` line on a solid's edge: use a damping frame.
- An area terminal outside the electromagnetic domain, and a region of the
  mechanical domain without a stiffness or a density.
- A frame region name that is not in the mesh, or a frame with a loss factor
  traced by JAX.

## Next

- {doc}`../tutorials/transducer`: this section at full size, the strain at the
  two arms and the pitch whose {term}`synchronous frequency` is a chosen one.
- {doc}`piezoelectric-gradient`: the gradients of the admittance with respect
  to the materials and to a node map of your own.
- {doc}`2d-models`: what a 2D model is and what it refuses.
- {doc}`../conventions`, "Piezoelectric problems": the signs, the units and the
  formulas of network parameters.
