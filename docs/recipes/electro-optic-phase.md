# Electro-optic phase shift

Find how much a voltage on an electrode shifts the phase of light in a
crystal waveguide: the change of the effective index per volt, the
voltage-length product $V_\pi L$ of one arm and of a push-pull pair, and the
electrodes' capacitance per length. An electrostatic solve gives the static
field; {py:obj}`gfem.electro_optic_shift` turns it into the
{term}`electro-optic shift` of a guided mode on a cross-section, through
each crystal's electro-optic tensor.

The device is the section of the {doc}`modulator tutorial
<../tutorials/modulator>`: two lithium niobate ridges on an x-cut film, one in
each gap of a ground-signal-ground electrode pattern, at $\lambda = 1.55$ µm.
The tutorial's {download}`tfln_section.py <../tutorials/tfln_section.py>`
meshes it with netgen; here at twice the tutorial's edge lengths, which is
quicker and moves the shift by under 1 %. The shift is first order in the
static field: one mode solve gives the weights of every element, and the
shift is their sum weighted by the change of permittivity the field causes,
with the mode held constant. The section must be lossless.

## A crystal with an electro-optic tensor

A material's `electro_optic_tensor` is the contracted {term}`electro-optic
tensor` $r_{Ik}$, a 6 × 3 array in m/V: rows $xx, yy, zz, yz, xz, xy$,
columns $x, y, z$, in the frame of the material's `axes`. The rows of `axes`
are the crystal's $x$, $y$ and $z$ in the model's coordinates. The static
field $E_\mathrm{s}$ changes the optical permittivity by
$\Delta\varepsilon_r = -\varepsilon_r (r \cdot E_\mathrm{s}) \varepsilon_r$
({doc}`../conventions`, "Electro-optic shift").

The electrostatic model and the optical one each have their own crystal
material: the {term}`static permittivity` in one, the optical permittivity
and the tensor in the other. Give both the same `axes` from one variable: the
shift cannot check that they agree.

```python
import urllib.request

import numpy as np

import gfem

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/tfln_section.py"
)
urllib.request.urlretrieve(url, "tfln_section.py")
from tfln_section import tfln_section  # noqa: E402

um = 1e-6
# x-cut: crystal x up (y), crystal y along the guide (z), crystal z across (x)
axes = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
r13, r22, r33, r51 = 9.6e-12, 6.8e-12, 30.9e-12, 32.6e-12  # m/V, class 3m
tensor = [
    [0.0, -r22, r13],  # xx
    [0.0, r22, r13],  # yy
    [0.0, 0.0, r33],  # zz
    [0.0, r51, 0.0],  # yz
    [r51, 0.0, 0.0],  # xz
    [-r22, 0.0, 0.0],  # xy
]
ln_static = gfem.Material(eps_r=(44.0, 44.0, 27.9), axes=axes)
ln_optical = gfem.Material(
    eps_r=(2.2111**2, 2.2111**2, 2.1376**2), axes=axes, electro_optic_tensor=tensor
)
oxide_static, oxide = gfem.Material(eps_r=3.9), gfem.Material(eps_r=1.444**2)
```

- The tensor is a property of the material, so a region of crystal carries it
  in every model it appears in; an element without one adds nothing to the
  shift.
- `axes` must be right-handed: the tensor's sign follows the crystal's
  frame, and a left-handed set flips it without an error.
- **Cited constants.** {py:obj}`gfem.materials.lithium_niobate` and
  {py:obj}`gfem.materials.lithium_niobate_static` build the two crystal
  materials from cited sources, with the clamped tensor and static
  permittivity, as the modulator tutorial does. The coefficients typed here
  are the unclamped set.

## The shift per volt

The electrostatic model holds the whole section, with the electrodes left
out of the solve (`excluded={"em": [...]}`): their surfaces carry the
potentials. One {term}`optical window` per ridge holds the ridge and its
surroundings out to the electrodes' inner edges, on a mesh of its own in the
same coordinates. A wave port on its end face `"front"` carries the mode.

```python
section = tfln_section(gap=3.5 * um, width=0.9 * um, coarsen=2.0)
electrodes = ["signal", "ground_r", "ground_l"]
static = gfem.Model(
    section.static,
    materials={
        "air": gfem.Material(),
        "cladding": oxide_static,
        "box": oxide_static,
        "slab": ln_static,
        "ridge_a": ln_static,
        "ridge_b": ln_static,
        **{name: gfem.Material() for name in electrodes},
    },
    excluded={"em": electrodes},
)
es = gfem.solve(
    static,
    gfem.Electrostatic({"signal": ["signal_s"]}, ground=["ground_s"]),
    options=gfem.SolverOptions(order=2),
)


def window(mesh):
    """One ridge's optical window, its mode on the end face "front"."""
    materials = {
        "cladding": oxide,
        "box": oxide,
        "slab": ln_optical,
        "ridge": ln_optical,
    }
    return gfem.Model(mesh, materials=materials, ports=[gfem.WavePort("arm", "front")])


frequency = 299_792_458.0 / 1.55e-6  # Hz
potential = es.potential["signal"]  # 1 V on the signal, 0 V on the grounds
a = gfem.electro_optic_shift(
    window(section.window_a), potential, port="arm", frequency=frequency, mode=2
)
b = gfem.electro_optic_shift(
    window(section.window_b), potential, port="arm", frequency=frequency, mode=2
)
print(a)
print(f"arm b / arm a: {float(b.delta_n_eff_per_volt / a.delta_n_eff_per_volt):.5f}")
print(f"V_pi L of arm a: {100 * float(a.v_pi_l):.3f} V cm")
```

```text
ElectroOpticShift(port 'arm', mode 2, 1.93414e+14 Hz, n_eff 1.907649, delta_n_eff_per_volt -2.41331e-05 /V, v_pi_l 0.0321136 V m)
arm b / arm a: -1.00262
V_pi L of arm a: 3.211 V cm
```

- `delta_n_eff_per_volt` is $\partial n_\mathrm{eff}/\partial V$ for 1 V on
  the terminal whose potential was given, every other terminal and the ground
  at 0 V. The shift is linear in the drive: at $V$ volts the index moves by
  $V$ times it, and one electrostatic solve serves every voltage.
- `phase_per_length_per_volt` is $k_0\,\partial n_\mathrm{eff}/\partial V$
  in rad/(V m), a phase delay; `n_eff` is the index without a field.
- `v_pi_l` is the {term}`voltage-length product` of the arm,
  $V_\pi L = \lambda / (2\,|\partial n_\mathrm{eff}/\partial V|)$ in V m:
  `100 * v_pi_l` is in V cm, and an arm of length $L$ needs
  $V_\pi = V_\pi L / L$.
- `mode=2` picks the TE mode, the one the crystal's $r_{33}$ acts on; see
  "Choose the window and the mode" below.

## Push-pull and several terminals

The signal sits between the two arms, so its field crosses them in opposite
directions and the two shifts have opposite signs. Driven {term}`push-pull`
in a Mach-Zehnder interferometer, the arms' phase difference is the
difference of the two shifts:

```python
wavelength = 1.55e-6  # m
v_pi_l = wavelength / (2 * abs(a.delta_n_eff_per_volt - b.delta_n_eff_per_volt))
length = 2e-3  # m of electrode
print(f"push-pull V_pi L: {100 * float(v_pi_l):.3f} V cm")
print(f"V_pi over {1e3 * length:.0f} mm: {float(v_pi_l / length):.2f} V")
```

```text
push-pull V_pi L: 1.604 V cm
V_pi over 2 mm: 8.02 V
```

With several signal terminals, solve once with all of them as terminals: each
`es.potential[name]` gives a per-volt shift, and the shifts add, each times
its own drive.

## The electrodes' capacitance per length

The same electrostatic solve gives the capacitance the drive sees. On a
cross-section it is per metre of electrode:

```python
print(es)
print(f"{1e12 * float(es.capacitance_per_length[0, 0]):.1f} pF/m")
```

```text
ElectrostaticResult(terminals=['signal'], slab capacitance in fF:
  signal:   0.009714
capacitance per length in pF/m:
  signal:      97.14)
97.1 pF/m
```

## Choose the window and the mode

- **A window out to the electrodes.** The shift weighs the mode's lateral
  tail, where the static field is strongest, so take the window out to the
  electrodes' inner edges. Solving the mode on the whole section instead,
  with the electrodes excluded, puts the electrodes' surfaces in the optical
  solve as magnetic walls: shells around metal that guide modes of their own
  in the film, which crowd the spectrum above the ridge's. The window and the
  whole section give shifts within a few percent (1.6 % measured on this
  kind of ridge), the difference of their boundaries.
- **The mode.** In this window the first mode is TM, polarized up the
  ridge, and the TE mode is the second. Check the polarization of the mode
  you pick from its field: the share of $|E_x|^2$ in
  $|E_x|^2 + |E_y|^2$ over the section is near 1 for TE.

```python
optical = window(section.window_a)
x = np.linspace(section.ridge_centre - 1.75 * um, section.ridge_centre + 1.75 * um, 71)
y = np.linspace(-1.2 * um, 1.7 * um, 59)
X, Y = np.meshgrid(x, y)
points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
for m in (1, 2):
    mode = gfem.solve(optical, gfem.PortMode("arm", frequency, mode=m))
    e = np.abs(np.asarray(mode.e_field.at(points))[0]) ** 2
    te = e[:, 0].sum() / (e[:, 0] + e[:, 1]).sum()
    print(f"mode {m}: n_eff {float(mode.n_eff[0].real):.4f}, TE fraction {te:.3f}")
```

```text
mode 1: n_eff 1.9250, TE fraction 0.011
mode 2: n_eff 1.9076, TE fraction 0.994
```

- **Closed-form checks want electric side walls.** A window's sides are
  magnetic walls by default. Against a slab's closed form, name the window's
  side faces and list them in the wave port's `pec_edges`: magnetic walls
  raise a wide slab's $n_\mathrm{eff}$ by about $(\lambda / 2W)^2 / (2n)$ for
  a width $W$.

## Check your results

- **Mirror symmetry.** In a symmetric ground-signal-ground section the two
  arms' shifts are equal and opposite, to the meshes' asymmetry: $3 \times
  10^{-3}$ above, and $2.5 \times 10^{-5}$ on the tutorial's finer section.
- **The bulk estimate.** A uniform field $V/g$ across a gap $g$ shifts a
  TE mode's index by about $\tfrac12 n_e^3 r_{33} V / g$. The ratio of the
  shift to it, the {term}`electro-optic overlap factor` $\Gamma$, is between
  about 0.3 and 0.7 for a ridge in a few-µm gap: the field in the crystal is
  weaker than $V/g$, and only part of the mode is in the crystal.
- **Convergence.** Halve `coarsen` (or refine the meshes of your own section)
  and raise `gfem.SolverOptions(order=...)` of the electrostatic solve: the
  shift reads the static field's gradient, which an order-1 solve biases by
  about 1 %.

## Limits

- A cross-section at DC: travelling-wave effects (velocity matching, RF loss,
  bandwidth) are outside it, and the tensor and the static permittivity are
  the values at the drive's frequency you give.
- First order in the field, with the mode held constant: about $10^{-6}$ of
  the shift at $V_\pi$ for lithium niobate.
- A lossless optical section: a complex `eps_r`, a `tan_delta` or a
  `conductivity` in the optical model raises, and so do a 2D model and a
  potential split across ranks.
- The optical model does not see the electrodes: no metal absorption. Keep
  the electrodes far enough from the mode, as the {doc}`modulator tutorial
  <../tutorials/modulator>` does with an {term}`electrode clearance`.

## Next

- {doc}`electro-optic-gradient`: the gradient of the shift with respect to
  the tensor, the static permittivity, the drive and the electrode gap.
- {doc}`../tutorials/modulator`: the push-pull modulator, its field, its
  $V_\pi L$ against a published device, and a co-design of the gap and the
  ridge width.
- {doc}`thermo-optic-phase`: the same first-order shift, driven by a
  temperature rise.
- {doc}`../modes-and-ports`: how wave ports pick and number their modes.
