# Electro-optic phase shift

Find how much a voltage on an electrode shifts the phase of light in a
crystal waveguide: the change of the effective index per volt, the
voltage-length product $V_\pi L$ of one arm and of a push-pull pair, and the
electrodes' capacitance per length, from an electrostatic solve and
{py:obj}`gfem.electro_optic_shift` (the {term}`electro-optic shift` of a mode).

The device is the {doc}`modulator tutorial <../tutorials/modulator>`'s
section, built by {download}`tfln_section.py <../tutorials/tfln_section.py>`
at twice its edge lengths: two lithium niobate ridges in the gaps of
ground-signal-ground electrodes, at 1.55 µm.

## A crystal with an electro-optic tensor

A material's `electro_optic_tensor` is the 6 × 3 {term}`electro-optic tensor`
in m/V, in the frame of its right-handed `axes` ({doc}`../conventions`,
"Electro-optic shift"). The electrostatic model holds the
{term}`static permittivity`, the optical one the permittivity and the tensor,
with the same `axes` from one variable (the shift cannot check that):

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

- An element without a tensor adds nothing to the shift.
- {py:obj}`gfem.materials.lithium_niobate` and
  {py:obj}`gfem.materials.lithium_niobate_static` build the two materials from
  cited sources with the clamped tensor, as the tutorial does; the coefficients
  typed here are the unclamped set.

## The shift per volt

The electrostatic model holds the whole section, with the electrodes out of the
solve (`excluded={"em": [...]}`): their surfaces carry the potentials. One
{term}`optical window` per ridge, on a mesh of its own, holds the ridge out to
the electrodes' inner edges; a wave port on its end face `"front"` carries the
mode.

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

- `delta_n_eff_per_volt` is $\partial n_\mathrm{eff}/\partial V$ for 1 V on the
  terminal whose potential was given: the shift is linear in the drive, and one
  electrostatic solve serves every voltage.
- `phase_per_length_per_volt` is $k_0\,\partial n_\mathrm{eff}/\partial V$ in
  rad/(V m).
- `v_pi_l` is the {term}`voltage-length product`
  $V_\pi L = \lambda / (2\,|\partial n_\mathrm{eff}/\partial V|)$ in V m (`100 *
  v_pi_l` in V cm); an arm of length $L$ needs $V_\pi = V_\pi L / L$.
- `mode=2` picks the TE mode, the one $r_{33}$ acts on (see the last section).

## Push-pull and several terminals

The signal sits between the two arms, so its field crosses them in opposite
directions and the two shifts have opposite signs. Driven {term}`push-pull`,
the arms' phase difference is the difference of the two shifts:

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

With several signal terminals, solve once with all of them: each
`es.potential[name]` gives a per-volt shift, and the shifts add, each times its
own drive.

## The electrodes' capacitance per length

The same electrostatic solve gives the capacitance the drive sees, per metre of
electrode on a cross-section:

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

- **A window out to the electrodes.** The shift weighs the mode's lateral tail,
  where the static field is strongest. Solving on the whole section puts the
  electrodes in the optical solve as magnetic walls whose shells guide spurious
  modes; the two shifts agree within a few percent (1.6 % measured).
- **The mode.** In this window the first mode is TM and the TE mode is the
  second: check the share of $|E_x|^2$ in $|E_x|^2 + |E_y|^2$, near 1 for TE.

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
  magnetic walls by default; against a slab's closed form, name them and list
  them in the wave port's `pec_edges`, since magnetic walls raise a wide slab's
  $n_\mathrm{eff}$ by about $(\lambda / 2W)^2 / (2n)$.

## Check your results

- **Mirror symmetry.** In a symmetric section the two arms' shifts are equal
  and opposite to the meshes' asymmetry: $3 \times 10^{-3}$ above, $2.5 \times
  10^{-5}$ on the tutorial's finer section.
- **The bulk estimate.** A uniform field $V/g$ across a gap $g$ shifts a TE
  mode's index by about $\tfrac12 n_e^3 r_{33} V / g$; the ratio of the shift to
  it, the {term}`electro-optic overlap factor` $\Gamma$, is between 0.3 and 0.7
  for a ridge in a few-µm gap.
- **Convergence.** Halve `coarsen` and raise the electrostatic solve's `order`:
  an order-1 solve biases the static field's gradient by about 1 %.

## Limits

- **DC cross-section.** No travelling-wave effects (velocity matching, RF loss,
  bandwidth).
- **First order.** In the field with the mode held constant, about $10^{-6}$ of
  the shift at $V_\pi$ for lithium niobate.
- **Lossless optics.** A complex `eps_r`, a `tan_delta` or a `conductivity`
  raises, and so do a 2D model and a potential split across ranks.
- **No electrodes in the optics.** No metal absorption: keep them clear of the
  mode, as the tutorial does with an {term}`electrode clearance`.

## Next

{doc}`electro-optic-gradient`, {doc}`../tutorials/modulator`,
{doc}`thermo-optic-phase`, {doc}`../modes-and-ports`.
