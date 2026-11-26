# Electro-optic phase gradients

Find how the phase a voltage puts on a crystal waveguide depends on what
sets it: each entry of the {term}`electro-optic tensor`, the
{term}`static permittivity` of every material, the drive, and the
electrode gap. One `jax.grad` call through
{py:obj}`gfem.Electrostatic` → {py:obj}`gfem.electro_optic_shift` gives the
first three, for one adjoint electrostatic solve.

## The phase as a function of its inputs

A {py:obj}`gfem.MaterialField` lets `jax.grad` reach a material's values: the
crystal's static principal values on its `axes`, the oxide's and the air's
static permittivity, and the crystal's tensor from its four class-3m entries.

```python
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

import gfem
from gfem.shape import boundary_morph

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/tfln_section.py"
)
urllib.request.urlretrieve(url, "tfln_section.py")
from tfln_section import tfln_section  # noqa: E402

um = 1e-6
axes = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))  # x-cut
frequency = 299_792_458.0 / 1.55e-6  # Hz
length = 2e-3  # m
section = tfln_section(gap=3.5 * um, width=0.9 * um, coarsen=2.0)
electrodes = ["signal", "ground_r", "ground_l"]
drive = gfem.Electrostatic({"signal": ["signal_s"]}, ground=["ground_s"])
options = gfem.SolverOptions(order=2)
oxide = gfem.Material(eps_r=1.444**2)


def tensor_3m(r13, r22, r33, r51):
    """The contracted tensor of class 3m, rows xx yy zz yz xz xy, m/V."""
    return jnp.array(
        [
            [0.0, -r22, r13],
            [0.0, r22, r13],
            [0.0, 0.0, r33],
            [0.0, r51, 0.0],
            [r51, 0.0, 0.0],
            [-r22, 0.0, 0.0],
        ]
    )


def static_model(mesh, eps_ln, eps_ox, eps_air):
    """The electrostatic model, its permittivities differentiable."""
    crystal = gfem.MaterialField(eps_r=eps_ln, axes=axes)
    materials = {
        "air": gfem.MaterialField(eps_r=eps_air),
        "cladding": gfem.MaterialField(eps_r=eps_ox),
        "box": gfem.MaterialField(eps_r=eps_ox),
        "slab": crystal,
        "ridge_a": crystal,
        "ridge_b": crystal,
        **{name: gfem.Material() for name in electrodes},
    }
    return gfem.Model(mesh, materials=materials, excluded={"em": electrodes})


def window_model(mesh, r):
    """Arm a's optical window, its crystal carrying the tensor of ``r``."""
    crystal = gfem.MaterialField(
        eps_r=jnp.array((2.2111**2, 2.2111**2, 2.1376**2)),
        axes=axes,
        electro_optic_tensor=tensor_3m(*r),
    )
    materials = {"cladding": oxide, "box": oxide, "slab": crystal, "ridge": crystal}
    return gfem.Model(mesh, materials=materials, ports=[gfem.WavePort("arm", "front")])


def phase(r, eps_ln, eps_ox, eps_air, volts):
    """The phase delay arm a gains over ``length`` at ``volts``, rad."""
    static = static_model(section.static, eps_ln, eps_ox, eps_air)
    potential = gfem.solve(static, drive, options=options).potential["signal"]
    shift = gfem.electro_optic_shift(
        window_model(section.window_a, r),
        potential,
        port="arm",
        frequency=frequency,
        mode=2,
    )
    return shift.phase_per_length_per_volt * volts * length


inputs = (
    jnp.array([9.6e-12, 6.8e-12, 30.9e-12, 32.6e-12]),  # r13, r22, r33, r51 (m/V)
    jnp.array([44.0, 44.0, 27.9]),  # the crystal's static eps along its x, y, z
    3.9,  # oxide
    1.0,  # air
    1.0,  # V
)
value, (d_r, d_ln, d_ox, d_air, d_volts) = jax.value_and_grad(
    phase, argnums=(0, 1, 2, 3, 4)
)(*inputs)
print(f"phase at 1 V: {float(value):.5f} rad")
```

```text
phase at 1 V: -0.19565 rad
```

The gap goes through a {py:func}`gfem.shape.boundary_morph` of the
electrostatic mesh. The ridge's width is the exception: it moves the mode,
which the shift holds constant, so it takes a difference over two remeshes.
The device is the section of {doc}`electro-optic-phase`, built by
{download}`tfln_section.py <../tutorials/tfln_section.py>` at twice its edge
lengths; the phase is that of one arm over 2 mm at 1 V.

- The phase is negative on arm a: the static field points from the signal to
  the ground, along the crystal's $+z$, and lowers the index by about
  $\tfrac12 n_e^3 r_{33} E_z$. On arm b both reverse.
- The shift holds its mode constant, so the gradient goes through the static
  field and the tensor, not through the mode: the optical model's permittivity
  and node coordinates cannot be inputs (see "What raises").

## Check the gradient against closed forms

- The phase is linear in the tensor, so $\sum_I r_I\, \partial\varphi/\partial r_I = \varphi$;
  each term over $\varphi$ is that entry's share.
- Scaling every static permittivity by one factor leaves the potential and the
  phase unchanged: $\sum_m \varepsilon_m\, \partial\varphi/\partial\varepsilon_m = 0$.
- The phase is linear in the drive: $V\, \partial\varphi/\partial V = \varphi$.

```python
r, eps_ln, eps_ox, eps_air, volts = inputs
shares = np.asarray(r * d_r / value)
eps = (eps_ln * d_ln).sum() + eps_ox * d_ox + eps_air * d_air
print(f"sum r dphi/dr / phi:     {shares.sum():.6f}")
for name, share in zip(("r13", "r22", "r33", "r51"), shares, strict=True):
    print(f"  {name}: {share:+.4f}")
print(f"sum eps dphi/deps / phi: {float(eps / value):+.1e}")
for name, part in zip("xyz", np.asarray(eps_ln * d_ln / value), strict=True):
    print(f"  crystal {name}: {part:+.4f}")
print(f"  oxide:     {float(eps_ox * d_ox / value):+.4f}")
print(f"  air:       {float(eps_air * d_air / value):+.4f}")
print(f"V dphi/dV / phi:         {float(volts * d_volts / value):.6f}")
```

```text
sum r dphi/dr / phi:     1.000000
  r13: +0.0188
  r22: -0.0000
  r33: +0.9645
  r51: +0.0167
sum eps dphi/deps / phi: +1.0e-15
  crystal x: +0.0232
  crystal y: +0.0002
  crystal z: -0.1086
  oxide:     +0.0901
  air:       -0.0050
V dphi/dV / phi:         1.000000
```

$r_{33}$ earns almost all of the phase, since the TE mode's field and the
static field both lie along the crystal's $z$; $r_{22}$ gives none, its axis
being along the guide. A larger static permittivity of the crystal pushes the
field into the oxide, so its share is negative and the oxide's positive.

## The electrode gap

{py:func}`gfem.shape.boundary_morph` moves a named surface of the electrostatic
mesh along its normal, and the static field follows; `jax.grad` counts both the
change of the potential and the static mesh moving under the optical window's
fixed points. Two rules make the move a change of the gap alone:

- **Name the crystal's interfaces so they slide, and hold the waveguide.** The
  section names the film's faces (`"ln_if"`) and the ridges' outlines;
  `fixed=` holds the ridges. An unnamed interface deforms with the electrode
  and gives a gradient about 5 % off.
- **Move both sides of the gap.** One morph moves the signal's inner walls and
  another the grounds', each by half; their sum is the move of both walls to
  first order.

A positive `t` grows each electrode, so a move `s` closes the gap by `s`:

```python
static = static_model(section.static, *inputs[1:4])
window = window_model(section.window_a, inputs[0])


def walls(x0):
    """The nodes on the vertical walls at |x| = x0."""
    return lambda x: np.abs(np.abs(x[:, 0]) - x0) < 1e-12


hold = ["ridge_a_s", "ridge_b_s"]
signal = boundary_morph(
    static, "signal_s", where=walls(section.signal_edge), fixed=hold
)
ground = boundary_morph(
    static, "ground_s", where=walls(section.ground_edge), fixed=hold
)


def closed_by(s):
    """Arm a's shift per volt with the gap closed by ``s`` m, the ridge centred."""
    moved = jax.tree.map(
        lambda p, q, p0: p + q - p0,
        signal(s / 2 * jnp.ones(signal.n_params)),
        ground(s / 2 * jnp.ones(ground.n_params)),
        static.mesh,
    )
    potential = gfem.solve(static.replace(mesh=moved), drive, options=options)
    return gfem.electro_optic_shift(
        window, potential.potential["signal"], port="arm", frequency=frequency, mode=2
    ).delta_n_eff_per_volt


s0, d_s = jax.value_and_grad(closed_by)(0.0)
print(f"d ln|dn/dV| / d gap: {-float(d_s / s0) * um:+.4f} per um")
for step in (1e-9, 2e-9):  # m
    fd = (closed_by(step) - closed_by(-step)) / (2 * step)
    print(
        f"against a difference at {step / 1e-9:.0f} nm: {abs(float(fd / d_s) - 1):.1e}"
    )
```

```text
d ln|dn/dV| / d gap: -0.2949 per um
against a difference at 1 nm: 9.6e-08
against a difference at 2 nm: 3.8e-07
```

A wider gap weakens the field, by about 30 % per µm of gap to first order, and
the difference agrees with the gradient at both steps. The window and the static
mesh are separate: a step that carries a window point across a face of the
moving mesh makes the shift jump slightly, so check with steps that stay on the
smooth piece, or with the optical model on the static model's own mesh.

## The ridge width, by a difference over remeshes

The width moves the mode, which the shift cannot differentiate. Remesh at two
widths and take a central difference, at a step large next to the mesh's noise
and small next to the width:

```python
def shift_at(width):
    """Arm a's shift per volt on a section remeshed at ``width``, m."""
    remeshed = tfln_section(gap=3.5 * um, width=width, coarsen=2.0)
    static = static_model(remeshed.static, *inputs[1:4])
    potential = gfem.solve(static, drive, options=options).potential["signal"]
    return gfem.electro_optic_shift(
        window_model(remeshed.window_a, inputs[0]),
        potential,
        port="arm",
        frequency=frequency,
        mode=2,
    ).delta_n_eff_per_volt


h = 20e-9  # m
d_width = (shift_at(0.9 * um + h) - shift_at(0.9 * um - h)) / (2 * h)
print(f"d ln|dn/dV| / d width: {float(d_width / s0) * um:+.4f} per um")
```

```text
d ln|dn/dV| / d width: +0.0703 per um
```

Each remesh moves every node, so the difference is noisy at about $10^{-3}$ of
the objective: stop an optimizer fed with it at that tolerance, as the
{doc}`modulator tutorial <../tutorials/modulator>` does.

## What raises

`jax.grad` of the shift raises {py:obj}`gfem.NonDifferentiableError` when it
reaches an input of the optical model other than the tensor: its `eps_r`,
`tan_delta`, `mu_r`, `conductivity` or node coordinates, each of which moves the
mode. To use such a value without its derivative, wrap it in
`jax.lax.stop_gradient`.

```python
def with_optical_eps(eps_z):
    crystal = gfem.MaterialField(
        eps_r=jnp.array([2.2111**2, 2.2111**2, eps_z]),
        axes=axes,
        electro_optic_tensor=tensor_3m(*inputs[0]),
    )
    optical = window.replace(
        materials={**window.materials, "slab": crystal, "ridge": crystal}
    )
    potential = gfem.solve(static, drive, options=options).potential["signal"]
    return gfem.electro_optic_shift(
        optical, potential, port="arm", frequency=frequency, mode=2
    ).delta_n_eff_per_volt


try:
    jax.grad(with_optical_eps)(2.1376**2)
except gfem.NonDifferentiableError as error:
    print(type(error).__name__)
```

```text
NonDifferentiableError
```

## Limits

- **No gradient through the mode,** so none with respect to the ridge's shape
  or the optical permittivity: difference over remeshes.
- **Gap moves** need the crystal's interfaces named in the electrostatic mesh;
  a window's objective is smooth only between faces of the moving mesh.

## Next

{doc}`electro-optic-phase`, {doc}`../tutorials/modulator`,
{doc}`shape-gradient`, {doc}`remesh-and-restart`, {doc}`capacitance-gradient`.
