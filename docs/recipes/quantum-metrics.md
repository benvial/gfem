# Qubit metrics and gradients

Get a qubit's anharmonicity and its dispersive shift on a resonator from an
eigenmode solve, and differentiate them with respect to materials, sheet fill,
shape and the junction's inductance. The solve treats the junction as a linear
inductor; {py:obj}`gfem.quantum.dressed_modes` puts its cosine back on the
solved modes and returns the {term}`dressed modes <Dressed mode>`.

## From a solve to the dispersive shift

The mesh `transmon.msh` holds two pads on a silicon chip inside a conducting
box, joined by a junction on the surface `"junction"`. The junction is a
{py:obj}`gfem.LumpedPort` with only an inductance: it excites nothing, and its
{term}`participation ratio` is the junction participation of each mode.

```python
import jax
import jax.numpy as jnp

import gfem

L_J = 7.5e-9  # H, the junction's linear inductance
mesh = gfem.Mesh.from_gmsh("transmon.msh", length_unit="mm")
model = gfem.Model(
    mesh,
    materials={"vacuum": gfem.Material(), "chip": gfem.Material(eps_r=11.7)},
    boundaries=[gfem.PEC(surface) for surface in ("walls", "pads", "pad_edge")],
    ports=[gfem.LumpedPort("jj", "junction", "+x", inductance=L_J)],
)
problem = gfem.Eigenmode(target=5.5e9, n_modes=2)  # the qubit, then the readout

modes = gfem.solve(model, problem)
dressed = gfem.quantum.dressed_modes(
    modes.omega, modes.participation["jj"], inductance=L_J
)
chi = dressed.dispersive_shift[0, 1]  # rad/s, the readout's shift per qubit photon
```

Every step is differentiable. The gradient of the dispersive shift with
respect to the chip's permittivity, element by element:

```python
def dispersive_shift(eps):
    """The dispersive shift between qubit and readout, Hz."""
    m = model.with_material("chip", gfem.MaterialField(eps_r=eps))
    modes = gfem.solve(m, problem)
    dressed = gfem.quantum.dressed_modes(
        modes.omega, modes.participation["jj"], inductance=L_J
    )
    return dressed.dispersive_shift[0, 1] / (2 * jnp.pi)


n = len(mesh.elements("chip"))
value, grad = jax.value_and_grad(dispersive_shift)(jnp.full(n, 11.7))
```

- `jax.grad` follows the mode frequencies and the junction participations, at
  one extra linear solve per mode the objective reads.
- Modes come nearest `target` first; follow one mode by its field as in
  {doc}`eigenmode-gradient`.
- The tutorial {doc}`../tutorials/transmon` differentiates the shift with
  respect to the pad gap.

## What you get

{py:obj}`gfem.quantum.DressedModes` holds, in the input mode order and in
rad/s (divide by $2\pi$ for Hz): `omega`, the dressed transition frequency;
`lamb_shift`, the {term}`Lamb shift`; `anharmonicity`, the second transition
frequency minus the first, negative for a transmon; `dispersive_shift`, the
{term}`dispersive shift`, `(n, n)` and symmetric, the full shift of mode `n`
per photon in mode `m` (its diagonal is `anharmonicity`); and `overlap`, close
to 1 when the Fock labels are clear. Other sources write the shift as positive
or as half of it: the conversions are in {py:obj}`gfem.conventions`, under
"Quantum metrics".

The step after the solve needs no mesh. On two typed modes, a qubit and a
resonator:

```python
import numpy as np

from gfem import quantum

omega = 2 * np.pi * np.array([5.6e9, 6.8e9])  # rad/s
participation = np.array([-0.97, -0.008])  # junction participation of each mode
dressed = quantum.dressed_modes(omega, participation, inductance=7.5e-9)
shift_hz = dressed.dispersive_shift[0, 1] / (2 * np.pi)
```

`inductance` is the junction's `L_J`, the value its `LumpedPort` was given;
the Josephson energy is `E_J = (ħ/2e)² / L_J`.

## The junction inductance as a design input

`L_J` differentiates through the solve and through the dressed modes. Make
it an argument of your objective, build the port from it, and pass the same
value to `dressed_modes`:

```python
morph = gfem.shape.boundary_morph(model, "pad_edge", fixed=("walls",))


def model_at(gap, L_J):
    """The device with its pad gap widened by ``gap`` (m), junction ``L_J`` (H)."""
    moved = model.replace(mesh=morph(-gap / 2 * jnp.ones(morph.n_params)))
    port = gfem.LumpedPort("jj", "junction", "+x", inductance=L_J)
    return moved.replace(ports=[port])


def dispersive_shift(gap, L_J):
    modes = gfem.solve(model_at(gap, L_J), problem)
    dressed = gfem.quantum.dressed_modes(
        modes.omega, modes.participation["jj"], inductance=L_J
    )
    return dressed.dispersive_shift[0, 1]


d_gap, d_lj = jax.grad(dispersive_shift, argnums=(0, 1))(0.0, 7.5e-9)
```

gfem cannot detect two different values passed to the port and to
`dressed_modes`: the gradient then misses the solve's share. Pass one
variable to both.

## Options

- `levels` (12) is the number of Fock levels per mode, one for all or one per
  mode, each at least 3, with a product at most `gfem.quantum.MAX_DIMENSION`.
  Raise it until the outputs stop changing.
- `method="first_order"` returns the closed perturbative forms, for comparison
  with published first-order numbers only: its dispersive shift is off by a
  factor of 1.5 to 1.9 at ordinary detunings.
- Select modes by slicing, `modes.omega[:2]` and the matching participations;
  a mode left out adds no Lamb or dispersive shift.

## When it refuses

- A level whose overlap with its Fock state is 1/2 or less raises
  {py:obj}`gfem.ModeLabelError`: two modes have hybridized. Move them apart,
  or leave one out.
- A mode with a zero junction participation raises `ValueError`; leave it
  out.
- Inside `jax.jit`, `jax.grad` or `jax.vmap` both arrive as
  `jax.errors.JaxRuntimeError`, with gfem's message ({ref}`gradient-errors`).
- A second derivative of the dressed modes raises
  {py:obj}`gfem.NonDifferentiableError`; use `method="first_order"`, which
  differentiates to any order.
- `overlap` is read for its value: reverse mode through it raises
  `gfem.NonDifferentiableError`, forward mode gives a NaN tangent. Wrap it in
  `jax.lax.stop_gradient`.

(quantum-sheet-participation)=
## Kinetic inductance of a film: sheet participation

A superconducting film with kinetic inductance `L_s` per square is an
{py:obj}`gfem.Impedance` with an inductance. Its
{term}`sheet participation` is the share of the mode's energy in that
inductance:

```python
film = gfem.Impedance("film", inductance=1e-12)  # H per square
modes = gfem.solve(model.replace(boundaries=[*model.boundaries, film]), problem)
p_film = modes.sheet_participation["film"]  # (n_modes,), in [0, 1]
```

- For a lossless mode it is the film's kinetic inductance fraction,
  `-2 d ln(omega) / d ln(L_s)`, so `d omega / omega = -(p / 2) d L_s / L_s`.
- It differentiates like the junction participation, also with respect to
  `L_s` and a {py:obj}`gfem.SheetField` fill.
- A heavily inductive sheet carries its own sheet-plasmon modes, which can
  lie near `target`: ask for a few more modes and pick yours by its field.
- At the edge of a patterned film the sheet current sits within `L_s / mu0`
  of the edge (0.8 µm for 1 pH per square). The participation converges once
  the cells at the edge are about that small (within about 1 % at twice that
  size); a coarser edge overestimates it, by 3 % at four times and 15 % at
  thirty times. Refine towards film edges; a uniform sheet needs no care.

## Check your results

- Solve again with `L_J` changed by 1 %: each junction participation is close
  to `-2 d ln(omega) / d ln(L_J)` from the two solves (exact for a field
  uniform across the port). The same check on `L_s` does not check the mesh at
  a film edge.
- Raise `levels` by a few: the outputs should not move.
- Compare the gradient with a central difference along a random direction.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **Transmon regime.** The junction's cosine on the solved linear modes has no
  charge dispersion: `E_J` must be much larger than the charging energy, and
  `Q` much larger than 1, since only `Re(omega)` is used ({doc}`loss-budget`).
  Quote `dressed.omega`, not the linearized circuit's frequency.
- Modes you do not solve or pass contribute nothing: check that adding one
  more does not move them.
- One junction: a 2-D `participation` raises `ValueError`.
- `Eigenmode.target` is not a gradient input; keep it constant inside the
  differentiated function.
- The gradient of a participation has the conditions of
  {doc}`eigenmode-gradient`.

## Next

{doc}`../tutorials/transmon`, {doc}`loss-budget`, {doc}`eigenmode-gradient`,
{doc}`shape-gradient`; {py:obj}`gfem.quantum.dressed_modes`,
{py:obj}`gfem.conventions`.
