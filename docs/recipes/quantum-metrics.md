# Qubit metrics and gradients

Get a qubit's anharmonicity and its dispersive shift on a resonator from an
eigenmode solve, and differentiate them with respect to materials, sheet fill,
shape and the junction's inductance. The solve treats the Josephson junction as
a linear inductor. {py:obj}`gfem.quantum.dressed_modes` then puts the
junction's cosine back on the solved modes and returns the
{term}`dressed modes <Dressed mode>`.

## From a solve to the dispersive shift

The mesh `transmon.msh` holds two pads on a silicon chip inside a conducting
box, joined by a junction on the surface `"junction"`. The junction is a
{py:obj}`gfem.LumpedPort` with only an inductance: on an eigenmode solve it
excites nothing, and its {term}`participation ratio` is the junction
participation of each mode.

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

- The shift depends on the design through the mode frequencies and through
  the junction participations, which are properties of the mode fields.
  `jax.grad` follows both. The participations' part costs one extra linear
  solve per mode the objective reads, on the factorization the eigensolve
  already made.
- Modes come nearest `target` first. Over an optimization two modes can swap
  places; follow one mode by its field as in {doc}`eigenmode-gradient`.
- The tutorial {doc}`../tutorials/transmon` differentiates the dispersive
  shift with respect to the pad gap, and splits the gradient into its
  frequency and mode-field parts.

## What you get

{py:obj}`gfem.quantum.DressedModes` holds, in the input mode order, signed
angular frequencies in rad/s; divide by $2\pi$ for Hz:

`omega`
: The dressed transition frequency of each mode.

`lamb_shift`
: The {term}`Lamb shift`: `omega` minus the solved mode's `Re(omega)`.

`anharmonicity`
: The {term}`anharmonicity`: the second transition frequency minus the
  first, negative for a transmon.

`dispersive_shift`
: The {term}`dispersive shift`, `(n, n)` and symmetric: the full shift of
  mode `n`'s frequency per photon in mode `m`, not half of it. Its diagonal is
  `anharmonicity`.

`overlap`
: The smallest overlap of a level used with the Fock state it is labelled
  by, in (1/2, 1]. Close to 1, the labels are clear.

These signs and the full shift are gfem's convention: other sources write
the shift as positive or as half of it. The conversions are in
{py:obj}`gfem.conventions`, under "Quantum metrics".

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
`dressed_modes`: the result is then a different circuit, and the gradient
misses the solve's share. Pass one variable to both.

## Options

- `levels` (12 by default) is the number of Fock levels per mode, one for
  all modes or one per mode, each at least 3. Their product over the modes
  is at most `gfem.quantum.MAX_DIMENSION`; select fewer modes or lower
  levels if it is not. Raise `levels` until the outputs stop changing.
- `method="first_order"` returns the closed perturbative forms. Use it only
  to compare with published first-order numbers: at ordinary detunings its
  dispersive shift is off by a factor of 1.5 to 1.9.
- Select modes by slicing: pass `modes.omega[:2]` and the matching
  participations. A mode you leave out adds no Lamb shift and no dispersive
  shift.

## When it refuses

- A level whose overlap with its Fock state is 1/2 or less raises
  {py:obj}`gfem.ModeLabelError`: two modes have hybridized. Move them apart in
  the design, or leave the hybridized mode out.
- A mode with a zero junction participation raises `ValueError`; leave it
  out.
- Inside `jax.jit`, `jax.grad` or `jax.vmap` both arrive as
  `jax.errors.JaxRuntimeError`, with gfem's message (see
  {ref}`gradient-errors`).
- A second derivative of the dressed modes raises
  {py:obj}`gfem.NonDifferentiableError`. Compute an inner gradient outside the
  differentiated function, or use `method="first_order"`, which
  differentiates to any order.
- `overlap` is read for its value, not differentiated. A reverse-mode
  gradient through it (`jax.grad`, `jax.vjp`) raises
  `gfem.NonDifferentiableError`. In forward mode (`jax.jvp`, `jax.jacfwd`)
  its tangent is NaN. Wrap it in `jax.lax.stop_gradient` to use it in a
  differentiated function.

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
  `-2 d ln(omega) / d ln(L_s)`. A small change of `L_s` therefore moves the
  frequency by `d omega / omega = -(p / 2) d L_s / L_s`.
- It differentiates like the junction participation, also with respect to
  `L_s` and the fill of a {py:obj}`gfem.SheetField`.
- A heavily inductive sheet carries its own sheet-plasmon modes, which can
  lie near `target`: ask for a few more modes and pick yours by its field.
- At the edge of a patterned film the sheet current is concentrated within a
  distance `L_s / mu0` of the edge (0.8 µm for 1 pH per square, 0.08 µm for
  0.1 pH per square). The participation converges once the mesh cells at the
  film edge are about that small. On a 10 µm strip between walls it was
  within about 1 % at twice that size. A coarser edge overestimates it, by
  3 % at four times `L_s / mu0` and 15 % at thirty times. Refine the mesh
  towards film edges rather than everywhere; a uniform sheet or wall needs no
  such care.

## Check your results

- Solve again with `L_J` changed by 1 %: the junction participation of each
  mode is close to `-2 d ln(omega) / d ln(L_J)` from the two solves. The
  identity is exact when the field is uniform across the port.
- The same check on `L_s` holds for `sheet_participation` on any mesh, so it
  does not check the mesh at a film edge.
- Raise `levels` by a few: the outputs should not move.
- Compare the gradient with a central difference of your objective along a
  random direction, at the default `eigen_tolerance`.

How gfem itself is checked: {doc}`../verification`.

## Validity

- The model is the junction's cosine on the solved linear modes, with no
  charge dispersion. It holds in the transmon regime, with `E_J` much larger
  than the charging energy; use a charge-basis model outside it.
- It uses `Re(omega)` only, so it holds for modes with `Q` much larger
  than 1. Read their loss separately ({doc}`loss-budget`).
- Modes you do not solve or do not pass contribute nothing. Solve enough
  modes around the ones you read, and check that adding one more does not move
  them.
- The solve's qubit frequency is that of the linearized circuit. The dressed
  one is lower by about the anharmonicity: quote `dressed.omega` as the qubit
  frequency.

## Limits

- One junction. `dressed_modes` takes one junction's participations; a 2-D
  `participation` raises `ValueError`.
- `Eigenmode.target` is not a gradient input; keep it constant inside the
  differentiated function.
- The gradient of a participation needs an isolated mode, the direct solver
  on the CPU and one process; with `eigen_tolerance` above 1e-10 it raises.
  The conditions are in {doc}`eigenmode-gradient`.

## Next

{doc}`../tutorials/transmon`, {doc}`loss-budget`, {doc}`eigenmode-gradient`,
{doc}`shape-gradient`; {py:obj}`gfem.quantum.dressed_modes`,
{py:obj}`gfem.conventions`.
