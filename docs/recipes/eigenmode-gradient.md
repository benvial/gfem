# Eigenfrequency and Q gradients

Tune a resonator's frequency and Q by differentiating them with respect to the
material in each element. `jax.grad` of an objective built from
`result.omega`, and so from `result.frequency` and `result.q`, reaches the
per-element `eps_r`, `mu_r`, `tan_delta` and `conductivity` of every
{py:obj}`gfem.MaterialField`; so do the participation ratios and port Q of a
mode, through its field. Batches, memory, `stop_gradient` and errors work as
for a capacitance ({doc}`capacitance-gradient`).

## A frequency and loss objective

The mesh `resonator.msh` holds a resonator on a substrate inside a
conducting box.

```python
import jax
import jax.numpy as jnp

import gfem

mesh = gfem.Mesh.from_gmsh("resonator.msh")
model = gfem.Model(
    mesh,
    materials={"substrate": gfem.Material(eps_r=11.45, tan_delta=1e-6)},
    boundaries=[gfem.PEC("metal"), gfem.Conductor("box", 5.8e7)],
)
problem = gfem.Eigenmode(target=7.5e9, n_modes=2)
options = gfem.SolverOptions(eigen_tolerance=1e-10)


def objective(eps):
    """Squared detuning from 7.6 GHz plus the loss 1 / Q of the first mode."""
    m = model.with_material("substrate", gfem.MaterialField(eps_r=eps, tan_delta=1e-6))
    modes = gfem.solve(m, problem, options=options)
    detuning = modes.frequency[0] / 7.6e9 - 1
    return detuning**2 + 1e3 / modes.q[0]


n = len(mesh.elements("substrate"))
value, grad = jax.value_and_grad(objective)(jnp.full(n, 11.45))
```

- Prefer `1 / q` to `q` in objectives: `q` is infinite for a lossless mode,
  and `|Im omega|` has a kink at zero loss.
- Run gradient-based design at `eigen_tolerance` 1e-10 or below: frequencies
  are only as smooth in the design as the eigensolve is converged.
- An eigenfrequency gradient needs no solve: for a mode `x` with wavenumber
  `k` of the operator `T(k)`, `dk/dp = -xᵀ (∂T/∂p) x / xᵀ T'(k) x`, one
  element-wise integral per mode your objective uses, with `T'(k)` including
  the frequency dependence of conductor surface impedance and of the
  second-order absorbing term.
- Next to a {term}`perfectly matched layer` the eigenfrequency gradients are
  exact ({doc}`perfectly-matched-layer`).
- For a field with `axes` the gradient has the field's shape, one value per
  principal axis ({doc}`anisotropic-materials`).

## Differentiate a degenerate pair

Modes whose frequencies agree to `SolverOptions.cluster_rtol` (1e-6) form a
degenerate cluster, labelled by `result.cluster` (the two modes of a square
cavity have `[0, 0]`). A member's own frequency has a kink where the design
breaks the symmetry, but a function symmetric in the cluster's modes is smooth
and differentiates:

```python
modes = gfem.solve(m, gfem.Eigenmode(target=17e9, n_modes=3), options=options)
pair = modes.frequency[:2]  # modes.cluster == [0, 0, 1]
mean = jnp.mean(pair)  # differentiable
soft_max = jax.nn.logsumexp(10 * pair / 1e9) / 10  # differentiable
first = pair[0]  # its gradient raises DegenerateModeError
```

- gfem accepts the gradient when the derivative of your objective with respect
  to each member's frequency is the same on every member of a cluster, to 1e-3
  relative; the gradient is then that value times the derivative of the
  cluster's sum.
- Otherwise it raises {py:obj}`gfem.DegenerateModeError`: make the objective
  symmetric in the cluster, or break the symmetry in the design (see
  {ref}`gradient-errors`). A cluster cut by `n_modes` also raises, but gfem
  cannot always detect it: ask for enough modes.

## Follow one mode across iterations

Mode indices are ordered by distance from `target`, so two modes can swap
places where their frequencies cross during an optimization. Pick the index of
the physical mode by overlap with the previous iteration's, under
`stop_gradient`:

```python
def tracked(modes, reference):
    """Index of the mode whose field is most like ``reference`` (coefficients)."""
    fields = jax.lax.stop_gradient(jnp.stack([f.coefficients for f in modes.e_fields]))
    overlap = jnp.abs(fields.conj() @ reference) / jnp.linalg.norm(fields, axis=1)
    return jnp.argmax(overlap)


def objective(eps, reference):
    m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
    modes = gfem.solve(m, problem, options=options)
    return modes.frequency[tracked(modes, reference)] / 1e9


# after each step, follow the mode the objective used
modes = gfem.solve(m, problem, options=options)
reference = modes.e_field(int(tracked(modes, reference))).coefficients
```

- Overlaps need the same mesh topology across iterations: material designs and
  moved nodes keep it, a remesh does not.
- Ask for a few more modes than you track, so the mode stays in the result as
  it drifts from `target`.

## Participation ratios and port Q

A mode's {term}`participation ratios <participation ratio>` and port Q are
read from its field and differentiate too. Keep the coupling through port `p1`
and the energy share in a lossy substrate low:

```python
coupled = model.replace(ports=[gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)])


def loss(eps):
    """The first mode's coupling loss plus its substrate loss, as 1 / Q."""
    m = coupled.with_material("substrate", gfem.MaterialField(eps_r=eps))
    modes = gfem.solve(m, problem, options=options)
    return 1 / modes.q_port["p1"][0] + modes.bulk_participation["substrate"][0] * 2e-6


g = jax.grad(loss)(jnp.full(n, 11.45))
```

- These outputs differentiate: `participation` (the junction participation of
  an inductive port), `q_port`, `bulk_participation`,
  `principal_participation` and `sheet_participation`, and the mode fields
  `e_field`, `field.at` and `gfem.interface_participation`, with respect to
  the materials, the sheet fill and the lumped values.
- Each mode read through its field costs one extra linear solve, on the
  factorization the eigensolve already made. The default `eigen_tolerance` is
  enough: gfem polishes each such mode, and above 1e-10 the gradient raises
  `gfem.NonDifferentiableError`.
- Prefer `1 / q_port` to `q_port`: a gradient through an infinite `q_port`
  raises `gfem.NonDifferentiableError`.
- The dressed modes of a junction (anharmonicity, dispersive shift) build on
  the junction participation: see {doc}`quantum-metrics`.

## Other inputs

- The fill of a {py:obj}`gfem.SheetField` differentiates ({doc}`sheet-design`).
- The mesh node coordinates differentiate, with every port surface held where
  it was imported ({doc}`shape-gradient`).
- The {term}`lumped values <lumped value>` that enter only the operator
  differentiate (a `LumpedPort`'s `inductance` and `capacitance`, an
  `Impedance`'s `resistance`, `inductance` and `capacitance`): build the port
  or the sheet inside the differentiated function.
- The numbers of a permittivity law differentiate the eigenfrequency and Q; by
  them, participations, port Q and the mode field raise
  `gfem.NonDifferentiableError` ({doc}`dispersive-materials`).

## Check your results

- Compare the gradient with a central difference of your objective along a
  random direction, at the default `eigen_tolerance`.
- For a junction port with a field uniform across it, the participation is
  `-2 d ln(omega) / d ln(L)`: two solves with the port's inductance changed by
  1 % check it.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **An isolated mode.** A gradient through a mode's field needs a relative
  frequency gap of at least 1e-4 (or `cluster_rtol`) to every other converged
  mode, those beyond `n_modes` included. A degenerate member, a cut cluster, a
  nearer pair, or a mode whose polishing did not converge raises
  `gfem.DegenerateModeError`, also for a symmetric objective: separate the
  modes in the design, or build the objective from frequencies.
- **Mode-field exceptions.** The mode field raises with respect to node
  coordinates (use finite differences over moved meshes, {doc}`loss-budget`),
  and `principal_participation` with respect to a lumped value (use
  `bulk_participation`). Periodic and Bloch models, `linear_solver` other than
  `"direct"`, `device="cuda"`, `result.error` and MPI ranks raise for a
  mode-field gradient; the frequencies still differentiate
  ({doc}`../capabilities`).
- **Wave ports.** A design region touching a wave port face raises
  `gfem.PortGeometryError`; a driven solve differentiates it. A wave port's
  `q_port` raises `gfem.NonDifferentiableError` for a mode with no amplitude
  on the port or degenerate port modes: differentiate `omega` there.
- `Eigenmode.target`, a lumped port's resistance and a `Conductor`'s values
  are not differentiable: keep them constant inside the differentiated function.

## Next

{py:obj}`gfem.Eigenmode`, {py:obj}`gfem.SolverOptions`; {doc}`quantum-metrics`,
{doc}`loss-budget`, {doc}`capacitance-gradient`, {doc}`s-parameter-gradient`,
{doc}`shape-gradient`; {doc}`../capabilities`.
