# Eigenfrequency and Q gradients

Tune a resonator's frequency and Q by differentiating them with respect to the
material in each element. `jax.grad` of an objective built from
`result.omega`, and so from `result.frequency` and `result.q`, reaches the
per-element `eps_r` (complex where lossy), `mu_r`, `tan_delta` and
`conductivity` of every {py:obj}`gfem.MaterialField`, also when the model has
`Conductor` plates or a second-order `Absorbing` boundary. The participation
ratios and the port Q of a mode differentiate too, through its field. Batches,
memory, `stop_gradient` and errors work as for a capacitance: see
{doc}`capacitance-gradient`.

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
- Run gradient-based design at a tight `eigen_tolerance` (1e-10 or below).
  The frequencies are only as smooth in the design as the eigensolve is
  converged.
- An eigenfrequency gradient needs no solve. For a mode `x` with wavenumber
  `k` of the operator `T(k)`, `dk/dp = -xᵀ (∂T/∂p) x / xᵀ T'(k) x`, so the
  backward pass (the part of `jax.grad` that runs after the solve) is one
  element-wise integral per mode your objective uses. `T'(k)` includes the
  frequency dependence of the conductor surface impedance and of the
  second-order absorbing term.
- Next to a {term}`perfectly matched layer` the eigenfrequency gradients
  are exact. The layer's background is not an input: it is a constant
  `Material`, and the layer's stretch is fixed
  ({doc}`perfectly-matched-layer`).
- A differentiated solve refines one mode beyond `n_modes`, to detect a
  degenerate cluster that `n_modes` cuts (see below).
- For a field with `axes` (an {term}`anisotropic material field`) the
  gradient has the field's shape, one value per principal axis; see
  {doc}`anisotropic-materials`.

## Differentiate a degenerate pair

Modes whose frequencies agree to `SolverOptions.cluster_rtol` (1e-6 by
default) form a degenerate cluster, labelled by `result.cluster`: the two
modes of a square cavity, for example, have labels `[0, 0]`. Each member is
then an arbitrary basis vector of the cluster's eigenspace. Its own frequency
has no derivative: it has a kink where the design breaks the symmetry. A
function symmetric in the cluster's modes is smooth and differentiates:

```python
modes = gfem.solve(m, gfem.Eigenmode(target=17e9, n_modes=3), options=options)
pair = modes.frequency[:2]  # modes.cluster == [0, 0, 1]
mean = jnp.mean(pair)  # differentiable
soft_max = jax.nn.logsumexp(10 * pair / 1e9) / 10  # differentiable
first = pair[0]  # its gradient raises DegenerateModeError
```

- gfem accepts the gradient when the derivative of your objective with
  respect to each member's frequency is the same on every member of a
  cluster, to 1e-3 relative. The gradient is then that value times the
  derivative of the cluster's sum.
- Otherwise the gradient raises {py:obj}`gfem.DegenerateModeError`, naming the
  modes, their frequencies and gap; make the objective symmetric in the
  cluster, or break the symmetry in the design (see {ref}`gradient-errors`).
- A cluster that extends beyond the `n_modes` modes you asked for also
  raises, but gfem cannot always detect it: ask for enough modes to hold the
  whole cluster.

## Follow one mode across iterations

Mode indices are ordered by distance from `target`, so during an optimization
two modes can swap places where their frequencies cross. The gradient of
`omega[i]` is right for whichever mode is at `i`, but an objective should
follow the same physical mode. Pick its index by overlap with the mode of the
previous iteration, under `stop_gradient`:

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

- Overlaps compare coefficient vectors, so they need the same mesh topology
  across iterations. Material designs and moved nodes keep it; a remesh does
  not.
- Ask for a few more modes than you track (`n_modes`), so the mode stays in
  the result when it drifts away from `target`.

## Participation ratios and port Q

A mode's {term}`participation ratios <participation ratio>` and the coupling
Q of its ports are read from its field, and they differentiate as well. Keep a
resonator's coupling through port `p1` and the share of its energy in a lossy
substrate low:

```python
coupled = model.replace(ports=[gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)])


def loss(eps):
    """The first mode's coupling loss plus its substrate loss, as 1 / Q."""
    m = coupled.with_material("substrate", gfem.MaterialField(eps_r=eps))
    modes = gfem.solve(m, problem, options=options)
    return 1 / modes.q_port["p1"][0] + modes.bulk_participation["substrate"][0] * 2e-6


g = jax.grad(loss)(jnp.full(n, 11.45))
```

- These outputs differentiate: `participation` (the junction participation
  of an inductive port), `q_port`, `bulk_participation`,
  `principal_participation` and `sheet_participation`. So do the mode fields
  themselves, `e_field`, `field.at` and `gfem.interface_participation`, with
  respect to the materials, the sheet fill and the lumped values.
- Each mode your objective reads through its field costs one extra linear
  solve, on the factorization the eigensolve already made: a fraction of the
  forward solve.
- The default `eigen_tolerance` is enough for participations and their
  gradients: gfem polishes each mode that can take such a gradient. Above
  1e-10 such a gradient raises `gfem.NonDifferentiableError`.
- Prefer `1 / q_port` to `q_port`: a port the mode does not reach has an
  infinite `q_port`, and a gradient through it raises
  `gfem.NonDifferentiableError`.
- A wave port's `q_port` differentiates the same way, with respect to the
  materials away from its face.
- The dressed modes of a junction (anharmonicity, dispersive shift) build on
  the junction participation: see {doc}`quantum-metrics`.

## Other inputs

- The fill of a {py:obj}`gfem.SheetField` on a `Conductor` or `Impedance`
  differentiates: see {doc}`sheet-design`.
- The mesh node coordinates differentiate, with every port surface held
  where it was imported: see {doc}`shape-gradient`. The gradient is exact for
  displacements that hold the port surfaces; a port's voltage path, width and
  capacitance stay as imported, also for a junction port between moving
  pads.
- The {term}`lumped values <lumped value>` that enter only the operator
  differentiate: a `LumpedPort`'s `inductance` and `capacitance` (a
  junction inductance, say) and an `Impedance`'s `resistance`, `inductance`
  and `capacitance`. Build the port or the sheet inside the differentiated
  function.
- The numbers of a permittivity law differentiate the eigenfrequency and Q.
  By them, participations, port Q and the mode field raise
  `gfem.NonDifferentiableError`: differentiate the eigenfrequency or Q. The
  inputs above still reach every output of a model with a law
  ({doc}`dispersive-materials`).

## Check your results

- Compare the gradient with a central difference of your objective along a
  random direction, at the default `eigen_tolerance`. The default is enough
  for frequencies and participations alike.
- For a junction port with a field uniform across it, the participation is
  `-2 d ln(omega) / d ln(L)`: two solves with the port's inductance changed by
  1 % check it.

How gfem itself is checked: {doc}`../verification`.

## Limits

- A gradient through a mode's field needs an isolated mode: a relative
  frequency gap of at least 1e-4 (or `cluster_rtol`, if larger) to every
  other converged mode the solve found, those beyond `n_modes` included. A member of a
  degenerate cluster, a cluster cut by `n_modes` or a nearer pair raises
  `gfem.DegenerateModeError`, also for a symmetric objective. Separate the
  modes in the design, or build that objective from frequencies.
- Mode-field outputs differentiate as the matrix on {doc}`../capabilities`
  shows, the lumped values included, with two exceptions. The mode field
  (`e_field`, `field.at`, `gfem.interface_participation`) raises with respect
  to the node coordinates: take finite differences over moved meshes
  ({doc}`loss-budget`). `principal_participation` raises with respect to a
  lumped value; use `bulk_participation` there.
- Periodic and Bloch models, `linear_solver` other than `"direct"`,
  `device="cuda"` and `result.error` raise
  `gfem.NonDifferentiableError` for a mode-field gradient. The frequencies
  still differentiate there, as {doc}`../capabilities` shows; or wrap the
  value in `jax.lax.stop_gradient` to use it without its derivative.
- With wave ports, a design region touching a wave port face raises
  `gfem.PortGeometryError`: the port's term at a complex eigenfrequency holds
  its mode. A driven solve differentiates such a design. A wave port's `q_port`
  raises `gfem.NonDifferentiableError` for a mode with no amplitude on the
  port, and for degenerate port modes on a shared port surface:
  differentiate `omega` there.
- Across MPI ranks a gradient through a mode's field raises
  `gfem.DistributedUnsupportedError` (feature `mode_field_gradient`) when
  `jax.grad` traces it; frequencies and Q differentiate there
  ({doc}`distributed-solve`).
- A mode whose polishing does not converge is left unpolished and is not
  isolated: a gradient through its field raises `gfem.DegenerateModeError`
  naming it. Separate the modes in the design, or build the objective from
  frequencies.
- `Eigenmode.target`, a lumped port's resistance and a `Conductor`'s values
  are not differentiable. Keep them constant inside the differentiated
  function.

## Next

{py:obj}`gfem.Eigenmode`, {py:obj}`gfem.SolverOptions`; {doc}`quantum-metrics`,
{doc}`loss-budget`, {doc}`capacitance-gradient`, {doc}`s-parameter-gradient`,
{doc}`shape-gradient`; {doc}`../capabilities`.
