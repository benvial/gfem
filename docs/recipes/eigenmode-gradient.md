# Differentiate eigenfrequencies and Q

Tune a resonator's frequency and Q by differentiating them with respect to the
material in each element. `jax.grad` of an objective built from
`result.omega`, and so from `result.frequency` and `result.q`, reaches the
per-element `eps_r` (complex where lossy), `mu_r`, `tan_delta` and
`conductivity` of every {py:obj}`gfem.MaterialField`, also when the model has
`Conductor` plates or a second-order `Absorbing` boundary. Batches, memory,
`stop_gradient` and errors work as for a capacitance: see
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

## Other inputs

- The fill of a {py:obj}`gfem.SheetField` on a `Conductor` or `Impedance`
  differentiates: see {doc}`sheet-design`.
- The mesh node coordinates differentiate, with every port surface held
  where it was imported: see {doc}`shape-gradient`.

## Check your results

- Compare the gradient with a central difference of your objective along a
  random direction. Set `eigen_tolerance` to about 1e-12 for this check:
  at the design default the eigensolve's own error can hide the agreement.

How gfem itself is checked: {doc}`../verification`.

## Limits

- Only the eigenfrequencies differentiate. A gradient through anything
  computed from the mode fields (`e_field`, `bulk_participation`,
  `participation`, `q_port`, `error`, `gfem.interface_participation`,
  `field.at`) raises `gfem.NonDifferentiableError` when `jax.grad` is called.
  Wrap such a value in `jax.lax.stop_gradient` to use it without its
  derivative.
- `Eigenmode.target` and boundary and port parameters are not differentiable.
  Keep them constant inside the differentiated function.

## Next

{py:obj}`gfem.Eigenmode`, {py:obj}`gfem.SolverOptions`; {doc}`capacitance-gradient`,
{doc}`s-parameter-gradient`, {doc}`shape-gradient`; {doc}`../capabilities`.
