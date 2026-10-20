# Differentiate eigenfrequencies and Q

An eigenmode solve is a JAX function of the model's `MaterialField` values
like any other (see [Differentiate a capacitance](capacitance-gradient.md) for
the transformations, the memory knob and `stop_gradient`). `jax.grad` of an
objective built from `result.omega`, and so from `result.frequency` and
`result.q`, reaches the per-element `eps_r` (complex where lossy), `mu_r`,
`tan_delta` and `conductivity`, also when the model has `Conductor` plates or
a second-order `Absorbing` boundary.

## A frequency and loss objective

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

Prefer `1 / q` to `q` in objectives: `q` is infinite for a lossless mode, and
`|Im omega|` has a kink at zero loss. Run gradient-based design at a tight
`eigen_tolerance` (1e-10 or below): the frequencies are only as smooth in the
design as the eigensolve is converged, and finite-difference checks of
eigenfrequency gradients need about 1e-12.

## Cost

An eigenfrequency gradient needs no solve. For a mode `x` with wavenumber `k`
of the operator `T(k)`, `dk/dp = -xᵀ (∂T/∂p) x / xᵀ T'(k) x`, so the backward
pass is one element-wise integral per mode with a nonzero cotangent. `T'(k)`
includes the frequency derivative of the conductor surface impedance and of
the second-order absorbing term. The forward keeps only the mode fields; a
differentiated solve also refines one guard mode beyond `n_modes` (see below).

## What raises

Only the eigenfrequencies differentiate in this version. A cotangent on
anything computed from the mode fields (`e_field`, `bulk_participation`,
`participation`, `q_port`, `error`, `gfem.interface_participation`,
`field.at`) raises `gfem.NonDifferentiableError` while JAX traces the
gradient; wrap such a value in `jax.lax.stop_gradient` to use it without its
derivative. The mesh node coordinates, `Eigenmode.target` and boundary and
port parameters are not differentiable either; the fill of a
`gfem.SheetField` on a `Conductor` or `Impedance` is (see
[Design a patterned metal sheet](sheet-design.md)).

## Degenerate modes

Modes whose frequencies agree to `SolverOptions.cluster_rtol` (1e-6 by
default) form a degenerate cluster, labelled by `result.cluster`: the two
modes of a square cavity, for example, have labels `[0, 0]`. Each member is
then an arbitrary basis vector of the cluster's eigenspace, and its own
frequency has no derivative (it has a kink where the design breaks the
symmetry). A function symmetric in the cluster's modes is smooth and
differentiates:

```python
modes = gfem.solve(m, gfem.Eigenmode(target=17e9, n_modes=3), options=options)
pair = modes.frequency[:2]  # modes.cluster == [0, 0, 1]
mean = jnp.mean(pair)  # differentiable
soft_max = jax.nn.logsumexp(10 * pair / 1e9) / 10  # differentiable
first = pair[0]  # its gradient raises DegenerateModeError
```

gfem decides from the cotangents: when they are equal on every member of a
cluster (to 1e-3 relative), the gradient is that cotangent times the
derivative of the cluster's sum, from a basis-invariant trace formula;
otherwise the backward pass raises `gfem.DegenerateModeError`, naming the
modes, their frequencies and gap, and the remedies. The error is raised on the
host during the backward pass, so under `jax.grad` it reaches you as a
`jax.errors.JaxRuntimeError` carrying that message. It also raises when the
cluster extends beyond the `n_modes` modes you asked for (the guard mode shows
it): ask for enough modes to hold the whole cluster.

## Mode tracking

Mode indices are ordered by distance from `target`, so during an optimization
two modes can swap places where their frequencies cross. That is not a
gradient issue (the gradient of `omega[i]` is right for whichever mode is at
`i`), but an objective should follow the same physical mode. Pick its index by
overlap with the mode of the previous iteration, under `stop_gradient`:

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

Overlaps compare coefficient vectors, so they need the same mesh topology
across iterations (material designs and moved nodes keep it; a remesh does
not). Ask for a few more modes than you track (`n_modes`), so the mode stays
in the result when it drifts away from `target`.
