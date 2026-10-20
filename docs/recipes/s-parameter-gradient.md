# Differentiate S-parameters and driven fields

A driven solve is a JAX function of the model's `MaterialField` values like
any other (see [Differentiate a capacitance](capacitance-gradient.md) for the
transformations, the memory knob and `stop_gradient`). `jax.grad` of an
objective built from `result.s` or from kept fields reaches the per-element
`eps_r` (complex where lossy), `mu_r`, `tan_delta` and `conductivity`.

## A transmission objective

```python
import jax
import jax.numpy as jnp

import gfem

mesh = gfem.Mesh.from_gmsh("filter.msh")
model = gfem.Model(
    mesh,
    materials={"substrate": gfem.Material(eps_r=4.4)},
    boundaries=[gfem.PEC("ground"), gfem.Absorbing("outer")],
    ports=[
        gfem.LumpedPort("p1", "port1", "+z"),
        gfem.LumpedPort("p2", "port2", "+z"),
    ],
)
problem = gfem.Driven(frequency=[4e9 + 0.1e9 * k for k in range(21)])


def passband_loss(eps):
    """Mean |S21|^2 deficit over the band, for a permittivity per element."""
    m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
    s = gfem.solve(m, problem).s
    return jnp.mean(1 - jnp.abs(s["p2", "p1"]) ** 2)


n = len(mesh.elements("substrate"))
value, grad = jax.value_and_grad(passband_loss)(jnp.full(n, 4.4))  # grad: (n,)
```

Anything computed from `result.s` in `jnp` differentiates: `s.db()`, `s.y()`,
`s.z()`, `s.renormalize(...)`. The reference impedances `s.z_ref` are the
port resistances, constants of the materials.

## Cost

An S gradient needs no extra solve when every port the objective reads is
excited (the default): by reciprocity each port's solution is its own adjoint,
and the backward pass is one element-wise integral per frequency. Reading
the row of a port left out of `Driven(excite=...)` costs one adjoint solve per
frequency.

A kept field (`Driven(keep_fields=...)`, then `result.e_field(f, port)`)
differentiates too: its cotangent takes one multi-column adjoint solve at that
frequency, reusing nothing from the forward but the mesh and spaces. The
adjoint solve runs at `SolverOptions(adjoint_tolerance=...)`, by default the
forward `tolerance`. Its operator is kept for later pullbacks of the same solve,
within `retain_memory`.

## What raises

The mesh node coordinates, port and boundary parameters (lumped `R`, `L`, `C`,
`Impedance`, `Conductor` conductivity) and the frequencies are read but not
differentiable in this version: a perturbed one raises
`gfem.NonDifferentiableError` while JAX traces the gradient. Thermal
properties are not read: their gradient is an exact zero.

## Wave ports

S differentiates through `WavePort`s, lossy ones included, as long as the
design stays off the port faces. A wave port's mode comes from its cross-section
(the materials of the elements beside the face), and gfem treats it as a
constant of the solve, so a perturbed material whose region has an element on a
wave port face raises `gfem.PortGeometryError` while JAX traces the gradient
(a forward solve of the same model is fine). Give the elements next to each
port a region of their own with a constant `gfem.Material`, and put the design
further in:

```python
model = gfem.Model(
    mesh,
    materials={"lead": gfem.Material(eps_r=4.0), "core": gfem.MaterialField(eps_r=eps)},
    boundaries=[gfem.PEC("ground"), gfem.PEC("signal")],
    ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
)
```

The port modes in `result.port_modes`, and the reference impedances `s.z_ref`
(the ports' `Z_PV`), are then constants: their gradient is an exact zero.

Cost: a lossless port reuses its own solution as the adjoint, as a lumped port
does. A lossy port's mode is complex (conductor or dielectric loss in the
cross-section), and its S row projects on the conjugated mode, which is not its
excitation: each frequency with a lossy port in the objective takes one adjoint
solve (all such rows in one multi-column call). An adaptive sweep solves those
S functionals at each sample along with the excitations and adds them to its
basis, so its frozen-basis gradient is exact at the samples; a lossy port adds
one right-hand side per sample.

## Adaptive sweeps

`Driven(sweep="adaptive")` differentiates too, with its reduced basis held
fixed and its samples frozen (`Driven(samples=...)`): see
[Differentiate an adaptive frequency sweep](adaptive-sweep-gradient.md).
