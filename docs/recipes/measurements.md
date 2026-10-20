# Measure fields after the solve

A result carries every quantity that needs no extra input. Anything that
needs more physics input (a probe point, a thin interface layer, a voltage
path) is a measurement: a function called on the result, also on a result
reloaded with `gfem.load_result`.

## Interface (surface) loss of a resonator

The thin oxides and residues on a superconducting resonator are too thin to
mesh. `gfem.interface_participation` gives the share of each mode's electric
energy in a layer of thickness `t` and permittivity `eps_r` on a surface,
with the metal-air (`"MA"`), metal-substrate (`"MS"`) and substrate-air
(`"SA"`) models of Wenner et al. (Palace's `Dielectric` postprocessing):

```python
import gfem

modes = gfem.solve(model, gfem.Eigenmode(target=7.8e9, n_modes=2))
ma = gfem.interface_participation(modes, "metal", 3e-9, 10.0, kind="MA", tan_delta=2e-3)
ms = gfem.interface_participation(modes, "metal", 3e-9, 10.0, kind="MS", tan_delta=2e-3)
sa = gfem.interface_participation(modes, "gaps", 3e-9, 10.0, kind="SA", tan_delta=2e-3)
ma.participation  # (n_modes,)
1 / (ma.inverse_q + ms.inverse_q + sa.inverse_q)  # the surface-loss Q
```

On a metal sheet between substrate and vacuum, `"MA"` reads the vacuum face
and `"MS"` the substrate face (its normal flux, with the substrate's
permittivity tensor), so one surface serves both.

## Probes

`field.at(points)` evaluates any result field at points in metres: kept
driven fields, mode fields, terminal potentials, `field.b()`:

```python
result = gfem.solve(model, gfem.Driven(frequency=28e9, keep_fields=True))
e = result.e_field(28e9, excite="p1").at([[0.0, 0.0, 5e-6], [1e-4, 0.0, 5e-6]])
```

A point inside a volume the field does not cover (an excluded metal volume)
raises `ValueError`.

## Characteristic impedance

`gfem.characteristic_impedance(mode, voltage_path)` is the power-voltage
impedance `|V|^2 / (2 P)` of a `gfem.PortMode` result along any path on the
port, when the wave port declares none:

```python
mode = gfem.solve(model, gfem.PortMode("p1", [10e9, 20e9]))
z = gfem.characteristic_impedance(mode, [(0.0, 0.0, 0.0), (0.0, 10e-6, 0.0)])
```

## Gradients

Probes and interface participation are host kernels behind gfem's gradient
wrapper: they run under `jax.jit` and `jax.vmap`, and `jax.grad` flows
through the field coefficients into the solve. A probe objective on a kept
driven field therefore has material-field gradients:

```python
def field_at_probe(eps):
    m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
    result = gfem.solve(m, gfem.Driven(frequency=28e9, keep_fields=True))
    return jnp.sum(jnp.abs(result.e_field(28e9).at(points)) ** 2)


grad = jax.grad(field_at_probe)(eps)
```

Mesh-node coordinates and probe points are fixed: perturbing them raises
`gfem.NonDifferentiableError` (`jax.lax.stop_gradient` is the opt-out).
Mode fields are not differentiable in this version, so a gradient of an
interface participation that reaches the eigenmode solve raises there; the
layer's thickness, permittivity and loss tangent differentiate.
