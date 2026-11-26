# Multimode ports

Get one S-matrix with a row and a column for each guided mode at each end of
a polarization rotator, a mode converter or a taper.
`S["out_TM0", "in_TE0"]`, for example, is the fraction of the TE0 wave
launched at the input that leaves as TM0. Several {py:obj}`gfem.WavePort`s on
one surface, one `mode` each, form a {term}`shared port surface`: one
boundary that absorbs and launches each {term}`requested mode`.

The example is a 500 × 220 nm silicon strip in oxide at 1.55 µm, with TE-like
and TM-like fundamental modes (TE0, TM0), on a mesh with two leads ending on
`port1` and `port2` and a `design` region between them.

## Set up the ports

Put one wave port per mode on each port surface, each with its own name:

```python
import gfem

f0 = 299792458.0 / 1.55e-6  # Hz

mesh = gfem.Mesh.from_gmsh("strip.msh", length_unit="um")
model = gfem.Model(
    mesh,
    materials={
        "cladding": gfem.Material(eps_r=1.444**2),
        "core": gfem.Material(eps_r=3.476**2),
        "design": gfem.Material(eps_r=1.444**2),
    },
    boundaries=[gfem.Absorbing("outer", order=2)],
    ports=[
        gfem.WavePort("in_TE0", "port1", mode=1, pec_edges=("outer",)),
        gfem.WavePort("in_TM0", "port1", mode=2, pec_edges=("outer",)),
        gfem.WavePort("out_TE0", "port2", mode=1, pec_edges=("outer",)),
        gfem.WavePort("out_TM0", "port2", mode=2, pec_edges=("outer",)),
    ],
)
result = gfem.solve(model, gfem.Driven(frequency=f0))
s = result.s
s["out_TE0", "in_TE0"]  # TE0 transmitted as TE0
s["out_TM0", "in_TE0"]  # TE0 converted to TM0
```

Every API keyed by port name works per mode: `Driven(excite=...)`,
`result.port_modes["in_TM0"]`, `s.z_ref`, Touchstone and xarray export, and
the saved job.

- **Each mode is its own port.** S is made of each mode's waves, normalized to
  1 W on a lossless line or pseudo-waves with a `voltage_path`
  ({py:obj}`gfem.conventions`); `s.y()`, `s.z()` and `s.renormalize()` act on
  modal voltages and currents.
- **One surface, one cross-section.** The wave ports of a surface request
  different `mode`s and give the same `pec_edges`; a
  {py:obj}`gfem.LumpedPort` or a boundary condition cannot share it
  (`gfem.Model` raises `ValueError`).
- **Sign of each mode.** `offset`, `voltage_path` and `polarity` belong to each
  port. A `voltage_path` fixes that mode's sign and gives its `z_ref`; without
  one the sign comes from the mesh, so `|S|` and powers are unaffected but the
  phase of a cross-mode entry does not compare between meshes. Give a
  `voltage_path` or a `polarity` (one arrow per port) to compare phases.
- **`mode` is a rank, not a polarization.** It is the mode's rank at the
  lowest frequency of the solve, 1 being the largest effective index. Check
  which rank is TM0 before naming a port after it (here TE0 is rank 1, TM0
  rank 2):

  ```python
  tm0 = gfem.solve(model, gfem.PortMode("in_TM0", f0))
  tm0.n_eff  # the TM-like strip mode has the lower index
  ```

  or look at `tm0.e_field`, which points mostly normal to the wafer for a
  {term}`TM-like mode`.
- **Driven fields.** A kept field is the field for 1 W incident in one
  requested mode, with every other port terminated and the other modes of the
  excited surface absorbed.

## Differentiate a conversion

In a straight strip whose stack is symmetric about its mid-plane, TE0 and TM0
do not couple, so `S["out_TM0", "in_TE0"]` is zero; a device that breaks that
symmetry converts one into the other. Make the device a `gfem.MaterialField`
on a region kept off both port faces and maximize the conversion:

```python
import jax
import jax.numpy as jnp

problem = gfem.Driven(frequency=f0)


def conversion_loss(eps):
    """1 - |S(out_TM0, in_TE0)|^2, for a permittivity per design element."""
    m = model.with_material("design", gfem.MaterialField(eps_r=eps))
    s = gfem.solve(m, problem).s
    return 1 - jnp.abs(s["out_TM0", "in_TE0"]) ** 2


n = len(mesh.elements("design"))
value, grad = jax.value_and_grad(conversion_loss)(jnp.full(n, 1.444**2))
```

The cost rule of {doc}`s-parameter-gradient` applies per mode: with every port
excited (the default) a lossless port line needs no extra solve, and with
`Driven(excite=["in_TE0"])` reading the `out_TM0` row costs one adjoint solve
per frequency ({doc}`adaptive-sweep-gradient` for adaptive sweeps).

## Check your results

- **Reciprocity.** `S` equals its transpose to the solver's tolerance.
- **Power.** For a lossless device with every propagating mode requested,
  `SᴴS` is the identity.
- **Power through a plane.** {py:obj}`gfem.power_flow` through a plane in the
  output lead ({doc}`measurements`) equals `Σ_k |S[out_k, in]|²` over the
  requested modes at that end.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **Unrequested modes are reflected, not converted.** A propagating mode no
  port requests is partly reflected, by about
  $(\beta_j - \beta_0)/(\beta_j + \beta_0)$, and S misses the power the device
  sends into it: request it to see where that goes. A mode ranked below the
  highest requested one and left out warns with
  {py:obj}`gfem.SkippedPortModeWarning`.
- **Mirror half models.** On a {term}`mirror half model` the ranks count only
  the modes the symmetry plane admits: a plane through the strip's
  mid-height admits only one of TE0 and TM0, so requesting both needs the full
  stack.
- **Lossy port lines.** Give every wave port a `voltage_path`, or none: on a
  lossy line the two kinds of waves differ and S is symmetric only between
  ports of one kind, to round-off with alike meshed faces and otherwise to the
  discretization error of the two modes' normalizations (about $10^{-5}$ on a
  coarse lossy microstrip).
- **Degenerate, crossing and cut-off modes** raise {py:obj}`gfem.PortModeError`
  naming the port and surface. For degenerate modes (TE10 and TE01 of a square
  guide) use a {term}`mirror half model` or break the symmetry with
  `WavePort(pec_edges=...)`; for a crossing or a mode below cutoff in the band,
  narrow the band. A clean crossing is followed and warns with
  {py:obj}`gfem.PortModeTrackingWarning`.
- **Eigenmodes and designs.** {py:obj}`gfem.Eigenmode` takes a shared port
  surface, dropping the cross terms between modes in each coupling Q (zero on
  a lossless section); a design on the face changes every requested mode
  ({doc}`port-mode-gradient`).

## Next

{py:obj}`gfem.WavePort`, {py:obj}`gfem.PortMode`, {py:obj}`gfem.conventions`,
{doc}`s-parameter-gradient`, {doc}`measurements`.
