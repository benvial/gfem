# Request several modes on one port surface

Get one S-matrix with a row and a column for each guided mode at each end of
a polarization rotator, a mode converter or a taper.
`S["out_TM0", "in_TE0"]`, for example, is the fraction of the TE0 wave
launched at the input that leaves as TM0. Several {py:obj}`gfem.WavePort`s on
one surface, one `mode` each, form a {term}`shared port surface`: one
boundary that absorbs and launches each {term}`requested mode`.

The example is a 500 × 220 nm silicon strip in oxide at 1.55 µm, which
carries a TE-like and a TM-like fundamental mode (TE0 and TM0). The gmsh
mesh has two leads of the strip ending on the exterior surfaces `port1` and
`port2`, and a `core` and a `cladding` region. It also has an outer surface
`outer` and, between the leads, the device: a region `design`. Each lead is uniform near its port face, so the port mode problem of
the face describes the guided modes of the lead.

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

- **Each mode is its own port.** S is made of power waves normalized to unit
  modal power ({py:obj}`gfem.conventions`). `s.y()`, `s.z()` and
  `s.renormalize()` act on modal voltages and currents, one per mode, not on
  the conductors' terminal ones.
- **One surface, one cross-section.** The wave ports of one surface must
  request different `mode`s and give the same `pec_edges`; a
  {py:obj}`gfem.LumpedPort` or a boundary condition cannot share the surface
  with them. `gfem.Model` raises `ValueError` otherwise.
- **Per port.** `offset` and `voltage_path` belong to each port. Each mode is
  de-embedded with its own propagation constant. A `voltage_path` fixes that
  mode's sign (`Re V > 0` along the path) and gives its `z_ref`. Without one,
  a mesh-dependent rule fixes the sign of the port's row and column of S.
  `|S|` and every power are unaffected, but the phase of a cross-mode entry
  is then not comparable between meshes: give a `voltage_path` to compare
  phases.
- **`mode` is a rank, not a polarization.** It is the mode's rank at the
  lowest frequency of the solve, 1 being the most confined (the largest
  effective index). Check which rank is TM0 before naming a port after it:

  ```python
  tm0 = gfem.solve(model, gfem.PortMode("in_TM0", f0))
  tm0.n_eff  # the TM-like strip mode has the lower index
  ```

  or look at `tm0.e_field`, which points mostly normal to the wafer for a
  {term}`TM-like mode`. On a shared port surface, `PortMode` runs the
  surface's joint sweep, so it gives the same mode the driven solve uses.
  For this strip TE0 is rank 1 and TM0 rank 2.
- **Driven fields.** A kept field (`Driven(keep_fields=...)`) is the field for
  1 W incident in one requested mode, with every other port terminated. The
  other modes of the excited port's own surface are absorbed (matched) like
  those of the far end.

## Differentiate a conversion

A conversion entry differentiates like any other S entry. In a straight strip
whose stack is symmetric about its mid-plane, TE0 and TM0 do not couple, so
`S["out_TM0", "in_TE0"]` is zero; a device that breaks that symmetry converts
one into the other. Make the device a `gfem.MaterialField` on a region kept
off both port faces and maximize the conversion:

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

The cost rule of {doc}`s-parameter-gradient` applies per mode. With every port
excited (the default), a lossless port line needs no extra solve: each
port's solution is its own adjoint. With `Driven(excite=["in_TE0"])`, reading
the `out_TM0` row costs one adjoint solve per frequency. Adaptive sweeps
differentiate as for one mode per port
({doc}`adaptive-sweep-gradient`).

## Check your results

- **Reciprocity.** `S` should equal its transpose to the solver's tolerance.
- **Power.** For a lossless device with every propagating mode requested,
  `SᴴS` should be the identity: no power goes anywhere S does not show.
- **Power through a plane.** {py:obj}`gfem.power_flow` through a plane in the
  output lead (see {doc}`measurements`) should equal `Σ_k |S[out_k, in]|²`,
  the sum over the requested modes at that end.

How gfem itself is checked on wave ports and shared port surfaces:
{doc}`../verification`.

## Limits

- **Unrequested modes are reflected, not converted.** A propagating mode that
  no wave port on its surface requests is partly reflected at the port, by
  about `(β_j − β₀)/(β_j + β₀)`. Here β_j is its propagation constant and β₀
  that of the surface's lowest-ranked requested mode. It is never converted
  into a requested mode there, so S misses the power the device sends into
  it. Request it to see where that power goes. A mode ranked below the
  highest requested one and left out (a lone `mode=2`, or modes 1 and 3)
  warns with {py:obj}`gfem.SkippedPortModeWarning`. Modes above the highest
  requested rank never warn.
- **Mirror half models.** On a {term}`mirror half model` the ranks count only
  the modes the symmetry plane admits, so a mode of the other parity is not a
  gap. A PMC or PEC plane through the strip's mid-height admits only one of
  TE0 and TM0: requesting both needs the full stack.
- **Lossy port lines.** S reads each mode by the conjugated projection, which
  separates the requested modes exactly only for a lossless port line. With
  loss in the cross-section, one port's S may read part of another port's
  mode (the {term}`mode cross-projection`); above 1e-3 the solve warns with
  {py:obj}`gfem.PortModeProjectionWarning`, which states the size of that
  error in S. Keep the port line lossless near the port.
- **Degenerate, crossing and cut-off modes.** The solve raises
  {py:obj}`gfem.PortModeError`, naming the port and the surface, in three
  cases. A requested mode degenerate with another (TE10 and TE01 of a square
  guide) has no defined field: break the cross-section's symmetry, with the
  mesh or a `pec_edges` wall. Two requested modes that follow the same mode
  through an unresolved crossing: narrow the band or break the symmetry. A
  requested mode below cutoff somewhere in the band: narrow the band. A lossy
  mode counts as below cutoff where its attenuation is at least its phase
  constant, Re β ≤ |Im β|. A clean crossing is followed and warns with
  {py:obj}`gfem.PortModeTrackingWarning`.
- **Eigenmodes.** {py:obj}`gfem.Eigenmode` raises `NotImplementedError` for
  any wave port, lone or shared: find resonances from a `gfem.Driven` sweep.
- **Designs on a port face.** The port modes are constants of the solve: a
  design region with elements on a port face raises
  {py:obj}`gfem.PortGeometryError` under `jax.grad`, naming the surface and
  its wave ports. Keep the design region off the port faces.

## Next

{py:obj}`gfem.WavePort`, {py:obj}`gfem.PortMode`, {py:obj}`gfem.conventions`,
{doc}`s-parameter-gradient` (S objectives, their cost and what raises),
{doc}`measurements` (power through a plane, probes, characteristic
impedance).
