# How gfem is checked

Each claim on {doc}`capabilities` and in the recipes is backed by a test that
solves a case with a known answer. The answer is a closed form, an identity
the physics must obey, an independent method, or finite differences for a
gradient. This
page lists those cases by area, in physical terms, and links the test files
on GitHub. The tolerance each check asserts is in its test, not on this page.

Most checks run on every change to gfem. Cases too large for that run before
each release, and the PETSc and SLEPc checks run twice a week; each section
says which.

## Gradients

Every FD-verified mark of the gradient matrix on {doc}`capabilities` is a set
of cells, one per problem, input, linear solver and sweep. Each cell compares
the gradient with fourth-order central differences of the same objective,
with tight solver tolerances. It then checks, at the default tolerances, that
the Taylor remainder shrinks at second order as the step shrinks.

```{gfem-gradient-checks}
```

The inputs are the per-element permittivity and conductivity (isotropic, and
as principal values on tilted axes), the permeability and the loss tangent
(driven and eigenmode problems), the fill of a sheet, the mesh node
coordinates moved through `gfem.shape.boundary_morph`, the parameters of a
permittivity law (its high-frequency permittivity, strengths and
frequencies) and the density of a metal-dielectric blend. The problems are
capacitance and the electric energy of each region, driven S on lumped ports, on wave ports (with the design away from the port
faces and on them) and between modes of a shared port surface, a port mode's
propagation constant, impedance and field, driven S on periodic and Bloch cells,
eigenfrequency and Q, and, on the direct solver, the participation ratios,
port Q and mode field of an isolated mode and the dispersive shift and
anharmonicity of a junction's dressed modes. The heat cells differentiate a
driven field's dissipated power and the temperature its heat gives, with
respect to the lossy materials, a sheet's fill and resistance, the drive and
the thermal conductivity, and the steady heat flow and the transient heat
out, with respect to the thermal properties and the heat.

An exact zero is checked to be zero. Each "raises" mark, and each row of
"Gradients that raise", is a call that must raise the named error. Both the
matrix and that list are built from these tests when the docs are built, so
they cannot drift from them.

The same cells run on `linear_solver="petsc"` and `eigen_solver="slepc"`
where the {term}`PETSc stack` is installed (see
[PETSc and SLEPc](#verification-petsc)), so the matrix does not list PETSc.
The cells between modes of a shared port surface are among the large cases
that run before each release; the other cells run on every change.

A driven optimization on `linear_solver="direct"` that solves designs on an
earlier design's factorization is checked against factorizing every design:
S agrees to a twentieth of `tolerance` and the gradient to `tolerance`.

Thermal and DC conduction designs on `linear_solver="direct"` reuse plans and
factors across designs as driven ones do, and agree bit for bit with
`retain_memory=0` at one solve thread.

Tests: [gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py),
[gradients that raise](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix_raise_table.py),
[reused factorizations](https://github.com/benvial/gfem/blob/main/tests/verification/test_factor_reuse_gradient.py).

## Low frequency

The {term}`tree-cotree gauge` is checked on a case whose S has a closed form
far below the low-frequency floor: a parallel-plate line between two 50 Ω
lumped ports, its S from the line's ABCD matrix. Its mesh's floor is about
95 kHz.

- **At 10 kHz** ($k_0 h_\mathrm{min} \approx 10^{-7}$), the gauged direct solve
  gives S within $10^{-13}$ of the closed form (measured:
  $4 \times 10^{-15}$), with $S_{12} = S_{21}$ to $10^{-14}$. The same solve
  with `gauge="none"` is off by more than $10^{-10}$ (measured:
  $2 \times 10^{-8}$).
- **At 1 MHz and 100 MHz**, the gauged S is within $10^{-13}$ of the closed
  form too.
- **The adjoint gradient** of $S_{11}$ with respect to the permittivity at
  10 kHz matches the derivative of the closed form to $10^{-10}$, relative.
- **An adaptive sweep** through the gauge keeps the same digits below the
  floor, and above it matches the uniform gauged sweep to the sweep
  tolerance.

A model with wave ports gives the same S gauged and ungauged above the
floor. Below it, the wave ports' mode problem is not gauged, and the solve warns.

Tests: [gauged solves](https://github.com/benvial/gfem/blob/main/tests/unit/test_gauge_solve.py),
[adaptive sweep](https://github.com/benvial/gfem/blob/main/tests/unit/test_gauge_adaptive.py),
[gauge basis](https://github.com/benvial/gfem/blob/main/tests/unit/test_gauge_basis.py).

## Wave ports and shared port surfaces

- A matched parallel-plate line transmits its TEM mode, and its first higher
  mode, with their analytic propagation.
- An open dielectric guide between two wave ports transmits its fundamental
  mode with its slab's effective index, almost without reflection, and S is
  passive and reciprocal.
- A rectangular guide with a dielectric slab against one wall converts TE10
  into TE20. S, with both modes on each end's shared port surface, matches an
  independent mode-matching solution. It is unitary and reciprocal, and the
  power through each lead equals the sum of the requested modes' |S|².
- Before each release: TE0 and TM0 of a 500 × 220 nm silicon strip, requested
  on one port surface at each end, are each transmitted with their own phase,
  without reflection or conversion.
- **Gradients of port modes.** The derivatives of a port mode's $\gamma$ and
  $Z_{PV}$ meet closed forms written with the discrete mode's own values, which
  hold on any mesh: a rectangular guide filled with a lossy dielectric,
  $\partial\gamma/\partial\varepsilon_r = -c\,k_0^2/(2\gamma)$ with
  $c = 1 - i\tan\delta$; the TEM mode of a parallel-plate line,
  $\partial\gamma/\partial\varepsilon_r = \gamma/(2\varepsilon_r)$ and
  $\partial Z_{PV}/\partial\varepsilon_r = -Z_{PV}/(2\varepsilon_r)$; a
  uniform stretch of a guide's face; and a sheet, whose fill and resistance
  enter only as their ratio. On a guide loaded by a dielectric slab, the
  derivative of the transverse-resonance root is reached as the element order
  rises.
- **Designs at a port face.** With the design on the port faces, driven S, the
  port modes, `z_ref`, port waves and kept fields match finite differences for
  every input. The inputs are the permittivity, conductivity, loss tangent and
  permeability (isotropic and on axes), a permittivity law, a density blend, a
  sheet's fill and an `Impedance`'s values, and node coordinates moved in and
  across the face's plane. The cases cover lossless and lossy lines, a lone
  port and a shared port surface, a voltage path, an absorbing wall beside the
  face and a face cut by a perfectly matched layer, on the direct and
  BDDC + GMRES solvers and the adaptive sweep. S's gradient does not depend on
  which ports are excited, and a driven solve's $\partial\gamma$ and
  $\partial Z_{PV}$ equal a `gfem.PortMode` solve's.
- **Moved port faces.** When a guide's faces stretch, the gradient's volume part
  and its port-mode part each match a difference taken with the other frozen,
  although the two nearly cancel. Between the frequencies a port mode sweep
  solves exactly, the gradient matches differences of exact solves. Two port
  modes $2\times10^{-3}$ apart keep exact gradients.

Tests: [shared port surface](https://github.com/benvial/gfem/blob/main/tests/verification/test_shared_port_surface.py),
[wave ports](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_wave_ports.py),
[silicon strip](https://github.com/benvial/gfem/blob/main/tests/reference/test_shared_port_surface.py),
[port mode gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_port_mode_gradient.py),
[shape gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_shape_gradient.py),
[recipe](https://github.com/benvial/gfem/blob/main/tests/unit/test_docs_port_mode_gradient_recipe.py).

## SAX circuits

- A wave port's reference plane moved inside the slab equals the plain slab
  followed by SAX's straight guide with the port mode's index, to $10^{-12}$.
- An RF block followed by SAX's series capacitor equals scikit-rf's cascade of
  the same two networks, to $10^{-12}$, on a reference of 50 Ω and of 75 Ω; a
  new reference impedance costs no solve.
- The gradient of a circuit, with respect to a gfem input and a SAX input in
  one `jax.grad` (an interferometer's cross power; an RF block's permittivity
  and the circuit's reference impedance), matches fourth-order central
  differences to a relative $10^{-6}$, and the same under `jax.jit`.
- A 3D E-plane bend, rotated by quarter turns and mirrored with its polarity
  reference (or voltage path), keeps $S_{21}$ to $10^{-8}$; an interferometer
  of a bend and its quarter turn sends all its power to the cross output, the
  bar output carrying less than $10^{-8}$ of it.

Tests: [circuits](https://github.com/benvial/gfem/blob/main/tests/unit/test_sax_circuits.py),
[polarity](https://github.com/benvial/gfem/blob/main/tests/verification/test_circuit_polarity.py),
[recipe](https://github.com/benvial/gfem/blob/main/tests/unit/test_docs_sax_recipe.py).

## Group index and dispersion

A port mode's group index $n_g$, its dispersion $\beta_2$ and $\beta_3$ and
the dispersion parameter $D$ are checked against closed forms, against
differences of exact solves and against the sweep's own exact solves:

- **Uniformly filled guides.** In a rectangular guide filled with air, a
  constant dielectric, or a lossless or lossy Lorentz or Drude medium, and on
  a parallel-plate line in both polarizations with a lossy Lorentz or a
  conductive fill, the discrete mode obeys the filled guide's dispersion
  relation with the mesh's own cutoff. $n_g$, $\beta_2$ and $\beta_3$ match
  that closed form to rounding, complex with loss.
- **Under the element order.** The TE0 and TM0 modes of a 220 nm silicon
  slab in oxide with Sellmeier laws, on a line port, converge to values
  computed to 40 digits from the slab's dispersion relation, each order at
  least five times closer. So do an empty rectangular guide's TE10 and a guide
  with a dielectric slab against one wall. On the lossless slab the
  reference group index is also the stored energy over the power.
- **Sections without a closed form.** On guides with copper walls, thick and
  thin, between lossy parallel plates, beside an absorbing end next to a
  Lorentz medium and on a strip cut by a perfectly matched layer, the values
  match central differences across three exact solves $10^{-4}$ apart in
  relative frequency.
- **Sweeps.** Rows served by the reduced port mode model match exact solves
  at the same frequencies, on a slab guide, a hybrid section and a coplanar
  line over 1-100 GHz, including rows that fall back to an exact solve for
  their dispersion. At one solve thread, a port mode solve and a driven
  solve on the same grid give the same propagation constants bit for bit:
  computing the dispersion leaves the modes unchanged. On a quasi-TEM line
  near DC, $\beta_2$ is limited by noise of about $10^{-3}$ relative.
- **Results.** $D$ follows from $\beta_2$; a driven solve's port modes carry
  none; the values survive saving, loading, `to_xarray` and `jax.jit`, and
  `jax.grad` through them raises while `gamma` of the same result
  differentiates. On two MPI ranks they agree with one process.

Two comparisons are observations, not checks, and live on
{doc}`recipes/waveguide-dispersion`: a 450 × 220 nm silicon wire's group
index against a measured one, and a 500 × 220 nm wire's $\beta_2$ against a
published simulation, which agree to about 10 %.

Not checked: a fabricated guide's measured dispersion, and modes at an exact
degeneracy (refused). All of these run on every change, the MPI case with
the MPI tests.

Tests: [unit checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_port_mode_dispersion.py),
[slab and order](https://github.com/benvial/gfem/blob/main/tests/verification/test_port_mode_dispersion.py),
[rectangular guides](https://github.com/benvial/gfem/blob/main/tests/verification/test_port_mode_convergence.py),
[sweeps](https://github.com/benvial/gfem/blob/main/tests/verification/test_port_mode_sweep.py),
[MPI](https://github.com/benvial/gfem/blob/main/tests/mpi/test_wave_ports.py).

## Kerr nonlinearity

A port mode's nonlinear coefficient $\gamma_\mathrm{NL}$ and the coupling
tensor between its waves are checked against closed forms, against identities
that hold to rounding and against finite differences:

- **Closed forms, and the element order.** The TE10 mode of a filled
  rectangular guide obeys
  $\gamma_\mathrm{NL} = \tfrac32\,(k_0 n_2/ab)\,(n/n_\mathrm{eff})^2$, with the
  discrete $n_\mathrm{eff}$; and the TE0 mode of a 220 nm silicon slab in
  oxide with line ports has a value computed to 30 digits. The error
  falls by at least a factor of 5 for each element order, and is below
  $10^{-10}$ at order 5.
- **Every row.** On a filled guide the coefficient of each row of a 41-row
  grid, reduced rows included, follows $(f_i/f_0)(n_{\mathrm{eff},0}/n_{\mathrm{eff},i})^2$
  to $10^{-10}$, with a lossless Lorentz fill too. A row of a sweep that the
  reduced model serves matches an exact solve at its frequency.
- **Two-photon absorption.** $\mathrm{Im}/\mathrm{Re} =
  \beta_\mathrm{TPA}/(2k_0 n_2)$ to $10^{-12}$; with $n_2 = 0$ the real part
  vanishes. On a lossy fill the value is the lossless one times
  $(n_\mathrm{eff}/\mathrm{Re}\,n_\mathrm{eff})^2$.
- **The coupling tensor.** Its diagonal is $\gamma_\mathrm{NL}$ row by row; it
  is symmetric in its two $+\omega$ slots, obeys the energy identity
  $\gamma_{abcd}/\omega_a = \overline{\gamma_{badc}}/\omega_b$, and for one
  mode at one frequency has 16 equal entries, so the cross-phase sum is
  twice the self-phase. Cross-phase modulation of TE10 by TE01 in a filled
  guide equals $\tfrac49\,n_m/n_n$ of the self-phase.
- **A silicon wire.** The 500 × 220 nm strip at three wavelengths gives
  a real tensor, in the gauge where the transverse field is real.
- **Gradients.** $\gamma_\mathrm{NL}$ differentiates with respect to
  $\varepsilon_r$, $n_2$ and $\beta_\mathrm{TPA}$ of a density blend, matching
  fourth-order central differences to $10^{-6}$; the value under
  `jax.grad` is the eager one.
- **Quadrature and layers.** The field's quadrature columns reproduce the
  field at their points, sum to the section's area, and a rule two degrees
  higher gives the same value; a perfectly matched layer's triangles
  carrying $n_2$ change nothing.
- **Refusals.** A region without a property, an anisotropic region, an Hz line
  port, traced node coordinates, a result without a field and waves of
  different ports each raise the error that names them.

Not checked: a measured nonlinear coefficient of a fabricated guide, and any
literature value: the silicon and nitride constants are inputs. The recipe
{doc}`recipes/waveguide-nonlinearity` prints the value of a silicon wire at
1.55 µm beside the figure of merit it implies.

Tests: [unit checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_kerr.py),
[closed forms, orders and gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_kerr.py),
[recipe](https://github.com/benvial/gfem/blob/main/tests/unit/test_docs_waveguide_nonlinearity_recipe.py).

## Second-order nonlinearity

The overlap $\kappa$ of a pump mode and its second harmonic, the normalized
efficiency $\eta_\mathrm{norm} = |\kappa|^2$, the phase mismatch and the
poling period are checked against closed forms, against identities and against
a finite difference:

- **Closed forms, and the element order.** The TE10 mode of a filled
  rectangular guide at $\omega$ coupled to its TE10 at $2\omega$ through one
  entry of $d$ has $\eta_\mathrm{norm} = 2k_0^2d^2Z_0/(n_{\mathrm{eff},\omega}^2
  n_{\mathrm{eff},2\omega}A_\mathrm{eff})$ with $A_\mathrm{eff} = (9\pi^2/128)\,ab$
  and the discrete effective indices; the TE mode of a filled slab on a line
  port of a 2D model has the same form per unit height. The error falls with the
  element order, as the fields' does, and the phase mismatch and the poling
  period $2\pi/\Delta\beta$ are those of the discrete indices.
- **Selection rules.** The pump's TE10 squared drives only the odd TE$m$0 modes
  at $2\omega$, with the overlap $-4a/(\pi m(m^2-4))$ of the sine profiles: the
  TE20 mode at $2\omega$ is uncoupled, to rounding.
- **Gauge.** On a lossless section the overlap of the closed-form cases is
  real: $\mathrm{Im}\,\kappa$ vanishes to rounding in the gauge where the
  transverse field is real.
- **The tensor between waves.** It is symmetric in its two amplitudes, and the
  entry of the second harmonic is `second_harmonic`'s $\kappa$.
- **The efficiency.** `efficiency` has the sinc$^2$, lossy and
  quasi-phase-matched forms, finite at zero mismatch, and its limits at zero
  loss.
- **Gradients.** The overlap differentiates with respect to $d$, matching a
  central finite difference, with the step and the tolerance derived from the
  round-off and the truncation of the difference.
- **Refusals.** A region without the tensor, a tensor beside a stiffness, an
  Hz line port, traced node coordinates, a result without a field, waves of
  different ports and a left-handed frame each raise the error that names them.

Not checked: a measured conversion efficiency, and any literature value of $d$
or of a device's efficiency: the tensor is an input. The recipe
{doc}`recipes/second-harmonic-generation` prints the figures of a thin-film
lithium niobate ridge beside the ideal theory of the paper it follows, as an
observation.

Tests: [unit checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_three_wave.py),
[closed forms, orders and gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_three_wave.py),
[recipe](https://github.com/benvial/gfem/blob/main/tests/unit/test_docs_second_harmonic_recipe.py).

## Eigenmodes with wave ports

Each case solves an eigenmode problem with wave ports and compares it with an
exact root or with a driven sweep of the same model:

- **Guides with exact roots.** A rectangular guide shorted at one end, with
  layers filling its whole section and a wave port at the other end, has
  resonances that are the roots of one transmission-line equation in
  complex frequency. The eigenfrequency converges at element orders 2 to 4,
  at Q 5.5 and Q 869, by at least ten times per order, to within $10^{-4}$
  at order 3 and $3\times10^{-7}$ at order 4. Within $10^{-4}$ at order 3
  are also a resonance just above the port mode's cutoff, a guide with a
  port at each end, both mode families of a shared port surface carrying
  TE10 and TE20 (each told apart by its field's symmetry), a lossy
  anisotropic port section and a coaxial line (TEM, in closed form).
- **The coupling Q.** With no other loss, the wave ports' $\sum_p 1/q_p$
  matches $1/Q$ to $O(1/Q^2)$, and two ports split the loss as the exact
  powers out of each end do. A mirror-symmetric guide gives its two ports
  the same coupling Q.
- **Against a driven sweep.** A pole fitted to a driven sweep through the
  same ports, on the same mesh, sits at the eigenfrequency, to the port mode
  tolerance and the fit's own bias: within $10^{-12}$ on a single port at
  Q 869, $5\times10^{-8}$ on the shared port surface's TE20 resonance, and
  $3\times10^{-7}$ on a guide partly filled with a dielectric, whose hybrid
  port mode has no closed form.
- **A line loaded by a guide.** The "Solve a device" recipe's line, open at
  one end and with a wave port at the other, resonates at the roots of
  $Z_\mathrm{in}(f) + Z_0 = 0$, and its `q` and `q_port` match the
  transmission line's.
- **Gradients.** The eigenfrequency and the wave port's coupling Q match
  finite differences with respect to the materials away from the port face.
- **Refusals.** A target below a port mode's cutoff raises, a real root
  below cutoff is dropped with a warning, and the unsupported combinations
  raise by name.

Tests: [exact roots and driven cross-check](https://github.com/benvial/gfem/blob/main/tests/verification/test_eigenmode_wave_ports.py),
[unit](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode_wave_ports.py),
[refusals](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode_wave_port_refusals.py),
[recipe](https://github.com/benvial/gfem/blob/main/tests/unit/test_docs_solve_recipe.py).

## 2D models

Each case solves a {term}`2D model` and compares it with a closed form, with
the same plane posed as a {term}`slab <Slab model>`, or with finite
differences:

- **A loaded line, both polarizations, both kinds of wall.** The topology
  tutorial's line in top view, a section of $\varepsilon_r = 10$ between air feeds,
  transmits and reflects as the closed-form three-part transmission line. Its
  TEM mode is checked in Ez between magnetic walls and in Hz between conducting
  walls; the mode that varies as $\sin(\pi y / w)$ across the width, above its cutoff,
  in Ez between conducting walls and in Hz between magnetic walls. A straight
  guide passes its mode with no reflection, and a conducting wall reflects the
  electric field with $-1$ in both polarizations.
- **Guide modes.** The {term}`line port <Line port>` across the bend tutorial's
  guide finds the effective index of the symmetric slab guide's closed form,
  even mode with E along z (Ez) and with H along z (Hz), at 1.50, 1.55 and
  1.60 µm.
- **Agreement with the slab.** The bend, with its design square all cladding or
  all core, gives the same S as a slab one cell thick between PEC plates in Ez.
  In Hz the bend solves exactly the plain $H_z$ equation with the line port's
  boundary term; the slab, between magnetic plates, solves it with its own port
  term, and agrees with that problem instead. The topology line's S agrees with
  the slab's lumped ports matched to the line and with the closed form, empty
  and with a uniform ceramic section. The field of the 2D model is
  $\sqrt{h}$ times the slab's, for a slab $h$ thick fed 1 W.
- **Gradients.** Gradients of S by per-element material fields (an index blend
  with a Drude end in Ez, constant permittivities in Hz, a loss tangent and a
  permeability) and by the pixels of a layer grid at element orders 2 and 3
  match central differences, with a Taylor remainder of slope 2. The pixels'
  permittivity, read inside each triangle, is their bilinear interpolation. A
  design area that owns a second-order absorbing line is differentiated too.
  The {doc}`tutorials/metal-bend` bend's band objective, through gold's law and
  the index blend, matches a difference on both the 2D model and the slab.
  A design that reaches a line port differentiates too, in Ez and Hz, lossless
  and lossy: the port's $\gamma$ and field match finite differences, and a
  uniform line its closed form
  $\partial\gamma/\partial\varepsilon_r = i k_0 c/(2\sqrt{c\,\varepsilon_r})$.
- Before each release, on that bend's 45° gold mirror: the tutorial mesh at
  element order 3 gives the S of the same 2D model on cells eight times
  smaller at order 2, a value computed once and frozen; and the band
  objective's gradient with respect to the pixels points the same way on the
  2D model and on the slab, though the two read the pixels at different
  points.

Tests: [closed forms and the slab](https://github.com/benvial/gfem/blob/main/tests/verification/test_2d_models.py),
[guide modes](https://github.com/benvial/gfem/blob/main/tests/verification/test_2d_line_port_modes.py),
[material-field gradients](https://github.com/benvial/gfem/blob/main/tests/unit/test_2d_gradients.py),
[layer-grid gradients](https://github.com/benvial/gfem/blob/main/tests/unit/test_2d_layer_gradients.py),
[line port gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_line_port_gradient.py),
[metal bend](https://github.com/benvial/gfem/blob/main/tests/verification/test_metal_bend.py),
[metal bend, fine mesh and slab gradient](https://github.com/benvial/gfem/blob/main/tests/reference/test_metal_bend_fine.py).

## Periodic cells

Each case compares a periodic solve with a closed form or a converged reference:

- **A Bloch two-port line.** A lossy parallel-plate line periodic along x,
  with two 50 Ω lumped ports, gives the S of the closed-form Bloch impedance
  matrix at +k, −k and 0, on the direct and BDDC + GMRES solvers. S at +k is
  far from the closed form at −k, which fixes the sign of the Bloch phase.
- **The same line, loaded.** Loaded by its two ports, the line is a damped
  Bloch eigenmode problem. Its complex resonance is the root of
  det(Z(f, k) + R) = 0.
- **An empty parallel-plate cell periodic along x and y.** Its four lowest
  bands are those of the empty lattice and are real. The lowest band's field
  advances by exp(−i k·Δr) between two points.
- **A layered (Bragg) line.** Its bands are the transfer-matrix roots,
  lossless and lossy, at both edges of the gap at k = π/a, at π/2a, and at
  k = 0 with plain periodicity.
- **An empty cube periodic along x, y and z.** Its lowest band is the
  light-line doublet c|k|/2π.
- **Band diagrams.** The empty square lattice's bands and band group
  velocities along Γ-X-M-Γ are the empty-lattice values, with the bands
  crossing and the static band filled in at Γ. A square lattice of alumina
  rods gives its Ez and Hz bands at Γ, X and M, its band group velocities and
  its Ez gap as converged two-dimensional finite-element and plane-wave
  solutions do. Every wave vector of a diagram equals the same wave vector
  solved alone, and a lossy crystal's sweep finds every band.

Further checks: a wave vector of (0, 0, 0) gives the solution with no wave
vector, plain periodic walls give the same S as PMC walls on a field uniform
across the pair, and S(k)ᵀ = S(−k) ({term}`Bloch reciprocity`). On PETSc,
the plain line's S matches the direct solver's, SLEPc finds the Bragg line's
bands, and a Bloch operator is refused.

Tests: [driven](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_driven.py),
[bands](https://github.com/benvial/gfem/blob/main/tests/verification/test_bands.py),
[eigenmode](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_eigen.py),
[identities, driven](https://github.com/benvial/gfem/blob/main/tests/unit/test_periodic_driven.py),
[identities, eigenmode](https://github.com/benvial/gfem/blob/main/tests/unit/test_periodic_eigen.py),
[PETSc](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_petsc.py).

## Floquet ports

Two cells with Floquet ports on both open faces, solved against references
that do not use gfem:

- **A dielectric slab on a substrate, under air.** Its reflection and
  transmission, complex, match the tangential-E transfer-matrix (Airy) form
  for TE and TM: at normal incidence, at conical incidence (40° from the
  normal, azimuth 30°) under a Bloch wave vector, and at a fixed angle over a
  band of three frequencies. The cross-polarized entries vanish, and each
  column's efficiencies add up to 1.
- **The same slab between air on both sides, at Brewster's angle.** Its TM
  reflection vanishes.
- **A lamellar grating.** The diffraction efficiency of every propagating
  order, reflected and transmitted, in both polarizations, matches a
  rigorous coupled-wave (RCWA) solution computed with many harmonics: at
  normal incidence, at 20°, at conical incidence (20°, azimuth 30°) and with
  a lossy ridge, whose missing power matches the RCWA absorption. In the
  lossless cases the efficiencies add up to 1, and the power flowing into
  the substrate, measured on the field, equals the transmitted efficiencies.
- **Reference planes.** Moving the open faces away from the slab, with the
  ports' offsets grown by the same distance, leaves S unchanged.

Identities checked on coarse cells: a flat glass-air interface reflects
(n − 1)/(n + 1) in both polarizations; a cell closed by a PEC wall reflects
all power; the full S of the grating obeys {term}`Bloch reciprocity` in its
Floquet form, with the order relabelling and the signs σ, at normal and
conical incidence, and fails it without the signs; a zero wave vector gives
the plain periodic S; a fixed-angle sweep gives the S of solves at each
frequency's own wave vector; S is continuous through a
{term}`Rayleigh anomaly`; the combination (S[:, TM] + i S[:, TE]) / √2
turns right-handed for incidence from the top face and left-handed from the
bottom one; and the BDDC + GMRES solver and frequency workers agree with the
serial direct solve. Gradients of the efficiencies through Floquet ports,
plain and Bloch, with respect to the permittivity and the node coordinates,
are in the gradient matrix ({doc}`capabilities`). All of these run on every
change to gfem.

Tests: [Floquet ports](https://github.com/benvial/gfem/blob/main/tests/verification/test_floquet_driven.py),
[identities](https://github.com/benvial/gfem/blob/main/tests/unit/test_floquet_driven.py),
[orders](https://github.com/benvial/gfem/blob/main/tests/unit/test_floquet_orders.py).

## Periodic surfaces

The workflow built on Floquet ports (cell outputs, angle sweeps, emissivity,
cell responses, aperture fields) is checked against closed forms and exact
identities, on cells small enough to solve in seconds. No frozen data is
added beyond the grating's RCWA reference.

- **Cell outputs.** On the slab cells of the Floquet ports, the Jones matrix
  from the bottom to the top face and the reflection Jones matrix of the top
  face match the Airy entries at normal, conical and Brewster incidence
  (about $10^{-6}$ in the entries), with vanishing cross-polarized entries,
  and the efficiencies are the squared moduli. A uniaxial slab with its optic
  axis in the plane, a wave plate with a Fabry-Perot amplitude mismatch,
  matches the per-axis Airy form rotated by the axis angle to about
  $10^{-7}$, in Jones, circular and Mueller form. The Mueller matrix is also
  compared with the one measured from Stokes vectors for horizontal,
  vertical, diagonal and circular inputs, 45° light on the wave plate
  turns into the closed-form mix of right- and left-handed light from either
  face, and the Pancharatnam-Berry phase of the cross-handed entry grows as
  $+2\psi$ from the top face and $-2\psi$ from the bottom. A PEC mirror gives
  $\mathrm{diag}(1,1,-1,-1)$ and an empty cell the identity. The power
  balance (reflectance, transmittance, absorptance and the power coupled to
  lumped or wave ports of a fed cell) adds to 1 to rounding for every
  polarization, and refuses a cell with an unreported propagating order.
- **Angle sweeps.** One sweep over normal, oblique and conical incidences of
  the slab gives, for each member, the Airy Jones matrix at its own wave
  vector, with the unpolarized efficiencies independent of the azimuth where
  the cell is symmetric. On the lamellar grating, the efficiencies of every
  order at normal incidence, at 20° and at conical incidence match the RCWA
  solution, planar incidence leaves the cross-polarized specular entry at
  $10^{-5}$ and conical incidence does not. A failed member leaves the
  earlier ones in the cache, so a rerun resumes.
- **Emissivity.** A lossy slab on a PEC wall has the emissivity
  $1 - |r|^2$ of the PEC-terminated Airy form, at normal and oblique
  incidence. On a lossy slab between air and glass, the emissivity at each
  node of the hemispherical rule matches $1 - |r|^2 - |t|^2$, and the
  hemispherical emissivity matches the weighted sum of the closed form on the
  same nodes to about $10^{-5}$; the rule itself, with eight polar nodes,
  reproduces the exact integral of that slab to about $10^{-5}$. The quadrature rule integrates
  powers of the cosine exactly and reproduces the unfolded rule on a
  symmetric integrand. Kirchhoff's law is checked on a grating without mirror
  symmetry: the power missing from the row of an outgoing wave equals, to
  rounding, the absorptance of the wave arriving from the opposite azimuth,
  while the absorptances at the two azimuths differ, so the label is
  discriminated.
- **Cell response.** A response tabulated from a smooth analytic function
  reproduces it to rounding, equals the stored values and derivatives at the
  samples and is differentiable across them. On a dielectric ridge cell
  with a morphed mesh, Hermite interpolation of the stored value and
  derivative matches solves at held-out widths to a few $10^{-4}$, and the
  error falls close to the fourth power of the sample spacing (a fitted order
  of about 3.9 between five and nine samples).
- **Propagation and the locally periodic approximation.** Plane waves
  advance by $e^{-ik_z d}$ in the propagating band and decay in the
  evanescent one. The focus of a one- and a two-dimensional lens matches the
  stationary-phase closed form and the Rayleigh-Sommerfeld sum of the
  samples to about $10^{-4}$, and a blazed deflector sends its energy to the
  grating-equation angle, with its centroid at $z\tan\theta$. Against a
  supercell solve of a ridge cell varied across eight periods, every
  transmitted order of the approximation is within a few percent of the full
  solution, with the sign of the negative orders fixed, and a supercell of
  four identical ridges reproduces the single cell to rounding.
- **Gradients.** The gradient of a figure of merit built from the cell
  outputs (circular efficiencies, the unpolarized absorptance, a circular
  Jones entry, a Mueller entry and the power of an order that does not
  propagate) with respect to the permittivity of each element of the lamellar
  grating at conical incidence matches high-order central differences, and
  the same under `jax.jit`. The emissivity of an angle sweep is checked the
  same way, the stored derivative of a cell response against fourth-order
  differences of its values, and the aperture field and its propagation
  through the design map, the envelope and the stored values and
  derivatives.

Gradients with respect to the incidence angle are not provided and raise
({doc}`capabilities`). All of these run on every change to gfem.

Tests: [cell outputs](https://github.com/benvial/gfem/blob/main/tests/verification/test_diffraction_cell_outputs.py),
[angle sweeps](https://github.com/benvial/gfem/blob/main/tests/verification/test_angle_sweep.py),
[emissivity](https://github.com/benvial/gfem/blob/main/tests/verification/test_emissivity.py),
[cell response](https://github.com/benvial/gfem/blob/main/tests/verification/test_cell_response.py),
[supercell](https://github.com/benvial/gfem/blob/main/tests/verification/test_lpa_supercell.py),
[gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_diffraction_gradient.py),
[identities](https://github.com/benvial/gfem/blob/main/tests/unit/test_diffraction_cell_outputs.py),
[propagation](https://github.com/benvial/gfem/blob/main/tests/unit/test_aperture.py).

## Perfectly matched layers

Each case ends a guide or a cell with a layer backed by a wall and compares
the reflection read at a port with the layer's closed form,
$|R| = R_0^{\,n f \cos\theta / f_\text{ref}}$, at order 3 on meshes layered
along the stretch:

- **A TEM line ended by a layer.** Its reflection is the target reflection
  at the reference frequency; with two frequencies and no reference
  frequency given, the lowest one is the reference and the other reflects
  less; a pinned reference frequency above the solve frequency reflects
  more, as the formula says; a background of index $n$ reflects $R_0^{\,n}$.
- **A TE10 guide ended by a layer.** Its mode is two plane waves at
  $\cos\theta = \beta/k_0$, and its reflection follows the formula, near
  cutoff too, where it rises; a reference frequency where $k_0 = \beta$
  brings it back below the mesh's floor.
- **A periodic cell over a layer, read at a Floquet port.** Its specular
  reflection follows the formula at normal, oblique and conical incidence,
  TE and TM, in vacuum and in a dielectric filling, with no cross-polarized
  reflection and no warning of an oblique absorber.
- **A leaky dielectric guide radiating into a layer.** Its complex
  resonance matches the root of its dispersion relation, plain and under a
  Bloch wave vector, and every returned mode stays put when the layer's
  target reflection changes, so no mode of the layer is returned.
- **A strip guide through side layers.** Between two wave ports cut by the
  layers, the guide transmits its own mode's propagation and reflects
  nothing, S is symmetric and passive, and the effective index matches a
  wide closed section's to the clearance rule.

Identities checked on coarse models: the layer absorbs the power the port
does not get back; a model without a layer is bitwise unchanged;
electrostatic and DC conduction solves see the layer's background material;
changing the target reflection changes the operator and its cache keys;
energies and participations count the physical regions; a port section cut
by side layers returns the guided mode; a solve across MPI ranks equals one
process's. Each refusal of the layer is a test of its own. Gradients next
to a layer (the permittivity of a slab beside it, nodes in the air, on the
layer's inner face and inside it, a boundary morph, a probe inside the
layer and an eigenfrequency) match finite differences. All of these run on
every change to gfem.

Tests: [closed forms](https://github.com/benvial/gfem/blob/main/tests/verification/test_pml.py),
[identities and refusals](https://github.com/benvial/gfem/blob/main/tests/unit/test_pml.py),
[gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_pml_gradient.py).

## Free space

Incident fields, point dipoles and the measurements read from them are
checked at order 3 against closed forms, against identities the physics must
obey, and against finite differences:

- **Spheres under a plane wave.** A lossy, a lossless and a perfectly
  conducting sphere lit at 30° in TE and TM, solved as two columns of one
  factorization: their extinction, absorption and scattering cross sections
  match the Mie series at the radius of the sphere of equal volume, a
  lossless sphere absorbs nothing, and a model cut by a symmetry plane
  reports half of each. A wide Gaussian beam over its focal intensity gives
  the plane wave's cross sections.
- **The optical theorem.** The extinction read from the forward far-field
  amplitude matches the power balance of the cross sections.
- **A particle on a layered substrate.** A lossy particle on a film over
  glass, lit from the air and from the glass past the critical angle (the
  particle then sits in the evanescent tail): extinction is absorption plus
  scattering, and each part is the same on any side holding the particle.
  The stack's own field matches Fresnel's coefficients at one interface, in
  total internal reflection too, Airy's sum for a lossy film and the
  standing wave over a ground plane, and a lossless stack conserves its
  normal power flux; with no particle, the scattered field vanishes to
  round-off whatever the incidence.
- **A Gaussian beam.** It solves Maxwell's equations (by finite
  differences), is the superposition of plane waves it is built from, and
  carries 1 W through any plane across its axis; its discrete residual in a
  box falls under refinement in the element size and in the order.
- **Dipoles in a homogeneous medium.** Electric and magnetic dipoles in
  vacuum and in a dielectric emit the bulk power of the medium, so their
  Purcell factor is 1.
- **Dipoles above a ground plane.** Normal and tangential dipoles at several
  heights above a perfect conductor and above a magnetic wall emit what
  image theory gives, and their far fields are those of the dipole and its
  image.
- **A dipole in a two-mode guide.** The power it sends into each guided mode
  over its emitted power, the beta factor, matches the guide's modal closed
  form, and the guided powers add up to the emitted power.
- **A dipole's power split.** What the ports carry away, what the layer takes
  and what lossy materials absorb add up to the emitted power, to round-off.
- **Port waves by reciprocity.** With `source_fields=False`, the port waves
  of dipoles on a lumped-port line and in a wave-port guide equal those of
  the dipoles' solved source fields (to round-off on the line and in the
  guide), on uniform and adaptive sweeps; gradients match theirs (uniform)
  and are exact at the samples and to the sweep's field error between
  (adaptive).
- **The far field of a Hertzian dipole.** Its pattern matches the closed
  form, the intensity integrated over the sphere is the radiated power, the
  peak directivity is 1.5, and two crossed dipoles in quadrature radiate
  circular polarization along their axis, of axial ratio 1.
- **A fiber over a grating coupler.** The coupling of the port-fed field into
  the fiber's beam equals the wave the port receives when that beam lights
  the grating (reciprocity), and it does not depend on the plane it is read
  on. A dipole couples into a Gaussian beam as reciprocity gives from the
  beam's field at the dipole.

Gradients match finite differences: the Purcell factor and the beta factor
by a design's permittivity, the emitted power by a boundary morph that moves
the dipole's element and on an adaptive sweep, the cross sections under
plane waves and a Gaussian beam, a plane wave's port wave and kept fields by
the scatterer's permittivity and by its permittivity law's parameters, the
radar cross section and the directivity by a design, the beam coupling by a
design, and the power through a surface, the far field and the beam coupling
by node coordinates (a morph of the device and a move of every node). Every
refusal of a source or of its measurements is a test of its own. Across
three MPI ranks, a plane wave on a homogeneous background and on a layered
substrate, an electric dipole held by a rank other than the first, and every
measurement read from them (cross sections, power through a surface, far
fields, beam coupling, emitted power, Purcell factor and its gradient) equal
one process's.

All of these run on every change to gfem, with two exceptions. Two cases
run before each release: the spheres on a refined mesh, against the Mie
series and in their bistatic radar cross section, and a dipole above a glass
half-space against Sommerfeld's reflection integrals. The checks across
ranks need the {term}`MPI PETSc stack` and run twice a week.

Tests: [spheres](https://github.com/benvial/gfem/blob/main/tests/verification/test_free_space_mie.py),
[port waves by reciprocity](https://github.com/benvial/gfem/blob/main/tests/verification/test_reciprocal_port_waves.py),
[far fields of spheres](https://github.com/benvial/gfem/blob/main/tests/verification/test_far_field_mie.py),
[layered substrate](https://github.com/benvial/gfem/blob/main/tests/verification/test_layered.py),
[stack fields](https://github.com/benvial/gfem/blob/main/tests/unit/test_transfer_matrix.py),
[Gaussian beam](https://github.com/benvial/gfem/blob/main/tests/unit/test_gaussian_beam.py),
[dipoles](https://github.com/benvial/gfem/blob/main/tests/verification/test_dipoles.py),
[far fields](https://github.com/benvial/gfem/blob/main/tests/verification/test_far_field.py),
[beam coupling](https://github.com/benvial/gfem/blob/main/tests/verification/test_beam_coupling.py),
[gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_free_space_gradient.py),
[far-field gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_far_field_gradient.py),
[dipole over glass](https://github.com/benvial/gfem/blob/main/tests/reference/test_layered_dipole.py),
[across ranks](https://github.com/benvial/gfem/blob/main/tests/mpi/test_free_space.py).

## Optical forces

The force and torque of a solved field on a body are checked at orders 2 and 3
against closed forms, against an identity the physics must obey, and against
finite differences:

- **The exact derivative.** On an eigenmode at 1 J the force and the torque
  equal $-(1\,\mathrm{J}/\omega)\,\mathrm{d}\omega/\mathrm{d}\xi$, read from
  the gradient of the eigenfrequency through a node map that translates or
  rotates the body, to the eigensolver's tolerance; moving the torque's origin
  adds the moment of the force.
- **A cavity wall.** The force of the TE101 mode of a perfectly conducting
  rectangular cavity on its end wall, held by a metal slab, matches
  $1\,\mathrm{J}/[a(1 + (a/d)^2)]$, and its components along the walls the
  slab slides on are NaN.
- **Spheres under a plane wave.** The radiation-pressure efficiency of a
  lossless and a lossy sphere lit at 30° in TE and TM matches the Mie series
  $Q_\mathrm{ext} - g\,Q_\mathrm{sca}$ at the radius of the sphere of equal
  volume, and a lossless cube is pushed along the wave.
- **Two waveguides.** The gradient of the effective index of a port mode
  through a node map that moves one of two silicon slabs matches the dispersion relation of the two slabs, differentiated in the gap,
  at gaps of 100 and 300 nm.
- **Refusals.** Every unsupported body, field or surrounding is a test of its
  own.

All of these run on every change to gfem.

Tests: [force and torque](https://github.com/benvial/gfem/blob/main/tests/unit/test_optical_force.py),
[closed forms](https://github.com/benvial/gfem/blob/main/tests/verification/test_optical_force.py).

## Multipoles

The multipole decomposition of a scatterer's field is checked at order 3
against closed forms, against identities the physics must obey, and against
finite differences:

- **Spheres under a plane wave.** Dielectric spheres lit by a plane wave,
  including the one that satisfies the first Kerker condition: their
  multipole coefficients match the Mie series up to the degree it is
  resolved, the partial scattering cross sections match it in any
  propagation direction, and they add up to the scattering cross section.
  A perfectly conducting sphere and a magnetic sphere match their own Mie
  series.
- **A wide beam.** A Gaussian beam wide against the scatterer gives the plane
  wave's multipoles, and an incident-field function that is a plane wave
  gives the plane wave's.
- **A point dipole read back.** An electric or a magnetic point dipole, in
  vacuum and in a magnetic medium, returns its own moment exactly.
- **A moved scatterer and an empty one.** Moving the scatterer multiplies the
  coefficients by the phase of the plane wave at its new position, the
  cross sections stay put, and with no scatterer the coefficients are
  round-off.
- **The toroidal correction.** The long-wavelength and toroidal dipoles of the
  induced current give the exact electric dipole as $p \approx p_0 - ikT$,
  and the anapole sphere has $p_0$ and $T$ at the values of the Mie
  internal field.
- **Gradients.** The multipoles' gradient by a design's permittivity matches
  finite differences, for a dipole, for a quadrupole and for the fraction of
  the scattering a multipole carries. Every refusal is a test of its own.
- **Across ranks.** On three MPI ranks the coefficients and the current
  moments equal one process's, and so does the gradient.

All of these run on every change to gfem, except the refined Kerker and
anapole spheres, which run before each release, and the checks across ranks,
which need the {term}`MPI PETSc stack` and run twice a week.

Tests: [multipoles](https://github.com/benvial/gfem/blob/main/tests/unit/test_multipoles.py),
[spheres](https://github.com/benvial/gfem/blob/main/tests/verification/test_multipoles.py),
[gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_multipoles_gradient.py),
[across ranks](https://github.com/benvial/gfem/blob/main/tests/mpi/test_multipoles.py).

## Anisotropic materials

- An absorbing boundary ending a biaxial, magnetic slab reflects as the
  half-space of transmission-line theory, for both polarizations and at two
  electrical slab thicknesses.
- A WR-28 guide filled with anisotropic ε and μ transmits its TE10 mode with
  the closed-form propagation constant, without reflecting. With its tensors
  turned about the guide axis, it does not reflect either.
- The port mode of that turned tensor propagates, with a propagation constant
  between those of the two guides filled with the tensor's principal values
  across the guide.
- The loss along each principal axis from {term}`principal participation`
  matches $1/Q$ from the complex eigenfrequency, on a material whose field lies
  off every principal axis.
- A model and its rigidly rotated copy, with the material given as rotated
  global matrices, give the same S, eigenfrequency and capacitance. A
  wave-port guide rotated to an oblique port normal gives the same S.
- A uniform anisotropic material field on tilted axes solves as the
  anisotropic material it holds.

Tests: [wave ports](https://github.com/benvial/gfem/blob/main/tests/verification/test_anisotropic_wave_ports.py),
[rotation](https://github.com/benvial/gfem/blob/main/tests/unit/test_anisotropic_rotation.py),
[absorbing boundary](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_boundaries.py),
[port mode](https://github.com/benvial/gfem/blob/main/tests/unit/test_port_mode.py),
[principal participation](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode.py),
[material field](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven.py).

## Dispersive materials

Each check below uses a region whose permittivity follows a
{term}`permittivity law`:

- **Equivalence.** A dispersive region solves exactly as a constant material
  set to the law's permittivity at each frequency. This holds for every term
  kind, with a conductivity, and beside first- and second-order absorbing
  boundaries. It also holds for a metal (Re ε < 0), against the same
  permittivity set directly. In the electrostatic problem the law gives its
  {term}`static permittivity`.
- **Filled cavities.** A closed box filled with a Lorentz or Drude medium
  resonates at the closed-form polariton and plasma branches of its cutoffs.
  A Drude medium returns no field at its plasma frequency. A lossless Drude
  metal solved as a polynomial problem gives the same modes.
- **Resonances.** A target or mode where a law region's wavelength is
  shorter than two of its longest edges raises before solving. Below that
  limit the modes are the exact ones of the discretization, at orders 1
  and 3.
- **Mode energy.** The {term}`energy permittivity` makes the mode energy
  equal to the eigenproblem's own normalization. Beside a dispersive region,
  a junction's participation obeys −2 d ln ω / d ln L, and the surrounding
  dielectric's share
  obeys −2 d ln ω / d ln ε. Where the energy permittivity is not positive,
  the solve refuses.
- **Wave ports.** A guide filled with a Debye or Lorentz medium transmits its
  fundamental mode with the closed-form propagation constant at every
  frequency of a sweep. A metal in a port's cross-section is refused before
  any solve.
- **Adaptive sweep.** A lossless resonance inside the band is sampled on both
  sides, and the reduced model matches the uniform sweep. So does the sweep
  of a Djordjevic-Sarkar substrate and of a law region behind an absorbing
  boundary.
- **Gradients.** Law parameters are checked by finite differences on driven
  S (uniform and adaptive sweeps, wave ports, a periodic cell, behind
  absorbing boundaries), on eigenfrequency and Q, and on capacitance.
  Existing inputs are checked beside a law, including participations and
  mode fields beside a lossy Lorentz region. The density of a
  metal-dielectric blend is checked under both the index and the linear
  interpolation. The gradient of {doc}`tutorials/metal-bend` through gold
  is checked by finite differences.
- **Solvers and ranks.** The driven and eigenmode law cells also run on
  `linear_solver="petsc"`. On two MPI ranks, S and its gradient by a law
  agree with one process.
- **Mesh refinement.** A line with a Drude metal block is one of the field
  estimator's cases (see [Mesh refinement error
  estimates](#verification-mesh-refinement)).
- **Material data.** Each library law is checked against its source: exactly
  for silica and lithium niobate, within the refit's stated error for
  silicon and silicon nitride, and for the metals against an independent
  evaluation of the source's model and against its published tabulation.
  Every library law and every fitted law is passive on a dense grid far
  beyond its {term}`fitted range`. The fit recovers known laws from their
  own samples, and reaches its target on dense gold and aluminium film
  tables. A solve outside a law's fitted range warns once, for each problem
  type that reads a law.

Not checked: port modes of plasmonic guides (refused), absorbed power in a
dispersive region, agreement with measured metal data, eigenmodes inside a
resonance's crowded band, convergence of the tutorial's unresolved skin
depth, library values against measurements other than their source, and the
fit's repeatability across platforms.

All of these run on every change. The PETSc cells run twice a week, as in
[PETSc and SLEPc](#verification-petsc).

Tests: [driven](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_dispersive.py),
[law gradients](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_law_gradient.py),
[cavities and bands](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode_dispersive.py),
[mode energy](https://github.com/benvial/gfem/blob/main/tests/unit/test_mode_energy_dispersive.py),
[electrostatic](https://github.com/benvial/gfem/blob/main/tests/unit/test_electrostatic_dispersive.py),
[index blend](https://github.com/benvial/gfem/blob/main/tests/unit/test_index_blend.py),
[wave ports](https://github.com/benvial/gfem/blob/main/tests/verification/test_dispersive_wave_ports.py),
[adaptive sweep](https://github.com/benvial/gfem/blob/main/tests/verification/test_dispersive_adaptive.py),
[metal bend](https://github.com/benvial/gfem/blob/main/tests/verification/test_metal_bend.py),
[field estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_field_estimate.py),
[MPI](https://github.com/benvial/gfem/blob/main/tests/mpi/test_dispersive.py),
[library materials](https://github.com/benvial/gfem/blob/main/tests/unit/test_library_materials.py),
[library sources](https://github.com/benvial/gfem/blob/main/tests/verification/test_library_sources.py),
[fit](https://github.com/benvial/gfem/blob/main/tests/unit/test_fit_permittivity.py),
[fit on measured tables](https://github.com/benvial/gfem/blob/main/tests/verification/test_fit_permittivity_data.py),
[fitted range](https://github.com/benvial/gfem/blob/main/tests/unit/test_fitted_range.py), and the law and blend cells of the
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py) and the
[gradients that raise](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix_raise_table.py).

## Patterned sheets

- On a parallel-plate line, a sheet of 2 µm of copper or of a 1 pH kinetic
  inductance, solid or with a void / solid pattern, takes BDDC + GMRES the
  iterations of an empty sheet, and its S matches the direct solver's.
- A copper pixel design graded across the default contrast passes the
  finite-difference check on both solvers.
- The whole design chain (pixels on a layer grid, filter, projection,
  exponential fill) differentiates end to end, and so do node coordinates
  that move the filled sheet.
- In a cavity with a resistive film of random fill, the eigenfrequency
  matches an independent iteration that refactorizes the operator at every
  step.

Tests: [sheets](https://github.com/benvial/gfem/blob/main/tests/verification/test_sheet_gradient.py).

## Layer patterns and GDS

- A disk's pattern follows its circle to within 0.05 pixel, with its area and its
  perimeter (to 0.5 %); a bar keeps its width to within 0.05 pixel. Area
  fractions of a pattern add up to its area to round-off.
- On a dumbbell bridge, two bars, a notch, an annulus and a small disk, the width
  and spacing check finds one violation at the true distance (0.05 pixel, 0.1 on
  the faceted disk), and agrees with an independent layout checker's projection
  metric to 1.5 database units.
- An independent GDS reader (gdstk) reads the file back: units, cell, layers and
  datatypes, polygon counts, the area per layer to within the perimeter times
  the database unit.
- The fabrication model moves a straight edge by the edge bias to $0.02\sigma$
  from $-1.5\sigma$ to $1.5\sigma$; it and the length-scale constraints pass
  second-order gradient checks. A bar or slot drawn at $0.9$ of the rule
  breaks its constraint and one at $1.1$ keeps it.
- On the 2D bend, the arc's pattern solved back as area fractions transmits as
  the same pattern drawn on 6.25 nm pixels, and as the $\beta = 32$ projection
  the optimizer solves, to 0.02 in $|S_{21}|^2$; the gradient through the
  fabrication model's three printed designs matches a finite difference.
- A toy design whose target holds bars, slots and a bridge of 0.3 to 0.7 of the
  rule fails the check unconstrained (narrowest 0.47 and 0.24 of the rule); with
  the constraints asked 10 % above the rule as a penalty, its narrowest width
  and spacing reach 1.051 and 0.906 of the rule.

Tests: [patterns](https://github.com/benvial/gfem/blob/main/tests/unit/test_pattern.py),
[checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_feature_check.py),
[GDS](https://github.com/benvial/gfem/blob/main/tests/unit/test_gds.py),
[fabrication model](https://github.com/benvial/gfem/blob/main/tests/unit/test_fabrication_model.py),
[constraints](https://github.com/benvial/gfem/blob/main/tests/unit/test_length_scale.py),
[end to end](https://github.com/benvial/gfem/blob/main/tests/verification/test_layer_patterns.py).

## Quantum metrics

- **Participations.** A junction's participation equals
  −2 d ln ω / d ln L_J from two solves, with and without a capacitance on the
  port and beside a capacitive sheet. A port with an inductance and a
  capacitance at the end of a line resonates at the closed-form frequency of
  the loaded line.
- **The dressed modes.** As the junction's phase shrinks they tend to the
  first-order closed forms. They do not move when more Fock levels are
  added, a transmon below its resonator gets negative shifts, and a
  hybridized pair raises. They reproduce two stored outputs of the pyEPR
  tutorials, for one and two modes.
- **Gradients.** The gradient of the dressed modes matches fourth-order
  differences and differentiating the eigenvalue problem directly. The
  dispersive shift's gradient through a solve is checked by finite
  differences on a small two-mode model on every change, and before each
  release on the tutorial's transmon, along its pad gap.
- **Interface layers.** `"MA"` on a film with vacuum on both faces equals the
  sum of its two faces read one at a time.

Not checked: agreement with a measured device, and convergence of a sheet's
or an interface's participation at a film edge.

Tests: [dressed modes](https://github.com/benvial/gfem/blob/main/tests/unit/test_quantum.py),
[participations](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode.py),
[transmon](https://github.com/benvial/gfem/blob/main/tests/verification/test_transmon_chi.py),
[interface layers](https://github.com/benvial/gfem/blob/main/tests/verification/test_interface_participation.py),
and the dispersive-shift cells of the
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py).

## Thermal

Thermal and DC conduction solves, and the electromagnetic heat and heat
flow measured on them, are checked against closed forms, against identities
of the discrete balance, against each other, and against two independent
codes, gsim and femwell, on the same heater. Each check runs on every change
to gfem.

- **Exact in the discrete space.** Layered slabs, in series, under
  convection and with a convection-only bottom, give their hand temperatures,
  on a cross-section and on unstructured 3D tetrahedra. A bar gives
  R = L/(σA), a two-conductor slab the hand formula for its resistance per
  length, heat and power, and the power a conductor dissipates equals the
  power its terminals deliver. Switching a source on and off equals two
  superposed steps, and twice the power gives twice the rise.
- **Closed forms.** A strip heater on a substrate follows the ln coth profile
  of its temperature, and its far field decays over 2H/π. A slab's slowest
  thermal mode gives its time constant. The late decay of a step response
  matches it, with TR-BDF2 and with the θ schemes, and the 10–90 % and 1/e
  times match the analytic trace normalised over the same window. After ten
  time constants a run reaches the steady temperature. The resistance of an
  annular bend matches its closed form, and the Joule heat of each element is
  the exact mean of σ|∇V|² over it.
- **The thermo-optic shift.** The first-order shift agrees with a mode
  re-solved on the heated permittivity at the drive that gives π. An optical
  window around the core gives the same shift as the whole section with the
  other regions excluded. A route through a z-invariant 3D rise gives the
  section's phase per length times its length, and the first order agrees
  with a re-solved mode at a station in a 3D heater.
- **3D against the cross-section.** A heater section extruded in layers of
  tetrahedra gives the slab's resistance per length and its core rise per
  watt per metre.
- **Two independent codes.** On the femwell TiN heater section, meshed by
  gsim and read by gfem from gsim's own file, gfem's P_π matches gsim's
  (22.39 mW) and femwell's published 22.6 mW, and the heater's resistance
  matches gsim's. Driven at P_π with the backward Euler scheme on gsim's time
  grid, the rise time and the time constant match gsim's.
- **Electromagnetic heat.** The power a driven field dissipates in a region
  equals the power the field brings into it, measured as the flow of the
  same field through the region's surface, for a loss tangent, a
  conductivity, a principal-axis tensor, a complex material field, a
  permittivity law (a gold bend, among others) and a blend. On a closed
  device whose ports have real references and sit on uniform stretches of
  line, solved at element order 3, the losses of every region and metal
  sheet add up to $1 - \sum_j |S_{j1}|^2$ to $10^{-9}$ (at order 2, S read
  on a lumped port's faces is about $10^{-6}$ off, and a port on a
  nonuniform stretch about $10^{-5}$); a perfectly matched layer,
  an absorbing face and a lumped port's load dissipate exactly nothing. The
  heat of a drive is the incident power times that loss, and two ports fed
  together heat as their superposed field. Against transmission-line closed
  forms: a lossy section of a parallel-plate line, a metal wall ending a
  line, as a surface conductor and as a volume whose skin depth the mesh
  resolves. A cross-section slab heats per length as the 2D model loses, in
  both polarizations, and as the closed form. A heater made of the metal
  wall rises as its absorbed power conducts to the sink, and a convection
  wall keeps its share.
- **Heat flow.** The heat out of every boundary surface adds up to the
  source power, with convection touching a fixed surface, a heat flux, and
  sheet heat on a fixed face; a set of elements gives minus its complement,
  and heat on a face lying on the cut belongs to neither. Layered slabs in
  series and under convection give their hand fluxes. Sheet heat on a
  slab's top goes all to the sink under an insulated top, splits by the
  conductances under convection, and leaves at once through a fixed top. A
  side with no heat sink measures its source power, with a conductivity
  gradient of exactly zero. The sink of a strip heater takes its whole
  heat. In a transient run, the heat out of the sinks plus the heat stored
  equals the heat put in at every time, under every time scheme; the heat
  out of a slab's sink matches its series solution, and after ten time
  constants its step-average flow is the steady one.
- **Gradients.** Every thermal and heat gradient of the matrix on
  {doc}`capabilities` (the dissipated power, the temperature it heats, the
  steady heat flow and the transient heat out), one chain from the heater's
  conductivity, every element's thermal conductivity, the core's
  thermo-optic coefficient and the drive current through DC conduction,
  Joule heating, the steady and the transient temperature to the phase, and
  one chain from a line's loss tangent, a sheet's fill and resistance, every
  element's thermal conductivity and a two-port drive through the driven
  solve and its electromagnetic heat to the temperature, match fourth-order
  central differences on the direct solver. The thermal cells and chain,
  and the dissipated power with respect to a conductivity, also run on
  BDDC + GMRES.

Two comparisons are observations, not checks, and live on the tutorials. The
TiN heater of Jacques et al. (2019): gfem's P_π falls inside the published
bands at both heater widths ({doc}`tutorials/heater`). The 3D spiral heater:
its P_π, resistance and per-pass phases on three meshes
({doc}`tutorials/heater3d`).

Tests: [closed forms](https://github.com/benvial/gfem/blob/main/tests/verification/test_thermal_closed_forms.py),
[electromagnetic heat closed forms](https://github.com/benvial/gfem/blob/main/tests/verification/test_electromagnetic_heat_closed_forms.py),
[dissipated power](https://github.com/benvial/gfem/blob/main/tests/unit/test_electromagnetic_heating.py),
[electromagnetic heat sources](https://github.com/benvial/gfem/blob/main/tests/unit/test_electromagnetic_heat_source.py),
[sheet heat](https://github.com/benvial/gfem/blob/main/tests/unit/test_face_heat.py),
[heat flow](https://github.com/benvial/gfem/blob/main/tests/unit/test_heat_flow.py),
[heat out](https://github.com/benvial/gfem/blob/main/tests/unit/test_heat_out_trace.py),
[gold bend](https://github.com/benvial/gfem/blob/main/tests/verification/test_metal_bend.py),
[gsim and femwell](https://github.com/benvial/gfem/blob/main/tests/verification/test_thermal_parity.py),
[extrusion](https://github.com/benvial/gfem/blob/main/tests/verification/test_thermal_extrusion.py),
[DC conduction](https://github.com/benvial/gfem/blob/main/tests/verification/test_dc_conduction.py),
[thermo-optic shift](https://github.com/benvial/gfem/blob/main/tests/verification/test_thermo_optic.py),
[route phase](https://github.com/benvial/gfem/blob/main/tests/verification/test_route_phase.py),
[steady](https://github.com/benvial/gfem/blob/main/tests/unit/test_steady_thermal.py),
[transient](https://github.com/benvial/gfem/blob/main/tests/unit/test_transient_thermal.py),
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py).

## Electro-optics

The electro-optic shift is checked against closed forms, against modes
re-solved on the perturbed permittivity, and against itself on rotated and
remeshed sections. Each check runs on every change to gfem.

- **The electro-optic shift.** A lithium niobate film between two electrodes,
  closed by electric side walls, gives the slab's closed-form shift, for an
  x-cut film through $r_{33}$ and for a film whose field reaches $r_{22}$
  alone. The first order agrees with a mode re-solved on the perturbed
  permittivity at $\pm 10$ V: on a ridge, for the diagonal part of the change,
  and on an L-shaped core that breaks every mirror symmetry, for $r_{51}$
  alone, the off-diagonal part. An x-cut section and the same section turned
  by 90° into z-cut, crystal axes and electrodes with it, give the same
  shift; $r_{22}$ adds nothing where the mode and the field leave it nothing
  to act on. The shift is linear in the tensor and in the drive, a uniform
  static field is read exactly, and a tensor of the same value on every
  diagonal entry gives the thermo-optic shift of the matching coefficient.
- **Across meshes.** The static field is read at fixed points of the optical
  model, wherever they fall in the static mesh: an optical model on a
  second mesh of the same section gives the same shift to
  $5.5 \times 10^{-4}$.
- **Capacitance per length.** Parallel plates on a section give
  $\varepsilon_0 \varepsilon_r H / d$ per metre, whatever the slab's thickness.
- **Gradients.** The gradients with respect to the tensor and to the static
  permittivity match fourth-order central differences, and so does the
  gradient with respect to the static mesh's nodes: alone, through an
  electrode gap moved by a shape morph with the optical model on the same
  mesh, and through an optical window on its own mesh, the ridge held. The
  optical model's permittivity and nodes raise.

Three comparisons are observations, not checks, and live on the pages. The
modulator of Wang et al. (2018): gfem's push-pull $V_\pi L$ is 1.62 V·cm
against the measured 1.8 V·cm, at assumed values of the dimensions the paper
withholds ({doc}`tutorials/modulator`). An optical window and the whole
section with the electrodes excluded give shifts within a few percent, and
an order-1 electrostatic solve biases the shift by about 1 %
({doc}`recipes/electro-optic-phase`).

Tests: [unit checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_electro_optic.py),
[closed forms and re-solves](https://github.com/benvial/gfem/blob/main/tests/verification/test_electro_optic.py),
[node gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_electro_optic_gradient.py).

## Piezoelectric problems

The piezoelectric problem is checked against closed forms of plates, against
identities that hold exactly in the discrete problem, and against finite
differences. Each check runs on every change to gfem.

- **Plates.** A plate of a hexagonal crystal between two electrodes, with free
  side edges, gives the closed-form admittance of the thickness-extensional and
  the thickness-shear resonator, with and without loss, from far below to just
  above resonance. The shear plate's displacement and engineering strain at
  points match their closed forms, which checks the strain's Voigt order and
  the per-volt normalization.
- **A finger array.** The capacitance a finger pair adds to a long array of line
  electrodes matches the closed form of a periodic array.
- **A damping frame.** The reflection of a plane wave at normal incidence from a
  frame of four wavelengths or more follows its design rule within a factor of
  two ({py:class}`gfem.DampingFrame`).
- **Identities.** The power a transducer takes is the power its losses dissipate.
  Without the piezoelectric tensor the admittance is that of the clamped
  capacitance. Without loss it is purely reactive. It is reciprocal and passive,
  and each row sums to zero when every electrode is a terminal. A section turned
  in its plane, with its crystal axes turned with it, gives the same admittance.
- **Gradients.** The gradients with respect to every material property, and with
  respect to the nodes through pitch, stretch, finger-thickness and local maps,
  with line and with meshed electrodes, match fourth-order central differences.
  So do readings of the displacement and strain at points.
- **Material tensors.** The rotated stiffness and piezoelectric tensor give
  lithium niobate's bulk wave speeds along the rotated axis.

Observations, not checks. The transducer of the {doc}`tutorial
<tutorials/transducer>` at full size (20 signal and 19 ground fingers, 40
grounded strips per side, assumed counts), 1.14 M unknowns on 32 cores, gives
$Z = 41.5 + 0.7i$ Ω at 0.8597 GHz and an S11 dip of about −38 dB near
0.8596 GHz; the device measured −34.3 dB near 0.84 GHz (Ni, Bhave and Xu,
arXiv 2606.05337), about 2 % lower in frequency. Its gradient costs one pass
over the nodes, about a fifth of one frequency's factorization.

Tests: [unit checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_piezoelectric.py),
[materials](https://github.com/benvial/gfem/blob/main/tests/unit/test_piezoelectric_material.py),
[network parameters and strain](https://github.com/benvial/gfem/blob/main/tests/unit/test_piezoelectric_network.py),
[closed forms](https://github.com/benvial/gfem/blob/main/tests/verification/test_piezoelectric.py),
[gradients](https://github.com/benvial/gfem/blob/main/tests/unit/test_piezoelectric_gradients.py).

## Mechanical eigenmodes

The elastic modes of a 2D section are checked against the closed forms of an
isotropic solid at order 4 on refined meshes, and against identities that hold
exactly in the discrete problem. Each check runs on every change to gfem.

- **Closed forms.** The anti-plane standing wave of a free rectangle, the shear
  horizontal modes of a strip with sliding walls (frequency and group
  velocity), the Lamb modes S0 and A0 of a plate (the roots of the
  Rayleigh-Lamb relations) and the Rayleigh wave of a deep block (frequency
  and group velocity) agree to better than $10^{-9}$ in frequency, and
  $10^{-7}$ for the Rayleigh wave, which also carries the finite depth. The
  error falls at least 100 times when the edge length halves.
- **Identities.** A uniform quality factor scales every frequency by
  $\sqrt{1 + i/Q_m}$; a lossless mode has a real frequency; $\omega(-q) =
  \omega(q)$ for an isotropic section; a mode is normalized to 1 J/m with a real
  largest coefficient; the group velocity matches a central difference of the
  band.
- **Piezoelectric plate.** The thickness-extensional plate of a 6mm crystal
  matches the stiffened half-wave frequency with open faces and the root of
  $\tan X / X = 1/k_t^2$ with grounded faces, to $10^{-9}$, and the grounded
  plate's mode is a pole of the `gfem.Piezoelectric` admittance.
- **Gradients.** The gradient of `omega` in the stiffness, density, quality
  factor, piezoelectric tensor, permittivity and node coordinates matches
  fourth-order central differences to $10^{-6}$, with and without an axial
  wavenumber.

Tests: [unit checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_mechanical_eigenmode.py),
[closed forms](https://github.com/benvial/gfem/blob/main/tests/verification/test_mechanical_eigenmode.py),
[piezoelectric plate](https://github.com/benvial/gfem/blob/main/tests/verification/test_mechanical_eigenmode_piezo.py),
[gradients](https://github.com/benvial/gfem/blob/main/tests/unit/test_mechanical_eigenmode_gradients.py).

(verification-acousto-optic)=
## Acousto-optic shift

The shift of a guided mode under an acoustic wave
({py:obj}`gfem.acousto_optic_shift`) is checked against closed forms and
against itself:

- **Closed form.** A slab in a uniform strain gives the weighted sum of its
  permittivity change exactly.
- **A second solve.** A separate mode solve gives the same shift to the
  eigen-solve's round-off.
- **Frame invariance.** The same device with its crystal and sections turned
  gives the same shift.
- **The electro-optic part.** With a static potential as the field, the electro-optic
  part equals {py:obj}`gfem.electro_optic_shift`.
- **Gradients.** The gradients in the photoelastic and electro-optic tensors, the
  offset, an acoustic node map and an acoustic material leaf match fourth-order
  central differences.

Observations, not checks. The tutorial's transducer at full size (20 signal and 19
ground fingers, 40 grounded strips per side, assumed counts), with optical windows
of 8 by 5 µm, gives a push-pull $V_\pi L$ of 0.48 V·cm per volt of incident wave at
0.861 GHz and an effective index of 1.8401; the device measured 1.004 V·cm at
0.842 GHz (Ni, Bhave and Xu, arXiv 2606.05337). The photoelastic and electro-optic
parts have opposite signs, the second 0.30 of the first.

Tests: [unit checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_acousto_optic.py),
[materials](https://github.com/benvial/gfem/blob/main/tests/unit/test_photoelastic_material.py).

(verification-mesh-refinement)=
## Mesh refinement error estimates

The {term}`error estimate` of {doc}`recipes/mesh-refinement` is checked by
its effectivity: the estimated error divided by the error against a known
answer, which must stay within a range set in the test for every mesh of
every case. Each kind of estimate (`ErrorEstimate.kind`) has its own range,
generated from its cases:

```{gfem-effectivity-table}
```

The quantity estimator's cases, all at `order=2`:

- **A square coaxial line** (a square conductor in a square shield), whose
  inner corners make the field singular: its capacitance over a mesh
  refinement run, against a constant extrapolated from fine uniform meshes
  and checked on a graded one.
- **A circular coaxial line**, faceted: its capacitance on meshes of 1 to 8
  times the base density, against a higher-order solve on the same facets.
  Against the exact formula for the circle the estimate stays below the
  true error, since it does not see the faceting.
- **A stepped parallel-plate line**, whose top plate steps down to half its
  height, with a singular field at the step's edge: $S_{21}$ at 3 and 8 GHz on
  uniform meshes and over a mesh refinement run with lumped ports, and at
  3 GHz with wave ports, against constants from fine uniform and graded
  meshes.
- **A parallel-plate line** between ports of 50 and 30 Ω: $S_{11}$, $S_{21}$ and $S_{22}$
  from 3 to 25 GHz on three meshes, against the line's closed form.
- **An L-shaped cavity**, whose re-entrant edge makes the field singular:
  its first resonance over a mesh refinement run, against a published
  high-accuracy value.
- **A rectangular cavity** with metal walls: four modes on two meshes,
  against the closed form.

The quantity estimator is also checked on cases that exercise one feature
each: a capacitance between two bars with a grounded screen between them,
and one between two conductors on either side of an interior metal sheet;
lines and cavities filled with anisotropic and magnetic materials, and a
coaxial line with an anisotropic fill; a stepped line with its wave ports
de-embedded, and a waveguide with two modes on each of two shared wave port
surfaces; the resonances of cavities with lossy fills and lossy walls
(impedance, absorbing and conducting); and a line closed by resistive ports
as a resonator. With loss in the volume: S of the parallel-plate line with
a lossy fill (dielectric loss and conductivity) between lumped ports,
against its closed form; and S between wave ports of a lossy waveguide:
uniformly filled, half filled by a lossy dielectric slab (whose port mode
is hybrid, so the estimate solves the ports' {term}`adjoint dual`), and
with that slab widening halfway along, each with and without
de-embedding, against a higher-order solve. A waveguide holding a
dielectric strip along its length, whose wave ports carry a hybrid mode,
is checked de-embedded over half its length against its closed form.

The S estimate is also checked on a line between conducting plates of
finite conductivity; on a line through a metal described by a permittivity
law deep below zero ($\mathrm{Re}\,\varepsilon$ about $-30$), at a wedge, on a flat face and
through a metal block; on wave ports whose face crosses a permeability
contrast and lumped ports beside a loss contrast; and on lines filled with
lossy magnetic and lossy anisotropic materials. The resonance estimate
covers permittivity laws with $\mathrm{Re}\,\varepsilon > 0$ (a dispersive fill and layered
cavities), second-order absorbing side walls, and lossy magnetic fills in
one and two layers.

The other kinds of quantity:

- **Q** of a mode: a line closed by resistive ports as a resonator, cavities
  with resistive, absorbing and conducting walls, and lossy fills whose Q
  is known exactly.
- **The mean frequency of a degenerate cluster**: pairs and triples of
  modes in symmetric cavities, a pair split slightly by the geometry, a
  pair in a cavity with conducting walls, and a pair with a third mode
  close by.
- **Objectives of S**: lossy and lossless lines (matched and nearly
  matched), the stepped line, waveguides with shared and lossy wave ports,
  and a line through a deep metal. On a lossless model with lumped ports
  the estimate goes through the {term}`lossless rewrite`.
- **Objectives of capacitance and electrostatic energy, and of potential
  means**: the two bars, with the space around them split into regions,
  and the coaxial line.
- **Objectives of frequency and Q**: the L-shaped cavity, the line closed
  by resistive ports, and a cavity with two modes.
- **Absorbed power and field means**: a waveguide with a lossy dielectric
  block, a waveguide with regions across half its width, and stepped lines
  with a lossy layer and with a small region at the step's edge.
- **Participations**: a cavity with a dielectric slab and one with a
  dielectric block.

A few references are extrapolated from finer meshes, with their
uncertainty kept below a fifth of every error checked.

Over the capacitance, S and resonance cases, the quantity estimate was
between {gfem-effectivity}`quantity low` and
{gfem-effectivity}`quantity high` times the true error; the higher ratios
came from fields with no sharp metal edge, such as the parallel-plate line
and the rectangular cavity, and from layered cavities. Some objectives are
estimated far more pessimistically near their target; the recipe's Limits
quote those ratios.

The field estimator is checked the same way, against the field's error in
the energy norm: on the square and circular coaxial lines, the coaxial
line with an anisotropic fill, the parallel-plate line, and the L-shaped
and rectangular cavities; and with loss, on the parallel-plate line with a
lossy fill (dielectric loss and conductivity) and with a Drude metal block
($\mathrm{Re}\,\varepsilon < 0$) against their closed forms, and against a higher-order solve on
the same mesh on a cavity in two layers, the upper one lossy, and on a
cavity whose lossy wall (impedance or absorbing) gives its resonance a
complex frequency. It is also checked on the line between conducting
plates, at a deep metal's wedge and flat face, in lossy magnetic and lossy
anisotropic fills, and in a cavity with second-order absorbing side walls.
Its estimate was between {gfem-effectivity}`field low` and
{gfem-effectivity}`field high` times the true error, the circular line
measured against a higher-order solve on its own facets; against the exact
circle it stays below the true error. These cases run at `order=2`, and
the original cases also at `order=3`.

Four configurations lie below that range and are not claimed; each test
holds the estimate to its measured band:

- on a line in two layers, the upper one lossy, with lumped ports spanning
  both: {gfem-effectivity}`lumped port across a loss jump low` to
  {gfem-effectivity}`lumped port across a loss jump high` times the true
  error on coarse meshes;
- next to impedance walls that carry the field's current:
  {gfem-effectivity}`impedance surface low` to
  {gfem-effectivity}`impedance surface high`;
- next to a lumped port spanning a permeability contrast:
  {gfem-effectivity}`lumped port across a permeability contrast low` to
  {gfem-effectivity}`lumped port across a permeability contrast high`;
- at a metal wedge with a permittivity law at $\mathrm{Re}\,\varepsilon = -1.5$, against a
  higher-order solve the field itself does not converge to:
  {gfem-effectivity}`plasmonic window low` to
  {gfem-effectivity}`plasmonic window high`.

The S estimate has one such configuration: a lumped port whose face
crosses a permittivity interface, on cells stretched 10:1 along the line,
measured
{gfem-effectivity}`stretched cells at a lumped port across a permittivity contrast low`
to
{gfem-effectivity}`stretched cells at a lumped port across a permittivity contrast high`
times the S error.

On the stepped line, whose field error has
no known value, the estimate is checked to keep a steady ratio to the $S_{21}$
error as the mesh is refined. On the three singular benchmark cases a mesh refinement run also
ends with a smaller error than uniform splits at the same number of
unknowns. Every range is read from the tests when the docs are built, and
every set of checks runs on every change.

Tests: [quantity estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_quantity_estimate.py),
[objective estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_objective_estimate.py),
[field estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_field_estimate.py).

## The p-multigrid solver

`linear_solver="pmg_gmres"` ({doc}`recipes/large-driven-model`) is checked
against the direct solver on models that need each part of its
{term}`direct set`:

- **Against the direct solver.** S of a strip line at orders 2 and 3, with
  each factorization of the direct set. A two-frequency sweep builds the
  solver once and matches fresh builds; the adjoint solve runs on the
  forward's solver. At 50 MHz on a metal line, where `"auto"` picks it once
  the direct solver does not fit, S matches the direct solver.
- **A guide between perfectly matched layers.** A strip guide cut by side
  layers solves in a few iterations, and its S matches the direct solver's.
  With the layer left to the smoother instead, the solve stalls.
- **A thin film.** A silicon guide with a film one nanometre thick along its
  outer faces, meshed one cell across, takes as many iterations as the guide
  without the film: the film's {term}`poor cells <poor cell>` go on the direct
  set.
- **Size in wavelengths.** A silicon guide one wavelength across at order 2,
  and half a wavelength across at order 3, run on every change. Before each
  release, the guide two and four wavelengths across takes about the same
  number of iterations; at two wavelengths S matches the direct solver, and at
  four it is reciprocal and passive.
- **Metal lines.** The coplanar lines of the reference cases, before each
  release, at the frequencies of their reference S, with fewer iterations than
  `"bddc_gmres"`.
- **Gradients.** S gradients on lumped and wave ports, the design's initial
  guess, and a gradient next to a layer are checked against finite
  differences on `"pmg_gmres"` too. The gradient matrix does not list it.
- **`"auto"` and refusals.** The memory estimate counts the same direct set
  as the solver builds. `"auto"` picks the solver only where the direct solver
  does not fit and the solver's own estimate does. Each refusal (order, problem
  type, Bloch, periodic and Floquet models, MPI ranks, 2D) is a test of its
  own.

### At scale

Before the solver was merged, silicon guides of 2.9 and 6.6 million unknowns
and a half model with a perfectly matched layer of 2.1 million unknowns were
checked against the direct solver. A bend of 17.2 million unknowns was checked
against the direct solver's outgoing waves from an earlier run on a
large-memory node. Times and memory are on {doc}`project/scaling`. This run is
not part of the release checks.

Tests: [p-multigrid](https://github.com/benvial/gfem/blob/main/tests/unit/test_pmg.py),
[mechanisms](https://github.com/benvial/gfem/blob/main/tests/verification/test_pmg.py),
[guides and metal lines](https://github.com/benvial/gfem/blob/main/tests/reference/test_driven.py),
[`"auto"`](https://github.com/benvial/gfem/blob/main/tests/unit/test_auto_solver.py).

## Broadband models and time responses

The time responses and rational models of S ({doc}`recipes/time-responses`,
{doc}`recipes/rational-model`) are checked against closed forms, and the
netlists against the model they were written from.

- **Time responses of a known line.** A lossless 75 Ω line behind a 50 Ω
  port, ended by 30 Ω, has a closed-form S. Its reflected step reads 75 Ω
  on the line and settles on the load's 30 Ω; its impulse peaks at each
  reflection's $\Gamma$, at its round trip, and conjugated data put the
  reflections at negative times. The rise time of each Kaiser window
  follows from the window, and S renormalized to another reference gives
  the same profile.
- **Rational fits.** Exact rational S, written out from known poles and
  residues, gives those poles back; lossless lines several phase wraps long
  fit within the tolerance, with a pole count that grows with the number of
  wraps. Passivity enforcement finds every interval where a singular value
  of S exceeds 1, and leaves a passive model close to the passive truth.
  The model's closed-form step matches the inverse Laplace transform of its
  terms, and the windowed step of its own S.
- **Netlists.** A small AC solver of the subcircuit's cards, in the tests,
  gives the model's S on every platform. In one CI job, ngspice runs the
  SPICE file and the Verilog-A module, and VACASK the module, with S equal
  to the model's to $10^{-9}$; a transient of the subcircuit follows the
  model's closed-form ramp response. These run on every change.
- **Solved devices.** The parallel-plate line of {doc}`get-started`, with
  its ceramic section, seen by a reflectometer: the impedance profile reads
  $Z_0/\sqrt{10}$ on the ceramic and $Z_0$ beyond it, the impulse peaks at
  the front face's reflection and round trip, and the first edge rises in
  about $0.99/f_\mathrm{max}$. Its rational model gives the same step as its
  S. A wave-port line 10 phase wraps long matches the exact TEM line, and
  its fit needs about two poles per wrap. A dielectric insert in a
  waveguide above cutoff gives the band-pass impulse of mode matching. These
  solved cases run nightly and before each release.
- **Wave ports.** On a lossy parallel-plate line, solved on every change,
  the fit asks for `z0=` on the mode's complex $Z_{PV}$, and S with more
  gain than the line's loss makes it warn that the data are not passive and
  still return a passive model.

Tests: [time responses](https://github.com/benvial/gfem/blob/main/tests/unit/test_time_responses.py),
[rational fits](https://github.com/benvial/gfem/blob/main/tests/unit/test_rational_fit.py),
[closed-form step](https://github.com/benvial/gfem/blob/main/tests/unit/test_rational_step.py),
[netlists](https://github.com/benvial/gfem/blob/main/tests/unit/test_netlist_export.py),
[circuit simulators](https://github.com/benvial/gfem/blob/main/tests/unit/test_netlist_simulators.py),
[wave ports](https://github.com/benvial/gfem/blob/main/tests/unit/test_broadband_wave_ports.py),
[solved devices](https://github.com/benvial/gfem/blob/main/tests/verification/test_broadband_fem.py).

(verification-petsc)=
## PETSc and SLEPc

- With no `petsc_options`, `linear_solver="petsc"` is an exact factorization:
  its results agree with `"direct"`.
- SLEPc's modes are refined as ARPACK's are: modes, Q, fields and gradients
  agree with ARPACK's.

These checks need the {term}`PETSc stack`. They run twice a week and on
every change to gfem's PETSc code.

Tests: [PETSc](https://github.com/benvial/gfem/blob/main/tests/unit/test_linalg_petsc.py),
[SLEPc](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode_slepc.py),
and the PETSc cells of the
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py).

## Several MPI ranks

Each thing a {term}`distributed solve` runs (see
{doc}`recipes/distributed-solve`) has a test launched on several ranks. It
compares the result with one process, on `"direct"` and on `"bddc_gmres"`,
and checks that every rank holds the same result, bit for bit. A driven solve
on `"bddc_gmres"` must also take the same number of iterations. An eigenmode
solve is compared with one process on SLEPc; its frequencies, Q and fields
must match, not its iteration count. Periodic driven cells are checked on
partitions that keep the pairs as given and on partitions that swap them. Some
things have no such test yet: a lumped port terminated by an inductance or a
capacitance, {py:obj}`gfem.characteristic_impedance`, `"petsc"` on an
electrostatic, eigenmode or periodic solve, and a periodic eigenmode solve on
`"bddc_gmres"`. They run the same code and do not raise.

The launched tests on `"bddc_gmres"` run on every change. Those on
`"direct"`, and every eigenmode test, need the {term}`MPI PETSc stack` and
run twice a week.

Tests: [launched tests](https://github.com/benvial/gfem/tree/main/tests/mpi).

### On many nodes

The runs of {doc}`project/scaling` were checked three ways:

- **Every rank count against one process.** On a 10,248,856 DOF coplanar
  line, from 1 to 32 nodes, each solver against its own one-process run. S
  agreed to rounding error on both solvers, and `bddc_gmres` took the same
  number of iterations.
- **The two solvers against each other.** On the largest mesh both solvers
  finished, 47,277,960 DOF on 16 nodes, `bddc_gmres` against `direct`. S
  agreed to well within the GMRES tolerance of that run. `direct` did not
  finish at 108,570,046 DOF, so this check did not run there.
- **The largest mesh against the 10,248,856 DOF line.** The meshes differ, so
  S differs by the change of mesh. This is an observation, not a check.

In every run, every rank held the same S, bit for bit.
