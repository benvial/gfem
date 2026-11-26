# Modes and ports

gfem solves two different eigenproblems, and both return "modes". A
{term}`port mode` is a wave travelling along a guide: the frequency is given and
its propagation constant is found. A {term}`resonant mode` is a field that rings
in a structure: its frequency is found. This page says which is which, when
each is real or complex, and what a {term}`port` does in a solve.

| | Port mode | Resonant mode |
| --- | --- | --- |
| Problem | {py:obj}`gfem.PortMode` | {py:obj}`gfem.Eigenmode` |
| Domain | a port's cross-section (2D, or 1D for a {term}`line port`) | the whole model |
| Given | the frequency $\omega$ | the Bloch wave vector, on a periodic cell |
| Found | the propagation constant $\gamma = \alpha + i\beta$ | the angular frequency $\omega$ |
| Result | `PortModeResult.gamma`, `n_eff` | `EigenmodeResult.omega`, `frequency`, `q` |
| Lossless and bound | $\gamma = i\beta$, $\alpha = 0$ | $\omega$ real, $Q = \infty$ |
| Losing power | $\alpha > 0$: the wave decays along the guide | $\mathrm{Im}\,\omega > 0$: the field decays in time |

```python
modes = gfem.solve(model, gfem.PortMode("p1", np.linspace(180e12, 200e12, 21)))
modes.gamma  # 1/m, alpha + i beta: the frequency was given
modes.n_eff  # gamma / (i k0)

resonances = gfem.solve(model, gfem.Eigenmode(target=193e12, n_modes=3))
resonances.omega  # rad/s, complex: the frequency was found
resonances.q  # |omega| / (2 Im omega)
```

## Port modes: bound and leaky

A port mode varies along the guide as $e^{-\gamma s}$ (see {doc}`conventions`).
Two things make $\alpha$ positive, and they are not the same:

- **Loss.** A lossy material or a conductor in the cross-section absorbs
  power as the wave travels.
- **Leakage.** The mode is not held by the guide and radiates sideways as it
  travels: a strip waveguide over a buried oxide too thin to isolate it from
  the substrate, a bend, a guide above the light line of a cladding. Such a
  mode is a {term}`leaky mode`. Its field decays along the guide but grows
  away from it, into the region it radiates into, so the cross-section must
  end in an absorbing edge or a {term}`perfectly matched layer`; behind a
  perfect conductor its radiation would reflect back and the section would
  have no leaky mode to find.

A {term}`bound mode` is one whose field decays away from the guide: below the
light line of every cladding, $\beta > k_0 n$ for each cladding index $n$. A
lossless bound mode has $\alpha = 0$. The fundamental mode of a silicon strip
waveguide is bound, even though the structure is open: nothing about it
radiates. A section that is open is not enough to make a mode leaky.

## Group index and dispersion

A port mode's propagation constant changes with frequency, and its
derivatives say how a pulse travels along the guide. Every row of a
{py:obj}`gfem.PortMode` grid carries them, computed exactly from the mode
problem at that frequency, not from differences across the grid:

| Result | Quantity | Unit |
| --- | --- | --- |
| `n_g` | {term}`group index` $n_g = c\,\mathrm{d}\beta/\mathrm{d}\omega = n_\mathrm{eff} + \omega\,\mathrm{d}n_\mathrm{eff}/\mathrm{d}\omega$ | 1 |
| `beta_2` | {term}`group-velocity dispersion` $\beta_2 = \mathrm{d}^2\beta/\mathrm{d}\omega^2$ | s²/m |
| `beta_3` | {term}`third-order dispersion` $\beta_3 = \mathrm{d}^3\beta/\mathrm{d}\omega^3$ | s³/m |
| `D` | {term}`dispersion parameter` $D = -(2\pi c/\lambda^2)\,\beta_2$, $\lambda = c/f$ | s/m² |

A pulse's envelope travels at the group velocity $v_g = c/n_g$. Where
$\beta_2 > 0$ ($D < 0$) the dispersion is normal: lower frequencies travel
faster. Where $\beta_2 < 0$ ($D > 0$) it is anomalous.

The materials and the guide's shape both disperse, and the one derivative
holds both. A region with a {term}`permittivity law` enters through the law's
own frequency dependence. On a lossless section the group index is also
$c$ times the energy the mode stores per unit length over the power it
carries, with the {term}`energy permittivity` weighting a law region. A
constant permittivity set to a law's value at one frequency gives the right
$n_\mathrm{eff}$ and the wrong $n_g$: 3 % low on a silicon wire.

With loss, $\gamma = \alpha + i\beta$ is complex, and so are its derivatives,
written $n' - in''$ like `n_eff` (the signs are on {doc}`conventions`). The
real parts describe the phase, as on a lossless guide, and the imaginary parts
say how the loss changes with frequency. What no longer holds with loss is the
energy picture: the stored energy over the power is then the speed of the
energy, not $c\,\mathrm{d}\beta/\mathrm{d}\omega$.

The error in these values is the mesh's and the section's truncation, never
a step. $\beta_2$ reacts to the section's edge far more than
$n_\mathrm{eff}$ does: around a silicon wire at its zero-dispersion width,
0.5 µm of oxide gets $n_\mathrm{eff}$ within $3 \times 10^{-4}$ and gives
$D$ the wrong sign, so converge $D$ in the {term}`clearance` on its own. The
calculation, and its checks: {doc}`recipes/waveguide-dispersion`.

These derivatives have no gradient: under `jax.grad` they raise
{py:obj}`gfem.NonDifferentiableError`, while $\gamma$ of the same result
follows a design at the port's face ({doc}`recipes/port-mode-gradient`).

## Bent port modes

A {term}`bent port mode` of {py:obj}`gfem.Bend` solves the bent guide exactly,
as the body of revolution about the bend's axis at the real frequency, on the
straight cross-section: the bend multiplies $\varepsilon$ and $\mu$ by
$\mathrm{diag}(h, h, 1/h)$ with $h = 1 + \xi/R$, where $\xi$ is the distance
from the {term}`reference radius` $R$'s arc along `outward`. In a perfectly
matched layer $h$ takes the stretched coordinate. The mode varies as
$e^{-\nu\varphi}$ with $\nu = -i\gamma R$, and $\gamma$, $n_\mathrm{eff}$,
$n_g$ and $\beta_k$ are per unit length of the arc. Moving $R$ moves the axis
and leaves the guide where it is.

- **Loss.** The {term}`bend loss` over 90° is
  $L_{90} = (10\pi/\ln 10)\,R\,\mathrm{Re}\,\gamma$ dB: radiation into the
  cladding plus absorption. The mode radiates beyond the {term}`caustic`
  $x_c = \mathrm{Re}\,\nu/(k_0 n_\mathrm{cl}) - R$, so the outer layer must
  start past it and a wavelength thick; gfem warns when it does not, or when
  an outer edge that is not a perfect conductor has no layer.
- **Junction.** Where a straight guide meets a bent one, the power the
  {term}`junction coupling` $\tau^2$ passes is
  $\big(\tfrac12(\langle s, b\rangle + \langle b, s\rangle)\big)^2 /
  (\langle s, s\rangle\langle b, b\rangle)$, with the reciprocity pairing of the
  port mode problem, unconjugated, so that it is 1 for a mode with itself. It
  neglects the reflection, and with unequal impedances $|\tau^2|$ can exceed 1
  by a little: 1.0047 for a WR-90 guide at $R = 30$ mm.
- **Fields.** The solve works with a field $e_n$ normal to the section; the
  physical normal component is $E_s = e_n/h$, which is what
  {py:obj}`gfem.PortModeField` and the plots return, for 1 W through the
  section.
- **Nonlinearity and rings.** The Kerr overlap weights the volume by $h$ and
  uses $E_s$. A ring of radius $R$ has resonances spaced by
  $D_1 = c/(n_g R)$ and $D_2 = -(c/n_g)\,D_1^2\,\beta_2$ with the bent $n_g$
  and $\beta_2$, and a radiation limit
  $Q_\mathrm{rad} = k_0\,\mathrm{Re}\,n_g/(2\,\mathrm{Re}\,\gamma)$.

The calculation, with its checks and limits: {doc}`recipes/waveguide-bends`.

## Resonant modes: closed and open

An {term}`eigenmode problem` finds the frequencies at which a source-free field
can exist. In a closed, lossless structure, such as a cavity with perfectly
conducting walls and lossless filling, these frequencies are real.

When the structure loses power, $\omega$ is complex and
$\mathrm{Im}\,\omega > 0$ (time dependence $e^{+i\omega t}$, so the field decays
as $e^{-\mathrm{Im}\,\omega\, t}$). Material loss does this, and so does
anything that lets the field leave: a resistive {term}`lumped port`, a
{term}`wave port`, a first-order `Absorbing` boundary, a perfectly matched
layer. A resonant mode of
an open structure is a {term}`quasi-normal mode` (QNM). Its field decays in
time and, because the power it loses travels outwards, grows away from the
structure; the perfectly matched layer damps that growth inside the layer.
Its quality factor is

$$
Q = \frac{|\omega|}{2\, \mathrm{Im}\,\omega}.
$$

The modes give the poles of a driven response, their Q and, under `jax.grad`,
how both move with the design ({doc}`recipes/eigenmode-gradient`). The response
over a band comes from a driven sweep ({doc}`recipes/frequency-sweep`).

On a periodic cell with a fixed Bloch wave vector, `Eigenmode(wave_vector=...)`
gives the modes of that wave vector; one problem per wave vector traces the
band diagram $\omega(k)$. In a lossless cell a band of bound modes is real and a
band of leaky modes is complex.

### A resonance through a wave port

A {term}`wave port` in an `Eigenmode` problem absorbs its mode at the
complex eigenfrequency. Its propagation constant there is the
{term}`outgoing branch`,

$$
\gamma = i\sqrt{\varepsilon_r \mu_r k_0^2 - k_c^2},
$$

continued from real frequencies above the cutoff $k_c$, where it is the wave
leaving the model. At a decaying mode, $\mathrm{Im}\,\omega > 0$, this gives
$\mathrm{Re}\,\gamma < 0$: the field grows along the guide away from the
structure. That is the guide's share of the spatial growth of a
quasi-normal mode.

The target must lie above the cutoff of every requested port mode. Below a
cutoff the outgoing branch also has roots that are not resonances (anti-bound
ones, or ones that grow in time); gfem drops them with
{py:obj}`gfem.AntiBoundRootWarning`. A root below cutoff that decays in time
is a resonance and is kept.

The port's $\gamma$ at the complex eigenfrequency is not reported;
{py:obj}`gfem.PortMode` gives it at real frequencies. Each wave port reports
its {term}`coupling Q` in `EigenmodeResult.q_port`: $Q_p = \mathrm{Re}\,\omega\,W / P_p$,
with $W$ the mode's stored energy and $P_p$ the time-averaged power port $p$
takes out (for a resistive lumped port the power its resistance draws, for a
wave port the power its requested mode carries out through the face). With
other losses, $1/Q \approx \sum_p 1/Q_p$ plus the material and wall losses,
closer as Q grows.

**Which resonances are found.** With a wave port, an `Eigenmode` solve returns
the `n_modes` resonances with $\mathrm{Im}\,\omega \le \omega_t / 4$
($\omega_t = 2\pi\,$`target`, so a Q above about 2 near the target) whose
$\mathrm{Re}\,f$ is nearest `target`. Asking for more modes adds modes at the end and never changes the first
ones. A resonance with a Q below about 2 is out of reach, even when it lies
nearest `target`: find it from a `gfem.Driven` sweep. If the strip holds more
modes than the search can list, `gfem.EigenmodeConvergenceError` says so; lower
`n_modes` or move `target`.

**Modes the ports do not absorb.** A wave port absorbs the modes it requests
exactly, and any other field on its face only partly. For each port surface
$s$, `EigenmodeResult.requested_power_share` gives the part of the surface's
absorbed power that its requested modes carry,

$$
\eta_s = \frac{\sum_{p \in s} P_p}{P_s},
$$

with $P_p$ the power the requested mode of port $p$ carries out and $P_s$ the
power the surface absorbs: the {term}`requested power share`. It is near 1
for a mode of a requested family. A mode carried by an
{term}`unrequested port mode`, or a {term}`trapped mode`, has a share near 0
and a Q that is the port's, not the device's. When the share is below one
half the solve warns `gfem.UnrequestedPortModeWarning` and tells which of two
things to do. If an unrequested mode propagates at the resonance, request it
on the surface with another `gfem.WavePort` and solve again; if the warning
stays, the mode is trapped. If none propagates, the mode is trapped inside the
model: its Q is a lower bound on the device's, so move the port away. A share
from one half to 1 is reported, not warned: a discontinuity close to the port
lowers both the share and the Q, and moving the port away restores them.

## One leak, seen from both sides

A leaky mode is one physical wave. Fix the frequency and it decays in space
($\alpha > 0$, a port mode); fix the wave number and it decays in time
($\mathrm{Im}\,\omega > 0$, a resonant mode of a periodic cell). For weak
leakage, $\alpha \ll \beta$, the two rates are tied by the group velocity
$v_g = \mathrm{d}\omega/\mathrm{d}\beta = c/n_g$ along the guide
(`PortModeResult.n_g`):

$$
\mathrm{Im}\,\omega \approx v_g\, \alpha .
$$

The same holds for the losses of a bound mode. A guide coupled to a cavity
mixes the two pictures: the guide's own mode stays bound, while the cavity's
resonance becomes a quasi-normal mode whose $\mathrm{Im}\,\omega$ counts the
power it loses into the guide. An `Eigenmode` problem with a wave port on the
guide finds that quasi-normal mode directly, and `q_port` gives the share of
its loss that leaves through the guide.

## The three roles of a port

A {term}`port` does three things, and only one of them depends on the problem.

1. **It is a boundary, always.** A wave port absorbs its own mode without
   reflection; a lumped port is a resistor, with an optional inductance and
   capacitance. Both are in the operator of every problem on the model, so
   they set the resonances and their Q. In an `Eigenmode` problem a port is
   only this boundary: a resistive lumped port and every wave port add a
   {term}`coupling Q` (`EigenmodeResult.q_port`), and an inductive lumped
   port a participation ratio.
2. **It is a source, when excited.** A {term}`driven problem` feeds each
   {term}`excited port <Excited port>` in turn with 1 W, one column of the
   {term}`S-parameters` each. `Driven(excite=...)` names those ports; the
   default excites every port.
3. **It is read, always.** S has a row for every port, excited or not.

```python
# Feed p1 only; p2 still terminates the device and S still reads it
result = gfem.solve(model, gfem.Driven(frequencies, excite="p1"))
```

Changing `excite=` changes only the right-hand side, never the operator: a
port that is not excited loads the device exactly as an excited one does.

### When the design reaches a port

A wave port reads S against its own mode, solved on its face. A design that
reaches the face (a material of the elements on it, a sheet on its edge, or
the face's nodes) changes that mode, and the port follows: $\gamma$,
$Z_{PV}$ and the reference impedance move with the design, and S stays in
the pseudo-waves of the port's new mode. That is the right reference when
the port's guide is part of the design, a taper's output for instance. When
the device feeds a fixed external line or guide, the reference must not move:
renormalize S to that line's impedance, or keep a stretch of the external
guide's own section between the design and the port
({doc}`recipes/port-mode-gradient`).
