# Multiphysics couplings

How does a temperature, an applied voltage or a strain change the optical permittivity or a mechanical mode, and which sign and unit does each coupling carry?

Each coupling is a first-order response about a solved background, with the
time dependence $e^{+i\omega t}$ of {doc}`conventions`. The rules are one entry
each there; this page derives them and checks the thermal and electro-optic
ones on a solve. The sign of loss and the S convention are those of
{doc}`phasors-and-power-waves`.

(theory-multiphysics-couplings-temperature-and-thermo-optic-phase)=
## Temperature and thermo-optic phase

A thermal solve's temperatures are absolute, in kelvin, and its rise is the
temperature minus the {term}`source-free temperature`: the temperature with
every heat source off, which the boundaries alone set. The device here is the
heater section of {doc}`recipes/heater-temperature`, a TiN strip over a
silicon waveguide, read as a {term}`slab cross-section`.

**Kelvin.** A temperature at or below 0 K raises, whatever its origin:

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/heater_w2.msh"
urllib.request.urlretrieve(url, "heater_w2.msh")
heater_section = gfem.Mesh.from_gmsh_cross_section("heater_w2.msh", length_unit="um")
try:
    gfem.FixedTemperature("bottom", temperature=25.0 - 273.15)  # 25 degC as -248 K
except ValueError as error:
    print(type(error).__name__)
```

```text
ValueError
```

**The rise.** With the sink at 300 K and the air above at 310 K, the
source-free temperature is not uniform. The rise is the heater's own part,
linear in its power, and the temperature is the rise plus the source-free
temperature. Every extensive quantity on a cross-section is per metre of
guide: the slab's totals are for its `section_thickness`, so read
`power_per_length`.

```python
oxide = gfem.Material(
    eps_r=1.444**2, thermal_conductivity=1.38, thermo_optic_coefficient=1e-5
)
silicon = gfem.Material(
    eps_r=3.4777**2, thermal_conductivity=90.0, thermo_optic_coefficient=1.86e-4
)
stack = {
    "substrate": gfem.Material(thermal_conductivity=148.0),
    "box": oxide,
    "cladding": oxide,
    "window": oxide,
    "neighbour_window": oxide,
    "core": silicon,
    "neighbour": silicon,
    "heater": gfem.Material(thermal_conductivity=28.0, conductivity=2.3e6),
}
heater = gfem.Model(
    heater_section,
    materials=stack,
    boundaries=[
        gfem.FixedTemperature("bottom", temperature=300.0),
        gfem.Convection("top", coefficient=10.0, ambient=310.0),
    ],
    ports=[gfem.WavePort("core", "front")],
    excluded={"em": ["substrate", "box", "cladding", "heater", "neighbour"]},
)
dc = gfem.solve(heater, gfem.DCConduction({"heater": "front"}, ground="back"))
heat = gfem.joule_heating(dc, power_per_length=10.0)  # W per metre of guide
steady = gfem.solve(heater, gfem.SteadyThermal(heat))
cold = gfem.solve(heater, gfem.SteadyThermal())  # no source: the source-free one
core = {
    name: float(gfem.region_mean(field, "core"))
    for name, field in (
        ("T", steady.temperature),
        ("rise", steady.rise),
        ("T0", cold.temperature),
    )
}
print(f"T {core['T']:.4f} K = rise {core['rise']:.4f} K + T0 {core['T0']:.4f} K")
print(float(steady.power) / heater_section.section_thickness)  # W/m, slab / thickness
print(float(steady.power_per_length))  # W/m
```

```text
T 301.7118 K = rise 1.7117 K + T0 300.0002 K
9.999999999999817
9.999999999999817
```

**Heat flow is positive out.** {py:obj}`gfem.heat_flow` is the heat leaving
a side, in W; on a cross-section {py:obj}`gfem.heat_flow_per_length` gives it
in W/m. It reads the absolute temperature, so a flow the boundaries drive
with no source on, here from the warmer air down to the sink, is part of it;
the flows out of every boundary surface still add up to the power the
sources put in:

```python
sink = float(gfem.heat_flow_per_length(steady, "bottom", model=heater))  # W/m
air = float(gfem.heat_flow_per_length(steady, "top", model=heater))
print(f"out through the sink {sink:.4f} W/m, through the top {air:.4f} W/m")
print(f"together: {sink + air:.6f} W/m")
```

```text
out through the sink 10.0037 W/m, through the top -0.0037 W/m
together: 10.000000 W/m
```

The top's flow is negative: heat enters there, from the air.

**Time** is in seconds: a transient's `duration`, its waveform's start times
and the times it returns ({doc}`recipes/thermal-step-response`).

A transient run's {py:meth}`gfem.Trace.heat_out` is the heat that has left
since $t = 0$, in J (J/m with {py:meth}`gfem.Trace.heat_out_per_length`), so
it starts at 0; its step difference over `dt` is the mean flow over the
step, in W.

**Electromagnetic heat** is time-averaged power.
{py:obj}`gfem.dissipated_power` gives a kept driven field's loss per region
and sheet in W, for its 1 W incident (`INCIDENT_POWER`): the fraction of the
incident power each one dissipates. On a port with a complex reference
impedance the field is per unit $|a|^2$, not exactly 1 W incident. A
{py:obj}`gfem.Conductor` sheet dissipates $\mathrm{Re}(1/Z_s)\, |E_t|^2/2$
per area and an {py:obj}`gfem.Impedance` sheet $|E_t|^2/(2R)$, times its
fill, with $E_t$ the peak tangential field. {py:obj}`gfem.electromagnetic_heating`
scales those losses by the incident power of each port, in W (W/m on a
cross-section), into a heat source ({doc}`recipes/electromagnetic-heating`).

**The thermo-optic phase is a delay.** A material's
`thermo_optic_coefficient` $dn/dT$ acts on the rise, about the index of its
`eps_r`: that index is the one at the source-free temperature, and no
reference temperature is stored. The phase per length a rise adds to a
guided mode is positive for a positive $dn/dT$ and a positive rise, like the
phase $\beta L$ that a length $L$ of guide delays light by. Under $e^{+i\omega t}$ the guide
transmits $e^{-i\beta L}$, so a transmission's argument moves by minus the phase:

```python
f_optical = 299_792_458.0 / 1.55e-6  # Hz
shift = gfem.thermo_optic_shift(heater, steady.rise, port="core", frequency=f_optical)
print(float(shift.phase_per_length) > 0)
length = 320e-6  # m of heated guide
print(np.angle(np.exp(-1j * float(shift.phase_per_length) * length)))  # rad
```

```text
True
-0.4458987292927867
```

The {doc}`recipes/thermo-optic-phase` recipe turns the phase into $P_\pi$, and
the {py:obj}`gfem.conventions` module, "Temperature and thermo-optic phase",
gives the first-order formula.

(theory-multiphysics-couplings-electro-optic-shift)=
## Electro-optic shift

A crystal's `electro_optic_tensor` is the contracted tensor $r_{Ik}$ in m/V,
a 6 × 3 array: rows $xx, yy, zz, yz, xz, xy$, columns $x, y, z$, in the frame
of the material's `axes`, whose rows are the crystal's $x$, $y$ and $z$ in the
model's coordinates (a right-handed set). A static field
$E_\mathrm{s} = -\nabla V$ changes the optical permittivity by
$\Delta\varepsilon_r = -\varepsilon_r (r \cdot E_\mathrm{s}) \varepsilon_r$.
The device here is the section of {doc}`recipes/electro-optic-phase`: x-cut
lithium niobate ridges between a signal and two grounds, coarsely meshed,
with only $r_{33}$ in the tensor.

**Per volt, and the phase is a delay.** The potential is one terminal's at
1 V, every other terminal and the ground at 0 V, and the shift is per volt
of it. `phase_per_length_per_volt` is $k_0\,\partial n_\mathrm{eff}/\partial V$, the extra
phase delay per metre and per volt, like the thermo-optic phase: a
transmission through a length $L$ at $V$ volts turns its argument by
$-\phi V L$, with $\phi$ the `phase_per_length_per_volt`. On the right-hand arm the field points
along the crystal's $+z$ and lowers the index, so the delay is negative and
the argument turns forward:

```python
url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/tfln_section.py"
)
urllib.request.urlretrieve(url, "tfln_section.py")
from tfln_section import tfln_section  # noqa: E402

axes = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))  # x-cut
r33_only = np.zeros((6, 3))
r33_only[2, 2] = 30.9e-12  # row zz, column z, m/V
ln_static = gfem.Material(eps_r=(44.0, 44.0, 27.9), axes=axes)
ln = gfem.Material(
    eps_r=(2.2111**2, 2.2111**2, 2.1376**2), axes=axes, electro_optic_tensor=r33_only
)
oxide_static, oxide = gfem.Material(eps_r=3.9), gfem.Material(eps_r=1.444**2)
modulator = tfln_section(gap=3.5e-6, width=0.9e-6, coarsen=2.0)
electrodes = ["signal", "ground_r", "ground_l"]
static = gfem.Model(
    modulator.static,
    materials={
        "air": gfem.Material(),
        **dict.fromkeys(("cladding", "box"), oxide_static),
        **dict.fromkeys(("slab", "ridge_a", "ridge_b"), ln_static),
        **{name: gfem.Material() for name in electrodes},
    },
    excluded={"em": electrodes},
)
es = gfem.solve(
    static,
    gfem.Electrostatic({"signal": ["signal_s"]}, ground=["ground_s"]),
    options=gfem.SolverOptions(order=2),
)
arms = {}
for arm, mesh in (("a", modulator.window_a), ("b", modulator.window_b)):
    window = gfem.Model(
        mesh,
        materials={"cladding": oxide, "box": oxide, "slab": ln, "ridge": ln},
        ports=[gfem.WavePort(arm, "front")],
    )
    arms[arm] = gfem.electro_optic_shift(
        window, es.potential["signal"], port=arm, frequency=f_optical, mode=2
    )
a = arms["a"]
print(float(a.delta_n_eff_per_volt) < 0)
volts, length = 5.0, 2e-3  # V, m
delay = float(a.phase_per_length_per_volt) * volts * length  # rad
print(f"{delay:.4f}")
print(f"{np.angle(np.exp(-1j * delay)):.4f}")  # the transmission's argument
```

```text
True
-0.9435
0.9435
```

**$V_\pi L$ per arm, and push-pull.** `v_pi_l` is one arm's
$V_\pi L = \lambda / (2\,|\partial n_\mathrm{eff}/\partial V|)$ in V m. A push-pull pair of arms
driven by one potential has $V_\pi L = \lambda / (2\,|s_a - s_b|)$, with $s$ each
arm's `delta_n_eff_per_volt`; here the arms' shifts are equal and opposite
to the coarse meshes' asymmetry, so the pair needs half an arm's:

```python
b = arms["b"]
pair = 1.55e-6 / (2 * abs(float(a.delta_n_eff_per_volt - b.delta_n_eff_per_volt)))
print(f"arm a {100 * float(a.v_pi_l):.3f} V cm, push-pull {100 * pair:.3f} V cm")
```

```text
arm a 3.330 V cm, push-pull 1.663 V cm
```

Solve each arm's mode on an optical window that reaches the electrodes'
inner edges, as here: the shift weighs the mode's tail, where the static
field is strongest. The {term}`electrode clearance` has a lower bound
$\ln(1/\delta) / (2\kappa)$, with
$\kappa = k_0 \sqrt{n_\mathrm{eff}^2 - n_\mathrm{slab}^2}$, so that the
mode's intensity at the metal is $\delta$ of its value at the ridge's edge.
The {py:obj}`gfem.conventions` module, "Electro-optic
shift", gives the formulas, and {doc}`recipes/electro-optic-phase` the
whole calculation.

(theory-multiphysics-couplings-photoelastic-tensor)=
## Photoelastic tensor

A crystal's `photoelastic_tensor` is the contracted tensor $p^E_{IJ}$, a
dimensionless 6 × 6 array: rows and columns $xx, yy, zz, yz, xz, xy$, in the
frame of the material's `axes` (a right-handed set), with the engineering shear
strain, so that the strain $S$ changes the inverse permittivity by
$\Delta(\varepsilon_r^{-1})_I = \sum_J p_{IJ} S_J$. It is rotated into the
model's frame like a stiffness, $M p M^\mathsf{T}$, and is not assumed
symmetric. It is the tensor at constant electric field: the piezoelectric field
of the wave acts through the clamped electro-optic tensor $r^S$, and a free
$r^T$ would count the strain twice. The optical material carries it; a material
with a stiffness or a piezoelectric tensor refuses it.

{py:obj}`gfem.acousto_optic_shift` turns the strain and field of a
piezoelectric result into the permittivity change
$\Delta\varepsilon_r = -\varepsilon_r (p S + r E)\varepsilon_r$, and weighs it
with a mode solved once. **The shift is a phasor.** With the terminal driven at
$\mathrm{Re}(V e^{i\Omega t})$, gfem's $e^{+i\omega t}$, the effective index
is $n_\mathrm{eff} + \mathrm{Re}(\delta n\,V e^{i\Omega t})$ to first order,
$\delta n$ the complex `delta_n_eff_per_volt` of the acoustic frequency
$\Omega/2\pi$. A push-pull pair has
$V_\pi L = \lambda / (2\,|\delta n_a - \delta n_b|)$ per volt on the fingers;
per volt of an incident wave on a source impedance $Z_0$ it is that times
$|(Z_L + Z_0)/(2Z)|$, with $Z = 1/(W Y')$ and $Z_L = Z + R_s$. Only the
photoelastic and electro-optic terms are counted, not the moving boundaries or
the crystal's rotation. {doc}`recipes/acousto-optic-shift` has the whole
calculation.

(theory-multiphysics-couplings-optomechanical-coupling)=
## Optomechanical coupling

{py:obj}`gfem.optomechanical_coupling` reads the modes of a
{py:class}`gfem.MechanicalEigenmode` solve at $q = 0$, each normalized to 1 J per
metre of guide, against the mode of a wave port. A displacement $\alpha u$ of
dimensionless amplitude $\alpha$ changes the effective index at the fixed optical
frequency $\omega$ by $\partial n_\mathrm{eff}/\partial\alpha$, the sum of a
moving-boundary and a photoelastic part
({py:obj}`gfem.optomechanical_overlap`); the coupling per root length is

$$
\tilde g_0 = \frac{\omega}{n_g}\,\frac{\partial n_\mathrm{eff}}{\partial\alpha}\,
\sqrt{\frac{\hbar\,\mathrm{Re}\,\Omega}{4}},
\qquad g_0(L) = \frac{\tilde g_0}{\sqrt{L}},
$$

with $\Omega$ the mechanical and $\omega$ the optical angular frequency and $n_g$
the port mode's group index. **Units are angular**: $\tilde g_0$ in
rad s$^{-1}$ m$^{1/2}$, $g_0$ in rad/s; divide by $2\pi$ for hertz.
**The sign** is that of $-\partial\omega_c/\partial\alpha$: positive when the
displacement raises the effective index, and it follows the mechanical mode's
phase convention (its largest displacement coefficient real and positive), so
read $|g_0|$; the two parts of one mode keep their relative sign. The values are
complex for a lossy mode and real to round-off otherwise. The forward
intramodal Brillouin gain of a guide is
$G_\mathrm{B} = 4 Q_m \tilde g_0^2/(\hbar\omega\,v_g^2\,\mathrm{Re}\,\Omega)$
with $v_g = c/n_g$ (Van Laer et al., Phys. Rev. A 93, 053828 (2016)). The
rotation of the crystal by the displacement is not counted.
{doc}`recipes/optomechanical-coupling` has the whole calculation.

(theory-multiphysics-couplings-piezoelectric-problems)=
## Piezoelectric problems

A {py:class}`gfem.Piezoelectric` problem solves linear piezoelectricity on a
2D section: the $(x, y)$ plane, $\partial_z = 0$, and three displacement
components $u_x, u_y, u_z$. With $e^{+i\omega t}$, the stress and the electric
displacement are

$$
T = c^E S - e^\mathsf{T} E, \qquad D = e S + \varepsilon^S E, \qquad
E = -\nabla\varphi,
$$

with $\nabla\cdot T = -\omega^2 \rho\, u$ and $\nabla\cdot D = 0$. A material's
`stiffness` is $c^E$ in Pa, 6 × 6, at constant field; its `piezoelectric_tensor`
is $e$ in C/m², 3 × 6, the stress constants and not the strain constants $d$;
its `eps_r` is $\varepsilon^S / \varepsilon_0$, at constant strain. All three
are in the frame of the material's `axes`, whose rows are the crystal's $x$, $y$
and $z$ in the model's coordinates (a right-handed set).

**Voigt order and engineering shear.** The six strain components are ordered
$xx, yy, zz, yz, xz, xy$, with the shear entries twice the tensor components:

$$
S = (\partial_x u_x,\ \partial_y u_y,\ 0,\ \partial_y u_z,\ \partial_x u_z,\
\partial_y u_x + \partial_x u_y).
$$

{py:func}`gfem.strain` returns this $S$ at points, per volt. The shear wave of
an x-cut film, polarized along the fingers, is the fifth entry,
$S_{xz} = \partial_x u_z$.

**Loss is a complex stiffness.** Under $e^{+i\omega t}$ the stiffness is
$c^E (1 + i(1/Q_m + \eta r^2))$, with $Q_m$ the material's
`mechanical_quality_factor` and $\eta r^2$ a {py:class}`gfem.DampingFrame`'s:
$r$ is the normalised depth into the frame and $\eta$ its `loss_factor`.
The imaginary part is positive, the opposite of a permittivity's
$\varepsilon_r (1 - i\tan\delta)$: a solid dissipates
$\tfrac{\omega}{2} S^* \,\mathrm{Im}(c)\, S \ge 0$, a dielectric
$\tfrac{\omega}{2}\,\mathrm{Im}(-\varepsilon)|E|^2 \ge 0$ ({doc}`"Sign of loss" <phasors-and-power-waves>`).
A frame of thickness $L$ in a medium of wavelength $\lambda$ reflects a plane
wave at normal incidence by $|R| \approx e^{-2\pi\eta (L/\lambda)/3}$, down to
a floor of about $3.4\times10^{-3}\,\eta\,(\lambda/L)^2$: four wavelengths at
the default $\eta = 1.2$ give $|R| < 10^{-3}$.

**Per metre along $z$.** For each terminal $j$ in turn, the potential is 1 V on
it and 0 V on every other terminal and the ground. The result's
`admittance_per_length` is $Y'_{ij}(f) = i\omega q'_i$ in S/m, the current into
terminal $i$ per volt on terminal $j$ per metre ($q'$ the charge per length),
indexed `[frequency, i, j]`: the response first, the excitation second, as in
S-parameters. `clamped_capacitance_per_length` is $C_0'$ in F/m, the capacitance
of the potential alone with $\varepsilon^S$ and no displacement. It is not the
low-frequency limit of $Y'/i\omega$, the free capacitance, which is larger by
$1 + K^2$ with $K^2$ the crystal's coupling. A device of aperture $W$ has
$Y = W Y'$, as every extensive quantity of a 2D model scales
({doc}`2D models <conventions>`).

**Network parameters.** `network_parameters(aperture, series_resistance=,
reference_impedance=)` puts the resistances $R = \mathrm{diag}(R_s)$ in the
leads and refers S to the reference impedances (50 Ω by default), as
pseudo-waves ({doc}`S-parameters and power <phasors-and-power-waves>`):

$$
Y = W Y', \qquad Y_t = Y (I + R Y)^{-1}.
$$

For one terminal on a real $Z_0$, $S_{11} = (Z_L - Z_0)/(Z_L + Z_0)$ with
$Z_L = 1/Y + R_s$.

**Per terminal volt, and incident referral.** The kept displacement and
potential and the strain are per volt across the terminal, after $R_s$. A
device fed by an incident wave $V_\mathrm{inc}$ on $Z_0$ has across its one
terminal

$$
V_\mathrm{IDT} = V_\mathrm{inc}\,\frac{2 Z_\mathrm{IDT}}{Z_L + Z_0}, \qquad
Z_\mathrm{IDT} = 1/Y, \quad Z_L = Z_\mathrm{IDT} + R_s,
$$

so a quantity per terminal volt is referred to the incident wave by that
factor. The {doc}`recipes/piezoelectric-admittance` recipe and the
{doc}`transducer tutorial <tutorials/transducer>` use these conventions; the
{py:obj}`gfem.conventions` module, "Piezoelectric problems", states them.

(theory-multiphysics-couplings-mechanical-eigenmodes)=
## Mechanical eigenmodes

A {py:class}`gfem.MechanicalEigenmode` problem finds the source-free solutions
of the same elasticity as a piezoelectric problem (and, when a material carries
a piezoelectric tensor, with its potential coupled), on a 2D section, for fields
of the form $u(x, y)\,e^{i\omega t - iqz}$: $q$ is the {term}`axial
wavenumber` in rad/m, and $\partial_z \to -iq$ in the strain. A mode with
$q > 0$ travels along $+z$ where its group velocity is positive. The stiffness
and the loss are the piezoelectric problem's, $c\,(1 + i/Q_m)$ and a
{py:class}`gfem.DampingFrame`'s, under $e^{+i\omega t}$, so a decaying mode has
$\mathrm{Im}\,\omega > 0$ and

$$
Q = \frac{\mathrm{Re}\,\omega}{2\,\mathrm{Im}\,\omega}.
$$

**Normalization.** A mode's displacement carries
$\tfrac12 (\mathrm{Re}\,\omega)^2 \int \rho\,|u|^2\,\mathrm{d}A = 1$ J per metre
along $z$, with its largest coefficient real and positive. It is the
time-averaged energy of a lossless mode, which splits equally between kinetic
and strain energy.

**Group velocity.** `group_velocity` is $\mathrm{Re}\,\mathrm{d}\omega/\mathrm{d}q$
in m/s, the exact derivative of the discrete band. Every section is
reciprocal: $\omega(-q) = \omega(q)$, the group velocity at $-q$ is the opposite,
and at $q = 0$ it is exactly 0 (no derivative is assembled there).

**Clamped components.** `clamped` fixes all three components on a line by its
name, or only the components named by a string of `"xyz"` in the lab axes: a
line of constant $x$ held with `"x"` keeps $u_x = 0$ and slides in $y$ and $z$.
Every other line is traction-free. A section free at $q = 0$ has rigid-body
motions at $\omega = 0$ that are not returned.

## Validity and limits

Every shift is first order in the field, strain or rise. The acousto-optic and
optomechanical couplings count the photoelastic and electro-optic terms and
the moving boundary, not the crystal's rotation by the displacement. A
transducer's admittance is per metre along $z$, so a device of aperture $W$
scales it by $W$.
