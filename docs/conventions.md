# Conventions

Which sign, unit and normalization is behind a number gfem returns? The full
statement is the {py:obj}`gfem.conventions` module, the one place where each
convention is written out. This page shows each of them on a solve, so you
can tell which one you are looking at, and names the constant that records
it. The device is the parallel-plate line of {doc}`get-started`.

| Constant | Value | What it fixes |
| --- | --- | --- |
| `gfem.conventions.TIME_DEPENDENCE` | `exp(+iωt)` | the time dependence of every complex phasor |
| `gfem.conventions.PHASOR` | `peak` | a phasor's modulus is the amplitude of the oscillation, not its RMS value |
| `gfem.conventions.WAVES` | `pseudo` | S relates pseudo-waves (power waves on a real reference) |
| `gfem.conventions.INCIDENT_POWER` | 1 W | the incident power behind every driven field |
| `gfem.conventions.MODE_ENERGY` | 1 J | the electric energy every mode field stores |
| `gfem.conventions.BLOCH_PHASE` | `E(r + a) = exp(-i k·a) E(r)` | the phase across a periodic pair |
| `gfem.conventions.INTERIOR_SHEET_ADMITTANCE` | 0.5 | a thin `Conductor` film on an interior surface, in units of $\sigma t$ |
| `gfem.conventions.EXTERIOR_THICKNESS_FACTOR` | 2 | how often a `Conductor`'s thickness counts on an exterior surface |

## Units

Everything you pass to gfem and everything it returns is in SI units: metres,
hertz, ohms, farads, watts, V/m. The one exception is the mesh file, whose
length unit you state once when you read it; gfem converts the coordinates.

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")
print(mesh.length_unit)  # the unit of the file
print(mesh.imported_nodes.max(axis=0))  # the far corner of the line, m
```

```text
mm
[0.04  0.006 0.001]
```

These quantities use the same SI units, with temperatures in kelvin (see
"Temperature and thermo-optic phase" below):

| Quantity | Unit |
| --- | --- |
| temperature, temperature rise, `ambient` | K |
| `thermal_conductivity` | W/m/K |
| `mass_density` | kg/m³ |
| `specific_heat` | J/kg/K |
| `Convection` `coefficient` | W/m²/K |
| `HeatFlux` `flux` | W/m² |
| electrical `conductivity` | S/m |
| `thermo_optic_coefficient` | 1/K |
| heat (`SteadyThermalResult.heat`), `power_density` | W/m³ |
| time, `duration` | s |
| `heat_flow`, `dissipated_power`, `incident_power` | W |
| `Trace.heat_out` | J |
| on a cross-section: `power_per_length`, `resistance_per_length`, `capacitance_per_length` | W/m, Ω/m, F/m |
| on a cross-section: `heat_flow_per_length`, `incident_power_per_length` | W/m |
| on a cross-section: `Trace.heat_out_per_length` | J/m |
| `electro_optic_tensor` | m/V |
| `delta_n_eff_per_volt`, `phase_per_length_per_volt`, `v_pi_l` | 1/V, rad/(V m), V m |
| a port mode's `n_g`, `beta_2`, `beta_3`, `D` | 1, s²/m, s³/m, s/m² |

## Time dependence

The physical field is $\mathrm{Re}(E\, e^{+i\omega t})$, the electrical engineering convention
(`TIME_DEPENDENCE`): write $j$ for $i$ and the formulas of a circuit textbook
apply. A wave travelling along $+x$ is then $e^{-i\beta x}$, so the phase of a
transmission falls with distance, and with frequency. On the line with its
ceramic section, from 2 to 8 GHz:

```python
z_line = 376.73 / 6  # ohm
plates = [gfem.PEC("bottom"), gfem.PEC("top")]
ports = [
    gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
    gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
]
ceramic = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)}
model = gfem.Model(mesh, materials=ceramic, boundaries=plates, ports=ports)

band = gfem.Driven(frequency=np.linspace(2e9, 8e9, 7), excite="p1", keep_fields=5e9)
result = gfem.solve(model, band)
print(np.unwrap(np.angle(result.s["p2", "p1"])))  # the phase of S21, rad
```

```text
[ 3.05180459  0.79326538 -0.47316251 -2.65371362 -4.14297302 -5.8252978
 -7.83283125]
```

A code that uses $e^{-i\omega t}$ gives the complex conjugate of every S entry and
every field.

## S-parameters and power

S relates {term}`pseudo-waves` (`WAVES`), normalized to each port's
{term}`reference impedance`. On a real reference they are the
{term}`power waves`: $|a|^2$ is the power a wave carries, in W. A wave port
with a `voltage_path` on a lossy line has a complex reference, its mode's
$Z_{PV}$, and the line is then matched to its own mode: $S_{11} = 0$.
Phasors are peak values (`PHASOR`), so a time-averaged power carries a
factor $\tfrac{1}{2}$, as in $|V|^2 / (2R)$. A kept driven field is the field for
`INCIDENT_POWER`, 1 W, sent into the excited port, with every other port
matched. The power that field carries through the feed line should then be
what port 1 does not reflect, $1 - |S_{11}|^2$ W:

```python
field = result.e_field(5e9, "p1")
through = gfem.power_flow(field, lambda c: c[:, 0] < 5e-3, model=model)
print(through)  # W crossing x = 5 mm towards the ceramic
print(1 - abs(result.s["p1", "p1"][3]) ** 2)  # what port 1 does not reflect at 5 GHz
```

```text
0.8123995326128145
0.8124000081969966
```

## SAX circuits

A solved S can be a block of a [SAX](https://github.com/flaport/sax) circuit,
through `result.s.to_sdict(...)` or a {py:obj}`gfem.SaxComponent`. What the
circuit receives depends on its kind, so you state it (`circuit="optical"` or
`"rf"`): the wrong kind still composes, with wrong interference.

```python
both = gfem.solve(model, gfem.Driven(frequency=5e9))  # every port excited
net = both.s
names = {"p1": "o1", "p2": "o2"}
photonic = net.to_sdict(circuit="optical", ports=names)
rf = net.to_sdict(circuit="rf", ports=names)  # renormalized to 50 ohm
print(photonic["o1", "o2"][0] == np.conj(net["p2", "p1"][0]))
print(abs(rf["o1", "o2"][0] - net.renormalize(50.0)["p2", "p1"][0]) < 1e-12)
```

```text
True
True
```

- A photonic component hands SAX $\overline{S}$: SAX's photonic models are
  $e^{-i\omega t}$ (a straight guide of length $L$ is $e^{+ikL}$), gfem's fields
  $e^{+i\omega t}$. S keeps its modal unit-power normalization, and the
  circuit's wavelength `wl` is in µm.
- An RF component hands S unchanged in time convention, as SAX's RF models, scikit-rf
  and Touchstone share gfem's $e^{+j\omega t}$, renormalized to one real
  reference impedance `z0` (50 Ω unless given) for the whole circuit
  ({term}`common reference impedance`), with the frequency `f` in Hz.
- Keys are `(input, output)` circuit port names, so `sd["o1", "o2"]` is
  $S_{21}$: the wave out of `o2` when `o1` is fed. Every pair is present, zeros
  included, because SAX reads a missing pair as zero.
- The sign of a wave port's row and column of S follows its
  {term}`polarity reference` (or its `voltage_path`), not the mesh.

The recipe {doc}`recipes/sax-circuit` builds a photonic and an RF circuit from
these blocks.

## Sign of loss

Under $e^{+i\omega t}$ a lossy permittivity has a negative imaginary part,
$\varepsilon = \varepsilon' (1 - i \tan\delta)$, and a mode that decays has $\mathrm{Im}\,\omega > 0$. Its quality factor
is $|\omega| / (2 |\mathrm{Im}\,\omega|)$. With the ports removed and a
{term}`loss tangent` of 1e-3 in the ceramic, the line's first resonance:

```python
lossy = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0, tan_delta=1e-3)}
resonator = gfem.Model(mesh, materials=lossy, boundaries=plates)
modes = gfem.solve(resonator, gfem.Eigenmode(target=2e9))
omega = modes.omega[0]  # rad/s, complex
print(omega.real / 2e9 / np.pi, omega.imag > 0)  # GHz, and whether it decays
print(modes.q[0], abs(omega) / (2 * omega.imag))
```

```text
2.1419711636402075 True
1251.6804868146692 1251.6804868146692
```

## Dispersive materials

A {term}`permittivity law` follows the same sign of loss. Its parameters are
ordinary frequencies in hertz ($\omega = 2\pi f$), and under $e^{+i\omega t}$ its terms are:

| Term | Adds to $\varepsilon_\infty$ |
| --- | --- |
| {py:obj}`gfem.Drude` | $-\omega_p^2 / (\omega (\omega - i\gamma))$ |
| {py:obj}`gfem.Lorentz` | $\Delta\varepsilon\, \omega_0^2 / (\omega_0^2 - \omega^2 + i\gamma\omega)$ |
| {py:obj}`gfem.Debye` | $\Delta\varepsilon / (1 + i f / f_r)$ |
| {py:obj}`gfem.DjordjevicSarkar` | $\Delta\varepsilon \ln((f_2 + i f) / (f_1 + i f)) / \ln(f_2 / f_1)$ |

Here $\omega_p$, $\omega_0$ and $\gamma$ are $2\pi$ times the term's plasma, resonance and damping
frequencies, and $f_r$, $f_1$ and $f_2$ are its relaxation and corner frequencies.
With every strength positive and every damping at least 0, each term is
passive alone, so $\mathrm{Im}\,\varepsilon \le 0$ at every real frequency. A Lorentz medium just
below its resonance:

```python
law = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=10e9, damping=1e9)],
)
eps = complex(law(9e9))
print(eps.real, eps.imag)  # at 9 GHz: Im eps < 0, a loss
print(law(0.0).real)  # the static permittivity, eps_inf + delta_eps
```

```text
6.298642533936651 -2.0361990950226243
3.0
```

Most optics papers write their fits under $e^{-i\omega t}$. There the same law is
the complex conjugate: keep the parameter values, and read their $-i\gamma$ as $+i\gamma$.
`law(0)` is the {term}`static permittivity`, which the electrostatic problem
reads. A mode's energy weighs a law region with its
{term}`energy permittivity` instead of $\mathrm{Re}\,\varepsilon$ (see below). How to write a law:
{doc}`recipes/dispersive-materials`.

A measured table gives the refractive index $n$ and the
{term}`extinction coefficient` $k \ge 0$, which does not depend on the time
convention. Under $e^{+i\omega t}$ the index is $n - ik$, so
$\varepsilon = (n - ik)^2$ and $\mathrm{Im}\,\varepsilon = -2nk$.
{py:obj}`gfem.fit_permittivity` takes $n$ and $k$ as tabulated; a
permittivity table written under $e^{-i\omega t}$ as
$\varepsilon_1 + i\varepsilon_2$ goes in as its complex conjugate
({doc}`recipes/fit-measured-data`).

## Eigenmode fields

A mode has no source to set its amplitude, so gfem scales every mode field to
store `MODE_ENERGY`, 1 J, of electric energy: $\tfrac{1}{2} \mathrm{Re} \int \mathbf{D}^* \cdot \mathbf{E}\, dV$, plus
$C|V|^2/2$ in each lumped port with a capacitance $C$, plus $\tfrac{1}{2} \int C_s |E_t|^2\, dS$ on
each {py:obj}`gfem.Impedance` sheet with a capacitance $C_s$ per square. For a
lossless mode this is its total energy. In a region with a permittivity
law, the volume term weighs $|E|^2$ with the energy permittivity at the mode's
real frequency, so it stays the total energy of a lossless mode. Elsewhere,
the time-averaged electric energy `gfem.electric_energy` measures is half of
the volume term. Participation
ratios are shares of the same energy. Over all the volumes, the capacitive
ports and the capacitive sheets they add up to 1, and over the volumes alone
when the model has neither, as here:

```python
energy = gfem.electric_energy(modes.e_field(0), model=resonator)  # J, per volume
print(2 * sum(energy.values()))  # J
print(sum(modes.bulk_participation.values())[0])
```

```text
1.0
1.0000000000000002
```

The inductive participations of a lumped port with an inductance $L$ and of an
inductive sheet are shares of the same energy too. For a lossless mode, a
sheet's `sheet_participation` is $-2\, d\ln\omega / d\ln L_s$ on any mesh. A port's
`participation` is $-2\, d\ln\omega / d\ln L$ when the field is uniform across the
port; otherwise it is read from the port's mean voltage and differs slightly.

## Periodic cells

Across a periodic pair of translation $\mathbf{a}$, a Bloch wave vector $\mathbf{k}$ sets
`BLOCH_PHASE`: `E(r + a) = exp(-i k·a) E(r)`, so the phase of a Bloch wave
travels along $+\mathbf{k}$. {doc}`recipes/periodic-cell` shows it on a cell, and
"Floquet ports" below gives the conventions of a cell lit by a plane wave.

Wave vectors are Cartesian, in rad/m. The named points of a
{py:func}`gfem.wave_vector_path` are in reduced coordinates of
`Model.reciprocal_lattice`, $\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi\delta_{ij}$.
Under `BLOCH_PHASE` the band group velocity $\mathbf{v}_g =
\nabla_{\mathbf{k}}\omega$ points the way energy travels, with no sign flip.
Bands are numbered from 0, the lowest.

## Floquet ports

A {py:obj}`gfem.FloquetPort` is one diffraction order in one polarization of
a plane wave on the open face of a periodic cell
({doc}`recipes/floquet-ports`). The cell here is one period of the
{doc}`metagrating tutorial <tutorials/metagrating>`'s grating with its
design layers left as air: a flat interface between glass ($n = 1.444$) below
and air above, 2.023 µm along x and 0.08 µm along y.

**The lattice frame.** The two pairs that rim a face, in `boundaries` order,
have translations $\mathbf{a}_1$ and $\mathbf{a}_2$. The face's frame is $\hat{\mathbf{e}}_1 = \mathbf{a}_1 / |\mathbf{a}_1|$, $\hat{\mathbf{e}}_3$ along
$\mathbf{a}_1 \times \mathbf{a}_2$ and $\hat{\mathbf{e}}_2 = \hat{\mathbf{e}}_3 \times \hat{\mathbf{e}}_1$; here $\hat{\mathbf{e}}_1 = +x$, $\hat{\mathbf{e}}_2 = +y$, $\hat{\mathbf{e}}_3 = +z$ on both faces.
Order $(m, n)$ has the in-plane wave vector $\boldsymbol{\kappa} = \mathbf{k}_t + m \mathbf{b}_1 + n \mathbf{b}_2$, with
$\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi \delta_{ij}$ and $\mathbf{k}_t$ the problem's `wave_vector` (zero at normal
incidence). An order propagates in the medium behind the face when
$|\boldsymbol{\kappa}| < k_0 n$; at 1.55 µm the orders −1, 0 and +1 do, at 3.1 µm only the
specular one:

```python
c = 299_792_458.0  # m/s
url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/metagrating.msh"
urllib.request.urlretrieve(url, "metagrating.msh")
cell_mesh = gfem.Mesh.from_gmsh("metagrating.msh", length_unit="um")
n_glass = 1.444
air = gfem.Material()
glass = gfem.Material(eps_r=n_glass**2)
flat = {"glass": glass, "lower": air, "upper": air, "air": air}
pairs = [gfem.Periodic(("x_min", "x_max")), gfem.Periodic(("y_min", "y_max"))]
cell = gfem.Model(cell_mesh, materials=flat, boundaries=pairs)
print(np.asarray(cell.periodic_translations))  # a1, a2, m
print(cell.propagating_orders("top", c / 1.55e-6))
print(cell.propagating_orders("top", c / 3.1e-6))
```

```text
[[2.0233813e-06 0.0000000e+00 0.0000000e+00]
 [0.0000000e+00 8.0000000e-08 0.0000000e+00]]
((0, 0), (-1, 0), (1, 0))
((0, 0),)
```

The TE reference of an order is $\hat{\mathbf{e}}_3 \times \hat{\boldsymbol{\kappa}}$ ($\mathbf{E}$ normal to its plane of
incidence, s) and the TM reference is $\hat{\boldsymbol{\kappa}}$ ($\mathbf{E}$ in it, p), both tangential to
the face. At $\boldsymbol{\kappa} = 0$, $\hat{\boldsymbol{\kappa}}$ is $\hat{\mathbf{e}}_1$: at normal incidence TE is $\mathbf{E}$ along $\hat{\mathbf{e}}_2$ ($+y$ here)
and TM is $\mathbf{E}$ along $\hat{\mathbf{e}}_1$ ($+x$).

**What S is.** A Floquet entry compares the tangential E of two orders at
their face planes, scaled to power: $S_{p\leftarrow q} = (c_p / c_q) \sqrt{Z_q / Z_p}$, with
$c$ the order's tangential-E coefficient and $Z$ its wave impedance in the medium
behind its own face, $\eta / \cos\theta$ for TE and $\eta \cos\theta$ for TM. $|S|^2$ is therefore
the {term}`diffraction efficiency`, and for a lossless cell the efficiencies
of every propagating order on both faces, in both polarizations, add up to 1.
The reflection of a flat interface is the tangential-E coefficient
$(n_1 - n_2) / (n_1 + n_2)$ in both polarizations at normal incidence. The
Fresnel $r_p$ of many optics texts has the opposite sign. With the ports'
`offset` moving their reference planes to the interface, at 3.1 µm:

```python
f = c / 3.1e-6  # Hz: only the specular order propagates
ports = [
    gfem.FloquetPort(f"glass {pol}", "bottom", polarization=pol, offset=0.5e-6)
    for pol in ("TE", "TM")
] + [
    gfem.FloquetPort(f"air {pol}", "top", polarization=pol, offset=1.2e-6)
    for pol in ("TE", "TM")
]
ports.append(gfem.FloquetPort("air -1", "top", order=(-1, 0), polarization="TE"))
interface = cell.replace(ports=ports)
direct = gfem.SolverOptions(linear_solver="direct")
lit = gfem.Driven(f, excite=["glass TE", "glass TM"])
floquet = gfem.solve(interface, lit, options=direct)
s = floquet.s
print(complex(s["glass TE", "glass TE"][0]), complex(s["glass TM", "glass TM"][0]))
print((n_glass - 1) / (n_glass + 1))  # (n1 - n2) / (n1 + n2), from the glass
print(abs(s["glass TE", "glass TE"][0]) ** 2 + abs(s["air TE", "glass TE"][0]) ** 2)
```

```text
(0.18167034107937008+2.0236696358788642e-07j) (0.1816703863297391+2.0964626335623908e-07j)
0.18166939443535188
0.9999999997457327
```

**Jones matrices.** The Jones matrix of a cell between two orders is a
$2\times2$ block of S in the Floquet references, ordered (TM, TE): rows the
outgoing order's polarizations, columns the incident order's,
$J_{ij} = S_{(p,i)\leftarrow(q,j)}$. The references are the same on both
faces and in both directions, so nothing flips on reflection, and at normal
incidence TM is $\hat{\mathbf{e}}_1$ and TE is $\hat{\mathbf{e}}_2$ on both faces:
the Jones matrix is then the lab Cartesian one, and a wave plate with its axis
at $\psi$ from $\hat{\mathbf{e}}_1$ reads at $\psi$ from either side. The entries are power-wave
ratios like S, so $|J_{ij}|^2$ is an efficiency; the ratio of the fields is
$J_{ij}\sqrt{Z_{\mathrm{out},i}/Z_{\mathrm{in},j}}$, and the two coincide when
the two orders' impedances are equal (same medium, same polarization, same
angle). Both polarizations of each order need a
port, and both columns of the incident order must be excited. The transmission
of the interface above, glass to air, is diagonal, with
$|J|^2 = 1 - r^2$ for the reflection $r$ of the
previous block:

```python
jones = gfem.diffraction.jones(floquet, "top", "bottom", model=interface)
print(complex(jones.values[0, 0, 0]), complex(jones.values[0, 1, 1]))  # TM, TE
print(np.sqrt(1 - ((n_glass - 1) / (n_glass + 1)) ** 2))
```

```text
(0.9833594819245479+4.923795437360556e-07j) (0.9833594901751661+4.858235650433676e-07j)
0.9833596651914763
```

**Rows that do not propagate.** The (−1, 0) order is evanescent at 3.1 µm. Its
row of S reads exactly 0 at every frequency where it does not propagate, and
its reference impedance is NaN there. `diffraction_orders` gives its $\boldsymbol{\kappa}$, here
$-\mathbf{b}_1$ along $-x$, and whether it propagates. The reference impedances of the
propagating rows are $\eta / n$ of their medium (TE and TM are equal at normal
incidence):

```python
minus_one = floquet.diffraction_orders["air -1"]
print(np.asarray(minus_one.wave_vector[0]), bool(minus_one.propagating[0]))
print(complex(s["air -1", "glass TE"][0]))
at = {name: i for i, name in enumerate(s.ports)}
print(np.asarray(s.z_ref).real[0, [at["glass TE"], at["air TE"], at["air -1"]]])  # ohm
```

```text
[-3105289.79977531        0.                0.        ] False
0j
[260.89356884 376.73031341          nan]
```

**Phase.** Under $e^{+i\omega t}$ a plane wave crossing an empty cell of thickness $L$
comes out as $e^{-i\beta L}$, a delay. With the glass made air and no offsets, the
1.7 µm cell:

```python
empty = cell.with_material("glass", air).replace(
    ports=[
        gfem.FloquetPort("down", "top", polarization="TE"),
        gfem.FloquetPort("out", "bottom", polarization="TE"),
    ]
)
s_empty = gfem.solve(empty, gfem.Driven(f, excite="down"), options=direct).s
print(complex(s_empty["out", "down"][0]))
print(np.exp(-1j * 2 * np.pi * f / c * 1.7e-6))  # exp(-i beta L)
```

```text
(-0.9541393055579418+0.2993629662200534j)
(-0.954139256400049+0.2993631229733574j)
```

The in-plane phase of every order is referred to the model origin.
Translating the whole model in-plane by $\mathbf{r}_0$ multiplies $S_{p\leftarrow q}$ by
$e^{-i (\boldsymbol{\kappa}_p - \boldsymbol{\kappa}_q) \cdot \mathbf{r}_0}$: entries between two specular orders do not change,
entries between different orders do.

**Bloch reciprocity.** A reversed wave has the opposite in-plane wave vector,
so order $(m, n)$ maps to $(-m, -n)$, and the TE and TM references, odd in $\hat{\boldsymbol{\kappa}}$,
flip sign with it. With $\sigma = -1$ for an order with $\boldsymbol{\kappa} \ne 0$ and $+1$ for $\boldsymbol{\kappa} = 0$,
{term}`Bloch reciprocity` reads $S_{p\leftarrow q}(\mathbf{k}) = \sigma_p \sigma_q S_{-q\leftarrow -p}(-\mathbf{k})$, where
$-p$ is port $p$'s face and polarization with the opposite order. The plain
transpose never holds as it stands; the
{doc}`metagrating tutorial <tutorials/metagrating>` checks the signed form
on its design.

**Circular polarization.** No port is circular: combine columns. With the TE
and TM ports of the same order on the top face (incidence along $-\hat{\mathbf{e}}_3$),
$(S[:, \mathrm{TM}] + i\, S[:, \mathrm{TE}]) / \sqrt{2}$ is the response to right-handed circular
incidence in the IEEE sense (the field turns counterclockwise seen with the
wave coming towards you). From the bottom face, incidence along $+\hat{\mathbf{e}}_3$, the same
combination is left-handed, and $(S[:, \mathrm{TM}] - i\, S[:, \mathrm{TE}]) / \sqrt{2}$ is right-handed.
`Jones.circular` does this for every beam:
$J_\mathrm{circ} = U_{h_\mathrm{out}}^\dagger J\, U_{h_\mathrm{in}}$ with

$$U_h = \frac{1}{\sqrt2}\begin{pmatrix}1 & 1\\ -ih & ih\end{pmatrix}$$

and $h = \pm1$ the direction of the beam along $\hat{\mathbf{e}}_3$: its columns,
right-handed (RCP) then left-handed (LCP), are the polarizations
$(\hat{\mathbf{E}}_\mathrm{TM} \mp ih\,\hat{\mathbf{E}}_\mathrm{TE})/\sqrt2$, handed about the beam's own direction.
Rows and columns of $J_\mathrm{circ}$ are (RCP, LCP). A mirror therefore swaps the
handedness, and transmission through an interface keeps it. On a cell closed by a PEC on
`"bottom"`, with the port on `"top"` moved down onto the mirror by its `offset`, the response
to right-handed incidence is all left-handed, and the Jones matrix is $-P$ with $P$ the
swap of the two circular polarizations, whose Mueller matrix is
$\mathrm{diag}(1, 1, -1, -1)$:

```python
mirror = cell.with_material("glass", air).replace(
    ports=[
        gfem.FloquetPort(f"mirror {pol}", "top", polarization=pol, offset=1.7e-6)
        for pol in ("TM", "TE")
    ],
    boundaries=[*pairs, gfem.PEC("bottom")],
)
closed = gfem.solve(
    mirror, gfem.Driven(f, excite=["mirror TM", "mirror TE"]), options=direct
)
sm = closed.s
column = [
    (sm[row, "mirror TM"] + 1j * sm[row, "mirror TE"])[0] / np.sqrt(2)
    for row in ("mirror TM", "mirror TE")
]
print(
    np.round(((column[0] + 1j * column[1]) / np.sqrt(2)).real, 4) + 0.0
)  # right-handed out
print(
    np.round(((column[0] - 1j * column[1]) / np.sqrt(2)).real, 4) + 0.0
)  # left-handed out
mirror_jones = gfem.diffraction.jones(closed, "top", "top", model=mirror)
print(np.round(np.asarray(mirror_jones.circular[0]).real, 4) + 0.0)  # -P
print(np.round(np.asarray(mirror_jones.mueller[0]), 4) + 0.0)  # diag(1, 1, -1, -1)
```

```text
0.0
-1.0
[[ 0. -1.]
 [-1.  0.]]
[[ 1.  0.  0.  0.]
 [ 0.  1.  0.  0.]
 [ 0.  0. -1.  0.]
 [ 0.  0.  0. -1.]]
```

The empty cell is the other extreme: it keeps the handedness and the Stokes
vector, so $|J_\mathrm{circ}|$ is the identity and $M = I$ whatever the
phase $e^{-i\beta L}$:

```python
free = cell.with_material("glass", air).replace(
    ports=[
        gfem.FloquetPort(f"{side} {pol}", side, polarization=pol)
        for side in ("top", "bottom")
        for pol in ("TM", "TE")
    ]
)
through = gfem.solve(
    free, gfem.Driven(f, excite=["bottom TM", "bottom TE"]), options=direct
)
free_jones = gfem.diffraction.jones(through, "top", "bottom", model=free)
print(np.round(np.abs(np.asarray(free_jones.circular[0])), 4))  # |.| of e^{-i beta L} I
print(np.round(np.asarray(free_jones.mueller[0]), 4) + 0.0)  # I
```

```text
[[1. 0.]
 [0. 1.]]
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
```

**Stokes and Mueller.** The Stokes vector $(S_0, S_1, S_2, S_3)$ of a beam is
taken in the beam's own right-handed frame $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})$,
with components $(b_\theta, b_\phi) = (h\, b_\mathrm{TM}, b_\mathrm{TE})$ for the
direction sign $h$ of the beam: $S = A\,(b \otimes b^*)$ with
$$A = \begin{pmatrix}1&0&0&1\\1&0&0&-1\\0&1&1&0\\0&-i&i&0\end{pmatrix}.$$
This is the $e^{+i\omega t}$ form: $S_3 = |b_\mathrm{R}|^2 - |b_\mathrm{L}|^2$
is positive for the right-handed light of the labels above (the $0, i, -i, 0$ row of
textbooks written for $e^{-i\omega t}$ gives the opposite sign). The Mueller matrix
is $M = A\,(J_w \otimes J_w^*)\,A^{-1}$ with
$J_w = \mathrm{diag}(h_\mathrm{out}, 1)\, J\, \mathrm{diag}(h_\mathrm{in}, 1)$:
real, $4\times4$, mapping the incident Stokes vector to the outgoing one,
and $M_{00} = \tfrac12 \sum_{ij} |J_{ij}|^2$ is the efficiency for unpolarized
light. Because each beam has its own frame, $M$ is always a Mueller-Jones matrix: a
mirror is $\mathrm{diag}(1, 1, -1, -1)$ and free transmission is the identity.
The cost is that for a downward beam $S_2$ is measured from
$\hat{\boldsymbol{\theta}} = -\hat{\mathbf{E}}_\mathrm{TM}$, so a plate with its axis at $\psi$
lit from the top shows its linear block at $-\psi$; lit from the bottom, where
the Floquet references, $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})$ and the usual
light-along-$+z$ frame coincide, it reads at $+\psi$. A
quarter-wave plate with its slow axis along $\hat{\mathbf{e}}_1$ turns light linearly
polarized at 45° into right-handed light when lit from the top, and into left-handed
light from the bottom.

**Normal incidence in a conical scan.** At $\boldsymbol{\kappa} = 0$ the references
are $\hat{\mathbf{e}}_1$ and $\hat{\mathbf{e}}_2$ whatever the azimuth $\varphi$ of the scan, while for
$\theta > 0$ they are those rotated by $\varphi$: a $\theta$ scan at $\varphi \ne 0$ sees
the linear entries of $J$ jump by that rotation at $\theta = 0$, and the linear block
of $M$ by twice the angle. Efficiencies do not change, and the circular entries
change only by phases $e^{\pm i\varphi}$. The jump is documented, not corrected: no
basis is continuous at $\boldsymbol{\kappa} = 0$.

**Emission direction.** An emitter in equilibrium radiates into a direction as it
absorbs the time-reversed wave (Kirchhoff's law). A wave with the in-plane wave
vector at azimuth $\varphi$ travels towards $(\cos\varphi, \sin\varphi)$, and the reversed
one leaves towards $-\hat{\boldsymbol{\kappa}}$, so the absorptance of the incidence
$(\theta, \varphi)$ is the emissivity towards $(\theta, \varphi + \pi)$, and
`gfem.diffraction.emissivity` labels its rows so.

## Aperture fields

`gfem.diffraction.aperture_field` samples a metasurface once per cell. The
cells of an axis of $n_1$ cells sit at $x_j = (j - \tfrac{n_1-1}{2})\,a_1$, measured
from the aperture centre along $\hat{\mathbf{e}}_1$ with the pitch $a_1$ of the
lattice, and likewise along $\hat{\mathbf{e}}_2$; with an even count the centre falls
between two samples. The values are power waves relative to the incident plane
wave, as the entries of S are, so $|E|^2$ of a wave that leaves normally is its
intensity relative to the incident one. The two components are those along
$\hat{\mathbf{e}}_1$ (TM) and $\hat{\mathbf{e}}_2$ (TE), which are the Floquet
references at normal incidence. `gfem.diffraction.propagate` multiplies each plane
wave of the field by $e^{-ik_zd}$ for a distance $d$ into the half-space behind the
surface, with $k_z = \sqrt{k^2 - q^2}$ where the wave propagates and
$k_z = -i\sqrt{q^2 - k^2}$ where it is evanescent, so that evanescent waves decay
under $e^{+i\omega t}$.

## Temperature and thermo-optic phase

A thermal solve's temperatures are absolute, in kelvin, and its rise is the
temperature minus the {term}`source-free temperature`: the temperature with
every heat source off, which the boundaries alone set. The device here is the
heater section of {doc}`recipes/heater-temperature`, a TiN strip over a
silicon waveguide, read as a {term}`slab cross-section`.

**Kelvin.** A temperature at or below 0 K raises, whatever its origin:

```python
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
field is strongest. The {py:obj}`gfem.conventions` module, "Electro-optic
shift", gives the formulas, and {doc}`recipes/electro-optic-phase` the
whole calculation.

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
$\tfrac{\omega}{2}\,\mathrm{Im}(-\varepsilon)|E|^2 \ge 0$ ("Sign of loss").
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
(the table below).

**Network parameters.** `network_parameters(aperture, series_resistance=,
reference_impedance=)` puts the resistances $R = \mathrm{diag}(R_s)$ in the
leads and refers S to the reference impedances (50 Ω by default), as
pseudo-waves ("S-parameters and power"):

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
in m/s, the exact derivative of the discrete band. A section with a mirror
symmetry along $z$ has $\omega(-q) = \omega(q)$ and the opposite group velocity.

**Clamped components.** `clamped` fixes all three components on a line by its
name, or only the components named by a string of `"xyz"` in the lab axes: a
line of constant $x$ held with `"x"` keeps $u_x = 0$ and slides in $y$ and $z$.
Every other line is traction-free. A section free at $q = 0$ has rigid-body
motions at $\omega = 0$ that are not returned.

## Group index and dispersion

A port mode's group index and dispersion are the frequency derivatives of
$\gamma / i$, with $\omega = 2\pi f$ and $\lambda = c/f$ the vacuum
wavelength:

$$
n_g = \frac{c}{i}\frac{\mathrm{d}\gamma}{\mathrm{d}\omega}, \qquad
\beta_k = \frac{1}{i}\frac{\mathrm{d}^k\gamma}{\mathrm{d}\omega^k}, \qquad
D = -\frac{2\pi c}{\lambda^2}\,\beta_2 .
$$

For a lossless mode they are real and the same under either time
dependence, so a published value compares directly. With loss they are
complex and written $n' - in''$ like `n_eff`; a code that uses
$e^{-i\omega t}$ reports their complex conjugates. $\beta_2 > 0$,
equivalently $D < 0$, is normal dispersion; $D > 0$ is anomalous. Papers
quote them in other units: 1 ps/(nm km) is $10^{-6}$ s/m², 1 ps²/km
(1 fs²/mm) is $10^{-27}$ s²/m and 1 ps³/km is $10^{-39}$ s³/m. The
{py:obj}`gfem.conventions` module, "Group index and dispersion", gives the
formulas, and {doc}`recipes/waveguide-dispersion` the calculation.

## Kerr nonlinearity

A material's `nonlinear_index` $n_2$ (m²/W) and `two_photon_absorption`
$\beta_\mathrm{TPA}$ (m/W) make its permittivity depend on the intensity,
$n = n_0 + n_2 I$ and $\alpha_I = \beta_\mathrm{TPA} I$ in the intensity
$I$. They combine in a complex index
$\tilde n_2 = n_2 - i\beta_\mathrm{TPA}/(2k_0)$, with $k_0 = \omega/c$. The
nonlinear coefficient of a port mode $E$ (all three components, normalized
to 1 W; $n^2 = \mathrm{Re}\,\varepsilon_r$ of each element at the frequency
of the row) is

$$
\gamma_\mathrm{NL} = \frac{k_0 c^2 \varepsilon_0^2}{12} \int_S n^2\,\tilde n_2
\left[2|E|^4 + |E\cdot E|^2\right] \mathrm{d}A,
$$

with $E\cdot E$ the product without conjugation. $\gamma_\mathrm{NL}$ is
reported in the convention of pulse-propagation tools, $e^{-i\omega t}$: the
envelope $A$ of power $P = |A|^2$ obeys $\mathrm{d}A/\mathrm{d}z =
i\gamma_\mathrm{NL}|A|^2 A$, the propagation constant moves by
$\delta\beta = \mathrm{Re}\,\gamma_\mathrm{NL}\,P$ and the field attenuation
by $\delta\alpha = \mathrm{Im}\,\gamma_\mathrm{NL}\,P \ge 0$, the loss of
two-photon absorption: $\mathrm{d}P/\mathrm{d}z = -2\,\mathrm{Im}\,
\gamma_\mathrm{NL}\,P^2$. It is the complex conjugate of the same integral in
gfem's own $e^{+i\omega t}$ phasors, so the sign of its imaginary part does
not follow `n_eff`'s $n' - in''$.

Between several waves $a, b, c, d$ (rows of port mode results) the
tensor $\gamma_{abcd}$ is the coefficient of $A_b A_c \overline{A_d}$ in the
equation of $A_a$, summed over every index tuple with no degeneracy factor,
so the factor 2 of cross-phase modulation and of four-wave mixing comes out
of the sum; $\gamma_{aaaa} = \gamma_\mathrm{NL}$ of the same row.

Units: $\gamma_\mathrm{NL}$ is in 1/(W m) on a 3D section. On a line port of
a 2D model the field is for 1 W/m and the integral runs along the line, so it
is in 1/W. Only the $E_z$ polarization of a line port has it. The
isotropic scalar $n_2$ is real and does not disperse over the grid; the
formula holds while the attenuation is small against the phase constant.
The {py:obj}`gfem.conventions` module, "Kerr nonlinearity", gives the same
formulas, and {doc}`recipes/waveguide-nonlinearity` the calculation.

## Second-order nonlinearity

A material's `second_order_nonlinear_tensor` is the contracted tensor
$d_{il} = d_{ijk}$ (m/V), with $d = \chi^{(2)}/2$ (the convention of data
sheets, Boyd and Roberts: half the susceptibility). Rows are the polarization
$x, y, z$, columns the field pair $xx, yy, zz, yz, xz, xy$ with no factor 2 on
the shear ones, in the frame of the material's `axes`. For a field $F$,
$\sum_{jk} d_{ijk}F_jF_k = \sum_l d_{il}V_l$ with $V = (F_x^2, F_y^2, F_z^2,
2F_yF_z, 2F_xF_z, 2F_xF_y)$. A domain of reversed polarization carries $-d$.
The tensor is real, constant over the grid, and read by no linear solve.

The coupling between the port modes $a, b, c$ of three waves with
$\omega_a = \omega_b + \omega_c$ (rows of results, each normalized to 1 W) is
$\kappa_{abc} = \overline{T_{abc}}$ with

$$
T_{abc} = \frac{\omega_a\varepsilon_0}{4} \int_S \sum_{ijk} d_{ijk}\,
\overline{E_{a,i}}\,E_{b,j}\,E_{c,k}\,\mathrm{d}A,
$$

all three components of each mode. It is reported in the convention of
coupled-mode tools, $e^{-i\omega t}$, the complex conjugate of the integral in
gfem's own phasors: the amplitude $A$ of a wave, with $|A|^2$ its power, obeys
$\mathrm{d}A_a/\mathrm{d}z \ni i\sum_{bc}\kappa_{abc}A_bA_c\,
e^{-i\Delta\beta_{abc}z}$, $\Delta\beta_{abc} = \beta_a - \beta_b - \beta_c$,
summed over every ordered pair with no degeneracy factor, so the factor 2 of
sum-frequency generation against second-harmonic generation comes out of the
sum. $\kappa_{abc}$ is symmetric in $b, c$ and has units
1/(W$^{1/2}$ m); on an Ez line port of a 2D model (1 W/m, integral along the
line) W$^{-1/2}$ m$^{-1/2}$.

For the second harmonic of a pump at $\omega$, $\kappa = \kappa_{hpp}$ with
$h$ the mode at $2\omega$, and the undepleted harmonic power is

$$
P_{2\omega}(L) = \eta_\mathrm{norm}\,P_\omega^2L^2\,
\mathrm{sinc}^2\!\left(\frac{\Delta\beta L}{2}\right),\qquad
\eta_\mathrm{norm} = |\kappa|^2,
$$

$\Delta\beta = \beta_{2\omega} - 2\beta_\omega$ in 1/m and
$\eta_\mathrm{norm}$ in 1/(W m²) (1/(W m) on a line port;
1 W$^{-1}$m$^{-2}$ is $10^{-2}$ %/(W cm²)). Quasi-phase matching with a
square-wave poling of period $\Lambda$, duty cycle $D$ and order $m$ multiplies
$\kappa$ by $G_m = 2\sin(\pi mD)/(\pi m)$ and replaces $\Delta\beta$ by
$|\Delta\beta| - 2\pi m/\Lambda$, matched at $\Lambda = 2\pi m/|\Delta\beta|$.
With the loss of the modes, $\alpha = \mathrm{Re}\,\gamma$ and
$\Delta\alpha = \alpha_{2\omega} - 2\alpha_\omega$,

$$
\frac{P_{2\omega}}{P_\omega} = |G|^2|\kappa|^2P_\omega\,
e^{-(\alpha_{2\omega} + 2\alpha_\omega)L}\,
\frac{\sinh^2(\Delta\alpha L/2) + \sin^2(\delta L/2)}
{(\Delta\alpha/2)^2 + (\delta/2)^2},
$$

with $\delta$ the residual mismatch. For difference-frequency generation and
parametric amplification of a pump $a$ with a signal $b$ and an idler $c$, the
amplitude of $b$ obeys $\mathrm{d}A_b/\mathrm{d}z \ni i\,(2\omega_b/\omega_a)\,
\overline{\kappa_{abc}}\,A_a\overline{A_c}\,e^{i\Delta\beta z}$ (and $b
\leftrightarrow c$): the rule conserves the photon flux when $d$ is symmetric
in its three indices (Kleinman symmetry), which holds when one $d$ serves
every process. The sum-frequency power of two waves in the undepleted
regime is $4|\kappa_{312}|^2P_1P_2L^2\,\mathrm{sinc}^2(\Delta\beta L/2)$.
The formulas hold while $\alpha \ll \beta$ and for a pump that is not
depleted. The {py:obj}`gfem.conventions` module, "Second-order nonlinearity",
gives the same formulas, and {doc}`recipes/second-harmonic-generation` the
calculation.

## Thin metals

A {py:obj}`gfem.Conductor` with a thickness describes one face of a film. On
a surface inside the mesh a film much thinner than its skin depth is
therefore a sheet of admittance `INTERIOR_SHEET_ADMITTANCE` times $\sigma t$, half
the film's conductance: give `thickness=2 * t` for the whole film. On the
outer surface of the mesh the thickness already counts
`EXTERIOR_THICKNESS_FACTOR` times, and the film is $\sigma t$. The module's "Surface
conductors" section has the formula.

## 2D models

On a {term}`2D model` the fields do not vary along z, and every extensive
quantity is per metre along z. The rows that change:

| Quantity | On a 3D model | On a 2D model |
| --- | --- | --- |
| power fed into an excited port (`INCIDENT_POWER`) | 1 W | 1 W/m |
| kept driven field | $\mathbf{E}$ in V/m | $E_z$ in V/m (Ez) or $H_z$ in A/m (Hz), for 1 W/m |
| S-parameters | dimensionless | dimensionless, and the same |
| `Field.at` | (m, 3) points, (m, 3) values | (m, 2) points, (m,) values |
| kept piezoelectric fields | | $u_x, u_y, u_z$ in m and $\varphi$ in V, per volt on the terminal |

S is a ratio of powers, so it does not depend on the height of the structure:
a {term}`slab <Slab model>` of thickness $h$ between plates and the 2D model of
its plane give the same S. The slab's 1 W is $1/h$ W/m, so the fields relate by
$E^{\mathrm{2D}} = \sqrt{h}\,E^{\mathrm{slab}}$. The time dependence, the phasors and
the sign of loss are those above. The module's "2D models" section gives the
equation of each polarization and which line is a conductor in each.

## Incident fields

A {py:obj}`gfem.PlaneWave` or {py:obj}`gfem.GaussianBeam` is given by its
propagation direction in spherical angles from $+z$, in rad:
$\hat{\mathbf{k}} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$. A
wave with $\theta < \pi/2$ travels up, one with $\theta > \pi/2$ down, so a
wave arriving 30° off the normal from above is `theta=np.pi - np.radians(30)`.

| Polarization | Field direction | Also called |
| --- | --- | --- |
| `"TM"` | $\hat{\boldsymbol{\theta}} = (\cos\theta\cos\phi, \cos\theta\sin\phi, -\sin\theta)$ | p |
| `"TE"` | $\hat{\boldsymbol{\phi}} = (-\sin\phi, \cos\phi, 0)$ | s |
| `"RCP"` | $(\hat{\boldsymbol{\theta}} - i\hat{\boldsymbol{\phi}})/\sqrt{2}$ | right-handed (IEEE) |
| `"LCP"` | $(\hat{\boldsymbol{\theta}} + i\hat{\boldsymbol{\phi}})/\sqrt{2}$ | left-handed (IEEE) |
| $(a_\mathrm{TM}, a_\mathrm{TE})$ | $(a_\mathrm{TM}\hat{\boldsymbol{\theta}} + a_\mathrm{TE}\hat{\boldsymbol{\phi}})/\lVert a \rVert$ | Jones pair |

$(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}}, \hat{\mathbf{k}})$ is
right-handed, and along $\pm z$ the azimuth $\phi$ still fixes the plane of
incidence. TE is the TE reference of a Floquet port in every direction; TM
points along the Floquet TM reference for a wave going up and against it for
a wave going down. Under `TIME_DEPENDENCE`, the real field of RCP turns from
$\hat{\boldsymbol{\theta}}$ to $\hat{\boldsymbol{\phi}}$ with the thumb along
$\hat{\mathbf{k}}$.
So for a wave going down the Floquet circular unit vector
$(\hat{\mathbf{E}}_\mathrm{TM} + i\hat{\mathbf{E}}_\mathrm{TE})/\sqrt2$ of the RCP column is
$-(\hat{\boldsymbol\theta} - i\hat{\boldsymbol\phi})/\sqrt2$: the same right-handed
wave as `"RCP"`, with the opposite phase.

A plane wave's `amplitude` $E_0$ is the peak field in V/m (`PHASOR`) in the
layer it comes from; in a homogeneous background its field is
$E_0\,\hat{\mathbf{e}}\,e^{-i k \hat{\mathbf{k}}\cdot\mathbf{r}}$, with the
phase referred to the model origin. A Gaussian beam carries `INCIDENT_POWER`
through any plane normal to its axis, like an excited port, with a real
positive field at its focus. The kept field of a source column is the
{term}`total field <Total field>`, the incident plus the scattered field; inside
a perfectly matched layer, where the incident field is zero, it is the
scattered field. `DrivenResult.e_field(..., part="scattered")` and
`part="incident"` give the two parts on the same space, adding exactly to the
kept field. Where the incident field is dropped or must satisfy a wall it must
stay at most $10^{-3}$ of its peak: where a scatterer meets the layer or an
absorbing face, on a wave port face, and across an exterior PEC or PMC wall.
In a source column every port is terminated, and `DrivenResult.port_waves` holds
the outgoing wave $b$ at each port in √W: $|b|^2$ is the power the source sends
into the port. The module's "Incident fields" section has the formulas.

## Point dipoles

A {py:obj}`gfem.ElectricDipole` of moment $p$ (C·m) at $\mathbf{r}_0$ is the
current $\mathbf{J} = i\omega \mathbf{p}\,\delta(\mathbf{r} - \mathbf{r}_0)$;
a {py:obj}`gfem.MagneticDipole` of moment $m$ (A·m²) the magnetic current
$\mathbf{M} = i\omega\mu_0 \mathbf{m}\,\delta(\mathbf{r} - \mathbf{r}_0)$.
Both moments are peak phasors, the same at every frequency, and an antenna's
current moment $Il$ (A·m) is $p = Il/(i\omega)$. The
{term}`emitted power <Emitted power>`
$P = -\tfrac{\omega}{2}\operatorname{Im}(\mathbf{p}^*\cdot\mathbf{E}(\mathbf{r}_0))$
is positive for a dipole that radiates; the Purcell factor divides it by the
power of the same dipole in the bulk material at $\mathbf{r}_0$. The module's
"Point dipoles" section gives that bulk power and the local density of
states.

## Far fields

{py:obj}`gfem.far_field` gives the {term}`far-field amplitude <Far-field amplitude>`
$\mathbf{F}(\hat{\mathbf{r}}) = \lim_{r\to\infty} r\,e^{+ikr}\mathbf{E}(r\hat{\mathbf{r}})$
in V: under `TIME_DEPENDENCE` an outgoing wave is $e^{-ikr}/r$, and the
phase is referred to the model origin. $k$ and $\eta$ are those of the
{term}`exterior medium <Exterior medium>`, the model's background. The
direction $\hat{\mathbf{r}}$ takes the angles of an incident field's
$\hat{\mathbf{k}}$, and `e_theta`, `e_phi` are the components along
$\hat{\boldsymbol{\theta}}$ and $\hat{\boldsymbol{\phi}}$; `e_right` and
`e_left`, $(F_\theta \pm iF_\phi)/\sqrt{2}$, those along RCP and LCP. The
radiation intensity is $U = |\mathbf{F}|^2/(2\eta)$ in W/sr (`PHASOR`), and
the directivity, gain and realized gain divide $4\pi U$ by the radiated,
accepted and incident power (IEEE 145): a port column is fed
`INCIDENT_POWER`, so its realized gain is $4\pi U$. A plane wave's bistatic
radar cross section is $4\pi|\mathbf{F}|^2/|E_0|^2$, of its scattered
field. The module's "Far fields" section has the formulas.

## Multipoles

`gfem.Multipoles` expands the scattered field of an incident-field column about an
origin (default the model origin) in outgoing vector spherical waves of the
background, $E_s = E_\text{inc}\sum_{lm}[\alpha_{lm}N^{(2)}_{lm} + \beta_{lm}M^{(2)}_{lm}]$,
with Condon-Shortley $Y_{lm}$, Jackson's $X_{lm}$ and $h^{(2)}_l$ (outgoing under
`TIME_DEPENDENCE`). The coefficients are dimensionless and power-normalized: the
partial scattering cross section of the electric (magnetic) multipole of degree $l$
is $k^{-2}\sum_m|\alpha_{lm}|^2$ ($k^{-2}\sum_m|\beta_{lm}|^2$), in m², over the
incident intensity of `cross_sections`. $E_\text{inc}$ is complex: a plane wave's
`amplitude` (phase at the model origin), a beam's focal field $\sqrt{2\eta I_0}$,
$\sqrt{2\eta I}$ for an incident-field function with `intensity=`. It is not
`FarField.incident_amplitude`, the real $|E_0|$ of a plane wave (`None` for a
beam). Coefficients are stored $l$-major, $m$ ascending, flat index
$j = l^2 + l + m - 1$; the default degree is Wiscombe's
$\max(2, \lceil x + 4x^{1/3} + 2\rceil)$ with $x = kR_s$.

For a sphere at the origin under $E_0\hat{x}e^{-ikz}$, with $\bar a_l, \bar b_l$ the
conjugates of Bohren and Huffman's coefficients,
$\alpha_{l,\pm1} = \pm(-i)^l\sqrt{\pi(2l+1)}\,\bar a_l$ and
$\beta_{l,\pm1} = -(-i)^l\sqrt{\pi(2l+1)}\,\bar b_l$, all other $m$ zero.

The exact moments are fixed linear maps of the $l \le 2$ coefficients, in the
model's $x, y, z$ (with $\hat e_{\pm1} = \mp(\hat x \pm i\hat y)/\sqrt2$,
$\hat e_0 = \hat z$ and $T_{2M}$ the orthonormal symmetric traceless rank-2 tensors):

- $p = -\sqrt{6\pi}\,\varepsilon E_\text{inc}k^{-3}\sum_m\alpha_{1m}\hat e_m$ (C·m);
- $m = -i\sqrt{6\pi}\,E_\text{inc}(k^3\eta)^{-1}\sum_m\beta_{1m}\hat e_m$ (A·m²);
- $Q^e = -12\sqrt{5\pi}\,\varepsilon E_\text{inc}k^{-4}\sum_M\alpha_{2M}T_{2M}$ (C·m²);
- $Q^m = -12\sqrt{5\pi}\,i\,E_\text{inc}(k^4\eta)^{-1}\sum_M\beta_{2M}T_{2M}$ (A·m³).

Two factor-3 traps: Evlyukhin's and the anapole review's $Q^m$ is $\tfrac13$ of
this one ($1/(80\pi)$ against $1/(720\pi)$ in $\sigma_{Q^m}$), and a Landau-style
$Q^e = \int\rho\,(r_\alpha r_\beta - \tfrac13 r^2\delta_{\alpha\beta})\,dV$ is $\tfrac13$
of this one. The long-wavelength moments of `CurrentMoments`,
$p_0 = \frac{1}{i\omega}\int J\,dV$ and $T = \frac{1}{10v}\int[(r\cdot J)r - 2r^2J]\,dV$
with $v = \omega/k$, approximate the exact dipole as $p \approx p_0 - ikT$; the
anapole review's $T$ (A·m³) is $vT$. The long-wavelength anapole is $p_0 = ikT$, the
exact one $p = 0$. The first Kerker condition for $\hat k = \hat z$ is
$p_x = m_y/v$.

From a paper in $e^{-i\omega t}$: conjugate every phasor and material; since
$\overline{Y_{lm}} = (-1)^m Y_{l,-m}$ and $\overline{X_{lm}} = (-1)^{m+1}X_{l,-m}$, a
coefficient defined with $Y_{lm}^*$ maps to the conjugate with $m \to -m$ and a
sign. Grahn's $a_E(l,m)$ is
$\alpha_{lm} = -(-1)^m(-i)^l\sqrt{\pi(2l+1)}\,\overline{a_E(l,-m)}$, likewise $\beta$;
Alaee's moments are gfem's with $-1/(i\omega) \to +1/(i\omega)$. The module's
"Multipoles" section has the formulas.

## Optical force

{py:obj}`gfem.optical_force` is the time-averaged force and torque on a body,
from peak phasors: the derivative of $L = \tfrac14\int(\varepsilon|E|^2 -
|B|^2/\mu)\,dV$ of the solved field under a rigid motion of the body. A driven
field is per solved excitation (1 W incident at its port, a plane wave at its
`amplitude`), so the force in N scales with the incident power; an eigenmode
field is at 1 J of stored energy and its force is
$-(1\,\mathrm{J}/\omega)\,\mathrm{d}\omega/\mathrm{d}\xi$. The torque is about
the body's volume centroid unless `origin=` says otherwise, in N m.

## Port waves and beam coupling

A source column's `port_waves` and {py:obj}`gfem.beam_coupling` are
amplitudes in √W, like S for a port fed `INCIDENT_POWER`. On a real
reference (a photonic wave port without a voltage path, on a lossless
guide) $|b|^2$ is the power a source sends into the port and $|c|^2$ the
power a port's field couples into a beam; with a voltage path or on a lossy
guide both are pseudo-waves, like S (`WAVES`). The
{term}`beam coupling <Beam coupling>` $c$ refers to the beam scaled to 1 W
through the side it is read on, with the beam's own phase (a Gaussian
beam's zero focal phase): by reciprocity it is the wave the port receives
when that beam lights the device. The module's "Port waves and beam
coupling" section has the formulas.

## Time responses

`NetworkParameters.step_response` and `impulse_response` turn S into
{term}`time responses <time response>` under the same `TIME_DEPENDENCE`, so
positive time is later: $h(t) = \int S(f)\, e^{+2\pi i f t}\, df$, the plain
inverse transform, with no conjugation. On a {term}`harmonic grid`
$f_k = k\,\Delta f$ the record lasts $1/\Delta f$, negative times included.
The impulse is dimensionless, scaled so that an isolated reflection $\Gamma$
peaks at $\Gamma$, and the step ends at $S(0)$. On the line, the ceramic's
front face reflects $2 \times 10\ \mathrm{mm}/c = 66.7$ ps after the edge
leaves port 1:

```python
grid = gfem.Driven(frequency=0.2e9 * np.arange(1, 101), sweep="adaptive")
impulse = gfem.solve(model, grid).s.impulse_response()
h = np.asarray(impulse["p1", "p1"])
print(f"{impulse.time[np.argmin(h)] * 1e12:.1f} ps")  # the most negative peak
```

```text
65.9 ps
```

Data from a code that writes $e^{-i\omega t}$ are the complex conjugate of
gfem's: conjugate them first, or this reflection lands at $-66$ ps. A
{term}`rational model` writes S as $D + \sum_n R_n/(2\pi i f - p_n)$, poles
$p_n$ in rad/s with $\mathrm{Re}\,p_n < 0$, on one real reference impedance
per port. The module's "Time responses" and "Rational models" sections have
the formulas, and {doc}`recipes/time-responses` the whole calculation.

## Next

- {py:obj}`gfem.conventions`: every convention in full, with the formulas
  for power waves, wave ports and energy densities.
- {doc}`recipes/solve-a-device`: the solves this page reads its numbers from.
- {doc}`recipes/measurements`: power through a surface and energy in a region.
- {doc}`recipes/2d-models`: solving a 2D model in either polarization.
