# Phasors and power waves

Why does a phasor, an S entry or a mode field carry the sign, unit and scale it does, and what does each show on a solve?

Every field is a time-harmonic phasor in a linear, time-invariant medium, at
one frequency or on a grid of them. The rules are one entry each on
{doc}`conventions`; this page derives them and checks each on the
parallel-plate line of {doc}`get-started`, whose printed numbers are those of
the repository's own line mesh.

## Units

The mesh file is the one length not in metres: you state its unit once when
you read it, and gfem converts the coordinates. The table of units is on
{doc}`conventions`.

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
{term}`reference impedance` $z$:
$a = \sqrt{\mathrm{Re}\, z} / (2|z|)\,(V + zI)$ and
$b = \sqrt{\mathrm{Re}\, z} / (2|z|)\,(V - zI)$ (Marks and Williams). On a real reference they are the
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

A lossy region's {term}`modal overlap` is about
$8\pi\,(V / \lambda_m^3)\,|\mathrm{Im}\,\varepsilon| / \mathrm{Re}\,\varepsilon$, with $V$ the region's volume
and $\lambda_m$ the wavelength in it
({doc}`recipes/dispersive-materials`).

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

## Validity and limits

These statements hold for linear media. A field-dependent material, such as a
Kerr or second-order one, is on {doc}`nonlinear-and-quantum-optics`. The
mode-energy scale holds for a mode, which has no source to set its amplitude;
a driven field is scaled by the incident power instead.
