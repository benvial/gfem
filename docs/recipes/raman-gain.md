# Raman gain between modes

Find how much power a pump mode hands to a Stokes mode through the Raman
effect: the modal gain between two guided waves, from a bulk gain spectrum of
the material. `gfem.raman_gain` reads the {py:obj}`gfem.PortModeResult` of a
wave port and the model's materials, at no cost beyond the modes' own solves.

The guide is the silicon strip of {doc}`waveguide-nonlinearity`, 500 × 220 nm
in oxide, with a Stokes wave at 1.55 µm. Units are SI: the gain is in 1/(W m),
the bulk gain in m/W.

## Material data

A material's `raman_spectrum` is a `gfem.RamanSpectrum`: the bulk gain
$g(\Omega)$ in m/W against the frequency shift, at a reference Stokes
frequency, with a required symmetry class. No spectrum is built in: the
values are yours, and they scatter.

| Material | Spectrum | Value | Source |
| --- | --- | --- | --- |
| Silicon | `lorentzian`, `"cubic"` | line at 15.6 THz, 105 GHz wide; 20 cm/GW at 1.55 µm (4.3 to 76 reported) | Lin, Painter and Agrawal, Opt. Express 15, 16604 (2007); Rey et al., Opt. Express 19, 6 (2011) |
| Silica | `lin_agrawal`, `"isotropic"` | peak at 13.04 THz, $6.2 \times 10^{-14}$ m/W at 1.55 µm | Lin and Agrawal, Opt. Lett. 31, 2086 (2006), scaled from $1.2 \times 10^{-13}$ m/W at 795.5 nm |
| Silicon nitride | none | no bulk spectrum | see below |

- Silicon nitride has no reliable bulk spectrum: its own response is weak, and
  the Raman lasers made on the platform use the silica cladding. A material
  without a `raman_spectrum` contributes nothing.
- Silicon's $g(\Omega)$ is the gain of co-polarized light along
  $\langle 110 \rangle$, and the cubic class needs the crystal frame. The
  `lin_agrawal` size is yours; only its shape is the model's.
- `nonlinear_index` is the $n_2$ you measured. A value measured with pulses
  much longer than the Raman response includes the Raman part, and gfem
  gives all of it the Kerr symmetry. `raman_gain` is separate and subtracts
  nothing. A pulse tool that adds a delayed Raman term splits $n_2$ itself,
  with $f_R n_2 = \frac{1}{\pi k_0}\int_0^\infty g(\Omega)/\Omega\,\mathrm{d}\Omega$.

## The gain of a silicon wire

Silicon's law has no crystal frame: a cubic spectrum and `axes` need a
constant permittivity, so the overlap runs on a copy of the model whose core
is the permittivity at 1.55 µm, with the axes of a wire along $[110]$. The
modes are solved on the dispersive model. Stokes, silicon pump and silica
pump are three frequencies of one grid:

```python
import dataclasses
import urllib.request

import numpy as np
import scipy.constants

import gfem

c = scipy.constants.c  # m/s


def sellmeier(strengths, wavelengths_um):
    """n^2 = 1 + sum of B lambda^2 / (lambda^2 - lambda_i^2), lambda_i in um."""
    terms = [
        gfem.Lorentz(delta_eps=b, resonance_frequency=c / (li * 1e-6), damping=0.0)
        for b, li in zip(strengths, wavelengths_um, strict=True)
    ]
    return gfem.PermittivityLaw(eps_inf=1.0, terms=terms)


# Salzberg and Villa (1957), three-term fit by Tatian (1984); Malitson (1965)
silicon = gfem.Material(
    eps_r=sellmeier(
        (10.6684293, 0.0030434748, 1.54133408), (0.301516485, 1.13475115, 1104.0)
    )
)
oxide = gfem.Material(
    eps_r=sellmeier((0.6961663, 0.4079426, 0.8974794), (0.0684043, 0.1162414, 9.896161))
)

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/strip_section.py"
)
urllib.request.urlretrieve(url, "strip_section.py")
from strip_section import strip_section  # noqa: E402

model = gfem.Model(
    strip_section(500e-9, height=220e-9),
    materials={"core": silicon, "cladding": oxide},
    ports=[gfem.WavePort("wg", "front")],
)
options = gfem.SolverOptions(order=3)
f_s = c / 1.55e-6  # Stokes
f_si, f_silica = f_s + 15.6e12, f_s + 13.04e12  # the two lines' pumps, Hz
waves = np.array([f_s, f_si, f_silica])
te = gfem.solve(model, gfem.PortMode("wg", waves, mode=1), options=options)
tm = gfem.solve(model, gfem.PortMode("wg", waves, mode=2), options=options)
print("n_eff TE0", np.round(te.n_eff.real, 3), "TM0", np.round(tm.n_eff.real, 3))
```

```text
n_eff TE0 [2.447 2.577 2.557] TM0 [1.772 1.943 1.913]
```

The silicon spectrum goes on the oriented core, the silica one on the oxide
(an isotropic spectrum needs no frame, so it sits beside the law), and each
material is read alone:

```python
s = 2**-0.5
# crystal axes in the lab frame (x across, y up, z along): a wire along [110]
axes = np.array([[s, 0, s], [s, 0, -s], [0, 1, 0]])
eps_si = float(silicon.eps_r(f_s).real)
si_line = gfem.RamanSpectrum.lorentzian(2e-10, 15.6e12, 105e9, f_s, symmetry="cubic")
oriented = dataclasses.replace(silicon, eps_r=eps_si, axes=axes, raman_spectrum=si_line)
plain = dataclasses.replace(oriented, axes=None)
silica_line = gfem.RamanSpectrum.lin_agrawal(6.18e-14, f_s)
cladding = dataclasses.replace(oxide, raman_spectrum=silica_line)
silicon_only = model.with_material("core", oriented)
along_100 = model.with_material("core", plain)
silica_only = model.with_material("cladding", cladding)

for name, result in (("TE0", te), ("TM0", tm)):
    g_si = gfem.raman_gain(silicon_only, result).gain[0, 1]
    g_100 = gfem.raman_gain(along_100, result).gain[0, 1]
    g_silica = gfem.raman_gain(silica_only, result).gain[0, 2]
    print(
        f"{name}: Si along [110] {g_si:.1f}, along [100] {g_100:.1f}, "
        f"silica cladding {g_silica:.2f} /(W m)"
    )
```

```text
TE0: Si along [110] 1749.3, along [100] 2.2, silica cladding 0.02 /(W m)
TM0: Si along [110] 380.7, along [100] 13.6, silica cladding 0.28 /(W m)
```

- The TE0 gain is 1.7 /(W mm) at the silicon line: 100 mW in the pump
  amplifies a Stokes wave by $e^{0.17\,L}$ over $L$ mm, before any loss.
- Turning the wire to $[100]$ takes the TE0 gain down by a factor 800: the
  selection rule of the cubic tensors, which a scalar gain does not have.
  It is not zero because the mode of a wire is hybrid, with a longitudinal
  field.
- The TM0 mode is not free of Raman gain along $[110]$: its longitudinal
  field couples to the tensor, and the gain is 22 % of TE0's. Rules that give
  none to TM neglect that field.
- The silica cladding adds a gain three to five orders below the silicon
  line's, over a line 100 times wider.

## Pump loss and the Stokes column

The matrix is square over all waves, so the pump's own loss is in it. By the
Manley-Rowe relation, a photon lost from the pump is one gained by the
Stokes wave:

```python
gain = gfem.raman_gain(silicon_only, te).gain
print(f"Stokes gain {gain[0, 1]:.2f}, pump loss {gain[1, 0]:.2f} /(W m)")
print(f"-(w_b/w_a) g_ab = {-f_si / f_s * gain[0, 1]:.2f}")
```

```text
Stokes gain 1749.27, pump loss -1890.36 /(W m)
-(w_b/w_a) g_ab = -1890.36
```

Entry $[a, b]$ is the gain of wave $a$ per watt of wave $b$: positive when
$a$ lies below $b$, negative above. A pump result and a Stokes result on a
frequency grid give the gain spectrum of the guide as a column of the matrix,
at the cost of the integrals only.

## Limits

- A material with `raman_spectrum` of class `"cubic"` needs a constant
  permittivity, and `axes` give its frame: the dispersive model is solved
  and the overlap is read on the second, as above.
- The gain is that of a continuous wave. The time-domain tensor of a
  multimode pulse equation is not computed.
- The hybrid Hz line port of a 2D model is refused, and so is a layered
  density behind a spectrum.
- No gradient with respect to the spectrum itself; the mode fields and the
  permittivity differentiate.

## Next

- {doc}`waveguide-nonlinearity`: the Kerr coefficient of the same guide.
- {doc}`anisotropic-materials`: the crystal frame.
- {py:obj}`gfem.raman_gain`, {py:obj}`gfem.RamanGain`,
  {py:obj}`gfem.RamanSpectrum`.
