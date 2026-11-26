# Plane waves and Floquet orders

How is a plane wave, a diffraction order or a multipole labelled and signed, and what does each convention do to a solved cell?

Every wave here is a phasor with the time dependence $e^{+i\omega t}$, so an
outgoing wave is $e^{-ikr}/r$ and a delay is $e^{-i\beta L}$
({doc}`conventions`). The rules are one entry each there; this page derives
them and checks the Floquet ones on a solved cell. The sign of loss and the S
convention are those of {doc}`phasors-and-power-waves`.

## Floquet ports

A {py:obj}`gfem.FloquetPort` is one diffraction order in one polarization of
a plane wave on the open face of a periodic cell
({doc}`recipes/floquet-ports`). Here the cell is one period of the
{doc}`metagrating tutorial <tutorials/metagrating>`'s grating with its
design layers left as air: a flat interface between glass ($n = 1.444$) below
and air above, 2.023 µm along x and 0.08 µm along y.

**The lattice frame.** The two pairs that rim a face, in `boundaries` order,
have translations $\mathbf{a}_1$ and $\mathbf{a}_2$. The face's frame is $\hat{\mathbf{e}}_1 = \mathbf{a}_1 / |\mathbf{a}_1|$, $\hat{\mathbf{e}}_3$ along
$\mathbf{a}_1 \times \mathbf{a}_2$ and $\hat{\mathbf{e}}_2 = \hat{\mathbf{e}}_3 \times \hat{\mathbf{e}}_1$: $+x$, $+y$, $+z$ on both faces here.
Order $(m, n)$ has the in-plane wave vector $\boldsymbol{\kappa} = \mathbf{k}_t + m \mathbf{b}_1 + n \mathbf{b}_2$, with
$\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi \delta_{ij}$ and $\mathbf{k}_t$ the problem's `wave_vector` (zero at normal
incidence). An order propagates in the medium behind the face when
$|\boldsymbol{\kappa}| < k_0 n$; at 1.55 µm the orders −1, 0 and +1 do, at 3.1 µm only the
specular one:

```python
import urllib.request

import numpy as np

import gfem

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
$J_{ij}\sqrt{Z_{\mathrm{out},i}/Z_{\mathrm{in},j}}$. Both polarizations of
each order need a port, and both columns of the incident order must be
excited. The transmission of the interface above, glass to air, is diagonal,
with $|J|^2 = 1 - r^2$ for the reflection $r$ of the previous block:

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

The in-plane phase of every order is referred to the model origin: translating
the model in-plane by $\mathbf{r}_0$ multiplies $S_{p\leftarrow q}$ by
$e^{-i (\boldsymbol{\kappa}_p - \boldsymbol{\kappa}_q) \cdot \mathbf{r}_0}$, so entries between two specular orders do not change.

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
lit from the top shows its linear block at $-\psi$; lit from the bottom it
reads at $+\psi$. A
quarter-wave plate with its slow axis along $\hat{\mathbf{e}}_1$ turns light linearly
polarized at 45° into right-handed light when lit from the top, and into left-handed
light from the bottom.

**Normal incidence in a conical scan.** At $\boldsymbol{\kappa} = 0$ the references
are $\hat{\mathbf{e}}_1$ and $\hat{\mathbf{e}}_2$ whatever the azimuth $\varphi$ of the scan, while for
$\theta > 0$ they are those rotated by $\varphi$: a $\theta$ scan at $\varphi \ne 0$ sees
the linear entries of $J$ jump by that rotation at $\theta = 0$, and the linear block
of $M$ by twice the angle. Efficiencies do not change, and the circular entries
change only by phases $e^{\pm i\varphi}$. No basis is continuous at $\boldsymbol{\kappa} = 0$, so the jump is not corrected.

**Emission direction.** An emitter in equilibrium radiates into a direction as it
absorbs the time-reversed wave (Kirchhoff's law). A wave with the in-plane wave
vector at azimuth $\varphi$ travels towards $(\cos\varphi, \sin\varphi)$, and the reversed
one leaves towards $-\hat{\boldsymbol{\kappa}}$, so the absorptance of the incidence
$(\theta, \varphi)$ is the emissivity towards $(\theta, \varphi + \pi)$, and
`gfem.diffraction.emissivity` labels its rows so.

## Aperture fields

`gfem.diffraction.aperture_field` samples a metasurface once per cell, with the
values power waves relative to the incident plane wave, as the entries of S
are. `gfem.diffraction.propagate` multiplies each plane wave of the field by
$e^{-ik_zd}$ for a distance $d$ into the half-space behind the surface, with
$k_z = \sqrt{k^2 - q^2}$ where the wave propagates and
$k_z = -i\sqrt{q^2 - k^2}$ where it is evanescent, so that evanescent waves
decay under $e^{+i\omega t}$. The centring of the cells and the two components
are on {doc}`conventions`.

## Incident fields

A plane wave's `amplitude` $E_0$ is the peak field in V/m in the layer it
comes from; in a homogeneous background its field is
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
into the port.

## Multipoles

A sphere at the origin under $E_0\hat{x}e^{-ikz}$, with $\bar a_l, \bar b_l$ the
conjugates of Bohren and Huffman's coefficients, has
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
Alaee's moments are gfem's with $-1/(i\omega) \to +1/(i\omega)$.

## Validity and limits

The Floquet checks use a flat interface, whose reflection is known in closed
form, on a coarse mesh: they agree to the mesh error, not to round-off. Away
from normal incidence the references are those of the order's own plane of
incidence, and they are discontinuous at $\boldsymbol{\kappa} = 0$. The
multipole expansion is about one origin, and its default degree is set by the
size parameter of the scatterer.
