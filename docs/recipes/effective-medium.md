# Effective medium of a slab

Replace a slab of a periodic cell, a layer of meta-atoms or a stack of
patterned films, by a homogeneous slab and read its permittivity
$\varepsilon$ and permeability $\mu$. The retrieval is the Nicolson-Ross-Weir
inversion of the reflection and transmission of the slab at normal incidence,
read from the specular {py:obj}`gfem.FloquetPort`s of two faces
({doc}`floquet-ports`), so it needs one solve per frequency band and no new
physics. The helper is {py:func}`gfem.diffraction.effective_medium`; the
cell is built as in {doc}`diffraction-efficiency`.

## Put the reference planes on the slab

The retrieval treats what lies between the two reference planes as the slab,
of thickness $d$. A Floquet port's plane is its face moved by `offset` along
the inward normal, so give both faces the offset that lands the plane on the
slab surface. The cell below has a layer of rods, thickness
$d = 0.2\ \mu\mathrm{m}$, centered between faces `"top"` and `"bottom"`
that are a distance `gap` away from it, meshed with the volumes `"rod"` and
`"air"`:

```python
import numpy as np

import gfem
from gfem import diffraction as dif

c = 299_792_458.0  # m/s
mesh = gfem.Mesh.from_gmsh("rod_cell.msh", length_unit="um")
cell = gfem.Model(
    mesh,
    materials={"rod": gfem.Material(eps_r=4.0), "air": gfem.Material()},
    boundaries=[
        gfem.Periodic(("x_min", "x_max")),
        gfem.Periodic(("y_min", "y_max")),
    ],
)
gap = 0.4e-6  # m, from each open face to the slab surface
band = np.linspace(0.8, 1.2, 9) * c / 1.0e-6  # Hz, ascending
cell = cell.replace(
    ports=[
        *dif.floquet_ports(cell, "top", band, offset=gap),
        *dif.floquet_ports(cell, "bottom", band, offset=gap),
    ]
)
lit = [f"{face} (0, 0) {pol}" for face in ("top", "bottom") for pol in ("TE", "TM")]
options = gfem.SolverOptions(order=2, linear_solver="direct")
res = gfem.solve(cell, gfem.Driven(band, excite=lit), options=options)
```

Light the specular order of the incident face in both polarizations; the
columns of the other face are optional (see Limits).

## Retrieve

```python
em = dif.effective_medium(res, "top", "bottom", model=cell, branch=0)
em.permittivity  # (n_freq, 2): (eps_11, eps_22) in the lattice frame
em.permeability  # (n_freq, 2): (mu_11, mu_22)
em.index, em.impedance  # (n_freq, 2): n and Z / Z_0 of each polarization column
em.thickness, em.frequency  # the slab thickness in m, the frequencies in Hz
```

The slab is read as a diagonal tensor in the lattice frame
$(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2)$. The wave with $\mathbf{E}$ along
$\hat{\mathbf{e}}_1$ (TM at normal incidence) has the refractive index $n_0$
and the relative impedance $z_0$, and gives $\varepsilon_{11} = n_0/z_0$ and
$\mu_{22} = n_0 z_0$; the wave along $\hat{\mathbf{e}}_2$ gives
$\varepsilon_{22} = n_1/z_1$ and $\mu_{11} = n_1 z_1$. Time dependence is
$e^{+i\omega t}$, so a passive slab has $\operatorname{Im} n \le 0$
({py:obj}`gfem.conventions`).

## Choose the branch

A transmission only fixes the phase $n k_0 d$ modulo $2\pi$, so the retrieval
needs the sheet:
$n = \left(2\pi m - \arg T + i\ln|T|\right)/(k_0 d)$ with $m$ = `branch`, the
integer sheet at the lowest frequency, and $T = e^{-i n k_0 d}$ the
transmission through the slab alone.

- `branch=0` is right when $\operatorname{Re}(n)\,k_0 d < \pi$ at the lowest
  frequency, a slab thinner than half a wavelength in its own medium. The rod
  layer above, with $n \le 2$ (an upper estimate: the rods do not fill the
  cell), has $n k_0 d \le 2.0\ \mathrm{rad}$ at the lowest frequency of the
  band ($\lambda = 1.25\ \mu\mathrm{m}$) and at most $3\ \mathrm{rad}$
  across it, below $\pi$, so it sits on branch 0. A slab of the same material
  $0.3\ \mu\mathrm{m}$ thick at a wavelength of $1\ \mu\mathrm{m}$ has
  $n k_0 d \approx 3.8\ \mathrm{rad} > \pi$ and needs `branch=1`.
- `branch` takes one int for both columns, or a pair `(m_0, m_1)`, one per
  polarization column, when the two waves are on different sheets (an
  anisotropic slab). `em.branch` is always the pair.
- Along a band the phase of $T$ is unwrapped, starting from its principal
  value at the lowest frequency, so one `branch` serves the whole sweep. The
  frequencies must increase strictly, and the grid fine enough that the phase
  moves less than $\pi$ between neighbours.
- When two branches both look reasonable, solve a thinner slab or a lower
  frequency: for $k_0 d \to 0$ the right branch is 0 and its result is
  smooth. A $\varepsilon(f)$ with jumps, or an $\operatorname{Im}\varepsilon$
  that changes sign in a passive material, points to the wrong branch.

## When the numbers mean something

- **Validity.** The values describe the slab only while
  $|n| k_0 a \ll 1$, $a$ the period along the propagation. Near the Bragg
  condition, or where the cell diffracts, the retrieved
  $\varepsilon$ and $\mu$ are not material constants, only the parameters
  that reproduce $S_{11}$ and $S_{21}$.
- **Half-wave thickness.** At $n k_0 d = m\pi$ ($m \ge 1$) a slab of low loss
  reflects nothing and transmits with unit modulus whatever its impedance, so
  $z$ is undetermined ($S_{11} \to 0$, $S_{21} \to \pm 1$) and
  $\varepsilon$, $\mu$ and their derivatives lose all accuracy near such
  frequencies. Keep a band clear of them, or use a thinner slab.
- **Check with the background.** Retrieving an empty slab gives back the
  permittivity and permeability of the medium behind the faces.

## Differentiate

Every step is `jax.numpy`, so a loss on the retrieved permittivity
differentiates toward the cell's parameters, here the rod permittivity,
toward a target $\varepsilon_\mathrm{target}$:

```python
import jax

base, air = cell, gfem.Material()  # the cell above, with a free rod permittivity
eps_target = 3.2  # target relative permittivity at the centre of the band


def loss(eps_rod):
    cell = base.replace(materials={"rod": gfem.Material(eps_r=eps_rod), "air": air})
    res = gfem.solve(cell, gfem.Driven(band, excite=lit), options=options)
    em = dif.effective_medium(res, "top", "bottom", model=cell)
    return abs(em.permittivity[4, 0] - eps_target) ** 2


grad = jax.grad(loss)(4.0)
```

`branch` is an integer sheet and is not differentiated
(`NonDifferentiableError` if traced); the unwrapping adds constants of
$2\pi$, which have no gradient.

## Limits

- **Normal incidence.** The retrieval refuses an angle sweep and a wave vector
  with a tangential part (`ValueError`).
- **Same medium on both faces.** Faces with different media behind them
  raise `NotImplementedError`.
- **Principal axes along the lattice.** The tensor is diagonal in
  $(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2)$. A cross-polarized reflection or
  transmission above 1 % of the co-polarized total
  $\sqrt{|r_{ii}|^2 + |t_{ii}|^2}$ of the same incident column warns: align
  the lattice with the axes of the structure. The total, not the block's own
  co-polarized entry, so a matched slab with a vanishing co-polarized
  reflection does not warn on round-off.
- **Symmetric cell.** The inversion assumes the same reflection from both
  sides. When the columns of the other face are excited too, a difference of
  more than 1 % warns that the cell is asymmetric or bianisotropic, and the
  result describes only the wave from the first face.
- **Planes.** The two faces must be antiparallel and the planes must not meet
  or cross (`ValueError`), and both polarizations of order $(0, 0)$ need a
  port on both faces.
- **`jax.jit`.** Put the solve inside the jitted function: the retrieval
  checks normal incidence on the result's wave vector, which a result passed
  as an argument traces (an error).

## Next

{doc}`diffraction-efficiency`, {doc}`floquet-ports`, {doc}`periodic-cell`,
{doc}`s-parameter-gradient`, {py:obj}`gfem.conventions`, {doc}`../capabilities`.
