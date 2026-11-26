# Effective medium of a slab

Replace a slab of a periodic cell, a layer of meta-atoms or a stack of patterned
films, by a homogeneous slab and read its permittivity $\varepsilon$ and
permeability $\mu$. The helper {py:func}`gfem.diffraction.effective_medium`
inverts (Nicolson-Ross-Weir) the normal-incidence reflection and transmission
read from the specular {py:obj}`gfem.FloquetPort` ports of two faces
({doc}`floquet-ports`): one solve per frequency band.

## Put the reference planes on the slab

What lies between the two reference planes is the slab, of thickness $d$. Give
both faces the `offset` that lands each port's plane on the slab surface. A
layer of rods, $d = 0.2\ \mu\mathrm{m}$, a distance `gap` from faces `"top"`
and `"bottom"` (built as in {doc}`diffraction-efficiency`):

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

Light the specular order of the incident face in both polarizations; the other
face's columns are optional.

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
$\hat{\mathbf{e}}_1$ has the refractive index $n_0$ and relative impedance
$z_0$, and gives $\varepsilon_{11} = n_0/z_0$ and $\mu_{22} = n_0 z_0$; the
wave along $\hat{\mathbf{e}}_2$ gives $\varepsilon_{22} = n_1/z_1$ and
$\mu_{11} = n_1 z_1$. A passive slab has $\operatorname{Im} n \le 0$
({py:obj}`gfem.conventions`).

## Choose the branch

A transmission only fixes the phase $n k_0 d$ modulo $2\pi$:
$n = \left(2\pi m - \arg T + i\ln|T|\right)/(k_0 d)$ with $m$ = `branch`, the
integer sheet at the lowest frequency, and $T = e^{-i n k_0 d}$ the transmission
through the slab alone.

- `branch=0` is right when $\operatorname{Re}(n)\,k_0 d < \pi$ at the lowest
  frequency. The rod layer above has $n k_0 d \le 3\ \mathrm{rad}$ across its
  band, so it sits on branch 0; the same material $0.3\ \mu\mathrm{m}$ thick at
  $1\ \mu\mathrm{m}$ has $n k_0 d \approx 3.8\ \mathrm{rad}$ and needs
  `branch=1`.
- `branch` takes one int, or a pair `(m_0, m_1)` when the two polarizations are
  on different sheets; `em.branch` is always the pair.
- The phase of $T$ is unwrapped along the band from the lowest frequency, so one
  `branch` serves the sweep. Frequencies must increase strictly, with the phase
  moving less than $\pi$ between neighbours.
- If two branches look reasonable, solve a thinner slab or a lower frequency
  (the right branch tends to 0). An $\varepsilon(f)$ with jumps, or a sign-changing
  $\operatorname{Im}\varepsilon$ in a passive material, points to the wrong
  branch.

## When the numbers mean something

- **Validity.** The values describe the slab only while $|n| k_0 a \ll 1$, $a$
  the period; near the Bragg condition or where the cell diffracts they are only
  the parameters that reproduce $S_{11}$ and $S_{21}$.
- **Half-wave thickness.** At $n k_0 d = m\pi$ a low-loss slab reflects nothing
  whatever its impedance, so $\varepsilon$ and $\mu$ lose all accuracy: keep the
  band clear, or use a thinner slab.
- **Check with the background.** Retrieving an empty slab gives back the medium
  behind the faces.

## Differentiate

Every step is `jax.numpy`, so a loss on the retrieved permittivity
differentiates toward the cell's parameters, here the rod permittivity toward a
target $\varepsilon_\mathrm{target}$:

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

`branch` is an integer and is not differentiated (`NonDifferentiableError` if
traced).

## Limits

- **Normal incidence.** An angle sweep or a wave vector with a tangential part
  raises `ValueError`.
- **Same medium on both faces**, or `NotImplementedError`.
- **Principal axes along the lattice.** A cross-polarized reflection or
  transmission above 1 % of the co-polarized total
  $\sqrt{|r_{ii}|^2 + |t_{ii}|^2}$ warns: align the lattice with the axes.
- **Symmetric cell.** The inversion assumes the same reflection from both sides;
  when the other face's columns are excited, a difference above 1 % warns that
  the cell is asymmetric or bianisotropic.
- **Planes.** The faces must be antiparallel and the planes must not meet or
  cross (`ValueError`); both polarizations of order $(0, 0)$ need a port on both
  faces.
- **`jax.jit`.** Put the solve inside the jitted function: a result passed as an
  argument traces its wave vector.

## Next

{doc}`diffraction-efficiency`, {doc}`floquet-ports`, {doc}`periodic-cell`,
{doc}`s-parameter-gradient`, {py:obj}`gfem.conventions`, {doc}`../capabilities`.
