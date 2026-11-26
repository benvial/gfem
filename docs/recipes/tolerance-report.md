# Fabrication tolerance and yield

A foundry gives a width as "±0.3 µm" and a substrate as $\varepsilon_r \pm 0.2$. How much do
the S-parameters you designed for move under those spreads, which spread
dominates, and what fraction of devices still meets the specification?
{py:obj}`gfem.tolerance_report` differentiates your forward model at the nominal
design and propagates the spreads through the derivatives: one solve and one
adjoint pass per quantity, whatever the number of parameters.

## The script

`model` is the line of {doc}`../get-started`. The parameters are the side walls'
edge bias (positive moves a wall into the line) and the ceramic's permittivity;
the quantities are the transmitted power at 5 and 8 GHz, which must stay above
0.75 and 0.3.

```python
import dataclasses
import urllib.request

import jax.numpy as jnp

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")
z_line = 376.73 / 6  # ohm: the 6 mm wide, 1 mm high parallel-plate line in air
model = gfem.Model(
    mesh,
    materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)},
    boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    ports=[
        gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
        gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
    ],
)

walls = gfem.shape.boundary_morph(model, "side")
ceramic = gfem.MaterialField.uniform("design", model.materials["design"])
problem = gfem.Driven(frequency=[5e9, 8e9], excite="p1")


def quantities(p):
    """Named real quantities of one fabricated device."""
    moved = model.replace(mesh=walls(p["edge bias"]))  # a scalar moves every wall node
    moved = moved.with_material(
        "design", dataclasses.replace(ceramic, eps_r=p["eps_r"])
    )
    s = gfem.solve(moved, problem).s
    return {
        f"|S21|² at {f / 1e9:g} GHz": jnp.abs(s["p2", "p1"][i]) ** 2
        for i, f in enumerate(problem.frequency)
    }


report = gfem.tolerance_report(
    quantities,
    {"edge bias": 0.0, "eps_r": 10.0},
    tolerance={"edge bias": 300e-6, "eps_r": 0.2},
    specs={"|S21|² at 5 GHz": (0.75, None), "|S21|² at 8 GHz": (0.3, None)},
)
print(report)
```

```text
Tolerance report: 2 quantities, 2 parameters
Linearized at the nominal design. Each tolerance ±t is 3 standard deviations of a Gaussian;
parameters independent.

Parameter   nominal   tolerance
edge bias   0         ±0.0003
eps_r       10        ±0.2

Quantity          nominal   ±3σ      worst case            spec      yield     Cpk
|S21|² at 5 GHz   0.812     ±0.07    [0.717, 0.908] fail   >= 0.75   99.6 %    0.9
|S21|² at 8 GHz   0.366     ±0.064   [0.282, 0.45] fail    >= 0.3    99.9 %*   1
All specs: 99.5 % ± 0.01 % (Monte Carlo on the linear model; bounds 99.5 % to 99.6 %)
* the spec limit lies beyond ±3σ, where the linear model was not checked.

Effect at +tolerance (share of variance), largest first
|S21|² at 5 GHz   eps_r -0.06 (74 %)   edge bias +0.0354 (26 %)
|S21|² at 8 GHz   edge bias -0.0582 (83 %)   eps_r -0.026 (17 %)

Linearity, checked on edge bias (forward solves at ±0.0003): linear error at most 0.0031 (|S21|² at 5 GHz), within a tenth of ±3σ for every quantity.
Mean shift: |S21|² at 5 GHz -0.00028, |S21|² at 8 GHz +3.8e-06.
```

- `quantities` is any JAX function of a dict of named real scalars returning a
  dict of named real scalars. A complex or array value is refused: take
  `jnp.abs`, `.real`, `jnp.angle` or `s.db()`.
- `walls(p["edge bias"])` is a {py:obj}`gfem.shape.boundary_morph` called with a
  scalar: every moving node moves by that length along its normal, positive
  when the conductor, void or `grows` volume grows ({doc}`shape-gradient`).
- A {py:obj}`gfem.Material` holds constants, so the permittivity goes through a
  {py:obj}`gfem.MaterialField.uniform`, rebuilt with `dataclasses.replace`.
- A tolerance $t$ is the half-width of "$\pm t$", read as three standard
  deviations of a Gaussian, $\sigma = t/3$; a fixed parameter has tolerance 0. A
  uniform spread $\pm a$ goes in as $t = \sqrt{3}\,a$, and the yield is then
  approximate.

## Reading the report

With the Jacobian $J_{ij} = \partial f_i / \partial p_j$ at the nominal design,
quantity $i$ has the standard deviation $s_i$ with

$$
s_i^2 = \sum_j J_{ij}^2 \sigma_j^2,
$$

for independent parameters, and the report prints $\pm 3 s_i$ beside the worst
case $\sum_j |J_{ij}|\, t_j$, the sure bound for a linear model.

| Entry | Meaning |
| --- | --- |
| `fail` | the worst case leaves the spec window, not that a device does |
| Effect at +tolerance | $J_{ij} t_j$, with the share $J_{ij}^2 \sigma_j^2 / s_i^2$ of the variance; here 5 GHz is mostly the permittivity and 8 GHz the width |
| Yield | $\Phi\big((f_i - l_i)/s_i\big)$ for a lower limit $l_i$, $\Phi$ the standard normal distribution function |
| Cpk | distance to the nearest limit in units of $3 s_i$; `*` marks a limit beyond $\pm 3\sigma$ |
| All specs | joint yield: a seeded Monte Carlo on the linear model with its standard error, and the Bonferroni and Fréchet bounds `joint_yield_bounds` |
| Linearity | the largest-share parameter evaluated at $\pm t$ with two more solves; a quantity is flagged when the linear prediction misses by over a tenth of $\pm 3 s_i$. `check_linearity=k` checks $k$ parameters, `0` turns it off |

Every field is on the returned {py:obj}`gfem.ToleranceReport`;
`report.to_markdown()` gives the tables.

## Other parameters

- **A gap** between two walls of one morph changes by $-2t$ when each wall grows
  by $t$: a gap tolerance $\pm g$ is an edge-bias tolerance $g/2$.
- **Two surfaces** are two morphs, one parameter each; displacements add to first
  order:
  `model.mesh.with_nodes(a(p["w"]).nodes + b(p["g"]).nodes - model.mesh.imported_nodes)`.
- **A fabrication model.** `fab(rho, beta=..., bias=p["bias"])` of a
  {py:obj}`gfem.design.FabricationModel` takes a traced bias.
- **Correlation.** `correlation={("edge bias", "eps_r"): 0.5}`; unlisted pairs
  are 0, and the variance share gets a remainder.

## Cost

$1 + 2k$ forward evaluations ($k$ the checked parameters, 1 by default) and one
backward pass per quantity, whatever the number of parameters.

## Check your results

The Jacobian is a derivative of the discrete solve: compare one entry with a
central difference:

```python
q, name = "|S21|² at 5 GHz", "eps_r"
i, j = report.quantities.index(q), report.parameters.index(name)
h = 1e-2
up = quantities({"edge bias": 0.0, "eps_r": 10.0 + h})[q]
down = quantities({"edge bias": 0.0, "eps_r": 10.0 - h})[q]
print(
    f"d{q}/d{name}: report {report.jacobian[i, j]:.5f}, difference {float((up - down) / (2 * h)):.5f}"
)
```

```text
d|S21|² at 5 GHz/deps_r: report -0.30018, difference -0.30017
```

- The report differentiates the discrete model, not the mesh's error: compare
  the nominal solve's error estimate ({doc}`mesh-refinement`) with the distance
  to the spec and with $\pm 3 s_i$.
- A flagged quantity needs a smaller tolerance: shrink until the flag goes away.

## Limits

- The yield is a linear-model number, exact for a linear response, degrading as
  it curves over $\pm t$ or a limit passes $\pm 3\sigma$ (`*`); it is not a
  Monte Carlo over re-solves.
- A parameter the function does not read is refused by name, as is a complex,
  array or non-finite quantity.
- The function is reverse-mode differentiated, so it goes through
  {py:obj}`gfem.solve`; a path that does not differentiate raises
  {py:obj}`gfem.NonDifferentiableError`. `tolerance_report` returns numpy
  values: call it outside `jax.jit` and `jax.grad`.
- A parameter is a real scalar, the same along the whole surface; a roughness is
  a field of parameters, which this report does not treat.

## Next

{py:obj}`gfem.tolerance_report`, {py:obj}`gfem.ToleranceReport`,
{py:obj}`gfem.shape.boundary_morph`; {doc}`shape-gradient`,
{doc}`s-parameter-gradient`, {doc}`robust-optimization`,
{doc}`fabricable-layout`; {doc}`../capabilities`.
