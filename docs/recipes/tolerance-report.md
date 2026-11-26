# Fabrication tolerance and yield

A foundry gives a width as "±0.3 µm" and a substrate as $\varepsilon_r \pm 0.2$. How much do
the S-parameters you designed for move under those spreads, which spread
dominates, and what fraction of devices still meets the specification?
{py:obj}`gfem.tolerance_report` answers from one solve and one adjoint pass per
quantity, whatever the number of parameters: it differentiates your forward model
at the nominal design and propagates the spreads through the derivatives.

`model` below is the getting-started line of {doc}`../get-started`: a parallel-plate
line 40 mm long, 6 mm wide and 1 mm high between magnetic side walls `"side"`,
with a lumped port `p1` and `p2` at each end matched to the empty line, and a
ceramic of $\varepsilon_r = 10$ in its middle 20 mm.

## The script

The parameters are the edge bias of the two side walls (every wall node moves
by the same length, and a positive bias moves a wall into the line, so the width
changes by minus twice the bias) and the ceramic's permittivity. The quantities are the transmitted power at 5 and
8 GHz, which must stay above 0.75 and 0.3.

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

- `quantities` is any JAX function of a dict of named real scalars that returns a
  dict of named real scalars. gfem never builds the model: you do, so anything
  that differentiates can be a parameter. A complex or array value is refused;
  take `jnp.abs`, `.real`, `jnp.angle` or `s.db()` and give one name per scalar.
- `walls(p["edge bias"])` is a {py:obj}`gfem.shape.boundary_morph` called with a
  scalar: every moving node of the surface is displaced by that length along its
  normal. This is the surface's **edge bias**, positive when the conductor, the
  void or the `grows` volume grows: the side walls bound the domain, so the
  void grows into the line ({doc}`shape-gradient`). A lumped port holds its
  face, so the walls bend back to the ports across the first elements.
- A {py:obj}`gfem.Material` holds constants, so the permittivity goes through a
  {py:obj}`gfem.MaterialField.uniform` of the region, rebuilt with
  `dataclasses.replace`, which keeps the loss tangent, permeability and
  conductivity of the material.
- A tolerance $t$ is the half-width of "$\pm t$" and is read as three standard
  deviations of a Gaussian, $\sigma = t/3$: a foundry's "±0.3 µm" goes in as
  written. The tolerance of a parameter held fixed is 0. A uniform spread
  $\pm a$ goes in as $t = \sqrt{3}\,a$ (its standard deviation is $a/\sqrt{3}$);
  the yield is then approximate.

## Reading the report

With the Jacobian $J_{ij} = \partial f_i / \partial p_j$ at the nominal design,
quantity $i$ has the standard deviation $s_i$ with

$$
s_i^2 = \sum_j J_{ij}^2 \sigma_j^2,
$$

for independent parameters, and the report prints $\pm 3 s_i$ beside the
worst case $\sum_j |J_{ij}|\, t_j$, which is the sure bound when the model is
linear. `fail` beside it means that the worst case leaves the spec window, not
that a device does.

- **Effect at +tolerance** is $J_{ij} t_j$, the signed change of the quantity
  when one parameter sits at the top of its range, with the share
  $J_{ij}^2 \sigma_j^2 / s_i^2$ of the variance. Here the 5 GHz transmission is
  mostly a matter of the permittivity, and the 8 GHz one of the line's width: the
  ranking depends on the quantity, which is why the table is per quantity.
- **Yield** of one spec is exact under the linear model,
  $\Phi\big((f_i - l_i)/s_i\big)$ for a lower limit $l_i$, with $\Phi$ the
  standard normal distribution function; **Cpk** is the distance from the nominal
  value to the nearest limit in units of $3 s_i$ (1 means the limit sits at
  $\pm 3\sigma$). A `*` marks a limit beyond $\pm 3\sigma$, where the linear model
  was not checked.
- **All specs** is the joint yield. With several specs it is a Monte Carlo on the
  linear model (seeded, so the same call gives the same number), with its standard
  error, and the Bonferroni and Fréchet bounds
  (`joint_yield_bounds`) that any joint probability with these marginals obeys.
- **Linearity** is a test: the parameter with the largest variance share is
  evaluated at $\pm t$ with two more forward solves, and a quantity is flagged
  when the linear prediction misses by more than a tenth of $\pm 3 s_i$.
  `check_linearity=k` checks $k$ parameters and `0` turns the check off. The
  mean shift is the average of the two evaluations against the nominal value:
  the first correction when a response curves.

Every field is on the returned {py:obj}`gfem.ToleranceReport`: `jacobian`,
`effect`, `share`, `std`, `worst_case`, `yields`, `cpk`, `joint_yield`,
`linear_error` and `nonlinear`. `report.to_markdown()` gives the same tables as
pipe tables.

## Other parameters

- **A gap.** The gap between two walls of one morph changes by $-2t$ when each
  wall grows by $t$, and a width between two walls by $+2t$. A gap tolerance of
  $\pm g$ is an edge-bias tolerance of $g/2$.
- **Two surfaces.** Two independent surfaces are two morphs, one parameter each.
  Their displacements add to first order, which is all the Jacobian reads:
  `model.mesh.with_nodes(a(p["w"]).nodes + b(p["g"]).nodes - model.mesh.imported_nodes)`.
- **A fabrication model.** `fab(rho, beta=..., bias=p["bias"])` of a
  {py:obj}`gfem.design.FabricationModel` takes a traced edge bias, so the bias of a
  blurred and thresholded layer pattern is a parameter like any other.
- **Correlation.** `correlation={("edge bias", "eps_r"): 0.5}` sets the
  correlation coefficient of a pair of parameters; unlisted pairs are 0. The
  variance share then has a remainder, the part that belongs to no single
  parameter.

## Cost

The report costs $1 + 2k$ forward evaluations ($k$ the number of checked
parameters, 1 by default) and one backward pass per quantity, whatever the
number of parameters; a backward pass costs well under a tenth of a forward
solve for a driven $S$. Add parameters freely, and keep the quantities to the
ones you have a spec for.

## Check your results

The Jacobian is a derivative of the discrete solve, so compare one entry with a
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

- The report differentiates the discrete model, not the mesh's error. Compare the
  discretization error estimate of the nominal solve ({doc}`mesh-refinement`) with
  the distance from the nominal value to the spec and with $\pm 3 s_i$: an
  estimate near either says the mesh, not the process, limits the answer.
- A quantity flagged by the linearity check needs a smaller tolerance or a second
  look: its yield is a linear-model number. Shrinking the tolerances until the
  flag goes away shows how far the linear range extends.

## Limits

- The yield is a linear-model number. It is exact for a linear response, and it
  degrades as the response curves over $\pm t$ or as a limit moves beyond
  $\pm 3\sigma$ (`*`). It is not a Monte Carlo over re-solves of the device.
- A parameter that the function does not read is refused by name, as is a
  quantity that is complex, an array or not finite.
- The function is reverse-mode differentiated, so it must go through
  {py:obj}`gfem.solve`, and a path that does not differentiate (a solver
  tolerance, a mesh remap) raises {py:obj}`gfem.NonDifferentiableError`.
  `tolerance_report` returns numpy values: call it outside `jax.jit` and
  `jax.grad`.
- A parameter is a real scalar, and a surface offset is the same along the
  whole surface. A roughness or a position-dependent bias is a field of
  parameters, which this report does not treat.

## Next

{py:obj}`gfem.tolerance_report`, {py:obj}`gfem.ToleranceReport`,
{py:obj}`gfem.shape.boundary_morph`; {doc}`shape-gradient`,
{doc}`s-parameter-gradient`, {doc}`robust-optimization`,
{doc}`fabricable-layout`; {doc}`../capabilities`.
