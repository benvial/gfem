# Robust optimization

A density design is fabricated with an etch bias: edges move outwards
(under-etch, more material) or inwards (over-etch, less material).
{py:obj}`gfem.design.robust` gives the eroded, nominal and dilated variants of
one filtered density and the optimizer works on the worst, so performance holds
across the bias and the filter radius sets a minimum feature size for solid and
void. The worked example is the
[topology tutorial](../tutorials/topology.ipynb); the later sections add a
{term}`fabrication model` (a blur and an {term}`edge bias` in metres), a
foundry's {term}`minimum width` and {term}`minimum spacing`, and the checks
that accept the design before {doc}`fabricable-layout`.

## The chain

```python
import jax
import jax.numpy as jnp

import gfem
from gfem import design

grid = design.LayerGrid(mesh, "design", pitch=0.5e-3)  # pixels over the layer
blur = design.helmholtz_filter(grid, radius=1.5e-3)  # minimum length scale
AIR, CERAMIC = gfem.Material(), gfem.Material(eps_r=10.0)


def transmission(density):
    """|S21|^2 of one design (one density per pixel)."""
    eps = gfem.MaterialField.interpolate(grid.to_elements(density), AIR, CERAMIC)
    s = gfem.solve(model.with_material("design", eps), problem).s
    return jnp.abs(s["p2", "p1"][0]) ** 2


def worst_loss(rho, beta):
    variants = design.robust(blur(rho), beta=beta, delta=0.15)
    losses = jax.vmap(lambda d: 1 - transmission(d))(jnp.stack(variants))
    return jnp.max(losses)  # eroded, nominal, dilated: the worst one


value, grad = jax.value_and_grad(worst_loss)(rho, 4.0)
```

- **Eroded** is the projection at the higher threshold `eta + delta`,
  **dilated** at `eta - delta`; `delta` sets the etch tolerance and has no
  default.
- `jax.vmap` over `gfem.solve` runs the three solves in sequence: three
  forward solves and three backward passes per step.
- A volume constraint goes on the **dilated** design.
- Raise `beta` in steps (2, 4, 8, 16, 32); beyond about 64 the gradients turn
  jagged.
- `delta` is a threshold shift, not a length: near a straight edge of a nearly
  binary design the edge moves by $|b| = -l\ln(1 - 2\delta)$,
  $l = R/(2\sqrt3)$ for a filter radius $R$ ($0.064R$ at $\delta = 0.1$). For
  a bias in metres use a fabrication model.

## A fabrication model

The optimizer designs the **drawn** design, `design.project(blur(rho), beta)`,
whose pattern goes to GDS. {py:obj}`gfem.design.FabricationModel` predicts what
the process prints: a Gaussian blur of standard deviation $\sigma$, then a
threshold set by the edge bias $b$ (m, $b > 0$ grows the solid),

$$
\hat\rho_b = P_{\beta,\,\eta(b)}(G_\sigma * \hat\rho), \qquad
\eta(b) = \tfrac12\operatorname{erfc}\frac{b}{\sqrt2\,\sigma},
$$

so a straight edge prints moved by $b$ and corners round off over about
$2\sigma$. The solves see the **printed** designs:

```python
grid = design.LayerGrid(mesh, "design", pitch=7.5e-9)  # sigma / 4
blur = design.helmholtz_filter(grid, radius=165e-9)
fab = design.FabricationModel(grid, sigma=30e-9, bias=-5e-9)  # a 5 nm over-etch


def worst_loss(rho, beta):
    drawn = design.project(blur(rho), beta)  # the design intent, to GDS
    printed = fab.robust(drawn, beta=beta, tolerance=10e-9)  # bias -15, -5, +5 nm
    losses = jax.vmap(lambda d: 1 - transmission(d))(jnp.stack(printed))
    return 0.01 * jax.nn.logsumexp(losses / 0.01)  # a smooth worst case
```

- `fab.robust` returns the same `Robust(eroded, nominal, dilated)`, printed at
  biases $b - t$, $b$ and $b + t$ for a tolerance $t$, with the drawn design's
  $\beta$ (a grey drawn design cannot go to GDS).
- **Pitch and bias.** Use a pitch of $\sigma/4$ or finer and keep
  $|b|$, $|b \pm t|$ within $1.5\sigma$; $\sigma$ below the pitch or biases
  above $2\sigma$ are refused.
- **Narrow features** print narrower than $w + 2b$ below about
  $3\sigma + 2\max(-b, 0)$ and vanish below about $2\sigma$.
- **Fixed guides are not biased.** Geometry outside the grid keeps its drawn
  width, so a printed variant steps by $b$ per side against it.
- `bias` may be traced, for the derivative in the bias.

## Worst case

The worst of the three variants is a maximum, not smooth where two tie.

- With Adam, use the smooth maximum `t * jax.nn.logsumexp(f / t)`, above the
  maximum by at most $t \ln 3$; $t = 0.01$ suits losses between 0.01 and 1.
- With MMA or CCSAQ, use the epigraph form: minimize a variable $s$ subject to
  $f_i(\rho) \le s$ per variant, starting $s$ 5 % above the worst.

## Minimum width and spacing

A foundry's minimum width and spacing become two differentiable constraints on
the filtered density, each at most 0 when the drawn pattern keeps its rule:

```python
rules = design.length_scale_constraints(blur, min_width=165e-9, min_spacing=165e-9)


def constraints(rho):
    return jnp.stack(rules(blur(rho)))  # [solid, void], each <= 0 when held
```

Run them in two stages: optimize without them, raising $\beta$ to 32, until the
topology forms; then switch them on at $\beta \ge 16$, never from a grey start
(a grey design has no lines or spaces to measure). A quadratic penalty with
Adam is the plain path, its weight raised in steps:

```python
import optax


def penalized(rho, mu):
    excess = jnp.maximum(constraints(rho), 0.0)
    return worst_loss(rho, 32.0) + mu * jnp.sum(excess**2)


optimizer = optax.adam(0.02)
for mu in (1.0, 10.0, 100.0):
    state = optimizer.init(rho)
    for _ in range(20):
        g = jax.grad(penalized)(rho, mu)
        updates, state = optimizer.update(g, state)
        rho = jnp.clip(optax.apply_updates(rho, updates), 0.0, 1.0)
```

With nlopt, CCSAQ takes the two values as inequality constraints and holds
them more closely; scale the objective into about 1 to 100:

```python
import nlopt
import numpy as np


def objective(x, grad):
    f = jax.value_and_grad(lambda r: 100.0 * worst_loss(r, 32.0))
    value, g = f(jnp.asarray(x.reshape(grid.shape)))
    if grad.size:
        grad[:] = np.ravel(g)
    return float(value)


def rule_values(result, x, grad):
    r = jnp.asarray(x.reshape(grid.shape))
    result[:] = np.asarray(constraints(r))
    if grad.size:
        grad[:] = np.asarray(jax.jacrev(constraints)(r)).reshape(2, -1)


opt = nlopt.opt(nlopt.LD_CCSAQ, rho.size)
opt.set_lower_bounds(0.0)
opt.set_upper_bounds(1.0)
opt.set_min_objective(objective)
opt.add_inequality_mconstraint(rule_values, [1e-3, 1e-3])
opt.set_maxeval(100)
rho = opt.optimize(np.ravel(rho)).reshape(grid.shape)
```

- **What $g \le 0$ guarantees.** Straight lines and spaces reach about 0.97 of
  their rule, compact features about 1.1 times it: constrain about 10 % above
  the foundry rule (165 nm for 150 nm), then confirm with
  `LayerPattern.check` ("Accept the design").
- **Range.** A rule above $1.25R$ or a pitch above $R/3$ is refused: raise the
  filter radius or refine the grid.
- On the single-filter path (`design.robust`) the radius alone holds a minimum
  length $\ell$: `design.radius_for_length_scale(150e-9, delta=0.1)` gives
  $2.50\,\ell$.

## Real neighbours, symmetry and footprint

The filter, the constraints and the blur treat the grid's edge as a mirror, so a
feed guide outside the grid is invisible to them. Mesh the design region a
margin beyond the free area, into the guides, and pin the margin's pixels to
the real neighbours before the filter:

```python
import numpy as np

guides = design.LayerPattern.from_polygons([feed_in, feed_out])  # (n, 2) arrays, m
fixed = grid.area_fraction(guides)  # the share of each pixel the guides cover
margin = 165e-9  # at least max(R, 3 sigma, min_spacing)
xmin, ymin, xmax, ymax = grid.bounds
x, y = np.meshgrid(grid.x, grid.y)
in_margin = (x < xmin + margin) | (x > xmax - margin)
in_margin |= (y < ymin + margin) | (y > ymax - margin)

rho = jnp.where(in_margin, fixed, rho)  # the real neighbours, pinned
rho = (rho + rho[:, ::-1]) / 2  # mirror symmetry about the grid's centre line in x
rho = jnp.where(grid.sampled, rho, 0.0)  # void outside a non-rectangular footprint
```

- The margin is at least $\max(R, 3\sigma, s_\text{min})$, $s_\text{min}$ the
  minimum spacing. The guide stubs in it are part of the pattern.
- Without a margin the edge stays a mirror: a guide meeting it at right angles
  continues exactly, but a sliver along the edge is left to the assembled
  layout's checks.

## Accept the design

Check the printed variants against the rules: the width on the **eroded**
pattern, the spacing on the **dilated** one. The fabrication model predicts
them best from the drawn design's exact area fractions (edges to about 0.015
pixel, against about 0.2 from a $\beta = 32$ projection):

```python
drawn_pattern = grid.pattern(blur(rho), eta=0.5)  # the drawn pattern, to GDS
drawn = grid.area_fraction(drawn_pattern)  # the drawn design, exact, as pixels
eroded = fab.pattern(drawn, bias=fab.bias - 10e-9)
dilated = fab.pattern(drawn, bias=fab.bias + 10e-9)
print(eroded.check(min_width=150e-9))
print(dilated.check(min_spacing=150e-9))
```

On the single-filter path the variants are contours of the filtered density at
shifted thresholds: `grid.pattern(blur(rho), eta=0.5 + delta)` for the width,
`eta=0.5 - delta` for the spacing. Then export and re-solve the drawn pattern:
{doc}`fabricable-layout`.

## Next

{doc}`fabricable-layout`, {py:obj}`gfem.design.FabricationModel`,
{py:obj}`gfem.design.length_scale_constraints`, {py:obj}`gfem.design.robust`,
{py:obj}`gfem.design.helmholtz_filter`, {py:obj}`gfem.design.project`,
{doc}`s-parameter-gradient`, {doc}`sheet-design`, the
[topology tutorial](../tutorials/topology.ipynb).
