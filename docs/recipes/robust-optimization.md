# Robust optimization

A density design is fabricated with an etch bias: edges move outwards
(under-etch, more material) or inwards (over-etch, less material).
{py:obj}`gfem.design.robust` gives the three variants of one filtered density, and
the optimizer works on the worst of them. The result keeps its performance
across the bias, and both the solid and the void get a minimum feature size
set by the filter radius. The full worked example is the
[topology tutorial](../tutorials/topology.ipynb).

The sections after the first add what a lithographic layer needs: a
{term}`fabrication model` of the process (a blur and an {term}`edge bias` in
metres), a foundry's {term}`minimum width` and {term}`minimum spacing` held as
constraints, the real neighbours of the design region, and the checks that
accept the final design. {doc}`fabricable-layout` then turns it into a GDS file.

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

- **Eroded** is the projection at the higher threshold `eta + delta` (less
  solid), **dilated** at `eta - delta` (more solid). `delta` sets the etch
  tolerance and has no default.
- `jax.vmap` over `gfem.solve` runs the three solves one after the other:
  each optimizer step costs three forward solves and three backward passes.
- `jnp.max` is not smooth where two variants tie. Adam copes; for a
  gradient-based constrained optimizer (MMA, SLSQP) use the epigraph form
  (see "Worst case" below).
- A volume constraint goes on the **dilated** design, the one with the most
  material.
- Raise `beta` in steps (2, 4, 8, 16, 32) as the design converges: a soft
  projection lets material move, a sharp one makes the design binary. End the
  continuation at $\beta = 32$; beyond about 64 the projection's gradients turn
  jagged.
- `delta` is a threshold shift, not a length. Near a straight edge of a nearly
  binary design the filtered profile falls off as $\tfrac12 e^{-x/l}$, with
  $l = R/(2\sqrt3)$ for a filter radius $R$, so the thresholds
  $\tfrac12 \pm \delta$ move the edge by $|b| = -l\ln(1 - 2\delta)$:

  | $\delta$ | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
  |---|---|---|---|---|---|
  | $\lvert b\rvert / R$ | 0.030 | 0.064 | 0.103 | 0.147 | 0.200 |

  Elsewhere (grey regions, curved or narrow features) the shift differs. For a
  bias in metres, use a fabrication model.

## A fabrication model

The optimizer designs the **drawn** design, `design.project(blur(rho), beta)`:
the design intent, whose pattern goes to GDS. The process prints something
else. {py:obj}`gfem.design.FabricationModel` predicts it: a Gaussian blur of
standard deviation $\sigma$, then a threshold set by the edge bias $b$ (m,
$b > 0$ grows the solid),

$$
\hat\rho_b = P_{\beta,\,\eta(b)}(G_\sigma * \hat\rho), \qquad
\eta(b) = \tfrac12\operatorname{erfc}\frac{b}{\sqrt2\,\sigma},
$$

so a straight edge prints moved outwards by $b$ and corners round off over
about $2\sigma$. The solves see the **printed** designs:

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

- `fab.robust` returns the same `Robust(eroded, nominal, dilated)` as
  `design.robust`, printed at biases $b - t$, $b$ and $b + t$ for a tolerance
  $t$. The printed projection uses the drawn design's $\beta$: a drawn design
  held grey while the printed one sharpens would rely on grey levels that a GDS
  file cannot carry.
- **Pitch.** At $\beta = 32$ a printed edge sits within a few hundredths of
  $\sigma$ of its intended place when the pitch is $\sigma/4$ or finer (about
  $0.1\sigma$ at $\sigma/2$). A $\sigma$ below the pitch is refused: the blur
  is then unresolved.
- **Bias range.** $|b|$ or $|b \pm t|$ above $2\sigma$ is refused: the edge
  would sit in the blur's tail. Keep them within $1.5\sigma$.
- **Narrow features.** A line of drawn width $w$ prints at $w + 2b$ once $w$ is
  above about $3\sigma + 2\max(-b, 0)$; narrower lines print narrower still and
  vanish below about $2\sigma$. Spaces are the mirror case.
- **Fixed guides are not biased.** Geometry outside the grid, such as feed
  guides, keeps its drawn width, so a printed variant steps by $b$ per side
  against it. On the bend tutorial's guides (0.55 µm, indices 2.25 and 1.444)
  a 0.55 µm guide meeting a 0.505 or 0.597 µm stub loses 6 to
  $8\times10^{-4}$ of the power per junction: for the bend's two junctions,
  about 1 % of the 0.07 to 0.11 drop of its variants. The blur mirrors at the grid's edge, which continues a guide
  meeting it at right angles exactly: keep features normal to the footprint
  edge there.
- `bias` may be traced (`fab(drawn, beta=beta, bias=b)`), for the derivative
  of an objective in the bias.

## Worst case

The worst of the three variants is a maximum, not smooth where two tie.

- With Adam (optax), use the smooth maximum $t \ln \sum_i e^{f_i / t}$,
  `t * jax.nn.logsumexp(f / t)`, as above. It lies above the maximum by at most
  $t \ln 3$ for three variants; $t = 0.01$ suits losses between 0.01 and 1.
- With a constrained optimizer (MMA, CCSAQ), use the epigraph form: add a
  variable $s$, minimize $s$ subject to $f_i(\rho) \le s$ for each variant, and
  start $s$ 5 % above the worst variant so that the start is feasible.

## Minimum width and spacing

A foundry's minimum width and spacing become two differentiable constraints on
the filtered density, each at most 0 when the drawn design's pattern keeps its
rule:

```python
rules = design.length_scale_constraints(blur, min_width=165e-9, min_spacing=165e-9)


def constraints(rho):
    return jnp.stack(rules(blur(rho)))  # [solid, void], each <= 0 when held
```

Run them in two stages:

1. Optimize without the constraints, raising $\beta$ up to 32, until the
   topology has formed.
2. Switch the constraints on at that $\beta$ (16 or more), never from a grey
   start: they measure the drawn design's lines and spaces, and a grey design
   has none yet.

A quadratic penalty with Adam is the plain path, its weight raised in steps:

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

With nlopt installed, its CCSAQ method takes the two values as inequality
constraints and holds them more closely. Scale the objective into about 1 to
100, where CCSAQ's steps behave:

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

- **What $g \le 0$ guarantees.** Straight lines and spaces of the drawn design
  are at least about 0.97 of their rule; compact features (a dot, a hole) are
  held stricter, to about 1.1 times it. After an optimization a few percent of
  a phase's area can still sit below the rule: constrain about 10 % above the
  foundry rule (165 nm here for a 150 nm rule), then confirm on the pattern
  with `LayerPattern.check` (see "Accept the design" below).
- **Range.** A rule above $1.25R$ or a pitch above $R/3$ is refused: beyond
  them the constraints let lines narrower than the rule pass. Raise the filter
  radius or refine the grid.
- The constraints act on the drawn design, the design intent; the fabrication
  model's printed designs go to the solves.
- On the single-filter path (`design.robust`), the filter radius alone can hold
  a minimum length $\ell$ through the eroded and dilated variants:
  `design.radius_for_length_scale(150e-9, delta=0.1)` gives that radius,
  $2.50\,\ell$ at $\delta = 0.1$.

## Real neighbours, symmetry and footprint

The filter, the constraints and the blur all treat the grid's edge as a mirror,
and the width and spacing check does not measure along it, so a feed guide
outside the grid is invisible to all four. Mesh the design region a margin
beyond the free design area, into the guides and the cladding, and pin the
margin's pixels to the real neighbours before the filter:

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

- The margin is at least $\max(R, 3\sigma, s_\text{min})$, with $s_\text{min}$
  the minimum spacing: the filter's kernel decays as $e^{-x/l}$ with
  $l = 0.29R$, so $R$ leaves 3 % of it; $3\sigma$ holds 99.7 % of the blur; and
  the check needs a rule's reach to see a sliver.
- The guide stubs inside the margin are part of the pattern, and the guides
  outside abut them on the footprint edge. The guide step of the printed
  variants moves to the margin's outer edge.
- Without a margin (an existing mesh), the edge stays a mirror: a guide meeting
  it at right angles is continued exactly, but a sliver along the edge is left
  to the assembled layout's checks.

## Accept the design

Check the printed variants against the rules: the width on the **eroded**
pattern, the spacing on the **dilated** one, each where its rule is tightest.
The fabrication model predicts them best from the exact drawn design, the area
fractions of the drawn pattern (edges to about 0.015 pixel, against about 0.2
pixel from a $\beta = 32$ projection):

```python
drawn_pattern = grid.pattern(blur(rho), eta=0.5)  # the drawn pattern, to GDS
drawn = grid.area_fraction(drawn_pattern)  # the drawn design, exact, as pixels
eroded = fab.pattern(drawn, bias=fab.bias - 10e-9)
dilated = fab.pattern(drawn, bias=fab.bias + 10e-9)
print(eroded.check(min_width=150e-9))
print(dilated.check(min_spacing=150e-9))
```

On the single-filter path the variants are contours of the filtered density at
the shifted thresholds: `grid.pattern(blur(rho), eta=0.5 + delta)` for the
width, `eta=0.5 - delta` for the spacing. Then export and re-solve the drawn
pattern: {doc}`fabricable-layout`.

## Next

{doc}`fabricable-layout`, {py:obj}`gfem.design.FabricationModel`,
{py:obj}`gfem.design.length_scale_constraints`, {py:obj}`gfem.design.robust`,
{py:obj}`gfem.design.helmholtz_filter`, {py:obj}`gfem.design.project`,
{doc}`s-parameter-gradient`, {doc}`sheet-design`, the
[topology tutorial](../tutorials/topology.ipynb).
