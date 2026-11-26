# Shape gradients of mode figures

Differentiate a port mode's group index and dispersion with respect to the
nodes of the port face. `PortModeResult.n_g` and `beta_2` have no backward
pass, but `gamma` has one, node coordinates included. A central difference in
frequency over the exact `gamma` rows of one `PortMode` gives $n_g$ and
$\beta_2$ that differentiate through everything `gamma` does, at one
eigenvector derivative per row and no extra solve.

The guide is a filled rectangular guide of index $N = 1.5$ with PEC walls, the
WR-90 section of 22.86 × 10.16 mm, at 8 GHz, just above the cutoff of its
TE10 mode. It has closed forms, so the gradients can be checked. Units are SI.

## The stencil

With $\gamma = \alpha + i\beta$ the propagation constant of a lossless mode,
$\beta = \mathrm{Im}\,\gamma$, and $n_g = c\,\mathrm{d}\beta/\mathrm{d}\omega$,
$\beta_2 = \mathrm{d}^2\beta/\mathrm{d}\omega^2$. Take rows at
$\omega(1 + k\,\delta)$:

$$
n_g \approx \frac{c\,(\beta_{+1} - \beta_{-1})}{2\,\omega\delta},
\qquad
\beta_2 \approx \frac{-\beta_{+2} + 16\beta_{+1} - 30\beta_0 + 16\beta_{-1} - \beta_{-2}}
{12\,(\omega\delta)^2}.
$$

Use $\delta = 10^{-3}$ for $n_g$ (three rows) and $3 \times 10^{-3}$ for
$\beta_2$ (five rows). The truncation is $O(\delta^2)$ for $n_g$ and
$O(\delta^4)$ for $\beta_2$, and the solve noise, about $10^{-14}$ of
$\beta$, is divided by $\delta$ or $\delta^2$; these two steps balance them
near the cutoff. The PML stretch, if any, stays fixed across the rows.

```python
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

import gfem

c = scipy.constants.c  # m/s
N, f0 = 1.5, 8e9

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/wr90_guide.msh"
urllib.request.urlretrieve(url, "wr90_guide.msh")
mesh = gfem.Mesh.from_gmsh("wr90_guide.msh", length_unit="mm")
options = gfem.SolverOptions(order=2)


def solve(nodes, frequency):
    model = gfem.Model(
        mesh.with_nodes(nodes),
        materials={"air": gfem.Material(eps_r=N**2)},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    return gfem.solve(model, gfem.PortMode("p1", frequency), options=options)


def stencil(nodes):
    """The group index and dispersion at f0, from rows of gamma."""
    d3, d5 = 1e-3, 3e-3
    beta = jnp.imag(solve(nodes, f0 * (1 + d3 * np.arange(-1, 2))).gamma)
    n_g = c * (beta[2] - beta[0]) / (2 * 2 * np.pi * f0 * d3)
    beta = jnp.imag(solve(nodes, f0 * (1 + d5 * np.arange(-2, 3))).gamma)
    w = 2 * np.pi * f0 * d5
    beta_2 = (-beta[4] + 16 * beta[3] - 30 * beta[2] + 16 * beta[1] - beta[0]) / (
        12 * w**2
    )
    return n_g, beta_2


nodes = jnp.asarray(mesh.nodes)
n_g, beta_2 = stencil(nodes)
mode = solve(nodes, [f0])
print(f"n_g: stencil {n_g:.6f}, reported {mode.n_g[0].real:.6f}")
print(f"beta_2: stencil {beta_2:.6e}, reported {mode.beta_2[0].real:.6e} s^2/m")
```

```text
n_g: stencil 1.791033, reported 1.791032
beta_2: stencil -5.059448e-20, reported -5.059448e-20 s^2/m
```

The stencil's $n_g$ differs from the reported one by $\delta^2$ times a
coefficient of order one, and its $\beta_2$ by about $10^{-9}$ of itself.

## The gradient with respect to the face

Move the nodes of the port face, $\mathbf x \to (1 + s)\,\mathbf x$ about the
face's centre, in its plane, and read $\mathrm{d}/\mathrm{d}s$ at $s = 0$.
Only the port face's nodes enter a port mode, so only they move. The
guide's width and height grow by the same factor, the cutoff falls as
$1/(1+s)$, and with $\kappa^2 = N^2 k_0^2 - \beta^2$ fixed by the cutoff,

$$
\frac{\mathrm{d}n_\mathrm{eff}}{\mathrm{d}s} = \frac{N^2 - n_\mathrm{eff}^2}{n_\mathrm{eff}},
\qquad
\frac{\mathrm{d}n_g}{\mathrm{d}s} = -\frac{N^2\,(N^2 - n_\mathrm{eff}^2)}{n_\mathrm{eff}^3}.
$$

```python
x = np.asarray(mesh.nodes)
face = np.isclose(x[:, 2], 0.0)  # the port face lies in z = 0
move = np.zeros_like(x)
move[face, :2] = x[face, :2] - x[face, :2].mean(axis=0)
move = jnp.asarray(move)

grad_n_g = jax.grad(lambda x: stencil(x)[0])(nodes)
grad_beta_2 = jax.grad(lambda x: stencil(x)[1])(nodes)
n_eff = mode.n_eff[0].real
closed = -(N**2) * (N**2 - n_eff**2) / n_eff**3
print(f"dn_g/ds: adjoint {jnp.sum(grad_n_g * move):.6f}, closed form {closed:.6f}")
print(f"dbeta_2/ds: adjoint {jnp.sum(grad_beta_2 * move):.4e} s^2/m")
```

```text
dn_g/ds: adjoint -0.762421, closed form -0.762419
dbeta_2/ds: adjoint 1.6580e-19 s^2/m
```

The gradient is the exact derivative of the stencil. It differs from the
derivative of the true $n_g$ by the stencil's own $O(\delta^2)$ truncation,
here a few parts in $10^{6}$.

## Limits

- The gradient with respect to node coordinates holds for the planar port of a
  3D mesh. A line port of a 2D mesh refuses node gradients.
- The frequencies of one `PortMode` must be a grid with equal steps; a
  mode that is not solved on a row, or switches branch across the stencil, spoils it.
- The step suits a guide a few percent above its cutoff or farther. Very
  near a cutoff $\beta$ is not smooth in $\omega$: reduce $\delta$ and
  compare the stencil with the reported value first.
- `n_g` and `beta_2` of the result, used directly in an objective, still
  raise under `jax.grad`; the error names this route.

## Next

- {doc}`waveguide-dispersion`: the reported $n_g$ and $\beta_2$ of a section.
- {doc}`port-mode-gradient`: the gradient of `gamma` and `n_eff`.
- {doc}`waveguide-nonlinearity`: the Kerr coefficient, which also differentiates through the nodes.
