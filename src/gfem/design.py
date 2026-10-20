"""Design helpers: layer grids, density filters and projections.

A density design runs through a fixed chain of differentiable maps, then
`gfem.MaterialField.interpolate` turns the result into a material::

    grid = gfem.design.LayerGrid(mesh, "design", pitch=2e-6)
    blur = gfem.design.helmholtz_filter(grid, radius=20e-6)
    rho_bar = gfem.design.project(blur(rho), beta=8.0)
    field = gfem.MaterialField.interpolate(grid.to_elements(rho_bar), air, ceramic)

Every helper is pure JAX or a fixed linear map with an exact adjoint, so
`jax.grad`, `jax.jit` and `jax.vmap` work through all of them. The optimizer
is the user's own (optax, nlopt, scipy, ...).

- The **filter** (`helmholtz_filter`) sets a minimum length scale: it smooths
  the density by a screened-Poisson (Helmholtz PDE) solve.
- The **projection** (`project`) pushes the filtered density towards 0 / 1;
  raise ``beta`` gradually during the optimization (continuation).
- The **robust triple** (`robust`) projects at three thresholds, giving the
  eroded, nominal and dilated designs of an etch tolerance. Optimizing the
  worst of the three guarantees the length scale on both phases.
- A **layer grid** (`LayerGrid`) holds a planar pixel design, constant
  through a volume's thickness along mesh z, as planar fabrication makes it,
  or the pattern of a horizontal sheet (`gfem.SheetField`, ``to_faces``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.scipy import fft

from gfem import _morph, _sparse
from gfem._checks import is_traced, static_float
from gfem.mesh import Mesh

__all__ = [
    "HelmholtzFilter",
    "LayerGrid",
    "Robust",
    "helmholtz_filter",
    "project",
    "robust",
]


def project(rho: jax.Array, beta: float, eta: float = 0.5) -> jax.Array:
    """Smoothed Heaviside (tanh) projection of a density towards 0 / 1.

    ``(tanh(beta eta) + tanh(beta (rho - eta))) / (tanh(beta eta)
    + tanh(beta (1 - eta)))``: 0 stays 0, 1 stays 1, and values above the
    threshold ``eta`` move towards 1 (below it, towards 0). ``beta`` sets the
    sharpness: about linear at 1, nearly a step at 64.

    Args:
        rho: Densities in [0, 1], any shape.
        beta: Sharpness, > 0. Usually raised in steps (1, 2, 4, ...) as the
            optimization converges (continuation).
        eta: Threshold in (0, 1).

    Raises:
        ValueError: ``beta <= 0`` or ``eta`` outside (0, 1).
    """
    _check_projection(beta, eta)
    a = jnp.tanh(beta * eta)
    b = jnp.tanh(beta * (1 - eta))
    return (a + jnp.tanh(beta * (jnp.asarray(rho) - eta))) / (a + b)


def _check_projection(beta: float, eta: float) -> None:
    if not is_traced(beta) and not float(beta) > 0:
        raise ValueError(f"project: beta must be > 0, got {beta}.")
    if not is_traced(eta) and not 0 < float(eta) < 1:
        raise ValueError(f"project: the threshold eta must be in (0, 1), got {eta}.")


class Robust(NamedTuple):
    """The eroded, nominal and dilated projections of one filtered density.

    A named tuple: ``jnp.stack(robust(...))`` gives a ``(3, ...)`` batch to
    `jax.vmap` a solve over.

    Attributes:
        eroded: Projected at ``eta + delta``: less solid (over-etched).
        nominal: Projected at ``eta``.
        dilated: Projected at ``eta - delta``: more solid (under-etched).
    """

    eroded: jax.Array
    nominal: jax.Array
    dilated: jax.Array


def robust(rho: jax.Array, *, beta: float, delta: float, eta: float = 0.5) -> Robust:
    """Project a filtered density at three thresholds (robust formulation).

    The eroded design is the nominal one after an over-etch, the dilated one
    after an under-etch. Optimizing the worst of the three objectives (and
    putting a volume constraint on the dilated design) makes the result
    tolerate the etch and guarantees a minimum length scale on both the
    solid and the void phase.

    Args:
        rho: Filtered densities in [0, 1] (the projection of an unfiltered
            density has no length scale).
        beta: Projection sharpness, > 0.
        delta: Threshold shift, > 0; it sets the etch tolerance, so it has no
            default. ``eta - delta`` and ``eta + delta`` must lie in (0, 1).
        eta: Nominal threshold.

    Returns:
        ``Robust(eroded, nominal, dilated)`` at ``eta + delta``, ``eta`` and
        ``eta - delta``.

    Raises:
        ValueError: ``delta <= 0``, or a threshold outside (0, 1).
    """
    if not is_traced(delta) and not is_traced(eta):
        d, e = float(delta), float(eta)
        if not d > 0 or not 0 < e - d < e + d < 1:
            raise ValueError(
                f"robust: delta must be > 0 with eta - delta and eta + delta in "
                f"(0, 1), got eta={e}, delta={d}."
            )
    return Robust(
        eroded=project(rho, beta, eta + delta),
        nominal=project(rho, beta, eta),
        dilated=project(rho, beta, eta - delta),
    )


# --- layer grid ---------------------------------------------------------------------


class LayerGrid:
    """A planar pixel grid of design values over a region's footprint.

    The grid covers the region's xy extent (mesh z is the wafer normal) with
    square pixels of side `pitch`, centred on the footprint. A design array
    ``rho`` has `shape` ``(ny, nx)``: row ``i`` is at ``y[i]``, column ``j``
    at ``x[j]``. On a volume region, `to_elements` gives each element the
    bilinear interpolation of the pixel values at its centroid's (x, y), so
    the design is constant through the region's thickness. Filter and
    project on the grid (`helmholtz_filter`, `project`), then map to the
    elements::

        grid = gfem.design.LayerGrid(mesh, "design", pitch=2e-6)
        blur = gfem.design.helmholtz_filter(grid, radius=20e-6)
        density = grid.to_elements(gfem.design.project(blur(rho), beta=8.0))

    On a horizontal surface region (a sheet design: every face normal along
    mesh z), `to_faces` does the same at each face centroid, the input of
    `gfem.SheetField.interpolate`::

        grid = gfem.design.LayerGrid(mesh, "M1_design", pitch=2e-6)
        fill = gfem.SheetField.interpolate(grid.to_faces(rho_bar))

    The map is linear and fixed (built from the mesh as given), so its
    gradient is its transpose. Pixels that no element centroid samples get
    no gradient: the constructor warns with their count. Pixels outside a
    non-rectangular footprint are such pixels; `sampled` masks them.

    Attributes:
        region: The volume or surface region.
        pitch: Pixel side (m).
        shape: ``(ny, nx)``.
        origin: Lower-left corner of the grid ``(x, y)`` (m).
        x: Pixel-centre x coordinates, ``(nx,)`` (m).
        y: Pixel-centre y coordinates, ``(ny,)`` (m).
        sampled: ``(ny, nx)`` bool, pixels some element (face) centroid
            samples.
        n_elements: On a volume, the number of its elements (the length of
            `to_elements`' output, in ``mesh.elements(region)`` order);
            ``None`` on a surface.
        n_faces: On a surface, the number of its faces (the length of
            `to_faces`' output, in ``mesh.elements(region)`` order); ``None``
            on a volume.
    """

    def __init__(self, mesh: Mesh, region: str, *, pitch: float) -> None:
        """Lay a pixel grid over a volume or horizontal surface's footprint.

        Args:
            mesh: The mesh, with concrete node coordinates.
            region: A volume region, or a surface whose faces are all
                horizontal (normal along mesh z).
            pitch: Pixel side (m), > 0.

        Raises:
            RegionNotFoundError: The mesh has no such region.
            ValueError: ``pitch <= 0``, the mesh nodes are traced, or a
                face of the surface is not horizontal.

        Warns:
            UserWarning: Some pixels are sampled by no element (face)
                centroid.
        """
        r = mesh.region(region)
        pitch = static_float(pitch, "LayerGrid.pitch", "The pixel grid is fixed.")
        if pitch <= 0:
            raise ValueError(f"LayerGrid: pitch must be > 0 m, got {pitch}.")
        if is_traced(mesh.nodes):
            raise ValueError(
                "LayerGrid: the mesh nodes are traced by JAX; build the grid "
                "outside jax.jit / jax.grad, from a mesh with concrete nodes."
            )
        nodes = np.asarray(mesh.nodes)
        if r.dim == 3:
            cells = mesh.tetrahedra[mesh.elements(region)]
        else:
            cells = mesh.triangles[mesh.elements(region)]
            _check_horizontal(nodes, cells, region)
        points = nodes[cells].mean(axis=1)[:, :2]
        lo = nodes[cells.ravel()].min(axis=0)[:2]
        hi = nodes[cells.ravel()].max(axis=0)[:2]
        n = [
            max(1, math.ceil((h - l) / pitch - 1e-9))
            for l, h in zip(lo, hi, strict=True)
        ]
        origin = (lo + hi) / 2 - np.array(n) * pitch / 2

        self.region = region
        self.pitch = pitch
        self.shape = (n[1], n[0])
        self.origin = origin
        self.x = origin[0] + (np.arange(n[0]) + 0.5) * pitch
        self.y = origin[1] + (np.arange(n[1]) + 0.5) * pitch
        self.n_elements = len(cells) if r.dim == 3 else None
        self.n_faces = len(cells) if r.dim == 2 else None
        self._what = "element" if r.dim == 3 else "face"
        self._index, self._weight = _bilinear(points, origin, pitch, self.shape)
        weight = np.bincount(
            self._index.ravel(), self._weight.ravel(), minlength=n[0] * n[1]
        )
        self.sampled = weight.reshape(self.shape) > 0
        for a in (self.origin, self.x, self.y, self.sampled, self._index, self._weight):
            a.flags.writeable = False
        dead = int((~self.sampled).sum())
        if dead:
            warnings.warn(
                f"LayerGrid on {region!r}: {dead} of {self.sampled.size} pixels "
                f"are sampled by no {self._what} centroid, so they get no gradient and "
                "do not reach the solve. Refine the mesh in the region, or coarsen "
                f"the pitch (now {pitch:g} m). Pixels outside a non-rectangular "
                "footprint are always unsampled: mask them with grid.sampled.",
                UserWarning,
                stacklevel=2,
            )

    def to_elements(self, rho: jax.Array) -> jax.Array:
        """Per-element values of a pixel array: bilinear at element centroids.

        Args:
            rho: Pixel values, shape `shape`.

        Returns:
            ``(n_elements,)`` values in ``mesh.elements(region)`` order, the
            input of `gfem.MaterialField.interpolate`.

        Raises:
            ValueError: ``rho`` does not have the grid's shape, or the grid
                is on a surface (use `to_faces`).
        """
        if self.n_elements is None:
            raise ValueError(
                f"LayerGrid.to_elements: the grid is on the surface {self.region!r}; "
                "map a sheet design to its faces with grid.to_faces(rho)."
            )
        return self._interpolate(rho, "to_elements")

    def to_faces(self, rho: jax.Array) -> jax.Array:
        """Per-face values of a pixel array: bilinear at face centroids.

        Args:
            rho: Pixel values, shape `shape`.

        Returns:
            ``(n_faces,)`` values in ``mesh.elements(region)`` order, the
            input of `gfem.SheetField.interpolate`.

        Raises:
            ValueError: ``rho`` does not have the grid's shape, or the grid
                is on a volume (use `to_elements`).
        """
        if self.n_faces is None:
            raise ValueError(
                f"LayerGrid.to_faces: the grid is on the volume {self.region!r}; "
                "map a layer design to its elements with grid.to_elements(rho)."
            )
        return self._interpolate(rho, "to_faces")

    def _interpolate(self, rho: jax.Array, name: str) -> jax.Array:
        rho = jnp.asarray(rho)
        if rho.shape != self.shape:
            raise ValueError(
                f"LayerGrid.{name}: rho has shape {rho.shape}, but the grid "
                f"on {self.region!r} has shape {self.shape} (ny, nx)."
            )
        return jnp.sum(rho.reshape(-1)[self._index] * self._weight, axis=1)

    def __repr__(self) -> str:
        n = self.n_elements if self.n_faces is None else self.n_faces
        return (
            f"LayerGrid({self.region!r}, pitch={self.pitch:g} m, shape={self.shape}, "
            f"{n} {self._what}s)"
        )


jax.tree_util.register_static(LayerGrid)


#: |n_z| below 1 - this of a face's unit normal: not horizontal.
_HORIZONTAL = 1e-9


def _check_horizontal(nodes: np.ndarray, triangles: np.ndarray, region: str) -> None:
    """Raise unless every face's normal is along mesh z."""
    p = nodes[triangles]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    nz = np.abs(normal[:, 2]) / np.linalg.norm(normal, axis=1)
    bad = np.flatnonzero(nz < 1 - _HORIZONTAL)
    if len(bad):
        first = ", ".join(str(int(i)) for i in bad[:5])
        raise ValueError(
            f"LayerGrid: the surface {region!r} is not horizontal: {len(bad)} of "
            f"{len(triangles)} faces have a normal off mesh z (faces {first}"
            f"{', ...' if len(bad) > 5 else ''} in mesh.elements({region!r}) "
            "order). A layer grid patterns a horizontal sheet only; vertical "
            "sidewalls are not layer-designable. Give the design its own "
            "horizontal surface region, or pass per-face densities to "
            "gfem.SheetField.interpolate directly."
        )


def _bilinear(
    points: np.ndarray, origin: np.ndarray, pitch: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Flat pixel indices and weights ``(n, 4)`` of bilinear interpolation.

    Points beyond the outermost pixel centres take the edge pixels' values.
    """
    corners = []
    for axis, size in ((0, shape[1]), (1, shape[0])):
        f = np.clip((points[:, axis] - origin[axis]) / pitch - 0.5, 0, size - 1)
        i0 = np.minimum(np.floor(f).astype(np.int64), max(size - 2, 0))
        t = f - i0
        i1 = np.minimum(i0 + 1, size - 1)
        corners.append((i0, i1, t))
    (ix0, ix1, tx), (iy0, iy1, ty) = corners
    nx = shape[1]
    index = np.stack(
        [iy0 * nx + ix0, iy0 * nx + ix1, iy1 * nx + ix0, iy1 * nx + ix1], axis=1
    )
    weight = np.stack(
        [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1
    )
    return index, weight


# --- Helmholtz filter ---------------------------------------------------------------

#: PDE length per filter radius: a cone (linear hat) filter of radius R and the
#: Helmholtz filter of length R / (2 sqrt(3)) have the same second moment.
PDE_LENGTH_PER_RADIUS = 1 / (2 * math.sqrt(3))


class HelmholtzFilter:
    """A density filter: a fixed, self-adjoint smoothing map.

    Call it on a density (``blur(rho)``); `helmholtz_filter` builds it.

    Attributes:
        radius: The filter radius (m).
        shape: The shape of the densities it takes and returns.
    """

    def __init__(
        self, radius: float, shape: tuple[int, ...], apply: Callable, where: str
    ) -> None:
        """Wrap a linear map; use `helmholtz_filter` to build one."""
        self.radius = radius
        self.shape = shape
        self._apply = apply
        self._where = where

    def __call__(self, rho: jax.Array) -> jax.Array:
        """The filtered density, same shape as ``rho``.

        Raises:
            ValueError: ``rho`` has the wrong shape or is complex.
        """
        rho = jnp.asarray(rho)
        if rho.shape != self.shape:
            raise ValueError(
                f"helmholtz_filter: rho has shape {rho.shape}, but the filter on "
                f"{self._where} takes shape {self.shape}."
            )
        if jnp.iscomplexobj(rho):
            raise ValueError("helmholtz_filter: densities must be real.")
        return self._apply(rho.astype(jnp.float64))

    def __repr__(self) -> str:
        return f"HelmholtzFilter({self._where}, radius={self.radius:g} m)"


jax.tree_util.register_static(HelmholtzFilter)


def helmholtz_filter(
    carrier: Mesh | LayerGrid, region: str | None = None, *, radius: float
) -> HelmholtzFilter:
    """A Helmholtz (PDE) density filter of a given radius.

    The filtered density ``u`` solves ``-r^2 lap(u) + u = rho`` with
    zero-flux (Neumann) edges, ``r = radius / (2 sqrt(3))``: the filter has
    the second moment of a cone filter of radius ``radius``, and features
    much smaller than ``radius`` are smoothed away. Constants pass unchanged
    and the (volume-weighted) total density is kept.

    - On a `LayerGrid`, the filter acts on the pixel array (a 2D screened
      Poisson solve with second differences, by cosine transforms).
    - On ``(mesh, region)``, it acts on one density per element of the
      volume region (free-form 3D designs): P1 finite elements on the
      region, a P0 load, and element averages back.

    Both are linear and symmetric, so the gradient is the same filter.

    Args:
        carrier: A `LayerGrid`, or the mesh of a free-form design.
        region: With a mesh, the design's volume region; ``None`` with a
            grid (it knows its region).
        radius: Filter radius (m), > 0.

    Raises:
        ValueError: A bad radius, a missing or extra ``region``, or a surface
            region (use a `LayerGrid`).
    """
    radius = static_float(radius, "helmholtz_filter radius", "The filter is fixed.")
    if radius <= 0:
        raise ValueError(f"helmholtz_filter: radius must be > 0 m, got {radius}.")
    length = radius * PDE_LENGTH_PER_RADIUS
    if isinstance(carrier, LayerGrid):
        if region is not None:
            raise ValueError(
                "helmholtz_filter: a LayerGrid already knows its region; call "
                "helmholtz_filter(grid, radius=...)."
            )
        return HelmholtzFilter(
            radius, carrier.shape, _grid_filter(carrier, length), repr(carrier)
        )
    if region is None:
        raise ValueError(
            "helmholtz_filter: name the design's volume region, "
            "helmholtz_filter(mesh, region, radius=...)."
        )
    if carrier.region(region).dim != 3:
        raise ValueError(
            f"helmholtz_filter: {region!r} is a surface; filter a layer design "
            "on its pixel grid instead: helmholtz_filter(LayerGrid(mesh, region, "
            "pitch=...), radius=...)."
        )
    apply, n = _mesh_filter(carrier, region, length)
    return HelmholtzFilter(radius, (n,), apply, f"region {region!r}")


def _grid_filter(grid: LayerGrid, length: float) -> Callable:
    # The second-difference Laplacian with reflecting (Neumann) edges is
    # diagonal in the orthonormal type-II cosine transform.
    h = grid.pitch
    ky, kx = ((2 / h * np.sin(np.pi * np.arange(n) / (2 * n))) ** 2 for n in grid.shape)
    gain = jnp.asarray(1 / (1 + length**2 * (ky[:, None] + kx[None, :])))

    def apply(rho: jax.Array) -> jax.Array:
        return fft.idctn(gain * fft.dctn(rho, type=2, norm="ortho"), norm="ortho")

    return apply


def _mesh_filter(mesh: Mesh, region: str, length: float) -> tuple[Callable, int]:
    from gfem._linalg import factorize

    if is_traced(mesh.nodes):
        raise ValueError(
            "helmholtz_filter: the mesh nodes are traced by JAX; build the filter "
            "outside jax.jit / jax.grad, from a mesh with concrete nodes."
        )
    tets = mesh.tetrahedra[mesh.elements(region)]
    used, local = np.unique(tets, return_inverse=True)
    local = local.reshape(tets.shape)
    g, vol = _morph.p1_gradients(np.asarray(mesh.nodes)[used], local)
    vol = vol / vol.mean()  # u = K^-1 b is invariant under a common scale
    stiff = vol[:, None, None] * np.einsum("eik,ejk->eij", g, g)
    mass = vol[:, None, None] / 20 * (np.ones((4, 4)) + np.eye(4))
    rows = np.repeat(local, 4, axis=1).ravel()
    cols = np.tile(local, (1, 4)).ravel()
    n_nodes, n_el = len(used), len(tets)
    matrix = sp.coo_array(
        ((length**2 * stiff + mass).ravel(), (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
    el = np.repeat(np.arange(n_el), 4)
    # P0 density -> P1 load (b_i = sum over elements at i of vol rho / 4), and
    # P1 -> element averages: transposes of each other up to the volumes.
    load = _sparse.coo(
        sp.coo_array((np.repeat(vol / 4, 4), (local.ravel(), el)), (n_nodes, n_el))
    )
    average = _sparse.coo(
        sp.coo_array((np.full(4 * n_el, 0.25), (el, local.ravel())), (n_el, n_nodes))
    )
    operator = _sparse.coo(matrix)
    factor = factorize(matrix)

    def apply(rho: jax.Array) -> jax.Array:
        return average.matvec(_sparse.spd_solve(operator, factor, load.matvec(rho)))

    return apply, n_el
