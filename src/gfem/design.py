"""Design helpers: layer grids, density filters and projections.

A density design runs through a fixed chain of differentiable maps, then
`gfem.MaterialField.interpolate` turns the result into a material::

    grid = gfem.design.LayerGrid(mesh, "design", pitch=2e-6)
    blur = gfem.design.helmholtz_filter(grid, radius=20e-6)
    rho_bar = gfem.design.project(blur(rho), beta=8.0)
    field = gfem.MaterialField.interpolate(grid.to_elements(rho_bar), air, ceramic)

Every helper is pure JAX or a fixed linear map with an exact adjoint, so
`jax.grad`, `jax.jit` and `jax.vmap` work through all of them. The optimizer
is your own (optax, nlopt, scipy, ...).

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

import dataclasses
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from gfem import _morph, _mpi, _pattern, _sparse, _spectral
from gfem._checks import is_traced, static_float
from gfem._dim2 import refuse_entry
from gfem._fabrication import FabricationModel
from gfem._feature_check import PatternCheck, Violation
from gfem._gds import write_gds
from gfem._length_scale import (
    LengthScaleConstraints,
    LengthScaleValues,
    length_scale_constraints,
    radius_for_length_scale,
)
from gfem._materials import PixelGeometry
from gfem._pattern import LayerPattern, Polygon
from gfem._pytree import register_fields
from gfem.errors import UnsampledPixelsWarning
from gfem.mesh import Mesh

__all__ = [
    "FabricationModel",
    "HelmholtzFilter",
    "LayerDensity",
    "LayerGrid",
    "LayerPattern",
    "LengthScaleConstraints",
    "LengthScaleValues",
    "PatternCheck",
    "Polygon",
    "Robust",
    "Violation",
    "helmholtz_filter",
    "length_scale_constraints",
    "project",
    "radius_for_length_scale",
    "robust",
    "write_gds",
]


def project(rho: jax.Array, beta: float, eta: float = 0.5) -> jax.Array:
    """Smoothed Heaviside (tanh) projection of a density towards 0 / 1.

    ``(tanh(beta eta) + tanh(beta (rho - eta))) / (tanh(beta eta)
    + tanh(beta (1 - eta)))``: 0 stays 0, 1 stays 1, and values above the
    threshold ``eta`` move towards 1 (below it, towards 0). ``beta`` sets the
    sharpness: about linear at 1, nearly a step at 64. Returns values in
    [0, 1] for rho in [0, 1] (clamped against round-off).

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
    p = (a + jnp.tanh(beta * (jnp.asarray(rho) - eta))) / (a + b)
    # Round-off can leave p a few ulp outside [0, 1]. Nested where, not
    # jnp.clip: clip halves the gradient where p sits exactly on a bound.
    return jnp.where(p < 0, 0.0, jnp.where(p > 1, 1.0, p))


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
    gradient is its transpose. On a curved (second-order) mesh a pixel is
    sampled at the mean of the element's corner nodes, not at a point of its
    curved map: that is a convention of the design map, not a measurement.

    Pixels that no element centroid samples get
    no gradient: the constructor warns with their count. Pixels outside a
    non-rectangular footprint are such pixels; `sampled` masks them.

    On a 2D mesh (a cross-section in the xy plane), a solve samples the
    pixels at its own quadrature points instead of at centroids:
    `to_elements` returns a `LayerDensity`, which keeps the pixels, and
    coverage counts the points of the lowest-order rule, the edge midpoints
    of each triangle. Surface regions (lines) are not designable there.

    On a periodic cell, pass ``periodic`` for the axes along which the
    region spans one period: the grid's edges there are one seam, so
    `helmholtz_filter` smooths across it and the bilinear map wraps around
    it, and the design keeps its length scale across the seam::

        grid = gfem.design.LayerGrid(
            mesh, "design", pitch=25e-9, periodic=(True, False)
        )

    Attributes:
        region: The volume or surface region.
        pitch: Pixel side (m).
        periodic: Per axis ``(x, y)``, whether the grid wraps around a seam.
        shape: ``(ny, nx)``.
        origin: Lower-left corner of the grid ``(x, y)`` (m).
        bounds: The region's footprint, its xy bounding rectangle
            ``(xmin, ymin, xmax, ymax)`` (m). The grid overhangs it by up to a
            pixel when `pitch` does not divide it; on a periodic axis its
            edges are the grid's.
        x: Pixel-centre x coordinates, ``(nx,)`` (m).
        y: Pixel-centre y coordinates, ``(ny,)`` (m).
        sampled: ``(ny, nx)`` bool, pixels some element (face) centroid
            samples; on a 2D mesh, pixels some point of the lowest-order rule
            (an edge midpoint) samples.
        n_elements: On a volume, the number of its elements (on a tetrahedral
            mesh the length of `to_elements`' output, in
            ``mesh.elements(region)`` order); ``None`` on a surface.
        geometry: Where the pixels lie (region, origin, pitch, shape,
            periodic), what a solve needs to sample them.
        n_faces: On a surface, the number of its faces (the length of
            `to_faces`' output, in ``mesh.elements(region)`` order); ``None``
            on a volume.
    """

    def __init__(
        self,
        mesh: Mesh,
        region: str,
        *,
        pitch: float,
        periodic: tuple[bool, bool] = (False, False),
        comm: Any = None,
    ) -> None:
        """Lay a pixel grid over a volume or horizontal surface's footprint.

        Args:
            mesh: The mesh, with concrete node coordinates.
            region: A volume region, or a surface whose faces are all
                horizontal (normal along mesh z).
            pitch: Pixel side (m), > 0.
            periodic: Per axis ``(x, y)``, whether the footprint spans one
                period of a periodic cell along it, so its two edges are one
                seam. On a periodic axis `helmholtz_filter` smooths across the
                seam and the bilinear map wraps around it, and `pitch` must
                divide the footprint's length.
            comm: The communicator, as for `gfem.solve`: only rank 0 shows the
                warning. ``MPI.COMM_SELF`` under a launcher shows it on each rank.

        Raises:
            NotImplementedError: ``region`` is a line of a 2D mesh.
            RegionNotFoundError: The mesh has no such region.
            ValueError: ``pitch <= 0``, ``pitch`` does not divide the
                footprint along a periodic axis, the mesh nodes are computed inside
                `jax.grad`, `jax.jit` or `jax.vmap` (build the grid from the
                fixed mesh), or a face of the surface is not horizontal.

        Warns:
            gfem.UnsampledPixelsWarning: Some pixels are sampled by no element
                (face) centroid, or on a 2D mesh by no edge midpoint.
        """
        comm = _mpi.resolve_comm(comm)
        r = mesh.region(region)
        pitch = static_float(pitch, "LayerGrid.pitch", "The pixel grid is fixed.")
        if pitch <= 0:
            raise ValueError(f"LayerGrid: pitch must be > 0 m, got {pitch}.")
        periodic = _check_periodic(periodic)
        if is_traced(mesh.nodes):
            raise ValueError(
                "LayerGrid: the mesh nodes are computed inside jax.grad, jax.jit "
                "or jax.vmap; build the grid outside the transformed function, "
                "from a mesh with concrete nodes."
            )
        nodes = np.asarray(mesh.nodes)
        volume = r.dim == mesh.dim
        if volume:
            cells = mesh.volume_elements[mesh.elements(region)]
        elif mesh.dim == 2:
            raise NotImplementedError(
                f"LayerGrid: {region!r} is a line of a 2D mesh, and a sheet "
                "design (gfem.SheetField) is not available on a 2D mesh. Lay the "
                "grid over an area region."
            )
        else:
            cells = mesh.surface_elements[mesh.elements(region)]
            _check_horizontal(nodes, cells, region)
        points = _sampling_points(nodes, cells, mesh.dim)
        lo = nodes[cells.ravel()].min(axis=0)[:2]
        hi = nodes[cells.ravel()].max(axis=0)[:2]
        n = [
            max(1, math.ceil((h - l) / pitch - 1e-9))
            for l, h in zip(lo, hi, strict=True)
        ]
        for axis, name in enumerate("xy"):
            length = hi[axis] - lo[axis]
            if periodic[axis] and abs(length / pitch - n[axis]) > _PERIOD_FIT:
                fit = length / max(1, round(length / pitch))
                raise ValueError(
                    f"LayerGrid on {region!r}: the footprint is periodic along "
                    f"{name}, so the pitch must divide its length {length:g} m "
                    f"into whole pixels, got pitch={pitch:g} m "
                    f"({length / pitch:.4g} pixels). Use pitch={fit:.6g} m."
                )
        origin = (lo + hi) / 2 - np.array(n) * pitch / 2

        self.region = region
        self.pitch = pitch
        self.periodic = periodic
        self.shape = (n[1], n[0])
        self.origin = origin
        self.bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
        self.x = origin[0] + (np.arange(n[0]) + 0.5) * pitch
        self.y = origin[1] + (np.arange(n[1]) + 0.5) * pitch
        self.n_elements = len(cells) if volume else None
        self.n_faces = None if volume else len(cells)
        self._what = "element" if volume else "face"
        self._points = mesh.dim == 2
        self.geometry = PixelGeometry(
            region, tuple(origin), pitch, self.shape, periodic
        )
        self._index, self._weight = _bilinear(
            points, origin, pitch, self.shape, periodic
        )
        weight = np.bincount(
            self._index.ravel(), self._weight.ravel(), minlength=n[0] * n[1]
        )
        self.sampled = weight.reshape(self.shape) > 0
        for a in (self.origin, self.x, self.y, self.sampled, self._index, self._weight):
            a.flags.writeable = False
        dead = int((~self.sampled).sum())
        if dead:
            by = (
                "are reached by no quadrature point of the lowest-order rule"
                if self._points
                else f"are sampled by no {self._what} centroid"
            )
            _mpi.warn(
                comm,
                f"LayerGrid on {region!r}: {dead} of {self.sampled.size} pixels "
                f"{by}, so they get no gradient and "
                "do not reach the solve. Refine the mesh in the region, or coarsen "
                f"the pitch (now {pitch:g} m). Pixels outside a non-rectangular "
                "footprint are always unsampled: mask them with grid.sampled.",
                UnsampledPixelsWarning,
                stacklevel=2,
            )

    def to_elements(self, rho: jax.Array) -> jax.Array | LayerDensity:
        """The design of a pixel array on the region's elements.

        On a tetrahedral mesh, each element's value is the bilinear
        interpolation of the pixels at its centroid. On a 2D mesh, a solve
        samples the pixels at its own quadrature points, which depend on its
        order: the result is a `LayerDensity` holding the pixels.

        Args:
            rho: Pixel values, shape `shape`.

        Returns:
            The input of `gfem.MaterialField.interpolate`: on a tetrahedral
            mesh ``(n_elements,)`` values in ``mesh.elements(region)``
            order; on a 2D mesh a `LayerDensity`.

        Raises:
            ValueError: ``rho`` does not have the grid's shape, or the grid
                is on a surface (use `to_faces`).
        """
        if self.n_elements is None:
            raise ValueError(
                f"LayerGrid.to_elements: the grid is on the surface {self.region!r}; "
                "map a sheet design to its faces with grid.to_faces(rho)."
            )
        if self._points:
            return LayerDensity(self._pixels(rho, "LayerGrid.to_elements"), self)
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

    def pattern(self, rho: Any, *, eta: float = 0.5) -> LayerPattern:
        r"""The layer pattern: the polygons where the filtered density reaches ``eta``.

        Contours the **filtered** density (``blur(rho)``, before `project`)
        at the projection threshold, read through the grid's own bilinear
        map, the field a solve samples. The edge is the
        :math:`\beta \to \infty` limit of ``project(filtered, beta, eta)``,
        the drawn design, placed between pixel centres to a few hundredths
        of a pixel. A projected or binary design works too, but only to
        about half a pixel (its values saturate at the centres).
        Solid is ``rho >= eta``; a saddle cell joins its two solid corners
        when the mean of its four corners reaches ``eta``.

        Solid touching the footprint closes exactly along `bounds`, and a
        grid overhanging the footprint is clipped to it. A periodic axis
        gives one period, cut at the seam with matching vertices on both
        sides. `sampled` is ignored: set pixels outside a non-rectangular
        footprint void before the filter (``jnp.where(grid.sampled, rho,
        0.0)``). The same on a volume, a sheet and a 2D model. Not
        differentiable: call it on the final design, outside `jax.grad`.

        Args:
            rho: Filtered pixel densities, shape `shape` (on a 2D model the
                pixel array, not the `LayerDensity`).
            eta: Projection threshold, in (0, 1), as given to `project`;
                ``eta + delta`` and ``eta - delta`` give the eroded and
                dilated patterns of `robust`.

        Returns:
            The pattern, with this grid's `region`, `bounds` and `periodic`.

        Raises:
            TypeError: ``rho`` is a `LayerDensity` (pass ``density.pixels``).
            ValueError: ``rho`` has the wrong shape, holds NaN, or is
                computed inside `jax.grad`, `jax.jit` or `jax.vmap`; ``eta``
                outside (0, 1).
        """
        return _pattern.contour(
            self,
            rho,
            eta,
            lambda points: _bilinear(
                points, self.origin, self.pitch, self.shape, self.periodic
            ),
        )

    def area_fraction(self, pattern: LayerPattern) -> np.ndarray:
        """Each pixel's share of its area inside the footprint that the pattern covers.

        The exact area fraction (no point sampling): the pattern's edges
        are split at the pixel lines and each piece's area accumulated in its
        pixel. The denominator is the pixel's area inside `bounds`, so a fully solid
        pattern gives 1 in the pixels overhanging the footprint too, and
        polygons reaching past the footprint (guides drawn long) are clipped.
        Uses: re-solving a pattern (``grid.to_elements(grid.area_fraction(
        pattern))``), hand-drawn designs, and fixed neighbours pinned in a
        margin band of the grid::

            fixed = grid.area_fraction(design.LayerPattern.from_polygons(guides))
            rho = jnp.where(in_margin, fixed, rho)

        Args:
            pattern: A layer pattern in metres, from any grid or
                `LayerPattern.from_polygons`.

        Returns:
            ``(ny, nx)`` float64 values in [0, 1], a plain array (not
            differentiable in the pattern).

        Raises:
            TypeError: ``pattern`` is not a `LayerPattern`.
            ValueError: The pattern has polygons but none overlaps the
                footprint (lengths are in metres).
        """
        return _pattern.area_fraction(self, pattern)

    def _pixels(
        self, rho: Any, where: str, *, real: bool = False, remedy: str = ""
    ) -> jax.Array:
        """``rho`` as a pixel array of this grid, or raise naming ``where``.

        Args:
            rho: The pixel values.
            where: The caller, named in the messages.
            real: Whether to refuse complex values too.
            remedy: Appended to the shape message.
        """
        if isinstance(rho, LayerDensity):
            raise TypeError(
                f"{where}: rho is a LayerDensity; it acts on the pixel array "
                "before to_elements: pass density.pixels."
            )
        rho = jnp.asarray(rho)
        if rho.shape != self.shape:
            raise ValueError(
                f"{where}: rho has shape {rho.shape}, but the grid on "
                f"{self.region!r} has shape {self.shape} (ny, nx).{remedy}"
            )
        if real and jnp.iscomplexobj(rho):
            raise ValueError(f"{where}: densities must be real.")
        return rho

    def _interpolate(self, rho: jax.Array, name: str) -> jax.Array:
        rho = self._pixels(rho, f"LayerGrid.{name}")
        return jnp.sum(rho.reshape(-1)[self._index] * self._weight, axis=1)

    def __repr__(self) -> str:
        n = self.n_elements if self.n_faces is None else self.n_faces
        wrap = f", periodic={self.periodic}" if any(self.periodic) else ""
        return (
            f"LayerGrid({self.region!r}, pitch={self.pitch:g} m, shape={self.shape}"
            f"{wrap}, {n} {self._what}s)"
        )


jax.tree_util.register_static(LayerGrid)


#: Why a layer density has no array value.
_NO_ARRAY = (
    "a layer density on a 2D model is sampled at the solve's quadrature points, "
    "so it has no value per element; transform the pixel array before "
    "to_elements, and plot density.pixels on grid.x, grid.y"
)


@dataclasses.dataclass(frozen=True, eq=False)
class LayerDensity:
    """A pixel design bound to its layer grid, on a 2D mesh.

    What `LayerGrid.to_elements` returns on a 2D mesh, the input of
    `gfem.MaterialField.interpolate`: a solve samples the pixels at its own
    quadrature points. A JAX pytree whose leaf is `pixels`. It has no
    arithmetic and no array value (``numpy.asarray`` raises `TypeError`):
    filter, project and combine the pixel array before `LayerGrid.to_elements`.

    Attributes:
        pixels: The pixel values, shape ``grid.shape``.
        grid: The layer grid.

    Raises:
        ValueError: ``pixels`` does not have the grid's shape.
    """

    pixels: jax.Array
    grid: LayerGrid

    def __post_init__(self) -> None:
        if np.shape(self.pixels) != self.grid.shape:
            raise ValueError(
                f"LayerDensity: pixels have shape {np.shape(self.pixels)}, but the "
                f"grid on {self.grid.region!r} has shape {self.grid.shape} (ny, nx)."
            )

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Raise: a layer density has no value per element."""
        raise TypeError(f"LayerDensity: {_NO_ARRAY}.")

    def __repr__(self) -> str:
        return f"LayerDensity({self.grid!r})"


register_fields(LayerDensity, ("pixels",))


def _sampling_points(nodes: np.ndarray, cells: np.ndarray, dim: int) -> np.ndarray:
    """The (x, y) points at which a grid's coverage is counted.

    The cells' centroids on a tetrahedral mesh; on a 2D mesh the points of
    the lowest-order rule a solve samples at, the edge midpoints (the
    degree-2 triangle rule).
    """
    if dim == 2:
        ends = nodes[cells[:, [1, 2, 0]]]
        return ((nodes[cells] + ends) / 2).reshape(-1, 2)
    return nodes[cells].mean(axis=1)[:, :2]


#: Pixels a periodic footprint's length may miss a whole number by (round-off of
#: mesh coordinates).
_PERIOD_FIT = 1e-6


def _check_periodic(periodic: Any) -> tuple[bool, bool]:
    """``periodic`` as a pair of bools, or raise."""
    try:
        ok = len(periodic) == 2 and all(
            isinstance(p, bool | np.bool_) for p in periodic
        )
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(
            "LayerGrid: periodic takes one bool per axis (x, y), e.g. "
            f"periodic=(True, False), got {periodic!r}."
        )
    return (bool(periodic[0]), bool(periodic[1]))


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
    points: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    shape: tuple[int, int],
    periodic: tuple[bool, bool] = (False, False),
) -> tuple[np.ndarray, np.ndarray]:
    """Flat pixel indices and weights ``(n, 4)`` of bilinear interpolation.

    Points beyond the outermost pixel centres take the edge pixels' values,
    except along a periodic axis, where they interpolate between the last
    and the first pixel (across the seam).
    """
    corners = []
    for axis, size in ((0, shape[1]), (1, shape[0])):
        f = (points[:, axis] - origin[axis]) / pitch - 0.5
        if periodic[axis]:
            i0 = np.floor(f).astype(np.int64)
            t = f - i0
            corners.append((i0 % size, (i0 + 1) % size, t))
            continue
        f = np.clip(f, 0, size - 1)
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

#: PDE length per filter radius: the 1D hat filter of radius :math:`R` and the 1D
#: Helmholtz filter of length :math:`R/(2\sqrt3)` have the same second moment,
#: :math:`R^2/6` (the convention of Lazarov and Sigmund 2011). In 2D the Helmholtz
#: filter is a little wider than a cone of the same radius:
#: :math:`\langle r^2\rangle = R^2/3` against :math:`0.3R^2`.
PDE_LENGTH_PER_RADIUS = 1 / (2 * math.sqrt(3))


class HelmholtzFilter:
    """A density filter: a fixed, self-adjoint smoothing map.

    Call it on a density (``blur(rho)``); `helmholtz_filter` builds it.

    Attributes:
        radius: The filter radius (m).
        shape: The shape of the densities it takes and returns.
    """

    def __init__(
        self,
        radius: float,
        shape: tuple[int, ...],
        apply: Callable,
        where: str,
        grid: LayerGrid | None = None,
    ) -> None:
        """Wrap a linear map; use `helmholtz_filter` to build one."""
        self.radius = radius
        self.shape = shape
        self._apply = apply
        self._where = where
        # The pixel grid it filters on (None on a free-form mesh): the
        # length-scale constraints read its pitch, seams and sampled pixels.
        self._grid = grid

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
    r"""A Helmholtz (PDE) density filter of a given radius.

    The filtered density :math:`u` solves :math:`-r^2 \nabla^2 u + u = \rho`
    with zero-flux (Neumann) edges, :math:`r` = ``radius`` /
    :math:`(2\sqrt{3})`: ``radius`` is that of the 1D hat (linear) filter
    with the same second moment; in 2D the Helmholtz filter is slightly wider
    than a cone filter of that radius. Features much smaller than ``radius``
    are smoothed away. Constants pass unchanged and the (volume-weighted)
    total density is kept.

    - On a `LayerGrid`, the filter acts on the pixel array (a 2D screened
      Poisson solve with second differences, by cosine transforms). Along a
      periodic axis of the grid (``LayerGrid(..., periodic=...)``) the edges
      are one seam instead of walls: the solve wraps around it (by a Fourier
      transform), so the length scale holds across the seam.
    - On ``(mesh, region)``, it acts on one density per element of the
      volume region (free-form 3D designs): P1 finite elements on the
      region, a P0 load, and element averages back.

    Both are linear and symmetric, so the gradient is the same filter. On a
    curved (second-order) mesh the filter works on the straight chords of the
    corner nodes: the radius is measured on that polyhedral geometry, a
    convention of the filter, not of the curved elements.

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
            radius,
            carrier.shape,
            _grid_filter(carrier, length),
            repr(carrier),
            grid=carrier,
        )
    refuse_entry(carrier, "helmholtz_filter")
    if region is None:
        raise ValueError(
            "helmholtz_filter: name the design's volume region, "
            "helmholtz_filter(mesh, region, radius=...)."
        )
    if carrier.region(region).dim != carrier.dim:
        raise ValueError(
            f"helmholtz_filter: {region!r} is a surface; filter a layer design "
            "on its pixel grid instead: helmholtz_filter(LayerGrid(mesh, region, "
            "pitch=...), radius=...)."
        )
    apply, n = _mesh_filter(carrier, region, length)
    return HelmholtzFilter(radius, (n,), apply, f"region {region!r}")


def _grid_filter(grid: LayerGrid, length: float) -> Callable:
    # The screened-Poisson solve is a gain per mode of the second differences.
    return _spectral.function_of_laplacian(
        grid.shape, grid.pitch, grid.periodic, lambda k: 1 / (1 + length**2 * k)
    )


def _mesh_filter(mesh: Mesh, region: str, length: float) -> tuple[Callable, int]:
    from gfem._linalg import factorize

    if is_traced(mesh.nodes):
        raise ValueError(
            "helmholtz_filter: the mesh nodes are computed inside jax.grad, "
            "jax.jit or jax.vmap; build the filter outside the transformed "
            "function, from a mesh with concrete nodes."
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
