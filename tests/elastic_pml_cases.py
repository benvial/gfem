r"""Leaky columns with a closed form: the cases of the elastic perfectly matched layer.

A silicon-like layer (shear speed :math:`c_1`, :math:`h = 1\,\mu\mathrm{m}` thick)
on a substrate that has the same wave speeds and a hundredth of the density
(impedance ratio :math:`\zeta = 0.01`), ended by a perfectly matched layer. A
shear wave standing in the layer leaks into the substrate with the exact
complex frequencies

.. math::

    \omega_n = \frac{c_1}{h}\,\big[n\pi + i\,\mathrm{artanh}\,\zeta\big],

so :math:`Q = n\pi / (2\,\mathrm{artanh}\,\zeta) = 157.0743965521` for
:math:`n = 1` (`leaky_roots`). The substrate is half a wavelength deep
(:math:`a = \lambda/2`) and the layer one quasi-longitudinal wavelength thick
(:math:`d = \lambda_P`), with :math:`R_0 = 10^{-8}`.

`leaky_column` meshes it: a 2D section (antiplane mode :math:`u_z(y)` with
rollers on the sides, optionally with layers on the sides too) or a 3D column
on a Kuhn grid with the stretch along z (a thickness-shear mode
:math:`u_x(z)`, :math:`u_z` fixed on the x faces). Lengths are in micrometres.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import structured_meshes as sm
import thermal_cases as tc

import gfem

#: The layer's silicon-like constants and the matched substrate's.
MU_LAYER, NU, RHO_LAYER = 79.6e9, 0.25, 2329.0
#: The substrate: a hundredth of the density and the modulus, the same speeds.
MU_SUB, RHO_SUB = 0.796e9, 23.29
#: The layer's thickness, um.
H = 1.0
UM = 1e-6
#: The target reflection of the layers.
R0 = 1e-8


def _speeds(mu: float, rho: float) -> tuple[float, float]:
    lam = 2 * mu * NU / (1 - 2 * NU)
    return float(np.sqrt(mu / rho)), float(np.sqrt((lam + 2 * mu) / rho))


C_T, C_L = _speeds(MU_LAYER, RHO_LAYER)
#: The impedance ratio substrate / layer (both shear waves).
ZETA = RHO_SUB * _speeds(MU_SUB, RHO_SUB)[0] / (RHO_LAYER * C_T)


def leaky_roots(n: int, zeta: float = ZETA) -> complex:
    r"""The root :math:`\omega_n = (c_1/h)[n\pi + i\,\mathrm{artanh}\,\zeta]`, rad/s."""
    return C_T / (H * UM) * (n * np.pi + 1j * np.arctanh(zeta))


def _material(mu: float, rho: float, quality: float | None) -> gfem.Material:
    return gfem.Material(
        stiffness=gfem.isotropic_stiffness(2 * mu * (1 + NU), NU),
        mass_density=rho,
        mechanical_quality_factor=quality,
    )


@dataclasses.dataclass(frozen=True)
class Column:
    """A leaky column and what its solve needs.

    Attributes:
        model: The model.
        pml: The layer, as `gfem.PML`.
        omega: The exact complex frequency of the first root, rad/s.
        clamped: The rollers of the mode.
        thickness: The layer's thickness ``d``, um.
    """

    model: gfem.Model
    pml: gfem.PML
    omega: complex
    clamped: dict[str, str]
    thickness: float

    @property
    def target(self) -> float:
        """The real frequency of the root, Hz."""
        return float(self.omega.real / (2 * np.pi))

    @property
    def quality(self) -> float:
        """The exact quality factor of the root."""
        return float(self.omega.real / (2 * self.omega.imag))


def _depths() -> tuple[float, float]:
    """The substrate's depth (half a shear wavelength), the layer's (one P wave), um."""
    omega = leaky_roots(1)
    shear = 2 * np.pi * C_T / omega.real / UM
    return 0.5 * shear, 2 * np.pi * C_L / omega.real / UM


def _section(path: Path, maxh: float, side: float) -> gfem.Mesh:
    a, d = _depths()
    width = 0.6
    columns = [(0.0, width, "")]
    if side:
        columns = [(-side, 0.0, "_l"), *columns, (width, width + side, "_r")]
    bands = [(-a - d, -a, "pml"), (-a, 0.0, "sub"), (0.0, H, "layer")]
    rects = [
        (x0, y0, x1, y1, name + tag, None)
        for x0, x1, tag in columns
        for y0, y1, name in bands
    ]
    left, right = columns[0][0], columns[-1][1]
    edges: list[tc.Namer] = [
        (lambda p: abs(p[0] - left) < 1e-9, "out_left"),
        (lambda p: abs(p[0] - right) < 1e-9, "out_right"),
        (lambda p: abs(p[1] + a + d) < 1e-9, "out_bottom"),
    ]
    nodes, tris, lines = tc._mesh2d(rects, maxh, edges)
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _body(path: Path, maxh: float) -> gfem.Mesh:
    """The 3D column on a Kuhn grid, cell edges about ``maxh`` along every axis."""
    a, d = _depths()
    width = 0.6
    z = np.unique(
        np.concatenate(
            [
                np.linspace(z0, z1, max(1, round((z1 - z0) / maxh)) + 1)
                for z0, z1 in ((-a - d, -a), (-a, 0.0), (0.0, H))
            ]
        )
    )
    xy = np.linspace(0.0, width, max(1, round(width / maxh)) + 1)

    def volume(c: np.ndarray) -> str:
        return "pml" if c[2] < -a else "sub" if c[2] < 0 else "layer"

    def face(c: np.ndarray) -> str:
        for i, axis in enumerate("xy"):
            if abs(c[i]) < 1e-9:
                return f"{axis}0"
            if abs(c[i] - width) < 1e-9:
                return f"{axis}1"
        return ""

    sm.write_msh22(path, sm.carve((xy, xy, z), volume, face, None))
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def leaky_column(
    path: Path,
    maxh: float,
    *,
    side_layers: bool = False,
    dim: int = 2,
    quality: float | None = None,
    reflection: float = R0,
) -> Column:
    """The leaky column of ``dim`` 2 (a section) or 3 (a body), meshed at ``maxh``.

    Args:
        path: The mesh file to write.
        maxh: The element size, um.
        side_layers: A 2D section with a layer on each side too (the corner
            layers stretch along x and y); the layer is 0.6 um thick there.
        dim: 2 for the antiplane section, 3 for the thickness-shear body.
        quality: The mechanical quality factor of every region, layers
            included.
        reflection: The layer's target reflection.
    """
    mesh = _section(path, maxh, 0.6 if side_layers else 0.0) if dim == 2 else None
    if mesh is None:
        mesh = _body(path, maxh)
    layer_names = [n for n in mesh.volumes if n.startswith("pml") or "_" in n]
    materials = {}
    for name in mesh.volumes:
        base = name.split("_")[0]
        mu, rho = (MU_LAYER, RHO_LAYER) if base == "layer" else (MU_SUB, RHO_SUB)
        materials[name] = _material(mu, rho, quality)
    clamped = (
        {"out_left": "x", "out_right": "x"} if dim == 2 else {"x0": "z", "x1": "z"}
    )
    return Column(
        model=gfem.Model(mesh, materials=materials),
        pml=gfem.PML(layer_names, target_reflection=reflection),
        omega=leaky_roots(1),
        clamped=clamped,
        thickness=_depths()[1],
    )
