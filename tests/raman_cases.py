"""Filled PEC guides at optical size and flat Raman spectra for the Raman tests.

`guide_model` writes the section of a guide of sides ``a`` x ``b`` (um) with
one cell layer along z, whose ports "p1" (z = 0) and "p2" are the section, and
returns a model with PEC walls. `flat_spectrum` is a gain that is constant above
a shift, so that a closed form can use one number at any pump frequency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.constants as sc
import structured_meshes as sm

import gfem

#: reference Stokes wavelength of the closed forms (m)
LAMBDA_S = 1.55e-6
F_S = sc.c / LAMBDA_S


def guide_mesh(
    path: Path, a: float, b: float, *, nx: int, ny: int, length: float = 0.2
) -> Path:
    """The guide's mesh: regions "air", surfaces "port1", "port2" and "walls" (um)."""
    edges = (
        np.linspace(0.0, a, nx + 1),
        np.linspace(0.0, b, ny + 1),
        np.linspace(0.0, length, 2),
    )
    eps = 1e-9

    def surface(c):
        if abs(c[2]) < eps:
            return "port1"
        if abs(c[2] - length) < eps:
            return "port2"
        return "walls"

    return sm.write_msh22(path, sm.structured(edges, lambda _: "air", surface))


def guide_model(path: Path, a: float, b: float, material, *, nx: int, ny: int):
    mesh = gfem.Mesh.from_gmsh(guide_mesh(path, a, b, nx=nx, ny=ny), length_unit="um")
    return gfem.Model(
        mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def flat_spectrum(gain: float, *, symmetry: str, perpendicular: float | None = None):
    """``gain`` (m/W) at every shift from 1 THz to 40 THz, referred to 1550 nm."""
    shift = np.array([0.0, 1e12, 40e12])
    kwargs = {}
    if perpendicular is not None:
        kwargs["perpendicular_gain"] = np.array([0.0, perpendicular, perpendicular])
    return gfem.RamanSpectrum(
        shift, np.array([0.0, gain, gain]), F_S, symmetry, **kwargs
    )


def te_index(f: float, n: float, a: float) -> float:
    """The effective index of the TE mode with one half-wave across ``a`` (um)."""
    k0 = 2 * np.pi * f / sc.c
    return float(np.sqrt(1 - (np.pi / (a * 1e-6) / (n * k0)) ** 2) * n)
