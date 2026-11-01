"""Built-wheel smoke: a parallel-plate line solved three ways, against closed forms.

The electrostatic capacitance and its gradient, a driven sweep through two
lumped ports, and the first two resonances of the open line, on a 120-tet
mesh: a few seconds on any platform. The run fails unless gfem was imported
from the environment's site-packages, so it proves the installed wheel works,
not the checkout. Run it with the interpreter of a fresh venv holding the
wheel, for example::

    .venv-smoke/bin/python -E -s tools/wheel_smoke.py

Not with ``-I``: that drops the script's directory from ``sys.path``, and
``structured_meshes`` is imported from there. The last line is
``OK <platform> <python>``.
"""

from __future__ import annotations

import math
import platform
import sys
import sysconfig
import tempfile
from pathlib import Path

import jax
import numpy as np
import scipy.constants as const
import structured_meshes as sm

import gfem


def check(ok, failure: str) -> None:
    """Exit with ``failure`` unless ``ok`` (an assert would vanish under -O)."""
    if not ok:
        raise SystemExit(f"FAILED: {failure}")


site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
print("gfem", gfem.__version__, "from", gfem.__file__)
check(sys.prefix != sys.base_prefix, f"not in a venv: {sys.prefix}")
check(
    Path(gfem.__file__).resolve().is_relative_to(site_packages),
    f"gfem imported from {gfem.__file__}, not from {site_packages}",
)

L, W, H, EPS = 10.0, 2.0, 1.0, 4.0  # mm
path = Path(tempfile.mkdtemp()) / "line.msh"
sm.write_msh22(path, sm.strip_line(L, W, H, (20, 1, 1)))
mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
materials = {"dielectric": gfem.Material(eps_r=EPS)}
plates = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")

# Electrostatic: C = eps0 eps_r W L / H, and dC/deps summed over the
# elements is C / eps_r.
model = gfem.Model(mesh, materials=materials)
c = float(np.asarray(gfem.solve(model, plates).capacitance)[0, 0])
c_exact = const.epsilon_0 * EPS * W * L * 1e-3 / H
rel = abs(c / c_exact - 1)
print(f"electrostatic C = {c:.6e} F, exact {c_exact:.6e}, rel {rel:.1e}")
check(rel < 1e-6, "electrostatic capacitance")


def capacitance(eps):
    """The plate capacitance with one permittivity per element."""
    field = model.with_material("dielectric", gfem.MaterialField(eps_r=eps))
    return gfem.solve(field, plates).capacitance[0, 0]


n = len(mesh.elements("dielectric"))
g = float(jax.grad(capacitance)(jax.numpy.full(n, EPS)).sum())
print(f"electrostatic dC/deps sum = {g:.6e}, expected {c_exact / EPS:.6e}")
check(abs(g / (c_exact / EPS) - 1) < 1e-6, "electrostatic gradient")

# Driven: a matched line between two lumped ports, so S is passive and
# reciprocal.
line = gfem.Model(
    mesh,
    materials=materials,
    boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    ports=[gfem.LumpedPort("p1", "port1", "+z"), gfem.LumpedPort("p2", "port2", "+z")],
)
s = np.asarray(gfem.solve(line, gfem.Driven(frequency=[2e9, 5e9])).s.values)
print("driven |S| at 2, 5 GHz:", np.round(np.abs(s), 4).tolist())
check(np.all(np.abs(s) <= 1 + 1e-9), "driven S is not passive")
check(np.allclose(s[:, 0, 1], s[:, 1, 0], atol=1e-9), "driven S is not reciprocal")

# Eigenmode: the open line resonates at f_m = m c / (2 L sqrt(eps_r)).
open_line = gfem.Model(
    mesh, materials=materials, boundaries=[gfem.PEC("bottom"), gfem.PEC("top")]
)
f1 = const.c / (2 * L * 1e-3 * math.sqrt(EPS))
modes = gfem.solve(open_line, gfem.Eigenmode(target=0.5 * f1, n_modes=2))
f = np.asarray(modes.frequency)
print(f"eigenmode f = {f.tolist()}, exact {[f1, 2 * f1]}")
np.testing.assert_allclose(f, [f1, 2 * f1], rtol=1e-3)

print("OK", sys.platform, platform.python_version())
