r"""Curved-wall verification cases: gmsh builders and closed forms.

Two geometries with a closed-form answer on a curved wall, each meshed by gmsh
at order 1 and then at order 2 from the same mesh, so the two files have
identical corners and differ only in the mid-edge nodes (put on the CAD
surface by ``setOrder(2)``, no high-order optimisation: the step a user takes
with ``Mesh.ElementOrder 2``):

- coax: a quarter of a circular coaxial line, radii $a = 1$ and $b = 2$,
  length $L = 0.5$ (volume ``fill``, surfaces ``inner``, ``outer`` and
  ``sym``, the planes and ends, left natural). Capacitance
  $C = \pi \varepsilon_0 L / (2 \ln(b/a))$, lengths in micrometres.
- sphere: one octant of a unit sphere (1 mm), volume ``ball``, surface
  ``sphere`` and the planes ``x0``, ``y0``, ``z0``. A metal sphere cavity
  resonates at $f = c\, x / (2 \pi R)$ with $x$ a root of
  $(x j_n(x))'$ (TM$_{n}$) or of $j_n$ (TE$_n$); PEC on the sphere and the
  planes the mode's field is tangential to, the others natural (PMC).

gmsh is imported when a case is built, never at import, so collecting a test
file needs no gmsh. ``General.NumThreads`` is 1 and the sizes are set with
``MeshSizeMin = MeshSizeMax = h``: the mesh is the one the verification
tables were measured on, up to the gmsh version.
"""

import math
from pathlib import Path

import numpy as np
import scipy.constants as const
import scipy.optimize as opt
import scipy.special as sps

import gfem

A, B, L = 1.0, 2.0, 0.5  # coax radii and length, um
R = 1.0  # sphere radius, mm

OPTIONS = {
    "linear_solver": "direct",
    "threads": 1,
}

COAX_PROBLEM = gfem.Electrostatic(terminals={"in": "inner"}, ground="outer")


def _group(name, tags, dim):
    import gmsh

    gmsh.model.addPhysicalGroup(dim, tags, name=name)


def _coax():
    import gmsh

    occ = gmsh.model.occ
    outer = occ.addCylinder(0, 0, 0, 0, 0, L, B, angle=math.pi / 2)
    inner = occ.addCylinder(0, 0, 0, 0, 0, L, A, angle=math.pi / 2)
    occ.cut([(3, outer)], [(3, inner)])
    occ.synchronize()
    _group("fill", [t for _, t in gmsh.model.getEntities(3)], 3)
    groups = {"inner": [], "outer": [], "sym": []}
    for _, tag in gmsh.model.getEntities(2):
        if gmsh.model.getType(2, tag) != "Cylinder":
            groups["sym"].append(tag)
            continue
        box = gmsh.model.getBoundingBox(2, tag)
        # the inner wall lies within radius 1, the outer reaches radius 2
        groups["inner" if max(box[3], box[4]) < 1.5 else "outer"].append(tag)
    for name, tags in groups.items():
        _group(name, tags, 2)


def _sphere():
    import gmsh

    occ = gmsh.model.occ
    ball = occ.addSphere(0, 0, 0, R)
    box = occ.addBox(0, 0, 0, 1.5 * R, 1.5 * R, 1.5 * R)
    occ.intersect([(3, ball)], [(3, box)])
    occ.synchronize()
    _group("ball", [t for _, t in gmsh.model.getEntities(3)], 3)
    groups = {"sphere": [], "x0": [], "y0": [], "z0": []}
    for _, tag in gmsh.model.getEntities(2):
        if gmsh.model.getType(2, tag) == "Sphere":
            groups["sphere"].append(tag)
            continue
        box = gmsh.model.getBoundingBox(2, tag)
        # a coordinate plane is the face with zero extent along its normal
        axis = int(np.argmin([box[3] - box[0], box[4] - box[1], box[5] - box[2]]))
        groups["xyz"[axis] + "0"].append(tag)
    for name, tags in groups.items():
        _group(name, tags, 2)


def write_twins(case, h, directory):
    """Mesh ``case`` ("coax" or "sphere") at size ``h``; write both orders.

    Returns ``(order1_path, order2_path)`` in ``directory``: one mesh, written
    before and after ``setOrder(2)``, so the corners are identical.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add(case)
        {"coax": _coax, "sphere": _sphere}[case]()
        gmsh.model.mesh.generate(3)
        paths = []
        for order in (1, 2):
            if order == 2:
                gmsh.model.mesh.setOrder(2)
            paths.append(Path(directory) / f"{case}_{h:g}_order{order}.msh")
            gmsh.write(str(paths[-1]))
    finally:
        gmsh.finalize()
    return tuple(paths)


# --- coax --------------------------------------------------------------------

C_COAX = const.epsilon_0 * math.pi * L * 1e-6 / (2 * math.log(B / A))


def coax_error(path, order):
    """Relative error of the quarter coax's capacitance read from ``path``."""
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    model = gfem.Model(mesh, materials={"fill": gfem.Material(eps_r=1.0)})
    options = gfem.SolverOptions(order=order, **OPTIONS)
    c = float(gfem.solve(model, COAX_PROBLEM, options=options).capacitance[0, 0])
    return (c - C_COAX) / C_COAX


# --- sphere ------------------------------------------------------------------


def _root(f, lo, hi):
    return opt.brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)


def _d_xjn(n):
    """The derivative of $x j_n(x)$, whose roots are the TM$_n$ resonances."""
    return lambda x: (
        sps.spherical_jn(n, x) + x * sps.spherical_jn(n, x, derivative=True)
    )


#: First roots $kR$ of the closed forms (brackets are the neighbouring
#: roots of the same functions, 1e-14 rtol; each is the lowest of its family
#: and the brackets are checked by the sign change brentq requires).
K_TM11 = _root(_d_xjn(1), 2.0, 3.5)
K_TM21 = _root(_d_xjn(2), 3.0, 4.2)
K_TE11 = _root(lambda x: sps.spherical_jn(1, x), 4.0, 5.0)

#: PEC walls and the modes searched, per class: TM$_{11}$ is $E_z$-dipole,
#: tangential on the sphere and $z = 0$, natural on $x = 0$ and $y = 0$;
#: TM$_{21}$ and TE$_{11}$ are the two lowest modes of the class with PEC on
#: the sphere, $x = 0$ and $y = 0$ and a natural $z = 0$.
TM_PEC = ("sphere", "z0")
TE_PEC = ("sphere", "x0", "y0")


def frequency(k):
    """The resonance of a root $kR = k$ on the sphere of radius ``R`` mm, Hz."""
    return const.c * k / (2 * math.pi * R * 1e-3)


def sphere_errors(path, order, pec, ks, n_modes):
    """Relative errors of the modes nearest the roots ``ks``.

    The search targets 0.9 times the highest root; the modes found are matched
    to the closest ``ks`` (the roots are well apart: 14 % at the closest) and
    the errors returned in the order of ``ks``.
    """
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"ball": gfem.Material(eps_r=1.0)},
        boundaries=[gfem.PEC(name) for name in pec],
    )
    options = gfem.SolverOptions(order=order, **OPTIONS)
    target = 0.9 * frequency(max(ks))
    result = gfem.solve(
        model, gfem.Eigenmode(target=target, n_modes=n_modes), options=options
    )
    found = np.sort(np.asarray(result.frequency, dtype=float))
    exact = np.array([frequency(k) for k in ks])
    nearest = [found[np.argmin(abs(found - f))] for f in exact]
    return (np.array(nearest) - exact) / exact
