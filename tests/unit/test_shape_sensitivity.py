"""Node-coordinate (shape) terms of the engine sensitivity kernel.

Volume terms use hand-written transport tensors: checked against NGSolve's
``DiffShape`` on real fields, where ``DiffShape`` is correct. Surface terms
use ``DiffShape`` on a real Lagrangian of Re/Im-split fields. The whole
kernel is checked against finite differences of the assembled driven
pieces on complex fields.
"""

import jax
import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import derivative

import gfem
from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.domain import em_region, h1_space, hcurl_space
from gfem._engine.driven import DrivenRule, driven_operator
from gfem._engine.sensitivity import FieldPair, ShapeTerm, sensitivity

NODES = ("nodes",)
ORDER = 2
PORTS = ("port1", "port2")


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    """A two-layer line with every kind of driven piece."""
    path = tmp_path_factory.mktemp("line") / "line.msh"
    layers = (("lower", 0.5), ("upper", 0.5))
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 2, 1), layers=layers))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    n = len(mesh.elements("lower"))
    ramp = np.linspace(1.0, 1.5, n)
    return gfem.Model(
        mesh,
        materials={
            "lower": gfem.MaterialField(
                eps_r=3.9 * (1 - 0.02j) * ramp, mu_r=1.2 * ramp, conductivity=0.5 * ramp
            ),
            "upper": gfem.Material(
                eps_r=(2.0, 3.0, 4.0),
                axes=((0.6, 0.8, 0.0), (-0.8, 0.6, 0.0), (0.0, 0.0, 1.0)),
                tan_delta=0.01,
            ),
        },
        boundaries=[
            gfem.PEC("bottom"),
            gfem.Conductor("top", conductivity=5e7, thickness=1e-6),
            gfem.Absorbing("side", order=2),
            gfem.Impedance("lower__upper", resistance=300.0, capacitance=1e-15),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _moved(model, nodes):
    mesh = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(model.mesh), [nodes]
    )
    return model.replace(mesh=mesh)


def _vector_h1_gradient(model, form):
    """``form.DiffShape(W)`` assembled per node, (n_nodes, 3) per file unit."""
    built = discretization(model.mesh).built
    space = ngs.VectorH1(built.mesh, order=1)
    lf = ngs.LinearForm(space)
    lf += form(space.TestFunction())
    lf.Assemble()
    return lf.vec.FV().NumPy().reshape(3, -1).T.copy()


def _tensor(model, *, symmetric=True):
    """An anisotropic, element-wise constant coefficient."""
    built = discretization(model.mesh).built
    rng = np.random.default_rng(5)
    n = len(model.mesh.tetrahedra)
    a = rng.uniform(-0.3, 0.3, (n, 3, 3))
    values = 2.0 * np.eye(3) + a + (np.swapaxes(a, 1, 2) if symmetric else 0)
    gfs = []
    for i in range(3):
        for j in range(3):
            gf = ngs.GridFunction(ngs.L2(built.mesh, order=0))
            gf.vec.FV().NumPy()[:] = values[built.tetrahedron_order, i, j]
            gfs.append(gf)
    return ngs.CF(tuple(gfs), dims=(3, 3))


def _scalar(model):
    built = discretization(model.mesh).built
    gf = ngs.GridFunction(ngs.L2(built.mesh, order=0))
    rng = np.random.default_rng(6)
    gf.vec.FV().NumPy()[:] = rng.uniform(1.0, 3.0, len(model.mesh.tetrahedra))
    return gf


DIFFERENTIAL = {
    "gradient": ngs.grad,
    "value": lambda u: u,
    "curl": ngs.curl,
}


@pytest.mark.parametrize("coefficient", ["scalar", "tensor", "general"])
@pytest.mark.parametrize("kind", ["gradient", "value", "curl"])
def test_transport_tensors_match_diffshape_on_real_fields(line, kind, coefficient):
    model = line
    built = discretization(model.mesh).built
    domain = em_region(model, built)
    if kind == "gradient":
        space = h1_space(model, built, ORDER)
    else:
        space = hcurl_space(model, built, ORDER, complex_valued=False)
    coef = {
        "scalar": lambda: _scalar(model),
        "tensor": lambda: _tensor(model),
        "general": lambda: _tensor(model, symmetric=False),
    }[coefficient]()
    rng = np.random.default_rng(1)
    left = rng.standard_normal((space.ndof, 2))
    right = rng.standard_normal((space.ndof, 2))
    theta = 0.7
    term = ShapeTerm("t", kind, coefficient=coef)
    pair = FieldPair(space, left, right, {"t": theta}, ORDER, domain)
    got = sensitivity(model, [pair], {NODES}, form="driven", shape=lambda: [term])
    # DiffShape's reference: the same Lagrangian with the fields transported
    gfs = [[ngs.GridFunction(space) for _ in range(2)] for _ in range(2)]
    for k in range(2):
        gfs[0][k].vec.FV().NumPy()[:] = left[:, k]
        gfs[1][k].vec.FV().NumPy()[:] = right[:, k]
    d = DIFFERENTIAL[kind]
    dx = quadrature.dx(ORDER, definedon=domain)
    lagrangian = sum(
        theta * ((coef * d(gfs[0][k])) * d(gfs[1][k])) * dx for k in range(2)
    )
    expected = _vector_h1_gradient(model, lagrangian.DiffShape) / built.scale
    assert got[NODES].shape == model.mesh.nodes.shape
    np.testing.assert_allclose(
        got[NODES], expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


def _pieces_value(model, left, right, omega, names):
    """``Re sum_t theta_t left^T A_t right`` over the named driven pieces."""
    op = driven_operator(model, ORDER)
    total = 0.0
    for k, piece in enumerate(op.pieces):
        if piece.name in names:
            total += complex(piece.theta(omega)) * np.sum(left * op.apply(k, right))
    return total.real


def _shape_terms(model):
    # built on the current discretization of the mesh (never cache them)
    rule = DrivenRule(
        model, gfem.Driven(frequency=20e9), gfem.SolverOptions(order=ORDER)
    )
    return rule.shape_terms(model)


def test_driven_shape_terms_cover_every_piece_but_the_ports(line):
    shape_terms = _shape_terms(line)
    names = {p.name for p in driven_operator(line, ORDER).pieces}
    assert {t.name for t in shape_terms} == {
        n for n in names if not n.startswith("port ")
    }


@pytest.mark.parametrize(
    "piece",
    [
        "curl_curl",
        "mass",
        "conductivity",
        "absorbing",
        "absorbing order 2",
        "conductor top (one-sided)",
        "impedance lower__upper R",
        "impedance lower__upper C",
    ],
)
def test_driven_node_gradient_matches_fd_on_complex_fields(line, piece):
    shape_terms = _shape_terms(line)
    op = driven_operator(line, ORDER)
    assert piece in {p.name for p in op.pieces}
    omega = 2 * np.pi * 20e9
    rng = np.random.default_rng(2)
    ndof = op.space.ndof
    left = rng.standard_normal((ndof, 2)) + 1j * rng.standard_normal((ndof, 2))
    right = rng.standard_normal((ndof, 2)) + 1j * rng.standard_normal((ndof, 2))
    thetas = {p.name: complex(p.theta(omega)) for p in op.pieces}
    pair = FieldPair(
        op.space, left, right, {piece: thetas[piece]}, ORDER, em_region(line, op.built)
    )
    got = sensitivity(line, [pair], {NODES}, form="driven", shape=lambda: shape_terms)
    nodes = np.asarray(line.mesh.nodes)
    direction = rng.standard_normal(nodes.shape)
    # the operator's lumped ports need tangent directions: hold their nodes
    mesh = line.mesh
    ports = np.isin(mesh.triangle_tags, [mesh.region(s).tag for s in PORTS])
    direction[np.unique(mesh.triangles[ports])] = 0.0
    direction /= np.linalg.norm(direction)

    def value(x):
        return _pieces_value(_moved(line, x), left, right, omega, {piece})

    fd = derivative(value, nodes, direction, 2e-7)
    ad = float(np.sum(got[NODES] * direction))
    assert ad == pytest.approx(fd, rel=1e-6)
