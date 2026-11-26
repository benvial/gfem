r"""The optomechanical overlap of a displacement with a wave port's mode.

A mechanical displacement :math:`u(x, y)` of a waveguide's cross-section,
scaled by a dimensionless amplitude :math:`\alpha`, changes the optical
mode's effective index at a fixed optical frequency :math:`\omega` by two
mechanisms, to first order in :math:`\alpha`:

.. math::

    \frac{\partial n_\mathrm{eff}}{\partial\alpha}
    = \delta n_\mathrm{MB} + \delta n_\mathrm{PE}

The moving-boundary part :math:`\delta n_\mathrm{MB}` is the displacement of
the section's nodes seen by the optical mode (Johnson's perturbation of a
shifted interface, discretized); the photoelastic part
:math:`\delta n_\mathrm{PE}` is the strain of the displacement acting through
the materials' ``photoelastic_tensor`` (as in `gfem.acousto_optic_shift`).
`optomechanical_overlap` gives both, per unit amplitude of the displacement
it is given.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._dim2 import refuse_entry
from gfem._optical_shift import (
    _check_model,
    _frequency,
    _mode_weights,
    _section,
    _wave_port,
)
from gfem._pytree import register_fields
from gfem.acousto_optic import (
    _check_displacement,
    _check_optical_fixed,
    _check_plane,
    _contribution,
    _crystal,
    _mean_strain_of,
    _weights_rule,
)
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from gfem._field import FieldMap
    from gfem.model import Model

__all__ = ["OptomechanicalOverlap", "optomechanical_overlap"]

_OWNER = "optomechanical_overlap"


@dataclasses.dataclass(frozen=True, eq=False)
class OptomechanicalOverlap:
    r"""The overlap of a displacement with a wave port's mode, per unit amplitude.

    With :math:`u \to \alpha u` for a dimensionless amplitude :math:`\alpha`,
    the effective index of the mode at the fixed optical frequency changes as
    :math:`\partial n_\mathrm{eff}/\partial\alpha` = ``total``: dimensionless
    (the displacement is in metres, the index per unit amplitude of that
    field). The phase of the displacement carries into the values: complex for
    a complex displacement (a lossy or a driven mode), real to round-off for a
    real one.

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: The optical frequency, Hz.
        n_eff: The unperturbed effective index (real; the section is
            lossless), a constant.
        n_g: The mode's group index (real), a constant.
        moving_boundary: :math:`\delta n_\mathrm{MB}`, the moving-boundary
            part, complex scalar.
        photoelastic: :math:`\delta n_\mathrm{PE}`, the photoelastic part,
            complex scalar.
    """

    port: str
    mode: int
    frequency: float
    n_eff: jax.Array
    n_g: jax.Array
    moving_boundary: jax.Array
    photoelastic: jax.Array

    @property
    def total(self) -> jax.Array:
        r""":math:`\partial n_\mathrm{eff}/\partial\alpha`: the two parts' sum."""
        return self.moving_boundary + self.photoelastic

    def __repr__(self) -> str:
        head = f"OptomechanicalOverlap(port {self.port!r}, mode {self.mode}"
        try:
            n, mb, pe = (
                complex(np.asarray(v))
                for v in (self.n_eff, self.moving_boundary, self.photoelastic)
            )
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"{head})"
        return (
            f"{head}, {self.frequency:.6g} Hz, n_eff {n.real:.6f}, "
            f"moving_boundary {mb:.6g}, photoelastic {pe:.6g})"
        )


register_fields(
    OptomechanicalOverlap,
    ("n_eff", "n_g", "moving_boundary", "photoelastic"),
)


def _optical_only(model: Model) -> Model:
    """``model`` without the photoelastic tensors, which the mode does not read.

    A tensor traced by `jax.grad` would send the mode solve down the traced
    path; the mode is held constant, so it never needs them.
    """
    return model.replace(
        materials={
            name: dataclasses.replace(material, photoelastic_tensor=None)
            for name, material in model.materials.items()
        }
    )


def _node_gradient(
    model: Model, port: str, frequency: float, mode: int | None
) -> tuple[np.ndarray, jax.Array, jax.Array, int]:
    r"""The gradient of :math:`\mathrm{Re}\,n_\mathrm{eff}` in the mesh nodes.

    One port mode solve and its backward pass. Returns ``(n_nodes, 3)`` in
    1/m (non-zero on the port's nodes), ``n_eff``, ``n_g`` and the mode
    number.
    """
    import gfem
    from gfem.problems import PortMode

    kept: dict[str, Any] = {}

    def n_eff(nodes: jax.Array) -> tuple[jax.Array, jax.Array]:
        moved = model.replace(mesh=model.mesh.with_nodes(nodes))
        out = gfem.solve(moved, PortMode(port, frequency, mode=mode))
        kept["mode"] = out.mode
        n_g = jnp.asarray(out.n_g)  # a PortMode solve always carries it
        kept["n_g"] = jax.lax.stop_gradient(jnp.real(n_g[0]))
        value = out.n_eff[0]
        return jnp.real(value), jax.lax.stop_gradient(jnp.real(value))

    grad, n = jax.grad(n_eff, has_aux=True)(jnp.asarray(model.mesh.nodes))
    return np.asarray(grad), n, kept["n_g"], kept["mode"]


def _supported_nodes(
    displacement: FieldMap, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The rows of ``nodes`` (``(n, 3)``, m) in the displacement's support.

    Returns the boolean mask over ``nodes`` and the points ``(m, 2)`` of the
    covered ones, the :math:`(x, y)` the in-plane displacement is read at.
    """
    from gfem import _probe

    xy = nodes[:, :2]
    mask = np.ones(len(xy), bool)
    for c in "xy":
        field = displacement[c]
        mask &= _probe.engine().covered(field.mesh, _probe.layout(field), xy)
    return mask, xy[mask]


def optomechanical_overlap(
    model: Model,
    displacement: FieldMap,
    *,
    port: str,
    frequency: float,
    mode: int | None = None,
) -> OptomechanicalOverlap:
    r"""The optomechanical overlap of a displacement with a wave port's mode.

    Solves the port's mode on ``model`` (a `gfem.PortMode` at one optical
    frequency) and gives the change of its effective index at that frequency
    per unit amplitude :math:`\alpha` of ``displacement`` :math:`u`, to first
    order:

    .. math::

        \frac{\partial n_\mathrm{eff}}{\partial\alpha}
        = \underbrace{\sum_{i\in\mathrm{section}}
            \frac{\partial\,\mathrm{Re}\,n_\mathrm{eff}}{\partial \mathbf r_i}
            \cdot \mathbf W_i}_{\delta n_\mathrm{MB}}
        + \underbrace{\sum_t W_t : \Delta\varepsilon_t}_{\delta n_\mathrm{PE}},
        \qquad
        \Delta\varepsilon_t = -\varepsilon_t\,(p_t \bar S_t)\,\varepsilon_t

    The moving-boundary part is the gradient of the port mode's effective
    index in the nodes :math:`\mathbf r_i` of its section, contracted with the
    in-plane displacement :math:`\mathbf W_i = (u_x, u_y)` read at each node
    inside the displacement's support (zero at the others, front and back
    nodes of the slab alike: the slab moves as a prism; :math:`u_z` moves no
    node of a section that is invariant along :math:`z`). It is the discrete
    form of Johnson's interface integral, exact for the discrete problem, for
    any permittivity (anisotropic included) and without a one-sided sampling of
    the field at a corner. It needs no conformity of the two meshes: an
    interface of the optical mesh carries the term, and a mechanical boundary
    inside a homogeneous optical region contributes discretization error only.

    The photoelastic part is the first-order shift of
    `gfem.acousto_optic_shift` for the strain :math:`\bar S_t` of the
    displacement (`gfem.strain`, mean over each tetrahedron :math:`t` behind a
    triangle of the port's section, read at its quadrature points) through the
    ``photoelastic_tensor`` :math:`p_t` of its material, with :math:`W_t`
    the mode's weight at 1 W (see `gfem.acousto_optic_shift`). By reciprocity
    the volume overlap of the strain already holds the electrostrictive
    pressure on the boundaries. The rotation of the crystal by the
    displacement (roto-optic) is not counted.

    The sign is that of the displacement: reversing :math:`u` reverses both
    parts, and a mechanical mode's own sign follows its phase convention.
    A positive value means the displacement raises the effective index.

    The optical mode is solved once and held constant. `jax.grad` reaches
    the displacement's fields (so every input of the mechanical solve that
    made them) and the ``photoelastic_tensor`` of a `gfem.MaterialField`; the
    optical model's other inputs are refused.

    Args:
        model: The optical model: a 3D slab (`gfem.Mesh.from_gmsh_cross_section`)
            with a lossless section.
        displacement: A displacement `gfem.FieldMap` with the components
            ``x``, ``y``, ``z`` (m) at :math:`q = 0`, on a native 2D mesh with
            the same global :math:`(x, y)` as the optical section (translate a
            mesh to align them).
        port: The wave port's name; its section lies in a plane
            :math:`z` = const.
        frequency: One optical frequency, Hz.
        mode: Which mode (1 = the most propagating), or ``None`` for the wave
            port's own.

    Returns:
        The two parts and the mode's effective and group index.

    Raises:
        TypeError: ``model`` is not a `gfem.Model`, or ``displacement`` is not
            a `gfem.FieldMap`.
        ValueError: ``port`` is not a wave port of the model, a sequence of
            frequencies is given, ``displacement`` lacks a component, is not
            on a 2D mesh or has an axial wavenumber other than 0, the port's
            section is not in a plane z = const, no node of the section is in
            the displacement's support and no element has a
            ``photoelastic_tensor``, or an element with one lies outside the
            displacement's support (the message names the region).
        NotImplementedError: The model is on a 2D mesh (a 2D `gfem.PortMode` is
            a line mode) or the section is lossy (a complex ``eps_r``, a
            ``tan_delta`` or ``conductivity`` other than 0, or a permittivity
            law).
        NonDifferentiableError: Under `jax.grad`, an input of the optical
            model other than the photoelastic tensors (its ``eps_r``,
            ``tan_delta``, ``mu_r``, ``conductivity`` or node coordinates):
            the mode is held constant.
    """
    refuse_entry(getattr(model, "mesh", None), _OWNER)
    _check_model(model, _OWNER)
    _check_displacement(displacement, _OWNER)
    f = _frequency(frequency, _OWNER)
    faces, tets = _section(model, _wave_port(model, port))
    crystal = _crystal(
        model, port, tets, _OWNER, tensors=("photoelastic_tensor",), require=False
    )
    _check_optical_fixed(
        model,
        _OWNER,
        "the optomechanical overlap",
        "the displacement or the photoelastic_tensor",
    )
    if is_traced(model.mesh.nodes):
        raise NonDifferentiableError(
            f"{_OWNER}: gfem cannot differentiate the optomechanical overlap with "
            "respect to the optical model's node coordinates: the mode and the "
            "points the displacement is read at are held constant. Wrap them "
            "in jax.lax.stop_gradient to use their value without their "
            "derivative."
        )
    _check_plane(model.mesh, faces, port, _OWNER, "mechanical")
    ip = np.flatnonzero(crystal.has_p)
    section = np.unique(np.asarray(model.mesh.triangles)[faces])
    nodes = np.asarray(model.mesh.nodes, float)
    inside, points = _supported_nodes(displacement, nodes[section])
    if not len(ip) and not inside.any():
        raise ValueError(
            f"{_OWNER}: no element of wave port {port!r}'s section has a "
            "photoelastic_tensor and no node of the section lies in the "
            "displacement's support, so the overlap is zero. Set a "
            "photoelastic_tensor on the optical material, or align the "
            "displacement's mesh with the section (same x, y in m)."
        )
    owner = f"{_OWNER} (wave port {port!r})"
    pe = jnp.zeros((), dtype=jnp.complex128)
    if len(ip):
        mean_s = _mean_strain_of(
            displacement,
            model.mesh,
            tets[crystal.rows[ip]],
            offset=jnp.zeros(2),
            owner=owner,
            moved={},
        )
    optical = _optical_only(model)
    grad, n_eff, n_g, found = _node_gradient(optical, port, f, mode)
    mb = jnp.zeros((), dtype=jnp.complex128)
    if inside.any():
        w = jnp.stack([displacement[c].at(points) for c in "xy"], axis=1)
        g = jnp.asarray(grad[section[inside]][:, :2])
        mb = jnp.sum(g * w).astype(jnp.complex128)
    if len(ip):
        _, _, weights, _ = _mode_weights(
            optical, port, f, mode, faces, tensor=True, rule=_weights_rule()
        )
        eta = jnp.einsum("nIJ,nJ->nI", crystal.p[ip], mean_s)
        pe = _contribution(weights[crystal.rows[ip]], crystal.eps[ip], eta[None])[0]
    return OptomechanicalOverlap(
        port=port,
        mode=found,
        frequency=f,
        n_eff=n_eff,
        n_g=n_g,
        moving_boundary=mb,
        photoelastic=pe.astype(jnp.complex128),
    )
