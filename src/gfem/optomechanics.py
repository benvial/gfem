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
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

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
    from collections.abc import Sequence

    from gfem._field import FieldMap
    from gfem.model import Model
    from gfem.results import MechanicalEigenmodeResult

__all__ = [
    "OptomechanicalCoupling",
    "OptomechanicalOverlap",
    "optomechanical_coupling",
    "optomechanical_overlap",
]

_OWNER = "optomechanical_overlap"
_COUPLING = "optomechanical_coupling"


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
            n = complex(np.asarray(self.n_eff))
            if np.ndim(self.moving_boundary):
                return (
                    f"{head}, {self.frequency:.6g} Hz, n_eff {n.real:.6f}, "
                    f"{np.shape(self.moving_boundary)[0]} displacements)"
                )
            mb, pe = (
                complex(np.asarray(v))
                for v in (self.moving_boundary, self.photoelastic)
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
        # the support is a place, not a design variable: its test reads the
        # value of node coordinates a node map traces
        mesh = field.mesh
        if is_traced(mesh.nodes):
            mesh = mesh.with_nodes(jax.lax.stop_gradient(mesh.nodes))
        mask &= _probe.engine().covered(mesh, _probe.layout(field), xy)
    return mask, xy[mask]


class _Setup(NamedTuple):
    """What the optical model fixes before any solve: the section and its crystal."""

    frequency: float
    faces: np.ndarray
    tets: np.ndarray
    crystal: Any
    section: np.ndarray
    nodes: np.ndarray
    ip: np.ndarray


def _setup(
    model: Model,
    port: str,
    frequency: float,
    owner: str,
    what: str,
    *,
    through: str,
    displacement: Any,
) -> _Setup:
    """Check the optical model and read its section (no solve).

    Raises what `optomechanical_overlap` documents for the model, the port,
    the frequency and ``displacement``.
    """
    refuse_entry(getattr(model, "mesh", None), owner)
    _check_model(model, owner)
    _check_displacement(displacement, owner)
    f = _frequency(frequency, owner)
    faces, tets = _section(model, _wave_port(model, port))
    crystal = _crystal(
        model, port, tets, owner, tensors=("photoelastic_tensor",), require=False
    )
    _check_optical_fixed(model, owner, what, through)
    if is_traced(model.mesh.nodes):
        raise NonDifferentiableError(
            f"{owner}: gfem cannot differentiate {what} "
            "with respect to the optical model's node coordinates: the mode and "
            "the points the displacement is read at are held constant. Wrap them "
            "in jax.lax.stop_gradient to use their value without their "
            "derivative."
        )
    _check_plane(model.mesh, faces, port, owner, "mechanical")
    section = np.unique(np.asarray(model.mesh.triangles)[faces])
    return _Setup(
        f,
        faces,
        tets,
        crystal,
        section,
        np.asarray(model.mesh.nodes, float),
        np.flatnonzero(crystal.has_p),
    )


def _check_support(
    setup: _Setup, displacement: FieldMap, owner: str, port: str
) -> tuple[np.ndarray, np.ndarray]:
    """The section's nodes in the displacement's support; raise if nothing couples."""
    inside, points = _supported_nodes(displacement, setup.nodes[setup.section])
    if not len(setup.ip) and not inside.any():
        raise ValueError(
            f"{owner}: no element of wave port {port!r}'s section has a "
            "photoelastic_tensor and no node of the section lies in the "
            "displacement's support, so the overlap is zero. Set a "
            "photoelastic_tensor on the optical material, or align the "
            "displacement's mesh with the section (same x, y in m)."
        )
    return inside, points


class _Solved(NamedTuple):
    """The optical mode's constants: the node gradient, indices and tensor weights."""

    grad: np.ndarray
    n_eff: jax.Array
    n_g: jax.Array
    mode: int
    weights: Any


def _solve_optical(model: Model, setup: _Setup, port: str, mode: int | None) -> _Solved:
    """The one optical solve set: the mode, its node gradient and its weights."""
    optical = _optical_only(model)
    grad, n_eff, n_g, found = _node_gradient(optical, port, setup.frequency, mode)
    weights = None
    if len(setup.ip):
        _, _, weights, _ = _mode_weights(
            optical,
            port,
            setup.frequency,
            mode,
            setup.faces,
            tensor=True,
            rule=_weights_rule(),
        )
    return _Solved(grad, n_eff, n_g, found, weights)


def _parts(
    model: Model,
    setup: _Setup,
    solved: _Solved,
    displacement: FieldMap,
    *,
    inside: np.ndarray,
    points: np.ndarray,
    owner: str,
    port: str,
) -> tuple[jax.Array, jax.Array]:
    """The moving-boundary and photoelastic overlaps of one displacement."""
    crystal, ip = setup.crystal, setup.ip
    mb = jnp.zeros((), dtype=jnp.complex128)
    pe = jnp.zeros((), dtype=jnp.complex128)
    if len(ip):
        mean_s = _mean_strain_of(
            displacement,
            model.mesh,
            setup.tets[crystal.rows[ip]],
            offset=jnp.zeros(2),
            owner=f"{owner} (wave port {port!r})",
            moved={},
        )
        eta = jnp.einsum("nIJ,nJ->nI", crystal.p[ip], mean_s)
        pe = _contribution(
            solved.weights[crystal.rows[ip]], crystal.eps[ip], eta[None]
        )[0]
    if inside.any():
        w = jnp.stack([displacement[c].at(points) for c in "xy"], axis=1)
        g = jnp.asarray(solved.grad[setup.section[inside]][:, :2])
        mb = jnp.sum(g * w).astype(jnp.complex128)
    return mb, pe.astype(jnp.complex128)


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
    setup = _setup(
        model,
        port,
        frequency,
        _OWNER,
        "the optomechanical overlap",
        through="the displacement or the photoelastic_tensor",
        displacement=displacement,
    )
    inside, points = _check_support(setup, displacement, _OWNER, port)
    solved = _solve_optical(model, setup, port, mode)
    mb, pe = _parts(
        model,
        setup,
        solved,
        displacement,
        inside=inside,
        points=points,
        owner=_OWNER,
        port=port,
    )
    return OptomechanicalOverlap(
        port=port,
        mode=solved.mode,
        frequency=setup.frequency,
        n_eff=solved.n_eff,
        n_g=solved.n_g,
        moving_boundary=mb,
        photoelastic=pe,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class OptomechanicalCoupling:
    r"""The vacuum optomechanical coupling of mechanical modes to a wave port's mode.

    For a mechanical mode of angular frequency :math:`\Omega` that stores
    1 J/m, with the optical mode at the fixed angular frequency
    :math:`\omega = 2\pi f`, ``total`` is

    .. math::

        \tilde g_0 = \frac{\omega}{n_g}\,
        \frac{\partial n_\mathrm{eff}}{\partial\alpha}\,
        \sqrt{\frac{\hbar\,\mathrm{Re}\,\Omega}{4}},
        \qquad g_0(L) = \frac{\tilde g_0}{\sqrt{L}}

    in rad s\ :sup:`-1` m\ :sup:`1/2`, with ``moving_boundary`` and
    ``photoelastic`` the two parts of :math:`\partial n_\mathrm{eff}/\partial\alpha`
    scaled the same way. The values are complex: real to round-off for a
    lossless mode, complex for a lossy one (the displacement is). Their sign
    follows the mode's phase convention (largest displacement coefficient
    real and positive) and flips with it: :math:`|g_0|` is the convention-free
    number.

    Attributes:
        overlap: The `OptomechanicalOverlap` of each mode, arrays of
            shape ``(n,)`` over the modes (``n_eff``, ``n_g`` are scalars:
            one optical solve serves every mode).
        omega_m: The mechanical modes' complex angular frequencies, rad/s,
            ``(n,)``.
        moving_boundary: The moving-boundary part of :math:`\tilde g_0`, ``(n,)``.
        photoelastic: The photoelastic part of :math:`\tilde g_0`, ``(n,)``.
        modes: The numbers of the mechanical modes (``MechanicalEigenmodeResult``
            order, 0 = nearest its target), in the order of the arrays.
    """

    overlap: OptomechanicalOverlap
    omega_m: jax.Array
    moving_boundary: jax.Array
    photoelastic: jax.Array
    modes: tuple[int, ...]

    @property
    def total(self) -> jax.Array:
        r""":math:`\tilde g_0`, ``(n,)``: the two parts' sum."""
        return self.moving_boundary + self.photoelastic

    def g0(self, length: float) -> jax.Array:
        r""":math:`g_0 = \tilde g_0/\sqrt{L}` of a guide of length ``length``, rad/s.

        The parts follow by the same division (``moving_boundary`` and
        ``photoelastic`` over :math:`\sqrt{L}`). The coupling of a length
        :math:`L` treats it as a cavity: the optical mode shifts by
        :math:`\partial\omega_c/\partial\alpha = -(\omega/n_g)\,
        \partial n_\mathrm{eff}/\partial\alpha` and the mode's effective
        mass is :math:`2L/(\mathrm{Re}\,\Omega)^2`. Angular units: divide by
        :math:`2\pi` for a rate in Hz.

        Raises:
            ValueError: ``length`` is not positive.
        """
        if not is_traced(length) and not float(length) > 0:
            raise ValueError(f"g0 needs a positive length in m, got {length!r}.")
        return self.total / jnp.sqrt(length)

    def __repr__(self) -> str:
        head = f"OptomechanicalCoupling(port {self.overlap.port!r}, modes {self.modes}"
        try:
            g = np.asarray(self.total)
            f = np.real(np.asarray(self.omega_m)) / (2 * np.pi)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"{head})"
        rows = ", ".join(
            f"{a:.6g} Hz: |g0 sqrt(L)| {abs(b):.4g}" for a, b in zip(f, g, strict=True)
        )
        return f"{head}, [{rows}] rad/s m^1/2)"


register_fields(
    OptomechanicalCoupling, ("overlap", "omega_m", "moving_boundary", "photoelastic")
)


def _modes(mechanical: MechanicalEigenmodeResult, selection: Any) -> tuple[int, ...]:
    """The non-negative mode numbers ``selection`` names, ``None`` for every mode."""
    n = mechanical.n_modes
    if selection is None:
        return tuple(range(n))
    picked = [selection] if isinstance(selection, int | np.integer) else selection
    modes = []
    for k in picked:
        if not isinstance(k, int | np.integer):
            raise TypeError(
                f"{_COUPLING}: mechanical_modes takes an int or a sequence of "
                f"ints, got {k!r}."
            )
        mechanical.displacement(int(k))  # raises IndexError for a missing mode
        modes.append(int(k) % n)
    if not modes:
        raise ValueError(f"{_COUPLING}: mechanical_modes is empty.")
    return tuple(modes)


def optomechanical_coupling(
    model: Model,
    mechanical: MechanicalEigenmodeResult,
    *,
    port: str,
    frequency: float,
    mode: int | None = None,
    mechanical_modes: int | Sequence[int] | None = None,
) -> OptomechanicalCoupling:
    r"""The vacuum coupling :math:`g_0` of mechanical modes to a wave port's mode.

    For each requested mode :math:`u` of a `gfem.MechanicalEigenmode` solve
    (:math:`q = 0`, stored energy 1 J/m) the overlap of
    `optomechanical_overlap` is turned into the coupling of a length
    :math:`L` of the guide treated as a cavity. A displacement amplitude
    :math:`\alpha` shifts the optical resonance by
    :math:`\partial\omega_c/\partial\alpha = -(\omega/n_g)\,
    \partial n_\mathrm{eff}/\partial\alpha` at fixed :math:`\omega`; the mode's
    effective mass is :math:`m_\alpha = L\int\rho|u|^2\,dA =
    2L/(\mathrm{Re}\,\Omega)^2`, so its zero-point amplitude is
    :math:`\alpha_\mathrm{zpf} = \sqrt{\hbar/(2 m_\alpha\,\mathrm{Re}\,\Omega)}
    = \sqrt{\hbar\,\mathrm{Re}\,\Omega/(4L)}` and

    .. math::

        \tilde g_0 = g_0\sqrt{L} = \frac{\omega}{n_g}\,
        \frac{\partial n_\mathrm{eff}}{\partial\alpha}\,
        \sqrt{\frac{\hbar\,\mathrm{Re}\,\Omega}{4}}

    with :math:`n_g` the port mode's group index (material dispersion
    included) and :math:`\hbar` from `scipy.constants`. The sign is the
    convention :math:`g_0 = -(\partial\omega_c/\partial\alpha)\,\alpha_\mathrm{zpf}`,
    positive when the mode's displacement raises the effective index, and it
    follows the mode's phase convention: read :math:`|g_0|`. The modes
    are intramodal forward couplings at :math:`q = 0`; the roto-optic term is
    not counted.

    The optical mode, its node gradient and its weights are solved once, for
    every requested mechanical mode. `jax.grad` reaches every input of the
    mechanical solve (stiffness, density, quality factor, piezoelectric
    tensor, permittivity, node coordinates of the mechanical mesh) through
    :math:`\mathrm{Re}\,\Omega` and the mode field, and the
    ``photoelastic_tensor``; a clustered (degenerate) mode is refused by the
    mechanical solve's backward pass, as for any scalar of the mode. The
    optical model's other inputs are refused (the mode is held constant).

    Args:
        model: The optical model, as for `optomechanical_overlap`.
        mechanical: A `gfem.MechanicalEigenmodeResult` at :math:`q = 0` on a
            native 2D mesh with the same global :math:`(x, y)` (m) as the
            optical section.
        port: The wave port's name.
        frequency: One optical frequency, Hz.
        mode: Which optical mode (1 = the most propagating), or ``None`` for
            the wave port's own.
        mechanical_modes: Which mechanical modes (numbers in ``mechanical``,
            negative counting from the last): an int, a sequence, or ``None``
            for every mode.

    Returns:
        The coupling, parts and total, per requested mode.

    Raises:
        TypeError: ``mechanical`` is not a `gfem.MechanicalEigenmodeResult`, or
            ``model`` is not a `gfem.Model`.
        IndexError: A number of ``mechanical_modes`` is not a mode of ``mechanical``.
        ValueError: ``mechanical`` has an axial wavenumber other than 0,
            ``mechanical_modes`` is empty, and every ``ValueError`` of
            `optomechanical_overlap`.
        NotImplementedError: As `optomechanical_overlap`.
        NonDifferentiableError: As `optomechanical_overlap`.
    """
    from gfem.results import MechanicalEigenmodeResult

    if not isinstance(mechanical, MechanicalEigenmodeResult):
        raise TypeError(
            f"{_COUPLING} takes a gfem.MechanicalEigenmodeResult (the solve of a "
            f"gfem.MechanicalEigenmode), got {type(mechanical).__name__}; give a "
            "displacement FieldMap to optomechanical_overlap instead."
        )
    if mechanical.wavenumber != 0:
        raise ValueError(
            f"{_COUPLING}: the mechanical modes have the axial wavenumber "
            f"q = {mechanical.wavenumber:g} rad/m; only q = 0 is read (a mode "
            "with q != 0 averages to zero against an optical mode with itself)."
        )
    modes = _modes(mechanical, mechanical_modes)
    first = mechanical.displacement(modes[0])
    setup = _setup(
        model,
        port,
        frequency,
        _COUPLING,
        "the optomechanical coupling",
        through="the mechanical model or the photoelastic_tensor",
        displacement=first,
    )
    inside, points = _check_support(setup, first, _COUPLING, port)
    solved = _solve_optical(model, setup, port, mode)
    parts = [
        _parts(
            model,
            setup,
            solved,
            mechanical.displacement(m),
            inside=inside,
            points=points,
            owner=_COUPLING,
            port=port,
        )
        for m in modes
    ]
    mb = jnp.stack([a for a, _ in parts])
    pe = jnp.stack([b for _, b in parts])
    omega_m = jnp.asarray(mechanical.omega)[jnp.asarray(modes)]
    scale = (
        2
        * jnp.pi
        * setup.frequency
        / solved.n_g
        * jnp.sqrt(scipy.constants.hbar * jnp.real(omega_m) / 4)
    )
    return OptomechanicalCoupling(
        overlap=OptomechanicalOverlap(
            port=port,
            mode=solved.mode,
            frequency=setup.frequency,
            n_eff=solved.n_eff,
            n_g=solved.n_g,
            moving_boundary=mb,
            photoelastic=pe,
        ),
        omega_m=omega_m,
        moving_boundary=scale * mb,
        photoelastic=scale * pe,
        modes=modes,
    )
