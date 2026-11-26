r"""The acousto-optic shift of a wave port's mode from a piezoelectric result.

An acoustic wave in a crystal strains it (:math:`S`, engineering Voigt strain)
and, through the piezoelectric effect, carries an electric field
:math:`E = -\nabla\phi`. Both change the optical permittivity of a material
that has a ``photoelastic_tensor`` (:math:`p^E_{IJ}`) and an
``electro_optic_tensor`` (:math:`r^S_{Ik}`, clamped):

.. math::

    \Delta(\varepsilon_r^{-1})_I = \sum_J p_{IJ} S_J + \sum_k r_{Ik} E_k,
    \qquad
    \Delta\varepsilon_r = -\varepsilon_r\,\Delta(\varepsilon_r^{-1})\,
    \varepsilon_r

`acousto_optic_shift` gives the change of a wave port's effective index to
first order in that change, from one mode solve, for every frequency a
`gfem.Piezoelectric` solve kept its fields at, per volt on one terminal: a
complex phasor under :math:`e^{+i\Omega t}`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._checks import is_traced
from gfem._dim2 import refuse_entry
from gfem._materials import (
    _VOIGT_ROW,
    Material,
    stiffness_in_frame,
)
from gfem._optical_shift import (
    _check_model,
    _frequency,
    _lossy,
    _mode_weights,
    _region_rows,
    _section,
    _wave_port,
)
from gfem._pytree import readonly, register_fields
from gfem.electro_optic import _permittivity, _tensor
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from gfem.model import Model
    from gfem.results import PiezoelectricResult

__all__ = ["AcoustoOpticShift", "acousto_optic_shift"]

_OWNER = "acousto_optic_shift"


@dataclasses.dataclass(frozen=True, eq=False)
class AcoustoOpticShift:
    r"""The first-order acousto-optic shift of a wave port's mode, per volt.

    The mode is solved once on the optical model and held constant: the
    shift is exactly linear in the acoustic fields and in the photoelastic
    and electro-optic tensors. Per volt means per volt on the terminal whose
    kept fields were read (1 V there, 0 V on the other terminals and the
    ground). With a drive :math:`\mathrm{Re}(V e^{i\Omega t})` on that
    terminal the effective index is, to first order,
    :math:`n(t) = n_\mathrm{eff} + \mathrm{Re}(\delta n\,V e^{i\Omega t})`,
    :math:`\delta n` the complex ``delta_n_eff_per_volt`` of the acoustic
    frequency :math:`\Omega/2\pi`.

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        terminal: The excited terminal whose fields were read.
        frequency: The optical frequency, Hz.
        acoustic_frequency: The acoustic frequencies, Hz, ``(n_f,)``, those the
            result kept fields at for ``terminal`` (static, not differentiable).
        n_eff: The unperturbed effective index (real; the section is
            lossless), a constant.
        delta_n_eff_per_volt: :math:`\delta n = \partial n_\mathrm{eff} /
            \partial V` at each acoustic frequency, 1/V, complex ``(n_f,)``:
            ``photoelastic_per_volt + electro_optic_per_volt``.
        photoelastic_per_volt: The part of ``delta_n_eff_per_volt`` through
            the ``photoelastic_tensor`` (the strain), complex ``(n_f,)``.
        electro_optic_per_volt: The part through the ``electro_optic_tensor``
            (the piezoelectric field), complex ``(n_f,)``.
        phase_per_length_per_volt: :math:`k_0\,\delta n`, rad/(V m), complex
            ``(n_f,)``: the phase delay per unit length of guide and per volt
            (`gfem.conventions`).
    """

    port: str
    mode: int
    terminal: str
    frequency: float
    acoustic_frequency: np.ndarray
    n_eff: jax.Array
    delta_n_eff_per_volt: jax.Array
    photoelastic_per_volt: jax.Array
    electro_optic_per_volt: jax.Array
    phase_per_length_per_volt: jax.Array

    @property
    def v_pi_l(self) -> jax.Array:
        r"""The voltage-length product of this arm at each acoustic frequency, V m.

        :math:`V_\pi L = \lambda / (2\,|\delta n|)`, :math:`\lambda = c/f` the
        optical wavelength: the amplitude of the drive times the length that
        turns the phase by :math:`\pi`; infinite for a zero shift. ``(n_f,)``.
        """
        wavelength = scipy.constants.c / self.frequency
        return wavelength / (2 * jnp.abs(self.delta_n_eff_per_volt))

    def __repr__(self) -> str:
        head = f"AcoustoOpticShift(port {self.port!r}, mode {self.mode}"
        try:
            n = float(np.asarray(self.n_eff))
            vl = np.asarray(self.v_pi_l)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"{head}, terminal {self.terminal!r})"
        f = self.acoustic_frequency
        return (
            f"{head}, terminal {self.terminal!r}, {self.frequency:.6g} Hz, "
            f"n_eff {n:.6f}, {len(f)} acoustic frequencies "
            f"{f.min():.6g} to {f.max():.6g} Hz, v_pi_l {vl.min():.6g} to "
            f"{vl.max():.6g} V m)"
        )


register_fields(
    AcoustoOpticShift,
    (
        "n_eff",
        "delta_n_eff_per_volt",
        "photoelastic_per_volt",
        "electro_optic_per_volt",
        "phase_per_length_per_volt",
    ),
)


def _check_acoustic(acoustic: Any, terminal: str | None) -> tuple[str, np.ndarray]:
    """The excited terminal and the kept acoustic frequencies for it.

    Raises:
        TypeError: ``acoustic`` is not a piezoelectric result.
        ValueError: The result kept no fields, ``terminal`` is ambiguous or
            unknown, or no field was kept for it.
    """
    from gfem.results import PiezoelectricResult

    if not isinstance(acoustic, PiezoelectricResult):
        raise TypeError(
            f"{_OWNER} takes a gfem.PiezoelectricResult (the result of a "
            f"gfem.Piezoelectric solve), got {type(acoustic).__name__}."
        )
    if not acoustic.kept:
        raise ValueError(
            f"{_OWNER}: the piezoelectric result kept no fields: solve with "
            "Piezoelectric(..., keep_fields=True), or keep_fields=[frequencies], "
            "to keep the displacement and potential the shift reads."
        )
    terminals = acoustic.terminals
    names = ", ".join(repr(t) for t in terminals)
    if terminal is None:
        if len(terminals) > 1:
            raise ValueError(
                f"{_OWNER}: the result has terminals {names}; pass terminal= to "
                "choose the one whose fields to read (the shift of several "
                "terminals is the sum of each terminal's shift times its drive)."
            )
        terminal = terminals[0]
    elif terminal not in terminals:
        raise ValueError(
            f"{_OWNER}: terminal {terminal!r} is not a terminal of the result "
            f"(terminals {names})."
        )
    kept = np.array([f for f, name in acoustic.kept if name == terminal], float)
    if not len(kept):
        listed = ", ".join(repr(t) for t in sorted({t for _, t in acoustic.kept}))
        raise ValueError(
            f"{_OWNER}: the result kept no fields for terminal {terminal!r} "
            f"(it kept {listed}): solve with keep_fields=True."
        )
    return terminal, kept


def _check_displacement(displacement: Any, owner: str) -> None:
    """Raise unless ``displacement`` is a q = 0 `gfem.FieldMap` of x, y, z on a 2D mesh.

    Raises:
        TypeError: ``displacement`` is not a `gfem.FieldMap`.
        ValueError: It lacks a component, is on a 3D mesh, or has an axial
            wavenumber other than 0.
    """
    from gfem._field import FieldMap

    if not isinstance(displacement, FieldMap):
        raise TypeError(
            f"{owner} takes a displacement gfem.FieldMap with the components "
            f"'x', 'y', 'z' (a mechanical mode's, or a piezoelectric "
            f"solve's), got {type(displacement).__name__}."
        )
    missing = [c for c in "xyz" if c not in displacement]
    if missing:
        raise ValueError(
            f"{owner}: the displacement has the components "
            f"{', '.join(repr(k) for k in displacement)}; it needs 'x', 'y' "
            "and 'z'."
        )
    if any(displacement[c].mesh.dim != 2 for c in "xyz"):
        raise ValueError(
            f"{owner}: the displacement must live on a native 2D mesh, the "
            "waveguide's cross-section."
        )
    if displacement.wavenumber != 0:
        raise ValueError(
            f"{owner}: the displacement has the axial wavenumber "
            f"q = {displacement.wavenumber:g} rad/m; only q = 0 is read (a mode "
            "with q != 0 averages to zero against an optical mode with itself)."
        )


def _offset(offset: Any) -> jax.Array:
    """The offset as a ``(2,)`` array of metres.

    Raises:
        ValueError: ``offset`` is not two finite numbers.
    """
    traced = is_traced(offset)
    # a tuple with a traced entry stacks through JAX; NumPy cannot convert it
    arr = jnp.asarray(offset, dtype=jnp.float64) if traced else np.asarray(offset)
    if arr.shape != (2,):
        raise ValueError(
            f"{_OWNER}: offset is the (dx, dy) shift in m of the acoustic section "
            f"against the optical one, shape (2,), got shape {arr.shape}."
        )
    if not traced and not np.isfinite(np.asarray(arr, float)).all():
        raise ValueError(f"{_OWNER}: offset must be finite, got {offset!r}.")
    return jnp.asarray(arr, dtype=jnp.float64)


def _check_plane(
    mesh: Any,
    faces: np.ndarray,
    port: str,
    owner: str = _OWNER,
    section: str = "acoustic",
) -> None:
    """Raise unless the port's section lies in a plane :math:`z` = const.

    The acoustic result is invariant along z, and the shift reads it at the
    section's points projected onto :math:`(x, y)`: another orientation of the
    section would read the fields along one line, with no error.
    """
    z = np.asarray(mesh.nodes, float)[np.asarray(mesh.triangles)[faces]][..., 2]
    extent = np.ptp(np.asarray(mesh.nodes, float), axis=0).max()
    if np.ptp(z) > 1e-9 * extent:
        raise ValueError(
            f"{owner}: wave port {port!r} is not in a plane z = const: the "
            f"{section} section is invariant along z, so the guide must run along z."
        )


def _check_optical_fixed(
    model: Model,
    owner: str = _OWNER,
    what: str = "the acousto-optic shift",
    through: str = "the acoustic result, the photoelastic_tensor, the "
    "electro_optic_tensor or the offset",
) -> None:
    """Raise `NonDifferentiableError` for a traced input of the port mode problem.

    The mode is held constant; the refusal comes before the acoustic fields
    are read rather than after.
    """
    names = ("eps_r", "tan_delta", "mu_r", "conductivity")
    for material in model.materials.values():
        for name in names:
            if is_traced(getattr(material, name, None)):
                raise NonDifferentiableError(
                    f"{owner}: gfem cannot differentiate {what} "
                    f"with respect to the optical model's {name}: the mode is held "
                    "constant (first order in the displacement). Wrap it in "
                    "jax.lax.stop_gradient to use its value without its "
                    f"derivative. Differentiate through {through}."
                )


def _photoelastic(material: Any, pos: np.ndarray) -> Any:
    """The photoelastic tensor of elements ``pos``, ``(n, 6, 6)``, global frame."""
    p = material.photoelastic_tensor
    p = np.asarray(p) if isinstance(material, Material) else jnp.asarray(p)
    if p.ndim == 3:
        p = p[pos]
    return jnp.broadcast_to(
        jnp.asarray(stiffness_in_frame(p, material.axes)), (len(pos), 6, 6)
    )


@dataclasses.dataclass(frozen=True)
class _Crystal:
    """The section's elements with a tensor: rows into the section's tetrahedra.

    ``rows`` indexes the tetrahedra (and triangles) of the section; ``eps`` is
    the optical permittivity of each, ``p`` and ``r`` its photoelastic and
    electro-optic tensors in the global frame (zero where it has none), and
    ``has_p`` and ``has_r`` say which of them it has.
    """

    rows: np.ndarray
    eps: Any
    p: Any
    r: Any
    has_p: np.ndarray
    has_r: np.ndarray


def _crystal(
    model: Model,
    port: str,
    tets: np.ndarray,
    owner: str = _OWNER,
    *,
    tensors: tuple[str, ...] = ("photoelastic_tensor", "electro_optic_tensor"),
    require: bool = True,
) -> _Crystal:
    """The section's elements with a photoelastic or electro-optic tensor.

    ``tensors`` names the tensors that count (the photoelastic one alone for the
    overlap of a displacement). Lossy regions raise whatever they carry. With no
    element holding a counted tensor, ``require`` raises and its absence gives
    an empty crystal.

    Raises:
        ValueError: No element behind the section has a tensor.
        NotImplementedError: The section is lossy.
    """
    mesh = model.mesh
    rows, eps, p, r, has_p, has_r = [], [], [], [], [], []
    regions = _region_rows(mesh, tets)
    for region, at in regions.items():
        material = model.materials[region]
        pos = np.searchsorted(mesh.elements(region), tets[at])
        loss = _lossy(material, pos)
        if loss is not None:
            raise NotImplementedError(
                f"{owner}: region {region!r} of wave port {port!r}'s section has "
                f"{loss}. The first-order shift weighs Re(E E^H), which holds for "
                "a lossless section only: give the section's regions a real eps_r "
                "with no tan_delta or conductivity (an optical window or "
                "EM-excluded electrodes keep the loss out of the section)."
            )
        photoelastic = (
            "photoelastic_tensor" in tensors
            and material.photoelastic_tensor is not None
        )
        electro_optic = (
            "electro_optic_tensor" in tensors
            and material.electro_optic_tensor is not None
        )
        if not (photoelastic or electro_optic):
            continue
        n = len(at)
        rows.append(at)
        eps.append(_permittivity(material, pos))
        p.append(_photoelastic(material, pos) if photoelastic else jnp.zeros((n, 6, 6)))
        r.append(_tensor(material, pos) if electro_optic else jnp.zeros((n, 6, 3)))
        has_p.append(np.full(n, photoelastic))
        has_r.append(np.full(n, electro_optic))
    if not rows:
        if not require:
            return _Crystal(
                np.zeros(0, int),
                jnp.zeros((0, 3, 3)),
                jnp.zeros((0, 6, 6)),
                jnp.zeros((0, 6, 3)),
                np.zeros(0, bool),
                np.zeros(0, bool),
            )
        names = ", ".join(repr(name) for name in regions)
        raise ValueError(
            f"{_OWNER}: no element of wave port {port!r}'s section has a "
            f"photoelastic_tensor or an electro_optic_tensor (its regions are "
            f"{names}), so the shift is zero. Set them on the crystal's optical "
            "material."
        )
    return _Crystal(
        np.concatenate(rows),
        jnp.concatenate(eps),
        jnp.concatenate(p),
        jnp.concatenate(r),
        np.concatenate(has_p),
        np.concatenate(has_r),
    )


def _points(mesh: Any, tets: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """The rule's points of ``tets`` in the optical section, ``(n q, 2)``, and weights.

    The points of the quadrature rule `_static_field` uses (exact for the
    gradient of a field of ``order``), projected onto :math:`(x, y)`.
    """
    from gfem import _probe

    bary, weights = _probe.engine().tetrahedron_rule(max(order - 1, 0))
    corners = np.asarray(mesh.nodes, float)[np.asarray(mesh.tetrahedra)[tets]]
    return np.einsum("qk,nkd->nqd", bary, corners)[..., :2].reshape(-1, 2), weights


def _moved(field: Any, offset: jax.Array, moved: dict[int, Any]) -> Any:
    r"""``field`` on its mesh translated by ``-offset``.

    Reading the translated field at :math:`x` is reading ``field`` at
    :math:`x + \mathrm{offset}`; the points stay fixed numbers and the
    translation is a node map, so the reading's Eulerian node derivative gives
    the exact derivative in ``offset``. A concrete zero offset leaves the field
    as it is. ``moved`` holds the translated meshes of this call, so a mesh
    shared by the fields and the frequencies is translated once.
    """
    if not is_traced(offset) and not np.any(np.asarray(offset)):
        return field
    mesh = field.mesh
    if id(mesh) not in moved:
        # the original is kept in the entry so its id stays unique
        moved[id(mesh)] = (mesh, mesh.with_nodes(mesh.nodes - offset))
    return dataclasses.replace(field, mesh=moved[id(mesh)][1])


def _read(what: str, owner: str, read: Any) -> Any:
    """``read()``, its out-of-domain error worded for the shift."""
    try:
        return read()
    except ValueError as err:
        raise ValueError(f"{owner}: reading the {what}: {err}") from err


def _mean_strain(
    acoustic: Any,
    frequencies: np.ndarray,
    terminal: str,
    mesh: Any,
    tets: np.ndarray,
    *,
    offset: jax.Array,
    owner: str,
    moved: dict[int, Any],
) -> jax.Array:
    r"""The mean engineering strain over each of ``tets``, ``(n_f, n, 6)``, per volt."""
    return jnp.stack(
        [
            _mean_strain_of(
                acoustic.displacement(f, terminal),
                mesh,
                tets,
                offset=offset,
                owner=owner,
                moved=moved,
            )
            for f in frequencies
        ]
    )


def _mean_strain_of(
    read: Any,
    mesh: Any,
    tets: np.ndarray,
    *,
    offset: jax.Array,
    owner: str,
    moved: dict[int, Any],
) -> jax.Array:
    r"""The mean engineering strain of a displacement over each of ``tets``, ``(n, 6)``.

    ``read`` is a `gfem.FieldMap` of the components ``x``, ``y``, ``z`` on a 2D
    mesh, read at the quadrature points of ``tets`` projected onto
    :math:`(x, y)`.
    """
    from gfem._field import FieldMap
    from gfem.measurements import strain

    points, weights = _points(mesh, tets, read["x"].order)
    displacement = FieldMap(
        {k: _moved(v, offset, moved) for k, v in read.items()},
        label="component",
        wavenumber=read.wavenumber,
    )
    s = _read("strain", owner, lambda: strain(displacement, points))
    return jnp.einsum("q,nqc->nc", weights, s.reshape(len(tets), len(weights), 6))


def _mean_field(
    acoustic: Any,
    frequencies: np.ndarray,
    terminal: str,
    mesh: Any,
    tets: np.ndarray,
    *,
    offset: jax.Array,
    owner: str,
    moved: dict[int, Any],
) -> jax.Array:
    r"""The mean :math:`-\nabla\phi` of each of ``tets``, ``(n_f, n, 3)``, V/m/V."""
    from gfem import _probe

    order = acoustic.potential(frequencies[0], terminal).order
    points, weights = _points(mesh, tets, order)
    out = []
    for f in frequencies:
        potential = _moved(acoustic.potential(f, terminal), offset, moved)
        g = _read("field", owner, lambda v=potential: _probe.gradient_at(v, points))
        mean = -jnp.einsum("q,nqc->nc", weights, g.reshape(len(tets), len(weights), 2))
        out.append(jnp.concatenate([mean, jnp.zeros((len(tets), 1))], axis=1))
    return jnp.stack(out)


def _contribution(w: Any, eps: Any, eta: Any) -> jax.Array:
    r"""The shift of a mean :math:`\eta = \Delta(\varepsilon^{-1})` per frequency.

    :math:`\sum_t W_t : (-\varepsilon_t \eta_t \varepsilon_t)`, ``(n_f,)``.
    """
    delta_eps = -eps @ eta[..., _VOIGT_ROW] @ eps
    return jnp.einsum("nij,fnij->f", w, delta_eps)


def acousto_optic_shift(
    model: Model,
    acoustic: PiezoelectricResult,
    *,
    port: str,
    frequency: float,
    terminal: str | None = None,
    mode: int | None = None,
    offset: Any = (0.0, 0.0),
) -> AcoustoOpticShift:
    r"""The first-order acousto-optic shift of a wave port's mode, per volt.

    Solves the port's mode on ``model`` (a `gfem.PortMode` at one frequency)
    and gives the change of its effective index caused by the strain and the
    piezoelectric field of every kept frequency of ``acoustic``, through each
    element's ``photoelastic_tensor`` and ``electro_optic_tensor``, to first
    order:

    .. math::

        \delta n_\mathrm{eff} = \sum_t W_t : \Delta\varepsilon_t, \qquad
        W_t = \frac{c\varepsilon_0}{4}\,\mathrm{Re}\int_t E_0 E_0^H\,dA,
        \qquad
        \Delta\varepsilon_t = -\varepsilon_t\,(p_t \bar S_t + r_t \bar E_t)\,
            \varepsilon_t

    :math:`E_0` is the mode at 1 W, :math:`t` runs over the triangles of the
    port's section, :math:`\varepsilon_t`, :math:`p_t` and :math:`r_t` are the
    optical permittivity, the photoelastic and the electro-optic tensor of
    the tetrahedron behind it (in the global frame, rotated from the
    material's ``axes``), and :math:`\bar S_t` and :math:`\bar E_t` are the
    means of the engineering strain and of the field
    :math:`E = -\nabla\phi = (-\partial_x\phi, -\partial_y\phi, 0)` over that
    tetrahedron, read in the acoustic section (a 2D mesh) at the tetrahedron's
    quadrature points projected onto :math:`(x, y)` and moved by ``offset``.
    The result is complex: the phasor of the acoustic frequency, with
    :math:`n(t) = n_\mathrm{eff} + \mathrm{Re}(\delta n\,V e^{i\Omega t})` for
    a drive :math:`\mathrm{Re}(V e^{i\Omega t})` on the terminal. The
    photoelastic part is the :math:`p` term alone and the electro-optic part
    the :math:`r` term alone.

    ``model`` is the optical model: a lossless section with ``eps_r`` at the
    optical frequency and the tensors, the acoustic crystal given as another
    `gfem.Material` of the acoustic model. ``photoelastic_tensor`` is
    :math:`p^E` and ``electro_optic_tensor`` the clamped :math:`r^S`: the
    strain is counted through :math:`p` and the piezoelectric field through
    :math:`r`, so a free :math:`r^T` would count the strain twice. Give the
    optical and the acoustic crystal the same ``axes``, which the shift cannot
    check. Elements with a ``photoelastic_tensor`` must lie in the acoustic
    result's mechanical domain, and those with an ``electro_optic_tensor``
    in its potential's support. Only the photoelastic and electro-optic terms
    are counted: the moving boundaries and the rotation of the crystal by
    the displacement (roto-optic) are not, which is exact to first order for
    an acoustic mode whose displacement moves no boundary of the optical
    section (shear polarized along the guide) and an approximation for
    in-plane motion.

    `jax.grad` reaches the ``photoelastic_tensor`` and ``electro_optic_tensor``
    of a `gfem.MaterialField`, every input of the piezoelectric solve through
    the kept fields (its materials, geometry and the acoustic node maps of
    `gfem.Mesh.with_nodes`, which are Eulerian as in `gfem.strain`: the fields
    are read at fixed optical points), and ``offset``. The optical mode is held
    constant, so the optical model's own inputs are not reached: the mode part
    of a waveguide width's derivative is a finite difference over a remesh.

    The shift is linear in the fields, so the shifts of several terminals add,
    each times its drive. A push-pull modulator's two arms, ``a`` and ``b``
    (two ports on one acoustic result), give :math:`V_\pi L = \lambda /
    (2\,|\delta n_a - \delta n_b|)` per volt on the terminal.

    Example:
        The photoelastic and electro-optic parts of one arm at every kept
        frequency::

            acoustic = gfem.solve(
                crystal_model,
                gfem.Piezoelectric(
                    freqs, terminals=["hot"], ground=["gnd"], keep_fields=True
                ),
            )
            f = scipy.constants.c / 1.55e-6
            a = gfem.acousto_optic_shift(optical, acoustic, port="a", frequency=f)
            print(
                a.acoustic_frequency,
                a.photoelastic_per_volt,
                a.electro_optic_per_volt,
                a.v_pi_l * 100,
            )

    Args:
        model: The optical model.
        acoustic: A `gfem.PiezoelectricResult` that kept its fields
            (``keep_fields=True``).
        port: The wave port's name.
        frequency: One optical frequency, Hz.
        terminal: The terminal whose kept fields (1 V on it) are read, or
            ``None`` for the result's only terminal.
        mode: Which mode (1 = the most propagating), or ``None`` for the wave
            port's own.
        offset: ``(dx, dy)`` in m, where the optical section sits relative to
            the acoustic one: the acoustic fields are read at the optical
            points plus ``offset``.

    Returns:
        The shift at each kept acoustic frequency of ``terminal``.

    Raises:
        TypeError: ``model`` is not a `gfem.Model`, or ``acoustic`` is not a
            `gfem.PiezoelectricResult`.
        ValueError: ``port`` is not a wave port of the model, a sequence of
            frequencies is given, ``acoustic`` kept no fields, ``terminal``
            is unknown or ambiguous, ``offset`` is not two finite numbers, no
            element of the section has a tensor, or an element lies outside
            the acoustic mechanical domain (with a ``photoelastic_tensor``)
            or the potential's support (with an ``electro_optic_tensor``; the
            message names the point), or the port's section is not in a plane
            z = const.
        NotImplementedError: The model is on a 2D mesh (a 2D `gfem.PortMode`
            is a line mode) or the section is lossy (a complex ``eps_r``, a
            ``tan_delta`` or ``conductivity`` other than 0, or a permittivity
            law).
        NonDifferentiableError: Under `jax.grad`, an input of the optical
            model other than the tensors (its ``eps_r``, ``tan_delta``,
            ``mu_r``, ``conductivity`` or node coordinates): the mode is held
            constant.
    """
    refuse_entry(getattr(model, "mesh", None), _OWNER)
    _check_model(model, _OWNER)
    name, kept = _check_acoustic(acoustic, terminal)
    f = _frequency(frequency, _OWNER)
    shift = _offset(offset)
    faces, tets = _section(model, _wave_port(model, port))
    crystal = _crystal(model, port, tets)
    _check_optical_fixed(model)
    if is_traced(model.mesh.nodes):
        raise NonDifferentiableError(
            f"{_OWNER}: gfem cannot differentiate the acousto-optic shift with "
            "respect to the optical model's node coordinates: the mode and the "
            "points the acoustic fields are read at are held constant. Wrap them "
            "in jax.lax.stop_gradient to use their value without their "
            "derivative."
        )
    _check_plane(model.mesh, faces, port)
    owner = f"{_OWNER} (wave port {port!r})"
    rows = crystal.rows
    ip, ir = np.flatnonzero(crystal.has_p), np.flatnonzero(crystal.has_r)
    n_f = len(kept)
    zero = jnp.zeros(n_f, dtype=jnp.complex128)
    pe = er = zero
    mean_s = mean_e = None
    moved: dict[int, Any] = {}
    if len(ip):
        mean_s = _mean_strain(
            acoustic,
            kept,
            name,
            model.mesh,
            tets[rows[ip]],
            offset=shift,
            owner=owner,
            moved=moved,
        )
    if len(ir):
        mean_e = _mean_field(
            acoustic,
            kept,
            name,
            model.mesh,
            tets[rows[ir]],
            offset=shift,
            owner=owner,
            moved=moved,
        )
    found, n_eff, weights, k0 = _mode_weights(
        model, port, f, mode, faces, tensor=True, rule=_weights_rule()
    )
    if mean_s is not None:
        eta = jnp.einsum("nIJ,fnJ->fnI", crystal.p[ip], mean_s)
        pe = _contribution(weights[rows[ip]], crystal.eps[ip], eta)
    if mean_e is not None:
        eta = jnp.einsum("nIk,fnk->fnI", crystal.r[ir], mean_e)
        er = _contribution(weights[rows[ir]], crystal.eps[ir], eta)
    delta = pe + er
    return AcoustoOpticShift(
        port=port,
        mode=found,
        terminal=name,
        frequency=f,
        acoustic_frequency=readonly(kept),
        n_eff=n_eff,
        delta_n_eff_per_volt=delta,
        photoelastic_per_volt=pe,
        electro_optic_per_volt=er,
        phase_per_length_per_volt=k0 * delta,
    )


def _weights_rule() -> type:
    from gfem._engine.electro_optic import AcoustoOpticWeightsRule

    return AcoustoOpticWeightsRule
