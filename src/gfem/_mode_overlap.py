"""The effect-free part of the overlaps of a port's modes: fields and columns.

`gfem.kerr` and `gfem.three_wave` integrate products of the modes that
`gfem.solve` found with `gfem.PortMode` over the port's cross-section. What
they share lives here: the elements behind the section, the DOF vectors of
the results, the checks that a model is the one the modes were solved on, the
mode at the quadrature points of the section and the cache of those points.
Each effect reads its own material values through a reader it passes in.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import partial_eval

from gfem._checks import is_traced
from gfem.errors import NonDifferentiableError
from gfem.results import PortModeResult

if TYPE_CHECKING:
    from gfem._engine.kerr_columns import Columns as _Columns
    from gfem.model import Model


_NODES = (
    "{owner} reads the mode at fixed quadrature points while gfem differentiates "
    "the node coordinates: the derivative of an overlap on a moving section is "
    "not available in this version. Differentiate with respect to the materials, "
    "or wrap the node coordinates in jax.lax.stop_gradient."
)
_SOLVED_CONSTANT = (
    "{owner} differentiates the model's node coordinates, but the mode was solved "
    "where they were constants (outside the differentiated function, or on another "
    "model): its field carries no node derivative and the gradient would miss it. "
    "Solve the mode with gfem.solve on the model passed here, inside the function."
)
_SOLVED_TRACED = (
    "{owner}: the mode was solved with node coordinates traced by jax.grad, but "
    "the model passed here has constant ones: the overlap would miss the motion of "
    "its section. Pass the model the mode was solved on."
)


def _columns(space: Any, raised: int = 0) -> _Columns:
    """The quadrature columns of ``space`` (`gfem._engine.kerr_columns.columns`)."""
    from gfem._engine.kerr_columns import columns

    return columns(space, raised)


def _elements(model: Model, section: Any) -> tuple[np.ndarray, np.ndarray]:
    """The solve-domain element behind each section element, and its region tag."""
    mesh = model.mesh
    if len(section.normal) == 3:
        from gfem._engine.port_mode_sensitivity import owner_tetrahedra

        cells = owner_tetrahedra(model, section)
        return cells, np.asarray(mesh.tetrahedron_tags)[cells]
    cells = np.asarray(section.cells)
    return cells, np.asarray(mesh._topology.volume_tags)[cells]  # noqa: SLF001


def _per_element(value: Any, pos: np.ndarray) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    return jnp.broadcast_to(arr, pos.shape) if arr.ndim == 0 else arr[pos]


@dataclasses.dataclass(frozen=True, eq=False)
class _Waves:
    """The waves of the results: DOF vectors, frequencies, columns, materials.

    ``materials`` is what the caller's reader returned; it has ``keep``
    ``(n_el,)``, False on a perfectly matched layer's triangles.
    """

    port: str
    modes: tuple[int, ...]
    frequency: np.ndarray
    vectors: jax.Array
    section: Any
    columns: _Columns
    materials: Any
    moved: tuple[jax.Array, jax.Array] | None = None
    bend: Any = None

    def metric(self) -> jax.Array | None:
        r"""``(n_el, n_q)`` the bend's :math:`h = 1 + \xi/R` at the points, or ``None``.

        :math:`R` is the results' (traced when the solve is), :math:`\xi` the
        distance of the real point from the reference arc.
        """
        if self.bend is None:
            return None
        bend, cols = self.bend, self.columns
        # the section of a traced result carries no bend: the real point is
        # read from the plane coordinates, the arc from the result's bend
        points = cols.points
        assert points is not None  # noqa: S101 - a planar section's columns
        shape = points.shape[:2]
        real = self.section.to_global(np.reshape(points, (-1, 2)))
        xi = (real - np.asarray(bend.at)) @ np.asarray(bend.outward)
        xi = jnp.asarray(xi.reshape(shape))
        return 1 + xi / self.bend.radius

    @property
    def weights(self) -> jax.Array:
        r"""``(n_el, n_q)`` quadrature weights, m² (m), zero on a layer.

        A bent section's carry the physical volume element, the factor :math:`h`.
        """
        w = jnp.asarray(self.columns.measure * self.materials.keep[:, None])
        if self.moved is not None:
            w = w * self.moved[1][:, None]
        h = self.metric()
        return w if h is None else w * h

    def row(self, vector: jax.Array) -> jax.Array:
        """``(n_el, n_q, n_c)`` the mode of one DOF vector, in the global frame."""
        cols = self.columns
        e = jnp.einsum("eqck,ek->eqc", cols.weights, vector[cols.dofs])
        if self.moved is not None:
            # covariant (Nedelec) transport of the tangential components
            e = jnp.einsum("eij,eqj->eqi", self.moved[0], e)
        if cols.normal_weights is None:
            return e
        e_n = jnp.einsum("eqk,ek->eq", cols.normal_weights, vector[cols.normal_dofs])
        h = self.metric()
        if h is not None:
            e_n = e_n / h  # the physical field, not the transformed one
        full = jnp.concatenate([e, e_n[..., None]], axis=-1)
        return full @ jnp.asarray(self.section.frame)

    def field(self) -> jax.Array:
        """``(n_wave, n_el, n_q, n_c)`` the mode at 1 W in the global frame."""
        return jax.lax.map(self.row, self.vectors)


def _vector(field: Any) -> jax.Array:
    """The full DOF vector of each frequency, ``(n_freq, n_dof)``."""
    if field._vector is not None:  # noqa: SLF001
        return jnp.asarray(field._vector)  # noqa: SLF001
    _, basis, coords = field._data()  # noqa: SLF001
    return jnp.asarray(np.asarray(coords) @ np.asarray(basis).T)


def _same_section(a: Any, b: Any) -> bool:
    if len(a.normal) != len(b.normal):
        return False
    names = ("points", "triangles") if len(a.normal) == 3 else ("points", "segments")
    return all(np.array_equal(getattr(a, n), getattr(b, n)) for n in names)


def refuse_bend(owner: str, *modes: Any) -> None:
    """Raise `NotImplementedError` if any of ``modes`` is a bent port mode.

    For the effects whose overlaps carry no bend metric.
    """
    flat = [
        m
        for entry in modes
        for m in (entry if isinstance(entry, Sequence) else (entry,))
    ]
    if any(getattr(m, "bend", None) is not None for m in flat):
        raise NotImplementedError(
            f"{owner}: a bent port mode (gfem.Bend) is not supported: the "
            "bend-weighted overlaps of this effect are not available in this "
            "version. gfem.kerr_nonlinearity and gfem.kerr_coupling take it."
        )


def _same_bend(a: Any, b: Any) -> bool:
    """Whether two results were solved on the same bend (or both on none).

    A radius compares by value when both are concrete and by identity when
    one is traced: two traced radii are known equal only when they are the
    same value of the transformation.
    """
    if a is None or b is None:
        return a is b
    if a.outward != b.outward or a.at != b.at:
        return False
    if is_traced(a.radius) or is_traced(b.radius):
        return a.radius is b.radius
    return float(a.radius) == float(b.radius)


def _check_bends(owner: str, results: Sequence[Any]) -> None:
    """Raise if the results are not all of one guide (one bend, or none)."""
    if not all(_same_bend(r.bend, results[0].bend) for r in results):
        raise ValueError(
            f"{owner}: the waves are of different guides: a straight mode "
            "and a bent one, or two different bends, share no section "
            "metric. Solve every wave with the same gfem.Bend."
        )


def _waves(
    model: Model,
    modes: Any,
    owner: str,
    read_materials: Callable[[Model, Any, np.ndarray, str, str], Any],
    cached_columns: Callable[[Model, Any, str, Any], _Columns] | None = None,
    *,
    shape: bool = False,
) -> _Waves:
    """Gather the waves of ``modes`` on one section.

    ``read_materials(model, section, frequency, port, owner)`` reads the
    material values of the section's elements the effect needs (and raises the
    effect's own refusals); ``cached_columns`` defaults to `_cached_columns`.

    With ``shape`` the node coordinates may be traced by `jax.grad`: the
    section, the columns and the materials are read at their values and the
    waves carry the map of each section triangle from there to the moved one
    (`_moved`), so the overlaps differentiate the moving cross-section. The
    materials are read by owner element, so a `gfem.MaterialField`'s values
    move with their elements, as the mode's own node derivative holds them.

    Raises:
        TypeError: An entry is not a `gfem.PortModeResult`, or has no electric
            field.
        NotImplementedError: An Hz line port.
        ValueError: The results are of different ports or discretizations, or
            the model is not the one they were solved on.
        NonDifferentiableError: The node coordinates are traced (without
            ``shape``, or on a line port), here or in a mode's solve; or the
            model's are traced and a mode's solve's were not, or the reverse.
        TypeError: With ``shape``, the node coordinates are traced by
            `jax.jit`.
    """
    from gfem.model import Model

    if not isinstance(model, Model):
        raise TypeError(f"{owner} needs a gfem.Model, got {type(model).__name__}.")
    results = [modes] if isinstance(modes, PortModeResult) else modes
    if isinstance(results, str) or not isinstance(results, Sequence) or not results:
        raise TypeError(
            f"{owner} needs a gfem.PortModeResult (from gfem.solve(model, "
            f"gfem.PortMode(...))) or a sequence of them, got {type(modes).__name__}."
        )
    for r in results:
        if not isinstance(r, PortModeResult):
            raise TypeError(
                f"{owner} needs gfem.PortModeResult entries, got {type(r).__name__}: "
                "solve the mode with gfem.solve(model, gfem.PortMode(...))."
            )
        if r.h_field is not None:
            raise NotImplementedError(
                f"{owner}: wave port {r.port!r} is an Hz line port, whose electric "
                "field needs the gradient of Hz; solve it with polarization='Ez'."
            )
        if r.e_field is None:
            raise TypeError(
                f"{owner}: the result of wave port {r.port!r} carries no electric "
                "field (a driven result's port modes do not): solve "
                "gfem.solve(model, gfem.PortMode(...)) and pass that result."
            )
    if any(r.bend is not None for r in results) and (
        is_traced(model.mesh.nodes) or any(r.e_field._traced_nodes for r in results)  # noqa: SLF001
    ):
        raise NonDifferentiableError(
            f"{owner}: the derivative of an overlap of a bent port mode with "
            "respect to the node coordinates is not available in this version "
            "(its metric reads the points); differentiate with respect to the "
            "bend's radius or the materials, or wrap the node coordinates in "
            "jax.lax.stop_gradient."
        )
    traced, refusal = _node_state(model, results, owner, shape=shape)
    host = _at_host_nodes(model, owner) if traced else model
    spaces = []
    for r in results:
        try:
            spaces.append(r.e_field._space)  # noqa: SLF001
        except NonDifferentiableError as err:
            raise NonDifferentiableError(_NODES.format(owner=owner)) from err
    first = spaces[0]
    for r, space in zip(results, spaces, strict=True):
        if (
            r.port != results[0].port
            or space.order != first.order
            or not _same_section(space.section, first.section)
        ):
            raise ValueError(
                f"{owner}: the waves must share one port, element order and "
                f"section geometry; got port {results[0].port!r} (order "
                f"{first.order}) and port {r.port!r} (order {space.order}). Solve "
                "them with the same SolverOptions on the same model."
            )
    _check_bends(owner, results)
    section = first.section
    frequency = np.concatenate(
        [np.atleast_1d(np.asarray(r.frequency, float)) for r in results]
    )
    counts = [len(np.atleast_1d(np.asarray(r.frequency))) for r in results]
    modes_of = tuple(
        int(r.mode) for r, n in zip(results, counts, strict=True) for _ in range(n)
    )
    vectors = jnp.concatenate([_vector(r.e_field) for r in results], axis=0)
    _check_model(host, section, results[0].port, owner)
    materials = read_materials(host, section, frequency, results[0].port, owner)
    columns = (cached_columns or _cached_columns)(host, section, results[0].port, first)
    moved = None if host is model else _moved(model, section, owner)
    if refusal is not None:  # after the line port's own refusal
        raise NonDifferentiableError(refusal)
    return _Waves(
        results[0].port,
        modes_of,
        frequency,
        vectors,
        section,
        columns,
        materials,
        moved,
        results[0].bend,
    )


def _node_state(
    model: Model, results: Sequence[Any], owner: str, *, shape: bool
) -> tuple[bool, str | None]:
    """Whether the model's node coordinates are traced, and a refusal to raise.

    Each solve records whether its node coordinates were traced. The concrete
    field of a solve outside the gradient holds no tracer to compare with the
    model's by identity, so tracedness, not the tracer, is the signal: the
    overlap's node derivative needs the mode's and the model's both traced, or
    neither.

    Raises:
        NonDifferentiableError: Without ``shape``, the model's or a solve's
            node coordinates are traced.
    """
    traced = is_traced(model.mesh.nodes)
    solved = [r.e_field._traced_nodes for r in results]  # noqa: SLF001
    if not shape and (traced or any(solved)):
        raise NonDifferentiableError(_NODES.format(owner=owner))
    if all(s == traced for s in solved):
        return traced, None
    return traced, (_SOLVED_CONSTANT if traced else _SOLVED_TRACED).format(owner=owner)


def _at_host_nodes(model: Model, owner: str) -> Model:
    """``model`` with its traced node coordinates replaced by their values."""
    from gfem.optomechanics import _host_nodes

    staged = partial_eval.DynamicJaxprTracer
    if any(isinstance(x, staged) for x in jax.tree.leaves(model.mesh.nodes)):
        raise TypeError(
            f"{owner}: the node coordinates are computed inside jax.jit, but the "
            "section's elements are read at their values. Differentiate with "
            "jax.grad outside jax.jit, or wrap the node coordinates in "
            "jax.lax.stop_gradient."
        )
    return model.replace(mesh=model.mesh.with_nodes(_host_nodes(model.mesh)))


def _moved(model: Model, section: Any, owner: str) -> tuple[jax.Array, jax.Array]:
    r"""Each section triangle's map from its read shape to its moved one.

    The columns are built on the triangles at the node values the mode was
    read at, :math:`X_0`. With :math:`A_e = J_e(X) J_e(X_0)^{-1}`, the
    Jacobians of the straight triangles at the moved nodes :math:`X` and at
    :math:`X_0`, the physical tangential columns map as
    :math:`W_t(X) = A_e^{-\mathsf{T}} W_t(X_0)` (the covariant Piola
    transport, the DOFs held) and the weights as
    :math:`w_q(X) = w_q(X_0)\,|\det A_e|`; the normal component is
    :math:`H^1` and stays. The section's points are
    :math:`(X/s - o)F_2^\mathsf{T}` with the frame, origin and scale held, as
    `chain_to_nodes` holds them.

    Returns:
        ``(n_el, 2, 2)`` the matrices :math:`A_e^{-\mathsf{T}}` and ``(n_el,)``
        the :math:`|\det A_e|`.

    Raises:
        NonDifferentiableError: A line port (the 2D solve refuses node
            coordinates too).
    """
    if len(section.normal) != 3:
        raise NonDifferentiableError(_NODES.format(owner=owner))
    used = np.unique(np.asarray(model.mesh.triangles)[section.faces])
    frame = jnp.asarray(section.frame[:2])
    points = (jnp.asarray(model.mesh.nodes)[used] / section.scale - section.origin) @ (
        frame.T
    )
    triangles = np.asarray(section.triangles)

    def jacobian(p: Any) -> Any:
        a = p[triangles]
        return jnp.stack([a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]], axis=-1)

    read = np.linalg.inv(np.asarray(jacobian(np.asarray(section.points))))
    a = jacobian(points) @ read
    return jnp.swapaxes(jnp.linalg.inv(a), -1, -2), jnp.abs(jnp.linalg.det(a))


def _check_model(model: Model, section: Any, port: str, owner: str) -> None:
    """Refuse a model whose port is not the section the mode was solved on.

    The section is rebuilt from the model's mesh and port; a mode of another
    mesh (or of moved nodes) would be read at the wrong elements.
    """
    from gfem._line_section import line_section
    from gfem._port_section import port_geometry

    planar = len(section.normal) == 3
    try:
        if planar:
            other = port_geometry(model, section.port)
        else:
            other = line_section(model, section.port, geometry=True)
        same = (
            other.scale == section.scale
            and _same_section(other, section)
            and (
                not planar
                or np.array_equal(np.asarray(other.faces), np.asarray(section.faces))
            )
        )
    except (ValueError, KeyError, IndexError, AttributeError):
        same = False
    if not same:
        raise ValueError(
            f"{owner}: the model is not the one wave port {port!r}'s mode was "
            "solved on: its mesh gives another cross-section."
        )


#: Columns of the sections already read, per mesh topology then ``(port, order)``,
#: with the section they were built on: a traced result rebuilds its space at
#: every call, and the columns cost a few NGSolve assemblies.
_COLUMNS: weakref.WeakKeyDictionary[Any, dict[tuple, tuple[Any, _Columns]]] = (
    weakref.WeakKeyDictionary()
)


def _cached_columns(
    model: Model,
    section: Any,
    port: str,
    space: Any,
    *,
    store: Any = None,
    build: Callable[..., _Columns] | None = None,
) -> _Columns:
    """The columns of ``space``, from the cache ``store`` or built by ``build``."""
    store = (_COLUMNS if store is None else store).setdefault(
        model.mesh._topology,  # noqa: SLF001
        {},
    )
    key = (port, space.order)
    hit = store.get(key)
    if (
        hit is not None
        and hit[0].scale == section.scale
        and _same_section(hit[0], section)
    ):
        return hit[1]
    columns = (build or _columns)(space)
    store[key] = (section, columns)
    return columns
