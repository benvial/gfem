r"""The second-order polarization a pump drives: a source for the generated solve.

A `Pump` is a kept field of a driven result (or a combination of its columns)
with the power and phase it is fed at. `second_order_polarization` forms from
one or two pumps the polarization :math:`P` at the generated frequency
:math:`\omega_3` that the contracted tensor :math:`d` of the model's materials
(``gfem._chi2``, in the global frame) gives at the points of a quadrature rule:

.. math::

    P = \varepsilon_0\, d\!:\!FF \quad (\omega_3 = 2\omega), \qquad
    P = 2\varepsilon_0\, d\!:\!F_1F_2 \quad (\omega_3 = \omega_1 + \omega_2), \qquad
    P = 2\varepsilon_0\, d\!:\!F_1\overline{F_2} \quad (\omega_3 = \omega_1 - \omega_2),

with :math:`F = \sum_p a_p E_p` the pump field, :math:`E_p` the kept field of
column :math:`p` and :math:`a_p = \sqrt{P_p}\,e^{i\phi_p}` its amplitude, as
`gfem.electromagnetic_heating` feeds ports. The pump is undepleted: it is the
solution at :math:`\omega` alone. The generated field solves the current
:math:`J = +i\omega_3 P` (`gfem.PolarizationSource`)::

    src = gfem.second_order_polarization(model, gfem.Pump(pump_result, f, ...))
    gen = gfem.solve(model_2w, gfem.Driven(src.frequency, excite=(), sources=src))

Both solves are on one mesh. A uniform guide of crystal length :math:`L` between
matched ports gives the generated wave :math:`b = -i\,\overline{\kappa}\,
a_1^2 L\,\mathrm{sinc}(\Delta\beta L/2)\,e^{-i(\beta_h + 2\beta_p)L/2}` with
:math:`\kappa` of `gfem.three_wave_coupling`.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _chi2
from gfem._checks import is_traced, same_lattice, static_float
from gfem._materials import Material
from gfem._pml import layer_tetrahedra
from gfem._pytree import register_fields
from gfem._sources import GaussianBeam, PolarizationSource
from gfem.errors import NonDifferentiableError
from gfem.thermal import _amplitude, _check_finite, _check_non_negative

if TYPE_CHECKING:
    from gfem.model import Model
    from gfem.results import DrivenResult

__all__ = ["Pump", "second_order_polarization"]

_EPS0 = const.epsilon_0
_PROPERTY = "second_order_nonlinear_tensor"
#: Relative tolerance of two frequencies being one (a wavelength round trip).
_SAME_FREQUENCY = 1e-9
_NAMES = {
    "second-harmonic": "second harmonic",
    "sum-frequency": "sum frequency",
    "difference-frequency": "difference frequency",
}


@dataclasses.dataclass(frozen=True, init=False, eq=False, repr=False)
class Pump:
    r"""A kept pump field of a driven result, with the drive that feeds it.

    The pump field is :math:`F = \sum_p a_p E_p` over the result's columns
    ``excite``, :math:`E_p` the kept electric field at ``frequency`` and
    :math:`a_p = \sqrt{P_p}\,e^{i\phi_p}`, as `gfem.electromagnetic_heating`
    feeds ports: a wave port column is 1 W incident (:math:`\sqrt{\mathrm W}`
    at its reference plane), a Floquet port column 1 W per cell and a
    `gfem.GaussianBeam` column 1 W. A `gfem.PlaneWave`, `gfem.IncidentField` or
    dipole column is absolute (its amplitude or moment is the drive), so it
    takes a phase and no power. The result must keep its fields
    (``Driven(keep_fields=...)``) at ``frequency``.

    Args:
        result: A `gfem.DrivenResult` of a 3D model with a kept field at
            ``frequency`` for every column of ``excite``.
        frequency: The pump frequency, Hz (matched to a kept one to
            :math:`10^{-9}` relative).
        excite: The column or columns to feed: a name, a sequence, or ``None``
            for the keys of a mapping ``incident_power`` or the result's only
            column.
        incident_power: The incident power of each port or beam column, W: a
            number for one column or a mapping from column name. Required for
            them (a default of 1 W would hide that the generated power is
            quadratic in it), and refused for a plane wave, an incident field
            and a dipole.
        incident_phase: The phase of each column, rad, 0 where left out: a
            number for one column or a mapping.

    Raises:
        TypeError: ``result`` is not a driven result, or a power or phase is
            not a real number.
        ValueError: No kept field at ``frequency``; a power missing, negative,
            or given to a column that takes none; ports and sources mixed in
            one pump; ``excite`` needed or repeating a column; a dipole of a
            ``source_fields=False`` result.
        KeyError: A column the result does not have, or a power or phase for
            a column not fed.
        NotImplementedError: The result is a 2D model's.
        NonDifferentiableError: ``frequency`` is traced.
    """

    result: DrivenResult
    frequency: float
    columns: tuple[str, ...]
    powers: tuple[Any, ...]
    phases: tuple[Any, ...]

    def __init__(
        self,
        result: DrivenResult,
        frequency: float,
        *,
        excite: str | Sequence[str] | None = None,
        incident_power: float | Mapping[str, float] | None = None,
        incident_phase: float | Mapping[str, float] | None = None,
    ) -> None:
        """Validate (see the class docstring)."""
        from gfem.results import DrivenResult

        if not isinstance(result, DrivenResult):
            raise TypeError(
                "Pump takes the result of gfem.solve(model, gfem.Driven(..., "
                f"keep_fields=...)), got {type(result).__name__}."
            )
        if result.polarization is not None:
            raise NotImplementedError(
                "A Pump of a 2D model's result: the second-order source is "
                "built on 3D meshes. Pose the pump on a slab model, one element "
                "thick."
            )
        f = static_float(
            frequency,
            "Pump.frequency",
            "Differentiate the pump through its result, whose fields are "
            "traced, and give the frequency as a number.",
        )
        _kept_index(result, f)
        names = _columns(result, excite, incident_power)
        powers, phases = _drives(result, names, incident_power, incident_phase)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "frequency", f)
        object.__setattr__(self, "columns", names)
        object.__setattr__(self, "powers", powers)
        object.__setattr__(self, "phases", phases)

    def __repr__(self) -> str:
        return f"Pump(frequency={self.frequency:.4g}, columns={self.columns!r})"

    def fields(self) -> list[Any]:
        """The kept fields of the columns, `gfem.Field` each."""
        return [self.result.e_field(self.frequency, c) for c in self.columns]

    def amplitudes(self) -> jax.Array:
        r"""``(n_columns,)``: :math:`a_p`, complex.

        A port column is :math:`\sqrt{P_p}\,e^{i\phi_p}` at its reference
        plane (`gfem.DrivenResult.with_offsets` moves it to the surface, as
        `gfem.electromagnetic_heating` does); an absolute column is
        :math:`e^{i\phi_p}`.
        """
        result = self.result
        index = _grid_index(result, self.frequency)
        offsets = dict(result.offsets)
        out = []
        for name, power, phase in zip(
            self.columns, self.powers, self.phases, strict=True
        ):
            a = jnp.exp(1j * phase)
            if power is not None:
                a = _amplitude(power) * a
            if name in result.s.excited and offsets.get(name, 0.0):
                shift = result._plane_shift(name, offsets[name])  # noqa: SLF001
                a = a * shift[index]
            out.append(a)
        return jnp.stack(out).astype(complex)


def _grid_index(result: DrivenResult, frequency: float) -> int:
    return next(
        k
        for k, g in enumerate(result.frequency)
        if math.isclose(float(g), frequency, rel_tol=_SAME_FREQUENCY)
    )


def _kept_index(result: DrivenResult, frequency: float) -> int:
    """The index of ``frequency`` among the kept ones, or raise."""
    for k, (f, _) in enumerate(result.kept):
        if math.isclose(float(f), frequency, rel_tol=_SAME_FREQUENCY):
            return k
    raise ValueError(
        f"The result keeps no field at {frequency:g} Hz: solve the pump with "
        "gfem.Driven(..., keep_fields=True) (or the frequency) and give "
        "Pump(frequency=) one of its frequencies."
    )


def _columns(
    result: DrivenResult, excite: str | Sequence[str] | None, power: Any
) -> tuple[str, ...]:
    """The columns a pump feeds, checked against the result's."""
    available = result._excitations  # noqa: SLF001
    if excite is None:
        if isinstance(power, Mapping):
            names = tuple(power)
        elif len(available) == 1:
            names = available
        else:
            raise ValueError(
                f"The result has {len(available)} columns "
                f"({', '.join(map(repr, available))}): name the ones the pump "
                "feeds with Pump(excite=...)."
            )
    else:
        names = (excite,) if isinstance(excite, str) else tuple(excite)
    if not names:
        raise ValueError("Pump.excite names no column.")
    if len(set(names)) != len(names):
        raise ValueError(f"Pump.excite lists a column twice: {names!r}.")
    for name in names:
        if name not in available:
            raise KeyError(
                f"Pump.excite names {name!r}, but the result's columns are "
                f"{', '.join(map(repr, available))}."
            )
    is_port = [n in result.s.excited for n in names]
    if any(is_port) and not all(is_port):
        raise ValueError(
            "A Pump feeds ports or sources, not both: make one Pump of the "
            f"ports and one of the sources (got {', '.join(map(repr, names))})."
        )
    for name in names:
        result._refuse_unsolved_source(  # noqa: SLF001
            name, "so the pump has no field to read", remedy="a pump"
        )
    return names


def _drives(
    result: DrivenResult, names: tuple[str, ...], power: Any, phase: Any
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """The power (``None`` where the column takes none) and phase of each column."""
    given = _per_column(power, names, "incident_power")
    phases = _per_column(phase, names, "incident_phase")
    powers: list[Any] = []
    for name in names:
        source = result.sources.get(name)
        takes = source is None or isinstance(source, GaussianBeam)
        if takes:
            if name not in given:
                raise ValueError(
                    f"Pump needs incident_power for {name!r}: the field is 1 W "
                    "per unit amplitude squared, and a silent default would hide "
                    "that the generated power is quadratic in it."
                )
            value = _check_finite(given[name], f"Pump incident_power[{name!r}]")
            _check_non_negative(value, f"Pump incident_power of {name!r}")
            powers.append(value)
        else:
            if name in given:
                kind = type(source).__name__
                remedy = (
                    "set PlaneWave(amplitude=...)"
                    if kind == "PlaneWave"
                    else "scale its amplitude or moment"
                )
                raise ValueError(
                    f"Pump incident_power for the {kind} column {name!r}: its "
                    f"field is already absolute (no power); {remedy}. A phase "
                    "is accepted."
                )
            powers.append(None)
    return tuple(powers), tuple(
        _check_finite(phases.get(n, 0.0), f"Pump incident_phase[{n!r}]") for n in names
    )


def _per_column(value: Any, names: tuple[str, ...], what: str) -> dict[str, Any]:
    """``value`` as a mapping from column: a number belongs to the only column."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        extra = [k for k in value if k not in names]
        if extra:
            raise KeyError(
                f"Pump.{what} names {extra[0]!r}, which the pump does not feed "
                f"(its columns are {', '.join(map(repr, names))})."
            )
        return dict(value)
    if len(names) != 1:
        raise ValueError(
            f"Pump.{what} is one number but the pump feeds {len(names)} "
            f"columns: give a mapping from column name, e.g. {what}="
            f"{{{names[0]!r}: ...}}."
        )
    return {names[0]: value}


register_fields(Pump, ("result", "powers", "phases"))


def second_order_polarization(
    model: Model,
    pump: Pump | tuple[Pump, Pump],
    *,
    process: Literal["sum-frequency", "difference-frequency"] | None = None,
    name: str | None = None,
    generated_order: int | None = None,
) -> PolarizationSource:
    r"""The polarization a pump drives through the model's second-order tensor.

    With one pump the process is second-harmonic generation,
    :math:`\omega_3 = 2\omega` and :math:`P = \varepsilon_0\, d\!:\!FF`. With
    two, ``process`` is ``"sum-frequency"`` (:math:`\omega_3 = \omega_1 +
    \omega_2`, :math:`P = 2\varepsilon_0\, d\!:\!F_1F_2`) or
    ``"difference-frequency"`` (:math:`\omega_3 = \omega_1 - \omega_2` with the
    first pump the higher, :math:`P = 2\varepsilon_0\, d\!:\!F_1\overline{F_2}`).
    :math:`d` is the contracted tensor of each region's material, in the global
    frame, and :math:`F` the pump's three-component field (`Pump`). :math:`P`
    is formed on the elements that have a tensor and lie outside any perfectly
    matched layer, at the points of a quadrature rule exact for the product of
    the pump fields and a generated basis function.

    The pump results may come from other models of the same mesh (a
    constant-:math:`\varepsilon` model per frequency, say); ``model`` supplies
    the tensors and the layer. Solve the result on a model of the same mesh at
    the generated frequency, with `gfem.Driven`::

        gen = gfem.solve(model_2w, gfem.Driven(src.frequency, excite=(), sources=src))
        gen.port_wave("z_max TM", src.name)  # sqrt(W)
        gen.emitted_power[src.name]  # W

    The generated field is the response to :math:`J = +i\omega_3 P` with the
    pump undepleted; the load of a mesh in um and of the same mesh in m are
    the same.

    Args:
        model: The model whose materials carry ``second_order_nonlinear_tensor``.
        pump: A `Pump`, or two for sum or difference frequency.
        process: ``None`` for one pump; for two, ``"sum-frequency"`` or
            ``"difference-frequency"``.
        name: The label of the source's column; defaults to ``"second
            harmonic"``, ``"sum frequency"`` or ``"difference frequency"``.
        generated_order: The highest Nédélec order of the generated solve; the
            quadrature rule integrates the load exactly up to it. Defaults to
            the pump field's order.

    Returns:
        A `gfem.PolarizationSource` at the generated frequency.

    Raises:
        TypeError: ``pump`` is not a `Pump` or a pair of them.
        ValueError: No region of ``model`` carries a tensor, or none outside
            a perfectly matched layer; ``process`` given with one pump or
            missing with two, or two pumps at one frequency (feed one
            `Pump` with both drives for sum frequency); a difference with
            the first pump below the second; ``generated_order`` below 1; a
            pump on another mesh than ``model``'s.
        NotImplementedError: A pump solved across MPI ranks; the pump orders and
            ``generated_order`` sum to more than 10 (an order-4 pump with the
            default ``generated_order``).
        NonDifferentiableError: The node coordinates of the model or of a pump's
            mesh are traced (the call is not traceable: make it outside
            ``jax.jit``).
    """
    pumps = _pumps(pump)
    kind = _process(pumps, process)
    mesh = model.mesh
    if is_traced(mesh.nodes):
        raise NonDifferentiableError(
            "second_order_polarization is not traceable: it reads the pump at "
            "the rule points of a fixed mesh, so call it outside jax.jit with "
            "concrete mesh nodes (gfem does not differentiate with respect to "
            "the node coordinates here)."
        )
    for p in pumps:
        _check_pump(p, mesh)
    f_a = pumps[0].frequency
    frequency = {
        "second-harmonic": 2 * f_a,
        "sum-frequency": f_a + pumps[-1].frequency,
        "difference-frequency": f_a - pumps[-1].frequency,
    }[kind]
    orders = [p.result.e_field(p.frequency, p.columns[0]).order for p in pumps]
    if generated_order is None:
        generated_order = max(orders)
    if isinstance(generated_order, bool) or not hasattr(generated_order, "__index__"):
        raise ValueError(
            f"generated_order must be an int >= 1, got {generated_order!r}."
        )
    generated_order = operator.index(generated_order)
    if generated_order < 1:
        raise ValueError(f"generated_order must be an int >= 1, got {generated_order}.")
    # the rule integrates (pump field)(pump field)(generated basis) exactly
    degree = (orders[0] + orders[-1] + generated_order) / 2
    rule_order = math.ceil(degree)
    from gfem._engine.volume_loads import MAX_RULE_ORDER

    if rule_order > MAX_RULE_ORDER:
        raise NotImplementedError(
            f"pump orders {orders[0]} and {orders[-1]} with generated_order="
            f"{generated_order} need an integration rule of order {rule_order}, "
            f"and gfem supports up to {MAX_RULE_ORDER} (the sum of the three "
            f"orders is at most {2 * MAX_RULE_ORDER}): lower generated_order "
            "or solve the pump at a lower order."
        )
    lattice, wave_vector = _bloch(pumps, kind)
    elements, tensors, regions = _tensors(model)
    values = _polarization(pumps, kind, elements, tensors, order=rule_order)
    p_values, points = values
    return PolarizationSource(
        name=name or _NAMES[kind],
        frequency=frequency,
        process=kind,
        pump_frequencies=tuple(p.frequency for p in pumps),
        regions=regions,
        order=rule_order,
        generated_order=generated_order,
        elements=elements,
        mesh_elements=len(mesh.volume_elements),
        points=points,
        re=jnp.real(p_values),
        im=jnp.imag(p_values),
        wave_vector=wave_vector,
        lattice=lattice,
    )


def _pumps(pump: Any) -> tuple[Pump, ...]:
    items = tuple(pump) if isinstance(pump, tuple | list) else (pump,)
    if len(items) not in (1, 2) or not all(isinstance(p, Pump) for p in items):
        raise TypeError(
            "second_order_polarization takes a gfem.Pump (second harmonic) or a "
            f"pair of them (sum or difference frequency), got {pump!r}."
        )
    return items


def _process(pumps: tuple[Pump, ...], process: str | None) -> str:
    if len(pumps) == 1:
        if process is not None:
            raise ValueError(
                f"process={process!r} needs two pumps: one Pump is second-harmonic "
                "generation, so leave process out."
            )
        return "second-harmonic"
    if process not in ("sum-frequency", "difference-frequency"):
        raise ValueError(
            "Two pumps need process='sum-frequency' or 'difference-frequency', "
            f"got {process!r}."
        )
    a, b = pumps[0].frequency, pumps[1].frequency
    if math.isclose(a, b, rel_tol=_SAME_FREQUENCY):
        if process == "sum-frequency":
            raise ValueError(
                "Two pumps at one frequency are second-harmonic generation, whose "
                "factor differs: feed one Pump with both drives "
                "(Pump(excite=[...], incident_power={...}))."
            )
        raise ValueError(
            "The difference of two pumps at one frequency is optical "
            "rectification (a static polarization), which is not supported."
        )
    if process == "difference-frequency" and a < b:
        raise ValueError(
            f"The first pump of a difference frequency is the higher one "
            f"({a:g} Hz is below {b:g} Hz): swap the pumps."
        )
    return process


def _check_pump(pump: Pump, mesh: Any) -> None:
    field = pump.result.e_field(pump.frequency, pump.columns[0])
    if field._comm is not None:  # noqa: SLF001
        raise NotImplementedError(
            "A pump solved across MPI ranks: the second-order source is built "
            "from the kept fields of one process."
        )
    other = field.mesh
    if other is not mesh and is_traced(other.nodes):
        raise NonDifferentiableError(
            "second_order_polarization is not traceable: the pump's mesh nodes "
            "are traced values. Call it outside jax.jit (or jax.grad with "
            "respect to the pump's coefficients only), on concrete mesh nodes."
        )
    same = other is mesh or (
        np.shape(other.nodes) == np.shape(mesh.nodes)
        and np.array_equal(other.volume_elements, mesh.volume_elements)
        and np.array_equal(other.nodes, mesh.nodes)
    )
    if not same:
        raise ValueError(
            "A pump was solved on another mesh than the model's: the source is "
            "formed element by element, so the pump results and the model share "
            "one mesh (their models may differ)."
        )


def _bloch(
    pumps: tuple[Pump, ...], kind: str
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float] | None]:
    r"""The pumps' lattice and the Bloch vector of the generated field.

    A pump of plain periodicity counts as the zero vector beside a Bloch one,
    and the product of Bloch factors adds the vectors:
    :math:`\mathbf k_1 + \mathbf k_2`, with :math:`-\mathbf k_2` for the
    conjugated pump of a difference.
    """
    fields = [p.fields()[0] for p in pumps]
    lattice = fields[0]._lattice  # noqa: SLF001
    for f in fields[1:]:
        other = f._lattice  # noqa: SLF001
        if not same_lattice(other, lattice):
            raise ValueError(
                "The two pumps were solved on cells with different periodic "
                "pairs (translations "
                f"{lattice!r} and {other!r} m): the generated field of one cell "
                "needs both pumps on the same lattice."
            )
    vectors = [f._wave_vector for f in fields]  # noqa: SLF001
    if all(v is None for v in vectors):
        return lattice, None
    k = [np.zeros(3) if v is None else np.asarray(v, float) for v in vectors]
    sign = -1.0 if kind == "difference-frequency" else 1.0
    total = k[0] + (k[0] if len(k) == 1 else sign * k[1])
    return lattice, (float(total[0]), float(total[1]), float(total[2]))


def _tensors(model: Model) -> tuple[np.ndarray, jax.Array, tuple[str, ...]]:
    """The elements with a tensor outside the layer, and their global-frame ``d``."""
    mesh = model.mesh
    carrying = [
        r for r, m in model.materials.items() if getattr(m, _PROPERTY, None) is not None
    ]
    if not carrying:
        names = ", ".join(map(repr, model.materials)) or "none"
        raise ValueError(
            f"No region of the model has a {_PROPERTY} (its regions: {names}): "
            "set one on the crystal's material."
        )
    layer = layer_tetrahedra(mesh, model.pml)
    per_region = {r: np.asarray(mesh.elements(r)) for r in carrying}
    elements = np.unique(np.concatenate(list(per_region.values())))
    if layer is not None:
        elements = elements[~layer[elements]]
    if not elements.size:
        raise ValueError(
            f"Every element of the regions {', '.join(map(repr, carrying))} that "
            f"carry a {_PROPERTY} lies in a perfectly matched layer: the source "
            "is built outside it. Set the tensor on a physical region."
        )
    d = jnp.zeros((len(elements), 3, 6))
    for region, cells in per_region.items():
        material = model.materials[region]
        value = getattr(material, _PROPERTY)
        mine = np.isin(elements, cells)
        rows = np.flatnonzero(mine)
        if not rows.size:
            continue
        pos = np.searchsorted(cells, elements[rows])
        if isinstance(material, Material):
            rotated = jnp.asarray(
                _chi2.in_global_frame(np.asarray(value), material.axes)
            )
        else:
            rotated = _chi2.in_global_frame(jnp.asarray(value), material.axes)
        per = (
            jnp.broadcast_to(rotated, (len(rows), 3, 6))
            if rotated.ndim == 2
            else rotated[pos]
        )
        d = d.at[rows].set(per)
    return elements, d, tuple(carrying)


def _polarization(
    pumps: tuple[Pump, ...],
    kind: str,
    elements: np.ndarray,
    d: jax.Array,
    *,
    order: int,
) -> tuple[jax.Array, np.ndarray]:
    """``(n_points, 3)`` complex :math:`P` and the points' coordinates."""
    from gfem._engine.volume_loads import pump_columns

    pump_fields = []
    points: np.ndarray | None = None
    for p in pumps:
        fields = p.fields()
        columns = pump_columns(fields[0], elements, order)
        if points is None:
            points = columns.points
        elif columns.points.shape != points.shape or not np.allclose(
            columns.points, points, rtol=0, atol=1e-9 * max(np.ptp(points), 1e-300)
        ):  # pragma: no cover - one mesh, one rule
            raise RuntimeError("the pumps' rule points differ.")
        a = p.amplitudes()
        total = sum(a[k] * columns.values(f.coefficients) for k, f in enumerate(fields))
        pump_fields.append(total)
    first = pump_fields[0]
    second = first if kind == "second-harmonic" else pump_fields[1]
    if kind == "difference-frequency":
        second = jnp.conj(second)
    factor = 1.0 if kind == "second-harmonic" else 2.0
    p_values = factor * _EPS0 * _chi2.contract(d[:, None], first, second)
    return p_values.reshape(-1, 3), np.asarray(points)
