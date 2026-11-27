r"""Mode cascades: devices whose blocks are joined through uniform sections.

A device made of discontinuities (a junction, a step, a taper facet) joined
by straight uniform guides is solved by FEM only at its discontinuities. Each
block ends in the kept modes of the guide (`gfem.WavePort.kept_modes`), the
propagating ones and enough evanescent ones to carry the field across the
block's short guide stub. `cascade` then joins blocks through the sections
analytically, mode by mode, with a length that is a plain (traced) number:
a length sweep needs no new mesh or solve and differentiates in each length.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants
from numpy.typing import ArrayLike

from gfem._checks import is_traced
from gfem._pytree import register_fields
from gfem._text import hertz_band
from gfem.errors import NonDifferentiableError
from gfem.results import DrivenResult, NetworkParameters, PortModeResult

if TYPE_CHECKING:
    from gfem.problems import Polarization

__all__ = ["CascadeResult", "cascade"]

# A joined pair must model one guide: |gamma_A - gamma_B| / k0 beyond this
# means two different cross-sections. Two discretizations of one guide differ
# by up to 4e-5 (measured), so this leaves a margin of about 250.
_MISMATCH = 1e-2

# two blocks carry the same mode when the mass-normalized overlap of their
# fields exceeds this: the same mode of two discretizations of one guide
# overlaps to 1 - O(h^2p), and the nearest other mode of a guide to well below
_SAME_MODE = 0.95

# the propagating threshold of the port mode sweeps, in units of k0
_PROPAGATES = 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class CascadeResult:
    r"""The result of `cascade`: S of the exposed ports of the joined blocks.

    A cascade result is a block of the same kind as a `DrivenResult` for
    `cascade` (it carries ``s``, ``port_modes``, ``offsets`` and
    ``polarization``), so cascades nest.

    Attributes:
        s: The scattering parameters of the exposed ports
            (`NetworkParameters`): the blocks' ports that are not section
            ports, in block order and then each block's port order, all
            excited, each with its block's ``z_ref``.
        port_modes: The mode of each exposed wave port over the frequencies
            (`PortModeResult`, by port name), as its block carries it.
        sections: The length of each section as given, m.
        gamma: The propagation constant :math:`\gamma_m` used for each mode
            of each section, ``(n_freq, N)``, 1/m: the mean of the two ends'.
        gamma_mismatch: :math:`|\gamma_A - \gamma_B| / k_0` of each mode of
            each section, ``(n_freq, N)``, with :math:`k_0 = \omega/c_0`: how
            far the two blocks are from modelling one guide.
        polarization: The field normal to the plane the blocks computed,
            ``"Ez"`` or ``"Hz"`` (a static label).
        offsets: ``(port name, distance in m)`` of each exposed wave port
            (static labels), as in `DrivenResult`.
    """

    s: NetworkParameters
    port_modes: Mapping[str, PortModeResult] = dataclasses.field(default_factory=dict)
    sections: Mapping[str, jax.Array] = dataclasses.field(default_factory=dict)
    gamma: Mapping[str, jax.Array] = dataclasses.field(default_factory=dict)
    gamma_mismatch: Mapping[str, jax.Array] = dataclasses.field(default_factory=dict)
    polarization: Polarization | None = None
    offsets: tuple[tuple[str, float], ...] = ()

    def __repr__(self) -> str:
        ports = ", ".join(map(repr, self.s.ports))
        parts = []
        for name, length in self.sections.items():
            n = np.shape(self.gamma[name])[-1]
            try:
                where = f"{float(np.asarray(length)):.6g} m"
            except jax.errors.TracerArrayConversionError:
                where = "traced length"
            except TypeError:  # a concrete batch of lengths (vmap)
                low, high = np.min(length), np.max(length)
                where = f"{low:.6g} to {high:.6g} m"
            parts.append(f"{name!r} ({n} modes, {where})")
        sections = ", ".join(parts) if parts else "no sections"
        f = self.s.frequency
        span = hertz_band(min(f), max(f), 4) if f else "no frequencies"
        return f"CascadeResult(ports {ports}; sections {sections}; {span})"


register_fields(
    CascadeResult,
    ("s", "port_modes", "sections", "gamma", "gamma_mismatch"),
)

#: What `cascade` accepts as a block.
Block = DrivenResult | CascadeResult


def cascade(
    blocks: Sequence[DrivenResult | CascadeResult],
    sections: Mapping[str, ArrayLike],
) -> CascadeResult:
    r"""S of a device whose blocks are joined through uniform sections.

    Each section ``s`` joins the two blocks that carry its ports ``s@1`` ...
    ``s@N`` (`gfem.WavePort.kept_modes`) and carries each kept mode across as
    :math:`e^{-\gamma_m \ell}`: the mode of one guide, whose two ends the
    blocks' port planes are. With :math:`S` the block-diagonal S of all
    blocks (exposed ports :math:`e`, joined ports :math:`i`) and
    :math:`\Gamma` the matrix that joins the two ends of each section mode
    through :math:`e^{-\gamma_m \ell}`, :math:`\gamma_m` the mean of the two
    ends' propagation constants,

    .. math::

        S_\mathrm{ext} = S_{ee} + S_{ei}\,\Gamma\,
        (I - S_{ii}\,\Gamma)^{-1}\, S_{ie},

    one linear solve per frequency, so the algebra stays finite at any length
    (no transfer matrices) and is exact at :math:`\ell = 0`, where the blocks
    are joined directly. An evanescent kept mode's row and column of S are
    pseudo-waves that carry no power (`gfem.WavePort`), so a cascade is
    passive on its exposed ports, not on the joined ones.

    Ports. A port ``s@m`` with ``s`` a key of ``sections`` is internal. Every
    other port name must occur in exactly one block and is exposed, in block
    order and then each block's port order; every exposed port is excited
    and keeps its ``z_ref`` (nothing is renormalized). A block is a
    `DrivenResult` of a 2D model solved with every port excited
    (``Driven(excite=None)``), or a `CascadeResult`.

    Gauge. A mode's sign is arbitrary in each block, so the second block's
    row and column :math:`m` are multiplied by :math:`t_m = \pm 1`. Where both
    ends of the mode carry a ``polarity`` (`gfem.WavePort`) it is
    :math:`t_m = \mathrm{sign}(\hat d_A \cdot \hat z)\,
    \mathrm{sign}(\hat d_B \cdot \hat z)`, with or without a trace. Inside
    `jax.grad`, `jax.jit` or `jax.vmap` the blocks solved inside the
    transformation carry no port fields (blocks solved outside it keep them),
    so a mode of such blocks without a polarity on both ends cannot be
    aligned: give
    ``WavePort.kept_modes(..., polarity={m: (point, direction)})``.

    Args:
        blocks: The solved blocks.
        sections: Section name to its length in m between the two blocks'
            port planes (plane to plane, not the physical length of the
            guide), a scalar :math:`\ge 0`: 0 joins the blocks directly. It
            may be traced; sweep it with `jax.vmap`. Differentiable.

    Returns:
        The cascade's `CascadeResult`.

    Raises:
        ValueError: ``sections`` is not a mapping; there are no blocks; the
            blocks' frequencies differ; a block does not have every port
            excited; a section name contains ``"@"``; a section's ports sit
            in no block, one block or three or more; its ends carry
            different numbers of modes, ranks other than :math:`1..N`, or a
            port that is not a wave port mode; a port name other than a
            section's occurs in two or more blocks (join through
            ``kept_modes`` and a length 0 section); a joined port has an
            offset, a voltage path or a perfectly matched layer; the blocks
            solved different polarizations; a length is not a scalar, is
            negative, not finite or complex; the last kept mode propagates
            at some frequency; the two ends' propagation constants differ by
            more than :math:`10^{-2} k_0`; a section's two planes do not face
            each other; a mode's polarity direction is not along :math:`z`;
            a section joins plane (3D) ports; the two blocks do not carry the
            same mode (the overlap of its fields is below 0.95); the gauge of
            a mode is undetermined or changes sign across the band; the
            polarity and the pairing gauge of a mode disagree; a mode has no
            polarity on both ends and no port fields to align (outside a
            transformation).
        NonDifferentiableError: As the last case, inside `jax.grad`,
            `jax.jit` or `jax.vmap`.

    Checks that read values (the voltage path, the lengths, propagation and
    index mismatch, the planes' normals, the overlap and sign of the fields,
    and the polarity cross-check) run only on concrete values, so blocks
    solved inside a transformation skip the field checks: run the cascade
    once outside the transformation to have them all.
    """
    blocks = list(blocks)
    if not isinstance(sections, Mapping):
        raise ValueError(  # noqa: TRY004 - a bad argument is a ValueError throughout
            f"cascade: sections must map each section name to its length, got "
            f"{type(sections).__name__}."
        )
    _check_blocks(blocks)
    plan = [_plan_section(blocks, name) for name in sections]
    _check_exposed(blocks, plan)
    _check_joined(blocks, plan)
    lengths = _lengths(sections)
    traced = is_traced(blocks) or any(is_traced(v) for v in lengths.values())
    # the value checks and the gauge read only the blocks, never the lengths:
    # computed eagerly they stay concrete under jax.jit and jax.vmap
    with jax.ensure_compile_time_eval():
        _check_values(blocks, plan)
        gauge = _gauges(blocks, plan, traced)
    return _join(blocks, plan, lengths, gauge)


# --- wiring -------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Section:
    """One section: the two blocks it joins and the port name of each mode."""

    name: str
    a: int  # index of the first block in the list, the gauge reference
    b: int
    n: int

    def port(self, m: int) -> str:
        return f"{self.name}@{m}"


def _ranks(block: Block, name: str) -> list[int | None]:
    """The ranks of the ports ``name@...`` of a block; None where not an integer."""
    out: list[int | None] = []
    for port in block.s.ports:
        if port.startswith(f"{name}@"):
            tail = port[len(name) + 1 :]
            out.append(int(tail) if tail.isdecimal() else None)
    return out


def _check_blocks(blocks: list[Block]) -> None:
    """The blocks agree on frequencies, excitation and polarization."""
    if not blocks:
        raise ValueError("cascade: give at least one block.")
    freq = blocks[0].s.frequency
    for i, blk in enumerate(blocks):
        if blk.s.frequency != freq:
            raise ValueError(
                f"cascade: block {i} has frequencies {blk.s.frequency} against "
                f"{freq} of block 0; the blocks must be solved at the same "
                "frequencies (there is no interpolation)."
            )
        if blk.s.excited != blk.s.ports:
            raise ValueError(
                f"cascade: block {i} excites {blk.s.excited} of its ports "
                f"{blk.s.ports}; every port must be excited "
                "(Driven(excite=None))."
            )
        if len(set(blk.s.ports)) != len(blk.s.ports):  # pragma: no cover - guarded
            raise ValueError(f"cascade: block {i} repeats a port name.")
    if len({blk.polarization for blk in blocks}) > 1:
        found = ", ".join(repr(blk.polarization) for blk in blocks)
        raise ValueError(
            f"cascade: the blocks solved different polarizations ({found}); Ez "
            "and Hz fields do not join."
        )


def _plan_section(blocks: list[Block], name: str) -> _Section:
    """The two blocks of a section, with the rank and count checks."""
    if "@" in name:
        raise ValueError(
            f"cascade: the section name {name!r} contains '@', which separates "
            "a section from its mode rank."
        )
    holders = [i for i, blk in enumerate(blocks) if _ranks(blk, name)]
    if len(holders) != 2:
        where = (
            "no block"
            if not holders
            else f"{len(holders)} block{'s' if len(holders) > 1 else ''} "
            f"({', '.join(map(str, holders))})"
        )
        raise ValueError(
            f"cascade: section {name!r} has ports {name}@1 ... in {where}; "
            "it must join exactly two blocks."
        )
    counts = []
    for i in holders:
        ranks = _ranks(blocks[i], name)
        if sorted(r for r in ranks if r is not None) != list(range(1, len(ranks) + 1)):
            raise ValueError(
                f"cascade: section {name!r} in block {i} has ranks "
                f"{ranks}; they must be 1..N, once each."
            )
        counts.append(len(ranks))
    if counts[0] != counts[1]:
        raise ValueError(
            f"cascade: section {name!r} has {counts[0]} modes in block "
            f"{holders[0]} and {counts[1]} in block {holders[1]}; the two "
            "ends must keep the same number."
        )
    return _Section(name, holders[0], holders[1], counts[0])


def _check_exposed(blocks: list[Block], plan: list[_Section]) -> None:
    """Every port that is not a section's occurs in one block only."""
    internal = {sec.port(m) for sec in plan for m in range(1, sec.n + 1)}
    seen: dict[str, int] = {}
    for i, blk in enumerate(blocks):
        for port in blk.s.ports:
            if port in internal:
                continue
            if port in seen:
                raise ValueError(
                    f"cascade: port {port!r} occurs in blocks {seen[port]} and "
                    f"{i} and is not a section's. To join two blocks at one "
                    "plane, give the ports WavePort.kept_modes(...) and a "
                    "section of length 0: sections={'name': 0.0}."
                )
            seen[port] = i


def _check_joined(blocks: list[Block], plan: list[_Section]) -> None:
    """Every joined port is a wave port mode of its rank, unshifted, no layer."""
    for sec in plan:
        for i in (sec.a, sec.b):
            blk = blocks[i]
            offsets = dict(blk.offsets)
            for m in range(1, sec.n + 1):
                port = sec.port(m)
                mode = blk.port_modes.get(port)
                if mode is None:
                    raise ValueError(
                        f"cascade: joined port {port!r} of block {i} is not a "
                        "wave port mode (no port_modes entry); build the "
                        "section's ports with WavePort.kept_modes."
                    )
                if mode.mode != m:
                    raise ValueError(
                        f"cascade: joined port {port!r} of block {i} carries "
                        f"mode {mode.mode}, not {m}."
                    )
                if offsets.get(port, 0.0) != 0.0:
                    raise ValueError(
                        f"cascade: joined port {port!r} of block {i} has offset "
                        f"{offsets[port]:g} m; a joined plane cannot be "
                        "shifted (the section's length is plane to plane)."
                    )
                if mode.pml is not None:
                    raise ValueError(
                        f"cascade: joined port {port!r} of block {i} has a "
                        "perfectly matched layer in its section; close the "
                        "section with walls, or solve the device whole."
                    )


def _lengths(sections: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
    """The section lengths as scalars; static shape and concrete value checks."""
    out = {}
    for name, value in sections.items():
        if np.ndim(value) != 0:
            raise ValueError(
                f"cascade: the length of section {name!r} must be a scalar "
                f"(sweep it with jax.vmap), got shape {np.shape(value)}."
            )
        if not is_traced(value):
            v = np.asarray(value)
            if np.iscomplexobj(v) or not np.isfinite(v) or v < 0:
                raise ValueError(
                    f"cascade: the length of section {name!r} must be real, "
                    f"finite and >= 0 (m), got {value!r}."
                )
        out[name] = jnp.asarray(value)
    return out


def _check_values(blocks: list[Block], plan: list[_Section]) -> None:
    """The checks that read values, on the concrete ones."""
    k0 = 2 * np.pi * np.asarray(blocks[0].s.frequency, dtype=float) / scipy.constants.c
    for sec in plan:
        ends = []
        for i in (sec.a, sec.b):
            modes = [blocks[i].port_modes[sec.port(m)] for m in range(1, sec.n + 1)]
            for mode in modes:
                z = mode.Z_PV
                if not is_traced(z) and np.isfinite(np.asarray(z)).any():
                    raise ValueError(
                        f"cascade: joined port {mode.port!r} of block {i} has a "
                        "voltage path; a joined port carries a mode, not a voltage."
                    )
            gammas = [mode.gamma for mode in modes]
            ends.append(None if is_traced(gammas) else np.stack(gammas, axis=-1))
        _check_normals(blocks, sec)
        for i, g in zip((sec.a, sec.b), ends, strict=True):
            if g is None:
                continue
            # beta = gamma / i; the last kept mode must not propagate (the port
            # mode sweeps' test, with k0 in place of the materials' k_max)
            beta = g[:, -1] / 1j
            if (beta.real > np.maximum(_PROPAGATES * k0, np.abs(beta.imag))).any():
                raise ValueError(
                    f"cascade: kept mode {sec.n} of section {sec.name!r} propagates "
                    f"at some frequency in block {i}; keep modes beyond the last "
                    "propagating one (a larger N in WavePort.kept_modes)."
                )
        if ends[0] is not None and ends[1] is not None:
            mismatch = np.abs(ends[0] - ends[1]) / k0[:, None]
            if mismatch.max() > _MISMATCH:
                raise ValueError(
                    f"cascade: the two blocks do not model the same guide at "
                    f"section {sec.name!r}: |gamma_A - gamma_B|/k0 reaches "
                    f"{mismatch.max():.3g} (limit {_MISMATCH:g})."
                )


def _field(mode: PortModeResult) -> Any:
    """The port field a mode carries, E or (an Hz line port's) H; None under a trace."""
    return mode.e_field if mode.e_field is not None else mode.h_field


def _check_normals(blocks: list[Block], sec: _Section) -> None:
    """The two planes of a section face each other (antiparallel inward normals)."""
    for m in range(1, sec.n + 1):
        fa = _field(blocks[sec.a].port_modes[sec.port(m)])
        fb = _field(blocks[sec.b].port_modes[sec.port(m)])
        if fa is None or fb is None:
            continue
        try:
            na, nb = np.asarray(fa.normal), np.asarray(fb.normal)
        except NonDifferentiableError:
            continue
        # antiparallel up to 1e-6 in the cosine (an angle of about 1.4e-3 rad):
        # far above the round-off of unit normals, far below any tilted plane
        if float(na @ nb) > -1 + 1e-6:
            raise ValueError(
                f"cascade: the planes of section {sec.name!r} do not face each "
                f"other: the inward normals {fa.normal} and {fb.normal} are not "
                "antiparallel."
            )


# --- gauge --------------------------------------------------------------------


def _concrete(mode: PortModeResult) -> bool:
    """The mode carries a port field with its basis and coordinates (not traced)."""
    field = _field(mode)
    return field is not None and getattr(field, "_vector", None) is None


def _kappa(mode: PortModeResult) -> np.ndarray:
    r"""``(n_freq,)``: the pseudo-wave scale of a mode, from its concrete field.

    :math:`\kappa = \sqrt{p(\omega)\,\gamma\,u^\mathsf{T} M(\omega) u}`,
    principal root, with the expression of the solve (the port without a
    voltage path): the engine recomputes it from the stored field, so a block
    solved outside a transformation needs no copy of it.
    """
    field = _field(mode)
    space, basis, coords = field._space, field._basis, field._coords  # noqa: SLF001
    omega = 2 * np.pi * np.asarray(mode.frequency, float)
    gamma = np.asarray(mode.gamma, complex) * space.section.scale
    pairing = [
        space.pairing(basis @ c, g, w)
        for c, g, w in zip(coords, gamma, omega, strict=True)
    ]
    return np.sqrt(np.array(pairing) + 0j)


def _overlap_grid(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Gauss points on block A's port line, the same points on B's, and weights.

    The line is cut at the nodes of both ends (B's nodes mapped along the
    common transverse coordinate), and each piece takes ``order + 1`` Gauss
    points. On a piece both fields are polynomials of degree at most ``order``,
    so their product, of degree :math:`2 \cdot \mathrm{order}`, is integrated
    exactly (Gauss with :math:`n` points is exact to degree :math:`2n - 1`),
    whether or not the two meshes conform. B's points are A's translated along
    the normal onto B's plane.

    Returns:
        ``(points_a (n, 2), points_b (n, 2), weights (n,))``; the points in
        metres, the weights in A's local length unit.
    """
    sa, sb = a._space.section, b._space.section  # noqa: SLF001
    order = max(a._space.order, b._space.order)  # noqa: SLF001
    ta = np.asarray(sa.points, float)
    nodes_b = sa.to_local(sb.to_global(np.asarray(sb.points, float)))[0]
    lo, hi = ta.min(), ta.max()
    tol = 1e-9 * (hi - lo)
    cuts = np.unique(np.concatenate([ta, np.clip(nodes_b, lo, hi)]))
    cuts = cuts[np.r_[True, np.diff(cuts) > tol]]
    x, w = np.polynomial.legendre.leggauss(order + 1)
    left, right = cuts[:-1, None], cuts[1:, None]
    t = (0.5 * (right - left) * x + 0.5 * (right + left)).ravel()
    weights = (0.5 * (right - left) * w).ravel()
    points_a = sa.to_global(t)
    normal = np.asarray(sa.normal, float)
    gap = (np.asarray(sb.origin, float) * sb.scale - points_a) @ normal
    return points_a, points_a + gap[:, None] * normal, weights


def _line_weight(field: Any, frequency: Any, points: np.ndarray) -> np.ndarray:
    r"""``(n_freq, n_points)``: the line weight :math:`w=\sum_j \theta_j w_j`.

    :math:`w` is :math:`1/\mu_r` for Ez and :math:`1/\varepsilon_r(\omega)` for
    Hz: the weight under which the modes of one line are orthogonal. The points
    lie inside one segment each (the Gauss pieces are cut at the line's nodes).
    """
    space = field._space  # noqa: SLF001
    section = space.section
    omega = 2 * np.pi * np.asarray(frequency, float)
    t = section.to_local(points)[0]
    ends = np.asarray(section.points, float)[np.asarray(section.segments, int)]
    order = np.argsort(ends.min(axis=1))
    low = ends.min(axis=1)[order]
    segment = order[np.searchsorted(low, t, side="right") - 1]
    values = np.array(space.weight_values)[:, segment]
    return space.weights(omega) @ values


def _pairing_sign(section: str, m: int, a: PortModeResult, b: PortModeResult) -> float:
    r"""The sign :math:`t_m` aligning mode ``m`` of block B to A, from their fields.

    :math:`t_m = \mathrm{sign}\,\mathrm{Re}(r_m \kappa_A/\kappa_B)` with
    :math:`r_m = \int w e_A e_B / \int w e_A^2`, per frequency (the overlaps
    are unconjugated, as the reciprocity pairing is), :math:`w` the weight of
    block A's line (:math:`1/\varepsilon_r` for Hz, under which its modes are
    orthogonal).
    """
    fa, fb = _field(a), _field(b)
    if len(fa.normal) != 2:
        raise ValueError(
            f"cascade: section {section!r} joins plane ports; a field gauge "
            "needs 2D line ports."
        )
    pa, pb, w = _overlap_grid(fa, fb)
    ea, eb = np.asarray(fa.at(pa)), np.asarray(fb.at(pb))
    wt = _line_weight(fa, a.frequency, pa) * w
    iaa, ibb, iab = (
        (ea * ea * wt).sum(-1),
        (eb * eb * wt).sum(-1),
        (ea * eb * wt).sum(-1),
    )
    c = iab / np.sqrt(iaa * ibb)
    if (np.abs(c) < _SAME_MODE).any():
        raise ValueError(
            f"cascade: the two blocks do not carry the same mode {m} at section "
            f"{section!r}: the overlap |c_{m}| of its fields is "
            f"{np.abs(c).min():.3g} (below {_SAME_MODE}); check the ranks of "
            "the kept modes and the sections' guides."
        )
    q = iab / iaa * _kappa(a) / _kappa(b)
    if (np.abs(q.real) < 0.5 * np.abs(q)).any():
        raise ValueError(
            f"cascade: gauge undetermined for mode {m} of section {section!r}: "
            f"r_m kappa_A / kappa_B = {q[np.argmin(np.abs(q.real) / np.abs(q))]:.3g} "
            "is closer to the imaginary than to the real axis."
        )
    signs = np.sign(q.real)
    if (signs != signs[0]).any():
        raise ValueError(
            f"cascade: gauge changes sign across the band for mode {m} of "
            f"section {section!r}: split the band or fix the sign with a "
            "polarity (WavePort.kept_modes(..., polarity={m: (point, direction)}))."
        )
    return float(signs[0])


def _z_sign(mode: PortModeResult) -> float:
    """The sign of the polarity arrow along :math:`z` of a line port."""
    z = mode.polarity[1][2]  # ty: ignore[not-subscriptable]
    if abs(z) < 0.5:
        raise ValueError(
            f"cascade: the polarity direction {mode.polarity[1]} of port "  # ty: ignore[not-subscriptable]
            f"{mode.port!r} is not along z; a cascade joins 2D line ports."
        )
    return 1.0 if z > 0 else -1.0


def _gauges(
    blocks: list[Block], plan: list[_Section], traced: bool
) -> dict[str, np.ndarray]:
    """The sign :math:`t_m` of each mode of each section, applied to block B."""
    out = {}
    for sec in plan:
        t = np.ones(sec.n)
        for m in range(1, sec.n + 1):
            a = blocks[sec.a].port_modes[sec.port(m)]
            b = blocks[sec.b].port_modes[sec.port(m)]
            both = _concrete(a) and _concrete(b)
            if a.polarity is not None and b.polarity is not None:
                t[m - 1] = _z_sign(a) * _z_sign(b)
                if both and _pairing_sign(sec.name, m, a, b) != t[m - 1]:
                    raise ValueError(
                        f"cascade: the polarity and the pairing gauge of mode {m} "
                        f"of section {sec.name!r} disagree: put both polarity "
                        "points at the same transverse position."
                    )
            elif both:
                t[m - 1] = _pairing_sign(sec.name, m, a, b)
            else:
                remedy = (
                    f"give both ends of mode {m} a polarity: "
                    "WavePort.kept_modes(..., polarity={m: (point, direction)})"
                )
                if traced:
                    raise NonDifferentiableError(
                        f"cascade: mode {m} of section {sec.name!r} has no "
                        "polarity on both ends and, inside jax.grad, jax.jit or "
                        "jax.vmap, the blocks carry no port fields to fix its "
                        f"sign; {remedy}."
                    )
                raise ValueError(
                    f"cascade: mode {m} of section {sec.name!r} has no polarity "
                    "on both ends and its blocks carry no port fields to fix its "
                    f"sign; {remedy}."
                )
        out[sec.name] = t
    return out


# --- algebra ------------------------------------------------------------------


def _section_gamma(
    blocks: list[Block], plan: list[_Section], lengths: dict[str, jax.Array]
) -> tuple[dict, dict, list]:
    r"""The section :math:`\gamma`, its mismatch in :math:`k_0` and the decay."""
    k0 = 2 * np.pi * np.asarray(blocks[0].s.frequency, dtype=float) / scipy.constants.c
    gamma, mismatch, decay = {}, {}, []
    for sec in plan:
        ends = [
            jnp.stack(
                [blocks[i].port_modes[sec.port(m)].gamma for m in range(1, sec.n + 1)],
                axis=-1,
            )
            for i in (sec.a, sec.b)
        ]
        gamma[sec.name] = 0.5 * (ends[0] + ends[1])
        mismatch[sec.name] = jnp.abs(ends[0] - ends[1]) / k0[:, None]
        # plane to plane: exp(-gamma * length), exactly 0 where it underflows
        decay.append(jnp.exp(-gamma[sec.name] * lengths[sec.name]))
    return gamma, mismatch, decay


def _exposed(
    blocks: list[Block], names: tuple[str, ...]
) -> tuple[dict[str, PortModeResult], tuple[tuple[str, float], ...]]:
    """The modes and offsets of the exposed ports, from their blocks."""
    modes, offsets = {}, []
    for blk in blocks:
        shift = dict(blk.offsets)
        for port in blk.s.ports:
            if port in names:
                if port in blk.port_modes:
                    modes[port] = blk.port_modes[port]
                if port in shift:
                    offsets.append((port, shift[port]))
    return modes, tuple(offsets)


def _interconnect(
    s_all: jax.Array,
    e: np.ndarray,
    i: np.ndarray,
    pairs: tuple[np.ndarray, np.ndarray],
    decay: jax.Array,
) -> jax.Array:
    r"""The one-solve interconnection of S over frequency (`cascade`).

    ``pairs = (ia, ib)``: ``ia[k]`` and ``ib[k]`` index the two joined ports
    (in ``i``) of the pair with decay ``decay[:, k]``.
    """
    ia, ib = pairs
    s_ee = s_all[:, e[:, None], e[None, :]]
    if len(i) == 0:
        return s_ee
    s_ei = s_all[:, e[:, None], i[None, :]]
    s_ie = s_all[:, i[:, None], e[None, :]]
    s_ii = s_all[:, i[:, None], i[None, :]]
    g = jnp.zeros((s_all.shape[0], len(i), len(i)), dtype=decay.dtype)
    g = g.at[:, ia, ib].set(decay).at[:, ib, ia].set(decay)
    x = jnp.linalg.solve(jnp.eye(len(i), dtype=g.dtype) - s_ii @ g, s_ie)
    return s_ee + s_ei @ g @ x


def _join(
    blocks: list[Block],
    plan: list[_Section],
    lengths: dict[str, jax.Array],
    gauge: dict[str, np.ndarray],
) -> CascadeResult:
    """Block-diagonal S, the section matrix Gamma, and the one-solve interconnection."""
    first = blocks[0].s
    n_f = len(first.frequency)
    start, total = [], 0
    for blk in blocks:
        start.append(total)
        total += len(blk.s.ports)
    # the sign of every port of the block-diagonal S: B's modes carry t_m
    sign = np.ones(total)
    where: dict[tuple[int, str], int] = {}
    for i, blk in enumerate(blocks):
        for j, port in enumerate(blk.s.ports):
            where[i, port] = start[i] + j
    for sec in plan:
        for m in range(1, sec.n + 1):
            sign[where[sec.b, sec.port(m)]] = gauge[sec.name][m - 1]
    dtype = jnp.result_type(*(blk.s.values.dtype for blk in blocks), jnp.complex64)
    s_all = jnp.zeros((n_f, total, total), dtype=dtype)
    for i, blk in enumerate(blocks):
        n = len(blk.s.ports)
        s_all = s_all.at[:, start[i] : start[i] + n, start[i] : start[i] + n].set(
            blk.s.values
        )
    s_all = s_all * (sign[:, None] * sign[None, :])

    joined = {
        (i, port)
        for sec in plan
        for i in (sec.a, sec.b)
        for port in (sec.port(m) for m in range(1, sec.n + 1))
    }
    ext_names, ext_idx, z_ref = [], [], []
    for i, blk in enumerate(blocks):
        for j, port in enumerate(blk.s.ports):
            if (i, port) not in joined:
                ext_names.append(port)
                ext_idx.append(start[i] + j)
                z_ref.append(blk.s.z_ref[:, j])
    int_idx = [
        where[i, sec.port(m)]
        for sec in plan
        for i in (sec.a, sec.b)
        for m in range(1, sec.n + 1)
    ]
    position = {g: k for k, g in enumerate(int_idx)}

    gamma, mismatch, decay = _section_gamma(blocks, plan, lengths)
    ia, ib = [], []
    for sec in plan:
        for m in range(1, sec.n + 1):
            ia.append(position[where[sec.a, sec.port(m)]])
            ib.append(position[where[sec.b, sec.port(m)]])
    e, i_ = np.asarray(ext_idx, dtype=int), np.asarray(int_idx, dtype=int)
    d = jnp.concatenate(decay, axis=-1) if decay else jnp.zeros((n_f, 0), complex)
    s_ext = _interconnect(s_all, e, i_, (np.asarray(ia), np.asarray(ib)), d)

    names = tuple(ext_names)
    s = NetworkParameters(
        values=s_ext,
        z_ref=jnp.stack(z_ref, axis=-1) if z_ref else jnp.zeros((n_f, 0)),
        frequency=first.frequency,
        ports=names,
        excited=names,
    )
    modes, offsets = _exposed(blocks, names)
    return CascadeResult(
        s=s,
        port_modes=modes,
        sections=lengths,
        gamma=gamma,
        gamma_mismatch=mismatch,
        polarization=blocks[0].polarization,
        offsets=offsets,
    )
