"""The electromagnetic solve domain on the NGSolve mesh.

Excluded volume entities are left out of every electromagnetic space
(``definedon`` the remaining domains, then ``Compress``), so their shells
become exterior boundaries of the solve, as in Palace, which cracks them.
"""

from __future__ import annotations

import ngsolve as ngs

from gfem._boundaries import PEC
from gfem._engine.mesh import EngineMesh
from gfem._periodic import pair_names
from gfem.model import Model


def em_region(model: Model, built: EngineMesh) -> ngs.Region:
    """The NGSolve volume region of the model's electromagnetic solve domain.

    Args:
        model: The model; ``built`` must come from ``model.mesh``.
        built: The engine mesh.
    """
    tags = {r.name: r.tag for r in model.mesh.regions if r.dim == 3}
    out = {(tags[name], ent) for name, ent in model.excluded_entities}
    bits = ngs.BitArray(len(built.domains))
    bits.Clear()
    for k, key in enumerate(built.domains):
        if key not in out:
            bits.Set(k)
    return ngs.Region(built.mesh, ngs.VOL, bits)


def hcurl_space(
    model: Model,
    built: EngineMesh,
    order: int,
    *,
    complex_valued: bool = True,
    dirichlet: tuple[str, ...] = (),
    periodic: bool = False,
    phases: tuple[complex, ...] | None = None,
) -> ngs.FESpace:
    """First-kind Nédélec space of ``order`` on the electromagnetic domain.

    DOFs of excluded volumes are compressed away, so the space's size is the
    system size. The tangential DOFs on the ``dirichlet`` surfaces (exact
    names, e.g. `PEC`) are fixed: left out of ``FreeDofs()``, which BDDC
    reads. The DOF numbering does not depend on ``dirichlet``.

    With ``periodic``, the space is periodic over the point identifications
    of ``built`` (the model's periodic pairs, `to_ngsolve`): each second
    surface's DOFs are its twins' on the first, times the pair's Bloch
    phase (``phases``, `gfem._periodic.bloch_phases`; ``None`` for plain
    periodicity), and are compressed away. Its DOF numbering is not the
    unpaired space's: map with the expansion P, never by index.
    """
    space = ngs.HCurl(
        built.mesh,
        order=order,
        type1=True,
        complex=complex_valued,
        definedon=em_region(model, built),
        dirichlet=_bc_numbers(built, dirichlet),
    )
    return _compressed(space, periodic=periodic, phases=phases)


def _compressed(
    space: ngs.FESpace, *, periodic: bool, phases: tuple[complex, ...] | None
) -> ngs.FESpace:
    """``space`` compressed, periodic first with ``periodic`` (Bloch ``phases``)."""
    if periodic:
        space = ngs.Periodic(space, phase=None if phases is None else list(phases))
    return ngs.Compress(space)


def _bc_numbers(built: EngineMesh, names: tuple[str, ...]) -> list[int]:
    """1-based NGSolve boundary numbers of these surface names (exact names)."""
    wanted = set(names)
    return [j + 1 for j, n in enumerate(built.mesh.GetBoundaries()) if n in wanted]


def pec_surfaces(model: Model) -> tuple[str, ...]:
    """The `PEC` surfaces of the model, sorted: the Dirichlet surfaces of a space."""
    return tuple(sorted({b.surface for b in model.boundaries if isinstance(b, PEC)}))


def domain_key(model: Model) -> tuple:
    """The electromagnetic physics domain, as a cache key.

    Two models with the same mesh topology and excluded entities share
    their spaces.
    """
    return ("em", model._domain.excluded)  # noqa: SLF001


def space_key(
    model: Model,
    kind: str,
    order: int,
    *,
    periodic: bool = False,
    phases: tuple[complex, ...] | None = None,
    real: bool = False,
) -> tuple:
    """The cache key of a ``kind`` space (``"hcurl"``, ``"h1"``) of ``order``.

    Its `PEC` surfaces fixed (`pec_surfaces`); with ``periodic``, periodic
    over the model's pairs with Bloch ``phases`` (``None``: plain
    periodicity); ``real`` for a real-valued space. The one key builder of
    `gfem._engine.cache.Discretization.space` for these spaces, so a paired
    and an unpaired space never share a key.
    """
    key = (domain_key(model), kind, order, pec_surfaces(model))
    if periodic:
        key = (*key, pair_names(model._periodic), phases)  # noqa: SLF001
    return (*key, "real") if real else key


def h1_space(
    model: Model,
    built: EngineMesh,
    order: int,
    dirichlet: tuple[str, ...] = (),
    *,
    periodic: bool = False,
    phases: tuple[complex, ...] | None = None,
) -> ngs.FESpace:
    """Real H1 space of ``order`` on the electromagnetic domain, compressed.

    The DOFs on the ``dirichlet`` surfaces (exact names) are fixed: they are
    left out of ``FreeDofs()``, which BDDC reads. With ``periodic``, periodic
    over the model's pairs as `hcurl_space`; Bloch ``phases`` make it
    complex.
    """
    space = ngs.H1(
        built.mesh,
        order=order,
        complex=phases is not None,
        definedon=em_region(model, built),
        dirichlet=_bc_numbers(built, dirichlet),
    )
    return _compressed(space, periodic=periodic, phases=phases)


def boundary_region(built: EngineMesh, surfaces: tuple[str, ...]) -> ngs.Region:
    """The NGSolve boundary region of these surface names (exact names)."""
    names = built.mesh.GetBoundaries()
    bits = ngs.BitArray(len(names))
    bits.Clear()
    wanted = set(surfaces)
    for j, name in enumerate(names):
        if name in wanted:
            bits.Set(j)
    return ngs.Region(built.mesh, ngs.BND, bits)
