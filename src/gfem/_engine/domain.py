"""The electromagnetic solve domain on the NGSolve mesh.

Excluded volume entities are left out of every electromagnetic space
(``definedon`` the remaining domains, then ``Compress``), so their shells
become exterior boundaries of the solve, as in Palace, which cracks them.
"""

from __future__ import annotations

import ngsolve as ngs

from gfem._engine.mesh import EngineMesh
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
) -> ngs.FESpace:
    """First-kind Nédélec space of ``order`` on the electromagnetic domain.

    DOFs of excluded volumes are compressed away, so the space's size is the
    system size. The tangential DOFs on the ``dirichlet`` surfaces (exact
    names, e.g. `PEC`) are fixed: left out of ``FreeDofs()``, which BDDC
    reads. The DOF numbering does not depend on ``dirichlet``.
    """
    space = ngs.HCurl(
        built.mesh,
        order=order,
        type1=True,
        complex=complex_valued,
        definedon=em_region(model, built),
        dirichlet=_bc_numbers(built, dirichlet),
    )
    return ngs.Compress(space)


def _bc_numbers(built: EngineMesh, names: tuple[str, ...]) -> list[int]:
    """1-based NGSolve boundary numbers of these surface names (exact names)."""
    wanted = set(names)
    return [j + 1 for j, n in enumerate(built.mesh.GetBoundaries()) if n in wanted]


def domain_key(model: Model) -> tuple:
    """The electromagnetic physics domain, as a cache key.

    Two models with the same mesh topology and excluded entities share
    their spaces.
    """
    return ("em", model._domain.excluded)  # noqa: SLF001


def h1_space(
    model: Model, built: EngineMesh, order: int, dirichlet: tuple[str, ...] = ()
) -> ngs.FESpace:
    """Real H1 space of ``order`` on the electromagnetic domain, compressed.

    The DOFs on the ``dirichlet`` surfaces (exact names) are fixed: they are
    left out of ``FreeDofs()``, which BDDC reads.
    """
    space = ngs.H1(
        built.mesh,
        order=order,
        definedon=em_region(model, built),
        dirichlet=_bc_numbers(built, dirichlet),
    )
    return ngs.Compress(space)


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
