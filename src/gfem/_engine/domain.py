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
    model: Model, built: EngineMesh, order: int, *, complex_valued: bool = True
) -> ngs.FESpace:
    """First-kind Nédélec space of ``order`` on the electromagnetic domain.

    DOFs of excluded volumes are compressed away, so the space's size is the
    system size.
    """
    space = ngs.HCurl(
        built.mesh,
        order=order,
        type1=True,
        complex=complex_valued,
        definedon=em_region(model, built),
    )
    return ngs.Compress(space)
