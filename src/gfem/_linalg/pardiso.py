"""gfem's MKL PARDISO inverse type, with gfem's flags (the ``mkl`` extra).

NGSolve's MKL PARDISO wrapper reads its flags from the matrix it factors
(``inverse_flags``). A matrix NGSolve builds itself, such as BDDC's coarse
(wirebasket) matrix, carries none, so the wrapper's defaults would factor it
as unsymmetric LU with minimum degree and up to 16 refinement steps. gfem
registers a subclass that applies `PARDISO_FLAGS` itself, under two inverse
names: one symmetric, one for general operators (without ``symmetric``).
`pardiso_inverse` gives the name to pass as ``inverse=``. NGSolve's own
``"pardiso"`` name is never used for them.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar

import ngsolve as ngs

from gfem._linalg import Symmetry

#: gfem's flags for NGSolve's MKL PARDISO wrapper: MKL's OpenMP parallel METIS
#: ordering (the same nested dissection and fill as serial METIS, with a
#: faster analysis, phase 11, at 2-4 threads), the symmetric matrix type
#: (gfem stores the full matrix, so NGSolve would otherwise factor it as
#: unsymmetric LU, with twice the fill), and no iterative refinement.
#: NGSolve's wrapper asks for up to 16 refinement steps with no
#: backward-error stop: they stagnate without lowering the residual and make
#: a solve cost about 4.5 substitutions instead of one. 0 is MKL's default
#: (two steps only after a perturbed pivot).
PARDISO_FLAGS = {"ordering": "parallel_metis", "symmetric": True, "refinement": 0}
#: The inverse name of gfem's PARDISO with `PARDISO_FLAGS` (``A^T = A``).
SYMMETRIC = "gfem_pardiso"
#: The inverse name of gfem's PARDISO for a general operator: `PARDISO_FLAGS`
#: without ``symmetric`` (LU).
GENERAL = "gfem_pardiso_general"


@functools.cache
def pardiso_available() -> bool:
    """Whether NGSolve found MKL PARDISO in a pip ``mkl`` (the ``mkl`` extra).

    Reads private attributes of NGSolve's wrapper: if they are missing (another
    NGSolve version), gfem's solvers stay sparse Cholesky.
    """
    try:
        from ngsolve.solvers import mkl_pardiso
    except ImportError:
        return False
    return (
        getattr(mkl_pardiso, "_pardiso", None) is not None
        and getattr(mkl_pardiso, "_mkl_free_buffers", None) is not None
    )


def pardiso_inverse(symmetry: Symmetry) -> str | None:
    """The PARDISO inverse name for an operator of ``symmetry``, or ``None``.

    `GENERAL` for ``"general"``, `SYMMETRIC` otherwise; ``None`` without MKL
    PARDISO (`pardiso_available`). The type is registered on first use.
    """
    if not pardiso_available():
        return None
    _register()
    return GENERAL if symmetry == "general" else SYMMETRIC


@functools.cache
def _register() -> None:
    """Register gfem's PARDISO inverse types with NGSolve, once."""
    from ngsolve.solvers.mkl_pardiso import MKLPardiso

    class GfemPardiso(MKLPardiso):
        """NGSolve's MKL PARDISO wrapper with `PARDISO_FLAGS`."""

        flags: ClassVar[dict[str, Any]] = PARDISO_FLAGS

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._apply_flags(self.flags)

        def _call_pardiso(self, phase: int) -> None:
            # PARDISO refuses an empty matrix: a coarse problem whose DOFs
            # are all fixed has nothing to factor or solve
            if self._n[0] > 0:  # ty: ignore[unresolved-attribute]
                super()._call_pardiso(phase)

    class GfemPardisoGeneral(GfemPardiso):
        """gfem's PARDISO for a general operator: unsymmetric LU."""

        flags: ClassVar[dict[str, Any]] = {
            k: v for k, v in PARDISO_FLAGS.items() if k != "symmetric"
        }

    ngs.la.RegisterInverseType(SYMMETRIC, GfemPardiso)
    ngs.la.RegisterInverseType(GENERAL, GfemPardisoGeneral)
