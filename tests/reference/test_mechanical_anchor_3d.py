r"""The anchor loss of a 3D cantilever normal to a half-space, against Wilson-Rae.

A beam :math:`l = 20\,h` long, :math:`h = 1` um thick (the bending direction) and
:math:`w = 2` um wide, on the free face of a silicon-like half-space ended by a
perfectly matched layer (`elastic_pml_cases.anchored_cantilever_3d`). Its
quality factor is compared with the rigid-root result of Wilson-Rae (2008) and
Judge et al. (2007),

.. math::

    Q^{-1} = A\,\frac{w}{l}\Big(\frac{h}{l}\Big)^4, \qquad A = 0.3314
    \quad (\nu = 0.25),

which prints as :math:`Q^{-1} / [(w/l)(h/l)^4]`. The published coefficients
differ by 7 to 12 % between the sources and the rigid-root theory leaves out
the root's compliance, so the test asserts only a factor 2; the ratio, the
unknowns, the time and the memory are an observation.
"""

import resource
import time

import elastic_pml_cases as ec
import pytest

import gfem

pytestmark = pytest.mark.reference

RATIO = 20.0


def test_the_anchor_loss_of_a_3d_cantilever_is_within_a_factor_two_of_wilson_rae(
    tmp_path,
):
    start = time.perf_counter()
    cantilever = ec.anchored_cantilever_3d(tmp_path / "beam.msh", RATIO)
    meshed = time.perf_counter() - start
    problem = gfem.MechanicalEigenmode(
        cantilever.target,
        pml=gfem.PML(["layer"], target_reflection=1e-8),
        target_quality_factor=cantilever.quality,
    )
    start = time.perf_counter()
    result = gfem.solve(cantilever.model, problem, options=gfem.SolverOptions(order=2))
    solved = time.perf_counter() - start
    omega = complex(result.omega[0])
    quality = omega.real / (2 * omega.imag)
    width, ratio = cantilever.width, cantilever.ratio
    coefficient = 1 / quality / ((width / ratio) * ratio**-4)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # GiB, Linux
    print(
        f"A = Q^-1 / [(w/l)(h/l)^4] = {coefficient:.4f} against "
        f"{ec.WILSON_RAE} (ratio {coefficient / ec.WILSON_RAE:.3f}); "
        f"Q = {quality:.4g}; {cantilever.tetrahedra} tetrahedra; mesh "
        f"{meshed:.0f} s, solve {solved:.0f} s; peak {peak:.1f} GiB"
    )
    # the published coefficients differ by 7 to 12 %: a factor 2 is a wrong mode
    # or a layer that reflects, not a spread
    assert 0.5 < coefficient / ec.WILSON_RAE < 2.0
