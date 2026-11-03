"""The problems with no distributed path: one case per refusal key (ADR 0072).

`tests/unit/test_mpi.py` raises each one on a stub communicator, and
`tests/mpi/test_launch.py` raises each one on every rank of a launched job.
Nothing here solves. Every model is built from a directory, so a launched
function can build it from arguments that pickle.
"""

from pathlib import Path

import gradient_cases as cases
import periodic_cases as pc

import gfem

FREQUENCY = 5e9

#: Two terminals of `stack`, which `cases.stack` names by its layers.
TERMINALS = {"top": "top", "mid": "oxide__nitride"}


def stack(directory: Path) -> gfem.Model:
    """A small electrostatic stack with its materials set."""
    return cases.stack(directory).replace(
        materials={
            "oxide": gfem.Material(eps_r=3.9),
            "nitride": gfem.Material(eps_r=7.5),
        }
    )


def _line(directory: Path) -> gfem.Model:
    return cases.layered_line(directory)


def _cell(directory: Path) -> gfem.Model:
    return pc.c1(pc.c1_mesh(directory))


def refused(key: str, directory: Path):  # noqa: PLR0911
    """``(model, problem, options)`` of a solve refused under ``key``."""
    direct = gfem.SolverOptions(linear_solver="direct")
    match key:
        case "adaptive_sweep":
            problem = gfem.Driven([1e9, 5e9], sweep="adaptive")
            return _line(directory), problem, direct
        case "eigenmode":
            return _line(directory), gfem.Eigenmode(target=FREQUENCY, n_modes=1), direct
        case "electrostatic":
            return _line(directory), gfem.Electrostatic(terminals=TERMINALS), direct
        case "periodic":
            return _cell(directory), gfem.Driven(FREQUENCY), direct
        case "bloch":
            problem = gfem.Driven(FREQUENCY, wave_vector=pc.C1_K)
            return _cell(directory), problem, direct
        case "frequency_workers":
            options = gfem.SolverOptions(linear_solver="direct", frequency_workers=2)
            return _line(directory), gfem.Driven([1e9, 5e9]), options
        case "device":
            options = gfem.SolverOptions(linear_solver="direct", device="cuda")
            return _line(directory), gfem.Driven(FREQUENCY), options
        case "bddc_order_1":
            options = gfem.SolverOptions(order=1, linear_solver="bddc_gmres")
            return _line(directory), gfem.Driven(FREQUENCY), options
        case "bddc_order_1_auto":
            options = gfem.SolverOptions(order=1)
            return _line(directory), gfem.Driven(FREQUENCY), options
        # No distributed solve exists yet, so these two raise as well.
        case "driven":
            return _line(directory), gfem.Driven(FREQUENCY), direct
        case "port_mode":
            return _line(directory), gfem.PortMode("p1", FREQUENCY), None
    raise ValueError(key)


#: Every case of `refused`.
KEYS = (
    "adaptive_sweep",
    "eigenmode",
    "electrostatic",
    "periodic",
    "bloch",
    "frequency_workers",
    "device",
    "bddc_order_1",
    "bddc_order_1_auto",
    "driven",
    "port_mode",
)


def feature(key: str) -> str:
    """The ``feature`` a case raises."""
    return key.removesuffix("_auto")
