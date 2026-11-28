"""The gradient verification matrix: every v0.1 gradient against differences.

One cell per problem type x input kind x linear solver x frequency sweep:

- problems: electrostatic capacitance (and a potential functional, and the
  region energies), driven
  S on lumped ports and on wave ports (lossless, lossy, and two modes per
  shared port surface, read through cross-mode entries), driven S (and
  probed fields) on a periodic cell with plain and Bloch periodicity,
  driven S through Floquet ports (diffraction efficiencies, cross-polarized
  entries and rows that read 0, plain and at a conical Bloch wave vector),
  eigenfrequency (frequency and loss of a simple mode); steady and
  transient heat conduction (TR-BDF2 and theta, on a kept trajectory and
  from checkpoints) and DC conduction (its conductance, and its
  dissipation through a Joule heat source); the dissipated power per
  region and sheet of a driven field, the steady rise its heat drives
  (`gfem.electromagnetic_heating`), and the heat through a sink and a cut,
  steady (`gfem.heat_flow`) and transient (`gfem.Trace.heat_out`); a
  plane wave's kept field, a point dipole's emitted power, and a dipole's
  port wave read by reciprocity (``driven_dipole_reciprocal``:
  ``source_fields=False``, uniform, and adaptive at frozen samples);
- inputs: per-element permittivity ``eps_r`` and conductivity, the same per
  principal value of a field with axes (``eps_r_axes``,
  ``conductivity_axes``: biaxial values on the `TILTED` axes, so every
  entry of each projector is exercised), the permeability ``mu_r`` and the
  loss tangent ``tan_delta`` (driven and eigenmode, on the direct solver:
  the solver and sweep paths are the permittivity's), the fill of a
  `gfem.SheetField`,
  and the mesh node coordinates (through `gfem.shape.boundary_morph`, or
  for the driven line, whose absorbing side walls touch every element, by
  moving its interface's nodes along z: `gradient_cases.interface_move`); the
  thermal conductivity, mass density and specific heat, and a heat
  source's amount and a waveform's scales (input ``heat``); a two-port
  drive's incident powers and phase (input ``drive``);
- solvers: ``direct`` and ``bddc_gmres``; sweeps: uniform and adaptive (with
  frozen samples; the objective reads S at the samples, where the reduced
  model reproduces the full solve of the perturbed design too, so the
  differences of the adaptive pipeline check the frozen-basis gradient).

The PETSc cells (`PETSC_CELLS`: ``linear_solver="petsc"``, and SLEPc seeds
for the eigenmode cell) run the same checks where the PETSc stack is
installed, and are skipped elsewhere (the ``petsc`` / ``slepc`` markers).
They stay out of `CELLS`, from which the docs build derives the Capabilities
page's claims: the verification job that runs on every PR has no PETSc.

Each cell checks the gradient along 3 random unit directions (the third
along ``i`` for a complex permittivity of principal values) against
fourth-order central differences (tests/finite_differences.py, gate 1e-6)
at tight tolerance (iterative solves to 1e-12, eigenpairs to 1e-12), and
that the Taylor remainder falls at second order at production tolerance
(`gfem.SolverOptions` defaults: iterative solves to 1e-6, eigenpairs to
1e-10), with gradient and differences both carrying that solver error.

The periodic eigenmode cells (``eigenmode_periodic_plain`` and
``eigenmode_periodic_bloch``, `_eigen_periodic_cells`) solve a damped,
lossy cell periodic along x and y (tests/periodic_cases.py, `eigen_cell`):
plain on both solvers, Bloch on ``direct`` (the only linear solver of a
Bloch eigenmode solve), where the left eigenvectors carry the gradient. Its
node direction moves the twins of every pair together (a periodic
direction, the raw node gradient).

The chain gate (``-k chain``) differentiates the whole thermal
chain on a heater slab: DC conduction -> Joule heating at a current ->
steady thermal -> thermo-optic shift (phase per length plus the core's mean
temperature), in the heater's per-element conductivity, every element's
thermal conductivity, the core's per-element thermo-optic coefficient and
the current at once, with unequal boundary temperatures (so the
source-free temperature enters), on both solvers; and its transient twin (a
phase trace and a mean-temperature trace from a uniform start). Its
electromagnetic twin (``em_chain``) differentiates driven -> electromagnetic
heating -> steady thermal -> region means in a per-element loss tangent, a
sheet's fill and resistance, every element's thermal conductivity and the
two-port drive at once.

Then the consistency checks (JAX transformations and evicted cache entries
per input kind, reciprocity against an explicit adjoint, adaptive against
uniform at the samples) and the degenerate eigenmode clusters. Pairs this
version does not differentiate are rows of the raise table
(tests/verification/test_gradient_matrix_raise_table.py), not cells: the
electrostatic conductivity (an exact zero) and sheet fill (not an
electrostatic object), eigenmode field quantities, the port mode problem.

Run the gate with ``pytest -m verification -k gradient_matrix``. The slowest
shared-surface cells (`REFERENCE_CELLS`) also carry the ``reference`` marker, so
the default tiers (``-m "not reference ..."``) leave them to the reference
tier.
"""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import piezo_cases as pz
import pml_cases
import pytest
import thermal_cases as tc
from anisotropy import TILTED, turned
from eigen_cases import steer_search
from finite_differences import (
    assert_gradient_matches,
    assert_taylor_second_order,
    unit_directions,
)
from gradient_cases import (  # the port mode cells
    BIAXIAL,
    BLEND_DIELECTRIC,
    BLEND_FREQUENCIES,
    BLEND_METAL,
    CAVITY_TARGET,
    COVER_LAW,
    EIGEN_LAWS,
    EPS0,
    GUIDE_TARGET,
    JUNCTION_CHI_L,
    JUNCTION_L,
    JUNCTION_TARGET,
    KINETIC_CAPACITANCE,
    KINETIC_TARGET,
    LAW_FREQUENCIES,
    LAWS,
    LOSSLESS_LUMPED,
    LUMPED,
    LUMPED_TARGET,
    MIRRORED_TARGET,
    OMEGA_WEIGHT,
    PORT_MODE_FREQUENCIES,
    PORT_MODE_POLARITY,
    RESONATOR_TARGET,
    SHEETS,
    STACK_PROBLEM,
    STATIC_LAW,
    WAVE_A4,
    WAVE_A4_ROOT,
    WAVE_A_ROOT,
    WAVE_A_SPLIT,
    WAVE_C,
    WAVE_C_HEIGHT,
    WAVE_C_ROOT,
    WAVE_L,
    WAVE_L_ROOT,
    WAVE_LUMPED,
    absorbing_guide,
    capacitor,
    cavity_fill,
    cavity_model,
    core_design,
    dispersive_line,
    dispersive_mesh,
    dispersive_resonator,
    face_functional,
    face_line,
    guide_model,
    hybrid_line,
    hybrid_mesh,
    interface_move,
    junction_law_line,
    junction_line,
    junction_mesh,
    junction_values,
    kinetic_cavity,
    layered_line,
    layered_resonator,
    line_axes_design,
    line_design,
    lossless_lumped_line,
    lumped_line,
    mirrored_cavity,
    on_the_sheet,
    port_mode_functional,
    port_mode_line,
    ramp,
    resonator_mesh,
    s_functional,
    scaled_law,
    shape_line,
    shared_surface_functional,
    sheet_cavity,
    sheet_fill,
    sheet_line,
    sheet_mesh,
    split_cavity,
    stack,
    stack_fields,
    terminated,
    tight,
    wave_guide,
    wave_line,
    wave_mesh,
    wave_resonator,
    with_fields,
)
from periodic_cases import (
    EG_A,
    EG_B,
    EG_BLOCH_TARGET,
    EG_H,
    EG_K,
    EG_PLAIN_TARGET,
    EG_TAN_DELTA,
    MM,
    coordinate_classes,
    eigen_cell,
    eigen_cell_design,
    eigen_cell_mesh,
)

import gfem
from gfem import _pml
from gfem._port_section import port_section
from gfem.shape import boundary_morph
from gfem.thermal import _tet_volumes

#: G's Bloch cells meet its absorbing top obliquely (`pc.G_OBLIQUE`).
pytestmark = pytest.mark.filterwarnings(pc.G_OBLIQUE)

SOLVERS = ("direct", "bddc_gmres")


@dataclasses.dataclass(frozen=True)
class Cell:
    """One cell: problem x input kind x solver x sweep (x variant)."""

    problem: str
    input: str
    solver: str
    sweep: str = ""
    variant: str = ""

    def __str__(self) -> str:
        parts = (self.problem, self.input, self.variant, self.solver, self.sweep)
        return "-".join(p for p in parts if p)


def _cells() -> list[Cell]:
    cells = []
    for solver in SOLVERS:
        cells += [
            Cell("electrostatic", "eps_r", solver),
            Cell("electrostatic", "eps_r_axes", solver),
            Cell("electrostatic", "nodes", solver, variant="capacitance"),
            Cell("electrostatic", "nodes", solver, variant="potential"),
            *(
                Cell("electrostatic", name, solver, variant="energy")
                for name in ("eps_r", "eps_r_axes", "nodes")
            ),
        ]
        for sweep in ("uniform", "adaptive"):
            cells += [
                Cell("driven", "eps_r", solver, sweep),
                Cell("driven", "conductivity", solver, sweep),
                # on first-order absorbing sides, owned by the design
                Cell("driven", "eps_r_axes", solver, sweep),
                Cell("driven", "conductivity_axes", solver, sweep),
                Cell("driven", "nodes", solver, sweep),
            ]
            sheets = (
                ("conductor", "impedance") if sweep == "uniform" else ("impedance",)
            )
            cells += [Cell("driven", "fill", solver, sweep, s) for s in sheets]
            # a lumped port's L and C, an impedance sheet's R and L
            cells.append(Cell("driven", "lumped", solver, sweep))
            # Wave ports: every port x input on the direct solver; on BDDC +
            # GMRES the lossy ports (complex port terms in the Woodbury update
            # and in the adaptive basis; the adjoint solve through the port
            # operator is the shared cells' and
            # test_the_s21_gradient_does_not_depend_on_which_ports_are_excited's)
            # in the permittivity, whose conductivity cells take
            # the same code path, and in the nodes on the uniform sweep (the
            # adaptive backward is the solver's only through those solves).
            bddc = {"eps_r"} | ({"nodes"} if sweep == "uniform" else set())
            cells += [
                Cell("driven_wave", name, solver, sweep, port)
                for port in ("lossless", "lossy")
                for name in ("eps_r", "conductivity", "nodes")
                if solver == "direct" or (port == "lossy" and name in bddc)
            ]
            # principal values beside a lossy port: the port terms read no
            # field, so one cell shows the principal-value path through them
            if solver == "direct" and sweep == "uniform":
                cells.append(Cell("driven_wave", "eps_r_axes", solver, sweep, "lossy"))
            # A shared port surface (two modes per end): every input on the
            # direct solver, the permittivity on BDDC + GMRES
            cells += [
                Cell("driven_wave", name, solver, sweep, "shared")
                for name in ("eps_r", "conductivity", "nodes")
                if solver == "direct" or (name == "eps_r" and sweep == "uniform")
            ]
            # the same surface with a loss tangent on the leads: the S
            # functional is still parallel to the excitation, so the cross-mode
            # entries' adjoint loads go through the same corrections, now complex
            if solver == "direct":
                cells += [
                    Cell("driven_wave", name, solver, sweep, "shared_lossy")
                    for name in ("eps_r", "conductivity", "nodes")
                ]
        cells += [
            Cell("eigenmode", name, solver)
            for name in (
                "eps_r",
                "conductivity",
                "eps_r_axes",
                "conductivity_axes",
                "fill",
                "lumped",
                "nodes",
            )
        ]
    # the curl (mu_r) and loss (tan_delta) paths: one solver and one sweep
    cells += [
        Cell("driven", f"{name}{form}", "direct", "uniform")
        for form in ("", "_axes")
        for name in ("mu_r", "tan_delta")
    ]
    cells += [Cell("eigenmode", name, "direct") for name in ("mu_r", "tan_delta")]
    return cells


def _driven_periodic_cells() -> list[Cell]:
    """The driven cells of a periodic model (the gradient cell G).

    Plain and Bloch periodicity on the uniform sweep for every input kind
    on the direct solver, Bloch on BDDC + GMRES too (its coarse LU and the
    transposed system), plain periodicity on the adaptive sweep (Bloch
    raises in its forward) in the permittivity and the nodes (conductivity
    and fill take the permittivity's code path), and a probed Bloch field
    (a field cotangent, loaded as P^T g on the transposed system). To keep
    the default tier's time, plain periodicity on BDDC + GMRES has the
    permittivity cell only (its pairs are the direct solver's).
    """
    inputs = ("eps_r", "conductivity", "fill", "nodes")
    return [
        *(
            Cell(problem, name, "direct", "uniform")
            for problem in ("driven_periodic_plain", "driven_periodic_bloch")
            for name in inputs
        ),
        Cell("driven_periodic_plain", "eps_r", "direct", "adaptive"),
        Cell("driven_periodic_plain", "nodes", "direct", "adaptive"),
        Cell("driven_periodic_bloch", "eps_r", "direct", "uniform", "field"),
        *(
            Cell("driven_periodic_bloch", name, "bddc_gmres", "uniform")
            for name in inputs
        ),
        Cell("driven_periodic_plain", "eps_r", "bddc_gmres", "uniform"),
    ]


FLOQUET_PROBLEMS = ("driven_floquet_plain", "driven_floquet_bloch")


def _driven_floquet_cells() -> list[Cell]:
    """The driven cells of a cell with Floquet ports (coarse F2).

    Plain periodicity and a conical Bloch wave vector (20°, 30° from the
    air) on the uniform sweep, in the permittivity and the nodes on the
    direct solver, and in the permittivity on BDDC + GMRES (the transposed
    system's Floquet term under Bloch). The objective (`_floquet_value`)
    reads a ±1 order's efficiency, a cross-polarized entry and a row that
    does not propagate at one frequency at least. One more Bloch cell keeps
    the angle fixed over its two frequencies (``fixed_angle``: ±k(f) per
    frequency), in the permittivity on the direct solver.
    """
    return [
        Cell("driven_floquet_bloch", "eps_r", "direct", "uniform", "fixed_angle"),
        *(
            Cell(problem, name, "direct", "uniform")
            for problem in FLOQUET_PROBLEMS
            for name in ("eps_r", "nodes")
        ),
        *(
            Cell(problem, "eps_r", "bddc_gmres", "uniform")
            for problem in FLOQUET_PROBLEMS
        ),
    ]


# --- periodic eigenmode cells ------------------------------------------------------


def _eigen_periodic_cells() -> list[Cell]:
    """The periodic eigenmode cells: plain on both solvers, Bloch on ``direct``.

    Bloch: the damped cell in every input, and a linear lossy one (``tan
    delta``, no damping) in the permittivity.
    """
    inputs = ("eps_r", "conductivity", "fill", "nodes")
    return [
        *(Cell("eigenmode_periodic_plain", name, "direct") for name in inputs),
        Cell("eigenmode_periodic_plain", "eps_r", "bddc_gmres"),
        *(Cell("eigenmode_periodic_bloch", name, "direct") for name in inputs),
        Cell("eigenmode_periodic_bloch", "eps_r", "direct", variant="linear"),
    ]


# --- eigenmode mode-field cells (quantum metrics) ----------------------------------


def _participation_cells() -> list[Cell]:
    """The ratios of the result (``eigenmode_participation``), on ``direct``.

    By fixture variant (`_participation`): fixture J (`junction_line`)
    lossless, with a resistive port, lossy (complex permittivity, a
    conductivity, a static loss tangent), with a `gfem.Conductor` top, with
    a capacitive junction, with fields of principal values; the shorted
    guide's second-order absorbing wall; the glued line's resistive
    `gfem.Impedance` sheet; the mirrored cavity's pair split to a relative
    gap of 2.5e-4 (above the isolation threshold 1e-4). The node cells move
    fixture J's interface (`_participation_nodes`), with the static values
    of each variant (the lossy one's loss tangent among them). The sheet
    cavity's inductive `gfem.Impedance`, with a capacitance or without,
    gives the sheet participation and the sheet's share of the mode energy
    (`_kinetic`); its node cells move the sheet. The ``"chi"`` cells read
    the dispersive shift of `gfem.quantum.dressed_modes` between fixture J's
    two lowest modes (`_chi`), the whole chain from the solve. The lumped
    cells read one ratio each of the lossy glued line in its port's L and C
    and its sheet's R, L and C, and the dressed anharmonicity of the
    lossless line's junction mode in its port's and sheet's L and C, the
    port's L also given to `gfem.quantum.dressed_modes` (`_lumped_ratio`).
    """
    key = "eigenmode_participation"
    return [
        Cell(key, "eps_r", "direct", variant="inductive"),
        Cell(key, "nodes", "direct", variant="inductive"),
        Cell(key, "eps_r", "direct", variant="resistive"),
        Cell(key, "nodes", "direct", variant="resistive"),
        Cell(key, "eps_r", "direct", variant="lossy"),
        Cell(key, "conductivity", "direct", variant="lossy"),
        Cell(key, "nodes", "direct", variant="lossy"),
        Cell(key, "eps_r", "direct", variant="conductor"),
        Cell(key, "nodes", "direct", variant="conductor"),
        Cell(key, "eps_r", "direct", variant="absorbing2"),
        Cell(key, "eps_r", "direct", variant="capacitive_port"),
        Cell(key, "nodes", "direct", variant="capacitive_port"),
        Cell(key, "fill", "direct", variant="resistive_sheet"),
        Cell(key, "eps_r", "direct", variant="near_pair"),
        Cell(key, "eps_r_axes", "direct", variant="principal"),
        Cell(key, "nodes", "direct", variant="principal"),
        Cell(key, "fill", "direct", variant="sheet"),
        Cell(key, "eps_r", "direct", variant="sheet"),
        Cell(key, "nodes", "direct", variant="sheet"),
        Cell(key, "eps_r", "direct", variant="capacitive_sheet"),
        Cell(key, "fill", "direct", variant="capacitive_sheet"),
        Cell(key, "nodes", "direct", variant="capacitive_sheet"),
        Cell(key, "eps_r", "direct", variant="chi"),
        Cell(key, "nodes", "direct", variant="chi"),
        # the lumped values: one cell per ratio (`_lumped_ratio`)
        *(
            Cell(key, "lumped", "direct", variant=v)
            for v in (
                "participation",
                "q_port",
                "bulk",
                "sheet_participation",
                "anharmonicity",
            )
        ),
    ]


def _field_cells() -> list[Cell]:
    """The normalized mode field (``eigenmode_field``), on ``direct``.

    By fixture variant (`_field`): the interface participation of fixture
    J's layer interface (lossless) and of the sheet cavity's walls (a
    `gfem.Conductor` film, in its fill); a probed field component times a
    phase, so the phase rule's multiplier is nonzero, on fixture J with a
    resistive port and on the terminated resonator (in its conductivity),
    and on the lossy glued line in its lumped values (`_lumped_ratio`).
    """
    key = "eigenmode_field"
    return [
        Cell(key, "eps_r", "direct", variant="interface"),
        Cell(key, "eps_r", "direct", variant="probe"),
        Cell(key, "conductivity", "direct", variant="probe"),
        Cell(key, "fill", "direct", variant="interface"),
        Cell(key, "lumped", "direct", variant="probe"),
    ]


# --- eigenmode with wave ports -------------------------------------------------------


def _eigen_wave_cells() -> list[Cell]:
    """An eigenmode with wave ports, on ``direct`` (`_eigen_wave`).

    ``eigenmode_wave``: case A of the shorted WR-90 guide (Q 5.8 with the
    lumped port), in the permittivity of a layer off the port face, the
    nodes of an interface off it, and an inductive lumped port's L and C;
    the objective reads omega, the wave port's coupling Q, the lumped port's
    participation and the layer's share. The ``"a4"`` variant: the windowed
    guide A4 (Q 869), omega and the coupling Q in the windows' permittivity.
    ``eigenmode_wave_shared``: case C's TE20 root, one port surface carrying
    TE10 and TE20 (its modal correction live), omega and both ports'
    coupling Q. ``eigenmode_wave_lossy``: case L, a lossy anisotropic port
    section, omega and the coupling Q.
    """
    return [
        *(
            Cell("eigenmode_wave", name, "direct")
            for name in ("eps_r", "nodes", "lumped")
        ),
        Cell("eigenmode_wave", "eps_r", "direct", variant="a4"),
        Cell("eigenmode_wave_shared", "eps_r", "direct"),
        Cell("eigenmode_wave_lossy", "eps_r", "direct"),
    ]


# --- dispersive materials ------------------------------------------------------------


def _law_cells() -> list[Cell]:
    """The driven cells of a permittivity law, on ``direct``.

    Input ``"law"``: the law's parameters, normalized as p0 (1 + t), on the
    strip line (`dispersive_line`) for each law of `LAWS` (``poles``,
    ``relaxation``) and behind first- and second-order absorbing walls on the
    law region (``absorbing1``, ``absorbing2``: their face terms), through
    wave ports (``lossy``) and on the plain periodic cell. The existing inputs
    beside a law (variant ``"law"``): the host's per-element permittivity, and
    the nodes of the shape line whose cover holds a law (its susceptibility
    and absorbing face pieces transported).
    """
    return [
        *(
            Cell("driven", "law", "direct", "uniform", variant)
            for variant in ("poles", "relaxation", "absorbing1", "absorbing2")
        ),
        Cell("driven", "eps_r", "direct", "uniform", "law"),
        Cell("driven", "nodes", "direct", "uniform", "law"),
        Cell("driven_wave", "law", "direct", "uniform", "lossy"),
        Cell("driven_periodic_plain", "law", "direct", "uniform"),
        # eigenmode omega (frequency and Q) in the cover's law parameters
        *(
            Cell("eigenmode", "law", "direct", variant=variant)
            for variant in ("lorentz", "drude_lorentz", "relaxation", "drude_lossless")
        ),
        # beside the cover's lossy Lorentz law: omega in the sub layer's
        # permittivity and the interface's nodes (E5, E6); fixture J's ratios
        # and a probed field (P1-P4: the energy permittivity per mode, the
        # kappa term, the phase rule on eps_inf)
        Cell("eigenmode", "eps_r", "direct", variant="lorentz"),
        Cell("eigenmode", "nodes", "direct", variant="lorentz"),
        *(
            Cell("eigenmode_participation", name, "direct", variant="lorentz")
            for name in ("eps_r", "lumped", "nodes")
        ),
        Cell("eigenmode_field", "eps_r", "direct", variant="lorentz"),
        *_adaptive_law_cells(),
    ]


def _adaptive_law_cells() -> list[Cell]:
    """The adaptive sweep's law cells, frozen samples.

    The ``poles`` law with its Lorentz pole moved into the band
    (`ADAPTIVE_LAWS`) and the ``lorentz`` law behind first-order absorbing
    walls, in their parameters; the host's per-element permittivity beside
    the moved ``poles`` law.
    """
    return [
        Cell("driven", "law", "direct", "adaptive", "poles"),
        Cell("driven", "law", "direct", "adaptive", "absorbing1"),
        Cell("driven", "eps_r", "direct", "adaptive", "law"),
    ]


def _blend_cells() -> list[Cell]:
    """The density of a metal-dielectric blend: cells D11 and D12.

    The strip line's core blends a constant dielectric and a Drude metal by a
    density in [0.2, 0.8], under the index scheme and the linear one.
    """
    return [
        Cell("driven", "blend", "direct", "uniform", scheme)
        for scheme in ("index", "linear")
    ]


#: The face cells' inputs by variant: a design at the wave port faces.
FACE_INPUTS = {
    "face_lossless": ("eps_r", "nodes"),
    "face_lossy": (
        "eps_r",
        "conductivity",
        "tan_delta",
        "mu_r",
        "eps_r_axes",
        "law",
        "nodes",
    ),
    "face_path": ("eps_r", "nodes"),
    "face_absorbing": ("eps_r", "law"),
    "face_shared": ("eps_r", "nodes"),
}


def _face_cells() -> list[Cell]:
    """The driven cells of a design at the wave port faces, on the uniform sweep.

    The two-layer line's lower layer "lo" touches both port faces: its
    per-element permittivity (lossless and lossy leads, with a voltage path,
    behind absorbing side walls, on a lossy shared surface), conductivity, loss
    tangent, permeability, principal values, a law's parameters, the fill and the
    `Impedance` values of a sheet beside the ports, a dipole's power through a
    port cut by a layer, and the permittivity on BDDC + GMRES.
    """
    cells = [
        Cell("driven_wave", name, "direct", "uniform", variant)
        for variant, names in FACE_INPUTS.items()
        for name in names
    ]
    cells += [
        Cell("driven_wave", "fill", "direct", "uniform", "face_sheet_conductor"),
        Cell("driven_wave", "fill", "direct", "uniform", "face_sheet_impedance"),
        Cell("driven_wave", "lumped", "direct", "uniform", "face_sheet"),
        Cell("driven_dipole", "eps_r", "direct", "uniform", "face"),
        Cell("driven_wave", "eps_r", "bddc_gmres", "uniform", "face_lossy"),
    ]
    return cells


def _electrostatic_law_cells() -> list[Cell]:
    """Cell S1: the capacitance in a law's parameters, on ``direct``.

    The capacitor's nitride holds `STATIC_LAW`: the static permittivity reads
    eps_inf and the strengths, and the frequency parameters are exact zeros.
    """
    return [Cell("electrostatic", "law", "direct", variant="static")]


def _thermal_cells() -> list[Cell]:
    """Steady and transient heat conduction, and DC conduction.

    Each transient input on both schemes, on a kept trajectory (``full``)
    and from checkpoints (``checkpoints``: ``retain_memory=0`` keeps no
    entry, so the backward reruns its forward with checkpoints).
    """
    cells = []
    for solver in SOLVERS:
        cells.append(Cell("steady_thermal", "thermal_conductivity", solver))
        for scheme in ("tr_bdf2", "theta"):
            for layout in ("full", "checkpoints"):
                variant = f"{scheme}-{layout}"
                cells += [
                    Cell("transient_thermal", name, solver, variant=variant)
                    for name in (
                        "thermal_conductivity",
                        "mass_density",
                        "specific_heat",
                    )
                ]
                cells += [
                    Cell(
                        "transient_thermal", "heat", solver, variant=f"{variant}-{what}"
                    )
                    for what in ("amount", "waveform")
                ]
        cells += [
            Cell("dc_conduction", "conductivity", solver, variant=v)
            for v in ("resistance", "dissipation")
        ]
    return cells


def _heat_cells() -> list[Cell]:
    """Electromagnetic heat and heat flow.

    The dissipated power per region and sheet, and the steady rise an
    electromagnetic heat source drives, in the electromagnetic inputs (a
    complex permittivity, the loss tangent per principal value, the
    conductivity, a law's parameters, the lumped values, a `Conductor` and an
    `Impedance` sheet's fill), the two-port drive's incident powers and phase
    and the thermal conductivity; the heat through a sink and a cut, steady
    and transient (each scheme and layout, as the transient cells), in what
    the thermal solve reads. Every cell on the direct solver: the heat kernel
    reads only the field, so one row (the dissipated power in the
    conductivity) repeats on BDDC + GMRES.
    """
    cells = [
        Cell("dissipated_power", name, "direct")
        for name in ("eps_r", "conductivity", "tan_delta_axes", "law", "lumped")
    ]
    cells.append(Cell("dissipated_power", "conductivity", "bddc_gmres"))
    cells += [
        Cell("dissipated_power", "blend", "direct", variant=scheme)
        for scheme in ("index", "linear")
    ]
    cells += [
        Cell("electromagnetic_heating", name, "direct")
        for name in ("eps_r", "conductivity", "lumped", "drive", "thermal_conductivity")
    ]
    for problem in ("dissipated_power", "electromagnetic_heating"):
        cells += [
            Cell(problem, "fill", "direct", variant=s)
            for s in ("conductor", "impedance")
        ]
    cells += [
        Cell("steady_heat_flow", name, "direct")
        for name in ("thermal_conductivity", "heat")
    ]
    for scheme in ("tr_bdf2", "theta"):
        for layout in ("full", "checkpoints"):
            variant = f"{scheme}-{layout}"
            cells += [
                Cell("transient_heat_out", name, "direct", variant=variant)
                for name in ("thermal_conductivity", "mass_density", "specific_heat")
            ]
            cells += [
                Cell("transient_heat_out", "heat", "direct", variant=f"{variant}-{w}")
                for w in ("amount", "waveform")
            ]
    return cells


def _energy_cells() -> list[Cell]:
    """The electric energy per region of a field, on ``direct``.

    ``electric_energy``: the driven line's energy in the permittivity (and its
    principal values), the conductivity, the loss tangent, a sheet's fill, the
    node coordinates of its interface and the parameters of a permittivity law
    (whose Brillouin weight is in JAX). ``electric_energy_eigenmode``: the
    energy of mode 0's field on fixture J in the sub layer's permittivity,
    plain and beside the cover's lossy Lorentz law (the weight moves with the
    mode's frequency). ``electric_energy_periodic``: the energy of the
    periodic cell G's field in its permittivity.
    """
    key = "electric_energy"
    return [
        *(
            Cell(key, name, "direct")
            for name in ("eps_r", "eps_r_axes", "conductivity", "tan_delta", "nodes")
        ),
        Cell(key, "law", "direct", variant="poles"),
        *(Cell(key, "fill", "direct", variant=s) for s in ("conductor", "impedance")),
        Cell("electric_energy_eigenmode", "eps_r", "direct", variant="probe"),
        Cell("electric_energy_eigenmode", "eps_r", "direct", variant="lorentz"),
        Cell("electric_energy_periodic", "eps_r", "direct", "uniform"),
    ]


def _source_cells() -> list[Cell]:
    """The source columns' cells: a plane wave's kept field and dipoles' power.

    On the direct solver and the uniform sweep: the load of a source column
    is the only new path, and it is the same for every solver. The node
    coordinates of a plane wave or a magnetic dipole raise (`OUT_OF_SCOPE`).
    ``driven_dipole_reciprocal``: the dipole's port wave read by reciprocity
    (``source_fields=False``, no source column), whose backward solves one
    adjoint column per port; on the adaptive sweep at frozen samples, where
    those columns are solved at full size.
    """
    return [
        *(
            Cell("driven_planewave", name, "direct", "uniform")
            for name in ("eps_r", "conductivity", "fill")
        ),
        *(
            Cell("driven_dipole", name, "direct", "uniform")
            for name in ("eps_r", "conductivity", "fill", "nodes")
        ),
        *(
            Cell("driven_magnetic_dipole", name, "direct", "uniform")
            for name in ("eps_r", "conductivity", "fill")
        ),
        *(
            Cell("driven_dipole_reciprocal", name, "direct", "uniform")
            for name in ("eps_r", "conductivity", "fill", "nodes")
        ),
        Cell("driven_dipole_reciprocal", "eps_r", "direct", "adaptive"),
    ]


def _piezoelectric_cells() -> list[Cell]:
    """The piezoelectric section (``piezoelectric``), on ``direct``.

    The admittance and the clamped capacitance of the line-electrode
    transducer of tests/piezo_cases.py, in the principal permittivities of
    its crystal (``eps_r_axes``) and in the nodes through a pitch map
    (``nodes``). The other material inputs and the meshed fingers' maps are
    gated in tests/unit/test_piezoelectric_gradients.py and
    tests/unit/test_piezoelectric_nodes.py. The direct solver is the only one
    of a piezoelectric solve.
    """
    return [
        Cell("piezoelectric", "eps_r_axes", "direct"),
        Cell("piezoelectric", "nodes", "direct"),
    ]


# --- port mode cells -----------------------------------------------------------------
# `gfem.PortMode` differentiated by its 2D adjoint: the design sits on the port
# face (the wave line's layer "lo" touches both ends), the objective reads gamma,
# Z_PV and the field at five points, so the quadrant gauge and the overlap chain
# are inside the gate. FD runs at port_mode_tolerance=1e-12, the Taylor test at
# the production tolerance.

PORT_MODE_STRIP_POINTS = np.array(
    [[0.1e-6, 0.05e-6, 0.0], [-0.3e-6, 0.1e-6, 0.0], [0.4e-6, -0.1e-6, 0.0]]
)
PORT_MODE_STRIP_E0 = 3e7
PORT_MODE_TAYLOR = {
    "eps_r": 3e-2,
    "conductivity": 1.0,
    "tan_delta": 3e-2,
    "mu_r": 3e-2,
}


def _port_mode_cells() -> list[Cell]:
    """The `gfem.PortMode` cells: every input a wave port's section reads."""
    cells = [
        Cell("port_mode", name, "direct")
        for name in ("eps_r", "conductivity", "tan_delta", "mu_r", "eps_r_axes", "law")
    ]
    cells += [
        # a hybrid mode of a shared port surface, a layer cutting the face
        Cell("port_mode", "eps_r", "direct", variant="shared"),
        Cell("port_mode", "eps_r", "direct", variant="pml"),
        # a polarity reference fixes the phase at every frequency on its own
        Cell("port_mode", "eps_r", "direct", variant="polarity"),
        # the face stretches in its plane; the strip's nodes inside the layer
        Cell("port_mode", "nodes", "direct"),
        Cell("port_mode", "nodes", "direct", variant="pml"),
        # the sheet's fill and the Impedance's R, L, C beside the face
        Cell("port_mode", "fill", "direct", variant="conductor"),
        Cell("port_mode", "fill", "direct", variant="impedance"),
        Cell("port_mode", "lumped", "direct", variant="sheet"),
    ]
    return cells


CELLS = [
    *_cells(),
    *_piezoelectric_cells(),
    *_source_cells(),
    *_driven_periodic_cells(),
    *_driven_floquet_cells(),
    *_eigen_periodic_cells(),
    *_participation_cells(),
    *_field_cells(),
    *_eigen_wave_cells(),
    *_law_cells(),
    *_blend_cells(),
    *_face_cells(),
    *_electrostatic_law_cells(),
    *_thermal_cells(),
    *_heat_cells(),
    *_energy_cells(),
    *_port_mode_cells(),
]
#: The PETSc stack's cells, one per path its solves take: electrostatic,
#: driven on both sweeps, the adjoint through a wave port operator's
#: Woodbury update, and the eigenmode seed on SLEPc.
PETSC_CELLS = [
    Cell("electrostatic", "eps_r", "petsc"),
    Cell("driven", "eps_r", "petsc", "uniform"),
    Cell("driven", "eps_r", "petsc", "adaptive"),
    Cell("driven_wave", "eps_r", "petsc", "uniform", "lossy"),
    # a design at the wave port faces: the port pass reads a PETSc adjoint
    Cell("driven_wave", "eps_r", "petsc", "uniform", "face_lossy"),
    Cell("eigenmode", "eps_r", "petsc"),
    # plain periodicity keeps every backend (Bloch on PETSc raises)
    Cell("driven_periodic_plain", "eps_r", "petsc", "uniform"),
    # the mode-field adjoint after SLEPc seeds (its linear solver stays
    # direct: a PETSc one raises for a mode-field cotangent)
    Cell("eigenmode_participation", "eps_r", "direct", variant="slepc"),
    # a permittivity law's parameters: the law pass after a PETSc
    # solve, and the eigenfrequency after SLEPc seeds
    Cell("driven", "law", "petsc", "uniform", "poles"),
    Cell("eigenmode", "law", "petsc", variant="lorentz"),
]
PETSC_PARAMS = [
    pytest.param(
        cell,
        marks=pytest.mark.slepc
        if cell.problem.startswith("eigenmode")
        else pytest.mark.petsc,
    )
    for cell in PETSC_CELLS
]
#: Cells run in the reference tier only: each shared-surface cell solves four
#: wave ports through two port mode sweeps, and these three take 2-4 min on
#: MN5 (8 cores). The default tiers keep the shared-surface permittivity on both
#: sweeps and every input on the uniform sweep.
REFERENCE_CELLS = {
    Cell("driven_wave", "eps_r", "bddc_gmres", "uniform", "shared"),
    Cell("driven_wave", "conductivity", "direct", "adaptive", "shared"),
    Cell("driven_wave", "nodes", "direct", "adaptive", "shared"),
    Cell("driven_wave", "conductivity", "direct", "adaptive", "shared_lossy"),
    Cell("driven_wave", "nodes", "direct", "adaptive", "shared_lossy"),
}
CELL_PARAMS = [
    pytest.param(cell, marks=pytest.mark.reference, id=str(cell))
    if cell in REFERENCE_CELLS
    else cell
    for cell in CELLS
]


@dataclasses.dataclass
class Case:
    """A cell's objective, its input and how to perturb it.

    ``step``: fourth-order difference step along a unit direction (see
    tests/finite_differences.py); ``taylor``: the largest of the Taylor
    test's three steps along one unit direction, each ``taylor_ratio``
    times the next, from ``taylor_at`` (``p`` when None). At production
    tolerance the objective carries the solver's error (about 1e-7 relative
    for GMRES at 1e-6): the remainders must stay far above it.
    """

    objective: Callable
    p: jax.Array
    rng: np.random.Generator
    step: float
    taylor: float
    reference: Callable | None = None
    taylor_at: jax.Array | None = None
    taylor_ratio: float = 10**0.5
    mask: Callable = lambda directions: directions
    imaginary: bool = False  # the third direction along i (a complex input)

    def directions(self, n: int) -> list[np.ndarray]:
        out = self.mask(unit_directions(self.rng, self.p.shape, n))
        if self.imaginary and n >= 3:
            out[2] = 1j * out[2]
        return out


class Designs:
    """The cells' designs, each built on first use (module-scoped)."""

    def __init__(self, factory) -> None:
        self._factory = factory

    def _dir(self, name):
        return self._factory.mktemp(name)

    @functools.cached_property
    def stack(self):
        return stack(self._dir("stack"))

    @functools.cached_property
    def capacitor(self):
        return capacitor(self._dir("capacitor"))

    @functools.cached_property
    def transducer(self):
        mesh = pz.mesh(self._dir("transducer") / "idt.msh", "lines")
        return mesh, pz.model(mesh, "lines")

    @functools.cached_property
    def line(self):
        return layered_line(self._dir("line"))

    @functools.cached_property
    def axes_line(self):
        return layered_line(self._dir("axes_line"), absorbing_order=1)

    @functools.cached_property
    def shape_line(self):
        return shape_line(self._dir("shape_line"))

    @functools.cached_property
    def wave_mesh(self):
        return wave_mesh(self._dir("wave"))

    @functools.cached_property
    def hybrid_mesh(self):
        # the coarsest grid that keeps the quadrant (96 tetrahedra): its two
        # modes and Gram coupling are the finer grids' to 3 digits
        return hybrid_mesh(self._dir("hybrid"), cells=(4, 2, 2))

    @functools.cached_property
    def sheet_mesh(self):
        return sheet_mesh(self._dir("sheet"))

    @functools.cached_property
    def sheet_cavity(self):
        return sheet_cavity(self._dir("sheet_cavity"))

    @functools.cached_property
    def resonator(self):
        return resonator_mesh(self._dir("resonator"))

    @functools.cached_property
    def layered_resonator(self):
        return layered_resonator(self._dir("layered_resonator"))

    @functools.cached_property
    def mirrored_cavity(self):
        return mirrored_cavity(self._dir("mirrored"))

    @functools.cached_property
    def corner_cavity(self):
        return mirrored_cavity(self._dir("corner"), corner=True)

    @functools.cached_property
    def junction(self):
        return junction_mesh(self._dir("junction"))

    @functools.cached_property
    def guide(self):
        return absorbing_guide(self._dir("guide"))

    @functools.cached_property
    def periodic(self):
        return pc.g_mesh(self._dir("periodic"))

    @functools.cached_property
    def floquet(self):
        return pc.f2_mesh(self._dir("floquet"))

    @functools.cached_property
    def eigen_cell(self):
        return eigen_cell_mesh(self._dir("eigen_cell"))

    @functools.cached_property
    def wave_a(self):
        return wave_guide(self._dir("wave_a"), WAVE_A_SPLIT, lumped=6.0, moving=10.0)

    @functools.cached_property
    def wave_a4(self):
        return wave_guide(self._dir("wave_a4"), WAVE_A4)

    @functools.cached_property
    def wave_c(self):
        return wave_guide(self._dir("wave_c"), WAVE_C, height=WAVE_C_HEIGHT)

    @functools.cached_property
    def wave_l(self):
        return wave_guide(self._dir("wave_l"), WAVE_L)

    @functools.cached_property
    def dispersive(self):
        return dispersive_mesh(self._dir("dispersive"))

    @functools.cached_property
    def thermal_slab(self):
        # a heater on a block of cladding, sunk below, cooled above (um)
        rects = [
            (0.0, 0.0, 3.0, 3.0, "clad", None),
            (1.0, 3.0, 2.0, 4.0, "heater", None),
        ]
        edges = [
            (lambda p: abs(p[1]) < 1e-9, "bottom"),
            (lambda p: abs(p[1] - 4.0) < 1e-9, "top"),
        ]
        return tc.section(self._dir("thermal_slab") / "slab.msh", rects, 0.6, edges)

    @functools.cached_property
    def dc_bar(self):
        # a conducting bar between two terminals, oxide above it (um)
        rects = [(0.0, 0.0, 3.0, 1.0, "bar", None), (0.0, 1.0, 3.0, 2.0, "ox", None)]
        edges = [
            (lambda p: abs(p[0]) < 1e-9 and p[1] < 1.0, "left"),
            (lambda p: abs(p[0] - 3.0) < 1e-9 and p[1] < 1.0, "right"),
        ]
        return tc.section(self._dir("dc_bar") / "bar.msh", rects, 0.4, edges)

    @functools.cached_property
    def chain_slab(self):
        # a TiN-like heater 1.5 um above a silicon core in oxide, sunk below
        # and cooled above (um); the heater's terminals on the end faces
        rects = [
            (-1.75, -1.0, 1.75, 1.8, "clad", None),
            (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
            (-0.5, 1.5, 0.5, 1.65, "heater", 0.1),
        ]
        edges = [
            (lambda p: abs(p[1] + 1.0) < 1e-9, "bottom"),
            (lambda p: abs(p[1] - 1.8) < 1e-9, "top"),
        ]
        return tc.section(self._dir("chain_slab") / "slab.msh", rects, 0.3, edges)

    @functools.cached_property
    def planewave_box(self):
        return planewave_box(self._dir("planewave"))

    @functools.cached_property
    def dipole_line(self):
        return dipole_line(self._dir("dipole"))

    @functools.cached_property
    def face_dipole_line(self):
        return face_dipole_line(self._dir("face_dipole"))

    @functools.cached_property
    def law_shape_line(self):
        # the shape line's magnetic cover as a Lorentz medium (mu_r kept)
        line = shape_line(self._dir("law_shape_line"))
        return line.with_material("cover", gfem.Material(eps_r=COVER_LAW, mu_r=1.3))


def _cube(c):
    if ((c > 0.25) & (c < 0.75)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


def _cube_sheet(c):
    """The cube scatterer's +x face: an interior sheet."""
    inside = 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75
    return "sheet" if abs(c[0] - 0.75) < 1e-9 and inside else ""


def _sheet(mesh, surface, fill=None):
    """A conducting sheet on ``surface``, of per-face ``fill``."""
    n = len(mesh.elements(surface))
    fill = np.linspace(0.2, 0.9, n) if fill is None else fill
    return gfem.Conductor(surface, 100.0, 1e-6, fill=gfem.SheetField(fill=fill))


def planewave_box(directory):
    """A cube scatterer with a sheet in an air box in a layer, no port (um)."""
    path = pml_cases.box(
        directory / "box.msh",
        pml_cases.SIDES,
        thickness=0.25,
        cells=4,
        volume=_cube,
        surface=_cube_sheet,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    air = gfem.Material()
    return gfem.Model(
        mesh,
        materials={"air": air, "pml": air, "scat": air},
        boundaries=[_sheet(mesh, "sheet")],
        pml=gfem.PML("pml"),
        background=air,
    )


def dipole_line(directory):
    """The layer's TEM line with a design slab and a wave port (um)."""
    mesh = gfem.Mesh.from_gmsh(pml_cases.slab_line(directory / "line.msh"))
    air = gfem.Material()
    return gfem.Model(
        mesh,
        materials={"air": air, "pml": air, "slab": air},
        boundaries=[gfem.PEC("pec"), _sheet(mesh, "front")],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )


def face_dipole_line(directory):
    """`dipole_line` with a region "lead" of its own on the port's face (um)."""
    mesh = gfem.Mesh.from_gmsh(
        pml_cases.slab_line(directory / "line.msh", lead=FACE_LEAD)
    )
    air = gfem.Material()
    return gfem.Model(
        mesh,
        materials={"lead": air, "air": air, "pml": air, "slab": air},
        boundaries=[gfem.PEC("pec"), _sheet(mesh, "front")],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )


#: The thickness (um) of the "lead" of `face_dipole_line`, below the dipole.
FACE_LEAD = 0.3


@pytest.fixture(scope="module")
def designs(tmp_path_factory):
    return Designs(tmp_path_factory)


#: The mode-field cells' keys: their eigenpairs to 1e-13 (the forward polishes
#: the modes anyway; the differences stay clean).
MODE_FIELD_KEYS = ("eigenmode_participation", "eigenmode_field")


def _options(cell: Cell, *, production: bool) -> gfem.SolverOptions:
    extra = {}
    if cell.problem == "eigenmode" and cell.solver == "petsc":
        extra["eigen_solver"] = "slepc"  # a PETSc cell runs the whole PETSc stack
    if cell.problem in MODE_FIELD_KEYS and cell.variant == "slepc":
        extra["eigen_solver"] = "slepc"  # SLEPc seeds, the direct adjoint
    if production:
        return gfem.SolverOptions(order=2, linear_solver=cell.solver, **extra)
    if cell.problem in MODE_FIELD_KEYS:
        extra["eigen_tolerance"] = 1e-13
    elif cell.problem.startswith("eigenmode"):
        extra["eigen_tolerance"] = 1e-12
    return tight(cell.solver, **extra)


# --- the cells' objectives ---------------------------------------------------------

#: Steps on unit directions: fourth-order truncation and round-off both well
#: below the 1e-6 gate (measured worst errors 1e-8 .. 1e-10).
STEPS = {
    "eps_r": 1e-2,
    "mu_r": 1e-2,
    "tan_delta": 1e-3,
    "conductivity": 5e-2,
    "nodes": 2e-6,
    # a law's parameters, normalized: p0 (1 + t)
    "law": 1e-3,
    # a blend's density, in [0.2, 0.8] (the linear blend's S is strongly
    # curved: 1e-2 leaves 6e-7 of truncation, 5e-3 4e-8)
    "blend": 5e-3,
    # thermal: k in [1, 2] W/(m K), rho in [2000, 2400] kg/m^3, c_p in
    # [600, 800] J/(kg K); the amount and the scales are O(1)
    "thermal_conductivity": 1e-2,
    "mass_density": 2.0,
    "specific_heat": 1.0,
    "heat": 1e-2,
    # a two-port drive's incident powers (W) and phase (rad), O(1)
    "drive": 1e-2,
}
#: Ranges a cell draws its own ``mu_r`` or ``tan_delta`` from (the other
#: cells' designs carry neither).
RANGES = {"mu_r": (1.0, 2.0), "tan_delta": (0.01, 0.2)}
#: A principal-value input's suffix: ``eps_r_axes`` holds principal values of eps_r.
AXES = "_axes"


def _property(name: str) -> str:
    return name.removesuffix(AXES)


#: Grids of the driven cells: uniform frequencies, adaptive grid and samples.
LINE_FREQUENCIES = (18e9, 27e9)
LINE_GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 21))
LINE_SAMPLES = (LINE_GRID[0], LINE_GRID[20], LINE_GRID[10], LINE_GRID[4], LINE_GRID[16])
GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))
WAVE_FREQUENCIES = (12e9, 24e9)
WAVE_SAMPLES = (GRID[0], GRID[10], GRID[5], GRID[2])
#: The hybrid line's grids: modes 1 and 2 propagate, mode 3 is cut off.
#: The loss tangent of the hybrid line's leads, by cell variant.
SHARED_LOSSY_TAN_DELTA = {"shared": 0.0, "shared_lossy": 0.05}
HYBRID_FREQUENCIES = (55e9, 65e9)
HYBRID_GRID = tuple(float(f) for f in np.linspace(52e9, 68e9, 9))
HYBRID_SAMPLES = (HYBRID_GRID[0], HYBRID_GRID[8], HYBRID_GRID[4], HYBRID_GRID[2])
#: Max relative gradient difference, adjoint vs reciprocity, per solver
#: (measured 6.2e-15 and 1.7e-13; direct reached 2.8e-14 on MKL in CI).
SHARED_SURFACE_RECIPROCITY = {"direct": 1e-13, "bddc_gmres": 5e-13}
#: Relative value and gradient differences, adaptive vs full, at the samples
#: (measured 5.6e-11 and below 1e-9).
SHARED_SURFACE_ADAPTIVE = (2e-10, 3e-9)
SHAPE_FREQUENCIES = (GRID[2], GRID[8])
SHAPE_SAMPLES = (GRID[0], GRID[10], GRID[5])
SHEET_FREQUENCIES = (6e9, 9e9)
SHEET_GRID = tuple(float(f) for f in np.linspace(5e9, 10e9, 6))
SHEET_SAMPLES = (SHEET_GRID[0], SHEET_GRID[5], SHEET_GRID[2])


def _sweep(cell: Cell, frequencies, grid, samples, **kwargs):
    """The cell's problem and the grid rows its objective reads."""
    if cell.sweep == "uniform":
        problem = gfem.Driven(frequency=frequencies, **kwargs)
        return problem, list(range(len(frequencies)))
    problem = gfem.Driven(frequency=grid, sweep="adaptive", samples=samples, **kwargs)
    return problem, [grid.index(f) for f in samples]


def _wave(d: Designs, cell: Cell):
    """A wave-port cell's mesh, model, problem and S objective at its rows.

    The ``"shared"`` variant excites p1_m1 of the hybrid line only: its
    cross-mode entries take adjoint loads through the shared surfaces' full
    modal corrections; ``"shared_lossy"`` puts a loss tangent on its leads.
    """
    if cell.variant in ("shared", "shared_lossy"):
        mesh = d.hybrid_mesh
        problem, rows = _sweep(
            cell, HYBRID_FREQUENCIES, HYBRID_GRID, HYBRID_SAMPLES, excite="p1_m1"
        )
        rows = jnp.asarray(rows)
        return (
            mesh,
            hybrid_line(mesh, tan_delta=SHARED_LOSSY_TAN_DELTA[cell.variant]),
            problem,
            lambda s: shared_surface_functional(s, rows),
        )
    mesh = d.wave_mesh
    problem, rows = _sweep(cell, WAVE_FREQUENCIES, GRID, WAVE_SAMPLES)
    rows = jnp.asarray(rows)
    return (
        mesh,
        wave_line(mesh, cell.variant),
        problem,
        lambda s: s_functional(s.values[rows]),
    )


#: Weights of the energy objective, (regions, terminals): all distinct.
ENERGY_WEIGHTS = np.array([[1.0, 0.3], [-0.7, 2.0]])


def _stack_functional(cell: Cell, result):
    """<W, C> of the stack, or <W, E> over its region energies (``energy``)."""
    weights = ENERGY_WEIGHTS  # not symmetric on purpose
    if cell.variant == "energy":
        value = jnp.stack(list(result.energy.values()))
    else:
        value = result.capacitance
    return jnp.sum(weights * value) / (EPS0 * 1e-6)  # O(1) values


def _electrostatic_eps(d: Designs, cell: Cell, options, seed: int) -> Case:
    """J(eps) = <W, C(eps)> on a two-terminal stack with a random P0 field.

    Per principal value (``eps_r_axes``): ``(n, 3)`` on the tilted axes.
    Variant ``energy``: ``<W, E(eps)>`` over the region energies.
    """
    model = d.stack
    axes = TILTED if cell.input.endswith(AXES) else None

    def objective(eps):
        fields = stack_fields(model, eps, axes)
        result = gfem.solve(fields, STACK_PROBLEM, options=options)
        return _stack_functional(cell, result)

    rng = np.random.default_rng(42 + seed)
    shape = (len(model.mesh.tetrahedra),) + ((3,) if axes else ())
    eps = jnp.asarray(rng.uniform(2.0, 9.0, shape))
    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=1e-1)


ELECTROSTATIC = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


def _energy_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """<W, E> over the stack's region energies in every node coordinate."""
    rng = np.random.default_rng(7 + seed)
    eps = jnp.asarray(rng.uniform(2.0, 9.0, len(d.stack.mesh.tetrahedra)))
    model = stack_fields(d.stack, eps)
    structure = jax.tree_util.tree_structure(model.mesh)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(structure, [nodes])
        result = gfem.solve(model.replace(mesh=mesh), STACK_PROBLEM, options=options)
        return _stack_functional(cell, result)

    # the stack is in um: its layers' elements are 1.25e-7 m thick
    return Case(objective, jnp.asarray(model.mesh.nodes), rng, step=1e-9, taylor=1e-8)


def _electrostatic_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """C (pF) of a capacitor whose dielectric interface moves (+ a potential).

    Variant ``energy``: see `_energy_nodes`.
    """
    if cell.variant == "energy":
        return _energy_nodes(d, cell, options, seed)
    model = d.capacitor
    morph = boundary_morph(model, "oxide__nitride", grows="oxide")
    weights = np.random.default_rng(4).uniform(0.5, 1.5, 200)

    def objective(t):
        result = gfem.solve(
            model.replace(mesh=morph(t)), ELECTROSTATIC, options=options
        )
        value = result.capacitance[0, 0] * 1e12
        if cell.variant == "potential":
            phi = result.potential["top"].coefficients
            value = value + jnp.sum(weights[: phi.shape[0]] * phi[:200])
        return value

    t = jnp.zeros(morph.n_params)
    return Case(
        objective,
        t,
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _driven_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the two-layer line in one per-element property.

    Per principal value (``_axes``): biaxial values on the tilted axes, on
    the line with first-order absorbing sides (each side face's admittance
    tensor moves with the values of the element behind it). A ``mu_r`` or
    ``tan_delta`` cell adds its property to the design of the others.
    """
    name = _property(cell.input)
    if cell.input.endswith(AXES):
        line, axes = d.axes_line, TILTED
        values, rng = line_axes_design(line, seed)
    else:
        line, axes = d.line, None
        values, rng = line_design(line, seed)
    if name in RANGES:
        drawn = rng.uniform(*RANGES[name], values["eps_r"].shape)
        values = {**values, name: jnp.asarray(drawn)}
    problem, rows = _sweep(cell, LINE_FREQUENCIES, LINE_GRID, LINE_SAMPLES)

    def objective(p):
        model = with_fields(line, {**values, name: p}, axes)
        s = gfem.solve(model, problem, options=options).s.values
        return s_functional(s[jnp.asarray(rows)])

    taylor = {"eps_r": 1e-1, "mu_r": 1e-1, "tan_delta": 3e-2, "conductivity": 1.0}
    p = values[name]
    return Case(
        objective,
        p,
        rng,
        step=STEPS[name],
        taylor=taylor[name],
        imaginary=axes is not None and jnp.iscomplexobj(p),
    )


def _driven_fill(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the glued line in the per-face fill of its sheet."""
    mesh = d.sheet_mesh
    problem, rows = _sweep(cell, SHEET_FREQUENCIES, SHEET_GRID, SHEET_SAMPLES)

    def objective_on(problem, rows):
        def objective(fill):
            model = sheet_line(mesh, SHEETS[cell.variant](gfem.SheetField(fill=fill)))
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    reference = None
    if cell.sweep == "adaptive":
        # the differences are taken on a uniform sweep of the samples
        reference = objective_on(gfem.Driven(frequency=SHEET_SAMPLES), [0, 1, 2])
    fill, rng = sheet_fill(mesh, seed)
    return Case(
        objective_on(problem, rows),
        fill,
        rng,
        step=5e-3,
        taylor=5e-2,
        reference=reference,
    )


def _lumped_values(t):
    """The lumped values `LUMPED` scaled by ``1 + t``: an O(1) input."""
    return jnp.asarray(LUMPED) * (1 + t)


def _driven_lumped(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the glued line in its port's L and C and its sheet's R, L and C."""
    mesh = d.sheet_mesh
    problem, rows = _sweep(cell, SHEET_FREQUENCIES, SHEET_GRID, SHEET_SAMPLES)

    def objective_on(problem, rows):
        def objective(t):
            model = lumped_line(mesh, _lumped_values(t))
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    reference = None
    if cell.sweep == "adaptive":
        # the differences are taken on a uniform sweep of the samples
        reference = objective_on(gfem.Driven(frequency=SHEET_SAMPLES), [0, 1, 2])
    return Case(
        objective_on(problem, rows),
        jnp.zeros(len(LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
        reference=reference,
    )


def _driven_nodes(d: Designs, cell: Cell, options, seed: int, *, line=None) -> Case:
    """S of the line with every kind of shape term; its interface moves."""
    line = d.shape_line if line is None else line
    n, moved = interface_move(line)
    problem, rows = _sweep(cell, SHAPE_FREQUENCIES, GRID, SHAPE_SAMPLES)

    def objective(t):
        s = gfem.solve(line.replace(mesh=moved(t)), problem, options=options).s.values
        return s_functional(s[jnp.asarray(rows)])

    return Case(
        objective,
        jnp.zeros(n),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        # the line's curvature in t is small: its remainder reaches the
        # BDDC + GMRES error (1e-6) below 5e-5, a third-order bend above 4e-4
        taylor=2e-4,
        taylor_ratio=2.0,
        taylor_at=jnp.full(n, 1e-5),
    )


#: The plane wave of the source cells, and the dipoles beside the line's port
#: (in its air, off the faces of its grid), with the powers that make their
#: objectives O(1).
SOURCE_WAVE = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization="TM")
DIPOLE_POSITION = np.array([0.07, 0.13, 0.44]) * 1e-6
SOURCE_DIPOLES = {
    "driven_dipole": (
        gfem.ElectricDipole("d", position=DIPOLE_POSITION, moment=(0, 1e-28, 3e-29)),
        1e-11,
    ),
    "driven_magnetic_dipole": (
        gfem.MagneticDipole("d", position=DIPOLE_POSITION, moment=(1e-20, 0, 0)),
        1e-12,
    ),
}
#: The source cells' conductivities, S/m: a loss tangent of about 0.1 at the
#: 1 um wavelength.
SOURCE_SIGMA = (1e3, 3e3)


def _source_design(model, region, seed):
    n = len(model.mesh.elements(region))
    rng = np.random.default_rng(31 + seed)
    sheet = model.boundaries[-1].surface
    values = {
        "eps_r": jnp.asarray(rng.uniform(2.0, 3.0, n)),
        "conductivity": jnp.asarray(rng.uniform(*SOURCE_SIGMA, n)),
        "fill": jnp.asarray(rng.uniform(0.2, 0.9, len(model.mesh.elements(sheet)))),
    }
    return values, rng


def _source_case(cell, objective, values, rng) -> Case:
    name = cell.input
    taylor = {"eps_r": 1e-1, "conductivity": 1e2, "fill": 5e-2}[name]
    # The conductivity step: the objectives vary on a scale of 1e4-1e5 S/m in
    # sigma (J / |grad J| is 2e5 S/m), so a random unit direction nearly
    # orthogonal to the gradient (|grad . d| / |grad| = 1.8e-3 on the plane
    # wave) moves J by only 9e-9 relative per S/m. The fourth-order
    # difference's round-off, 1.5 delta J / (h |grad . d|) for a relative
    # noise delta of J (up to 1e-14: threaded sums, sparse Cholesky's
    # unpivoted factors), is then 1.7e-6 / h: at h = 1 S/m it crossed the
    # 1e-6 gate. At h = 100 S/m it is 1.7e-8; the truncation, from the
    # second-order difference's 9e-7 there, is about (6 x 9e-7)^2 / 30 =
    # 1e-12. Every element stays above 800 S/m (sigma >= 1e3, |d_i| <= 1).
    step = {"eps_r": STEPS["eps_r"], "conductivity": 1e2, "fill": 5e-3}[name]
    return Case(objective, values[name], rng, step=step, taylor=taylor)


def _with_source_design(model, region, values, name, p):
    """``model`` with its region's material fields and its sheet's fill."""
    design = {**values, name: p}
    fill = design.pop("fill")
    sheet = model.boundaries[-1]
    model = model.replace(
        boundaries=[*model.boundaries[:-1], _sheet(model.mesh, sheet.surface, fill)]
    )
    return model.with_material(region, gfem.MaterialField(**design))


def _planewave_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """|E|^2 over a plane wave's kept field, in the scatterer's property or fill."""
    model = d.planewave_box
    values, rng = _source_design(model, "scat", seed)
    problem = gfem.Driven(pml_cases.F0, sources=SOURCE_WAVE, keep_fields=True)

    def objective(p):
        moved = _with_source_design(model, "scat", values, cell.input, p)
        result = gfem.solve(moved, problem, options=options)
        coefficients = result.e_field(pml_cases.F0).coefficients
        return jnp.sum(jnp.abs(coefficients) ** 2) / coefficients.shape[0]

    return _source_case(cell, objective, values, rng)


def _dipole_problem(cell):
    dipole, scale = SOURCE_DIPOLES[cell.problem]
    return gfem.Driven(pml_cases.F0, excite=(), sources=dipole), scale


def _dipole_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """A dipole's emitted power in the slab's property or its sheet's fill."""
    model = d.dipole_line
    values, rng = _source_design(model, "slab", seed)
    problem, scale = _dipole_problem(cell)

    def objective(p):
        moved = _with_source_design(model, "slab", values, cell.input, p)
        result = gfem.solve(moved, problem, options=options)
        return result.emitted_power["d"][0] / scale

    return _source_case(cell, objective, values, rng)


def _dipole_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """An electric dipole's power as the slab's face, and the dipole's element, move."""
    values, _ = _source_design(d.dipole_line, "slab", seed)
    model = _with_source_design(d.dipole_line, "slab", values, "fill", values["fill"])
    morph = boundary_morph(model, "front", grows="slab")
    problem, scale = _dipole_problem(cell)

    def objective(t):
        result = gfem.solve(model.replace(mesh=morph(t)), problem, options=options)
        return result.emitted_power["d"][0] / scale

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=1e-9,
        taylor=3e-8,
        taylor_at=jnp.full(morph.n_params, 1e-10),
    )


#: The reciprocal port wave's grid on the adaptive sweep, its frozen samples
#: and the sample its objective reads.
RECIPROCAL_GRID = tuple(float(f) for f in np.linspace(0.9, 1.1, 5) * pml_cases.F0)
RECIPROCAL_SAMPLES = (RECIPROCAL_GRID[0], RECIPROCAL_GRID[4], RECIPROCAL_GRID[2])
#: |W|^2 of the dipole's port wave is 3e-11 W on the eps_r design: the scale
#: that makes the objective O(1).
RECIPROCAL_SCALE = 1e-11


def _reciprocal_wave(cell: Cell, model, options):
    """|W_p1,d|^2 / scale, the dipole's port wave read by reciprocity."""
    dipole, _ = SOURCE_DIPOLES["driven_dipole"]
    if cell.sweep == "adaptive":
        problem = gfem.Driven(
            RECIPROCAL_GRID,
            excite=(),
            sources=dipole,
            source_fields=False,
            sweep="adaptive",
            samples=RECIPROCAL_SAMPLES,
        )
        row = 2
    else:
        problem = gfem.Driven(
            pml_cases.F0, excite=(), sources=dipole, source_fields=False
        )
        row = 0
    result = gfem.solve(model, problem, options=options)
    return jnp.abs(result.port_wave("p1", "d")[row]) ** 2 / RECIPROCAL_SCALE


def _reciprocal_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The dipole's port wave in the slab's property or its sheet's fill."""
    model = d.dipole_line
    values, rng = _source_design(model, "slab", seed)

    def objective(p):
        moved = _with_source_design(model, "slab", values, cell.input, p)
        return _reciprocal_wave(cell, moved, options)

    return _source_case(cell, objective, values, rng)


def _reciprocal_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The dipole's port wave as the slab's face, and the dipole's element, move."""
    values, _ = _source_design(d.dipole_line, "slab", seed)
    model = _with_source_design(d.dipole_line, "slab", values, "fill", values["fill"])
    morph = boundary_morph(model, "front", grows="slab")

    def objective(t):
        return _reciprocal_wave(cell, model.replace(mesh=morph(t)), options)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=1e-9,
        # the port wave's third-order term cancels its second near 1e-8 (the
        # remainders' slope reads 5.4 from 3e-8 to 1e-8): second order from
        # 1e-9 down (slopes 1.98 and 1.99 measured, remainders 7e-7 to 8e-9)
        taylor=1e-9,
        taylor_at=jnp.full(morph.n_params, 1e-10),
    )


def _wave_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S through wave ports in the core's per-element eps_r or conductivity.

    Per principal value (``_axes``): biaxial values on the tilted axes.
    """
    mesh, model, problem, functional = _wave(d, cell)
    values, rng = core_design(mesh, seed)
    name = _property(cell.input)
    axes = TILTED if cell.input.endswith(AXES) else None
    if axes:
        values = {k: v[:, None] * BIAXIAL for k, v in values.items()}

    def objective(p):
        field = gfem.MaterialField(**{**values, name: p}, axes=axes)
        m = model.with_material("core", field)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return functional(gfem.solve(m, problem, options=options).s)

    # (the lossy uniform BDDC cell's remainder bends above 3e-2)
    taylor = {"eps_r": 3e-2, "conductivity": 1.0}[name]
    p = values[name]
    return Case(
        objective,
        p,
        rng,
        step=STEPS[name],
        taylor=taylor,
        imaginary=axes is not None and jnp.iscomplexobj(p),
    )


def _wave_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S through wave ports; the top plate moves, held on the port faces."""
    _, model, problem, functional = _wave(d, cell)
    morph = boundary_morph(model, "top")

    def objective(t):
        moved = model.replace(mesh=morph(t))
        return functional(gfem.solve(moved, problem, options=options).s)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _omega(model, options, target):
    result = gfem.solve(model, gfem.Eigenmode(target=target), options=options)
    return jnp.real(result.omega[0] * OMEGA_WEIGHT)


def _eigen_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """omega of the terminated line resonator in one per-element property.

    Per principal value (``_axes``): the ramps times distinct factors per
    axis (biaxial), on the tilted axes. A ``mu_r`` or ``tan_delta`` cell
    adds its property, a ramp over its range, to the design of the others.
    """
    mesh = d.resonator
    values = {
        "eps_r": ramp(mesh, "dielectric", 3.5, 4.5),
        "conductivity": ramp(mesh, "dielectric", 0.2, 0.6),
    }
    name = _property(cell.input)
    if name in RANGES:
        values[name] = ramp(mesh, "dielectric", *RANGES[name])
    axes = TILTED if cell.input.endswith(AXES) else None
    if axes:
        values = {k: v[:, None] * BIAXIAL for k, v in values.items()}

    def objective(p):
        fields = values | {name: p}
        eps, sigma = fields.pop("eps_r"), fields.pop("conductivity")
        model = terminated(mesh, eps, sigma, axes, **fields)
        return _omega(model, options, RESONATOR_TARGET)

    # a unit step of 1 S/m would make the 0.2 S/m end of the ramp negative
    # (and one of 0.1 the 0.01 end of the loss tangent's)
    taylor = {"eps_r": 1e-1, "mu_r": 1e-1, "tan_delta": 3e-2, "conductivity": 3e-1}
    taylor = taylor[name]
    return Case(
        objective,
        values[name],
        np.random.default_rng(seed),
        step=STEPS[name],
        taylor=taylor,
    )


def _eigen_fill(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """omega of the sheet cavity in the per-face fill of its resistive film."""
    mesh = d.sheet_cavity
    fill, rng = cavity_fill(mesh, seed=1 + seed)

    def objective(p):
        return _omega(cavity_model(mesh, p), options, CAVITY_TARGET)

    return Case(
        objective,
        fill,
        rng,
        step=5e-2,
        taylor=1e-1,
        mask=lambda directions: on_the_sheet(directions, fill),
    )


def _eigen_lumped(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """omega of the glued line in its port's L and C and its sheet's R, L and C."""
    mesh = d.sheet_mesh

    def objective(t):
        return _omega(lumped_line(mesh, _lumped_values(t)), options, LUMPED_TARGET)

    return Case(
        objective,
        jnp.zeros(len(LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
    )


def _eigen_nodes(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """omega of the two-layer resonator; its interface moves."""
    model = d.layered_resonator
    morph = boundary_morph(model, "sub__cover", grows="sub")

    def objective(t):
        return _omega(model.replace(mesh=morph(t)), options, RESONATOR_TARGET)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


# --- eigenmode with wave ports ------------------------------------------------------

#: Weights of the coupling Q's inverse in the wave cells' objectives.
WAVE_Q_WEIGHT = 3.0


def _wave_mode(model, problem, options, root) -> int:
    """The index of the mode nearest ``root`` (GHz) at the cell's design."""
    omega = np.asarray(gfem.solve(model, problem, options=options).omega)
    return int(np.argmin(np.abs(omega / (2 * np.pi * 1e9) - root)))


def _eigen_wave(d: Designs, cell: Cell, options, seed: int) -> Case:
    """An eigenmode cell with wave ports (`_eigen_wave_cells`)."""
    rng = np.random.default_rng(seed)
    wave = [gfem.WavePort("p1", "port")]
    if cell.problem == "eigenmode_wave_shared":
        mesh, layers, root = d.wave_c, WAVE_C, WAVE_C_ROOT
        wave = [gfem.WavePort("te10", "port"), gfem.WavePort("te20", "port", mode=2)]
        problem = gfem.Eigenmode(target=15.2e9, n_modes=3)
    elif cell.problem == "eigenmode_wave_lossy":
        mesh, layers, root = d.wave_l, WAVE_L, WAVE_L_ROOT
        problem = gfem.Eigenmode(target=11.67e9, n_modes=3)
    elif cell.variant == "a4":
        mesh, layers, root = d.wave_a4, WAVE_A4, WAVE_A4_ROOT
        problem = gfem.Eigenmode(target=9.97e9)
    else:
        return _eigen_wave_a(d, cell, options, seed)
    region = "window" if cell.variant == "a4" else "dielectric"
    eps = ramp(mesh, region, 3.8, 4.2) * (2.5 if cell.variant == "a4" else 1.0)

    def model(eps):
        design = {region: gfem.MaterialField(eps_r=eps)}
        return wave_resonator(mesh, layers, design, wave)

    j = _wave_mode(model(eps), problem, options, root)

    def objective(p):
        result = gfem.solve(model(p), problem, options=options)
        out = jnp.real(result.omega[j] * OMEGA_WEIGHT)
        for q in result.q_port.values():
            out = out + WAVE_Q_WEIGHT / q[j]
        return out

    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=1e-1)


def _eigen_wave_a(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Case A with an inductive lumped port, in its permittivity, nodes or L, C."""
    mesh = d.wave_a
    rng = np.random.default_rng(seed)
    problem = gfem.Eigenmode(target=11.6e9, n_modes=4)
    eps0 = ramp(mesh, "dielectric", 3.8, 4.2)

    def model(eps, t=0.0, mesh=mesh):
        inductance, capacitance = jnp.asarray(WAVE_LUMPED) * (1 + t)
        ports = [
            gfem.WavePort("p1", "port"),
            gfem.LumpedPort(
                "l1", "lumped", "+y", inductance=inductance, capacitance=capacitance
            ),
        ]
        design = {"dielectric": gfem.MaterialField(eps_r=eps)}
        return wave_resonator(mesh, WAVE_A_SPLIT, design, ports)

    j = _wave_mode(model(eps0), problem, options, WAVE_A_ROOT)

    def read(model):
        result = gfem.solve(model, problem, options=options)
        return (
            jnp.real(result.omega[j] * OMEGA_WEIGHT)
            + WAVE_Q_WEIGHT / result.q_port["p1"][j]
            + result.participation["l1"][j]
            + 0.7 * result.bulk_participation["dielectric"][j]
        )

    if cell.input == "eps_r":
        return Case(
            lambda p: read(model(p)), eps0, rng, step=STEPS["eps_r"], taylor=1e-1
        )
    if cell.input == "lumped":
        # a 10 % step reorders the modes nearest the target (the remainder
        # jumps to 30); from 3 % on the order holds
        return Case(
            lambda t: read(model(eps0, t)), jnp.zeros(2), rng, step=1e-3, taylor=3e-2
        )
    morph = boundary_morph(model(eps0), "moving", grows="spacer")
    return Case(
        lambda t: read(model(eps0, mesh=morph(t))),
        jnp.zeros(morph.n_params),
        rng,
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


# --- eigenmode ratios: participation, port Q, region and principal shares ----------

#: The split cavity's pair: a relative gap of 2.49e-4 (above the isolation
#: threshold 1e-4); differences take steps well below the gap.
NEAR_PAIR_DELTA, NEAR_PAIR_STEP = 1e-3, 5e-5
#: The R-only sheet's mode on the glued line (13.47 + 4.28i GHz, Q 1.6).
SHEET_LINE_TARGET = 13.5e9
#: Weights of the principal shares of a cell's principal objective.
PRINCIPAL_WEIGHTS = jnp.array([1.0, -0.6, 0.8])


def _ratios(result, variant: str):
    """A cell's objective: a weighted sum of mode 0's ratios.

    Every J variant reads the junction's participation and the sub layer's
    share; a resistive port adds ``1 / q_port`` (the loss term an objective
    uses, never ``q_port``); the principal variant the sub layer's shares
    along its axes.
    """
    out = result.participation["jj"][0] + 0.7 * result.bulk_participation["sub"][0]
    if "r" in result.q_port:
        out = out + 3.0 / result.q_port["r"][0]
    if variant == "principal":
        shares = result.principal_participation["sub"][0]
        out = out + jnp.dot(PRINCIPAL_WEIGHTS, shares)
    return out


def _kinetic(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The kinetic cavity's mode 0 in its sheet's fill, "lower"'s eps_r or nodes.

    The ``"sheet"`` variant reads the sheet participation; the
    ``"capacitive_sheet"`` one adds a capacitance to the sheet and reads
    "lower"'s share of the mode energy beside it (both move with the
    sheet's term of the energy). The node cells move the sheet
    with its glue plane, "lower" growing.
    """
    mesh = d.sheet_cavity
    fill, rng = cavity_fill(mesh, seed=1 + seed)
    capacitance = KINETIC_CAPACITANCE if cell.variant == "capacitive_sheet" else None
    problem = gfem.Eigenmode(target=KINETIC_TARGET)

    def read(model):
        result = gfem.solve(model, problem, options=options)
        out = result.sheet_participation["sheet"][0]
        if capacitance is not None:
            out = result.bulk_participation["lower"][0] + 0.5 * out
        return out

    def model(fill, eps=None, mesh=mesh):
        return kinetic_cavity(mesh, fill, eps, capacitance=capacitance)

    if cell.input == "fill":
        return Case(
            lambda p: read(model(p)),
            fill,
            rng,
            step=1e-2,
            taylor=1e-1,
            mask=lambda directions: on_the_sheet(directions, fill),
        )
    if cell.input == "eps_r":
        eps = ramp(mesh, "lower", 1.0, 2.0)
        return Case(
            lambda p: read(model(fill, p)),
            eps,
            rng,
            step=STEPS["eps_r"],
            taylor=1e-1,
        )
    morph = boundary_morph(model(fill), "sheet", grows="lower")
    return Case(
        lambda t: read(model(fill, mesh=morph(t))),
        jnp.zeros(morph.n_params),
        rng,
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _participation(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Mode 0's ratios in one input (`_participation_cells`)."""
    variant = cell.variant
    if variant in ("sheet", "capacitive_sheet"):
        return _kinetic(d, cell, options, seed)
    if variant == "absorbing2":
        mesh = d.guide
        core = ramp(mesh, "core", 9.0, 11.0)

        def objective(p):
            problem = gfem.Eigenmode(target=GUIDE_TARGET)
            result = gfem.solve(guide_model(mesh, p), problem, options=options)
            return result.bulk_participation["core"][0]

        return Case(
            objective, core, np.random.default_rng(seed), step=1e-2, taylor=1e-1
        )
    if variant == "resistive_sheet":
        mesh = d.sheet_mesh
        fill, rng = sheet_fill(mesh, seed=1 + seed)

        def objective(p):
            sheet = gfem.Impedance(
                "sheet", resistance=2e3, fill=gfem.SheetField(fill=p)
            )
            problem = gfem.Eigenmode(target=SHEET_LINE_TARGET)
            result = gfem.solve(sheet_line(mesh, sheet), problem, options=options)
            return 1.0 / result.q_port["p1"][0] + 0.5 / result.q_port["p2"][0]

        return Case(objective, fill, rng, step=5e-2, taylor=1e-1)
    if variant == "chi":
        return _chi(d, cell, options, seed)
    if variant == "near_pair":
        mesh = d.corner_cavity
        base = jnp.ones(len(mesh.tetrahedra))

        def objective(p):
            model = split_cavity(mesh, p, NEAR_PAIR_DELTA)
            problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=2)
            result = gfem.solve(model, problem, options=options)
            # mode 0, the nearer the target, is the lower of the pair
            return result.bulk_participation["corner"][0]

        return Case(
            objective,
            base,
            np.random.default_rng(seed),
            step=NEAR_PAIR_STEP,
            taylor=1e-4,
        )
    mesh = d.junction
    fixture = "inductive" if variant == "slepc" else variant
    values = junction_values(mesh, fixture)
    name = _property(cell.input)

    def objective(p):
        moved = {r: dict(v) for r, v in values.items()}
        moved["sub"][name] = p
        model = junction_line(mesh, fixture, moved)
        problem = gfem.Eigenmode(target=JUNCTION_TARGET)
        return _ratios(gfem.solve(model, problem, options=options), fixture)

    p = values["sub"][name]
    return Case(
        objective,
        p,
        np.random.default_rng(seed),
        step=STEPS[name],
        taylor={"eps_r": 1e-1, "conductivity": 3e-1}[name],
        imaginary=bool(jnp.iscomplexobj(p)),
    )


# --- eigenmode normalized field: probes and interface participation ----------------

#: The probed point of the field cells, m: inside the lower layer of fixture
#: J and of the resonator (both 10 x 2 x 1 mm) and of the glued line (8 x 2 x
#: 1 mm), off every element face.
FIELD_POINT = np.array([3.37, 0.93, 0.41]) * 1e-3
#: The probe objective's phase multiplier: ``Re(exp(0.7i) E_z)`` reads the
#: imaginary part of a lossy mode's field, so the phase rule's ``mu2`` is
#: nonzero; and its scale (``|E|`` is about 1e9 V/m at 1 J).
FIELD_PHASE, FIELD_SCALE = np.exp(0.7j), 1e-9
#: The interface layers: J's sub / cover interface (``"default"``), and an
#: oxide on the cavity's metal walls (``"MA"``): thickness (m) and eps_r.
FIELD_LAYER = (3e-9, 10.0)


def _probe(result) -> jax.Array:
    """Mode 0's probed z component, times the phase, scaled."""
    e = result.e_fields[0].at(FIELD_POINT)
    return FIELD_SCALE * jnp.real(FIELD_PHASE * e[2])


def _field(d: Designs, cell: Cell, options, seed: int) -> Case:
    """A measurement of mode 0's normalized field in one input (`_field_cells`)."""
    thickness, eps_layer = FIELD_LAYER
    if cell.input == "fill":
        mesh = d.sheet_cavity
        fill, rng = cavity_fill(mesh, seed=1 + seed)

        def objective(p):
            result = gfem.solve(
                cavity_model(mesh, p),
                gfem.Eigenmode(target=CAVITY_TARGET),
                options=options,
            )
            return (
                1e3
                * gfem.interface_participation(
                    result, "pec", thickness, eps_layer, kind="MA"
                ).participation[0]
            )

        return Case(
            objective,
            fill,
            rng,
            step=5e-2,
            taylor=1e-1,
            mask=lambda directions: on_the_sheet(directions, fill),
        )
    if cell.input == "conductivity":
        mesh = d.resonator
        eps = ramp(mesh, "dielectric", 3.5, 4.5)
        sigma = ramp(mesh, "dielectric", 0.2, 0.6)

        def objective(p):
            problem = gfem.Eigenmode(target=RESONATOR_TARGET)
            return _probe(
                gfem.solve(terminated(mesh, eps, p), problem, options=options)
            )

        return Case(
            objective,
            sigma,
            np.random.default_rng(seed),
            step=STEPS["conductivity"],
            taylor=3e-1,
        )
    mesh = d.junction
    fixture = "resistive" if cell.variant == "probe" else "inductive"
    values = junction_values(mesh, fixture)

    def objective(p):
        moved = {r: dict(v) for r, v in values.items()}
        moved["sub"]["eps_r"] = p
        model = junction_line(mesh, fixture, moved)
        result = gfem.solve(
            model, gfem.Eigenmode(target=JUNCTION_TARGET), options=options
        )
        if cell.variant == "probe":
            return _probe(result)
        return (
            1e3
            * gfem.interface_participation(
                result, "sub__cover", thickness, eps_layer
            ).participation[0]
        )

    return Case(
        objective,
        values["sub"]["eps_r"],
        np.random.default_rng(seed),
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


def _lumped_ratio(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Mode 0 of the lossy glued line in its five lumped values.

    All five values scaled by ``1 + t`` (`_lumped_values`). By variant: port
    p1's participation, its port Q, "lower"'s share of the mode energy, the
    sheet's participation, or (``eigenmode_field``) the probed field's
    component along the plates (y) times a phase. The ``"anharmonicity"``
    variant is the lossless line's instead (`_lumped_anharmonicity`).
    """
    mesh = d.sheet_mesh
    problem = gfem.Eigenmode(target=LUMPED_TARGET)
    if cell.variant == "anharmonicity":
        return _lumped_anharmonicity(mesh, problem, options, seed)
    read = {
        "participation": lambda r: r.participation["p1"][0],
        "q_port": lambda r: r.q_port["p1"][0],
        "bulk": lambda r: r.bulk_participation["lower"][0],
        "sheet_participation": lambda r: r.sheet_participation["sheet"][0],
        "probe": lambda r: (
            FIELD_SCALE * jnp.real(FIELD_PHASE * r.e_fields[0].at(FIELD_POINT)[1])
        ),
    }[cell.variant]

    def objective(t):
        model = lumped_line(mesh, _lumped_values(t))
        return read(gfem.solve(model, problem, options=options))

    return Case(
        objective,
        jnp.zeros(len(LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
    )


#: The dressed outputs' scale: rad/s to MHz.
TO_MHZ = 1 / (2 * np.pi * 1e6)


def _lumped_anharmonicity(mesh, problem, options, seed: int) -> Case:
    """Cell L6: the dressed anharmonicity of the lossless line's junction mode.

    Port p1 (L, C) is the linearized junction; its four values
    (`LOSSLESS_LUMPED`) are scaled by ``1 + t``, and the same traced port
    inductance is the ``inductance`` of `gfem.quantum.dressed_modes`, so the
    gradient runs through the solve (``omega``, ``participation``) and
    through the junction energy alike.
    """

    def objective(t):
        values = jnp.asarray(LOSSLESS_LUMPED) * (1 + t)
        result = gfem.solve(
            lossless_lumped_line(mesh, values), problem, options=options
        )
        dressed = gfem.quantum.dressed_modes(
            result.omega, result.participation["p1"], inductance=values[0]
        )
        return TO_MHZ * dressed.anharmonicity[0]

    return Case(
        objective,
        jnp.zeros(len(LOSSLESS_LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
    )


def _dispersive_shift(result) -> jax.Array:
    """The dressed dispersive shift between fixture J's modes 0 and 1, MHz."""
    dressed = gfem.quantum.dressed_modes(
        result.omega, result.participation["jj"], inductance=JUNCTION_CHI_L
    )
    return TO_MHZ * dressed.dispersive_shift[0, 1]


def _chi(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Cells 15 and 16: the dispersive shift of fixture J's two lowest modes.

    The ``"chi"`` variant's junction (`JUNCTION_CHI_L`): both modes (1.70
    and 7.98 GHz) carry it (participations 0.81 and 0.11), and
    `gfem.quantum.dressed_modes` diagonalizes its cosine on them, with the
    solve's inductance. In the lower layer's permittivity, or as the
    layers' interface moves (the ports held). The node step is 5x the other
    node cells': the dressed step's round-off (4e-14 of the shift) then sits
    further under the gate (measured errors 1e-8 to 1e-7, against up to
    5e-7 at 2e-6 m).
    """
    mesh = d.junction
    problem = gfem.Eigenmode(target=JUNCTION_TARGET, n_modes=2)
    rng = np.random.default_rng(seed)
    if cell.input == "nodes":
        model = junction_line(mesh, "chi")
        morph = boundary_morph(model, "sub__cover", grows="sub")

        def moved(t):
            result = gfem.solve(model.replace(mesh=morph(t)), problem, options=options)
            return _dispersive_shift(result)

        return Case(
            moved,
            jnp.zeros(morph.n_params),
            rng,
            step=5 * STEPS["nodes"],
            taylor=4e-5,
            taylor_at=jnp.full(morph.n_params, 1e-5),
        )
    values = junction_values(mesh)

    def objective(p):
        changed = {r: dict(v) for r, v in values.items()}
        changed["sub"]["eps_r"] = p
        model = junction_line(mesh, "chi", changed)
        result = gfem.solve(model, problem, options=options)
        return _dispersive_shift(result)

    return Case(
        objective,
        values["sub"]["eps_r"],
        rng,
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


def _participation_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Mode 0's ratios of fixture J as its interface "sub__cover" moves.

    The morph holds the ports' surfaces (both junction and resistive port
    sit on the line's ends), so the node gradient is exact along it.
    """
    variant = cell.variant
    if variant in ("sheet", "capacitive_sheet"):
        return _kinetic(d, cell, options, seed)
    if variant == "chi":
        return _chi(d, cell, options, seed)
    model = junction_line(d.junction, variant)
    morph = boundary_morph(model, "sub__cover", grows="sub")
    problem = gfem.Eigenmode(target=JUNCTION_TARGET)

    def objective(t):
        moved = model.replace(mesh=morph(t))
        return _ratios(gfem.solve(moved, problem, options=options), variant)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


# --- driven on a periodic model (the gradient cell G) ------------------------------

#: The adaptive grid and samples of the periodic cells.
PERIODIC_GRID = tuple(float(f) for f in np.linspace(16e9, 32e9, 9))
PERIODIC_SAMPLES = (PERIODIC_GRID[0], PERIODIC_GRID[8], PERIODIC_GRID[4])
#: Probe points of the field cell, m (inside the cell, off the seams).
PERIODIC_PROBES = np.array([[1.3, 0.7, 0.5], [2.6, 4.2, 1.6], [3.4, 2.5, 2.4]]) * 1e-3


def _periodic_sweep(cell: Cell):
    """The periodic cell's problem and the grid rows its objective reads.

    ``wave_vector``: `pc.G_K` for the Bloch problem, (0, 0, 0) for its
    variant ``"k0"`` (the general path at the plain field), none for the
    plain one.
    """
    wave_vector = None
    if cell.problem == "driven_periodic_bloch":
        wave_vector = (0.0, 0.0, 0.0) if cell.variant == "k0" else pc.G_K
    if cell.sweep == "uniform":
        problem = gfem.Driven(
            frequency=pc.G_FREQUENCIES,
            wave_vector=wave_vector,
            keep_fields=pc.G_FREQUENCIES[1] if cell.variant == "field" else False,
        )
        return problem, list(range(len(pc.G_FREQUENCIES)))
    problem = gfem.Driven(
        frequency=PERIODIC_GRID, sweep="adaptive", samples=PERIODIC_SAMPLES
    )
    return problem, [PERIODIC_GRID.index(f) for f in PERIODIC_SAMPLES]


def _periodic_value(cell: Cell, result, rows):
    """S at ``rows``, or for the ``"field"`` variant a functional of probed E."""
    if cell.variant != "field":
        return s_functional(result.s.values[jnp.asarray(rows)])
    weights = np.array([[1.0, 0.5, -0.8], [0.3, 1.2, 0.7], [-0.4, 0.9, 1.1]])
    value = 0.0
    for port in ("p1", "p2"):
        e = result.e_field(pc.G_FREQUENCIES[1], port).at(PERIODIC_PROBES)
        value = value + jnp.sum(weights * jnp.abs(e) ** 2) / 1e4
        value = value + jnp.real((0.6 - 0.4j) * e[1, 2]) / 1e2
    return value


def _periodic_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S (or probed E) of G in the per-element eps_r or conductivity of its floor."""
    mesh = d.periodic
    rng = np.random.default_rng(10 + seed)
    n = len(mesh.elements("core")) + len(mesh.elements("sub"))
    values = {
        "eps_r": jnp.asarray(rng.uniform(2.0, 6.0, n) * (1 - 0.02j)),
        "conductivity": jnp.asarray(rng.uniform(0.5, 2.0, n)),
    }
    problem, rows = _periodic_sweep(cell)
    name = cell.input

    def objective(p):
        model = pc.g(mesh, **{**values, name: p})
        return _periodic_value(cell, gfem.solve(model, problem, options=options), rows)

    # G's S bends little in eps_r: below 1e-1 the remainder reaches the
    # BDDC + GMRES error (1e-9 at 3e-2)
    taylor = {"eps_r": 3e-1, "conductivity": 1.0}[name]
    return Case(objective, values[name], rng, step=STEPS[name], taylor=taylor)


def _periodic_fill(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of G in the per-face fill of its sheet, which touches both pairs."""
    mesh = d.periodic
    rng = np.random.default_rng(20 + seed)
    fill = jnp.asarray(rng.uniform(0.2, 0.8, len(mesh.elements("sheet"))))
    problem, rows = _periodic_sweep(cell)

    def objective(p):
        model = pc.g(mesh, fill=p)
        return _periodic_value(cell, gfem.solve(model, problem, options=options), rows)

    # (the Bloch BDDC + GMRES remainder reaches the solver's error below 5e-2)
    return Case(objective, fill, rng, step=5e-3, taylor=1e-1)


def _periodic_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of G; its sidewall, across the x seam, moves (tied motion)."""
    model = pc.g(d.periodic)
    morph = boundary_morph(model, "wall", grows="core")
    problem, rows = _periodic_sweep(cell)

    def objective(t):
        result = gfem.solve(model.replace(mesh=morph(t)), problem, options=options)
        return _periodic_value(cell, result, rows)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


# --- driven with Floquet ports (coarse F2) ----------------------------------------


def _floquet_value(result) -> jax.Array:
    """|S|^2 of the -1 transmitted order, a cross-polarized entry, a zero row."""
    s = result.s
    efficiency = jnp.sum(jnp.abs(s["T-1,0_TE", "R0,0_TE"]) ** 2)
    cross = jnp.sum(jnp.real((0.7 - 0.3j) * s["R0,0_TM", "R0,0_TE"]))
    zero = jnp.sum(jnp.abs(s["R1,0_TE", "R0,0_TM"]) ** 2)
    return efficiency + cross + zero


def _floquet_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """F2's objective in the per-element eps_r of its ridge layer."""
    mesh = d.floquet
    rng = np.random.default_rng(30 + seed)
    eps = pc.floquet_gradient_design(mesh, rng)
    problem = pc.floquet_gradient_problem(
        pc.floquet_gradient_model(mesh, eps),
        bloch=cell.problem == "driven_floquet_bloch",
        fixed_angle=cell.variant == "fixed_angle",
    )

    def objective(p):
        result = gfem.solve(
            pc.floquet_gradient_model(mesh, p), problem, options=options
        )
        return _floquet_value(result)

    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=3e-1, imaginary=True)


def _floquet_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """F2's objective in the nodes of its ridge layer (interfaces included).

    The raw node gradient, along directions on the nodes of the ridge layer
    off the seams: the ridge's walls, top and bottom move, the Floquet
    surfaces and the pairs do not.
    """
    mesh = d.floquet
    model = pc.floquet_gradient_model(
        mesh, pc.floquet_gradient_design(mesh, np.random.default_rng(30))
    )
    problem = pc.floquet_gradient_problem(
        model,
        bloch=cell.problem == "driven_floquet_bloch",
        fixed_angle=cell.variant == "fixed_angle",
    )
    x = np.asarray(mesh.imported_nodes)
    lo, hi = pc.F2_D * pc.UM, (pc.F2_D + pc.F2_H) * pc.UM
    a = np.array(pc.F2_A) * pc.UM
    tol = 1e-3 * pc.UM
    layer = (x[:, 2] > lo - tol) & (x[:, 2] < hi + tol)
    inside = layer & (x[:, :2] > tol).all(axis=1) & (x[:, :2] < a - tol).all(axis=1)
    weight = inside[:, None].astype(float)

    def mask(directions):
        return [w / np.linalg.norm(w) for w in (dd * weight for dd in directions)]

    def objective(nodes):
        moved = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(mesh), [nodes]
        )
        result = gfem.solve(model.replace(mesh=moved), problem, options=options)
        return _floquet_value(result)

    # elements are 0.1 um: the remainder is second order below 3e-9 m
    return Case(
        objective,
        jnp.asarray(x),
        np.random.default_rng(seed),
        step=2e-9,
        taylor=1e-9,
        mask=mask,
    )


# --- the law cells' objectives --------------------------------------------------------

#: The smallest normalized law-gradient component a law cell admits, relative
#: to the gradient's norm: no parameter hides under the 1e-6 gate.
LAW_COMPONENT = 1e-4


def _law_case(objective, law, seed: int) -> Case:
    """A law cell's case: ``objective`` of the normalized parameters t."""
    return Case(
        objective,
        jnp.zeros(len(jax.tree.leaves(law))),
        np.random.default_rng(seed),
        step=STEPS["law"],
        taylor=1e-1,
    )


def _driven_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the strip line in its core's law parameters (`LAWS`).

    ``absorbing1`` / ``absorbing2``: the ``lorentz`` law behind absorbing y
    walls of that order (their law face pieces' terms).
    """
    if cell.sweep == "adaptive":
        return _adaptive_law(d, cell, options, seed)
    name, absorbing = cell.variant, None
    if name.startswith("absorbing"):
        name, absorbing = "lorentz", int(name.removeprefix("absorbing"))
    mesh, law = d.dispersive, LAWS[name]
    problem = gfem.Driven(frequency=LAW_FREQUENCIES[name])

    def objective(t):
        model = dispersive_line(mesh, scaled_law(law, t), absorbing=absorbing)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    return _law_case(objective, law, seed)


def _electrostatic_law(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """C (pF) of the capacitor whose nitride holds `STATIC_LAW`, in its parameters."""
    model = d.capacitor

    def objective(t):
        law = scaled_law(STATIC_LAW, t)
        m = model.with_material("nitride", gfem.Material(eps_r=law))
        return gfem.solve(m, ELECTROSTATIC, options=options).capacitance[0, 0] * 1e12

    return _law_case(objective, STATIC_LAW, seed)


def _static_leaves(law: gfem.PermittivityLaw) -> np.ndarray:
    """Which leaves of ``law`` the static permittivity reads (eps_inf, strengths)."""
    paths = jax.tree_util.tree_flatten_with_path(law)[0]
    return np.array(
        [jax.tree_util.keystr(p).endswith(("eps_inf", "delta_eps")) for p, _ in paths]
    )


def _driven_host_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the strip line in the host's per-element eps_r, beside the ``poles`` law."""
    if cell.sweep == "adaptive":
        return _adaptive_host_law(d, cell, options, seed)
    mesh = d.dispersive
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("dielectric"))
    eps = jnp.asarray(rng.uniform(2.0, 6.0, n) * (1 - 0.02j))
    problem = gfem.Driven(frequency=LAW_FREQUENCIES["poles"])

    def objective(p):
        host = gfem.MaterialField(eps_r=p)
        model = dispersive_line(mesh, LAWS["poles"], host=host)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=1e-1)


#: The adaptive law cells' laws: ``poles`` with its Lorentz pole moved from
#: 30 GHz into the band, and ``lorentz`` (behind the absorbing walls).
ADAPTIVE_LAWS = {
    "poles": dataclasses.replace(
        LAWS["poles"],
        terms=(
            LAWS["poles"].terms[0],
            dataclasses.replace(LAWS["poles"].terms[1], resonance_frequency=13.1e9),
        ),
    ),
    "absorbing1": LAWS["lorentz"],
}
#: Their grids and frozen samples: the endpoints, the two grid neighbours of
#: the pole, two more. Behind an absorbing face Re eps must stay > 0, so the
#: ``lorentz`` grid skips its metallic band (8 to 12.6 GHz) and its margin.
ADAPTIVE_LAW_GRIDS = {
    "poles": tuple(float(f) for f in np.linspace(4e9, 24e9, 21)),
    "absorbing1": tuple(
        float(f) for f in (*np.linspace(4e9, 7e9, 7), *np.linspace(15e9, 24e9, 10))
    ),
}
ADAPTIVE_LAW_SAMPLES = {
    "poles": (4e9, 24e9, 13e9, 14e9, 8e9, 19e9),
    "absorbing1": (4e9, 24e9, 7e9, 15e9, 5.5e9, 19e9),
}


def _adaptive_law_sweep(name: str) -> tuple[gfem.Driven, list[int], gfem.Driven]:
    """The adaptive problem, its sample rows, and the uniform sweep of the samples."""
    grid, samples = ADAPTIVE_LAW_GRIDS[name], ADAPTIVE_LAW_SAMPLES[name]
    problem = gfem.Driven(frequency=grid, sweep="adaptive", samples=samples)
    return problem, [grid.index(f) for f in samples], gfem.Driven(frequency=samples)


def _adaptive_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S at the samples of an adaptive sweep, in the core's law parameters.

    The differences are taken on a uniform sweep of the samples
    (`_driven_fill`'s pattern): at a sample the full solution lies in the
    frozen basis.
    """
    name = cell.variant
    law = ADAPTIVE_LAWS[name]
    absorbing = 1 if name == "absorbing1" else None
    problem, rows, uniform = _adaptive_law_sweep(name)

    def objective_on(problem, rows):
        def objective(t):
            model = dispersive_line(
                d.dispersive, scaled_law(law, t), absorbing=absorbing
            )
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    case = _law_case(objective_on(problem, rows), law, seed)
    case.reference = objective_on(uniform, list(range(len(rows))))
    if name == "absorbing1":
        # the 7 GHz sample sits 1 GHz under a pole 0.4 GHz wide: at 10 %
        # steps the third-order term still shows (slopes 1.66, 1.91)
        case.taylor = 1e-2
    return case


def _adaptive_host_law(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """The host's per-element eps_r beside the moved ``poles`` law, adaptive."""
    mesh = d.dispersive
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("dielectric"))
    eps = jnp.asarray(rng.uniform(2.0, 6.0, n) * (1 - 0.02j))
    problem, rows, uniform = _adaptive_law_sweep("poles")

    def objective_on(problem, rows):
        def objective(p):
            host = gfem.MaterialField(eps_r=p)
            model = dispersive_line(mesh, ADAPTIVE_LAWS["poles"], host=host)
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    return Case(
        objective_on(problem, rows),
        eps,
        rng,
        step=STEPS["eps_r"],
        taylor=1e-1,
        reference=objective_on(uniform, list(range(len(rows)))),
    )


def _driven_blend(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the strip line whose core blends a dielectric and a Drude metal.

    The input is the density, one per core element, in [0.2, 0.8]; the
    variant is the scheme (``index`` or ``linear``).
    """
    mesh = d.dispersive
    rng = np.random.default_rng(seed)
    rho = jnp.asarray(rng.uniform(0.2, 0.8, len(mesh.elements("core"))))
    problem = gfem.Driven(frequency=BLEND_FREQUENCIES)

    def objective(p):
        core = gfem.MaterialField.interpolate(
            p, BLEND_DIELECTRIC, BLEND_METAL, scheme=cell.variant
        )
        model = dispersive_line(mesh, BLEND_DIELECTRIC.eps_r)
        model = model.with_material("core", core)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    # the linear blend's S is strongly curved: its third-order term shows at a
    # Taylor step of 1e-1 (slope 2.11), not from 3e-2
    return Case(objective, rho, rng, step=STEPS["blend"], taylor=3e-2)


def _driven_nodes_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the shape line whose cover holds a law; its interface moves."""
    return _driven_nodes(d, cell, options, seed, line=d.law_shape_line)


def _wave_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S through wave ports in the law parameters of the core (``poles``)."""
    _, model, problem, functional = _wave(d, cell)
    law = LAWS["poles"]

    def objective(t):
        m = model.with_material("core", gfem.Material(eps_r=scaled_law(law, t)))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return functional(gfem.solve(m, problem, options=options).s)

    return _law_case(objective, law, seed)


#: Frequencies of the sheet beside wave ports: the glued line's.
FACE_SHEET_FREQUENCIES = SHEET_FREQUENCIES
#: The sheet's values with wave ports, in `LUMPED`: its R, L and C.
FACE_SHEET_VALUES = LUMPED[2:]


#: The face cells' grids where they are not `WAVE_FREQUENCIES`: the line with
#: absorbing side walls has its first mode's cutoff at 18 GHz.
FACE_FREQUENCIES = {"face_absorbing": (24e9, 30e9)}
#: The loss tangent of the face cells' shared surface.
FACE_SHARED_TAN_DELTA = 0.02


def _face_options(options):
    """``options`` with the port modes solved to the differences' accuracy.

    A reduced port row is off by up to the port mode tolerance, which a central
    difference sees and the (polished) gradient does not: the checks solve the
    modes tightly, the Taylor test keeps the production tolerance.
    """
    if options.tolerance <= 1e-12:
        return dataclasses.replace(options, port_mode_tolerance=1e-12)
    return options


def _face_problem(d: Designs, cell: Cell):
    """The face cell's mesh, model, problem, rows and voltage-path ports."""
    if cell.variant == "face_shared":
        mesh = d.hybrid_mesh
        problem = gfem.Driven(frequency=HYBRID_FREQUENCIES, excite="p1_m1")
        model = hybrid_line(mesh, tan_delta=FACE_SHARED_TAN_DELTA)
        return mesh, model, problem, [0, 1], ()
    mesh = d.wave_mesh
    model = face_line(mesh, cell.variant)
    frequencies = FACE_FREQUENCIES.get(cell.variant, WAVE_FREQUENCIES)
    problem = gfem.Driven(frequency=frequencies)
    paths = ("p1",) if cell.variant == "face_path" else ()
    return mesh, model, problem, [0, 1], paths


def _face_objective(rows, paths, *, shared: bool):
    """``result -> objective`` of a face cell: S, gamma and, with paths, Z_PV."""
    rows = jnp.asarray(rows)
    if shared:
        return lambda result: face_functional(
            result, rows, s_part=shared_surface_functional(result.s, rows)
        )
    return lambda result: face_functional(result, rows, paths)


def _face_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S, gamma and Z_PV in the lower layer's per-element property, at both faces.

    Per principal value (``_axes``): biaxial values on axes turned about the
    ports' normal. A ``mu_r``, ``tan_delta`` or ``conductivity`` cell adds its
    property to the design of the others; a ``blend`` cell is the density
    between two constant lossy ends.
    """
    options = _face_options(options)
    mesh, model, problem, rows, paths = _face_problem(d, cell)
    shared = cell.variant == "face_shared"
    functional = _face_objective(rows, paths, shared=shared)
    region = "block" if shared else "lo"
    n = len(mesh.elements(region))
    name = _property(cell.input)
    axes = turned(0, 30.0) if cell.input.endswith(AXES) else None
    rng = np.random.default_rng(seed)
    if name == "blend":
        ends = (
            gfem.Material(eps_r=3.0, tan_delta=0.05),
            gfem.Material(eps_r=5.0, tan_delta=0.05),
        )
        values = {"blend": jnp.asarray(rng.uniform(0.2, 0.8, n))}

        def design(p):
            return gfem.MaterialField.interpolate(p, *ends, scheme="linear")

        taylor = 3e-2
    else:
        lossy = shared or cell.variant != "face_lossless"
        eps = ramp(mesh, region, 3.5, 4.5)
        values = {"eps_r": eps}
        if lossy:
            values["tan_delta"] = jnp.full(n, FACE_SHARED_TAN_DELTA if shared else 0.05)
        if name == "conductivity":
            values[name] = ramp(mesh, region, 0.2, 0.6)
        elif name in RANGES:
            values[name] = ramp(mesh, region, *RANGES[name])
        if axes is not None:
            values = {k: v[:, None] * BIAXIAL for k, v in values.items()}

        def design(p):
            return gfem.MaterialField(**{**values, name: p}, axes=axes)

        taylor = {
            "eps_r": 3e-2,
            "conductivity": 1e-1,
            "mu_r": 1e-1,
            "tan_delta": 3e-2,
        }[name]

    def objective(p):
        m = model.with_material(region, design(p))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return functional(gfem.solve(m, problem, options=options))

    p0 = values[name]
    return Case(objective, p0, rng, step=STEPS[name], taylor=taylor)


def _face_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S, gamma and Z_PV as the port faces stretch in their plane.

    The top plate moves with the wave ports' faces sliding (their gap grows
    and shrinks); with a voltage path, a side wall does (the path is a fixed
    set of points between the plates, which must stay).
    """
    options = _face_options(options)
    mesh, model, problem, rows, paths = _face_problem(d, cell)
    functional = _face_objective(rows, paths, shared=cell.variant == "face_shared")
    ports = tuple(p.name for p in model.ports if isinstance(p, gfem.WavePort))
    surface = "side" if paths else "top"
    morph = boundary_morph(model, surface, sliding_ports=ports)
    del mesh

    def objective(t):
        moved = model.replace(mesh=morph(t))
        return functional(gfem.solve(moved, problem, options=options))

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _face_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S, gamma and Z_PV in the law parameters of the lower layer (`COVER_LAW`)."""
    options = _face_options(options)
    _, model, problem, rows, paths = _face_problem(d, cell)
    functional = _face_objective(rows, paths, shared=False)

    def objective(t):
        m = model.with_material("lo", gfem.Material(eps_r=scaled_law(COVER_LAW, t)))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return functional(gfem.solve(m, problem, options=options))

    return _law_case(objective, COVER_LAW, seed)


def _face_sheet_line(mesh, sheet):
    """The glued line with wave ports at both ends: the sheet meets both faces."""
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(eps_r=3.0), "upper": gfem.Material()},
        boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), sheet],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def _face_sheet(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S and gamma through wave ports in a sheet's fill, or an Impedance's R, L, C."""
    options = _face_options(options)
    mesh = d.sheet_mesh
    problem = gfem.Driven(frequency=FACE_SHEET_FREQUENCIES)
    rows = list(range(len(FACE_SHEET_FREQUENCIES)))

    def functional(result):
        return face_functional(result, rows)

    if cell.input == "lumped":

        def objective(t):
            r, l, c = jnp.asarray(FACE_SHEET_VALUES) * (1 + t)
            sheet = gfem.Impedance("sheet", resistance=r, inductance=l, capacitance=c)
            m = _face_sheet_line(mesh, sheet)
            return functional(gfem.solve(m, problem, options=options))

        return Case(
            objective,
            jnp.zeros(3),
            np.random.default_rng(seed),
            step=1e-3,
            taylor=1e-1,
        )
    kind = cell.variant.removeprefix("face_sheet_")
    fill, rng = sheet_fill(mesh, seed)

    def objective(p):
        sheet = SHEETS[kind](gfem.SheetField(fill=p))
        m = _face_sheet_line(mesh, sheet)
        return functional(gfem.solve(m, problem, options=options))

    return Case(objective, fill, rng, step=5e-3, taylor=1e-2)


def _face_dipole(d: Designs, cell: Cell, options, seed: int) -> Case:
    """A dipole's emitted power in the lead that holds the port's face."""
    options = _face_options(options)
    model = d.face_dipole_line
    n = len(model.mesh.elements("lead"))
    rng = np.random.default_rng(31 + seed)
    eps = jnp.asarray(rng.uniform(2.0, 3.0, n))
    problem, scale = _dipole_problem(Cell("driven_dipole", "eps_r", cell.solver))

    def objective(p):
        moved = model.with_material("lead", gfem.MaterialField(eps_r=p))
        result = gfem.solve(moved, problem, options=options)
        return result.emitted_power["d"][0] / scale

    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=1e-1)


#: The face cells' builders, by (problem, input, variant).
FACE_BUILDERS = {
    **{
        ("driven_wave", name, variant): _face_material
        for variant, names in FACE_INPUTS.items()
        for name in names
        if name not in ("law", "nodes")
    },
    **{
        ("driven_wave", "nodes", variant): _face_nodes
        for variant, names in FACE_INPUTS.items()
        if "nodes" in names
    },
    **{
        ("driven_wave", "law", variant): _face_law
        for variant in ("face_lossy", "face_absorbing")
    },
    ("driven_wave", "fill", "face_sheet_conductor"): _face_sheet,
    ("driven_wave", "fill", "face_sheet_impedance"): _face_sheet,
    ("driven_wave", "lumped", "face_sheet"): _face_sheet,
    ("driven_dipole", "eps_r", "face"): _face_dipole,
}


def _periodic_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of G whose floor "core" holds the ``poles`` law, in its parameters."""
    mesh = d.periodic
    law = LAWS["poles"]
    problem, rows = _periodic_sweep(cell)

    def objective(t):
        model = pc.g(mesh).with_material(
            "core", gfem.Material(eps_r=scaled_law(law, t))
        )
        return _periodic_value(cell, gfem.solve(model, problem, options=options), rows)

    return _law_case(objective, law, seed)


def _eigen_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """omega of the layered resonator in its cover's law parameters (`EIGEN_LAWS`).

    ``drude_lossless`` (the polynomial path) moves its eps_inf and plasma
    frequency: its damping, 0, has no normalized direction.
    """
    law = EIGEN_LAWS[cell.variant]
    resonator = d.layered_resonator
    if cell.variant == "drude_lossless":
        (term,) = law.terms

        def build(t):
            drude = dataclasses.replace(
                term, plasma_frequency=term.plasma_frequency * (1 + t[1])
            )
            return dataclasses.replace(
                law, eps_inf=law.eps_inf * (1 + t[0]), terms=[drude]
            )

        size = 2
    else:

        def build(t):
            return scaled_law(law, t)

        size = len(jax.tree.leaves(law))

    def objective(t):
        model = dispersive_resonator(resonator, build(t))
        return _omega(model, options, RESONATOR_TARGET)

    return Case(
        objective,
        jnp.zeros(size),
        np.random.default_rng(seed),
        step=STEPS["law"],
        taylor=1e-1,
    )


def _eigen_beside_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """omega of the layered resonator beside its cover's Lorentz law (E5, E6).

    ``eps_r``: the sub layer as a field (a ramp, its conductivity kept);
    ``nodes``: the interface "sub__cover" moves, through the law region.
    """
    resonator = dispersive_resonator(d.layered_resonator, EIGEN_LAWS["lorentz"])
    if cell.input == "nodes":
        morph = boundary_morph(resonator, "sub__cover", grows="sub")

        def moved(t):
            return _omega(resonator.replace(mesh=morph(t)), options, RESONATOR_TARGET)

        return Case(
            moved,
            jnp.zeros(morph.n_params),
            np.random.default_rng(seed),
            step=STEPS["nodes"],
            taylor=4e-5,
            taylor_at=jnp.full(morph.n_params, 1e-5),
        )
    eps = ramp(resonator.mesh, "sub", 3.5, 4.5)

    def objective(p):
        sub = gfem.MaterialField(eps_r=p, conductivity=0.3)
        return _omega(resonator.with_material("sub", sub), options, RESONATOR_TARGET)

    return Case(
        objective,
        eps,
        np.random.default_rng(seed),
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


def _law_ratios(result):
    """P1-P3's objective: mode 0's ratios beside fixture J's law cover.

    The two shares are weighted unequally: they add up to 1 (no capacitor),
    so an equal sum would have no derivative.
    """
    bulk = result.bulk_participation
    return (
        result.participation["jj"][0]
        + 0.7 * bulk["sub"][0]
        - 0.4 * bulk["cover"][0]
        + 3.0 / result.q_port["r"][0]
    )


def _junction_beside_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Fixture J with its cover's lossy Lorentz law (P1-P4, `junction_law_line`).

    Ratios (`_law_ratios`) in the sub layer's eps_r, the junction's L (scaled
    by ``1 + t``) or the interface's nodes; the probed field (`_probe`, the
    phase rule's multiplier nonzero) in the sub layer's eps_r.
    """
    mesh = d.junction
    problem = gfem.Eigenmode(target=JUNCTION_TARGET)
    read = _probe if cell.problem == "eigenmode_field" else _law_ratios

    def solve(model):
        return read(gfem.solve(model, problem, options=options))

    rng = np.random.default_rng(seed)
    if cell.input == "nodes":
        model = junction_law_line(mesh)
        morph = boundary_morph(model, "sub__cover", grows="sub")
        return Case(
            lambda t: solve(model.replace(mesh=morph(t))),
            jnp.zeros(morph.n_params),
            rng,
            step=STEPS["nodes"],
            taylor=4e-5,
            taylor_at=jnp.full(morph.n_params, 1e-5),
        )
    if cell.input == "lumped":
        return Case(
            lambda t: solve(
                junction_law_line(mesh, inductance=JUNCTION_L * (1 + t[0]))
            ),
            jnp.zeros(1),
            rng,
            step=1e-3,
            taylor=1e-1,
        )
    return Case(
        lambda p: solve(junction_law_line(mesh, p)),
        junction_values(mesh)["sub"]["eps_r"],
        rng,
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


# --- the periodic eigenmode cells' objectives ---------------------------------------

#: The design inputs of the periodic eigen cell, by cell input.
EIGEN_CELL_INPUTS = {"eps_r": "eps", "conductivity": "sigma", "fill": "fill"}


def _eigen_periodic(
    d: Designs, cell: Cell, options, seed: int, *, wave_vector=None
) -> Case:
    """omega of the periodic eigen cell (plain or Bloch) in one input.

    ``wave_vector`` solves the plain cell's mode with that wave vector.
    """
    mesh = d.eigen_cell
    design = eigen_cell_design(mesh)
    bloch = cell.problem == "eigenmode_periodic_bloch"
    problem = gfem.Eigenmode(
        target=EG_BLOCH_TARGET if bloch else EG_PLAIN_TARGET,
        wave_vector=EG_K if bloch else wave_vector,
    )

    def omega(**changes):
        values = design | changes
        if cell.variant == "linear":
            model = eigen_cell(values["mesh"], values["eps"], tan_delta=EG_TAN_DELTA)
        else:
            model = eigen_cell(
                values["mesh"], values["eps"], values["sigma"], values["fill"]
            )
        result = gfem.solve(model, problem, options=options)
        return jnp.real(result.omega[0] * OMEGA_WEIGHT)

    design["mesh"] = mesh
    rng = np.random.default_rng(seed)
    if cell.input == "nodes":
        # the mid-plane nodes move along z, each tie class (twins across
        # the pairs, the corners of both) as one: a periodic direction
        nodes = np.asarray(mesh.imported_nodes)
        mid = np.isclose(nodes[:, 2], 0.5 * EG_H * MM)
        labels = coordinate_classes(mesh, (EG_A * MM, EG_B * MM, None))
        classes = np.unique(labels[mid])
        tied = jnp.asarray((labels[:, None] == classes[None, :]) & mid[:, None], float)
        x0 = jnp.asarray(nodes)
        tree = jax.tree_util.tree_structure(mesh)

        def objective(t):
            moved = x0.at[:, 2].add(tied @ t)
            return omega(mesh=jax.tree_util.tree_unflatten(tree, [moved]))

        return Case(
            objective,
            jnp.zeros(len(classes)),
            rng,
            step=STEPS["nodes"],
            # smaller steps' remainders reach the eigen tolerance's error (1e-9)
            taylor=2e-4,
            taylor_ratio=2.0,
            taylor_at=jnp.full(len(classes), 1e-5),
        )
    name = EIGEN_CELL_INPUTS[cell.input]

    def objective(p):
        return omega(**{name: p})

    step = {"eps": STEPS["eps_r"], "sigma": STEPS["conductivity"], "fill": 5e-3}
    # a unit step in the 0.02-0.06 S/m ramp would make it negative
    taylor = {"eps": 1e-1, "sigma": 3e-2, "fill": 5e-2}
    return Case(
        objective,
        jnp.asarray(design[name]),
        rng,
        step=step[name],
        taylor=taylor[name],
    )


THERMAL_RANGES = {
    "thermal_conductivity": (1.0, 2.0),
    "mass_density": (2000.0, 2400.0),
    "specific_heat": (600.0, 800.0),
}
#: The transient cells' time grid: 9 steps over about one diffusion time
#: of the slab, the waveform switching down after 5.
TRANSIENT_DURATION, TRANSIENT_STEPS = 2e-6, 9
TRANSIENT_SWITCH = TRANSIENT_DURATION * 5 / TRANSIENT_STEPS


def _thermal_model(mesh, values, coefficient=1e4):
    """The slab with per-element thermal properties (missing: their midrange).

    ``coefficient``: the top's convection, W/(m^2 K).
    """
    fields = {}
    for region in ("clad", "heater"):
        tets = mesh.elements(region)
        fields[region] = gfem.MaterialField(
            **{
                name: values[name][tets]
                if name in values
                else jnp.full(len(tets), sum(THERMAL_RANGES[name]) / 2)
                for name in THERMAL_RANGES
            }
        )
    return gfem.Model(
        mesh,
        materials=fields,
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            # another ambient: the source-free temperature is not uniform,
            # so its own adjoint pair enters
            gfem.Convection("top", coefficient=coefficient, ambient=290.0),
        ],
    )


def _thermal_options(cell: Cell, options):
    scheme, layout = cell.variant.split("-")[:2]
    return dataclasses.replace(
        options,
        time_scheme=scheme,
        theta=0.5 if scheme == "theta" else 1.0,
        retain_memory=0 if layout == "checkpoints" else options.retain_memory,
    )


def _steady_thermal(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """<w, T - 300> plus the heater's mean rise, in the per-element k."""
    mesh = d.thermal_slab
    heat = gfem.HeatSource.uniform("heater", power_per_length=1.0)
    rng = np.random.default_rng(11 + seed)

    def objective(k):
        model = _thermal_model(mesh, {"thermal_conductivity": k})
        result = gfem.solve(model, gfem.SteadyThermal(heat), options=options)
        t = result.temperature.coefficients
        weights = np.linspace(0.5, 1.5, t.shape[0])
        return jnp.sum(weights * (t - 300.0)) / t.shape[0] + gfem.region_mean(
            result.rise, "heater"
        )

    k = jnp.asarray(
        rng.uniform(*THERMAL_RANGES["thermal_conductivity"], len(mesh.tetrahedra))
    )
    return Case(objective, k, rng, step=STEPS["thermal_conductivity"], taylor=1e-1)


def _transient_thermal(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Traces and a kept field of a switched heater, in one input.

    A material property per element, or (input ``heat``) the source's
    amount (``amount``) or the waveform's two scales (``waveform``); from a
    uniform start, so T0 enters through the start too.
    """
    mesh = d.thermal_slab
    options = _thermal_options(cell, options)
    rng = np.random.default_rng(13 + seed)
    weights = np.linspace(1.0, 2.0, TRANSIENT_STEPS + 1)
    what = cell.variant.split("-")[2] if cell.input == "heat" else None

    def objective(p):
        values, amount, scales = {}, 1.0, (1.0, 0.3)
        if what == "amount":
            amount = p[0]
        elif what == "waveform":
            scales = (p[0], p[1])
        else:
            values = {cell.input: p}
        problem = gfem.TransientThermal(
            gfem.HeatSource.uniform("heater", power_per_length=amount),
            duration=TRANSIENT_DURATION,
            steps=TRANSIENT_STEPS,
            waveform=[(0.0, scales[0]), (TRANSIENT_SWITCH, scales[1])],
            initial_temperature=295.0,
            traces={
                "heater": gfem.Trace.mean("heater"),
                "clad": gfem.Trace.point((1.5e-6, 1.0e-6, mesh.section_thickness / 2)),
            },
            keep_fields=[TRANSIENT_DURATION * 3 / TRANSIENT_STEPS],
        )
        result = gfem.solve(_thermal_model(mesh, values), problem, options=options)
        kept = result.temperature(TRANSIENT_DURATION * 3 / TRANSIENT_STEPS)
        value = (
            jnp.sum(weights * (result.traces["heater"] - 295.0))
            + jnp.sum(result.traces["clad"] - 295.0)
            + gfem.region_mean(kept, "clad")
        )
        # curved, so the Taylor test sees a second-order remainder in the
        # amount and the scales too (the solve is linear in them)
        return value + 1e-2 * value**2

    if what == "amount":
        p = jnp.asarray([1.0])
    elif what == "waveform":
        p = jnp.asarray([1.0, 0.3])
    else:
        p = jnp.asarray(rng.uniform(*THERMAL_RANGES[cell.input], len(mesh.tetrahedra)))
    step = STEPS[cell.input]
    return Case(objective, p, rng, step=step, taylor=10 * step)


def _dc_conduction(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The bar's resistance, or (``dissipation``) its weighted Joule heat at 1 mA.

    In the bar's per-element conductivity, scaled by 1e5 S/m.
    """
    mesh = d.dc_bar
    rng = np.random.default_rng(17 + seed)
    tets = mesh.elements("bar")
    oxide = gfem.Material(thermal_conductivity=1.4)
    weights = np.linspace(1.0, 2.0, len(mesh.tetrahedra))

    def objective(x):
        model = gfem.Model(
            mesh,
            materials={"bar": gfem.MaterialField(conductivity=1e5 * x), "ox": oxide},
            excluded={"dc_conduction": ("ox",)},
        )
        result = gfem.solve(
            model, gfem.DCConduction({"t": "left"}, "right"), options=options
        )
        if cell.variant == "resistance":
            return 1.0 / result.conductance[0, 0]
        heat = gfem.joule_heating(result, current=1e-3)
        density = heat._density_on(mesh)
        volumes = jnp.asarray(_tet_volumes(mesh))
        return 1e3 * jnp.sum(weights * density * volumes)

    x = jnp.asarray(rng.uniform(1.0, 2.0, len(tets)))
    return Case(objective, x, rng, step=STEPS["heat"], taylor=1e-1)


# --- electromagnetic heat and heat flow ---------------------------------------------

#: The heated models' thermal conductivity (W/(m K)) where no cell draws one.
HEAT_K = 1.4
#: Weights of the dissipated power per region and sheet (all distinct).
LOSS_WEIGHTS = {"sub": 1.0, "cover": 0.6, "sheet": 1.3, "core": 1.1}
#: The two-port drive of the ``drive`` cell: p1's and p2's incident powers (W)
#: and p2's phase (rad). The other heating cells drive p1 with 1 W.
DRIVE = (1.0, 0.5, 0.7)
#: The heating cells' frequency: the line's second, the sheet line's first.
HEAT_FREQUENCY = {"line": LINE_FREQUENCIES[1], "sheet": SHEET_FREQUENCIES[0]}


def _with_thermal(model: gfem.Model, sink: str, k=None) -> gfem.Model:
    """``model`` conducting heat to a fixed temperature on ``sink``.

    ``k``: per element in mesh order (every region a `gfem.MaterialField`),
    or None for `HEAT_K` everywhere.
    """
    materials = {}
    for region, material in model.materials.items():
        if k is None:
            materials[region] = dataclasses.replace(
                material,
                thermal_conductivity=HEAT_K
                if isinstance(material, gfem.Material)
                else jnp.full(len(model.mesh.elements(region)), HEAT_K),
            )
        else:
            values = k[model.mesh.elements(region)]
            materials[region] = dataclasses.replace(
                material, thermal_conductivity=values
            )
    sink_boundary = gfem.FixedTemperature(sink, temperature=300.0)
    return model.replace(
        materials=materials, boundaries=[*model.boundaries, sink_boundary]
    )


def _loss(model: gfem.Model, result, frequencies, measure=None) -> jax.Array:
    """The weighted dissipated power (or ``measure``) of p1's fields.

    ``measure(field, model=model)``: a per-region measurement, the dissipated
    power by default.
    """
    measure = gfem.dissipated_power if measure is None else measure
    total = 0.0
    for i, f in enumerate(frequencies):
        lost = measure(result.e_field(f, "p1"), model=model)
        total = total + (1 + 0.3 * i) * sum(
            LOSS_WEIGHTS.get(name, 0.9) * value for name, value in lost.items()
        )
    return total


def _heated_rise(model: gfem.Model, result, frequency, drive=None) -> jax.Array:
    """The mean rise (K / 100) of every region under the heat of ``drive``."""
    if drive is None:
        powers, phase = {"p1": 1.0}, None
    else:
        powers, phase = {"p1": drive[0], "p2": drive[1]}, {"p2": drive[2]}
    heat = gfem.electromagnetic_heating(
        result,
        model=model,
        frequency=frequency,
        incident_power=powers,
        incident_phase=phase,
    )
    steady = gfem.solve(model, gfem.SteadyThermal(heat), options=HEAT_THERMAL)
    rises = [gfem.region_mean(steady.rise, r) for r in model.materials]
    return sum((1 + 0.4 * i) * r for i, r in enumerate(rises)) / 100.0


#: The thermal solve of the heating cells: direct, whatever the driven solver.
HEAT_THERMAL = gfem.SolverOptions(order=2, linear_solver="direct")


#: The per-region measurements of the loss cells, by problem key. The energy
#: is in pJ-scale joules: ``ENERGY_SCALE`` makes its objective O(1).
ENERGY_SCALE = 1e12
LOSSES = {
    "dissipated_power": gfem.dissipated_power,
    "electric_energy": lambda field, model: {
        k: ENERGY_SCALE * v for k, v in gfem.electric_energy(field, model=model).items()
    },
}


def _heat_value(
    cell: Cell, model: gfem.Model, options, frequencies, kind, *, drive=None
):
    """The cell's heat objective of a model: its loss, or its heated rise."""
    if cell.problem in LOSSES:
        problem = gfem.Driven(frequency=frequencies, excite="p1", keep_fields=True)
        result = gfem.solve(model, problem, options=options)
        return _loss(model, result, frequencies, LOSSES[cell.problem])
    sink = "bottom" if kind == "line" else "plate1"
    excite = None if drive is not None else "p1"
    f = HEAT_FREQUENCY[kind]
    model = (
        model if cell.input == "thermal_conductivity" else _with_thermal(model, sink)
    )
    problem = gfem.Driven(frequency=f, excite=excite, keep_fields=True)
    result = gfem.solve(model, problem, options=options)
    return _heated_rise(model, result, f, drive)


def _heat_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The two-layer line's loss (or heated rise) in one per-element property.

    ``tan_delta_axes``: per principal value on the tilted axes (the line
    with first-order absorbing sides). ``eps_r``: complex, so the third
    direction is along i. ``drive`` and ``thermal_conductivity``: the design
    is fixed, the input is the drive or every element's k.
    """
    name = _property(cell.input)
    if cell.input.endswith(AXES):
        line, axes = d.axes_line, TILTED
        values, rng = line_axes_design(line, seed)
    else:
        line, axes = d.line, None
        values, rng = line_design(line, seed)
    if name in RANGES:
        drawn = rng.uniform(*RANGES[name], values["eps_r"].shape)
        values = {**values, name: jnp.asarray(drawn)}

    if cell.input == "drive":

        def objective(p):
            model = with_fields(line, values, axes)
            return _heat_value(cell, model, options, None, "line", drive=p)

        return Case(
            objective, jnp.asarray(DRIVE), rng, step=STEPS["drive"], taylor=1e-1
        )
    if cell.input == "thermal_conductivity":
        n = len(line.mesh.tetrahedra)

        def objective(k):
            model = _with_thermal(with_fields(line, values, axes), "bottom", k)
            return _heat_value(cell, model, options, None, "line")

        k = jnp.asarray(rng.uniform(*THERMAL_RANGES["thermal_conductivity"], n))
        return Case(objective, k, rng, step=STEPS["thermal_conductivity"], taylor=1e-1)

    def objective(p):
        model = with_fields(line, {**values, name: p}, axes)
        return _heat_value(cell, model, options, LINE_FREQUENCIES, "line")

    taylor = {"eps_r": 1e-1, "tan_delta": 3e-2, "conductivity": 1.0}
    p = values[name]
    return Case(
        objective,
        p,
        rng,
        step=STEPS[name],
        taylor=taylor[name],
        imaginary=bool(jnp.iscomplexobj(p)),
    )


def _heat_fill(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The glued line's sheet loss (or heated rise) in the sheet's per-face fill."""
    mesh = d.sheet_mesh

    def objective(fill):
        model = sheet_line(mesh, SHEETS[cell.variant](gfem.SheetField(fill=fill)))
        return _heat_value(cell, model, options, SHEET_FREQUENCIES, "sheet")

    fill, rng = sheet_fill(mesh, seed)
    return Case(objective, fill, rng, step=5e-3, taylor=5e-2)


def _heat_lumped(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The glued line's loss (or heated rise) in its lumped values.

    Its port's L and C, its sheet's R, L and C, as `_driven_lumped`.
    """
    mesh = d.sheet_mesh

    def objective(t):
        model = lumped_line(mesh, _lumped_values(t))
        return _heat_value(cell, model, options, SHEET_FREQUENCIES, "sheet")

    return Case(
        objective,
        jnp.zeros(len(LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
    )


def _heat_law(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """The strip line's core loss in its law's parameters (``poles``)."""
    mesh, law = d.dispersive, LAWS["poles"]
    frequencies = LAW_FREQUENCIES["poles"]
    problem = gfem.Driven(frequency=frequencies, excite="p1", keep_fields=True)

    def objective(t):
        model = dispersive_line(mesh, scaled_law(law, t))
        return _loss(model, gfem.solve(model, problem, options=options), frequencies)

    return _law_case(objective, law, seed)


def _energy_law(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """The strip line's energy per region in its core's law parameters (``poles``)."""
    mesh, law = d.dispersive, LAWS["poles"]
    frequencies = LAW_FREQUENCIES["poles"]
    problem = gfem.Driven(frequency=frequencies, excite="p1", keep_fields=True)

    def objective(t):
        model = dispersive_line(mesh, scaled_law(law, t))
        result = gfem.solve(model, problem, options=options)
        return _loss(model, result, frequencies, LOSSES["electric_energy"])

    return _law_case(objective, law, seed)


def _energy_nodes(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """The shape line's energy per region; its interface moves (as `_driven_nodes`).

    The field and the model move together: the energy is defined where their
    meshes are one.
    """
    line = d.shape_line
    n, moved = interface_move(line)
    problem = gfem.Driven(frequency=SHAPE_FREQUENCIES, excite="p1", keep_fields=True)

    def objective(t):
        model = line.replace(mesh=moved(t))
        result = gfem.solve(model, problem, options=options)
        return _loss(model, result, SHAPE_FREQUENCIES, LOSSES["electric_energy"])

    return Case(
        objective,
        jnp.zeros(n),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=2e-4,
        taylor_ratio=2.0,
        taylor_at=jnp.full(n, 1e-5),
    )


def _energy_eigen(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The energy per region of mode 0's field on fixture J, in the sub layer's eps_r.

    Plain (variant ``probe``), or beside the cover's lossy Lorentz law
    (``lorentz``): the Brillouin weight then moves with the mode's frequency.
    """
    mesh = d.junction
    values = junction_values(mesh)
    problem = gfem.Eigenmode(target=JUNCTION_TARGET)

    def objective(p):
        if cell.variant == "lorentz":
            model = junction_law_line(mesh, p)
        else:
            moved = {r: dict(v) for r, v in values.items()}
            moved["sub"]["eps_r"] = p
            model = junction_line(mesh, "resistive", moved)
        result = gfem.solve(model, problem, options=options)
        energy = gfem.electric_energy(result.e_fields[0], model=model)
        return ENERGY_SCALE * sum(
            LOSS_WEIGHTS.get(name, 0.9) * value for name, value in energy.items()
        )

    return Case(
        objective,
        values["sub"]["eps_r"],
        np.random.default_rng(seed),
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


def _energy_periodic(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The energy per region of G's p1 field in the per-element eps_r of its floor."""
    mesh = d.periodic
    rng = np.random.default_rng(10 + seed)
    n = len(mesh.elements("core")) + len(mesh.elements("sub"))
    eps = jnp.asarray(rng.uniform(2.0, 6.0, n) * (1 - 0.02j))
    problem, _ = _periodic_sweep(dataclasses.replace(cell, variant="field"))

    def objective(p):
        model = pc.g(mesh, eps_r=p)
        result = gfem.solve(model, problem, options=options)
        return _loss(model, result, [pc.G_FREQUENCIES[1]], LOSSES["electric_energy"])

    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=3e-1)


def _heat_blend(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The strip line's core loss in its blend's density (the scheme the variant)."""
    mesh = d.dispersive
    rng = np.random.default_rng(seed)
    rho = jnp.asarray(rng.uniform(0.2, 0.8, len(mesh.elements("core"))))
    problem = gfem.Driven(frequency=BLEND_FREQUENCIES, excite="p1", keep_fields=True)

    def objective(p):
        core = gfem.MaterialField.interpolate(
            p, BLEND_DIELECTRIC, BLEND_METAL, scheme=cell.variant
        )
        model = dispersive_line(mesh, BLEND_DIELECTRIC.eps_r).with_material(
            "core", core
        )
        return _loss(
            model, gfem.solve(model, problem, options=options), BLEND_FREQUENCIES
        )

    return Case(objective, rho, rng, step=STEPS["blend"], taylor=3e-2)


#: The heat-flow cells' top convection, W/(m^2 K): the top takes about half
#: the heat, so the flows move with k by tens of %. At the slab cells' 1e4 it
#: takes 1 %, and a flow's derivative sinks to 1e-4 of the flow, where the
#: fixed sink's reaction carries the direct solve's 3e-12 (a difference of
#: 1e-2 then reads it at 1e-5 relative).
FLOW_CONVECTION = 1e6


def _heat_flow_value(result, model, kind):
    """The heat through the slab's bottom and through a cut above the heater's foot."""
    cut = lambda c: c[:, 1] > 1.7e-6  # noqa: E731 - through the cladding
    if kind == "steady":
        return gfem.heat_flow_per_length(
            result, "bottom", model=model
        ) + 0.7 * gfem.heat_flow_per_length(result, cut, model=model)
    return {
        "sink": gfem.Trace.heat_out_per_length("bottom"),
        "cut": gfem.Trace.heat_out_per_length(cut),
    }


def _steady_heat_flow(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The heat through the slab's sink and a cut, in the per-element k or amount."""
    mesh = d.thermal_slab
    rng = np.random.default_rng(19 + seed)

    def objective(p):
        values, amount = {}, 1.0
        if cell.input == "heat":
            amount = p[0]
        else:
            values = {cell.input: p}
        heat = gfem.HeatSource.uniform("heater", power_per_length=amount)
        model = _thermal_model(mesh, values, FLOW_CONVECTION)
        result = gfem.solve(model, gfem.SteadyThermal(heat), options=options)
        value = _heat_flow_value(result, model, "steady")
        # curved in the amount too (the solve is linear in it)
        return value + 1e-2 * value**2

    if cell.input == "heat":
        return Case(objective, jnp.asarray([1.0]), rng, step=STEPS["heat"], taylor=1e-1)
    k = jnp.asarray(
        rng.uniform(*THERMAL_RANGES["thermal_conductivity"], len(mesh.tetrahedra))
    )
    return Case(objective, k, rng, step=STEPS["thermal_conductivity"], taylor=1e-1)


def _transient_heat_out(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The heat out of the slab's sink and a cut, as `_transient_thermal`'s inputs."""
    mesh = d.thermal_slab
    options = _thermal_options(cell, options)
    rng = np.random.default_rng(29 + seed)
    weights = np.linspace(1.0, 2.0, TRANSIENT_STEPS + 1)
    what = cell.variant.split("-")[2] if cell.input == "heat" else None

    def objective(p):
        values, amount, scales = {}, 1.0, (1.0, 0.3)
        if what == "amount":
            amount = p[0]
        elif what == "waveform":
            scales = (p[0], p[1])
        else:
            values = {cell.input: p}
        problem = gfem.TransientThermal(
            gfem.HeatSource.uniform("heater", power_per_length=amount),
            duration=TRANSIENT_DURATION,
            steps=TRANSIENT_STEPS,
            waveform=[(0.0, scales[0]), (TRANSIENT_SWITCH, scales[1])],
            initial_temperature=295.0,
            traces=_heat_flow_value(None, None, "transient"),
        )
        model = _thermal_model(mesh, values, FLOW_CONVECTION)
        result = gfem.solve(model, problem, options=options)
        # J/m over about a diffusion time of 1 W/m: O(1e-6)
        value = 1e6 * (
            jnp.sum(weights * result.traces["sink"]) + jnp.sum(result.traces["cut"])
        )
        return value + 1e-2 * value**2

    if what == "amount":
        p = jnp.asarray([1.0])
    elif what == "waveform":
        p = jnp.asarray([1.0, 0.3])
    else:
        p = jnp.asarray(rng.uniform(*THERMAL_RANGES[cell.input], len(mesh.tetrahedra)))
    step = STEPS[cell.input]
    return Case(objective, p, rng, step=step, taylor=10 * step)


def _piezoelectric(d: Designs, cell: Cell, options, seed: int) -> Case:
    """``<W, [Re Y', Im Y', C0']>`` of the transducer, in its permittivity or pitch.

    Per principal value of the crystal (``eps_r_axes``, ``(3,)``; the damping
    frame keeps its own values), or through the pitch map (``nodes``: the area
    between the frames stretched by ``1 + p``, the frames translated rigidly).
    """
    mesh, model = d.transducer
    problem = pz.problem()
    rng = np.random.default_rng(23 + seed)
    weights = rng.uniform(0.5, 1.5, 3)

    def reading(result):
        y = result.admittance_per_length[0, 0, 0]
        c0 = result.clamped_capacitance_per_length[0, 0] * 1e10  # O(1)
        return weights @ jnp.stack([y.real, y.imag, c0])

    if cell.input == "nodes":
        left = (-pz.PAIRS * pz.PERIOD / 2 - pz.MARGIN) * 1e-6
        nodes = jnp.asarray(mesh.imported_nodes)

        def objective(p):
            x = nodes[:, 0]
            inside = jnp.clip(x, left, -left)
            moved = nodes.at[:, 0].set(x - inside + (1.0 + p[0]) * inside)
            result = gfem.solve(
                model.replace(mesh=mesh.with_nodes(moved)), problem, options=options
            )
            return reading(result)

        return Case(objective, jnp.zeros(1), rng, step=3e-4, taylor=1e-2)

    def objective(eps):
        film = dataclasses.replace(
            gfem.MaterialField.uniform("ln", model.materials["ln"]), eps_r=eps
        )
        result = gfem.solve(model.with_material("ln", film), problem, options=options)
        return reading(result)

    return Case(
        objective, jnp.asarray(pz.EPS_LN), rng, step=STEPS["eps_r"], taylor=1e-1
    )


BUILDERS = {
    ("piezoelectric", "eps_r_axes"): _piezoelectric,
    ("piezoelectric", "nodes"): _piezoelectric,
    ("steady_thermal", "thermal_conductivity"): _steady_thermal,
    **{
        ("transient_thermal", name): _transient_thermal
        for name in (*THERMAL_RANGES, "heat")
    },
    ("dc_conduction", "conductivity"): _dc_conduction,
    **{
        ("dissipated_power", name): _heat_material
        for name in ("eps_r", "conductivity", "tan_delta_axes")
    },
    **{
        ("electric_energy", name): _heat_material
        for name in ("eps_r", "eps_r_axes", "conductivity", "tan_delta")
    },
    ("electric_energy", "fill"): _heat_fill,
    ("electric_energy", "law"): _energy_law,
    ("electric_energy", "nodes"): _energy_nodes,
    ("electric_energy_eigenmode", "eps_r"): _energy_eigen,
    ("electric_energy_periodic", "eps_r"): _energy_periodic,
    ("dissipated_power", "fill"): _heat_fill,
    ("dissipated_power", "lumped"): _heat_lumped,
    ("dissipated_power", "law"): _heat_law,
    ("dissipated_power", "blend"): _heat_blend,
    **{
        ("electromagnetic_heating", name): _heat_material
        for name in ("eps_r", "conductivity", "drive", "thermal_conductivity")
    },
    ("electromagnetic_heating", "fill"): _heat_fill,
    ("electromagnetic_heating", "lumped"): _heat_lumped,
    ("steady_heat_flow", "thermal_conductivity"): _steady_heat_flow,
    ("steady_heat_flow", "heat"): _steady_heat_flow,
    **{
        ("transient_heat_out", name): _transient_heat_out
        for name in (*THERMAL_RANGES, "heat")
    },
    ("electrostatic", "eps_r"): _electrostatic_eps,
    ("electrostatic", "eps_r_axes"): _electrostatic_eps,
    ("electrostatic", "nodes"): _electrostatic_nodes,
    ("driven", "eps_r"): _driven_material,
    ("driven", "conductivity"): _driven_material,
    ("driven", "eps_r_axes"): _driven_material,
    ("driven", "conductivity_axes"): _driven_material,
    **{
        ("driven", f"{name}{form}"): _driven_material
        for name in RANGES
        for form in ("", AXES)
    },
    ("driven", "fill"): _driven_fill,
    ("driven", "lumped"): _driven_lumped,
    ("driven", "nodes"): _driven_nodes,
    ("driven_wave", "eps_r"): _wave_material,
    ("driven_wave", "conductivity"): _wave_material,
    ("driven_wave", "eps_r_axes"): _wave_material,
    ("driven_wave", "nodes"): _wave_nodes,
    ("eigenmode", "eps_r"): _eigen_material,
    ("eigenmode", "conductivity"): _eigen_material,
    ("eigenmode", "eps_r_axes"): _eigen_material,
    ("eigenmode", "conductivity_axes"): _eigen_material,
    ("eigenmode", "mu_r"): _eigen_material,
    ("eigenmode", "tan_delta"): _eigen_material,
    ("eigenmode", "fill"): _eigen_fill,
    ("eigenmode", "lumped"): _eigen_lumped,
    ("eigenmode", "nodes"): _eigen_nodes,
    **{
        (problem, name): _eigen_wave
        for problem in (
            "eigenmode_wave",
            "eigenmode_wave_shared",
            "eigenmode_wave_lossy",
        )
        for name in ("eps_r", "nodes", "lumped")
    },
    **{
        (problem, name): builder
        for problem in ("driven_periodic_plain", "driven_periodic_bloch")
        for name, builder in (
            ("eps_r", _periodic_material),
            ("conductivity", _periodic_material),
            ("fill", _periodic_fill),
            ("nodes", _periodic_nodes),
        )
    },
    **{
        (problem, name): builder
        for problem in FLOQUET_PROBLEMS
        for name, builder in (("eps_r", _floquet_material), ("nodes", _floquet_nodes))
    },
    **{
        (problem, name): _eigen_periodic
        for problem in ("eigenmode_periodic_plain", "eigenmode_periodic_bloch")
        for name in ("eps_r", "conductivity", "fill", "nodes")
    },
    **{
        ("eigenmode_participation", name): _participation
        for name in ("eps_r", "conductivity", "eps_r_axes", "fill")
    },
    **{("eigenmode_field", name): _field for name in ("eps_r", "conductivity", "fill")},
    ("eigenmode_participation", "nodes"): _participation_nodes,
    ("eigenmode_participation", "lumped"): _lumped_ratio,
    ("eigenmode_field", "lumped"): _lumped_ratio,
    ("driven", "law"): _driven_law,
    ("driven_wave", "law"): _wave_law,
    ("driven_periodic_plain", "law"): _periodic_law,
    ("eigenmode", "law"): _eigen_law,
    ("driven", "blend"): _driven_blend,
    ("electrostatic", "law"): _electrostatic_law,
    **{
        ("driven_planewave", name): _planewave_material
        for name in ("eps_r", "conductivity", "fill")
    },
    **{
        (problem, name): _dipole_material
        for problem in SOURCE_DIPOLES
        for name in ("eps_r", "conductivity", "fill")
    },
    ("driven_dipole", "nodes"): _dipole_nodes,
    **{
        ("driven_dipole_reciprocal", name): _reciprocal_material
        for name in ("eps_r", "conductivity", "fill")
    },
    ("driven_dipole_reciprocal", "nodes"): _reciprocal_nodes,
}
# --- port mode builders (the cells are listed with `_port_mode_cells`) ---


def _port_mode_tight(options):
    """Options for the differences: exact modes (the Taylor test keeps the default)."""
    if options.tolerance > 1e-12:
        return options
    return dataclasses.replace(options, port_mode_tolerance=1e-12)


def _port_mode_lower(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Gamma, Z_PV and the field of p1 in the lower layer's per-element property."""
    if cell.variant == "shared":
        return _port_mode_shared(d, cell, options, seed)
    if cell.variant == "pml":
        return _port_mode_pml(d, cell, options, seed)
    options = _port_mode_tight(options)
    mesh = d.wave_mesh
    name = _property(cell.input)
    n = len(mesh.elements("lo"))
    ones = jnp.ones(n)
    base = {"eps_r": ramp(mesh, "lo", 3.5, 4.5), "tan_delta": 0.05 * ones}
    designs = {
        "eps_r": base["eps_r"],
        "conductivity": ramp(mesh, "lo", 0.5, 1.5),
        "tan_delta": ramp(mesh, "lo", *RANGES["tan_delta"]),
        "mu_r": ramp(mesh, "lo", 1.0, 2.0),
    }
    axes = turned(0, 30.0) if cell.input.endswith(AXES) else None
    problem = gfem.PortMode("p1", PORT_MODE_FREQUENCIES)
    if axes:
        # principal values biaxial in the face's plane, on axes turned about the
        # port normal x (a tilt off it would break the transverse anisotropy)
        p = (base["eps_r"] * (1 - 0.05j))[:, None] * BIAXIAL
        others: dict = {}
    else:
        p = designs[name]
        others = {k: v for k, v in base.items() if k != name}
        if name == "conductivity":
            others["eps_r"] = 4.0 * ones

    # a polarity reference replaces the voltage path: the two cannot combine
    polarity = PORT_MODE_POLARITY if cell.variant == "polarity" else None

    def objective(p):
        field = gfem.MaterialField(**{**others, name: p}, axes=axes)
        model = port_mode_line(mesh, field, path=polarity is None, polarity=polarity)
        result = gfem.solve(model, problem, options=options)
        return port_mode_functional(result, PORT_MODE_FREQUENCIES)

    return Case(
        objective,
        p,
        np.random.default_rng(seed),
        step=STEPS[name],
        taylor=PORT_MODE_TAYLOR[name],
        imaginary=axes is not None,
    )


def _port_mode_shared(
    d: Designs,
    cell: Cell,  # noqa: ARG001
    options,
    seed: int,
) -> Case:
    """Mode 2 of the hybrid line's shared port surface in the quadrant block's eps."""
    options = _port_mode_tight(options)
    mesh = d.hybrid_mesh
    n = len(mesh.elements("block"))
    ones = jnp.ones(n)
    problem = gfem.PortMode("p1_m2", HYBRID_FREQUENCIES)
    line = hybrid_line(mesh, tan_delta=0.02)

    def objective(p):
        field = gfem.MaterialField(eps_r=p, tan_delta=0.02 * ones)
        result = gfem.solve(
            line.with_material("block", field), problem, options=options
        )
        return port_mode_functional(result, HYBRID_FREQUENCIES)

    return Case(
        objective,
        ramp(mesh, "block", 3.5, 4.5),
        np.random.default_rng(seed),
        step=STEPS["eps_r"],
        taylor=PORT_MODE_TAYLOR["eps_r"],
    )


def _port_mode_pml(
    d: Designs,
    cell: Cell,  # noqa: ARG001
    options,
    seed: int,
) -> Case:
    """The strip guide cut by a perfectly matched layer, in its core's eps."""
    options = _port_mode_tight(options)
    path = pml_cases.strip_guide(
        d._dir("portmode_strip") / "strip.msh", 0.4, quarter=False, length=0.15
    )
    model = pml_cases.strip_model(path)
    n = len(model.mesh.elements("core"))
    problem = gfem.PortMode("p1", pml_cases.STRIP_F)

    def objective(p):
        field = gfem.MaterialField(eps_r=p)
        result = gfem.solve(
            model.with_material("core", field), problem, options=options
        )
        return port_mode_functional(
            result,
            [pml_cases.STRIP_F],
            PORT_MODE_STRIP_POINTS,
            PORT_MODE_STRIP_E0,
        )

    return Case(
        objective,
        jnp.asarray(np.linspace(3.8, 4.2, n)),
        np.random.default_rng(seed),
        step=STEPS["eps_r"],
        taylor=PORT_MODE_TAYLOR["eps_r"],
    )


def _port_mode_sheet(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Gamma and the field of p1 beside the glued line's sheet: its fill or R, L, C."""
    options = _port_mode_tight(options)
    mesh = d.sheet_mesh
    problem = gfem.PortMode("p1", SHEET_FREQUENCIES)
    waves = [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")]
    points = np.array([[0.0, 0.5e-3, 0.25e-3], [0.0, 1.4e-3, 0.7e-3]])

    def value(model):
        result = gfem.solve(model, problem, options=options)
        return port_mode_functional(result, SHEET_FREQUENCIES, points)

    if cell.input == "fill":
        fill, rng = sheet_fill(mesh, seed)

        def objective(fill):
            sheet = SHEETS[cell.variant](gfem.SheetField(fill=fill))
            return value(sheet_line(mesh, sheet).replace(ports=waves))

        return Case(objective, fill, rng, step=5e-3, taylor=5e-2)

    def objective(t):
        values = _lumped_values(t)
        # the port p2's L and C are read by no mode of p1: exact zeros
        line = lumped_line(mesh, values)
        sheet = line.boundaries[-1]
        tail = gfem.LumpedPort(
            "p2",
            "port2",
            "+y",
            resistance=1000.0,
            inductance=values[0],
            capacitance=values[1],
        )
        return value(
            line.replace(
                boundaries=[*line.boundaries[:-1], sheet], ports=[waves[0], tail]
            )
        )

    return Case(
        objective,
        jnp.zeros(len(LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
    )


def _port_mode_law(
    d: Designs,
    cell: Cell,  # noqa: ARG001
    options,
    seed: int,
) -> Case:
    """The lower layer's permittivity-law parameters, normalized (`COVER_LAW`)."""
    options = _port_mode_tight(options)
    mesh = d.wave_mesh
    problem = gfem.PortMode("p1", PORT_MODE_FREQUENCIES)

    def objective(t):
        model = port_mode_line(mesh, gfem.Material(eps_r=scaled_law(COVER_LAW, t)))
        result = gfem.solve(model, problem, options=options)
        return port_mode_functional(result, PORT_MODE_FREQUENCIES)

    return _law_case(objective, COVER_LAW, seed)


def _port_mode_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Gamma and Z_PV of p1 as the side walls move, with the faces sliding."""
    if cell.variant == "pml":
        return _port_mode_pml_nodes(d, cell, options, seed)
    options = _port_mode_tight(options)
    model = port_mode_line(d.wave_mesh, gfem.Material(eps_r=4.0, tan_delta=0.05))
    problem = gfem.PortMode("p1", PORT_MODE_FREQUENCIES)
    morph = boundary_morph(model, "side", sliding_ports=("p1", "p2"))

    def objective(t):
        result = gfem.solve(model.replace(mesh=morph(t)), problem, options=options)
        return port_mode_functional(result, PORT_MODE_FREQUENCIES, field=False)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _port_mode_pml_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:  # noqa: ARG001
    """The strip's face nodes inside the layer move in the face's plane."""
    options = _port_mode_tight(options)
    path = pml_cases.strip_guide(
        d._dir("portmode_strip_nodes") / "strip.msh", 0.4, quarter=False, length=0.15
    )
    model = pml_cases.strip_model(path)
    problem = gfem.PortMode("p1", pml_cases.STRIP_F)
    mesh = model.mesh
    section = port_section(
        model, model.ports[0], _pml.stretch(model, [pml_cases.STRIP_F])
    )
    assert section.layer is not None
    used = np.unique(mesh.triangles[section.faces])
    nodes0 = np.asarray(mesh.nodes)
    layer = used[np.unique(section.triangles[section.layer])]
    # deep in the layer: the forward refuses a tetrahedron moved past its inner face
    sides = section.stretch.layers.sides
    depth = np.max([s.sign * (nodes0[layer, s.axis] - s.face) for s in sides], axis=0)
    inside = layer[depth > 1e-7]
    assert len(inside)
    frame = section.frame[:2]
    treedef = jax.tree_util.tree_structure(mesh)

    def objective(t):
        # t in millimetres, in the face's two directions
        shift = (jnp.reshape(t, (-1, 2)) @ jnp.asarray(frame)) * 1e-3
        moved = jnp.asarray(nodes0).at[inside].add(shift)
        result = gfem.solve(
            model.replace(mesh=jax.tree_util.tree_unflatten(treedef, [moved])),
            problem,
            options=options,
        )
        return port_mode_functional(result, [pml_cases.STRIP_F], field=False)

    n = 2 * len(inside)
    return Case(
        objective,
        jnp.zeros(n),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=3e-5,
        taylor_at=jnp.full(n, 1e-6),
    )


PORT_MODE_BUILDERS = {
    ("port_mode", "nodes"): _port_mode_nodes,
    **{
        ("port_mode", name): _port_mode_lower
        for name in ("eps_r", "conductivity", "tan_delta", "mu_r", "eps_r_axes")
    },
    ("port_mode", "law"): _port_mode_law,
    ("port_mode", "fill"): _port_mode_sheet,
    ("port_mode", "lumped"): _port_mode_sheet,
}

BUILDERS.update(PORT_MODE_BUILDERS)


#: Builders of the existing inputs beside a law, by (problem, input, variant).
VARIANT_BUILDERS = {
    ("driven", "eps_r", "law"): _driven_host_law,
    ("driven", "nodes", "law"): _driven_nodes_law,
    ("eigenmode", "eps_r", "lorentz"): _eigen_beside_law,
    ("eigenmode", "nodes", "lorentz"): _eigen_beside_law,
    **{
        ("eigenmode_participation", name, "lorentz"): _junction_beside_law
        for name in ("eps_r", "lumped", "nodes")
    },
    ("eigenmode_field", "eps_r", "lorentz"): _junction_beside_law,
    ("electric_energy_eigenmode", "eps_r", "lorentz"): _energy_eigen,
}


VARIANT_BUILDERS |= FACE_BUILDERS


def _builder(cell: Cell) -> Callable:
    variant = VARIANT_BUILDERS.get((cell.problem, cell.input, cell.variant))
    return variant or BUILDERS[cell.problem, cell.input]


def _case(designs: Designs, cell: Cell, *, production: bool, seed: int = 0) -> Case:
    build = _builder(cell)
    return build(designs, cell, _options(cell, production=production), seed)


# --- the cells -----------------------------------------------------------------------


@pytest.mark.parametrize("cell", [*CELL_PARAMS, *PETSC_PARAMS], ids=str)
def test_gradient_matrix_matches_central_differences(designs, cell):
    case = _case(designs, cell, production=False)
    grad = jax.grad(case.objective)(case.p)
    if cell.input == "law":
        # every normalized law parameter moves the objective; on the
        # electrostatic problem, every one its static permittivity reads, and
        # the frequency parameters are exact zeros
        g = np.abs(np.asarray(grad))
        if cell.problem == "electrostatic":
            read = _static_leaves(STATIC_LAW)
            assert not g[~read].any(), f"{cell}: {g}"
            g = g[read]
        assert g.min() >= LAW_COMPONENT * np.linalg.norm(g), f"{cell}: {g}"
    assert_gradient_matches(
        case.objective,
        case.p,
        case.directions(3),
        step=case.step,
        reference=case.reference,
        grad=grad,
    )


@pytest.mark.parametrize("cell", [*CELL_PARAMS, *PETSC_PARAMS], ids=str)
def test_gradient_matrix_taylor_remainder_is_second_order(designs, cell):
    case = _case(designs, cell, production=True, seed=1)
    (d,) = case.directions(1)
    at = case.p if case.taylor_at is None else case.taylor_at
    steps = [case.taylor / case.taylor_ratio**k for k in range(3)]
    assert_taylor_second_order(case.objective, at, d, steps)


# --- consistency: the same gradient however it is asked for --------------------------

#: One cell per input kind and problem (direct solver), for the transformations.
FAMILIES = [
    Cell("electrostatic", "eps_r", "direct"),
    Cell("electrostatic", "nodes", "direct", variant="potential"),
    Cell("driven", "eps_r", "direct", "uniform"),
    Cell("driven", "eps_r", "direct", "adaptive"),
    Cell("driven", "eps_r_axes", "direct", "uniform"),
    Cell("driven", "fill", "direct", "uniform", "conductor"),
    Cell("driven", "nodes", "direct", "uniform"),
    Cell("driven_wave", "eps_r", "direct", "uniform", "lossy"),
    # a design at the wave port faces: the port pass rebuilds its 2D factors
    Cell("driven_wave", "eps_r", "direct", "uniform", "face_lossy"),
    Cell("driven_wave", "nodes", "direct", "uniform", "face_lossy"),
    Cell("port_mode", "eps_r", "direct"),
    # port waves read by reciprocity: the ports' own columns of the entry, and
    # the adaptive sweep's full-size adjoint columns
    Cell("driven_dipole_reciprocal", "eps_r", "direct", "uniform"),
    Cell("driven_dipole_reciprocal", "eps_r", "direct", "adaptive"),
    Cell("eigenmode", "conductivity", "direct"),
    Cell("eigenmode", "fill", "direct"),
    Cell("eigenmode", "nodes", "direct"),
    # the mode-field adjoint: ratios (cell 1), a probed field (cell 26)
    Cell("eigenmode_participation", "eps_r", "direct", variant="inductive"),
    Cell("eigenmode_field", "eps_r", "direct", variant="probe"),
    Cell("eigenmode_participation", "nodes", "direct", variant="inductive"),
    Cell("driven_periodic_bloch", "eps_r", "direct", "uniform"),
    Cell("eigenmode_periodic_bloch", "eps_r", "direct"),
    # a permittivity law as a pytree inside `gfem.Material` (cells D1 and E1)
    Cell("driven", "law", "direct", "uniform", "poles"),
    Cell("eigenmode", "law", "direct", variant="lorentz"),
    # thermal and DC conduction: the reverse sweep's kept
    # trajectory, and the dissipation's adjoint
    Cell("steady_thermal", "thermal_conductivity", "direct"),
    Cell("transient_thermal", "thermal_conductivity", "direct", variant="tr_bdf2-full"),
    Cell("transient_thermal", "heat", "direct", variant="theta-full-waveform"),
    Cell("dc_conduction", "conductivity", "direct", variant="dissipation"),
]
#: Two solves of one design agree to about 1 ulp (threaded assembly), so do
#: their gradients. An eigenmode gradient is only as reproducible as the
#: refined eigenvector, to the eigen tolerance (1e-12) times the operator's
#: condition: each solve's Arnoldi run starts from a random vector, so two
#: solves' gradients differ by up to ~1e-7 (the value by 1e-15).
MODE_FIELD = frozenset({"eigenmode_participation", "eigenmode_field"})

RTOL = {
    "eigenmode": 1e-6,
    "eigenmode_participation": 1e-6,
    "eigenmode_field": 1e-6,
    "eigenmode_periodic_plain": 1e-6,
    "eigenmode_periodic_bloch": 1e-6,
}


def _assert_same(got, want, cell):
    got, want = np.asarray(got), np.asarray(want)
    error = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert error <= RTOL.get(cell.problem, 1e-9), f"{cell}: relative error {error:.2e}"


@pytest.mark.parametrize("cell", FAMILIES, ids=str)
def test_gradient_matrix_transformations_and_eviction_agree(designs, cell):
    # vmap(grad) is a loop of solves, a pullback can be called twice, and a
    # backward without its cache entry recomputes it from the residual design
    case = _case(designs, cell, production=False)
    (d,) = case.directions(1)
    batch = jnp.stack([case.p, case.p + 10 * case.step * d])
    grad = jax.grad(case.objective)
    looped = jnp.stack([grad(p) for p in batch])
    assert np.abs(np.asarray(looped)).max() > 0
    vmapped = jax.vmap(grad)(batch)
    for k in range(2):
        _assert_same(vmapped[k], looped[k], cell)

    _, pullback = jax.vjp(case.objective, case.p)
    first = pullback(1.0)[0]
    _assert_same(first, looped[0], cell)
    # the same kept entry: the second pullback repeats the first exactly, but
    # for a mode-field adjoint, an iterative solve on the kept operator that a
    # first pullback may have reshifted: there it agrees to the
    # solve's acceptance (measured 1e-11 on MN5), not bit for bit
    repeat = 1e-9 if cell.problem in MODE_FIELD else 1e-12
    np.testing.assert_allclose(pullback(2.0)[0], 2 * first, rtol=repeat)

    options = dataclasses.replace(_options(cell, production=False), retain_memory=0)
    evicted = _builder(cell)(designs, cell, options, 0)
    _assert_same(jax.grad(evicted.objective)(case.p), looped[0], cell)


@pytest.mark.parametrize(
    "solver", [*SOLVERS, pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_gradient_matrix_reciprocity_equals_the_explicit_adjoint(designs, solver):
    # Exciting p1 only leaves S21 without p2's solution: its gradient takes
    # an adjoint solve. With both ports excited it comes from reciprocity.
    line = designs.line
    values, _ = line_design(line, seed=3)
    options = tight(solver)

    def s21_gradient(problem):
        def objective(eps):
            model = with_fields(line, {**values, "eps_r": eps})
            s = gfem.solve(model, problem, options=options).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return jax.grad(objective)(values["eps_r"])

    full = s21_gradient(gfem.Driven(frequency=LINE_FREQUENCIES))
    adjoint = s21_gradient(gfem.Driven(frequency=LINE_FREQUENCIES, excite="p1"))
    np.testing.assert_allclose(adjoint, full, rtol=1e-8, atol=1e-10 * abs(full).max())


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity", "law"])
def test_gradient_matrix_adaptive_equals_the_full_solve_at_the_samples(
    designs, solver, name
):
    # at a sample the full solution lies in the frozen basis
    if name == "law":
        # the strip line's moved poles law (cell D3), in its parameters
        cell = Cell("driven", "law", solver, "adaptive", "poles")
        case = _adaptive_law(designs, cell, tight(solver), 0)
        value, grad = jax.value_and_grad(case.objective)(case.p)
        full_value, full_grad = jax.value_and_grad(case.reference)(case.p)
        assert float(value) == pytest.approx(float(full_value), rel=1e-9)
        assert _relative_norm(grad, full_grad) <= 1e-7
        return
    line = designs.line
    values, _ = line_design(line)
    options = tight(solver)
    adaptive_cell = Cell("driven", name, solver, "adaptive")
    adaptive = _driven_material(designs, adaptive_cell, options, 0).objective

    def full(p):
        model = with_fields(line, {**values, name: p})
        problem = gfem.Driven(frequency=LINE_SAMPLES)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    p = values[name]
    value, grad = jax.value_and_grad(adaptive)(p)
    full_value, full_grad = jax.value_and_grad(full)(p)
    assert float(value) == pytest.approx(float(full_value), rel=1e-9)
    assert _relative_norm(grad, full_grad) <= 1e-7


def _relative_norm(a, b) -> float:
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def _shared_surface_core(designs, problem, options, *, seed=0, tan_delta=0.0):
    """The shared-surface objective in the core's eps_r, on ``problem``'s rows."""
    mesh = designs.hybrid_mesh
    model = hybrid_line(mesh, tan_delta=tan_delta)
    values, _ = core_design(mesh, seed)
    rows = jnp.arange(len(problem.frequency))

    def objective(eps):
        m = model.with_material("core", gfem.MaterialField(**{**values, "eps_r": eps}))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return shared_surface_functional(
            gfem.solve(m, problem, options=options).s, rows
        )

    return objective, values["eps_r"]


@pytest.mark.parametrize("tan_delta", [0.0, 0.05], ids=["lossless", "lossy"])
@pytest.mark.parametrize("solver", SOLVERS)
def test_gradient_matrix_shared_surface_reciprocity_equals_the_explicit_adjoint(
    designs, solver, tan_delta
):
    # Exciting p1_m1 only leaves the cross-mode entries without p1_m2's and
    # p2_m2's solutions: their gradients take adjoint solves through the
    # shared surfaces' full modal corrections. With every port excited they
    # come from reciprocity, which holds only if each correction is symmetric.
    options = tight(solver)

    def gradient(problem):
        objective, eps = _shared_surface_core(
            designs, problem, options, seed=3, tan_delta=tan_delta
        )
        return jax.grad(objective)(eps)

    full = gradient(gfem.Driven(frequency=HYBRID_FREQUENCIES))
    adjoint = gradient(gfem.Driven(frequency=HYBRID_FREQUENCIES, excite="p1_m1"))
    error = np.abs(np.asarray(adjoint - full)).max() / np.abs(np.asarray(full)).max()
    assert error <= SHARED_SURFACE_RECIPROCITY[solver]


@pytest.mark.parametrize("variant", ["shared", "shared_lossy"])
def test_gradient_matrix_shared_surface_adaptive_equals_the_full_solve_at_the_samples(
    designs, variant
):
    # at a sample the full solution lies in the frozen basis
    options = tight("direct")
    cell = Cell("driven_wave", "eps_r", "direct", "adaptive", variant)
    adaptive = _wave_material(designs, cell, options, 0).objective
    problem = gfem.Driven(frequency=HYBRID_SAMPLES, excite="p1_m1")
    full, eps = _shared_surface_core(
        designs, problem, options, tan_delta=SHARED_LOSSY_TAN_DELTA[variant]
    )
    value, grad = jax.value_and_grad(adaptive)(eps)
    full_value, full_grad = jax.value_and_grad(full)(eps)
    value_error = abs(float(value) - float(full_value)) / abs(float(full_value))
    error = float(jnp.linalg.norm(grad - full_grad) / jnp.linalg.norm(full_grad))
    assert value_error <= SHARED_SURFACE_ADAPTIVE[0]
    assert error <= SHARED_SURFACE_ADAPTIVE[1]


# --- consistency on a periodic model (the gradient cell G) -------------------------


@pytest.mark.parametrize("name", ["eps_r", "nodes"])
def test_gradient_matrix_bloch_at_zero_wave_vector_equals_plain(designs, name):
    # k = 0 takes the general path (LU, the transposed system, no reciprocity
    # shortcut) to the plain periodic field
    plain_cell = Cell("driven_periodic_plain", name, "direct", "uniform")
    bloch_cell = Cell("driven_periodic_bloch", name, "direct", "uniform", "k0")
    plain = _case(designs, plain_cell, production=False)
    bloch = _case(designs, bloch_cell, production=False)
    want = np.asarray(jax.grad(plain.objective)(plain.p))
    got = np.asarray(jax.grad(bloch.objective)(plain.p))
    assert np.abs(want).max() > 0
    assert np.linalg.norm(got - want) <= 1e-8 * np.linalg.norm(want)


# --- the band diagram sweep over the eigen cell --------------------------------------

#: A target below half the lowest frequency found along the path (1.5-15 GHz).
BAND_TARGET = 0.5e9
#: The weights of every band's frequency in the band objective, wave vector by band.
BAND_WEIGHTS = np.array([[0.3, 1.0], [0.7, 0.4], [0.5, 0.9]])


def _band_path():
    """Γ, half way and X along x: the eigen cell's raw path of three wave vectors."""
    x = np.pi / (EG_A * MM)
    return np.array([[0.0, 0.0, 0.0], [0.5 * x, 0.0, 0.0], [x, 0.0, 0.0]])


def _band_objective(d: Designs, *, independent: bool):
    r"""The gap and a weighted sum of the bands of the eigen cell, in its eps.

    :math:`\mathrm{Re}[\min_k \omega_1 - \max_k \omega_0]
    + \sum w\,\mathrm{Re}\,\omega` over the three wave vectors of
    `_band_path`. With ``independent`` each wave vector is a `gfem.solve` on a
    fresh discretization (the caches released between them), sorted by hand,
    with the static band at Γ left out.
    """
    mesh, path = d.eigen_cell, _band_path()
    options = tight("direct")

    def sweep(eps):
        model = eigen_cell(mesh, eps, tan_delta=EG_TAN_DELTA)
        if not independent:
            bd = gfem.band_diagram(
                model, path, target=BAND_TARGET, n_bands=2, options=options
            )
            return bd.omega
        rows = []
        for k in path:
            gfem.release_memory()
            problem = gfem.Eigenmode(BAND_TARGET, n_modes=2, wave_vector=tuple(k))
            omega = gfem.solve(model, problem, options=options).omega
            order = np.argsort(np.asarray(jax.lax.stop_gradient(omega)).real)
            rows.append(omega[order])
        # Γ drops its static band: the band diagram fills it in at 0
        rows[0] = jnp.concatenate([jnp.zeros(1, complex), rows[0][:1]])
        return jnp.stack(rows)

    def objective(eps):
        w = jnp.real(sweep(eps) * OMEGA_WEIGHT)
        return w[:, 1].min() - w[:, 0].max() + jnp.sum(BAND_WEIGHTS * w)

    return objective


def test_gradient_matrix_band_diagram_matches_central_differences(designs):
    objective = _band_objective(designs, independent=False)
    eps = jnp.asarray(eigen_cell_design(designs.eigen_cell)["eps"])
    rng = np.random.default_rng(5)
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=STEPS["eps_r"])


def test_gradient_matrix_band_diagram_equals_independent_gradients(designs):
    eps = jnp.asarray(eigen_cell_design(designs.eigen_cell)["eps"])
    swept = jax.grad(_band_objective(designs, independent=False))(eps)
    apart = jax.grad(_band_objective(designs, independent=True))(eps)
    assert np.abs(np.asarray(apart)).max() > 0
    np.testing.assert_allclose(
        np.asarray(swept), np.asarray(apart), rtol=1e-9, atol=1e-9 * np.abs(apart).max()
    )


def _periodic_nodes_objective(model):
    """The Bloch S objective of ``model`` as a function of its raw node coordinates."""
    problem = gfem.Driven(frequency=pc.G_FREQUENCIES, wave_vector=pc.G_K)
    treedef = jax.tree_util.tree_structure(model.mesh)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(treedef, [nodes])
        result = gfem.solve(model.replace(mesh=mesh), problem, options=tight("direct"))
        return s_functional(result.s.values)

    return objective


def test_gradient_matrix_rigid_translation_of_a_periodic_cell_is_free(designs):
    # every term is invariant under a rigid move of the whole mesh, twins and
    # all: the raw node gradient has no component along it
    model = pc.g(designs.periodic)
    nodes = jnp.asarray(model.mesh.nodes)
    grad = np.asarray(jax.grad(_periodic_nodes_objective(model))(nodes))
    t = np.broadcast_to(np.array([0.3, -0.5, 0.8]), grad.shape)
    assert np.abs(grad).max() > 0
    assert abs(np.sum(grad * t)) <= 1e-10 * np.linalg.norm(grad) * np.linalg.norm(t)


def test_gradient_matrix_tied_morph_gradient_is_its_transpose_on_the_raw_one(designs):
    # the morph's shared columns sum each tie class's raw cotangents
    model = pc.g(designs.periodic)
    morph = boundary_morph(model, "wall", grows="core")
    objective = _periodic_nodes_objective(model)
    t = jnp.zeros(morph.n_params)
    tied = np.asarray(jax.grad(lambda t: objective(morph(t).nodes))(t))
    raw = np.asarray(jax.grad(objective)(jnp.asarray(model.mesh.nodes)))
    jacobian = np.asarray(jax.jacfwd(lambda t: morph(t).nodes)(t))  # (n, 3, p)
    want = np.einsum("nip,ni->p", jacobian, raw)
    assert np.abs(want).max() > 0
    np.testing.assert_allclose(tied, want, rtol=0, atol=1e-12 * np.abs(want).max())


# --- degenerate eigenmode clusters -------------------------------------------------
# The TE101 / TE011 pair of a square-based PEC box on a mesh with the
# square's symmetry: exactly degenerate. Only functions symmetric in the pair
# differentiate; one member alone raises gfem.DegenerateModeError (on the
# host, in the backward pass: JAX re-raises it as a JaxRuntimeError that
# keeps the message).

DEGENERATE = jax.errors.JaxRuntimeError
CLUSTER_OPTIONS = gfem.SolverOptions(eigen_tolerance=1e-12)


def _cavity_modes(mesh, eps, n_modes):
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.MaterialField(eps_r=eps, tan_delta=1e-3)},
        boundaries=[gfem.PEC("wall")],
    )
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=n_modes)
    return gfem.solve(model, problem, options=CLUSTER_OPTIONS)


def _uniform(mesh):
    return jnp.ones(len(mesh.elements("fill")))


def test_gradient_matrix_degenerate_pair_is_labelled_as_one_cluster(designs):
    cavity = designs.mirrored_cavity
    result = _cavity_modes(cavity, _uniform(cavity), 3)
    omega = np.asarray(result.omega)
    assert abs(omega[0] - omega[1]) <= 1e-12 * abs(omega[0])
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 0, 1])


@pytest.mark.parametrize("overlap", [0.95, 0.999])
def test_gradient_matrix_nearly_parallel_ritz_vectors_keep_the_pair(
    designs, monkeypatch, overlap
):
    # ARPACK returns an arbitrary basis of a multiple eigenvalue's eigenspace:
    # how parallel its two vectors are depends on round-off (process state)
    def nearly_parallel(shift_invert):
        def seed_solve(*args, **kwargs):
            k, vectors = shift_invert(*args, **kwargs)
            i, j = np.argsort(np.where(k.real > 0, k.real, np.inf))[:2]
            assert abs(k[i] - k[j]) <= 1e-12 * abs(k[i])  # the pair
            q = np.linalg.qr(vectors[:, [i, j]])[0]
            vectors = vectors.copy()
            vectors[:, i] = q[:, 0]
            vectors[:, j] = overlap * q[:, 0] + np.sqrt(1 - overlap**2) * q[:, 1]
            return k, vectors

        return seed_solve

    steer_search(monkeypatch, eigen_solver=nearly_parallel)
    cavity = designs.mirrored_cavity
    result = _cavity_modes(cavity, _uniform(cavity), 3)
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 0, 1])


def test_gradient_matrix_cluster_sum_matches_central_differences(designs):
    cavity = designs.mirrored_cavity

    def objective(eps):
        omega = _cavity_modes(cavity, eps, 2).omega
        return jnp.real(jnp.sum(omega) * OMEGA_WEIGHT)

    eps = _uniform(cavity)
    rng = np.random.default_rng(3)
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_cluster_smooth_max_matches_central_differences(designs):
    # any function symmetric in the pair: here a smooth max of the frequencies
    cavity = designs.mirrored_cavity

    def objective(eps):
        f = _cavity_modes(cavity, eps, 3).frequency[:2] / 1e9
        return jax.nn.logsumexp(10 * f) / 10

    eps = _uniform(cavity)
    rng = np.random.default_rng(4)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_single_mode_of_a_cluster_raises(designs):
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 2).omega[0])

    with pytest.raises(DEGENERATE, match="DegenerateModeError") as info:
        jax.grad(objective)(_uniform(cavity))
    message = str(info.value)
    assert "18.0" in message  # the cluster's frequencies
    assert "sum or mean" in message  # the remedy


def test_gradient_matrix_cluster_cut_by_n_modes_raises(designs):
    # n_modes=1 keeps one member of the pair: the guard mode shows the rest
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 1).omega[0])

    with pytest.raises(DEGENERATE, match=r"DegenerateModeError.*n_modes"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_conductor_cluster_cut_by_n_modes_raises(designs):
    # a conductor's seed asks for one pair beyond n_modes: the pair's other
    # member takes that slot and shows the cut
    cavity = designs.mirrored_cavity

    def objective(eps):
        model = gfem.Model(
            cavity,
            materials={"fill": gfem.MaterialField(eps_r=eps)},
            boundaries=[gfem.Conductor("wall", 5.8e7)],
        )
        problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=1)
        return jnp.real(gfem.solve(model, problem, options=CLUSTER_OPTIONS).omega[0])

    with pytest.raises(DEGENERATE, match=r"DegenerateModeError.*n_modes"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_conductor_cluster_cut_with_split_ritz_values_raises(
    designs, monkeypatch
):
    # the frozen seed's Ritz values of the pair split by 3e-8 (relative, at
    # seed tolerance 1e-3: CI run 41300599403) and its vectors were nearly
    # parallel: both seeds refined to one mode, and the guard showed no cut
    def split_and_parallel(shift_invert):
        def seed_solve(*args, **kwargs):
            k, vectors = shift_invert(*args, **kwargs)
            i, j = np.argsort(np.where(k.real > 0, k.real, np.inf))[:2]
            assert abs(k[i] - k[j]) <= 1e-6 * abs(k[i])  # the pair
            k = k.copy()
            k[j] = k[i] * (1 + 3e-8)
            q = np.linalg.qr(vectors[:, [i, j]])[0]
            vectors = vectors.copy()
            vectors[:, i] = q[:, 0]
            vectors[:, j] = 0.95 * q[:, 0] + np.sqrt(1 - 0.95**2) * q[:, 1]
            return k, vectors

        return seed_solve

    steer_search(monkeypatch, eigen_solver=split_and_parallel)
    cavity = designs.mirrored_cavity

    def objective(eps):
        model = gfem.Model(
            cavity,
            materials={"fill": gfem.MaterialField(eps_r=eps)},
            boundaries=[gfem.Conductor("wall", 5.8e7)],
        )
        problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=1)
        return jnp.real(gfem.solve(model, problem, options=CLUSTER_OPTIONS).omega[0])

    with pytest.raises(DEGENERATE, match=r"DegenerateModeError.*n_modes"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_simple_mode_beside_a_cluster_differentiates(designs):
    # TM110 alone: no cotangent reaches the pair
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 3).omega[2] * OMEGA_WEIGHT)

    eps = _uniform(cavity)
    rng = np.random.default_rng(5)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_stop_gradient_on_the_other_member_still_raises(designs):
    # the cotangents differ (one is zero): not symmetric, whatever the opt-out
    cavity = designs.mirrored_cavity

    def objective(eps):
        omega = _cavity_modes(cavity, eps, 2).omega
        return jnp.real(omega[0] + jax.lax.stop_gradient(omega[1]))

    with pytest.raises(DEGENERATE, match="DegenerateModeError"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_cluster_sum_in_the_node_coordinates(designs):
    # a shape design that keeps the square's symmetry: one side wall moves
    # (so the pair splits along the design direction); the cluster sum is
    # symmetric whatever the split
    cavity = designs.mirrored_cavity
    model = gfem.Model(
        cavity,
        materials={"fill": gfem.Material(eps_r=1.0, tan_delta=1e-3)},
        boundaries=[gfem.PEC("wall")],
    )
    morph = boundary_morph(model, "wall", where=lambda x: np.abs(x[:, 2]) < 1e-9)
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=2)

    def objective(t):
        moved = model.replace(mesh=morph(t))
        omega = gfem.solve(moved, problem, options=CLUSTER_OPTIONS).omega
        return jnp.real(jnp.sum(omega) * OMEGA_WEIGHT)

    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(6)
    directions = unit_directions(rng, t.shape, 2)
    assert_gradient_matches(objective, t, directions, step=STEPS["nodes"])


# --- periodic eigenmodes: Bloch at k = 0 is plain periodicity ----------------------


@pytest.mark.parametrize("name", ["eps_r", "nodes"])
def test_gradient_matrix_bloch_eigen_at_k_zero_equals_plain(designs, name):
    # the general path (LU, two-sided refinement, complex projection) at
    # k = 0 and the plain one solve one operator; their gradients agree to
    # the eigen gradients' reproducibility (`RTOL`)
    cell = Cell("eigenmode_periodic_plain", name, "direct")
    options = _options(cell, production=False)
    plain = _eigen_periodic(designs, cell, options, 0)
    at_zero = _eigen_periodic(designs, cell, options, 0, wave_vector=(0.0, 0.0, 0.0))
    want = jax.grad(plain.objective)(plain.p)
    assert np.abs(np.asarray(want)).max() > 0
    _assert_same(jax.grad(at_zero.objective)(plain.p), want, cell)


# --- the chain: DC conduction -> Joule heating -> thermal -> thermo-optic shift ------

FREQUENCY_OPTICAL = 299792458.0 / 1.55e-6
#: The chain's inputs, normalised to O(1): sigma = 1e6 s S/m (heater), k =
#: base k times a factor (every element), dn/dT = 1e-4 d /K (core), I = 4e-4 i
#: A (about 1 W/m in the heater).
CHAIN_K = {"clad": 1.38, "core": 90.0, "heater": 28.0}
CHAIN_TRANSIENT_DURATION, CHAIN_TRANSIENT_STEPS = 1e-6, 6
#: The difference step on the normalised inputs. The start's offset from the
#: source-free temperature (a few K, in the phase trace's rise and the
#: core's temperature) carries the solves' round-off, about 1e-11 relative:
#: at 1e-2 it leaves 3e-6 in the k differences, at 3e-2 1e-7 (fourth-order
#: truncation 4e-8).
CHAIN_STEP = 3e-2


def _chain(d: Designs, solver: str, transient: bool):
    """The chain's objective of the normalised inputs, and a random design."""
    mesh = d.chain_slab
    options = tight(solver)
    tets = {r: mesh.elements(r) for r in CHAIN_K}
    sizes = [len(tets["heater"]), len(mesh.tetrahedra), len(tets["core"]), 1]
    cuts = np.cumsum(sizes)[:-1]

    def thermal(s, kk):
        def k(region):
            return CHAIN_K[region] * kk[tets[region]]

        def material(region, rho, cp, **extra):
            n = len(tets[region])
            return gfem.MaterialField(
                thermal_conductivity=k(region),
                mass_density=jnp.full(n, rho),
                specific_heat=jnp.full(n, cp),
                **extra,
            )

        return gfem.Model(
            mesh,
            materials={
                "clad": material("clad", 2203.0, 709.0),
                "core": material("core", 2330.0, 711.0),
                "heater": material("heater", 5240.0, 598.0, conductivity=1e6 * s),
            },
            boundaries=[
                gfem.FixedTemperature("bottom", temperature=300.0),
                # another ambient: the source-free temperature is not uniform
                gfem.Convection("top", coefficient=1e4, ambient=290.0),
            ],
        )

    def optical(dn):
        # the same slab with the heater out of the mode solve (a traced
        # conductivity on the model the shift reads raises)
        n_clad = len(tets["clad"])
        materials = {
            "clad": gfem.MaterialField(
                eps_r=jnp.full(n_clad, 1.444**2),
                thermo_optic_coefficient=jnp.full(n_clad, 1e-5),
            ),
            "core": gfem.MaterialField(
                eps_r=jnp.full(len(tets["core"]), 3.4777**2),
                thermo_optic_coefficient=1e-4 * dn,
            ),
        }
        return gfem.Model(
            mesh,
            materials=materials,
            ports=[gfem.WavePort("p", "front")],
            excluded={"em": ["heater"]},
        )

    def objective(p):
        s, kk, dn, i = jnp.split(p, cuts)
        m, o = thermal(s, kk), optical(dn)
        dc = gfem.solve(
            m, gfem.DCConduction({"h": "front"}, ground="back"), options=options
        )
        heat = gfem.joule_heating(dc, current=4e-4 * i[0])
        if not transient:
            steady = gfem.solve(m, gfem.SteadyThermal(heat), options=options)
            shift = gfem.thermo_optic_shift(
                o, steady.rise, port="p", frequency=FREQUENCY_OPTICAL
            )
            core = gfem.region_mean(steady.temperature, "core")
            return 1e-2 * shift.phase_per_length + core - 295.0
        shift = gfem.thermo_optic_shift(o, None, port="p", frequency=FREQUENCY_OPTICAL)
        problem = gfem.TransientThermal(
            heat,
            duration=CHAIN_TRANSIENT_DURATION,
            steps=CHAIN_TRANSIENT_STEPS,
            initial_temperature=295.0,
            traces={"phase": shift.trace(), "core": gfem.Trace.mean("core")},
        )
        result = gfem.solve(m, problem, options=options)
        weights = np.linspace(1.0, 2.0, CHAIN_TRANSIENT_STEPS + 1)
        return 1e-2 * jnp.sum(weights * result.traces["phase"]) + jnp.sum(
            result.traces["core"] - 295.0
        )

    rng = np.random.default_rng(23)
    p = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(0.8, 1.2, sizes[0]),
                rng.uniform(0.8, 1.2, sizes[1]),
                rng.uniform(1.5, 2.0, sizes[2]),
                [1.0],
            ]
        )
    )
    return objective, p, rng, cuts


@pytest.mark.parametrize("variant", ["steady", "transient"])
@pytest.mark.parametrize("solver", SOLVERS)
def test_gradient_matrix_chain_matches_central_differences(designs, solver, variant):
    objective, p, rng, cuts = _chain(designs, solver, variant == "transient")
    grad = np.asarray(jax.grad(objective)(p))
    # every input kind moves it: the conductivity through the heat, k through
    # the rise and the source-free temperature, dn/dT through the shift (on
    # the core's tetrahedra behind the port's triangles, which the mode reads)
    sigma, k, dn, current = np.split(grad, cuts)
    assert np.all(sigma != 0.0)
    assert np.all(k != 0.0)
    assert np.count_nonzero(dn) >= len(dn) // 3
    assert current[0] != 0.0
    # worst of 3 directions on ostau: steady 5e-9, transient 7e-8
    assert_gradient_matches(
        objective, p, unit_directions(rng, p.shape, 3), step=CHAIN_STEP, grad=grad
    )


# --- the electromagnetic chain: driven -> electromagnetic heat -> thermal ------------

#: The glued line's heat chain, on the direct solver: the lower half's
#: per-element loss tangent (0.05 t), the sheet's per-face fill and its
#: resistance (2 kohm r), every element's thermal conductivity (HEAT_K k)
#: and the two-port drive (p1's and p2's incident powers, p2's phase).
EM_CHAIN_F = SHEET_FREQUENCIES[0]
EM_CHAIN_STEP = 1e-2


def _em_chain(d: Designs):
    """The EM chain's objective of the normalised inputs, and a random design."""
    mesh = d.sheet_mesh
    options = tight("direct")
    lower = mesh.elements("lower")
    sizes = [len(lower), len(mesh.elements("sheet")), 1, len(mesh.tetrahedra), 3]
    cuts = np.cumsum(sizes)[:-1]

    def model(t, fill, r, k):
        def material(region, eps, **extra):
            tets = mesh.elements(region)
            return gfem.MaterialField(
                eps_r=jnp.full(len(tets), eps),
                thermal_conductivity=HEAT_K * k[tets],
                **extra,
            )

        sheet = gfem.Impedance(
            "sheet",
            resistance=2e3 * r[0],
            inductance=2e-8,
            fill=gfem.SheetField(fill=fill),
        )
        return gfem.Model(
            mesh,
            materials={
                "lower": material("lower", 3.0, tan_delta=0.05 * t),
                "upper": material("upper", 1.0),
            },
            boundaries=[
                gfem.PEC("plate1"),
                gfem.PEC("plate2"),
                sheet,
                gfem.FixedTemperature("plate1", temperature=300.0),
            ],
            ports=[
                gfem.LumpedPort("p1", "port1", "+y", impedance=200.0),
                gfem.LumpedPort("p2", "port2", "+y", impedance=200.0),
            ],
        )

    def objective(p):
        t, fill, r, k, drive = jnp.split(p, cuts)
        m = model(t, fill, r, k)
        driven = gfem.solve(
            m, gfem.Driven(frequency=EM_CHAIN_F, keep_fields=True), options=options
        )
        return _heated_rise(m, driven, EM_CHAIN_F, drive)

    rng = np.random.default_rng(31)
    p = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(0.6, 1.4, sizes[0]),
                rng.uniform(0.2, 0.8, sizes[1]),
                [1.0],
                rng.uniform(0.8, 1.2, sizes[3]),
                DRIVE,
            ]
        )
    )
    return objective, p, rng, cuts


def test_gradient_matrix_em_chain_matches_central_differences(designs):
    objective, p, rng, cuts = _em_chain(designs)
    grad = np.asarray(jax.grad(objective)(p))
    # every input kind moves it: the loss tangent and the fill and resistance
    # through the heat, k through the rise, the drive through the heat
    t, fill, r, k, drive = np.split(grad, cuts)
    assert np.all(t != 0.0)
    assert np.all(fill != 0.0)
    assert r[0] != 0.0
    assert np.all(k != 0.0)
    assert np.all(drive != 0.0)
    assert_gradient_matches(
        objective, p, unit_directions(rng, p.shape, 3), step=EM_CHAIN_STEP, grad=grad
    )


# --- a design at the wave port faces on an adaptive sweep ----------------------------

#: The adaptive face cells: the lower layer's per-element property, which both
#: port faces see, on the frozen samples.
FACE_ADAPTIVE_CELLS = [
    Cell("driven_wave", "eps_r", "direct", "adaptive", "face_lossless"),
    Cell("driven_wave", "eps_r", "direct", "adaptive", "face_lossy"),
    Cell("driven_wave", "conductivity", "direct", "adaptive", "face_lossy"),
    Cell("driven_wave", "eps_r", "direct", "adaptive", "face_path"),
    # the faces stretch in their plane (sliding wave ports)
    Cell("driven_wave", "nodes", "direct", "adaptive", "face_lossless"),
    Cell("driven_wave", "nodes", "direct", "adaptive", "face_lossy"),
]


def _face_adaptive_nodes(  # noqa: PLR0917
    model, options, adaptive, uniform, rows, paths, seed
) -> Case:
    """The faces stretch in their plane (the top plate moves, the ports slide)."""
    ports = tuple(p.name for p in model.ports if isinstance(p, gfem.WavePort))
    morph = boundary_morph(model, "top", sliding_ports=ports)

    def objective_on(problem, rows):
        functional = _face_objective(rows, paths, shared=False)
        return lambda t: functional(
            gfem.solve(model.replace(mesh=morph(t)), problem, options=options)
        )

    case = Case(
        objective_on(adaptive, rows),
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )
    case.reference = objective_on(uniform, list(range(len(rows))))
    return case


def _face_adaptive_case(d: Designs, cell: Cell, seed: int) -> Case:
    """`_face_material` on frozen samples of an adaptive sweep.

    The differences are taken on a uniform sweep of the samples: there the full
    solution lies in the frozen basis, and the port modes are solved at the
    rows themselves.
    """
    options = _face_options(_options(cell, production=False))
    mesh = d.wave_mesh
    model = face_line(mesh, cell.variant)
    paths = ("p1",) if cell.variant == "face_path" else ()
    adaptive = gfem.Driven(frequency=GRID, sweep="adaptive", samples=WAVE_SAMPLES)
    uniform = gfem.Driven(frequency=WAVE_SAMPLES)
    rows = [GRID.index(f) for f in WAVE_SAMPLES]
    if cell.input == "nodes":
        return _face_adaptive_nodes(
            model, options, adaptive, uniform, rows, paths, seed
        )
    name = _property(cell.input)
    n = len(mesh.elements("lo"))
    values = {"eps_r": ramp(mesh, "lo", 3.5, 4.5)}
    if cell.variant != "face_lossless":
        values["tan_delta"] = jnp.full(n, 0.05)
    if name == "conductivity":
        values[name] = ramp(mesh, "lo", 0.2, 0.6)

    def objective_on(problem, rows):
        functional = _face_objective(rows, paths, shared=False)

        def objective(p):
            field = gfem.MaterialField(**{**values, name: p})
            m = model.with_material("lo", field)
            m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
            return functional(gfem.solve(m, problem, options=options))

        return objective

    case = Case(
        objective_on(adaptive, rows),
        values[name],
        np.random.default_rng(seed),
        step=STEPS[name],
        taylor=3e-2,
    )
    case.reference = objective_on(uniform, list(range(len(rows))))
    return case


@pytest.mark.parametrize("cell", FACE_ADAPTIVE_CELLS, ids=str)
def test_gradient_matrix_adaptive_face_cells_match_uniform_differences_at_samples(
    designs, cell
):
    case = _face_adaptive_case(designs, cell, 0)
    grad = jax.grad(case.objective)(case.p)
    assert_gradient_matches(
        case.objective,
        case.p,
        case.directions(3),
        step=case.step,
        reference=case.reference,
        grad=grad,
    )


def test_gradient_matrix_adaptive_face_gamma_alone_matches_the_uniform_gradient(
    designs,
):
    # a row whose only cotangent is on gamma loads no reduced solve but still
    # runs the port pass
    cell = Cell("driven_wave", "eps_r", "direct", "adaptive", "face_lossy")
    options = _face_options(_options(cell, production=False))
    mesh = designs.wave_mesh
    model = face_line(mesh, cell.variant)
    eps = ramp(mesh, "lo", 3.5, 4.5)
    tan_delta = jnp.full(len(eps), 0.05)
    adaptive = gfem.Driven(frequency=GRID, sweep="adaptive", samples=WAVE_SAMPLES)
    uniform = gfem.Driven(frequency=WAVE_SAMPLES)

    def gamma_gradient(problem):
        def objective(p):
            field = gfem.MaterialField(eps_r=p, tan_delta=tan_delta)
            m = model.with_material("lo", field)
            m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
            result = gfem.solve(m, problem, options=options)
            gamma = result.port_modes["p1"].gamma
            return jnp.imag(gamma[jnp.asarray(rows)]).sum()

        rows = (
            [GRID.index(f) for f in WAVE_SAMPLES]
            if problem is adaptive
            else list(range(len(WAVE_SAMPLES)))
        )
        return jax.grad(objective)(eps)

    got, want = gamma_gradient(adaptive), gamma_gradient(uniform)
    assert _relative_norm(got, want) <= 1e-6
