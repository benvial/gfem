"""The Capabilities page's claims, derived from the tests at docs build time.

The gradient matrix (problem type x gradient input) and the raises list are
never typed by hand: they come from the verification tests' module-level
data, imported by path without running a solve: ``CELLS``
(tests/verification/test_gradient_matrix.py), ``EXPECTED`` and
``OUT_OF_SCOPE`` (tests/verification/test_gradient_matrix_raise_table.py).
The table of what raises above one rank comes the same way: its rows are the
cases of tests/mpi_cases.py, its text gfem's own refusal messages. So does
the ranges of the mesh refinement error estimates over the true error that
the mesh refinement recipe and docs/verification.md quote, one per estimate
kind (``ErrorEstimate.kind``): each module of ``ESTIMATE_TESTS``
(tests/verification/test_quantity_estimate.py,
tests/verification/test_objective_estimate.py,
tests/verification/test_field_estimate.py) holds ``EFFECTIVITY_BY_KIND``
(per kind, per case), ``OVERESTIMATES`` and ``WITNESSES``. A kind's range is
the span of its cases; ``quantity`` is the span of the entry kinds; a
witness is a measured band reaching under its kind's range, quoted as
unclaimed; an overestimate a claimed band whose high end is quoted on its
own. The ``gfem-effectivity`` role renders one end of a range by name
(``quantity low``, ``field high``, ``lumped port across a loss jump low``),
and the ``gfem-effectivity-table`` directive every kind's range. Every kind
of ``gfem.refinement.ESTIMATE_KINDS`` must be emitted (``EMITTED_KINDS``)
and verified. ``PORT_MODE_SHARE`` (tests/verification/test_quantity_estimate.py)
bounds a wave port's 2D mode share of the S estimate (``port mode share
high``).

A cell input with the suffix ``_axes`` (``eps_r_axes``) is the same
property per principal value of a field with fixed axes: it folds into its
property's column, whose cells then name the forms their tests cover
("isotropic and principal values").

The permeability, the loss tangent, the lumped values (``OFF_GRID``: a
lumped port's inductance and capacitance, an `Impedance`'s values), the
parameters of a permittivity law (input kinds ``("law", ...)``) and the
density of a metal-dielectric blend (``("material", region, "blend")``) have cells
but no column: the rows that verify them are listed in one sentence under the
matrix, one phrase per row and group (``OFF_GRID_GROUPS``), and their cells
count in the provenance.

Every claim that cannot be derived raises `CapabilitiesError`, which the
Sphinx hook turns into an `ExtensionError`: a problem key or input kind the
mapping below does not know, a grid cell with no outcome or with
conflicting ones, a missing ``OUT_OF_SCOPE`` row the grid needs, and an
``OUT_OF_SCOPE`` row off the grid with no site label (or a label with no row).

A cell whose variant starts with ``face`` puts the design on a wave port
face; a verified cell with one says so (``faces``), and a face variant on a
problem without wave ports raises `CapabilitiesError`.

`derive` returns one plain dict (lists, dicts, strings and numbers). The
directives on docs/capabilities.md (the matrix, the raises list in the site's
words of ``RAISES_LABELS``) and docs/verification.md (the matrix's provenance
sentence) render it, and every page template
gets it as ``gfem_capabilities`` (the template context, as if set in
``html_context``) for the front page's compact matrix, which shows the rows
marked ``front`` only. Only
`setup` imports Sphinx, so the derivation runs (and is unit-tested) without
it.
"""

import dataclasses
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX_TEST = "tests/verification/test_gradient_matrix.py"
RAISE_TEST = "tests/verification/test_gradient_matrix_raise_table.py"
REFUSAL_CASES = "tests/mpi_cases.py"
ESTIMATE_TEST = "tests/verification/test_quantity_estimate.py"
FIELD_ESTIMATE_TEST = "tests/verification/test_field_estimate.py"
OBJECTIVE_ESTIMATE_TEST = "tests/verification/test_objective_estimate.py"
#: The estimate test modules, each holding ``EFFECTIVITY_BY_KIND``,
#: ``OVERESTIMATES`` and ``WITNESSES`` (and optionally
#: ``ORDER_3_EFFECTIVITY_BY_KIND``, claimed with order 2).
ESTIMATE_TESTS = (ESTIMATE_TEST, OBJECTIVE_ESTIMATE_TEST, FIELD_ESTIMATE_TEST)
CAPABILITIES_DOC = "capabilities"
VERIFICATION_DOC = "verification"  # the matrix's provenance, the effectivity range
FRONT_DOC = "index"  # the gfem-home layout renders the compact matrix
DISTRIBUTED_DOC = "recipes/distributed-solve"
REFINEMENT_DOC = "recipes/mesh-refinement"  # quotes the effectivity range
CONTEXT_KEY = "gfem_capabilities"  # the template context name


class CapabilitiesError(Exception):
    """A Capabilities claim cannot be derived from the tests."""


# --- Mapping: test vocabulary -> site vocabulary ------------------------------


@dataclasses.dataclass(frozen=True)
class Row:
    """A row of the grid: one site problem type."""

    label: str
    detail: str  # the headline result, as the row header's second line
    recipe: str = ""  # docname of the recipe that backs the row
    front: bool = True  # also a row of the front page's compact matrix


@dataclasses.dataclass(frozen=True)
class Problem:
    """A problem key of the tests: its row, headline result leaf, port kind."""

    row: str
    headline: str  # the result leaf of EXPECTED that answers the row
    ports: str = ""  # port kind of its verification cells, if it has ports


ROWS = {
    "driven": Row(
        "Driven",
        "S-parameters, uniform and adaptive sweep",
        recipe="recipes/s-parameter-gradient",
    ),
    "eigenmode": Row(
        "Eigenmode", "frequency and Q", recipe="recipes/eigenmode-gradient"
    ),
    "electrostatic": Row(
        "Electrostatic", "capacitance", recipe="recipes/capacitance-gradient"
    ),
    "port_mode": Row(
        "Port mode problem",
        "effective index, impedance and field",
        recipe="recipes/port-mode-gradient",
    ),
    "piezoelectric": Row(
        "Piezoelectric",
        "admittance and clamped capacitance, 2D",
        recipe="recipes/piezoelectric-gradient",
        front=False,
    ),
    # periodic models: one path per row, so each row lists its own solvers
    # and sweeps (Capabilities page only)
    "driven_periodic_plain": Row(
        "Driven, plain periodic",
        "S-parameters",
        recipe="recipes/periodic-cell",
        front=False,
    ),
    "driven_periodic_bloch": Row(
        "Driven, Bloch",
        "S-parameters at a Bloch wave vector",
        recipe="recipes/periodic-cell",
        front=False,
    ),
    "eigenmode_periodic_plain": Row(
        "Eigenmode, plain periodic",
        "frequency and Q",
        recipe="recipes/periodic-cell",
        front=False,
    ),
    "eigenmode_periodic_bloch": Row(
        "Eigenmode, Bloch",
        "band frequencies and Q",
        recipe="recipes/periodic-cell",
        front=False,
    ),
    # the ratios and normalized fields of isolated modes:
    # Capabilities page only
    "eigenmode_participation": Row(
        "Eigenmode, participations",
        "participation ratios and port Q",
        recipe="recipes/eigenmode-gradient",
        front=False,
    ),
    "eigenmode_field": Row(
        "Eigenmode, mode field",
        "the normalized field at points and interface participation",
        recipe="recipes/measurements",
        front=False,
    ),
    # the energy of a field, the loss of a driven field and the heat it
    # gives: Capabilities page only
    "electric_energy": Row(
        "Electric energy",
        "energy per region of a driven or mode field",
        recipe="recipes/measurements",
        front=False,
    ),
    "dissipated_power": Row(
        "Dissipated power",
        "loss per region and sheet of a driven field",
        recipe="recipes/measurements",
        front=False,
    ),
    "electromagnetic_heating": Row(
        "Electromagnetic heating",
        "temperature from a driven result's loss",
        recipe="recipes/electromagnetic-heating",
        front=False,
    ),
    # sources: Capabilities page only; their node coordinates raise where
    # the plain driven row's differentiate, so each has its own row
    "driven_incident": Row(
        "Driven, incident field",
        "kept fields and cross sections",
        recipe="recipes/cross-sections",
        front=False,
    ),
    "driven_dipole": Row(
        "Driven, electric dipole",
        "emitted power and port waves (also by reciprocity)",
        front=False,
    ),
    "driven_magnetic_dipole": Row(
        "Driven, magnetic dipole", "emitted power and port waves", front=False
    ),
    # thermal and DC conduction: Capabilities page only
    "steady_thermal": Row(
        "Steady thermal",
        "temperature, rise, heat, power and heat flow",
        front=False,
    ),
    "transient_thermal": Row(
        "Transient thermal",
        "traces (heat out too) and kept temperatures, TR-BDF2 and theta",
        front=False,
    ),
    "dc_conduction": Row(
        "DC conduction", "conductance, dissipation and potentials", front=False
    ),
    # measurements of a rise on a held mode: the mode's
    # inputs raise, the rise and the thermo-optic coefficient differentiate
    "thermo_optic_shift": Row(
        "Thermo-optic shift", "phase per length of a section", front=False
    ),
    "route_phase": Row(
        "Route phase", "phase along routes through a 3D rise", front=False
    ),
}
COLUMNS = {
    "eps_r": Row("Permittivity", ""),
    "conductivity": Row("Conductivity", ""),
    "fill": Row("Sheet fill", "", "recipes/sheet-design"),
    "nodes": Row("Node coordinates", "", "recipes/shape-gradient"),
}
PROBLEMS = {
    "driven": Problem("driven", "s.values", "lumped"),
    "driven_wave": Problem("driven", "s.values", "wave"),
    "driven_wave_shared": Problem("driven", "s.values", "wave"),
    "driven_wave_face": Problem("driven", "s.values", "wave"),
    "driven_wave_shared_face": Problem("driven", "s.values", "wave"),
    "driven_wave_face_sheet": Problem("driven", "s.values", "wave"),
    "driven_wave_face_law": Problem("driven", "s.values", "wave"),
    "driven_sheet": Problem("driven", "s.values", "lumped"),
    "driven_adaptive": Problem("driven", "s.values", "lumped"),
    "driven_periodic_plain": Problem("driven_periodic_plain", "s.values", "lumped"),
    "driven_periodic_bloch": Problem("driven_periodic_bloch", "s.values", "lumped"),
    # Floquet ports on a periodic cell: S on the periodic rows
    "driven_floquet_plain": Problem("driven_periodic_plain", "s.values", "floquet"),
    "driven_floquet_bloch": Problem("driven_periodic_bloch", "s.values", "floquet"),
    "eigenmode": Problem("eigenmode", "omega"),
    "eigenmode_sheet": Problem("eigenmode", "omega"),
    "eigenmode_periodic_plain": Problem("eigenmode_periodic_plain", "omega"),
    "eigenmode_periodic_bloch": Problem("eigenmode_periodic_bloch", "omega"),
    # wave ports load the resonance: omega on the eigenmode row, the
    # wave port's q_port under eigenmode_participation
    "eigenmode_wave": Problem("eigenmode", "omega", "wave"),
    "eigenmode_wave_shared": Problem("eigenmode", "omega", "wave"),
    "eigenmode_wave_lossy": Problem("eigenmode", "omega", "wave"),
    # cell-only key: the raise table's eigenmode keys answer for its leaves
    "eigenmode_participation": Problem("eigenmode_participation", "participation"),
    "eigenmode_field": Problem("eigenmode_field", "e_fields.0.coefficients"),
    "electrostatic": Problem("electrostatic", "capacitance"),
    "piezoelectric": Problem("piezoelectric", "admittance_per_length"),
    "port_mode": Problem("port_mode", "gamma"),
    "port_mode_sheet": Problem("port_mode", "gamma"),
    "driven_lumped": Problem("driven", "s.values", "lumped"),
    "driven_lumped_adaptive": Problem("driven", "s.values", "lumped"),
    "eigenmode_lumped": Problem("eigenmode", "omega"),
    "electrostatic_lumped": Problem("electrostatic", "capacitance"),
    "port_mode_lumped": Problem("port_mode", "gamma"),
    "port_mode_law": Problem("port_mode", "gamma"),
    # a permittivity law beside a material field
    "driven_law": Problem("driven", "s.values", "lumped"),
    "driven_law_absorbing": Problem("driven", "s.values", "lumped"),
    "driven_law_adaptive": Problem("driven", "s.values", "lumped"),
    "driven_periodic_law": Problem("driven_periodic_plain", "s.values", "lumped"),
    "eigenmode_law": Problem("eigenmode", "omega"),
    # a metal-dielectric density blend
    "driven_blend": Problem("driven", "s.values", "lumped"),
    # the static permittivity of a law
    "electrostatic_law": Problem("electrostatic", "capacitance"),
    # a perfectly matched layer: a model option on the driven and eigenmode rows
    "driven_pml": Problem("driven", "s.values", "wave"),
    "eigenmode_pml": Problem("eigenmode", "omega"),
    # sources: a plane wave on a scatterer (no port; its kept field), and an
    # electric or magnetic dipole beside a wave port (its emitted power)
    "driven_planewave": Problem("driven_incident", "e_fields.0.coefficients"),
    "driven_dipole": Problem("driven_dipole", "emitted_power.d", "wave"),
    # the dipole's port waves read by reciprocity (no source column)
    "driven_dipole_reciprocal": Problem("driven_dipole", "port_waves", "wave"),
    "driven_magnetic_dipole": Problem(
        "driven_magnetic_dipole", "emitted_power.d", "wave"
    ),
    # the energy of a field (also beside a law; a mode field and a periodic
    # cell are cell-only keys), the loss of a driven field and its heat
    "electric_energy": Problem("electric_energy", "lower"),
    "electric_energy_law": Problem("electric_energy", "lower"),
    "electric_energy_eigenmode": Problem("electric_energy", "lower"),
    "electric_energy_periodic": Problem("electric_energy", "lower"),
    "dissipated_power": Problem("dissipated_power", "lower"),
    "dissipated_power_law": Problem("dissipated_power", "lower"),
    "dissipated_power_blend": Problem("dissipated_power", "lower"),
    "electromagnetic_heating": Problem("electromagnetic_heating", "rise.coefficients"),
    # thermal and DC conduction; the heat through a sink and a cut, steady
    # and transient, on the thermal rows
    "steady_thermal": Problem("steady_thermal", "temperature.coefficients"),
    "transient_thermal": Problem("transient_thermal", "traces.upper"),
    "steady_heat_flow": Problem("steady_thermal", "temperature.coefficients"),
    "transient_heat_out": Problem("transient_thermal", "traces.upper"),
    "dc_conduction": Problem("dc_conduction", "conductance"),
    "thermo_optic_shift": Problem("thermo_optic_shift", "phase_per_length"),
    "route_phase": Problem("route_phase", "phase.r"),
}
#: The rows whose cell-only keys have no ``EXPECTED`` entries of their own:
#: the raise table's leaves of their row's keys answer for them, by prefix
#: (the principal shares, which raise by a lumped value, are not among them).
CELL_ONLY_LEAVES = {
    "eigenmode_participation": (
        "eigenmode",
        ("q_port.", "participation.", "bulk_participation.", "sheet_participation."),
    ),
    "eigenmode_field": ("eigenmode", ("e_fields.0.coefficients",)),
}
# CELLS input names, and the material properties of EXPECTED's input kinds
# ("material", <region>, <property>); None: known, but not a grid column.
# A suffix AXES names the property per principal value (fixed axes).
AXES = "_axes"
CELL_INPUTS = {name: name for name in COLUMNS} | {
    f"{name}{AXES}": name for name in ("eps_r", "conductivity")
}
# The value forms of a material column's cells, by cell input.
FORMS = {"eps_r": "isotropic", "conductivity": "isotropic"} | {
    f"{name}{AXES}": "principal values" for name in ("eps_r", "conductivity")
}
# Cell inputs with no grid column, in the site's words. Their cells are
# listed under the matrix (`also_verified`).
OFF_GRID = {
    "mu_r": "permeability",
    "tan_delta": "loss tangent",
    "lumped": "lumped values",
    "law": "permittivity-law parameters",
    # the density leaf of a law blend, ("material", region, "blend")
    "blend": "density of a metal-dielectric blend",
    # thermal problems: their properties, and the problem leaves
    # ("problem", ...): a heat source's amount, the waveform's scales
    "thermal_conductivity": "thermal conductivity",
    "mass_density": "mass density",
    "specific_heat": "specific heat",
    "thermo_optic_coefficient": "thermo-optic coefficient",
    "heat": "heat-source amount and waveform scales",
    # an electromagnetic heat source's drive, ("drive", ...)
    "drive": "incident powers and phases",
    # a thermal boundary's value, ("boundary", <index>, <value>)
    "boundary_value": "thermal boundary values",
}
#: The off-grid inputs listed together, one phrase per group and row.
OFF_GRID_GROUPS = (
    ("mu_r", "tan_delta"),
    ("lumped",),
    ("law",),
    ("blend",),
    ("thermal_conductivity", "mass_density", "specific_heat"),
    ("thermo_optic_coefficient",),
    ("heat",),
    ("drive",),
    ("boundary_value",),
)
#: The material properties among them (the others have no value forms).
OFF_GRID_PROPERTIES = ("mu_r", "tan_delta")
OFF_GRID_INPUTS = {name: name for name in OFF_GRID} | {
    f"{name}{AXES}": name for name in OFF_GRID_PROPERTIES
}
FORMS |= dict.fromkeys(OFF_GRID_PROPERTIES, "isotropic") | {
    f"{name}{AXES}": "principal values" for name in OFF_GRID_PROPERTIES
}
#: The lumped values of EXPECTED's input kinds ("port", <index>, <value>) and
#: ("boundary", <index>, <value>): the off-grid input "lumped".
LUMPED_VALUES = {
    "port": ("inductance", "capacitance"),
    "boundary": ("resistance", "inductance", "capacitance"),
}
#: The thermal boundary values among EXPECTED's ("boundary", <index>, <value>)
#: input kinds: the off-grid input "boundary_value".
THERMAL_BOUNDARY_VALUES = ("temperature", "coefficient", "ambient", "flux")
MATERIAL_PROPERTIES = {
    "eps_r": "eps_r",
    "conductivity": "conductivity",
    "mu_r": None,
    "tan_delta": None,
    "thermal_conductivity": None,
    "mass_density": None,
    "specific_heat": None,
    "thermo_optic_coefficient": None,
    # a solid's constants: exact zeros of every problem in the raise table
    "stiffness": None,
    "piezoelectric_tensor": None,
    "mechanical_quality_factor": None,
}
SOLVERS = {"direct": "direct solver", "bddc_gmres": "BDDC + GMRES"}
# OUT_OF_SCOPE rows the grid needs (by their label): the row's cell raises.
GRID_LABELS = {
    "electrostatic x sheet fill": ("electrostatic", "fill"),
    "piezoelectric x sheet fill": ("piezoelectric", "fill"),
    "eigenmode mode field x node coordinates": ("eigenmode_field", "nodes"),
    # the shape-gradient refusal of the thermal and conduction forms
    "steady_thermal x node coordinates": ("steady_thermal", "nodes"),
    "transient_thermal x node coordinates": ("transient_thermal", "nodes"),
    "dc_conduction x node coordinates": ("dc_conduction", "nodes"),
}
#: What each raises row asks for, in the site's words, by the test's label (the
#: match key). A row off the grid with no label here, or a label with no row,
#: fails the build.
RAISES_LABELS = {
    "eigenmode mode-field output x law parameter": (
        "Eigenmode participations, port Q and mode fields, with respect to the "
        "parameters of a permittivity law (the frequency and Q differentiate)"
    ),
    "eigenmode x absorbing face on a law region": (
        "Eigenmode with an `Absorbing` boundary on a region with a permittivity "
        "law (raised by the solve)"
    ),
    "periodic eigenmode x law parameter": (
        "Eigenmode of a periodic model, with respect to the parameters of a "
        "permittivity law"
    ),
    "LumpedPort resistance": (
        "Any result, with respect to a lumped port's resistance (its reference "
        "impedance)"
    ),
    "PML target reflection (target_reflection)": (
        "Any result, with respect to a perfectly matched layer's target "
        "reflection (a fixed value of the layer)"
    ),
    "PML target reflection (reference_frequency)": (
        "Any result, with respect to a perfectly matched layer's reference "
        "frequency (a fixed value of the layer)"
    ),
    "FloquetPort offset": (
        "Any result, with respect to a Floquet port's de-embedding offset"
    ),
    "Floquet driven x adaptive sweep": (
        '`Driven(sweep="adaptive")` on a model with Floquet ports'
    ),
    "Floquet driven x petsc": (
        '`Driven` with Floquet ports on `linear_solver="petsc"`'
    ),
    "eigenmode x Floquet port": "`Eigenmode` on a model with Floquet ports",
    "diffraction outputs x incident polarization": (
        "`gfem.diffraction` efficiencies, power balances, Jones matrices and "
        "aperture fields, with respect to the incident polarization (a constant "
        "of the measurement; to optimize it, combine the columns of a Jones "
        "matrix with your own vector)"
    ),
    "angle_sweep under a traced angle": (
        "`gfem.diffraction.angle_sweep`, with respect to the incidence angles "
        "(each incidence is a constant of its solve)"
    ),
    "driven x material of a Floquet port's half-space": (
        "Driven S-parameters, with respect to the material of the half-space "
        "behind a Floquet port (it fixes the diffraction orders)"
    ),
    "driven x nodes of a Floquet port surface": (
        "Driven S-parameters, with respect to node coordinates on a Floquet "
        "port's surface"
    ),
    "cell_response x the cell's own design": (
        "`gfem.diffraction.cell_response`, with respect to the cell's own design "
        "or its sample parameters (a cell response tabulates fixed cells; "
        "differentiating it in the design would need second derivatives of the "
        "solve)"
    ),
    "propagate x distance": (
        "`gfem.diffraction.propagate`, with respect to the propagation distance "
        "or the padding (the transfer function is built once, outside the trace)"
    ),
    "Model.bloch_wave_vector under a traced angle": (
        "`Model.bloch_wave_vector`, with respect to the incidence angles (the "
        "incidence of a Floquet port is a constant of the solve)"
    ),
    "law parameter of a blend's end": (
        "Any result, with respect to the parameters of a permittivity law at an "
        "end of a metal-dielectric blend"
    ),
    "driven x law parameter under a Bloch wave vector": (
        "Driven S-parameters at a Bloch `wave_vector`, with respect to the "
        "parameters of a permittivity law"
    ),
    "material axes (orientation)": (
        "Any result, with respect to a material's principal axes"
    ),
    "values of an anisotropic Material": (
        "Any result, with respect to the values of an anisotropic `Material`"
    ),
    "nodes beside an anisotropic absorbing face": (
        "Any result, with respect to node coordinates, when an `Absorbing` "
        "surface borders an anisotropic material"
    ),
    "2D PortMode x law region at the line port": (
        "A line port's `PortMode`, with respect to the parameters of a "
        "permittivity law in a region at the port (the material field values "
        "differentiate)"
    ),
    "port mode residual": (
        "The backward error of a port mode (`PortModeResult.residual`), with "
        "respect to anything (a diagnostic)"
    ),
    "port mode group index and dispersion": (
        "A port mode's group index and dispersion (`PortModeResult.n_g`, "
        "`beta_2`, `beta_3` and `D`), with respect to anything (`gamma`, "
        "`n_eff` and `Z_PV` differentiate)"
    ),
    "port mode field at points x node coordinates": (
        "A port mode's field at points (`PortModeField.at`), with respect to "
        "node coordinates (`gamma`, `n_eff` and `Z_PV` differentiate)"
    ),
    "characteristic_impedance of a traced mode": (
        "`characteristic_impedance` of a port mode inside `jax.grad` (`Z_PV` on "
        "the result differentiates)"
    ),
    "electro-optic shift x optical permittivity": (
        "`electro_optic_shift`, with respect to the optical model's "
        "permittivity, loss tangent, permeability or conductivity (the shift "
        "holds the mode constant; the tensor, the static permittivity, the "
        "drive and the electrode gap differentiate)"
    ),
    "driven x port mode residual at a port-face design": (
        "The `residual` of a wave port's mode in a driven result, with respect "
        "to a design a wave port's mode reads (the backward error of the mode "
        "solve is a diagnostic)"
    ),
    "driven x port-face design under a Bloch wave vector": (
        "Driven results at a Bloch `wave_vector`, with respect to a design a wave "
        "port's mode reads"
    ),
    "driven x port-face design on a fixed-angle sweep": (
        "Driven results on a fixed-angle sweep, with respect to a design a wave "
        "port's mode reads"
    ),
    "driven x nodes of a port surface": (
        "Driven S-parameters, with respect to node coordinates on a lumped or "
        "Floquet port surface"
    ),
    "eigenmode x nodes of a port surface": (
        "Eigenmode results, with respect to node coordinates on a port surface"
    ),
    "eigenmode mode-field output x nodes of a port surface": (
        "Eigenmode participations, port Q and mode fields, with respect to node "
        "coordinates on a port surface"
    ),
    "eigenmode x material on a wave port face": (
        "Eigenmode results, with respect to the material of a region with "
        "elements on a wave port's face"
    ),
    "eigenmode x law parameter of a region on a wave port face": (
        "Eigenmode results, with respect to the parameters of a permittivity "
        "law in a region with elements on a wave port's face"
    ),
    "eigenmode x Impedance value beside a wave port": (
        "Eigenmode results, with respect to the resistance, inductance or "
        "capacitance of an `Impedance` that shares an edge with a wave port's face"
    ),
    "eigenmode x nodes of a wave port surface": (
        "Eigenmode results with wave ports, with respect to node coordinates on "
        "a wave port's face"
    ),
    "greedy adaptive sweep under a transformation": (
        '`Driven(sweep="adaptive")` without `samples`, inside `jax.grad`, '
        "`jax.jit` or `jax.vmap`"
    ),
    "forward mode": "Forward mode: `jax.jvp` of a solve",
    "probe x node coordinates": (
        "A field value at points (`Field.at`), with respect to node coordinates"
    ),
    "electric energy of an adaptive sweep's field": (
        "The electric energy of a field per region (`electric_energy`) of a "
        "field of an adaptive sweep, with respect to anything the field depends on"
    ),
    "cross_sections x design in the truncation layer": (
        "Cross sections (`cross_sections`), with respect to a material field "
        "with elements in the truncation layer (the elements touching the "
        "perfectly matched layer or an absorbing face)"
    ),
    "design in the far-field layer": (
        "A far field (`far_field`) whose layer just outside its side holds a "
        "design region"
    ),
    "design in the beam_coupling layer": (
        "A beam coupling (`beam_coupling`) whose layer just outside its side "
        "holds a design region"
    ),
    "beam_coupling with an IncidentField on a moved layer": (
        "A beam coupling (`beam_coupling`) into a `gfem.IncidentField` beam "
        "once node coordinates of its layer have moved"
    ),
    "traced beam parameters in beam_coupling's test beam": (
        "A beam coupling (`beam_coupling`), with respect to the parameters of "
        "its `gfem.GaussianBeam` (waist, focus, angles, polarization)"
    ),
    "power_flow x material in its layer": (
        "The power through a surface (`power_flow`), with respect to a material "
        "field with elements in the element layer just outside its side"
    ),
    "cross_sections x node coordinates": (
        "Cross sections (`cross_sections`), with respect to node coordinates (the "
        "incident-field solve refuses them first)"
    ),
    "multipoles x node coordinates": (
        "Multipoles (`multipoles`), with respect to node coordinates through a "
        "solve (the incident-field solve refuses them first)"
    ),
    "multipoles x node coordinates at fixed coefficients": (
        "Multipoles (`multipoles`), with respect to node coordinates at fixed "
        "coefficients"
    ),
    "multipoles x design in the truncation layer": (
        "Multipoles (`multipoles`), with respect to a material field with "
        "elements in the truncation layer (the elements touching the perfectly "
        "matched layer or an absorbing face)"
    ),
    "multipoles x traced origin": (
        "Multipoles (`multipoles`), with respect to the multipole origin"
    ),
    "multipoles x traced intensity": (
        "Multipoles (`multipoles`), with respect to the intensity of a "
        "`gfem.IncidentField`"
    ),
    "current_moments of a traced design": (
        "Current moments (`current_moments`) of a traced result or model "
        "(read the exact moments with `multipoles`)"
    ),
    "heat_flow x node coordinates": (
        "The heat through a surface (`heat_flow`, `heat_flow_per_length`), with "
        "respect to node coordinates"
    ),
    "power_flow x reduced field": (
        "The power through a surface (`power_flow`) of a field of an adaptive "
        "sweep, with respect to anything the field depends on"
    ),
    "dissipated_power x node coordinates": (
        "The dissipated power per region and sheet (`dissipated_power`), with "
        "respect to node coordinates"
    ),
    "dissipated_power x reduced field": (
        "The dissipated power per region and sheet (`dissipated_power`) of a "
        "field of an adaptive sweep, with respect to anything the field depends on"
    ),
    "dissipated_power x frequency": (
        "The dissipated power per region and sheet (`dissipated_power`), with "
        "respect to the field's frequency (a fixed input of the driven solve)"
    ),
    "electromagnetic_heating x reduced field": (
        "The heat of a driven result (`electromagnetic_heating`) of an adaptive "
        "sweep, with respect to anything its fields depend on"
    ),
    "Conductor conductivity": (
        "Any result, with respect to a `Conductor`'s conductivity or thickness "
        "(fixed values of the sheet: its fill, or an `Impedance` resistance, is "
        "the design)"
    ),
    "electromagnetic_heating x frequency": (
        "The heat of a driven result (`electromagnetic_heating`), with respect "
        "to its frequency (a fixed input of the driven solve)"
    ),
    "driven x wave_vector": (
        "Driven S-parameters, with respect to the Bloch wave vector"
    ),
    "driven x traced wave_vector_frequency": (
        "Driven S-parameters of a fixed-angle sweep, with respect to the "
        "reference frequency of its wave vector (`wave_vector_frequency`)"
    ),
    "periodic model x node move that breaks a pair": (
        "A node move that breaks a `Periodic` pair (the two surfaces no longer match)"
    ),
    "boundary_morph x paired moving surface": (
        "`boundary_morph` with a surface of a `Periodic` pair as the moving surface"
    ),
    "Bloch driven x adaptive sweep": (
        '`Driven(sweep="adaptive")` with a Bloch `wave_vector`'
    ),
    "eigenmode x wave_vector": (
        "Eigenmode results, with respect to the Bloch wave vector"
    ),
    "Bloch eigenmode x bddc_gmres": (
        '`Eigenmode` with a Bloch `wave_vector` on `linear_solver="bddc_gmres"`'
    ),
    "eigenmode band group velocity": (
        "The band group velocity of Bloch eigenmodes (`EigenmodeResult.v_g`, "
        "`BandDiagram.v_g`), with respect to anything"
    ),
    "band_diagram inside jax.jit or jax.vmap": (
        "`band_diagram` inside `jax.jit` or `jax.vmap` (it runs under `jax.grad`)"
    ),
    "band_diagram x wave vectors or target": (
        "`band_diagram`, with respect to the wave vectors or the target"
    ),
    "eigenmode backward error": (
        "The backward error of eigenmodes (`EigenmodeResult.error`), with "
        "respect to anything"
    ),
    "eigenmode mode-field output x bddc_gmres": (
        "Eigenmode participations, port Q and mode fields on "
        '`linear_solver="bddc_gmres"`'
    ),
    "eigenmode mode-field output x petsc": (
        'Eigenmode participations, port Q and mode fields on `linear_solver="petsc"`'
    ),
    "eigenmode mode-field output x cuda": (
        'Eigenmode participations, port Q and mode fields on a GPU (`device="cuda"`)'
    ),
    "eigenmode mode-field output x eigen_tolerance above 1e-10": (
        "Eigenmode participations, port Q and mode fields solved with "
        "`eigen_tolerance` above 1e-10"
    ),
    "eigenmode mode-field output x plain periodic model": (
        "Eigenmode participations, port Q and mode fields of a periodic model"
    ),
    "eigenmode mode-field output x Bloch wave vector": (
        "Eigenmode participations, port Q and mode fields at a Bloch wave vector"
    ),
    "eigenmode mode-field output x member of a cluster": (
        "Eigenmode participations, port Q and mode fields of one mode of a "
        "degenerate cluster, or a sum over it"
    ),
    "eigenmode mode-field output x cluster cut by n_modes": (
        "Eigenmode participations, port Q and mode fields of a mode whose "
        "cluster `n_modes` cuts"
    ),
    "eigenmode mode-field output x nearly degenerate mode": (
        "Eigenmode participations, port Q and mode fields of a mode within a "
        "relative gap of 1e-4 of another"
    ),
    "eigenmode q_port of a mode with no voltage on the port": (
        "Port Q of a mode that does not load the port (differentiate `1 / q_port`)"
    ),
    "eigenmode principal_participation x lumped value": (
        "Eigenmode principal participation, with respect to a lumped port's or "
        "an `Impedance`'s values"
    ),
    **{
        f"TransientThermal {what}": (
            f"`TransientThermal`, with respect to its {words} (a fixed value of "
            "the time grid; raised when the problem is built)"
        )
        for what, words in (
            ("duration", "`duration`"),
            ("steps", "`steps`"),
            ("waveform start time", "waveform start times"),
            ("initial_temperature", "`initial_temperature`"),
        )
    },
    **{
        f"{name} under jax.grad": (
            f"`{name}` of a trace (a crossing time found by search; "
            "differentiate the trace's values)"
        )
        for name in ("rise_time", "time_constant")
    },
    "fit_rational under jax.grad": (
        "`NetworkParameters.fit_rational`, with respect to S, its reference "
        "impedance or the fit's weights (the time responses of S differentiate)"
    ),
    "thermo-optic shift x optical permittivity": (
        "`thermo_optic_shift`, with respect to the optical model's permittivity, "
        "loss tangent, permeability or conductivity (the mode is held constant; "
        "the rise and the thermo-optic coefficient differentiate)"
    ),
    "thermo-optic shift x optical node coordinates": (
        "`thermo_optic_shift`, with respect to the optical model's node coordinates"
    ),
    "route phase x window permittivity": (
        "`route_phase`, with respect to the window's permittivity, loss tangent, "
        "permeability or conductivity (the mode is held constant; the rise and "
        "the thermo-optic coefficient differentiate)"
    ),
    "route phase x window node coordinates": (
        "`route_phase`, with respect to the window's node coordinates"
    ),
}
STATUS_TEXT = {"verified": "FD-verified", "zero": "exact zero", "raises": "raises"}
OUTCOMES = {"gradient", "zero", "raises"}
#: How a gfem error raised in the backward pass reaches the caller.
GRADIENT = "while the gradient is computed"
WRAPPED = f"{GRADIENT}, inside a `jax.errors.JaxRuntimeError`"
#: The label of the recipes' "When a gradient raises" section, which says how
#: to catch an error raised while the gradient is computed; each such row
#: links it (`links_gradient_errors`).
GRADIENT_ERRORS = "gradient-errors"


def forms_text(forms: list[str]) -> str:
    """A column's value forms as a cell detail ("" for isotropic only)."""
    if forms == ["isotropic", "principal values"]:
        return "isotropic and principal values"
    if forms == ["principal values"]:
        return "principal values only"
    return ""


def _input_column(kind) -> str | None:
    """The grid column of an EXPECTED input kind (None: off the grid)."""
    if not isinstance(kind, tuple):
        kind = (kind,)
    if kind == ("nodes",):
        return "nodes"
    if len(kind) == 3 and kind[0] == "material" and kind[2] in MATERIAL_PROPERTIES:
        return MATERIAL_PROPERTIES[kind[2]]
    if len(kind) == 4 and kind[0] == "boundary" and kind[2:] == ("fill", "fill"):
        return "fill"
    if _off_grid_input(kind) in (
        "lumped",
        "law",
        "blend",
        "heat",
        "drive",
        "rise",
        "boundary_value",
    ):
        return None
    raise CapabilitiesError(
        f"{RAISE_TEST}: input kind {kind!r} is not in the Capabilities mapping "
        "(docs/_ext/gfem_capabilities.py)"
    )


#: The off-grid input of an input kind that is no model leaf, by its head.
LEAF_INPUTS = {"problem": "heat", "drive": "drive"}


def _off_grid_input(kind: tuple) -> str | None:
    """The `OFF_GRID` input of an EXPECTED input kind, or None."""
    # a law's ("law", region, "eps_inf") or ("law", region, term, parameter)
    if kind[0] == "law" and len(kind) in (3, 4):
        return "law"
    # a thermal problem's leaves (a heat source's amount, the waveform's
    # scales) and an electromagnetic heat source's incident powers and phases
    if kind[0] in LEAF_INPUTS:
        return LEAF_INPUTS[kind[0]]
    # the rise a thermo-optic measurement reads (no cell, no group)
    if kind == ("rise",):
        return "rise"
    if len(kind) == 3 and kind[2] in LUMPED_VALUES.get(kind[0], ()):
        return "lumped"
    if len(kind) == 3 and kind[0] == "material" and kind[2] in OFF_GRID:
        return kind[2]
    thermal_value = (
        len(kind) == 3 and kind[0] == "boundary" and kind[2] in THERMAL_BOUNDARY_VALUES
    )
    return "boundary_value" if thermal_value else None


def _claiming_rows(problem: Problem, leaf: str) -> list[str]:
    """The rows an ``EXPECTED`` leaf of ``problem`` answers for.

    Its own row for the headline leaf, and each cell-only row whose leaves
    (`CELL_ONLY_LEAVES`) it is.
    """
    rows = [
        row
        for row, (base, prefixes) in CELL_ONLY_LEAVES.items()
        if problem.row == base and leaf.startswith(prefixes)
    ]
    if leaf == problem.headline:
        rows.append(problem.row)
    return rows


def _claim(claims: dict, rows: list[str], column: str | None, outcome: str) -> None:
    """Add ``outcome`` to each row's claims in ``column`` (none if ``None``)."""
    if column is None:
        return
    for row in rows:
        claims.setdefault((row, column), set()).add(outcome)


def _problem(key: str, source: str) -> Problem:
    if key not in PROBLEMS:
        raise CapabilitiesError(
            f"{source}: problem {key!r} is not in the Capabilities mapping "
            "(docs/_ext/gfem_capabilities.py): add it to PROBLEMS"
        )
    return PROBLEMS[key]


def _cell_column(cell) -> str:
    if cell.input not in CELL_INPUTS:
        raise CapabilitiesError(
            f"{MATRIX_TEST}: input {cell.input!r} of cell {cell} is not in the "
            "Capabilities mapping (docs/_ext/gfem_capabilities.py)"
        )
    return CELL_INPUTS[cell.input]


# --- Gradient matrix ----------------------------------------------------------


def gradient_matrix(cells, expected, out_of_scope) -> dict:
    """The grid, counts and raises list from the verification tests' data.

    Args:
        cells: ``CELLS``, each with ``problem``, ``input``, ``solver``,
            ``sweep`` and ``variant``.
        expected: ``EXPECTED``, ``(problem, result leaf, input kind) ->
            "gradient" | "zero" | "raises"``.
        out_of_scope: ``OUT_OF_SCOPE``, rows ``(label, call, error, match)``.

    Returns:
        ``columns``, ``rows`` (each with one cell per column), ``also`` (see
        `also_verified`), ``counts``, ``provenance`` and ``raises``.

    Raises:
        CapabilitiesError: when a claim cannot be derived.
    """
    verified: dict[tuple[str, str], list] = {}
    off_grid: dict[tuple[str, str], list] = {}
    for cell in cells:
        if cell.solver not in SOLVERS:
            raise CapabilitiesError(
                f"{MATRIX_TEST}: solver {cell.solver!r} of cell {cell} is not in "
                "the Capabilities mapping (docs/_ext/gfem_capabilities.py)"
            )
        problem = _problem(cell.problem, MATRIX_TEST)
        if cell.input in OFF_GRID_INPUTS:
            key = (problem.row, OFF_GRID_INPUTS[cell.input])
            off_grid.setdefault(key, []).append(cell)
            continue
        verified.setdefault((problem.row, _cell_column(cell)), []).append(cell)

    claimed: dict[tuple[str, str], set[str]] = {}
    claimed_off: dict[tuple[str, str], set[str]] = {}
    headlines: set[str] = set()
    for (key, leaf, kind), outcome in expected.items():
        problem = _problem(key, RAISE_TEST)
        column = _input_column(kind)
        if outcome not in OUTCOMES:
            raise CapabilitiesError(
                f"{RAISE_TEST}: outcome {outcome!r} of {(key, leaf, kind)!r} is not "
                f"one of {sorted(OUTCOMES)}"
            )
        rows = _claiming_rows(problem, leaf)
        if leaf == problem.headline:
            headlines.add(key)
        if column is not None:
            _claim(claimed, rows, column, outcome)
        else:
            _claim(claimed_off, rows, _off_grid_input(kind), outcome)
    missing = sorted({k for k, _, _ in expected} - headlines)
    if missing:
        leaves = ", ".join(f"{k!r} ({PROBLEMS[k].headline!r})" for k in missing)
        raise CapabilitiesError(
            f"{RAISE_TEST}: no row for the headline result leaf of problem {leaves}"
        )

    labels = {}
    for row in out_of_scope:
        label, error, match = _out_of_scope_row(row)
        labels[label] = (error, match)

    rows = []
    for row_key, row in ROWS.items():
        ports_of_row = {
            PROBLEMS[c.problem].ports
            for (r, _), cs in verified.items()
            if r == row_key
            for c in cs
        } - {""}
        out = [
            _resolve(
                row_key,
                column,
                cells=verified.get((row_key, column), []),
                claimed=claimed.get((row_key, column), set()),
                labels=labels,
                ports_of_row=ports_of_row,
            )
            for column in COLUMNS
        ]
        rows.append(
            {
                "key": row_key,
                "label": row.label,
                "detail": row.detail,
                "recipe": row.recipe,
                "front": row.front,
                "cells": out,
            }
        )

    counts = {"cells": len(cells)} | {
        s: sum(c.solver == s for c in cells) for s in SOLVERS
    }
    provenance = (
        f"{counts['cells']} cells ({counts['direct']} on the direct solver, "
        f"{counts['bddc_gmres']} repeated on BDDC + GMRES), each checked along "
        "3 random directions against fourth-order central differences"
    )
    also = also_verified(off_grid, claimed_off)
    off_labels = [label for label in labels if label not in GRID_LABELS]
    unlabelled = [label for label in off_labels if label not in RAISES_LABELS]
    stale = [label for label in RAISES_LABELS if label not in off_labels]
    if unlabelled or stale:
        raise CapabilitiesError(
            f"{RAISE_TEST}: OUT_OF_SCOPE rows with no site label {unlabelled}, "
            f"site labels with no row {stale} (RAISES_LABELS in "
            "docs/_ext/gfem_capabilities.py)"
        )
    raises = [
        {"key": label, "label": RAISES_LABELS[label], **_error(*labels[label])}
        for label in off_labels
    ]
    return {
        "columns": [
            {"key": k, "label": c.label, "recipe": c.recipe} for k, c in COLUMNS.items()
        ],
        "rows": rows,
        "also": also,
        "counts": counts,
        "provenance": provenance,
        "raises": raises,
    }


#: The ``EXPECTED`` outcomes an off-grid input's cells may verify, by input:
#: gradients, and for a permittivity law also the exact zeros of the
#: parameters a problem never reads (the electrostatic static permittivity
#: reads no frequency), and for lumped values the exact zeros of a lumped
#: port's own L and C beside a wave port's mode (which never reads them).
_VERIFIED_CLAIMS = {name: ({"gradient"},) for name in OFF_GRID} | {
    "law": ({"gradient"}, {"gradient", "zero"}),
    "lumped": ({"gradient"}, {"gradient", "zero"}),
}


def also_verified(off_grid: dict, claimed: dict) -> list[dict]:
    """The FD-verified pairs with no grid column, one entry per row and group.

    Args:
        off_grid: ``(row, property) -> cells`` of the `OFF_GRID` inputs.
        claimed: ``(row, property) -> outcomes`` of ``EXPECTED`` for them.

    Returns:
        Per row, in `ROWS` order, and per group of `OFF_GRID_GROUPS` with
        cells there: ``key``, ``label``, ``inputs`` (site
        words), ``cells``, ``forms_text``, ``solvers``, ``sweeps`` and
        ``text``, the entry as one phrase.

    Raises:
        CapabilitiesError: a cell whose pair the raise table does not expect
            to be a gradient.
    """
    out = []
    for row_key, row in ROWS.items():
        for group in OFF_GRID_GROUPS:
            entry = _also_entry(row_key, row, group, off_grid, claimed)
            if entry is not None:
                out.append(entry)
    return out


def _also_entry(row_key, row, group, off_grid, claimed) -> dict | None:
    """One `also_verified` entry: a row's cells of one group of inputs."""
    cells = []
    inputs = []
    for name in group:
        words = OFF_GRID[name]
        found = off_grid.get((row_key, name), [])
        if not found:
            continue
        if claimed.get((row_key, name), set()) not in _VERIFIED_CLAIMS[name]:
            raise CapabilitiesError(
                f"{row.label} x {words}: {MATRIX_TEST} verifies a gradient "
                f"that {RAISE_TEST} does not expect"
            )
        cells += found
        inputs.append(words)
    if not cells:
        return None
    forms = forms_text(sorted({FORMS[c.input] for c in cells if c.input in FORMS}))
    solvers = [SOLVERS[s] for s in dict.fromkeys(c.solver for c in cells)]
    sweeps = [s for s in dict.fromkeys(c.sweep for c in cells) if s]
    parts = [forms] if forms else []
    parts.append(", ".join(solvers))
    if sweeps:
        parts.append(f"{' and '.join(sweeps)} sweep")
    return {
        "key": row_key,
        "label": row.label,
        "inputs": inputs,
        "cells": len(cells),
        "forms_text": forms,
        "solvers": solvers,
        "sweeps": sweeps,
        "text": (
            f"{row.label} with respect to the {' and the '.join(inputs)} "
            f"({', '.join(parts)})"
        ),
    }


def _resolve(row, column, *, cells, claimed, labels, ports_of_row) -> dict:
    """One grid cell: FD-verified, exact zero or raises (and only one)."""
    where = f"grid cell {ROWS[row].label} x {COLUMNS[column].label}"
    statuses = set()
    if cells:
        statuses.add("verified")
    if "gradient" in claimed and not cells:
        raise CapabilitiesError(
            f"{where}: {RAISE_TEST} expects a gradient that no cell of "
            f"{MATRIX_TEST} verifies against finite differences"
        )
    statuses |= {{"gradient": "verified"}.get(o, o) for o in claimed}
    grid_labels = [lab for lab, cell in GRID_LABELS.items() if cell == (row, column)]
    error = ""
    for label in grid_labels:
        if label in labels:
            statuses.add("raises")
            error = _error(*labels[label])["error"]
    if not statuses:
        missing = [lab for lab in grid_labels if lab not in labels]
        hint = f" (the OUT_OF_SCOPE row {missing[0]!r} is gone)" if missing else ""
        raise CapabilitiesError(f"{where}: no outcome can be derived{hint}")
    if len(statuses) > 1:
        raise CapabilitiesError(
            f"{where}: conflicting outcomes {sorted(statuses)} between "
            f"{MATRIX_TEST} and {RAISE_TEST}"
        )
    (status,) = statuses
    ports = [p for p in dict.fromkeys(PROBLEMS[c.problem].ports for c in cells) if p]
    face_cells = [
        c for c in cells if c.variant == "face" or c.variant.startswith("face_")
    ]
    for c in face_cells:
        if PROBLEMS[c.problem].ports != "wave":
            raise CapabilitiesError(
                f"{where}: face variant {c.variant!r} on problem {c.problem!r}, "
                "which has no wave ports"
            )
    forms = sorted({FORMS[c.input] for c in cells if c.input in FORMS})
    note = f"{' and '.join(ports)} ports" if ports and set(ports) < ports_of_row else ""
    label = STATUS_TEXT[status]
    if error:
        label += f" {error}"
    elif note:
        label += f", {note}"
    return {
        "column": column,
        "status": status,
        "text": STATUS_TEXT[status],
        "label": label,
        "note": note,
        "error": error,
        "cells": len(cells),
        "solvers": [SOLVERS[s] for s in dict.fromkeys(c.solver for c in cells)],
        "sweeps": [s for s in dict.fromkeys(c.sweep for c in cells) if s],
        "ports": ports,
        "faces": bool(face_cells),
        "forms": forms,
        "forms_text": forms_text(forms),
        "source": MATRIX_TEST if cells else RAISE_TEST,
    }


def _out_of_scope_row(row) -> tuple[str, type, str]:
    if not (
        isinstance(row, tuple)
        and len(row) == 4
        and isinstance(row[0], str)
        and isinstance(row[2], type)
    ):
        raise CapabilitiesError(
            f"{RAISE_TEST}: OUT_OF_SCOPE row {row!r} is not (label, call, error, match)"
        )
    return row[0], row[2], str(row[3])


def _error(error: type, match: str) -> dict:
    """The error a raises row names.

    A gfem error raised in the backward pass reaches the caller wrapped in
    JAX's runtime error (a ``ValueError`` once a jitted function has already
    succeeded, which no test of the raise table reaches), so its own name (the
    head of the test's ``match``) is the one to publish.
    """
    if error.__name__ == "JaxRuntimeError":
        head = re.match(r"\w+Error\b", match)
        if head:
            return {"error": head.group(), "when": WRAPPED}
        return {"error": error.__name__, "when": GRADIENT}
    return {"error": error.__name__, "when": ""}


def links_gradient_errors(row: dict) -> bool:
    """Whether a raises row links the "When a gradient raises" section."""
    return row["when"].startswith(GRADIENT)


# --- What raises above one rank --------------------------------------------------

#: What each refusal case asks for, in the site's words. A case with no label here,
#: or a label with no case, fails the build.
REFUSAL_LABELS = {
    "curved_mesh": "A curved (second-order) mesh, with any problem",
    "adaptive_sweep": '`Driven(sweep="adaptive")`, with or without `samples`',
    "arpack": '`eigen_solver="arpack"` on an `Eigenmode` problem',
    "thermal": "`SteadyThermal`",
    "thermal_transient": "`TransientThermal`",
    "dc_conduction": "`DCConduction`",
    "periodic": (
        "`Periodic` pairs on two or three axes of a `Driven` or an `Eigenmode` "
        "problem; `Electrostatic` on a periodic model"
    ),
    "bloch": "`Eigenmode(wave_vector=...)` and `band_diagram`",
    "eigenmode_wave_port": "A `WavePort` on an `Eigenmode` problem",
    "floquet_port": "A `FloquetPort` on a `Driven` problem",
    "frequency_workers": "`frequency_workers` above 1, on more than one frequency",
    "device": '`device="cuda"`',
    "bddc_order_1": '`linear_solver="bddc_gmres"` at `order=1`',
    "bddc_order_1_auto": '`linear_solver="auto"` at `order=1`',
    "pmg_gmres": '`linear_solver="pmg_gmres"`',
    "petsc_preconditioner": (
        '`linear_solver="petsc"` with a `pc_type` that has no distributed form '
        "(`ilu`, `icc`, or `lu` or `cholesky` on a package other than MUMPS)"
    ),
    "field_export": "`Field.to_meshio`, `Field.to_pyvista`",
    "load_result_fields": "`load_result` of a bundle with fields",
    "region_mean": "`region_mean`",
    "element_mean": "`element_mean`",
    "electric_energy": "`electric_energy`",
    "dissipated_power": "`dissipated_power`",
    "optical_force": "`optical_force`",
    "electromagnetic_heating": "`electromagnetic_heating`",
    "error_estimate": "`error_estimate` of a result solved across ranks",
    "mesh_refinement": "`refine_mesh`",
    "mode_field_gradient": (
        "A gradient through an `Eigenmode` result's mode field: `q_port`, the "
        "participations, `e_fields`, `interface_participation`"
    ),
    "cad_morph": "`gfem.shape.cad_morph`, `CadMorph.remesh`",
}


def distributed_refusals(keys, methods, guards, feature, messages) -> list[dict]:
    """One row per refusal case: what it asks for, its ``feature``, its message.

    Args:
        keys: The solves refused, ``tests/mpi_cases.py``'s ``KEYS``.
        methods: The method calls refused, its ``METHOD_KEYS``.
        guards: The refusal keys no public call reaches, its ``GUARD_KEYS``. No row.
        feature: The ``feature`` a case raises.
        messages: gfem's own refusal messages by ``feature``.

    Raises:
        CapabilitiesError: The cases, the labels and the messages differ.
    """
    cases = [*keys, *methods]
    if set(cases) != set(REFUSAL_LABELS):
        raise CapabilitiesError(
            f"{REFUSAL_CASES}: the cases {sorted(cases)} are not the labelled ones "
            f"{sorted(REFUSAL_LABELS)}"
        )
    features = {feature(case) for case in cases} | set(guards)
    if features != set(messages):
        raise CapabilitiesError(
            f"{REFUSAL_CASES}: the cases raise {sorted(features)}, and gfem has "
            f"messages for {sorted(messages)}"
        )
    return [
        {
            "label": REFUSAL_LABELS[case],
            "feature": feature(case),
            "message": messages[feature(case)],
        }
        for case in cases
    ]


# --- The error estimates' effectivity -------------------------------------------

#: The lowest effectivity any claimed case may assert: below it a
#: band is a witness, not a claim.
CLAIMS_FLOOR = 0.9

#: The entry kinds the ``quantity`` range spans: the kinds admitted on their
#: measured effectivity.
ENTRY_KINDS = ("capacitance", "s", "omega")

#: Each estimate kind (``gfem.refinement.ESTIMATE_KINDS``) in words, for the
#: ``gfem-effectivity-table`` directive. It must cover the kinds exactly.
KIND_WORDS = {
    "field": "a field, in the energy norm",
    "capacitance": "a capacitance entry",
    "s": "an S entry",
    "omega": "the complex frequency of a mode",
    "q": "the Q of a mode",
    "cluster mean": "the mean frequency of a degenerate cluster",
    "objective of s": "an objective of S",
    "objective of s, lossless rewrite": "an objective of S on a lossless model",
    "objective of omega": "an objective of mode frequencies",
    "objective of capacitance": "an objective of capacitance",
    "objective of electrostatic energy": "an objective of electrostatic energy",
    "objective of potential means": "an objective of potential means",
    "objective of absorbed power": "an objective of absorbed power",
    "objective of field means": "an objective of field means",
    "objective of participation": "an objective of participations",
}


def _ranges(by_case, source: str) -> dict:
    """``by_case`` as floats, each positive and ordered."""
    try:
        ranges = {str(k): tuple(float(v) for v in r) for k, r in by_case.items()}
    except (TypeError, ValueError, AttributeError) as exc:
        raise CapabilitiesError(f"{source}: unreadable ranges: {exc}") from exc
    if not ranges or any(len(r) != 2 or not 0 < r[0] <= r[1] for r in ranges.values()):
        raise CapabilitiesError(f"{source}: bad case ranges {ranges}")
    return ranges


def _quoted(ranges: dict) -> dict:
    """The span of ``ranges`` as the site writes it, with the case names."""
    low = min(r[0] for r in ranges.values())
    high = max(r[1] for r in ranges.values())
    return {"low": f"{low:g}", "high": f"{high:g}", "cases": list(ranges)}


def effectivity_by_kind(by_kind, kinds, emitted, source: str = ESTIMATE_TEST) -> dict:
    """The range the docs quote per estimate kind: the span of its cases.

    Args:
        by_kind: ``EFFECTIVITY_BY_KIND`` (merged over the estimate test
            modules): per kind, one ``(low, high)`` per case.
        kinds: ``gfem.refinement.ESTIMATE_KINDS``, every kind the code knows.
        emitted: ``gfem.refinement.EMITTED_KINDS``, the kinds the
            estimators emit today; each needs a case, and no other may have
            one.
        source: Where the ranges come from, for the messages.

    Returns:
        Per kind of ``emitted`` with cases, ``low`` and ``high`` as the site
        writes them (``"0.9"``, ``"15"``) and ``cases``, the case names in
        the tests' order.

    Raises:
        CapabilitiesError: A key of ``by_kind`` is not in ``kinds`` or not
            emitted; an emitted kind has no case; a case's range is not
            positive and ordered, or starts below `CLAIMS_FLOOR`.
    """
    unknown = [k for k in by_kind if k not in kinds]
    if unknown:
        raise CapabilitiesError(f"{source}: unknown estimate kinds {unknown}")
    silent = [k for k in by_kind if k not in emitted]
    if silent:
        raise CapabilitiesError(
            f"{source}: cases for kinds the estimators do not emit: {silent}"
        )
    missing = [k for k in emitted if not by_kind.get(k)]
    if missing:
        raise CapabilitiesError(
            f"{source}: emitted kinds without a verification case: {missing}"
        )
    out = {}
    for kind in emitted:
        ranges = _ranges(by_kind[kind], f"{source} [{kind!r}]")
        low = [case for case, r in ranges.items() if r[0] < CLAIMS_FLOOR]
        if low:
            raise CapabilitiesError(
                f"{source} [{kind!r}]: cases {low} start below the claims floor "
                f"{CLAIMS_FLOOR:g}; a band reaching under it is a witness"
            )
        out[kind] = _quoted(ranges)
    return out


def entry_span(by_kind: dict, entries=ENTRY_KINDS) -> dict:
    """The ``quantity`` range: the span of the entry kinds' cases."""
    ranges = {}
    for kind in entries:
        ranges.update(by_kind.get(kind, {}))
    if not ranges:
        raise CapabilitiesError(f"no cases for the entry kinds {list(entries)}")
    return _quoted(_ranges(ranges, "the entry kinds"))


def _band(band, what: str, source: str) -> tuple[float, float]:
    try:
        low, high = (float(v) for v in band)
    except (TypeError, ValueError) as exc:
        raise CapabilitiesError(f"{source}: unreadable {what}: {exc}") from exc
    if not 0 < low <= high:
        raise CapabilitiesError(f"{source}: bad {what} band {(low, high)}")
    return low, high


def witness(band, quoted: dict, source: str = FIELD_ESTIMATE_TEST) -> dict:
    """A measured band the docs quote as unclaimed: it reaches under a range.

    Args:
        band: ``(low, high)``, the effectivity a witness test asserts (a
            ``WITNESSES`` entry's band).
        quoted: Its kind's range, as `effectivity_by_kind` returns it.
        source: The test module the band comes from, for the messages.

    Returns:
        ``low`` and ``high`` as the site writes them.

    Raises:
        CapabilitiesError: ``band`` is not positive and ordered, or its low
            end is not below the low end of ``quoted`` (then it is a claim).
    """
    low, high = _band(band, "witness", source)
    if low >= float(quoted["low"]):
        raise CapabilitiesError(
            f"{source}: witness {(low, high)} does not reach below the quoted "
            f"range from {quoted['low']}"
        )
    return {"low": f"{low:g}", "high": f"{high:g}"}


def overestimate(band, quoted: dict, source: str = ESTIMATE_TEST) -> dict:
    """A claimed band whose high end the docs quote on its own (Limits).

    Args:
        band: ``(low, high)``, an ``OVERESTIMATES`` entry's band.
        quoted: Its kind's range, as `effectivity_by_kind` returns it.
        source: The test module the band comes from, for the messages.

    Returns:
        ``low`` and ``high`` as the site writes them.

    Raises:
        CapabilitiesError: ``band`` is not positive and ordered; its low end
            is below the kind's (a witness); or its high end lies inside the
            kind's range (an ordinary case of ``EFFECTIVITY_BY_KIND``).
    """
    low, high = _band(band, "overestimate", source)
    if low < float(quoted["low"]):
        raise CapabilitiesError(
            f"{source}: overestimate {(low, high)} starts below its kind's "
            f"range from {quoted['low']}"
        )
    if high <= float(quoted["high"]):
        raise CapabilitiesError(
            f"{source}: overestimate {(low, high)} lies inside its kind's range "
            f"up to {quoted['high']}: make it a case of EFFECTIVITY_BY_KIND"
        )
    return {"low": f"{low:g}", "high": f"{high:g}"}


def port_mode_share(share, source: str = ESTIMATE_TEST) -> dict:
    """The bound on a wave port's 2D mode share of the S estimate.

    Args:
        share: ``PORT_MODE_SHARE``, the bound the hybrid strip test holds the
            de-embedding term to, as a fraction of the S estimate.
        source: The test module it comes from, for the messages.

    Returns:
        ``high`` as the site writes it; the share has no low end.

    Raises:
        CapabilitiesError: ``share`` is not a number in (0, 1).
    """
    try:
        value = float(share)
    except (TypeError, ValueError) as exc:
        raise CapabilitiesError(f"{source}: unreadable port mode share: {exc}") from exc
    if not 0 < value < 1:
        raise CapabilitiesError(
            f"{source}: port mode share {value:g} is not a fraction in (0, 1)"
        )
    return {"high": f"{value:g}"}


def effectivity_table(ranges: dict, kinds, emitted, words=None) -> list[dict]:
    """The rows of the ``gfem-effectivity-table`` directive, one per emitted kind.

    Args:
        ranges: The quoted ranges, as `load_effectivity` returns them.
        kinds: ``gfem.refinement.ESTIMATE_KINDS``.
        emitted: ``gfem.refinement.EMITTED_KINDS``, the rows in order.
        words: Each kind in words; `KIND_WORDS` by default.

    Raises:
        CapabilitiesError: ``words`` does not cover ``kinds`` exactly.
    """
    words = KIND_WORDS if words is None else words
    if set(words) != set(kinds):
        raise CapabilitiesError(
            f"the kind words cover {sorted(words)}, not the estimate kinds "
            f"{sorted(kinds)}"
        )
    return [
        {
            "kind": kind,
            "words": words[kind],
            "low": ranges[kind]["low"],
            "high": ranges[kind]["high"],
            "cases": len(ranges[kind]["cases"]),
        }
        for kind in emitted
    ]


# --- Loading the verification tests --------------------------------------------


def load_refusals(repo: Path) -> list[dict]:
    """The rows of `distributed_refusals`, from the cases under ``repo``."""
    from gfem import _mpi

    paths = [str(repo / "tests"), str(repo / "tools")]
    sys.path[:0] = paths
    try:
        cases = _import(repo / REFUSAL_CASES)
    finally:
        for p in paths:
            sys.path.remove(p)
    try:
        return distributed_refusals(
            cases.KEYS,
            cases.METHOD_KEYS,
            cases.GUARD_KEYS,
            cases.feature,
            _mpi._REFUSALS,  # noqa: SLF001
        )
    except AttributeError as exc:
        raise CapabilitiesError(f"the refusal cases are gone: {exc}") from exc


def load_verification(repo: Path) -> tuple[list, dict, list]:
    """``CELLS``, ``EXPECTED`` and ``OUT_OF_SCOPE``, imported by path.

    The modules import their helpers from tests/ and tools/ (as pytest's
    ``pythonpath`` does), and gfem, JAX and pytest; nothing is solved.
    """
    paths = [str(repo / "tests"), str(repo / "tools")]
    sys.path[:0] = paths
    try:
        matrix = _import(repo / MATRIX_TEST)
        table = _import(repo / RAISE_TEST)
    finally:
        for p in paths:
            sys.path.remove(p)
    try:
        return list(matrix.CELLS), dict(table.EXPECTED), list(table.OUT_OF_SCOPE)
    except AttributeError as exc:
        raise CapabilitiesError(f"verification data is gone: {exc}") from exc


def _estimate_kinds() -> tuple[tuple, tuple]:
    """``ESTIMATE_KINDS`` and ``EMITTED_KINDS`` of ``gfem.refinement``.

    Raises:
        CapabilitiesError: They are gone, or differ: every kind the code
            knows must be emitted (and so verified and in the table).
    """
    from gfem import refinement

    try:
        kinds = tuple(refinement.ESTIMATE_KINDS)
        emitted = tuple(refinement.EMITTED_KINDS)
    except AttributeError as exc:
        raise CapabilitiesError(f"the estimate kinds are gone: {exc}") from exc
    if emitted != kinds:
        silent = [k for k in kinds if k not in emitted]
        raise CapabilitiesError(
            f"EMITTED_KINDS {list(emitted)} is not ESTIMATE_KINDS: the "
            f"estimators do not emit {silent}, or emit them in another order"
        )
    return kinds, emitted


def _merge(target: dict, name: str, values, source: str, owners: dict) -> None:
    """Add ``values`` (a dict) under ``name``s; a key from two modules raises."""
    for key, value in dict(values).items():
        if key in owners:
            raise CapabilitiesError(
                f"{name} {key!r} is defined in both {owners[key]} and {source}"
            )
        owners[key] = source
        target[key] = value


def load_effectivity(repo: Path) -> dict:
    """The quoted effectivity ranges, from the estimate tests under ``repo``.

    Each module of `ESTIMATE_TESTS` holds ``EFFECTIVITY_BY_KIND``,
    ``OVERESTIMATES`` and ``WITNESSES``; a kind or band name defined in two
    modules raises. A module's ``ORDER_3_EFFECTIVITY_BY_KIND`` is claimed
    with order 2, its cases named ``"<case>, order 3"``.

    Returns:
        Per emitted kind, as `effectivity_by_kind` returns it; ``quantity``,
        the span of the entry kinds (`entry_span`); and each witness (see
        `witness`) and overestimate (see `overestimate`) by its name; and
        ``port mode share``, the bound on a wave port's 2D mode share of the
        S estimate (see `port_mode_share`).
    """
    paths = [str(repo / "tests"), str(repo / "tools")]
    sys.path[:0] = paths
    try:
        modules = {source: _import(repo / source) for source in ESTIMATE_TESTS}
    finally:
        for p in paths:
            sys.path.remove(p)
    by_kind: dict = {}
    bands: dict = {"WITNESSES": {}, "OVERESTIMATES": {}}
    kind_owners: dict = {}
    band_owners: dict = {}
    try:
        for source, module in modules.items():
            cases = {kind: dict(r) for kind, r in module.EFFECTIVITY_BY_KIND.items()}
            for kind, r in getattr(module, "ORDER_3_EFFECTIVITY_BY_KIND", {}).items():
                cases.setdefault(kind, {}).update(
                    {f"{case}, order 3": v for case, v in r.items()}
                )
            _merge(by_kind, "estimate kind", cases, source, kind_owners)
            for name, collected in bands.items():
                values = {k: (source, v) for k, v in getattr(module, name).items()}
                _merge(collected, name, values, source, band_owners)
    except AttributeError as exc:
        raise CapabilitiesError(f"the effectivity ranges are gone: {exc}") from exc
    kinds, emitted = _estimate_kinds()
    out = effectivity_by_kind(by_kind, kinds, emitted, " and ".join(ESTIMATE_TESTS))
    out["quantity"] = entry_span(by_kind)
    for name, check in (("WITNESSES", witness), ("OVERESTIMATES", overestimate)):
        for key, (source, (kind, band)) in bands[name].items():
            if key in out:
                raise CapabilitiesError(f"{source}: {name} {key!r} names a range")
            if kind not in out:
                raise CapabilitiesError(
                    f"{source}: {name} {key!r} is of kind {kind!r}, which has no "
                    "quoted range"
                )
            out[key] = check(band, out[kind], source)
    try:
        share = modules[ESTIMATE_TEST].PORT_MODE_SHARE
    except AttributeError as exc:
        raise CapabilitiesError(f"the port mode share is gone: {exc}") from exc
    out["port mode share"] = port_mode_share(share)
    return out


def _import(path: Path):
    name = f"_gfem_capabilities_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None or not path.is_file():
        raise CapabilitiesError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CapabilitiesError(f"cannot import {path}: {exc!r}") from exc
    return module


def derive(repo: Path = REPO) -> dict:
    """Every Capabilities claim, from the tests under ``repo``.

    Returns:
        ``columns``, ``rows``, ``also``, ``counts``, ``provenance`` and ``raises`` (see
        `gradient_matrix`), ``refusals`` (see `distributed_refusals`),
        ``effectivity`` (see `load_effectivity`) and ``effectivity_table``
        (see `effectivity_table`).
    """
    effectivity = load_effectivity(repo)
    return gradient_matrix(*load_verification(repo)) | {
        "refusals": load_refusals(repo),
        "effectivity": effectivity,
        "effectivity_table": effectivity_table(effectivity, *_estimate_kinds()),
    }


# --- Sphinx --------------------------------------------------------------------


def setup(app):
    """Register the directives and derive the claims once per build."""
    from sphinx.errors import ExtensionError

    def derive_claims(app) -> None:
        try:
            data = derive(REPO)
        except CapabilitiesError as exc:
            raise ExtensionError(
                f"Capabilities: {exc}", modname="gfem_capabilities"
            ) from exc
        app.env.gfem_capabilities = data  # the directives' source

    def add_context(app, _pagename, _templatename, context, _doctree) -> None:
        # Every page template sees the claims (the gfem-home layout renders
        # the compact matrix from them).
        context[CONTEXT_KEY] = app.env.gfem_capabilities

    def reread(_app, env, *_changes) -> list[str]:
        # These pages render the tests' data, not their own source: read them
        # again on every build so an incremental build never serves stale claims.
        pages = (
            CAPABILITIES_DOC,
            VERIFICATION_DOC,
            FRONT_DOC,
            DISTRIBUTED_DOC,
            REFINEMENT_DOC,
        )
        return [d for d in pages if d in env.found_docs]

    app.add_config_value("gfem_capabilities_source_url", "", "html", str)
    app.connect("builder-inited", derive_claims)
    app.connect("env-get-outdated", reread)
    app.connect("html-page-context", add_context)
    for name, directive in _directives().items():
        app.add_directive(name, directive)
    app.add_role("gfem-effectivity", _effectivity_role)
    app.add_role("gfem-test", _test_role)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}


def _effectivity_role(_name, rawtext, text, lineno, inliner, *_args):
    """``{gfem-effectivity}`field low```: one end of a quoted range.

    The argument is the range's name and the end (``low`` or ``high``). The
    names are the estimate kinds (``field``, ``s``, ...; a name may hold
    spaces and commas), ``quantity`` (the entry kinds' span), and the
    ``WITNESSES`` and ``OVERESTIMATES`` keys of the estimate tests (``lumped
    port across a loss jump``, ...). A bare end is an error: several ranges
    exist.
    """
    from docutils import nodes

    data = inliner.document.settings.env.gfem_capabilities["effectivity"]
    name, _, end = text.rpartition(" ")
    if name not in data or end not in ("low", "high") or end not in data[name]:
        message = inliner.reporter.error(
            f"gfem-effectivity takes a range name and low|high, not {text!r}",
            line=lineno,
        )
        return [inliner.problematic(rawtext, rawtext, message)], [message]
    return [nodes.Text(data[name][end])], []


def _test_role(_name, rawtext, text, lineno, inliner, *_args):
    """``{gfem-test}`tests/unit/test_x.py```: a link to a test on GitHub.

    The argument is a test file, a test node (``path::name``, where ``name``
    is a function or class the file defines) or a directory, relative to the
    repository root. The link text is the argument; the target is
    ``gfem_capabilities_source_url`` plus the path. A target that does not
    exist is an error, so a renamed or deleted test fails the strict build.
    """
    from docutils import nodes

    path, _, node = text.strip().partition("::")
    problem = _test_target_problem(path, node)
    if problem:
        message = inliner.reporter.error(
            f"gfem-test target {text!r}: {problem}", line=lineno
        )
        return [inliner.problematic(rawtext, rawtext, message)], [message]
    literal = nodes.literal(text.strip(), text.strip())
    base = inliner.document.settings.env.config.gfem_capabilities_source_url
    if not base:
        return [literal], []
    ref = nodes.reference(rawtext, "", refuri=f"{base}{path}", internal=False)
    ref += literal
    return [ref], []


def _test_target_problem(path: str, node: str) -> str:
    """Why ``path`` (and ``node``) is no test of the repository, or ``""``."""
    target = (REPO / path).resolve()
    if not path or not target.is_relative_to(REPO / "tests"):
        return "not a path under tests/"
    if not target.exists():
        return "no such file or directory"
    if not node:
        return ""
    # A parametrize id after "[" is not part of the defined name.
    name = node.split("[", 1)[0].rsplit("::", maxsplit=1)[-1]
    pattern = rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b"
    if target.is_file() and re.search(
        pattern, target.read_text(encoding="utf-8"), re.MULTILINE
    ):
        return ""
    return f"no function or class {name!r} in that file"


def _directives() -> dict:
    """The directives (defined here so the module imports without Sphinx)."""
    from docutils import nodes
    from sphinx import addnodes
    from sphinx.util.docutils import SphinxDirective

    class _Claims(SphinxDirective):
        has_content = False

        @property
        def data(self) -> dict:
            return self.env.gfem_capabilities

        def source(self, file: str, *children):
            """``children`` linked to ``file`` once a source URL is set."""
            base = self.config.gfem_capabilities_source_url
            if not base:
                return list(children)
            ref = nodes.reference("", "", refuri=f"{base}{file}", internal=False)
            ref.extend(children)
            return [ref]

        def ref(self, label: str, *children):
            """``children`` linked to the section labelled ``label``."""
            xref = addnodes.pending_xref(
                "",
                refdomain="std",
                reftype="ref",
                reftarget=label,
                refdoc=self.env.docname,
                refexplicit=True,
                refwarn=True,
            )
            xref.extend(children)
            return xref

        def doc(self, docname: str, *children):
            """``children`` linked to the page ``docname``."""
            if not docname:
                return list(children)
            xref = addnodes.pending_xref(
                "",
                refdomain="std",
                reftype="doc",
                reftarget=f"/{docname}",
                refdoc=self.env.docname,
                refexplicit=True,
                refwarn=True,
            )
            xref.extend(children)
            return [xref]

    class GradientMatrix(_Claims):
        """The full gradient matrix (its provenance is `GradientChecks`)."""

        def run(self):
            data = self.data
            table = nodes.table(classes=["gfem-matrix"])
            group = nodes.tgroup(cols=1 + len(data["columns"]))
            table += group
            group += nodes.colspec(colwidth=24, stub=1)
            group.extend(nodes.colspec(colwidth=19) for _ in data["columns"])
            head = nodes.row()
            head += _entry(nodes.Text("Problem"))
            for column in data["columns"]:
                head += _entry(*self.doc(column["recipe"], nodes.Text(column["label"])))
            group += nodes.thead("", head)
            body = nodes.tbody()
            for row in data["rows"]:
                line = nodes.row()
                line += _entry(
                    nodes.strong(
                        "", "", *self.doc(row["recipe"], nodes.Text(row["label"]))
                    ),
                    nodes.Text(" "),
                    nodes.inline(text=row["detail"], classes=["gfem-detail"]),
                )
                for cell in row["cells"]:
                    line += _entry(*self.cell(cell))
                body += line
            group += body
            if not data["also"]:
                return [table]
            also = "; ".join(entry["text"] for entry in data["also"])
            text = f"Also FD-verified, with no column above: {also}."
            return [table, nodes.paragraph(text=text)]

        def cell(self, cell: dict) -> list:
            mark = nodes.inline(classes=["gfem-mark", f"gfem-{cell['status']}"])
            mark += nodes.inline(classes=["gfem-glyph"])
            mark.extend(self.source(cell["source"], nodes.Text(cell["label"])))
            out = [mark]
            if cell["status"] == "verified":
                parts = [cell["forms_text"]] if cell["forms_text"] else []
                parts.append(", ".join(cell["solvers"]))
                if cell["sweeps"]:
                    parts.append(f"{' and '.join(cell['sweeps'])} sweep")
                if cell["ports"]:
                    parts.append(f"{' and '.join(cell['ports'])} ports")
                if cell["faces"]:
                    parts.append("at wave port faces too")
                out += [
                    nodes.Text(" "),
                    nodes.inline(text=" · ".join(parts), classes=["gfem-detail"]),
                ]
            return out

    class GradientChecks(_Claims):
        """How many cells the matrix checks, and how: one generated sentence."""

        def run(self):
            text = f"{self.data['provenance']}."
            return [nodes.paragraph(text=text[0].upper() + text[1:])]

    class Raises(_Claims):
        """The out-of-scope pairs off the grid, each with its named error."""

        def run(self):
            items = nodes.bullet_list(classes=["gfem-status"])
            for row in self.data["raises"]:
                tag = nodes.inline(classes=["tag", "raises"])
                tag += nodes.Text("raises ")
                tag += nodes.literal(text=row["error"])
                if row["when"]:
                    tag.extend(_with_code(f" {row['when']}"))
                para = nodes.paragraph()
                what = nodes.inline(classes=["what"])
                what.extend(_with_code(row["label"]))
                para += what
                para += nodes.Text(" ")
                # one flex item: the tag and its catch link wrap together
                status = nodes.inline(classes=["status"])
                status.extend(self.source(RAISE_TEST, tag))
                if links_gradient_errors(row):  # outside the source link: no nesting
                    status += nodes.Text(" (")
                    status += self.ref(GRADIENT_ERRORS, nodes.Text("how to catch it"))
                    status += nodes.Text(")")
                para += status
                items += nodes.list_item("", para)
            return [items]

    class DistributedRefusals(_Claims):
        """What raises above one rank, with its ``feature`` and gfem's message."""

        def run(self):
            table = nodes.table()
            group = nodes.tgroup(cols=3)
            table += group
            group += nodes.colspec(colwidth=30)
            group += nodes.colspec(colwidth=15)
            group += nodes.colspec(colwidth=55)
            head = nodes.row()
            for title in ("You ask for", "feature", "The error says"):
                head += _entry(nodes.Text(title))
            group += nodes.thead("", head)
            body = nodes.tbody()
            for row in self.data["refusals"]:
                line = nodes.row()
                line += _entry(*_with_code(row["label"]))
                line += _entry(nodes.literal(text=row["feature"]))
                line += _entry(nodes.Text(row["message"]))
                body += line
            group += body
            return [table]

    class EffectivityTable(_Claims):
        """Each estimate kind's quoted effectivity range, one row per kind."""

        def run(self):
            table = nodes.table()
            group = nodes.tgroup(cols=4)
            table += group
            for width in (45, 15, 15, 25):
                group += nodes.colspec(colwidth=width)
            head = nodes.row()
            for title in ("Estimate of", "Lowest", "Highest", "Cases"):
                head += _entry(nodes.Text(title))
            group += nodes.thead("", head)
            body = nodes.tbody()
            for row in self.data["effectivity_table"]:
                line = nodes.row()
                line += _entry(
                    nodes.Text(f"{row['words']} ("),
                    nodes.literal(text=row["kind"]),
                    nodes.Text(")"),
                )
                line += _entry(nodes.Text(row["low"]))
                line += _entry(nodes.Text(row["high"]))
                line += _entry(nodes.Text(str(row["cases"])))
                body += line
            group += body
            return [table]

    def _with_code(text: str) -> list:
        """``text``'s backticked parts as literals."""
        return [
            nodes.literal(text=part) if i % 2 else nodes.Text(part)
            for i, part in enumerate(text.split("`"))
            if part
        ]

    def _entry(*children):
        return nodes.entry("", nodes.paragraph("", "", *children))

    return {
        "gfem-gradient-matrix": GradientMatrix,
        "gfem-gradient-checks": GradientChecks,
        "gfem-raises": Raises,
        "gfem-distributed-refusals": DistributedRefusals,
        "gfem-effectivity-table": EffectivityTable,
    }
