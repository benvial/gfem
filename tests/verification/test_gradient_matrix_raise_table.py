"""The raise table: every result leaf x every model / problem leaf.

Each (problem, result leaf, input leaf) cell is one of three outcomes:
a finite, nonzero gradient; an exact zero, because the problem's physics
never reads that input (or the leaf is a constant of it, like a port's
reference impedance); or `gfem.NonDifferentiableError` at trace time.
Never a silent zero for an input the physics reads. The cells are
generated from the pytrees, so a new leaf without an expected outcome
fails here.

Then the pairs outside v0.1's differentiable set that are not cells of a
result leaf by an input leaf (`OUT_OF_SCOPE`): each raises a named error,
at trace time where the cause is static, on the host in the backward pass
where it depends on the data. Together with the gradient verification
matrix (tests/verification/test_gradient_matrix.py) this is the gate for
v0.1's gradients (``pytest -m verification -k gradient_matrix``).
"""

import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import line_port_cases as lpc
import numpy as np
import periodic_cases as pc
import piezo_cases as pz
import pml_cases
import pytest
import structured_meshes as sm
import thermal_cases as tc
from gradient_cases import (
    HYBRID_PORTS,
    MIRRORED_TARGET,
    WAVE_A_SPLIT,
    WAVE_LUMPED,
    hybrid_line,
    hybrid_mesh,
    mirrored_cavity,
    split_cavity,
    wave_guide,
    wave_resonator,
)

import gfem
from gfem._adjoint import solve_vjp
from gfem._adjoint.solve_vjp import input_kind
from gfem._engine.electrostatic import ElectrostaticRule
from gfem.shape import boundary_morph

DATA = Path(__file__).parents[1] / "unit" / "data"
GRADIENT, ZERO, RAISES = "gradient", "zero", "raises"

ALL_PROPERTIES = {
    "eps_r": 3.9 * (1 - 0.01j),
    "mu_r": 1.0,
    "tan_delta": 0.01,
    "conductivity": 2.0,
    "thermal_conductivity": 1.4,
    "mass_density": 2200.0,
    "specific_heat": 700.0,
    "thermo_optic_coefficient": 1.0e-5,
    # a solid's constants, unread by every problem below
    "stiffness": gfem.isotropic_stiffness(72e9, 0.17),
    "piezoelectric_tensor": np.arange(1.0, 19.0).reshape(3, 6) / 10,
    "mechanical_quality_factor": 500.0,
}


def _ramped(values, ramp):
    """``values`` per element, each scaled by ``ramp`` (tensors as a whole)."""
    return {k: np.multiply.outer(ramp, v) for k, v in values.items()}


# (problem name, result leaf, input kind) -> outcome. Input kinds name the
# model leaf: ("nodes",) or ("material", <region>, <property>).
EXPECTED = {
    ("electrostatic", "capacitance", ("nodes",)): GRADIENT,
    ("electrostatic", "capacitance", ("material", "oxide", "eps_r")): GRADIENT,
    ("electrostatic", "capacitance", ("material", "oxide", "mu_r")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "tan_delta")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "conductivity")): ZERO,
    (
        "electrostatic",
        "capacitance",
        ("material", "oxide", "thermal_conductivity"),
    ): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "mass_density")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "specific_heat")): ZERO,
    (
        "electrostatic",
        "capacitance",
        ("material", "oxide", "thermo_optic_coefficient"),
    ): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "stiffness")): ZERO,
    (
        "electrostatic",
        "capacitance",
        ("material", "oxide", "piezoelectric_tensor"),
    ): ZERO,
    (
        "electrostatic",
        "capacitance",
        ("material", "oxide", "mechanical_quality_factor"),
    ): ZERO,
}
# Terminal potentials: their coefficients differentiate like the capacitance;
# their mesh is the input mesh passed through (identity in the nodes).
EXPECTED |= {
    ("electrostatic", f"potential.{t}.{leaf}", kind): (
        EXPECTED[("electrostatic", "capacitance", kind)]
        if leaf == "coefficients" or kind == ("nodes",)
        else ZERO
    )
    for t in ("top",)
    for leaf in ("coefficients", "mesh.nodes")
    for kind in [("nodes",)] + [("material", "oxide", p) for p in ALL_PROPERTIES]
}
# Region energies: each differentiates with respect to what the solve reads
# (the nitride's energy moves with the oxide's permittivity through the
# potential), and is an exact zero for the inputs it never reads.
EXPECTED |= {
    ("electrostatic", f"energy.{region}", kind): (
        ZERO if kind[0] == "material" and kind[2] != "eps_r" else GRADIENT
    )
    for region in ("oxide", "nitride")
    for kind in [("nodes",)] + [("material", "oxide", p) for p in ALL_PROPERTIES]
}
EM_PROPERTIES = ("eps_r", "mu_r", "tan_delta", "conductivity")
# The port mode problem: gamma, Z_PV (with a voltage path) and the field
# differentiate with respect to the material, law, fill and Impedance inputs
# its section reads, a diagnostic (residual) and the frequency derivatives of
# gamma (n_g, beta_2, beta_3) raise, constants (frequency, solved) and the
# thermal properties get exact zeros. The node coordinates of the face move
# the mode like any other input. These cases run in
# `test_port_face_raise_table`, one item per case.
PORT_MODE_DISPERSION = ("n_g", "beta_2", "beta_3")
PORT_MODE_OUTPUTS = (
    "frequency",
    "gamma",
    "Z_PV",
    "residual",
    "solved",
    *PORT_MODE_DISPERSION,
)
PORT_FACE_CASES = (
    "driven_wave_face",
    "driven_wave_shared_face",
    "driven_wave_face_sheet",
    "driven_wave_face_law",
)
PORT_FACE_NAMES = (
    "port_mode",
    "port_mode_law",
    "port_mode_sheet",
    "port_mode_lumped",
    *PORT_FACE_CASES,
)


def _port_mode_outcome(out, kind, *, path, zero=()):
    """The expected outcome of a port mode output by an input kind.

    ``path``: whether p1 has a voltage path (else Z_PV is a NaN constant
    whose cotangent is an exact zero); ``zero``: the kinds the mode never
    reads (a lumped port's own L and C).
    """
    if kind[0] == "material" and kind[2] not in EM_PROPERTIES:
        return ZERO
    if out == "residual" or out in PORT_MODE_DISPERSION:
        return RAISES
    if out in ("frequency", "solved") or kind in zero:
        return ZERO
    if out == "Z_PV" and not path:
        return ZERO
    return GRADIENT


EXPECTED |= {
    ("port_mode", out, kind): _port_mode_outcome(out, kind, path=True)
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",)] + [("material", "fill", p) for p in ALL_PROPERTIES]
}

# Driven: S and the kept field coefficients differentiate with respect to
# every electromagnetic property and the node coordinates, and are exact
# zeros for the thermal properties. z_ref (the port resistances) and the
# kept fields' angular frequency are constants; the kept fields' mesh is the
# input mesh passed through (identity in the nodes, constant otherwise).
_DRIVEN_DEPENDENT = (
    "s.values",
    *(f"e_fields.{k}.coefficients" for k in (0, 1)),
)
_NODE_DEPENDENT = (
    *_DRIVEN_DEPENDENT,
    *(f"e_fields.{k}.mesh.nodes" for k in (0, 1)),
    "e_fields.coordinates",
    "e_fields.mesh.nodes",
)
EXPECTED |= {
    ("driven", out, kind): (
        (GRADIENT if out in _NODE_DEPENDENT else ZERO)
        if kind == ("nodes",)
        else GRADIENT
        if out in _DRIVEN_DEPENDENT and kind[2] in EM_PROPERTIES
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        # kept fields (one per excited port)
        *(
            f"e_fields.{k}.{leaf}"
            for k in (0, 1)
            for leaf in ("coefficients", "mesh.nodes", "omega")
        ),
    )
    for kind in [("nodes",)] + [("material", "dielectric", p) for p in ALL_PROPERTIES]
}


# Driven on a periodic model (the C1 line, plain and Bloch): as the
# non-periodic line, region "cell".
def _in_cell(kind: tuple) -> tuple:
    return kind if kind == ("nodes",) else ("material", "cell", kind[2])


EXPECTED |= {
    (problem, out, _in_cell(kind)): outcome
    for problem in ("driven_periodic_plain", "driven_periodic_bloch")
    for (name, out, kind), outcome in list(EXPECTED.items())
    if name == "driven"
}
# A permittivity law on a second region "core": one term of each
# kind (Re eps > 0 at the table's 5 GHz), 11 leaves. S and the kept fields'
# coefficients differentiate by every law parameter; z_ref and the fields'
# omega and mesh are constants. The host field and the nodes are as on the
# plain line. ``driven_law_absorbing``: the same behind first-order absorbing
# y walls on both regions (the law's face pieces).
TABLE_LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[
        gfem.Drude(plasma_frequency=3e9, damping=0.5e9),
        gfem.Lorentz(delta_eps=1.0, resonance_frequency=20e9, damping=1e9),
        gfem.Debye(delta_eps=2.0, relaxation_frequency=5e9),
        gfem.DjordjevicSarkar(delta_eps=1.1, low_frequency=1e3, high_frequency=1e12),
    ],
)
#: `TABLE_LAW` with the pole off the heat cases' frequency: the energy is
#: undefined inside a pole's linewidth.
ENERGY_LAW = dataclasses.replace(
    TABLE_LAW,
    terms=[
        dataclasses.replace(t, resonance_frequency=30e9)
        if isinstance(t, gfem.Lorentz)
        else t
        for t in TABLE_LAW.terms
    ],
)
LAW_KINDS = [
    ("law", "core", "eps_inf"),
    *(
        ("law", "core", i, f.name)
        for i, term in enumerate(TABLE_LAW.terms)
        for f in dataclasses.fields(term)
    ),
]
EXPECTED |= {
    (problem, out, kind): (
        EXPECTED[("driven", out, kind)]
        if kind[0] != "law"
        else GRADIENT
        if out in _DRIVEN_DEPENDENT
        else ZERO
    )
    for problem in ("driven_law", "driven_law_absorbing")
    for out in sorted({out for name, out, _ in EXPECTED if name == "driven"})
    for kind in [
        ("nodes",),
        *(("material", "dielectric", p) for p in ALL_PROPERTIES),
        *LAW_KINDS,
    ]
}
# A density blend of a constant end and TABLE_LAW on "core" (the
# index scheme): S and the kept fields' coefficients differentiate by its
# density and by the mu_r and conductivity the blend interpolates; the host
# field and the nodes are as on the plain line.
EXPECTED |= {
    ("driven_blend", out, kind): (
        EXPECTED[("driven", out, kind)]
        if kind[1:2] != ("core",)
        else GRADIENT
        if out in _DRIVEN_DEPENDENT
        else ZERO
    )
    for out in sorted({out for name, out, _ in EXPECTED if name == "driven"})
    for kind in [
        ("nodes",),
        *(("material", "dielectric", p) for p in ALL_PROPERTIES),
        *(("material", "core", p) for p in ("mu_r", "conductivity", "blend")),
    ]
}
# The C1 periodic line with the law on its cell (plain periodicity): as
# the plain periodic line in the nodes, the law's leaves as above.
EXPECTED |= {
    ("driven_periodic_law", out, kind): (
        EXPECTED[("driven_periodic_plain", out, kind)]
        if kind[0] != "law"
        else GRADIENT
        if out in _DRIVEN_DEPENDENT
        else ZERO
    )
    for out in sorted({out for name, out, _ in EXPECTED if name == "driven"})
    for kind in [("nodes",), *(("law", "cell", *k[2:]) for k in LAW_KINDS)]
}
# Adaptive sweep (frozen samples, so it traces): S and the kept fields'
# reduced coordinates differentiate like the uniform sweep's S and fields.
# The reduced basis is frozen by definition of the gradient, so a cotangent
# on it raises for every input the solve reads.
ADAPTIVE = gfem.Driven(
    frequency=(4e9, 5e9, 6e9), sweep="adaptive", samples=(4e9, 6e9), keep_fields=5e9
)
EXPECTED |= {
    ("driven_adaptive", out, kind): (
        (
            RAISES
            if out == "e_fields.basis"
            else GRADIENT
            if out in _NODE_DEPENDENT
            else ZERO
        )
        if kind == ("nodes",)
        else ZERO
        if kind[2] not in EM_PROPERTIES
        else GRADIENT
        if out in ("s.values", "e_fields.coordinates")
        else RAISES
        if out == "e_fields.basis"
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        "e_fields.basis",
        "e_fields.coordinates",
        "e_fields.omega",
        "e_fields.mesh.nodes",
    )
    for kind in [("nodes",)] + [("material", "dielectric", p) for p in ALL_PROPERTIES]
}
# The law line on the adaptive sweep: the law's leaves
# differentiate S and the kept fields' reduced coordinates, like the host's
# material; the frozen basis raises; the other inputs as `driven_adaptive`.
EXPECTED |= {
    ("driven_law_adaptive", out, kind): (
        EXPECTED[("driven_adaptive", out, kind)]
        if kind[0] != "law"
        else GRADIENT
        if out in ("s.values", "e_fields.coordinates")
        else RAISES
        if out == "e_fields.basis"
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        "e_fields.basis",
        "e_fields.coordinates",
        "e_fields.omega",
        "e_fields.mesh.nodes",
    )
    for kind in [
        ("nodes",),
        *(("material", "dielectric", p) for p in ALL_PROPERTIES),
        *LAW_KINDS,
    ]
}

# Wave ports (design region "core" away from the port faces): S
# differentiates like the lumped-port line's, in the materials and the node
# coordinates; the port modes, and so z_ref (Z_PV), are constants of the
# solve: the design never touches the port faces (nor moves their nodes),
# which is the condition for differentiating at all.
WAVE_PORT_MODE_OUTPUTS = ("frequency", "gamma", "Z_PV", "residual", "solved")
EXPECTED |= {
    ("driven_wave", out, kind): (
        (GRADIENT if out == "s.values" else ZERO)
        if kind == ("nodes",)
        else GRADIENT
        if out == "s.values" and kind[2] in EM_PROPERTIES
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        *(
            f"port_modes.{p}.{leaf}"
            for p in ("p1", "p2")
            for leaf in WAVE_PORT_MODE_OUTPUTS
        ),
    )
    for kind in [("nodes",)] + [("material", "core", p) for p in ALL_PROPERTIES]
}

# A design at the wave port faces (cases `port_face_cases`, run by
# `test_port_face_raise_table`): the port modes follow the design, so S, the
# modes' gamma and, where a port has a voltage path, Z_PV and z_ref
# differentiate with respect to every electromagnetic input a port's section
# reads. The residual of a mode is a diagnostic: it raises beside such a design.
# A thermal property is not read: exact zeros. The node coordinates of the
# faces move the modes like any other input.
#: The 2-term law of the face region of `driven_wave_face_law`.
FACE_LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=40e9, damping=2e9)],
)
FACE_LAW_KINDS = [
    ("law", "dielectric", "eps_inf"),
    *(("law", "dielectric", 0, f.name) for f in dataclasses.fields(FACE_LAW.terms[0])),
]
FACE_SHEET_KINDS = [
    ("boundary", 2, "fill", "fill"),
    *(("boundary", 2, v) for v in ("resistance", "inductance", "capacitance")),
]
FACE_KINDS = {
    "driven_wave_face": [
        ("nodes",),
        *(("material", "dielectric", p) for p in ALL_PROPERTIES),
    ],
    "driven_wave_shared_face": [("nodes",), ("material", "fill", "eps_r")],
    "driven_wave_face_sheet": [("nodes",), *FACE_SHEET_KINDS],
    "driven_wave_face_law": [("nodes",), *FACE_LAW_KINDS],
}


def _face_outcome(name, out, kind):
    """The outcome of port-face case ``name``'s leaf ``out`` in input ``kind``."""
    if kind[0] == "material" and kind[2] not in EM_PROPERTIES:
        return ZERO
    leaf = out.rsplit(".", 1)[-1]
    # p1 has a voltage path on the lone-port line
    path = name in ("driven_wave_face", "driven_wave_face_law")
    if out == "s.values" or leaf == "gamma":
        return GRADIENT
    if out == "s.z_ref" or leaf == "Z_PV":
        return GRADIENT if path and not out.startswith("port_modes.p2") else ZERO
    return RAISES if leaf == "residual" else ZERO


EXPECTED |= {
    (name, out, kind): _face_outcome(name, out, kind)
    for name in PORT_FACE_CASES
    for out in (
        "s.values",
        "s.z_ref",
        *(
            f"port_modes.{p}.{leaf}"
            for p in (
                HYBRID_PORTS if name == "driven_wave_shared_face" else ("p1", "p2")
            )
            for leaf in WAVE_PORT_MODE_OUTPUTS
        ),
    )
    for kind in FACE_KINDS[name]
}

# A perfectly matched layer behind a wave port (the layer's TEM line with a
# design slab in its air): S differentiates in the slab's material and the
# node coordinates, whose rows next to the layer read its stretch at the
# moved points; the port mode and z_ref are constants. The layer's background
# carries no leaf (a constant Material), and its knobs are static.
EXPECTED |= {
    ("driven_pml", out, kind): (
        (GRADIENT if out == "s.values" else ZERO)
        if kind == ("nodes",)
        else GRADIENT
        if out == "s.values" and kind[2] in EM_PROPERTIES
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        *(f"port_modes.p1.{leaf}" for leaf in WAVE_PORT_MODE_OUTPUTS),
    )
    for kind in [("nodes",)] + [("material", "slab", p) for p in ALL_PROPERTIES]
}

# Point dipoles in the air of the layer's line (beside its wave port, every
# column kept): an electric dipole of real moment, and a magnetic one. The
# emitted power, the dipole's port waves and both columns' kept coefficients
# differentiate like S in the slab's electromagnetic properties and the fill
# of a conducting sheet on its front face and, for the
# electric dipole, in the node coordinates (its load moves with its element);
# a magnetic dipole's load has no node derivative, so every output raises in
# the nodes. The port mode, z_ref and the fields' omega are constants, and
# the thermal properties exact zeros.
#: The fill of the conducting sheet on the slab's front face.
DIPOLE_FILL = ("boundary", 1, "fill", "fill")
_DIPOLE_DEPENDENT = (
    "s.values",
    "port_waves",
    "emitted_power.d",
    *(f"e_fields.{k}.coefficients" for k in (0, 1)),
)
for _problem, _nodes in (
    ("driven_dipole", GRADIENT),
    ("driven_magnetic_dipole", RAISES),
):
    EXPECTED |= {
        (_problem, out, kind): (
            (
                _nodes
                if _nodes == RAISES
                else GRADIENT
                if out in _DIPOLE_DEPENDENT or out.endswith("mesh.nodes")
                else ZERO
            )
            if kind == ("nodes",)
            else GRADIENT
            if out in _DIPOLE_DEPENDENT
            and (kind == DIPOLE_FILL or kind[2] in EM_PROPERTIES)
            else ZERO
        )
        for out in (
            *_DIPOLE_DEPENDENT,
            "s.z_ref",
            *(f"port_modes.p1.{leaf}" for leaf in WAVE_PORT_MODE_OUTPUTS),
            *(
                f"e_fields.{k}.{leaf}"
                for k in (0, 1)
                for leaf in ("mesh.nodes", "omega")
            ),
        )
        for kind in [
            ("nodes",),
            *(("material", "slab", p) for p in ALL_PROPERTIES),
            DIPOLE_FILL,
        ]
    }
# A plane wave on a dielectric cube in an air box wrapped in a layer, no port:
# the kept field differentiates in the cube's electromagnetic properties and
# the fill of a conducting sheet on its +x face (the load reads the background,
# never the cube or its sheet); S is empty. The node coordinates raise: the
# load is the wave's interpolant on the mesh.
PLANEWAVE_FILL = ("boundary", 0, "fill", "fill")
EXPECTED |= {
    ("driven_planewave", out, kind): (
        RAISES
        if kind == ("nodes",)
        else GRADIENT
        if out == "e_fields.0.coefficients"
        and (kind == PLANEWAVE_FILL or kind[2] in EM_PROPERTIES)
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in [
        ("nodes",),
        *(("material", "scat", p) for p in ALL_PROPERTIES),
        PLANEWAVE_FILL,
    ]
}

# Floquet ports (F1's slab between Floquet ports on both faces, the design
# the slab, away from both faces; plain and at a conical Bloch wave vector):
# S differentiates like the periodic line's, in the materials and the node
# coordinates; z_ref (the orders' wave impedances) and every leaf of the
# diffraction orders are constants of the solve. The half-space's
# medium, the incidence and the Floquet surfaces' nodes raise (`OUT_OF_SCOPE`).
FLOQUET_PORTS = ("R_TE", "R_TM", "T_TE", "T_TM")
DIFFRACTION_ORDER_OUTPUTS = (
    "frequency",
    "wave_vector",
    "gamma",
    "propagating",
    "theta",
    "phi",
)
EXPECTED |= {
    (problem, out, kind): (
        (GRADIENT if out == "s.values" else ZERO)
        if kind == ("nodes",)
        else GRADIENT
        if out == "s.values" and kind[2] in EM_PROPERTIES
        else ZERO
    )
    for problem in ("driven_floquet_plain", "driven_floquet_bloch")
    for out in (
        "s.values",
        "s.z_ref",
        *(
            f"diffraction_orders.{p}.{leaf}"
            for p in FLOQUET_PORTS
            for leaf in DIFFRACTION_ORDER_OUTPUTS
        ),
    )
    for kind in [("nodes",)] + [("material", "slab", p) for p in ALL_PROPERTIES]
}

# A shared port surface (two requested modes at each end of a hybrid line):
# S, cross-mode entries included, differentiates like a lone wave port's;
# every port's mode is a constant of the solve. The core carries only a
# permittivity: each cell costs a 4-port solve, and the other properties'
# outcomes are the lone wave ports' above.
EXPECTED |= {
    ("driven_wave_shared", out, kind): GRADIENT if out == "s.values" else ZERO
    for out in (
        "s.values",
        "s.z_ref",
        *(
            f"port_modes.{p}.{leaf}"
            for p in HYBRID_PORTS
            for leaf in WAVE_PORT_MODE_OUTPUTS
        ),
    )
    for kind in [("nodes",), ("material", "core", "eps_r")]
}

# Eigenmode: the eigenvalues (omega, also as each mode field's omega)
# differentiate with respect to every electromagnetic property, the node
# coordinates and the lumped values. The ratios of the mode fields (port Q,
# participations, region, principal and sheet shares) differentiate with respect to
# the electromagnetic properties, sheet fills, node coordinates and lumped
# values, and the normalized mode fields'
# coefficients with respect to the electromagnetic properties, sheet fills
# and lumped values (never the node coordinates); the principal shares raise
# with respect to the lumped values in this version, as does the backward
# error with respect to anything. The
# cluster labels are integers: exact zeros. The mode fields' mesh is the
# input mesh passed through (identity in the nodes, constant otherwise). A
# periodic model's mode-field outputs raise whatever the input.
_EIGEN_DEPENDENT = ("omega", "e_fields.0.omega")
_EIGEN_CONSTANT = ("cluster", "e_fields.0.mesh.nodes")
_EIGEN_RATIOS = (
    "q_port.",
    "participation.",
    "bulk_participation.",
    "principal_participation.",
    "sheet_participation.",
    "e_fields.0.coefficients",
)


def _eigen_outcome(out, kind, *, periodic=False):
    if kind == ("nodes",):
        if out == "cluster":
            return ZERO
        dependent = (*_EIGEN_DEPENDENT, "e_fields.0.mesh.nodes")
        # the normalized field has no node gradient
        ratio = out.startswith(_EIGEN_RATIOS) and not periodic
        field = out == "e_fields.0.coefficients"
        return GRADIENT if out in dependent or (ratio and not field) else RAISES
    if (kind[0] == "material" and kind[2] not in EM_PROPERTIES) or (
        out in _EIGEN_CONSTANT
    ):
        return ZERO
    if out in _EIGEN_DEPENDENT:
        return GRADIENT
    ratio = out.startswith(_EIGEN_RATIOS)
    lumped = kind[0] in ("port", "boundary") and kind[2:] != ("fill", "fill")
    principal = out.startswith("principal_participation.")
    return GRADIENT if ratio and not periodic and not (lumped and principal) else RAISES


EIGEN_KINDS = [
    ("nodes",),
    *(("material", "dielectric", p) for p in ALL_PROPERTIES),
    ("port", 0, "inductance"),
    ("port", 0, "capacitance"),
]
EXPECTED |= {
    ("eigenmode", out, kind): _eigen_outcome(out, kind)
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "q_port.p2",
        "participation.p1",
        "bulk_participation.dielectric",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in EIGEN_KINDS
}
# The result carries the model's materials as solved (for measurements): each
# leaf is its input passed through (identity), a constant in the nodes.
EXPECTED |= {
    ("eigenmode", f"materials.dielectric.{out}", kind): (
        GRADIENT if kind[0] == "material" and kind[2] == out else ZERO
    )
    for out in ALL_PROPERTIES
    for kind in EIGEN_KINDS
}

# Eigenmode with a wave port (case A of the shorted guide, its eps_r = 4
# layer split by an inductive lumped port "l1"; the design the first part,
# off the port face): every result leaf as on the lumped resonator, the wave
# port's coupling Q a ratio like a resistive port's. The inputs of the port
# section (its regions' materials and laws, a sheet beside the face, the
# face's nodes) raise `gfem.PortGeometryError` (`OUT_OF_SCOPE`).
EIGEN_WAVE_KINDS = [
    ("nodes",),
    *(("material", "dielectric", p) for p in ALL_PROPERTIES),
    ("port", 1, "inductance"),
    ("port", 1, "capacitance"),
]
EXPECTED |= {
    ("eigenmode_wave", out, kind): _eigen_outcome(out, kind)
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "requested_power_share.port",
        "participation.l1",
        *(f"bulk_participation.{r}" for r in ("dielectric", "spacer", "air")),
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in EIGEN_WAVE_KINDS
}
EXPECTED |= {
    ("eigenmode_wave", f"materials.dielectric.{out}", kind): (
        GRADIENT if kind[0] == "material" and kind[2] == out else ZERO
    )
    for out in ALL_PROPERTIES
    for kind in EIGEN_WAVE_KINDS
}
EIGEN_WAVE_TARGET = 11.6e9

# Eigenmode with TABLE_LAW on a second region "core":
# omega (so frequency and Q) differentiates by every law parameter; the
# mode-field outputs and the backward error raise by them ("differentiate
# the eigenfrequency or Q"); the cluster labels and the fields' mesh are
# constants. The result's law leaves are the inputs passed through. The
# other inputs are as on the plain resonator, the core's share like the
# dielectric's.
EIGEN_LAW_KINDS = [*EIGEN_KINDS, *LAW_KINDS]
_EIGEN_LAW_LEAVES = {
    ("eps_inf",): "materials.core.eps_r.eps_inf",
    **{
        (i, f.name): f"materials.core.eps_r.terms.{i}.{f.name}"
        for i, term in enumerate(TABLE_LAW.terms)
        for f in dataclasses.fields(term)
    },
}


def _eigen_law_outcome(out, kind):
    if out.startswith("materials.core."):
        return GRADIENT if _EIGEN_LAW_LEAVES.get(tuple(kind[2:])) == out else ZERO
    if kind[0] != "law":
        if out.startswith("materials.dielectric."):
            return EXPECTED[("eigenmode", out, kind)]
        base = out.replace("bulk_participation.core", "bulk_participation.dielectric")
        return EXPECTED[("eigenmode", base, kind)]
    if out.startswith("materials.dielectric.") or out in _EIGEN_CONSTANT:
        return ZERO
    return GRADIENT if out in _EIGEN_DEPENDENT else RAISES


EXPECTED |= {
    ("eigenmode_law", out, kind): _eigen_law_outcome(out, kind)
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "q_port.p2",
        "participation.p1",
        "bulk_participation.dielectric",
        "bulk_participation.core",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
        *(f"materials.dielectric.{p}" for p in ALL_PROPERTIES),
        *_EIGEN_LAW_LEAVES.values(),
    )
    for kind in EIGEN_LAW_KINDS
}

# Eigenmode on a periodic model (the C3 Bragg line, plain at Γ and Bloch at
# the zone edge; no ports): as the non-periodic resonator, region "low".
# Its material field carries only the permittivity and the conductivity
# (the properties above are covered by "eigenmode"): each cell is an
# eigenmode solve and its gradient.
PERIODIC_EIGEN_PROPERTIES = ("eps_r", "conductivity")
EXPECTED |= {
    (problem, out, kind): (
        (
            ZERO
            if kind == ("nodes",)
            else GRADIENT
            if kind[2] == out.removeprefix("materials.low.")
            else ZERO
        )
        if out.startswith("materials.low.")
        else _eigen_outcome(out, kind, periodic=True)
    )
    for problem in ("eigenmode_periodic_plain", "eigenmode_periodic_bloch")
    for out in (
        "omega",
        "error",
        "bulk_participation.low",
        "bulk_participation.high",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
        *(f"materials.low.{p}" for p in PERIODIC_EIGEN_PROPERTIES),
    )
    for kind in [("nodes",)]
    + [("material", "low", p) for p in PERIODIC_EIGEN_PROPERTIES]
}

# The Bloch band group velocity is read from the modes, not differentiable
# by anything: every cell of its leaf raises (OUT_OF_SCOPE, "eigenmode band
# group velocity"). Only a Bloch solve has the leaf.
EXPECTED |= {
    ("eigenmode_periodic_bloch", "v_g", kind): _eigen_outcome(
        "v_g", kind, periodic=True
    )
    for kind in [("nodes",)]
    + [("material", "low", p) for p in PERIODIC_EIGEN_PROPERTIES]
}

# Eigenmode with a perfectly matched layer (the leaky guide: a dielectric
# core radiating through an air tail into a PEC-backed layer; no ports): as
# the non-periodic resonator, region "core", whose material field carries
# only the permittivity and the conductivity. The node rows next to the layer
# read its stretch at the moved points; the participations count the
# physical regions only, so the layer has none.
EXPECTED |= {
    ("eigenmode_pml", out, kind): (
        (
            ZERO
            if kind == ("nodes",)
            else GRADIENT
            if kind[2] == out.removeprefix("materials.core.")
            else ZERO
        )
        if out.startswith("materials.core.")
        else _eigen_outcome(out, kind)
    )
    for out in (
        "omega",
        "error",
        "bulk_participation.core",
        "bulk_participation.tail",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
        *(f"materials.core.{p}" for p in PERIODIC_EIGEN_PROPERTIES),
    )
    for kind in [("nodes",)]
    + [("material", "core", p) for p in PERIODIC_EIGEN_PROPERTIES]
}

# Sheet fields: the fill of a sheet on the glued line (model.boundaries[2],
# static materials). Driven S and eigenfrequencies differentiate with respect
# to it like a material; the port mode is not differentiable at all.
FILL = ("boundary", 2, "fill", "fill")
EXPECTED |= {
    ("driven_sheet", out, kind): GRADIENT if out == "s.values" else ZERO
    for out in ("s.values", "s.z_ref")
    for kind in [("nodes",), FILL]
}
EXPECTED |= {
    ("eigenmode_sheet", out, kind): _eigen_outcome(out, kind)
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "q_port.p2",
        "bulk_participation.lower",
        "bulk_participation.upper",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in [("nodes",), FILL]
}
EXPECTED |= {
    ("port_mode_sheet", out, kind): _port_mode_outcome(out, kind, path=False)
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",), FILL]
}

# Lumped values: port p1's inductance and capacitance and the
# sheet's resistance, inductance and capacitance on the glued line (static
# materials).
# Driven S, the kept fields and the eigenfrequencies differentiate with
# respect to them; z_ref (p1's resistance) and the fields' omega and mesh do
# not move with them; electrostatics never reads them; the port mode problem
# is not differentiable. Only a model with no Impedance and no lumped-port L
# or C has none of these leaves: every Impedance adds them, R-only ones too.
LUMPED_KINDS = [
    ("port", 0, "inductance"),
    ("port", 0, "capacitance"),
    ("boundary", 2, "resistance"),
    ("boundary", 2, "inductance"),
    ("boundary", 2, "capacitance"),
]
EXPECTED |= {
    ("driven_lumped", out, kind): (
        EXPECTED[("driven", out, ("nodes",))]
        if kind == ("nodes",)
        else GRADIENT
        if out in _DRIVEN_DEPENDENT
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        *(
            f"e_fields.{k}.{leaf}"
            for k in (0, 1)
            for leaf in ("coefficients", "mesh.nodes", "omega")
        ),
    )
    for kind in [("nodes",), *LUMPED_KINDS]
}
EXPECTED |= {
    ("driven_lumped_adaptive", out, kind): (
        EXPECTED[("driven_adaptive", out, ("nodes",))]
        if kind == ("nodes",)
        else GRADIENT
        if out in ("s.values", "e_fields.coordinates")
        else RAISES
        if out == "e_fields.basis"
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        "e_fields.basis",
        "e_fields.coordinates",
        "e_fields.omega",
        "e_fields.mesh.nodes",
    )
    for kind in [("nodes",), *LUMPED_KINDS]
}
EXPECTED |= {
    ("eigenmode_lumped", out, kind): _eigen_outcome(out, kind)
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "q_port.p2",
        "participation.p1",
        "bulk_participation.lower",
        "bulk_participation.upper",
        "sheet_participation.sheet",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in [("nodes",), *LUMPED_KINDS]
}
EXPECTED |= {
    ("electrostatic_lumped", out, kind): (
        EXPECTED[("electrostatic", "capacitance", ("nodes",))]
        if kind == ("nodes",)
        else ZERO
    )
    for out in (
        "capacitance",
        "energy.lower",
        "energy.upper",
        "potential.top.coefficients",
        "potential.top.mesh.nodes",
    )
    for kind in [("nodes",), *LUMPED_KINDS]
}
# the port mode's model: wave port p1, the lumped port p2 holds L and C
PORT_MODE_LUMPED_KINDS = [
    ("port", 1, "inductance"),
    ("port", 1, "capacitance"),
    ("boundary", 2, "resistance"),
    ("boundary", 2, "inductance"),
    ("boundary", 2, "capacitance"),
]
# the wave port's mode does not read the lumped port's own L and C: exact zeros
EXPECTED |= {
    ("port_mode_lumped", out, kind): _port_mode_outcome(
        out, kind, path=False, zero=PORT_MODE_LUMPED_KINDS[:2]
    )
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",), *PORT_MODE_LUMPED_KINDS]
}


# A permittivity law in a wave port's cross-section: every law parameter
# differentiates gamma and Z_PV, the node coordinates raise. One term of each
# kind, each with Re eps > 0 at 10 GHz.
PORT_MODE_LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[
        gfem.Drude(plasma_frequency=5e9, damping=1e9),
        gfem.Lorentz(delta_eps=1.0, resonance_frequency=50e9, damping=1e9),
        gfem.Debye(delta_eps=0.5, relaxation_frequency=20e9),
        gfem.DjordjevicSarkar(delta_eps=0.3, low_frequency=1e3, high_frequency=1e12),
    ],
)
PORT_MODE_LAW_KINDS = [
    ("law", "fill", "eps_inf"),
    ("law", "fill", 0, "plasma_frequency"),
    ("law", "fill", 0, "damping"),
    ("law", "fill", 1, "delta_eps"),
    ("law", "fill", 1, "resonance_frequency"),
    ("law", "fill", 1, "damping"),
    ("law", "fill", 2, "delta_eps"),
    ("law", "fill", 2, "relaxation_frequency"),
    ("law", "fill", 3, "delta_eps"),
    ("law", "fill", 3, "low_frequency"),
    ("law", "fill", 3, "high_frequency"),
]
EXPECTED |= {
    ("port_mode_law", out, kind): _port_mode_outcome(out, kind, path=True)
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",), *PORT_MODE_LAW_KINDS]
}

# The stack with a permittivity law on its nitride: the table's
# law without its Drude term (a Drude metal has no static permittivity). The
# capacitance, the energies and the potentials differentiate by eps_inf and
# every strength, which the static permittivity reads; every frequency
# parameter is an exact zero (a leaf the physics never reads has a zero
# gradient, not an error). The oxide field and the nodes are as on the plain
# stack.
ELECTROSTATIC_LAW = dataclasses.replace(TABLE_LAW, terms=TABLE_LAW.terms[1:])
ELECTROSTATIC_LAW_KINDS = [
    ("law", "nitride", "eps_inf"),
    *(
        ("law", "nitride", i, f.name)
        for i, term in enumerate(ELECTROSTATIC_LAW.terms)
        for f in dataclasses.fields(term)
    ),
]
EXPECTED |= {
    ("electrostatic_law", out, kind): (
        EXPECTED[("electrostatic", out, kind)]
        if kind[0] != "law"
        else GRADIENT
        if kind[-1] in ("eps_inf", "delta_eps") and not out.endswith("mesh.nodes")
        else ZERO
    )
    for out in sorted({out for name, out, _ in EXPECTED if name == "electrostatic"})
    for kind in [
        ("nodes",),
        *(("material", "oxide", p) for p in ALL_PROPERTIES),
        *ELECTROSTATIC_LAW_KINDS,
    ]
}


# Thermal and DC conduction: each output differentiates with
# respect to what its solve reads (the thermal properties it reads, the heat
# source amount, the waveform scales; the conductivity for DC conduction)
# and is an exact zero for every other material property and the sheet
# fill; the node coordinates raise (the forms have no shape terms), for
# every output, since the solve itself is traced.
#: The thermal boundary values of the thermal and DC cases (a fixed
#: temperature, a convection's coefficient and ambient, a heat flux). Steady
#: thermal reads them (the temperature reads all four, the rise only the
#: coefficient); DC conduction does not read them (exact zeros); transient
#: thermal reads them as steady does (the rises only the coefficient, the
#: source-free temperature all four).
THERMAL_VALUE_KINDS = [
    ("boundary", 1, "temperature"),
    ("boundary", 2, "coefficient"),
    ("boundary", 2, "ambient"),
    ("boundary", 3, "flux"),
]
_T_FIXED, _H, _T_AMBIENT, _FLUX = THERMAL_VALUE_KINDS
THERMAL_KINDS = [
    ("nodes",),
    *(("material", "lower", p) for p in ALL_PROPERTIES),
    ("boundary", 0, "fill", "fill"),
    *THERMAL_VALUE_KINDS,
]
_K = ("material", "lower", "thermal_conductivity")
_RHO_CP = (
    ("material", "lower", "mass_density"),
    ("material", "lower", "specific_heat"),
)
AMOUNT = ("problem", ".heat[0]._amount")
SCALES = ("problem", "._scales")
#: Per problem: each result leaf's inputs read (GRADIENT), the rest ZERO.
THERMAL_READS = {
    "steady_thermal": {
        "temperature.coefficients": {_K, AMOUNT, _T_FIXED, _H, _T_AMBIENT, _FLUX},
        "temperature.mesh.nodes": set(),
        "rise.coefficients": {_K, AMOUNT, _H},
        "rise.mesh.nodes": set(),
        "heat.coefficients": {AMOUNT},
        "heat.mesh.nodes": set(),
        "power": {AMOUNT},
    },
    "transient_thermal": {
        "time": set(),
        "traces.upper": {_K, *_RHO_CP, AMOUNT, SCALES, _T_FIXED, _H, _T_AMBIENT, _FLUX},
        "_rises": {_K, *_RHO_CP, AMOUNT, SCALES, _H},
        "_source_free.coefficients": {_K, _T_FIXED, _H, _T_AMBIENT, _FLUX},
        "_source_free.mesh.nodes": set(),
        "power": {AMOUNT},
    },
    "dc_conduction": {
        "conductance": {("material", "lower", "conductivity")},
        "potential.t.coefficients": {("material", "lower", "conductivity")},
        "potential.t.mesh.nodes": set(),
        "dissipation": {("material", "lower", "conductivity")},
    },
}
_PROBLEM_KINDS = {
    "steady_thermal": (AMOUNT,),
    "transient_thermal": (AMOUNT, SCALES),
    "dc_conduction": (),
}
EXPECTED |= {
    (name, out, kind): (
        RAISES
        if kind == ("nodes",) and not name.startswith("electric_energy")
        else GRADIENT
        if kind in reads or kind == ("nodes",)
        else ZERO
    )
    for name, leaves in THERMAL_READS.items()
    for out, reads in leaves.items()
    for kind in [*THERMAL_KINDS, *_PROBLEM_KINDS[name]]
}

# The piezoelectric problem on the film's field: the admittance, the kept
# displacement and potential read the nodes and five properties (the clamped
# capacitance only the permittivity), a kept field's mesh passes the nodes
# through, and its frequency is a constant. Every other property is unread.
PIEZO_KINDS = [("nodes",), *(("material", "ln", p) for p in ALL_PROPERTIES)]
_PZ_MECHANICS = {
    ("material", "ln", p)
    for p in (
        "eps_r",
        "stiffness",
        "piezoelectric_tensor",
        "mass_density",
        "mechanical_quality_factor",
    )
}
_PZ_FIELDS = {("nodes",), *_PZ_MECHANICS}
PIEZO_READS = {
    "admittance_per_length": _PZ_FIELDS,
    "clamped_capacitance_per_length": {("nodes",), ("material", "ln", "eps_r")},
    **{
        f"displacement_fields.0.{axis}.{leaf}": reads
        for axis in "xyz"
        for leaf, reads in (
            ("coefficients", _PZ_FIELDS),
            ("mesh.nodes", {("nodes",)}),
            ("omega", set()),
        )
    },
    "potential_fields.0.coefficients": _PZ_FIELDS,
    "potential_fields.0.mesh.nodes": {("nodes",)},
    "potential_fields.0.omega": set(),
}
EXPECTED |= {
    ("piezoelectric", out, kind): GRADIENT if kind in reads else ZERO
    for out, reads in PIEZO_READS.items()
    for kind in PIEZO_KINDS
}


# The thermo-optic shift and the route phase: measurement
# rows, as the power flow's are, in the optical model's leaves and the
# rise's coefficients (input kind ("rise",)). The mode is a constant of the
# measurement: every input its port mode solve reads raises (the
# electromagnetic properties, the node coordinates, a boundary's sheet fill),
# for every leaf; the thermal properties are exact zeros. The thermo-optic
# coefficient moves the phase and the trace's weights; the rise moves the
# phase only; n_eff moves with neither.
OPTICAL_EM = ("eps_r", "mu_r", "tan_delta", "conductivity")
OPTICAL_KINDS = [
    ("nodes",),
    *(("material", "core", p) for p in ALL_PROPERTIES),
    ("boundary", 0, "fill", "fill"),
    ("rise",),
]
_DN = ("material", "core", "thermo_optic_coefficient")
MEASUREMENT_READS = {
    "thermo_optic_shift": {
        "n_eff": set(),
        "delta_n_eff": {_DN, ("rise",)},
        "phase_per_length": {_DN, ("rise",)},
        "_trace._weights": {_DN},
    },
    "route_phase": {
        "n_eff": set(),
        "phase.r": {_DN, ("rise",)},
        "phase_per_length.r": {_DN, ("rise",)},
        "_traces.r._weights": {_DN},
        # the stations' arc lengths and the route's length: constants
        "arc_length.r": set(),
        "length.r": set(),
    },
}


def _optical_outcome(kind, reads):
    if kind[0] in ("nodes", "boundary") or (
        kind[0] == "material" and kind[2] in OPTICAL_EM
    ):
        return RAISES
    return GRADIENT if kind in reads else ZERO


EXPECTED |= {
    (name, out, kind): _optical_outcome(kind, reads)
    for name, leaves in MEASUREMENT_READS.items()
    for out, reads in leaves.items()
    for kind in OPTICAL_KINDS
}


# Electromagnetic heat: the dissipated power per region and sheet of a driven
# field, and the steady temperature the heat of a two-port drive raises, as
# functions of the model (through the driven solve) and of the drive
# (`heat_cases`). Each differentiates with respect to every electromagnetic
# input the solve reads (the permeability and the `Impedance` inductance and
# capacitance through the field), the sheet's fill, and the incident powers
# and phase; a lossless region's power is an exact zero for every input; the
# thermal properties are exact zeros of the heat, and the temperature reads
# the thermal conductivity. The node coordinates raise for every leaf (the
# heat kernel has no shape terms).
HEAT_KINDS = [
    ("nodes",),
    *(("material", "lower", p) for p in ALL_PROPERTIES),
    *(("boundary", 2, v) for v in ("resistance", "inductance", "capacitance")),
    ("boundary", 2, "fill", "fill"),
    ("boundary", 3, "temperature"),
]
#: The drive's leaves: the incident powers and p2's phase.
DRIVE_KINDS = [
    ("drive", "power", "p1"),
    ("drive", "power", "p2"),
    ("drive", "phase", "p2"),
]
#: The heat sink's fixed temperature: only the temperature reads it (the
#: electromagnetic solve and the rise do not).
_SINK = HEAT_KINDS[-1]
_EM_READS = {
    *(("material", "lower", p) for p in EM_PROPERTIES),
    *HEAT_KINDS[-5:-1],
}
_HEATING_READS = _EM_READS | set(DRIVE_KINDS)
#: ``dissipated_power_law``: the upper half a permittivity law (`TABLE_LAW`).
HEAT_LAW_KINDS = [("law", "upper", *k[2:]) for k in LAW_KINDS]
_LAW_READS = _EM_READS | set(HEAT_LAW_KINDS)
#: ``dissipated_power_blend``: the upper half an index blend of a constant end
#: and `TABLE_LAW`; its density moves the loss.
HEAT_BLEND_KINDS = [("material", "upper", p) for p in ("mu_r", "conductivity", "blend")]
_BLEND_READS = _EM_READS | set(HEAT_BLEND_KINDS)
HEAT_READS = {
    "dissipated_power": {"lower": _EM_READS, "upper": set(), "sheet": _EM_READS},
    "dissipated_power_law": dict.fromkeys(("lower", "upper", "sheet"), _LAW_READS),
    "dissipated_power_blend": dict.fromkeys(("lower", "upper", "sheet"), _BLEND_READS),
    # the energy of every region reads the solve's inputs through the field
    # and its own permittivity (a law's weight in JAX), and has node terms
    "electric_energy": dict.fromkeys(("lower", "upper"), _EM_READS),
    "electric_energy_law": dict.fromkeys(("lower", "upper"), _LAW_READS),
    "electromagnetic_heating": {
        "temperature.coefficients": _HEATING_READS | {_K, _SINK},
        "temperature.mesh.nodes": set(),
        "rise.coefficients": _HEATING_READS | {_K},
        "rise.mesh.nodes": set(),
        "heat.coefficients": _HEATING_READS,
        "heat.mesh.nodes": set(),
        "power": _HEATING_READS,
        "_face_heat": _HEATING_READS,
    },
}
EXPECTED |= {
    (name, out, kind): (
        RAISES
        if kind == ("nodes",) and not name.startswith("electric_energy")
        else GRADIENT
        if kind in reads or kind == ("nodes",)
        else ZERO
    )
    for name, leaves in HEAT_READS.items()
    for out, reads in leaves.items()
    for kind in [
        *HEAT_KINDS,
        *(DRIVE_KINDS if name == "electromagnetic_heating" else ()),
        *(
            HEAT_LAW_KINDS
            if name in ("dissipated_power_law", "electric_energy_law")
            else ()
        ),
        *(HEAT_BLEND_KINDS if name == "dissipated_power_blend" else ()),
    ]
}


def _field(mesh, region, **overrides):
    # values vary per element, so the terminal potentials depend on them
    n = len(mesh.elements(region))
    ramp = np.linspace(1.0, 1.5, n)
    values = ALL_PROPERTIES | overrides
    return gfem.MaterialField(**_ramped(values, ramp))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    layers = (("oxide", 0.5), ("nitride", 0.25))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    return gfem.Model(
        mesh,
        materials={"oxide": _field(mesh, "oxide"), "nitride": gfem.Material(eps_r=7.5)},
        boundaries=[gfem.PEC("bottom")],
    )


@pytest.fixture(scope="module")
def driven_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (2, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": _field(mesh, "dielectric")},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def law_cases(tmp_path_factory):
    path = tmp_path_factory.mktemp("law_line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 1, 1), core=(1.0, 3.0)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={
            "dielectric": _field(mesh, "dielectric"),
            "core": gfem.Material(eps_r=TABLE_LAW),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    absorbing = model.replace(
        boundaries=[*model.boundaries, gfem.Absorbing("side", order=1)]
    )
    problem = gfem.Driven(frequency=5e9, keep_fields=True)
    n = len(mesh.elements("core"))
    blend = gfem.MaterialField.interpolate(
        jnp.linspace(0.3, 0.7, n),
        gfem.Material(eps_r=2.0),
        gfem.Material(eps_r=TABLE_LAW),
        scheme="index",
    )
    return {
        "driven_law": (model, problem),
        "driven_law_absorbing": (absorbing, problem),
        "driven_law_adaptive": (model, ADAPTIVE),
        "driven_blend": (model.with_material("core", blend), problem),
    }


@pytest.fixture(scope="module")
def eigen_model(tmp_path_factory):
    # weakly damped (resistive ports, little conduction): the modes are sharp
    path = tmp_path_factory.mktemp("resonator") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    # p1 also holds an inductance and a capacitance: a participation, two
    # lumped values, and a share of the mode energy outside the dielectric
    # (with the dielectric's alone, its bulk participation is exactly 1)
    return gfem.Model(
        mesh,
        materials={"dielectric": _field(mesh, "dielectric", conductivity=1e-3)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort(
                "p1",
                "port1",
                "+z",
                resistance=1000.0,
                inductance=1e-9,
                capacitance=1e-14,
            ),
            gfem.LumpedPort("p2", "port2", "+z", impedance=1000.0),
        ],
    )


def _eigen_wave_model(mesh, dielectric):
    ports = [
        gfem.WavePort("p1", "port"),
        gfem.LumpedPort(
            "l1",
            "lumped",
            "+y",
            inductance=WAVE_LUMPED[0],
            capacitance=WAVE_LUMPED[1],
        ),
    ]
    return wave_resonator(mesh, WAVE_A_SPLIT, {"dielectric": dielectric}, ports)


@pytest.fixture(scope="module")
def eigen_wave_model(tmp_path_factory):
    # little conduction: the mode stays sharp
    mesh = wave_guide(
        tmp_path_factory.mktemp("wave_guide"), WAVE_A_SPLIT, lumped=6.0, nx=4
    )
    return _eigen_wave_model(mesh, _field(mesh, "dielectric", conductivity=1e-3))


@pytest.fixture(scope="module")
def eigen_law_model(tmp_path_factory):
    # the eigenmode resonator with TABLE_LAW on a 2 mm "core": its mode near
    # 18.6 GHz sees eps about 9.1 - 3.0i (below the unresolved band)
    path = tmp_path_factory.mktemp("law_resonator") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 1, 1), core=(1.0, 3.0)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={
            "dielectric": _field(mesh, "dielectric", conductivity=1e-3),
            "core": gfem.Material(eps_r=TABLE_LAW),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort(
                "p1",
                "port1",
                "+z",
                resistance=1000.0,
                inductance=1e-9,
                capacitance=1e-14,
            ),
            gfem.LumpedPort("p2", "port2", "+z", impedance=1000.0),
        ],
    )


@pytest.fixture(scope="module")
def wave_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("wave") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 1, 1), core=(1.0, 2.0)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=2.0),
            "core": _field(mesh, "core"),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


@pytest.fixture(scope="module")
def pml_model(tmp_path_factory):
    path = pml_cases.slab_line(tmp_path_factory.mktemp("pml") / "line.msh", cells=4)
    mesh = gfem.Mesh.from_gmsh(path)
    return gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "pml": gfem.Material(),
            "slab": _field(mesh, "slab"),
        },
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )


def _cube(c):
    if ((c > 0.25) & (c < 0.75)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


def _cube_sheet(c):
    """The cube scatterer's +x face: an interior sheet."""
    inside = 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75
    return "sheet" if abs(c[0] - 0.75) < 1e-9 and inside else ""


def _source_sheet(mesh, surface):
    """A conducting sheet of per-face fill: the source rows' fill input."""
    fill = gfem.SheetField(fill=np.linspace(0.2, 0.9, len(mesh.elements(surface))))
    return gfem.Conductor(surface, 100.0, 1e-6, fill=fill)


@pytest.fixture(scope="module")
def planewave_case(tmp_path_factory):
    sides = ("-x", "+x", "-y", "+y", "-z", "+z")
    path = pml_cases.box(
        tmp_path_factory.mktemp("planewave") / "box.msh",
        sides,
        thickness=0.25,
        cells=4,
        volume=_cube,
        surface=_cube_sheet,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "pml": gfem.Material(),
            "scat": _field(mesh, "scat"),
        },
        boundaries=[_source_sheet(mesh, "sheet")],
        pml=gfem.PML("pml"),
        background=gfem.Material(),
    )
    wave = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization="TM")
    return model, gfem.Driven(frequency=pml_cases.F0, sources=wave, keep_fields=True)


#: In the line's air, off the faces of its grid (um).
DIPOLE_POSITION = np.array([0.07, 0.13, 0.3]) * 1e-6


@pytest.fixture(scope="module")
def dipole_model(pml_model):
    # the layer's line with a conducting sheet on the slab's front face
    sheet = _source_sheet(pml_model.mesh, "front")
    return pml_model.replace(boundaries=[*pml_model.boundaries, sheet])


def _dipole_cases(model):
    electric = gfem.ElectricDipole("d", position=DIPOLE_POSITION, moment=(0, 1e-28, 0))
    magnetic = gfem.MagneticDipole("d", position=DIPOLE_POSITION, moment=(1e-20, 0, 0))
    return {
        name: (model, gfem.Driven(pml_cases.F0, keep_fields=True, sources=dipole))
        for name, dipole in (
            ("driven_dipole", electric),
            ("driven_magnetic_dipole", magnetic),
        )
    }


@pytest.fixture(scope="module")
def eigen_pml_case(tmp_path_factory):
    # the leaky guide, weakly lossy, its core a material field (a gentle ramp,
    # so the mode nearest the target stays the leaky one, Q about 10)
    model, _ = pml_cases.leaky_guide_model(
        tmp_path_factory.mktemp("leaky") / "guide.msh"
    )
    ramp = np.linspace(1.0, 1.05, len(model.mesh.elements("core")))
    core = gfem.MaterialField(
        eps_r=pml_cases.GUIDE_EPS * (1 - 0.01j) * ramp, conductivity=1e-3 * ramp
    )
    target = pml_cases.guide_root().real * pml_cases.C0 * 1e3 / (2 * math.pi)
    return model.with_material("core", core), gfem.Eigenmode(target=target)


@pytest.fixture(scope="module")
def shared_surface_model(tmp_path_factory):
    # the gradient matrix's hybrid line on the coarsest grid that keeps its
    # quadrant: modes 1 and 2 of each lead on one shared port surface
    mesh = hybrid_mesh(tmp_path_factory.mktemp("hybrid"), cells=(4, 2, 2))
    eps = 3.0 * np.linspace(1.0, 1.5, len(mesh.elements("core")))
    return hybrid_line(mesh).with_material("core", gfem.MaterialField(eps_r=eps))


@pytest.fixture(scope="module")
def sheet_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("sheet") / "sheet.msh"
    sm.write_msh22(path, sm.sheet_line(4.0, 2.0, 1.0, (4, 1, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _sheet_model(mesh, ports):
    n = len(mesh.elements("sheet"))
    fill = gfem.SheetField(fill=np.linspace(0.2, 0.9, n))
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(eps_r=2.0), "upper": gfem.Material()},
        boundaries=[
            gfem.PEC("plate1"),
            gfem.PEC("plate2"),
            gfem.Conductor("sheet", 100.0, 1e-6, fill=fill),
        ],
        ports=ports,
    )


@pytest.fixture(scope="module")
def sheet_cases(sheet_mesh):
    lumped = [
        gfem.LumpedPort("p1", "port1", "+y", impedance=1000.0),
        gfem.LumpedPort("p2", "port2", "+y", impedance=1000.0),
    ]
    waves = [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")]
    return {
        "driven_sheet": (
            _sheet_model(sheet_mesh, lumped),
            gfem.Driven(frequency=20e9),
        ),
        "eigenmode_sheet": (
            _sheet_model(sheet_mesh, lumped),
            gfem.Eigenmode(target=30e9),
        ),
        "port_mode_sheet": (
            _sheet_model(sheet_mesh, waves),
            gfem.PortMode("p1", 20e9),
        ),
    }


#: The lumped values of the lumped cases: port inductance (H) and capacitance
#: (F), sheet resistance (ohm), inductance (H) and capacitance (F) per square.
LUMPED = (2e-9, 5e-14, 2e4, 1e-6, 2e-16)
#: Near the lumped line's sharpest mode, 15.07 + 1.46j GHz (the next ones,
#: 11.7 + 4.2j and 19.8 + 4.8j GHz, are 4.4 and 5.8 GHz off).
LUMPED_TARGET = 15e9
LUMPED_ADAPTIVE = gfem.Driven(
    frequency=(18e9, 20e9, 22e9),
    sweep="adaptive",
    samples=(18e9, 22e9),
    keep_fields=20e9,
)


def _lumped_model(mesh, ports, lower=None):
    sheet = gfem.Impedance(
        "sheet", resistance=LUMPED[2], inductance=LUMPED[3], capacitance=LUMPED[4]
    )
    return gfem.Model(
        mesh,
        materials={
            "lower": gfem.Material(eps_r=2.0) if lower is None else lower,
            "upper": gfem.Material(),
        },
        boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), sheet],
        ports=ports,
    )


def _rlc_port(name, surface):
    return gfem.LumpedPort(
        name,
        surface,
        "+y",
        resistance=1000.0,
        inductance=LUMPED[0],
        capacitance=LUMPED[1],
    )


@pytest.fixture(scope="module")
def lumped_cases(sheet_mesh):
    lumped = _lumped_model(
        sheet_mesh,
        [_rlc_port("p1", "port1"), gfem.LumpedPort("p2", "port2", "+y", impedance=1e3)],
    )
    mixed = _lumped_model(
        sheet_mesh, [gfem.WavePort("p1", "port1"), _rlc_port("p2", "port2")]
    )
    return {
        "driven_lumped": (lumped, gfem.Driven(frequency=20e9, keep_fields=True)),
        "driven_lumped_adaptive": (lumped, LUMPED_ADAPTIVE),
        "eigenmode_lumped": (lumped, gfem.Eigenmode(target=LUMPED_TARGET)),
        "electrostatic_lumped": (
            lumped,
            gfem.Electrostatic(terminals={"top": "plate2"}, ground="plate1"),
        ),
        "port_mode_lumped": (mixed, gfem.PortMode("p1", 20e9)),
    }


@pytest.fixture(scope="module")
def wave_cases(wave_model, shared_surface_model):
    return {
        "driven_wave": (wave_model, gfem.Driven(frequency=20e9)),
        "driven_wave_shared": (shared_surface_model, SHARED_SURFACE_PROBLEM),
    }


@pytest.fixture(scope="module")
def driven_port_face_cases(tmp_path_factory, sheet_mesh):
    """The driven cases with a design at the wave port faces, by name."""
    path = tmp_path_factory.mktemp("wave_face") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 1, 1), core=(1.0, 2.0)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    voltage = ((0.0, 0.5e-3, 0.0), (0.0, 0.5e-3, 1e-3))
    ports = [gfem.WavePort("p1", "port1", voltage_path=voltage)]
    ports.append(gfem.WavePort("p2", "port2"))

    def line(dielectric):
        return gfem.Model(
            mesh,
            materials={"dielectric": dielectric, "core": gfem.Material(eps_r=3.0)},
            boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
            ports=ports,
        )

    n = len(sheet_mesh.elements("sheet"))
    sheet = gfem.Impedance(
        "sheet",
        resistance=LUMPED[2],
        inductance=LUMPED[3],
        capacitance=LUMPED[4],
        fill=gfem.SheetField(fill=np.linspace(0.2, 0.9, n)),
    )
    waves = [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")]
    hybrid = hybrid_mesh(tmp_path_factory.mktemp("hybrid_face"), cells=(4, 2, 2))
    fill = 2.0 * np.linspace(1.0, 1.5, len(hybrid.elements("fill")))
    return {
        "driven_wave_face": (
            line(_field(mesh, "dielectric")),
            gfem.Driven(frequency=20e9),
        ),
        "driven_wave_shared_face": (
            # lossy leads: the modes' gamma has a real part to differentiate
            hybrid_line(hybrid, tan_delta=0.02).with_material(
                "fill", gfem.MaterialField(eps_r=fill)
            ),
            SHARED_SURFACE_PROBLEM,
        ),
        "driven_wave_face_sheet": (
            gfem.Model(
                sheet_mesh,
                materials={"lower": gfem.Material(eps_r=2.0), "upper": gfem.Material()},
                boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), sheet],
                ports=waves,
            ),
            gfem.Driven(frequency=20e9),
        ),
        "driven_wave_face_law": (
            line(gfem.Material(eps_r=FACE_LAW)),
            gfem.Driven(frequency=20e9),
        ),
    }


@pytest.fixture(scope="module")
def bloch_wave_port_model(tmp_path_factory):
    """A periodic cell along x with a wave port on a patch of z_min, off the pair."""
    edges = tuple(
        np.linspace(0, size, n + 1)
        for size, n in zip((10.0, 7.0, 1.0), (4, 3, 1), strict=True)
    )

    def surface(c):
        return "port" if abs(c[2]) < 1e-9 and 2.5 < c[0] < 7.5 else ""

    path = tmp_path_factory.mktemp("bloch_port") / "cell.msh"
    sm.write_msh22(path, sm.periodic_cell(edges, surface=surface))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    eps = np.full(len(mesh.elements("cell")), 2.0)
    return gfem.Model(
        mesh,
        materials={"cell": gfem.MaterialField(eps_r=eps)},
        boundaries=[gfem.Periodic(("x_min", "x_max"))],
        ports=[gfem.WavePort("w", "port")],
    )


@pytest.fixture(scope="module")
def driven_periodic_cases(tmp_path_factory):
    # the C1 line (periodic along x) on 3 cells, its port sheets the inner
    # grid planes, a material field over its cell
    path = tmp_path_factory.mktemp("c1") / "c1.msh"
    edges = ((0.0, pc.C1_X1, pc.C1_X2, pc.C1_A), (0.0, pc.C1_W), (0.0, pc.C1_H))
    sheets = (("sheet1", pc.C1_X1), ("sheet2", pc.C1_X2))
    sm.write_msh22(path, sm.periodic_cell(tuple(map(np.asarray, edges)), sheets=sheets))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = pc.c1(mesh).with_material("cell", _field(mesh, "cell"))
    return {
        "driven_periodic_plain": (
            model,
            gfem.Driven(frequency=5e9, keep_fields=True),
        ),
        "driven_periodic_law": (
            model.with_material("cell", gfem.Material(eps_r=TABLE_LAW)),
            gfem.Driven(frequency=5e9, keep_fields=True),
        ),
        "driven_periodic_bloch": (
            model,
            gfem.Driven(frequency=5e9, keep_fields=True, wave_vector=pc.C1_K),
        ),
    }


@pytest.fixture(scope="module")
def driven_floquet_cases(tmp_path_factory):
    # F1 (coarse): a material field over its slab, plain and at F1's conical
    # incidence from the air
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("f1_table"))
    model = pc.f1(mesh).with_material("slab", _field(mesh, "slab"))
    k = model.bloch_wave_vector(
        "z_max", pc.F1_FREQUENCY, theta=pc.F1_THETA, phi=pc.F1_PHI
    )
    return {
        "driven_floquet_plain": (model, gfem.Driven(frequency=pc.F1_FREQUENCY)),
        "driven_floquet_bloch": (
            model,
            gfem.Driven(frequency=pc.F1_FREQUENCY, wave_vector=k),
        ),
    }


@pytest.fixture(scope="module")
def eigen_periodic_cases(tmp_path_factory):
    # the C3 Bragg line (periodic along x), weakly lossy: a material field
    # over its low layer, the high layer a constant material
    mesh = pc.c3_mesh(tmp_path_factory.mktemp("c3"), (5, 1, 1))
    ramp = np.linspace(1.0, 1.5, len(mesh.elements("low")))
    low = gfem.MaterialField(
        eps_r=ALL_PROPERTIES["eps_r"] * ramp, conductivity=1e-3 * ramp
    )
    model = pc.c3(mesh).with_material("low", low)
    zone_edge = (math.pi / (pc.C3_A * pc.MM), 0.0, 0.0)
    return {
        "eigenmode_periodic_plain": (
            model,
            gfem.Eigenmode(target=pc.C3_PLAIN_TARGET),
        ),
        "eigenmode_periodic_bloch": (
            model,
            gfem.Eigenmode(target=5e9, wave_vector=zone_edge),
        ),
    }


@pytest.fixture(scope="module")
def thermal_problem_cases(sheet_mesh):
    """The steady, transient and DC conduction problems on the sheet line.

    Every material property on the lower half (conducting, the heat source
    there), a sheet fill between the halves, a fixed temperature, a
    convection with another ambient (so the source-free temperature is not
    uniform: its own adjoint pair enters) and a heat flux through an end.
    """
    mesh = sheet_mesh
    fill = gfem.SheetField(fill=np.linspace(0.2, 0.9, len(mesh.elements("sheet"))))
    upper = gfem.Material(
        eps_r=2.0, thermal_conductivity=1.0, mass_density=2000.0, specific_heat=800.0
    )
    model = gfem.Model(
        mesh,
        materials={"lower": _field(mesh, "lower"), "upper": upper},
        boundaries=[
            gfem.Conductor("sheet", 100.0, 1e-6, fill=fill),
            gfem.FixedTemperature("plate1", temperature=300.0),
            gfem.Convection("plate2", coefficient=1e3, ambient=290.0),
            gfem.HeatFlux("port1", flux=50.0),
        ],
    )
    heat = gfem.HeatSource.uniform("lower", power=1e-3)
    transient = gfem.TransientThermal(
        heat,
        duration=0.5,
        steps=4,
        waveform=[(0.0, 1.0), (0.25, 0.5)],
        traces={"upper": gfem.Trace.mean("upper")},
        keep_fields=True,
    )
    return {
        "steady_thermal": (model, gfem.SteadyThermal(heat)),
        "transient_thermal": (model, transient),
        "dc_conduction": (model, gfem.DCConduction({"t": "port1"}, "port2")),
    }


@pytest.fixture(scope="module")
def piezoelectric_case(tmp_path_factory):
    """The transducer section of the piezoelectric tests: lines on lithium niobate.

    The film is a field holding every material property (its tensors scaled
    per element, on the crystal's axes, the unread ones given arbitrary
    values); its damping frame is a constant material beside it. The mesh is
    the coarsest the layout allows (the electrode edges, not ``maxh``, set its
    size): the table counts one adjoint solve per cell, so a cell's cost is the
    section's.
    """
    mesh = pz.mesh(
        tmp_path_factory.mktemp("piezoelectric") / "idt.msh", "lines", maxh=2.5
    )
    ramp = np.linspace(1.0, 1.5, len(mesh.elements("ln")))
    values = ALL_PROPERTIES | {
        "eps_r": np.array(pz.EPS_LN),
        "mu_r": np.ones(3),
        "tan_delta": np.full(3, 0.01),
        "conductivity": np.full(3, 2.0),
        "stiffness": pz.C_LN,
        "piezoelectric_tensor": pz.E_LN,
        "mass_density": pz.RHO_LN,
        "mechanical_quality_factor": pz.QM,
    }
    film = gfem.MaterialField(**_ramped(values, ramp), axes=pz.AXES)
    model = pz.model(mesh, "lines").with_material("ln", film)
    return {"piezoelectric": (model, pz.problem(keep_fields=True))}


@pytest.fixture(scope="module")
def measurement_cases(tmp_path_factory):
    """The thermo-optic shift on a section, the route phase through a 3D block.

    Each as ``(model, rise, measure)``: ``measure(model, rise)`` is the
    measurement. The core holds every material property (lossless: the
    shift refuses a lossy section), a sheet inside the cladding.
    """
    directory = tmp_path_factory.mktemp("measurements")
    rects = [
        (-1.25, -0.6, 1.25, -0.3, "clad", None),
        (-1.25, -0.3, 1.25, 0.82, "clad", None),
        (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
    ]
    edges = [
        (lambda p: abs(p[1] + 0.6) < 1e-9, "bottom"),
        (lambda p: abs(p[1] + 0.3) < 1e-9, "sheet"),
    ]
    section = tc.section(directory / "section.msh", rects, 0.3, edges)
    n = len(section.elements("core"))
    ramp = np.linspace(1.0, 1.5, n)
    values = ALL_PROPERTIES | {
        "eps_r": 12.1,
        "tan_delta": 0.0,
        "conductivity": 0.0,
        "thermo_optic_coefficient": 1.86e-4,
    }
    core = gfem.MaterialField(**_ramped(values, ramp))
    clad = gfem.Material(
        eps_r=2.085,
        thermal_conductivity=1.38,
        mass_density=2200.0,
        specific_heat=700.0,
        thermo_optic_coefficient=1e-5,
    )
    fill = gfem.SheetField(fill=np.linspace(0.2, 0.9, len(section.elements("sheet"))))
    optical = gfem.Model(
        section,
        materials={"clad": clad, "core": core},
        boundaries=[gfem.Conductor("sheet", 100.0, 1e-6, fill=fill)],
        ports=[gfem.WavePort("p", "front")],
    )
    thermal = gfem.Model(
        section,
        materials={"clad": clad, "core": core},
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
    )
    heat = gfem.HeatSource.uniform("core", power_per_length=1.0)
    section_rise = gfem.solve(thermal, gfem.SteadyThermal(heat)).rise
    block_path = directory / "block.msh"
    sink = [(lambda p: abs(p[0]) < 1e-9, "sink")]
    block = tc.brick(block_path, (4.0, 3.0), [("block", 3.0)], 1.0, sink)
    block_model = gfem.Model(
        block,
        materials={"block": gfem.Material(thermal_conductivity=1.38)},
        boundaries=[gfem.FixedTemperature("sink", temperature=300.0)],
    )
    source = gfem.HeatSource.uniform("block", power_density=1e12)
    block_rise = gfem.solve(block_model, gfem.SteadyThermal(source)).rise
    route = {"r": np.array([[2.0, 0.5, 1.5], [2.0, 2.5, 1.5]]) * 1e-6}

    def shift(model, rise):
        return gfem.thermo_optic_shift(model, rise, port="p", frequency=1.934e14)

    def phase(model, rise):
        return gfem.route_phase(model, rise, routes=route, port="p", frequency=1.934e14)

    return {
        "thermo_optic_shift": (optical, section_rise, shift),
        "route_phase": (optical, block_rise, phase),
    }


def _measurement_cells(cases):
    for name, (model, rise, measure) in cases.items():
        result = measure(model, rise)
        kinds = [
            input_kind(path, "model")
            for path, _ in jax.tree_util.tree_flatten_with_path(model)[0]
        ] + [("rise",)]
        for out_path, _ in jax.tree_util.tree_flatten_with_path(result)[0]:
            out_name = jax.tree_util.keystr(out_path, simple=True, separator=".")
            for index, kind in enumerate(kinds):
                yield name, model, rise, measure, out_name, out_path, index, kind


def _measurement_objective(model, rise, measure, out_path, index):
    """One measurement leaf's weighted sum in one input leaf (rise last)."""
    leaves, treedef = jax.tree.flatten(model)
    inputs = [*leaves, rise.coefficients]

    def objective(x):
        moved = [*inputs[:index], x, *inputs[index + 1 :]]
        result = measure(
            jax.tree.unflatten(treedef, moved[:-1]),
            dataclasses.replace(rise, coefficients=moved[-1]),
        )
        (value,) = [
            v
            for p, v in jax.tree_util.tree_flatten_with_path(result)[0]
            if p == out_path
        ]
        weights = np.random.default_rng(1).uniform(0.5, 1.5, np.shape(value))
        return jnp.sum(jnp.real(value) * weights)

    return objective, inputs[index]


def test_every_measurement_cell_has_an_expected_outcome(measurement_cases):
    cells = {(n, o, k) for n, *_, o, _, _, k in _measurement_cells(measurement_cases)}
    assert cells == {c for c in EXPECTED if c[0] in measurement_cases}


def test_measurement_raise_table(measurement_cases):
    for (
        name,
        model,
        rise,
        measure,
        out_name,
        out_path,
        index,
        kind,
    ) in _measurement_cells(measurement_cases):
        expected = EXPECTED[(name, out_name, kind)]
        objective, x = _measurement_objective(model, rise, measure, out_path, index)
        cell = f"{name} {out_name} x {kind}"
        if expected == RAISES:
            with pytest.raises(gfem.NonDifferentiableError):  # noqa: PT012
                jax.make_jaxpr(jax.grad(objective))(x)
                pytest.fail(f"{cell}: no NonDifferentiableError")
            continue
        grad = np.asarray(jax.grad(objective)(x))
        assert np.isfinite(grad).all(), cell
        if expected == ZERO:
            assert not grad.any(), cell
        else:
            assert np.abs(grad).max() > 0, cell


HEAT_FREQUENCY = 20e9
#: The heat cases' drive: p1's and p2's incident powers (W), p2's phase (rad).
HEAT_DRIVE = {"power": {"p1": 1.0, "p2": 0.5}, "phase": {"p2": 0.7}}


@pytest.fixture(scope="module")
def heat_cases(sheet_mesh):
    """The dissipated power (also beside a law) and the heating on the sheet line.

    Each as ``(model, measure)``: ``measure(model, drive)`` is the
    measurement. The lower half holds every material property, the upper
    half is lossless (a plain material), the sheet an `Impedance` with every
    value and a fill; the lower plate is the heat sink.
    """
    n = len(sheet_mesh.elements("sheet"))
    fill = gfem.SheetField(fill=np.linspace(0.2, 0.9, n))
    model = gfem.Model(
        sheet_mesh,
        materials={
            "lower": _field(sheet_mesh, "lower"),
            "upper": gfem.Material(thermal_conductivity=1.0),
        },
        boundaries=[
            gfem.PEC("plate1"),
            gfem.PEC("plate2"),
            gfem.Impedance(
                "sheet", resistance=2e3, inductance=2e-8, capacitance=2e-15, fill=fill
            ),
            gfem.FixedTemperature("plate1", temperature=300.0),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+y", impedance=1000.0),
            gfem.LumpedPort("p2", "port2", "+y", impedance=1000.0),
        ],
    )
    problem = gfem.Driven(frequency=HEAT_FREQUENCY, keep_fields=True)

    def lost(model, _drive):
        field = gfem.solve(model, problem).e_field(HEAT_FREQUENCY, "p1")
        return gfem.dissipated_power(field, model=model)

    def heated(model, drive):
        heat = gfem.electromagnetic_heating(
            gfem.solve(model, problem),
            model=model,
            frequency=HEAT_FREQUENCY,
            incident_power=drive["power"],
            incident_phase=drive["phase"],
        )
        return gfem.solve(model, gfem.SteadyThermal(heat))

    law = model.with_material(
        "upper", gfem.Material(eps_r=TABLE_LAW, thermal_conductivity=1.0)
    )
    n_upper = len(sheet_mesh.elements("upper"))
    blend = model.with_material(
        "upper",
        gfem.MaterialField.interpolate(
            jnp.linspace(0.3, 0.7, n_upper),
            gfem.Material(eps_r=2.0),
            gfem.Material(eps_r=TABLE_LAW),
            scheme="index",
        ),
    )

    def stored(model, _drive):
        field = gfem.solve(model, problem).e_field(HEAT_FREQUENCY, "p1")
        return gfem.electric_energy(field, model=model)

    energy_law = model.with_material(
        "upper", gfem.Material(eps_r=ENERGY_LAW, thermal_conductivity=1.0)
    )
    return {
        "electric_energy": (model, stored),
        "electric_energy_law": (energy_law, stored),
        "dissipated_power": (model, lost),
        "dissipated_power_law": (law, lost),
        "dissipated_power_blend": (blend, lost),
        "electromagnetic_heating": (model, heated),
    }


def _heat_cells(cases):
    drive_paths = jax.tree_util.tree_flatten_with_path(HEAT_DRIVE)[0]
    for name, (model, measure) in cases.items():
        result = measure(model, HEAT_DRIVE)
        kinds = [
            input_kind(path, "model")
            for path, _ in jax.tree_util.tree_flatten_with_path(model)[0]
        ]
        if name == "electromagnetic_heating":
            kinds += [("drive", *(str(k.key) for k in path)) for path, _ in drive_paths]
        for out_path, _ in jax.tree_util.tree_flatten_with_path(result)[0]:
            out_name = jax.tree_util.keystr(out_path, simple=True, separator=".")
            for index, kind in enumerate(kinds):
                yield name, model, measure, out_name, out_path, index, kind


def _heat_objective(model, measure, out_path, index):
    """One heat leaf's weighted sum in one input leaf (the drive's last)."""
    leaves, treedef = jax.tree.flatten(model)
    drive_leaves, drive_tree = jax.tree.flatten(HEAT_DRIVE)
    inputs = [*leaves, *drive_leaves]

    def objective(x):
        moved = [*inputs[:index], x, *inputs[index + 1 :]]
        result = measure(
            jax.tree.unflatten(treedef, moved[: len(leaves)]),
            jax.tree.unflatten(drive_tree, moved[len(leaves) :]),
        )
        (value,) = [
            v
            for p, v in jax.tree_util.tree_flatten_with_path(result)[0]
            if p == out_path
        ]
        weights = np.random.default_rng(1).uniform(0.5, 1.5, np.shape(value))
        return jnp.sum(jnp.real(value) * weights)

    return objective, jnp.asarray(inputs[index])


def test_every_heat_cell_has_an_expected_outcome(heat_cases):
    cells = {(n, o, k) for n, *_, o, _, _, k in _heat_cells(heat_cases)}
    assert cells == {c for c in EXPECTED if c[0] in HEAT_READS}


def test_heat_raise_table(heat_cases):
    for name, model, measure, out_name, out_path, index, kind in _heat_cells(
        heat_cases
    ):
        expected = EXPECTED[(name, out_name, kind)]
        objective, x = _heat_objective(model, measure, out_path, index)
        cell = f"{name} {out_name} x {kind}"
        if expected == RAISES:
            with pytest.raises(gfem.NonDifferentiableError):  # noqa: PT012
                jax.make_jaxpr(jax.grad(objective))(x)
                pytest.fail(f"{cell}: no NonDifferentiableError")
            continue
        grad = np.asarray(jax.grad(objective)(x))
        assert np.isfinite(grad).all(), cell
        if expected == ZERO:
            assert not grad.any(), cell
        else:
            assert np.abs(grad).max() > 0, cell


# --- the heat kernel's explicit term: what is never heat ----------------------------


def _excluded_metal_model(directory):
    """A 4 x 2 x 1 mm line beside a 1 mm metal block left out of the solve.

    Plates at z = 0 and 1 mm over the line, lumped ports at its ends, PEC
    on the face it shares with the block; the block is excluded from the
    electromagnetic problem.
    """
    edges = (np.linspace(0, 4, 5), np.linspace(0, 3, 4), np.array([0.0, 1.0]))

    def region(c):
        return "metal" if c[1] > 2.0 else "line"

    def face(c):
        x, y, z = c
        if y > 2.0 + 1e-9:
            return ""
        if abs(y - 2.0) < 1e-9:
            return "wall"
        if abs(x) < 1e-9:
            return "port1"
        if abs(x - 4.0) < 1e-9:
            return "port2"
        return "plates" if abs(z) < 1e-9 or abs(z - 1.0) < 1e-9 else ""

    path = sm.write_msh22(directory / "beside.msh", sm.structured(edges, region, face))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={
            "line": _field(mesh, "line"),
            "metal": gfem.MaterialField(
                eps_r=np.full(len(mesh.elements("metal")), 1.0 - 0.5j),
                conductivity=np.full(len(mesh.elements("metal")), 1e3),
            ),
        },
        boundaries=[gfem.PEC("plates"), gfem.PEC("wall")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
        excluded={"em": ("metal",)},
    )


def _explicit_case(request, case):
    """``(model, frequency, leaf)``: the model whose field is held, and the leaf."""
    if case == "em-excluded tetrahedra":
        directory = request.getfixturevalue("tmp_path")
        return _excluded_metal_model(directory), 20e9, ("metal", "conductivity")
    model, _ = request.getfixturevalue("heat_cases")["dissipated_power"]
    leaf = {
        "mu_r": ("lower", "mu_r"),
        "Impedance inductance": ("sheet", "inductance"),
        "Impedance capacitance": ("sheet", "capacitance"),
        "loss tangent (control)": ("lower", "tan_delta"),
        "Impedance resistance (control)": ("sheet", "resistance"),
    }[case]
    return model, HEAT_FREQUENCY, leaf


def _moved(model, leaf, value):
    owner, name = leaf
    if owner == "sheet":
        (i,) = [
            i for i, b in enumerate(model.boundaries) if isinstance(b, gfem.Impedance)
        ]
        boundaries = list(model.boundaries)
        boundaries[i] = dataclasses.replace(boundaries[i], **{name: value})
        return model.replace(boundaries=boundaries)
    material = dataclasses.replace(model.materials[owner], **{name: value})
    return model.with_material(owner, material)


def _held_value(model, leaf):
    owner, name = leaf
    if owner == "sheet":
        (b,) = [b for b in model.boundaries if isinstance(b, gfem.Impedance)]
        return jnp.asarray(getattr(b, name))
    return jnp.asarray(getattr(model.materials[owner], name))


EXPLICIT = {
    # a real permeability stores energy, it does not dissipate
    "mu_r": ZERO,
    # a sheet's L and C are reactive: its loss is |E_t|^2 / (2 R) per fill
    "Impedance inductance": ZERO,
    "Impedance capacitance": ZERO,
    # a region left out of the solve has no field to dissipate
    "em-excluded tetrahedra": ZERO,
    "loss tangent (control)": GRADIENT,
    "Impedance resistance (control)": GRADIENT,
}


def test_a_lossy_perfectly_matched_layer_heats_nothing_whatever_the_design(pml_model):
    # the layer's background is lossy, the slab lossless: the layer's
    # absorption leaves the model, so the heat is an exact zero of the slab's
    # permittivity, though the field moves with it. (The background is a
    # constant `Material`, so it is no input itself.)
    model = pml_model.with_material("pml", gfem.Material(tan_delta=0.1))
    problem = gfem.Driven(frequency=pml_cases.F0, keep_fields=True)
    eps = jnp.linspace(2.0, 3.0, len(model.mesh.elements("slab")))

    def lost(e):
        moved = model.with_material("slab", gfem.MaterialField(eps_r=e))
        result = gfem.solve(moved, problem)
        field = result.e_field(pml_cases.F0, "p1")
        heat = gfem.dissipated_power(field, model=moved)
        return sum(jnp.real(v) for v in heat.values()), result.s.values

    assert float(lost(eps)[0]) == 0.0
    grad = np.asarray(jax.grad(lambda e: lost(e)[0])(eps))
    assert np.isfinite(grad).all()
    assert not grad.any()
    # the field (and so S) does move with the design
    moved = jax.grad(lambda e: jnp.sum(jnp.abs(lost(e)[1]) ** 2))(eps)
    assert np.abs(moved).max() > 0


@pytest.mark.parametrize("case", EXPLICIT)
def test_the_heat_kernel_s_explicit_term(request, case):
    # the field held: what the dissipated power reads of the model itself,
    # beside what the field reads (the solve's own gradient)
    model, frequency, leaf = _explicit_case(request, case)
    field = gfem.solve(model, gfem.Driven(frequency=frequency, keep_fields=True))
    field = field.e_field(frequency, model.ports[0].name)

    def objective(value):
        lost = gfem.dissipated_power(field, model=_moved(model, leaf, value))
        return sum(jnp.real(v) for v in lost.values())

    grad = np.asarray(jax.grad(objective)(_held_value(model, leaf)))
    assert np.isfinite(grad).all()
    if EXPLICIT[case] == ZERO:
        assert not grad.any(), case
    else:
        assert np.abs(grad).max() > 0, case


@pytest.fixture(scope="module")
def dipole_cases(dipole_model):
    return _dipole_cases(dipole_model)


#: The fixture that builds a case, for the cases a fixture builds several of: a
#: case's item requests only that fixture, not every case of the table.
CASE_GROUPS = {
    "wave_cases": ("driven_wave", "driven_wave_shared"),
    "dipole_cases": ("driven_dipole", "driven_magnetic_dipole"),
    "driven_port_face_cases": (
        "driven_wave_face",
        "driven_wave_shared_face",
        "driven_wave_face_sheet",
        "driven_wave_face_law",
    ),
    "sheet_cases": ("driven_sheet", "eigenmode_sheet", "port_mode_sheet"),
    "lumped_cases": (
        "driven_lumped",
        "driven_lumped_adaptive",
        "eigenmode_lumped",
        "electrostatic_lumped",
        "port_mode_lumped",
    ),
    "driven_periodic_cases": (
        "driven_periodic_plain",
        "driven_periodic_law",
        "driven_periodic_bloch",
    ),
    "driven_floquet_cases": ("driven_floquet_plain", "driven_floquet_bloch"),
    "eigen_periodic_cases": ("eigenmode_periodic_plain", "eigenmode_periodic_bloch"),
    "law_cases": (
        "driven_law",
        "driven_law_absorbing",
        "driven_law_adaptive",
        "driven_blend",
    ),
    "thermal_problem_cases": ("steady_thermal", "transient_thermal", "dc_conduction"),
    "piezoelectric_case": ("piezoelectric",),
    "eigen_pml_case": ("eigenmode_pml",),
    "planewave_case": ("driven_planewave",),
}
#: These two fixtures return their one case itself, not a dict of cases.
BARE_CASES = ("eigen_pml_case", "planewave_case")


def _port_model(*, law):
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    path = ((20e-6, 0.0, 0.0), (20e-6, 10e-6, 0.0))
    port_model = gfem.Model(
        mesh,
        materials={"fill": _field(mesh, "fill")},
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort("p1", "port1", voltage_path=path),
            gfem.WavePort("p2", "port2"),
        ],
    )
    if law:
        port_model = port_model.with_material(
            "fill", gfem.Material(eps_r=PORT_MODE_LAW)
        )
    return port_model, gfem.PortMode("p1", 10e9)


def _fixture(request, name):
    return request.getfixturevalue(name)


#: The cases no group fixture builds: name -> builder(request).
SINGLE_CASES = {
    "electrostatic": lambda r: (_fixture(r, "model"), PROBLEMS["electrostatic"]),
    "electrostatic_law": lambda r: (
        _fixture(r, "model").with_material(
            "nitride", gfem.Material(eps_r=ELECTROSTATIC_LAW)
        ),
        PROBLEMS["electrostatic"],
    ),
    "port_mode": lambda _: _port_model(law=False),
    "port_mode_law": lambda _: _port_model(law=True),
    "driven": lambda r: (
        _fixture(r, "driven_model"),
        gfem.Driven(frequency=5e9, keep_fields=True),
    ),
    "driven_adaptive": lambda r: (_fixture(r, "driven_model"), ADAPTIVE),
    "eigenmode": lambda r: (
        _fixture(r, "eigen_model"),
        gfem.Eigenmode(target=17e9),
    ),
    "eigenmode_wave": lambda r: (
        _fixture(r, "eigen_wave_model"),
        gfem.Eigenmode(target=EIGEN_WAVE_TARGET),
    ),
    "driven_pml": lambda r: (
        _fixture(r, "pml_model"),
        gfem.Driven(frequency=pml_cases.F0),
    ),
    "eigenmode_law": lambda r: (
        _fixture(r, "eigen_law_model"),
        gfem.Eigenmode(target=17e9),
    ),
}
GROUP_OF = {name: group for group, names in CASE_GROUPS.items() for name in names}
CASE_NAMES = tuple(sorted({*SINGLE_CASES, *GROUP_OF}))


def _case(request, name):
    """The ``(model, problem)`` of one case, building only the fixtures it needs."""
    if name in SINGLE_CASES:
        return SINGLE_CASES[name](request)
    group = request.getfixturevalue(GROUP_OF[name])
    if GROUP_OF[name] in BARE_CASES:
        return group
    _check_group_keys(GROUP_OF[name], group)
    return group[name]


def _check_group_keys(fixture, built):
    """A group fixture builds exactly the cases ``CASE_GROUPS`` lists for it.

    A case a fixture builds that no entry lists would never be requested, so the
    table would skip it without a word; one listed but not built would be a
    KeyError far from the cause.
    """
    listed = set(CASE_GROUPS[fixture])
    returned = set(built)
    if returned != listed:
        raise AssertionError(
            f"group fixture {fixture!r} returns cases that differ from CASE_GROUPS: "
            f"built but not listed {sorted(returned - listed)}, "
            f"listed but not built {sorted(listed - returned)}"
        )


@pytest.fixture(scope="module")
def all_cases(request):
    return {name: _case(request, name) for name in CASE_NAMES}


@pytest.fixture(scope="module")
def cases(all_cases):
    return {k: v for k, v in all_cases.items() if k not in PORT_FACE_NAMES}


@pytest.fixture(scope="module")
def port_face_cases(all_cases):
    return {k: all_cases[k] for k in PORT_FACE_NAMES}


#: Both modes of the hybrid line propagate (mode 2 from about 48 GHz). One
#: excitation: S's other rows (cross-mode included) take adjoint loads.
SHARED_SURFACE_PROBLEM = gfem.Driven(frequency=60e9, excite="p1_m1")
PROBLEMS = {
    "electrostatic": gfem.Electrostatic(terminals={"top": "top"}, ground="bottom"),
}


# A wave port's face nodes are inputs of its mode wherever the design is: gamma
# follows them, a diagnostic raises, and the rest (a constant, an impedance
# without a path) is an exact zero.
LOSSLESS_WAVE_CASES = (
    "driven_wave",
    "driven_wave_shared",
    "driven_pml",
    "driven_dipole",
)
for _key in list(EXPECTED):
    _name, _out, _kind = _key
    if (
        _kind == ("nodes",)
        and _out.startswith("port_modes.")
        and _name not in PORT_FACE_NAMES
        and EXPECTED[_key] != RAISES
    ):
        _leaf = _out.rsplit(".", 1)[-1]
        # (the objective reads the real part: a lossless mode's gamma is imaginary)
        _gamma = ZERO if _name in LOSSLESS_WAVE_CASES else GRADIENT
        EXPECTED[_key] = {"gamma": _gamma, "residual": RAISES}.get(_leaf, ZERO)


def _cells(cases):
    for problem_name, (model, problem) in cases.items():
        result = gfem.solve(model, problem)
        model_leaves = jax.tree_util.tree_flatten_with_path(model)[0]
        kinds = [input_kind(path, "model") for path, _ in model_leaves] + [
            input_kind(path, "problem")
            for path, _ in jax.tree_util.tree_flatten_with_path(problem)[0]
        ]
        for out_path, _ in jax.tree_util.tree_flatten_with_path(result)[0]:
            out_name = jax.tree_util.keystr(out_path, simple=True, separator=".")
            for index, kind in enumerate(kinds):
                yield problem_name, model, problem, out_name, out_path, index, kind


def _objective(model, problem, out_path, index):
    """The weighted sum of one result leaf as a function of one input leaf."""
    leaves, treedef = jax.tree.flatten((model, problem))

    def objective(x):
        moved, moved_problem = jax.tree.unflatten(
            treedef, [*leaves[:index], x, *leaves[index + 1 :]]
        )
        result = gfem.solve(moved, moved_problem)
        (value,) = [
            v
            for p, v in jax.tree_util.tree_flatten_with_path(result)[0]
            if p == out_path
        ]
        weights = np.random.default_rng(1).uniform(0.5, 1.5, np.shape(value))
        return jnp.sum(jnp.real(value) * weights)

    return objective, leaves[index]


#: The raise table's cases, one test item each (their gradients solve per cell).
RAISE_TABLE_NAMES = tuple(
    sorted(
        {c[0] for c in EXPECTED}
        - set(MEASUREMENT_READS)
        - set(HEAT_READS)
        - set(PORT_FACE_NAMES)
    )
)


#: The cases whose cells take over 15 minutes as one item: one item per result leaf.
SPLIT_BY_LEAF = ("eigenmode_wave",)
RAISE_TABLE_ITEMS = tuple(
    (name, leaf)
    for name in RAISE_TABLE_NAMES
    for leaf in (
        sorted({c[1] for c in EXPECTED if c[0] == name})
        if name in SPLIT_BY_LEAF
        else [None]
    )
)


def test_the_raise_table_names_are_the_built_cases(cases):
    assert set(RAISE_TABLE_NAMES) == set(cases)


def test_every_cell_has_an_expected_outcome(cases):
    cells = {(p, o, k) for p, _, _, o, _, _, k in _cells(cases)}
    assert cells == {
        c
        for c in EXPECTED
        if c[0] not in MEASUREMENT_READS
        and c[0] not in HEAT_READS
        and c[0] not in PORT_FACE_NAMES
    }


def test_every_port_face_cell_has_an_expected_outcome(port_face_cases):
    cells = {(p, o, k) for p, _, _, o, _, _, k in _cells(port_face_cases)}
    assert cells == {c for c in EXPECTED if c[0] in PORT_FACE_NAMES}


@pytest.mark.parametrize("name", PORT_FACE_NAMES)
def test_port_face_raise_table(request, name):
    """The port mode's cases, one item each (their gradients solve a mode per cell)."""
    only = {name: _case(request, name)}
    for _, model, problem, out_name, out_path, index, kind in _cells(only):
        expected = EXPECTED[(name, out_name, kind)]
        objective, x = _objective(model, problem, out_path, index)
        cell = f"{name} {out_name} x {kind}"
        if expected == RAISES:
            with pytest.raises(gfem.NonDifferentiableError):  # noqa: PT012
                jax.make_jaxpr(jax.grad(objective))(x)
                pytest.fail(f"{cell}: no NonDifferentiableError")
            continue
        grad = np.asarray(jax.grad(objective)(x))
        assert np.isfinite(grad).all(), cell
        if expected == ZERO:
            assert not grad.any(), cell
        else:
            assert np.abs(grad).max() > 0, cell


PORT_FACE_PATH = ((20e-6, 0.0, 0.0), (20e-6, 10e-6, 0.0))


def _plate_mode(eps, path=True):
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    n = len(mesh.elements("fill"))
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.MaterialField(eps_r=eps * jnp.ones(n))},
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort("p1", "port1", voltage_path=PORT_FACE_PATH if path else None),
            gfem.WavePort("p2", "port2"),
        ],
    )
    return gfem.solve(model, gfem.PortMode("p1", 10e9)), model


def test_a_port_mode_field_under_batched_nodes_is_refused():
    """`PortModeField.at` reads fixed points: traced nodes are not available."""
    _, model = _plate_mode(4.0)
    nodes = jnp.asarray(model.mesh.nodes)

    def field(x):
        moved = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [x]
        )
        result = gfem.solve(model.replace(mesh=moved), gfem.PortMode("p1", 10e9))
        return result.e_field.at([20e-6, 5e-6, 0.0])

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"PortModeField\.at reads the mode at fixed"
    ):
        jax.vmap(field)(nodes[None])


def test_a_field_cotangent_under_wanted_nodes_has_the_same_refusal():
    from gfem._sweep.port_mode import PortModeRule

    _, model = _plate_mode(4.0)
    rule = PortModeRule(model, gfem.PortMode("p1", 10e9), gfem.SolverOptions())
    text = rule.nondifferentiable_output("field", {("nodes",)})
    assert "PortModeField.at reads the mode at fixed points" in text
    assert (
        rule.nondifferentiable_output("field", {("material", "fill", "eps_r")}) is None
    )
    assert "diagnostic" in rule.nondifferentiable_output("residual", ())


def test_the_characteristic_impedance_of_a_traced_mode_raises_at_call_time():
    def impedance(eps):
        mode, _ = _plate_mode(eps)
        return jnp.real(
            gfem.characteristic_impedance(mode, voltage_path=PORT_FACE_PATH)[0]
        )

    with pytest.raises(
        gfem.NonDifferentiableError,
        match="characteristic_impedance is not differentiable",
    ):
        jax.grad(impedance)(4.0)


@pytest.mark.parametrize("rule", ["thermo", "electro"])
def test_a_shift_keeps_refusing_the_optical_inputs_of_its_mode(rule):
    from gfem._engine.electro_optic import ElectroOpticWeightsRule
    from gfem._engine.thermo_optic import ShiftWeightsRule

    _, model = _plate_mode(4.0)
    made = {"thermo": ShiftWeightsRule, "electro": ElectroOpticWeightsRule}[rule](
        model, gfem.PortMode("p1", 10e9), gfem.SolverOptions(), np.array([0])
    )
    reason = made.unsupported_input(("material", "fill", "eps_r"))
    assert "shift holds the port mode constant" in reason
    assert "is not wired through the shift's weights" in reason


@pytest.fixture
def one_forward_per_state(monkeypatch):
    """Run a forward that keeps nothing once per design state.

    A cell whose input the solve does not read (or whose output the table
    only reads) pulls nothing back: its host forward keeps no entry, and its
    result is the same for every output of the cell's problem. The table
    repeats it per output, which the dear solves (a wave port eigenmode's
    coverage search) cannot afford. A forward that keeps an entry for the
    backward, and every backward, still runs per cell.
    """
    forward = solve_vjp._forward
    memo = {}

    def memoized(rule, model, *, keep, request):
        if keep:
            return forward(rule, model, keep=keep, request=request)
        leaves = [
            *jax.tree_util.tree_leaves(model),
            *jax.tree_util.tree_leaves(rule.problem),
        ]
        key = (
            type(rule),
            str(jax.tree_util.tree_structure((model, rule.problem))),
            request.wanted,
            tuple(np.asarray(x).tobytes() for x in leaves),
        )
        if key not in memo:
            memo[key] = forward(rule, model, keep=keep, request=request)
        return memo[key]

    monkeypatch.setattr(solve_vjp, "_forward", memoized)


@pytest.mark.bitwise  # its low-Q lumped mode converges run to run only on one thread
@pytest.mark.parametrize(
    ("name", "leaf"),
    RAISE_TABLE_ITEMS,
    ids=[n if leaf is None else f"{n}-{leaf}" for n, leaf in RAISE_TABLE_ITEMS],
)
@pytest.mark.usefixtures("one_forward_per_state")
def test_raise_table(request, name, leaf):
    """One case (a slow one, one result leaf) per item: it builds only its fixtures."""
    only = {name: _case(request, name)}
    for _, model, problem, out_name, out_path, index, kind in _cells(only):
        if leaf is not None and out_name != leaf:
            continue
        expected = EXPECTED[(name, out_name, kind)]
        objective, x = _objective(model, problem, out_path, index)
        cell = f"{name} {out_name} x {kind}"
        if expected == RAISES:
            # raised while tracing, before any host work
            with pytest.raises(gfem.NonDifferentiableError):  # noqa: PT012
                jax.make_jaxpr(jax.grad(objective))(x)
                pytest.fail(f"{cell}: no NonDifferentiableError")
            continue
        grad = np.asarray(jax.grad(objective)(x))
        assert np.isfinite(grad).all(), cell
        if expected == ZERO:
            assert not grad.any(), cell
        else:
            assert np.abs(grad).max() > 0, cell


class _RuleWithoutShapeTerms(ElectrostaticRule):
    """The electrostatic rule as an operator family without shape terms."""

    form = "electrostatic without shape terms"


def test_unsupported_input_error_names_the_input_and_the_opt_out(model):
    # every v0.1 rule differentiates every input its physics reads: a rule
    # of a form the kernel has no node terms for stands in for a future one
    problem = PROBLEMS["electrostatic"]
    rule = _RuleWithoutShapeTerms(model, problem, gfem.SolverOptions())

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        return solve_vjp(rule, model.replace(mesh=mesh), problem)["capacitance"][0, 0]

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"mesh\.nodes.*stop_gradient"
    ):
        jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


# --- out of scope: pairs v0.1 does not differentiate, not leaf x leaf cells ------


def _out_electrostatic_sheet(request):
    # a partly filled sheet is not an electrostatic object (raised by any
    # solve, so under every transformation too)
    mesh = request.getfixturevalue("sheet_mesh")
    model = _sheet_model(mesh, [])
    problem = gfem.Electrostatic(terminals={"top": "plate2"}, ground="plate1")
    jax.grad(lambda m: gfem.solve(m, problem).capacitance[0, 0])(model)


def _out_piezoelectric_sheet(request):
    # a partly filled sheet needs the slab: a 2D mesh refuses it when the
    # model is built, and a piezoelectric section is always 2D
    model = request.getfixturevalue("piezoelectric_case")["piezoelectric"][0]
    fill = gfem.SheetField(
        fill=np.linspace(0.2, 0.9, len(model.mesh.elements("bottom")))
    )
    model.replace(boundaries=[gfem.Conductor("bottom", 100.0, 1e-6, fill=fill)])


def _sheet_fill_objective(model, problem):
    sheet = model.boundaries[2]

    def objective(fill):
        filled = dataclasses.replace(sheet, fill=gfem.SheetField(fill=fill))
        boundaries = [*model.boundaries[:2], filled]
        s = gfem.solve(model.replace(boundaries=boundaries), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    return objective, sheet.fill.fill


def _out_lumped_port_resistance(request):
    # a port's resistance is its reference impedance: a fixed value (raised
    # by the constructor)
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9)

    def objective(r):
        ports = [gfem.LumpedPort("p1", "port1", "+z", resistance=r), model.ports[1]]
        s = gfem.solve(model.replace(ports=ports), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    jax.grad(objective)(50.0)


def _out_far_field_layer_design(request):
    # the far field reads the field over its layer in the exterior medium: a
    # design there is refused (a side around one cell of the scatterer)
    model, problem = request.getfixturevalue("planewave_case")
    eps = jnp.asarray(model.materials["scat"].eps_r)

    def one_cell(c):
        return ((c > 0.25e-6) & (c < 0.5e-6)).all(axis=1)

    def objective(e):
        moved = model.with_material("scat", gfem.MaterialField(eps_r=e))
        result = gfem.solve(moved, problem)
        ff = gfem.far_field(result, "w", 0.3, 0.2, model=moved, side=one_cell)
        return ff.rcs[0]

    jax.grad(objective)(eps)


#: A beam coming down onto the unit cube, for the beam-coupling rows.
_COUPLING_BEAM = dict(focus=(0.5e-6, 0.5e-6, 1e-6), theta=np.pi, polarization="TM")


def _out_beam_coupling_layer_design(request):
    # the coupling reads the beam's medium over its layer: a design there is
    # refused (a side around one cell of the scatterer)
    model, problem = request.getfixturevalue("planewave_case")
    eps = jnp.asarray(model.materials["scat"].eps_r)
    beam = gfem.GaussianBeam("b", waist=0.3e-6, **_COUPLING_BEAM)

    def one_cell(c):
        return ((c > 0.25e-6) & (c < 0.5e-6)).all(axis=1)

    def objective(e):
        moved = model.with_material("scat", gfem.MaterialField(eps_r=e))
        result = gfem.solve(moved, problem)
        c = gfem.beam_coupling(result, "w", beam, model=moved, side=one_cell)
        return jnp.abs(c[0]) ** 2

    jax.grad(objective)(eps)


def _down_wave(points, frequency):
    """A plane wave coming down along -z, along x, 1 V/m."""
    k = 2 * np.pi * frequency / 299792458.0
    e = np.zeros((len(points), 3), complex)
    e[:, 0] = np.exp(1j * k * points[:, 2])
    return e


def _out_beam_coupling_incident_field_on_a_moved_layer(request):
    # an IncidentField beam is interpolated on the imported mesh: once its
    # layer's nodes move it would move with them, so the host forward refuses
    model, _ = request.getfixturevalue("planewave_case")
    nodes = np.asarray(model.mesh.nodes)
    inner = ((nodes > 1e-15) & (nodes < 1e-6 - 1e-15)).all(axis=1)
    moved = nodes + np.where(inner[:, None], [0.0, 0.0, 1e-9], 0.0)
    model = model.replace(mesh=gfem.Mesh(jnp.asarray(moved), model.mesh._topology))
    dipole = gfem.ElectricDipole(
        "d", position=(0.1e-6, 0.12e-6, 0.14e-6), moment=(1e-30, 0, 0)
    )
    result = gfem.solve(
        model, gfem.Driven(pml_cases.F0, sources=dipole, keep_fields=True)
    )
    gfem.beam_coupling(result, "d", gfem.IncidentField("down", _down_wave), model=model)


def _out_beam_coupling_traced_beam(_request):
    # a beam's parameters are fixed values (raised by the constructor)
    def objective(waist):
        beam = gfem.GaussianBeam("b", waist=waist, **_COUPLING_BEAM)
        return beam.waist

    jax.grad(objective)(0.3e-6)


def _out_floquet_port_offset(_request):
    # a Floquet port's de-embedding offset is static (raised by the constructor)
    def objective(offset):
        gfem.FloquetPort("R", "z_max", polarization="TE", offset=offset)
        return offset

    jax.grad(objective)(1e-7)


def _out_propagate_distance(_request):
    # the transfer function is built once, outside the trace (raised by propagate)
    from gfem import diffraction as dif

    x = (np.arange(8) - 3.5) * 0.5e-6
    values = np.ones((1, 2, 8), dtype=complex)
    field = dif.ApertureField(values, [3e14], x, None, 1.0, 0.0)
    jax.grad(lambda d: dif.propagate(field, d).intensity.sum())(1e-5)


def _out_cell_response_design(request):
    # a cell response tabulates fixed cells: raised before any solve
    from gfem import diffraction as dif

    mesh = pc.ridge_mesh(request.getfixturevalue("tmp_path"))

    def objective(eps):
        def make_cell(_p, _dp):
            model = pc.ridge(mesh)
            n = len(mesh.elements("atom"))
            return model.with_material(
                "atom", gfem.MaterialField(eps_r=eps * jnp.ones(n))
            )

        response = dif.cell_response(
            make_cell, [0.25], frequency=[pc.RIDGE_FREQUENCY], excite=["sub TE"]
        )
        return jnp.sum(response.values.real)

    jax.grad(objective)(5.0)


def _out_bloch_wave_vector_angle(request):
    # the incidence of a Floquet port is a constant of the solve
    edges = (np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    path = sm.write_msh22(
        request.getfixturevalue("tmp_path") / "cell.msh", sm.periodic_cell(edges)
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"cell": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
    )

    def objective(theta):
        return model.bloch_wave_vector("z_max", 2e14, theta=theta)[0]

    jax.grad(objective)(0.3)


def _out_law_under_a_bloch_wave_vector(request):
    model, problem = request.getfixturevalue("driven_periodic_cases")[
        "driven_periodic_bloch"
    ]

    def objective(eps_inf):
        law = dataclasses.replace(TABLE_LAW, eps_inf=eps_inf)
        m = model.with_material("cell", gfem.Material(eps_r=law))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(2.0)


def _out_law_of_a_blend_end(_request):
    # the ends of a blend are fixed: a traced law raises while
    # the field is built, with the remedy
    def objective(eps_inf):
        law = dataclasses.replace(TABLE_LAW, eps_inf=eps_inf)
        field = gfem.MaterialField.interpolate(
            jnp.array([0.5]),
            gfem.Material(eps_r=2.0),
            gfem.Material(eps_r=law),
            scheme="index",
        )
        return jnp.sum(field.blend)

    jax.make_jaxpr(jax.grad(objective))(2.0)


def _out_material_axes(request):
    # the principal directions are static data, never a design input
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9)
    n = len(model.mesh.elements("dielectric"))

    def objective(axes):
        field = gfem.MaterialField(eps_r=jnp.full((n, 3), 2.0), axes=axes)
        m = model.with_material("dielectric", field)
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.grad(objective)(jnp.eye(3))


def _out_anisotropic_material_value(request):
    # a Material is a constant, anisotropic or not: the error names the
    # field that holds its principal values as leaves
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9)

    def objective(eps):
        m = model.with_material("dielectric", gfem.Material(eps_r=(eps, 2.5, 3.0)))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.grad(objective)(2.0)


def _out_nodes_beside_a_tensor_absorbing_face(request):
    # the node pass transports scalar absorbing coefficients only
    driven = request.getfixturevalue("driven_model")
    model = driven.replace(
        materials={"dielectric": gfem.Material(eps_r=(2.0, 2.5, 3.0))},
        boundaries=[*driven.boundaries, gfem.Absorbing("side")],
    )
    problem = gfem.Driven(frequency=5e9)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        s = gfem.solve(model.replace(mesh=mesh), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_line_port_mode_law_parameter(request):
    # a line port's mode does not differentiate a law's parameters: a silent
    # zero would pass for the true derivative
    path = request.getfixturevalue("tmp_path_factory").mktemp("law_line_port")
    mesh = gfem.Mesh.from_gmsh(lpc.write_guide(path / "guide.msh"), length_unit="mm")

    def objective(eps_inf):
        law = gfem.PermittivityLaw(
            eps_inf=eps_inf,
            terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=200e9, damping=2e9)],
        )
        model = gfem.Model(
            mesh,
            materials={
                "clad": gfem.Material(),
                "core1": gfem.Material(eps_r=law),
                "core2": gfem.Material(eps_r=4.0),
            },
            boundaries=[gfem.PEC("side")],
            ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
        )
        result = gfem.solve(
            model,
            gfem.PortMode("p1", lpc.FREQUENCY, polarization="Ez"),
            options=gfem.SolverOptions(threads=1),
        )
        return jnp.imag(result.gamma[0])

    jax.grad(objective)(4.0)


def _out_port_mode_residual(_request):
    # the backward error of a mode is a diagnostic: its cotangent raises
    def residual(eps):
        mode, _ = _plate_mode(eps)
        return jnp.sum(mode.residual)

    jax.make_jaxpr(jax.grad(residual))(4.0)


def _out_port_mode_dispersion(_request):
    # the group index and dispersion have no backward pass, beside gamma's
    def group_index(eps):
        mode, _ = _plate_mode(eps)
        return jnp.real(mode.n_g[0])

    jax.make_jaxpr(jax.grad(group_index))(4.0)


def _out_port_mode_field_at_nodes(_request):
    # `PortModeField.at` reads fixed points: traced nodes are not available
    _, model = _plate_mode(4.0)

    def field(nodes):
        moved = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        result = gfem.solve(model.replace(mesh=moved), gfem.PortMode("p1", 10e9))
        return jnp.real(result.e_field.at([20e-6, 5e-6, 0.0])[0])

    jax.grad(field)(jnp.asarray(model.mesh.nodes))


def _out_characteristic_impedance_of_a_traced_mode(_request):
    def impedance(eps):
        mode, _ = _plate_mode(eps)
        return jnp.real(
            gfem.characteristic_impedance(mode, voltage_path=PORT_FACE_PATH)[0]
        )

    jax.grad(impedance)(4.0)


def _out_electro_optic_shift_by_the_optical_eps(request):
    # the mode is a constant of the shift
    import electro_optic_cases as eo

    path = request.getfixturevalue("tmp_path_factory").mktemp("eo") / "lchannel.msh"
    mesh = eo.l_core(path, 0.5, 0.15)
    static = gfem.Model(
        mesh,
        materials={
            r: gfem.Material(eps_r=eo.LN_STATIC, axes=eo.X_CUT)
            if r in ("core", "foot")
            else gfem.Material(eps_r=3.9)
            for r in mesh._region_names().values()
        },
    )
    potential = gfem.solve(
        static,
        gfem.Electrostatic(terminals={"t": ["top"]}, ground=["bottom"]),
        options=gfem.SolverOptions(order=2, linear_solver="direct"),
    ).potential["t"]

    def shift(eps):
        crystal = gfem.MaterialField(
            eps_r=eps, axes=eo.X_CUT, electro_optic_tensor=jnp.asarray(eo.R_LN)
        )
        model = gfem.Model(
            mesh,
            materials={
                "clad": gfem.Material(eps_r=eo.N_OX**2),
                "core": crystal,
                "foot": crystal,
            },
            ports=[gfem.WavePort("p", "front")],
        )
        return gfem.electro_optic_shift(
            model, potential, port="p", frequency=eo.FREQUENCY
        ).delta_n_eff_per_volt

    jax.grad(shift)(jnp.asarray(eo.LN_OPTICAL))


def _out_driven_port_mode_residual(request):
    # the backward error of a mode is a diagnostic: a cotangent on it raises
    # where a wanted input moves the mode
    model, problem = request.getfixturevalue("driven_port_face_cases")[
        "driven_wave_face"
    ]

    def objective(eps):
        m = model.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return jnp.sum(gfem.solve(m, problem).port_modes["p1"].residual)

    jax.make_jaxpr(jax.grad(objective))(
        jnp.full(len(model.mesh.elements("dielectric")), 2.0)
    )


def _out_port_face_design_under_bloch(**problem):
    def call(request):
        model = request.getfixturevalue("bloch_wave_port_model")
        driven = gfem.Driven(frequency=10e9, wave_vector=(1e2, 0.0, 0.0), **problem)

        def objective(eps):
            m = model.with_material("cell", gfem.MaterialField(eps_r=eps))
            m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
            return jnp.sum(jnp.abs(gfem.solve(m, driven).s.values) ** 2)

        jax.make_jaxpr(jax.grad(objective))(
            jnp.full(len(model.mesh.elements("cell")), 2.0)
        )

    return call


def _moved_port_objective(model, value):
    """``value(result)`` with one node of port1 moved by ``dx`` along x."""
    mesh = model.mesh
    node = int(mesh.triangles[mesh.triangle_tags == mesh.region("port1").tag][0, 0])

    def objective(dx):
        x = jnp.asarray(mesh.nodes).at[node, 0].add(dx)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        return value(model.replace(mesh=moved))

    return objective


def _out_driven_port_nodes_moved(request):
    # shape gradients hold every port's geometry: raised on the host
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9)
    objective = _moved_port_objective(
        model, lambda m: jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)
    )
    jax.grad(objective)(1e-6)


def _out_eigenmode_port_nodes_moved(request):
    model = request.getfixturevalue("eigen_model")
    problem = gfem.Eigenmode(target=17e9)
    objective = _moved_port_objective(
        model, lambda m: jnp.real(gfem.solve(m, problem).omega[0]) * 1e-9
    )
    jax.grad(objective)(1e-6)


def _out_eigenmode_field_port_nodes_moved(request):
    # the ratios' node gradient holds the port surfaces too: a moved port node
    # raises on the host, as for omega
    model = request.getfixturevalue("eigen_model")
    problem = gfem.Eigenmode(target=17e9)
    objective = _moved_port_objective(
        model, lambda m: gfem.solve(m, problem).q_port["p1"][0]
    )
    jax.grad(objective)(1e-6)


def _out_eigenmode_material_on_a_wave_port_face(request):
    # the wave port's mode reads the air region: a constant of the solve
    model = request.getfixturevalue("eigen_wave_model")
    n = len(model.mesh.elements("air"))
    problem = gfem.Eigenmode(target=EIGEN_WAVE_TARGET)

    def objective(eps):
        m = model.with_material("air", gfem.MaterialField(eps_r=eps))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return jnp.real(gfem.solve(m, problem).omega[0]) * 1e-9

    jax.make_jaxpr(jax.grad(objective))(jnp.full(n, 1.0))


def _out_eigenmode_law_on_a_wave_port_face(request):
    model = request.getfixturevalue("eigen_wave_model")
    problem = gfem.Eigenmode(target=EIGEN_WAVE_TARGET)

    def objective(eps_inf):
        law = dataclasses.replace(TABLE_LAW, eps_inf=eps_inf)
        m = model.with_material("air", gfem.Material(eps_r=law))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return 1 / gfem.solve(m, problem).q_port["p1"][0]

    jax.make_jaxpr(jax.grad(objective))(2.0)


def _out_eigenmode_impedance_beside_a_wave_port(request):
    # a sheet on the walls shares an edge with the port face: its values
    # are constants of the solve there
    model = request.getfixturevalue("eigen_wave_model")
    problem = gfem.Eigenmode(target=EIGEN_WAVE_TARGET)

    def objective(inductance):
        sheet = gfem.Impedance("walls", resistance=1e-3, inductance=inductance)
        m = model.replace(boundaries=[sheet])
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return jnp.real(gfem.solve(m, problem).omega[0]) * 1e-9

    jax.make_jaxpr(jax.grad(objective))(1e-12)


def _out_eigenmode_wave_port_nodes_moved(request):
    # the wave port's face moves: raised on the host, for omega and q_port
    model = request.getfixturevalue("eigen_wave_model")
    model = model.replace(
        materials={**model.materials, "dielectric": gfem.Material(eps_r=4.0)}
    )
    problem = gfem.Eigenmode(target=EIGEN_WAVE_TARGET)
    mesh = model.mesh
    node = int(mesh.triangles[mesh.triangle_tags == mesh.region("port").tag][0, 0])

    def objective(dx):
        x = jnp.asarray(mesh.nodes).at[node, 0].add(dx)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        result = gfem.solve(model.replace(mesh=moved), problem)
        return jnp.real(result.omega[0]) * 1e-9 + 1 / result.q_port["p1"][0]

    jax.grad(objective)(1e-6)


def _out_greedy_adaptive_sweep(request):
    # the sample set must be frozen (Driven(samples=...)) under a transform
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=(4e9, 5e9, 6e9), sweep="adaptive")
    eps = model.materials["dielectric"].eps_r

    def objective(e):
        m = model.with_material("dielectric", gfem.MaterialField(eps_r=e))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_forward_mode(request):
    # reverse mode only: jax.jvp of a solve is JAX's own TypeError
    model = request.getfixturevalue("model")
    problem = PROBLEMS["electrostatic"]
    eps = model.materials["oxide"].eps_r

    def objective(e):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=e))
        return gfem.solve(m, problem).capacitance[0, 0]

    jax.jvp(objective, (eps,), (jnp.ones_like(eps),))


def _out_probe_in_the_node_coordinates(request):
    # measurements read fields on the mesh: no node gradients through them
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9, keep_fields=True)
    points = np.array([[2.0, 1.0, 0.5]]) * 1e-3

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        frozen = model.replace(mesh=jax.lax.stop_gradient(mesh))
        field = gfem.solve(frozen, problem).e_field(5e9, "p1")
        field = dataclasses.replace(field, mesh=mesh)
        return jnp.sum(jnp.abs(field.at(points)) ** 2)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_electric_energy_of_a_reduced_field(request):
    # an adaptive sweep's basis holds no adjoint of the energy
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(
        frequency=(18e9, 20e9, 22e9),
        sweep="adaptive",
        samples=(18e9, 22e9),
        keep_fields=20e9,
    )
    eps = model.materials["core"].eps_r

    def objective(e):
        m = model.with_material("core", gfem.MaterialField(eps_r=e))
        field = gfem.solve(m, problem).e_field(20e9, "p1")
        return gfem.electric_energy(field, model=m)["core"]

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_power_flow_of_a_material_in_its_layer(request):
    # the side is the leads: the core field is in the layer power_flow reads
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(frequency=20e9, keep_fields=True)
    eps = model.materials["core"].eps_r

    def objective(e):
        m = model.with_material("core", gfem.MaterialField(eps_r=e))
        field = gfem.solve(m, problem).e_field(20e9, "p1")
        return gfem.power_flow(field, "dielectric", model=m)

    jax.make_jaxpr(jax.grad(objective))(eps)


def _cut_off_wave(points, frequency):
    """A wave along +z, zero from x = 0.7 um on: none where the slab meets the layer."""
    k = 2 * np.pi * frequency / pml_cases.C0
    x = points[:, 0] * 1e6
    window = np.where(x < 0.7, np.cos(np.clip(x - 0.6, 0, 0.1) * 5 * np.pi) ** 2, 0)
    return (np.exp(-1j * k * points[:, 2]) * window)[:, None] * np.array([0, 1, 0])


def _out_cross_sections_of_a_design_in_the_truncation_layer(request):
    # a slab at the cube's +x end touches the layer: the cross sections read
    # its elements, so its material cannot be differentiated there
    def volume(c):
        if c[0] > 0.75 and ((c > 0) & (c < 1)).all():
            return "slab"
        return "air" if ((c > 0) & (c < 1)).all() else "pml"

    path = pml_cases.box(
        request.getfixturevalue("tmp_path") / "slab.msh",
        ("-x", "+x", "-y", "+y", "-z", "+z"),
        thickness=0.25,
        cells=4,
        volume=volume,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "pml": gfem.Material(),
            "slab": _field(mesh, "slab"),
        },
        pml=gfem.PML("pml"),
        background=gfem.Material(),
    )
    wave = gfem.IncidentField("w", _cut_off_wave)
    problem = gfem.Driven(frequency=pml_cases.F0, sources=wave, keep_fields=True)
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=1))

    def objective(e):
        m = model.with_material("slab", gfem.MaterialField(eps_r=e))
        xs = gfem.cross_sections(result, "w", model=m, intensity=1.0)
        return xs.scattering[0]

    jax.make_jaxpr(jax.grad(objective))(model.materials["slab"].eps_r)


def _out_cross_sections_in_the_node_coordinates(request):
    # the absorption reads the incident load, whose interpolant has no node
    # derivative: the solve of the incident column refuses the nodes first
    model, problem = request.getfixturevalue("planewave_case")

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        moved = model.replace(mesh=mesh)
        result = gfem.solve(moved, problem)
        return gfem.cross_sections(result, "w", model=moved, intensity=1.0).absorption[
            0
        ]

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _multipoles_cube(request):
    model, problem = request.getfixturevalue("planewave_case")
    return model, gfem.solve(model, problem, options=gfem.SolverOptions(order=1))


def _out_multipoles_in_the_node_coordinates(request):
    # the scattered field subtracts the incident interpolant: the solve of the
    # incident column refuses the nodes first
    model, problem = request.getfixturevalue("planewave_case")

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        moved = model.replace(mesh=mesh)
        result = gfem.solve(moved, problem)
        return gfem.multipoles(result, "w", model=moved).scattering[0]

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_multipoles_in_the_node_coordinates_at_fixed_coefficients(request):
    # the kernel's node pass would drop the incident interpolant's term: the
    # entry refuses traced nodes before reading the field
    model, result = _multipoles_cube(request)

    def objective(x):
        leaves, treedef = jax.tree_util.tree_flatten_with_path(result)
        moved = [
            x if jax.tree_util.keystr(p).endswith("mesh.nodes") else v
            for p, v in leaves
        ]
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [x]
        )
        return gfem.multipoles(
            jax.tree_util.tree_unflatten(treedef, moved),
            "w",
            model=model.replace(mesh=mesh),
        ).scattering[0]

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_multipoles_of_a_design_in_the_truncation_layer(request):
    # a slab at the cube's +x end touches the layer: the multipoles read its
    # elements through the far-field layer, so its material cannot be
    # differentiated there
    def volume(c):
        if c[0] > 0.75 and ((c > 0) & (c < 1)).all():
            return "slab"
        return "air" if ((c > 0) & (c < 1)).all() else "pml"

    path = pml_cases.box(
        request.getfixturevalue("tmp_path") / "slab.msh",
        ("-x", "+x", "-y", "+y", "-z", "+z"),
        thickness=0.25,
        cells=4,
        volume=volume,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "pml": gfem.Material(),
            "slab": _field(mesh, "slab"),
        },
        pml=gfem.PML("pml"),
        background=gfem.Material(),
    )
    wave = gfem.IncidentField("w", _cut_off_wave)
    problem = gfem.Driven(frequency=pml_cases.F0, sources=wave, keep_fields=True)
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=1))

    def objective(e):
        m = model.with_material("slab", gfem.MaterialField(eps_r=e))
        return gfem.multipoles(result, "w", model=m, intensity=1.0).scattering[0]

    jax.make_jaxpr(jax.grad(objective))(model.materials["slab"].eps_r)


def _out_multipoles_of_a_traced_origin(request):
    model, result = _multipoles_cube(request)

    def objective(origin):
        return gfem.multipoles(result, "w", model=model, origin=origin).scattering[0]

    jax.make_jaxpr(jax.grad(objective))(jnp.full(3, 0.5e-6))


def _out_multipoles_of_a_traced_intensity(request):
    model, _ = request.getfixturevalue("planewave_case")
    wave = gfem.IncidentField("w", _down_wave)
    problem = gfem.Driven(frequency=pml_cases.F0, sources=wave, keep_fields=True)
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=1))

    def objective(intensity):
        return gfem.multipoles(
            result, "w", model=model, intensity=intensity
        ).scattering[0]

    jax.make_jaxpr(jax.grad(objective))(1.0)


def _out_current_moments_of_a_traced_design(request):
    model, result = _multipoles_cube(request)

    def objective(e):
        m = model.with_material("scat", gfem.MaterialField(eps_r=e))
        return jnp.abs(gfem.current_moments(result, "w", model=m).toroidal_dipole).sum()

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.materials["scat"].eps_r))


def _out_power_flow_of_a_reduced_field(request):
    # an adaptive sweep's basis holds no adjoint of the power
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(
        frequency=(18e9, 20e9, 22e9),
        sweep="adaptive",
        samples=(18e9, 22e9),
        keep_fields=20e9,
    )
    eps = model.materials["core"].eps_r

    def objective(e):
        m = model.with_material("core", gfem.MaterialField(eps_r=e))
        field = gfem.solve(m, problem).e_field(20e9, "p1")
        return gfem.power_flow(field, "core", model=m)

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_heat_flow_in_the_node_coordinates(request):
    # the residual integrates on the elements the nodes move
    path = request.getfixturevalue("tmp_path") / "heat.msh"
    mesh = tc.section(
        path,
        [(0.0, 0.0, 2.0, 1.0, "slab", None)],
        0.5,
        [(lambda p: abs(p[1]) < 1e-9, "bottom")],
    )
    model = gfem.Model(
        mesh,
        materials={"slab": gfem.Material(thermal_conductivity=1.4)},
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
    )
    heat = gfem.HeatSource.uniform("slab", power_per_length=1.0)
    result = gfem.solve(model, gfem.SteadyThermal(heat))

    def objective(nodes):
        moved = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        return gfem.heat_flow_per_length(
            result, "bottom", model=model.replace(mesh=moved)
        )

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_dissipated_power_in_the_node_coordinates(request):
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(frequency=20e9, keep_fields=True)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        frozen = model.replace(mesh=jax.lax.stop_gradient(mesh))
        field = gfem.solve(frozen, problem).e_field(20e9, "p1")
        field = dataclasses.replace(field, mesh=mesh)
        return gfem.dissipated_power(field, model=frozen)["core"]

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_dissipated_power_of_a_reduced_field(request):
    # an adaptive sweep's basis holds no adjoint of the dissipated power
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(
        frequency=(18e9, 20e9, 22e9),
        sweep="adaptive",
        samples=(18e9, 22e9),
        keep_fields=20e9,
    )
    eps = model.materials["core"].eps_r

    def objective(e):
        m = model.with_material("core", gfem.MaterialField(eps_r=e))
        field = gfem.solve(m, problem).e_field(20e9, "p1")
        return gfem.dissipated_power(field, model=m)["core"]

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_dissipated_power_in_the_frequency(request):
    # the frequency is a fixed input of the driven solve
    model = request.getfixturevalue("wave_model")
    field = gfem.solve(model, gfem.Driven(frequency=20e9, keep_fields=True)).e_field(
        20e9, "p1"
    )

    def objective(scale):
        moved = dataclasses.replace(field, omega=field.omega * scale)
        return gfem.dissipated_power(moved, model=model)["core"]

    jax.make_jaxpr(jax.grad(objective))(1.0)


def _out_electromagnetic_heating_in_the_frequency(request):
    # the heat's frequency picks a kept field: a fixed input, as the solve's
    model = request.getfixturevalue("wave_model")
    result = gfem.solve(model, gfem.Driven(frequency=20e9, keep_fields=True))

    def objective(f):
        heat = gfem.electromagnetic_heating(
            result, model=model, frequency=f, incident_power={"p1": 1.0}
        )
        return jnp.sum(heat._density_on(model.mesh))

    jax.make_jaxpr(jax.grad(objective))(20e9)


def _out_electromagnetic_heating_of_a_reduced_field(request):
    # the heat reads the adaptive sweep's field as the dissipated power does
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(
        frequency=(18e9, 20e9, 22e9),
        sweep="adaptive",
        samples=(18e9, 22e9),
        keep_fields=20e9,
    )
    eps = model.materials["core"].eps_r

    def objective(e):
        m = model.with_material("core", gfem.MaterialField(eps_r=e))
        heat = gfem.electromagnetic_heating(
            gfem.solve(m, problem),
            model=m,
            frequency=20e9,
            incident_power={"p1": 1.0},
        )
        return jnp.sum(heat._density_on(m.mesh))

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_conductor_conductivity(request):
    # a Conductor's conductivity is a fixed value (raised by the
    # constructor): a sheet's fill or an Impedance's resistance is the design
    model, _ = request.getfixturevalue("heat_cases")["dissipated_power"]
    problem = gfem.Driven(frequency=HEAT_FREQUENCY, keep_fields=True)

    def objective(sigma):
        boundaries = [*model.boundaries[:2], gfem.Conductor("sheet", sigma, 1e-6)]
        m = model.replace(boundaries=boundaries)
        field = gfem.solve(m, problem).e_field(HEAT_FREQUENCY, "p1")
        return gfem.dissipated_power(field, model=m)["sheet"]

    jax.grad(objective)(100.0)


def _out_traced_pml_knob(knob):
    # the stretch is fixed: its knobs are plain numbers, never design inputs
    def call(request):
        model = request.getfixturevalue("pml_model")

        def objective(value):
            layer = gfem.PML("pml", **{knob: value})
            result = gfem.solve(model.replace(pml=layer), gfem.Driven(pml_cases.F0))
            return jnp.sum(jnp.abs(result.s.values) ** 2)

        start = 1e-6 if knob == "target_reflection" else pml_cases.F0
        jax.make_jaxpr(jax.grad(objective))(start)

    return call


def _out_driven_traced_wave_vector(request):
    # the wave vector is static: a Bloch phase is not a design variable
    model, _ = request.getfixturevalue("driven_periodic_cases")["driven_periodic_bloch"]

    def objective(kx):
        problem = gfem.Driven(frequency=5e9, wave_vector=(kx, 0.0, 0.0))
        return jnp.sum(jnp.abs(gfem.solve(model, problem).s.values) ** 2)

    jax.grad(objective)(pc.C1_K[0])


def _out_driven_traced_wave_vector_frequency(request):
    # a fixed-angle sweep's reference frequency is static, as the wave vector
    model = request.getfixturevalue("floquet_cell")
    k = model.bloch_wave_vector(
        "z_max", pc.F1_FREQUENCY, theta=pc.F1_THETA, phi=pc.F1_PHI
    )

    def objective(f_ref):
        problem = gfem.Driven(
            frequency=[pc.F1_FREQUENCY, 1.1 * pc.F1_FREQUENCY],
            wave_vector=k,
            wave_vector_frequency=f_ref,
        )
        return jnp.sum(jnp.abs(gfem.solve(model, problem).s.values) ** 2)

    jax.grad(objective)(pc.F1_FREQUENCY)


def _out_periodic_pair_broken(request):
    # a node move that breaks a pair has no forward: raised on the host
    model, problem = request.getfixturevalue("driven_periodic_cases")[
        "driven_periodic_plain"
    ]
    mesh = model.mesh
    node = int(model._periodic[0].second_nodes[0])

    def objective(dy):
        x = jnp.asarray(mesh.nodes).at[node, 1].add(dy)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(model.replace(mesh=moved), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    jax.grad(objective)(1e-5)


def _out_morph_of_a_paired_surface(request):
    model, _ = request.getfixturevalue("driven_periodic_cases")["driven_periodic_plain"]
    boundary_morph(model, "x_max")


def _out_bloch_adaptive_sweep(request):
    model, _ = request.getfixturevalue("driven_periodic_cases")["driven_periodic_bloch"]
    problem = gfem.Driven(
        frequency=(4e9, 5e9, 6e9),
        sweep="adaptive",
        samples=(4e9, 6e9),
        wave_vector=pc.C1_K,
    )
    eps = model.materials["cell"].eps_r

    def objective(e):
        m = model.with_material("cell", gfem.MaterialField(eps_r=e))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.grad(objective)(eps)


# --- Floquet path rows -----------------------------------------------------------


@pytest.fixture(scope="module")
def floquet_cell(tmp_path_factory):
    """F1 (tests/periodic_cases.py): a slab with Floquet ports on both faces."""
    return pc.f1(pc.f1_mesh(tmp_path_factory.mktemp("f1")))


def _slab(model, e):
    """``model`` with the slab's eps_r ``e`` on every element, traceable."""
    n = len(model.mesh.elements("slab"))
    return model.with_material("slab", gfem.MaterialField(eps_r=jnp.full(n, e)))


def _floquet_slab_objective(model, problem, options=None):
    def objective(e):
        m = _slab(model, e)
        return jnp.sum(jnp.abs(gfem.solve(m, problem, options=options).s.values) ** 2)

    return objective


def _out_material_of_a_floquet_half_space(request):
    # the medium behind a Floquet surface fixes its orders: raised while tracing
    model = request.getfixturevalue("floquet_cell")
    n = len(model.mesh.elements("air"))
    problem = gfem.Driven(frequency=pc.F1_FREQUENCY)

    def objective(e):
        m = model.with_material("air", gfem.MaterialField(eps_r=jnp.full(n, e)))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(1.0)


def _out_floquet_port_nodes_moved(request):
    # shape gradients hold a Floquet surface too: raised on the host
    model = request.getfixturevalue("floquet_cell")
    mesh = model.mesh
    face = np.unique(mesh.triangles[mesh.triangle_tags == mesh.region("z_max").tag])
    x = np.asarray(mesh.imported_nodes)[face, :2]
    a = pc.F1_A * pc.UM
    (inside,) = np.nonzero(((x > 1e-3 * a) & (x < (1 - 1e-3) * a)).all(axis=1))
    node = int(face[inside[0]])
    problem = gfem.Driven(frequency=pc.F1_FREQUENCY)

    def objective(dx):
        nodes = jnp.asarray(mesh.nodes).at[node, 0].add(dx)
        moved = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(mesh), [nodes]
        )
        s = gfem.solve(model.replace(mesh=moved), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    jax.grad(objective)(1e-9)


def _out_floquet_adaptive_sweep(request):
    # the adaptive sweep has no Floquet port path: refused before assembly
    model = request.getfixturevalue("floquet_cell")
    f = (0.9 * pc.F1_FREQUENCY, pc.F1_FREQUENCY, 1.1 * pc.F1_FREQUENCY)
    problem = gfem.Driven(frequency=f, sweep="adaptive", samples=(f[0], f[2]))
    jax.grad(_floquet_slab_objective(model, problem))(pc.F1_EPS_SLAB)


def _out_floquet_petsc(request):
    # nor has PETSc: refused before the linear solver is resolved
    model = request.getfixturevalue("floquet_cell")
    options = gfem.SolverOptions(linear_solver="petsc")
    problem = gfem.Driven(frequency=pc.F1_FREQUENCY)
    jax.grad(_floquet_slab_objective(model, problem, options))(pc.F1_EPS_SLAB)


def _out_diffraction_polarization(request):
    # the incident polarization is a constant of the measurement
    from gfem import diffraction as dif

    model, problem = request.getfixturevalue("driven_floquet_cases")[
        "driven_floquet_plain"
    ]
    options = gfem.SolverOptions(order=2, threads=2, linear_solver="direct")
    result = gfem.solve(model, problem, options=options)

    def objective(a):
        eff = dif.efficiencies(result, "z_max", polarization=(a, 1j), model=model)
        return eff.values.sum()

    jax.make_jaxpr(jax.grad(objective))(1.0)


def _out_angle_sweep_traced_angle(request):
    # each incidence is a constant of its solve: refused before any solve
    from gfem import diffraction as dif

    model, _ = request.getfixturevalue("driven_floquet_cases")["driven_floquet_plain"]

    def objective(t):
        sweep = dif.angle_sweep(model, [pc.F1_FREQUENCY], surface="z_max", theta=t)
        return sweep[0].s.values.real.sum()

    jax.grad(objective)(0.3)


def _out_eigenmode_floquet_port(request):
    # an order's propagation constant at a complex frequency: refused
    model = request.getfixturevalue("floquet_cell")
    problem = gfem.Eigenmode(target=pc.F1_FREQUENCY, n_modes=1)

    def objective(e):
        return jnp.real(gfem.solve(_slab(model, e), problem).omega[0])

    jax.grad(objective)(pc.F1_EPS_SLAB)


# --- periodic eigenmode rows ----------------------------------------------------------


@pytest.fixture(scope="module")
def bragg_cell(tmp_path_factory):
    """The C3 Bragg line (tests/periodic_cases.py): one periodic pair."""
    return pc.c3(pc.c3_mesh(tmp_path_factory.mktemp("bragg")))


def _bloch_eigenmode(kx):
    return gfem.Eigenmode(target=5e9, wave_vector=(kx, 0.0, 0.0))


def _out_eigenmode_traced_wave_vector(request):
    # the wave vector is a static problem field, never a leaf
    model = request.getfixturevalue("bragg_cell")

    def objective(kx):
        return jnp.real(gfem.solve(model, _bloch_eigenmode(kx)).omega[0])

    jax.make_jaxpr(jax.grad(objective))(math.pi / (pc.C3_A * pc.MM))


def _out_eigenmode_band_group_velocity(request):
    # the band group velocity is read from the modes: a cotangent on it raises
    # at the trace, before any host work
    model = request.getfixturevalue("bragg_cell")
    n = len(model.mesh.elements("low"))
    problem = _bloch_eigenmode(math.pi / (pc.C3_A * pc.MM))

    def objective(e):
        m = model.with_material("low", gfem.MaterialField(eps_r=e))
        return jnp.real(gfem.solve(m, problem).v_g[0, 0])

    jax.make_jaxpr(jax.grad(objective))(jnp.full(n, pc.C3_EPS_LOW))


def _out_band_diagram_staged(request):
    # a band diagram orders and connects the bands from concrete values: it
    # runs under jax.grad, and raises at the trace inside jax.jit
    model = request.getfixturevalue("bragg_cell")
    n = len(model.mesh.elements("low"))
    path = np.array([[math.pi / (pc.C3_A * pc.MM), 0.0, 0.0]])

    def objective(e):
        m = model.with_material("low", gfem.MaterialField(eps_r=e))
        return gfem.band_diagram(m, path, target=5e9, n_bands=1).frequency[0, 0]

    jax.make_jaxpr(objective)(jnp.full(n, pc.C3_EPS_LOW))


def _out_band_diagram_wave_vectors(request):
    # the wave vectors and the target are fixed numbers, never a leaf
    model = request.getfixturevalue("bragg_cell")

    def objective(kx):
        path = jnp.stack([kx, 0.0, 0.0])[None]
        return gfem.band_diagram(model, path, target=5e9, n_bands=1).frequency[0, 0]

    jax.make_jaxpr(jax.grad(objective))(math.pi / (pc.C3_A * pc.MM))


def _out_bloch_eigenmode_on_bddc(request):
    # two-sided refinement solves the transposed system on the shift's
    # factors: a Bloch eigenmode solve refuses BDDC + GMRES before assembly
    model = request.getfixturevalue("bragg_cell")
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    k = math.pi / (pc.C3_A * pc.MM)
    gfem.solve(model, _bloch_eigenmode(k), options=options)


# --- eigenmode rows of a permittivity law ---------------------------------------------


def _eigen_law_objective(request, read, *, model=None, problem=None):
    """``read(result)`` as a function of the core law's eps_inf."""
    model = model or request.getfixturevalue("eigen_law_model")
    problem = problem or gfem.Eigenmode(target=17e9)

    def objective(eps_inf):
        law = dataclasses.replace(TABLE_LAW, eps_inf=eps_inf)
        m = model.with_material("core", gfem.Material(eps_r=law))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return read(gfem.solve(m, problem))

    return objective


def _out_eigenmode_probe_by_a_law_parameter(request):
    # a phase-dependent measurement of the normalized mode field
    point = np.array([2.1, 0.9, 0.4]) * 1e-3
    objective = _eigen_law_objective(
        request, lambda r: jnp.real(r.e_fields[0].at(point)[2])
    )
    jax.make_jaxpr(jax.grad(objective))(2.0)


def _out_eigenmode_absorbing_law_face(request):
    # the analytic sqrt(eps(omega)) wall would differ from the driven one
    model = request.getfixturevalue("eigen_law_model")
    model = model.replace(
        boundaries=[*model.boundaries, gfem.Absorbing("side", order=1)]
    )
    gfem.solve(model, gfem.Eigenmode(target=17e9))


def _periodic_law_model(request):
    model = request.getfixturevalue("bragg_cell")
    return model.with_material("low", gfem.Material(eps_r=TABLE_LAW))


def _periodic_law_objective(request, problem):
    model = _periodic_law_model(request)

    def objective(eps_inf):
        law = dataclasses.replace(TABLE_LAW, eps_inf=eps_inf)
        m = model.with_material("low", gfem.Material(eps_r=law))
        return jnp.real(gfem.solve(m, problem).omega[0]) * 1e-9

    return objective


def _out_periodic_eigenmode_by_a_law_parameter(request):
    objective = _periodic_law_objective(
        request, gfem.Eigenmode(target=pc.C3_PLAIN_TARGET)
    )
    jax.make_jaxpr(jax.grad(objective))(2.0)


# --- eigenmode mode-field rows --------------------------------------------------------


def _eigen_ratio_objective(model, problem, options, read):
    """``read(result)`` as a function of the dielectric's permittivity."""
    eps = model.materials["dielectric"].eps_r

    def objective(e):
        m = model.with_material(
            "dielectric", dataclasses.replace(model.materials["dielectric"], eps_r=e)
        )
        return read(gfem.solve(m, problem, options=options))

    return objective, eps


def _trace_eigen_q_port(request, **options):
    # the static refusals: raised while tracing, before any host work
    model = request.getfixturevalue("eigen_model")
    objective, eps = _eigen_ratio_objective(
        model,
        gfem.Eigenmode(target=17e9),
        gfem.SolverOptions(**options),
        lambda r: jnp.sum(r.q_port["p1"]),
    )
    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_eigenmode_error(request):
    model = request.getfixturevalue("eigen_model")
    objective, eps = _eigen_ratio_objective(
        model,
        gfem.Eigenmode(target=17e9),
        gfem.SolverOptions(),
        lambda r: jnp.sum(r.error),
    )
    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_eigenmode_field_bddc(request):
    _trace_eigen_q_port(request, linear_solver="bddc_gmres")


def _out_eigenmode_field_petsc(request):
    # the rule resolves the PETSc linear solver before tracing: without a
    # usable PETSc stack that is the error, not the refusal
    from gfem import _petsc

    try:
        _petsc.require_petsc("linear_solver")
    except gfem.SolverUnavailableError as err:
        pytest.skip(str(err))
    _trace_eigen_q_port(request, linear_solver="petsc")


def _out_eigenmode_field_cuda(request):
    # the GPU is first used by the forward, which the trace does not run
    _trace_eigen_q_port(request, device="cuda")


def _out_eigenmode_field_tolerance(request):
    _trace_eigen_q_port(request, eigen_tolerance=1e-8)


def _out_eigenmode_mode_field_nodes(request):
    # the normalized field (so Field.at, interface_participation) by the node
    # coordinates: never in reach (finite differences over morphed meshes)
    model = request.getfixturevalue("eigen_model")
    problem = gfem.Eigenmode(target=17e9)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        field = gfem.solve(model.replace(mesh=mesh), problem).e_fields[0]
        return jnp.sum(jnp.abs(field.coefficients) ** 2)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _bragg_bulk(request, problem):
    model = request.getfixturevalue("bragg_cell")
    n = len(model.mesh.elements("low"))

    def objective(e):
        m = model.with_material("low", gfem.MaterialField(eps_r=e))
        return gfem.solve(m, problem).bulk_participation["low"][0]

    jax.make_jaxpr(jax.grad(objective))(jnp.full(n, pc.C3_EPS_LOW))


def _out_eigenmode_field_periodic(request):
    _bragg_bulk(request, gfem.Eigenmode(target=pc.C3_PLAIN_TARGET))


def _out_eigenmode_field_bloch(request):
    _bragg_bulk(request, _bloch_eigenmode(math.pi / (pc.C3_A * pc.MM)))


@pytest.fixture(scope="module")
def corner_cavity(tmp_path_factory):
    """The mirrored cavity with a corner region (an energy share)."""
    return mirrored_cavity(tmp_path_factory.mktemp("corner"), corner=True)


def _corner_share(request, *, delta, n_modes, read):
    mesh = request.getfixturevalue("corner_cavity")
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=n_modes)

    def objective(eps):
        result = gfem.solve(split_cavity(mesh, eps, delta), problem)
        return read(result.bulk_participation["corner"])

    jax.grad(objective)(jnp.ones(len(mesh.tetrahedra)))


def _out_eigenmode_field_cluster_member(request):
    # TE101 = TE011 on the mirrored mesh: one mode's share, and their sum
    # (symmetric in the pair) too, since each member's field is arbitrary
    with pytest.raises(HOST, match=r"DegenerateModeError.*degenerate cluster"):
        _corner_share(request, delta=0.0, n_modes=2, read=jnp.sum)
    _corner_share(request, delta=0.0, n_modes=2, read=lambda share: share[0])


def _out_eigenmode_field_cut_cluster(request):
    _corner_share(request, delta=0.0, n_modes=1, read=lambda share: share[0])


def _out_eigenmode_field_near_pair(request):
    # split to a relative gap of 2.5e-5: outside cluster_rtol, below 1e-4
    _corner_share(request, delta=1e-4, n_modes=2, read=lambda share: share[0])


def _out_eigenmode_q_port_without_voltage(request):
    # no physical port reads exactly zero voltage on a mode (round-off), so
    # p2's voltage functional is replaced by zero: its q_port is infinite
    from gfem._engine import terms

    monkeypatch = request.getfixturevalue("monkeypatch")
    model = request.getfixturevalue("eigen_model")
    original = terms.port_vectors

    def deaf(space, built, order, data):
        b, v = original(space, built, order, data)
        return (b, np.zeros_like(v)) if data.port.name == "p2" else (b, v)

    monkeypatch.setattr(terms, "port_vectors", deaf)
    objective, eps = _eigen_ratio_objective(
        model,
        gfem.Eigenmode(target=17e9),
        gfem.SolverOptions(),
        lambda r: r.q_port["p2"][0],
    )
    jax.grad(objective)(eps)


def _out_eigenmode_principal_by_a_lumped_value(request):
    # the glued line's lumped model with principal values in "lower": its
    # principal shares by port p1's inductance (no cell checks that pair yet)
    mesh = request.getfixturevalue("sheet_mesh")
    lower = gfem.Material(eps_r=(2.0, 2.4, 1.8))

    def objective(inductance):
        p1 = dataclasses.replace(_rlc_port("p1", "port1"), inductance=inductance)
        p2 = gfem.LumpedPort("p2", "port2", "+y", impedance=1e3)
        model = _lumped_model(mesh, [p1, p2], lower=lower)
        result = gfem.solve(model, gfem.Eigenmode(target=LUMPED_TARGET))
        return jnp.sum(result.principal_participation["lower"][0])

    jax.make_jaxpr(jax.grad(objective))(LUMPED[0])


@pytest.fixture(scope="module")
def thermo_optic_section(tmp_path_factory):
    """A silicon core in oxide on a cross-section slab, dn/dT in both."""
    rects = [
        (-1.5, -1.0, 1.5, 1.2, "clad", None),
        (-0.25, 0.0, 0.25, 0.22, "core", 0.1),
    ]
    path = tmp_path_factory.mktemp("thermo_optic") / "section.msh"
    return tc.section(path, rects, 0.5, [])


def _thermo_optic_objective(mesh, core):
    def objective(x):
        materials = {
            "clad": gfem.Material(eps_r=2.085, thermo_optic_coefficient=1e-5),
            "core": core(x),
        }
        model = gfem.Model(
            mesh, materials=materials, ports=[gfem.WavePort("p", "front")]
        )
        shift = gfem.thermo_optic_shift(model, None, port="p", frequency=1.934e14)
        return jnp.sum(shift.trace()._weights) + shift.n_eff

    return objective


def _out_thermo_optic_shift_by_the_optical_eps(request):
    # the mode is a constant of the shift: a frozen-mode derivative is 2x off
    mesh = request.getfixturevalue("thermo_optic_section")
    objective = _thermo_optic_objective(
        mesh,
        lambda eps: gfem.MaterialField(eps_r=eps, thermo_optic_coefficient=1.86e-4),
    )
    jax.make_jaxpr(jax.grad(objective))(12.1)


def _out_thermo_optic_shift_by_the_optical_nodes(request):
    mesh = request.getfixturevalue("thermo_optic_section")
    core = gfem.Material(eps_r=12.1, thermo_optic_coefficient=1.86e-4)

    def objective(nodes):
        moved = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(mesh), [nodes]
        )
        return _thermo_optic_objective(moved, lambda _: core)(None)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(mesh.nodes))


def _route_objective(mesh, core):
    route = {"r": np.array([[0.0, 0.0, 0.0], [0.0, 2e-6, 0.0]])}

    def objective(x):
        materials = {
            "clad": gfem.Material(eps_r=2.085, thermo_optic_coefficient=1e-5),
            "core": core(x),
        }
        window = gfem.Model(
            mesh, materials=materials, ports=[gfem.WavePort("p", "front")]
        )
        phase = gfem.route_phase(
            window, None, routes=route, port="p", frequency=1.934e14
        )
        return jnp.sum(phase.trace("r")._weights) + phase.n_eff

    return objective


def _out_route_phase_by_the_window_eps(request):
    # the window's mode is a constant of the route phase, as of the shift
    mesh = request.getfixturevalue("thermo_optic_section")
    objective = _route_objective(
        mesh,
        lambda eps: gfem.MaterialField(eps_r=eps, thermo_optic_coefficient=1.86e-4),
    )
    jax.make_jaxpr(jax.grad(objective))(12.1)


def _out_route_phase_by_the_window_nodes(request):
    mesh = request.getfixturevalue("thermo_optic_section")
    core = gfem.Material(eps_r=12.1, thermo_optic_coefficient=1.86e-4)

    def objective(nodes):
        moved = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(mesh), [nodes]
        )
        return _route_objective(moved, lambda _: core)(None)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(mesh.nodes))


def _transient(**overrides):
    kwargs = {"duration": 1e-6, "steps": 4} | overrides
    return gfem.TransientThermal(**kwargs)


def _out_traced_time(**build):
    ((name, value),) = build.items()

    def call(_request):
        def objective(x):
            arg = x if name != "waveform" else [(0.0, 1.0), (x, 0.5)]
            _transient(**{name: arg})
            return x

        jax.make_jaxpr(jax.grad(objective))(value)

    return call


def _out_switching_time(name):
    def call(_request):
        t = np.linspace(0.0, 50.0, 501)
        helper = getattr(gfem, name)
        jax.make_jaxpr(jax.grad(lambda a: helper(t, a * (1.0 - np.exp(-t / 5.0)))))(1.0)

    return call


def _out_fit_rational(_request):
    f = np.linspace(1e9, 10e9, 20)
    pole = -2 * np.pi * 1e9

    def objective(a):
        s = jnp.zeros((len(f), 2, 2), complex)
        through = a / (2j * np.pi * f - pole)
        s = s.at[:, 0, 1].set(through).at[:, 1, 0].set(through)
        net = gfem.NetworkParameters(
            values=s,
            z_ref=jnp.full((len(f), 2), 50.0),
            frequency=tuple(float(x) for x in f),
            ports=("p1", "p2"),
            excited=("p1", "p2"),
        )
        return net.fit_rational().error

    jax.make_jaxpr(jax.grad(objective))(1e9)


def _out_thermal_nodes(name):
    def call(request):
        model, problem = request.getfixturevalue("thermal_problem_cases")[name]
        structure = jax.tree_util.tree_structure(model.mesh)

        def objective(nodes):
            mesh = jax.tree_util.tree_unflatten(structure, [nodes])
            result = gfem.solve(model.replace(mesh=mesh), problem)
            return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(result))

        jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))

    return call


# (row, raising call, error, message): DegenerateModeError on one mode of a
# degenerate eigenmode cluster is in tests/verification/test_gradient_matrix.py.
HOST = jax.errors.JaxRuntimeError  # a gfem error raised in the backward pass
OUT_OF_SCOPE = [
    *(
        (
            f"PML target reflection ({knob})",
            _out_traced_pml_knob(knob),
            gfem.NonDifferentiableError,
            "The stretch of a perfectly matched layer is fixed",
        )
        for knob in ("target_reflection", "reference_frequency")
    ),
    (
        "electrostatic x sheet fill",
        _out_electrostatic_sheet,
        NotImplementedError,
        "sheet field",
    ),
    (
        "piezoelectric x sheet fill",
        _out_piezoelectric_sheet,
        NotImplementedError,
        "SheetField.*2D mesh",
    ),
    (
        "LumpedPort resistance",
        _out_lumped_port_resistance,
        gfem.NonDifferentiableError,
        r"LumpedPort\.resistance is a fixed value.*inductance and capacitance",
    ),
    (
        "design in the far-field layer",
        _out_far_field_layer_design,
        ValueError,
        r"far-field layer just outside this side holds the design region 'scat'",
    ),
    (
        "design in the beam_coupling layer",
        _out_beam_coupling_layer_design,
        ValueError,
        r"beam_coupling's layer just outside the side holds region 'scat'",
    ),
    (
        "beam_coupling with an IncidentField on a moved layer",
        _out_beam_coupling_incident_field_on_a_moved_layer,
        ValueError,
        r"IncidentField 'down' as interpolated on the imported mesh.*have moved",
    ),
    (
        "traced beam parameters in beam_coupling's test beam",
        _out_beam_coupling_traced_beam,
        gfem.NonDifferentiableError,
        r"GaussianBeam\.waist is a fixed value",
    ),
    (
        "FloquetPort offset",
        _out_floquet_port_offset,
        gfem.NonDifferentiableError,
        r"FloquetPort\.offset is a fixed value",
    ),
    (
        "Model.bloch_wave_vector under a traced angle",
        _out_bloch_wave_vector_angle,
        gfem.NonDifferentiableError,
        r"Model\.bloch_wave_vector: theta and phi must be fixed numbers",
    ),
    (
        "driven x law parameter under a Bloch wave vector",
        _out_law_under_a_bloch_wave_vector,
        gfem.NonDifferentiableError,
        r"eps_r\.eps_inf.*Bloch wave vector",
    ),
    (
        "material axes (orientation)",
        _out_material_axes,
        gfem.NonDifferentiableError,
        r"axes are fixed.*principal values",
    ),
    (
        "values of an anisotropic Material",
        _out_anisotropic_material_value,
        gfem.NonDifferentiableError,
        r"MaterialField\.uniform.*principal values along its axes",
    ),
    (
        "nodes beside an anisotropic absorbing face",
        _out_nodes_beside_a_tensor_absorbing_face,
        gfem.NonDifferentiableError,
        r"Absorbing surface 'side' has an anisotropic material.*stop_gradient",
    ),
    (
        "driven x nodes of a port surface",
        _out_driven_port_nodes_moved,
        HOST,
        r"PortGeometryError.*port1",
    ),
    (
        "eigenmode x nodes of a port surface",
        _out_eigenmode_port_nodes_moved,
        HOST,
        r"PortGeometryError.*port1",
    ),
    (
        "2D PortMode x law region at the line port",
        _out_line_port_mode_law_parameter,
        gfem.NonDifferentiableError,
        r"gradients of a 2D model are available in this version with respect to",
    ),
    (
        "driven x port mode residual at a port-face design",
        _out_driven_port_mode_residual,
        gfem.NonDifferentiableError,
        r"'residual'.*diagnostic.*stop_gradient",
    ),
    (
        "driven x port-face design under a Bloch wave vector",
        _out_port_face_design_under_bloch(),
        gfem.NonDifferentiableError,
        r"mode of wave port 'w' reads it.*not verified under a Bloch wave vector",
    ),
    (
        "driven x port-face design on a fixed-angle sweep",
        _out_port_face_design_under_bloch(wave_vector_frequency=10e9),
        gfem.NonDifferentiableError,
        r"mode of wave port 'w' reads it.*not verified on a fixed-angle sweep",
    ),
    (
        "eigenmode x material on a wave port face",
        _out_eigenmode_material_on_a_wave_port_face,
        gfem.PortGeometryError,
        (
            r"WavePort 'p1'.*region 'air' has elements on the port face.*"
            r"A gfem\.Driven solve differentiates"
        ),
    ),
    (
        "eigenmode x law parameter of a region on a wave port face",
        _out_eigenmode_law_on_a_wave_port_face,
        gfem.PortGeometryError,
        (
            r"has elements on the port face.*permittivity law.*"
            r"A gfem\.Driven solve differentiates"
        ),
    ),
    (
        "eigenmode x Impedance value beside a wave port",
        _out_eigenmode_impedance_beside_a_wave_port,
        gfem.PortGeometryError,
        (
            r"Impedance model\.boundaries\[0\] shares an edge with the port face.*"
            r"A gfem\.Driven solve differentiates"
        ),
    ),
    (
        "eigenmode x nodes of a wave port surface",
        _out_eigenmode_wave_port_nodes_moved,
        HOST,
        r"PortGeometryError.*port",
    ),
    (
        "greedy adaptive sweep under a transformation",
        _out_greedy_adaptive_sweep,
        NotImplementedError,
        "samples=",
    ),
    ("forward mode", _out_forward_mode, TypeError, "jvp"),
    (
        "probe x node coordinates",
        _out_probe_in_the_node_coordinates,
        gfem.NonDifferentiableError,
        r"mesh node coordinates.*stop_gradient",
    ),
    (
        "electric energy of an adaptive sweep's field",
        _out_electric_energy_of_a_reduced_field,
        gfem.NonDifferentiableError,
        r"adaptive sweep.*frozen reduced basis.*sweep=\"uniform\"",
    ),
    (
        "cross_sections x design in the truncation layer",
        _out_cross_sections_of_a_design_in_the_truncation_layer,
        gfem.NonDifferentiableError,
        r"cross_sections with respect to the material of region 'slab'.*truncation",
    ),
    (
        "power_flow x material in its layer",
        _out_power_flow_of_a_material_in_its_layer,
        gfem.NonDifferentiableError,
        r"materials\['core'\]\.eps_r.*element layer.*move the surface",
    ),
    (
        "cross_sections x node coordinates",
        _out_cross_sections_in_the_node_coordinates,
        gfem.NonDifferentiableError,
        r"incident field 'w' enters the load.*node coordinates",
    ),
    (
        "multipoles x node coordinates",
        _out_multipoles_in_the_node_coordinates,
        gfem.NonDifferentiableError,
        r"incident field 'w' enters the load.*node coordinates",
    ),
    (
        "multipoles x node coordinates at fixed coefficients",
        _out_multipoles_in_the_node_coordinates_at_fixed_coefficients,
        gfem.NonDifferentiableError,
        r"multipoles needs fixed node coordinates.*stop_gradient",
    ),
    (
        "multipoles x design in the truncation layer",
        _out_multipoles_of_a_design_in_the_truncation_layer,
        gfem.NonDifferentiableError,
        r"multipoles.*truncation layer",
    ),
    (
        "multipoles x traced origin",
        _out_multipoles_of_a_traced_origin,
        gfem.NonDifferentiableError,
        r"multipoles' origin must be fixed",
    ),
    (
        "multipoles x traced intensity",
        _out_multipoles_of_a_traced_intensity,
        gfem.NonDifferentiableError,
        r"multipoles' intensity is static",
    ),
    (
        "current_moments of a traced design",
        _out_current_moments_of_a_traced_design,
        gfem.NonDifferentiableError,
        r"gfem\.current_moments is not differentiable",
    ),
    (
        "heat_flow x node coordinates",
        _out_heat_flow_in_the_node_coordinates,
        gfem.NonDifferentiableError,
        r"mesh node coordinates.*stop_gradient",
    ),
    (
        "power_flow x reduced field",
        _out_power_flow_of_a_reduced_field,
        gfem.NonDifferentiableError,
        r"adaptive sweep.*frozen reduced basis.*sweep=\"uniform\"",
    ),
    (
        "electromagnetic_heating x reduced field",
        _out_electromagnetic_heating_of_a_reduced_field,
        gfem.NonDifferentiableError,
        r"adaptive sweep.*frozen reduced basis.*sweep=\"uniform\"",
    ),
    (
        "Conductor conductivity",
        _out_conductor_conductivity,
        gfem.NonDifferentiableError,
        r"Conductor\.conductivity is a fixed value",
    ),
    (
        "dissipated_power x node coordinates",
        _out_dissipated_power_in_the_node_coordinates,
        gfem.NonDifferentiableError,
        r"mesh node coordinates.*stop_gradient",
    ),
    (
        "dissipated_power x reduced field",
        _out_dissipated_power_of_a_reduced_field,
        gfem.NonDifferentiableError,
        r"adaptive sweep.*frozen reduced basis.*sweep=\"uniform\"",
    ),
    (
        "dissipated_power x frequency",
        _out_dissipated_power_in_the_frequency,
        gfem.NonDifferentiableError,
        r"omega.*frequency is a fixed input of the driven solve",
    ),
    (
        "electromagnetic_heating x frequency",
        _out_electromagnetic_heating_in_the_frequency,
        gfem.NonDifferentiableError,
        r"electromagnetic_heating: frequency is a fixed value.*driven solve",
    ),
    # driven on a periodic model
    (
        "driven x wave_vector",
        _out_driven_traced_wave_vector,
        gfem.NonDifferentiableError,
        "wave_vector must be fixed",
    ),
    (
        "driven x traced wave_vector_frequency",
        _out_driven_traced_wave_vector_frequency,
        gfem.NonDifferentiableError,
        r"Driven\.wave_vector_frequency must be a fixed number",
    ),
    (
        "periodic model x node move that breaks a pair",
        _out_periodic_pair_broken,
        HOST,
        r"ValueError.*breaks Periodic\(",
    ),
    (
        "boundary_morph x paired moving surface",
        _out_morph_of_a_paired_surface,
        ValueError,
        "cannot be the moving surface",
    ),
    (
        "Bloch driven x adaptive sweep",
        _out_bloch_adaptive_sweep,
        NotImplementedError,
        "sweep='uniform'",
    ),
    # Floquet gradient rows
    (
        "driven x material of a Floquet port's half-space",
        _out_material_of_a_floquet_half_space,
        gfem.NonDifferentiableError,
        (
            r"FloquetPort 'R_TE' on surface 'z_max': region 'air' is the half-space "
            r"behind it, and its eps_r fixes the diffraction orders"
        ),
    ),
    (
        "driven x nodes of a Floquet port surface",
        _out_floquet_port_nodes_moved,
        HOST,
        r"PortGeometryError: FloquetPort 'R_TE': 1 node\(s\) of its surface 'z_max'",
    ),
    (
        "propagate x distance",
        _out_propagate_distance,
        gfem.NonDifferentiableError,
        r"propagate: distance and padding must be fixed numbers",
    ),
    # Floquet path rows
    (
        "Floquet driven x adaptive sweep",
        _out_floquet_adaptive_sweep,
        NotImplementedError,
        r"Floquet ports \('R_TE'.*the adaptive sweep cannot solve them",
    ),
    (
        "Floquet driven x petsc",
        _out_floquet_petsc,
        NotImplementedError,
        r"linear_solver='petsc' has no Floquet port path",
    ),
    (
        "eigenmode x Floquet port",
        _out_eigenmode_floquet_port,
        NotImplementedError,
        r"Eigenmode does not support Floquet port 'R_TE'",
    ),
    (
        "cell_response x the cell's own design",
        _out_cell_response_design,
        gfem.NonDifferentiableError,
        r"cell_response: the cell's design is traced by JAX.*second derivatives",
    ),
    (
        "diffraction outputs x incident polarization",
        _out_diffraction_polarization,
        gfem.NonDifferentiableError,
        r"efficiencies: polarization must be fixed numbers",
    ),
    (
        "angle_sweep under a traced angle",
        _out_angle_sweep_traced_angle,
        gfem.NonDifferentiableError,
        r"angle_sweep: theta and phi must be fixed numbers",
    ),
    # periodic eigenmode rows
    (
        "eigenmode x wave_vector",
        _out_eigenmode_traced_wave_vector,
        gfem.NonDifferentiableError,
        "wave_vector must be fixed",
    ),
    (
        "Bloch eigenmode x bddc_gmres",
        _out_bloch_eigenmode_on_bddc,
        ValueError,
        "linear_solver='direct'",
    ),
    (
        "eigenmode band group velocity",
        _out_eigenmode_band_group_velocity,
        gfem.NonDifferentiableError,
        "'v_g'",
    ),
    (
        "band_diagram inside jax.jit or jax.vmap",
        _out_band_diagram_staged,
        gfem.NonDifferentiableError,
        "not inside jax.jit or jax.vmap",
    ),
    (
        "band_diagram x wave vectors or target",
        _out_band_diagram_wave_vectors,
        gfem.NonDifferentiableError,
        "must be fixed numbers",
    ),
    # eigenmode mode-field outputs
    (
        "eigenmode backward error",
        _out_eigenmode_error,
        gfem.NonDifferentiableError,
        "'error'",
    ),
    (
        "eigenmode mode-field output x bddc_gmres",
        _out_eigenmode_field_bddc,
        gfem.NonDifferentiableError,
        'linear_solver="bddc_gmres"',
    ),
    (
        "eigenmode mode-field output x petsc",
        _out_eigenmode_field_petsc,
        gfem.NonDifferentiableError,
        'linear_solver="petsc"',
    ),
    (
        "eigenmode mode-field output x cuda",
        _out_eigenmode_field_cuda,
        gfem.NonDifferentiableError,
        'device="cuda"',
    ),
    (
        "eigenmode mode-field output x eigen_tolerance above 1e-10",
        _out_eigenmode_field_tolerance,
        gfem.NonDifferentiableError,
        "eigen_tolerance = 1e-08",
    ),
    (
        "eigenmode mode-field output x plain periodic model",
        _out_eigenmode_field_periodic,
        gfem.NonDifferentiableError,
        "periodic model",
    ),
    (
        "eigenmode mode-field output x Bloch wave vector",
        _out_eigenmode_field_bloch,
        gfem.NonDifferentiableError,
        "Bloch wave vector",
    ),
    (
        "eigenmode mode-field output x nodes of a port surface",
        _out_eigenmode_field_port_nodes_moved,
        HOST,
        r"PortGeometryError.*port1",
    ),
    (
        "eigenmode mode field x node coordinates",
        _out_eigenmode_mode_field_nodes,
        gfem.NonDifferentiableError,
        r"'e_fields.*' with respect to the mesh node coordinates.*finite differences",
    ),
    (
        "eigenmode mode-field output x member of a cluster",
        _out_eigenmode_field_cluster_member,
        HOST,
        "DegenerateModeError.*degenerate cluster",
    ),
    (
        "eigenmode mode-field output x cluster cut by n_modes",
        _out_eigenmode_field_cut_cluster,
        HOST,
        "DegenerateModeError.*n_modes",
    ),
    (
        "eigenmode mode-field output x nearly degenerate mode",
        _out_eigenmode_field_near_pair,
        HOST,
        "DegenerateModeError.*nearly degenerate",
    ),
    (
        "eigenmode q_port of a mode with no voltage on the port",
        _out_eigenmode_q_port_without_voltage,
        HOST,
        r"NonDifferentiableError.*1 / q_port",
    ),
    (
        "law parameter of a blend's end",
        _out_law_of_a_blend_end,
        gfem.NonDifferentiableError,
        "differentiate the law on a `Material` region",
    ),
    (
        "eigenmode principal_participation x lumped value",
        _out_eigenmode_principal_by_a_lumped_value,
        gfem.NonDifferentiableError,
        r"'principal_participation'.*model\.ports\[0\]\.inductance.*stop_gradient",
    ),
    (
        "eigenmode mode-field output x law parameter",
        _out_eigenmode_probe_by_a_law_parameter,
        gfem.NonDifferentiableError,
        r"permittivity law of model\.materials\['core'\].*eigenfrequency or Q",
    ),
    (
        "eigenmode x absorbing face on a law region",
        _out_eigenmode_absorbing_law_face,
        NotImplementedError,
        "absorbing boundary on a dispersive region",
    ),
    (
        "periodic eigenmode x law parameter",
        _out_periodic_eigenmode_by_a_law_parameter,
        gfem.NonDifferentiableError,
        r"eps_r\.eps_inf.*periodic eigenmode",
    ),
    (
        "TransientThermal duration",
        _out_traced_time(duration=1e-6),
        gfem.NonDifferentiableError,
        r"TransientThermal\.duration is a fixed value",
    ),
    (
        "TransientThermal steps",
        _out_traced_time(steps=4.0),
        gfem.NonDifferentiableError,
        r"TransientThermal\.steps must be a fixed integer",
    ),
    (
        "TransientThermal waveform start time",
        _out_traced_time(waveform=5e-7),
        gfem.NonDifferentiableError,
        r"waveform start time is a fixed value",
    ),
    (
        "TransientThermal initial_temperature",
        _out_traced_time(initial_temperature=300.0),
        gfem.NonDifferentiableError,
        r"TransientThermal\.initial_temperature is a fixed value",
    ),
    *(
        (
            f"{name} under jax.grad",
            _out_switching_time(name),
            gfem.NonDifferentiableError,
            rf"gfem\.{name} is a measurement on recorded samples",
        )
        for name in ("rise_time", "time_constant")
    ),
    (
        "fit_rational under jax.grad",
        _out_fit_rational,
        gfem.NonDifferentiableError,
        r"NetworkParameters\.fit_rational relocates poles iteratively",
    ),
    *(
        (
            f"{name} x node coordinates",
            _out_thermal_nodes(name),
            gfem.NonDifferentiableError,
            rf"{name} results with respect to the mesh node coordinates.*shape",
        )
        for name in ("steady_thermal", "transient_thermal", "dc_conduction")
    ),
    (
        "thermo-optic shift x optical permittivity",
        _out_thermo_optic_shift_by_the_optical_eps,
        gfem.NonDifferentiableError,
        r"model\.materials\['core'\]\.eps_r.*thermo-optic shift holds the port mode",
    ),
    (
        "thermo-optic shift x optical node coordinates",
        _out_thermo_optic_shift_by_the_optical_nodes,
        gfem.NonDifferentiableError,
        r"mesh\.nodes.*thermo-optic shift holds the port mode",
    ),
    (
        "route phase x window permittivity",
        _out_route_phase_by_the_window_eps,
        gfem.NonDifferentiableError,
        r"model\.materials\['core'\]\.eps_r.*thermo-optic shift holds the port mode",
    ),
    (
        "route phase x window node coordinates",
        _out_route_phase_by_the_window_nodes,
        gfem.NonDifferentiableError,
        r"mesh\.nodes.*thermo-optic shift holds the port mode",
    ),
    (
        "port mode residual",
        _out_port_mode_residual,
        gfem.NonDifferentiableError,
        r"'residual'.*diagnostic.*stop_gradient",
    ),
    (
        "port mode group index and dispersion",
        _out_port_mode_dispersion,
        gfem.NonDifferentiableError,
        r"'n_g'.*group index and dispersion.*stop_gradient",
    ),
    (
        "port mode field at points x node coordinates",
        _out_port_mode_field_at_nodes,
        gfem.NonDifferentiableError,
        r"PortModeField\.at reads the mode at fixed",
    ),
    (
        "characteristic_impedance of a traced mode",
        _out_characteristic_impedance_of_a_traced_mode,
        gfem.NonDifferentiableError,
        "characteristic_impedance is not differentiable",
    ),
    (
        "electro-optic shift x optical permittivity",
        _out_electro_optic_shift_by_the_optical_eps,
        gfem.NonDifferentiableError,
        r"electro-optic shift holds the port mode constant",
    ),
]


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [row[1:] for row in OUT_OF_SCOPE],
    ids=[row[0] for row in OUT_OF_SCOPE],
)
def test_out_of_scope_pairs_raise_named_errors(request, call, error, match):
    with pytest.raises(error, match=match):
        call(request)


# --- the frequency path beside the mode-field rows --------------------------------


def _omega_options(request, **options):
    # rows 3 to 6: the refusals are static, so tracing is the whole check
    model = request.getfixturevalue("eigen_model")
    objective, eps = _eigen_ratio_objective(
        model,
        gfem.Eigenmode(target=17e9),
        gfem.SolverOptions(**options),
        lambda r: jnp.real(r.omega[0]) * 1e-9,
    )
    jax.make_jaxpr(jax.grad(objective))(eps)


def _omega_petsc(request):
    from gfem import _petsc

    try:
        _petsc.require_petsc("linear_solver")
    except gfem.SolverUnavailableError as err:
        pytest.skip(str(err))
    _omega_options(request, linear_solver="petsc")


def _omega_bragg(request, problem):
    model = request.getfixturevalue("bragg_cell")
    n = len(model.mesh.elements("low"))

    def objective(e):
        m = model.with_material("low", gfem.MaterialField(eps_r=e))
        return jnp.real(gfem.solve(m, problem).omega[0]) * 1e-9

    jax.make_jaxpr(jax.grad(objective))(jnp.full(n, pc.C3_EPS_LOW))


def _omega_corner(request, *, delta, read):
    # rows 10 and 12 refuse in the backward: these run it
    mesh = request.getfixturevalue("corner_cavity")
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=2)

    def objective(eps):
        result = gfem.solve(split_cavity(mesh, eps, delta), problem)
        return read(jnp.real(result.omega)) * 1e-9

    grad = jax.grad(objective)(jnp.ones(len(mesh.tetrahedra)))
    assert np.isfinite(np.asarray(grad)).all()
    assert np.abs(np.asarray(grad)).max() > 0


#: The models of the mode-field rows 3 to 12 with
#: an omega-only objective: the new refusals stay off the frequency path.
#: Rows 9 (a moved port node) and 11 (a cluster cut by n_modes) are left
#: out: omega raises there too, by its own rows ("eigenmode x nodes of a
#: port surface", and tests/verification/test_gradient_matrix.py). On the
#: cluster of row 10 omega differentiates as a sum over the cluster.
OMEGA_BESIDE_ROWS = [
    ("bddc_gmres", lambda r: _omega_options(r, linear_solver="bddc_gmres")),
    ("petsc", _omega_petsc),
    ("cuda", lambda r: _omega_options(r, device="cuda")),
    ("eigen_tolerance above 1e-10", lambda r: _omega_options(r, eigen_tolerance=1e-8)),
    (
        "plain periodic model",
        lambda r: _omega_bragg(r, gfem.Eigenmode(target=pc.C3_PLAIN_TARGET)),
    ),
    (
        "Bloch wave vector",
        lambda r: _omega_bragg(r, _bloch_eigenmode(math.pi / (pc.C3_A * pc.MM))),
    ),
    ("cluster sum", lambda r: _omega_corner(r, delta=0.0, read=jnp.sum)),
    (
        "nearly degenerate mode",
        lambda r: _omega_corner(r, delta=1e-4, read=lambda omega: omega[0]),
    ),
]


#: The model of the law rows whose omega still differentiates: the
#: mode-field row's resonator (its probe raises, its frequency does not). The
#: periodic law row raises for omega too, and the absorbing row in the
#: forward, so neither has an omega companion.
def _omega_by_a_law_parameter(request):
    objective = _eigen_law_objective(request, lambda r: jnp.real(r.omega[0]) * 1e-9)
    grad = jax.grad(objective)(2.0)
    assert np.isfinite(grad)
    assert grad != 0


OMEGA_BESIDE_ROWS.append(("law parameter", _omega_by_a_law_parameter))


@pytest.mark.parametrize(
    "call",
    [row[1] for row in OMEGA_BESIDE_ROWS],
    ids=[row[0] for row in OMEGA_BESIDE_ROWS],
)
def test_omega_on_the_mode_field_rows_models_still_differentiates(request, call):
    call(request)


def test_stop_gradient_opts_a_read_input_out(model):
    problem = PROBLEMS["electrostatic"]

    def objective(m):
        frozen = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return gfem.solve(frozen, problem).capacitance[0, 0]  # in series with oxide

    grads = jax.grad(objective)(model)
    assert not np.asarray(grads.mesh.nodes).any()
    assert np.abs(np.asarray(grads.materials["oxide"].eps_r)).max() > 0


class _RuleWithExtraOutput(ElectrostaticRule):
    """The electrostatic rule plus an output its backward cannot handle."""

    def output_specs(self, model):
        return {**super().output_specs(model), "extra": jax.ShapeDtypeStruct((), float)}

    def forward(self, model, *, keep, request=None):
        outputs, entry = super().forward(model, keep=keep, request=request)
        return {**outputs, "extra": np.trace(outputs["capacitance"])}, entry


def test_nondifferentiable_output_raises_unless_stop_gradient(model):
    problem = PROBLEMS["electrostatic"]
    rule = _RuleWithExtraOutput(model, problem, gfem.SolverOptions())
    n = len(model.mesh.elements("oxide"))

    def objective(eps, *, stop):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=eps))
        out = solve_vjp(rule, m, problem)
        extra = jax.lax.stop_gradient(out["extra"]) if stop else out["extra"]
        return out["capacitance"][0, 0] + extra

    eps = jnp.full(n, 3.9)
    with pytest.raises(gfem.NonDifferentiableError, match="'extra'"):
        jax.make_jaxpr(jax.grad(lambda e: objective(e, stop=False)))(eps)
    grad = jax.grad(lambda e: objective(e, stop=True))(eps)

    def c_top(e):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=e))
        return gfem.solve(m, problem).capacitance[0, 0]

    assert np.abs(np.asarray(grad)).max() > 0
    np.testing.assert_allclose(grad, jax.grad(c_top)(eps), rtol=1e-14)
