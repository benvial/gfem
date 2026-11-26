"""The refusals of features with no path on a mesh of dimension 2.

On a 2D mesh, a feature that only has a tetrahedral path raises
`NotImplementedError` before any of its code runs, naming the remedy: a
supported option, or the slab model through `gfem.Mesh.from_gmsh_cross_section`.
Each table holds one entry per refused feature, so giving a feature its 2D path
deletes its entry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem._boundaries import (
    Conductor,
    Convection,
    FixedTemperature,
    FloquetPort,
    HeatFlux,
    Impedance,
    LumpedPort,
    Periodic,
    WavePort,
)
from gfem._checks import is_traced
from gfem._materials import Material, MaterialField

if TYPE_CHECKING:
    from gfem.mesh import Mesh
    from gfem.model import Model
    from gfem.options import SolverOptions

SLAB = "read the section as a slab with gfem.Mesh.from_gmsh_cross_section"
_GRATING = (
    "A 1D grating is modelled as a slab: pair its two walls with gfem.Periodic, "
    "keep the cell one element thick across them, give the wave vector no "
    f"component along that pair, and {SLAB}. A native 2D periodic cell is not "
    "in this version of gfem."
)
_THERMAL = f"Thermal problems on a cross-section run on the slab: {SLAB}."
_SHEET = (
    "Name a perfect conductor's line PEC, or mesh a lossy metal as an area of "
    f"its own with a conducting Material; or {SLAB}."
)
_GEOMETRY = "Change the geometry in gmsh and read the mesh again."
_BANDS = (
    "A 2D photonic crystal runs as a slab one element thick: PEC faces give the "
    f"Ez bands, natural faces the Hz bands; {SLAB}."
)


def _refusal(what: str, remedy: str) -> NotImplementedError:
    return NotImplementedError(f"{what} is not available on a 2D mesh. {remedy}")


def _named(item: Any) -> str:
    """``LumpedPort 'p1'`` or ``Conductor on 'strip'``."""
    name = getattr(item, "name", None)
    what = type(item).__name__
    if name is not None:
        return f"{what} {name!r}"
    return f"{what} on {getattr(item, 'surface', None)!r}"


#: Boundaries and ports with no 2D path: (refused?, what, remedy), first match.
_ITEMS: tuple[tuple[Callable[[Any], bool], Callable[[Any], str], str], ...] = (
    (
        lambda i: isinstance(i, LumpedPort),
        _named,
        f"Feed the line with a WavePort on a straight exterior line, or {SLAB}.",
    ),
    (lambda i: isinstance(i, FloquetPort), _named, _GRATING),
    (
        lambda i: isinstance(i, Periodic),
        lambda i: f"Periodic({i.surfaces!r})",
        _GRATING,
    ),
    (
        lambda i: isinstance(i, (Conductor, Impedance)) and i.fill is not None,
        lambda i: f"A SheetField on {_named(i)}",
        f"A patterned sheet needs the slab: {SLAB}.",
    ),
    (lambda i: isinstance(i, (Conductor, Impedance)), _named, _SHEET),
    (
        lambda i: isinstance(i, (FixedTemperature, Convection, HeatFlux)),
        _named,
        _THERMAL,
    ),
    (
        lambda i: isinstance(i, WavePort) and i.voltage_path is not None,
        lambda i: f"{_named(i)} with a voltage_path",
        (
            "A line port's mode carries 1 W/m and its S-parameters need no "
            "reference impedance: leave it out."
        ),
    ),
)


def refuse_anisotropic(model: Model) -> None:
    """Raise for an anisotropic material of a 2D model an EM rule is about to read.

    The 2D electromagnetic rules (`gfem.Driven`, `gfem.PortMode`) solve a
    scalar field with one value per property; a piezoelectric solve reads the
    tensors itself, so it does not call this.

    Raises:
        NotImplementedError: A region's material is anisotropic.
    """
    if model.mesh.dim != 2:
        return
    for region, material in model.materials.items():
        if isinstance(material, Material) and not material.isotropic:
            raise _refusal(
                f"The anisotropic material of {region!r}",
                f"Give it one value per property, or {SLAB}.",
            )
        if isinstance(material, MaterialField) and material.axes is not None:
            raise _refusal(
                f"The anisotropic material field of {region!r}",
                f"Build the field without axes, or {SLAB}.",
            )


def refuse_model(
    mesh: Mesh,
    items: tuple[Any, ...],
    excluded: tuple[tuple[Any, ...], ...],
    pml: Any = None,
) -> None:
    """Raise for a `gfem.Model` part with no 2D path, on a 2D mesh.

    Args:
        mesh: The model's mesh.
        items: The boundaries and ports.
        excluded: The ``excluded=`` entries of each physics.
        pml: The user's ``pml=``.

    Raises:
        NotImplementedError: A part has no 2D path; the message names the remedy.
    """
    if mesh.dim != 2:
        return
    for item in items:
        for refused, what, remedy in _ITEMS:
            if refused(item):
                raise _refusal(what(item), remedy)
    if pml is not None:
        regions = ", ".join(map(repr, pml.regions))
        raise _refusal(
            f"A perfectly matched layer (gfem.PML on {regions})",
            f"Close the domain with gfem.Absorbing on its outer lines, or {SLAB}.",
        )
    for entries in excluded:
        for entry in entries:
            if not isinstance(entry, str):
                raise _refusal(
                    f"{entry!r} in excluded=",
                    "Name the enclosed areas as regions of their own and exclude "
                    "them by name: excluded=('name', ...).",
                )


#: Problem types with no 2D solve, by class name. A `PortMode` solves the mode
#: of a line port (every wave port of a 2D model is one).
_PROBLEMS = {
    "Eigenmode": f"For its modes, {SLAB}.",
    "Electrostatic": f"For its capacitances, {SLAB}.",
    "DCConduction": f"DC conduction on a cross-section runs on the slab: {SLAB}.",
    "SteadyThermal": _THERMAL,
    "TransientThermal": _THERMAL,
}


def _moved(mesh: Mesh) -> bool:
    """Whether the nodes are traced or moved from the imported ones."""
    nodes = mesh.nodes
    return is_traced(nodes) or not np.array_equal(nodes, mesh.imported_nodes)


def refuse_solve(model: Model, problem: Any, options: SolverOptions, comm: Any) -> None:
    """Raise for a `gfem.solve` on a 2D mesh that has no 2D path.

    Raises:
        NotImplementedError: An input or the problem type has no 2D path; the
            message names the remedy.
    """
    if model.mesh.dim != 2:
        return
    # one check per refused input, in order
    if comm is not None:
        raise _refusal(
            "A solve under several processes (comm=)",
            "Solve on one process: leave comm out.",
        )
    from gfem._linalg import KRYLOV_KINDS  # loads NGSolve, which `import gfem` defers

    if options.linear_solver in KRYLOV_KINDS:
        raise _refusal(
            f"linear_solver={options.linear_solver!r}",
            'Use linear_solver="direct", "auto" or "petsc".',
        )
    if options.device != "cpu":
        raise _refusal(f"device={options.device!r}", 'Solve on device="cpu".')
    if getattr(problem, "wave_vector", None) is not None:
        raise _refusal(
            "A wave vector (wave_vector=)",
            "A 2D model is solved at normal incidence: leave wave_vector out. "
            f"For oblique or conical incidence, {SLAB}.",
        )
    if getattr(problem, "sweep", None) == "adaptive":
        raise _refusal(
            'The adaptive sweep (sweep="adaptive")',
            'Solve every frequency with sweep="uniform".',
        )
    if _moved(model.mesh) and not getattr(problem, "_moves_nodes_2d", False):
        raise _refusal("A mesh whose nodes moved from the imported ones", _GEOMETRY)
    name = type(problem).__name__
    if name in _PROBLEMS:
        raise _refusal(f"A {name} problem", _PROBLEMS[name])


#: Public entries with no 2D path, by name, and their remedies.
_ENTRIES = {
    "power_flow": f"Read the power through the ports from the S-parameters, or {SLAB}.",
    "optical_force": "A 2D model has no body to push: pose the bodies on a 3D "
    "mesh, or read the force per length between two guides from the gradient "
    "of a port mode's effective index in their separation.",
    "region_mean": f"Read the field at points with Field.at, or {SLAB}.",
    "element_mean": f"Read the field at points with Field.at, or {SLAB}.",
    "electric_energy": f"For the energies, {SLAB}.",
    "dissipated_power": "Pose the driven solve on the slab from "
    "gfem.Mesh.from_gmsh_cross_section, the mesh the thermal problem runs on.",
    "electromagnetic_heating": "Pose the driven solve on the slab from "
    "gfem.Mesh.from_gmsh_cross_section, the mesh the thermal problem runs on, "
    "and give incident_power_per_length=.",
    "electro_optic_shift": f"The electro-optic shift runs on the slab: {SLAB}.",
    "acousto_optic_shift": f"The acousto-optic shift runs on the slab: {SLAB}.",
    "optomechanical_overlap": f"The optomechanical overlap runs on the slab: {SLAB}.",
    "optomechanical_coupling": f"The optomechanical coupling runs on the slab: {SLAB}.",
    "brillouin_gain": f"The Brillouin gain runs on the slab: {SLAB}.",
    "interface_participation": f"For the participations, {SLAB}.",
    "characteristic_impedance": "A line port's mode carries 1 W/m and has no "
    "characteristic impedance: its S-parameters need no reference impedance.",
    "refine_mesh": f"Refine the mesh in gmsh and read it again. {_GEOMETRY}",
    "error_estimate": f"Refine the mesh in gmsh and read it again. {_GEOMETRY}",
    "boundary_morph": _GEOMETRY,
    "helmholtz_filter": "Filter the pixels of a LayerGrid instead: "
    "helmholtz_filter(grid, radius=...).",
    "wave_vector_path": _BANDS,
    "band_diagram": _BANDS,
    "multipoles": "A 2D model takes no incident field: pose the scatterer on a "
    "3D mesh.",
    "current_moments": "A 2D model takes no incident field: pose the scatterer "
    "on a 3D mesh.",
}


def refuse_entry(mesh: Any, entry: str) -> None:
    """Raise when ``mesh`` is a 2D mesh and ``entry`` has no 2D path.

    ``mesh`` may be any user input: only a mesh of dimension 2 raises.

    Raises:
        NotImplementedError: Naming ``gfem.<entry>`` and its remedy.
    """
    if getattr(mesh, "dim", None) == 2:
        raise entry_refusal(entry)


def entry_refusal(entry: str) -> NotImplementedError:
    """The error of ``gfem.<entry>`` called on a 2D model."""
    return _refusal(f"gfem.{entry}", _ENTRIES[entry])
