"""`Field`: a solution quantity on the model's mesh, carried by a result."""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.core import Tracer

from gfem._pytree import register_fields
from gfem.mesh import Mesh

if TYPE_CHECKING:
    import meshio

__all__ = ["SPACES", "Field", "FieldMap"]

#: Finite-element spaces a `Field` can live on, by name: H1 (continuous
#: scalar), H(curl) (tangentially continuous vector), H(div) (normally
#: continuous vector) and L2 (discontinuous scalar).
SPACES = ("h1", "hcurl", "hdiv", "l2")


@dataclasses.dataclass(frozen=True, eq=False)
class Field:
    """A solution quantity on the model's mesh, as finite-element coefficients.

    Results carry their fields as `Field`s, for example
    ``cap.potential["trace"]`` for an electrostatic terminal potential. A
    field knows what it is (``quantity``, ``unit``), how it is discretized
    (``space``, ``order``) and where it is defined (``support``); its values
    live in ``coefficients``. Write it to ParaView with `to_vtu`, or open it
    in a notebook with `to_pyvista`.

    Complex fields are phasors under the e^{+iωt} time convention. A driven
    field is the peak phasor for 1 W time-averaged incident power at the
    excited port, every other port terminated in its reference impedance, so
    time-averaged densities carry a factor 1/2 (w_e = Re(E·D*)/4). An
    eigenmode field has an arbitrary normalization.

    A field is a JAX pytree: its leaves are ``coefficients``, the mesh node
    coordinates and ``omega``; the rest is static.

    Attributes:
        coefficients: The finite-element coefficients, shape ``(n_dofs,)``,
            float64 or complex128. Their order is the engine's for this mesh
            topology, support, space and order: read values through the
            field's methods, not by index.
        mesh: The mesh the field lives on (the model's mesh, as solved).
        quantity: What the field is: ``"V"`` (electric potential), ``"E"``
            (electric field), ``"B"`` (magnetic flux density), and later
            ``"H"``, ``"T"`` (temperature) or a loss density.
        unit: The SI unit of the field's values, e.g. ``"V"``, ``"V/m"``.
        space: The finite-element space, one of `SPACES`.
        order: The polynomial order of the space.
        support: The volume regions the field is defined on. Volume entities
            removed from the solve (the model's ``excluded=``) are outside the
            support even when they belong to a listed region.
        omega: The angular frequency in rad/s of a time-harmonic field
            (complex for an eigenmode), ``None`` for a static field.
    """

    coefficients: jax.Array
    mesh: Mesh
    quantity: str
    unit: str
    space: str
    order: int
    support: tuple[str, ...]
    omega: jax.Array | None = None
    _excluded: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        """Check the descriptor against the mesh."""
        if not isinstance(self.mesh, Mesh):
            raise TypeError(f"mesh must be a gfem.Mesh, got {type(self.mesh)}.")
        for name in ("quantity", "unit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Field {name} must be a non-empty string.")
        if self.space not in SPACES:
            raise ValueError(
                f"Unknown field space {self.space!r}; expected one of {SPACES}."
            )
        if not isinstance(self.order, int) or self.order < 0:
            raise ValueError(f"Field order must be an int >= 0, got {self.order!r}.")
        support = tuple(self.support)
        for region in support:
            if region not in self.mesh.volumes:
                raise ValueError(
                    f"Field support {region!r} is not a volume region of the mesh."
                )
        object.__setattr__(self, "support", support)
        coefficients = jnp.asarray(self.coefficients)
        if coefficients.ndim != 1:
            raise ValueError(
                "Field coefficients must be one-dimensional, got shape "
                f"{coefficients.shape}."
            )
        object.__setattr__(self, "coefficients", coefficients)
        if self.omega is not None:
            object.__setattr__(self, "omega", jnp.asarray(self.omega))

    @property
    def is_complex(self) -> bool:
        """Whether the coefficients are complex (a phasor)."""
        return jnp.iscomplexobj(self.coefficients)

    def b(self) -> Field:
        """The magnetic flux density B = curl E / (-iω) of an electric field.

        Exact on the discrete field: B lives in H(div) of one order lower.
        Differentiable with respect to the coefficients and ``omega``.

        Returns:
            A `Field` with quantity ``"B"`` in tesla.

        Raises:
            ValueError: The field is not an electric field (quantity
                ``"E"``), or it has no angular frequency.
        """
        if self.quantity != "E":
            raise ValueError(
                "b() is defined for electric fields (quantity 'E'); this field's "
                f"quantity is {self.quantity!r}."
            )
        if self.space != "hcurl":
            raise ValueError(
                f"b() needs an H(curl) electric field; this one is in {self.space!r}."
            )
        if self.omega is None:
            raise ValueError(
                "b() needs the angular frequency of a time-harmonic field; this "
                "electric field is static (omega is None)."
            )
        rows, cols, values, size = _engine().curl_matrix(self)
        curl = jax.ops.segment_sum(
            values * self.coefficients[cols], rows, num_segments=size
        )
        return dataclasses.replace(
            self,
            coefficients=curl / (-1j * self.omega),
            quantity="B",
            unit="T",
            space="hdiv",
            order=max(self.order - 1, 0),
        )

    def to_meshio(self) -> meshio.Mesh:
        """The field sampled for visualization, as a `meshio.Mesh`.

        Every tetrahedron of the support is cut into ``order**3`` (at least
        one) sub-tetrahedra, and the field is sampled at their vertices,
        separately per tetrahedron (so fields that jump across elements show
        their jumps). Coordinates are in metres and values in SI units. A
        real field is the point data ``quantity`` (e.g. ``"V"``); a complex
        one is ``"<quantity>_real"`` and ``"<quantity>_imag"`` (Palace's
        names). The cell data ``region`` and ``entity`` give the gmsh
        physical and entity numbers of the parent tetrahedron.

        Raises:
            TypeError: The field holds JAX tracers: export concrete values,
                outside `jax.jit` / `jax.grad`.
        """
        import meshio

        self._check_concrete("to_meshio")
        points, cells, point_data, cell_data = _engine().sample(self)
        return meshio.Mesh(
            points,
            [("tetra", cells)],
            point_data=point_data,
            cell_data={k: [v] for k, v in cell_data.items()},
        )

    def to_vtu(self, path: str | Path) -> None:
        """Write the field to a VTU file for ParaView (see `to_meshio`)."""
        self.to_meshio().write(Path(path), file_format="vtu")

    def to_pyvista(self) -> Any:
        """The field as a ``pyvista.UnstructuredGrid`` (see `to_meshio`).

        Needs the optional ``viz`` extra: ``pip install "gfem[viz]"``.
        """
        try:
            pyvista = importlib.import_module("pyvista")
        except ImportError:
            raise ImportError(
                'to_pyvista needs pyvista: pip install "gfem[viz]".'
            ) from None
        return pyvista.from_meshio(self.to_meshio())

    def _check_concrete(self, method: str) -> None:
        if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(self)):
            raise TypeError(
                f"Field.{method} needs concrete values and cannot run under "
                "jax.jit, jax.grad or jax.vmap; export the field outside the "
                "transformed function."
            )

    def __repr__(self) -> str:
        kind = "complex" if self.is_complex else "real"
        omega = "" if self.omega is None else ", time-harmonic"
        return (
            f"Field({self.quantity} in {self.unit}, {self.space} order "
            f"{self.order}, {kind}{omega}, {self.coefficients.shape[0]} "
            f"coefficients, support={list(self.support)})"
        )


register_fields(Field, ("coefficients", "mesh", "omega"))


class FieldMap(Mapping[str, Field]):
    """A read-only mapping of labels to fields, in a fixed order (a pytree).

    For example a result's terminal potentials, by terminal name.
    """

    def __init__(self, fields: Mapping[str, Field], *, label: str = "field") -> None:
        """Map ``fields`` in their given order; ``label`` names a key in errors."""
        self._fields = dict(fields)
        self._label = label

    def __getitem__(self, key: str) -> Field:
        """The field ``key``.

        Raises:
            KeyError: No such label; the message lists the known ones.
        """
        try:
            return self._fields[key]
        except KeyError:
            known = ", ".join(repr(k) for k in self._fields)
            raise KeyError(
                f"No {self._label} {key!r}; the known ones are {known}."
            ) from None

    def __iter__(self) -> Iterator[str]:
        """The labels, in order."""
        return iter(self._fields)

    def __len__(self) -> int:
        """The number of fields."""
        return len(self._fields)

    def __repr__(self) -> str:
        return f"FieldMap({self._fields!r})"


def _flatten_map(m: FieldMap):
    keys = tuple(m)
    children = tuple((jax.tree_util.DictKey(k), m[k]) for k in keys)
    return children, (keys, m._label)  # noqa: SLF001


def _unflatten_map(aux, children) -> FieldMap:
    keys, label = aux
    return FieldMap(dict(zip(keys, children, strict=True)), label=label)


jax.tree_util.register_pytree_with_keys(FieldMap, _flatten_map, _unflatten_map)


def _engine() -> Any:
    """The engine's field kernels, imported on first use (they load NGSolve)."""
    return importlib.import_module("gfem._engine.fields")
