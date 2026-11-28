r"""The elastic perfectly matched layer of a mechanical solve: stretch and forms.

A layer region of a `gfem.MechanicalEigenmode(pml=)` problem stretches each
coordinate beyond a face of the interior box, :math:`s_j = 1 - i\sigma_j(\xi)`
(`gfem._pml` for the geometry and the profile), with the reference
wavenumber :math:`k_\text{ref} = 2\pi f_\text{ref}/c_\text{fast}` and
:math:`c_\text{fast}` the largest Christoffel speed along the side's normal:
every slower wave then reflects at or below :math:`R_0`. The weak form is

.. math::

    K = \int \tilde S(v)^{\mathsf T} C\,\tilde S(u)\,J, \qquad
    M = \int \rho\,J\,u\cdot v, \qquad J = \prod_j s_j, \qquad
    \tilde G_{ij} = \partial_j u_i / s_j,

bilinear (no conjugation), so the pencil stays complex symmetric at
:math:`q = 0`. A 2D section stretches x and y only: :math:`\partial_z` is
:math:`\mp iq` as without a layer.

The stretch is a coefficient of P1 functions of the imported coordinates,
masked to the layer elements, so it rides with the nodes. Every stretched
piece is written as a pair of forms with real coefficients, ``<piece>`` and
``<piece>_im``, whose weights differ by a factor :math:`i`: the real and
imaginary parts of each factor are real coefficient functions, never read from
a complex one.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
import warnings
from collections.abc import Callable
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem import _mpi
from gfem._engine import quadrature
from gfem._engine.coefficients import p0
from gfem._engine.piezoelectric import _element_arrays, _per_length
from gfem._engine.runtime import serial_spaces
from gfem._materials import MaterialField
from gfem._pml import PML, THIN, Layers, stretch_maximum
from gfem.errors import RegionNotFoundError, ThinLayerWarning

__all__ = ["ElasticStretch", "check_layer", "fastest_speed"]

_AXES = "xyz"
#: Voigt index (xx, yy, zz, yz, xz, xy) of the strain pair ``(i, j)``.
_VOIGT = ((0, 5, 4), (5, 1, 3), (4, 3, 2))


def fastest_speed(c: np.ndarray, rho: float, axis: int) -> float:
    r"""The largest bulk speed (m/s) along the coordinate axis ``axis``.

    The square root of the largest eigenvalue of the Christoffel matrix
    :math:`\Gamma_{ik} = c_{ijkl}\,n_j n_l/\rho`, ``c`` the lab-frame 6x6
    stiffness (Pa); the piezoelectric stiffening is ignored.
    """
    rows = [_VOIGT[i][axis] for i in range(3)]
    gamma = np.asarray(c, float)[np.ix_(rows, rows)] / rho
    return float(math.sqrt(np.linalg.eigvalsh(0.5 * (gamma + gamma.T)).max()))


def check_layer(model: Any, pml: PML, active: np.ndarray) -> None:
    """Raise for a layer region the elastic layer cannot stretch.

    Raises:
        ValueError: A region is a surface, has an element outside the
            mechanical domain, or the mesh is curved.
        RegionNotFoundError: A region is not in the mesh.
        NotImplementedError: A region's material is a `MaterialField`.
    """
    mesh = model.mesh
    for name in pml.regions:
        if name in mesh.surfaces and name not in mesh.volumes:
            raise ValueError(
                f"a perfectly matched layer is a volume region; {name!r} is a surface."
            )
        if name not in mesh.volumes:
            raise RegionNotFoundError(name, mesh.volumes, kind="volume")
        if not active[mesh.elements(name)].all():
            raise ValueError(
                f"perfectly matched layer {name!r}: the region is outside the "
                'mechanical domain (excluded from "mechanical"), so the layer '
                "would stretch nothing there. Remove it from excluded= or from "
                "the layer."
            )
        if isinstance(model.materials.get(name), MaterialField):
            raise NotImplementedError(
                f"perfectly matched layer {name!r}: a layer region takes a "
                "constant Material, got a MaterialField. Give the layer region "
                "a Material."
            )
    if mesh.curved:
        raise NotImplementedError(
            "a perfectly matched layer on a curved mesh is not supported: its "
            "stretch is read from the straight elements' coordinates. Use a "
            "straight mesh."
        )


@dataclasses.dataclass(frozen=True)
class _Cx:
    """A complex number as two real coefficients (floats or coefficient functions)."""

    re: Any
    im: Any = 0.0

    def __mul__(self, o: _Cx) -> _Cx:
        return _Cx(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    def __add__(self, o: _Cx) -> _Cx:
        return _Cx(self.re + o.re, self.im + o.im)

    def __sub__(self, o: _Cx) -> _Cx:
        return _Cx(self.re - o.re, self.im - o.im)

    def part(self, k: int) -> Any:
        """The real (0) or imaginary (1) coefficient."""
        return self.im if k else self.re


def _is_zero(x: Any) -> bool:
    """Whether a coefficient is the plain number 0."""
    return isinstance(x, int | float) and x == 0


def _dot(a: Any, b: Any) -> Any:
    return None if a is None or b is None else a * b


def _scale(w: Any, x: Any) -> Any:
    return None if x is None or _is_zero(w) else w * x


def _lin(*terms: tuple[float, Any]) -> Any:
    """The sum of ``sign * x`` over the terms that are not ``None`` (else ``None``)."""
    parts = [sign * x for sign, x in terms if x is not None]
    return functools.reduce(operator.add, parts) if parts else None


_ONE = _Cx(1.0, 0.0)
_ZERO = _Cx(0.0, 0.0)


#: The bilinear pieces, as ``(sign, test field, trial field, matrix)`` terms
#: (`ElasticStretch._strain_fields`): the stiffness, the coupling and the
#: dielectric, and the first (``1``: the cross terms) and second (``2``: along
#: :math:`z`) orders of the wavenumber of a 2D section.
_PIECES: dict[str, tuple[tuple[float, str, str, str], ...]] = {
    "stiffness": ((1.0, "strain", "strain", "c"),),
    "coupling": (
        (1.0, "strain", "gradient", "e_t"),
        (1.0, "gradient", "strain", "e"),
    ),
    "dielectric": ((-1.0, "gradient", "gradient", "eps"),),
    "stiffness1": (
        (1.0, "strain", "axial_strain", "c"),
        (-1.0, "axial_strain", "strain", "c"),
    ),
    "stiffness2": ((1.0, "axial_strain", "axial_strain", "c"),),
    "coupling1": (
        (1.0, "strain", "axial_gradient", "e_t"),
        (1.0, "gradient", "axial_strain", "e"),
        (-1.0, "axial_strain", "gradient", "e_t"),
        (-1.0, "axial_gradient", "strain", "e"),
    ),
    "coupling2": (
        (1.0, "axial_strain", "axial_gradient", "e_t"),
        (1.0, "axial_gradient", "axial_strain", "e"),
    ),
    "dielectric1": (
        (-1.0, "gradient", "axial_gradient", "eps"),
        (1.0, "axial_gradient", "gradient", "eps"),
    ),
    "dielectric2": ((-1.0, "axial_gradient", "axial_gradient", "eps"),),
}


class ElasticStretch:
    r"""One solve's stretch: sides, :math:`\sigma_\max`, and its profile.

    Attributes:
        sides: ``(axis, sign, face, thickness, k_ref, sigma_max)`` per
            covered side, lengths in m.
        axes: The stretch axes.
    """

    def __init__(
        self,
        model: Any,
        built: Any,
        layers: Layers,
        pml: PML,
        f_ref: float,
    ) -> None:
        mesh = model.mesh
        self.dim = mesh.dim
        self.scale = built.scale
        self.built = built
        self.layers = layers
        self.f_ref = f_ref
        arrays = _element_arrays(model)
        sides = []
        for side in layers.sides:
            speed = max(
                fastest_speed(
                    arrays["c"][mesh.elements(r)[0]],
                    float(arrays["rho"][mesh.elements(r)[0]]),
                    side.axis,
                )
                for r in side.regions
            )
            k_ref = 2 * math.pi * f_ref / speed
            sides.append(
                (
                    side.axis,
                    side.sign,
                    side.face,
                    side.thickness,
                    k_ref,
                    stretch_maximum(pml.target_reflection, k_ref, side.thickness),
                )
            )
        self.sides = tuple(sides)
        self.axes = tuple(sorted({s[0] for s in sides}))
        self._pml_tags = [mesh.region(r).tag for r in pml.regions]
        tags = mesh._topology.volume_tags  # noqa: SLF001
        self.in_layer = np.isin(tags, self._pml_tags)
        nodes = np.asarray(mesh.imported_nodes, float)[:, : self.dim] / self.scale
        with serial_spaces():
            space = ngs.H1(built.mesh, order=1)
        self._coordinates = []
        for axis in range(self.dim):
            gf = ngs.GridFunction(space)
            gf.vec.FV().NumPy()[:] = nodes[built.node_order, axis]
            self._coordinates.append(gf)
        self._mask = p0(built, self.in_layer.astype(float))
        self._cx: dict[str, Any] = {}

    def warn_thin(self) -> None:
        r"""Warn `ThinLayerWarning` for a side with :math:`k_\text{ref}d < 2\pi/100`."""
        for (_axis, _sign, _face, d, k_ref, smax), side in zip(
            self.sides, self.layers.sides, strict=True
        ):
            if k_ref * d < THIN:
                warnings.warn(
                    f"perfectly matched layer, {side.label} side "
                    f"({', '.join(side.regions)}): it is k_ref d = {k_ref * d:.3g} "
                    f"< 2 pi / 100 at {self.f_ref:.6g} Hz for the fastest bulk "
                    f"wave, so its stretch maximum {smax:.3g} makes the operator "
                    "badly conditioned and loses digits. Make the layer about "
                    "one quasi-longitudinal wavelength thick.",
                    ThinLayerWarning,
                    stacklevel=_mpi.outside_gfem(),
                )

    # --- the regions ---------------------------------------------------------------

    def region(self, model: Any, physics: str, which: str) -> ngs.Region:
        """The ``physics`` domain's ``"all"``, ``"layer"`` or ``"physical"`` part."""
        out = set(model._domain_of(physics).excluded)  # noqa: SLF001
        bits = ngs.BitArray(len(self.built.domains))
        bits.Clear()
        for k, key in enumerate(self.built.domains):
            layer = key[0] in self._pml_tags
            if key not in out and which in ("all", "layer" if layer else "physical"):
                bits.Set(k)
        return ngs.Region(self.built.mesh, ngs.VOL, bits)

    # --- the stretch as coefficients -----------------------------------------------

    def _sigma(self, axis: int) -> Any:
        r"""The profile :math:`\sigma_j` along ``axis`` (zero outside the layer)."""
        out: Any = ngs.CF(0.0)
        for ax, sign, face, d, _k, smax in self.sides:
            if ax != axis:
                continue
            depth = sign * (self._coordinates[axis] - face / self.scale)
            out = out + ngs.IfPos(
                depth, smax * (depth / (d / self.scale)) ** 2, ngs.CF(0.0)
            )
        return self._mask * out

    def _axis(self, axis: int) -> tuple[_Cx, _Cx, _Cx]:
        r"""``(s, 1/s, t)`` of an axis, :math:`t = \partial_g \ln s`.

        :math:`g` scales every :math:`\sigma_\max`: :math:`s = 1 - i g\sigma`,
        so :math:`t = -i\sigma/s`.
        """
        if axis not in self.axes:
            return _ONE, _ONE, _ZERO
        sig = self._sigma(axis)
        n = 1.0 + sig * sig
        inv = _Cx(1.0 / n, sig / n)
        return _Cx(1.0, -1.0 * sig), inv, _Cx(sig * inv.im, -1.0 * sig * inv.re)

    def _factors(self) -> list[tuple[_Cx, _Cx, _Cx]]:
        if "axes" not in self._cx:
            self._cx["axes"] = [self._axis(a) for a in range(3)]
        return self._cx["axes"]

    def jacobian(self, *, derivative: bool = False) -> _Cx:
        r""":math:`J = \prod_j s_j`, or its derivative in the gain :math:`g`."""
        f = self._factors()
        j = f[0][0] * f[1][0] * f[2][0]
        if not derivative:
            return j
        return j * (f[0][2] + f[1][2] + f[2][2])

    # --- the forms -----------------------------------------------------------------

    def _strain_fields(self) -> dict[str, tuple[Callable, list]]:
        r"""The fields a piece is built from: ``(make, [(axis, coefficient)])``.

        ``strain`` and ``gradient`` are the sums :math:`\sum_a G_a / s_a` over the
        section's (or body's) axes, ``axial_strain`` and ``axial_gradient`` the
        unstretched :math:`z` derivative (:math:`\mp iq` of a 2D section).
        """
        dim = self.dim
        factors = self._factors()

        def strain(u: Any, j: int) -> Any:
            """The strain of the derivative along axis ``j`` alone (axial: unit)."""
            d = [ngs.grad(u[i])[j] if j < dim else u[i] for i in range(3)]
            z = 0
            rows = {
                0: (d[0], z, z, z, d[2], d[1]),
                1: (z, d[1], z, d[2], z, d[0]),
                2: (z, z, d[2], d[1], d[0], z),
            }
            return ngs.CF(rows[j])

        def gradient(u: Any, j: int) -> Any:
            """The potential's derivative along axis ``j`` in the slot ``j``."""
            d = ngs.grad(u[3])[j] if j < dim else u[3]
            return ngs.CF(tuple(d if i == j else 0 for i in range(3)))

        inverse = [(a, factors[a][1]) for a in range(dim)]
        axial = [(2, _ONE)]
        return {
            "strain": (strain, inverse),
            "gradient": (gradient, inverse),
            "axial_strain": (strain, axial),
            "axial_gradient": (gradient, axial),
        }

    def _piece(
        self, terms: tuple, fields: dict, matrices: dict, *, derivative: bool
    ) -> Callable[[int], Callable[[Any, Any], Any]]:
        r"""The part-:math:`k` integrand of :math:`J \sum \pm\, f_v^T M f_u`.

        Each term is a sign, the test and trial fields and the matrix (names
        of ``fields`` and ``matrices``). The weight :math:`J/(s_a s_b)` of the
        pair of axes :math:`(a, b)` is of rank one, so the sum over the pairs is
        one product of two stretched sums. With ``derivative`` the integrand
        is the derivative in the gain :math:`g`: :math:`\partial_g J` on the
        product, and :math:`-t_a` on the factors of each side in turn
        (:math:`\partial_g (1/s_a) = -t_a/s_a`).
        """
        factors = self._factors()
        jac = self.jacobian()
        jac_t = self.jacobian(derivative=True) if derivative else jac

        def field(make: Callable, u: Any, coefs: list) -> tuple[Any, Any]:
            """The real and imaginary coefficient parts of ``sum_a coef_a G_a``."""
            out = []
            for k in (0, 1):
                parts = [
                    cf.part(k) * make(u, a)
                    for a, cf in coefs
                    if not _is_zero(cf.part(k))
                ]
                out.append(functools.reduce(operator.add, parts) if parts else None)
            return out[0], out[1]

        def quad(fv: Any, fu: Any, apply: Callable) -> tuple[Any, Any]:
            """The real and imaginary parts of ``fv^T M fu`` (no conjugation)."""
            mr, mi = (None if x is None else apply(x) for x in fu)
            vr, vi = fv
            return (
                _lin((1.0, _dot(vr, mr)), (-1.0, _dot(vi, mi))),
                _lin((1.0, _dot(vr, mi)), (1.0, _dot(vi, mr))),
            )

        def mix(w: _Cx, part: tuple[Any, Any], k: int) -> Any:
            """The real (0) or imaginary (1) part of ``w`` times a complex ``part``."""
            re, im = part
            if k == 0:
                out = _lin((1.0, _scale(w.re, re)), (-1.0, _scale(w.im, im)))
            else:
                out = _lin((1.0, _scale(w.re, im)), (1.0, _scale(w.im, re)))
            return 0.0 if out is None else out

        def make(k: int) -> Callable[[Any, Any], Any]:
            def form(u: Any, v: Any) -> Any:
                acc: Any = 0.0
                for sign, test, trial, name in terms:
                    (mv, cv), (mu, cu) = fields[test], fields[trial]
                    apply = matrices[name]
                    fv, fu = field(mv, v, cv), field(mu, u, cu)
                    acc = acc + sign * mix(jac_t, quad(fv, fu, apply), k)
                    if not derivative:
                        continue
                    hat_v = [(a, w * factors[a][2]) for a, w in cv]
                    hat_u = [(a, w * factors[a][2]) for a, w in cu]
                    for hv, hu in (
                        (field(mv, v, hat_v), fu),
                        (fv, field(mu, u, hat_u)),
                    ):
                        acc = acc - sign * mix(jac, quad(hv, hu, apply), k)
                return acc

            return form

        return make

    def forms(
        self,
        model: Any,
        co: dict[str, Any],
        order: int,
        *,
        wavenumber_pieces: bool,
        coupled: bool,
        lossy: bool = True,
        derivative: bool = False,
    ) -> dict[str, tuple[Callable[[Any, Any], Any], Any]]:
        r"""The stretched pieces of :math:`A`, ``(integrand, measure)`` by name.

        The pieces of `gfem._engine.piezoelectric._weak_forms` with every
        stretch factor in them: ``<piece>`` carries the real part of the
        factor and ``<piece>_im`` the imaginary part, on the layer elements
        (the physical elements have no imaginary part), plus ``mass_phys``,
        the unstretched mass of the physical elements. With ``derivative``
        the factors are their derivative in the gain :math:`g` (the pieces
        of :math:`\partial_g K` and :math:`\partial_g M`, on the layer
        elements only, no ``mass_phys``). Without ``lossy`` the loss pieces
        are left out (their coefficient is zero everywhere).
        """
        scale, rho, loss = self.scale, co["rho"], co["loss"]
        mech_layer = self.region(model, "mechanical", "layer")
        diel_layer = self.region(model, "em", "layer")
        mech_all = mech_layer if derivative else self.region(model, "mechanical", "all")
        diel_all = diel_layer if derivative else self.region(model, "em", "all")

        def dx(region: Any) -> Any:
            return quadrature.dx(order, definedon=region)

        # the real part lives on every element, the imaginary one on the layer's
        measures = {"mechanical": (dx(mech_all), dx(mech_layer))}
        measures["dielectric"] = (dx(diel_all), dx(diel_layer))
        fields = self._strain_fields()
        matrices = {
            "c": functools.partial(operator.mul, co["c"]),
            "e": functools.partial(operator.mul, co["e"]),
            "e_t": functools.partial(operator.mul, co["e"].trans),
            "eps": functools.partial(operator.mul, co["eps"]),
        }
        jac_t = self.jacobian(derivative=derivative)

        def mass(k: int) -> Callable[[Any, Any], Any]:
            weight = jac_t.part(k)
            return lambda u, v: (
                scale**2 * weight * rho * (u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
            )

        def loss_of(make: Callable) -> Callable:
            return lambda k: lambda u, v: loss * make(k)(u, v)

        forms: dict[str, tuple[Callable[[Any, Any], Any], Any]] = {}

        def add(name: str, make: Callable, which: str) -> None:
            for k, suffix in ((0, ""), (1, "_im")):
                forms[name + suffix] = (make(k), measures[which][k])

        add("mass", mass, "mechanical")
        for name, terms in _PIECES.items():
            if name[-1] in "12" and not wavenumber_pieces:
                continue
            if name.startswith(("coupling", "dielectric")) and not coupled:
                continue
            make = self._piece(terms, fields, matrices, derivative=derivative)
            add(name, make, "dielectric" if name.startswith("diel") else "mechanical")
            if lossy and name.startswith("stiffness"):
                add("loss" + name[len("stiffness") :], loss_of(make), "mechanical")
        if not derivative:
            forms["mass_phys"] = (
                lambda u, v: scale**2 * rho * (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]),
                dx(self.region(model, "mechanical", "physical")),
            )
        if self.dim == 3:
            # a 3D mesh's volume carries one more length
            forms = {k: (_per_length(f, scale), d) for k, (f, d) in forms.items()}
        return forms
