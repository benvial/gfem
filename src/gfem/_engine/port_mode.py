"""The 2D port mode problem on NGSolve: mesh, spaces and the affine pencil.

Formulation (Lee, Sun and Cendes): a port mode is
``E = (E_t + n E_n) exp(-gamma s)``, ``s`` the distance along the inward
port normal ``n``. With ``e_t = gamma E_t`` and ``e_n = E_n`` in
first-kind Nédélec x H1 elements of the same order, it solves
``A x = lam B x`` with ``lam = gamma^2`` and

    A = [ (nu_nn curl e_t, curl v_t) - k0^2 (eps_tt e_t, v_t), 0 ; 0, 0 ]
    B = [ (nu_tt (e_t + grad e_n), v_t + grad v_n) - k0^2 (eps_nn e_n, v_n) ]

(``B`` holds the whole second row and column), where ``eps_tt`` and
``eps_nn`` include ``-i sigma_tt / (omega eps0)`` and ``-i sigma_nn /
(omega eps0)`` (the conductivity tensor split like the permittivity), and
an absorbing edge adds ``i k0 Y_tau (E_T, V_T)`` to the first diagonal
block and ``i k0 Y_n (E_T, V_T)`` to the second
(`gfem._engine.anisotropy.edge_admittance`). A `Conductor` or `Impedance`
edge adds its admittance to both, as the driven problem's face term
(`gfem._engine.terms`): ``i k0 eta0 / Z_s(omega)`` for a conductor, ``i k0
eta0 / R``, ``mu0 scale / L`` and ``-k0^2 C / (eps0 scale)`` for an
impedance. A region whose eps_r is a permittivity law holds eps_inf in
``eps_tt`` and ``eps_nn`` and adds its susceptibility as one more piece,
``-k0^2 chi(omega) ((e_t, v_t) + (e_n, v_n))`` on its triangles, and an
absorbing edge beside it ``i k0 sqrt(Re eps(omega) / Re mu)`` on both blocks.
Both matrices are affine: each is a sum of fixed matrices times
1, -k0^2, i k0, a conductor's own scalar or a law's, assembled once.
Coordinates are in the mesh file's unit, so ``k0`` is per local unit.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Protocol

import netgen.meshing as ngm
import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp

from gfem._boundaries import Conductor
from gfem._engine import quadrature
from gfem._engine.dispersion import (
    d_susceptibility,
    susceptibility,
    susceptibility_taylor,
)
from gfem._engine.runtime import serial_spaces
from gfem._engine.stretch import stretch_factors
from gfem._engine.taylor import Taylor
from gfem._engine.terms import _THICK, surface_impedance, surface_impedance_slope
from gfem._linalg.base import free_mask
from gfem._port_section import PortSection

# Points this close to a midline of the phase-reference quadrant (relative to
# the port's diameter) count as inside it.
_QUADRANT_RTOL = 1e-10

Z0 = scipy.constants.mu_0 * scipy.constants.c  # vacuum wave impedance, ohm


class PencilScalar(Protocol):
    """A further scalar of the pencil: its value and real-axis Taylor series."""

    def __call__(self, k0: float) -> complex:
        """The value at wavenumber ``k0`` (per local unit)."""
        ...

    def taylor(self, k0: float) -> np.ndarray:
        r""":math:`\theta^{(n)}(k_0)/n!` for :math:`n \le 3` along real ``k0``."""
        ...


@dataclasses.dataclass(frozen=True)
class Pencil:
    """``A(k0) = sum theta_j(k0) A_j`` and ``B(k0)`` likewise, on free DOFs.

    Attributes:
        a: Matrices multiplying 1, -k0^2, i k0, then each of ``extra``.
        b: Likewise for ``B``.
        extra: The further scalars of k0 (one per conductor edge group, then
            one per law and one per law with absorbing edges), each with its
            real-axis Taylor series.
        slopes: The k0-derivative of each of ``extra``, analytic in a
            complex k0, or ``None`` for a scalar that is not analytic (an
            absorbing edge beside a law, which reads Re eps); empty when the
            problem gives none (a line port's).
    """

    a: tuple[sp.csr_matrix, ...]
    b: tuple[sp.csr_matrix, ...]
    extra: tuple[PencilScalar, ...] = ()
    slopes: tuple[Callable[[complex], complex] | None, ...] = ()

    def theta(self, k0: float) -> tuple[complex, ...]:
        """The scalar of each matrix at wavenumber ``k0`` (per local unit)."""
        return (1.0, -(k0**2), 1j * k0, *(f(k0) for f in self.extra))

    def d_theta(self, k0: complex) -> tuple[complex, ...]:
        """The k0-derivative of each scalar of `theta`, at a complex ``k0``.

        Raises:
            NotImplementedError: A scalar is not analytic in k0 (an absorbing
                edge beside a permittivity law), or has no slope.
        """
        if len(self.slopes) != len(self.extra):
            raise NotImplementedError(
                "This port mode problem gives no frequency slope of its "
                "material scalars, so it has no value at a complex frequency."
            )
        if any(f is None for f in self.slopes):
            raise NotImplementedError(
                "An absorbing edge beside a permittivity law in a wave port's "
                "cross-section takes the real part of the law's permittivity, "
                "which is not analytic in the frequency, so the port term has no "
                "value at a complex frequency (an eigenmode). Move the absorbing "
                "edge off the law's region, or give that region a constant "
                "permittivity."
            )
        return (0.0, -2 * k0, 1j, *(f(k0) for f in self.slopes if f is not None))

    def taylor(self, k0: float) -> np.ndarray:
        r"""The Taylor coefficients of each scalar of `theta` along real ``k0``.

        Shape (J, 4): row :math:`j` holds :math:`\theta_j^{(n)}(k_0)/n!`,
        :math:`n = 0..3`, so that
        :math:`A(k_0 + h) = \sum_n h^n \sum_j c_{jn} A_j + O(h^4)`. Real
        ``k0`` only: an absorbing edge beside a law reads
        :math:`\operatorname{Re}\varepsilon`, whose series holds along the
        real axis and nowhere else.
        """
        k = Taylor.variable(k0)
        base = (Taylor.constant(1.0), -(k * k), 1j * k)
        rows = [t.c for t in base] + [f.taylor(k0) for f in self.extra]
        return np.array(rows, dtype=complex)

    def at(self, k0: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """``A`` and ``B`` at wavenumber ``k0`` (per local length unit)."""
        theta = self.theta(k0)
        a = sum(t * m for t, m in zip(theta, self.a, strict=True))
        b = sum(t * m for t, m in zip(theta, self.b, strict=True))
        return sp.csr_matrix(a), sp.csr_matrix(b)


def omega_of(k0: float, scale: float) -> float:
    """The angular frequency (rad/s) of a wavenumber per local unit.

    ``scale`` is the local unit in metres.
    """
    return k0 * scipy.constants.c / scale


def admittance(eps: Any, mu: Any) -> Any:
    r"""The wave admittance relative to vacuum, :math:`\sqrt{\varepsilon_r/\mu_r}`.

    Real ``eps`` and ``mu``, scalars or arrays.
    """
    return np.sqrt(eps / mu)


def susceptibility_series(law: Any, scale: float, k0: float) -> Taylor:
    r""":math:`\chi(\omega(k_0 + h))` of a law as a `Taylor` series in :math:`h`.

    :math:`d\omega/dk_0 = c/s`, ``s`` the ``scale``.
    """
    rate = scipy.constants.c / scale
    omega = omega_of(k0, scale)
    return Taylor(susceptibility_taylor(law, omega) * rate ** np.arange(4))


@dataclasses.dataclass(frozen=True)
class LawScalar:
    r""":math:`-k_0^2\chi(\omega)` of a law's susceptibility piece.

    Attributes:
        law: The permittivity law.
        scale: The local unit in metres.
    """

    law: Any
    scale: float

    def __call__(self, k0: float) -> complex:
        """The value at wavenumber ``k0`` (per local unit)."""
        return -(k0**2) * susceptibility(self.law, omega_of(k0, self.scale))

    def taylor(self, k0: float) -> np.ndarray:
        r""":math:`\theta^{(n)}(k_0)/n!` for :math:`n \le 3` along real ``k0``."""
        k = Taylor.variable(k0)
        return (-(k * k) * susceptibility_series(self.law, self.scale, k0)).c


def law_slope(law: Any, scale: float) -> Callable[[complex], complex]:
    r"""The k0-derivative of `LawScalar`, :math:`-2k_0\chi - k_0^2\chi'(\omega)\,c/s`.

    ``s`` is ``scale``, so :math:`d\omega/dk_0 = c/s`.
    """
    rate = scipy.constants.c / scale

    def slope(k0: complex) -> complex:
        omega = k0 * rate
        return -2 * k0 * susceptibility(law, omega) - k0**2 * rate * d_susceptibility(
            law, omega
        )

    return slope


@dataclasses.dataclass(frozen=True)
class LawEdgeScalar:
    r""":math:`ik_0 Y(\omega)` of the absorbing edges beside a law.

    :math:`Y` is the `admittance` of :math:`\operatorname{Re}\varepsilon_r(\omega)`
    and ``mu``, or its inverse with ``inverse``. Not analytic in
    :math:`k_0`: its series takes the real part of each coefficient of the
    law's, which holds along the real axis only.

    Attributes:
        law: The permittivity law.
        mu: The real relative permeability beside the edges.
        scale: The local unit in metres.
        inverse: Whether the scalar holds :math:`1/Y`.
    """

    law: Any
    mu: float
    scale: float
    inverse: bool = False

    def __call__(self, k0: float) -> complex:
        """The value at wavenumber ``k0`` (per local unit)."""
        eps = (
            float(self.law.eps_inf)
            + susceptibility(self.law, omega_of(k0, self.scale)).real
        )
        mu = self.mu
        return 1j * k0 * (admittance(mu, eps) if self.inverse else admittance(eps, mu))

    def taylor(self, k0: float) -> np.ndarray:
        r""":math:`\theta^{(n)}(k_0)/n!` for :math:`n \le 3` along real ``k0``."""
        k = Taylor.variable(k0)
        chi = susceptibility_series(self.law, self.scale, k0)
        eps = float(self.law.eps_inf) + chi.real
        y = (self.mu / eps).sqrt() if self.inverse else (eps / self.mu).sqrt()
        return (1j * k * y).c


@dataclasses.dataclass(frozen=True)
class ConductorScalar:
    r""":math:`ik_0\eta_0/Z_s(\omega)` of a conductor edge group.

    Attributes:
        sigma: Conductivity, S/m.
        thickness: Thickness in m, or ``None`` for thick metal.
        scale: The local unit in metres.
    """

    sigma: float
    thickness: float | None
    scale: float

    def __call__(self, k0: float) -> complex:
        """The value at wavenumber ``k0`` (per local unit)."""
        omega = omega_of(k0, self.scale)
        return 1j * k0 * Z0 / surface_impedance(self.sigma, self.thickness, omega)

    def taylor(self, k0: float) -> np.ndarray:
        r""":math:`\theta^{(n)}(k_0)/n!` for :math:`n \le 3` along real ``k0``.

        `gfem._engine.terms.surface_impedance` as a series, on the same
        thick-film branch.
        """
        k = Taylor.variable(k0)
        omega = k * (scipy.constants.c / self.scale)
        g = (omega * (1j * scipy.constants.mu_0 * self.sigma)).sqrt()
        z = g / self.sigma
        if self.thickness is not None:
            u = g * (0.5 * self.thickness)
            if u.c[0].real < _THICK:
                z = z / u.tanh()
        return (1j * Z0 * k / z).c


@dataclasses.dataclass(frozen=True)
class FormItem:
    r"""One term of the port mode pencil.

    The pencil is the sum of its items, ``A`` and ``B`` slot by slot; every
    pass over the pencil walks the same list.

    Attributes:
        slot: The pencil matrix it adds to: 0, 1, 2 for the scalars 1,
            :math:`-k_0^2`, :math:`i k_0`, then one per extra scalar.
        side: ``"A"`` or ``"B"``.
        kind: The form. ``"curl"``: :math:`c\,\operatorname{curl} e_t
            \operatorname{curl} v_t`; ``"grad_t"``: :math:`(c\,w_t)\cdot w_v` with
            :math:`w = e_t + \nabla e_n`; ``"value_t"``: :math:`(c\,e_t)\cdot
            v_t`; ``"value_n"``: :math:`c\,e_n v_n`; ``"edge_t"``,
            ``"edge_n"``: the same products of the traces along a boundary
            group.
        coefficient: The coefficient :math:`c` (``None``: 1).
        region: ``"dx"`` (the triangles) or the name of a boundary group.
        input: What the coefficient reads: ``("eps",)``, ``("sigma",)``,
            ``("nu",)``, ``("absorbing",)`` (the admittance, which reads both
            :math:`\varepsilon` and :math:`\nu`), ``("law", g)`` and
            ``("absorbing_law", g)`` (the indicator of law ``g``, and the
            admittance beside it), ``("fill", g)`` (a conductor group's edge
            fill) and ``("impedance", g, name)`` (the weight of an
            `Impedance` group's ``"resistance"``, ``"inductance"`` or
            ``"capacitance"``).
        weight: A real factor of the item (``None``: none): the conductivity's
            :math:`\eta_0 s`, an impedance value's weight.
    """

    slot: int
    side: str
    kind: str
    coefficient: Any
    region: str
    input: tuple
    weight: float | None = None


class PortModeSpace:
    """The discrete port mode problem of one `PortSection`.

    Attributes:
        section: The cross-section.
        order: Element order (Nédélec first kind and H1).
        quantity: The mode field's name, ``"E"``.
        unit: Its unit, ``"V/m"``.
        mesh: The 2D NGSolve mesh in local plane coordinates.
        space: The compound space ``HCurl x H1``.
        free: Indices of the free (non-Dirichlet) DOFs.
        n_t: Number of Nédélec DOFs (the first block of ``space``).
        pencil: The affine pieces of ``A`` and ``B``.
        form_items: The `FormItem` list the pencil is assembled from.
    """

    quantity = "E"
    unit = "V/m"

    def __init__(
        self, section: PortSection, order: int, *, geometry: bool = False
    ) -> None:
        """Build the mesh and spaces and assemble the pencil.

        With ``geometry`` only the mesh and spaces are built, to locate a field
        (`evaluate`, `point_columns`): no coefficient is read and no pencil
        is assembled, so the section's material arrays may be placeholders.
        """
        if not isinstance(section, PortSection):
            # a backstop: a 2D mesh's line ports never reach this space
            raise TypeError(
                "PortModeSpace solves the modes of a planar PortSection, got "
                f"{type(section).__name__}."
            )
        self.section = section
        self.order = order
        self.mesh = _build_mesh(section)
        dirichlet = "pec" if len(section.pec_edges) else ""
        with serial_spaces():
            nd = ngs.HCurl(
                self.mesh, order=order, type1=True, complex=True, dirichlet=dirichlet
            )
            h1 = ngs.H1(self.mesh, order=order, complex=True, dirichlet=dirichlet)
            self.space = ngs.FESpace([nd, h1])
        self.n_t = nd.ndof
        self.free = np.flatnonzero(free_mask(self.space.FreeDofs(), self.space.ndof))
        if geometry:
            return
        # u = s / R, the bend's inverse radius per local unit: a parameter so
        # that the forms can be differentiated in it
        self.bend_u = (
            None
            if section.bend is None
            else ngs.Parameter(section.scale / section.bend.radius)
        )
        self._coef = _coefficients(self.mesh, section, self.bend_u)
        self.pencil = self._assemble()

    def __reduce__(self) -> tuple[type[PortModeSpace], tuple[PortSection, int]]:
        """Pickle as the section and order, rebuilt on load.

        The NGSolve mesh and spaces do not pickle; the rebuild from the same
        section is deterministic, so a basis on the old space (the mode
        fields of a result sent across processes) stands on the new one.
        """
        return (PortModeSpace, (self.section, self.order))

    # --- assembly -------------------------------------------------------------

    def _assemble(self) -> Pencil:
        items, extra, slopes = self._form_list()
        self.form_items = items
        n = 3 + len(extra)
        forms = {side: [[] for _ in range(n)] for side in ("A", "B")}
        for item in items:
            forms[item.side][item.slot].append(self._integrand(item))
        a = tuple(self._matrix(f) for f in forms["A"])
        b = tuple(self._matrix(f) for f in forms["B"])
        return Pencil(a=a, b=b, extra=tuple(extra), slopes=tuple(slopes))

    def _form_list(
        self,
    ) -> tuple[list[FormItem], list[PencilScalar], list[Any]]:
        """The pencil's form items, and the scalar and slope of each extra slot.

        The one list that assembly reads (`_integrand`): a pass that
        differentiates the pencil's coefficients or the section's nodes walks
        the same items, so it cannot read a different pencil.
        """
        c = self._coef
        sigma_scale = Z0 * self.section.scale
        items = [
            FormItem(0, "A", "curl", c["nu_nn"], "dx", ("nu",)),
            FormItem(1, "A", "value_t", c["eps_tt"], "dx", ("eps",)),
            FormItem(2, "A", "value_t", c["sigma_tt"], "dx", ("sigma",), sigma_scale),
            FormItem(0, "B", "grad_t", c["nu_tt"], "dx", ("nu",)),
            FormItem(1, "B", "value_n", c["eps_nn"], "dx", ("eps",)),
            FormItem(2, "B", "value_n", c["sigma_nn"], "dx", ("sigma",), sigma_scale),
        ]
        if len(_admittance_groups(self.section)) > 0:
            items += [
                FormItem(2, "A", "edge_t", c["Y_tau"], "absorbing.*", ("absorbing",)),
                FormItem(2, "B", "edge_n", c["Y_n"], "absorbing.*", ("absorbing",)),
            ]
        extra: list[PencilScalar] = []
        slopes: list[Any] = []
        fill = _edge_fill(self.mesh, self.section)
        for g, (item, factor) in enumerate(self.section.impedance_terms):
            region = f"impedance{g}"
            if isinstance(item, Conductor):
                slot = 3 + len(extra)
                items += [
                    FormItem(slot, "A", "edge_t", fill, region, ("fill", g)),
                    FormItem(slot, "B", "edge_n", fill, region, ("fill", g)),
                ]
                h = None if item.thickness is None else factor * item.thickness
                extra.append(ConductorScalar(item.conductivity, h, self.section.scale))
                slopes.append(self._conductor_slope(item, factor))
                continue
            for slot, name, weight in self._impedance_weights(item):
                read = ("impedance", g, name)
                items += [
                    FormItem(slot, "A", "edge_t", fill, region, read, weight),
                    FormItem(slot, "B", "edge_n", fill, region, read, weight),
                ]
        index = self.section.law_index
        absorbing_law = self.section.absorbing_law
        for g, (_, law) in enumerate(self.section.laws):
            slot = 3 + len(extra)
            inside = _p0(self.mesh, (np.asarray(index) == g).astype(float))
            items += [
                FormItem(slot, "A", "value_t", inside, "dx", ("law", g)),
                FormItem(slot, "B", "value_n", inside, "dx", ("law", g)),
            ]
            extra.append(LawScalar(law, self.section.scale))
            slopes.append(law_slope(law, self.section.scale))
            if absorbing_law is not None and (absorbing_law == g).any():
                slot = 3 + len(extra)
                read = ("absorbing_law", g)
                items += [
                    FormItem(slot, "A", "edge_t", None, f"law{g}", read),
                    FormItem(slot, "B", "edge_n", None, f"law{g}", read),
                ]
                rows = np.asarray(index) == g
                mu = 1 / float(self.section.nu_nn[rows][0].real)
                extra.append(LawEdgeScalar(law, mu, self.section.scale))
                slopes.append(None)  # reads Re eps: not analytic
        return items, extra, slopes

    def _integrand(self, item: FormItem) -> Any:
        """The bilinear form term of one item (`FormItem` lists the kinds)."""
        (et, en), (vt, vn) = self.space.TnT()
        q = quadrature.degree(self.order)
        c, w, kind = item.coefficient, item.weight, item.kind
        if item.region == "dx":
            measure = ngs.dx(intrules={ngs.TRIG: ngs.IntegrationRule(ngs.TRIG, q)})
        else:
            measure = ngs.ds(
                item.region, intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)}
            )
        if kind == "curl":
            return c * ngs.curl(et) * ngs.curl(vt) * measure
        if kind == "grad_t":
            return (c * (et + ngs.grad(en))) * (vt + ngs.grad(vn)) * measure
        if kind == "value_t":
            # the weight multiplies the coefficient's product, not the form
            return (c * et if w is None else w * (c * et)) * vt * measure
        if kind == "value_n":
            return (c if w is None else w * c) * en * vn * measure
        trace, test = (et, vt) if kind == "edge_t" else (en, vn)
        # the weight multiplies the coefficient: a form has no scalar product
        if c is None:
            c = 1.0 if w is None else w
        elif w is not None:
            c = w * c
        return c * trace.Trace() * test.Trace() * measure

    def _conductor_slope(
        self, item: Conductor, factor: float
    ) -> Callable[[complex], complex]:
        r"""The k0-derivative of `ConductorScalar`, :math:`\theta (1 - s) / k_0`.

        :math:`s = (\omega / Z_s)\,dZ_s/d\omega`
        (`gfem._engine.terms.surface_impedance_slope`).
        """
        scale = self.section.scale
        h = None if item.thickness is None else factor * item.thickness
        sigma = item.conductivity

        def slope(k0: complex) -> complex:
            omega = k0 * scipy.constants.c / scale
            z = surface_impedance(sigma, h, omega)
            return 1j * Z0 / z * (1 - surface_impedance_slope(sigma, h, omega))

        return slope

    def _impedance_weights(self, item) -> list[tuple[int, str, float]]:
        """(pencil slot, name, weight) of an `Impedance` edge's R, L and C."""
        scale = self.section.scale
        out = []
        # Python floats: NGSolve integrals do not multiply with NumPy scalars,
        # which a design's host values (an Impedance beside the port) are
        if item.resistance is not None:
            out.append((2, "resistance", float(Z0 / item.resistance)))
        if item.inductance is not None:
            out.append(
                (0, "inductance", float(scipy.constants.mu_0 * scale / item.inductance))
            )
        if item.capacitance is not None:
            out.append(
                (
                    1,
                    "capacitance",
                    float(item.capacitance / (scipy.constants.epsilon_0 * scale)),
                )
            )
        return out

    def _matrix(self, integrands) -> sp.csr_matrix:
        n = self.space.ndof
        form = ngs.BilinearForm(self.space)
        for term in integrands:
            if term is not None:
                form += term
        form.Assemble()
        rows, cols, vals = form.mat.COO()
        m = sp.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
        )
        return m[self.free][:, self.free].tocsr()

    def _vector(self, integrand) -> np.ndarray:
        form = ngs.LinearForm(self.space)
        form += integrand
        form.Assemble()
        return np.array(form.vec.FV().NumPy())[self.free]

    def _dx(self):
        q = quadrature.degree(self.order)
        return ngs.dx(intrules={ngs.TRIG: ngs.IntegrationRule(ngs.TRIG, q)})

    @functools.cached_property
    def mass(self) -> sp.csr_matrix:
        """Unweighted L2 mass of ``(e_t, e_n)`` on free DOFs (real SPD).

        The inner product that tracks a mode across frequencies.
        """
        (et, en), (vt, vn) = self.space.TnT()
        return self._matrix([(et * vt + en * vn) * self._dx()]).real.tocsr()

    @functools.cached_property
    def _energy_forms(self) -> tuple[sp.csr_matrix, ...]:
        """Background-weighted electric energy matrices, on free DOFs.

        ``(T_layer, T, N_layer, N)``: ``int Re(eps_tt) e_t . v_t`` and ``int
        Re(eps_nn) e_n v_n``, over the layer triangles and over all.
        """
        (et, en), (vt, vn) = self.space.TnT()
        c, dx = self._coef, self._dx()
        layer = c["layer"]
        t, n = c["eps_bg_tt"] * et, c["eps_bg_nn"] * en
        return (
            self._matrix([layer * t * vt * dx]),
            self._matrix([t * vt * dx]),
            self._matrix([layer * n * vn * dx]),
            self._matrix([n * vn * dx]),
        )

    def layer_share(self, vecs: np.ndarray, lam: np.ndarray) -> np.ndarray:
        r"""The layer energy share of each pencil eigenvector.

        :math:`\int_\text{layer} \operatorname{Re}\varepsilon\,|E|^2 /
        \int \operatorname{Re}\varepsilon\,|E|^2` with the background
        permittivity, :math:`E_t = e_t / \gamma` and :math:`\gamma^2 =
        \lambda`. Zero without a layer.

        Args:
            vecs: ``(n_free, m)`` eigenvectors ``(e_t, e_n)`` on free DOFs.
            lam: ``(m,)`` their eigenvalues :math:`\gamma^2`.
        """
        vecs = np.asarray(vecs).reshape(len(self.free), -1)
        if self.section.layer is None:
            return np.zeros(vecs.shape[1])
        tl, ta, nl, na = self._energy_forms

        def quad(m: sp.csr_matrix) -> np.ndarray:
            return np.einsum("ik,ik->k", vecs.conj(), m @ vecs).real

        weight = 1.0 / np.abs(np.asarray(lam))
        return (weight * quad(tl) + quad(nl)) / (weight * quad(ta) + quad(na))

    @functools.cached_property
    def power_forms(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """The matrices of the power integral, on free DOFs.

        ``int E_t . conj(nu_tt (grad E_n + gamma E_t)) = c^T G conj(c) +
        conj(gamma) c^T T conj(c)``; rows belong to the unconjugated factor
        (see `power`). Over the physical triangles only, as `power`.
        """
        (et, en), (vt, _vn) = self.space.TnT()
        nu = self._coef["physical"] * ngs.Conj(self._coef["nu_tt"])
        g = self._matrix([(nu * ngs.grad(en)) * vt * self._dx()])
        t = self._matrix([(nu * et) * vt * self._dx()])
        return g, t

    @functools.cached_property
    def coupling_forms(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """The matrices that couple a mode to the 3D field on the port face.

        ``M_G[i, j] = int (nu_tt grad psi_j) . phi_i`` and ``M_T[i, j] = int
        (nu_tt phi_j) . phi_i``: rows over every Nédélec DOF ``phi_i`` (the
        trace of the 3D field, Dirichlet DOFs included), columns over the free
        DOFs (the mode). ``nu_tt`` is the port's tangential inverse-permeability
        tensor ``A`` in the port frame, the tensor of the mode's n x H (as in
        `power`) and of the 3D Robin term (`gfem._engine.terms`).
        """
        (et, en), (vt, _vn) = self.space.TnT()
        nu = self._coef["nu_tt"]
        dx = self._dx()
        out = []
        for integrand in ((nu * ngs.grad(en)) * vt * dx, (nu * et) * vt * dx):
            form = ngs.BilinearForm(self.space)
            form += integrand
            form.Assemble()
            rows, cols, vals = form.mat.COO()
            n = self.space.ndof
            m = sp.csr_matrix(
                (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
            )
            out.append(m[: self.n_t][:, self.free].tocsr())
        return out[0], out[1]

    @functools.cached_property
    def gauge_forms(self) -> tuple[np.ndarray, np.ndarray]:
        """``l_n`` and ``l_t`` with ``gauge(c, gamma) = -i (l_n + gamma l_t) . c``.

        Free-DOF vectors; see `gauge`.
        """
        _, (vt, vn) = self.space.TnT()
        weight = self._quadrant() * self._coef["nu_tt"]
        t = self._gauge_direction()
        dx = self._dx()
        ln = self._vector((t * (weight * ngs.grad(vn))) * dx)
        lt = self._vector((t * (weight * vt)) * dx)
        return ln, lt

    def _quadrant(self) -> ngs.CoefficientFunction:
        s = self.section
        glob = [
            s.origin[k] + ngs.x * s.frame[0, k] + ngs.y * s.frame[1, k]
            for k in (0, 1, 2)
        ]
        corners = s.to_global(s.points) / s.scale
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        da, db = np.argsort(hi - lo)[::-1][:2]
        # Closed (x >= c), with a margin: symmetric quadrature
        # rules put points on the medians of a structured mesh, which can lie
        # on a midline, and rounding would put them on either side.
        tol = _QUADRANT_RTOL * float((hi - lo).max())
        ca, cb = 0.5 * (lo + hi)[da] - tol, 0.5 * (lo + hi)[db] - tol
        return ngs.IfPos(glob[da] - ca, 1.0, 0.0) * ngs.IfPos(glob[db] - cb, 1.0, 0.0)

    def _gauge_direction(self) -> ngs.CoefficientFunction:
        s = self.section
        return ngs.CF((float(s.frame[0].sum()), float(s.frame[1].sum())))

    # --- the sweep's protocol -------------------------------------------------

    def reduce(self, v: np.ndarray) -> SectionReduction:
        """The sweep's fixed field basis on the reduced basis ``v``.

        The pencil vector is ``(gamma E_t, E_n)``: ``v`` splits into a
        tangential and a normal block, so the field is affine in its
        coordinates.
        """
        return SectionReduction(self, v)

    def port_columns(self, basis: np.ndarray) -> dict[str, np.ndarray]:
        """What the 2D forms give per column of the sweep's ``basis``.

        ``r``: the entries of the coupling's ``vectors`` on the free 2D DOFs;
        ``reaction``: hᵀ e on the port face (the mode vanishes on fixed
        DOFs).
        """
        g_form, t_form = self.coupling_forms
        w = basis[self.free]
        m = w.shape[1] // 2
        r = np.concatenate([t_form @ w[:, :m], g_form @ w[:, m:]], axis=1)
        tangential = basis[: self.n_t, :m]
        return {"r": r, "reaction": r.T @ tangential}

    def phi(self, coords: np.ndarray, gamma: np.ndarray, omega: np.ndarray):
        """``(n_freq, 2 m)``: the mode's n x H on the columns of ``r``.

        ``gamma c_t`` and ``c_n``, ``gamma`` per file unit.
        """
        del omega
        m = coords.shape[1] // 2
        return np.concatenate([gamma[:, None] * coords[:, :m], coords[:, m:]], 1)

    def scalar(self, trace: np.ndarray, gamma0: complex, omega: float) -> np.ndarray:
        """``gamma0 M E`` on the columns of ``r``, for traces ``(m, n)``."""
        del omega
        return np.concatenate([gamma0 * trace, np.zeros_like(trace)])

    def pairing_factor(self, omegas: np.ndarray) -> np.ndarray:
        """``factor hᵀ e`` is the reciprocity pairing ``(1/2) int (e x h) . n``, W."""
        return -0.5j * self.section.scale / (omegas[:, None] * scipy.constants.mu_0)

    # --- fields ---------------------------------------------------------------

    def full(self, x: np.ndarray) -> np.ndarray:
        """A full coefficient vector from free-DOF values (zeros elsewhere)."""
        out = np.zeros(self.space.ndof, complex)
        out[self.free] = x
        return out

    def _gridfunction(self, coeffs: np.ndarray) -> ngs.GridFunction:
        gf = ngs.GridFunction(self.space)
        gf.vec.FV().NumPy()[:] = coeffs
        return gf

    def evaluate(self, coeffs: np.ndarray, local: np.ndarray) -> np.ndarray:
        """``(E_u, E_v, E_n)`` at plane points; zero off the cross-section.

        Args:
            coeffs: Full coefficient vectors of ``(E_t, E_n)``, ``(m, ndof)``.
            local: Plane coordinates ``(n, 2)`` in local units.

        Returns:
            ``(m, n, 3)`` complex.
        """
        coeffs = np.atleast_2d(coeffs)
        mips = [self.mesh(float(x), float(y)) for x, y in np.asarray(local, float)]
        found = [i for i, mip in enumerate(mips) if mip.nr >= 0]
        out = np.zeros((len(coeffs), len(mips), 3), complex)
        gf = ngs.GridFunction(self.space)
        et, en = gf.components
        for j, c in enumerate(coeffs):
            gf.vec.FV().NumPy()[:] = c
            for i in found:
                out[j, i, :2] = et(mips[i])
                out[j, i, 2] = en(mips[i])
        return out

    def point_columns(self, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The field at plane points as a linear map of the full DOF vector.

        Args:
            local: Plane coordinates ``(n, 2)`` in local units.

        Returns:
            ``dofs (n, k)`` the DOF numbers of the element holding each point
            (padded with 0) and ``weights (n, 3, k)`` such that ``(E_u, E_v,
            E_n)`` at point ``p`` is ``weights[p] @ x[dofs[p]]``; zero off the
            cross-section.
        """
        points = np.asarray(local, float)
        mips = [self.mesh(float(x), float(y)) for x, y in points]
        owners = [
            list(self.space.GetDofNrs(ngs.ElementId(ngs.VOL, mip.nr)))
            if mip.nr >= 0
            else []
            for mip in mips
        ]
        width = max((len(d) for d in owners), default=0)
        dofs = np.zeros((len(mips), width), np.int64)
        weights = np.zeros((len(mips), 3, width), complex)
        gf = ngs.GridFunction(self.space)
        et, en = gf.components
        values = gf.vec.FV().NumPy()
        for i, (mip, element) in enumerate(zip(mips, owners, strict=True)):
            for j, dof in enumerate(element):
                if dof < 0:  # an unnumbered DOF of the element
                    continue
                dofs[i, j] = dof
                values[:] = 0.0
                values[dof] = 1.0
                weights[i, :2, j] = et(mip)
                weights[i, 2, j] = en(mip)
        return dofs, weights

    def magnetic_parts(
        self, coeffs: np.ndarray, local: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""The two parts of the in-plane magnetic field at plane points.

        :math:`H_t = \frac{i}{\omega \mu_0 s} (h_n + \gamma h_t)` with
        :math:`h_n = R\, \nu_{tt} \nabla E_n` and :math:`h_t = R\, \nu_{tt}
        E_t` (Faraday under :math:`e^{+i\omega t}`, :math:`R a = a \times n`,
        :math:`s` the scale and :math:`\gamma` per local unit). Zero off the
        cross-section.

        Args:
            coeffs: Full coefficient vectors of ``(E_t, E_n)``, ``(m, ndof)``.
            local: Plane coordinates ``(n, 2)`` in local units.

        Returns:
            ``h_n`` and ``h_t``, each ``(m, n, 2)`` complex in ``(u, v)``.
        """
        coeffs = np.atleast_2d(coeffs)
        mips = [self.mesh(float(x), float(y)) for x, y in np.asarray(local, float)]
        found = [i for i, mip in enumerate(mips) if mip.nr >= 0]
        hn = np.zeros((len(coeffs), len(mips), 2), complex)
        ht = np.zeros_like(hn)
        gf = ngs.GridFunction(self.space)
        et, en = gf.components
        nu = self._coef["nu_tt"]
        cf_n, cf_t = nu * ngs.grad(en), nu * et
        for j, c in enumerate(coeffs):
            gf.vec.FV().NumPy()[:] = c
            for i in found:
                hn[j, i] = cf_n(mips[i])
                ht[j, i] = cf_t(mips[i])
        # R a = a x n = (a_v, -a_u) in the right-handed frame (u, v, n)
        rot = np.array([[0.0, 1.0], [-1.0, 0.0]])
        return hn @ rot.T, ht @ rot.T

    def power(self, coeffs: np.ndarray, gamma: complex, omega: float) -> complex:
        """Complex power ``1/2 int (E x H*) . n dA`` through the port, W.

        Over the triangles outside a perfectly matched layer: the field in
        the layer is not physical, so the 1 W of a mode does not depend on
        the layer's thickness or target reflection.

        Args:
            coeffs: Full coefficient vector of ``(E_t, E_n)`` (V/m).
            gamma: Propagation constant per local unit.
            omega: Angular frequency, rad/s.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        mu0 = scipy.constants.mu_0
        cf = self._coef["physical"] * (et * ngs.Conj(self._coef["nu_tt"] * a))
        value = ngs.Integrate(cf, self.mesh, order=quadrature.degree(self.order))
        return 0.5 * self.section.scale * 1j / (omega * mu0) * complex(value)

    def pairing(self, coeffs: np.ndarray, gamma: complex, omega: float) -> complex:
        """Reciprocity pairing ``1/2 int (E x H) . n dA``, unconjugated, W.

        `power` without the conjugate: 1 W for a lossless 1 W mode of real
        phase; the normalization of a wave port's modal waves.

        Args:
            coeffs: Full coefficient vector of ``(E_t, E_n)`` (V/m).
            gamma: Propagation constant per local unit.
            omega: Angular frequency, rad/s.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        mu0 = scipy.constants.mu_0
        cf = et * (self._coef["nu_tt"] * a)
        value = ngs.Integrate(cf, self.mesh, order=quadrature.degree(self.order))
        return -0.5j * self.section.scale / (omega * mu0) * complex(value)

    def gauge(self, coeffs: np.ndarray, gamma: complex) -> complex:
        """Phase reference: ``int_Q t . (n_out x H) dA`` up to a factor.

        ``Q`` is the closed upper-right quadrant of the port's bounding box
        along its two largest global extents and ``t`` the in-plane part of
        (1, 1, 1); the factor is real and positive.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        inside = self._quadrant()
        t = self._gauge_direction()
        cf = inside * (t * (-1j * (self._coef["nu_tt"] * a)))
        return complex(
            ngs.Integrate(cf, self.mesh, order=quadrature.degree(self.order))
        )


@dataclasses.dataclass(eq=False)
class SectionReduction:
    """A port mode's field basis, power and gauge on a fixed reduced basis.

    Attributes:
        space: The port mode problem.
        v: ``(n_free, m)`` the reduced basis of pencil vectors.
    """

    space: PortModeSpace
    v: np.ndarray

    def __post_init__(self) -> None:
        space, v = self.space, self.v
        m = v.shape[1]
        tangential = space.free < space.n_t
        w = np.zeros((len(space.free), 2 * m))
        w[tangential, :m] = v[tangential]
        w[~tangential, m:] = v[~tangential]
        self.w = w
        g_form, t_form = space.power_forms
        self._g, self._t = w.T @ (g_form @ w), w.T @ (t_form @ w)
        ln, lt = space.gauge_forms
        self._ln, self._lt = ln @ w, lt @ w
        # coordinate k of the field is pencil coordinate `source[k]`, over
        # gamma where `tangential`
        self.source = np.concatenate([np.arange(m), np.arange(m)])
        self.tangential = np.arange(2 * m) < m

    def forms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The power forms ``G`` and ``T`` and the gauge ``l_n`` and ``l_t``.

        On the field coordinates (:math:`w^T G w` and so on).
        """
        return self._g, self._t, self._ln, self._lt

    def coords(self, y: np.ndarray, gamma: complex) -> np.ndarray:
        """The field's coordinates ``(y / gamma, y)`` of a pencil vector's."""
        return np.concatenate([y / gamma, y])

    def power(self, z: np.ndarray, gamma: complex, omega: complex) -> complex:
        r"""Complex power through the port, W (`PortModeSpace.power`).

        ``omega`` may be complex: :math:`H^*` carries :math:`1/\omega^*`.
        """
        return (
            0.5
            * self.space.section.scale
            * 1j
            / (np.conj(omega) * scipy.constants.mu_0)
            * (
                z @ (self._g @ z.conj())
                + gamma.conjugate() * (z @ (self._t @ z.conj()))
            )
        )

    def gauge(self, z: np.ndarray, gamma: complex, omega: float) -> complex:
        """Phase reference (`PortModeSpace.gauge`)."""
        del omega
        return -1j * (self._ln @ z + gamma * (self._lt @ z))


@dataclasses.dataclass(eq=False)
class FullReduction:
    """`SectionReduction`'s protocol on the whole free-DOF vector, no reduced basis.

    The coordinates are the field's own ``(E_t, E_n)`` on the free DOFs, so
    the forms are the space's sparse matrices and a cotangent may point
    anywhere in the space: a backward pass that pulls a normalization back
    to the pencil vector needs that, since a reduced basis (one column for a
    lossless mode) holds only part of the directions the field can move.

    Attributes:
        space: The port mode problem.
    """

    space: PortModeSpace

    def __post_init__(self) -> None:
        space = self.space
        n = len(space.free)
        self.w = sp.identity(n, format="csr")
        self._g, self._t = space.power_forms
        self._ln, self._lt = space.gauge_forms
        self.source = np.arange(n)
        self.tangential = space.free < space.n_t

    def forms(self) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        """The sparse power forms ``G``, ``T`` and the gauge ``l_n``, ``l_t``."""
        return self._g, self._t, self._ln, self._lt

    def coords(self, y: np.ndarray, gamma: complex) -> np.ndarray:
        """The field ``(y_t / gamma, y_n)`` of a pencil vector."""
        return np.where(self.tangential, y / gamma, y)

    def power(self, z: np.ndarray, gamma: complex, omega: complex) -> complex:
        """Complex power through the port, W (`SectionReduction.power`)."""
        return (
            0.5
            * self.space.section.scale
            * 1j
            / (np.conj(omega) * scipy.constants.mu_0)
            * (
                z @ (self._g @ z.conj())
                + gamma.conjugate() * (z @ (self._t @ z.conj()))
            )
        )

    def gauge(self, z: np.ndarray, gamma: complex, omega: float) -> complex:
        """Phase reference (`PortModeSpace.gauge`)."""
        del omega
        return -1j * (self._ln @ z + gamma * (self._lt @ z))


def _build_mesh(section: PortSection) -> ngs.Mesh:
    m = ngm.Mesh(dim=2)  # ty: ignore[unresolved-attribute]
    pts = np.zeros((len(section.points), 3))
    pts[:, :2] = section.points
    m.AddPoints(pts)
    m.Add(ngm.FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))  # ty: ignore[unresolved-attribute]
    m.SetMaterial(1, "port")
    m.AddElements(dim=2, index=1, data=section.triangles, base=0)
    m.SetBCName(0, "pec")
    for edge in section.pec_edges:
        m.Add(ngm.Element1D([int(edge[0]) + 1, int(edge[1]) + 1], index=1))  # ty: ignore[unresolved-attribute]
    groups = _admittance_groups(section)
    for g, (_, edges) in enumerate(groups):
        m.SetBCName(g + 1, f"absorbing{g}")
        for edge in edges:
            m.Add(ngm.Element1D([int(edge[0]) + 1, int(edge[1]) + 1], index=g + 2))  # ty: ignore[unresolved-attribute]
    first = len(groups) + 1
    for g in range(len(section.impedance_terms)):
        m.SetBCName(first + g, f"impedance{g}")
        mask = section.impedance_group == g
        for edge in section.impedance_edges[mask]:
            m.Add(
                ngm.Element1D(  # ty: ignore[unresolved-attribute]
                    [int(edge[0]) + 1, int(edge[1]) + 1], index=first + g + 1
                )
            )
    first += len(section.impedance_terms)
    for g in range(len(section.laws)):
        m.SetBCName(first + g, f"law{g}")
        if section.absorbing_law is None:
            continue
        for edge in section.absorbing_edges[section.absorbing_law == g]:
            m.Add(
                ngm.Element1D(  # ty: ignore[unresolved-attribute]
                    [int(edge[0]) + 1, int(edge[1]) + 1], index=first + g + 1
                )
            )
    return ngs.Mesh(m)


def _edge_fill(mesh: ngs.Mesh, section: PortSection) -> ngs.CoefficientFunction:
    """The sheet fields' fill per impedance edge (1 without a sheet field)."""
    if section.impedance_fill is None or (section.impedance_fill == 1.0).all():
        return ngs.CF(1.0)
    fill = {
        (int(min(a, b)), int(max(a, b))): float(w)
        for (a, b), w in zip(
            section.impedance_edges, section.impedance_fill, strict=True
        )
    }
    space = ngs.SurfaceL2(mesh, order=0)
    gf = ngs.GridFunction(space)
    values = gf.vec.FV().NumPy()
    values[:] = 1.0
    for el in mesh.Elements(ngs.BND):
        a, b = sorted(v.nr for v in el.vertices)
        if (a, b) in fill:
            values[space.GetDofNrs(el)[0]] = fill[a, b]
    return gf


def _admittance_groups(
    section: PortSection,
) -> list[tuple[tuple[float, float], np.ndarray]]:
    """The absorbing edges grouped by their ``(Y_tau, Y_n)``.

    Not those beside a law triangle: each law's own edge piece holds them.
    """
    keep = np.ones(len(section.absorbing_edges), bool)
    if section.absorbing_law is not None:
        keep = section.absorbing_law < 0
    values, inverse = np.unique(
        section.absorbing_admittance[keep].reshape(-1, 2), axis=0, return_inverse=True
    )
    edges = section.absorbing_edges[keep]
    return [
        ((float(v[0]), float(v[1])), edges[inverse.ravel() == g])
        for g, v in enumerate(values)
    ]


def _p0(mesh: ngs.Mesh, values: np.ndarray) -> ngs.CoefficientFunction:
    """A piecewise-constant coefficient, one value per element (mesh order)."""
    values = np.asarray(values)
    if np.abs(values - values[0]).max() <= 1e-14 * max(np.abs(values).max(), 1.0):
        return ngs.CF(complex(values[0]))
    gf = ngs.GridFunction(ngs.L2(mesh, order=0, complex=True))
    gf.vec.FV().NumPy()[:] = values
    return gf


def _coefficients(
    mesh: ngs.Mesh, s: PortSection, bend_u: ngs.Parameter | None = None
) -> dict[str, ngs.CoefficientFunction]:
    def matrix(t: np.ndarray) -> ngs.CoefficientFunction:
        parts = [_p0(mesh, t[:, i, j]) for i in (0, 1) for j in (0, 1)]
        return ngs.CF(tuple(parts), dims=(2, 2))

    groups = _admittance_groups(s)
    y_tau, y_n = (
        mesh.BoundaryCF(
            {f"absorbing{g}": v[k] for g, (v, _) in enumerate(groups)},
            default=ngs.CF(0.0),
        )
        for k in (0, 1)
    )
    out = {
        "eps_tt": matrix(s.eps_tt),
        "eps_nn": _p0(mesh, s.eps_nn),
        "nu_tt": matrix(s.nu_tt),
        "nu_nn": _p0(mesh, s.nu_nn),
        "sigma_tt": matrix(s.sigma_tt.astype(complex)),
        "sigma_nn": _p0(mesh, s.sigma_nn.astype(complex)),
        "Y_tau": y_tau,
        "Y_n": y_n,
        "physical": ngs.CF(1.0),
    }
    layered = s.stretch is not None and s.layer is not None
    if layered:
        layer = np.asarray(s.layer).astype(float)
        out["physical"] = _p0(mesh, 1.0 - layer)
        out["layer"] = _p0(mesh, layer)
        out["eps_bg_tt"] = matrix(s.eps_tt.real.astype(complex))
        out["eps_bg_nn"] = _p0(mesh, s.eps_nn.real.astype(complex))
    elif s.bend is None:
        return out
    lam_tt, lam_nn = _section_stretch(s, bend_u)
    det = lam_tt[0, 0] * lam_tt[1, 1] - lam_tt[0, 1] * lam_tt[1, 0]
    for name in ("eps_tt", "sigma_tt"):
        out[name] = out[name] * lam_tt
    for name in ("eps_nn", "sigma_nn"):
        out[name] = out[name] * lam_nn
    # n x H of the stretched medium:
    # $R^\mathsf{T}(\mu_{tt}\Lambda_{tt})^{-1}R$ with $Ra = a\times n$ is
    # $(R^\mathsf{T}\Lambda_{tt}^{-1}R)\,\nu_{tt}
    # = (\Lambda_{tt}^\mathsf{T}/\det\Lambda_{tt})\,\nu_{tt}$
    out["nu_tt"] = ((1.0 / det) * lam_tt.trans) * out["nu_tt"]
    out["nu_nn"] = out["nu_nn"] / lam_nn
    return out


def _section_stretch(
    s: PortSection, bend_u: ngs.Parameter | None = None
) -> tuple[ngs.CoefficientFunction, ...]:
    r"""The section's metric, port frame: :math:`\Lambda_{tt}`, :math:`\Lambda_{nn}`.

    The PML tensor :math:`F \Lambda F^\mathsf{T}` with :math:`F` the frame's
    rows, the 3D tensor evaluated at the global position of each plane
    point (the identity without a layer), times the bend's metric
    :math:`\mathrm{diag}(h, h, 1/h)`: :math:`\Lambda_{tt}\to h\Lambda_{tt}`,
    :math:`\Lambda_{nn}\to\Lambda_{nn}/h` (:math:`h` from the stretched
    coordinate inside the layer, `_bend_factor`). A port normal across every
    stretch axis it meets (checked when the model is built) leaves no
    tangential-normal block. ``bend_u`` is the parameter :math:`1/R` of
    the bend, per local unit.
    """
    stretch, f = s.stretch, s.frame
    if stretch is None or s.layer is None:
        tt = ngs.CF((1, 0, 0, 1), dims=(2, 2))
        nn = ngs.CF(1.0)
    else:
        tt, nn = _pml_metric(s, stretch, f)
    if s.bend is None:
        return tt, nn
    h = _bend_factor(s, bend_u)
    return h * tt, nn / h


def _bend_factor(
    s: PortSection, bend_u: ngs.Parameter | None
) -> ngs.CoefficientFunction:
    r"""The bend's :math:`h = 1 + \tilde\xi/R` on the section's plane points.

    :math:`\tilde\xi` is the distance from the reference arc along
    ``outward`` in the stretched coordinate: a layer's complex shift along a
    stretch axis enters through that axis' component of ``outward``, so a
    guided mode sees the real :math:`h` and the layer a complex one that
    keeps :math:`\mathrm{Re}\,h` the physical value.
    """
    bend, f = s.bend, s.frame
    assert bend is not None  # noqa: S101 - callers check
    u = ngs.Parameter(s.scale / bend.radius) if bend_u is None else bend_u
    xi = bend.outward[0] * ngs.x + bend.outward[1] * ngs.y - bend.offset
    if s.stretch is not None and s.layer is not None:
        glob = tuple(s.origin[k] + ngs.x * f[0, k] + ngs.y * f[1, k] for k in range(3))
        outward = np.asarray(bend.outward) @ f[:2]
        for side in s.stretch.layers.sides:
            if abs(outward[side.axis]) < 1e-12:
                continue
            smax = s.stretch.stretch_maximum(side)
            d = side.thickness / s.scale
            depth = side.sign * (glob[side.axis] - side.face / s.scale)
            shift = ngs.IfPos(depth, smax * d / 3 * (depth / d) ** 3, 0.0)
            xi = xi - 1j * float(outward[side.axis]) * side.sign * shift
    return 1 + xi * u


def _pml_metric(
    s: PortSection, stretch: Any, f: np.ndarray
) -> tuple[ngs.CoefficientFunction, ngs.CoefficientFunction]:
    r"""The PML's :math:`\Lambda_{tt}` and :math:`\Lambda_{nn}` in the port frame."""
    glob = tuple(s.origin[k] + ngs.x * f[0, k] + ngs.y * f[1, k] for k in range(3))
    sx, sy, sz = stretch_factors(stretch, s.scale, coords=glob)
    diag = (sy * sz / sx, sx * sz / sy, sx * sy / sz)

    def entry(a: int, b: int) -> ngs.CoefficientFunction:
        terms = [
            float(f[a, i] * f[b, i]) * diag[i]
            for i in range(3)
            if abs(f[a, i] * f[b, i]) > 0
        ]
        return sum(terms[1:], terms[0]) if terms else ngs.CF(0.0)

    tt = ngs.CF(tuple(entry(a, b) for a in (0, 1) for b in (0, 1)), dims=(2, 2))
    return tt, entry(2, 2)
