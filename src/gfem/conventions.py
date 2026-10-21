"""Sign, unit and normalization conventions of gfem's results.

Every complex result and every docstring that mentions a phasor follows
the conventions below. Each section is self-contained; new physics adds a
section rather than changing an existing one, because changing a
convention silently changes users' numbers.

Units
-----
SI at the API: metres, seconds, hertz, volts, amperes, ohms, siemens,
farads, joules, watts, kelvin (reserved for thermal problems). Only the
mesh file declares its own length unit (``Mesh.from_gmsh(...,
length_unit=)``); gfem converts on import, and every field value is in SI
(an electric field in V/m, whatever the mesh unit).

Time dependence
---------------
Time-harmonic quantities are complex phasors under e^{+iωt}: the physical
field is ``Re(E exp(+iωt))``. This is Palace's convention (its operator
is ``K + iωC - ω²M``) and the electrical engineering one (Touchstone,
scikit-rf, SPICE write ``j`` for ``i``). Consequences:

- a wave travelling along ``+s`` is ``exp(-gamma s)``, with
  ``gamma = alpha + i beta`` and attenuation ``alpha >= 0``;
- a lossy permittivity is ``ε_r (1 - i tan δ)`` (negative imaginary part);
- an inductor's impedance is ``+iωL``, a capacitor's ``1 / (iωC)``;
- a decaying eigenmode has ``Im ω > 0``.

Phasors and field amplitude
---------------------------
Phasors are peak values, not RMS: ``|E|`` is the amplitude of the
oscillation. A driven field (``Driven(keep_fields=...)``) is the field
for 1 W time-averaged incident power at the excited port, with every other
port terminated in its reference impedance ``z_ref`` (a lumped port's L
and C, if any, belong to the network and load it in parallel). It is the
field behind one column of S: the excited port's incident wave is
``a = 1`` (√W), every other port's is 0.
Eigenmode fields store 1 J (see Eigenmodes). A port mode field is
normalized to 1 W time-averaged power through the port.

Network parameters
------------------
S-parameters are power waves (Kurokawa) with a reference impedance per
port and per frequency, ``z_ref``:

    a_i = (V_i + z_i I_i) / (2 sqrt(Re z_i)),
    b_i = (V_i - conj(z_i) I_i) / (2 sqrt(Re z_i)),   b = S a,

with ``V``, ``I`` peak port voltage and current, so ``|a_i|^2`` is the
time-averaged incident power in W (``|a| = 1`` is 1 W). When ``z_ref`` is
real, power waves and pseudo-waves coincide.

- A `LumpedPort`'s ``z_ref`` is its resistance. A port without a
  resistance has no reference impedance: its ``z_ref`` and its row of S are
  NaN, and it cannot be excited.
- A `WavePort`'s S is normalized to unit modal power; its ``z_ref`` is the
  power-voltage impedance ``Z_PV`` along its ``voltage_path`` (NaN without
  one). As in Palace, a wave port's S is the projection of the field on
  the conjugated port mode, with the mode's magnetic field taken at its
  complex wavenumber (Palace v0.18.1), and the mode scaled so that this projection
  of the mode itself carries 1 W; for a lossless mode this is the modal
  power wave. ``WavePort(offset=d)`` refers S to a plane ``d`` inside the
  domain: ``S_ij`` times ``exp(gamma_i d_i) exp(gamma_j d_j)``.
- ``NetworkParameters.y()``, ``.z()`` and ``.renormalize(z_new)`` use the
  power-wave formulas; ``to_touchstone`` labels a file with ``z_ref`` only
  when it is one real value, and otherwise needs ``z0=`` to renormalize
  first. gfem never labels a port 50 Ω unless S was computed or
  renormalized for 50 Ω.

Time-averaged densities
-----------------------
For peak phasors, the time average of a product of two harmonic quantities
carries a factor 1/2:

- electric energy density ``w_e = Re(E · D*) / 4`` (1/2 for the time
  average of ``(E · D) / 2``), magnetic ``w_m = Re(B · H*) / 4``;
- Poynting vector ``S = Re(E x H*) / 2``;
- ohmic loss density ``p = Re(sigma_eff) |E|^2 / 2``;
- a lumped port's power ``|V|^2 / (2 R)``.

With 1 W incident, the power a driven field delivers into the network
through the excited port is ``1 - |S_ii|^2`` W, and the power arriving at
port j's termination is ``|S_ji|^2`` W.

Surface conductors
------------------
A `Conductor` face adds the boundary term ``iω/Z_s(ω) ∫ E_t · v_t``, with
``Z_s`` the surface impedance of a metal of conductivity sigma and
thickness t in the finite-thickness form of HFSS and Palace
(``surfaceconductivityoperator.cpp``)::

    Z_s = (1 / (sigma δ)) [(sinh nu + sin nu) + i (sinh nu - sin nu)]
          / (cosh nu - cos nu)

with skin depth ``δ = sqrt(2 / (μ0 sigma ω))`` and ``nu = t / δ``
(``thickness=None``: ``Z_s = (1 + i) / (sigma δ)``). This form describes ONE
face of a film whose other face is modelled separately, as the shell of an
excluded metal volume is: for a film much thinner than its skin depth,
``Z_s → 2 / (sigma t)``.

- **Interior surface** (solve domain on both sides, e.g. a zero-thickness
  sheet): one term for the whole sheet, so a thin film is a sheet
  admittance of about ``sigma t / 2`` (`INTERIOR_SHEET_ADMITTANCE` times
  sigma t), not the film's conductance sigma t. To model a zero-thickness
  film of thickness t with its full conductance, give ``thickness=2 * t``.
- **Exterior surface** (mesh boundary, one-sided): the thickness counts
  twice (`EXTERIOR_THICKNESS_FACTOR`), so a thin film is ``sigma t``.
- **Shell of an excluded volume** (dielectric on one side, the excluded
  metal on the other): an interior face of the mesh, one term per face with
  the given thickness; the top and bottom faces of a trace together carry
  ``sigma t``.

This is Palace's convention face for face, so gsim layouts give the same
discrete problem in both codes.

Eigenmodes
----------
An eigenmode is a complex angular frequency ``omega`` and a field ``E``
with ``Re(E exp(+iωt))`` source-free; ``Im omega > 0`` decays, and
``Q = |omega| / (2 |Im omega|)`` (Palace's definition). The mode's
electric energy is::

    E_elec = 1/2 Re ∫ D* · E dV + Σ C_j |V_j|^2 / 2

over the solve domain and the capacitive lumped ports (peak phasors: the
total time-averaged energy of a lossless mode, electric plus magnetic, as
Palace's ``E_elec``). Each mode field is scaled to ``E_elec`` =
`MODE_ENERGY` and phased so that ``∫ D · E dV`` (no conjugate) is real
and positive, which makes a lossless mode's field real; the remaining sign
puts the largest coefficient's real part positive (among coefficients equal
to the largest within a relative 1e-12, the lowest index). Participation ratios
and coupling Q are ratios to ``E_elec`` and do not depend on this scale.
"""

from __future__ import annotations

__all__ = [
    "EXTERIOR_THICKNESS_FACTOR",
    "INCIDENT_POWER",
    "INTERIOR_SHEET_ADMITTANCE",
    "MODE_ENERGY",
    "PHASOR",
    "TIME_DEPENDENCE",
    "WAVES",
]

#: The time dependence of every complex phasor.
TIME_DEPENDENCE = "exp(+iωt)"
#: Phasors are peak values (not RMS).
PHASOR = "peak"
#: Time-averaged incident power at the excited port of a driven field, W.
INCIDENT_POWER = 1.0
#: The wave definition of S-parameters.
WAVES = "power"
#: Sheet admittance of one `Conductor` term on an interior surface for a film
#: much thinner than its skin depth, in units of sigma t (conductivity times
#: thickness).
INTERIOR_SHEET_ADMITTANCE = 0.5
#: Factor on a `Conductor`'s thickness on mesh-exterior faces, where the metal
#: is one-sided.
EXTERIOR_THICKNESS_FACTOR = 2.0
#: Electric energy of every eigenmode field (see Eigenmodes), J.
MODE_ENERGY = 1.0
