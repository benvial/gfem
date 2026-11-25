r"""Sign, unit and normalization conventions of gfem's results.

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
Time-harmonic quantities are complex phasors under :math:`e^{+i\omega t}`:
the physical field is :math:`\mathrm{Re}(E\, e^{+i\omega t})`, so the
operator is :math:`K + i\omega C - \omega^2 M`. This
is the electrical engineering convention (Touchstone, scikit-rf, SPICE write
:math:`j` for :math:`i`). Consequences:

- a wave travelling along :math:`+s` is :math:`e^{-\gamma s}`, with
  :math:`\gamma = \alpha + i\beta` and attenuation :math:`\alpha \ge 0`;
- a lossy permittivity is
  :math:`\varepsilon_r (1 - i \tan\delta) - i \sigma / (\varepsilon_0 \omega)`,
  per principal axis for an anisotropic material (negative imaginary part);
- an inductor's impedance is :math:`+i\omega L`, a capacitor's
  :math:`1 / (i\omega C)`;
- a decaying eigenmode has :math:`\mathrm{Im}\,\omega > 0`.

Phasors and field amplitude
---------------------------
Phasors are peak values, not RMS: :math:`|E|` is the amplitude of the
oscillation. A driven field (``Driven(keep_fields=...)``) is the field
for 1 W time-averaged incident power at the excited port, with every other
port terminated in its reference impedance ``z_ref`` (a lumped port's L
and C, if any, belong to the network and load it in parallel). A wave
port is terminated by absorbing its mode (matched), including the other
modes of the excited port's own surface. It is the field behind one
column of S: the excited port's incident wave is
:math:`a = 1` (√W), every other port's is 0.
Eigenmode fields store 1 J (see Eigenmodes). A port mode field is
normalized to 1 W time-averaged power through the port.

Network parameters
------------------
S-parameters are pseudo-waves (Marks and Williams) with a reference
impedance per port and per frequency, ``z_ref``:

.. math::

    a_i = \frac{\sqrt{\mathrm{Re}\, z_i}}{2 |z_i|} (V_i + z_i I_i), \qquad
    b_i = \frac{\sqrt{\mathrm{Re}\, z_i}}{2 |z_i|} (V_i - z_i I_i), \qquad
    b = S a,

with :math:`V`, :math:`I` peak port voltage and current, scaled so that
:math:`|a| = 1` is 1 W incident on a real reference. When ``z_ref`` is
real, pseudo-waves and power waves (Kurokawa) coincide; on a complex
reference, a line matched to its own characteristic impedance has
:math:`S_{11} = 0`, which power waves do not give.

- A `LumpedPort`'s ``z_ref`` is its resistance. A port without a
  resistance has no reference impedance: its ``z_ref`` and its row of S are
  NaN, and it cannot be excited.
- A `WavePort`'s S reads the field's modal amplitudes by the biorthogonal
  projection on the port surface's modes, :math:`c = R^{-1} H^T E` with
  :math:`R_{lk} = \int (n \times h_l) \cdot e_k` unconjugated, which by
  reciprocity separates the modes and matches a uniform line to its own
  mode, lossy or not. With a ``voltage_path`` its ``z_ref`` is the complex
  power-voltage impedance :math:`Z_{PV} = |V|^2 / (2 P^*)` of its mode, so
  the port's waves are the pseudo-waves of the mode's voltage and current;
  a lossless line has a real ``z_ref`` and unit-power modal waves. Without
  one, ``z_ref`` is NaN and the port's waves are its modal amplitudes
  normalized by reciprocity, :math:`\tfrac12 \int (e \times h) \cdot n = 1`
  unconjugated (1 W on a lossless line), which keeps S symmetric for a
  reciprocal device. S is symmetric when every port has the same kind of
  reference (all real, all wave ports with equal ``z_ref``, or all modal);
  a wave port with a ``voltage_path`` beside one without breaks the
  symmetry by the ratio of their normalizations on a lossy line.
  ``WavePort(offset=d)`` refers S to a plane :math:`d` inside the
  domain: :math:`S_{ij}` times :math:`e^{\gamma_i d_i} e^{\gamma_j d_j}`.
  Without a ``voltage_path`` or a ``polarity``, the sign of a wave port's
  row and column of S is fixed by a mesh-dependent rule and, in 3D, by the
  model's orientation in the global frame; every :math:`|S|` and power is
  unaffected. With a ``polarity``, the mode's field at the given point
  along the given direction (the field normal to the plane on a line port)
  is real and positive at every frequency.
- ``NetworkParameters.y()``, ``.z()`` and ``.renormalize(z_new)`` use the
  pseudo-wave formulas; ``to_touchstone`` labels a file with ``z_ref`` only
  when it is one real value, and otherwise needs ``z0=`` to renormalize
  first. gfem never labels a port 50 Ω unless S was computed or
  renormalized for 50 Ω, or no port has a reference impedance (wave ports
  without ``voltage_path``): then an explicit ``z0=`` is a nominal label,
  S is written unchanged and the file says so in a comment. On a shared
  port surface each mode is its own port, so ``y()``, ``z()`` and
  ``renormalize()`` act on modal voltages and currents, not on the
  conductors' terminal ones.
- Under a Bloch wave vector (see Bloch periodicity), S is referenced to the
  field E. A solver that reports the periodic envelope
  :math:`u = E\, e^{+i k \cdot r}` gives
  :math:`S_{ij}\, e^{+i k \cdot (r_i - r_j)}`, with i the output port, j the
  excited port and :math:`r_p` a point of port p, when :math:`k \cdot r` is
  constant over each port.

Time responses
--------------
``NetworkParameters.impulse_response`` and ``step_response`` transform S
to time under the same :math:`e^{+i\omega t}`, so positive time is later:
:math:`h(t) = \int S(f)\, e^{+2\pi i f t}\, df`, the plain inverse transform
with no conjugation. Data from a code that writes :math:`e^{-i\omega t}`
must be conjugated first, or its reflections land at negative times.

- Low pass needs a harmonic grid, :math:`f_k = k\,\Delta f` for
  :math:`k = 1..N`, plus an optional 0 Hz point. Without one, DC is
  :math:`S(0) = \mathrm{Re}(15 S_1 - 6 S_2 + S_3)/10`, the even polynomial
  in :math:`f` through the first three points. S is weighted by the upper
  half of a Kaiser window of parameter :math:`\beta` (6 by default).
- The record is two-sided and periodic,
  :math:`t_m = (m - n_t/2) / (n_t\,\Delta f)` for :math:`m = 0..n_t - 1`,
  of length :math:`1/\Delta f`.
- The impulse is divided by the window kernel's peak, so it is
  dimensionless and an isolated reflection :math:`\Gamma` peaks at
  :math:`\Gamma`. The step is the cumulative sum of the impulse over the
  whole record, before that division, and ends at :math:`S(0)`.
- Band pass (impulse only): any evenly spaced band, a full Kaiser window,
  and the magnitude :math:`|h(t)|`, normalized the same way.
- The impedance profile of a port with a real reference :math:`Z_0` is
  :math:`Z(t) = Z_0 \frac{1 + \rho(t)}{1 - \rho(t)}`, with :math:`\rho` its
  reflected step.

Rational models
---------------
A `gfem.RationalModel` writes S as poles shared by every element,
residue matrices and a constant:

.. math::

    S(f) = D + \sum_n \frac{R_n}{2\pi i f - p_n},

with :math:`s = i\omega` the Laplace variable under :math:`e^{+i\omega t}`,
poles :math:`p_n` in rad/s with :math:`\mathrm{Re}\, p_n < 0` (stable), real
or in conjugate pairs with conjugate residues (a response real in time),
and :math:`D` real, S as :math:`f \to \infty`. There is no term
proportional to :math:`s`. The DC limit is :math:`D - \sum_n R_n / p_n`.
The response to a unit step of incident wave at :math:`t = 0` is
:math:`D + \sum_n \frac{R_n}{p_n} (e^{p_n t} - 1)` for :math:`t \ge 0`
and 0 before.

S is referred to one constant real reference impedance per port
(``z_ref``), fixed when the model is fitted
(`gfem.NetworkParameters.fit_rational`): a real ``z_ref`` constant over
frequency is kept, anything else is renormalized to ``z0=`` first; a network
without reference impedances (wave ports without ``voltage_path``) is
fitted unchanged and its ``z0`` is only a label. On that reference the model
is passive when the largest singular value of :math:`S(f)` is at most 1 at
every frequency.

Time-averaged densities
-----------------------
For peak phasors, the time average of a product of two harmonic quantities
carries a factor 1/2:

- electric energy density :math:`w_e = \mathrm{Re}(E \cdot D^*) / 4`
  (1/2 for the time average of :math:`(E \cdot D) / 2`), magnetic
  :math:`w_m = \mathrm{Re}(B \cdot H^*) / 4`;
- Poynting vector :math:`S = \mathrm{Re}(E \times H^*) / 2`;
- ohmic loss density :math:`p = \mathrm{Re}(\sigma_\mathrm{eff})\, |E|^2 / 2`;
- sheet loss per area :math:`\mathrm{Re}(1 / Z_s)\, |E_t|^2 / 2` on a
  `Conductor` and :math:`|E_t|^2 / (2 R)` on an `Impedance`, times the
  sheet's fill (`gfem.dissipated_power` integrates both densities);
- a lumped port's power :math:`|V|^2 / (2 R)`.

With 1 W incident, the power a driven field delivers into the network
through the excited port is :math:`1 - |S_{ii}|^2` W, and the power arriving
at port j's termination is :math:`|S_{ji}|^2` W. On a model whose ports all
have real references and that radiates nothing, the power the field
dissipates inside, :math:`1 - \sum_j |S_{ji}|^2`, is the sum of
`gfem.dissipated_power`. On a complex reference a field is per unit
:math:`|a|^2`, which is not exactly 1 W incident.

Surface conductors
------------------
A `Conductor` face adds the boundary term
:math:`\frac{i\omega}{Z_s(\omega)} \int E_t \cdot v_t`, with
:math:`Z_s` the surface impedance of a metal of conductivity :math:`\sigma`
and thickness :math:`t` in the finite-thickness form of HFSS:

.. math::

    Z_s = \frac{1}{\sigma \delta}
          \frac{(\sinh\nu + \sin\nu) + i (\sinh\nu - \sin\nu)}
               {\cosh\nu - \cos\nu}

with skin depth :math:`\delta = \sqrt{2 / (\mu_0 \sigma \omega)}` and
:math:`\nu = t / \delta` (``thickness=None``:
:math:`Z_s = (1 + i) / (\sigma \delta)`). This form describes ONE
face of a film whose other face is modelled separately, as the shell of an
excluded metal volume is: for a film much thinner than its skin depth,
:math:`Z_s \to 2 / (\sigma t)`.

- **Interior surface** (solve domain on both sides, e.g. a zero-thickness
  sheet): one term for the whole sheet, so a thin film is a sheet
  admittance of about :math:`\sigma t / 2` (`INTERIOR_SHEET_ADMITTANCE`
  times :math:`\sigma t`), not the film's conductance :math:`\sigma t`. To
  model a zero-thickness film of thickness t with its full conductance,
  give ``thickness=2 * t``.
- **Exterior surface** (mesh boundary, one-sided): the thickness counts
  twice (`EXTERIOR_THICKNESS_FACTOR`), so a thin film is :math:`\sigma t`.
- **Shell of an excluded volume** (dielectric on one side, the excluded
  metal on the other): an interior face of the mesh, one term per face with
  the given thickness; the top and bottom faces of a trace together carry
  :math:`\sigma t`.

gsim layouts assume this convention face for face.

Dispersive permittivity
-----------------------
A `gfem.PermittivityLaw` is a relative permittivity that depends on
frequency,
:math:`\varepsilon_r(f) = \varepsilon_\infty + \sum \mathrm{term}(f)`.
Every law parameter that is a frequency is an ordinary frequency in hertz
(:math:`\omega = 2\pi f`), not rad/s or eV (convert an energy as
:math:`f = E / h`). Under :math:`e^{+i\omega t}` the terms are:

- `gfem.Drude`: :math:`-\omega_p^2 / (\omega (\omega - i\gamma))`,
  :math:`\omega_p = 2\pi \cdot` ``plasma_frequency``,
  :math:`\gamma = 2\pi \cdot` ``damping``; no static value (it diverges as
  :math:`f \to 0`);
- `gfem.Lorentz`:
  :math:`\Delta\varepsilon\, \omega_0^2 / (\omega_0^2 - \omega^2 + i\gamma\omega)`,
  :math:`\omega_0 = 2\pi \cdot` ``resonance_frequency``;
- `gfem.Debye`: :math:`\Delta\varepsilon / (1 + i f / f_r)`,
  :math:`f_r` = ``relaxation_frequency`` (a relaxation time :math:`\tau` is
  :math:`f_r = 1 / (2\pi\tau)`);
- `gfem.DjordjevicSarkar`:
  :math:`\frac{\Delta\varepsilon}{\ln(f_2 / f_1)} \ln\frac{f_2 + i f}{f_1 + i f}`,
  the principal log of the ratio, corners :math:`f_1` = ``low_frequency``
  and :math:`f_2` = ``high_frequency``.

:math:`\Delta\varepsilon` is each term's ``delta_eps``. With every strength
and damping :math:`\ge 0` each term is passive alone, so a law has
:math:`\mathrm{Im}\,\varepsilon_r \le 0` at every real frequency; a negative
strength is refused. A law evaluates at a complex frequency as the analytic
continuation (:math:`\mathrm{Im} f > 0` decays), and
``law(0)`` is the static permittivity. Most optics fits (Drude, Lorentz)
are written under :math:`e^{-i\omega t}`: there the same law is the complex
conjugate, so their :math:`+i\gamma` reads :math:`-i\gamma` here, with the
same parameter values. Circuit-board laws (Djordjevic-Sarkar) are usually
written under :math:`e^{+j\omega t}` already.

A measured table gives the refractive index :math:`n` and the
extinction coefficient :math:`k \ge 0`, which does not depend on the time
convention. Under :math:`e^{+i\omega t}` the index is :math:`n - ik`, so
:math:`\varepsilon = (n - ik)^2` and
:math:`\mathrm{Im}\,\varepsilon = -2nk`. `gfem.fit_permittivity` takes
:math:`n` and :math:`k` as tabulated; a permittivity table written under
:math:`e^{-i\omega t}` as :math:`\varepsilon_1 + i\varepsilon_2` goes in
as its complex conjugate.

Eigenmodes
----------
An eigenmode is a complex angular frequency :math:`\omega` and a field
:math:`E` with :math:`\mathrm{Re}(E\, e^{+i\omega t})` source-free;
:math:`\mathrm{Im}\,\omega > 0` decays, and
:math:`Q = |\omega| / (2 |\mathrm{Im}\,\omega|)`. The mode's
electric energy is:

.. math::

    E_\mathrm{elec} = \frac{1}{2} \mathrm{Re} \int D^* \cdot E \, dV
        + \sum_j \frac{C_j |V_j|^2}{2}
        + \sum \frac{1}{2} \int w\, C_s |E_t|^2 \, dS

over the solve domain, the capacitive lumped ports (capacitance :math:`C_j`,
voltage :math:`V_j`) and the capacitive `Impedance` sheets (capacitance per
square :math:`C_s`, tangential field :math:`E_t`, fill :math:`w`, 1 without
a `SheetField`) (peak phasors: the total time-averaged energy of a lossless
mode, electric plus magnetic).
In a region with a `gfem.PermittivityLaw`, :math:`\mathrm{Re}(D^* \cdot E)`
reads the energy permittivity
:math:`\mathrm{Re}[\varepsilon + (\omega / 2)\, d\varepsilon / d\omega]` at
the mode's real frequency, so that :math:`E_\mathrm{elec}` stays the total
energy of a lossless mode (Brillouin); for a lossy law it is the weak-loss
value. An eigenmode solve raises where that weight is not positive (loss
dominates the law there). The phase below reads the law's ``eps_inf`` in
:math:`D`. Each mode field is scaled to :math:`E_\mathrm{elec}` =
`MODE_ENERGY` and phased so that :math:`\int D \cdot E \, dV` (no conjugate)
is real and positive, which makes a lossless mode's field real; the
remaining sign puts the largest coefficient's real part positive (among
coefficients equal to the largest within a relative 1e-12, the lowest
index). Participation ratios and coupling Q are ratios to
:math:`E_\mathrm{elec}` and do not depend on this scale.

Bloch periodicity
-----------------
A `gfem.Periodic` pair relates two surfaces by the translation :math:`a`,
from its first surface to its second (`Model.periodic_translations`, m).
With a Bloch wave vector :math:`k` (rad/m, ``wave_vector=`` on `gfem.Driven`
and `gfem.Eigenmode`), the fields obey `BLOCH_PHASE`:

.. math::

    E(r + a) = e^{-i k \cdot a} E(r)

so, under :math:`e^{+i\omega t}`, the phase of a Bloch wave travels along
:math:`+k`. Without a wave vector the fields are equal on both surfaces
(plain periodicity). Swapping a pair's surfaces flips :math:`a` and gives
the same field.

Under a Bloch wave vector, :math:`\int D \cdot E \, dV` over the cell can
vanish, so an eigenmode field is phased to make its largest coefficient real
and positive instead (still scaled to `MODE_ENERGY`).

Floquet ports
-------------
A `gfem.FloquetPort` is one diffraction order in one polarization of a plane
wave on an open face of a periodic cell (a Floquet surface).

- Lattice frame: :math:`a_1` and :math:`a_2` are the translations of the
  face's two pairs, in ``boundaries`` order;
  :math:`\hat e_1 = a_1 / |a_1|`,
  :math:`\hat e_3 = a_1 \times a_2 / |a_1 \times a_2|`,
  :math:`\hat e_2 = \hat e_3 \times \hat e_1`. Order :math:`(m, n)` has the
  in-plane wave vector :math:`\kappa = k_t + m b_1 + n b_2` with
  :math:`b_i \cdot a_j = 2\pi \delta_{ij}` and :math:`k_t` the problem's
  ``wave_vector`` (the specular order's). Its field on the face is
  :math:`c\, \hat e\, e^{-i \kappa \cdot r}`, with :math:`r` the model
  coordinates: the in-plane phase is referred to the model origin, and
  translating the model in-plane by :math:`r_0` multiplies :math:`S_{pq}`
  by :math:`e^{-i (\kappa_p - \kappa_q) \cdot r_0}`.
- References: TE :math:`\hat e = \hat e_3 \times \hat\kappa` (s), TM
  :math:`\hat e = \hat\kappa` (p), the same on both faces of a cell, with
  :math:`\hat\kappa = \hat e_1` at :math:`\kappa = 0` (at normal incidence
  TE is E along :math:`\hat e_2`, TM along :math:`\hat e_1`).
- Normalization: :math:`S_{pq} = (c_p / c_q) \sqrt{Z_q / Z_p}`, :math:`c`
  the tangential-E coefficients at the face plane and :math:`Z` the order's
  wave impedance in the homogeneous medium behind its own face,
  :math:`\eta / \cos\theta` (TE) or :math:`\eta \cos\theta` (TM),
  :math:`\eta = \eta_0 \sqrt{\mu_r / \varepsilon_r}` and :math:`\theta` the
  order's angle from the face normal. The incident wave carries
  `INCIDENT_POWER` per cell, :math:`|S|^2` is the diffraction efficiency,
  and for a lossless cell the efficiencies of every propagating order on
  both faces, both polarizations, add up to 1. ``z_ref`` is :math:`Z`.
- Signs: tangential-E coefficients, so a flat interface reflects
  :math:`(Z_2 - Z_1) / (Z_2 + Z_1)`, :math:`(n_1 - n_2) / (n_1 + n_2)` in
  both polarizations at normal incidence from medium 1 (the Fresnel
  :math:`r_p` of many optics texts has the opposite sign), and an empty cell
  of thickness :math:`L` transmits :math:`e^{-i\beta L}`.
- Orders that do not propagate: where an order is evanescent or grazing
  (:math:`\gamma^2 \ge 0`), its row of S is exactly 0 and its ``z_ref`` is
  NaN, and it cannot be excited. ``FloquetPort(offset=d)`` multiplies the
  row and column by :math:`e^{\gamma d}` where the order propagates, 1
  elsewhere.
- Bloch reciprocity:
  :math:`S_{pq}(k) = \sigma_p \sigma_q S_{-q,-p}(-k)`, where
  :math:`-p` is port p's face and polarization with order
  :math:`(-m, -n)` and :math:`\sigma = -1` for an order with
  :math:`\kappa \ne 0`, :math:`+1` for :math:`\kappa = 0` (the references
  are odd in :math:`\hat\kappa`). The plain transpose does not hold as it
  stands.
- Circular polarization: with the TE and TM ports of one order on a face whose
  outward normal is :math:`+\hat e_3` (incidence along :math:`-\hat e_3`),
  :math:`(S_{:,\mathrm{TM}} + i S_{:,\mathrm{TE}}) / \sqrt{2}` is the response
  to IEEE right-handed circular incidence; on a face whose outward normal is
  :math:`-\hat e_3` the same combination is left-handed.

2D models
---------
A model whose mesh has dimension 2 (``Mesh.dim == 2``, read from a
triangle-only gmsh file) lies in the :math:`(x, y)` plane: its nodes are
:math:`(x, y)`, :math:`z` is normal to the plane, the fields do not vary
along :math:`z` (:math:`\partial_z = 0`), and waves travel in the plane.
A driven solve computes one polarization, the field component normal to the
plane, chosen by ``Driven(polarization=...)``:

.. math::

    \text{Ez:}\quad & \nabla \cdot (\mu_r^{-1} \nabla E_z)
        + k_0^2 \varepsilon_r E_z = 0 \\
    \text{Hz:}\quad & \nabla \cdot (\varepsilon_r^{-1} \nabla H_z)
        + k_0^2 \mu_r H_z = 0

with :math:`k_0 = \omega / c` and :math:`\varepsilon_r` including a
conductivity as :math:`-i\sigma / (\omega \varepsilon_0)` under
:math:`e^{+i\omega t}`. Ez has :math:`E` normal to the plane and
:math:`H` in it; Hz the reverse. A `gfem.PEC` line sets
:math:`E_z = 0` in Ez and leaves :math:`H_z` free (zero normal derivative)
in Hz; a `gfem.PMC` line does the reverse.

Every extensive quantity of a 2D model is per metre along :math:`z`: each
excited port is fed 1 W/m time-averaged incident power, and a kept field
(``"Ez"`` in V/m, ``"Hz"`` in A/m) is the peak phasor for that 1 W/m. S is
dimensionless. Against a slab of thickness :math:`h` between plates
(``Mesh.from_gmsh_cross_section``) fed 1 W, which is :math:`1/h` W/m, S is
the same and the fields relate by

.. math::

    E^{\mathrm{2D}} = \sqrt{h}\, E^{\mathrm{slab}}.

Quantum metrics
---------------
`gfem.quantum.dressed_modes` puts one junction's cosine on the solved
linear modes:

.. math::

    H / \hbar &= \sum_m \omega_m a_m^\dagger a_m
        - (E_J / \hbar) \left[\cos\varphi - 1 + \varphi^2 / 2\right] \\
    \varphi &= \sum_m \varphi_m (a_m + a_m^\dagger) \\
    \varphi_m &= \mathrm{sign}(p_m) \sqrt{|p_m| \hbar \omega_m / (2 E_J)} \\
    E_J &= (\hbar / 2e)^2 / L_J

with :math:`\omega_m` the real part of the mode's angular frequency,
:math:`p_m` its signed junction participation and :math:`L_J` the junction's
linear inductance. With :math:`E` its levels (:math:`E(0)` the ground
state, :math:`E(1_m)` one photon in mode :math:`m`), every output is a
signed angular frequency in rad/s and a difference of transition
frequencies, so its sign is an observable:

- dressed frequency :math:`E(1_m) - E(0)`; the Lamb shift is that minus
  :math:`\omega_m`;
- anharmonicity :math:`\alpha = E(2_m) - 2 E(1_m) + E(0)`, the second
  transition frequency minus the first;
- dispersive shift :math:`\chi = E(1_m 1_n) - E(1_m) - E(1_n) + E(0)`: the
  full shift of mode :math:`n`'s frequency per photon in mode :math:`m`, not
  half of it.

For an ordinary transmon all three are negative. Other sources' conventions
convert as
:math:`\chi = -\chi_\mathrm{Minev} = \chi_\mathrm{Nigg} = 2 \chi_\mathrm{Koch}`
and :math:`\alpha = -\alpha_\mathrm{Minev} = \alpha_\mathrm{Koch}` (Minev et
al. and pyEPR report a downward shift as positive; Koch et al. write half
the shift).

Temperature and thermo-optic phase
----------------------------------
Temperatures are absolute, in kelvin. A thermal result's rise is the
temperature minus the source-free temperature (the temperature with every
`gfem.HeatSource` off, which boundary temperatures and fluxes still set),
so it is the heaters' response even where the boundaries differ.

A material's ``thermo_optic_coefficient`` :math:`dn/dT` (1/K, any sign) acts
on the rise about the index :math:`n_0 = \sqrt{\varepsilon_r}` of its
``eps_r``, the index at the source-free temperature; no reference
temperature is stored. `gfem.Model.with_temperature_rise` gives each element
:math:`\varepsilon_r = (n_0 + (dn/dT)\, \Delta T_e)^2`, :math:`\Delta T_e`
(``rise_e``) the element's mean rise.

`gfem.thermo_optic_shift` is first order in the rise, about the port's
mode :math:`E_0` at 1 W on a lossless section:

.. math::

    \delta n_\mathrm{eff} &= \frac{c \varepsilon_0}{4} \sum_e 2 n_{0,e}
        (dn/dT)_e\, \Delta T_e \int_e |E_0|^2 \, dA \\
    \text{phase per length} &= k_0\, \mathrm{Re}(\delta n_\mathrm{eff})
        \quad \text{(rad/m)}

``phase_per_length`` is the extra phase delay per unit length of guide:
positive for a positive coefficient and a positive rise. Under
:math:`e^{+i\omega t}` a guide of length :math:`L` transmits
:math:`e^{-i\beta L}`, so a transmission's argument moves by minus
``phase_per_length * L``. The power that gives a :math:`\pi` shift is
:math:`P_\pi = \pi P' /` ``phase_per_length`` for a heater dissipating
:math:`P'` per unit length (``power_per_length``), which equals
:math:`\lambda / (2 s)` with :math:`s = \delta n_\mathrm{eff} / P'`.

`gfem.heat_flow` is the heat leaving a side of a steady temperature, in W,
positive out (`gfem.heat_flow_per_length`, W/m, on a cross-section), read
from the absolute temperature. `gfem.Trace.heat_out` is the heat that has
left a side since :math:`t = 0`, in J (J/m on a cross-section).

`gfem.route_phase` integrates the same first-order phase per length along
a route through a 3D rise (rad, a phase delay, positive for a positive
coefficient and rise; :math:`P_\pi = \pi P /` ``phase`` for a heater
dissipating :math:`P`). Its window is in route-local coordinates: the
section's x points to the left of travel (``up x tangent``), its y along
``up``, and the station sits at the section's origin.

Electro-optic shift
-------------------
A material's ``electro_optic_tensor`` is the contracted tensor
:math:`r_{Ik}` in m/V, shape (6, 3): rows :math:`I` the Voigt pairs
:math:`xx, yy, zz, yz, xz, xy`, columns :math:`k` the field components
:math:`x, y, z`, all in the frame whose axes are the rows of the material's
``axes`` (right-handed), rotated into the model's frame with the
permittivity. A static field :math:`E_\mathrm{s} = -\nabla V` changes the
optical permittivity by

.. math::

    \Delta\varepsilon_r = -\varepsilon_r\, (r \cdot E_\mathrm{s})\,
        \varepsilon_r, \qquad (r \cdot E_\mathrm{s})_{ij} = \sum_k r_{ijk}
        E_{\mathrm{s},k},

which is :math:`\Delta(1/n^2)_{ij} = \sum_k r_{ijk} E_{\mathrm{s},k}` to
first order. The static permittivity is the ``eps_r`` of a separate
electrostatic model; give both models' crystal the same ``axes``.

`gfem.electro_optic_shift` is first order in the field, about the port's
mode :math:`E_0` at 1 W on a lossless section, and per volt: the potential
is one terminal's at 1 V, every other terminal and the ground at 0 V. A
drive :math:`V` multiplies every per-volt value by :math:`V`, and several
terminals' shifts add, each times its drive.
``phase_per_length_per_volt`` is :math:`k_0\, \partial n_\mathrm{eff} /
\partial V` in rad/(V m), a phase delay like the thermo-optic phase: a
transmission through a length :math:`L` at :math:`V` turns its argument by
:math:`-\phi V L`, with :math:`\phi` the ``phase_per_length_per_volt``.

``v_pi_l`` is one arm's voltage-length product,
:math:`V_\pi L = \lambda / (2\, |\partial n_\mathrm{eff} / \partial V|)` in
V m. A push-pull pair of arms :math:`a` and :math:`b` driven by one
potential has :math:`V_\pi L = \lambda / (2\, |s_a - s_b|)`, with :math:`s`
each arm's ``delta_n_eff_per_volt``. Solve the mode on an optical window
that reaches the electrodes' inner edges: the shift weighs the mode's tail
there.

Piezoelectric problems
----------------------
A `gfem.Piezoelectric` problem solves linear piezoelectricity on a 2D
section: the :math:`(x, y)` plane, :math:`\partial_z = 0`, and three
displacement components :math:`u_x, u_y, u_z`. With :math:`e^{+i\omega t}`,

.. math::

    \nabla\cdot T = -\omega^2 \rho\, u, \qquad \nabla\cdot D = 0, \qquad
    T = c^E S - e^{\mathsf T} E, \qquad D = e S + \varepsilon^S E,
    \qquad E = -\nabla\varphi.

The stiffness ``stiffness`` is :math:`c^E` (Pa, 6 x 6, at constant field), the
``piezoelectric_tensor`` is :math:`e` (C/m^2, 3 x 6, the stress constants, not
the strain constants :math:`d`) and ``eps_r`` is :math:`\varepsilon^S /
\varepsilon_0` (at constant strain), all in the frame whose axes are the rows
of the material's ``axes`` (right-handed), rotated into the model's frame.

- Voigt order is :math:`xx, yy, zz, yz, xz, xy` with engineering shear (twice
  the tensor components): :math:`S = (\partial_x u_x,\ \partial_y u_y,\ 0,\
  \partial_y u_z,\ \partial_x u_z,\ \partial_y u_x + \partial_x u_y)`.
  `gfem.strain` returns this :math:`S` per volt.
- Loss is a complex stiffness under :math:`e^{+i\omega t}`,
  :math:`c^E (1 + i(1/Q_m + \eta r^2))`, with :math:`Q_m` the material's
  ``mechanical_quality_factor``, and :math:`\eta r^2` a `gfem.DampingFrame`'s
  (:math:`r` the normalised depth into the frame, :math:`\eta` its
  ``loss_factor``). Its imaginary part is positive: the power dissipated per
  volume, :math:`\tfrac{\omega}{2}\, S^* \mathrm{Im}(c)\, S`, is non-negative
  when :math:`\mathrm{Im}\,c \ge 0`. A permittivity's loss has the opposite
  sign, :math:`\varepsilon_r (1 - i\tan\delta)`, because there the dissipation
  is :math:`\tfrac{\omega}{2}\,\mathrm{Im}(-\varepsilon)|E|^2` (see "Time
  dependence").
- A damping frame of thickness :math:`L` in a medium of wavelength
  :math:`\lambda` reflects a plane wave at normal incidence by
  :math:`|R| \approx e^{-2\pi\eta (L/\lambda)/3}`, down to a floor of about
  :math:`3.4\times10^{-3}\,\eta\,(\lambda/L)^2`: four wavelengths at the
  default :math:`\eta = 1.2` give :math:`|R| < 10^{-3}`.

Every extensive result is per metre along :math:`z`. For each terminal :math:`j`
in turn the potential is 1 V on it and 0 V on every other terminal and the
ground; ``admittance_per_length`` is :math:`Y'_{ij}(f) = i\omega q'_i` in S/m,
the current into terminal :math:`i` per volt on terminal :math:`j` per metre
(:math:`q'` the charge per length), indexed ``[frequency, i, j]``: the response
first, the excitation second. ``clamped_capacitance_per_length`` is
:math:`C_0'` in F/m, the Maxwell matrix of the potential alone with
:math:`\varepsilon^S` and no displacement; it is not the low-frequency limit
of :math:`Y'/i\omega`, the free capacitance, which is larger by :math:`1 + K^2`.

`PiezoelectricResult.network_parameters` scales to an aperture :math:`W` and
puts the series resistances :math:`R = \mathrm{diag}(R_s)` in the leads:

.. math::

    Y = W Y', \qquad Y_t = Y (I + R Y)^{-1},

and S is that of :math:`Y_t` on the reference impedances, as pseudo-waves
(see "Network parameters"; 50 Ω by default). For one terminal on a real :math:`Z_0`,
:math:`S_{11} = (Z_L - Z_0)/(Z_L + Z_0)` with :math:`Z_L = 1/Y + R_s`.

The displacement and potential fields and the strain are per volt across the
terminal, after :math:`R_s`. A device fed by an incident wave :math:`V_\mathrm{inc}`
on :math:`Z_0` has, across its one terminal, :math:`V_\mathrm{IDT} =
V_\mathrm{inc}\, 2 Z_\mathrm{IDT}/(Z_L + Z_0)`, with :math:`Z_\mathrm{IDT} =
1/Y` and :math:`Z_L = Z_\mathrm{IDT} + R_s`. A quantity per terminal volt is
referred to the incident wave by that factor.

Mechanical eigenmodes
---------------------
A `gfem.MechanicalEigenmode` problem finds the source-free solutions of the
elasticity of "Piezoelectric problems" on a 2D section (with the potential
coupled when a material carries a piezoelectric tensor), for fields
:math:`u(x, y)\,e^{i\omega t - iqz}`: :math:`q` is the axial wavenumber in rad/m
and :math:`\partial_z \to -iq` in the strain, so a mode with :math:`q > 0`
travels along :math:`+z` where its group velocity is positive. The stiffness and
the loss are the piezoelectric problem's, under :math:`e^{+i\omega t}`: a
decaying mode has :math:`\mathrm{Im}\,\omega > 0` and its quality factor is

.. math::

    Q = \frac{\mathrm{Re}\,\omega}{2\,\mathrm{Im}\,\omega}.

- A mode is normalized to :math:`\tfrac12 (\mathrm{Re}\,\omega)^2 \int \rho\,|u|^2\,
  \mathrm{d}A = 1` J per metre along :math:`z`, with its largest coefficient
  real and positive: for a lossless mode the time-averaged energy.
- ``group_velocity`` is :math:`\mathrm{Re}\,\mathrm{d}\omega/\mathrm{d}q` in m/s,
  the exact derivative of the discrete band; a section with a mirror symmetry
  along :math:`z` has :math:`\omega(-q) = \omega(q)` and the opposite group
  velocity at :math:`-q`.
- ``clamped`` fixes all three components on a line given by its name, or the
  components of a string of ``"xyz"`` (the lab axes): ``"x"`` on a line of
  constant :math:`x` keeps :math:`u_x = 0` and slides in :math:`y` and :math:`z`.
  Every other line is traction-free; the rigid-body motions of a free section
  at :math:`q = 0` have :math:`\omega = 0` and are not returned.

Group index and dispersion
--------------------------
A port mode's group index and dispersion are the frequency derivatives of
:math:`\gamma / i` at each frequency of a `gfem.PortMode` grid:

.. math::

    n_g = \frac{c}{i}\frac{\mathrm{d}\gamma}{\mathrm{d}\omega}, \qquad
    \beta_k = \frac{1}{i}\frac{\mathrm{d}^k\gamma}{\mathrm{d}\omega^k}
    \quad (k = 2, 3), \qquad
    D = -\frac{2\pi c}{\lambda^2}\,\beta_2, \quad \lambda = c / f.

For a lossless mode they are real: :math:`n_g = c\,\mathrm{d}\beta/\mathrm{d}\omega`
and :math:`\beta_k = \mathrm{d}^k\beta/\mathrm{d}\omega^k`, the same under
either time dependence. With loss they are complex, with real parts
:math:`c\beta'` and :math:`\beta^{(k)}` and imaginary parts :math:`-c\alpha'`
and :math:`-\alpha^{(k)}`; a code using :math:`e^{-i\omega t}` reports the
complex conjugates. :math:`\beta_2 > 0` (:math:`D < 0`) is normal
dispersion. Units: :math:`\beta_2` in s²/m, :math:`\beta_3` in s³/m,
:math:`D` in s/m². A perfectly matched layer keeps its reference frequency's
stretch over the whole grid, and the derivatives are those of that section.

Kerr nonlinearity
-----------------
A material's ``nonlinear_index`` :math:`n_2` (m²/W) and
``two_photon_absorption`` :math:`\beta_\mathrm{TPA}` (m/W) give a complex
:math:`\tilde n_2 = n_2 - i\beta_\mathrm{TPA}/(2k_0)`, :math:`k_0 = \omega/c`.
For a port mode :math:`E` of 1 W, all three components, and
:math:`n^2 = \mathrm{Re}\,\varepsilon_r` of each element at the row's
frequency:

.. math::

    \gamma_\mathrm{NL} = \frac{k_0 c^2 \varepsilon_0^2}{12} \int_S n^2\,
    \tilde n_2 \left[2|E|^4 + |E\cdot E|^2\right] \mathrm{d}A,

:math:`E\cdot E` without conjugation. It is reported in the convention of
pulse-propagation tools, :math:`e^{-i\omega t}`: a power :math:`P` moves the
mode's phase constant by :math:`\delta\beta = \mathrm{Re}\,
\gamma_\mathrm{NL}\,P` and its field attenuation by :math:`\delta\alpha =
\mathrm{Im}\,\gamma_\mathrm{NL}\,P \ge 0`, the two-photon loss, and the
envelope obeys :math:`\mathrm{d}A/\mathrm{d}z = i\gamma_\mathrm{NL}|A|^2A` as
in the generalized nonlinear Schrödinger equation. It is the complex
conjugate of the integral in gfem's own :math:`e^{+i\omega t}` phasors.
Between waves :math:`a, b, c, d` the tensor
:math:`\gamma_{abcd}` is the coefficient of :math:`A_b A_c \overline{A_d}` in
the equation of :math:`A_a`, summed over all tuples with no degeneracy
factor; :math:`\gamma_{aaaa} = \gamma_\mathrm{NL}`. Units: 1/(W m), and 1/W
on an Ez line port of a 2D model (1 W/m, integral along the line). The
isotropic scalar :math:`n_2` is real and constant over the grid, and the
formula holds while :math:`\alpha \ll \beta`.

Second-order nonlinearity
-------------------------
A material's ``second_order_nonlinear_tensor`` is the contracted
:math:`d_{il} = d_{ijk}` in m/V, :math:`d = \chi^{(2)}/2`: rows the polarization
:math:`x, y, z`, columns the field pair :math:`xx, yy, zz, yz, xz, xy` with no
factor 2 on the shear ones, in the frame of the material's ``axes``. A domain
of reversed polarization carries :math:`-d`. For port modes :math:`a, b, c` of
1 W with :math:`\omega_a = \omega_b + \omega_c`, :math:`\kappa_{abc} =
\overline{T_{abc}}` with

.. math::

    T_{abc} = \frac{\omega_a \varepsilon_0}{4} \int_S \sum_{ijk} d_{ijk}\,
    \overline{E_{a,i}}\, E_{b,j}\, E_{c,k}\, \mathrm{d}A,

all three components of each mode. It is reported in the convention of
coupled-mode tools, :math:`e^{-i\omega t}`, the complex conjugate of the
integral in gfem's own phasors: :math:`\mathrm{d}A_a/\mathrm{d}z \ni i
\sum_{bc} \kappa_{abc} A_b A_c\, e^{-i\Delta\beta_{abc} z}`, with
:math:`|A|^2` a power, :math:`\Delta\beta_{abc} = \beta_a - \beta_b -
\beta_c`, summed over all ordered pairs with no degeneracy factor.
:math:`\kappa_{abc}` is symmetric in :math:`b, c`; units 1/(:math:`\sqrt{\mathrm{W}}`
m), and W\ :sup:`-1/2` m\ :sup:`-1/2` on an Ez line port. For the second
harmonic, :math:`\kappa = \kappa_{hpp}`, :math:`\eta_\mathrm{norm} =
|\kappa|^2` in 1/(W m²) (:math:`10^{-2}` %/(W cm²) per unit), the undepleted
harmonic power is :math:`\eta_\mathrm{norm} P_\omega^2 L^2
\mathrm{sinc}^2(\Delta\beta L/2)` with :math:`\Delta\beta = \beta_{2\omega} -
2\beta_\omega`, and quasi-phase matching of period :math:`\Lambda`, duty cycle
:math:`D` and order :math:`m` multiplies :math:`\kappa` by :math:`G_m =
2\sin(\pi m D)/(\pi m)` and replaces :math:`\Delta\beta` by
:math:`|\Delta\beta| - 2\pi m/\Lambda`. With the loss of the modes,

.. math::

    \frac{P_{2\omega}}{P_\omega} = |G|^2 |\kappa|^2 P_\omega\,
    e^{-(\alpha_{2\omega} + 2\alpha_\omega) L}\,
    \frac{\sinh^2(\Delta\alpha L/2) + \sin^2(\delta L/2)}
    {(\Delta\alpha/2)^2 + (\delta/2)^2},

:math:`\alpha = \mathrm{Re}\,\gamma`, :math:`\Delta\alpha = \alpha_{2\omega}
- 2\alpha_\omega`, :math:`\delta` the residual mismatch. A pump :math:`a` with
a signal :math:`b` and an idler :math:`c` (:math:`\omega_a = \omega_b +
\omega_c`) obeys :math:`\mathrm{d}A_b/\mathrm{d}z \ni i (2\omega_b/\omega_a)
\overline{\kappa_{abc}} A_a \overline{A_c}\, e^{i\Delta\beta z}`, which
conserves the photon flux when :math:`d` is symmetric in its three indices
(Kleinman symmetry). The formulas hold while :math:`\alpha \ll \beta` and for
an undepleted pump.

Incident fields
---------------
A `gfem.PlaneWave` or `gfem.GaussianBeam` travels along

.. math::

    \hat k = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta),

the propagation direction in spherical angles from :math:`+z`, in rad:
:math:`\theta < \pi/2` travels up, :math:`\theta > \pi/2` down (30° off
the normal from above is :math:`\theta = \pi - 30°`).

- Polarization basis: TM (p) is :math:`\hat\theta = \partial_\theta \hat k
  = (\cos\theta\cos\phi, \cos\theta\sin\phi, -\sin\theta)`, TE (s) is
  :math:`\hat\phi = \hat z \times \hat\kappa = (-\sin\phi, \cos\phi, 0)`
  with :math:`\hat\kappa = (\cos\phi, \sin\phi, 0)`; :math:`(\hat\theta,
  \hat\phi, \hat k)` is right-handed. Along :math:`\pm z`, :math:`\phi`
  still fixes the plane of incidence: TM is :math:`(\cos\phi, \sin\phi, 0)`
  at :math:`\theta = 0` and :math:`-(\cos\phi, \sin\phi, 0)` at
  :math:`\theta = \pi`.
- Floquet references: TE equals a `gfem.FloquetPort`'s TE reference in
  every direction. TM's tangential part is :math:`\cos\theta\,\hat\kappa`,
  along the Floquet TM reference for a wave going up and opposite to it for
  a wave going down.
- A polarization is a Jones pair :math:`(a_\mathrm{TM}, a_\mathrm{TE})`,
  normalized: the field direction is :math:`(a_\mathrm{TM}\hat\theta +
  a_\mathrm{TE}\hat\phi) / \lVert a\rVert`. ``"TM"`` is :math:`(1, 0)`,
  ``"TE"`` :math:`(0, 1)`, ``"RCP"`` :math:`(1, -i)/\sqrt2` and ``"LCP"``
  :math:`(1, i)/\sqrt2`, IEEE handedness under :math:`e^{+i\omega t}`: with
  the thumb along :math:`\hat k`, the real field of RCP turns from
  :math:`\hat\theta` to :math:`\hat\phi` (right hand), as for the Floquet
  ports' circular combination.
- Plane wave amplitude: :math:`E_\mathrm{inc}(r) = E_0\, \hat e\,
  e^{-i k \hat k \cdot r}` in a homogeneous background of wavenumber
  :math:`k`, with :math:`E_0` (``amplitude``) the peak field in V/m in the
  layer the wave comes from, complex for a phase, and :math:`r` the model
  coordinates: the phase is referred to the model origin, as the Floquet
  in-plane phase is.
- Layered background (`gfem.Background` of several layers, or over a
  ground plane): the incident field is the stack's whole solution for that
  plane wave, its reflected and transmitted waves included. The wave comes
  from the incidence layer (the bottom for :math:`\theta < \pi/2`, the top
  for :math:`\theta > \pi/2`), which must be lossless; :math:`E_0` and its
  phase at the model origin are those of the incident wave in that layer,
  extended over all space. Its lateral wave vector is real, so total
  internal reflection gives an evanescent field across the other layers. A
  layer's :math:`\varepsilon_r(1 - i\tan\delta) - i\sigma/(\omega
  \varepsilon_0)` is evaluated at each frequency.
- Gaussian beam: 1 W (`INCIDENT_POWER`) time-averaged power through any
  plane normal to its axis, and the field at the focus real and positive
  along the polarization (zero phase at the focus); the waist is a length,
  the same at every frequency.
- Total field = incident field + scattered field. A kept field of a source
  column is the total field; the scattered field is outgoing at the
  truncation. In a perfectly matched layer the kept field of an incident
  column is the scattered field: the incident field is zero there. Its
  incident part (``DrivenResult.e_field(..., part="incident")``) is the
  incident field interpolated on the kept field's space outside the layer,
  and its scattered part the kept field minus it, so the three add exactly.
- Truncation and walls: where the incident field is dropped or must hold a
  wall, it must stay at most :math:`10^{-3}` of its peak over the physical
  region: where a scatterer (an element whose region's material differs
  from the background stack at its height, a material field always, or a
  sheet, conductor or lumped face) meets the layer or an absorbing face, on
  a wave port face, and as :math:`|n\times E|` on an exterior PEC wall or
  :math:`\eta|n\times H|` on an exterior PMC wall (named, or by default).
- Port waves: in a source column every port is terminated in its reference
  impedance, and ``DrivenResult.port_waves`` is the outgoing pseudo-wave
  :math:`b` at each port, in √W as a column of S: on a real ``z_ref``,
  :math:`|b|^2` is the power in W the source sends into the port.

Point dipoles
-------------
A `gfem.ElectricDipole` of moment :math:`p` (C·m) at :math:`r_0` is the
current :math:`J = i\omega p\, \delta(r - r_0)` under :math:`e^{+i\omega t}`;
a `gfem.MagneticDipole` of moment :math:`m` (A·m²) is the current
:math:`J = \nabla \times (m\, \delta(r - r_0))`, i.e. the magnetic current
:math:`M = i\omega\mu_0 m\, \delta(r - r_0)`. Both moments are peak
phasors, the same at every frequency. An antenna current moment :math:`Il`
(A·m) is :math:`p = Il / (i\omega)`.

- Emitted power: the time-averaged power the dipole gives the field,

  .. math::

      P = -\frac{\omega}{2} \operatorname{Im}\left(p^* \cdot E(r_0)\right),

  positive for a dipole that radiates or drives a lossy medium; gfem pairs
  the dipole's load with the discrete solution, which is finite although the
  exact field at :math:`r_0` is not. Its real counterpart, the
  self-reaction, has no limit and is never reported.
- Bulk power: the emitted power of the same dipole in the unbounded,
  lossless material at :math:`r_0`, with :math:`n = \sqrt{\varepsilon_r
  \mu_r}` and :math:`c` the speed of light in vacuum,

  .. math::

      P_0^{(p)} = \frac{\mu_0 \mu_r n\, \omega^4 |p|^2}{12 \pi c}, \qquad
      P_0^{(m)} = \frac{\mu_0 \varepsilon_r n\, \omega^4 |m|^2}{12 \pi c^3}.

- Purcell factor: :math:`F = P / P_0`. In a non-magnetic medium the partial
  local density of states along :math:`p` is :math:`\rho_p = \rho_0 F`, with
  :math:`\rho_0 = \omega^2 n / (\pi^2 c^3)`.

Far fields
----------
`gfem.far_field` reads the field radiated into the exterior medium, the
model's background (one constant, isotropic, lossless `gfem.Material`), of
wavenumber :math:`k = n\omega/c` and wave impedance :math:`\eta =
\eta_0\sqrt{\mu_r/\varepsilon_r}`.

- Far-field amplitude: under :math:`e^{+i\omega t}` an outgoing wave is
  :math:`e^{-ikr}/r`, and

  .. math::

      F(\hat r) = \lim_{r\to\infty} r\,e^{+ikr} E(r\hat r),

  in V, with :math:`r` measured from the model origin, the phase reference.
  :math:`\hat r` is given by the same angles as an incident field's
  :math:`\hat k`, and :math:`F = F_\theta\hat\theta + F_\phi\hat\phi`
  in that basis. For an incident-field column it is the scattered field's.
- Circular components: :math:`F_R = (F_\theta + iF_\phi)/\sqrt2` and
  :math:`F_L = (F_\theta - iF_\phi)/\sqrt2`, the field's components along
  ``"RCP"`` and ``"LCP"`` (IEEE handedness); axial ratio
  :math:`(|F_R| + |F_L|)/\big||F_R| - |F_L|\big|`, 1 for circular and
  infinite for linear polarization.
- Radiation intensity :math:`U = |F|^2/(2\eta)` in W/sr (peak phasors);
  radiated power :math:`P_\mathrm{rad}`, the power flowing out through the
  far-field layer. The IEEE 145 ratios: directivity :math:`D = 4\pi
  U/P_\mathrm{rad}`, gain :math:`G = 4\pi U/P_\mathrm{acc}`, realized gain
  :math:`G_r = 4\pi U/P_\mathrm{inc}`, radiation efficiency
  :math:`P_\mathrm{rad}/P_\mathrm{acc}`. For a port column
  :math:`P_\mathrm{inc}` is `INCIDENT_POWER` and :math:`P_\mathrm{acc} = 1 -
  |S_{ii}|^2`, so :math:`G_r = 4\pi U`; for a point dipole both are its
  emitted power.
- Bistatic radar cross section of a `gfem.PlaneWave` column, in m²:
  :math:`\sigma(\hat r) = 4\pi|F(\hat r)|^2/|E_0|^2`.

Multipoles
----------
`gfem.Multipoles` expands the scattered field of an incident-field column
about an origin :math:`r_o` (default the model origin), outside a sphere
that holds every source, in the background's outgoing vector spherical
waves, of wavenumber :math:`k`, impedance :math:`\eta`, permittivity
:math:`\varepsilon` and :math:`v = \omega/k` (all real).

- Basis: Condon-Shortley :math:`Y_{lm}` (``scipy.special.sph_harm_y``),
  Jackson's :math:`X_{lm} = L Y_{lm}/\sqrt{l(l+1)}` with :math:`L = -i\,r\times\nabla`,
  :math:`M^{(2)}_{lm} = h^{(2)}_l(kr)X_{lm}` and :math:`N^{(2)}_{lm} =
  k^{-1}\nabla\times M^{(2)}_{lm}`; :math:`h^{(2)}_l \sim i^{l+1}e^{-ix}/x`
  is outgoing under :math:`e^{+i\omega t}`, like the far field's
  :math:`e^{-ikr}/r`.
- Expansion and normalization:

  .. math::

      E_s = E_\mathrm{inc}\sum_{l\ge1}\sum_{m=-l}^{l}
      \big[\alpha_{lm}N^{(2)}_{lm} + \beta_{lm}M^{(2)}_{lm}\big],
      \qquad
      \sigma_\mathrm{sca} = \frac1{k^2}\sum_{lm}
      \big(|\alpha_{lm}|^2 + |\beta_{lm}|^2\big),

  with no :math:`i^l` or :math:`\sqrt{\pi(2l+1)}` in the expansion, so
  :math:`\alpha_{lm}` and :math:`\beta_{lm}` are dimensionless and the
  partial scattering cross section of the electric (magnetic) multipole of
  degree :math:`l` is :math:`k^{-2}\sum_m|\alpha_{lm}|^2`
  (:math:`k^{-2}\sum_m|\beta_{lm}|^2`), in m², over the incident intensity
  :math:`I = |E_\mathrm{inc}|^2/(2\eta)` of `gfem.cross_sections`.
- Incident amplitude :math:`E_\mathrm{inc}`, V/m, complex: a
  `gfem.PlaneWave`'s ``amplitude`` (the phase at the model origin, not at
  the multipole origin); a `gfem.GaussianBeam`'s focal field
  :math:`\sqrt{2\eta I_0}` (real, positive); :math:`\sqrt{2\eta I}` for a
  `gfem.IncidentField` with ``intensity=``. This is not
  ``gfem.FarField.incident_amplitude``, the real :math:`|E_0|` of a plane
  wave (``None`` for a beam): the two share a name, not a meaning.
- Storage: :math:`\alpha, \beta` are ``(n_freq, lmax * (lmax + 2))``,
  :math:`l`-major with :math:`m` ascending, flat index
  :math:`j = l^2 + l + m - 1`; the degree-1 entries are the first three
  whatever ``lmax``. The default ``lmax`` is Wiscombe's
  :math:`\max(2, \lceil x + 4x^{1/3} + 2\rceil)`, :math:`x = kR_s` at the
  highest kept frequency and :math:`R_s` the scatterer's circumradius about
  the origin; ``lmax`` is at least 2 so every result carries the
  quadrupoles.
- Mie map. For a sphere at the origin under
  ``PlaneWave(theta=0, phi=0, polarization="TM")``
  (:math:`E_0\hat x e^{-ikz}`), with :math:`\bar a_l, \bar b_l` the
  conjugates of Bohren and Huffman's coefficients (gfem's
  :math:`\varepsilon` is the conjugate of theirs),

  .. math::

      \alpha_{l,\pm1} = \pm(-i)^l\sqrt{\pi(2l+1)}\,\bar a_l,\qquad
      \beta_{l,\pm1} = -(-i)^l\sqrt{\pi(2l+1)}\,\bar b_l,

  and every other :math:`m` is zero, so :math:`\sigma^E_l = 2\pi(2l+1)
  |a_l|^2/k^2`. A sphere centred at :math:`r_o` has these times
  :math:`e^{-ik\hat k\cdot r_o}`.
- Exact moments, fixed linear maps of the :math:`l\le2` coefficients, in the
  model's :math:`x, y, z`, with :math:`\hat e_{\pm1} = \mp(\hat x \pm
  i\hat y)/\sqrt2`, :math:`\hat e_0 = \hat z` and :math:`T_{2M}` the
  orthonormal symmetric traceless rank-2 tensors built from them:

  - :math:`p = -\sqrt{6\pi}\,\varepsilon E_\mathrm{inc}k^{-3}
    \sum_m\alpha_{1m}\hat e_m`, C·m;
  - :math:`m = -i\sqrt{6\pi}\,E_\mathrm{inc}(k^3\eta)^{-1}
    \sum_m\beta_{1m}\hat e_m`, A·m²;
  - :math:`Q^e = -12\sqrt{5\pi}\,\varepsilon E_\mathrm{inc}k^{-4}
    \sum_M\alpha_{2M}T_{2M}`, C·m²;
  - :math:`Q^m = -12\sqrt{5\pi}\,i\,E_\mathrm{inc}(k^4\eta)^{-1}
    \sum_M\beta_{2M}T_{2M}`, A·m³.

  A point `gfem.ElectricDipole` or `gfem.MagneticDipole` of moment
  :math:`p` or :math:`m` at the origin gives back exactly that moment and
  only :math:`l = 1`. The partial cross sections of the four moments are the
  degree-1 and degree-2 partials, :math:`\sigma_p = k^4|p|^2/(6\pi
  \varepsilon^2|E_\mathrm{inc}|^2)` and so on. :math:`Q^e` has Jackson's
  long-wavelength limit :math:`\int\rho\,(3r_\alpha r_\beta -
  r^2\delta_{\alpha\beta})\,dV`; :math:`Q^m` has
  :math:`\int[r_\alpha(r\times J)_\beta + r_\beta(r\times J)_\alpha]\,dV`.
  Two factor-3 traps: Evlyukhin's and the anapole review's :math:`Q^m` is
  :math:`\tfrac13` of this one (their :math:`\sigma_{Q^m}` has
  :math:`1/(80\pi)` where this has :math:`1/(720\pi)`), and a Landau-style
  :math:`Q^e = \int\rho\,(r_\alpha r_\beta - \tfrac13r^2\delta_{\alpha\beta})\,dV`
  is :math:`\tfrac13` of this one.
- Long-wavelength moments, `gfem.CurrentMoments`, from the induced current
  :math:`J = +i\omega\varepsilon_0(\varepsilon_r - \varepsilon_b)E` of the
  total field over the scatterer, :math:`r` from the origin:

  .. math::

      p_0 = \frac1{i\omega}\int J\,dV,\qquad
      T = \frac1{10v}\int\big[(r\cdot J)\,r - 2r^2J\big]\,dV,\qquad
      p \approx p_0 - ikT,

  in C·m and C·m². The literature's :math:`p \approx p_0 + ikT`
  (:math:`e^{-i\omega t}`) becomes this one. The anapole review's
  :math:`T` in A·m³ is :math:`vT`. The long-wavelength anapole condition is
  :math:`p_0 = ikT`; the exact one is :math:`p = 0` (:math:`\alpha_{1m} = 0`),
  and the two differ by several percent of :math:`|p_0|` for a sphere. The
  first Kerker condition, for :math:`\hat k = \hat z`, is
  :math:`\alpha_{1,\pm1}` against :math:`\beta_{1,\pm1}` as the Mie map has
  them, equivalently :math:`p_x = m_y/v`.
- Reading a paper in :math:`e^{-i\omega t}`: conjugate every phasor and
  material, and because :math:`\overline{Y_{lm}} = (-1)^mY_{l,-m}` and
  :math:`\overline{X_{lm}} = (-1)^{m+1}X_{l,-m}`, a coefficient defined with
  :math:`Y_{lm}^*` maps to the conjugate with :math:`m \to -m` and a sign.
  Bohren and Huffman's :math:`a_l, b_l` are conjugated and enter through
  the Mie map; Grahn's :math:`a_E(l,m)` is
  :math:`\alpha_{lm} = -(-1)^m(-i)^l\sqrt{\pi(2l+1)}\,\overline{a_E(l,-m)}`
  and likewise :math:`\beta`; Jackson's :math:`a_E(l,m)` (A, absolute) is
  :math:`E_\mathrm{inc}\alpha_{lm} = iZ_0(-1)^m\overline{a_E(l,-m)}` and
  :math:`E_\mathrm{inc}\beta_{lm} = -(-1)^mZ_0\overline{a_M(l,-m)}`; Alaee's
  moments are gfem's with the kernels :math:`-1/(i\omega) \to +1/(i\omega)`.

Port waves and beam coupling
----------------------------
A source column's port waves (``DrivenResult.port_waves``) and
`gfem.beam_coupling` are amplitudes in √W, like S for a port fed
`INCIDENT_POWER`.

- Port wave: :math:`b`, the outgoing wave at a port for the source's
  column. On a real reference (a photonic `gfem.WavePort` without a voltage
  path on a lossless guide, whose modal waves carry 1 W) :math:`|b|^2` is
  the power the source sends into the port, in W; with a voltage path, or
  on a lossy guide, :math:`b` is a pseudo-wave, like S, and :math:`|b|^2` is
  not exactly a power.
- Beam coupling: :math:`c`, the amplitude a field couples into an incoming
  beam :math:`w` (described travelling toward the device, as a fiber
  launches it) through the element layer :math:`L` just outside a side of
  P1 indicator :math:`\chi`, in the beam's medium of permeability
  :math:`\mu`:

  .. math::

      c = \frac{1}{4i\omega\mu\sqrt P}\int_L \Big[
          E\cdot(\nabla\chi\times\nabla\times w)
          + (\nabla\times E)\cdot(\nabla\chi\times w)\Big]\,dV,
      \qquad
      P = \frac{1}{2\omega\mu}\operatorname{Im}\int_L
          \nabla\chi\cdot(w\times\overline{\nabla\times w})\,dV,

  with :math:`P` the beam's power into the side, so that :math:`c` refers
  to the beam scaled to 1 W through the side. Its phase is the beam's (a
  `gfem.GaussianBeam`'s zero focal phase). By reciprocity, for a port
  column :math:`c` is the port wave :math:`b` of the beam's column when the
  beam lights the device; :math:`|c|^2` is a coupled power on a real
  reference, a pseudo-wave otherwise, as for :math:`b`.

Circuit components
------------------
`gfem.SaxComponent` and `gfem.NetworkParameters.to_sdict` hand S to a SAX
circuit as ``{(input, output): S}`` over circuit port names, so
``sd["o1", "o2"]`` is :math:`S_{21}`. The circuit's kind sets the convention:

- ``"optical"``: SAX's photonic models are :math:`e^{-i\omega t}`, so S is
  returned complex-conjugated, :math:`\overline{S}`, with the normalization
  of S (unit modal power on a wave port). The circuit's wavelength is ``wl``
  in µm.
- ``"rf"``: SAX's RF models are :math:`e^{+i\omega t}`, as gfem, so S keeps
  its time convention. It is renormalized to one real reference impedance
  ``z0`` (ohm, 50 by default) shared by every model of the circuit, and the
  circuit's frequency is ``f`` in Hz.

The sign of a wave port's row and column of S is physical only with a
``polarity`` reference or a ``voltage_path``, which a component requires on
every wave port it exposes. Between the frequencies of its band a component
interpolates the real and imaginary parts of S, so a circuit's derivative
with respect to ``wl`` or ``f`` is the interpolant's.
"""

from __future__ import annotations

__all__ = [
    "BLOCH_PHASE",
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
#: The wave definition of S-parameters (Marks and Williams; power waves on a
#: real reference).
WAVES = "pseudo"
#: Sheet admittance of one `Conductor` term on an interior surface for a film
#: much thinner than its skin depth, in units of :math:`\sigma t` (conductivity
#: times thickness).
INTERIOR_SHEET_ADMITTANCE = 0.5
#: Factor on a `Conductor`'s thickness on mesh-exterior faces, where the metal
#: is one-sided.
EXTERIOR_THICKNESS_FACTOR = 2.0
#: Electric energy of every eigenmode field (see Eigenmodes), J.
MODE_ENERGY = 1.0
#: The Bloch phase across a periodic pair of translation a (see Bloch
#: periodicity).
BLOCH_PHASE = "E(r + a) = exp(-i k·a) E(r)"
