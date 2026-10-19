"""Sign, unit and normalization conventions (time dependence e^{+iωt}, power waves).

Surface conductors
------------------

A `gfem.Conductor` face adds the boundary term ``iω/Z_s(ω) ∫ E_t · v_t``
(e^{+iωt}), with ``Z_s`` the surface impedance of a metal of conductivity
sigma and thickness t in the finite-thickness form of HFSS and Palace
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
"""

#: Sheet admittance of one `gfem.Conductor` term on an interior surface for a
#: film much thinner than its skin depth, in units of sigma t (conductivity
#: times thickness).
INTERIOR_SHEET_ADMITTANCE = 0.5

#: Factor on a `gfem.Conductor`'s thickness on mesh-exterior faces, where the
#: metal is one-sided.
EXTERIOR_THICKNESS_FACTOR = 2.0
