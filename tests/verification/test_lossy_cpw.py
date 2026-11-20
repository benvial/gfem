"""A lossy shielded coplanar waveguide: S21 against ``exp(-gamma L)`` (benvial/gfem#2).

The case is a CPW on a generic silicon / oxide / aluminium stack: a 10 um
signal and 6 um gaps in 2 um of Al (3.77e7 S/m, meshed as a volume) on
6.2 um of SiO2 (eps_r 4.1, filling the gaps) over lossless silicon (eps_r
11.9), under 100 um of air, inside a PEC box 300 um wide.
A PMC plane at the middle of the signal (the even mode) halves the cost:
the full line's impedance is half the half model's. The attenuation is 0.29
of the phase constant at 1 GHz and 0.066 at 10 GHz (``alpha / beta``). The
mesh is built here, from glued rectangles, with the helpers of
``tests/thermal_cases.py``.

A wave port's S is the unconjugated (biorthogonal) projection on the port
mode, in pseudo-waves at the mode's complex Z_PV (`gfem.conventions`), so a
uniform lossy line is matched to its own mode: ``S11 = 0`` and
``S21 = exp(-gamma L)`` up to discretisation. A power-wave (conjugated)
reference reflects at the order of ``alpha / beta`` instead.

- A 500 um straight section, two lossy wave ports, 1 and 10 GHz, on three
  cross-section meshes (maximal edge length in the metal 2, 1, 0.5 um,
  one prism layer along z): ``gamma`` of the Driven result is the 2D
  `gfem.PortMode`'s on the same port triangles.
- The finest section is matched: ``|S11|``, ``|S22|`` and ``|S12 - S21|``
  are small, and ``|S11|`` is far below ``alpha / beta``.
- Its ``S21`` equals ``exp(-gamma L)`` in magnitude and phase. This is a
  3D against 2D consistency check on the same triangles: the residual is the
  bulk difference of the two propagation constants (at 10 GHz, alpha +1.1e-2
  and beta -1.1e-3 relative), which refining along z does not change (one,
  two and four layers gave the same error) and only the cross-section
  refines.
- ``|S11|`` and ``|S21 - exp(-gamma L)|`` fall by at least 3x per halving
  of the metal's edge length (measured 3.7 to 5).
- The 2D mode on a finer section (0.25 um in the metal, whose skin depth
  at 50 GHz is 0.37 um) against frozen femwell and Palace numbers at 1, 10,
  25 and 50 GHz: the effective index, the attenuation and the impedance.

Measured here (Windows 11, 16 threads, order 2, direct solver): 1392, 2925
and 6501 tetrahedra; |S11| at 1 / 10 GHz on the three meshes 3.1e-4 / 4.8e-3,
6.5e-5 / 1.3e-3, 1.4e-5 / 2.7e-4; the Driven gamma equal to the 2D
PortMode's within 4.5e-8; the 2D section (6949 triangles) against Palace
within 5.5e-4 in Re n, 3.4e-4 in alpha and 4.4e-4 in Z_PV; the file 86 s.
"""

import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc

import gfem

# Geometry, um (y vertical, x across the half section from the symmetry plane).
X_BOX, Y_BOT, Y_TOP = 150.0, -203.0, 105.2
Y_SI, Y_M0, Y_M1 = -3.0, 3.2, 5.2
EDGE, INNER = 5.0, 11.0
#: The voltage path's height, and how far it stays off the metal.
Y_V, INSET = 4.2, 0.002
SIGMA = 3.77e7  # S/m
LENGTH_UM = 500.0
LENGTH = LENGTH_UM * 1e-6
FREQUENCIES = (1e9, 10e9)
LADDER = (2.0, 1.0, 0.5)  # maximal edge length in the metal, um
FINE = 0.25  # the 2D section compared with the frozen numbers
UM = 1e-6
NEPER_TO_DB = 20 / np.log(10)

MATERIALS = {
    "si": gfem.Material(eps_r=11.9),
    "oxide": gfem.Material(eps_r=4.1),
    "air": gfem.Material(),
    "signal": gfem.Material(conductivity=SIGMA),
    "ground": gfem.Material(conductivity=SIGMA),
}


def _rects(h_metal, h_near, *, split=31.0):
    """Glued rectangles of the half section: (x0, y0, x1, y1, name, maxh).

    Fine in the signal, the gap and the ground up to ``split``; an oxide and
    air band at ``h_near`` around the slot. The gaps are oxide to the top of
    the metal.
    """
    return [
        (0.0, Y_BOT, X_BOX, Y_SI, "si", None),
        (0.0, Y_SI, split, Y_M0, "oxide", h_near),
        (split, Y_SI, X_BOX, Y_M0, "oxide", None),
        (0.0, Y_M0, EDGE, Y_M1, "signal", h_metal),
        (EDGE, Y_M0, INNER, Y_M1, "oxide", h_metal),
        (INNER, Y_M0, split, Y_M1, "ground", h_metal),
        (split, Y_M0, X_BOX, Y_M1, "ground", 4 * h_metal),
        (0.0, Y_M1, split, Y_M1 + 10.0, "air", h_near),
        (split, Y_M1, X_BOX, Y_M1 + 10.0, "air", None),
        (0.0, Y_M1 + 10.0, X_BOX, Y_TOP, "air", None),
    ]


def _on_box(c):
    x, y, _ = c
    eps = 1e-6
    return abs(x - X_BOX) < eps or abs(y - Y_BOT) < eps or abs(y - Y_TOP) < eps


EDGES = [(_on_box, "box"), (lambda c: abs(c[0]) < 1e-6, "symmetry")]


def _path(z):
    """Voltage path from the signal to the ground across the gap, at ``z`` m."""
    return (
        ((EDGE + INSET) * UM, Y_V * UM, z),
        ((INNER - INSET) * UM, Y_V * UM, z),
    )


def _ports(*, line):
    if line:
        return [
            gfem.WavePort("p1", "front", voltage_path=_path(0.0)),
            gfem.WavePort("p2", "back", voltage_path=_path(LENGTH)),
        ]
    return [gfem.WavePort("p1", "front", voltage_path=_path(0.0))]


def _model(mesh, *, line):
    return gfem.Model(
        mesh,
        materials=MATERIALS,
        boundaries=[gfem.PEC("box"), gfem.PMC("symmetry")],
        ports=_ports(line=line),
    )


OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("cpw")


@pytest.fixture(scope="module")
def ladder(workdir):
    """The straight line at each mesh of ``LADDER``: model, S, gamma, z_ref."""
    out = []
    for h in LADDER:
        mesh = tc.extruded(
            workdir / f"line_{h}.msh",
            _rects(h, 4 * h),
            30.0,
            EDGES,
            length=LENGTH_UM,
            layers=1,
            ends=True,
        )
        assert set(mesh.volumes) == {"si", "oxide", "signal", "ground", "air"}
        assert {"box", "symmetry", "front", "back"} <= set(mesh.surfaces)
        model = _model(mesh, line=True)
        result = gfem.solve(model, gfem.Driven(FREQUENCIES), options=OPTIONS)
        out.append(
            {
                "h": h,
                "tets": len(mesh.tetrahedra),
                "model": model,
                "result": result,
                "s": np.asarray(result.s.values),
                "gamma": np.asarray(result.port_modes["p1"].gamma),
                "z_ref": np.asarray(result.s.z_ref),
            }
        )
    print("\ntetrahedra:", [r["tets"] for r in out])  # noqa: T201
    return out


def _s21_error(level):
    """``|S21 - exp(-gamma L)|`` and ``|S11|`` at each frequency."""
    e = np.exp(-level["gamma"] * LENGTH)
    return np.abs(level["s"][:, 1, 0] - e), np.abs(level["s"][:, 0, 0])


def _alpha_db_per_cm(gamma):
    return NEPER_TO_DB * np.asarray(gamma).real / 100


def _alpha_of_index(n, frequency):
    """Attenuation, dB/cm, of the effective index ``n' - i n''``."""
    k0 = 2 * np.pi * np.asarray(frequency) / scipy.constants.c
    return _alpha_db_per_cm(1j * k0 * np.asarray(n))


def test_the_attenuation_formula_on_a_known_index():
    # n = 2 - 0.1j at 10 GHz attenuates 0.1 k0 Np/m, 0.1 k0 (20 / ln 10) / 100 dB/cm
    k0 = 2 * np.pi * 10e9 / scipy.constants.c
    assert _alpha_of_index(2 - 0.1j, 10e9) == pytest.approx(
        0.1 * k0 * 8.685889638065035 / 100, rel=1e-12
    )
    assert _alpha_of_index(2 - 0.1j, 10e9) == pytest.approx(1.8204, rel=1e-4)


def test_the_driven_gamma_is_the_2d_port_modes(ladder):
    finest = ladder[-1]
    mode = gfem.solve(
        finest["model"], gfem.PortMode("p1", FREQUENCIES), options=OPTIONS
    )
    driven = finest["result"].port_modes["p1"]
    assert np.asarray(mode.solved).all()
    assert np.asarray(driven.solved).all()
    np.testing.assert_allclose(
        np.asarray(mode.gamma), np.asarray(driven.gamma), rtol=1e-6
    )
    print(  # noqa: T201
        "\nalpha/beta:",
        [float(g.real / g.imag) for g in np.asarray(mode.gamma)],
    )


#: Bounds on the finest mesh: 1 GHz, 10 GHz (measured 1.4e-5, 2.7e-4).
S_BOUND = np.array([3e-5, 5e-4])


def test_the_lossy_line_is_matched_to_its_own_mode(ladder):
    """S11 vanishes up to discretisation, far below ``alpha / beta``.

    The S functional is the unconjugated projection, so a uniform line
    reflects nothing; the error falls at about second order in the metal's
    edge length (see the convergence test). A conjugated (power-wave)
    reference would reflect ``|Im Z| / |Z|``, 0.28 and 0.066 here.
    """
    finest = ladder[-1]
    s, gamma = finest["s"], finest["gamma"]
    z = np.asarray(finest["result"].port_modes["p1"].Z_PV)
    ratio = gamma.real / gamma.imag
    for i, f in enumerate(FREQUENCIES):
        print(  # noqa: T201
            f"\n{f / 1e9:g} GHz: |S11| {abs(s[i, 0, 0]):.2e}, "
            f"alpha/beta {ratio[i]:.4f}, |Im Z_PV|/|Z_PV| {abs(z[i].imag / z[i]):.4f}"
        )
    s11 = np.abs(s[:, 0, 0])
    s22 = np.abs(s[:, 1, 1])
    assert (s11 <= S_BOUND).all()
    assert (s22 <= S_BOUND).all()
    assert (s11 < 0.01 * ratio).all()
    assert (np.abs(s[:, 0, 1] - s[:, 1, 0]) <= 1e-7).all()


def test_s21_is_exp_minus_gamma_l_in_magnitude_and_phase(ladder):
    """The 3D transmission against the 2D mode's ``exp(-gamma L)``.

    Both live on the same face triangles, so the residual is the bulk
    difference between the 3D and the 2D propagation constant: refining along
    z does not change it, refining the cross-section does.
    """
    finest = ladder[-1]
    e = np.exp(-finest["gamma"] * LENGTH)
    s21 = finest["s"][:, 1, 0]
    db = 20 * np.log10(np.abs(s21) / np.abs(e))
    phase = np.degrees(np.angle(s21 / e))
    for i, f in enumerate(FREQUENCIES):
        print(  # noqa: T201
            f"\n{f / 1e9:g} GHz: d|S21| {db[i]:+.2e} dB, dphase {phase[i]:+.2e} deg, "
            f"|S21 - e| {abs(s21[i] - e[i]):.2e}"
        )
    assert (np.abs(db) <= [1e-4, 2.5e-3]).all()
    assert (np.abs(phase) <= [2e-3, 3e-2]).all()
    assert (np.abs(s21 - e) <= S_BOUND).all()


def test_the_errors_fall_with_the_cross_section_mesh(ladder):
    errors = np.array([np.concatenate(_s21_error(level)) for level in ladder])
    for level, row in zip(ladder, errors, strict=True):
        print(  # noqa: T201
            f"\nh_metal {level['h']} um: |S21 - e| {row[:2]}, |S11| {row[2:]}"
        )
    ratios = errors[:-1] / errors[1:]
    print("\nratios per halving:\n", ratios)  # noqa: T201
    assert (ratios >= 3.0).all()


# --- the 2D mode against frozen femwell and Palace numbers ---------------------
#
# Reference: the same shielded CPW, the full line; Palace's 2D port mode on
# 27 522 triangles of order 2, and femwell on its own mesh.
# Provenance: the comparison of benvial/gfem#2 (its table rounds these
# numbers). Palace 2D boundary mode, build 295645f, driven by gsim 0.9.0;
# femwell 0.1.12; both on the same 27 522-triangle cross-section (metal
# 0.15 um, order 2). Source files: doplaydo/simulation-benchmarks at 177b68f
# (private; available on request).
#
# Frozen at 1, 10, 25 and 50 GHz: n_eff = n' - i n''.
GHZ = (1e9, 10e9, 25e9, 50e9)
PALACE_N = np.array(
    [
        2.133006767 - 0.613861182j,
        1.910114000 - 0.126555648j,
        1.873480977 - 0.078183035j,
        1.851462947 - 0.056388136j,
    ]
)
FEMWELL_N = np.array(
    [
        2.133005966 - 0.613858958j,
        1.910114697 - 0.126551903j,
        1.873482470 - 0.078173871j,
        1.851467452 - 0.056370523j,
    ]
)
#: Palace's power-voltage impedance of the full line, ohm.
PALACE_Z = np.array(
    [
        66.76036 - 19.21304j,
        59.78197 - 3.96087j,
        58.62420 - 2.44640j,
        57.89154 - 1.76294j,
    ]
)
#: femwell's power-current impedance (real and imaginary part), ohm: reported only.
FEMWELL_Z_PI = np.array(
    [
        66.81729 - 19.23126j,
        59.83637 - 3.96488j,
        58.68211 - 2.44833j,
        57.96759 - 1.76397j,
    ]
)
#: Attenuation of those indices, dB/cm: from n'' at the frequency.
PALACE_ALPHA = (1.1175, 2.3039, 3.5582, 5.1325)
FEMWELL_ALPHA = (1.1175, 2.3038, 3.5577, 5.1309)

#: Relative agreement with the references; measured at most 5.5e-4 (Re n'),
#: 3.4e-4 (alpha) and 4.4e-4 (Z_PV) against Palace, 5.5e-4 and 6.5e-4
#: against femwell. The two references differ by 3.1e-4 in n'' at 50 GHz,
#: and gfem on Palace's own mesh matched Palace to 5e-6 in n_eff at 1 GHz and
#: 1.2e-8 at 10 to 50 GHz (the comparison of benvial/gfem#2), so the
#: remainder is this in-test mesh.
REFERENCE_RTOL = 1.5e-3


@pytest.fixture(scope="module")
def section_mode(workdir):
    mesh = tc.section(workdir / "section.msh", _rects(FINE, 1.0), 30.0, EDGES)
    model = _model(mesh, line=False)
    mode = gfem.solve(model, gfem.PortMode("p1", GHZ), options=OPTIONS)
    print(f"\n2D section: {len(mesh.tetrahedra) // 3} triangles")  # noqa: T201
    return mode


def test_the_frozen_attenuations_follow_from_the_frozen_indices():
    for n, alpha in ((PALACE_N, PALACE_ALPHA), (FEMWELL_N, FEMWELL_ALPHA)):
        np.testing.assert_allclose(_alpha_of_index(n, GHZ), alpha, rtol=1e-4)


def test_the_2d_mode_matches_the_frozen_palace_and_femwell_numbers(section_mode):
    mode = section_mode
    assert np.asarray(mode.solved).all()
    n = np.asarray(mode.n_eff)
    alpha = _alpha_db_per_cm(mode.gamma)
    # the PMC plane halves the power, so the full line's impedance is half
    z = np.asarray(mode.Z_PV) / 2
    zpi = _z_pi(mode)
    for i, f in enumerate(GHZ):
        print(  # noqa: T201
            f"\n{f / 1e9:g} GHz: n {n[i]:.6f}, alpha {alpha[i]:.4f} dB/cm, "
            f"Z_PV {z[i]:.3f} (Palace {PALACE_Z[i]:.3f}), "
            f"Z_PI {zpi[i]:.3f} (femwell {FEMWELL_Z_PI[i]:.3f}, "
            f"rel {abs(zpi[i] / FEMWELL_Z_PI[i] - 1):.1e}, reported only)"
        )
    for reference in (PALACE_N, FEMWELL_N):
        np.testing.assert_allclose(n.real, reference.real, rtol=REFERENCE_RTOL)
        np.testing.assert_allclose(
            alpha, _alpha_of_index(reference, GHZ), rtol=REFERENCE_RTOL
        )
    assert (np.abs(z / PALACE_Z - 1) <= REFERENCE_RTOL).all()


def _z_pi(mode):
    """Z_PI of the full line, from a loop round the half signal to the PMC plane."""
    loop = [
        (0.0, 1.7 * UM, 0.0),
        (8.0 * UM, 1.7 * UM, 0.0),
        (8.0 * UM, 6.7 * UM, 0.0),
        (0.0, 6.7 * UM, 0.0),
    ]
    return np.asarray(gfem.characteristic_impedance(mode, current_path=loop)) / 2
