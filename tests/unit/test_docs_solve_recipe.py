"""The "Solve a device" recipe runs and prints what the page shows.

docs/recipes/solve-a-device.md runs nothing at docs build time. This runs its
blocks in order, as a reader does, on the repository's own
docs/tutorials/data/line.msh, and compares every printed number with a hand
formula coded here (the matched-line transmission, the plate attenuation, the
open-ended line's resonances, those of the line loaded by a guide, the plate
capacitance) and with the output block the page shows under the code.
"""

import docs_pages as dp
import numpy as np
import pytest
from scipy.optimize import brentq, newton

PAGE = dp.DOCS / "recipes" / "solve-a-device.md"
C0 = 299792458.0
ETA0 = 376.73
BAND = np.linspace(2e9, 8e9, 7)


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("solve_a_device"))


def _printed(ran, section: str) -> list[np.ndarray]:
    """The numbers each block of ``section`` printed, one array per block."""
    found = [
        dp.numbers(example.printed) for example in ran if example.section == section
    ]
    assert found, f"no Python block under {section!r}"
    return found


def _open_line_resonances(count: int) -> np.ndarray:
    """Resonances (Hz) of 10 mm air, 20 mm of eps_r = 10, 10 mm air, open ends.

    A TEM line with a magnetic wall at each end; by symmetry the middle plane
    is a magnetic or an electric wall, which gives one equation each.
    """
    n, a = np.sqrt(10.0), 10e-3

    def magnetic(f):
        k = 2 * np.pi * f / C0
        return np.tan(k * a) + n * np.tan(k * n * a)

    def electric(f):
        k = 2 * np.pi * f / C0
        return np.tan(k * a) * np.tan(k * n * a) - n

    grid = np.linspace(0.2e9, 9e9, 4000)
    roots = []
    for equation in (magnetic, electric):
        values = equation(grid)
        for lo, hi, at_lo, at_hi in zip(
            grid[:-1], grid[1:], values[:-1], values[1:], strict=True
        ):
            # A sign change between two small values is a root, not a pole.
            if at_lo * at_hi < 0 and abs(at_lo) < 50 and abs(at_hi) < 50:
                roots.append(brentq(equation, lo, hi))
    return np.sort(roots)[:count]


def _guide_loaded_resonance(guess: complex) -> complex:
    """A complex resonance (Hz) of the line open at one end, a guide at the other.

    Air 10 mm, eps_r = 10 for 20 mm, air 10 mm, open (a magnetic wall) at the
    far end; the guide is the air line, of impedance 1 here. Under
    exp(+i w t) a resonance is a root of Z_in(f) + 1 = 0 at the port, with
    Z_in looking into the line.
    """
    n = np.sqrt(10.0)

    def load(z_load, z, k, length):
        t = np.tan(k * length)
        return z * (z_load + 1j * z * t) / (z + 1j * z_load * t)

    def residual(f):
        k = 2 * np.pi * f / C0
        z = -1j / np.tan(k * 10e-3)  # the open-ended 10 mm of air
        z = load(z, 1 / n, n * k, 20e-3)
        return load(z, 1.0, k, 10e-3) + 1.0

    return complex(newton(residual, guess, tol=1e-3, maxiter=100))


def _coupling_q(f: complex) -> float:
    """The guide's coupling Q, Re w E / P, at the resonance ``f`` (Hz).

    The line's voltage V(z) runs from the open end (V = 1, I = 0) back to the
    port through each layer's exact transfer matrix; E = (1/2) integral of
    C |V|^2 dz with C = eps_r / c, the power out of the port
    P = -(1/2) Re(V I*) with the guide's impedance 1.
    """
    w = 2 * np.pi * f
    state = np.array([1.0 + 0j, 0.0 + 0j])
    energy = 0.0
    for length, eps_r in ((10e-3, 1.0), (20e-3, 10.0), (10e-3, 1.0)):
        k, z = w * np.sqrt(eps_r) / C0, 1 / np.sqrt(eps_r)
        steps = np.linspace(0.0, length, 2001)
        v_end, i_end = state
        # back from the layer's far face: V(s), s the distance from that face
        v = v_end * np.cos(k * steps) + 1j * z * i_end * np.sin(k * steps)
        energy += 0.5 * (eps_r / C0) * np.trapezoid(abs(v) ** 2, steps)
        i = i_end * np.cos(k * length) + 1j * v_end / z * np.sin(k * length)
        state = np.array([v[-1], i])
    v_port, i_port = state
    power = -0.5 * np.real(v_port * np.conj(i_port))
    return w.real * energy / power


def test_the_page_downloads_the_tutorial_mesh():
    first = dp.examples(PAGE.read_text(encoding="utf-8"))[0]
    assert f'url = "{dp.MESH_URL}"' in first.code
    assert dp.MESH.is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_every_section_with_a_number_prints_it(ran):
    printing = {example.section for example in ran if example.printed.strip()}
    assert printing == {
        "S-parameters over a band",
        "Walls and metals",
        "Resonances and Q",
        "Coupling Q of a port",
        "A resonator loaded by a guide",
        "Capacitance",
        "Time each phase",
        "Check your results",
    }


def test_the_band_is_the_matched_line_transmission(ran):
    (printed,) = _printed(ran, "S-parameters over a band")
    frequency, s21, db, z_ref = printed[:7], printed[7:14], printed[14:21], printed[21:]
    assert frequency * 1e9 == pytest.approx(BAND)
    # The page says "within 0.00002".
    assert s21 == pytest.approx(dp.line_formula(BAND), abs=2e-5)
    assert s21[[0, 3, 6]] == pytest.approx([0.8306, 0.9013, 0.6052], abs=1e-4)
    assert db == pytest.approx(20 * np.log10(s21), abs=1e-6)
    assert z_ref == pytest.approx([ETA0 / 6, ETA0 / 6])


def test_an_unnamed_surface_is_a_magnetic_wall(ran):
    walls, _ = _printed(ran, "Walls and metals")
    unnamed, named, formula = walls[:2], walls[2:4], walls[4]
    assert unnamed == pytest.approx(named, abs=1e-12)
    k = 2 * np.pi * 5e9 / C0
    assert formula == pytest.approx(np.angle(np.exp(-1j * k * 40e-3)))
    assert named == pytest.approx([1.0, formula], abs=1e-6)


def test_a_copper_plate_absorbs_what_the_attenuation_formula_says(ran):
    _, (solved, formula) = _printed(ran, "Walls and metals")
    r_s = np.sqrt(np.pi * 5e9 * 4e-7 * np.pi / 5.8e7)
    alpha = r_s / (2 * ETA0 * 1e-3)
    assert formula == pytest.approx(np.exp(-2 * alpha * 40e-3), rel=1e-12)
    assert formula == pytest.approx(0.998043, abs=1e-6)
    assert solved == pytest.approx(formula, abs=2e-6)


def test_the_resonances_are_the_open_ended_line_and_q_the_ceramic_loss(ran):
    (printed,) = _printed(ran, "Resonances and Q")
    frequency, q, share, from_share = (printed[i : i + 3] for i in (0, 3, 6, 9))
    roots = _open_line_resonances(3)
    assert roots / 1e9 == pytest.approx([2.14197, 4.19063, 5.96308], abs=1e-5)
    # The page says "within 5 kHz".
    assert frequency * 1e9 == pytest.approx(roots, abs=5e3)
    assert q == pytest.approx([1251.7, 1379.5, 1748.3], abs=0.05)
    assert from_share == pytest.approx(1 / (1e-3 * share), rel=1e-7)
    assert q == pytest.approx(from_share, rel=1e-6)
    assert "True" in ran[3].printed


def test_two_equal_ports_halve_the_coupling_q(ran):
    (printed,) = _printed(ran, "Coupling Q of a port")
    q_p1, q_p2, q, parallel = (printed[i : i + 2] for i in (0, 2, 4, 6))
    assert q_p1 == pytest.approx([665.28, 792.96], abs=0.01)
    assert q_p2 == pytest.approx(q_p1, rel=1e-9)
    assert parallel == pytest.approx(q_p1 / 2, rel=1e-9)
    assert q == pytest.approx(parallel, rel=1e-5)


def test_the_guide_loaded_resonances_are_the_line_roots(ran):
    (printed,) = _printed(ran, "A resonator loaded by a guide")
    frequency, q, q_port, share = (printed[i : i + 2] for i in (0, 2, 4, 6))
    # The page says "near 1": a lossless line has only its requested mode.
    assert share == pytest.approx([1.0, 1.0], abs=1e-6)
    roots = np.array(
        [_guide_loaded_resonance(g) for g in (4.4e9 + 0.2e9j, 6.4e9 + 0.2e9j)]
    )
    assert roots.real / 1e9 == pytest.approx([4.438665405, 6.406274154], abs=1e-9)
    assert roots.imag / 1e9 == pytest.approx([0.220892787, 0.184070298], abs=1e-9)
    # The page says "within 10 kHz".
    assert frequency * 1e9 == pytest.approx(roots.real, abs=1e4)
    exact_q = abs(roots) / (2 * roots.imag)
    exact_q_port = [_coupling_q(f) for f in roots]
    assert exact_q == pytest.approx([10.0595, 17.4089], abs=1e-4)
    assert exact_q_port == pytest.approx([10.0471, 17.4017], abs=1e-4)
    # The page says "the same to the third decimal".
    assert q == pytest.approx(exact_q, abs=5e-4)
    assert q_port == pytest.approx(exact_q_port, abs=5e-4)
    assert (q_port < q).all()


def test_the_capacitance_is_two_plate_capacitors(ran):
    (printed,) = _printed(ran, "Capacitance")
    # eps_0 is the page's own constant, written 8.8541878128e-12.
    solved, formula = printed[-2:]
    hand = 8.8541878128e-12 * (6e-3 / 1e-3) * (20e-3 + 10.0 * 20e-3) * 1e12
    assert formula == pytest.approx(hand, rel=1e-12)
    assert formula == pytest.approx(11.68753, abs=1e-5)
    assert solved == pytest.approx(formula, rel=1e-8)


def test_the_checks_hold_on_the_first_solve(ran):
    (printed,) = _printed(ran, "Check your results")
    balance, (order_2, order_3) = printed[:7], printed[-2:]
    assert balance == pytest.approx(1.0, abs=1e-6)  # the page: "better than 1e-6"
    (band,) = _printed(ran, "S-parameters over a band")
    assert order_2 == pytest.approx(band[7 + 3], abs=1e-8)
    # The page: "differ in the sixth digit".
    assert 1e-7 < abs(order_3 - order_2) < 1e-5
    (check,) = [e for e in ran if e.section == "Check your results"]
    assert check.printed.splitlines()[-2] == "True"
