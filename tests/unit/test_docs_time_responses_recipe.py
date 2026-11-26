r"""The "TDR and time responses" recipe runs and prints what the page shows.

docs/recipes/time-responses.md runs nothing at docs build time. This runs its
blocks in order, as a reader does, on the repository's own
docs/tutorials/data/line.msh, and checks what they print against the line's
closed forms: the ceramic's impedance :math:`Z_0/\sqrt{10}`, the far line's
:math:`Z_0`, the front face's reflection and its round trip, the transmitted
step's end and the rise time of the default window. The page's rise-time
table is checked on a synthetic constant reflection, and its record rule on
the same line solved on a 5 ns record.
"""

import math
import re

import docs_pages as dp
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "time-responses.md"
C0 = 299792458.0
Z_LINE = 376.73 / 6
F_MAX = 20e9
T1 = 2 * 10e-3 / C0  # round trip to the ceramic's front face, s
GAMMA = (1 / math.sqrt(10) - 1) / (1 / math.sqrt(10) + 1)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("time_responses"))


def _printed(ran, section: str) -> np.ndarray:
    (example,) = [example for example in ran if example.section == section]
    return dp.numbers(example.printed)


def test_the_page_downloads_the_tutorial_mesh(page):
    first = dp.examples(page)[0]
    assert f'url = "{dp.MESH_URL}"' in first.code


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_every_section_with_a_number_prints_it(ran):
    printing = {example.section for example in ran if example.printed.strip()}
    assert printing == {
        "A step and an impedance profile",
        "The transmitted step and the rise time",
        "The impulse response",
    }


def test_the_profile_reads_the_ceramic_and_then_the_far_line(ran):
    ceramic, late, by_hand_ceramic, by_hand_late, solves = _printed(
        ran, "A step and an impedance profile"
    )
    assert by_hand_ceramic == pytest.approx(Z_LINE / math.sqrt(10), abs=0.005)
    assert by_hand_late == pytest.approx(Z_LINE, abs=0.005)
    assert ceramic == pytest.approx(Z_LINE / math.sqrt(10), abs=0.05)
    assert late == pytest.approx(Z_LINE, abs=0.05)
    # The page: "solves 11 of the 400 frequencies in full".
    assert solves == 11


def test_the_ceramic_is_read_between_the_round_trips_to_its_faces(page):
    code = dp.examples(page)[0].code
    (at,) = re.findall(r"np\.interp\(\[([\d.e-]+),", code)
    back = T1 + 2 * 20e-3 * math.sqrt(10) / C0
    assert T1 < float(at) < back
    # The page's numbers for the two round trips.
    prose = dp._prose(dp.section(page, "A step and an impedance profile"))
    prose = prose.replace("$", "")
    assert f"{T1 * 1e12:.1f} ps" in prose
    assert f"{back * 1e12:.0f} ps" in prose


def test_the_transmitted_step_ends_at_one_and_the_edge_rises_in_0_99_over_f_max(ran):
    end, rise, _, by_hand = _printed(ran, "The transmitted step and the rise time")
    assert end == pytest.approx(1.0, abs=1e-3)  # S21(0) of a lossless line
    assert by_hand == pytest.approx(0.99 / F_MAX * 1e12, abs=0.05)
    assert rise == pytest.approx(by_hand, rel=0.02)


def test_the_impulse_peaks_at_the_front_faces_reflection_and_round_trip(ran):
    peak, at, gamma, round_trip = _printed(ran, "The impulse response")
    assert gamma == pytest.approx(GAMMA, abs=5e-4)
    assert round_trip == pytest.approx(T1 * 1e12, abs=0.05)
    assert peak == pytest.approx(GAMMA, abs=3e-3)
    assert at == pytest.approx(T1 * 1e12, abs=3)  # within one rise time


def _rise_table(page: str) -> dict[float, float]:
    """The page's table: Kaiser beta -> rise time times f_max."""
    rows = {}
    for line in dp.section(page, "Choose the grid").splitlines():
        match = re.match(r"\s*\| (\d+)[^|]*\| \$([\d.]+)/f_\\mathrm\{max\}\$", line)
        if match:
            rows[float(match[1])] = float(match[2])
    return rows


def test_the_rise_time_table_is_that_of_the_window(page):
    """A constant reflection's step through each window rises as the table says."""
    table = _rise_table(page)
    assert set(table) == {0.0, 6.0, 13.0}
    n, df = 400, F_MAX / 400
    frequency = tuple(df * np.arange(1, n + 1))
    flat = gfem.NetworkParameters(
        values=jnp.full((n, 1, 1), 0.5 + 0j),
        z_ref=jnp.full((n, 1), 50.0),
        frequency=frequency,
        ports=("p",),
        excited=("p",),
    )
    for beta, factor in table.items():
        step = flat.step_response(kaiser_beta=beta)
        # 10 % and 90 % of the jump, first crossings: the unwindowed step
        # overshoots (Gibbs), so the 0.5 it settles at is the reference
        rise = np.subtract(*_crossings(step.time, np.asarray(step["p", "p"]) / 0.5))
        assert -rise * F_MAX == pytest.approx(factor, rel=0.02), beta


def _crossings(time, trace):
    """The first times ``trace`` reaches 0.1 and 0.9, interpolated linearly."""
    found = []
    for level in (0.1, 0.9):
        i = int(np.argmax(trace >= level))
        found.append(np.interp(level, trace[i - 1 : i + 1], time[i - 1 : i + 1]))
    return found


def test_a_record_too_short_for_the_ceramic_shifts_the_profile(ran):
    """The page: a 5 ns record left a 4 % error, a 20 ns one none to read."""
    model = gfem.Model(
        gfem.Mesh.from_gmsh(dp.MESH, length_unit="mm"),
        materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=Z_LINE),
            gfem.LumpedPort("p2", "port2", "+z", impedance=Z_LINE),
        ],
    )
    short = gfem.Driven(frequency=0.2e9 * np.arange(1, 101), sweep="adaptive")
    step = gfem.solve(model, short).s.step_response()
    ceramic = float(np.interp(0.28e-9, step.time, step.impedance("p1")))
    error = abs(ceramic / (Z_LINE / math.sqrt(10)) - 1)
    assert 0.03 < error < 0.08  # 6 % on the plateau, from a 4 % error at DC
    (long_ceramic, *_) = _printed(ran, "A step and an impedance profile")
    assert abs(long_ceramic / (Z_LINE / math.sqrt(10)) - 1) < 2e-3


def test_the_page_links_its_terms_and_pages(page):
    for link in (
        "{term}`harmonic grid`",
        "{term}`time response`",
        "{term}`impedance profile`",
        "{term}`rational model`",
        "{doc}`rational-model`",
        "{doc}`../conventions`",
        "{doc}`adaptive-sweep-gradient`",
    ):
        assert link in page, link
