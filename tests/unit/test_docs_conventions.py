"""The conventions page names every convention; the theory pages show each on a solve.

docs/conventions.md holds one entry per convention. This checks that it
names every constant of `gfem.conventions` (so a constant added there without
a line on the page fails here) with its value, and runs the blocks of the
theory pages that check the conventions on the repository's own
docs/tutorials/data/line.msh (and heater and modulator sections): what they
print is what the pages show, and is what each convention says it must be.
"""

import re

import docs_pages as dp
import numpy as np
import pytest

import gfem

PAGE = dp.DOCS / "conventions.md"
#: The theory pages that hold the executed checks of the conventions.
THEORY = (
    "phasors-and-power-waves",
    "plane-waves-and-floquet-orders",
    "multiphysics-couplings",
    "nonlinear-and-quantum-optics",
)
RUN = THEORY[:3]  # the pages with executed blocks


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def floquet_page():
    return (dp.DOCS / "plane-waves-and-floquet-orders.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    found = []
    for name in RUN:
        workdir = tmp_path_factory.mktemp(name)
        with pytest.MonkeyPatch.context() as patch:
            patch.syspath_prepend(str(workdir))  # the section builder it downloads
            found += dp.run_page(dp.DOCS / f"{name}.md", workdir)
    return found


def _printed(ran, section: str) -> np.ndarray:
    (example,) = [example for example in ran if example.section == section]
    return dp.numbers(example.printed)


def _table(page: str) -> dict[str, str]:
    """The page's table of constants: name -> the text of its value cell."""
    rows = {}
    for line in page.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0].startswith("`gfem.conventions."):
            rows[cells[0].strip("`").removeprefix("gfem.conventions.")] = cells[1]
    return rows


@pytest.mark.parametrize("name", gfem.conventions.__all__)
def test_the_page_states_every_constant_with_its_value(page, name):
    value = getattr(gfem.conventions, name)
    table = _table(page)
    assert name in table, f"no row for gfem.conventions.{name} in the page's table"
    if isinstance(value, str):
        assert table[name] == f"`{value}`"
    else:
        (shown,) = dp.numbers(table[name])
        assert shown == value
    # Beyond the table, a section says what the constant means on a solve.
    # (plain, or as the `{py:data}` link to the module's constant)
    prose = dp._FENCE.sub("", page.split("## Units")[1])
    assert re.search(rf"`(~?gfem\.conventions\.)?{name}`", prose)


def test_the_table_has_no_constant_the_module_lacks(page):
    assert set(_table(page)) == set(gfem.conventions.__all__)


def test_the_page_links_the_module(page):
    assert "{py:obj}`gfem.conventions`" in page


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_mesh_coordinates_are_in_metres(ran):
    (example,) = [example for example in ran if example.section == "Units"]
    assert example.printed.splitlines()[0] == "mm"
    assert dp.numbers(example.printed) == pytest.approx([40e-3, 6e-3, 1e-3])


def test_the_phase_of_a_transmission_falls_with_frequency(ran):
    phase = _printed(ran, "Time dependence")
    assert len(phase) == 7
    assert np.all(np.diff(phase) < 0)
    assert phase[[0, -1]] == pytest.approx([3.05, -7.83], abs=5e-3)


def test_a_driven_field_carries_what_the_port_does_not_reflect(ran):
    through, unreflected = _printed(ran, "S-parameters and power")
    # 1 W incident: the transmitted fraction of the matched-line formula.
    assert unreflected == pytest.approx(dp.line_formula(5e9) ** 2, abs=1e-4)
    assert through == pytest.approx(unreflected, abs=2e-6)


def test_a_circuit_receives_the_conjugate_when_photonic_and_the_50_ohm_s_when_rf(ran):
    (example,) = [e for e in ran if e.section == "SAX circuits"]
    assert example.printed.split() == ["True", "True"]


def test_a_decaying_mode_has_a_positive_imaginary_frequency(ran):
    (example,) = [example for example in ran if example.section == "Sign of loss"]
    assert example.printed.split()[1] == "True"
    frequency, q, from_omega = dp.numbers(example.printed)
    assert frequency == pytest.approx(2.14197, abs=1e-5)
    assert q == pytest.approx(1251.7, abs=0.05)
    assert from_omega == pytest.approx(q, rel=1e-12)


def test_a_mode_field_stores_the_mode_energy(ran):
    energy, shares = _printed(ran, "Eigenmode fields")
    assert energy == pytest.approx(gfem.conventions.MODE_ENERGY, rel=1e-9)
    assert shares == pytest.approx(1.0, rel=1e-9)


def test_a_lossy_law_has_a_negative_imaginary_part_and_a_static_value(ran):
    (example,) = [e for e in ran if e.section == "Dispersive materials"]
    real, imag, static = dp.numbers(example.printed)
    # 2 + 1 * f0^2 / (f0^2 - f^2 + i gamma f) at f = 9, f0 = 10, gamma = 1 GHz
    assert complex(real, imag) == pytest.approx(2 + 100 / (19 + 9j), rel=1e-12)
    assert imag < 0
    assert static == 3.0


def test_the_dispersive_section_names_the_two_derived_permittivities(page):
    section = dp._prose(dp.section(page, "Dispersive materials"))
    assert "{term}`static permittivity`" in section
    assert "{term}`energy permittivity`" in section
    assert "{doc}`recipes/dispersive-materials`" in section


# -- Floquet ports: a flat glass/air interface in one period of the metagrating cell


def _floquet(ran, block: int) -> np.ndarray:
    found = [example for example in ran if example.section == "Floquet ports"]
    return dp.numbers(found[block].printed)


def test_the_cell_lists_its_orders_on_the_lattice_of_its_pairs(ran):
    numbers = _floquet(ran, 0)
    # a1 along x (2.023 um), a2 along y (0.08 um); (0,0), (-1,0), (1,0) at
    # 1.55 um, only (0, 0) at 3.1 um
    translations = [2.0234e-6, 0, 0, 0, 0.08e-6, 0]
    np.testing.assert_allclose(numbers[:6], translations, atol=1e-10)
    assert list(numbers[6:]) == [0, 0, -1, 0, 1, 0, 0, 0]


def test_a_flat_interface_reflects_n1_minus_n2_over_their_sum_in_both_polarizations(
    ran,
):
    te_re, te_im, tm_re, tm_im, closed, balance = _floquet(ran, 1)
    assert closed == pytest.approx((1.444 - 1) / (1.444 + 1), rel=1e-12)
    assert complex(te_re, te_im) == pytest.approx(closed, abs=1e-3)
    assert complex(tm_re, tm_im) == pytest.approx(closed, abs=1e-3)
    assert balance == pytest.approx(1.0, abs=1e-4)


def test_an_order_that_does_not_propagate_reads_zero_and_has_no_reference_impedance(
    ran,
):
    (example,) = [e for e in ran if e.section == "Floquet ports"][3:4]
    lines = example.printed.splitlines()
    kx, ky, kz = dp.numbers(lines[0])
    assert (kx, ky, kz) == pytest.approx((-2 * np.pi / 2.0234e-6, 0, 0), rel=1e-4)
    assert "False" in lines[0]
    assert lines[1] == "0j"  # exactly zero
    z_glass, z_air = dp.numbers(lines[2])
    assert z_air == pytest.approx(376.73, rel=1e-4)
    assert z_glass == pytest.approx(z_air / 1.444, rel=1e-6)
    assert lines[2].split()[-1] == "nan]"


def test_an_empty_cell_delays_the_phase_by_its_thickness(ran):
    s_re, s_im, airy_re, airy_im = _floquet(ran, 4)
    assert complex(airy_re, airy_im) == pytest.approx(
        np.exp(-2j * np.pi * 1.7 / 3.1), rel=1e-12
    )
    assert complex(s_re, s_im) == pytest.approx(complex(airy_re, airy_im), abs=1e-3)


def test_the_floquet_section_states_the_frame_sigma_and_circular_handedness(
    floquet_page,
):
    section = dp._prose(dp.section(floquet_page, "Floquet ports"))
    for text in (
        r"\hat{\mathbf{e}}_3 \times \hat{\boldsymbol{\kappa}}",  # the TE reference
        r"\sigma_p \sigma_q",  # Bloch reciprocity with Floquet ports
        # IEEE right-handed circular from the top
        r"(S[:, \mathrm{TM}] + i\, S[:, \mathrm{TE}]) / \sqrt{2}",
        "left-handed",  # the same combination from the bottom
        "{doc}`recipes/floquet-ports`",
    ):
        assert text in section, text


# -- Temperature and thermo-optic phase: the heater section of the recipes


def _thermal(ran) -> list:
    return [e for e in ran if e.section == "Temperature and thermo-optic phase"]


def test_a_temperature_at_or_below_zero_kelvin_raises(ran):
    assert _thermal(ran)[0].printed.strip() == "ValueError"


def test_the_temperature_is_the_rise_plus_the_source_free_temperature(ran):
    lines = _thermal(ran)[1].printed.splitlines()
    temperature, rise, source_free = dp.numbers(lines[0])
    assert temperature == pytest.approx(rise + source_free, abs=2e-4)
    assert rise > 0
    # the air above at 310 K warms the source-free core above the 300 K sink
    assert 300.0 < source_free < 310.0
    # a cross-section's totals are per slab thickness: per length, the drive
    per_slab, per_length = (dp.numbers(line)[0] for line in lines[1:])
    assert per_slab == pytest.approx(10.0, rel=1e-12)
    assert per_length == pytest.approx(10.0, rel=1e-12)


def test_the_phase_is_a_delay_that_lowers_a_transmissions_argument(ran):
    positive, angle = _thermal(ran)[3].printed.splitlines()
    assert positive == "True"
    assert float(angle) < 0


def test_the_thermal_section_links_its_recipes(page):
    section = dp.section(page, "Temperature and thermo-optic phase")
    for link in (
        "{doc}`recipes/heater-temperature`",
        "{doc}`recipes/thermal-step-response`",
        "{doc}`recipes/thermo-optic-phase`",
        "{term}`source-free temperature`",
    ):
        assert link in section, link
    assert (
        "no\nreference temperature" in section
        or "no reference temperature" in " ".join(section.split())
    )


def test_the_units_table_lists_the_thermal_units(page):
    units = dp.section(page, "Units")
    for unit in ("W/m/K", "kg/m³", "J/kg/K", "W/m²/K", "S/m", "1/K", "W/m³", "| K |"):
        assert unit in units, unit


def test_the_heat_flows_out_of_the_boundaries_add_up_to_the_source_power(ran):
    blocks = [e for e in ran if e.section == "Temperature and thermo-optic phase"]
    (example,) = [e for e in blocks if "heat_flow_per_length" in e.code]
    sink, air, together = dp.numbers(example.printed)
    assert air < 0  # the air at 310 K warms the section from the top
    assert sink > 10.0
    assert together == pytest.approx(10.0, abs=1e-6)  # the 10 W/m of the heater


# -- Electro-optic shift: the modulator section of the electro-optic recipes


def _electro_optic(ran) -> list:
    return [e for e in ran if e.section == "Electro-optic shift"]


def test_the_electro_optic_phase_is_a_delay_per_volt(ran):
    lowers, delay, angle = _electro_optic(ran)[0].printed.splitlines()
    assert lowers == "True"  # a field along the crystal's +z lowers n_e
    assert float(delay) < 0
    # the argument turns by minus the delay
    assert float(angle) == pytest.approx(-float(delay), abs=1e-4)


def test_push_pull_needs_half_an_arms_v_pi_l(ran):
    arm, pair = dp.numbers(_electro_optic(ran)[1].printed)
    # the coarse arms are mirror images to a few 1e-3
    assert pair == pytest.approx(arm / 2, rel=5e-3)


def test_the_electro_optic_section_states_the_tensor_frame(page):
    section = " ".join(dp._prose(dp.section(page, "Electro-optic shift")).split())
    for text in (
        "m/V",
        "rows $xx, yy, zz, yz, xz, xy$",
        "`axes`",
        "right-handed",
        r"-\varepsilon_r (r \cdot E_\mathrm{s}) \varepsilon_r",
        "{doc}`recipes/electro-optic-phase`",
        "optical window",
    ):
        assert text in section, text
    module = " ".join(gfem.conventions.__doc__.split())
    assert "Electro-optic shift" in module
    assert "Voigt" in module


# -- Time responses: the line's impulse response on a 5 ns record


def test_a_reflection_arrives_at_a_positive_time(ran, page):
    """Under exp(+i omega t) the ceramic's echo comes 2 x 10 mm / c after the edge."""
    (at,) = _printed(ran, "Time responses")
    assert at == pytest.approx(2 * 10e-3 / 299792458.0 * 1e12, abs=3)  # 66.7 ps
    section = dp._prose(dp.section(page, "Time responses"))
    for text in (
        "e^{+2\\pi i f t}",
        "no conjugation",
        "{doc}`recipes/time-responses`",
        "{term}`harmonic grid`",
    ):
        assert text in section, text


# -- Cell outputs: Jones, circular and Mueller on the page's small cells


def _outputs(ran) -> list:
    return [e for e in ran if e.section == "Floquet ports"]


def test_the_interface_transmits_the_power_the_reflection_leaves_in_the_jones_block(
    ran,
):
    tm_re, tm_im, te_re, te_im, expected = dp.numbers(_outputs(ran)[2].printed)
    r = (1.444 - 1) / (1.444 + 1)
    assert expected == pytest.approx(np.sqrt(1 - r**2), rel=1e-12)
    # power-wave entries, diagonal in (TM, TE), within the coarse mesh of the
    # tabulated S
    assert complex(tm_re, tm_im) == pytest.approx(expected, abs=1e-4)
    assert complex(te_re, te_im) == pytest.approx(expected, abs=1e-4)


def _mirror(ran) -> np.ndarray:
    (example,) = [e for e in _outputs(ran) if "mirror_jones" in e.code]
    return dp.numbers(example.printed)


def test_the_documented_top_face_combination_is_the_rcp_column_of_circular(ran):
    """(S_TM + i S_TE)/sqrt2 through U_+1^dagger is the first column of .circular."""
    right, left, *circular = _mirror(ran)
    assert (right, left) == pytest.approx((0.0, -1.0), abs=1e-4)
    # the same column of the circular matrix: right-handed out, then left-handed
    assert (circular[0], circular[2]) == pytest.approx((right, left), abs=1e-4)


def test_a_mirror_swaps_handedness_with_a_minus_sign_and_has_the_polarimetry_mueller(
    ran,
):
    numbers = _mirror(ran)
    minus_p = np.array([[0.0, -1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(numbers[2:6].reshape(2, 2), minus_p, atol=1e-4)
    np.testing.assert_allclose(
        numbers[6:].reshape(4, 4), np.diag([1.0, 1.0, -1.0, -1.0]), atol=1e-4
    )


def test_an_empty_cell_keeps_handedness_and_its_mueller_matrix_is_the_identity(ran):
    (example,) = [e for e in _outputs(ran) if "free_jones" in e.code]
    numbers = dp.numbers(example.printed)
    np.testing.assert_allclose(numbers[:4].reshape(2, 2), np.eye(2), atol=1e-4)
    np.testing.assert_allclose(numbers[4:].reshape(4, 4), np.eye(4), atol=1e-4)


def test_the_page_states_the_cell_output_conventions(floquet_page):
    text = " ".join(dp._prose(dp.section(floquet_page, "Floquet ports")).split())
    for expected in (
        "**Jones matrices.**",
        "ordered (TM, TE)",
        "nothing flips on reflection",
        r"U_h = \frac{1}{\sqrt2}\begin{pmatrix}1 & 1\\ -ih & ih\end{pmatrix}",
        "a mirror therefore swaps the handedness",
        "**Stokes and Mueller.**",
        r"M = A\,(J_w \otimes J_w^*)\,A^{-1}",
        r"\mathrm{diag}(1, 1, -1, -1)",
        "at $-\\psi$",
        "slow axis along $\\hat{\\mathbf{e}}_1$",
        "**Normal incidence in a conical scan.**",
        "**Emission direction.**",
        r"$(\theta, \varphi + \pi)$",
    ):
        assert expected.lower() in text.lower(), expected


def test_incident_fields_gives_the_downward_rcp_phase(page):
    text = " ".join(dp.section(page, "Incident fields").split())
    assert (
        r"So for a wave going down the Floquet circular unit vector "
        r"$(\hat{\mathbf{E}}_\mathrm{TM} + i\hat{\mathbf{E}}_\mathrm{TE})/\sqrt2$ "
        r"of the "
        r"RCP column is $-(\hat{\boldsymbol\theta} - i\hat{\boldsymbol\phi})/\sqrt2$: "
        r'the same right-handed wave as `"RCP"`, with the opposite phase.'
    ) in text


@pytest.mark.parametrize("name", ["conventions", *THEORY])
def test_the_page_names_no_solver_or_vendor_it_must_not(name):
    lowered = (dp.DOCS / f"{name}.md").read_text(encoding="utf-8").lower()
    for word in ("palace", "flexcompute", "tidy3d"):
        assert word not in lowered
