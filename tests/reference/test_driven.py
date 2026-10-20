"""Driven problems against Palace on the IHP coplanar line with lumped ports.

An 800 um TopMetal2 CPW on the IHP SG13G2 stack, two 50 ohm two-element
lumped CPW ports, surface-impedance conductors and an absorbing boundary;
order 2. The model is built from each case's Palace ``config.json`` as gsim
would build it: the metal volumes are excluded (their shells carry the
`Conductor`), materials take the permittivity and loss tangent, and the
absorbing surfaces (with the case's absorbing order), conductors and ports
take Palace's attributes.

- Parity (``driven/r8_abc1``, ``driven/r4_abc1``, first-order absorbing):
  gfem and Palace discretize the same mesh at the same order, so S agrees
  to the discretization gap between the codes (conductor cracking, the
  solver tolerance): max |dS| <= 1e-5 at Palace's own output frequencies.
- Physics (``driven/r8_box150``, second-order absorbing on an air box 150 um
  wider and 300 um taller): the box-size-converged |S21| = -0.300 dB at
  100 GHz. gfem's textbook second-order form and Palace's opposite sign
  agree here to about 0.001 dB.
- Timing (``driven/r8``, ``driven/r4``, the gsim defaults): one frequency
  sample, both excitations, against Palace's per-sample linear solve.
  gfem's order 2 differs from Palace's on these tight boxes, by design:
  the difference is printed, never asserted.
"""

import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import pytest
from isolated import isolated

import gfem
from gfem._engine import terms
from gfem._engine.driven import DrivenRule
from gfem._engine.runtime import task_manager
from gfem._linalg import Operator, prepare

TOLERANCE = 1e-5
#: Palace output frequencies (GHz) checked: rows of port-S.csv near 10, 50 and
#: 100 GHz.
ROWS = (27, 148, 299)
THREADS = 4


@pytest.fixture(scope="module")
def reference(reference_dir):
    return reference_dir / "driven"


def _model(case: Path) -> gfem.Model:
    config = json.loads((case / "config.json").read_text())
    unit = config["Model"]["L0"]
    mesh = gfem.Mesh.from_gmsh(case / "palace.msh", length_unit="um")
    assert unit == 1e-6
    name = {(r.dim, r.tag): r.name for r in mesh.regions}
    materials = {
        name[3, tag]: gfem.Material(
            eps_r=m["Permittivity"], tan_delta=m.get("LossTan", 0.0)
        )
        for m in config["Domains"]["Materials"]
        for tag in m["Attributes"]
    }
    bnd = config["Boundaries"]
    conductors = [
        gfem.Conductor(name[2, tag], c["Conductivity"], c["Thickness"] * unit)
        for c in bnd["Conductivity"]
        for tag in c["Attributes"]
    ]
    order = bnd["Absorbing"]["Order"]
    absorbing = [
        gfem.Absorbing(name[2, tag], order=order)
        for tag in bnd["Absorbing"]["Attributes"]
    ]
    ports = [
        gfem.LumpedPort(
            f"p{p['Index']}",
            tuple(name[2, e["Attributes"][0]] for e in p["Elements"]),
            tuple(e["Direction"] for e in p["Elements"]),
            impedance=p["R"],
        )
        for p in sorted(bnd["LumpedPort"], key=lambda p: p["Index"])
    ]
    # gsim's WavePortPEC: perfect conductors in the port mode problem only
    pec_edges = tuple(
        name[2, t] for t in bnd.get("WavePortPEC", {}).get("Attributes", [])
    )
    ports += [
        gfem.WavePort(
            f"p{p['Index']}",
            name[2, p["Attributes"][0]],
            mode=p["Mode"],
            offset=p["Offset"] * unit,
            pec_edges=pec_edges,
        )
        for p in sorted(bnd.get("WavePort", []), key=lambda p: p["Index"])
    ]
    shells = [c.surface for c in conductors]
    # with wave ports the metal reaches the line's end faces, which absorb
    ends = tuple(a.surface for a in absorbing) if bnd.get("WavePort") else ()
    excluded = tuple(
        gfem.enclosed_by(s, s.replace("_xy", "_z"), *ends)
        for s in shells
        if s.endswith("_xy")
    )
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[*conductors, *absorbing],
        ports=ports,
        excluded=excluded,
    )


def _palace(case: Path) -> tuple[np.ndarray, np.ndarray]:
    """Palace's S over its whole output grid: (Hz, (n, 2, 2) complex)."""
    data = np.loadtxt(
        case / "output" / "palace" / "port-S.csv", delimiter=",", skiprows=1
    )
    s = np.zeros((len(data), 2, 2), complex)
    # columns: f, then |S11| dB, arg S11, |S21|, arg S21, |S12|, arg S12, |S22|, ...
    for k, (i, j) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        db, deg = data[:, 1 + 2 * k], data[:, 2 + 2 * k]
        s[:, i, j] = 10 ** (db / 20) * np.exp(1j * np.radians(deg))
    return data[:, 0] * 1e9, s


def _db(s):
    return 20 * np.log10(np.abs(s))


def _peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# --- parity -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ndof", "max_iterations"),
    [
        # Palace and the solver spike: 48-49 (r8) and 43-45 (r4) GMRES
        # iterations at this tolerance; allowed +10 %
        ("r8_abc1", 162_640, 53),
        ("r4_abc1", 338_338, 49),
    ],
)
def test_driven_parity_s_matches_palace(reference, name, ndof, max_iterations):
    case = reference / name
    model = _model(case)
    frequency, palace = _palace(case)
    frequency, palace = frequency[list(ROWS)], palace[list(ROWS)]
    options = gfem.SolverOptions(threads=THREADS)
    problem = gfem.Driven(frequency=frequency)
    rule = DrivenRule(model, problem, options)
    assert rule.kind == "bddc_gmres"
    assert rule.operator(model).space.ndof == ndof
    start = time.perf_counter()
    outputs, _ = rule.forward(model, keep=False)
    wall = time.perf_counter() - start
    s = outputs["s"]
    diff = np.abs(s - palace).max()
    print(  # noqa: T201
        f"{name}: max |dS| {diff:.3g}, GMRES iterations {outputs['iterations']}, "
        f"{wall:.1f} s for {len(frequency)} frequencies, peak RSS "
        f"{_peak_rss_gib():.2f} GiB, load {os.getloadavg()}"
    )
    assert diff <= TOLERANCE, (diff, outputs["iterations"])
    assert max(max(its) for its in outputs["iterations"]) <= max_iterations, outputs
    # reciprocal to the GMRES tolerance
    np.testing.assert_allclose(s, np.swapaxes(s, 1, 2), atol=1e-6)
    # the public path (GMRES from zero) gives the same S to solver tolerance
    result = gfem.solve(model, gfem.Driven(frequency=frequency[1]), options=options)
    np.testing.assert_allclose(result.s.values[0], s[1], atol=2e-6)


# --- physics ----------------------------------------------------------------------

#: S at 10, 50 and 100 GHz on driven/r8_box150 with the textbook second-order
#: absorbing boundary, from an independent NGSolve script of the same discrete
#: problem (BDDC + GMRES at 1e-6).
INDEPENDENT = np.array(
    [
        [
            0.03936780451951272 + 0.08591739289705763j,
            0.9354205267421537 - 0.3083735470714336j,
            0.9354205364087202 - 0.308373554743944j,
            0.03938885692635363 + 0.08591059479992527j,
        ],
        [
            0.27463674383158 + 0.0052733089998297745j,
            0.06640690209329778 - 0.9372448023784595j,
            0.06640678836635616 - 0.9372448952288323j,
            0.27466843199596624 + 0.004898789510454887j,
        ],
        [
            0.010932174162803676 - 0.018377031197986317j,
            -0.9549648373111155 - 0.14661176364603762j,
            -0.9549646654245747 - 0.14661169139287622j,
            0.010757627235371325 - 0.018435592650570286j,
        ],
    ]
).reshape(3, 2, 2)


def test_driven_physics_box150_converged_transmission(reference):
    case = reference / "r8_box150"
    model = _model(case)
    assert {b.order for b in model.boundaries if isinstance(b, gfem.Absorbing)} == {2}
    frequency = np.array([10e9, 50e9, 100e9])
    options = gfem.SolverOptions(threads=THREADS)
    rule = DrivenRule(model, gfem.Driven(frequency=frequency), options)
    assert rule.operator(model).space.ndof == 333_910
    start = time.perf_counter()
    outputs, _ = rule.forward(model, keep=False)
    wall = time.perf_counter() - start
    s = outputs["s"]
    expected = INDEPENDENT  # [[S11, S12], [S21, S22]]
    s21_db = _db(s[:, 1, 0])
    f_palace, palace = _palace(case)
    palace21 = np.interp(frequency, f_palace, palace[:, 1, 0].real) + 1j * np.interp(
        frequency, f_palace, palace[:, 1, 0].imag
    )
    print(  # noqa: T201
        f"r8_box150: |S21| {np.round(s21_db, 4)} dB (Palace order 2 "
        f"{np.round(_db(palace21), 4)}), max |dS| vs independent "
        f"{np.abs(s - expected).max():.3g}, GMRES {outputs['iterations']}, "
        f"{wall:.1f} s, peak RSS {_peak_rss_gib():.2f} GiB, load {os.getloadavg()}"
    )
    assert s21_db[2] == pytest.approx(-0.300, abs=0.002)
    np.testing.assert_allclose(s, expected, atol=TOLERANCE)
    # Palace's (opposite-sign) second order agrees on this converged box
    np.testing.assert_allclose(s21_db, _db(palace21), atol=0.01)


# --- timing -----------------------------------------------------------------------

#: Palace's per-sample linear solve (setup, preconditioner and GMRES per
#: excitation) on driven/r8 and driven/r4 on this laptop.
PALACE_PER_SAMPLE = {"r8": 43.8, "r4": 104.6}


@pytest.mark.parametrize("name", ["r8", "r4"])
def test_driven_per_sample_time_against_palace(reference, name):
    case = reference / name
    model = _model(case)
    f = 50e9
    options = gfem.SolverOptions(threads=THREADS)
    rule = DrivenRule(model, gfem.Driven(frequency=f), options)
    with task_manager(THREADS):
        op = rule.operator(model)
        vectors = [
            terms.port_vectors(op.space, op.built, op.order, data)
            for data in op.ports  # both excitations, as Palace
        ]
        excite = np.stack([b for b, _ in vectors], axis=1)
        voltage = np.stack([v for _, v in vectors], axis=1)
        omega = 2 * math.pi * f
        start = time.perf_counter()
        handle = prepare(
            Operator(op.form(omega), op.space.FreeDofs()),
            symmetry="complex_symmetric",
            kind=rule.kind,
            tolerance=options.tolerance,
            width=2,
        )
        x = handle.solve(1j * op.wavenumber(omega) * excite)
        elapsed = time.perf_counter() - start
        iterations = list(handle.iterations)
        handle.free()
    per_sample = elapsed / 2
    s = (voltage.T @ x) / math.sqrt(2 * 50.0) - np.eye(2)
    f_palace, palace = _palace(case)
    palace_s = np.array(
        [
            [
                np.interp(f, f_palace, palace[:, i, j].real)
                + 1j * np.interp(f, f_palace, palace[:, i, j].imag)
                for j in range(2)
            ]
            for i in range(2)
        ]
    )
    ratio = per_sample / PALACE_PER_SAMPLE[name]
    print(  # noqa: T201
        f"{name}: ndof {op.space.ndof}, per sample {per_sample:.1f} s "
        f"({ratio:.2f}x Palace {PALACE_PER_SAMPLE[name]} s), GMRES {iterations}, "
        f"peak RSS {_peak_rss_gib():.2f} GiB, load {os.getloadavg()}; "
        f"order 2 at 50 GHz: |S21| {_db(s[1, 0]):.4f} dB vs Palace "
        f"{_db(palace_s[1, 0]):.4f} dB (sign of the curl term; not asserted)"
    )
    assert ratio <= 1.5


# --- adaptive sweep ---------------------------------------------------------------

#: Palace's 300-point adaptive sweep on this laptop: wall time (s) and peak
#: RSS (GiB), from the reference README.
PALACE_SWEEP = {"r8": (592.0, 1.60), "r4": (1359.0, 3.36)}
#: Frequencies (GHz) near which adaptive is checked against full solves, at
#: the nearest grid frequency the sweep did not solve.
CHECKS = (10.0, 25.0, 50.0, 66.6, 90.0)

_ADAPTIVE: dict[str, tuple] = {}


def _sweep(case: Path):
    """``(result, wall s)`` of the adaptive sweep on Palace's grid."""
    model = _model(case)
    frequency, _ = _palace(case)
    options = gfem.SolverOptions(threads=THREADS)
    problem = gfem.Driven(frequency=frequency, sweep="adaptive")
    start = time.perf_counter()
    result = gfem.solve(model, problem, options=options)
    return result, time.perf_counter() - start


def _adaptive(reference, name):
    """The adaptive sweep of a case on Palace's 300-point grid (cached).

    It runs in a fresh process, so the peak RSS is the sweep's own (as for
    Palace), not the high-water mark of the tests before it.
    """
    if name not in _ADAPTIVE:
        case = reference / name
        (result, wall), rss = isolated(_sweep, case)
        _ADAPTIVE[name] = (_model(case), *_palace(case), result, wall, rss)
    return _ADAPTIVE[name]


def _report(name, result, wall, rss):
    info = result.sweep
    errors = info.sample_errors
    confirming = next(
        (k for k, e in enumerate(reversed(errors)) if not e < info.tolerance),
        len(errors),
    )
    share = confirming / len(errors)
    print(  # noqa: T201
        f"{name} adaptive: {len(errors)} samples at "
        f"{[round(f / 1e9, 3) for f in info.sampled_frequencies]} GHz, errors "
        f"{[f'{e:.2e}' for e in errors]}, basis {info.basis_size}, converged "
        f"{info.converged}, confirmation share {share:.2f}; {wall:.1f} s, peak RSS "
        f"{rss:.2f} GiB, load {os.getloadavg()}"
    )


@pytest.mark.parametrize("name", ["r8_abc1", "r4_abc1"])
def test_adaptive_parity_s_matches_palace_over_the_grid(reference, name):
    _, frequency, palace, result, wall, rss = _adaptive(reference, name)
    _report(name, result, wall, rss)
    assert len(frequency) == 300
    assert result.sweep.converged
    diff = np.abs(np.asarray(result.s.values) - palace).max(axis=(1, 2))
    print(  # noqa: T201
        f"{name}: max |dS| vs Palace {diff.max():.3g} at "
        f"{frequency[diff.argmax()] / 1e9:.3f} GHz"
    )
    assert diff.max() <= TOLERANCE


def test_adaptive_matches_uniform_solves_between_samples(reference):
    model, frequency, _, result, wall, rss = _adaptive(reference, "r8")
    _report("r8", result, wall, rss)
    solved = np.array(result.sweep.solved)
    rows = []
    for f in CHECKS:
        nearest = np.argsort(np.abs(frequency - f * 1e9))
        rows.append(next(int(i) for i in nearest if not solved[i]))
    options = gfem.SolverOptions(threads=THREADS)
    uniform = gfem.solve(model, gfem.Driven(frequency=frequency[rows]), options=options)
    adaptive = np.asarray(result.s.values)[rows]
    diff = np.abs(np.asarray(uniform.s.values) - adaptive).max(axis=(1, 2))
    print(  # noqa: T201
        f"r8: |dS| adaptive vs uniform {np.round(diff, 9).tolist()} at "
        f"{np.round(frequency[rows] / 1e9, 3).tolist()} GHz"
    )
    assert diff.max() <= TOLERANCE


@pytest.mark.parametrize("name", ["r8", "r4"])
def test_adaptive_sweep_time_against_palace(reference, name):
    _, _, _, result, wall, rss = _adaptive(reference, name)
    _report(name, result, wall, rss)
    palace_wall, palace_rss = PALACE_SWEEP[name]
    print(  # noqa: T201
        f"{name}: 300-point sweep {wall:.1f} s = {wall / palace_wall:.3f}x Palace "
        f"({palace_wall:.0f} s); peak RSS {rss:.2f} GiB = {rss / palace_rss:.2f}x "
        f"Palace ({palace_rss} GiB)"
    )
    assert result.sweep.converged
    assert wall <= 1.5 * palace_wall
    assert rss <= 1.5 * palace_rss


# --- wave ports -------------------------------------------------------------------

#: driven/r8_wp grid rows checked by uniform solves: 1, 9.94, 50.0 and 100 GHz.
WP_ROWS = (0, 27, 148, 299)
#: Palace v0.18.1's port boundary (the baseline): the Robin term plus a
#: rank-two modal correction per wave port. Without the correction max |dS|
#: was 1e-5, 7.9e-4 and 1.5e-3 at 1, 50 and 100 GHz.


def test_wave_port_parity_s_matches_palace(reference):
    case = reference / "r8_wp"
    model = _model(case)
    assert {type(p) for p in model.ports} == {gfem.WavePort}
    frequency, palace = _palace(case)
    rows = list(WP_ROWS)
    options = gfem.SolverOptions(threads=THREADS)
    start = time.perf_counter()
    result = gfem.solve(model, gfem.Driven(frequency=frequency[rows]), options=options)
    wall = time.perf_counter() - start
    s = np.asarray(result.s.values)
    diff = np.abs(s - palace[rows]).max(axis=(1, 2))
    print(  # noqa: T201
        f"r8_wp uniform: max |dS| vs Palace {np.round(diff, 9).tolist()} at "
        f"{np.round(frequency[rows] / 1e9, 3).tolist()} GHz; n_eff "
        f"{np.round(np.asarray(result.port_modes['p1'].n_eff), 6).tolist()}; "
        f"{wall:.1f} s, peak RSS {_peak_rss_gib():.2f} GiB, load {os.getloadavg()}"
    )
    assert diff.max() <= TOLERANCE
    np.testing.assert_allclose(s, np.swapaxes(s, 1, 2), atol=1e-6)


def test_wave_port_adaptive_parity_s_matches_palace_over_the_grid(reference):
    _, frequency, palace, result, wall, rss = _adaptive(reference, "r8_wp")
    _report("r8_wp", result, wall, rss)
    diff = np.abs(np.asarray(result.s.values) - palace).max(axis=(1, 2))
    print(  # noqa: T201
        f"r8_wp adaptive: max |dS| vs Palace {diff.max():.3g} at "
        f"{frequency[diff.argmax()] / 1e9:.3f} GHz"
    )
    assert diff.max() <= TOLERANCE


def test_wave_port_adaptive_matches_uniform_solves_between_samples(reference):
    model, frequency, _, result, wall, rss = _adaptive(reference, "r8_wp")
    _report("r8_wp", result, wall, rss)
    assert result.sweep.converged
    for name, mode in result.port_modes.items():
        # recorded: 7-14 per port over repeated runs (ARPACK's random start
        # vector moves the greedy's low-frequency samples)
        exact = int(np.asarray(mode.solved).sum())
        print(f"r8_wp port {name}: {exact} exact port mode solves")  # noqa: T201
    adaptive = np.asarray(result.s.values)
    solved = np.array(result.sweep.solved)
    rows = []
    for f in CHECKS:
        nearest = np.argsort(np.abs(frequency - f * 1e9))
        rows.append(next(int(i) for i in nearest if not solved[i]))
    options = gfem.SolverOptions(threads=THREADS)
    uniform = gfem.solve(model, gfem.Driven(frequency=frequency[rows]), options=options)
    du = np.abs(np.asarray(uniform.s.values) - adaptive[rows]).max(axis=(1, 2))
    print(  # noqa: T201
        f"r8_wp: |dS| adaptive vs uniform {np.round(du, 9).tolist()} at "
        f"{np.round(frequency[rows] / 1e9, 3).tolist()} GHz"
    )
    assert du.max() <= TOLERANCE


def test_wave_port_low_frequency_backward_errors(reference):
    """Recorded, not a gate: the E-field formulation's low-frequency floor."""
    case = reference / "r8_wp"
    model = _model(case)
    frequency = np.array([1e9, 10e9, 100e9])
    options = gfem.SolverOptions(threads=THREADS, linear_solver="direct")
    rule = DrivenRule(model, gfem.Driven(frequency=frequency), options)
    with task_manager(THREADS):
        ports = rule._ports(model)
        op = ports.operator
        free = np.array(list(op.space.FreeDofs()), dtype=bool)
        for k, f in enumerate(frequency):
            omega = 2 * math.pi * f
            b = ports.system.rhs(omega)
            handle = ports.prepare(
                omega, kind="direct", device="cpu", tolerance=1e-6, width=b.shape[1]
            )
            x = handle.solve(b)
            handle.free()
            u, c = ports.update(omega)
            parts = [
                complex(piece.theta(omega)) * op.apply(j, x)
                for j, piece in enumerate(op.pieces)
            ] + [u @ (c @ (u.T @ x))]  # the modal correction
            r = (sum(parts) - b)[free]
            scale = sum(np.linalg.norm(p[free]) for p in parts) + np.linalg.norm(
                b[free]
            )
            port = [
                float(c.sweep.residual[k]) if c.sweep.solved[k] else math.nan
                for c in ports.waves
            ]
            print(  # noqa: T201
                f"r8_wp {f / 1e9:g} GHz: 3D backward error (direct) "
                f"{np.linalg.norm(r) / scale:.2e}; port mode backward error "
                f"(exact solves) {port}"
            )
