from pathlib import Path

import licence_gate as gate
import pytest


def _install(site: Path, name: str, *headers: str) -> None:
    """Write a minimal installed distribution (``*.dist-info/METADATA``)."""
    info = site / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True)
    lines = ["Metadata-Version: 2.4", f"Name: {name}", "Version: 1.0", *headers]
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _violations(site: Path, policy: gate.Policy | None = None) -> dict[str, str]:
    report = gate.audit([str(site)], policy or gate.Policy())
    return {entry.name: entry.verdict for entry in report if not entry.ok}


@pytest.mark.parametrize(
    "headers",
    [
        ["License-Expression: MIT"],
        ["License-Expression: LGPL-2.1-only"],
        ["License-Expression: GPL-3.0-or-later"],
        ["License-Expression: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0"],
        ["License-Expression: Apache-2.0 OR BSD-2-Clause"],
        ["License-Expression: GPL-2.0-or-later WITH Classpath-exception-2.0"],
        ["License: Apache-2.0"],
        ["Classifier: License :: OSI Approved :: BSD License"],
        [
            "License: Copyright (c) 2001 Enthought, Inc. All rights reserved.",
            "Classifier: License :: OSI Approved :: BSD License",
        ],
        [
            "Classifier: License :: OSI Approved :: LGPLv2+",
        ],
    ],
)
def test_free_licences_pass_including_gpl(tmp_path, headers):
    _install(tmp_path, "pkg", *headers)
    assert _violations(tmp_path) == {}


@pytest.mark.parametrize(
    "headers",
    [
        ["License-Expression: CC-BY-NC-4.0"],
        ["License-Expression: MIT AND CC-BY-NC-SA-4.0"],
        ["Classifier: License :: Free for non-commercial use"],
        ["License: Free for academic and non-commercial use"],
        ["License-Expression: LicenseRef-Proprietary"],
        ["Classifier: License :: Other/Proprietary License"],
    ],
)
def test_non_free_and_non_commercial_licences_fail(tmp_path, headers):
    _install(tmp_path, "pkg", *headers)
    assert _violations(tmp_path) == {"pkg": "non-free"}


def test_a_free_alternative_in_an_or_expression_passes(tmp_path):
    _install(tmp_path, "pkg", "License-Expression: CC-BY-NC-4.0 OR MIT")
    assert _violations(tmp_path) == {}


def test_missing_licence_metadata_fails_until_reviewed(tmp_path):
    _install(tmp_path, "cuda-toolkit")
    assert _violations(tmp_path) == {"cuda-toolkit": "unknown"}

    reviewed = gate.Policy(overrides={"cuda-toolkit": "MIT"})
    assert _violations(tmp_path, reviewed) == {}


def test_proprietary_licences_pass_only_for_allowed_distributions(tmp_path):
    _install(
        tmp_path,
        "nvidia-cublas-cu12",
        "License-Expression: LicenseRef-NVIDIA-Proprietary",
    )
    _install(tmp_path, "other-lib", "License-Expression: LicenseRef-NVIDIA-Proprietary")
    policy = gate.Policy(allow_proprietary=("nvidia-*",))
    assert _violations(tmp_path, policy) == {"other-lib": "non-free"}


def test_allowed_proprietary_distributions_still_fail_on_non_commercial(tmp_path):
    _install(tmp_path, "nvidia-thing", "License-Expression: CC-BY-NC-4.0")
    policy = gate.Policy(allow_proprietary=("nvidia-*",))
    assert _violations(tmp_path, policy) == {"nvidia-thing": "non-free"}


def test_repository_policy_accepts_the_development_environment():
    report = gate.audit(None, gate.load_policy())
    assert [e for e in report if not e.ok] == []
    assert any(e.name == "ngsolve" for e in report)


def _wheel(tmp_path: Path, filename: str, *members: str) -> Path:
    import zipfile

    path = tmp_path / filename
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "")
    return path


GOOD = ("gfem/__init__.py", "gfem-0.1.0.dist-info/licenses/LICENSE")


def test_a_pure_python_wheel_with_its_licence_passes(tmp_path):
    assert (
        gate.check_wheel(_wheel(tmp_path, "gfem-0.1.0-py3-none-any.whl", *GOOD)) == []
    )


@pytest.mark.parametrize(
    ("filename", "members", "problem"),
    [
        ("gfem-0.1.0-cp312-cp312-linux_x86_64.whl", GOOD, "py3-none-any"),
        (
            "gfem-0.1.0-py3-none-any.whl",
            (*GOOD, "gfem/_kernel.cpython-312.so"),
            "native",
        ),
        ("gfem-0.1.0-py3-none-any.whl", (*GOOD, "gfem/kernel.pyd"), "native"),
        ("gfem-0.1.0-py3-none-any.whl", ("gfem/__init__.py",), "LICENSE"),
    ],
)
def test_wheel_problems_are_reported(tmp_path, filename, members, problem):
    problems = gate.check_wheel(_wheel(tmp_path, filename, *members))
    assert len(problems) == 1
    assert problem in problems[0]


def test_cli_fails_on_a_non_free_dependency(tmp_path, capsys):
    _install(tmp_path, "pkg", "License-Expression: CC-BY-NC-4.0")
    assert gate.main(["--path", str(tmp_path)]) == 1
    assert "FAIL pkg 1.0: non-free" in capsys.readouterr().out
