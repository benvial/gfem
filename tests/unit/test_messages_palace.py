"""gfem's runtime messages never name Palace (ADR 0080).

Exception, warning and log messages are user-facing but not in the docs
build, so the build's Palace check does not see them. This test scans every
string literal in the code of ``src/gfem``, private modules included: a
message is often built away from its ``raise`` or ``warn`` (``msg = ...``, a
``reason`` passed down, ``err = Error(message)``). Docstrings, bare string
statements and comments are not runtime text and are skipped; private modules
may name Palace there.
"""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "gfem"
PALACE = re.compile("palace", re.IGNORECASE)


def runtime_strings(source):
    """(line, text) of every string literal in ``source`` that is not a docstring."""
    tree = ast.parse(source)
    docs = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docs
    ]


def palace_strings(source):
    return [(line, s) for line, s in runtime_strings(source) if PALACE.search(s)]


@pytest.mark.parametrize(
    "source",
    [
        'raise ValueError("as in Palace")',
        'raise ValueError(f"{x} as in palace")',
        'import warnings\nwarnings.warn("PALACE form", UserWarning)',
        'def f():\n    msg = "Palace form"\n    raise ValueError(msg)',
        'err = Error(message="the Palace sign")',
    ],
)
def test_a_message_naming_palace_is_found(source):
    assert len(palace_strings(source)) == 1


def test_docstrings_and_comments_are_not_messages():
    source = (
        '"""Palace\'s form."""\n'
        "def f():\n"
        '    """As in Palace."""\n'
        "    x = 1  # Palace's sign\n"
        '    """Attribute note: Palace."""\n'
        "    return x\n"
    )
    assert palace_strings(source) == []


def test_no_runtime_message_names_palace():
    found = [
        f"{path.relative_to(SRC)}:{line}: {text!r}"
        for path in sorted(SRC.rglob("*.py"))
        for line, text in palace_strings(path.read_text(encoding="utf-8"))
    ]
    assert found == []
