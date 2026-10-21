"""Licence gate: audit an installed dependency tree and gfem's own wheel.

Policy: GPL and LGPL dependencies are allowed; non-free and
non-commercial licences are not. Proprietary licences pass only where
``licences.toml`` allows them: any proprietary licence for the NVIDIA
runtime distributions reached through the ``cuda12`` / ``cuda13`` extras,
and the Intel Simplified Software License alone for the distributions of
the ``mkl`` extra. A distribution whose licence cannot be
read from its metadata fails until a human records it in ``licences.toml``.

Run it with the interpreter of the environment to audit, for example::

    .venv-gate/bin/python tools/licence_gate.py --wheel dist/gfem-*.whl

It uses the standard library only, so the audited environment needs nothing
beyond what it is meant to contain.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import re
import sys
import tomllib
import zipfile
from dataclasses import dataclass, field
from enum import IntEnum
from importlib.metadata import distributions
from pathlib import Path

DEFAULT_POLICY = Path(__file__).with_name("licences.toml")


class Kind(IntEnum):
    """Licence category, ordered from most to least permissive."""

    FREE = 0
    PROPRIETARY = 1
    UNKNOWN = 2
    NON_COMMERCIAL = 3


# SPDX identifiers (lower case) or prefixes of free-software licence families,
# copyleft included. Anything not listed is UNKNOWN and needs a review.
_FREE_IDS = {
    "0bsd",
    "afl-3.0",
    "artistic-2.0",
    "blueoak-1.0.0",
    "bsl-1.0",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "cnri-python",
    "curl",
    "hpnd",
    "isc",
    "libpng-2.0",
    "mit",
    "ms-pl",
    "ncsa",
    "ofl-1.1",
    "psf-2.0",
    "python-2.0",
    "unlicense",
    "upl-1.0",
    "w3c",
    "x11",
    "zlib",
    "zpl-2.1",
}
_FREE_PREFIXES = (
    "agpl-",
    "apache-",
    "bsd-",
    "cecill-",
    "epl-",
    "eupl-",
    "gpl-",
    "lgpl-",
    "mit-",
    "mpl-",
    "unicode-",
)
_NON_COMMERCIAL = re.compile(
    r"non[- ]?commercial|-nc\b|-nc-|academic|educational|home use", re.IGNORECASE
)
_PROPRIETARY = re.compile(r"proprietary|freeware|restricted|eula", re.IGNORECASE)

_FREE_CLASSIFIERS = (
    "License :: OSI Approved",
    "License :: Public Domain",
    "License :: DFSG approved",
    "License :: CC0 1.0 Universal",
)
_FREE_ALIASES = {
    "apache 2.0",
    "apache license 2.0",
    "apache license, version 2.0",
    "apache software license",
    "bsd",
    "bsd license",
    "mit license",
    "new bsd license",
    "the mit license",
    "the mit license (mit)",
}


def _kind_of_text(text: str) -> Kind:
    """Classify a licence name that is not an SPDX expression."""
    if _NON_COMMERCIAL.search(text):
        return Kind.NON_COMMERCIAL
    if _PROPRIETARY.search(text):
        return Kind.PROPRIETARY
    if text.strip().lower() in _FREE_ALIASES:
        return Kind.FREE
    return Kind.UNKNOWN


def _kind_of_id(spdx_id: str) -> Kind:
    """Classify one SPDX licence identifier."""
    lowered = spdx_id.lower().removesuffix("+")
    if _NON_COMMERCIAL.search(lowered):
        return Kind.NON_COMMERCIAL
    if lowered in _FREE_IDS or lowered.startswith(_FREE_PREFIXES):
        return Kind.FREE
    if lowered.startswith("licenseref-"):
        return _kind_of_text(lowered)
    return Kind.UNKNOWN


_TOKEN = re.compile(r"\s*(\(|\)|[A-Za-z0-9.+:-]+)")


def spdx_kind(expression: str) -> Kind | None:
    """Classify an SPDX expression; None when it does not parse as one.

    ``OR`` takes the most permissive alternative, ``AND`` the least, and
    ``WITH`` keeps the licence and drops the exception.
    """
    tokens: list[str] = []
    pos = 0
    text = expression.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            return None
        tokens.append(match.group(1))
        pos = match.end()
    if not tokens:
        return None

    def parse(i: int, operator: str) -> tuple[Kind, int]:
        # operator precedence: WITH > AND > OR
        if operator == "OR":
            kind, i = parse(i, "AND")
            while i < len(tokens) and tokens[i].upper() == "OR":
                other, i = parse(i + 1, "AND")
                kind = min(kind, other)
            return kind, i
        if operator == "AND":
            kind, i = parse(i, "WITH")
            while i < len(tokens) and tokens[i].upper() == "AND":
                other, i = parse(i + 1, "WITH")
                kind = max(kind, other)
            return kind, i
        if i >= len(tokens):
            raise ValueError
        if tokens[i] == "(":
            kind, i = parse(i + 1, "OR")
            if i >= len(tokens) or tokens[i] != ")":
                raise ValueError
            i += 1
        elif tokens[i] == ")" or tokens[i].upper() in {"AND", "OR", "WITH"}:
            raise ValueError
        else:
            kind, i = _kind_of_id(tokens[i]), i + 1
        if i < len(tokens) and tokens[i].upper() == "WITH":
            i += 2
        return kind, i

    try:
        kind, end = parse(0, "OR")
    except ValueError:
        return None
    return kind if end == len(tokens) else None


def _expression_kind(expression: str) -> Kind:
    """Classify an SPDX expression; UNKNOWN when it does not parse."""
    kind = spdx_kind(expression)
    return Kind.UNKNOWN if kind is None else kind


def _classifier_kind(classifiers: list[str]) -> Kind | None:
    """Classify the ``License ::`` trove classifiers; None when there are none."""
    kinds = []
    for classifier in classifiers:
        if not classifier.startswith("License ::"):
            continue
        if classifier.startswith(_FREE_CLASSIFIERS):
            kinds.append(Kind.FREE)
        else:
            kinds.append(_kind_of_text(classifier))
    return min(kinds) if kinds else None


def licence_kind(
    expression: str | None, text: str | None, classifiers: list[str]
) -> Kind:
    """Classify a distribution from its core metadata licence fields.

    ``License-Expression`` (PEP 639) wins when present. Otherwise a short
    ``License`` field and the trove classifiers are both read, and the least
    permissive of the readable ones counts.
    """
    if expression:
        return _expression_kind(expression)
    known = []
    if text and "\n" not in text.strip() and len(text) <= 100:
        kind = spdx_kind(text)
        if kind in {None, Kind.UNKNOWN}:
            kind = _kind_of_text(text)
        if kind is not Kind.UNKNOWN:
            known.append(kind)
    kind = _classifier_kind(classifiers)
    if kind is not None and kind is not Kind.UNKNOWN:
        known.append(kind)
    return max(known) if known else Kind.UNKNOWN


def normalize(name: str) -> str:
    """Normalize a distribution name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Policy:
    """Reviewed exceptions to the metadata-only classification.

    Attributes:
        overrides: distribution name to the SPDX expression a human read from
            its actual licence, for distributions with missing or unreadable
            metadata.
        allow_proprietary: ``fnmatch`` patterns of distributions allowed to
            carry a proprietary (but never a non-commercial) licence.
        allow_licences: a proprietary licence, by the name its metadata
            gives, to the ``fnmatch`` patterns of distributions allowed to
            carry that licence and no other proprietary one.
    """

    overrides: dict[str, str] = field(default_factory=dict)
    allow_proprietary: tuple[str, ...] = ()
    allow_licences: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def allows(self, name: str, licence: str | None) -> bool:
        """Whether distribution ``name`` may carry proprietary ``licence``."""
        patterns = [*self.allow_proprietary]
        if licence:
            for allowed, names in self.allow_licences.items():
                if allowed.casefold() == licence.strip().casefold():
                    patterns += names
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def load_policy(path: Path = DEFAULT_POLICY) -> Policy:
    """Read the policy from ``licences.toml``."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    overrides = {
        normalize(k): v["licence"] for k, v in data.get("overrides", {}).items()
    }
    licences = {
        k: tuple(v["distributions"]) for k, v in data.get("allow-licence", {}).items()
    }
    return Policy(
        overrides=overrides,
        allow_proprietary=tuple(data.get("allow-proprietary", [])),
        allow_licences=licences,
    )


def _describe(expression: str | None, text: str | None, classifiers: list[str]) -> str:
    """One-line licence description for the report."""
    if expression:
        return expression
    parts = [
        c.removeprefix("License :: ") for c in classifiers if c.startswith("License ::")
    ]
    lines = (text or "").strip().splitlines()
    if lines:
        parts.insert(0, lines[0][:60])
    return "; ".join(parts) or "-"


@dataclass(frozen=True)
class Entry:
    """Audit result for one installed distribution."""

    name: str
    version: str
    licence: str
    verdict: str
    ok: bool


def audit(paths: list[str] | None, policy: Policy) -> list[Entry]:
    """Audit every distribution installed on ``paths`` (default: ``sys.path``)."""
    found = distributions(path=paths) if paths is not None else distributions()
    entries: dict[str, Entry] = {}
    for dist in found:
        meta = dist.metadata
        if not meta.get("Name"):
            continue
        name = normalize(meta["Name"])
        if name in entries:
            continue
        classifiers = meta.get_all("Classifier") or []
        named = meta.get("License-Expression") or meta.get("License")
        if name in policy.overrides:
            licence = named = policy.overrides[name]
            kind = _expression_kind(licence)
            licence += " (reviewed)"
        else:
            expression, text = meta.get("License-Expression"), meta.get("License")
            kind = licence_kind(expression, text, classifiers)
            licence = _describe(expression, text, classifiers)
        allowed = policy.allows(name, named)
        if kind is Kind.FREE:
            verdict, ok = "free", True
        elif kind is Kind.PROPRIETARY and allowed:
            verdict, ok = "proprietary (allowed)", True
        elif kind is Kind.UNKNOWN:
            verdict, ok = "unknown", False
        else:
            verdict, ok = "non-free", False
        entries[name] = Entry(name, meta.get("Version") or "?", licence, verdict, ok)
    return sorted(entries.values(), key=lambda e: e.name)


_NATIVE = re.compile(r"\.(so|pyd|dylib|dll)(\.[0-9.]+)?$", re.IGNORECASE)


def check_wheel(wheel: Path) -> list[str]:
    """Return the problems with gfem's own wheel (an empty list when it is fine)."""
    problems = []
    if not wheel.name.endswith("-py3-none-any.whl"):
        problems.append(f"{wheel.name}: wheel tag is not py3-none-any")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        problems += [
            f"{wheel.name}: native file {n}" for n in names if _NATIVE.search(n)
        ]
        if not any(
            re.fullmatch(r"[^/]+\.dist-info/licenses/LICENSE", n) for n in names
        ):
            problems.append(f"{wheel.name}: no LICENSE in .dist-info/licenses/")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the gate; return 1 when any check fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--wheel",
        type=Path,
        action="append",
        default=[],
        help="also check this gfem wheel",
    )
    parser.add_argument(
        "--path", action="append", help="site directory to audit (default: sys.path)"
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args(argv)
    # Licence texts can hold characters a Windows cp1252 pipe cannot encode.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")

    problems = [p for wheel in args.wheel for p in check_wheel(wheel)]
    report = audit(args.path, load_policy(args.policy))
    width = max((len(e.name) for e in report), default=10)
    for e in report:
        flag = "  " if e.ok else "!!"
        print(f"{flag} {e.name:<{width}} {e.version:<14} {e.verdict:<22} {e.licence}")
    problems += [
        f"{e.name} {e.version}: {e.verdict} licence ({e.licence}); "
        + (
            "record the reviewed licence in licences.toml"
            if e.verdict == "unknown"
            else "non-free and non-commercial licences are not allowed"
        )
        for e in report
        if not e.ok
    ]
    print(f"\n{len(report)} distributions audited on Python {sys.version.split()[0]}")
    for problem in problems:
        print(f"FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
