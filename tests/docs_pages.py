"""Read the user pages' code without running it.

The drift tests of the user pages share these helpers: cut a page into
sections and fenced blocks, and check its code against the installed gfem
without solving anything (the static page check: blocks parse, gfem names
resolve, call keywords bind to the signatures).
"""

import ast
import contextlib
import dataclasses
import importlib
import inspect
import io
import json
import re
import shutil
import textwrap
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
#: The mesh every executed page downloads, and the address it downloads it from.
MESH = DOCS / "tutorials" / "data" / "line.msh"
MESH_URL = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
)
#: Every file an executed page may download (meshes, a measured table, and
#: the modulator tutorial's section builder): its address -> the repository's
#: copy.
MESHES = {
    MESH_URL: MESH,
    MESH_URL.replace("line.msh", "metagrating.msh"): MESH.with_name("metagrating.msh"),
    MESH_URL.replace("line.msh", "heater_w2.msh"): MESH.with_name("heater_w2.msh"),
    MESH_URL.replace("line.msh", "heater_straight.msh"): MESH.with_name(
        "heater_straight.msh"
    ),
    MESH_URL.replace("line.msh", "pml_line.msh"): MESH.with_name("pml_line.msh"),
    MESH_URL.replace("line.msh", "gold_mcpeak2015.csv"): MESH.with_name(
        "gold_mcpeak2015.csv"
    ),
    MESH_URL.replace("line.msh", "bend2d.msh"): MESH.with_name("bend2d.msh"),
    MESH_URL.replace("data/line.msh", "tfln_section.py"): DOCS
    / "tutorials"
    / "tfln_section.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/strip_section.py"): DOCS
    / "recipes"
    / "strip_section.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/mmi_blocks.py"): DOCS
    / "recipes"
    / "mmi_blocks.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/taper_blocks.py"): DOCS
    / "recipes"
    / "taper_blocks.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/ridge_section.py"): DOCS
    / "recipes"
    / "ridge_section.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/grating_cell.py"): DOCS
    / "recipes"
    / "grating_cell.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/slab_pair_section.py"): DOCS
    / "recipes"
    / "slab_pair_section.py",
    MESH_URL.replace("line.msh", "antenna_box.msh"): MESH.with_name("antenna_box.msh"),
    MESH_URL.replace("line.msh", "sphere.msh"): MESH.with_name("sphere.msh"),
    MESH_URL.replace("line.msh", "wr90_guide.msh"): MESH.with_name("wr90_guide.msh"),
    MESH_URL.replace("data/line.msh", "transducer_section.py"): DOCS
    / "tutorials"
    / "transducer_section.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/plate_section.py"): DOCS
    / "recipes"
    / "plate_section.py",
    MESH_URL.replace("data/line.msh", "aom_section.py"): DOCS
    / "tutorials"
    / "aom_section.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/wire_section.py"): DOCS
    / "recipes"
    / "wire_section.py",
    MESH_URL.replace("tutorials/data/line.msh", "recipes/resonator_meshes.py"): DOCS
    / "recipes"
    / "resonator_meshes.py",
}

_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>[^\n`]*)\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_ROLE = re.compile(r"\{py:[a-z]+\}`([^`]+)`")
_DOTTED = re.compile(r"(?<![\w./-])gfem(?:\.[A-Za-z_]\w*)+")
# A dotted gfem name that is a file, not a Python name (`gfem.json`).
_FILE_SUFFIXES = {"json", "npz", "msh", "py", "md", "toml", "txt", "yml", "html"}


def root_tree() -> dict[str | None, list[str]]:
    """The documents of each root toctree of ``docs/index.md``, by caption."""
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    tree: dict[str | None, list[str]] = {}
    for body in blocks(index, "{toctree}"):
        options = dict(re.findall(r"^:(\w+):[ \t]*(.*)$", body, re.MULTILINE))
        tree[options.get("caption")] = [
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.startswith(":")
        ]
    return tree


def source(doc: str) -> Path:
    """The file of a document of the site tree: Markdown or a notebook."""
    return next(
        path
        for suffix in (".md", ".ipynb")
        if (path := DOCS / f"{doc}{suffix}").is_file()
    )


def page_text(path: Path) -> str:
    """A page as Markdown: a notebook's code cells fenced as python, magics dropped."""
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return text
    parts = []
    for cell in json.loads(text)["cells"]:
        body = "".join(cell["source"])
        if cell["cell_type"] == "code":
            kept = [
                line
                for line in body.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            ]
            body = "```python\n" + "\n".join(kept) + "\n```"
        parts.append(body)
    return "\n\n".join(parts)


#: Documents of the site tree that are not user pages with code to check: the
#: release notes name removed API on purpose.
NOT_USER_PAGES = {"project/release-notes"}


def user_pages() -> list[Path]:
    """Every user page, notebooks included, read from the root site tree.

    A page is covered from the day it joins the tree. Read one with
    ``page_text``, which also reads a notebook.
    """
    return [
        source(doc)
        for docs in root_tree().values()
        for doc in docs
        if doc not in NOT_USER_PAGES
    ]


_DISPLAY = re.compile(r"^\$\$.*?\$\$[ \t]*$", re.MULTILINE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def prose_words(text: str) -> int:
    """The words a reader reads, as the docs budgets count them.

    Fenced blocks, display math, comments, headings, labels and front matter
    are left out; a role counts as its text and inline math as one word;
    tables and admonitions count.
    """
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    text = _COMMENT.sub("", _DISPLAY.sub("", _prose(text)))
    text = re.sub(r"\{[\w:-]+\}`([^`<]*)(<[^>]*>)?`", r"\1", text)
    text = re.sub(r"\$[^$\n]+\$", "X", text)
    lines = [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
        and not re.fullmatch(r"\s*\(.+\)=\s*", line)
    ]
    return sum(1 for word in " ".join(lines).split() if re.search(r"\w", word))


def title(text: str) -> str | None:
    """The page's first-level heading, or None."""
    match = re.search(r"^# (.+)$", _prose(text), re.MULTILINE)
    return match[1].strip() if match else None


def section(page: str, heading: str) -> str:
    """The text under ``## heading`` up to the next ``## ``."""
    pattern = rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, page, re.MULTILINE | re.DOTALL)
    assert match, f"no section {heading!r} on the page"
    return match[1]


def blocks(text: str, lang: str) -> list[str]:
    """The bodies of the fenced blocks of one language, dedented."""
    return [
        textwrap.dedent(m["body"])
        for m in _FENCE.finditer(text)
        if m["lang"].strip() == lang
    ]


def line_formula(frequencies) -> np.ndarray:
    """|S21| of a 20 mm, eps_r = 10 parallel-plate section between matched lines."""
    c, eps, length = 299792458.0, 10.0, 20e-3
    z1 = 376.73 / 6
    z2 = z1 / np.sqrt(eps)
    bl = 2 * np.pi * np.asarray(frequencies) * np.sqrt(eps) / c * length
    s21 = 2 / (2 * np.cos(bl) + 1j * np.sin(bl) * (z2 / z1 + z1 / z2))
    return np.abs(s21)


def toctree_entries(page: str, caption: str | None = None) -> list[str]:
    """The documents of a page's toctrees (of the one with ``caption``, if given)."""
    entries = []
    for body in blocks(page, "{toctree}"):
        options = dict(re.findall(r"^:(\w+):[ \t]*(.*)$", body, re.MULTILINE))
        if caption is not None and options.get("caption") != caption:
            continue
        entries += [
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.startswith(":")
        ]
    return entries


def doc_links(text: str) -> set[str]:
    """The targets of the ``{doc}`` roles in ``text``."""
    targets = set()
    for role in re.findall(r"\{doc\}`([^`]+)`", text):
        explicit = re.search(r"<([^>]+)>\s*$", role)
        targets.add((explicit[1] if explicit else role).strip())
    return targets


# -- Executed pages -----------------------------------------------------------

_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclasses.dataclass(frozen=True)
class Example:
    """One Python block of a page, with what the page shows and what it printed."""

    section: str
    code: str
    shown: str | None
    printed: str = ""
    start: int = 0  # where the block starts on the page


def examples(page: str) -> list[Example]:
    """The Python blocks of ``page`` in order, each with its output block.

    The output block of a Python block is the ``text`` block that follows it
    with nothing but blank lines in between.
    """
    titles = re.finditer(r"^## (.+)$", page, re.MULTILINE)
    headings = [(m.start(), m[1]) for m in titles]
    fences = list(_FENCE.finditer(page))
    found = []
    for fence, after in zip(fences, [*fences[1:], None], strict=True):
        if fence["lang"].strip() != "python":
            continue
        section = next((h for at, h in reversed(headings) if at < fence.start()), "")
        shown = None
        if (
            after is not None
            and after["lang"].strip() == "text"
            and not page[fence.end() : after.start()].strip()
        ):
            shown = textwrap.dedent(after["body"])
        found.append(
            Example(section, textwrap.dedent(fence["body"]), shown, start=fence.start())
        )
    return found


def run_page(
    path: Path,
    workdir: Path,
    *,
    skip: tuple[str, ...] = (),
    shell: Callable[[str], None] | None = None,
) -> list[Example]:
    """Run a page's Python blocks as a reader does, and record what each prints.

    The blocks run in order in one namespace, in ``workdir``, with each mesh
    download swapped for the repository's own copy (`MESHES`). The blocks of the
    sections named in ``skip`` are left out (code the test cannot run, such
    as a gmsh script). With ``shell``, each ``bash`` block is passed to it in
    its place among the Python blocks, in ``workdir``.
    """

    def urlretrieve(url, filename):
        assert url in MESHES, url
        shutil.copy(MESHES[url], Path(workdir) / filename)

    namespace = {"__name__": path.stem}
    found = [
        example
        for example in examples(path.read_text(encoding="utf-8"))
        if example.section not in skip
    ]
    commands = []
    run_shell = shell or (lambda _command: None)
    if shell is not None:
        page = path.read_text(encoding="utf-8")
        commands = [
            (m.start(), textwrap.dedent(m["body"]))
            for m in _FENCE.finditer(page)
            if m["lang"].strip() == "bash"
        ]
    ran = []
    original = urllib.request.urlretrieve
    urllib.request.urlretrieve = urlretrieve
    try:
        with contextlib.chdir(workdir):
            for example in found:
                while commands and commands[0][0] < example.start:
                    run_shell(commands.pop(0)[1])
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    exec(compile(example.code, str(path), "exec"), namespace)  # noqa: S102
                ran.append(dataclasses.replace(example, printed=out.getvalue()))
            for _, command in commands:
                run_shell(command)
    finally:
        urllib.request.urlretrieve = original
    return ran


def numbers(text: str) -> np.ndarray:
    """Every number written in ``text``, in order."""
    return np.array([float(n) for n in _NUMBER.findall(text)])


def output_problems(ran: list[Example], *, tolerance: float = 1e-5) -> list[str]:
    """Where a page's output blocks differ from what its code printed.

    A block that prints has an output block and one that does not has none;
    the words agree, and the numbers agree to ``tolerance`` (relative, and
    absolute for numbers near zero). The default is ten times the iterative
    solver's default tolerance: a driven S changes in its seventh digit with
    the ports excited together.
    """
    problems = []
    for number, example in enumerate(ran, start=1):
        where = f"python block {number} ({example.section})"
        if not example.printed.strip():
            if example.shown is not None:
                problems.append(f"{where} prints nothing but shows an output block")
            continue
        if example.shown is None:
            problems.append(f"{where} prints but the page shows no output block")
            continue
        words = [
            _NUMBER.sub("#", text).split() for text in (example.printed, example.shown)
        ]
        printed, shown = numbers(example.printed), numbers(example.shown)
        if words[0] != words[1] or not np.allclose(
            shown, printed, rtol=tolerance, atol=tolerance
        ):
            problems.append(
                f"{where} printed\n{example.printed}but the page shows\n{example.shown}"
            )
    return problems


# -- Static page check --------------------------------------------------------


class _MissingError(Exception):
    pass


def resolve(dotted: str):
    """The object a dotted name starting with ``gfem`` refers to.

    Raises `_MissingError` naming the first part that is not there. A dataclass
    field or annotated attribute without a class-level value counts as there.
    """
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for i, part in enumerate(parts[1:], start=1):
        try:
            obj = getattr(obj, part)
            continue
        except AttributeError:
            pass
        if inspect.ismodule(obj):
            try:
                obj = importlib.import_module(".".join(parts[: i + 1]))
                continue
            except ImportError:
                pass
        declared = inspect.isclass(obj) and any(
            part in getattr(base, "__annotations__", {}) for base in obj.__mro__
        )
        if declared and i == len(parts) - 1:
            return None  # a declared attribute: nothing further to inspect
        raise _MissingError(f"{dotted}: no {part!r} in {'.'.join(parts[:i])}")
    return obj


def _chain(node: ast.AST) -> list[str] | None:
    """``["a", "b", "c"]`` for the expression ``a.b.c``; None for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(parts)]
    return None


class _Code:
    """The gfem names and calls of one page's Python, read from its syntax trees."""

    def __init__(self, bare: frozenset[str] = frozenset()):
        # Local name -> dotted gfem name. `bare` names (inline code written
        # without the `gfem.` prefix) resolve in the package namespace.
        self.aliases = {"gfem": "gfem"} | {name: f"gfem.{name}" for name in bare}
        self.problems: list[str] = []

    def read_imports(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] != "gfem":
                        continue
                    self._check(alias.name)
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                if module.split(".")[0] != "gfem":
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    dotted = f"{module}.{alias.name}"
                    self._check(dotted)
                    self.aliases[alias.asname or alias.name] = dotted

    def forget_rebound(self, tree: ast.AST) -> None:
        """A name the page assigns to is the page's own, whatever was imported."""
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names = [node.id]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.FunctionDef | ast.ClassDef):
                names = [node.name]
            for name in names:
                if name != "gfem":
                    self.aliases.pop(name, None)

    def _dotted(self, node: ast.AST) -> str | None:
        chain = _chain(node)
        if chain is None or chain[0] not in self.aliases:
            return None
        return ".".join([self.aliases[chain[0]], *chain[1:]])

    def _check(self, dotted: str):
        try:
            return resolve(dotted)
        except _MissingError as missing:
            self.problems.append(str(missing))
            return None

    def check(self, tree: ast.AST, *, complete_calls: bool) -> None:
        """Resolve every gfem name of ``tree`` and bind every gfem call.

        With ``complete_calls`` a call must also give every required argument
        (a code block); without, only the arguments it shows are checked
        (inline code, which abbreviates).
        """
        inner = {
            id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and id(node) not in inner:
                dotted = self._dotted(node)
                if dotted is not None:
                    self._check(dotted)
            if isinstance(node, ast.Call):
                dotted = self._dotted(node.func)
                if dotted is None:
                    continue
                try:
                    callee = resolve(dotted)
                except _MissingError:
                    continue  # reported as a name above (or below, for a bare name)
                if callable(callee):
                    self._bind(dotted, callee, node, complete=complete_calls)

    def _bind(self, dotted: str, callee, call: ast.Call, *, complete: bool) -> None:
        try:
            signature = inspect.signature(callee)
        except (TypeError, ValueError):
            return  # no signature to read (a builtin)
        starred = any(isinstance(arg, ast.Starred) for arg in call.args)
        unpacked = any(keyword.arg is None for keyword in call.keywords)
        positional = [] if starred else [None] * len(call.args)
        keywords = {k.arg: None for k in call.keywords if k.arg is not None}
        try:
            # The arguments shown must fit; in a complete call, none is missing.
            signature.bind_partial(*positional, **keywords)
            if complete and not starred and not unpacked:
                signature.bind(*positional, **keywords)
        except TypeError as error:
            shown = ast.unparse(call)
            self.problems.append(f"{shown}: {error} (signature {dotted}{signature})")


def _public_names() -> frozenset[str]:
    import gfem

    return frozenset(gfem.__all__)


def _prose(page: str) -> str:
    """The page without its fenced blocks."""
    return _FENCE.sub("", page)


def static_problems(page: str) -> list[str]:
    """What a reader copying from ``page`` would trip on, found without solving.

    - a ``python`` block that does not parse;
    - a ``gfem.`` name (in a code block, in inline code or in a ``{py:...}``
      role) that the package does not have;
    - a call of a gfem callable with a keyword its signature does not take,
      too many positional arguments or, in a code block, a required argument
      left out.

    The blocks of a page are read in order as one program: a name imported
    from gfem in one block is known in the next.
    """
    problems = []
    code = _Code()
    for number, block in enumerate(blocks(page, "python"), start=1):
        try:
            tree = ast.parse(block)
        except SyntaxError as error:
            problems.append(f"python block {number} does not parse: {error}")
            continue
        code.read_imports(tree)
        code.forget_rebound(tree)
        code.check(tree, complete_calls=True)
    problems += code.problems

    prose = _prose(page)
    inline = _Code(bare=_public_names())
    for text in _INLINE.findall(prose):
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError:
            tree = None
        if tree is not None:
            # A bare name is only read as gfem's when it is called or dotted:
            # `solve` alone may be an English word, `solve(...)` is not.
            inline.check(tree, complete_calls=False)
        else:
            for dotted in _DOTTED.findall(text):
                inline._check(dotted)
    problems += inline.problems

    roles = _Code()
    for role in _ROLE.findall(prose):
        explicit = re.search(r"<([^>]+)>\s*$", role)
        target = (explicit[1] if explicit else role).strip().lstrip("~!")
        if target.split(".")[0] == "gfem":
            roles._check(target.removesuffix("()"))
    problems += roles.problems

    def is_file(problem: str) -> bool:
        name = problem.split(":", maxsplit=1)[0]
        return name.rsplit(".", 1)[-1] in _FILE_SUFFIXES

    return sorted({p for p in problems if not is_file(p)})
