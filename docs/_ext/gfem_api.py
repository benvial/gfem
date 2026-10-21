"""Sphinx extension for the gfem API reference (sphinx-autoapi).

Every public name is documented once, on its home module page, and references
resolve under every spelling users write:

- ``autoapi-skip-member`` skips the copies of a name that are not on its home
  page (``gfem.Model`` is documented on ``gfem.model`` only).
- ``missing-reference`` resolves ``gfem.X``, a bare ``X`` and
  ``gfem._private.X`` to the documented ``gfem.<home>.X``.
- ``env-before-read-docs`` rewrites private annotation types to their public
  home and drops the ones that stay private.
- A post-transform points ``gfem.solve`` at the function, not the module of the
  same name (only ``:py:mod:`` keeps the module).

The home map is computed statically with ``ast`` from the first entry of
``autoapi_dirs``, so the build needs no gfem install.
"""

import ast
import re
from pathlib import Path

from sphinx.addnodes import pending_xref
from sphinx.transforms.post_transforms import SphinxPostTransform

# Filled on config-inited from autoapi_dirs.
_API_MODULES: set[str] = set()  # "gfem" and its public modules
_API_HOMES: dict[str, str] = {}  # public name -> module page that documents it
_SHADOWED: dict[str, str] = {}  # module -> the function of the same name it exports

_PRIVATE_PATH = re.compile(r"gfem(?:\._\w+)+\.(\w+)")
_PRIVATE_NAME = re.compile(r"(?<![\w])_[A-Za-z]\w*|\._\w")


def _parse_module(path: Path) -> tuple[set[str], list[str]]:
    """Return the names a module defines and the names it exports.

    A module without ``__all__`` exports its defined names that do not start
    with ``_``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined: set[str] = set()
    exported: list[str] | None = None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__all__" and node.value is not None:
                    exported = list(ast.literal_eval(node.value))
                else:
                    defined.add(target.id)
    if exported is None:
        exported = sorted(name for name in defined if not name.startswith("_"))
    return defined, exported


def public_homes(pkg: Path) -> dict[str, str]:
    """Map each public gfem name to the one module page that documents it.

    The home is the public module that defines the name; for a name defined in
    a private module, the first public submodule whose ``__all__`` lists it;
    otherwise the top-level ``gfem`` page (for example ``cuda_available``).
    """
    modules = {"gfem": pkg / "__init__.py"}
    modules |= {f"gfem.{p.stem}": p for p in sorted(pkg.glob("[!_]*.py"))}
    info = {module: _parse_module(path) for module, path in modules.items()}
    homes: dict[str, str] = {}
    for module, (defined, exported) in info.items():  # 1. defined in a public module
        for name in exported:
            if name in defined and module != "gfem":
                homes[name] = module
    for module, (_, exported) in info.items():  # 2. re-exported from a private module
        if module == "gfem":
            continue
        for name in exported:
            if name not in homes and f"gfem.{name}" not in modules:
                homes[name] = module
    for name in info["gfem"][1]:  # 3. only the top-level namespace has it
        if f"gfem.{name}" not in modules and not name.startswith("__"):
            homes.setdefault(name, "gfem")
    return homes


def _load_homes(app, config) -> None:
    pkg = (Path(app.confdir) / config.autoapi_dirs[0]).resolve()
    _API_MODULES.clear()
    _API_MODULES.update({"gfem"} | {f"gfem.{p.stem}" for p in pkg.glob("[!_]*.py")})
    _API_HOMES.clear()
    _API_HOMES.update(public_homes(pkg))
    _SHADOWED.clear()
    for module in _API_MODULES:
        short = module.rpartition(".")[2]
        if _API_HOMES.get(short) == module:
            _SHADOWED[module] = f"{module}.{short}"


def _skip_duplicates(app, what, name, obj, skip, options):  # noqa: ARG001, PLR0917
    """Skip every copy of a public name that is not on its home page.

    The signature is the ``autoapi-skip-member`` event's.
    """
    if what in ("module", "package"):
        return None
    parent, _, short = name.rpartition(".")
    if parent in _API_MODULES and _API_HOMES.get(short, parent) != parent:
        return True
    return None


def _resolve_alias(app, env, node, contnode):
    """Resolve ``gfem.X``, bare ``X`` and ``gfem._private.X`` to its home."""
    if node.get("refdomain") != "py":
        return None
    parts = node["reftarget"].lstrip(".~").split(".")
    for i, part in enumerate(parts):
        prefix_hidden = all(p == "gfem" or p.startswith("_") for p in parts[:i])
        if part in _API_HOMES and prefix_hidden:
            home = _API_HOMES[part]
            if i == 0 and len(parts) == 1 and node.get("py:module") == home:
                return None  # already in its home module: the lookup failed otherwise
            target = ".".join([home, *parts[i:]])
            if target == node["reftarget"]:
                return None
            domain = env.get_domain("py")
            return domain.resolve_xref(
                env, node["refdoc"], app.builder, "obj", target, node, contnode
            )
    return None


def _hide_private_types(app, env, docnames):  # noqa: ARG001
    """Rewrite ``gfem._x.Name`` annotations to the public home; drop private ones.

    Private types only annotate constructor parameters users do not pass
    (``Mesh``'s topology, ``PortModeField``'s NGSolve-side space). The parameter
    name stays in the signature.
    """

    def public(match):
        name = match[1]
        return f"{_API_HOMES[name]}.{name}" if name in _API_HOMES else match[0]

    for fields in getattr(env, "autoapi_annotations", {}).values():
        for key, annotation in list(fields.items()):
            rewritten = _PRIVATE_PATH.sub(public, annotation)
            if _PRIVATE_NAME.search(rewritten):
                del fields[key]
            else:
                fields[key] = rewritten


class _PreferShadowingFunction(SphinxPostTransform):
    """Point ``gfem.solve`` at the function; only ``:py:mod:`` keeps the module."""

    default_priority = 5  # before ReferencesResolver (10)

    def run(self, **kwargs) -> None:  # noqa: ARG002
        for node in self.document.findall(pending_xref):
            target = node.get("reftarget", "").lstrip("~")
            if (
                node.get("refdomain") == "py"
                and node.get("reftype") != "mod"
                and target in _SHADOWED
            ):
                node["reftarget"] = node["reftarget"].replace(target, _SHADOWED[target])


def setup(app):
    """Register the gfem API hooks."""
    app.connect("config-inited", _load_homes)
    app.add_post_transform(_PreferShadowingFunction)
    app.connect("env-before-read-docs", _hide_private_types)
    app.connect("autoapi-skip-member", _skip_duplicates)
    app.connect("missing-reference", _resolve_alias)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
