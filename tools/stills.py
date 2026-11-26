"""Save the card stills of the physics sections: docs/_static/cards/*.

Usage: python tools/stills.py [name ...]    (``just stills [name ...]``)

A name is a tutorial stem or a standalone still (``waveguide-nonlinearity``);
none means every still. For a tutorial the notebook is read, cut after its
``card`` cell and executed in memory with the docs IPython profile and
``GFEM_STILLS_ONLY=1``: the startup hook (docs/_ext/gfem_scenes.py) runs
``docs/_stills/<stem>.py`` after that cell and writes the pair, and shows no
view, so no scene file changes. The notebook is never written. A script no
tutorial anchors runs directly. Only ``docs/_static/cards/`` changes.
"""

import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

DOCS = Path(__file__).resolve().parents[1] / "docs"
sys.path.insert(0, str(DOCS / "_ext"))
import gfem_stills  # noqa: E402  (a sibling of the docs build, not a package)

TIMEOUT = 3600  # s per cell: the longest tutorial cell before an anchor is minutes


def anchored() -> dict[str, int]:
    """The index of the ``card`` cell of each tutorial that has one, by stem."""
    found = {}
    for path in sorted(gfem_stills.TUTORIALS.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if gfem_stills.TAG in cell.metadata.get("tags", ()):
                found[path.stem] = index
    return found


def standalone(anchors: dict[str, int]) -> list[str]:
    """The scripts of ``docs/_stills`` that belong to no tutorial.

    A script named after a tutorial without a ``card`` cell is not standalone
    (it would be called with no namespace): `main` reports it.
    """
    tutorials = {path.stem for path in gfem_stills.TUTORIALS.glob("*.ipynb")}
    scripts = sorted(path.stem for path in gfem_stills.SCRIPTS.glob("*.py"))
    return [stem for stem in scripts if stem not in anchors and stem not in tutorials]


def unanchored(anchors: dict[str, int]) -> list[str]:
    """The tutorials that have a card script but no ``card`` cell."""
    scripts = {path.stem for path in gfem_stills.SCRIPTS.glob("*.py")}
    tutorials = {path.stem for path in gfem_stills.TUTORIALS.glob("*.ipynb")}
    return sorted((scripts & tutorials) - set(anchors))


def run_notebook(stem: str, index: int) -> None:
    """Execute tutorial ``stem`` up to and including its card cell, in memory."""
    notebook = nbformat.read(gfem_stills.TUTORIALS / f"{stem}.ipynb", as_version=4)
    notebook.cells = notebook.cells[: index + 1]
    NotebookClient(
        notebook,
        timeout=TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(gfem_stills.TUTORIALS)}},
    ).execute()
    # The hook reports a failed card script on stderr after the card cell ran.
    for output in notebook.cells[index].get("outputs", ()):
        if output.get("name") == "stderr" and "gfem still" in output.get("text", ""):
            print(output["text"])


def main(names: list[str]) -> int:
    """Save the named stills (all of them with no name); return the exit status."""
    anchors = anchored()
    alone = standalone(anchors)
    known = [*anchors, *alone]
    orphans = unanchored(anchors)
    for stem in orphans:
        print(f"tutorial {stem} has no card cell")
    unknown = [name for name in names if name not in known]
    if unknown:
        print(f"no such still: {', '.join(unknown)}; known: {', '.join(known)}")
        return 2
    chosen = names or known
    os.environ["IPYTHONDIR"] = str(DOCS / "_ipython")
    os.environ["GFEM_STILLS_ONLY"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    failed = []
    for stem in chosen:
        sidecar = gfem_stills.CARDS / f"{stem}.json"
        before = sidecar.stat().st_mtime_ns if sidecar.exists() else 0
        tic = time.perf_counter()
        try:
            if stem in anchors:
                run_notebook(stem, anchors[stem])
            else:
                gfem_stills.run_card(stem, None)
        except Exception as exc:  # noqa: BLE001  one broken tutorial must not stop the rest
            print(f"{stem}: FAILED, {type(exc).__name__}: {exc}")
            failed.append(stem)
            continue
        # A card script's error inside a kernel is only printed there: the
        # still is not saved if its sidecar is not new.
        fresh = sidecar.exists() and sidecar.stat().st_mtime_ns > before
        seconds = time.perf_counter() - tic
        print(f"{stem}: {'saved' if fresh else 'NOT saved'} in {seconds:.0f} s")
        if not fresh:
            failed.append(stem)
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 1 if failed or (orphans and not names) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
