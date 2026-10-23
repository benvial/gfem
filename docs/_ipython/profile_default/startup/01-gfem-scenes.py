"""Docs notebooks only: show PyVista plotters as interactive vtk.js scenes.

IPython runs this file in every kernel started with ``IPYTHONDIR`` pointing
at ``docs/_ipython``, which ``just notebooks`` sets when it executes the
tutorials for the docs. A plain ``jupyter`` or ``nbconvert`` run never sees
it, so the tutorial code stays plain PyVista. The work is in
``docs/_ext/gfem_scenes.py``.
"""


def _install_gfem_scenes():
    import importlib.util
    import sys
    import traceback
    from pathlib import Path

    shell = get_ipython()  # noqa: F821 (an IPython builtin)
    try:
        docs = Path(shell.profile_dir.location).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "gfem_scenes", docs / "_ext" / "gfem_scenes.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install(shell)
    except Exception:  # noqa: BLE001
        # IPython only logs a failing startup file. Say so on every cell's
        # stderr instead: it lands in the committed outputs, and the docs
        # build turns it into a warning.
        error = traceback.format_exc()
        shell.events.register(
            "pre_run_cell",
            lambda _info: sys.stderr.write(f"gfem scene hook failed:\n{error}"),
        )


_install_gfem_scenes()
del _install_gfem_scenes
