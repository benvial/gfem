"""Docs build only: draw figures in the gfem style, as light/dark pairs.

IPython runs this file in every kernel started with ``IPYTHONDIR`` pointing
at ``docs/_ipython``, which ``docs/conf.py`` sets for the notebook execution.
A plain ``jupyter`` or ``nbconvert`` run never sees it, so the tutorials stay
plain matplotlib. The work is in ``docs/_ext/gfem_figures.py``.
"""


def _install_gfem_figures():
    import importlib.util
    import sys
    import traceback
    from pathlib import Path

    shell = get_ipython()  # noqa: F821 (an IPython builtin)
    try:
        docs = Path(shell.profile_dir.location).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "gfem_figures", docs / "_ext" / "gfem_figures.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install(shell)
    except Exception:  # noqa: BLE001
        # IPython only logs a failing startup file. Say so on every cell's
        # stderr instead, which the docs build turns into a warning.
        error = traceback.format_exc()
        shell.events.register(
            "pre_run_cell",
            lambda _info: sys.stderr.write(f"gfem figure hook failed:\n{error}"),
        )


_install_gfem_figures()
del _install_gfem_figures
