"""Interactive 3D views (PyVista) for the docs build.

When ``just notebooks`` executes the tutorials for the docs, every kernel runs
``docs/_ipython/profile_default/startup/01-gfem-scenes.py``, which calls
`install`. From then on ``Plotter.show()`` exports the scene as a vtk.js
``.vtksz`` file under ``docs/_static/scenes/`` and displays, next to the plain
``image/png`` screenshot, a ``text/html`` iframe of trame-vtk's offline viewer
on that file (the scheme of PyVista's sphinx-gallery scraper). Each scene is
exported twice, on the site's light and dark grounds with black lines and text
in the theme's ink, and the two iframes are swapped by Shibuya's
``light-only`` / ``dark-only`` classes, as the figure pairs of
``gfem_figures.py``; the PNG is the light one. The notebooks are committed
with those outputs and the scenes; conf.py copies the viewer into the built
``_static/``. The notebooks themselves stay plain PyVista.

A scene is named after the source of the cell that shows it (a hash: nbconvert
gives the kernel no cell id) and its rank in that cell, so a re-run overwrites
it in place; ``just notebooks`` then runs ``tools/prune_scenes.py``, which
deletes the scenes no notebook output names any more (those of edited cells).

The same hook saves the card stills (``gfem_stills.py``): after the cell a
tutorial tags ``card`` the tutorial's script in ``docs/_stills/`` builds its
still from the kernel's namespace. With ``GFEM_STILLS_ONLY=1`` (``just
stills``) ``show()`` only closes the plotter, so earlier visible cells write no
scene and display nothing.
"""

import base64
import hashlib
import importlib.util
import io
import os
import sys
import traceback
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
SCENES = DOCS / "_static" / "scenes"
VIEWER = "static_viewer.html"  # trame_vtk.tools.vtksz2html.HTML_VIEWER_PATH
HEIGHT = 400  # px, as PyVista's offlineviewer directive

# The site's colour tokens (docs/_static/gfem.css): the ground behind a scene
# and the ink its black lines and text take.
THEMES = {
    "light": {"ground": "#f2f2f2", "ink": "#262626"},
    "dark": {"ground": "#1b1b1b", "ink": "#ececea"},
}
INKS = {"#000000", *(theme["ink"] for theme in THEMES.values())}


def _paint(plotter, theme: str) -> None:
    """Put the scene on the theme's ground, its black lines and text in its ink."""
    import pyvista as pv

    ground, ink = THEMES[theme]["ground"], THEMES[theme]["ink"]
    for renderer in plotter.renderers:
        renderer.set_background(ground)
        for actor in renderer.actors.values():
            prop = getattr(actor, "prop", None)
            if isinstance(prop, pv.Property):
                if prop.color.hex_rgb in INKS:
                    prop.color = ink
                if prop.edge_color.hex_rgb in INKS:
                    prop.edge_color = ink
            elif hasattr(actor, "GetTextProperty"):  # add_text
                actor.GetTextProperty().SetColor(pv.Color(ink).float_rgb)
    for bar in plotter.scalar_bars.values():
        for text in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            text.SetColor(pv.Color(ink).float_rgb)
    plotter.render()


def _iframe(scene: Path, theme: str, alt: str) -> str:
    # The page sits next to the notebook (cwd), so the viewer is reached from
    # there; the scene URL is resolved by the viewer, from _static/.
    viewer = Path(os.path.relpath(DOCS / "_static" / VIEWER, Path.cwd())).as_posix()
    data = scene.relative_to(DOCS / "_static").as_posix()
    return (
        f'<iframe class="gfem-scene {theme}-only" src="{viewer}?fileURL={data}" '
        f'title="{alt}" width="100%" height="{HEIGHT}" '
        'style="border: 0" loading="lazy"></iframe>'
    )


def _stills():
    """The ``gfem_stills`` module, loaded by path like this one."""
    spec = importlib.util.spec_from_file_location(
        "gfem_stills", DOCS / "_ext" / "gfem_stills.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gfem_stills"] = module  # its helpers load siblings by name
    spec.loader.exec_module(module)
    return module


def install(shell):
    """Export each shown PyVista plotter as an interactive scene in ``shell``."""
    import pyvista as pv
    from IPython.display import display

    stills = _stills()
    anchors = stills.anchors()  # sha256 of a ``card`` cell's source -> notebook stem
    cell = {"id": None, "rank": 0, "stem": None}

    def pre_run_cell(info):
        digest = hashlib.sha256(info.raw_cell.encode()).hexdigest()
        cell["id"] = digest[:12]
        cell["rank"] = 0
        cell["stem"] = anchors.get(digest)

    def post_run_cell(result):
        stem, cell["stem"] = cell["stem"], None
        if stem is None or not result.success:
            return
        try:
            stills.run_card(stem, shell.user_ns)
        except Exception:  # noqa: BLE001
            # An event callback's error is only logged: say it on stderr, and
            # `tools/stills.py` sees that the still is not new.
            sys.stderr.write(f"gfem still {stem} failed:\n{traceback.format_exc()}")

    def show(plotter, *_args, **_kwargs):
        if os.environ.get("GFEM_STILLS_ONLY") == "1":
            plotter.close()
            return
        SCENES.mkdir(parents=True, exist_ok=True)
        name = f"{cell['id']}-{cell['rank']}"
        cell["rank"] += 1
        alt = "Interactive 3D view: drag to rotate, scroll to zoom."
        iframes = []
        buf = io.BytesIO()
        for theme in THEMES:
            _paint(plotter, theme)
            scene = SCENES / f"{name}-{theme}.vtksz"
            plotter.trame.export_vtksz(scene)
            iframes.append(_iframe(scene, theme, alt))
            if theme == "light":
                plotter.screenshot(buf)
        plotter.close()
        display(
            {
                "image/png": base64.b64encode(buf.getvalue()).decode(),
                "text/html": "\n".join(iframes),
            },
            raw=True,
        )

    shell.events.register("pre_run_cell", pre_run_cell)
    shell.events.register("post_run_cell", post_run_cell)
    pv.Plotter.show = show
