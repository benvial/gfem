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
it in place; ``just notebooks`` clears the directory first, which drops the
scenes of edited cells.
"""

import base64
import hashlib
import io
import os
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


def install(shell):
    """Export each shown PyVista plotter as an interactive scene in ``shell``."""
    import pyvista as pv
    from IPython.display import display

    cell = {"id": None, "rank": 0}

    def pre_run_cell(info):
        cell["id"] = hashlib.sha256(info.raw_cell.encode()).hexdigest()[:12]
        cell["rank"] = 0

    def show(plotter, *_args, **_kwargs):
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
    pv.Plotter.show = show
