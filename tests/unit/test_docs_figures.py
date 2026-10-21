"""The docs build's themed-figure hook (docs/_ext/gfem_figures.py).

Needs matplotlib, which only the ``docs`` dependency group installs, so the
module is skipped in the default environment. Run it with
``uv run --python 3.13 --group docs pytest tests/unit/test_docs_figures.py``.
"""

import base64
import importlib.util
import io
import re
from pathlib import Path

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOCS = Path(__file__).resolve().parents[2] / "docs"
INK_DARK = (0xE6, 0xE8, 0xEB)
IMG = re.compile(r'<img class="(light|dark)-only" src="data:image/png;base64,([^"]+)"')


@pytest.fixture(scope="module")
def gfem_figures():
    spec = importlib.util.spec_from_file_location(
        "gfem_figures", DOCS / "_ext" / "gfem_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with mpl.rc_context():
        module.apply_style()
        yield module


@pytest.fixture
def fig():
    figure, ax = plt.subplots(figsize=(3, 2))
    yield figure, ax
    plt.close(figure)


def renders(html):
    """The light and dark renders in ``html``, as RGBA arrays in [0, 255]."""
    found = dict(IMG.findall(html))
    return {
        theme: mpimg.imread(io.BytesIO(base64.b64decode(data)), format="png") * 255
        for theme, data in found.items()
    }


def has_colour(image, rgb, tol=3):
    opaque = image[..., 3] > 250
    return bool(np.any(np.all(np.abs(image[..., :3] - rgb) <= tol, axis=-1) & opaque))


def test_one_figure_gives_a_light_and_a_dark_render(gfem_figures, fig):
    figure, ax = fig
    ax.plot([0, 1], [0, 1])

    html = gfem_figures.figure_html(figure)

    assert html.count('class="light-only"') == 1
    assert html.count('class="dark-only"') == 1
    assert set(renders(html)) == {"light", "dark"}


def test_black_lines_turn_ink_dark_in_the_dark_render_only(gfem_figures, fig):
    figure, ax = fig
    ax.plot([0, 1], [0.5, 0.5], color="k", lw=12)

    images = renders(gfem_figures.figure_html(figure))

    assert has_colour(images["dark"], INK_DARK)
    assert not has_colour(images["light"], INK_DARK)
    assert has_colour(images["light"], (0, 0, 0))


def test_the_figure_is_restored_after_rendering(gfem_figures, fig):
    figure, ax = fig
    (line,) = ax.plot([0, 1], [0, 1], color="k")
    label = ax.set_xlabel("x")

    gfem_figures.figure_html(figure)

    assert mpl.colors.to_hex(line.get_color()) == "#000000"
    assert mpl.colors.to_hex(label.get_color()) == "#3a3f46"  # ink-2


def test_a_colour_outside_the_table_is_reported_on_stderr(gfem_figures, fig, capsys):
    figure, ax = fig
    ax.plot([0, 1], [0, 1], color="#123456")

    gfem_figures.figure_html(figure)

    assert "#123456" in capsys.readouterr().err


def test_the_gfem_figure_hook_reports_nothing_for_palette_colours(
    gfem_figures, fig, capsys
):
    figure, ax = fig
    for k in range(6):
        ax.plot([0, 1], [k, k], label=f"C{k}")
    ax.axvline(0.5, color="gray", ls="--")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("t")

    gfem_figures.figure_html(figure)

    assert capsys.readouterr().err == ""


def test_greyscale_density_maps_stay_unchanged(gfem_figures, fig, capsys):
    figure, ax = fig
    ax.imshow(np.ones((4, 4)), cmap="gray", vmin=0, vmax=1)
    ax.set_axis_off()

    images = renders(gfem_figures.figure_html(figure))

    assert has_colour(images["dark"], (255, 255, 255))
    assert capsys.readouterr().err == ""


def test_the_sensitivity_ramp_swaps_to_its_dark_colormap(gfem_figures, fig):
    figure, ax = fig
    ax.imshow(np.zeros((4, 4)), cmap="gfem_sensitivity", vmin=0, vmax=1)
    ax.set_axis_off()

    images = renders(gfem_figures.figure_html(figure))

    assert has_colour(images["light"], (0xFF, 0xFF, 0xFF))  # ground
    assert has_colour(images["dark"], (0x11, 0x13, 0x17))  # ground-dark


def test_figures_stay_paired_after_pyplot_resets_the_formatters(gfem_figures, fig):
    from IPython.core.interactiveshell import InteractiveShell

    figure, _ = fig
    shell = InteractiveShell.instance()
    try:
        gfem_figures.install(shell)
        # Importing pyplot in a kernel selects the inline backend's figure
        # formats, which drops every Figure formatter, ours included.
        shell.run_cell(
            "from IPython.core.pylabtools import select_figure_formats\n"
            "select_figure_formats(get_ipython(), {'png'})"
        )

        data, _ = shell.display_formatter.format(figure)
    finally:
        InteractiveShell.clear_instance()

    assert 'class="dark-only"' in data["text/html"]


def test_axes_holding_an_image_draw_no_grid(gfem_figures, fig):
    figure, ax = fig
    ax.imshow(np.zeros((4, 4)), cmap="gray", vmin=0, vmax=1)
    assert any(line.get_visible() for line in ax.get_xgridlines())  # the style's grid

    gfem_figures.figure_html(figure)

    gridlines = [*ax.get_xgridlines(), *ax.get_ygridlines()]
    assert not any(line.get_visible() for line in gridlines)


def test_axes_without_an_image_keep_their_grid(gfem_figures, fig):
    figure, ax = fig
    ax.plot([0, 1], [0, 1])

    gfem_figures.figure_html(figure)

    assert any(line.get_visible() for line in ax.get_xgridlines())
