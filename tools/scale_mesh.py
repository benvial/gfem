"""One source mesh of the scaling family: an IHP TopMetal2 GSG CPW with lumped ports.

    python scale_mesh.py OUT r0.5     # or B1, B2, or a refined size in um

Written with gsim, gmsh and the IHP PDK in the versions of `PINS`, and refused on any
others: a current gsim writes the same mesh but a different config (conductivity 3.03e7
against 3.77e7 S/m), which would move S for a reason unrelated to the mesh. So write
`PINS`, never "current gsim", in a recipe. B2 was built on gsim 0.8.0 with
`r10M_void`'s config copied over it; the scaling record carries the voided mesh itself
for that reason.

Next `scale_void.py OUT VOIDED` removes the conductors, and `split_mesh.py` makes the
larger rungs. Source: gsim's ``nbs/palace_cpw_lumped.ipynb``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from collections.abc import Callable

#: The packages that reproduce the family's mesh and config.
PINS = {"gsim": "0.7.0", "ihp-gdsfactory": "0.2.8", "gmsh": "4.15.2"}
#: The refined size in um of each case: about 3.06, 6.24 and 14.2 M DOF once voided.
SIZES = {"r0.5": 0.366, "B1": 0.125, "B2": 0.024}


def check_pins(version: Callable[[str], str] = importlib.metadata.version) -> None:
    """Refuse any versions but `PINS`.

    Args:
        version: Reads an installed package's version.

    Raises:
        SystemExit: A package is missing or at another version.
    """
    wrong = {}
    for package, pinned in PINS.items():
        try:
            found = version(package)
        except importlib.metadata.PackageNotFoundError:
            found = "not installed"
        if found != pinned:
            wrong[package] = found
    if wrong:
        raise SystemExit(
            f"scale_mesh.py builds on {PINS}, and found {wrong}: another gsim writes "
            "another config, so S would move for a reason unrelated to the mesh."
        )


def refined_size(text: str) -> float:
    """A case's name, or a size in um."""
    return SIZES[text] if text in SIZES else float(text)


def build(out: str, refined: float) -> None:
    """Build the case in ``out``: palace.msh, config.json, metadata.json."""
    import gdsfactory as gf
    from gsim.common.stack import get_stack
    from gsim.palace import DrivenSim
    from ihp import LAYER, PDK

    PDK.activate()

    @gf.cell
    def gsg_electrode(
        length=800.0,
        s_width=20.0,
        g_width=40.0,
        gap_width=15.0,
        layer=LAYER.TopMetal2drawing,
    ):
        c = gf.Component()
        lower = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        lower.move((0, (g_width + s_width) / 2 + gap_width))
        c << gf.c.rectangle((length, s_width), centered=True, layer=layer)
        upper = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        upper.move((0, -(g_width + s_width) / 2 - gap_width))
        for name, x, o in (("o1", -length / 2, 180), ("o2", length / 2, 0)):
            c.add_port(
                name=name,
                center=(x, 0),
                width=s_width,
                orientation=o,
                port_type="electrical",
                layer=layer,
            )
        return c

    sim = DrivenSim()
    sim.set_output_dir(out)
    sim.set_geometry(gsg_electrode())
    sim.set_stack(get_stack())
    sim.set_airbox(margin_x=50, margin_y=0, z_above=100, z_below=100)
    for port in ("o1", "o2"):
        sim.add_cpw_port(
            port, layer="topmetal2", s_width=20, gap_width=15, excited=True
        )
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=300)
    print(sim.validate_config())
    print(
        sim.mesh(
            preset="default",
            refined_mesh_size=refined,
            max_mesh_size=40.0,
            fmax=150e9,
            verbose=False,
        )
    )
    print(sim.write_config())


def main() -> None:
    """Check the pins, then build."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("refined", help=f"one of {sorted(SIZES)}, or a size in um")
    args = parser.parse_args()
    check_pins()
    build(args.out, refined_size(args.refined))


if __name__ == "__main__":
    main()
