# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Draw the quaxed logo: quax's duck, wearing a tick.

quax's duck is a black rubber duck; quaxed's carries a white badge with a blue
tick on its body, for libraries that come already quaxified. The shapes are
measured from the original 48 px icon and drawn in its 48-unit grid, so the
picture is the same at any size. The docs use it as an SVG; for a bitmap, name a
.png and give its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
import io
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath

BLACK, WHITE, BLUE = "#000000", "#ffffff", "#1473f0"

# The duck in the original icon's 48-unit grid, x right and y down. The body's
# outline, clockwise: up the neck into the head, along the back to the tail's
# tip, down the breast, along the bottom and up the front.
BODY = [
    (12.7, 19.5), (14, 16), (27.5, 16), (27.2, 20), (28.6, 22.4), (32, 22.9),
    (35, 24), (38, 25.2), (41, 24.8), (43.8, 23.6), (43.6, 27), (42.8, 30),
    (41.5, 33.5), (39.5, 36.5), (37, 39), (33.5, 41.3), (28, 42), (20, 42),
    (14.5, 41.3), (11.5, 39.5), (9.3, 37), (8.3, 33.5), (8.4, 29.5), (9.4, 26.5),
    (10.8, 24.6), (12.2, 22.2),
]  # fmt: skip
HEAD = ((20.3, 12.7), 9.1)  # centre, radius
EYE = ((17.1, 12.6), 2.6)
BEAK = [(12, 12.6), (7, 12.9), (4.6, 13.1), (4.3, 14.5), (5, 16.3), (6.5, 17.8),
        (9.5, 19.6), (12.5, 20.2)]  # fmt: skip
BADGE = ((19.6, 31.2), 8.4)  # the white disc on the body
TICK = [(15.1, 30.8), (18.3, 34.3), (24.3, 27.8)]


def smooth(points: list[tuple[float, float]]) -> MplPath:
    """Return a closed Catmull-Rom curve through ``points``, as Bezier curves.

    Each span of the curve is exactly one cubic Bezier, which SVG stores as is,
    so the outline is smooth at any size and costs four points a span.
    """
    p = np.asarray(points, dtype=float)
    p0, p2, p3 = np.roll(p, 1, 0), np.roll(p, -1, 0), np.roll(p, -2, 0)
    c1, c2 = p + (p2 - p0) / 6, p2 - (p3 - p) / 6
    verts = np.concatenate(
        [p[:1], np.stack([c1, c2, p2], axis=1).reshape(-1, 2), p[:1]]
    )
    codes = [MplPath.MOVETO, *[MplPath.CURVE4] * (3 * len(p)), MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def disc(centre: tuple[float, float], radius: float, *, hole: bool = False) -> MplPath:
    """Return a circle as Bezier curves, wound the other way for a ``hole``."""
    unit = MplPath.unit_circle()
    verts = unit.vertices[:-1] * radius + centre
    if hole:  # reversed, so it cuts out of a shape it is combined with
        verts = verts[::-1]
    return MplPath(np.concatenate([verts, verts[:1]]), unit.codes)


def duck(ax: plt.Axes) -> None:
    """Draw quax's duck: body, beak and head in black, the eye a hole."""
    ax.add_patch(PathPatch(smooth(BODY), color=BLACK, lw=0))
    ax.add_patch(PathPatch(smooth(BEAK), color=BLACK, lw=0))
    # The eye is cut out of the head, so it shows whatever the logo sits on.
    head = MplPath.make_compound_path(disc(*HEAD), disc(*EYE, hole=True))
    ax.add_patch(PathPatch(head, color=BLACK, lw=0))


def badge(ax: plt.Axes) -> None:
    """Draw the white badge on the body, with its blue tick."""
    centre, radius = BADGE
    ax.add_patch(Circle(centre, radius, color=WHITE, lw=0))
    points_per_unit = ax.figure.get_figwidth() * 72 / 48  # the axes span 48
    ax.plot(
        *np.transpose(TICK),
        color=BLUE,
        lw=2.6 * points_per_unit,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="pixels per side, for a PNG"
    )
    args = parser.parse_args()

    dpi = 100
    fig = plt.figure(figsize=(args.size / dpi, args.size / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 48), ylim=(48, 0), aspect="equal")  # y down, as the icon's
    ax.axis("off")
    duck(ax)
    badge(ax)
    # No timestamp and fixed element ids, so a re-run gives the same file.
    mpl.rcParams["svg.hashsalt"] = "logo"
    if args.out.suffix == ".svg":
        svg = io.StringIO()
        fig.savefig(svg, format="svg", transparent=True, metadata={"Date": None})
        # matplotlib ends path lines with a space, which pre-commit would strip.
        lines = svg.getvalue().splitlines()
        args.out.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    else:
        fig.savefig(args.out, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
