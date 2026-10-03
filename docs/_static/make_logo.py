# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Draw the quaxed logo: quax's duck, wearing a tick.

quax's duck is a black rubber duck; quaxed's carries a white badge with a blue
tick on its body, for libraries that come already quaxified. The shapes are
measured from the original 48 px icon and drawn in its 48-unit grid, so a larger
image is the same picture, only sharper. Re-run it for a larger image::

    uv run docs/_static/make_logo.py                    # favicon.png, 512 px
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch, Polygon
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


def smooth(points: list[tuple[float, float]], n: int = 400) -> np.ndarray:
    """Return a closed Catmull-Rom curve through ``points``."""
    p = np.asarray(points, dtype=float)
    k = len(p)
    t = np.linspace(0, k, n, endpoint=False)
    i = t.astype(int)
    s = (t - i)[:, None]
    p0, p1, p2, p3 = (p[(i + d) % k] for d in (-1, 0, 1, 2))
    return 0.5 * (
        2 * p1
        + (p2 - p0) * s
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s**2
        + (3 * p1 - p0 - 3 * p2 + p3) * s**3
    )


def duck(ax: plt.Axes) -> None:
    """Draw quax's duck: body, beak and head in black, the eye a hole."""
    body = smooth(BODY)
    ax.add_patch(Polygon(body, color=BLACK, lw=0))
    ax.add_patch(Polygon(smooth(BEAK), color=BLACK, lw=0))
    # The head with the eye cut out, so it shows whatever the logo sits on.
    t = np.linspace(0, 2 * np.pi, 200)
    (hx, hy), hr = HEAD
    (ex, ey), er = EYE
    head = np.column_stack([hx + hr * np.cos(t), hy + hr * np.sin(t)])
    eye = np.column_stack([ex + er * np.cos(t), ey - er * np.sin(t)])  # reversed
    codes = np.full(400, MplPath.LINETO)
    codes[[0, 200]] = MplPath.MOVETO
    path = MplPath(np.concatenate([head, eye]), codes)
    ax.add_patch(PathPatch(path, color=BLACK, lw=0))


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
        default=Path(__file__).with_name("favicon.png"),
        help="output file (default: favicon.png)",
    )
    parser.add_argument("--size", type=int, default=512, help="pixels per side")
    args = parser.parse_args()

    dpi = 100
    fig = plt.figure(figsize=(args.size / dpi, args.size / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 48), ylim=(48, 0), aspect="equal")  # y down, as the icon's
    ax.axis("off")
    duck(ax)
    badge(ax)
    fig.savefig(args.out, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
