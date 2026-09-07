"""Colour maps, as data.

Here rather than in ``triton.gui`` because two things need them and only one of them
has a display: the viewer's image panels, and :func:`triton.export.spectrogram_rgb`,
which writes the spectrogram as raw pixels and must work on a machine with no Qt. A
core module that reached into the GUI for a lookup table would break the property that
the whole data path runs headless -- there is a test that would catch it.

Defined as control points rather than 256-entry tables so the same definition can
produce a numpy lookup table for an image, a pyqtgraph ``ColorMap`` for a colour bar,
and an interpolation at any resolution. One source, so a bar can never disagree with
the image beside it.

``jet`` is first because it is Triton's default and what every existing figure in the
lab's published work was made with. It is a poor perceptual choice by modern taste --
non-monotonic in lightness, which invents edges that are not in the data -- but changing
the default would make new figures incomparable with old ones, so it stays, and the
alternatives are offered beside it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MAPS", "COLORMAPS", "colormap_lut"]

MAPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "jet": [
        (0.000, (0, 0, 143)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
        (0.625, (255, 255, 0)), (0.875, (255, 0, 0)), (1.000, (128, 0, 0)),
    ],
    "grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "inverse grey": [(0.0, (255, 255, 255)), (1.0, (0, 0, 0))],
    "hot": [
        (0.0, (0, 0, 0)), (0.365, (255, 0, 0)), (0.746, (255, 255, 0)),
        (1.0, (255, 255, 255)),
    ],
    "bone": [
        (0.0, (0, 0, 0)), (0.376, (81, 81, 113)), (0.753, (166, 198, 198)),
        (1.0, (255, 255, 255)),
    ],
}

#: Menu order, jet first.
COLORMAPS: tuple[str, ...] = tuple(MAPS)


def colormap_lut(name: str, n: int = 256) -> np.ndarray:
    """An ``n`` x 3 uint8 lookup table.

    An unknown name falls back to ``jet`` rather than raising: the name can arrive from
    a settings file or an old session, and a colour map nobody recognises should not
    take the window down.
    """
    pts = MAPS.get(name) or MAPS["jet"]
    xs = np.array([p for p, _ in pts])
    cols = np.array([c for _, c in pts], dtype=float)
    grid = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for k in range(3):
        lut[:, k] = np.interp(grid, xs, cols[:, k]).round().astype(np.uint8)
    return lut
