"""The four display panels.

Deliberately dumb: a panel is handed a :class:`~triton.session.Frame` and draws it. It
never reaches into the session for data, never mutates anything, and knows nothing about
the other panels.

That is not tidiness for its own sake. MATLAB's panel routines are the opposite -- each
one reads the ``DATA`` global, applies the display filter, **writes the filtered result
back**, and decrements ``PARAMS.ch``. Three panels on screen therefore filter the data
three times and can end up showing three different channels (OPEN_DECISIONS §1.6). One
frame in, no shared state, and that class of bug cannot occur.

Stacking order is fixed at LTSA, spectrogram, time series, spectra, regardless of the
order the panels were switched on, matching ``plot_triton.m:44-70`` so that muscle
memory transfers.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QWidget

from .. import dsp
from ..session import Frame, LtsaTile

__all__ = [
    "PANEL_ORDER",
    "LtsaPanel",
    "SpectrogramPanel",
    "TimeSeriesPanel",
    "SpectraPanel",
    "colormap_lut",
    "pg_colormap",
]

#: Fixed stacking order, top to bottom -- ``plot_triton.m:44-70``.
PANEL_ORDER = ("ltsa", "specgram", "timeseries", "spectra")

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


#: Colour maps as control points, so no matplotlib dependency is needed for the one
#: map everyone actually uses. `jet` is what Triton defaults to and what every existing
#: figure in the lab's papers was made with, so it has to be here and it has to look
#: right; the others are offered because analysts asked for them over the years.
_MAPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
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


def pg_colormap(name: str) -> pg.ColorMap:
    """The same control points as :func:`colormap_lut`, as a pyqtgraph ColorMap.

    One source of truth for both: the image LUT and the colour bar beside it must agree,
    and building them from the same list is how that stays true.
    """
    pts = _MAPS.get(name) or _MAPS["jet"]
    return pg.ColorMap(
        pos=np.array([p for p, _ in pts]),
        color=np.array([(*c, 255) for _, c in pts], dtype=np.ubyte),
    )


def colormap_lut(name: str, n: int = 256) -> np.ndarray:
    """An ``n`` x 3 uint8 lookup table for one of the named maps."""
    pts = _MAPS.get(name) or _MAPS["jet"]
    xs = np.array([p for p, _ in pts])
    cols = np.array([c for _, c in pts], dtype=float)
    grid = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for k in range(3):
        lut[:, k] = np.interp(grid, xs, cols[:, k]).round().astype(np.uint8)
    return lut


class _Panel(pg.PlotWidget):
    """Shared axis setup and mouse reporting.  Not a panel on its own."""

    #: Shown as the y-axis label and used in the panel's title.
    y_label = ""
    y_units = ""
    #: Which panel this is, in the session's vocabulary.
    kind = ""

    #: ``(kind, x, y)`` in data coordinates. Hover drives the readout; click drives
    #: actions -- on the LTSA, opening the audio behind the point.
    hovered = Signal(str, float, float)
    picked = Signal(str, float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.showGrid(x=False, y=False)
        self.setMenuEnabled(False)
        self.getPlotItem().setLabel("left", self.y_label, units=self.y_units or None)
        self.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        self._boundary_lines: list[pg.InfiniteLine] = []

        self.scene().sigMouseMoved.connect(self._mouse_moved)
        self.scene().sigMouseClicked.connect(self._mouse_clicked)

        # Provenance stamps, in the row below the bottom axis. MATLAB writes the start
        # time bottom-left and the parameters bottom-right of every panel, and the
        # reason is worth keeping: a screenshot of a spectrogram ends up in a paper, a
        # thesis, or a Slack thread, and this is what makes it self-documenting --
        # the fs, nfft and colour settings that produced it travel with the picture.
        self._stamp = pg.LabelItem(justify="left", size="8pt", color="#333")
        self.getPlotItem().layout.addItem(self._stamp, 4, 1)
        self._stamp_parts: tuple[str, str, str] = ("", "", "")

    def set_stamp(self, title: str = "", left: str = "", right: str = "") -> None:
        """Title above, start time bottom-left, parameters bottom-right."""
        self._stamp_parts = (title, left, right)
        self.getPlotItem().setTitle(title or None, size="9pt", color="#222")
        # One label, two cells: a table is the only way a QGraphicsTextItem can
        # left- and right-justify on the same line.
        self._stamp.setText(
            f'<table width="100%"><tr>'
            f'<td align="left">{left}</td>'
            f'<td align="right">{right}</td>'
            f"</tr></table>"
        )

    @property
    def stamp(self) -> tuple[str, str, str]:
        """``(title, left, right)`` as last set -- for tests, since text is unreadable
        offscreen."""
        return self._stamp_parts

    def _to_data(self, scene_pos) -> tuple[float, float] | None:
        """Scene coordinates to data coordinates, or None if outside the plot."""
        vb = self.getPlotItem().getViewBox()
        if not self.getPlotItem().sceneBoundingRect().contains(scene_pos):
            return None
        p = vb.mapSceneToView(scene_pos)
        return float(p.x()), float(p.y())

    def _mouse_moved(self, pos) -> None:
        at = self._to_data(pos)
        if at is not None:
            self.hovered.emit(self.kind, at[0], at[1])

    def _mouse_clicked(self, event) -> None:
        at = self._to_data(event.scenePos())
        if at is not None:
            self.picked.emit(self.kind, at[0], at[1])

    def _draw_boundaries(self, offsets: list[float], x_max: float) -> None:
        """Red dashed lines where a raw file ends -- ``plot_specgram.m:116-122``.

        Offsets beyond the right edge are dropped rather than drawn off-screen;
        ``readseg`` deliberately extrapolates one boundary past the window
        (``readseg.m:129-131``), so the list routinely contains one that does not
        belong on this plot.
        """
        for line in self._boundary_lines:
            self.removeItem(line)
        self._boundary_lines.clear()
        pen = QPen(QColor(220, 0, 0))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        for x in offsets:
            if 0.0 < x < x_max:
                line = pg.InfiniteLine(pos=x, angle=90, pen=pen)
                self.addItem(line, ignoreBounds=True)
                self._boundary_lines.append(line)


class _ImagePanel(_Panel):
    """Common machinery for the two image panels."""

    y_label = "Frequency"
    y_units = "Hz"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.image = pg.ImageItem()
        self.addItem(self.image)
        self._lut_name: str | None = None
        # A spectrogram without a scale is a picture, not a measurement. The bar owns
        # the colour map and drives the image's lookup table, so the two cannot
        # disagree. Not interactive: the colour range is the session's to hold (issue
        # #111), so dragging the bar would fight the policy that keeps levels
        # comparable from one window to the next.
        self.colorbar = pg.ColorBarItem(
            values=(0.0, 1.0), width=14, interactive=False,
            label="dB re counts\u00b2/Hz",
        )
        self.colorbar.setImageItem(self.image, insert_in=self.getPlotItem())

    def _set_colormap(self, name: str) -> None:
        if name != self._lut_name:
            self.colorbar.setColorMap(pg_colormap(name))
            self._lut_name = name

    def _show(self, db: np.ndarray, x_max: float, f: np.ndarray,
              clim: tuple[float, float], colormap: str, log_freq: bool,
              fs: float | None = None, nfft: int | None = None) -> None:
        self._set_colormap(colormap)

        if log_freq and fs and nfft and f.size > 4:
            # An ImageItem is drawn on a uniform grid, and pyqtgraph's log mode means
            # the coordinates *are* log10 values. So the rows have to be resampled onto
            # a grid that is uniform in log10(f) before the axis is switched over --
            # simply setting log mode on a linearly spaced image asks the axis to read
            # 0 Hz as 10**0 and 100 kHz as 10**100000, which is where the 10**273 tick
            # labels came from.
            db, f = self._to_log_rows(db, f, fs, nfft)
            y0, y1 = float(np.log10(f[0])), float(np.log10(f[-1]))
        else:
            y0 = float(f[0]) if f.size else 0.0
            y1 = float(f[-1]) if f.size else 1.0
            log_freq = False

        # NaN reads as "no data here" -- a duty-cycle gap in gap-aware mode, or a
        # truncated LTSA. Left transparent rather than mapped to an end of the colour
        # scale, where it would look like real data at an extreme level.
        self.image.setImage(db, autoLevels=False)
        self.colorbar.setLevels(clim)
        self.image.setRect(QRectF(0.0, y0, x_max, max(y1 - y0, 1e-9)))
        self.getPlotItem().setLogMode(x=False, y=log_freq)
        self.setXRange(0.0, x_max, padding=0)
        self.setYRange(y0, y1, padding=0)

    @staticmethod
    def _to_log_rows(db: np.ndarray, f: np.ndarray, fs: float, nfft: int):
        """Resample rows onto a log-spaced frequency grid.  ``plot_specgram.m:96-100``.

        Uses the same sinc matrix MATLAB uses, over the band actually displayed. The
        matrix is built for the full bin count and then restricted, because its geometry
        depends on the total rather than on the crop.
        """
        n_full = nfft // 2 + 1
        m, rows = dsp.log_frequency_rows(n_full, fs, nfft)
        # The displayed band may be a crop of the full spectrum; line the matrix up with
        # the bins actually present.
        first = int(round(f[0] * nfft / fs))
        cols = slice(first, first + db.shape[0])
        m = m[:, cols]
        keep = (rows >= f[0]) & (rows <= f[-1]) & (rows > 0)
        if keep.sum() < 2:
            return db, f
        return m[keep] @ db, rows[keep]


class SpectrogramPanel(_ImagePanel):
    """The main event.  Port of ``plot_specgram.m``."""

    kind = "specgram"

    def render(self, frame: Frame, colormap: str = "jet", log_freq: bool = False,
               delimiters: bool = True, stamp: tuple[str, str, str] | None = None
               ) -> None:
        if stamp:
            self.set_stamp(*stamp)
        tile = frame.spectrogram
        x_max = float(frame.samples.shape[0]) / frame.fs
        nfft = int(round(frame.fs / (tile.f[1] - tile.f[0]))) if tile.f.size > 1 else 0
        self._show(tile.db, x_max, tile.f, tile.clim, colormap, log_freq,
                   fs=frame.fs, nfft=nfft)
        self.getPlotItem().setLabel("bottom", "Time", units="s")
        self._draw_boundaries(
            [b.offset_sec for b in tile.boundaries] if delimiters else [], x_max
        )


class LtsaPanel(_ImagePanel):
    """The long-term view.  Port of ``plot_ltsa.m``."""

    kind = "ltsa"

    def render(self, tile: LtsaTile, colormap: str = "jet",
               stamp: tuple[str, str, str] | None = None) -> None:
        if stamp:
            self.set_stamp(*stamp)
        x_max = float(tile.t[-1]) if tile.t.size else 1.0
        self._show(tile.db, x_max, tile.f, tile.clim, colormap, log_freq=False)
        self.getPlotItem().setLabel("bottom", "Time", units="hr")


class TimeSeriesPanel(_Panel):
    """Raw waveform.  Port of ``plot_timeseries.m``."""

    kind = "timeseries"

    y_label = "Amplitude"
    y_units = "counts"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.curve = self.plot(pen=pg.mkPen(QColor(0, 0, 160), width=1))

    def render(self, frame: Frame, delimiters: bool = True,
               stamp: tuple[str, str, str] | None = None) -> None:
        if stamp:
            self.set_stamp(*stamp)
        x = frame.time_axis
        y = frame.samples[:, frame.channel - 1]
        # A full window at 200 kHz is millions of points and Qt will draw every one.
        # autoDownsample plus clipToView keeps a pan responsive; peak mode is the right
        # choice because it preserves transients, which are the whole point of looking
        # at a waveform in this field.
        self.curve.setData(x, y, autoDownsample=True, downsampleMethod="peak",
                           clipToView=True)
        self.getPlotItem().setLabel("bottom", "Time", units="s")
        if x.size:
            self.setXRange(0.0, float(x[-1]), padding=0)
        self._draw_boundaries(
            [b.offset_sec for b in frame.boundaries] if delimiters else [],
            float(x[-1]) if x.size else 1.0,
        )


class SpectraPanel(_Panel):
    """Averaged spectrum for the window.  Port of ``plot_spectra.m``.

    Not a mean of the spectrogram: ``plot_spectra.m:32-38`` removes the mean and then
    calls ``pwelch``, which is a different estimate. The session computes it separately
    on the same samples.
    """

    kind = "spectra"

    y_label = "Spectrum level"
    y_units = "dB"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.curve = self.plot(pen=pg.mkPen(QColor(0, 110, 0), width=1))
        self.getPlotItem().setLabel("bottom", "Frequency", units="Hz")

    def render(self, frame: Frame, log_freq: bool = False,
               freq0: float = 0.0, freq1: float | None = None,
               stamp: tuple[str, str, str] | None = None) -> None:
        if stamp:
            self.set_stamp(*stamp)
        f, db = frame.spectra_f, frame.spectra_db
        if f.size == 0:
            self.curve.setData([], [])
            return
        hi = freq1 if freq1 is not None else float(f[-1])
        keep = (f >= freq0) & (f <= hi)
        # Log mode cannot show DC; dropping the zero bin is what MATLAB's log frequency
        # axis effectively does too, and keeping it would collapse the whole axis.
        if log_freq:
            keep &= f > 0
        self.curve.setData(f[keep], db[keep])
        self.getPlotItem().setLogMode(x=log_freq, y=False)
        if keep.any():
            self.setXRange(float(f[keep][0]), float(f[keep][-1]), padding=0)
            finite = db[keep][np.isfinite(db[keep])]
            if finite.size:
                pad = max(1.0, 0.05 * (finite.max() - finite.min()))
                self.setYRange(float(finite.min()) - pad, float(finite.max()) + pad,
                               padding=0)
