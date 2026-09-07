"""The main window.

MATLAB puts the plots in one figure and the controls in a second, separate one. That
split is either the best thing about the interface or the worst, depending on whether
someone has two monitors -- so this uses a Qt dock, which is the same content in the
same groupings and can be floated off into its own window or docked beside the plots.
That is the "improve quietly" latitude: nothing moves, but the arrangement stops being
a decision made in 2005.

Everything else about the layout follows ``plot_triton.m``: panels stack vertically in
a fixed order with equal heights, only the enabled ones are shown, and when none are
enabled the window says so rather than going blank.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QAction,
    QGuiApplication,
    QImage,
    QKeySequence,
    QPainter,
)
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..playback import Player
from ..session import TritonSession
from .bridge import Dirty, SessionBridge
from .controls import ControlPanel
from .panels import (
    PANEL_ORDER,
    LtsaPanel,
    SpectraPanel,
    SpectrogramPanel,
    TimeSeriesPanel,
)

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    def __init__(self, session: TritonSession | None = None):
        super().__init__()
        self.session = session or TritonSession()
        self.bridge = SessionBridge(self.session, self)

        self.setWindowTitle("Triton")
        self.resize(1180, 820)

        self.panels = {
            "ltsa": LtsaPanel(),
            "specgram": SpectrogramPanel(),
            "timeseries": TimeSeriesPanel(),
            "spectra": SpectraPanel(),
        }
        # A splitter rather than a plain layout so an analyst can give the spectrogram
        # more room than the waveform, which is what everyone wants within a minute of
        # turning the time series on. MATLAB's equal-height subplots cannot be resized.
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        for name in PANEL_ORDER:
            self.splitter.addWidget(self.panels[name])
        #: Which panels were visible last time the layout was applied. Sizes are only
        #: redistributed when this *changes*, so a manual drag survives any repaint
        #: that happens to carry a layout flag.
        self._visible_set: tuple[str, ...] = ()

        self.empty_label = QLabel("No plot type selected")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #777; font-size: 15px;")

        centre = QWidget()
        lay = QVBoxLayout(centre)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.splitter)
        lay.addWidget(self.empty_label)
        self.setCentralWidget(centre)

        self.controls = ControlPanel(self.session, self.bridge)
        # In a scroll area, because the panel is taller than a laptop screen and the
        # lower groups were simply unreachable otherwise. Horizontal scrolling is off:
        # the panel has a sensible width and a horizontal bar would only ever appear
        # because something failed to fit, which is a layout bug to fix rather than to
        # let the user scroll around.
        scroller = QScrollArea()
        scroller.setWidget(self.controls)
        scroller.setWidgetResizable(True)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setMinimumWidth(self.controls.sizeHint().width() + 24)

        dock = QDockWidget("Controls", self)
        dock.setWidget(scroller)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.controls_dock = dock

        # The pick log: MATLAB's Pickxyz text area, as a dock rather than a third
        # window. Read-only but selectable, so a run of points can be copied out --
        # which is what it is for. Tab-separated so it pastes straight into a
        # spreadsheet.
        self.pick_log = QPlainTextEdit()
        self.pick_log.setReadOnly(True)
        self.pick_log.setMaximumBlockCount(5000)
        self.pick_log.setPlaceholderText(
            "Click a panel to record the point under the cursor here.\n"
            "Columns: time, frequency (Hz), level (dB) or counts, panel, raw file, file."
        )
        pick_widget = QWidget()
        pv = QVBoxLayout(pick_widget)
        pv.setContentsMargins(2, 2, 2, 2)
        pv.addWidget(self.pick_log)
        pick_buttons = QHBoxLayout()
        copy_all = QPushButton("Copy all")
        copy_all.clicked.connect(self._copy_picks)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.pick_log.clear)
        pick_buttons.addWidget(copy_all)
        pick_buttons.addWidget(clear)
        pick_buttons.addStretch(1)
        pv.addLayout(pick_buttons)
        self.picks_dock = QDockWidget("Picks", self)
        self.picks_dock.setWidget(pick_widget)
        # Under the controls, in the same column, rather than across the bottom of the
        # window. In the bottom dock area it spanned the full width and took its height
        # from the plots -- too prominent for something used occasionally. Splitting the
        # controls column means it only ever costs control-panel space.
        self.picks_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.picks_dock)
        self.splitDockWidget(dock, self.picks_dock, Qt.Orientation.Vertical)
        self.resizeDocks([dock, self.picks_dock], [640, 200],
                         Qt.Orientation.Vertical)
        self.picks_dock.setVisible(False)          # appears on the first pick

        #: One player for the window. Constructed eagerly but it touches no audio
        #: device until asked to play, so a machine with no sound stack is fine.
        self.player = Player()

        self.setStatusBar(QStatusBar())
        self._image_acts: dict[str, QAction] = {}
        self._build_menus()

        for panel in self.panels.values():
            panel.hovered.connect(self._on_hover)
            panel.picked.connect(self._on_pick)

        self.bridge.repaint_needed.connect(self._repaint)
        self.bridge.changed.connect(self._sync_export_menus)
        self._frame = None
        self._sync_export_menus()
        self._apply_layout()
        if self.session.audio.is_open:
            self.bridge.force()

    # ----------------------------------------------------------------------- menus

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for label, shortcut, fn in (
            ("Open audio…", QKeySequence.StandardKey.Open, self.open_audio_dialog),
            ("Open LTSA…", "Ctrl+L", self.open_ltsa_dialog),
        ):
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(fn)
            file_menu.addAction(act)
        file_menu.addSeparator()

        # Grouped as initpulldowns.m:18-48 groups them, so the muscle memory carries,
        # and greyed out until there is something to export, as there.
        self.export_menu = file_menu.addMenu("Export plotted &data")
        for label, fn in (
            ("&WAV\u2026 (counts, for measurement)",
             lambda: self._export_audio("wav")),
            ("&Normalized WAV\u2026 (for listening)",
             lambda: self._export_audio("normwav")),
            ("&x.wav\u2026 (keeps time and deployment)",
             lambda: self._export_audio("xwav")),
            ("NumPy .np&z\u2026", lambda: self._export_audio("npz")),
            ("MATLAB .&mat\u2026", lambda: self._export_audio("mat")),
        ):
            act = QAction(label, self)
            act.triggered.connect(fn)
            self.export_menu.addAction(act)

        self.savefig_menu = file_menu.addMenu("Save plot window &as")
        for label, fn in (
            ("&PNG\u2026", lambda: self._save_window("png")),
            ("&JPEG\u2026", lambda: self._save_window("jpg")),
            ("P&DF\u2026", lambda: self._save_window("pdf")),
        ):
            act = QAction(label, self)
            act.triggered.connect(fn)
            self.savefig_menu.addAction(act)

        self.saveimage_menu = file_menu.addMenu("Save as &image (raw pixels)")
        self.saveimage_menu.setToolTipsVisible(True)
        for label, kind, tip in (
            ("&Spectrogram\u2026", "specgram",
             "One pixel per time bin per frequency bin. No axes, no labels."),
            ("&LTSA\u2026", "ltsa",
             "The LTSA window at its native bin resolution."),
        ):
            act = QAction(label, self)
            act.setToolTip(tip)
            act.triggered.connect(lambda _=False, k=kind: self._save_image(k))
            self.saveimage_menu.addAction(act)
            self._image_acts[kind] = act

        meta_menu = file_menu.addMenu("Export session m&etadata")
        for label, fn in (("&JSON\u2026", lambda: self._export_metadata("json")),
                          ("MATLAB .m&at\u2026",
                           lambda: self._export_metadata("mat"))):
            act = QAction(label, self)
            act.triggered.connect(fn)
            meta_menu.addAction(act)

        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = self.menuBar().addMenu("&View")
        # Motion on the keyboard as well as the buttons: an analyst paging through a
        # deployment does it hundreds of times an hour and should not have to aim.
        for label, keys, fn in (
            ("Step forward", ("Right",), lambda: self.session.step(+1)),
            ("Step back", ("Left",), lambda: self.session.step(-1)),
            ("Start of file", ("Home",), self.controls._goto_start),
            ("End of file", ("End",), self.controls._goto_end),
        ):
            act = QAction(label, self)
            act.setShortcuts([QKeySequence(k) for k in keys])
            act.triggered.connect(fn)
            view_menu.addAction(act)
            self.addAction(act)
        view_menu.addSeparator()
        view_menu.addAction(self.controls_dock.toggleViewAction())
        view_menu.addAction(self.picks_dock.toggleViewAction())

        sound_menu = self.menuBar().addMenu("&Sound")
        for label, keys, fn in (
            ("Play", ("Space",), self.controls._play),
            ("Stop", ("Escape",), self.controls._stop),
            ("Fit speed to hearing", ("Ctrl+H",), self.controls._fit_speed),
        ):
            act = QAction(label, self)
            act.setShortcuts([QKeySequence(k) for k in keys])
            act.triggered.connect(fn)
            sound_menu.addAction(act)
            self.addAction(act)

    # ------------------------------------------------------------------- open files

    def open_audio_dialog(self) -> None:
        start = str(self.session.config.last_audio_dir or Path.cwd())
        name, _ = QFileDialog.getOpenFileName(
            self, "Open audio", start,
            "Audio (*.x.wav *.wav *.flac);;All files (*)",
        )
        if name:
            self.open_audio(name)

    #: Whether :meth:`report_error` may open a modal dialog. Tests and batch runs set
    #: this False: a modal dialog with nobody to dismiss it blocks the process
    #: **forever**, and because it blocks inside a C call the Python traceback shows
    #: only the line that opened it -- which reads like the underlying operation
    #: hanging rather than the error handler.
    modal_errors = True

    def report_error(self, title: str, message: str) -> None:
        """Tell the user something failed, without ever blocking indefinitely."""
        self.statusBar().showMessage(message, 10000)
        if self.modal_errors:
            QMessageBox.warning(self, title, message)

    def open_audio(self, path: str | Path) -> None:
        try:
            self.session.open_audio(path)
        except Exception as exc:                      # noqa: BLE001 -- shown to a user
            # A reader raises with a specific reason (a plain wav where an x.wav was
            # expected, a sample-rate disagreement, a truncated header). Those messages
            # are written to be read by people, so show them rather than a traceback.
            self.report_error("Could not open file", str(exc))
            return
        self.statusBar().showMessage(str(path), 5000)
        self.setWindowTitle(f"Triton — {Path(path).name}")

    def open_ltsa_dialog(self) -> None:
        start = str(self.session.config.last_ltsa_dir or Path.cwd())
        name, _ = QFileDialog.getOpenFileName(
            self, "Open LTSA", start, "LTSA (*.ltsa);;All files (*)"
        )
        if not name:
            return
        try:
            self.session.open_ltsa(name)
        except Exception as exc:                      # noqa: BLE001
            self.report_error("Could not open LTSA", str(exc))
            return
        self.session.view.show_ltsa = True
        self.statusBar().showMessage(str(name), 5000)

    # --------------------------------------------------------------------- painting

    def _apply_layout(self) -> None:
        v = self.session.view
        wanted = {
            "ltsa": v.show_ltsa and self.session.ltsa.is_open,
            "specgram": v.show_specgram and self.session.audio.is_open,
            "timeseries": v.show_timeseries and self.session.audio.is_open,
            "spectra": v.show_spectra and self.session.audio.is_open,
        }
        for name, panel in self.panels.items():
            panel.setVisible(wanted[name])

        # Only when the set of visible panels changes -- otherwise a repaint would
        # stomp a manual drag.
        now = tuple(n for n in PANEL_ORDER if wanted[n])
        if now != self._visible_set:
            self._visible_set = now
            self._share_equally(now)
        any_shown = any(wanted.values())
        self.splitter.setVisible(any_shown)
        # plot_triton.m:36-38 shows a logo and says "No plot type selected". Saying so
        # is the useful half.
        self.empty_label.setVisible(not any_shown)
        if not any_shown and not self.session.audio.is_open:
            self.empty_label.setText("Open a file to begin  (File → Open audio)")
        elif not any_shown:
            self.empty_label.setText("No plot type selected")

    def _share_equally(self, visible: tuple[str, ...]) -> None:
        """Split the available height equally between the visible panels.

        What ``plot_triton.m`` does -- ``subplot(m, 1, k)`` gives every panel the same
        height -- and what makes toggling predictable.

        An earlier version tried to remember each panel's height and give it back. That
        fixed one bug and caused a worse one: the panels already on screen kept their
        full heights, so there was nothing left to give a newly shown panel and it
        arrived at the 80 px floor. Reported from use as panels "starting out
        compressed". Equal shares cannot do that, and a manual drag is still safe
        because this only runs when the visible set changes.
        """
        if not visible:
            return
        total = self.splitter.height() or 600
        share = total // len(visible)
        self.splitter.setSizes(
            [share if n in visible else 0 for n in PANEL_ORDER]
        )

    def _repaint(self, flags: Dirty) -> None:
        """Honour the most expensive flag we were given, and no more."""
        if Dirty.LAYOUT in flags:
            # Before painting, not after: a panel that has just been switched on has
            # nothing in it, and one switched off should not be handed a frame.
            self._apply_layout()

        v = self.session.view
        # DATA means re-read and re-transform; DISPLAY on its own means re-map values we
        # already have. Honouring that distinction is the whole point of classifying
        # changes: a 30 s window at 200 kHz costs 242 ms to compute and under a
        # millisecond to re-map, so a brightness slider is either smooth or unusable
        # depending on this one branch.
        if self.session.audio.is_open:
            try:
                if flags & (Dirty.DATA | Dirty.AXES) or self._frame is None:
                    self._frame = self.session.frame()
                elif Dirty.DISPLAY in flags:
                    self._frame = self.session.remap(self._frame)
            except Exception as exc:                  # noqa: BLE001
                # A window can legitimately be unreadable -- too few samples for the
                # chosen nfft, a position in a duty-cycle gap. Report it and leave the
                # last good frame on screen rather than blanking the display.
                self.statusBar().showMessage(str(exc), 8000)
                return

        if self._frame is not None:
            title = self._audio_title()
            start = str(self._frame.start)
            if self.panels["specgram"].isVisible():
                self.panels["specgram"].render(
                    self._frame, colormap=v.colormap, log_freq=v.log_freq,
                    delimiters=v.show_delimiters,
                    stamp=(title, start,
                           f"Fs = {self._frame.fs:,} Hz, NFFT = {v.nfft}, "
                           f"overlap {v.overlap_pct:g}%  |  "
                           f"B = {v.brightness:g} dB, C = {v.contrast:g}%"
                           + ("  |  band-pass "
                              f"{v.filter_low:g}-{v.filter_high:g} Hz"
                              if v.filter_on else "")),
                )
            if self.panels["timeseries"].isVisible():
                self.panels["timeseries"].render(
                    self._frame, delimiters=v.show_delimiters,
                    stamp=(title if not self.panels["specgram"].isVisible() else "",
                           start, f"Fs = {self._frame.fs:,} Hz"
                           + ("  |  band-pass "
                              f"{v.filter_low:g}-{v.filter_high:g} Hz"
                              if v.filter_on else "")),
                )
            if self.panels["spectra"].isVisible():
                self.panels["spectra"].render(
                    self._frame, log_freq=v.log_freq, freq0=v.freq0, freq1=v.freq1,
                    stamp=("", f"{start}  ({v.tseg_sec:g} s averaged)",
                           f"NFFT = {v.nfft}, overlap {v.overlap_pct:g}%"),
                )

        if self.panels["ltsa"].isVisible() and self.session.ltsa.is_open:
            try:
                lt = self.session.ltsa
                h = lt.source.header
                tile = self.session.ltsa_tile()
                self.panels["ltsa"].render(
                    tile, colormap=lt.colormap,
                    stamp=(f"{lt.path}   CH={h.channel}" if lt.path else "",
                           str(tile.start),
                           f"Fs = {h.fs:,} Hz, Tave = {h.tave:g} s, NFFT = {h.nfft}  |  "
                           f"B = {lt.brightness:g} dB, C = {lt.contrast:g}%"),
                )
            except Exception as exc:                  # noqa: BLE001
                self.statusBar().showMessage(str(exc), 8000)

        self.controls._refresh_readouts()

    # ---------------------------------------------------------------------- export

    def _sync_export_menus(self, path: str = "") -> None:
        """Grey the export menus out until there is something to export.

        What ``initpulldowns.m`` does with ``Enable, off`` -- except that MATLAB
        then has to switch them back on by hand from every site that opens or closes a
        file (``control.m``'s ``menuon``/``menuoff``), and misses some. Here it follows
        the change notification, so it cannot drift.
        """
        if path and not path.startswith(("audio.", "ltsa.")):
            return
        audio = self.session.audio.is_open
        for menu in (self.export_menu, self.savefig_menu):
            menu.setEnabled(audio)
        self._image_acts["specgram"].setEnabled(audio)
        self._image_acts["ltsa"].setEnabled(self.session.ltsa.is_open)
        self.saveimage_menu.setEnabled(audio or self.session.ltsa.is_open)

    def _ask_where(self, title: str, suffix: str, filt: str) -> Path | None:
        """A save dialog, offering a timestamped name in the last-used directory.

        The default name carries the source file and the window's start time, because
        a folder of clips called ``export1.wav`` is unusable a month later and that is
        what MATLAB's ``uiputfile`` default produces.
        """
        from ..export import default_stem

        start = (self.session.config.export_dir
                 or self.session.config.last_audio_dir or Path.cwd())
        stem = default_stem(
            self.session.audio.path,
            self._frame.start if self._frame is not None
            else np.datetime64("now"),
        )
        name, _ = QFileDialog.getSaveFileName(
            self, title, str(Path(start) / f"{stem}{suffix}"), filt)
        if not name:
            return None
        out = Path(name)
        if not out.name.lower().endswith(suffix.lower()):
            out = out.with_name(out.name + suffix)
        self.session.config.export_dir = out.parent
        return out

    def _export_audio(self, kind: str) -> None:
        from .. import export

        if self._frame is None:
            self.statusBar().showMessage("nothing plotted to export", 5000)
            return
        title, suffix, filt = {
            "wav": ("Export window as WAV", ".wav", "WAV (*.wav)"),
            "normwav": ("Export window as normalized WAV", ".wav", "WAV (*.wav)"),
            "xwav": ("Export window as x.wav", ".x.wav", "x.wav (*.x.wav)"),
            "npz": ("Export window arrays", ".npz", "NumPy archive (*.npz)"),
            "mat": ("Export window arrays", ".mat", "MATLAB file (*.mat)"),
        }[kind]
        out = self._ask_where(title, suffix, filt)
        if out is None:
            return

        frame = self._frame
        channel = None if self.session.view.export_all_channels else frame.channel
        meta = export.provenance(self.session, frame)
        try:
            if kind == "wav":
                export.write_wav(out, frame, channel=channel, sidecar=meta)
            elif kind == "normwav":
                export.write_wav(out, frame, normalise=True, channel=channel,
                                 sidecar=meta)
            elif kind == "xwav":
                export.write_xwav(out, frame, self.session, channel=channel,
                                  sidecar=meta)
            elif kind == "npz":
                export.write_arrays(out, frame, self.session, channel=channel)
            else:
                export.write_mat(out, frame, self.session, channel=channel)
        except Exception as exc:                      # noqa: BLE001
            self.report_error("Could not export", str(exc))
            return
        n = 1 if channel else int(frame.samples.shape[1])
        plural = "" if n == 1 else "s"
        plural = "" if n == 1 else "s"
        self.statusBar().showMessage(
            f"wrote {out.name}  --  {n} channel{plural}, "
            f"{frame.samples.shape[0]:,} samples", 8000)

    def _save_window(self, fmt: str) -> None:
        """The plots as a picture of themselves, at better than screen resolution.

        MATLAB prints at 300 dpi (``filepd.m``). A Qt widget grab has no dpi to set, so
        PNG and JPEG come out at the window's device pixel ratio and PDF goes through
        ``QPdfWriter`` at 300 dpi -- which still carries the spectrogram as a raster,
        because a spectrogram *is* a raster. Truly vector axes would mean re-plotting
        through matplotlib; see EXPORT_PLAN §4 for why that is not worth it yet.
        """
        suffix = f".{fmt}"
        filt = {"png": "PNG image (*.png)", "jpg": "JPEG image (*.jpg)",
                "pdf": "PDF document (*.pdf)"}[fmt]
        out = self._ask_where("Save plot window", suffix, filt)
        if out is None:
            return
        target = self.splitter if self.splitter.isVisible() else self.centralWidget()
        try:
            pm = target.grab()
            if fmt == "pdf":
                from PySide6.QtGui import QPageSize, QPdfWriter

                writer = QPdfWriter(str(out))
                writer.setResolution(300)
                writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
                painter = QPainter(writer)
                try:
                    page = painter.viewport().size()
                    size = pm.size().scaled(page, Qt.AspectRatioMode.KeepAspectRatio)
                    painter.drawPixmap(0, 0, size.width(), size.height(), pm)
                finally:
                    painter.end()
            elif not pm.save(str(out)):
                raise RuntimeError(f"Qt would not write {out.suffix} here")
        except Exception as exc:                      # noqa: BLE001
            self.report_error("Could not save the plot window", str(exc))
            return
        self.statusBar().showMessage(f"wrote {out.name}", 8000)

    def _save_image(self, kind: str) -> None:
        """The data as raw pixels -- one per bin, no axes, no labels.

        Revives the menu item MATLAB creates invisible and disabled
        (``initpulldowns.m:46``, ``filepd.m:403``). Not a screenshot: at native
        resolution these pixels *are* the numbers, which is what figure assembly and
        downstream image tools want, and it is the one export whose size does not
        depend on how big the window happened to be.
        """
        from .. import export

        if kind == "ltsa":
            if not self.session.ltsa.is_open:
                self.statusBar().showMessage("no LTSA open", 5000)
                return
            tile, cmap = self.session.ltsa_tile(), self.session.ltsa.colormap
        else:
            if self._frame is None:
                self.statusBar().showMessage("nothing plotted to export", 5000)
                return
            tile, cmap = self._frame.spectrogram, self.session.view.colormap

        out = self._ask_where(f"Save {kind} image", ".png",
                              "PNG image (*.png);;JPEG image (*.jpg)")
        if out is None:
            return
        try:
            rgb = np.ascontiguousarray(export.spectrogram_rgb(tile, cmap))
            h, w, _ = rgb.shape
            # bytes(), not the array: QImage does not copy or take ownership, and a
            # view onto a temporary would be freed out from under it.
            img = QImage(bytes(rgb.data), w, h, 3 * w,
                         QImage.Format.Format_RGB888)
            if not img.save(str(out)):
                raise RuntimeError(f"Qt would not write {out.suffix} here")
        except Exception as exc:                      # noqa: BLE001
            self.report_error("Could not save the image", str(exc))
            return
        export.write_sidecar(out, export.provenance(self.session, self._frame))
        self.statusBar().showMessage(f"wrote {out.name}  --  {w} x {h} pixels", 8000)

    def _export_metadata(self, fmt: str) -> None:
        """Every setting behind what is on screen, without the data.

        No MATLAB equivalent. It is here because the provenance sidecar answers "what
        made this file" and this answers "what was I looking at" -- which is the
        question a half-finished analysis leaves behind.
        """
        from .. import export

        out = self._ask_where("Export session metadata", f".{fmt}",
                              {"json": "JSON (*.json)",
                               "mat": "MATLAB file (*.mat)"}[fmt])
        if out is None:
            return
        meta = export.provenance(self.session, self._frame)
        try:
            if fmt == "json":
                out.write_text(json.dumps(meta, indent=2, sort_keys=True,
                                          default=str),
                               encoding="utf-8")
            else:
                from scipy.io import savemat

                savemat(str(out), {"provenance": json.dumps(meta, default=str)})
        except Exception as exc:                      # noqa: BLE001
            self.report_error("Could not export metadata", str(exc))
            return
        self.statusBar().showMessage(f"wrote {out.name}", 8000)

    # ---------------------------------------------------------------------- cursor

    def _on_hover(self, kind: str, x: float, y: float) -> None:
        """Update the readout for whichever panel the pointer is over."""
        try:
            if kind == "ltsa":
                if not self.session.ltsa.is_open:
                    return
                readout = self.session.probe_ltsa(self.session.ltsa_tile(), x, y)
            else:
                if self._frame is None:
                    return
                readout = self.session.probe(self._frame, kind, x, y)
        except Exception:                             # noqa: BLE001
            # Hover fires continuously, including over a window that cannot currently
            # be read. Failing silently is right here -- an error message per mouse
            # movement would bury everything else in the status bar.
            return
        self.controls.show_readout(readout)

    def _copy_picks(self) -> None:
        QGuiApplication.clipboard().setText(self.pick_log.toPlainText())
        self.statusBar().showMessage("picks copied", 3000)

    def _log_pick(self, kind: str, x: float, y: float) -> None:
        """Append the point under the cursor to the pick log, tab-separated."""
        try:
            if kind == "ltsa":
                if not self.session.ltsa.is_open:
                    return
                r = self.session.probe_ltsa(self.session.ltsa_tile(), x, y)
            else:
                if self._frame is None:
                    return
                r = self.session.probe(self._frame, kind, x, y)
        except Exception:                             # noqa: BLE001
            return
        cols = [
            str(r.time) if r.time is not None else "",
            f"{r.frequency:.1f}" if r.frequency is not None else "",
            f"{r.value:.1f}" if r.value is not None else "",
            r.panel,
            str(r.segment + 1) if r.segment is not None else "",
            r.source_file or (self.session.audio.path.name
                              if self.session.audio.path else ""),
        ]
        self.pick_log.appendPlainText("\t".join(cols))
        if not self.picks_dock.isVisible():
            self.picks_dock.setVisible(True)

    def _on_pick(self, kind: str, x: float, y: float) -> None:
        """A click: log the point if logging is on; on the LTSA, open the audio if armed.

        Two independent modes, as in MATLAB (Pickxyz and Expand), because they serve
        different moments. Reading values off an LTSA is a browsing activity and
        should not change which file is open; jumping into the audio is a decision.
        """
        if self.controls.log_picks.isChecked():
            self._log_pick(kind, x, y)
        if kind != "ltsa" or not self.session.ltsa.is_open or not self.session.ltsa.expand:
            return
        try:
            t = self.session.open_from_ltsa(x)
        except FileNotFoundError as exc:
            # The LTSA names the file; it just is not where the LTSA is. Offer to find
            # it, which is what pickxwav.m does inline.
            self.statusBar().showMessage(str(exc), 10000)
            if not self.modal_errors:
                return
            name, _ = QFileDialog.getOpenFileName(
                self, "Locate the audio file named by this LTSA",
                str(self.session.config.last_audio_dir or Path.cwd()),
                "Audio (*.x.wav *.wav *.flac);;All files (*)",
            )
            if not name:
                return
            try:
                t = self.session.open_from_ltsa(
                    x, search_dirs=[Path(name).parent]
                )
            except Exception as exc2:                 # noqa: BLE001
                self.report_error("Could not open the audio", str(exc2))
                return
        except Exception as exc:                      # noqa: BLE001
            self.report_error("Could not open the audio", str(exc))
            return
        self.statusBar().showMessage(
            f"{self.session.audio.path.name} at {t}", 6000
        )

    def _audio_title(self) -> str:
        """Full path plus channel, as MATLAB's plot title.

        The whole path rather than the file name, deliberately: the directory is the
        deployment, and a screenshot that says which deployment it came from is worth
        the title being long.
        """
        p = self.session.audio.path
        return f"{p}   CH={self.session.view.channel}" if p else ""

    def closeEvent(self, event) -> None:              # noqa: N802 -- Qt naming
        self.player.stop()
        self.bridge.detach()
        super().closeEvent(event)
