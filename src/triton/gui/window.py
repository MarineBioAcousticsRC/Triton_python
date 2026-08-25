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

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

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
        dock = QDockWidget("Controls", self)
        dock.setWidget(self.controls)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.controls_dock = dock

        self.setStatusBar(QStatusBar())
        self._build_menus()

        self.bridge.repaint_needed.connect(self._repaint)
        self._frame = None
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
        any_shown = any(wanted.values())
        self.splitter.setVisible(any_shown)
        # plot_triton.m:36-38 shows a logo and says "No plot type selected". Saying so
        # is the useful half.
        self.empty_label.setVisible(not any_shown)
        if not any_shown and not self.session.audio.is_open:
            self.empty_label.setText("Open a file to begin  (File → Open audio)")
        elif not any_shown:
            self.empty_label.setText("No plot type selected")

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
            if self.panels["specgram"].isVisible():
                self.panels["specgram"].render(
                    self._frame, colormap=v.colormap, log_freq=v.log_freq,
                    delimiters=v.show_delimiters,
                )
            if self.panels["timeseries"].isVisible():
                self.panels["timeseries"].render(
                    self._frame, delimiters=v.show_delimiters
                )
            if self.panels["spectra"].isVisible():
                self.panels["spectra"].render(
                    self._frame, log_freq=v.log_freq, freq0=v.freq0, freq1=v.freq1
                )

        if self.panels["ltsa"].isVisible() and self.session.ltsa.is_open:
            try:
                self.panels["ltsa"].render(self.session.ltsa_tile(), colormap=v.colormap)
            except Exception as exc:                  # noqa: BLE001
                self.statusBar().showMessage(str(exc), 8000)

        self.controls._refresh_readouts()

    def closeEvent(self, event) -> None:              # noqa: N802 -- Qt naming
        self.bridge.detach()
        super().closeEvent(event)
