"""The control panel.

MATLAB builds this in ``initcontrol.m`` -- 2,076 lines of hand-placed widgets -- and
handles it in ``control.m``, 976 lines of ``elseif strcmp(action, ...)`` covering 48
actions. About a quarter of those actions are pure ``Enable``/``Disable`` bookkeeping
(``buttoff``, ``menuon``, ``timeon``, ``ampon``, ``freqon``, ``logon`` and their twins),
which exists only because nothing announces when state changes.

Here a control is *bound* to a session field: edits flow one way, notifications flow
back the other, and neither side needs to know about the widgets on the far side of the
panel. That is what makes this file short.

The one genuinely fiddly part is loop prevention. A control writes to the session, the
session notifies, the bridge signals, and the control would write to itself again --
which for a spin box means fighting the user's cursor. :class:`_Binding` guards both
directions with a re-entrancy flag rather than by disconnecting signals, because
disconnecting loses events that arrive while disconnected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..session import _TIME_FORMS, TritonSession, ValidationError
from .bridge import SessionBridge

__all__ = ["ControlPanel"]


class _Binding(QObject):
    """Two-way link between one widget and one session field."""

    def __init__(self, session: TritonSession, path: str, widget: QWidget,
                 getter: Callable[[], Any], setter: Callable[[Any], None],
                 signal: Any):
        super().__init__(widget)
        self.session, self.path = session, path
        self.get, self.set = getter, setter
        self._busy = False
        signal.connect(self._widget_changed)
        self.refresh()

    def _target(self) -> tuple[Any, str]:
        group, _, field = self.path.partition(".")
        return getattr(self.session, group), field

    def _widget_changed(self, *_: object) -> None:
        if self._busy:
            return
        obj, field = self._target()
        self._busy = True
        refused = False
        try:
            setattr(obj, field, self.get())
        except ValidationError:
            refused = True
        finally:
            self._busy = False
        if refused:
            # Put the widget back rather than leaving the two out of step. Done after
            # clearing _busy, because refresh() is itself guarded by that flag and
            # would otherwise return without doing anything. Validation lives in one
            # place and the UI defers to it; duplicating the rules in widget ranges is
            # how the two drift apart.
            self.refresh()

    def refresh(self) -> None:
        if self._busy:
            return
        obj, field = self._target()
        value = getattr(obj, field, None)
        if value is None:
            return
        self._busy = True
        try:
            self.set(value)
        finally:
            self._busy = False


class ControlPanel(QWidget):
    """Everything that writes to the session, grouped as MATLAB groups it."""

    def __init__(self, session: TritonSession, bridge: SessionBridge,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.bridge = bridge
        self._bindings: dict[str, list[_Binding]] = {}

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.addWidget(self._cursor_group())
        root.addWidget(self._time_group())
        root.addWidget(self._display_group())
        root.addWidget(self._spectrogram_group())
        root.addWidget(self._colour_group())
        root.addWidget(self._filter_group())
        root.addWidget(self._ltsa_group())
        root.addWidget(self._sound_group())
        root.addStretch(1)

        bridge.changed.connect(self._on_change)

    # ------------------------------------------------------------------- plumbing

    def _bind(self, path: str, widget: QWidget, getter, setter, signal) -> None:
        self._bindings.setdefault(path, []).append(
            _Binding(self.session, path, widget, getter, setter, signal)
        )

    def _on_change(self, path: str) -> None:
        for b in self._bindings.get(path, ()):
            b.refresh()
        if path in ("audio.source", "view.tseg_sec", "audio.segment", "audio.offset"):
            self._refresh_readouts()
        if path.startswith("ltsa."):
            self._refresh_ltsa_position()

    def _spin(self, path: str, lo: float, hi: float, step: float = 1.0,
              decimals: int = 0, suffix: str = "") -> QWidget:
        w: QSpinBox | QDoubleSpinBox
        if decimals:
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
        else:
            w = QSpinBox()
        w.setRange(lo, hi)          # type: ignore[arg-type]
        w.setSingleStep(step)       # type: ignore[arg-type]
        w.setKeyboardTracking(False)   # don't fire on every keystroke mid-typing
        if suffix:
            w.setSuffix(f" {suffix}")
        self._bind(path, w, w.value, w.setValue, w.valueChanged)
        return w

    def _check(self, path: str, label: str) -> QCheckBox:
        w = QCheckBox(label)
        self._bind(path, w, w.isChecked, w.setChecked, w.toggled)
        return w

    # --------------------------------------------------------------------- groups

    def _cursor_group(self) -> QGroupBox:
        """Where the cursor readout lands.  ``coorddisp.m`` puts it in this window too.

        A fixed set of rows rather than rows added and removed per panel type: the
        readout changes as the pointer crosses between panels, and a control panel that
        reflows on mouse-move is unusable. Irrelevant rows are blanked, not hidden.
        """
        box = QGroupBox("Cursor")
        form = QFormLayout(box)
        self.cursor_rows: dict[str, QLabel] = {}
        for key in ("Time", "Frequency", "Spectrum level [dB]", "Counts", "Raw file",
                    "File"):
            lab = QLabel("--")
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(key, lab)
            self.cursor_rows[key] = lab
        return box

    def show_readout(self, readout) -> None:
        """Display a :class:`~triton.session.Readout`, blanking what does not apply."""
        values = dict(readout.lines())
        for key, lab in self.cursor_rows.items():
            lab.setText(values.get(key, "--"))

    def _time_group(self) -> QGroupBox:
        box = QGroupBox("Time")
        form = QFormLayout(box)

        self.time_label = QLabel("--")
        self.time_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Position", self.time_label)
        self.position_label = QLabel("--")
        form.addRow("Sample", self.position_label)

        # Go-to-time. Without this the only way to reach a specific moment is to step
        # to it, which is useless when the time came from a detection log or a paper.
        goto = QHBoxLayout()
        goto.setContentsMargins(0, 0, 0, 0)
        self.goto_edit = QLineEdit()
        self.goto_edit.setPlaceholderText("time, +secs, or @secs")
        # From the session rather than repeated here, so the tooltip and the
        # error message cannot drift apart.
        self.goto_edit.setToolTip(
            "Jump to a time. Any of:\n" + _TIME_FORMS + "\n\nEnter to go."
        )
        self.goto_edit.returnPressed.connect(self._goto_typed)
        go = QPushButton("Go")
        go.setMaximumWidth(40)
        go.clicked.connect(self._goto_typed)
        goto.addWidget(self.goto_edit)
        goto.addWidget(go)
        holder = QWidget()
        holder.setLayout(goto)
        form.addRow("Go to", holder)

        form.addRow("Segment length", self._spin("view.tseg_sec", 0.001, 86400.0,
                                                 step=0.5, decimals=3, suffix="s"))
        # -1 keeps PARAMS.tseg.step's convention: step by the window's own length.
        step = self._spin("view.step_sec", -1.0, 86400.0, step=0.5, decimals=3,
                          suffix="s")
        step.setToolTip("-1 means step by the segment length")
        form.addRow("Step", step)

        motion = QHBoxLayout()
        for label, tip, fn in (
            ("|◀", "start of file", lambda: self._goto_start()),
            ("◀◀", "back one step", lambda: self.session.step(-1)),
            ("▶▶", "forward one step", lambda: self.session.step(+1)),
            ("▶|", "end of file", lambda: self._goto_end()),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMaximumWidth(48)
            b.clicked.connect(fn)
            motion.addWidget(b)
        holder = QWidget()
        holder.setLayout(motion)
        form.addRow("", holder)
        return box

    def _display_group(self) -> QGroupBox:
        box = QGroupBox("Panels")
        grid = QGridLayout(box)
        # Listed in the order they are stacked, not the order MATLAB lists them in its
        # control window, so the panel list reads like the screen.
        for row, (path, label) in enumerate((
            ("view.show_ltsa", "LTSA"),
            ("view.show_specgram", "Spectrogram"),
            ("view.show_timeseries", "Time series"),
            ("view.show_spectra", "Spectra"),
        )):
            grid.addWidget(self._check(path, label), row, 0)
        grid.addWidget(self._check("view.show_delimiters", "Raw-file delimiters"), 4, 0)
        return box

    def _spectrogram_group(self) -> QGroupBox:
        box = QGroupBox("Spectrogram")
        form = QFormLayout(box)
        form.addRow("FFT length", self._spin("view.nfft", 8, 1 << 20, step=1))
        form.addRow("Overlap", self._spin("view.overlap_pct", 0, 99, step=5,
                                          suffix="%"))
        form.addRow("Low frequency", self._spin("view.freq0", 0.0, 1e6, step=100.0,
                                                decimals=1, suffix="Hz"))
        form.addRow("High frequency", self._spin("view.freq1", 1.0, 1e6, step=100.0,
                                                decimals=1, suffix="Hz"))
        form.addRow("", self._check("view.log_freq", "Log frequency axis"))
        self.channel_spin = self._spin("view.channel", 1, 64, step=1)
        form.addRow("Channel", self.channel_spin)
        return box

    def _colour_group(self) -> QGroupBox:
        box = QGroupBox("Colour")
        form = QFormLayout(box)
        form.addRow("Brightness", self._spin("view.brightness", -200.0, 200.0,
                                             step=1.0, decimals=1, suffix="dB"))
        form.addRow("Contrast", self._spin("view.contrast", 1.0, 1000.0, step=5.0,
                                           decimals=1, suffix="%"))
        combo = QComboBox()
        combo.addItems(["jet", "grey", "inverse grey", "hot", "bone"])
        self._bind("view.colormap", combo, combo.currentText, combo.setCurrentText,
                   combo.currentTextChanged)
        form.addRow("Map", combo)

        reset = QPushButton("Re-derive range")
        reset.setToolTip(
            "Recompute the colour range from the current window.\n"
            "The range is otherwise held while you scroll, so that levels stay\n"
            "comparable from one window to the next."
        )
        reset.clicked.connect(self._reset_clim)
        form.addRow("", reset)
        return box

    def _filter_group(self) -> QGroupBox:
        box = QGroupBox("Display filter")
        form = QFormLayout(box)
        on = self._check("view.filter_on", "Band-pass (display only)")
        on.setToolTip("Affects what is drawn, never what is measured or exported")
        form.addRow(on)
        form.addRow("From", self._spin("view.filter_low", 0.0, 1e6, step=50.0,
                                       decimals=1, suffix="Hz"))
        form.addRow("To", self._spin("view.filter_high", 1.0, 1e6, step=50.0,
                                     decimals=1, suffix="Hz"))
        return box

    def _ltsa_group(self) -> QGroupBox:
        """Motion for the LTSA panel, which has its own window and its own step."""
        box = QGroupBox("LTSA")
        form = QFormLayout(box)
        form.addRow("Window", self._spin("ltsa.tseg_hr", 0.0001, 8760.0, step=0.5,
                                         decimals=4, suffix="hr"))
        step = self._spin("ltsa.step_hr", -2.0, 8760.0, step=0.5, decimals=4,
                          suffix="hr")
        step.setToolTip(
            "-1  step by one window of time bins, skipping duty-cycle gaps (default)\n"
            "-2  step by one window of hours\n"
            "n   step by n hours"
        )
        form.addRow("Step", step)

        motion = QHBoxLayout()
        motion.setContentsMargins(0, 0, 0, 0)
        for label, tip, fn in (
            ("|◀", "start of LTSA", self._ltsa_start),
            ("◀◀", "back one step", lambda: self._ltsa_step(-1)),
            ("▶▶", "forward one step", lambda: self._ltsa_step(+1)),
            ("▶|", "end of LTSA", self._ltsa_end),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMaximumWidth(48)
            b.clicked.connect(fn)
            motion.addWidget(b)
        holder = QWidget()
        holder.setLayout(motion)
        form.addRow("", holder)

        self.ltsa_position_label = QLabel("--")
        form.addRow("Position", self.ltsa_position_label)
        return box

    def _sound_group(self) -> QGroupBox:
        """Playback of the window on screen."""
        box = QGroupBox("Sound")
        form = QFormLayout(box)

        speed = self._spin("view.play_speed", 0.001, 100.0, step=0.05, decimals=3,
                           suffix="x")
        speed.setToolTip(
            "Multiplies the sample rate, as Triton's speed factor does.\n"
            "Below 1 slows the audio and pitches it into hearing range,\n"
            "which is how a 60 kHz click becomes something you can hear."
        )
        form.addRow("Speed", speed)
        form.addRow("Volume", self._spin("view.play_volume", 0.0, 1.0, step=0.1,
                                         decimals=2))

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._play)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)
        fit = QPushButton("Fit to hearing")
        fit.setToolTip("Set the speed so the whole recorded band is audible")
        fit.clicked.connect(self._fit_speed)
        for b in (self.play_button, self.stop_button, fit):
            buttons.addWidget(b)
        holder = QWidget()
        holder.setLayout(buttons)
        form.addRow("", holder)
        return box

    # --------------------------------------------------------------------- actions

    def _status(self, text: str) -> None:
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(text, 8000)

    # ------------------------------------------------------------ LTSA motion actions

    def _ltsa_step(self, n: int) -> None:
        if not self.session.ltsa.is_open:
            return
        self.session.step_ltsa(n)
        self._refresh_ltsa_position()

    def _ltsa_start(self) -> None:
        if self.session.ltsa.is_open:
            self.session.ltsa_to_start()
            self._refresh_ltsa_position()

    def _ltsa_end(self) -> None:
        if self.session.ltsa.is_open:
            self.session.ltsa_to_end()
            self._refresh_ltsa_position()

    def _refresh_ltsa_position(self) -> None:
        if not self.session.ltsa.is_open:
            self.ltsa_position_label.setText("--")
            return
        src = self.session.ltsa_source
        here = self.session.ltsa.position or src.start_time
        self.ltsa_position_label.setText(
            f"{here}  (bin {src.bin_index(here) + 1} of {src.total_bins})"
        )

    # ---------------------------------------------------------------- sound actions

    def _play(self) -> None:
        if not self.session.audio.is_open:
            return
        win = self.window()
        player = getattr(win, "player", None)
        if player is None:
            return
        try:
            playable = self.session.playable(out_rate=player.default_rate())
            player.play(playable)
        except Exception as exc:                      # noqa: BLE001
            # No device, no driver, a speed that leaves nothing to play. All ordinary
            # on a compute node, and none of them worth a dialog.
            self._status(f"cannot play: {exc}")
            return
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        note = ""
        if playable.discarded_above_hz:
            note = (f"; content above {playable.discarded_above_hz:,.0f} Hz not "
                    f"audible at this speed -- try Fit to hearing")
        self._status(
            f"playing {playable.duration_sec:.1f} s at {playable.speed:g}x{note}"
        )

    def _stop(self) -> None:
        player = getattr(self.window(), "player", None)
        if player is not None:
            player.stop()
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _fit_speed(self) -> None:
        if not self.session.audio.is_open:
            return
        from ..playback import suggest_speed
        player = getattr(self.window(), "player", None)
        rate = player.default_rate() if player is not None else 48_000
        self.session.view.play_speed = suggest_speed(
            self.session.audio.source.sample_rate, rate
        )
        self._status(f"speed set to {self.session.view.play_speed:g}x for {rate} Hz out")

    def _goto_typed(self) -> None:
        """Act on the go-to-time box.

        The message on a bad entry goes in the box's own tooltip and the status bar
        rather than a dialog: someone typing a time is mid-flow, and a modal
        interruption for a typo is out of proportion. The text is left in place so it
        can be corrected rather than retyped.
        """
        text = self.goto_edit.text()
        if not text.strip():
            return
        try:
            landed = self.session.seek_text(text)
        except (ValueError, RuntimeError) as exc:
            win = self.window()
            if hasattr(win, "statusBar"):
                win.statusBar().showMessage(str(exc).splitlines()[0], 8000)
            self.goto_edit.setStyleSheet("background: #ffe8e8;")
            return
        self.goto_edit.setStyleSheet("")
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(f"moved to {landed}", 4000)

    def _goto_start(self) -> None:
        if self.session.audio.is_open:
            self.session.seek_samples(0, 0)

    def _goto_end(self) -> None:
        if not self.session.audio.is_open:
            return
        last = len(self.session.source.segments) - 1
        self.session.seek_samples(last, self.session._max_start_offset(last))

    def _reset_clim(self) -> None:
        self.session.view.clim = None

    def _refresh_readouts(self) -> None:
        if not self.session.audio.is_open:
            self.time_label.setText("--")
            self.position_label.setText("--")
            return
        self.time_label.setText(str(self.session.audio.time))
        self.position_label.setText(
            f"raw file {self.session.audio.segment + 1} of "
            f"{len(self.session.source.segments)}, "
            f"sample {self.session.audio.offset:,}"
        )

    def refresh_all(self) -> None:
        for group in self._bindings.values():
            for b in group:
                b.refresh()
        self._refresh_readouts()
