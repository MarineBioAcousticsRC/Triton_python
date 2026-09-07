"""The control panel.

Laid out as MATLAB lays it out, because that layout is dense for a reason. ``initcontrol.m``
packs about forty controls into a 600 x 700 window with a four-column grid -- a label,
then up to three fields, with a header row naming the fields beneath it -- and two
sections, LTSA and audio, built from the **same rows in the same order**: start time,
length / step, frequency start / end, brightness / contrast, colour map, motion. Learn one
and you know the other. An earlier version of this file used a form layout with one
control per row and came out three times taller than the screen it had to fit on.

What is *not* carried over from MATLAB: the separate pop-up windows for band-pass and
sound, which get lost behind other windows (both are inline here); radio buttons for the
panel toggles, which are independent and so are check boxes; and the ``p``/``n``
buttons, which are a Remora's previous/next-detection controls living permanently in the
core window and doing nothing unless that Remora is loaded.

Binding is the mechanism that keeps this file short. A control is bound to a session
field: edits flow one way, notifications flow back the other, and neither side knows
about the widgets on the far side of the panel. Loop prevention is a re-entrancy flag
rather than signal disconnection, because disconnecting loses events that arrive while
disconnected. A slider and a spin box can be bound to the *same* field, which is how
brightness gets both -- drag to explore, type for precision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..session import _TIME_FORMS, TritonSession, ValidationError
from .bridge import SessionBridge

__all__ = ["ControlPanel", "COLORMAPS"]

COLORMAPS = ("jet", "grey", "inverse grey", "hot", "bone")


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
            # clearing _busy, because refresh() is itself guarded by that flag and would
            # otherwise return without doing anything.
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


def _header(text: str) -> QLabel:
    """A column header, styled to read as a heading rather than a value."""
    lab = QLabel(text)
    lab.setStyleSheet("color: #444; font-size: 11px;")
    lab.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
    return lab


def _row_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lab.setStyleSheet("font-weight: 600;")
    return lab


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #ccc;")
    return line


class ControlPanel(QWidget):
    """Everything that writes to the session, in MATLAB's arrangement."""

    def __init__(self, session: TritonSession, bridge: SessionBridge,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.bridge = bridge
        self._bindings: dict[str, list[_Binding]] = {}

        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)
        root.addLayout(self._panel_toggles())
        root.addWidget(self._section("ltsa"))
        root.addWidget(self._section("audio"))
        root.addWidget(self._cursor_group())
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
              decimals: int = 0, suffix: str = "", width: int = 0) -> QWidget:
        w: QSpinBox | QDoubleSpinBox
        if decimals:
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
        else:
            w = QSpinBox()
        w.setRange(lo, hi)              # type: ignore[arg-type]
        w.setSingleStep(step)           # type: ignore[arg-type]
        w.setKeyboardTracking(False)    # don't fire on every keystroke mid-typing
        if suffix:
            w.setSuffix(f" {suffix}")
        if width:
            w.setMaximumWidth(width)
        self._bind(path, w, w.value, w.setValue, w.valueChanged)
        return w

    def _check(self, path: str, label: str) -> QCheckBox:
        w = QCheckBox(label)
        self._bind(path, w, w.isChecked, w.setChecked, w.toggled)
        return w

    def _combo(self, path: str, items: tuple[str, ...]) -> QComboBox:
        w = QComboBox()
        w.addItems(list(items))
        self._bind(path, w, w.currentText, w.setCurrentText, w.currentTextChanged)
        return w

    def _slider_spin(self, path: str, lo: float, hi: float, slider_lo: float,
                     slider_hi: float, decimals: int, suffix: str
                     ) -> tuple[QSlider, QWidget]:
        """A slider and a spin box bound to the same field.

        The slider covers the range people actually use; the spin box covers the range
        the session allows. When a typed value is outside the slider's range the slider
        sits at its end, which is correct -- it is an approximate control.

        QSlider is integer-valued, so it works in units of ``10**-decimals``.
        """
        scale = 10 ** decimals
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(round(slider_lo * scale)), int(round(slider_hi * scale)))
        slider.setSingleStep(1)
        slider.setPageStep(max(1, int(scale)))
        self._bind(path, slider,
                   lambda: slider.value() / scale,
                   lambda v: slider.setValue(int(round(v * scale))),
                   slider.valueChanged)
        spin = self._spin(path, lo, hi, step=1.0 if decimals == 0 else 10 ** -decimals * 5,
                          decimals=decimals, suffix=suffix, width=90)
        return slider, spin

    def _motion_row(self, first, back, forward, last) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        for label, tip, fn in (
            ("|◀", "start", first),
            ("◀◀", "back one step", back),
            ("▶▶", "forward one step", forward),
            ("▶|", "end", last),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMinimumWidth(44)
            b.clicked.connect(fn)
            row.addWidget(b)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    # ------------------------------------------------------------- panel toggles

    def _panel_toggles(self) -> QHBoxLayout:
        """One row across the top, in stacking order -- check boxes, not radios."""
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        for path, label in (("view.show_ltsa", "LTSA"),
                            ("view.show_specgram", "Spectrogram"),
                            ("view.show_timeseries", "Time series"),
                            ("view.show_spectra", "Spectra")):
            row.addWidget(self._check(path, label))
        row.addStretch(1)
        return row

    # ------------------------------------------------------- the mirrored sections

    def _section(self, kind: str) -> QGroupBox:
        """One of the two parallel sections.  Same rows, same order, for both.

        ``kind`` is ``"ltsa"`` or ``"audio"``. Where the two differ -- the audio section
        has spectral parameters, a channel, a display filter and sound; the LTSA has
        none of those -- the extra rows are appended *after* the shared ones, so the
        shared part lines up visually.
        """
        is_ltsa = kind == "ltsa"
        box = QGroupBox("LTSA" if is_ltsa else "Audio (x.wav / wav)")
        g = QGridLayout(box)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        g.setColumnMinimumWidth(0, 78)
        for c in (1, 2, 3):
            g.setColumnStretch(c, 1)
        r = 0

        # -- position readout, and the go-to box on the audio side
        g.addWidget(_row_label("Position"), r, 0)
        if is_ltsa:
            self.ltsa_position_label = QLabel("--")
            self.ltsa_position_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            g.addWidget(self.ltsa_position_label, r, 1, 1, 3)
        else:
            self.time_label = QLabel("--")
            self.time_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            g.addWidget(self.time_label, r, 1, 1, 3)
            r += 1
            self.position_label = QLabel("--")
            self.position_label.setStyleSheet("color: #555; font-size: 11px;")
            g.addWidget(self.position_label, r, 1, 1, 3)
            r += 1
            g.addWidget(_row_label("Go to"), r, 0)
            self.goto_edit = QLineEdit()
            self.goto_edit.setPlaceholderText("time, +secs, or @secs")
            self.goto_edit.setToolTip(
                "Jump to a time. Any of:\n" + _TIME_FORMS + "\n\nEnter to go."
            )
            self.goto_edit.returnPressed.connect(self._goto_typed)
            g.addWidget(self.goto_edit, r, 1, 1, 2)
            go = QPushButton("Go")
            go.setMaximumWidth(48)
            go.clicked.connect(self._goto_typed)
            g.addWidget(go, r, 3)
        r += 1

        # -- length / step, with headers naming the fields beneath
        unit = "hr" if is_ltsa else "s"
        g.addWidget(_header(f"Length ({unit})"), r, 1)
        g.addWidget(_header(f"Step ({unit})"), r, 2)
        r += 1
        g.addWidget(_row_label("Window"), r, 0)
        if is_ltsa:
            g.addWidget(self._spin("ltsa.tseg_hr", 0.0001, 8760.0, step=0.5,
                                   decimals=4), r, 1)
            step = self._spin("ltsa.step_hr", -2.0, 8760.0, step=0.5, decimals=4)
            step.setToolTip("-1  one window of time bins, skipping gaps (default)\n"
                            "-2  one window of hours\n n   n hours")
        else:
            g.addWidget(self._spin("view.tseg_sec", 0.001, 86400.0, step=0.5,
                                   decimals=3), r, 1)
            step = self._spin("view.step_sec", -1.0, 86400.0, step=0.5, decimals=3)
            step.setToolTip("-1 means step by the window length")
        g.addWidget(step, r, 2)
        r += 1

        # -- frequency
        g.addWidget(_header("Start (Hz)"), r, 1)
        g.addWidget(_header("End (Hz)"), r, 2)
        r += 1
        g.addWidget(_row_label("Frequency"), r, 0)
        pre = "ltsa" if is_ltsa else "view"
        g.addWidget(self._spin(f"{pre}.freq0", 0.0, 1e6, step=100.0, decimals=1), r, 1)
        g.addWidget(self._spin(f"{pre}.freq1", 1.0, 1e6, step=100.0, decimals=1), r, 2)
        if is_ltsa:
            pass
        else:
            # Linear/log are mutually exclusive, so these are correctly radios.
            axis = QWidget()
            ah = QHBoxLayout(axis)
            ah.setContentsMargins(0, 0, 0, 0)
            lin, log = QRadioButton("Linear"), QRadioButton("Log")
            grp = QButtonGroup(axis)
            grp.addButton(lin)
            grp.addButton(log)
            self._bind("view.log_freq", log, log.isChecked,
                       lambda v: (log.setChecked(bool(v)), lin.setChecked(not v)),
                       log.toggled)
            ah.addWidget(lin)
            ah.addWidget(log)
            g.addWidget(axis, r, 3)
        r += 1

        # -- brightness / contrast: slider plus value
        for label, field, lo, hi, slo, shi, dec, suf in (
            ("Brightness", "brightness", -200.0, 200.0, -60.0, 60.0, 1, "dB"),
            ("Contrast", "contrast", 1.0, 1000.0, 10.0, 400.0, 0, "%"),
        ):
            g.addWidget(_row_label(label), r, 0)
            slider, spin = self._slider_spin(f"{pre}.{field}", lo, hi, slo, shi, dec, suf)
            g.addWidget(slider, r, 1, 1, 2)
            g.addWidget(spin, r, 3)
            r += 1

        # -- colour map and delimiters
        g.addWidget(_row_label("Colour"), r, 0)
        g.addWidget(self._combo(f"{pre}.colormap", COLORMAPS), r, 1)
        if is_ltsa:
            reset = QPushButton("Re-derive range")
            reset.setToolTip("Recompute the colour range from this window.\n"
                             "It is otherwise held while you scroll, so levels stay\n"
                             "comparable from one window to the next.")
            reset.clicked.connect(self._reset_ltsa_clim)
            g.addWidget(reset, r, 2)
            expand = self._check("ltsa.expand", "Expand")
            expand.setToolTip(
                "When on, clicking the LTSA opens the audio recording behind that\n"
                "point and jumps to it. Off, a click only records the point in the\n"
                "pick log, so you can read values without changing what is open."
            )
            g.addWidget(expand, r, 3)
        else:
            reset = QPushButton("Re-derive range")
            reset.setToolTip("Recompute the colour range from this window.\n"
                             "It is otherwise held while you scroll, so levels stay\n"
                             "comparable from one window to the next.")
            reset.clicked.connect(self._reset_clim)
            g.addWidget(reset, r, 2)
            g.addWidget(self._check("view.show_delimiters", "Delimiters"), r, 3)
        r += 1

        # -- audio-only rows
        if not is_ltsa:
            g.addWidget(_hline(), r, 0, 1, 4)
            r += 1
            g.addWidget(_header("FFT length"), r, 1)
            g.addWidget(_header("Overlap (%)"), r, 2)
            g.addWidget(_header("Channel"), r, 3)
            r += 1
            g.addWidget(_row_label("Spectral"), r, 0)
            g.addWidget(self._spin("view.nfft", 8, 1 << 20, step=1), r, 1)
            g.addWidget(self._spin("view.overlap_pct", 0, 99, step=5), r, 2)
            self.channel_spin = self._spin("view.channel", 1, 64, step=1)
            g.addWidget(self.channel_spin, r, 3)
            r += 1

            g.addWidget(_header("From (Hz)"), r, 2)
            g.addWidget(_header("To (Hz)"), r, 3)
            r += 1
            g.addWidget(_row_label("Band-pass"), r, 0)
            on = self._check("view.filter_on", "On (display only)")
            on.setToolTip("Affects what is drawn and heard, never what is measured")
            g.addWidget(on, r, 1)
            g.addWidget(self._spin("view.filter_low", 0.0, 1e6, step=50.0, decimals=1),
                        r, 2)
            g.addWidget(self._spin("view.filter_high", 1.0, 1e6, step=50.0, decimals=1),
                        r, 3)
            r += 1

            g.addWidget(_header("Speed (x)"), r, 2)
            g.addWidget(_header("Volume"), r, 3)
            r += 1
            g.addWidget(_row_label("Sound"), r, 0)
            snd = QWidget()
            sh = QHBoxLayout(snd)
            sh.setContentsMargins(0, 0, 0, 0)
            sh.setSpacing(2)
            self.play_button = QPushButton("Play")
            self.play_button.clicked.connect(self._play)
            self.stop_button = QPushButton("Stop")
            self.stop_button.clicked.connect(self._stop)
            self.stop_button.setEnabled(False)
            fit = QPushButton("Fit")
            fit.setToolTip("Set the speed so the whole recorded band is audible")
            fit.clicked.connect(self._fit_speed)
            for b in (self.play_button, self.stop_button, fit):
                sh.addWidget(b)
            g.addWidget(snd, r, 1)
            speed = self._spin("view.play_speed", 0.001, 100.0, step=0.05, decimals=3)
            speed.setToolTip("Multiplies the sample rate. Below 1 slows the audio and\n"
                             "pitches it into hearing range.")
            g.addWidget(speed, r, 2)
            g.addWidget(self._spin("view.play_volume", 0.0, 1.0, step=0.1, decimals=2),
                        r, 3)
            r += 1

        # -- motion, last, as in MATLAB
        g.addWidget(_hline(), r, 0, 1, 4)
        r += 1
        if is_ltsa:
            g.addWidget(self._motion_row(self._ltsa_start,
                                         lambda: self._ltsa_step(-1),
                                         lambda: self._ltsa_step(+1),
                                         self._ltsa_end), r, 0, 1, 4)
        else:
            g.addWidget(self._motion_row(self._goto_start,
                                         lambda: self.session.step(-1),
                                         lambda: self.session.step(+1),
                                         self._goto_end), r, 0, 1, 4)
        return box

    # ------------------------------------------------------------------- cursor

    def _cursor_group(self) -> QGroupBox:
        """The live readout.  ``coorddisp.m`` puts it in this window too.

        Fixed rows, blanked rather than hidden when they do not apply: the readout
        changes as the pointer crosses between panels, and a panel that reflows on
        mouse-move is unusable. Two columns, so it takes half the height it used to.
        """
        box = QGroupBox("Cursor")
        g = QGridLayout(box)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(2)
        self.cursor_rows: dict[str, QLabel] = {}
        keys = ("Time", "Frequency", "Spectrum level [dB]", "Counts", "Raw file", "File")
        for i, key in enumerate(keys):
            row, col = divmod(i, 2)
            lab = QLabel("--")
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            g.addWidget(_row_label(key), row, col * 2)
            g.addWidget(lab, row, col * 2 + 1)
            self.cursor_rows[key] = lab
        self.log_picks = QCheckBox("Log picks (click a panel to record a point)")
        self.log_picks.setChecked(True)
        g.addWidget(self.log_picks, 3, 0, 1, 4)
        return box

    def show_readout(self, readout) -> None:
        values = dict(readout.lines())
        for key, lab in self.cursor_rows.items():
            lab.setText(values.get(key, "--"))

    # --------------------------------------------------------------------- actions

    def _status(self, text: str) -> None:
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(text, 8000)

    def _ltsa_step(self, n: int) -> None:
        if self.session.ltsa.is_open:
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

    def _play(self) -> None:
        if not self.session.audio.is_open:
            return
        player = getattr(self.window(), "player", None)
        if player is None:
            return
        try:
            playable = self.session.playable(out_rate=player.default_rate())
            player.play(playable)
        except Exception as exc:                      # noqa: BLE001
            self._status(f"cannot play: {exc}")
            return
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        note = ""
        if playable.discarded_above_hz:
            note = (f"; content above {playable.discarded_above_hz:,.0f} Hz not "
                    f"audible at this speed -- try Fit")
        self._status(f"playing {playable.duration_sec:.1f} s at {playable.speed:g}x{note}")

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
            self.session.audio.source.sample_rate, rate)
        self._status(f"speed set to {self.session.view.play_speed:g}x for {rate} Hz out")

    def _goto_typed(self) -> None:
        text = self.goto_edit.text()
        if not text.strip():
            return
        try:
            landed = self.session.seek_text(text)
        except (ValueError, RuntimeError) as exc:
            self._status(str(exc).splitlines()[0])
            self.goto_edit.setStyleSheet("background: #ffe8e8;")
            return
        self.goto_edit.setStyleSheet("")
        self._status(f"moved to {landed}")

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

    def _reset_ltsa_clim(self) -> None:
        self.session.ltsa.clim = None

    def _refresh_readouts(self) -> None:
        if not self.session.audio.is_open:
            self.time_label.setText("--")
            self.position_label.setText("--")
            return
        self.time_label.setText(str(self.session.audio.time))
        self.position_label.setText(
            f"raw file {self.session.audio.segment + 1} of "
            f"{len(self.session.source.segments)}, sample {self.session.audio.offset:,}"
        )

    def refresh_all(self) -> None:
        for group in self._bindings.values():
            for b in group:
                b.refresh()
        self._refresh_readouts()
        self._refresh_ltsa_position()
