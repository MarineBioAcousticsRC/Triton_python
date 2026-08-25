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
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..session import TritonSession, ValidationError
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
        root.addWidget(self._time_group())
        root.addWidget(self._display_group())
        root.addWidget(self._spectrogram_group())
        root.addWidget(self._colour_group())
        root.addWidget(self._filter_group())
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

    # --------------------------------------------------------------------- actions

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
