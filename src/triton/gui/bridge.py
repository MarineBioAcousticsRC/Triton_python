"""The one place Qt meets the session.

``triton.session`` deliberately imports no GUI toolkit -- there is a test that scans
its imports and fails if one appears -- so something has to translate the session's
framework-agnostic callbacks into Qt signals. That is this file, and keeping it small
and singular is the point: everything above it can be ordinary Qt, everything below it
stays usable from a script or a test with no display.

It also does the job ``control.m`` does by hand fifty times over. In MATLAB every
action re-implements its own "update the field, call ``plot_triton``, then toggle
twenty widget ``Enable`` states" sequence, which is where most of that file's 976
lines go and where most of its bugs live. Here a change to session state is announced
once and whoever cares subscribes.

Two things it adds beyond plain forwarding, both because a GUI needs them and a batch
script does not:

**Coalescing.** A session ``batch()`` already groups mutations, but a user dragging a
slider generates one notification per pixel. Repaints are scheduled on a zero-delay
timer, so any number of notifications arriving before Qt next processes events produce
exactly one repaint.

**Dirtiness classification.** Not every field costs the same to honour. Moving the
playback position or changing ``nfft`` means re-reading samples and re-running FFTs;
changing the colour map means neither. :class:`Dirty` says which, so the window can
repaint without recomputing when that is all that is needed.
"""

from __future__ import annotations

import enum
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from ..session import TritonSession

__all__ = ["Dirty", "SessionBridge", "classify"]


class Dirty(enum.Flag):
    """What a change invalidates.

    Ordered by cost. ``DATA`` implies re-reading samples from disk and re-running the
    spectrogram; ``DISPLAY`` is a re-map of numbers already computed; ``AXES`` is a
    relabel. A repaint honours the most expensive flag it is given.
    """

    NOTHING = 0
    AXES = enum.auto()        # tick labels, units, time format
    DISPLAY = enum.auto()     # colour map, brightness, contrast, colour limits
    DATA = enum.auto()        # position, nfft, overlap, tseg, channel, band, filter
    LTSA = enum.auto()        # anything about the LTSA panel
    LAYOUT = enum.auto()      # which panels are shown at all


#: Which session paths invalidate what. Prefix match, longest first.
#:
#: Kept as data rather than a chain of ``if`` statements precisely because
#: ``control.m`` is the chain-of-ifs version and it is 976 lines. Adding a control
#: here is adding a row.
_RULES: tuple[tuple[str, Dirty], ...] = (
    ("audio.source", Dirty.DATA | Dirty.AXES | Dirty.LAYOUT),
    ("audio.segment", Dirty.DATA),
    ("audio.offset", Dirty.DATA),
    ("view.nfft", Dirty.DATA),
    ("view.overlap_pct", Dirty.DATA),
    ("view.tseg_sec", Dirty.DATA | Dirty.AXES),
    ("view.channel", Dirty.DATA),
    ("view.freq0", Dirty.DATA | Dirty.AXES),
    ("view.freq1", Dirty.DATA | Dirty.AXES),
    ("view.step_sec", Dirty.NOTHING),          # affects the next step, not this frame
    # Panel toggles change the layout *and* need a frame, since a panel that was hidden
    # has nothing drawn in it yet. Listed one by one rather than as a "view.show_"
    # prefix because show_delimiters is not a layout change and prefix rules are
    # checked in order -- an easy way to get a subtly wrong classification.
    ("view.show_ltsa", Dirty.LAYOUT | Dirty.DATA | Dirty.LTSA),
    ("view.show_specgram", Dirty.LAYOUT | Dirty.DATA),
    ("view.show_timeseries", Dirty.LAYOUT | Dirty.DATA),
    ("view.show_spectra", Dirty.LAYOUT | Dirty.DATA),
    ("view.show_delimiters", Dirty.DISPLAY),
    ("view.log_freq", Dirty.AXES),
    # Playback settings change what is heard, never what is drawn.
    ("view.export_all_channels", Dirty.NOTHING),   # changes files, not the display
    ("view.play_speed", Dirty.NOTHING),
    ("view.play_volume", Dirty.NOTHING),
    ("view.filter_on", Dirty.DATA),
    ("view.filter_low", Dirty.DATA),
    ("view.filter_high", Dirty.DATA),
    ("view.brightness", Dirty.DISPLAY),
    ("view.contrast", Dirty.DISPLAY),
    ("view.clim", Dirty.DISPLAY),
    ("view.colormap", Dirty.DISPLAY),
    ("calibration.", Dirty.DATA),              # a transfer function shifts every value
    # Opening or closing an LTSA changes whether its panel *can* be shown, so it is
    # a layout change as well. Must precede the general "ltsa." rule, which is a
    # prefix match taken in order.
    ("ltsa.source", Dirty.LTSA | Dirty.LAYOUT),
    ("ltsa.expand", Dirty.NOTHING),             # a mode, draws nothing
    ("ltsa.clim", Dirty.LTSA),
    ("ltsa.", Dirty.LTSA),
    ("config.", Dirty.NOTHING),
)


def classify(path: str) -> Dirty:
    """What does a change at ``path`` invalidate?

    Unknown paths return ``DATA``, the expensive answer. A new field that nobody
    remembered to classify then repaints too often, which is a performance bug someone
    will notice; the alternative default of ``NOTHING`` would mean it silently never
    repaints, which looks like the data being wrong.
    """
    for prefix, flags in _RULES:
        if path == prefix or path.startswith(prefix):
            return flags
    return Dirty.DATA


class SessionBridge(QObject):
    """Turns session notifications into Qt signals, coalesced and classified.

    ``repaint_needed`` carries the accumulated :class:`Dirty` flags since the last
    repaint, so a listener can decide how much work to do.
    """

    #: Emitted once per event-loop turn, with everything that changed since the last.
    repaint_needed = Signal(Dirty)
    #: Every path, uncoalesced, for anything that needs to see individual changes --
    #: a Session Inspector, or a control keeping itself in sync with the state.
    changed = Signal(str)

    def __init__(self, session: TritonSession, parent: QObject | None = None):
        super().__init__(parent)
        self.session = session
        self._pending = Dirty.NOTHING
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._flush)
        self._unsubscribe: Callable[[], None] | None = session.subscribe(self._on_change)

    def _on_change(self, path: str) -> None:
        # Called from whatever thread mutated the session. Qt widgets may only be
        # touched from the GUI thread, so the signal is emitted here (thread-safe) and
        # the timer start is queued rather than called directly.
        self.changed.emit(path)
        flags = classify(path)
        if flags is Dirty.NOTHING:
            return
        self._pending |= flags
        # start() on the single-shot timer created in __init__, rather than the static
        # QTimer.singleShot: the static form has no (msec, timerType, callable)
        # overload, and calling it that way raises inside the session's dispatch --
        # where the exception surfaces from open_audio and looks like a reader bug.
        if not self._timer.isActive():
            self._timer.start()

    def _flush(self) -> None:
        flags, self._pending = self._pending, Dirty.NOTHING
        if flags is not Dirty.NOTHING:
            self.repaint_needed.emit(flags)

    def force(self, flags: Dirty = Dirty.DATA | Dirty.AXES | Dirty.LAYOUT) -> None:
        """Request a repaint without a session change -- for the initial draw."""
        self._pending |= flags
        self._flush()

    def detach(self) -> None:
        """Stop listening.  Idempotent, and safe to call from a destructor."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
