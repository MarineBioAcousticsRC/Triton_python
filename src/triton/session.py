"""The session: one object describing what is open and how it is being looked at.

This replaces the ``PARAMS`` global. The argument for doing so is in PORTING_PLAN §3
and is not repeated here; what matters at the code level is the distinction it rests
on. ``PARAMS`` is criticised for being global, but the useful thing about it --
that there is always one inspectable description of the loaded data -- is worth
keeping. What is not worth keeping is that it is *scoped* globally, so that forty
functions can reach in and mutate it, only one dataset can be open at a time, and
nothing can be tested without booting a GUI.

So a session is globally *reachable* (:func:`current_session`, one documented seam)
and not globally *scoped*: everything else takes a session argument.

Four properties this file exists to provide, in rough order of how much they matter.

**Position is an integer sample index, never a float time.** ``AudioState`` holds a
segment index and a sample offset within it; wall-clock time is derived. MATLAB
instead accumulates ``PARAMS.plot.dnum`` as the analyst steps around, so the same
nominal time reached two different ways can land on different samples -- see
``AudioSource.skip_for``. Here, stepping forward and back returns to the identical
sample by construction, and :func:`step` cannot drift however long a session runs.

**Mutation is observable.** Assigning to a state field validates the value and
notifies subscribers with a dotted path (``"view.nfft"``). This is what deletes most
of ``control.m``, where every one of ~50 actions hand-rolls its own "update field,
call plot_triton, toggle twenty widget Enable states" sequence.

**No GUI toolkit is imported.** Deliberately not Qt signals, though Phase 3 is Qt: a
callback registry keeps the whole data path usable from a test, a batch script, or CI
with no display. Phase 3 adapts this to Qt rather than the reverse.

**Nothing here changes a number.** The session composes the verified Phase 1 code and
adds no arithmetic of its own. ``tests/test_session.py`` asserts that tiles obtained
through the session are identical to the same data obtained by calling
``triton.io`` and ``triton.dsp`` directly, which is the only way to be sure this
layer stays a convenience rather than becoming a second implementation.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from . import dsp
from .io import audio as _audio
from .io import ltsa as _ltsa

__all__ = [
    "TritonSession",
    "AudioState",
    "ViewState",
    "LtsaState",
    "CalibrationState",
    "AppConfig",
    "SpectrogramTile",
    "LtsaTile",
    "Frame",
    "Readout",
    "current_session",
    "set_current_session",
    "open_session",
]

_MISSING = object()


# --------------------------------------------------------------------- validation

class ValidationError(ValueError):
    """An assignment to a session field that the field will not accept.

    Raised at assignment time rather than discovered later in a plot routine, which
    is where MATLAB finds these -- ``control.m`` reads a value out of a text box,
    stores it, and the consequence surfaces as an error inside ``plot_specgram``.
    """


def _positive(name: str, v: Any) -> Any:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
        raise ValidationError(f"{name} must be a positive number, got {v!r}")
    return v


def _positive_int(name: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {v!r}")
    return int(v)


def _percent(name: str, v: Any) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v < 100:
        raise ValidationError(
            f"{name} must be a percentage in [0, 100), got {v!r}. "
            f"100% overlap would advance zero samples per frame."
        )
    return float(v)


def _non_negative(name: str, v: Any) -> Any:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
        raise ValidationError(f"{name} must be zero or more, got {v!r}")
    return v


def _non_negative_int(name: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0:
        raise ValidationError(f"{name} must be a non-negative integer, got {v!r}")
    return int(v)


# ------------------------------------------------------------------- state groups

class _State:
    """Base for the session's state groups.

    Assignment goes through :meth:`__setattr__` so that values are validated and
    subscribers notified. Private attributes (leading underscore) bypass both, which
    is how the owner back-reference and the group's own name are stored without
    generating notifications for them.

    The owner reference is **weak**, and the reason is ownership rather than
    performance -- readers open and close their files per read, so there are no
    long-lived handles a cycle would pin. It is that a state group must not keep its
    session alive: ``view = TritonSession().view`` should collect the session, not
    leave an immortal one reachable through a leaked sub-object. The visible
    consequence is that a detached group stops notifying, which is correct, since
    there is no longer anything for it to notify.
    """

    #: Class-level defaults, not instance state set in ``__init__``. Subclasses are
    #: dataclasses, so their generated ``__init__`` assigns fields directly and never
    #: calls a base ``__init__`` -- meaning anything set there would not exist yet when
    #: ``__setattr__`` runs for the first field. Class attributes are always there.
    _owner_ref: weakref.ref[TritonSession] | None = None
    _group: str = ""
    _validators: dict[str, Callable[[str, Any], Any]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        cls._group = cls.__name__

    def _attach(self, owner: TritonSession, group: str) -> None:
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))
        object.__setattr__(self, "_group", group)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        validator = self._validators.get(key)
        if validator is not None:
            value = validator(f"{self._group}.{key}", value)
        old = getattr(self, key, _MISSING)
        object.__setattr__(self, key, value)
        if old is _MISSING or not _same(old, value):
            self._emit(key)

    def _emit(self, key: str) -> None:
        ref = getattr(self, "_owner_ref", None)
        owner = ref() if ref is not None else None
        if owner is not None:
            owner._notify(f"{self._group}.{key}")

    def as_dict(self) -> dict[str, Any]:
        """Public fields only, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]


def _same(a: Any, b: Any) -> bool:
    """Equality that survives numpy arrays and NaT, for change detection."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            and a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
        )
    try:
        return bool(a == b)
    except Exception:
        return False


@dataclass(eq=False)
class AudioState(_State):
    """What audio is open, and where in it we are.

    ``segment`` and ``offset`` are the position: a raw-file index and a sample count
    within it. Not a time. Every seek converts once, on the way in; everything
    downstream is integer arithmetic, so a session cannot accumulate drift no matter
    how much the analyst moves around.
    """

    source: _audio.AudioSource | None = None
    path: Path | None = None
    segment: int = 0
    offset: int = 0                      # samples into `segment`

    _validators = {"segment": _non_negative_int, "offset": _non_negative_int}

    @property
    def is_open(self) -> bool:
        return self.source is not None

    @property
    def time(self) -> np.datetime64:
        """Wall-clock time of the current position.  Derived, never stored."""
        if self.source is None:
            raise RuntimeError("no audio open")
        seg = self.source.segments[self.segment]
        ns = int(round(self.offset * 1e9 / self.source.sample_rate))
        return seg.start + np.timedelta64(ns, "ns")


@dataclass(eq=False)
class ViewState(_State):
    """How the audio is being displayed.

    Defaults match ``initparams.m`` so that a fresh session shows what a fresh Triton
    shows. ``clim`` empty means "derive from the first frame and then hold", which is
    the behaviour added to ``plot_specgram.m`` for issue #111 -- levels must stay
    comparable while an analyst scrolls, or feature salience changes under them.
    """

    tseg_sec: float = 1.0
    step_sec: float = -1.0               # -1 means "step by tseg", as PARAMS.tseg.step
    nfft: int = 1000
    overlap_pct: float = 0.0
    freq0: float = 0.0
    freq1: float | None = None           # None means Nyquist
    channel: int = 1                     # 1-based, as PARAMS.ch
    brightness: float = 0.0              # dB
    contrast: float = 100.0              # percent
    colormap: str = "jet"
    clim: tuple[float, float] | None = None

    # Display band-pass. Display only -- it never touches what is measured or written.
    filter_on: bool = False
    filter_low: float = 100.0
    filter_high: float = 1000.0

    # Which panels are shown. MATLAB stacks them in a fixed order regardless of the
    # order they were switched on (plot_triton.m:44-70), so this is four flags rather
    # than a list.
    show_ltsa: bool = False
    show_specgram: bool = True
    show_timeseries: bool = False
    show_spectra: bool = False

    # Log frequency axis, as PARAMS.fax / the specgramlog control.
    log_freq: bool = False
    show_delimiters: bool = True

    _validators = {
        "tseg_sec": _positive,
        "nfft": _positive_int,
        "overlap_pct": _percent,
        "freq0": _non_negative,
        "channel": _positive_int,
        "contrast": _positive,
        "filter_low": _non_negative,
        "filter_high": _positive,
    }


@dataclass(eq=False)
class LtsaState(_State):
    """The open LTSA and its own view settings, which are separate from the audio's."""

    source: _ltsa.LtsaSource | None = None
    path: Path | None = None
    tseg_hr: float = 1.0
    freq0: float = 0.0
    freq1: float | None = None
    brightness: float = 0.0
    contrast: float = 100.0
    position: np.datetime64 | None = None

    _validators = {"tseg_hr": _positive, "freq0": _non_negative, "contrast": _positive}

    @property
    def is_open(self) -> bool:
        return self.source is not None


@dataclass(eq=False)
class CalibrationState(_State):
    """Transfer function and gain.

    ``tf_freq``/``tf_value`` are the measured curve; :meth:`curve_for` interpolates it
    onto a frequency axis with the linear extrapolation ``plot_specgram.m`` relies on.
    """

    tf_freq: np.ndarray | None = None
    tf_value: np.ndarray | None = None
    tf_path: Path | None = None

    @property
    def has_tf(self) -> bool:
        return self.tf_freq is not None and self.tf_value is not None

    def curve_for(self, freq: np.ndarray) -> np.ndarray | None:
        if not self.has_tf:
            return None
        return dsp.apply_transfer_function_curve(self.tf_freq, self.tf_value, freq)


@dataclass(eq=False)
class AppConfig(_State):
    """Paths and defaults that outlive any one file being open."""

    last_audio_dir: Path | None = None
    last_ltsa_dir: Path | None = None
    export_dir: Path | None = None


# ------------------------------------------------------------------------- tiles

@dataclass(frozen=True)
class SpectrogramTile:
    """One window of spectrogram, with everything needed to draw it.

    ``boundaries`` replaces ``PARAMS.raw.delimit_time``: offsets in seconds from the
    left edge where a raw file ends. Entries past the right edge are normal -- see
    ``AudioSource._boundaries``.
    """

    start: np.datetime64
    f: np.ndarray
    t: np.ndarray                        # seconds from the left edge
    db: np.ndarray                       # (n_freq, n_time)
    boundaries: tuple[_audio.Boundary, ...]
    clim: tuple[float, float]
    segment: int
    offset: int


@dataclass(frozen=True)
class Frame:
    """Everything the four panels need for one window, from **one** read.

    This exists for two reasons, and the second is a bug fix.

    Reading a window costs a seek and a few megabytes; computing a spectrogram costs
    thousands of FFTs. MATLAB reads once into the ``DATA`` global and each panel takes
    what it needs from there, which is efficient. Doing the same here without a global
    means handing the panels one object.

    And in MATLAB each of ``plot_specgram``, ``plot_timeseries`` and ``plot_spectra``
    applies the display filter to ``DATA`` **and assigns the result back**, so showing
    three panels band-passes the data three times over; the panel drawn first sees
    different numbers from the panel drawn last, by up to ~20 dB at the filter corners
    (OPEN_DECISIONS §1.6). The same routines also decrement ``PARAMS.ch`` once each, so
    in multichannel mode three panels can show three different channels. Computing
    once and passing the result removes both by construction rather than by care.
    """

    start: np.datetime64
    samples: np.ndarray                  # (n, n_channels), after the display filter
    fs: int
    spectrogram: SpectrogramTile
    spectra_f: np.ndarray                # averaged spectrum: frequency axis
    spectra_db: np.ndarray               # ...and its values
    boundaries: tuple[_audio.Boundary, ...]
    channel: int                         # 1-based, the channel every panel is showing

    #: The same two arrays *before* contrast, brightness and the colour range were
    #: applied. Kept so that changing a display knob costs a multiply rather than a
    #: re-read and several thousand FFTs -- measured, a 30 s window at 200 kHz takes
    #: 242 ms to compute and under a millisecond to re-map, which is the difference
    #: between a brightness slider that works and one that does not.
    #: :meth:`TritonSession.remap` is what consumes them.
    raw_db: np.ndarray = field(default_factory=lambda: np.empty(0))
    raw_spectra_db: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def time_axis(self) -> np.ndarray:
        """Seconds from the left edge, as ``plot_timeseries.m:24`` computes it."""
        return np.arange(self.samples.shape[0]) / self.fs


@dataclass(frozen=True)
class LtsaTile:
    """One window of LTSA."""

    start: np.datetime64
    f: np.ndarray
    t: np.ndarray                        # hours from the left edge
    db: np.ndarray                       # (n_freq, n_bins)
    clim: tuple[float, float]


@dataclass(frozen=True)
class Readout:
    """What the cursor is over.  The port of ``coorddisp.m``'s readout half.

    Computed in the session rather than in the widget, so it can be tested without a
    display and so a Remora can ask the same question the cursor asks.
    """

    panel: str                          # 'specgram' | 'timeseries' | 'spectra' | 'ltsa'
    time: np.datetime64 | None = None
    frequency: float | None = None       # Hz
    value: float | None = None           # dB, or counts on the time series
    value_label: str = ""
    segment: int | None = None           # raw file the point falls in, 0-based
    source_file: str | None = None        # for LTSA picks: the audio file it came from

    def lines(self) -> list[tuple[str, str]]:
        """Label/text pairs, formatted as ``coorddisp.m`` formats them."""
        out: list[tuple[str, str]] = []
        if self.time is not None:
            out.append(("Time", str(self.time)))
        if self.frequency is not None:
            out.append(("Frequency", f"{self.frequency:.1f} Hz"))
        if self.value is not None:
            out.append((self.value_label or "Value", f"{self.value:.1f}"))
        if self.segment is not None:
            out.append(("Raw file", str(self.segment + 1)))     # 1-based, as MATLAB
        if self.source_file:
            out.append(("File", self.source_file))
        return out


# ----------------------------------------------------------------------- session

class TritonSession:
    """One open dataset and the view onto it.

    Construct directly for a headless or batch job; :func:`open_session` is the
    one-liner that also makes it current.
    """

    def __init__(self) -> None:
        self.audio = AudioState()
        self.view = ViewState()
        self.ltsa = LtsaState()
        self.calibration = CalibrationState()
        self.config = AppConfig()
        #: Per-plugin namespace.  Replaces the ``REMORA`` global; a plugin gets its
        #: own key and cannot reach another's, which is what stops two Remoras
        #: silently clobbering each other's keymaps the way they do today.
        self.plugins: dict[str, Any] = {}

        self._subscribers: list[tuple[str | None, Callable[[str], None]]] = []
        self._depth = 0
        self._pending: list[str] = []

        for name in ("audio", "view", "ltsa", "calibration", "config"):
            getattr(self, name)._attach(self, name)

    # -------------------------------------------------------------- notification

    def subscribe(
        self, callback: Callable[[str], None], prefix: str | None = None
    ) -> Callable[[], None]:
        """Call ``callback(path)`` when a field changes.  Returns an unsubscriber.

        ``prefix`` filters by dotted path, so a frequency-axis widget can subscribe to
        ``"view"`` and ignore LTSA changes. Returning the unsubscriber rather than
        requiring a matching ``unsubscribe(callback)`` means a caller cannot
        accidentally remove the wrong one of two identical bound methods.
        """
        entry = (prefix, callback)
        self._subscribers.append(entry)

        def _off() -> None:
            # Idempotent: unsubscribing twice, or after the session is torn down, is a
            # normal thing for a widget's destructor to do.
            with contextlib.suppress(ValueError):
                self._subscribers.remove(entry)
        return _off

    def _notify(self, path: str) -> None:
        if self._depth:
            self._pending.append(path)
            return
        self._dispatch([path])

    def _dispatch(self, paths: list[str]) -> None:
        for path in paths:
            for prefix, cb in list(self._subscribers):
                if prefix is None or path == prefix or path.startswith(prefix + "."):
                    cb(path)

    class _Batch:
        def __init__(self, session: TritonSession):
            self._s = session

        def __enter__(self) -> TritonSession:
            self._s._depth += 1
            return self._s

        def __exit__(self, *exc: object) -> None:
            self._s._depth -= 1
            if self._s._depth == 0:
                pending, self._s._pending = self._s._pending, []
                # Collapse repeats but keep first-seen order, so a subscriber that
                # redraws on any change redraws once rather than once per field.
                seen: set[str] = set()
                unique = [p for p in pending if not (p in seen or seen.add(p))]
                self._s._dispatch(unique)

    def batch(self) -> _Batch:
        """Group several mutations into one round of notifications.

        Without this, setting ``nfft`` and then ``overlap_pct`` redraws twice, and the
        intermediate redraw uses a combination the analyst never asked for. MATLAB has
        the same problem and solves it by having each ``control.m`` branch call
        ``plot_triton`` exactly once at the end, which is why adding a control means
        remembering to do that.
        """
        return TritonSession._Batch(self)

    # ---------------------------------------------------------------- open / close

    def open_audio(self, path: str | Path, *, channel: int | None = None) -> None:
        """Open a wav, x.wav or flac and position at its start.

        Resets the colour range, matching ``filepd.m``: an explicit open is where a new
        dynamic range should be derived. Advancing to the next file deliberately does
        *not* reset it, because that is the case where comparability matters most.
        """
        path = Path(path)
        src = _audio.open_audio(path, display_channel=(channel or self.view.channel) - 1)
        with self.batch():
            self.audio.source = src
            self.audio.path = path
            self.audio.segment = 0
            self.audio.offset = 0
            if channel is not None:
                self.view.channel = channel
            self.view.clim = None
            if self.view.freq1 is None:
                self.view.freq1 = src.sample_rate / 2
            self.config.last_audio_dir = path.parent

    def open_ltsa(self, path: str | Path) -> None:
        """Open an LTSA and position at its start."""
        path = Path(path)
        src = _ltsa.open_ltsa(path)
        with self.batch():
            self.ltsa.source = src
            self.ltsa.path = path
            self.ltsa.position = src.start_time
            if self.ltsa.freq1 is None:
                self.ltsa.freq1 = src.header.fmax
            self.config.last_ltsa_dir = path.parent

    def close(self) -> None:
        with self.batch():
            self.audio.source = None
            self.audio.path = None
            self.ltsa.source = None
            self.ltsa.path = None

    # ------------------------------------------------------------------ position

    @property
    def source(self) -> _audio.AudioSource:
        if self.audio.source is None:
            raise RuntimeError("no audio open; call open_audio() first")
        return self.audio.source

    def seek(self, t: np.datetime64) -> None:
        """Move to a wall-clock time, converting to a sample index once, here.

        A time in a duty-cycle gap snaps the way ``check_time.m`` does, using the
        current position to decide direction -- forward if the requested time is later
        than where we are, back otherwise.
        """
        src = self.source
        here = self.audio.time if self.audio.is_open else None
        index, snapped = src.resolve_time(t, previous=here, tseg_sec=self.view.tseg_sec)
        offset = src.skip_for(index, snapped)
        with self.batch():
            self.audio.segment = index
            self.audio.offset = max(0, offset)

    def seek_samples(self, segment: int, offset: int) -> None:
        """Move to an exact sample.  The primitive; :meth:`seek` is built on it."""
        src = self.source
        if not 0 <= segment < len(src.segments):
            raise ValidationError(
                f"segment {segment} out of range; this file has {len(src.segments)}"
            )
        with self.batch():
            self.audio.segment = segment
            self.audio.offset = offset

    def step(self, n: int = 1) -> None:
        """Move ``n`` steps forward (or back, if negative).

        A step is ``view.step_sec``, or ``view.tseg_sec`` when that is -1, matching
        ``PARAMS.tseg.step``'s convention.

        The arithmetic is in samples, so this is exact **while the position stays
        inside the file**: ``step(1)`` then ``step(-1)`` returns to the identical
        sample, and ten thousand steps accumulate no error. MATLAB adds and subtracts
        float datenums instead, which guarantees neither.

        Stepping past the end of a segment carries into the next one. Stepping past
        either end of the *file* clamps rather than raising -- an analyst holding a
        motion button expects it to stop, not to throw. Note that clamping is
        necessarily not reversible: once clamped, the information about how far past
        the end the step went is gone, so a matching step back does not return to
        where it came from. That is the correct behaviour and the reason the exactness
        claim above is qualified.
        """
        src = self.source
        step_sec = self.view.tseg_sec if self.view.step_sec == -1 else self.view.step_sec
        delta = int(round(n * step_sec * src.sample_rate))

        seg, off = self.audio.segment, self.audio.offset + delta
        while off < 0 and seg > 0:
            seg -= 1
            off += src.segments[seg].n_samples
        while seg < len(src.segments) - 1 and off >= src.segments[seg].n_samples:
            off -= src.segments[seg].n_samples
            seg += 1
        # Clamp inside the file at both ends. The high end matters as much as the low:
        # without it `offset` runs arbitrarily far past the last segment's length, and
        # the position looks valid while being unreadable.
        off = max(0, min(off, self._max_start_offset(seg)))
        self.seek_samples(seg, off)

    def _max_start_offset(self, segment: int) -> int:
        """The furthest into ``segment`` a window may start.

        Two bounds, and both are needed.

        :func:`_max_offset` is the last *addressable* sample -- which is not the last
        sample present, because of the one-sample-short end time in
        OPEN_DECISIONS §1.5.

        The window must also fit. In the **last** segment there is nothing after it to
        spill into, so a start position within one window of the end produces a short
        read that ``dsp.spectrogram`` rightly refuses. ``check_time.m:34-50`` handles
        this by reporting "too late" and reverting to the previous position; clamping
        to the last position where a full window fits is the same rule expressed as
        motion rather than as an error, and it is what a motion button should do.
        Earlier segments are not bounded this way on purpose: a window there is
        *meant* to spill into the next raw file, which is the splice behaviour
        analysts rely on.
        """
        src = self.source
        cap = _max_offset(src, segment)
        if segment == len(src.segments) - 1:
            window = int(np.ceil(self.view.tseg_sec * src.sample_rate))
            cap = min(cap, src.segments[segment].n_samples - window)
        return max(0, cap)

    def at_start(self) -> bool:
        return (self.audio.segment, self.audio.offset) == (0, 0)

    def at_end(self) -> bool:
        """At the last *addressable* sample, which is not the last sample present.

        Uses the same bound as :meth:`step`'s clamp -- see :func:`_max_offset` for why
        those differ.
        """
        src = self.source
        last = len(src.segments) - 1
        return (self.audio.segment == last
                and self.audio.offset >= self._max_start_offset(last))

    # --------------------------------------------------------------------- tiles

    def frame(self) -> Frame:
        """Read the current window once and compute everything the panels need.

        The single entry point for a repaint. See :class:`Frame` for why it is one call
        rather than one per panel.
        """
        src = self.source
        v = self.view
        start = self.audio.time
        data, boundaries = src.read_at(start, v.tseg_sec, splice_gaps=True)

        # Applied once, here, to the channel being displayed -- not once per panel.
        ch = min(v.channel, data.shape[1])
        if v.filter_on:
            data = data.copy()
            data[:, ch - 1] = dsp.display_filter(
                data[:, ch - 1], src.sample_rate, v.filter_low, v.filter_high
            )
        x = data[:, ch - 1]

        freq1 = v.freq1 if v.freq1 is not None else src.sample_rate / 2
        sg = dsp.spectrogram(x, fs=src.sample_rate, nfft=v.nfft,
                             overlap_pct=v.overlap_pct, freq0=v.freq0, freq1=freq1)

        # The transfer function is calibration rather than display, so it is folded in
        # here and treated as part of the raw values.
        raw_db = sg.db
        tf = self.calibration.curve_for(sg.f)
        if tf is not None:
            raw_db = raw_db + tf[:, None]
        db = self._apply_display(raw_db)

        tile = SpectrogramTile(
            start=start, f=sg.f, t=sg.t, db=db, boundaries=tuple(boundaries),
            clim=v.clim, segment=self.audio.segment, offset=self.audio.offset,
        )

        # The spectra panel is NOT a mean of the spectrogram: plot_spectra.m:32-38
        # detrends first and then calls pwelch, so it is its own computation on the
        # same samples. Cheap, since the samples are already in hand.
        if x.size >= v.nfft:
            noverlap = int(round((v.overlap_pct / 100.0) * v.nfft))
            sp_db = dsp.welch_db(x - x.mean(), fs=src.sample_rate, nfft=v.nfft,
                                 noverlap=noverlap)
            sp_f = np.arange(sp_db.size) * (src.sample_rate / v.nfft)
            if tf is not None:
                sp_db = sp_db + self.calibration.curve_for(sp_f)
        else:
            sp_f = np.empty(0)
            sp_db = np.empty(0)

        return Frame(
            start=start, samples=data, fs=src.sample_rate, spectrogram=tile,
            spectra_f=sp_f, spectra_db=sp_db, boundaries=tuple(boundaries), channel=ch,
            raw_db=raw_db, raw_spectra_db=sp_db,
        )

    def _apply_display(self, raw_db: np.ndarray) -> np.ndarray:
        """Contrast, brightness, and the held colour range.

        ``plot_specgram.m`` scales by contrast as a percentage and then shifts by
        brightness in dB, in that order, before the colour range is taken. Split out
        from :meth:`frame` so :meth:`remap` can redo just this part.
        """
        v = self.view
        db = raw_db * (v.contrast / 100.0) + v.brightness
        if v.clim is None:
            v.clim = _derive_clim(db)
        return db

    def remap(self, frame: Frame) -> Frame:
        """Re-apply the display mapping to an existing frame.

        For when contrast, brightness or the colour range changed and nothing else did.
        No file read and no FFT, so it is roughly three orders of magnitude cheaper
        than :meth:`frame` on a long window -- which is what makes a brightness control
        usable at 200 kHz. The samples, the frequency and time axes and the raw dB are
        all reused unchanged, so this cannot disagree with the frame it came from.
        """
        if frame.raw_db.size == 0:
            return self.frame()
        old = frame.spectrogram
        tile = SpectrogramTile(
            start=old.start, f=old.f, t=old.t, db=self._apply_display(frame.raw_db),
            boundaries=old.boundaries, clim=self.view.clim,
            segment=old.segment, offset=old.offset,
        )
        return Frame(
            start=frame.start, samples=frame.samples, fs=frame.fs, spectrogram=tile,
            spectra_f=frame.spectra_f, spectra_db=frame.raw_spectra_db,
            boundaries=frame.boundaries, channel=frame.channel,
            raw_db=frame.raw_db, raw_spectra_db=frame.raw_spectra_db,
        )

    def spectrogram_tile(self) -> SpectrogramTile:
        """The spectrogram alone.  Convenience over :meth:`frame`.

        Kept because it is the narrow thing most callers and tests want, and because
        the Phase 2 equivalence tests are written against it.
        """
        return self.frame().spectrogram

    def ltsa_tile(self) -> LtsaTile:
        """Read the current LTSA window."""
        if self.ltsa.source is None:
            raise RuntimeError("no LTSA open; call open_ltsa() first")
        src = self.ltsa.source
        st = self.ltsa
        start = st.position if st.position is not None else src.start_time
        db = src.read_block(start, hours=st.tseg_hr)
        f = src.f

        f1 = st.freq1 if st.freq1 is not None else src.header.fmax
        keep = (f >= st.freq0) & (f <= f1)
        f, db = f[keep], db[keep]

        db = db * (st.contrast / 100.0) + st.brightness
        return LtsaTile(start=start, f=f, t=src.bin_times_hours(st.tseg_hr),
                        db=db, clim=_derive_clim(db))

    # ------------------------------------------------------------------- the cursor

    def time_at(self, x_sec: float) -> tuple[np.datetime64, int]:
        """Wall-clock time ``x_sec`` into the current window, and its raw file.

        Not ``window start + x``. The window is read byte-contiguously across raw-file
        boundaries, so on duty-cycled data x is continuous while time jumps -- a point
        after a boundary is later than the naive sum by the whole gap.
        ``coorddisp.m``'s ``get_time_xwav`` handles this by walking the delimiter list;
        this walks the actual segment lengths, which is the same idea done from the
        source data.

        More accurate than the MATLAB in one respect: ``readseg.m:129-131``
        extrapolates its delimiters at a fixed spacing taken from raw file 1, and its
        own comment says it is "assuming that all raw files are the same length". When
        they are not, MATLAB's readout drifts after the second boundary and this does
        not. Same answer whenever the assumption holds.
        """
        src = self.source
        k = int(np.floor(x_sec * src.sample_rate))
        seg = self.audio.segment
        off = self.audio.offset + k
        while seg < len(src.segments) - 1 and off >= src.segments[seg].n_samples:
            off -= src.segments[seg].n_samples
            seg += 1
        ns = int(round(off * 1e9 / src.sample_rate))
        return src.segments[seg].start + np.timedelta64(ns, "ns"), seg

    def probe(self, frame: Frame, panel: str, x: float, y: float) -> Readout:
        """What is at ``(x, y)`` in one of the audio panels.

        ``x`` is seconds from the left edge; ``y`` is Hz for the spectrogram, counts
        for the time series, and Hz for the spectra panel.
        """
        t, seg = self.time_at(x)
        if panel == "timeseries":
            n = int(round(x * frame.fs))
            n = max(0, min(n, frame.samples.shape[0] - 1))
            return Readout("timeseries", time=t,
                           value=float(frame.samples[n, frame.channel - 1]),
                           value_label="Counts", segment=seg)

        if panel == "spectra":
            f, db = frame.spectra_f, frame.spectra_db
            if f.size == 0:
                return Readout("spectra", time=t, segment=seg)
            j = int(np.argmin(np.abs(f - y)))
            return Readout("spectra", time=t, frequency=float(f[j]),
                           value=float(db[j]), value_label="Spectrum level [dB]",
                           segment=seg)

        tile = frame.spectrogram
        # Nearest bin in each axis, as coorddisp.m:307-314 does with its floor(+half)
        # arithmetic. Expressed as a nearest-neighbour search because the axes are
        # already in hand and that cannot go out of range.
        if tile.f.size == 0 or tile.t.size == 0:
            return Readout("specgram", time=t, segment=seg)
        j = int(np.argmin(np.abs(tile.f - y)))
        i = int(np.argmin(np.abs(tile.t - x)))
        return Readout("specgram", time=t, frequency=float(tile.f[j]),
                       value=float(tile.db[j, i]),
                       value_label="Spectrum level [dB]", segment=seg)

    def probe_ltsa(self, tile: LtsaTile, x_hr: float, y_hz: float) -> Readout:
        """What is at ``(x, y)`` in the LTSA panel."""
        if self.ltsa.source is None:
            raise RuntimeError("no LTSA open")
        src = self.ltsa.source
        start = self.ltsa.position or src.start_time
        index, _bin, t = src.locate(start, x_hr, self.ltsa.tseg_hr)
        entry = src.header.entries[index]

        value = None
        if tile.f.size and tile.t.size:
            j = int(np.argmin(np.abs(tile.f - y_hz)))
            i = int(np.argmin(np.abs(tile.t - x_hr)))
            if j < tile.db.shape[0] and i < tile.db.shape[1]:
                value = float(tile.db[j, i])
        return Readout("ltsa", time=t,
                       frequency=float(tile.f[int(np.argmin(np.abs(tile.f - y_hz)))])
                       if tile.f.size else None,
                       value=value, value_label="Spectrum level [dB]",
                       segment=index, source_file=entry.filename)

    def open_from_ltsa(self, x_hr: float, *, search_dirs: list[Path] | None = None
                       ) -> np.datetime64:
        """Open the audio file behind a point in the LTSA and seek to it.

        Port of ``pickxwav.m``, and the reason it works at all is that an LTSA is a
        self-describing index: each directory entry records which audio file its raw
        file came from and when that raw file started, so no searching or filename
        parsing is needed.

        Returns the time sought to. Raises ``FileNotFoundError`` naming the file when
        it cannot be found, so a caller can offer a file dialog -- which is what
        ``pickxwav.m`` does inline, and which does not belong this far down.
        """
        if self.ltsa.source is None:
            raise RuntimeError("no LTSA open")
        src = self.ltsa.source
        start = self.ltsa.position or src.start_time
        index, _bin, t = src.locate(start, x_hr, self.ltsa.tseg_hr)
        entry = src.header.entries[index]

        candidates = [src.path.parent, *(search_dirs or [])]
        if self.config.last_audio_dir:
            candidates.append(self.config.last_audio_dir)
        for d in candidates:
            candidate = Path(d) / entry.filename
            if candidate.exists():
                with self.batch():
                    if (self.audio.path is None
                            or self.audio.path.name != entry.filename):
                        self.open_audio(candidate)
                    self.seek(t)
                    if not self.view.show_specgram:
                        # pickxwav.m:82-83 turns the spectrogram on when nothing that
                        # could show the audio is enabled. Landing on a blank window
                        # after a click reads as the click having failed.
                        self.view.show_specgram = True
                return t
        raise FileNotFoundError(
            f"{entry.filename} is named by {src.path.name} but was not found in "
            f"{', '.join(str(d) for d in candidates)}"
        )

    # -------------------------------------------------------- inspection surfaces

    def as_params(self) -> dict[str, Any]:
        """A nested dict shaped like MATLAB's ``PARAMS``, for people who think in it.

        **Read-only, best-effort, and for inspection only** (decided 2026-08-23; see
        docs/OPEN_DECISIONS.md §4). It is not a compatibility layer -- Remoras are
        modified to take a session. That decision is what lets this be *clearer* than
        the original rather than bug-compatible with it: the quirks catalogued in
        OPEN_DECISIONS §1 are deliberately **not** reproduced here, so a value read
        from this dict may differ from what MATLAB would have had in the same field.
        Where that is true it is noted below.

        Mutating the result does nothing. That is the point: the complaint about
        ``PARAMS`` was never that it was inspectable.
        """
        out: dict[str, Any] = {
            "fs": None, "nch": None, "nBits": None, "ftype": None,
            "nfft": self.view.nfft, "overlap": self.view.overlap_pct,
            "freq0": self.view.freq0, "freq1": self.view.freq1,
            "ch": self.view.channel,
            "bright": self.view.brightness, "contrast": self.view.contrast,
            "tseg": {"sec": self.view.tseg_sec, "step": self.view.step_sec},
        }
        if self.audio.source is not None:
            src = self.audio.source
            out.update(fs=src.sample_rate, nch=src.n_channels,
                       nBits=src.bits_per_sample,
                       ftype=2 if isinstance(src, _audio.XwavSource) else 1)
            out["infile"] = self.audio.path.name if self.audio.path else None
            out["inpath"] = str(self.audio.path.parent) if self.audio.path else None
            out["plot"] = {"dnum": self.audio.time}
            out["start"] = {"dnum": src.start}
            out["end"] = {"dnum": src.end}
            out["raw"] = {
                # NOT MATLAB's shifted datenums: real instants. The 2000-year offset
                # exists to buy float precision (timebase.md 2) and is meaningless
                # once times are integers.
                "dnumStart": [s.start for s in src.segments],
                "dnumEnd": [s.end for s in src.segments],
                "currentIndex": self.audio.segment + 1,      # MATLAB is 1-based
            }
            if isinstance(src, _audio.XwavSource):
                h = src.header
                out["xhd"] = {
                    "WavVersionNumber": h.wav_version,
                    "FirmwareVersionNumber": h.firmware_version,
                    "InstrumentID": h.instrument_id, "SiteName": h.site_name,
                    "ExperimentName": h.experiment_name,
                    "NumOfRawFiles": h.n_raw_files,
                    "Longitude": h.longitude, "Latitude": h.latitude,
                    "Depth": h.depth,
                    # From the header's raw-file table, not from `segments`.
                    # `write_length` is sector bookkeeping that only x.wav has, so it
                    # is deliberately absent from the generic `Segment` -- a plain wav
                    # has no sectors to count.
                    "byte_loc": [r.byte_loc for r in h.raw_files],
                    "byte_length": [r.byte_length for r in h.raw_files],
                    "write_length": [r.write_length for r in h.raw_files],
                    "sample_rate": [r.sample_rate for r in h.raw_files],
                    "gain": [r.gain for r in h.raw_files],
                    # Per raw file, unlike rdxwavhd.m:135, which keeps only the last
                    # (OPEN_DECISIONS §1.3). Strictly more information.
                    "dt": [r.dt for r in h.raw_files],
                }
        if self.ltsa.source is not None:
            h = self.ltsa.source.header
            out["ltsa"] = {
                "infile": h.path.name, "ver": h.version, "tave": h.tave,
                "dfreq": h.dfreq, "fs": h.fs, "nfft": h.nfft, "nf": h.n_freq,
                "nrftot": h.n_raw_total, "nxwav": h.n_xwav, "ch": h.channel,
                "byteloc": list(h.byte_loc), "nave": list(h.n_ave),
                "tseg": {"hr": self.ltsa.tseg_hr},
                "freq0": self.ltsa.freq0, "freq1": self.ltsa.freq1,
            }
        return out

    def snapshot(self) -> dict[str, Any]:
        """State plus provenance, for a bug report.

        The thing that makes a report actionable is not the state but knowing which
        code produced it, so the versions go in too. JSON-safe: times become ISO
        strings and paths become strings, so this can be pasted into an issue.
        """
        import platform
        import sys

        from . import __version__ as triton_version

        def plain(v: Any) -> Any:
            if isinstance(v, np.datetime64):
                return str(v)
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, np.ndarray):
                return {"shape": list(v.shape), "dtype": str(v.dtype)}
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            if isinstance(v, (np.integer, np.floating)):
                return v.item()
            if isinstance(v, (_audio.AudioSource, _ltsa.LtsaSource)):
                return type(v).__name__
            return v

        return {
            "triton": triton_version,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "audio": plain(self.audio.as_dict()),
            "position": {
                "segment": self.audio.segment,
                "offset": self.audio.offset,
                "time": str(self.audio.time) if self.audio.is_open else None,
            },
            "view": plain(self.view.as_dict()),
            "ltsa": plain(self.ltsa.as_dict()),
            "calibration": plain(self.calibration.as_dict()),
            "config": plain(self.config.as_dict()),
            "plugins": sorted(self.plugins),
        }

    def walk(self) -> Iterator[tuple[str, Any]]:
        """Every field as ``(dotted path, value)``.  Backs a Session Inspector panel."""
        for group in ("audio", "view", "ltsa", "calibration", "config"):
            for k, v in getattr(self, group).as_dict().items():
                yield f"{group}.{k}", v

    def __repr__(self) -> str:
        a = self.audio.path.name if self.audio.path else "no audio"
        lt = f", {self.ltsa.path.name}" if self.ltsa.path else ""
        return f"<TritonSession {a}{lt}>"


def _max_offset(src: _audio.AudioSource, segment: int) -> int:
    """The last sample offset in ``segment`` that is actually addressable.

    Not simply ``n_samples - 1``, and the reason is one of the quirks in
    OPEN_DECISIONS §1.5: a segment's declared end is ``start + (byte_length - 2) /
    ByteRate``, one sample short of the data it holds, and ``segment_containing``
    tests ``t < end`` **strictly**. So the final sample present in the file resolves to
    no segment at all, and clamping a motion button there produces a position that
    cannot be read.

    Derived rather than hardcoded as ``n_samples - 2``, because the shortfall depends
    on the relationship between ``byte_length``, ``ByteRate`` and the sample rate,
    which differ between one- and four-channel files and between bit depths.
    """
    seg = src.segments[segment]
    span_ns = seg.end_ns - seg.start_ns
    off = min(seg.n_samples - 1, (span_ns * src.sample_rate) // 1_000_000_000)
    # Walk back to strict inequality. At most a sample or two, and correct whatever
    # the rounding does at the boundary.
    while off > 0 and int(round(off * 1e9 / src.sample_rate)) >= span_ns:
        off -= 1
    return off


def _derive_clim(db: np.ndarray) -> tuple[float, float]:
    """Colour limits from the data, with the guards ``plot_specgram.m`` needs.

    Non-finite values are excluded -- a spectrogram of a silent stretch contains
    ``-inf`` -- and equal limits are separated, because ``caxis`` rejects them. The
    historical ``[1, 65]`` is the fallback when nothing is finite, which is what the
    fixed range used to be before issue #111.
    """
    finite = db[np.isfinite(db)]
    if finite.size == 0:
        return (1.0, 65.0)
    lo, hi = float(finite.min()), float(finite.max())
    return (lo, hi if hi > lo else lo + 1.0)


# --------------------------------------------------------------- the one global seam

_current: TritonSession | None = None


def current_session() -> TritonSession:
    """The process-wide session, created on first use.

    Deliberately *a* global seam -- one, documented and typed -- rather than the forty
    ``global PARAMS`` declarations it replaces. For a console or a plugin's
    convenience. Library code should take a session argument instead, and everything
    in this package does.
    """
    global _current
    if _current is None:
        _current = TritonSession()
    return _current


def set_current_session(session: TritonSession | None) -> None:
    """Set (or clear, with ``None``) the process-wide session."""
    global _current
    _current = session


def open_session(
    audio: str | Path | None = None,
    ltsa: str | Path | None = None,
    *,
    make_current: bool = True,
) -> TritonSession:
    """Build a session, optionally open files in it, optionally make it current."""
    s = TritonSession()
    if audio is not None:
        s.open_audio(audio)
    if ltsa is not None:
        s.open_ltsa(ltsa)
    if make_current:
        set_current_session(s)
    return s
