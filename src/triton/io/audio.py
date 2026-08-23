"""Opening an audio source and reading a time-addressed segment from it.

This is the port of ``readseg.m`` and ``check_time.m``, and it is where the
awkward parts of Triton's data model live. An x.wav file is not a single
recording: it is a sequence of *raw files* -- contiguous runs of samples written
by the recorder, each with its own timestamp, sample rate and gain. Duty-cycled
deployments leave real wall-clock gaps between them.

Two facts about how MATLAB Triton reads such a file drive the design here.

**The read is byte-contiguous and runs straight through raw-file boundaries.**
``readseg.m:112-114`` seeks once and reads ``tseg.samp`` frames, so when the
window crosses a boundary the samples after it come from a later wall-clock time
with no gap inserted. Triton then draws red dashed delimiters at the boundaries
(``plot_specgram.m:116-122``). This is the behaviour analysts read spectrograms
against, so it is the **default** here: ``splice_gaps=True``. Honouring gaps is
offered as an option, not as a correction.

**Time is compared in float datenums, and that costs about 39 ns.** At the
shifted epoch a datenum near 4048 has an ulp of 2**-41 days = 39.3 ns
(timebase.md 2). Everything in this module works in exact integer nanoseconds
instead, so our answers are the more accurate ones; where a reference value from
MATLAB is compared against, the tolerance is one quantum of that float rather
than a fixed nanosecond count.

Departures from MATLAB, all deliberate:

* **Unresolvable times raise instead of crashing later.** In ``check_time.m``
  conditions 2b (time past the last raw file) and 3 (time inside two raw files at
  once) leave ``PARAMS.raw.currentIndex`` empty or of length 2. ``readseg.m:108``
  then indexes ``byte_loc`` with it and hands a non-scalar offset to ``fseek``,
  which errors out with nothing that names the cause. Both cases raise
  :class:`TimeNotInData` / :class:`AmbiguousTime` here, from the place that knows
  why.
* **Direction-dependent snapping is a separate call.** ``check_time.m:83-93``
  snaps a time that lands in a gap forward or backward depending on which way the
  analyst was travelling, which it reads from GUI state. That is display
  behaviour, so :meth:`AudioSource.segment_containing` stays pure and
  :meth:`AudioSource.resolve_time` does the snapping when given the previous
  position.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .xwav import XwavHeader, read_xwav_header

__all__ = [
    "AudioSource",
    "XwavSource",
    "WavSource",
    "Segment",
    "Boundary",
    "TimeNotInData",
    "AmbiguousTime",
    "open_audio",
]

_NS_PER_SEC = 1_000_000_000


class TimeNotInData(ValueError):
    """The requested time falls in no segment -- a gap, or past the end."""


class AmbiguousTime(ValueError):
    """The requested time falls in more than one segment.

    ``check_time.m:104-111`` calls this out as a symptom of a wrong sample rate or
    bad header times, and it does happen in field data.
    """


@dataclass(frozen=True)
class Segment:
    """One contiguous run of samples: a *raw file* in x.wav terms.

    A plain wav or flac file has exactly one of these, spanning the whole file.

    ``end`` is computed as ``rdxwavhd.m:160`` does it:
    ``start + (byte_length - 2) / ByteRate``. The ``- 2`` is one 16-bit sample,
    and the divisor is the **fmt chunk** ByteRate, not anything derived from this
    raw file's own ``sample_rate``. Both look like slips and both decide which raw
    file a given time resolves to, so both are reproduced exactly
    (xwav.md 5.2, 6.4).
    """

    index: int
    start: np.datetime64           # exact, from the header's integer fields
    byte_loc: int
    byte_length: int
    sample_rate: int
    gain: int
    n_samples: int                 # per channel
    end: np.datetime64 = np.datetime64("NaT", "ns")

    @property
    def start_ns(self) -> int:
        return _as_ns(self.start)

    @property
    def end_ns(self) -> int:
        return _as_ns(self.end)


@dataclass(frozen=True)
class Boundary:
    """A raw-file boundary inside the plotted window.

    ``PARAMS.raw.delimit_time``: seconds from the left edge of the window. Note
    that entries **past the right edge are normal** -- ``readseg.m:129-131``
    extrapolates one boundary beyond the window on the assumption that all raw
    files are the same length, which is why ``plot_specgram.m:116`` tests for a
    list longer than one rather than for a non-empty list.
    """

    offset_sec: float
    index: int | None = None       # segment whose end this is, when known


class AudioSource:
    """Common surface: a list of segments and time-addressed reads."""

    segments: tuple[Segment, ...]
    sample_rate: int
    n_channels: int
    bits_per_sample: int
    bytes_per_sample: int
    byte_rate: int
    display_channel: int
    path: Path

    # -------------------------------------------------------------------- timing

    @property
    def start(self) -> np.datetime64:
        return self.segments[0].start

    @property
    def end(self) -> np.datetime64:
        return self.segments[-1].end

    def segment_containing(self, t: np.datetime64) -> int:
        """Which segment contains ``t``.  0-based; MATLAB's index minus one.

        Ports ``check_time.m:65-66`` exactly, including its one-sample grace at
        the leading edge::

            find(plot.dnum - raw.dnumStart > -datenum([0 0 0 0 0 1/fs])
                 & plot.dnum < raw.dnumEnd)

        so a time up to one sample period *before* a segment starts still counts
        as inside it, while the trailing comparison is strict.
        """
        t_ns = _as_ns(t)
        grace = _round_div(_NS_PER_SEC, self.sample_rate)
        hits = [
            s.index
            for s in self.segments
            if (t_ns - s.start_ns) > -grace and t_ns < s.end_ns
        ]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise TimeNotInData(self._why_not(t_ns))
        raise AmbiguousTime(
            f"{np.datetime64(t_ns, 'ns')} falls inside {len(hits)} raw files "
            f"({', '.join(str(h) for h in hits)}). This means a raw file's computed "
            f"end is after the next one's start, i.e. the sample rate "
            f"({self.sample_rate} Hz) or the header times are wrong "
            f"(check_time.m condition 3)."
        )

    def _why_not(self, t_ns: int) -> str:
        """Say which of check_time.m's two failure modes this is, and where to go."""
        if t_ns >= self.segments[-1].end_ns:
            return (
                f"{np.datetime64(t_ns, 'ns')} is past the end of the last raw file "
                f"({self.segments[-1].end}). check_time.m:98-102 warns that the sample "
                f"rate may be wrong; here it is {self.sample_rate} Hz."
            )
        for a, b in zip(self.segments, self.segments[1:], strict=False):
            if a.end_ns <= t_ns < b.start_ns:
                gap = (b.start_ns - a.end_ns) / 1e9
                return (
                    f"{np.datetime64(t_ns, 'ns')} falls in the {gap:g} s gap after raw "
                    f"file {a.index}. Duty-cycled data does this; use resolve_time() to "
                    f"snap to raw file {a.index} or {b.index}."
                )
        return f"{np.datetime64(t_ns, 'ns')} is not inside any raw file"

    def resolve_time(
        self,
        t: np.datetime64,
        previous: np.datetime64 | None = None,
        tseg_sec: float = 1.0,
    ) -> tuple[int, np.datetime64]:
        """Resolve ``t`` to a readable position, snapping out of gaps.

        Ports ``check_time.m:81-93``. A time in a gap snaps *forward* to the next
        raw file's start if the analyst was moving forward, and *backward* to
        ``tseg`` before the previous raw file's end if they were moving back --
        which is what makes paging through duty-cycled data land on data rather
        than on silence. ``previous`` stands in for ``PARAMS.save.dnum``; without
        it, forward is assumed, matching a fresh open.
        """
        try:
            return self.segment_containing(t), t
        except AmbiguousTime:
            raise
        except TimeNotInData:
            pass

        t_ns = _as_ns(t)
        prev_ns = _as_ns(previous) if previous is not None else t_ns - 1
        for a, b in zip(self.segments, self.segments[1:], strict=False):
            if a.end_ns <= t_ns < b.start_ns:
                if t_ns > prev_ns:                     # travelling forward
                    return b.index, b.start
                back_ns = a.end_ns - int(
                    round((tseg_sec - 1 / self.sample_rate) * _NS_PER_SEC)
                )
                return a.index, np.datetime64(back_ns, "ns")
        raise TimeNotInData(self._why_not(t_ns))

    # ------------------------------------------------------------------- reading

    def read_at(
        self,
        t: np.datetime64,
        tseg_sec: float,
        splice_gaps: bool = True,
    ) -> tuple[np.ndarray, list[Boundary]]:
        """Read ``tseg_sec`` of audio starting at ``t``.

        Returns ``(data, boundaries)`` where ``data`` is ``(n_samples,
        n_channels)`` float64 and ``boundaries`` are the raw-file delimiters
        Triton draws.

        With ``splice_gaps=True`` (the default, and what MATLAB does) the read is
        byte-contiguous: it crosses raw-file boundaries and the samples after one
        come from a later wall-clock time. With ``splice_gaps=False`` the window
        is laid out on true wall-clock time and gaps are filled with NaN, which no
        MATLAB path does and which therefore has no reference to check against.
        """
        index = self.segment_containing(t)
        n = _tseg_samples(tseg_sec, self.sample_rate)

        if splice_gaps:
            data = self._read_contiguous(self.segments[index], t, n)
        else:
            data = self._read_gap_aware(t, n)

        return self._apply_gain(data), self._boundaries(index, t, tseg_sec)

    def read_samples(self, index: int, skip: int, n_frames: int) -> np.ndarray:
        """Read ``n_frames`` frames starting ``skip`` samples into segment ``index``.

        The honest primitive: sample offsets are exact where times are not. It
        reads straight through the end of the segment, as ``readseg.m:112-114``
        does, so ``skip + n_frames`` may exceed the segment's own length.

        This exists as public API because it is the only way to ask for *exactly*
        the samples a given MATLAB session read. Time-addressed reads cannot do
        that: see :meth:`read_at`.
        """
        frame = self.n_channels * self.bytes_per_sample
        return self._apply_gain(
            self._read_frames(self.segments[index].byte_loc + skip * frame, n_frames)
        )

    def skip_for(self, index: int, t: np.datetime64) -> int:
        """Sample offset into segment ``index`` for time ``t``.  ``readseg.m:111``.

        Exact, where MATLAB's is not -- and that is the one place this port can
        legitimately disagree with it by a whole sample. MATLAB computes
        ``floor((plot.dnum - dnumStart) * 86400 * fs)`` from float datenums
        carrying about 40 ns of representation error (timebase.md 2). Forty
        nanoseconds is a tiny fraction of a sample at any rate Triton handles, so
        it changes the answer **only when the requested time lands within that
        distance of a sample boundary** -- and there the noise decides the floor.
        Measured over the 27 reference cases, 7 land exactly on a boundary and all
        7 fall one sample low in MATLAB.

        There is no way to reproduce that bit-for-bit, and it is worth being clear
        why: ``plot.dnum`` is not recomputed from the requested time, it is
        *accumulated* -- the file's start plus however many forward and backward
        steps the analyst took (``plot_triton.m``). Two routes to the same nominal
        time carry different accumulated error, so the same time in the same file
        can read different samples depending on how the analyst got there. "The
        MATLAB answer" is not a well-defined thing to match.

        So this returns the exact answer, and the port may differ from any given
        MATLAB session by at most one sample, only at exact sample boundaries.
        That is 100 microseconds at 10 kHz and 5 at 200 kHz -- immaterial for
        spectrograms and detection, and it is the *correct* sample for anything
        that cares, such as cross-correlation time-of-arrival work.
        """
        return _skip_samples(_as_ns(t) - self.segments[index].start_ns, self.sample_rate)

    def _read_contiguous(self, seg: Segment, t: np.datetime64, n: int) -> np.ndarray:
        """One seek, one read, straight through any boundary.  ``readseg.m:111-114``."""
        frame = self.n_channels * self.bytes_per_sample
        skip = self.skip_for(seg.index, t)
        return self._read_frames(seg.byte_loc + skip * frame, n)

    def _read_gap_aware(self, t: np.datetime64, n: int) -> np.ndarray:
        """Lay the window out on wall-clock time; NaN where no segment covers it."""
        out = np.full((n, self.n_channels), np.nan)
        filled = np.zeros(n, dtype=bool)
        t_ns = _as_ns(t)
        frame = self.n_channels * self.bytes_per_sample
        for seg in self.segments:
            d = _skip_samples(t_ns - seg.start_ns, self.sample_rate)
            lo = max(0, -d)
            hi = min(n, seg.n_samples - d)
            if hi <= lo:
                continue
            block = self._read_frames(seg.byte_loc + (d + lo) * frame, hi - lo)
            rows = np.arange(lo, lo + block.shape[0])
            # First segment to cover a row wins, so overlapping raw files (the
            # AmbiguousTime case) still give a deterministic answer.
            take = rows[~filled[rows]]
            out[take] = block[take - lo]
            filled[take] = True
        return out

    def _apply_gain(self, data: np.ndarray) -> np.ndarray:
        """``readseg.m:117-119``.

        Two quirks are reproduced. Only ``gain[0]`` is ever used, though the
        header carries one gain per raw file (xwav.md 6.6); and the division is
        applied to column ``PARAMS.ch`` alone -- the channel being displayed --
        so on a multichannel file with real gain the other channels come back
        unscaled. Both look wrong and both are what analysts' existing results
        were produced with, so changing them is a decision for the team rather
        than for the port.
        """
        gain = self.segments[0].gain
        if gain > 0:
            ch = self.display_channel
            data[:, ch] = data[:, ch] / gain
        return data

    def _boundaries(self, index: int, t: np.datetime64, tseg_sec: float) -> list[Boundary]:
        """``readseg.m:120-132``, verbatim including the extrapolation.

        The first entry is where the current raw file ends, in seconds from the
        left edge of the window. If any of the window extends past that, further
        boundaries are extrapolated at a fixed spacing taken from **raw file 1's**
        ``byte_length`` -- the code's own comment says it assumes every raw file
        is the same length, which duty-cycled data does not guarantee.
        """
        seg = self.segments[index]
        first = (seg.end_ns - _as_ns(t)) / 1e9
        out = [Boundary(first, seg.index)]

        bytes_on_plot = tseg_sec * self.byte_rate
        if bytes_on_plot - first * self.byte_rate > 0:      # window spans a boundary
            seconds_per_raw = self.segments[0].byte_length / self.byte_rate
            n_raw = math.ceil((tseg_sec - first) / seconds_per_raw) + 1
            out += [
                Boundary(first + k * seconds_per_raw, None) for k in range(1, n_raw)
            ]
        return out

    # ------------------------------------------------------------------- helpers

    def _read_frames(self, byte_offset: int, n_frames: int) -> np.ndarray:
        """Read ``n_frames`` interleaved frames from ``byte_offset``, as float64.

        Short reads shrink the result rather than zero-filling to the requested
        length, which is what MATLAB's ``fread(fid,[nch,N],dtype)`` does at EOF:
        it returns fewer columns, zero-padding only the final partial one.
        """
        want = n_frames * self.n_channels
        with open(self.path, "rb") as fh:
            fh.seek(byte_offset)
            raw = fh.read(want * self.bytes_per_sample)
        flat = _decode(raw, self.bits_per_sample)
        if flat.size < want:
            frames = -(-flat.size // self.n_channels)       # ceil
            padded = np.zeros(frames * self.n_channels)
            padded[: flat.size] = flat
            flat = padded
            n_frames = frames
        return flat[: n_frames * self.n_channels].reshape(n_frames, self.n_channels)


class XwavSource(AudioSource):
    """An x.wav file: many raw files, one per recorder buffer flush."""

    def __init__(self, header: XwavHeader, display_channel: int = 0):
        self.header = header
        self.path = header.path
        self.n_channels = header.n_channels
        self.bits_per_sample = header.bits_per_sample
        self.bytes_per_sample = header.bits_per_sample // 8
        self.display_channel = display_channel
        # rdxwavhd.m:188 -- the raw-file table's rate, not the fmt chunk's, whose
        # own comment says it "could be fake". A generic WAVE reader gets this
        # wrong on exactly the files where it matters.
        self.sample_rate = header.raw_files[0].sample_rate
        # ...but the *end times* and the boundary spacing divide by the fmt
        # chunk's ByteRate (rdxwavhd.m:160, readseg.m:129). When the two rates
        # disagree -- the fakefs fixture -- these are genuinely different numbers,
        # and using one for the other shifts every delimiter.
        self.byte_rate = header.byte_rate

        frame = self.n_channels * self.bytes_per_sample
        self.segments = tuple(
            Segment(
                index=r.index,
                start=r.start,
                byte_loc=r.byte_loc,
                byte_length=r.byte_length,
                sample_rate=r.sample_rate,
                gain=r.gain,
                n_samples=r.byte_length // frame,
                end=np.datetime64(
                    _as_ns(r.start)
                    + _round_div((r.byte_length - 2) * _NS_PER_SEC, self.byte_rate),
                    "ns",
                ),
            )
            for r in header.raw_files
        )


class WavSource(AudioSource):
    """A plain wav file: one segment, start time taken from the filename.

    ``readseg.m:85-92`` treats wav and flac identically and offsets from
    ``PARAMS.start.dnum``, which ``filepd.m`` fills in from the filename via
    ``wavname2dnum.m``. A file whose name carries no recognised timestamp has no
    absolute time at all, so the Unix epoch stands in and only relative offsets
    mean anything.
    """

    def __init__(
        self,
        path: str | Path,
        start: np.datetime64 | None = None,
        display_channel: int = 0,
    ):
        from ..timebase import parse_filename_time

        self.path = Path(path)
        with wave.open(str(self.path), "rb") as w:
            self.n_channels = w.getnchannels()
            self.bytes_per_sample = w.getsampwidth()
            self.sample_rate = w.getframerate()
            n_frames = w.getnframes()
        self.bits_per_sample = self.bytes_per_sample * 8
        self.byte_rate = self.sample_rate * self.n_channels * self.bytes_per_sample
        self.display_channel = display_channel
        data_off = self._find_data_offset()

        t0 = start if start is not None else parse_filename_time(self.path.name)
        if t0 is None:
            t0 = np.datetime64("1970-01-01T00:00:00", "ns")
        byte_length = n_frames * self.n_channels * self.bytes_per_sample
        self.segments = (
            Segment(
                index=0,
                start=t0,
                byte_loc=data_off,
                byte_length=byte_length,
                sample_rate=self.sample_rate,
                gain=0,
                n_samples=n_frames,
                end=np.datetime64(
                    _as_ns(t0)
                    + _round_div((byte_length - 2) * _NS_PER_SEC, self.byte_rate),
                    "ns",
                ),
            ),
        )

    def _find_data_offset(self) -> int:
        """Byte offset of the data chunk payload.

        Walked rather than assumed: wav files from other tools carry ``LIST``,
        ``fact`` and vendor chunks before ``data``, and a hardcoded 44 silently
        reads metadata as audio.
        """
        with open(self.path, "rb") as fh:
            fh.seek(12)
            while True:
                head = fh.read(8)
                if len(head) < 8:
                    raise ValueError(f"{self.path.name}: no data chunk")
                cid = head[:4]
                (size,) = struct.unpack("<I", head[4:])
                if cid == b"data":
                    return fh.tell()
                fh.seek(size + (size & 1), 1)


def open_audio(path: str | Path, display_channel: int = 0) -> AudioSource:
    """Open an audio file as a time-addressable source.

    Dispatches on the name, as ``filepd.m`` does: ``.x.wav`` is an x.wav,
    anything else is a plain audio file.
    """
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".x.flac"):
        raise NotImplementedError(
            "x.flac needs the harp chunk out of a FLAC APPLICATION block; that is "
            "Phase 2, and the byte_loc question in HANDOFF.md has to be settled first"
        )
    if name.endswith(".x.wav"):
        return XwavSource(read_xwav_header(path), display_channel=display_channel)
    if name.endswith(".flac"):
        raise NotImplementedError("flac needs soundfile; install triton[io] (Phase 2)")
    return WavSource(path, display_channel=display_channel)


# -------------------------------------------------------------------- numerics

def _as_ns(t: np.datetime64) -> int:
    return int(np.datetime64(t, "ns").astype("int64"))


def _round_div(num: int, den: int) -> int:
    """Nearest-integer division, ties away from zero.  Stays in integers."""
    if num >= 0:
        return (num + den // 2) // den
    return -((-num + den // 2) // den)


def _skip_samples(delta_ns: int, fs: int) -> int:
    """``floor((plot.dnum - dnumStart) * 86400 * fs)`` in exact integers.

    Python's ``//`` floors toward negative infinity, which is what MATLAB's
    ``floor`` does, so a negative delta behaves the same in both.
    """
    return (delta_ns * fs) // _NS_PER_SEC


def _tseg_samples(tseg_sec: float, fs: int) -> int:
    """``ceil(tseg.sec * fs)``, ``readseg.m:110``.

    Left in floating point on purpose. The product is float-fragile -- 0.31 is not
    exactly representable, so ``0.31 * 10000`` is not exactly 3100 in decimal --
    but MATLAB and Python both round it the same way in IEEE 754, so reproducing
    the arithmetic reproduces the sample count. Computing it "properly" with
    rationals would return 3101 for that case and disagree with every result the
    lab has ever produced.
    """
    return math.ceil(tseg_sec * fs)


def _decode(raw: bytes, bits: int) -> np.ndarray:
    """Bytes to float64 samples.

    32-bit is read as ``int32`` because ``readseg.m:100`` does, even though
    ``wrxwavhd.m:107`` writes ``AudioFormat = 3`` (IEEE float) for such files.
    One of the two is wrong and it needs a real 32-bit HARP file to settle
    (xwav.md 6.1); until then the reader matches the reader.

    24-bit is implemented, which MATLAB cannot do at all -- ``readseg.m:98`` asks
    ``fread`` for a nonexistent ``'int24'`` type. No 24-bit x.wav is believed to
    exist, so this path is unchecked against any reference and was nearly free.
    """
    if bits == 16:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if bits == 32:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64)
    if bits == 8:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
    if bits == 24:
        n = len(raw) // 3
        b = np.frombuffer(raw[: n * 3], dtype=np.uint8).reshape(n, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        return np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64)
    raise ValueError(f"{bits}-bit samples are not supported")
