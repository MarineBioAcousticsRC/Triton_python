"""Reading x.wav headers.

Byte layout and every known inconsistency in the MATLAB implementation are
documented in ``docs/formats/xwav.md``; this module implements that spec. Where
MATLAB and the spec disagree the spec wins, and the difference is noted here.

Deliberate departures from ``rdxwavhd.m``:

* **v2 headers are read.** ``io/ioReadXWAVHeader.m`` has no v2 branch and returns
  wrong ``byte_loc``/``byte_length`` silently on such a file (xwav.md 6.8). The
  per-channel ``drate`` and ``dt`` fields are parsed here.
* **Per-raw-file ``dt`` is kept.** ``rdxwavhd.m`` lines 135 and 137 assign
  ``PARAMS.xhd.dt`` unsubscripted inside the loop, so only the last raw file's
  value survives (xwav.md 6.7). Exposing the whole array is strictly more
  information, so nothing downstream can break by having it; some Remoras are
  believed to compensate for the loss, and that is a Phase 4 audit.
* **24-bit is read.** ``readseg.m`` asks ``fread`` for a nonexistent ``'int24'``
  type, so MATLAB cannot read 24-bit x.wav at all. No such files are believed to
  exist, but supporting it costs nothing.

The sample rate that matters is the **raw-file table** value, not the ``fmt``
chunk's. They differ on real files and Triton uses the former (xwav.md 5.1).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..timebase import _to_datetime64, from_triton_datenum  # noqa: F401

__all__ = ["RawFile", "XwavHeader", "read_xwav_header"]


@dataclass(frozen=True)
class RawFile:
    """One raw file: a contiguous run of samples written by the recorder."""

    index: int
    start: np.datetime64          # true instant, ns precision
    byte_loc: int                 # offset of this run's samples within the file
    byte_length: int              # bytes of sample data, all channels
    write_length: int             # sectors, as the recorder counted them
    sample_rate: int              # Hz, from the raw-file table
    gain: int
    dt: tuple[float, ...] | None = None   # v2 only: per-channel offset from ch1

    @property
    def n_samples(self) -> int:
        """Samples per channel.  Derived, so it cannot disagree with the bytes."""
        return self.byte_length // (self._bytes_per_frame or 1)

    _bytes_per_frame: int = field(default=0, repr=False, compare=False)

    @property
    def duration(self) -> np.timedelta64:
        return np.timedelta64(
            self.n_samples * 1_000_000_000 // self.sample_rate, "ns"
        )

    @property
    def end(self) -> np.datetime64:
        return self.start + self.duration


@dataclass(frozen=True)
class XwavHeader:
    """Everything the header says, with nothing derived away."""

    path: Path
    # fmt chunk
    audio_format: int
    n_channels: int
    fmt_sample_rate: int
    byte_rate: int
    block_align: int
    bits_per_sample: int
    # harp chunk
    wav_version: int
    firmware_version: str
    instrument_id: str
    site_name: str
    experiment_name: str
    disk_sequence_number: int
    disk_serial_number: str
    n_raw_files: int
    longitude: float
    latitude: float
    depth: int
    raw_files: tuple[RawFile, ...]
    drate: tuple[float, ...] | None = None   # v2 only: per-channel true rate
    header_size: int = 0

    @property
    def sample_rate(self) -> int:
        """The rate Triton actually uses: the raw-file table's, not ``fmt``'s."""
        return self.raw_files[0].sample_rate

    @property
    def bytes_per_sample(self) -> int:
        return self.bits_per_sample // 8

    @property
    def start(self) -> np.datetime64:
        return self.raw_files[0].start

    @property
    def end(self) -> np.datetime64:
        return self.raw_files[-1].end


def read_xwav_header(path: str | Path) -> XwavHeader:
    """Parse the RIFF, fmt and harp chunks of an x.wav file.

    Raises ``ValueError`` with a specific reason rather than returning something
    partly filled in: a caller that mistakes a truncated header for a good one
    produces wrong numbers instead of an error, which is worse.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        buf = fh.read(_MAX_HEADER_PROBE)

    if buf[:4] != b"RIFF":
        raise ValueError(f"{path.name}: not a RIFF file (starts with {buf[:4]!r})")
    if buf[8:12] != b"WAVE":
        raise ValueError(f"{path.name}: RIFF file is not WAVE ({buf[8:12]!r})")

    chunks = _walk_chunks(buf, path.name)
    if "fmt " not in chunks:
        raise ValueError(f"{path.name}: no fmt chunk")
    if "harp" not in chunks:
        raise ValueError(f"{path.name}: no harp chunk -- this is a plain wav, not an x.wav")

    fmt_off, fmt_size = chunks["fmt "]
    (audio_format, n_channels, fmt_sample_rate, byte_rate, block_align,
     bits_per_sample) = struct.unpack_from("<HHIIHH", buf, fmt_off)

    harp_off, harp_size = chunks["harp"]
    hdr = _parse_harp(buf, harp_off, harp_size, n_channels, bits_per_sample,
                      path.name)

    data_off, _ = chunks.get("data", (None, None))
    header_size = data_off if data_off is not None else 0

    return XwavHeader(
        path=path,
        audio_format=audio_format,
        n_channels=n_channels,
        fmt_sample_rate=fmt_sample_rate,
        byte_rate=byte_rate,
        block_align=block_align,
        bits_per_sample=bits_per_sample,
        header_size=header_size,
        **hdr,
    )


_MAX_HEADER_PROBE = 1 << 24  # 16 MB: a 500k-raw-file header still fits


def _walk_chunks(buf: bytes, name: str) -> dict[str, tuple[int, int]]:
    """Map chunk id -> (offset of its payload, declared size).

    Walks rather than assuming fixed offsets, because the harp chunk's size varies
    with raw-file count and with header version, and real files have been seen with
    chunks in a different order.
    """
    out: dict[str, tuple[int, int]] = {}
    pos = 12
    while pos + 8 <= len(buf):
        cid = buf[pos:pos + 4].decode("latin-1")
        (size,) = struct.unpack_from("<I", buf, pos + 4)
        payload = pos + 8
        out.setdefault(cid, (payload, size))
        if cid == "data":
            break                     # sample data follows; stop before it
        pos = payload + size + (size & 1)   # RIFF chunks pad to even length
        if size == 0:
            raise ValueError(f"{name}: zero-length {cid!r} chunk, header is corrupt")
    return out


def _wav_version(raw: int, name: str) -> int:
    """Normalise the harp chunk's version byte.

    Real HARP files store this as an **ASCII digit** -- 0x31 for version 1 -- while
    the synthetic Phase 0 fixtures store the integer 1. Both are accepted here and
    reported as an integer.

    This matters beyond parsing. ``rdxwavhd.m:90`` reads the field with ``uchar``
    and ``rdxwavhd.m:101`` then tests ``== 2``, which on a real file holding '2'
    (decimal 50) is false. If v2 files exist in the archive at all, MATLAB has
    never taken its own v2 branch for them and has been silently skipping the
    per-channel ``drate`` and ``dt`` fields -- a stronger version of the gap
    already recorded for ``io/ioReadXWAVHeader.m`` in xwav.md 6.8. Worth settling
    against a real v2 file; see fixtures/README.md.
    """
    if raw in (0, 1, 2):
        return raw
    if raw in (0x30, 0x31, 0x32):          # ASCII '0', '1', '2'
        return raw - 0x30
    raise ValueError(
        f"{name}: unknown WavVersionNumber {raw} "
        f"(neither an integer 0-2 nor an ASCII digit)"
    )


def _parse_harp(buf: bytes, off: int, size: int, n_channels: int,
                bits_per_sample: int, name: str) -> dict:
    """Parse the harp chunk, including the per-raw-file directory."""
    wav_version = _wav_version(buf[off], name)

    fixed = 64 + (4 * n_channels if wav_version == 2 else 0)
    entry = 32 + (4 * n_channels if wav_version == 2 else 0)

    firmware = _text(buf, off + 1, 10)
    instrument = _text(buf, off + 11, 4)
    site = _text(buf, off + 15, 4)
    experiment = _text(buf, off + 19, 8)
    disk_seq = buf[off + 27]
    disk_serial = _text(buf, off + 28, 8)
    (n_raw,) = struct.unpack_from("<H", buf, off + 36)
    (longitude,) = struct.unpack_from("<i", buf, off + 38)
    (latitude,) = struct.unpack_from("<i", buf, off + 42)
    (depth,) = struct.unpack_from("<h", buf, off + 46)

    drate = None
    if wav_version == 2:
        drate = struct.unpack_from(f"<{n_channels}f", buf, off + 48)

    declared = fixed - 8 + n_raw * entry
    if size != declared:
        raise ValueError(
            f"{name}: harp chunk says {size} bytes but {n_raw} raw files at "
            f"version {wav_version} need {declared}"
        )

    bytes_per_frame = n_channels * (bits_per_sample // 8)
    table_off = off + fixed - 8
    raw_files = []
    for k in range(n_raw):
        e = table_off + k * entry
        year, month, day, hour, minute, secs = buf[e:e + 6]
        (ticks,) = struct.unpack_from("<H", buf, e + 6)
        byte_loc, byte_length, write_length, sample_rate = struct.unpack_from(
            "<IIII", buf, e + 8
        )
        gain = buf[e + 24]

        dt = None
        if wav_version == 2:
            dt = struct.unpack_from(f"<{n_channels}f", buf, e + 32)

        raw_files.append(
            RawFile(
                index=k,
                start=_start_from_fields(year, month, day, hour, minute, secs,
                                        ticks, name, k),
                byte_loc=byte_loc,
                byte_length=byte_length,
                write_length=write_length,
                sample_rate=sample_rate,
                gain=gain,
                dt=dt,
                _bytes_per_frame=bytes_per_frame,
            )
        )

    return dict(
        wav_version=wav_version,
        firmware_version=firmware,
        instrument_id=instrument,
        site_name=site,
        experiment_name=experiment,
        disk_sequence_number=disk_seq,
        disk_serial_number=disk_serial,
        n_raw_files=n_raw,
        longitude=longitude / 1e5,
        latitude=latitude / 1e5,
        depth=depth,
        raw_files=tuple(raw_files),
        drate=drate,
    )


def _start_from_fields(year: int, month: int, day: int, hour: int, minute: int,
                       secs: int, ticks: int, name: str, k: int) -> np.datetime64:
    """Build the true start instant from the header's integer fields.

    Built from integers, not by converting a datenum: the header stores whole
    milliseconds, and going via a float would introduce tens of nanoseconds of
    noise that is not in the file.  The 2000-year offset is added here, once, at
    the boundary -- ``year`` on disk is 11 for a 2011 recording.
    """
    try:
        t = datetime(2000 + year, month, day, hour, minute, secs)
    except ValueError as exc:
        raise ValueError(
            f"{name}: raw file {k} has an impossible timestamp "
            f"(year {2000 + year}, month {month}, day {day}, "
            f"{hour:02d}:{minute:02d}:{secs:02d}): {exc}"
        ) from None
    return _to_datetime64(t, extra_ns=ticks * 1_000_000)


def _text(buf: bytes, off: int, n: int) -> str:
    """Fixed-width zero-padded field, as MATLAB's ``char(fread(...))`` would give."""
    return buf[off:off + n].split(b"\x00")[0].decode("latin-1").strip()
