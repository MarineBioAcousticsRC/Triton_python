"""Reading Long-Term Spectral Average files.

An LTSA is a pre-computed, heavily reduced spectrogram: one averaged spectrum per
``tave`` seconds, quantised to signed bytes. A year of continuous 200 kHz recording
becomes a file small enough to scroll through, which is the only reason browsing a
deployment at that scale is possible at all.

Three things about the format shape this module.

**Field widths depend on the version, in the middle of the struct.** ``nrftot`` is
uint16 in v1/v2 and uint32 in v3/v4; ``nave`` likewise; the filename field is 40
bytes through v3 and 80 in v4; ``rfileid`` is one byte through v3 and four in v4.
Get any of them wrong and every subsequent field is read from the wrong offset, so
the widths are derived once, from the version, in :func:`_layout`.

**``nxwav`` stayed uint16 when ``nrftot`` was widened.** So a v4 file can describe
four billion raw files but only 65,535 source audio files. That is the cap behind
the "too many files in one LTSA" problem, and it is a format limit, not a bug in the
writer — see ltsa.md.

**A truncated LTSA reads as plausible data unless checked.** If generation was
interrupted, the header still promises the full spectrum count. MATLAB's
``read_ltsadata`` seeks past the end, and a failed ``fseek`` used to fall through to
``fread``, returning whatever the file position happened to hold — real-looking
values from the wrong time. Both the header check and the NaN-on-failed-seek
behaviour are reproduced here.
"""

from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..timebase import _to_datetime64

__all__ = [
    "LtsaEntry",
    "LtsaHeader",
    "LtsaSource",
    "read_ltsa_header",
    "open_ltsa",
]

_NS_PER_SEC = 1_000_000_000


def _as_ns_local(t: np.datetime64) -> int:
    return int(np.datetime64(t, "ns").astype("int64"))


@dataclass(frozen=True)
class _Layout:
    """Version-dependent field widths and offsets."""

    nrftot_fmt: str
    nave_fmt: str
    fname_len: int
    rfileid_fmt: str
    entry_size: int
    nxwav_off: int
    ch_off: int


def _layout(version: int) -> _Layout:
    """Field widths for a header version (ltsa.md 2 and 3).

    Both header and directory-entry geometry change at v3, and the changes are
    interleaved with unchanged fields rather than appended, so this cannot be
    handled by reading a common prefix.

    The two offsets are worth spelling out, because getting one wrong is silent --
    it reads a padding byte, which is zero, and zero is a plausible channel number::

        v1/v2   nrftot uint16 @32-33   nxwav @34-35   ch @36   + 27 pad = 64
        v3/v4   nrftot uint32 @32-35   nxwav @36-37   ch @38   + 25 pad = 64

    ``_check_layout`` below asserts both add to 64, so an edit that breaks the
    arithmetic fails at import rather than returning zeros at run time.
    """
    if version in (1, 2):
        return _Layout("<H", "<H", 40, "<B", 64, 34, 36)
    if version == 3:
        return _Layout("<I", "<I", 40, "<B", 64, 36, 38)
    if version == 4:
        return _Layout("<I", "<I", 80, "<I", 104, 36, 38)
    raise ValueError(
        f"unknown LTSA version {version}; the format defines 1 through 4 and the "
        f"writer only emits 4"
    )


def _check_layout() -> None:
    """Assert the header and entry geometry of every version adds up to its size.

    Cheap insurance against exactly the mistake this function invites: an offset
    that is one byte out reads padding, padding is zero, and zero passes for a real
    value. Runs once at import.
    """
    for ver, entry_total in ((1, 64), (2, 64), (3, 64), (4, 104)):
        lay = _layout(ver)
        nrftot_w = struct.calcsize(lay.nrftot_fmt)
        # header: 32 fixed bytes, then nrftot, nxwav (2), ch (1), then padding to 64
        assert lay.nxwav_off == 32 + nrftot_w, f"v{ver} nxwav offset"
        assert lay.ch_off == lay.nxwav_off + 2, f"v{ver} ch offset"
        pad = 64 - (lay.ch_off + 1)
        assert pad == (27 if ver in (1, 2) else 25), f"v{ver} header padding is {pad}"
        # entry: 6 date bytes + ticks(2) + byteloc(4) + nave + fname + rfileid + pad
        used = 6 + 2 + 4 + struct.calcsize(lay.nave_fmt) + lay.fname_len \
            + struct.calcsize(lay.rfileid_fmt)
        assert lay.entry_size == entry_total, f"v{ver} entry size"
        assert entry_total - used == (9 if ver in (1, 2) else 7 if ver == 3 else 4), \
            f"v{ver} entry padding is {entry_total - used}"


_check_layout()


@dataclass(frozen=True)
class LtsaEntry:
    """One directory entry: the spectra computed from one raw file."""

    index: int
    start: np.datetime64
    byte_loc: int              # absolute offset of this raw file's first spectrum
    n_ave: int                 # time bins
    filename: str              # source audio file
    raw_file_id: int           # index of the raw file within that audio file
    duration_sec: float
    end: np.datetime64

    @property
    def start_ns(self) -> int:
        return int(np.datetime64(self.start, "ns").astype("int64"))

    @property
    def end_ns(self) -> int:
        return int(np.datetime64(self.end, "ns").astype("int64"))


@dataclass(frozen=True)
class LtsaHeader:
    """Everything the LTSA header and directory say."""

    path: Path
    version: int
    dir_start_loc: int
    data_start_loc: int
    tave: float                # seconds per time bin
    dfreq: float               # Hz per frequency bin
    fs: int
    nfft: int
    n_raw_total: int
    n_xwav: int
    channel: int
    entries: tuple[LtsaEntry, ...]
    missing_bytes: int = 0     # >0 if the file is shorter than the header promises

    # ---- convenience views, in the order the directory stores them

    @property
    def byte_loc(self) -> tuple[int, ...]:
        return tuple(e.byte_loc for e in self.entries)

    @property
    def n_ave(self) -> tuple[int, ...]:
        return tuple(e.n_ave for e in self.entries)

    @property
    def n_freq(self) -> int:
        """Frequency bins per spectrum.  ``read_ltsahead.m:172-176``."""
        return (self.nfft + 1) // 2 if self.nfft % 2 else self.nfft // 2 + 1

    @property
    def fmax(self) -> float:
        """``read_ltsahead.m:205`` -- derived from ``nf`` and ``dfreq``, not ``fs/2``.

        The commented-out line above it in the MATLAB shows the older
        ``floor(fs/2)`` definition. The two agree when ``dfreq == fs/nfft`` and can
        disagree otherwise, so this follows the live one.
        """
        return (self.n_freq - 1) * self.dfreq

    @property
    def freq(self) -> np.ndarray:
        return np.arange(self.n_freq, dtype=np.float64) * self.dfreq

    @property
    def start(self) -> np.datetime64:
        return self.entries[0].start

    @property
    def end(self) -> np.datetime64:
        return self.entries[-1].end

    @property
    def is_complete(self) -> bool:
        return self.missing_bytes == 0


def read_ltsa_header(path: str | Path) -> LtsaHeader:
    """Parse an LTSA's 64-byte header and its directory.

    Warns rather than raises when the file is shorter than the header promises: the
    header itself is still valid and worth having, and the caller may only want
    metadata. Reading data from such a file is where it becomes an error, and
    :meth:`LtsaSource.read_block` returns NaN there rather than plausible-looking
    values from the wrong time.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(64)
        if len(head) < 64:
            raise ValueError(f"{path.name}: only {len(head)} bytes, too short for a header")
        if head[:4] != b"LTSA":
            raise ValueError(f"{path.name}: not an LTSA file (starts with {head[:4]!r})")

        version = head[4]
        lay = _layout(version)

        (dir_start_loc,) = struct.unpack_from("<I", head, 8)
        (data_start_loc,) = struct.unpack_from("<I", head, 12)
        (tave,) = struct.unpack_from("<f", head, 16)
        (dfreq,) = struct.unpack_from("<f", head, 20)
        (fs,) = struct.unpack_from("<I", head, 24)
        (nfft,) = struct.unpack_from("<I", head, 28)
        (n_raw_total,) = struct.unpack_from(lay.nrftot_fmt, head, 32)
        (n_xwav,) = struct.unpack_from("<H", head, lay.nxwav_off)
        channel = head[lay.ch_off]

        # dir_start_loc is 1-based, unlike data_start_loc (ltsa.md 2).
        fh.seek(dir_start_loc - 1)
        dir_bytes = fh.read(n_raw_total * lay.entry_size)

    if len(dir_bytes) < n_raw_total * lay.entry_size:
        raise ValueError(
            f"{path.name}: directory is truncated -- {n_raw_total} entries at "
            f"{lay.entry_size} bytes need {n_raw_total * lay.entry_size}, found "
            f"{len(dir_bytes)}"
        )

    tave = float(tave)
    dfreq = float(dfreq)
    tick_ns = _NS_PER_SEC // fs if fs else 0
    entries = []
    for k in range(n_raw_total):
        off = k * lay.entry_size
        year, month, day, hour, minute, secs = dir_bytes[off:off + 6]
        (ticks,) = struct.unpack_from("<H", dir_bytes, off + 6)
        (byte_loc,) = struct.unpack_from("<I", dir_bytes, off + 8)
        (n_ave,) = struct.unpack_from(lay.nave_fmt, dir_bytes, off + 12)
        name_off = off + 12 + struct.calcsize(lay.nave_fmt)
        fname = dir_bytes[name_off:name_off + lay.fname_len]
        fname = fname.split(b"\x00")[0].decode("latin-1").strip()
        (rfileid,) = struct.unpack_from(lay.rfileid_fmt, dir_bytes,
                                        name_off + lay.fname_len)

        start = _entry_start(year, month, day, hour, minute, secs, ticks, path.name, k)
        # read_ltsahead.m:155-159. The end subtracts one *sample* period, 1/fs, not
        # one time bin -- which looks like a slip and decides which raw file a
        # requested time resolves to, so it is reproduced exactly (ltsa.md 3.1).
        dur_sec = tave * n_ave
        end_ns = (
            int(np.datetime64(start, "ns").astype("int64"))
            + int(round(dur_sec * _NS_PER_SEC))
            - tick_ns
        )
        entries.append(
            LtsaEntry(
                index=k, start=start, byte_loc=byte_loc, n_ave=n_ave,
                filename=fname, raw_file_id=rfileid, duration_sec=dur_sec,
                end=np.datetime64(end_ns, "ns"),
            )
        )

    n_freq = (nfft + 1) // 2 if nfft % 2 else nfft // 2 + 1
    expected = data_start_loc + sum(e.n_ave for e in entries) * n_freq
    actual = path.stat().st_size
    missing = max(0, expected - actual)
    if missing:
        total = expected - data_start_loc
        warnings.warn(
            f"{path.name} is incomplete: the header declares "
            f"{sum(e.n_ave for e in entries)} spectral averages needing {expected} "
            f"bytes, the file holds {actual}, so "
            f"{100 * missing / total:.1f}% of the spectra are missing. It was "
            f"probably interrupted while being made; consider remaking it.",
            stacklevel=2,
        )

    return LtsaHeader(
        path=path, version=version, dir_start_loc=dir_start_loc,
        data_start_loc=data_start_loc, tave=tave, dfreq=dfreq, fs=fs, nfft=nfft,
        n_raw_total=n_raw_total, n_xwav=n_xwav, channel=channel,
        entries=tuple(entries), missing_bytes=missing,
    )


def _entry_start(year, month, day, hour, minute, secs, ticks, name, k) -> np.datetime64:
    """Directory timestamp to a true instant.

    Built from the integer fields with the 2000-year offset added here, once, so no
    float enters the value -- same rule as the x.wav reader.
    """
    try:
        t = datetime(2000 + year, month, day, hour, minute, secs)
    except ValueError as exc:
        raise ValueError(
            f"{name}: directory entry {k} has an impossible timestamp "
            f"(year {2000 + year}, month {month}, day {day}, "
            f"{hour:02d}:{minute:02d}:{secs:02d}): {exc}"
        ) from None
    return _to_datetime64(t, extra_ns=ticks * 1_000_000)


class LtsaSource:
    """An open LTSA, addressable by time."""

    def __init__(self, header: LtsaHeader):
        self.header = header
        self.path = header.path

    # -------------------------------------------------------------------- timing

    @property
    def start_time(self) -> np.datetime64:
        return self.header.start

    @property
    def end_time(self) -> np.datetime64:
        return self.header.end

    @property
    def f(self) -> np.ndarray:
        return self.header.freq

    def bins_for(self, hours: float) -> int:
        """``nbin = floor(tseg.hr * 3600 / tave)``, ``read_ltsadata.m:17``."""
        return int(np.floor(hours * 3600.0 / self.header.tave))

    def entry_containing(self, t: np.datetime64) -> tuple[int, np.datetime64]:
        """Which directory entry a time falls in, and the possibly-snapped time.

        Ports ``read_ltsadata.m:23-32``. Two things worth noticing about the
        MATLAB, both reproduced:

        * The test is ``plot.dnum >= dnumStart & plot.dnum + tave <= dnumEnd``, so a
          time in the last ``tave`` of an entry does **not** match it -- there has
          to be room for a whole time bin.
        * When nothing matches, it snaps *forward* to the first entry that starts at
          or after the requested time, unconditionally. There is no
          direction-dependent choice as there is for x.wav, so a request landing in
          a duty-cycle gap always jumps ahead, never back.
        """
        t_ns = int(np.datetime64(t, "ns").astype("int64"))
        tave_ns = int(round(self.header.tave * _NS_PER_SEC))

        for e in self.header.entries:
            if t_ns >= e.start_ns and t_ns + tave_ns <= e.end_ns:
                return e.index, t

        for e in self.header.entries:
            if t_ns <= e.start_ns:
                return e.index, e.start
        raise ValueError(
            f"{np.datetime64(t_ns, 'ns')} is past the end of "
            f"{self.path.name} ({self.header.end})"
        )

    # -------------------------------------------------------------------- reading

    def read_block(self, start: np.datetime64, hours: float) -> np.ndarray:
        """Read ``hours`` of spectra starting at ``start``.

        Returns ``(n_freq, n_bins)`` float64, matching MATLAB's orientation and its
        ``fread`` return type. Like the x.wav reader, this runs straight through
        entry boundaries: the read is byte-contiguous, so on duty-cycled data the
        columns after a boundary come from a later wall-clock time with no gap
        inserted.

        A short file gives fewer columns rather than an error, which is what
        ``fread(fid,[nf,nbin],'int8')`` does at EOF. A seek that lands past the end
        of the file gives all-NaN, deliberately: the pre-2026 MATLAB fell through to
        ``fread`` there and returned whatever the file pointer was sitting on, which
        looks exactly like real spectra from the wrong time.
        """
        hdr = self.header
        nbin = self.bins_for(hours)
        nf = hdr.n_freq

        index, snapped = self.entry_containing(start)
        e = hdr.entries[index]

        # read_ltsadata.m:36-38, in exact integers rather than via float datenums.
        tave_ns = int(round(hdr.tave * _NS_PER_SEC))
        t_ns = int(np.datetime64(snapped, "ns").astype("int64"))
        bin_index = (t_ns - e.start_ns) // tave_ns          # 0-based; MATLAB adds 1
        skip = e.byte_loc + bin_index * nf

        size = self.path.stat().st_size
        if skip >= size:
            # MATLAB reaches this through a failed fseek (read_ltsadata.m:44-63).
            return np.full((nf, nbin), np.nan)

        with open(self.path, "rb") as fh:
            fh.seek(skip)
            raw = fh.read(nf * nbin)
        vals = np.frombuffer(raw, dtype=np.int8).astype(np.float64)

        # MATLAB's fread([nf,nbin],...) fills column-major and returns whole columns
        # only, zero-padding the final partial one.
        cols = -(-vals.size // nf)
        if vals.size < cols * nf:
            padded = np.zeros(cols * nf)
            padded[: vals.size] = vals
            vals = padded
        return vals.reshape((nf, cols), order="F")

    @property
    def total_bins(self) -> int:
        """Every time bin in the file, across all entries."""
        return sum(e.n_ave for e in self.header.entries)

    def bin_index(self, t: np.datetime64) -> int:
        """Global bin number containing ``t``, counting continuously across entries.

        The LTSA's x axis is bins laid side by side, so a *global* bin number is the
        natural coordinate for navigation: stepping is then addition, and the
        raw-file walk happens once in :meth:`time_of_bin` rather than at every call
        site. Clamps into the file.
        """
        tave_ns = int(round(self.header.tave * _NS_PER_SEC))
        t_ns = _as_ns_local(t)
        base = 0
        for e in self.header.entries:
            if t_ns < e.start_ns:
                return base                       # in the gap before this entry
            within = (t_ns - e.start_ns) // tave_ns
            if within < e.n_ave:
                return base + int(within)
            base += e.n_ave
        return max(0, self.total_bins - 1)

    def time_of_bin(self, k: int) -> np.datetime64:
        """Start time of global bin ``k``.  The inverse of :meth:`bin_index`."""
        k = max(0, min(int(k), max(self.total_bins - 1, 0)))
        tave_ns = int(round(self.header.tave * _NS_PER_SEC))
        for e in self.header.entries:
            if k < e.n_ave:
                return np.datetime64(e.start_ns + k * tave_ns, "ns")
            k -= e.n_ave
        last = self.header.entries[-1]
        return np.datetime64(last.start_ns + (last.n_ave - 1) * tave_ns, "ns")

    def advance_bins(self, t: np.datetime64, n_bins: int) -> np.datetime64:
        """Move ``n_bins`` through the data from ``t``, skipping duty-cycle gaps.

        Port of ``stepPlotTimeLTSA.m``, which is what ``motion_ltsa.m`` uses when the
        step is -1 -- the default. Stepping by *bins* rather than by hours is the
        difference between paging through the data and paging through the clock: on a
        duty-cycled deployment an hour of wall-clock may contain a few minutes of
        recording, so an hours-based step lands on empty windows while a bins-based one
        always shows data.
        """
        return self.time_of_bin(self.bin_index(t) + n_bins)

    def locate(self, start: np.datetime64, hours_from_left: float,
               hours: float) -> tuple[int, int, np.datetime64]:
        """Which entry and time bin a point in a plotted window falls in.

        Port of ``getIndexBin.m``. Returns ``(entry index, 0-based bin within that
        entry, the bin's centre time)``.

        The walk is the point. An LTSA plot's x axis is bins, laid side by side with
        no gaps, but consecutive entries can be separated by hours of duty-cycle
        silence -- so x maps to time **piecewise**, one linear stretch per raw file,
        and a naive ``start + x`` is wrong by the accumulated gap for any point after
        the first boundary. This consumes ``n_ave`` bins per entry until the requested
        bin is inside one.

        The half-bin offset matches ``getIndexBin.m:21``: the plot is an image whose
        first column is *centred* on the first bin, so a click at x=0 is the middle of
        bin 0 rather than its leading edge. Clamps at the last entry rather than
        raising, because a click a pixel past the end of the data is an ordinary thing
        for a user to do.
        """
        tbinsz = self.header.tave / 3600.0
        if tbinsz <= 0:
            raise ValueError(f"{self.path.name}: tave is {self.header.tave}")
        n_bins = self.bins_for(hours)
        cursor = int(np.floor(hours_from_left / tbinsz))
        cursor = max(0, min(cursor, max(n_bins - 1, 0)))

        index, first_bin = self.entry_containing(start)
        # entry_containing returns the entry; the bin the window starts at within it
        # is derived the same way read_block derives it.
        tave_ns = int(round(self.header.tave * _NS_PER_SEC))
        e = self.header.entries[index]
        start_bin = (int(np.datetime64(first_bin, "ns").astype("int64"))
                     - e.start_ns) // tave_ns

        remaining = self.header.entries[index].n_ave - start_bin
        if cursor < remaining:
            bin_in_entry = start_bin + cursor
        else:
            cursor -= remaining
            index += 1
            while (index < len(self.header.entries)
                   and cursor >= self.header.entries[index].n_ave):
                cursor -= self.header.entries[index].n_ave
                index += 1
            if index >= len(self.header.entries):
                index = len(self.header.entries) - 1
                cursor = self.header.entries[index].n_ave - 1
            bin_in_entry = cursor

        entry = self.header.entries[index]
        centre_ns = entry.start_ns + int(round((bin_in_entry + 0.5) * tave_ns))
        return index, bin_in_entry, np.datetime64(centre_ns, "ns")

    def bin_times_hours(self, hours: float) -> np.ndarray:
        """Bin centres in hours from the window's left edge.

        ``read_ltsadata.m:72``: ``0.5*tbinsz : tbinsz : (nbin-0.5)*tbinsz``. The
        MATLAB comment notes this is "only good for continuous data" -- the axis is
        a uniform grid regardless of any gaps the columns actually span.
        """
        nbin = self.bins_for(hours)
        tbinsz = self.header.tave / 3600.0
        return (np.arange(nbin) + 0.5) * tbinsz


def open_ltsa(path: str | Path) -> LtsaSource:
    """Open an LTSA for time-addressed reading."""
    return LtsaSource(read_ltsa_header(path))
