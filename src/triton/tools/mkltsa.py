"""Building an LTSA: the port of ``get_headers`` + ``ck_ltsaparams`` + ``write_ltsahead``
+ ``calc_ltsa``.

This is Phase 1's milestone, and the reason it is the milestone is that byte-identical
output is the only test that pins down the whole numeric stack at once -- the window
definition, the PSD normalisation, the int8 quantiser, *and* the parameter derivation.
The readers cannot check that last part at all: they read ``nfft``, ``tave`` and
``nave`` out of a header instead of computing them.

Four behaviours here look like mistakes and are reproduced deliberately, because
existing LTSAs were produced with them and a reader has to agree with the writer.
Each is marked BUG-COMPATIBLE at its site.

1. **The last spectral average of a raw file re-reads earlier data.**
   ``calc_ltsa.m:99`` advances the input file pointer by the *current* average's
   sample count, but the distance to advance is the *previous* average's length. They
   agree for every average except the last one of a raw file that does not divide
   evenly, where the last one is short -- so the pointer lands early and the final
   average overlaps the one before it. It never reads outside the raw file, so there
   is no visible corruption; the last time bin of each raw file is just not the data
   it claims to be. Exercised by the 10 kHz fixtures (``nave`` 3 against an exact 2.5).

2. **``fs`` comes from the ``fmt`` chunk here, not the raw-file table.** Everywhere
   else in Triton the raw-file table wins, and ``rdxwavhd.m``'s own comment says the
   ``fmt`` value "could be fake". LTSA creation is the documented exception
   (``ck_ltsaparams.m:20-21``): it takes the ``fmt`` rate and then *aborts* if any raw
   file disagrees. So a file the display path reads happily can be refused here.

3. **100 spare directory slots are reserved.** ``dataStartLoc`` allows for
   ``nrftot + 100`` entries and the unused ones are zero-filled
   (``write_ltsahead.m:41,148-150``), so a four-raw-file LTSA holding 612 bytes of
   spectra is 11,492 bytes on disk. Presumably for appending, which nothing does.

4. **The filename field is padded twice.** ``get_headers.m:113`` copies
   ``fnsz(2)`` characters, the width of the input filename matrix, which MATLAB
   space-pads to the longest name; the field is then NUL-padded to 80. So with inputs
   of differing name lengths the shorter names carry trailing spaces before the NULs.

The ``nxwav`` ceiling is a format limit rather than a bug, and it is the one worth
fixing rather than reproducing: see :func:`plan`.
"""

from __future__ import annotations

import argparse
import itertools
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..dsp import to_int8_db, welch_db
from ..io.xwav import read_xwav_header

__all__ = ["LtsaPlan", "RawEntry", "plan", "build", "main"]

#: ``write_ltsahead.m:43-46``
_HEADER_SIZE = 64
_ENTRY_SIZE = 64 + 40          # v4: 40 extra bytes for 80-character filenames
_SPARE_ENTRIES = 100           # reserved, never used -- see module docstring
_NXWAV_MAX = 65535             # the uint16 field, see plan()
_BYTELOC_MAX = 2**32           # write_ltsahead.m:131


@dataclass(frozen=True)
class RawEntry:
    """One raw file's contribution: where to read it, and what to write about it."""

    source: Path
    rfileid: int               # 1-based index within its own x.wav
    year: int                  # as stored, i.e. year - 2000
    month: int
    day: int
    hour: int
    minute: int
    secs: int
    ticks: int
    byte_loc: int
    byte_length: int
    write_length: int
    sample_rate: int
    n_ave: int = 0             # filled in by plan()
    byte_loc_out: int = 0      # filled in by plan()


@dataclass
class LtsaPlan:
    """Every derived number, computed before a byte is written.

    Separating the derivation from the writing is not tidiness: the derivation is the
    part with no other test, so it is worth being able to inspect and assert on
    without producing a file.
    """

    inputs: tuple[Path, ...]
    fs: int
    n_channels: int
    bits_per_sample: int
    channel: int               # 1-based, as PARAMS.ltsa.ch
    blksz: int
    tave: float
    tave_requested: float
    dfreq: float
    nfft: int
    cfact: float
    nfreq: int
    version: int
    entries: list[RawEntry] = field(default_factory=list)
    name_width: int = 80

    @property
    def n_xwav(self) -> int:
        return len(self.inputs)

    @property
    def n_raw_total(self) -> int:
        return len(self.entries)

    @property
    def dir_start_loc(self) -> int:
        return _HEADER_SIZE + 1                      # 1-based, unlike data_start_loc

    @property
    def data_start_loc(self) -> int:
        return _ENTRY_SIZE * (self.n_raw_total + _SPARE_ENTRIES) + _HEADER_SIZE

    @property
    def samples_per_average(self) -> int:
        """``sampPerAve``, ``calc_ltsa.m:16``.

        Required to be a whole number of samples. MATLAB lets a fractional value
        through into ``fread``'s size argument, where it does something
        implementation-defined; refusing is better than guessing which.
        """
        exact = self.tave * self.fs
        n = int(round(exact))
        if abs(exact - n) > 1e-9:
            raise ValueError(
                f"tave={self.tave} at {self.fs} Hz gives {exact} samples per average, "
                f"which is not a whole number. Choose a tave that divides evenly."
            )
        return n

    @property
    def expected_size(self) -> int:
        return self.data_start_loc + sum(e.n_ave for e in self.entries) * self.nfreq


def _gather(inputs: list[Path]) -> tuple[list[RawEntry], int, int, int, int]:
    """Port of ``get_headers.m`` for the x.wav case.

    Returns the flattened raw-file list plus the geometry taken from the first file.
    Reuses :func:`read_xwav_header` rather than re-walking the chunks, so the writer
    and the reader cannot drift apart in how they interpret a header.
    """
    entries: list[RawEntry] = []
    fmt_fs = n_channels = bits = None
    name_width = max(len(p.name) for p in inputs)

    for path in inputs:
        hdr = read_xwav_header(path)
        if fmt_fs is None:
            fmt_fs, n_channels, bits = hdr.fmt_sample_rate, hdr.n_channels, hdr.bits_per_sample
        elif (hdr.n_channels, hdr.bits_per_sample) != (n_channels, bits):
            raise ValueError(
                f"{path.name} has {hdr.n_channels} channels at {hdr.bits_per_sample} "
                f"bits; the first file has {n_channels} at {bits}. One LTSA cannot "
                f"mix them."
            )
        for r in hdr.raw_files:
            entries.append(RawEntry(
                source=path, rfileid=r.index + 1,
                year=int(str(r.start.astype("datetime64[Y]"))[:4]) - 2000,
                month=int(str(r.start.astype("datetime64[M]"))[5:7]),
                day=int(str(r.start.astype("datetime64[D]"))[8:10]),
                hour=int(str(r.start.astype("datetime64[h]"))[11:13]),
                minute=int(str(r.start.astype("datetime64[m]"))[14:16]),
                secs=int(str(r.start.astype("datetime64[s]"))[17:19]),
                ticks=int(
                    (r.start - r.start.astype("datetime64[s]"))
                    / np.timedelta64(1, "ms")
                ),
                byte_loc=r.byte_loc, byte_length=r.byte_length,
                write_length=r.write_length, sample_rate=r.sample_rate,
            ))
    return entries, fmt_fs, n_channels, bits, name_width


def plan(
    inputs: list[str | Path],
    tave: float,
    dfreq: float,
    channel: int = 1,
) -> LtsaPlan:
    """Derive every LTSA parameter, without writing anything.

    Ports ``ck_ltsaparams.m`` and the derivation half of ``write_ltsahead.m``.

    Raises rather than returning a half-built plan. MATLAB's equivalents ``disp`` a
    message and ``return``, leaving ``PARAMS`` partly updated and the caller none the
    wiser -- ``ck_ltsaparams.m:32`` on mismatched sample rates is the clearest case,
    since execution continues into ``write_ltsahead`` with a stale ``fs``.

    On too many input files: the ``nxwav`` header field is **uint16**, so an LTSA can
    name at most 65535 source files no matter how many raw files it contains --
    ``nrftot`` was widened to uint32 in v4 and this was not. It is a format limit, not
    a writer bug, which is why the fix is to split the work across several LTSAs
    rather than to widen anything. :func:`split_inputs` does that arithmetic.
    """
    paths = [Path(p) for p in inputs]
    if not paths:
        raise ValueError("no input files")
    if len(paths) > _NXWAV_MAX:
        raise ValueError(
            f"{len(paths)} input files, but an LTSA header records the count in a "
            f"uint16 and so can name at most {_NXWAV_MAX}. Split them across "
            f"several LTSAs -- see split_inputs()."
        )

    entries, fmt_fs, nch, bits, name_width = _gather(paths)

    # ck_ltsaparams.m:27-33. BUG-COMPATIBLE in spirit but not in consequence: MATLAB
    # prints and returns, then carries on into the writer. Refusing is the point --
    # a raw file recorded at a different rate would be averaged at the wrong one.
    bad = [e for e in entries if e.sample_rate != fmt_fs]
    if bad:
        rates = sorted({e.sample_rate for e in bad})
        raise ValueError(
            f"the fmt chunk says {fmt_fs} Hz but {len(bad)} raw file(s) report "
            f"{rates}. LTSA creation requires them to agree (ck_ltsaparams.m:27); "
            f"note the display path would read this file happily, because it trusts "
            f"the raw-file table instead."
        )

    # ck_ltsaparams.m:48-57. HARP sector geometry: 512-byte sectors with a 12-byte
    # header, and 4 more bytes of header when there are 4 channels.
    if nch == 1:
        blksz = (512 - 12) // 2
    elif nch == 4:
        blksz = (512 - 12 - 4) // 2
    else:
        raise ValueError(
            f"{nch} channels: Triton's sector geometry only defines 1 and 4 "
            f"(ck_ltsaparams.m:55), so blksz is unknown here"
        )

    # ck_ltsaparams.m:75-82. Silently clamped in MATLAB; recorded here so a caller
    # can see it happened, because it changes the meaning of every time bin.
    tave_requested = float(tave)
    max_tave = (entries[0].write_length * blksz) / fmt_fs
    tave = min(float(tave), max_tave)

    nfft = int(math.floor(fmt_fs / dfreq))          # ck_ltsaparams.m:85
    if nfft < 1:
        raise ValueError(f"dfreq={dfreq} Hz at {fmt_fs} Hz gives nfft={nfft}")
    cfact = tave * fmt_fs / nfft                    # ck_ltsaparams.m:89
    nfreq = (nfft + 1) // 2 if nfft % 2 else nfft // 2 + 1

    p = LtsaPlan(
        inputs=tuple(paths), fs=fmt_fs, n_channels=nch, bits_per_sample=bits,
        channel=channel, blksz=blksz, tave=tave, tave_requested=tave_requested,
        dfreq=float(dfreq), nfft=nfft, cfact=cfact, nfreq=nfreq, version=4,
        entries=entries, name_width=name_width,
    )

    # write_ltsahead.m:105-125. nave and the byte-location chain.
    loc = p.data_start_loc
    for k, e in enumerate(entries):
        n_samp = (e.write_length * blksz) / nch
        n_ave = int(math.ceil(n_samp / (nfft * cfact)))
        if k:
            loc = entries[k - 1].byte_loc_out + entries[k - 1].n_ave * nfreq
        if loc > _BYTELOC_MAX:
            raise ValueError(
                f"raw file {k} would start at byte {loc}, past the uint32 limit of "
                f"{_BYTELOC_MAX}. Use fewer files in this LTSA "
                f"(write_ltsahead.m:132)."
            )
        entries[k] = RawEntry(**{**e.__dict__, "n_ave": n_ave, "byte_loc_out": loc})
    return p


def split_inputs(inputs: list[str | Path], max_files: int = _NXWAV_MAX) -> list[list[Path]]:
    """Chunk an input list so each chunk fits in one LTSA's ``nxwav`` field.

    The known failure mode when a deployment is many small files rather than a few
    large ones: the limit is on the *file* count, not the data volume, so it bites
    exactly when each file contributes little. Splitting is the fix rather than a
    workaround, because the field cannot be widened without a version 5.

    Chunks are contiguous and in the order given, so each output LTSA covers one
    unbroken stretch of time provided the inputs were sorted.
    """
    paths = [Path(p) for p in inputs]
    if max_files < 1:
        raise ValueError("max_files must be at least 1")
    return [paths[i:i + max_files] for i in range(0, len(paths), max_files)]


def write_header(p: LtsaPlan, out: str | Path) -> None:
    """Write the 64-byte file header, the directory, and the spare-slot zero fill.

    Port of ``write_ltsahead.m``. Creates the file at its full final length so
    :func:`calc` can seek into it, which is how MATLAB works too -- it reopens with
    ``'r+'``.
    """
    out = Path(out)
    buf = bytearray()
    buf += b"LTSA"
    buf += struct.pack("<B", p.version)
    buf += b"xxx"
    buf += struct.pack("<I", p.dir_start_loc)
    buf += struct.pack("<I", p.data_start_loc)
    # float32, not float64: the header field is 4 bytes, so tave and dfreq are
    # truncated on the way to disk and a reader gets the truncated value back.
    buf += struct.pack("<f", p.tave)
    buf += struct.pack("<f", p.dfreq)
    buf += struct.pack("<I", p.fs)
    buf += struct.pack("<I", p.nfft)
    buf += struct.pack("<I", p.n_raw_total)          # v4: uint32
    buf += struct.pack("<H", p.n_xwav)               # still uint16 -- see plan()
    buf += struct.pack("<B", p.channel)
    buf += bytes(25)                                 # pad to 64
    assert len(buf) == _HEADER_SIZE, len(buf)

    for e in p.entries:
        entry = bytearray()
        entry += bytes([e.year, e.month, e.day, e.hour, e.minute, e.secs])
        entry += struct.pack("<H", e.ticks)
        entry += struct.pack("<I", e.byte_loc_out)
        entry += struct.pack("<I", e.n_ave)
        # get_headers.m:113 copies fnsz(2) characters -- the width MATLAB space-padded
        # the input list to -- and the field is then NUL-filled to 80.
        name = e.source.name.ljust(p.name_width)[: p.name_width]
        entry += name.encode("latin-1").ljust(80, b"\x00")[:80]
        entry += struct.pack("<I", e.rfileid)
        entry += bytes(4)
        assert len(entry) == _ENTRY_SIZE, len(entry)
        buf += entry

    spare = (p.n_raw_total + _SPARE_ENTRIES) - p.n_raw_total
    buf += bytes(_ENTRY_SIZE * spare)
    assert len(buf) == p.data_start_loc, (len(buf), p.data_start_loc)
    out.write_bytes(bytes(buf))


def calc(p: LtsaPlan, out: str | Path) -> int:
    """Compute the spectra and write them into the reserved data region.

    Port of ``calc_ltsa.m``. Returns the number of spectra written.
    """
    out = Path(out)
    samp_per_ave = p.samples_per_average
    bytes_per_sample = p.bits_per_sample // 8
    dtype = {16: "<i2", 32: "<i4", 8: "u1"}.get(p.bits_per_sample)
    if dtype is None:
        raise ValueError(f"{p.bits_per_sample}-bit LTSA input is not supported")

    # Group consecutive entries by source so each input file is opened once, not once
    # per raw file. get_headers walks the inputs in order, so entries from one file are
    # always contiguous; groupby relies on that rather than re-sorting, which would
    # change the output order and therefore the file.
    groups = [
        (src, list(grp))
        for src, grp in itertools.groupby(p.entries, key=lambda e: e.source)
    ]

    written = 0
    with open(out, "r+b") as fod:
        for src, entries in groups:
            with open(src, "rb") as fid:
                written += _calc_one_source(p, fod, fid, entries, samp_per_ave,
                                            bytes_per_sample, dtype)
    return written


def _calc_one_source(
    p: LtsaPlan,
    fod,
    fid,
    entries: list[RawEntry],
    samp_per_ave: int,
    bytes_per_sample: int,
    dtype: str,
) -> int:
    """The per-raw-file loop of ``calc_ltsa.m``, for the raw files of one input file."""
    written = 0
    for e in entries:
        n_samp_raw = (e.write_length * p.blksz) / p.n_channels
        nave1 = n_samp_raw / (p.nfft * p.cfact)
        dnave = e.n_ave - nave1

        fod.seek(e.byte_loc_out)
        xi = 0
        for n in range(1, e.n_ave + 1):
            if dnave == 0:
                nsamp = samp_per_ave
            elif n == e.n_ave:
                nsamp = int(n_samp_raw - (e.n_ave - 1) * samp_per_ave)
            else:
                nsamp = samp_per_ave

            # BUG-COMPATIBLE, calc_ltsa.m:95-100. The advance uses *this* average's
            # nsamp, but the distance to move is the *previous* average's length.
            # Identical for every average except a short final one, where the pointer
            # lands early and this average re-reads data the previous one covered.
            xi = e.byte_loc if n == 1 else xi + nsamp * bytes_per_sample * p.n_channels

            fid.seek(xi)
            raw = fid.read(nsamp * p.n_channels * bytes_per_sample)
            frames = np.frombuffer(raw, dtype=dtype).astype(np.float64)
            if frames.size:
                usable = (frames.size // p.n_channels) * p.n_channels
                data = frames[:usable].reshape(-1, p.n_channels)[:, p.channel - 1]
            else:
                # calc_ltsa.m:127-131 -- reports and substitutes silence rather than
                # aborting, so one unreadable raw file does not lose the whole LTSA.
                data = np.zeros(nsamp)

            if data.size < p.nfft:          # calc_ltsa.m:146-153
                data = np.concatenate([data, np.zeros(p.nfft - data.size)])

            db = welch_db(data, fs=p.fs, nfft=p.nfft, noverlap=0)
            fod.write(to_int8_db(db).tobytes())
            written += 1
    return written


def build(
    inputs: list[str | Path],
    out: str | Path,
    tave: float = 5.0,
    dfreq: float = 100.0,
    channel: int = 1,
) -> LtsaPlan:
    """Derive, write the header, compute the spectra.  Returns the plan used."""
    p = plan(inputs, tave=tave, dfreq=dfreq, channel=channel)
    write_header(p, out)
    calc(p, out)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m triton.tools.mkltsa",
        description="Build an LTSA from x.wav files, byte-identical to MATLAB Triton's.",
    )
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="x.wav files, or a directory containing them")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .ltsa path")
    ap.add_argument("--tave", type=float, default=5.0,
                    help="seconds per time bin (clamped to one raw file's length)")
    ap.add_argument("--dfreq", type=float, default=100.0, help="Hz per frequency bin")
    ap.add_argument("--channel", type=int, default=1, help="1-based channel to average")
    ap.add_argument("--split", action="store_true",
                    help=f"if more than {_NXWAV_MAX} inputs, write several LTSAs "
                         f"instead of refusing")
    a = ap.parse_args(argv)

    files: list[Path] = []
    for item in a.inputs:
        if item.is_dir():
            files += sorted(p for p in item.iterdir() if p.name.lower().endswith(".x.wav"))
        else:
            files.append(item)
    if not files:
        print("no x.wav files found", file=sys.stderr)
        return 2

    chunks = split_inputs(files) if a.split else [files]
    for i, chunk in enumerate(chunks):
        out = a.out if len(chunks) == 1 else a.out.with_name(
            f"{a.out.stem}_part{i + 1:03d}{a.out.suffix}"
        )
        # The refusals in plan() are all things a user can act on -- mixed bit depths,
        # a sample-rate disagreement, too many files -- so they get a message rather
        # than a traceback. This is a tool analysts run, not a library call.
        try:
            p = build(chunk, out, tave=a.tave, dfreq=a.dfreq, channel=a.channel)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        note = "" if p.tave == p.tave_requested else (
            f"  (tave clamped from {p.tave_requested:g} to {p.tave:g})"
        )
        print(
            f"{out.name}: {len(chunk)} file(s), {p.n_raw_total} raw files, "
            f"nfft {p.nfft}, {p.nfreq} freqs, "
            f"{sum(e.n_ave for e in p.entries)} spectra, "
            f"{out.stat().st_size} bytes{note}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
