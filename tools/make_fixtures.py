#!/usr/bin/env python
"""Generate the Phase 0 golden-fixture corpus.

These files are the inputs to the MATLAB<->Python parity suite.  MATLAB reads them with the
existing Triton code and dumps reference outputs (tools/matlab/dump_reference.m); the Python
implementation must reproduce those outputs exactly.

IMPORTANT -- this generator is deliberately standalone.  It writes x.wav bytes directly from the
spec in docs/formats/xwav.md and must NOT be refactored to use triton.io once that exists: a
fixture generator that shares code with the implementation under test proves nothing.

Signal content is deterministic (fixed seed, fixed tones) so every regeneration is byte-identical.

Usage:
    python tools/make_fixtures.py                    # write to fixtures/generated/
    python tools/make_fixtures.py --outdir some/dir
    python tools/make_fixtures.py --list             # describe the corpus, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------------
# Constants from the format spec.  See docs/formats/xwav.md.
# --------------------------------------------------------------------------------------------

YEAR_OFFSET = 2000  # the x.wav header year byte holds (year - 2000)
SECTOR_BYTES = 512
SECTOR_HEADER_BYTES = 12
SECTOR_HEADER_BYTES_4CH = 16  # 12 + 4 extra, per ck_ltsaparams.m:53

#: Deterministic content: a few tones plus seeded noise, so spectra have structure to compare.
TONE_HZ = (0.031, 0.113, 0.286)  # as a fraction of the sample rate
NOISE_SEED = 20260804


def blksz_for(nch: int) -> int:
    """Samples per 512-byte sector, summed over channels (ck_ltsaparams.m:48-57).

    Only 1 and 4 channels are defined by Triton; the formula assumes 16-bit samples.
    """
    if nch == 1:
        return (SECTOR_BYTES - SECTOR_HEADER_BYTES) // 2
    if nch == 4:
        return (SECTOR_BYTES - SECTOR_HEADER_BYTES_4CH) // 2
    raise ValueError(f"Triton only defines sector geometry for 1 or 4 channels, not {nch}")


# --------------------------------------------------------------------------------------------
# Fixture descriptions
# --------------------------------------------------------------------------------------------


@dataclass
class XWavSpec:
    """One synthetic x.wav file."""

    name: str
    purpose: str
    fs: int  # true sample rate (goes in the raw-file table)
    nch: int
    bits: int
    n_raw: int  # number of raw files
    raw_seconds: float  # duration of each raw file
    raw_period_seconds: float  # start-to-start spacing; > raw_seconds means duty-cycled
    start: datetime
    version: int = 1  # harp chunk WavVersionNumber: 0, 1 or 2
    gain: int = 1
    fmt_fs: int | None = None  # fmt-chunk rate if it should differ from the true rate
    start_ticks_ms: int = 0  # sub-second offset on the first raw file
    audio_format: int = 1  # 1 = PCM

    @property
    def bytes_per_sample(self) -> int:
        return self.bits // 8

    @property
    def samples_per_raw_per_ch(self) -> int:
        return int(round(self.raw_seconds * self.fs))

    @property
    def raw_byte_length(self) -> int:
        return self.samples_per_raw_per_ch * self.nch * self.bytes_per_sample

    @property
    def write_length(self) -> int:
        """Sector count.  Must divide evenly or MATLAB's nave arithmetic goes non-integer."""
        denom = self.bytes_per_sample * blksz_for(self.nch)
        wl, rem = divmod(self.raw_byte_length, denom)
        if rem:
            raise ValueError(
                f"{self.name}: raw_byte_length {self.raw_byte_length} is not a multiple of "
                f"{denom} (bytes_per_sample * blksz); pick a different raw_seconds"
            )
        return wl


@dataclass
class WavSpec:
    """One synthetic plain .wav file, named so wavname2dnum.m can parse its start time."""

    name: str
    purpose: str
    fs: int
    nch: int
    bits: int
    seconds: float


@dataclass
class Corpus:
    xwavs: list[XWavSpec] = field(default_factory=list)
    wavs: list[WavSpec] = field(default_factory=list)


# A realistic deployment timestamp, echoing the SOCAL41N example in Triton-master/metadata/.
BASE_START = datetime(2011, 1, 30, 8, 45, 0)


def build_corpus() -> Corpus:
    return Corpus(
        xwavs=[
            XWavSpec(
                name="xwav_v1_cont_1ch_16b_10k",
                purpose="Baseline: continuous, single channel, 16-bit, v1 header.",
                fs=10_000, nch=1, bits=16,
                n_raw=4, raw_seconds=2.5, raw_period_seconds=2.5,
                start=BASE_START,
            ),
            XWavSpec(
                name="xwav_v1_duty_1ch_16b_10k",
                purpose=(
                    "Duty-cycled: 2.5 s on, 7.5 s off.  Exercises gap handling, delimiter "
                    "placement, and readseg's splice across a raw-file boundary."
                ),
                fs=10_000, nch=1, bits=16,
                n_raw=4, raw_seconds=2.5, raw_period_seconds=10.0,
                start=BASE_START,
            ),
            XWavSpec(
                name="xwav_v1_ticks_1ch_16b_10k",
                purpose=(
                    "Sub-second start times (ticks = 125 ms).  Exercises millisecond timing "
                    "and the timestr/timenum round trip."
                ),
                fs=10_000, nch=1, bits=16,
                n_raw=3, raw_seconds=2.5, raw_period_seconds=3.7,
                start=BASE_START, start_ticks_ms=125,
            ),
            XWavSpec(
                name="xwav_v1_cont_4ch_16b_10k",
                purpose="Four channels: the other sector geometry (blksz = 248).",
                fs=10_000, nch=4, bits=16,
                n_raw=3, raw_seconds=0.62, raw_period_seconds=0.62,
                start=BASE_START,
            ),
            XWavSpec(
                name="xwav_v1_cont_1ch_16b_200k",
                purpose=(
                    "Real HARP sample rate.  This is the fixture that fails if timing is "
                    "carried as a float datenum at the true epoch (5 us per sample vs ~20 us "
                    "of datenum resolution)."
                ),
                fs=200_000, nch=1, bits=16,
                n_raw=4, raw_seconds=0.25, raw_period_seconds=0.25,
                start=BASE_START,
            ),
            XWavSpec(
                name="xwav_v2_cont_1ch_16b_10k",
                purpose=(
                    "v2 header: adds per-channel drate in the harp chunk and per-channel dt "
                    "in each raw-file record, and changes the subchunk-size formula."
                ),
                fs=10_000, nch=1, bits=16,
                n_raw=3, raw_seconds=2.5, raw_period_seconds=2.5,
                start=BASE_START, version=2,
            ),
            XWavSpec(
                name="xwav_v1_cont_1ch_32b_10k",
                purpose=(
                    "32-bit.  NOTE: wrxwavhd.m writes AudioFormat=3 (float) for 32-bit while "
                    "readseg.m reads int32.  This fixture writes int32 samples with "
                    "AudioFormat=1; see xwav.md section 6 item 1 -- OPEN QUESTION, needs a real "
                    "32-bit HARP file to settle."
                ),
                fs=10_000, nch=1, bits=32,
                n_raw=2, raw_seconds=2.5, raw_period_seconds=2.5,
                start=BASE_START,
            ),
            XWavSpec(
                name="xwav_v1_fakefs_1ch_16b",
                purpose=(
                    "fmt-chunk SampleRate (100000) disagrees with the raw-file table rate "
                    "(200000).  The display path must use the raw-file rate; mk_ltsa must "
                    "reject the file.  See xwav.md section 5.1."
                ),
                fs=200_000, nch=1, bits=16,
                n_raw=2, raw_seconds=0.25, raw_period_seconds=0.25,
                start=BASE_START, fmt_fs=100_000,
            ),
            XWavSpec(
                name="xwav_v1_gain4_1ch_16b_10k",
                purpose="gain = 4, so readseg must divide samples by 4.",
                fs=10_000, nch=1, bits=16,
                n_raw=2, raw_seconds=2.5, raw_period_seconds=2.5,
                start=BASE_START, gain=4,
            ),
        ],
        wavs=[
            WavSpec(
                name="wav_110130-084500_1ch_16b_10k",
                purpose="Plain wav, yymmdd-HHMMSS filename (wavname2dnum pattern 1).",
                fs=10_000, nch=1, bits=16, seconds=10.0,
            ),
            WavSpec(
                name="wav_20110130_084500_2ch_16b_10k",
                purpose="Plain wav, PAMGuard yyyymmdd_HHMMSS filename (pattern 4), 2 channels.",
                fs=10_000, nch=2, bits=16, seconds=6.0,
            ),
            WavSpec(
                name="wav_110130084500_1ch_24b_10k",
                purpose=(
                    "24-bit wav, SoundTrap yymmddHHMMSS filename (pattern 3).  24-bit is the "
                    "depth readseg.m cannot currently handle for x.wav."
                ),
                fs=10_000, nch=1, bits=24, seconds=4.0,
            ),
        ],
    )


# --------------------------------------------------------------------------------------------
# Signal synthesis
# --------------------------------------------------------------------------------------------


def make_samples(n: int, nch: int, fs: int, bits: int, seed_tag: str) -> np.ndarray:
    """Deterministic test signal, shape (n, nch), integer dtype for `bits`.

    Tones at fixed fractions of fs (so they land in the same bins at any rate) plus seeded
    noise.  Each channel is phase-offset so multichannel files are distinguishable.
    """
    peak = 2 ** (bits - 1) - 1
    # RandomState (not default_rng): NEP 19 guarantees its stream is stable across numpy
    # versions forever.  Generator makes no such promise, and these fixtures must stay
    # byte-identical for the life of the project.
    seed = (NOISE_SEED + int(hashlib.sha256(seed_tag.encode()).hexdigest()[:8], 16)) % (2**32)
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64)
    out = np.zeros((n, nch), dtype=np.float64)
    for ch in range(nch):
        sig = np.zeros(n)
        for k, frac in enumerate(TONE_HZ):
            amp = 0.25 / (k + 1)
            phase = 0.37 * ch + 0.11 * k
            sig += amp * np.sin(2 * np.pi * frac * t + phase)
        sig += 0.02 * rng.standard_normal(n)
        out[:, ch] = sig
    out = np.clip(out, -1.0, 1.0) * (peak * 0.8)
    dtype = {16: np.int16, 24: np.int32, 32: np.int32}[bits]
    return np.round(out).astype(dtype)


def pack_samples(x: np.ndarray, bits: int) -> bytes:
    """Interleave and pack to little-endian PCM bytes."""
    flat = x.reshape(-1)  # row-major == interleaved, since x is (n, nch)
    if bits == 16:
        return flat.astype("<i2").tobytes()
    if bits == 32:
        return flat.astype("<i4").tobytes()
    if bits == 24:
        as32 = flat.astype("<i4").tobytes()
        return b"".join(as32[i : i + 3] for i in range(0, len(as32), 4))
    raise ValueError(bits)


# --------------------------------------------------------------------------------------------
# x.wav writer -- follows docs/formats/xwav.md byte for byte
# --------------------------------------------------------------------------------------------


def _fixed(s: str, n: int) -> bytes:
    b = s.encode("ascii")
    if len(b) > n:
        raise ValueError(f"{s!r} exceeds {n} bytes")
    return b.ljust(n, b"\x00")


def write_xwav(spec: XWavSpec, path: Path) -> dict:
    nch, bps = spec.nch, spec.bytes_per_sample
    n_raw = spec.n_raw
    fmt_fs = spec.fmt_fs if spec.fmt_fs is not None else spec.fs
    block_align = nch * bps
    byte_rate = fmt_fs * block_align

    # --- header geometry -------------------------------------------------------------------
    if spec.version == 2:
        harp_fixed = 64 + 4 * nch
        raw_entry = 32 + 4 * nch
    else:
        harp_fixed = 64
        raw_entry = 32
    harp_subchunk_size = harp_fixed - 8 + n_raw * raw_entry
    header_size = 12 + 24 + harp_fixed + n_raw * raw_entry + 8

    raw_bytes = spec.raw_byte_length
    data_size = raw_bytes * n_raw

    # --- raw-file table --------------------------------------------------------------------
    raws = []
    for i in range(n_raw):
        t = spec.start + timedelta(
            seconds=i * spec.raw_period_seconds, milliseconds=spec.start_ticks_ms
        )
        raws.append(
            {
                "year": t.year - YEAR_OFFSET,
                "month": t.month,
                "day": t.day,
                "hour": t.hour,
                "minute": t.minute,
                "secs": t.second,
                "ticks": t.microsecond // 1000,
                "byte_loc": header_size + i * raw_bytes,
                "byte_length": raw_bytes,
                "write_length": spec.write_length,
                "sample_rate": spec.fs,
                "gain": spec.gain,
                "start_iso": t.isoformat(timespec="milliseconds"),
            }
        )

    # --- bytes -----------------------------------------------------------------------------
    buf = bytearray()
    buf += b"RIFF"
    buf += struct.pack("<I", header_size - 8 + data_size)  # ChunkSize == filesize - 8
    buf += b"WAVE"

    buf += b"fmt "
    buf += struct.pack("<I", 16)
    buf += struct.pack("<HHIIHH", spec.audio_format, nch, fmt_fs, byte_rate,
                       block_align, spec.bits)

    buf += b"harp"
    buf += struct.pack("<I", harp_subchunk_size)
    buf += struct.pack("<B", spec.version)
    buf += _fixed("1.0test001", 10)   # FirmwareVersionNumber
    buf += _fixed("T001", 4)          # InstrumentID
    buf += _fixed("FIXT", 4)          # SiteName
    buf += _fixed("PHASE0FX", 8)      # ExperimentName
    buf += struct.pack("<B", 1)       # DiskSequenceNumber
    buf += _fixed("FX000001", 8)      # DiskSerialNumber
    buf += struct.pack("<H", n_raw)
    buf += struct.pack("<i", -11_795_000)  # Longitude: -117.95 deg * 1e5
    buf += struct.pack("<i", 3_275_000)    # Latitude:   32.75 deg * 1e5
    buf += struct.pack("<h", 1000)         # Depth [m]
    if spec.version == 2:
        buf += struct.pack(f"<{nch}f", *([float(spec.fs)] * nch))  # drate
    buf += b"\x00" * 8  # Reserved

    assert len(buf) == 12 + 24 + harp_fixed, (len(buf), 12 + 24 + harp_fixed)

    for r in raws:
        buf += struct.pack(
            "<BBBBBBHIIIIB",
            r["year"], r["month"], r["day"], r["hour"], r["minute"], r["secs"],
            r["ticks"], r["byte_loc"], r["byte_length"], r["write_length"],
            r["sample_rate"], r["gain"],
        )
        buf += b"\x00" * 7  # padding to 32
        if spec.version == 2:
            buf += struct.pack(f"<{nch}f", *([0.0] * nch))  # dt, ch1-relative

    buf += b"data"
    buf += struct.pack("<I", data_size)
    assert len(buf) == header_size, (len(buf), header_size)
    assert raws[0]["byte_loc"] == header_size

    for i in range(n_raw):
        x = make_samples(spec.samples_per_raw_per_ch, nch, spec.fs, spec.bits,
                         f"{spec.name}:{i}")
        buf += pack_samples(x, spec.bits)

    path.write_bytes(bytes(buf))

    return {
        "name": spec.name,
        "file": path.name,
        "purpose": spec.purpose,
        "kind": "xwav",
        "bytes": len(buf),
        "sha256": hashlib.sha256(buf).hexdigest(),
        "header_size": header_size,
        "wav_version": spec.version,
        "fmt_sample_rate": fmt_fs,
        "true_sample_rate": spec.fs,
        "n_channels": nch,
        "bits_per_sample": spec.bits,
        "audio_format": spec.audio_format,
        "n_raw_files": n_raw,
        "samples_per_raw_per_channel": spec.samples_per_raw_per_ch,
        "write_length": spec.write_length,
        "blksz": blksz_for(nch),
        "gain": spec.gain,
        "duty_cycled": spec.raw_period_seconds > spec.raw_seconds,
        "raw_files": raws,
    }


# --------------------------------------------------------------------------------------------
# plain wav writer (stdlib-only; no soundfile dependency for fixture generation)
# --------------------------------------------------------------------------------------------


def write_wav(spec: WavSpec, path: Path) -> dict:
    n = int(round(spec.seconds * spec.fs))
    x = make_samples(n, spec.nch, spec.fs, spec.bits, spec.name)
    payload = pack_samples(x, spec.bits)
    block_align = spec.nch * spec.bits // 8

    buf = bytearray()
    buf += b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    buf += b"fmt " + struct.pack("<I", 16)
    buf += struct.pack("<HHIIHH", 1, spec.nch, spec.fs, spec.fs * block_align,
                       block_align, spec.bits)
    buf += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(bytes(buf))

    return {
        "name": spec.name,
        "file": path.name,
        "purpose": spec.purpose,
        "kind": "wav",
        "bytes": len(buf),
        "sha256": hashlib.sha256(buf).hexdigest(),
        "sample_rate": spec.fs,
        "n_channels": spec.nch,
        "bits_per_sample": spec.bits,
        "n_samples_per_channel": n,
    }


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "fixtures" / "generated")
    ap.add_argument("--list", action="store_true", help="describe the corpus and exit")
    args = ap.parse_args()

    corpus = build_corpus()

    if args.list:
        for s in corpus.xwavs:
            print(f"{s.name}.x.wav\n    {s.purpose}")
        for w in corpus.wavs:
            print(f"{w.name}.wav\n    {w.purpose}")
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in corpus.xwavs:
        manifest.append(write_xwav(s, args.outdir / f"{s.name}.x.wav"))
    for w in corpus.wavs:
        manifest.append(write_wav(w, args.outdir / f"{w.name}.wav"))

    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "generator": "tools/make_fixtures.py",
                "spec": "docs/formats/xwav.md",
                "note": "Regenerating must be byte-identical; sha256 is checked by the tests.",
                "fixtures": manifest,
            },
            indent=2,
        )
        + "\n"
    )

    total = sum(m["bytes"] for m in manifest)
    print(f"Wrote {len(manifest)} fixtures ({total / 1e6:.2f} MB) to {args.outdir}")
    for m in manifest:
        print(f"  {m['file']:<44} {m['bytes']:>9,} B  {m['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
