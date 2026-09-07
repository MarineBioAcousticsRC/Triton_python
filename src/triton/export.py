"""Writing the window on screen out to a file.

The port of MATLAB's File menu exports (``filepd.m:227-470``, ``miscpd.m:171-181``),
with none of the arithmetic in the GUI, so a batch script can clip a deployment without
a display and the tests can check the bytes without a screenshot.

Six bugs in the MATLAB versions are **not** reproduced. They are catalogued with
file:line in docs/EXPORT_PLAN.md §2; the three that matter are that
``wrxwavhd.m:20`` hardcodes two bytes per sample regardless of bit depth, that it sizes
the header from one channel while ``filepd.m:334`` writes all of them, and that MATLAB's
``fwrite`` walks a matrix column-major so those samples come out de-interleaved. The net
effect is that multichannel x.wav export does not currently work. Every function here
handles channel count and bit depth explicitly, and there are round-trip tests that read
back what was written.

Two conventions this adds, neither of which MATLAB has for its reachable exports:

**Every export can carry a provenance sidecar.** A wav clip is the artefact most likely
to be emailed to a collaborator and least likely to arrive with any record of where it
came from. :func:`provenance` produces that record and :func:`write_sidecar` puts it
beside the file.

**Default filenames carry the timestamp.** ``{source stem}@{ISO time}``, so exports are
self-identifying and sort chronologically. MATLAB's unreachable ``saveimageas`` already
did this (``filepd.m:407``); its reachable exports all default to ``data.wav``.
"""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .colormaps import colormap_lut
from .io import audio as _audio

if TYPE_CHECKING:                                    # pragma: no cover
    from .session import Frame, LtsaTile, TritonSession

__all__ = [
    "default_stem",
    "provenance",
    "write_sidecar",
    "write_wav",
    "write_xwav",
    "write_arrays",
    "write_mat",
    "spectrogram_rgb",
]


# ------------------------------------------------------------------------ naming

def default_stem(source: Path | None, start: np.datetime64) -> str:
    """``{source stem}@{ISO time}``, safe for a filename on any platform.

    Colons become dashes because Windows forbids them in paths, and the fractional
    second is kept to milliseconds -- enough to distinguish adjacent windows without
    a 27-character name.
    """
    stem = "clip"
    if source is not None:
        stem = source.name
        for suffix in (".x.wav", ".x.flac", ".wav", ".flac", ".ltsa"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
    t = str(np.datetime64(start, "ms")).replace(":", "-")
    return f"{stem}@{t}"


# ------------------------------------------------------------------ provenance

def provenance(session: TritonSession, frame: Frame | None = None) -> dict[str, Any]:
    """Everything needed to say where an exported file came from.

    The same content as the stamps drawn on the plots, plus the session snapshot. JSON
    safe: times are ISO strings and paths are strings, so this can be written beside
    the export or pasted into an issue.
    """
    out: dict[str, Any] = {
        "exported_by": "triton (Python port)",
        "source": str(session.audio.path) if session.audio.path else None,
        "channel": session.view.channel,
    }
    if frame is not None:
        src = session.audio.source
        out.update(
            window_start=str(frame.start),
            window_seconds=session.view.tseg_sec,
            sample_rate=frame.fs,
            n_channels=int(frame.samples.shape[1]),
            n_samples=int(frame.samples.shape[0]),
            raw_file=session.audio.segment + 1,       # 1-based, as MATLAB counts
            nfft=session.view.nfft,
            overlap_percent=session.view.overlap_pct,
            brightness_db=session.view.brightness,
            contrast_percent=session.view.contrast,
            colormap=session.view.colormap,
        )
        if session.view.filter_on:
            out["display_filter_hz"] = [session.view.filter_low,
                                        session.view.filter_high]
        if src is not None and isinstance(src, _audio.XwavSource):
            h = src.header
            out["deployment"] = {
                "instrument": h.instrument_id, "site": h.site_name,
                "experiment": h.experiment_name, "firmware": h.firmware_version,
                "latitude": h.latitude, "longitude": h.longitude, "depth": h.depth,
            }
    if session.calibration.has_tf:
        out["transfer_function"] = str(session.calibration.tf_path or "in memory")
    return out


def _check_writable(path: Path) -> None:
    """Fail before opening anything, with a sentence a person can act on.

    Worth the four lines: a user can type a path whose folder does not exist, and
    ``wave.open`` on one raises from inside its constructor, leaving a half-built object
    whose destructor then raises a second, unrelated ``AttributeError`` -- so the error
    that reaches the status bar is about ``_file``, not about the folder.
    """
    if not path.parent.is_dir():
        raise FileNotFoundError(f"there is no folder {path.parent}")


def write_sidecar(path: str | Path, meta: dict[str, Any]) -> Path:
    """Write ``<path>.json`` next to an export."""
    side = Path(path).with_suffix(Path(path).suffix + ".json")
    side.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return side


# --------------------------------------------------------------------- helpers

def _select(frame: Frame, channel: int | None) -> np.ndarray:
    """Samples as ``(n, nch)``: every channel, or just the one asked for.

    Defaults to every channel, because a multichannel clip is what a multichannel
    recording is. MATLAB intends one channel and writes a broken version of all
    (EXPORT_PLAN §2.5).
    """
    if channel is None:
        return frame.samples
    if not 1 <= channel <= frame.samples.shape[1]:
        raise ValueError(
            f"channel {channel} is out of range; this window has "
            f"{frame.samples.shape[1]}"
        )
    return frame.samples[:, channel - 1: channel]


def _normalise(x: np.ndarray, bits: int) -> np.ndarray:
    """Peak-normalise to full scale, at any bit depth.

    Three things MATLAB gets wrong here, all fixed (EXPORT_PLAN §2.1-2.3):
    it only scales 16-bit and silently leaves 24- and 32-bit untouched; its
    ``max(abs(DATA))`` on a multichannel array makes the scaling a matrix multiply;
    and it does not remove the mean first, so a DC offset -- which hydrophone data
    usually has -- means one side clips before the other.

    Scaled by a **single** factor across all channels, not per channel, so the relative
    levels between channels survive. Normalising each channel to its own peak would
    destroy exactly the comparison a multichannel recording exists to support.
    """
    x = x - x.mean(axis=0, keepdims=True)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0:
        return np.zeros_like(x)
    return x * ((2 ** (bits - 1) - 1) / peak)


def _to_bytes(x: np.ndarray, bits: int) -> bytes:
    """Interleaved PCM bytes, little-endian.

    ``x`` is ``(n, nch)``. C-order ``ravel`` interleaves -- frame 0 all channels, then
    frame 1 -- which is what both WAV and x.wav require. MATLAB's ``fwrite`` walks
    column-major and so writes channel-major, which is the de-interleaving bug.

    24-bit is packed by hand because numpy has no 24-bit integer type; MATLAB cannot
    write it at all, since ``fwrite`` has no ``'int24'``.
    """
    flat = np.asarray(x, dtype=np.float64).ravel(order="C")
    limit = 2 ** (bits - 1)
    clipped = np.clip(np.rint(flat), -limit, limit - 1)
    if bits == 8:
        return (clipped + 128).astype(np.uint8).tobytes()
    if bits == 16:
        return clipped.astype("<i2").tobytes()
    if bits == 32:
        return clipped.astype("<i4").tobytes()
    if bits == 24:
        as32 = clipped.astype("<i4").view(np.uint8).reshape(-1, 4)
        return as32[:, :3].tobytes()                 # drop the high byte, little-endian
    raise ValueError(f"{bits}-bit output is not supported")


# --------------------------------------------------------------------- wav / x.wav

def write_wav(
    path: str | Path,
    frame: Frame,
    *,
    normalise: bool = False,
    channel: int | None = None,
    bits: int | None = None,
    sidecar: dict[str, Any] | None = None,
) -> Path:
    """Write the window as a plain WAV.

    ``normalise=False`` -- the default -- writes the sample **counts** unchanged, so
    levels read from the file downstream still mean something. ``normalise=True`` scales
    the peak to full scale for listening, which changes the numbers and is therefore
    wrong for measurement. The menu labels say which is which.

    Uses the standard library's ``wave``, so no dependency is added.
    """
    path = Path(path)
    _check_writable(path)
    x = _select(frame, channel)
    bits = bits or 16
    if normalise:
        x = _normalise(x, bits)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(int(x.shape[1]))
        w.setsampwidth(bits // 8)
        w.setframerate(int(frame.fs))
        w.writeframes(_to_bytes(x, bits))
    if sidecar is not None:
        write_sidecar(path, {**sidecar, "normalised": normalise, "bits": bits})
    return path


def write_xwav(
    path: str | Path,
    frame: Frame,
    session: TritonSession,
    *,
    channel: int | None = None,
    sidecar: dict[str, Any] | None = None,
) -> Path:
    """Write the window as an x.wav, keeping its absolute time and its deployment.

    The most useful of the exports, and the reason is the format: an x.wav carries the
    clip's real start time to the millisecond plus the instrument, site, experiment,
    position and gain it came from. A wav clip is a sound; an x.wav clip still knows
    which deployment and which moment it is.

    One raw file, ``byte_length`` computed from the actual bit depth and channel count,
    samples interleaved. All three are things ``wrxwavhd.m`` gets wrong.

    Refuses a non-x.wav source, because there is no harp metadata to carry and a file
    with invented identity fields is worse than no file.
    """
    path = Path(path)
    src = session.audio.source
    if not isinstance(src, _audio.XwavSource):
        raise ValueError(
            "x.wav export needs an x.wav source: the harp header carries the "
            "instrument, site and position, and inventing those would be worse than "
            "refusing. Export as WAV instead."
        )
    h = src.header
    x = _select(frame, channel)
    n_samples, n_ch = int(x.shape[0]), int(x.shape[1])
    bits = h.bits_per_sample
    bytes_per_sample = bits // 8
    byte_length = n_samples * n_ch * bytes_per_sample
    byte_rate = int(frame.fs) * n_ch * bytes_per_sample

    # v1, one raw file: 64 - 8 + 1 * 32 = 88 bytes of harp payload, which puts the data
    # at byte 140 -- the same offset a real single-raw-file HARP file has.
    harp_size = 64 - 8 + 32
    data_offset = 12 + 24 + 8 + harp_size + 8

    t = np.datetime64(frame.start, "ms")
    fields = [int(str(t)[0:4]) - 2000, int(str(t)[5:7]), int(str(t)[8:10]),
              int(str(t)[11:13]), int(str(t)[14:16]), int(str(t)[17:19])]
    ticks = int((t - np.datetime64(t, "s")) / np.timedelta64(1, "ms"))

    buf = bytearray()
    buf += b"RIFF" + struct.pack("<I", 4 + 24 + 8 + harp_size + 8 + byte_length)
    buf += b"WAVE"
    buf += b"fmt " + struct.pack("<I", 16)
    buf += struct.pack("<HHIIHH", 1, n_ch, int(frame.fs), byte_rate,
                       n_ch * bytes_per_sample, bits)
    buf += b"harp" + struct.pack("<I", harp_size)
    # Identity carried straight from the source, so the clip stays traceable.
    buf += struct.pack("<B", h.wav_version if h.wav_version in (0, 1) else 1)
    buf += h.firmware_version.encode("latin-1").ljust(10, b"\x00")[:10]
    buf += h.instrument_id.encode("latin-1").ljust(4, b"\x00")[:4]
    buf += h.site_name.encode("latin-1").ljust(4, b"\x00")[:4]
    buf += h.experiment_name.encode("latin-1").ljust(8, b"\x00")[:8]
    buf += struct.pack("<B", h.disk_sequence_number)
    buf += h.disk_serial_number.encode("latin-1").ljust(8, b"\x00")[:8]
    buf += struct.pack("<H", 1)                       # NumOfRawFiles
    buf += struct.pack("<i", int(round(h.longitude * 100000)))
    buf += struct.pack("<i", int(round(h.latitude * 100000)))
    buf += struct.pack("<h", int(h.depth))
    buf += bytes(8)                                   # Reserved, to 64
    # The single raw-file directory entry.
    buf += bytes(fields) + struct.pack("<H", ticks)
    buf += struct.pack("<I", data_offset)
    buf += struct.pack("<I", byte_length)
    # write_length is in recorder sectors, which only mean anything for data that came
    # off a HARP disk. A clip did not, so it is derived from the geometry rather than
    # copied: blksz is 250 samples per sector for one channel, 248 for four.
    blksz = 248 if n_ch == 4 else 250
    buf += struct.pack("<I", max(1, -(-n_samples * n_ch // blksz)))
    buf += struct.pack("<I", int(frame.fs))
    buf += struct.pack("<B", src.segments[session.audio.segment].gain)
    buf += bytes(7)                                   # padding to 32
    buf += b"data" + struct.pack("<I", byte_length)
    assert len(buf) == data_offset, (len(buf), data_offset)

    path.write_bytes(bytes(buf) + _to_bytes(x, bits))
    if sidecar is not None:
        write_sidecar(path, {**sidecar, "format": "x.wav", "bits": bits})
    return path


# ------------------------------------------------------------------ arrays / mat

def _array_payload(frame: Frame, session: TritonSession,
                   channel: int | None) -> dict[str, Any]:
    tile = frame.spectrogram
    return {
        "samples": _select(frame, channel),
        "sample_rate": np.array([frame.fs]),
        "spectrogram_db": tile.db,
        "spectrogram_f_hz": tile.f,
        "spectrogram_t_sec": tile.t,
        "spectra_f_hz": frame.spectra_f,
        "spectra_db": frame.spectra_db,
        "window_start": str(frame.start),
        "provenance": json.dumps(provenance(session, frame)),
    }


def write_arrays(path: str | Path, frame: Frame, session: TritonSession, *,
                 channel: int | None = None) -> Path:
    """Write the window's arrays as ``.npz``.

    The substitute for MATLAB's saved ``.fig``: it re-creates the *data* rather than the
    *figure*, which is more useful and far less brittle. Provenance travels inside, so
    the file is self-describing without a sidecar.
    """
    path = Path(path)
    np.savez_compressed(path, **_array_payload(frame, session, channel))
    return path


def write_mat(path: str | Path, frame: Frame, session: TritonSession, *,
              channel: int | None = None) -> Path:
    """Write the same arrays as a MATLAB ``.mat``.

    Free: ``scipy.io.savemat`` and scipy is already a core dependency. Kept because
    people who have MATLAB analysis scripts should not have to stop using them to look
    at data through this viewer.
    """
    from scipy.io import savemat

    path = Path(path)
    savemat(str(path), _array_payload(frame, session, channel), do_compression=True)
    return path


# ------------------------------------------------------------------- raw image

def spectrogram_rgb(
    tile: SpectrogramTile | LtsaTile,          # noqa: F821 -- either tile type
    colormap: str = "jet",
    *,
    flip: bool = True,
    gap_colour: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """The spectrogram as raw pixels: one per time bin per frequency bin.

    Revives ``filepd.m:403``, which MATLAB creates with ``'Visible','off'`` and
    ``'Enable','off'`` (``initpulldowns.m:46``) and so cannot be reached. It is the only
    export that writes the data as an image rather than a picture of a plot -- no axes,
    no labels, native resolution -- which is what figure assembly and downstream image
    tools want.

    Returns ``(n_freq, n_time, 3)`` uint8. ``flip`` puts low frequencies at the bottom,
    as they appear on screen, since row 0 of an array is the top of an image.

    Colour mapping goes through the same lookup table the viewer uses, so the pixels
    match what was on screen. MATLAB's version instead used the dB values directly as
    colour-map *indices*, clipped to 1..256, which is why its output does not match its
    own display.

    Bins with no data -- a duty-cycle gap in an LTSA is NaN, a silent bin is -inf --
    are painted ``gap_colour``, a neutral grey by default, rather than mapped to an end
    of the scale. The panel leaves them transparent on screen, which an RGB array
    cannot do; colouring them as the quietest possible bin would make a gap look like
    real data, which is the one thing an exported image must not do.
    """
    lut = colormap_lut(colormap)
    lo, hi = tile.clim
    span = (hi - lo) or 1.0
    db = np.asarray(tile.db, dtype=np.float64)
    good = np.isfinite(db)
    # Masked *before* the cast, not after: rint/clip pass NaN straight through and
    # casting NaN to an integer is undefined, which numpy warns about and which would
    # produce a platform-dependent index.
    norm = np.where(good, (db - lo) / span, 0.0)
    idx = np.clip(np.rint(norm * (len(lut) - 1)), 0, len(lut) - 1).astype(np.intp)
    rgb = lut[idx]
    rgb[~good] = np.asarray(gap_colour, dtype=np.uint8)
    return np.flipud(rgb) if flip else rgb
