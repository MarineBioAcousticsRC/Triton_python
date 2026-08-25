"""Spectra: the window, the spectrogram, Welch averaging, and the int8 quantiser.

Small module, high stakes. Everything Triton displays and everything it writes into
an LTSA comes through here, so a fraction-of-a-dB error is not cosmetic -- it changes
published numbers while looking completely normal on screen.

**The window is the trap.** MATLAB's ``hanning(N)`` is the symmetric Hann window *with
its zero endpoints removed*. It is neither ``scipy.signal.windows.hann(N)`` (which
keeps the zeros) nor ``hann(N, sym=False)`` (the periodic variant). Reaching for
either shifts every value by a fraction of a dB. See :func:`hanning` and
ltsa.md 5.1.

**The normalisation is the second trap.** MATLAB's ``spectrogram`` and ``pwelch``
return a one-sided power *spectral density*: the squared magnitude divided by
``fs * sum(w**2)``, with the interior bins doubled to account for the negative
frequencies that were folded away. Get the divisor wrong -- ``sum(w)**2`` instead of
``sum(w**2)`` is the classic slip, and ``mkspecgram.m``'s own commented-out lines 26-28
show someone once wrestling with exactly that -- and everything is off by a constant
offset that is easy to mistake for a calibration difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Spectrogram",
    "hanning",
    "spectrogram",
    "welch_db",
    "to_int8_db",
    "apply_transfer_function_curve",
    "display_filter",
]


def hanning(n: int) -> np.ndarray:
    """MATLAB's ``hanning(n)``: the symmetric Hann window without its zero endpoints.

    Equivalent to ``scipy.signal.windows.hann(n + 2, sym=True)[1:-1]``, and written
    out here so the definition is visible rather than delegated::

        w[k] = 0.5 * (1 - cos(2*pi*k / (n+1)))      k = 1 .. n

    The distinction from ``hann(n)`` matters because no sample of this window is zero,
    so ``sum(w**2)`` differs and every resulting dB value shifts. ltsa.md 5.1 spells
    out the consequence. ``hanning(1)`` is 1, matching MATLAB.
    """
    if n < 1:
        raise ValueError(f"window length must be at least 1, got {n}")
    k = np.arange(1, n + 1, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (n + 1)))


def _onesided_psd(seg: np.ndarray, window: np.ndarray, nfft: int, fs: float) -> np.ndarray:
    """One-sided PSD of each column of ``seg``, as MATLAB computes it.

    ``seg`` is (window_length, n_segments), already windowed by the caller? No -- the
    windowing happens here, so the caller cannot forget it.

    The scale factor is ``fs * sum(w**2)``, and the interior bins are doubled to fold
    in the negative frequencies. DC is never doubled; Nyquist is doubled only when
    ``nfft`` is odd, because an odd-length FFT has no Nyquist bin to leave alone.
    """
    win = window[:, None]
    x = seg * win
    if x.shape[0] < nfft:                     # zero-pad, as MATLAB's fft(x, nfft) does
        x = np.vstack([x, np.zeros((nfft - x.shape[0], x.shape[1]))])
    spec = np.fft.rfft(x, n=nfft, axis=0)
    psd = (spec.real**2 + spec.imag**2) / (fs * np.sum(window**2))
    # Fold the negative frequencies in. rfft gives nfft//2 + 1 bins; the last one is
    # a true Nyquist bin only for even nfft, and a Nyquist bin has no partner to add.
    if nfft % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    return psd


def _frame(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split ``x`` into overlapping columns, dropping any short tail.

    Dropping rather than padding is what MATLAB does here: ``spectrogram`` and
    ``pwelch`` both use ``fix((n - noverlap) / hop)`` segments and ignore the
    remainder. (``calc_ltsa.m`` *does* zero-pad its final bin, but it does that to the
    data before calling ``pwelch``, not inside it -- see ltsa.md 5.)
    """
    n = x.size
    if n < frame_len:
        return np.empty((frame_len, 0))
    count = (n - frame_len) // hop + 1
    idx = np.arange(frame_len)[:, None] + hop * np.arange(count)[None, :]
    return x[idx]


@dataclass(frozen=True)
class Spectrogram:
    """A spectrogram, cropped to the displayed frequency band.

    ``db`` is ``(n_freq, n_time)`` -- frequency down, time across, matching MATLAB's
    orientation so a reference dump can be compared without transposing.
    """

    f: np.ndarray            # Hz, the retained bins
    t: np.ndarray            # s, segment midpoints
    db: np.ndarray           # 10*log10(PSD), (n_freq, n_time)
    fimin: int               # 1-based first retained bin, as PARAMS.fimin
    fimax: int               # 1-based last retained bin, as PARAMS.fimax


def spectrogram(
    x: np.ndarray,
    fs: float,
    nfft: int,
    overlap_pct: float = 0.0,
    freq0: float = 0.0,
    freq1: float | None = None,
) -> Spectrogram:
    """Port of ``mkspecgram.m``.

    ``mkspecgram`` calls MATLAB's ``spectrogram`` for its 4th output -- the one-sided
    PSD -- then crops to the displayed band and takes ``10*log10(abs(...))``. Note the
    factor of 10 rather than 20: the value being logged is already a power, so the
    ``abs`` is defensive rather than meaningful.

    The band crop uses ``floor`` on both edges (``mkspecgram.m:22-23``)::

        fimin = floor(freq0 / df) + 1
        fimax = floor(freq1 / df) + 1

    so the top edge is inclusive and the displayed band is very slightly wider than
    asked for. Reproduced as written; ``fimin`` and ``fimax`` are returned so a caller
    can see which bins it actually got.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if freq1 is None:
        freq1 = fs / 2.0

    window = hanning(nfft)
    noverlap = int(round((overlap_pct / 100.0) * nfft))
    hop = nfft - noverlap
    if hop < 1:
        raise ValueError(f"overlap of {overlap_pct}% leaves no advance between segments")

    seg = _frame(x, nfft, hop)
    if seg.shape[1] == 0:
        raise ValueError(f"{x.size} samples is fewer than one {nfft}-point window")

    psd = _onesided_psd(seg, window, nfft, fs)
    f_all = np.arange(psd.shape[0], dtype=np.float64) * (fs / nfft)
    # Segment midpoints, which is what MATLAB's spectrogram reports for t.
    t = (nfft / 2.0 + hop * np.arange(seg.shape[1])) / fs

    df = fs / nfft
    fimin = int(np.floor(freq0 / df)) + 1
    fimax = int(np.floor(freq1 / df)) + 1
    fimax = min(fimax, psd.shape[0])          # a freq1 at Nyquist would run one past

    band = slice(fimin - 1, fimax)
    with np.errstate(divide="ignore"):        # a true zero bin is -inf, not an error
        db = 10.0 * np.log10(np.abs(psd[band]))
    return Spectrogram(f=f_all[band], t=t, db=db, fimin=fimin, fimax=fimax)


def welch_db(
    x: np.ndarray,
    fs: float,
    nfft: int,
    noverlap: int = 0,
) -> np.ndarray:
    """Port of the ``pwelch`` call in ``calc_ltsa.m:159-162``, in dB.

    ``pwelch(data, hanning(nfft), noverlap, nfft, fs)`` averages the one-sided PSD
    over segments; this returns ``10*log10`` of that. ``noverlap`` is 0 in the LTSA
    path, but it is a parameter because nothing about the maths requires that.

    No detrending and no mean removal, matching MATLAB's defaults. A DC offset in the
    data therefore lands in bin 0, which is correct and occasionally surprising.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    window = hanning(nfft)
    hop = nfft - noverlap
    if hop < 1:
        raise ValueError(f"noverlap={noverlap} leaves no advance for nfft={nfft}")

    seg = _frame(x, nfft, hop)
    if seg.shape[1] == 0:
        # calc_ltsa.m zero-pads a short final bin to nfft before calling pwelch
        # (lines 146-154), so accept a short input the same way rather than failing.
        seg = np.zeros((nfft, 1))
        seg[: x.size, 0] = x

    psd = _onesided_psd(seg, window, nfft, fs).mean(axis=1)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(psd)


def to_int8_db(db: np.ndarray) -> np.ndarray:
    """Quantise dB values the way ``fwrite(fid, x, 'int8')`` does.

    Two behaviours to match, and neither is numpy's default:

    * **Rounding is half-away-from-zero**, not half-to-even. ``np.round(0.5)`` is 0;
      MATLAB writes 1. On real spectra the difference shows up on maybe one value in
      a few thousand, which is exactly the kind of discrepancy that is dismissed as
      noise instead of chased.
    * **Out-of-range values saturate** at -128 and 127 rather than wrapping.

    This is the lossy step in the LTSA format (ltsa.md 4): a spectrum is stored to
    the nearest whole dB, and anything outside a 255 dB window is clipped. It has to
    be reproduced exactly for byte-identical output, so it lives in its own function
    with its own test rather than inlined at the write site.
    """
    x = np.asarray(db, dtype=np.float64)
    # NaN would propagate through the comparison below and land somewhere arbitrary;
    # MATLAB's fwrite turns it into 0, so be explicit about it.
    rounded = np.where(np.isnan(x), 0.0, np.copysign(np.floor(np.abs(x) + 0.5), x))
    return np.clip(rounded, -128, 127).astype(np.int8)


def apply_transfer_function_curve(
    tf_freq: np.ndarray,
    tf_value: np.ndarray,
    freq: np.ndarray,
) -> np.ndarray:
    """Interpolate a hydrophone transfer function onto a frequency axis.

    ``plot_specgram.m:64-77`` uses ``interp1(f, v, fq, 'linear', 'extrap')``.

    The ``'extrap'`` is load-bearing and cannot be delegated to ``np.interp``, which
    silently *clamps* to the end values instead of extrapolating. A transfer function
    typically starts at 10 Hz and stops at 100 kHz while the spectrogram axis runs
    from 0 to Nyquist, so the extrapolated tails are used on every plot -- clamping
    instead would quietly flatten the correction at both ends of the band.

    Extrapolation off the end of a measured calibration curve is a questionable thing
    to do, but it is what every existing Triton result was produced with, so it is
    reproduced rather than second-guessed.
    """
    fp = np.asarray(tf_freq, dtype=np.float64).ravel()
    vp = np.asarray(tf_value, dtype=np.float64).ravel()
    fq = np.asarray(freq, dtype=np.float64).ravel()
    if fp.size != vp.size:
        raise ValueError(f"transfer function has {fp.size} frequencies and {vp.size} values")
    if fp.size < 2:
        raise ValueError("linear interpolation needs at least two points")
    order = np.argsort(fp, kind="stable")
    fp, vp = fp[order], vp[order]

    out = np.interp(fq, fp, vp)
    # Replace the clamped tails with a linear continuation of the end segments.
    lo = fq < fp[0]
    if lo.any():
        slope = (vp[1] - vp[0]) / (fp[1] - fp[0])
        out[lo] = vp[0] + slope * (fq[lo] - fp[0])
    hi = fq > fp[-1]
    if hi.any():
        slope = (vp[-1] - vp[-2]) / (fp[-1] - fp[-2])
        out[hi] = vp[-1] + slope * (fq[hi] - fp[-1])
    return out


def display_filter(
    x: np.ndarray, fs: float, f1: float, f2: float
) -> np.ndarray:
    """Zero-phase FIR band-pass for display only.  Port of ``display_filter.m``.

    Display only, and that matters: it is applied to what is drawn, never to what is
    measured or written. ``filtfilt`` runs the filter forwards and backwards, which
    doubles the effective order and gives zero phase distortion -- appropriate for
    looking at a spectrogram, wrong for anything timing-sensitive.

    The order adapts to the data because ``filtfilt`` needs more than three times the
    filter order in samples; a short window would otherwise raise rather than draw.
    A degenerate band is returned unfiltered rather than refused, since a user dragging
    a frequency control through an invalid state should see the unfiltered data, not an
    error dialog.

    **Apply this once per window, not once per panel.** In MATLAB each of
    ``plot_specgram``, ``plot_timeseries`` and ``plot_spectra`` filters the shared
    ``DATA`` global and assigns the result back, so showing three panels band-passes the
    data three times and the panels disagree by up to ~20 dB at the filter corners
    (OPEN_DECISIONS §1.6). :meth:`triton.session.TritonSession.frame` calls this once
    and hands the same array to every panel.
    """
    from scipy.signal import filtfilt, firwin

    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()                        # remove DC before filtering
    wn = np.clip(np.asarray([f1, f2], dtype=np.float64) / (fs / 2.0), 1e-6, 1 - 1e-6)
    if wn[1] <= wn[0]:
        return x                            # degenerate band; leave as-is
    n_order = min(200, 2 * ((x.size - 1) // 6))
    if n_order < 4:
        return x                            # too short to filter meaningfully
    b = firwin(n_order + 1, wn, pass_zero=False)
    return filtfilt(b, np.asarray([1.0]), x)
