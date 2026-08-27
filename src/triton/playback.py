"""Listening to what is on screen.

Marine acoustic data is mostly not directly audible: a 200 kHz recording has ten
times the bandwidth of human hearing, and the things people look for -- echolocation
clicks, high-frequency whistles -- sit well above it. So playback is not simply
"send the samples to the sound card"; it is a resampling problem with a *speed*
control, and the speed control is the point rather than a convenience.

``speed`` follows ``PARAMS.speedFactor``'s meaning: it multiplies the effective sample
rate. ``speed = 0.1`` plays ten times slower and shifts everything down by a factor of
ten, which is how a 60 kHz click becomes a 6 kHz tick you can hear.

Two departures from ``audvidplayer.m``, both deliberate:

**Resampling instead of integer decimation.** MATLAB decimates by ``ceil(fs/200000)``
and then snaps ``speedFactor`` to an integer with ``floor``/``ceil``
(``audvidplayer.m:39-63``), so asking for 0.1x on a 200 kHz file gets you 1x. Here the
ratio is rational and arbitrary, so the requested speed is the speed you get.

**Level set from peak amplitude, not peak-to-peak, and with the mean removed.**
``audvidplayer.m:68`` computes ``vol * 2^16 / (max - min)``, which biases asymmetrically
on any signal with a DC offset -- real hydrophone data usually has one -- and divides by
zero on a silent window. Nothing measured depends on playback gain, so there is no
reason to reproduce that.

The 200 kHz ceiling in the MATLAB is not reproduced either; it was a MATLAB
``audioplayer`` limit, and the real constraint is whatever the output device supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

__all__ = ["Playable", "prepare", "suggest_speed", "Player"]

#: Below this, playback is inaudible on most equipment and the resampler is doing
#: pointless work. ``audvidplayer.m:35`` uses the same figure.
MIN_RATE = 1000


@dataclass(frozen=True)
class Playable:
    """Audio ready to hand to a device, and what had to be done to get it there."""

    samples: np.ndarray          # float32 in [-1, 1]
    rate: int                    # the device rate to play it at
    speed: float                 # the speed actually applied
    duration_sec: float
    #: Highest original frequency that survived. Below Nyquist of ``rate/speed`` means
    #: the resampler's anti-alias filter discarded content, which is worth telling the
    #: user -- they are listening for things that may not be there any more.
    audible_up_to_hz: float
    discarded_above_hz: float | None


def prepare(
    samples: np.ndarray,
    fs: float,
    *,
    speed: float = 1.0,
    volume: float = 1.0,
    out_rate: int = 48_000,
) -> Playable:
    """Resample and normalise a window for playback.

    Pure and device-free, so the arithmetic can be tested without a sound card --
    which matters because a sound card is exactly the thing CI does not have.

    The resampling ratio is ``out_rate / (fs * speed)`` expressed as a fraction, so any
    speed is reachable. ``scipy.signal.resample_poly`` supplies the anti-alias filter;
    without it, slowing a 200 kHz recording down would fold everything above the new
    Nyquist back into the audible band as noise that sounds like biology.
    """
    from scipy.signal import resample_poly

    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("nothing to play")
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if not 0 <= volume <= 1:
        raise ValueError(f"volume must be in [0, 1], got {volume}")

    effective = fs * speed
    if effective < MIN_RATE:
        raise ValueError(
            f"{fs:g} Hz at {speed:g}x is an effective rate of {effective:g} Hz, "
            f"below the {MIN_RATE} Hz floor -- raise the speed"
        )

    # Remove the mean before anything else: hydrophone data commonly carries a DC
    # offset, and normalising around it wastes headroom and clips one side first.
    x = x - x.mean()

    ratio = Fraction(out_rate / effective).limit_denominator(1000)
    if ratio != 1:
        x = resample_poly(x, ratio.numerator, ratio.denominator)

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x * (volume / peak)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)

    # What survived. Content above the output Nyquist, referred back to original
    # frequencies, was removed by the anti-alias filter.
    nyq_original = (out_rate / 2) / speed
    audible = min(fs / 2, nyq_original)
    discarded = nyq_original if nyq_original < fs / 2 else None
    return Playable(
        samples=x, rate=out_rate, speed=speed,
        duration_sec=x.size / out_rate,
        audible_up_to_hz=audible, discarded_above_hz=discarded,
    )


def suggest_speed(fs: float, out_rate: int = 48_000) -> float:
    """A speed at which the whole recorded band becomes audible.

    ``fs/2`` of original bandwidth has to fit below ``out_rate/2``, so the speed is
    ``out_rate / fs`` -- 0.24 for a 200 kHz recording on a 48 kHz device. Rounded down
    to something a person would choose, because a spin box reading 0.24 invites
    fiddling while 0.2 does not.
    """
    exact = out_rate / fs
    if exact >= 1:
        return 1.0
    for nice in (0.5, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01):
        if nice <= exact:
            return nice
    return max(exact, MIN_RATE / fs)


class Player:
    """Non-blocking playback with a stop.

    Thin on purpose: it owns no audio logic, only the device. ``sounddevice`` is
    imported lazily so that importing :mod:`triton.playback` -- and therefore the GUI --
    works on a machine with no audio stack at all, which is the normal case for a
    compute node or a CI runner.
    """

    def __init__(self) -> None:
        self._sd = None
        self.playing = False

    def _device(self):
        if self._sd is None:
            import sounddevice
            self._sd = sounddevice
        return self._sd

    def default_rate(self, fallback: int = 48_000) -> int:
        """The output device's preferred rate, or ``fallback`` if there is no device."""
        try:
            info = self._device().query_devices(kind="output")
            return int(info["default_samplerate"])
        except Exception:                             # noqa: BLE001
            return fallback

    def play(self, playable: Playable) -> None:
        sd = self._device()
        sd.stop()
        sd.play(playable.samples, playable.rate)
        self.playing = True

    def stop(self) -> None:
        if self._sd is not None:
            self._sd.stop()
        self.playing = False
