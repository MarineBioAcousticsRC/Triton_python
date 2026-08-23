"""Triton — Python port of the Scripps Whale Acoustics Lab passive-acoustic analysis package.

Phase 0 (specification and golden fixtures) is complete; see README.md and PORTING_PLAN.md.
Nothing is implemented yet.  Phase 1 lands three subpackages here:

    triton.io      x.wav / wav / flac / ltsa readers and writers, no application state
    triton.timing  SampleClock, segment table, datenum interop at the boundaries
    triton.dsp     spectrogram, welch/LTSA, transfer functions, display mapping

`tests/test_parity_phase1.py` is the definition of done for that phase: it is written already
and currently skips.
"""

__version__ = "0.0.1.dev0"
