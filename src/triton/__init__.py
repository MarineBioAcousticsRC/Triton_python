"""Triton — Python port of the Scripps Whale Acoustics Lab passive-acoustic analysis package.

Phases 0 and 1 are complete. See README.md, PORTING_PLAN.md, and HANDOFF.md for state.

    triton.io       x.wav / wav / ltsa readers, no application state
    triton.timebase the year-2000 datenum shift and filename time parsing
    triton.dsp      window, spectrogram, Welch averaging, int8 quantiser, transfer functions
    triton.tools    mkltsa -- byte-identical to MATLAB's calc_ltsa
    triton.session  TritonSession: what is open and how it is being looked at

Two things a newcomer should read before changing anything here:

* **docs/formats/timebase.md.** Triton stores times as MATLAB datenums shifted back
  2000 years. It looks like a bug; it is what buys sub-microsecond resolution at real
  sample rates. Never store a time as a float in this package -- position is an
  integer sample index and wall-clock time is derived.
* **docs/OPEN_DECISIONS.md.** Several MATLAB behaviours reproduced here look wrong and
  are load-bearing, because changing them would alter numbers the lab has published.
  They are not to be tidied unilaterally.

`tests/test_parity_phase1.py` is the parity suite against MATLAB and should stay green;
`tests/test_session.py` additionally asserts that the session layer changes no numbers.
"""

from . import colormaps, dsp, export, io, session, timebase
from .session import (
    TritonSession,
    current_session,
    open_session,
    set_current_session,
)

__version__ = "0.0.2.dev0"

__all__ = [
    "io",
    "dsp",
    "export",
    "colormaps",
    "timebase",
    "session",
    "TritonSession",
    "current_session",
    "set_current_session",
    "open_session",
    "__version__",
]
