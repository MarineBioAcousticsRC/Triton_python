"""Readers and writers for Triton's file formats.

``triton.io`` is the only import surface for format work; the modules beneath it
are implementation detail and may be reorganised.
"""

from ..timebase import (
    from_matlab_datenum,
    from_triton_datenum,
    parse_filename_time,
    to_matlab_datenum,
    to_triton_datenum,
)
from .audio import (
    AmbiguousTime,
    AudioSource,
    Boundary,
    Segment,
    TimeNotInData,
    WavSource,
    XwavSource,
    open_audio,
)
from .xwav import RawFile, XwavHeader, read_xwav_header

__all__ = [
    "open_audio",
    "AudioSource",
    "XwavSource",
    "WavSource",
    "Segment",
    "Boundary",
    "TimeNotInData",
    "AmbiguousTime",
    "read_xwav_header",
    "XwavHeader",
    "RawFile",
    "from_triton_datenum",
    "to_triton_datenum",
    "from_matlab_datenum",
    "to_matlab_datenum",
    "parse_filename_time",
]
