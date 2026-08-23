"""Time conversion at the boundaries, and only there.

Triton stores wall-clock times as MATLAB datenums shifted back by 2000 years: a
recording from 2011 has year 11 in its header. That looks like a bug and is not.
A double has 52 mantissa bits, so the spacing of representable values grows with
magnitude; at the real 2011 epoch a float datenum resolves 10.06 microseconds,
while one sample at 200 kHz is 5 microseconds. The shift buys about eight bits and
takes the granularity to 1/128 of a sample, which is why sample-accurate seeking
works at all. See ``docs/formats/timebase.md`` section 2, whose numbers come from
MATLAB itself rather than from arithmetic here.

The consequence for this port: **never store a time as a float**. Position is an
integer sample index within a known segment; wall-clock time is derived. Datenums
appear only when reading or writing legacy headers and files.

One practical constraint shapes the code below. ``numpy.datetime64[ns]`` spans
roughly 1678 to 2262, so it **cannot represent a shifted date at all** -- year 11
is far outside it. A shifted value therefore never becomes a ``datetime64``; the
offset is applied while the value is still a day count.

That offset is exactly 730485 days. 2000 years is a whole number of days in the
proleptic Gregorian calendar because 2000 is a multiple of the 400-year cycle of
146097 days, so the shift needs no calendar arithmetic and has no 29-February
edge case. It is applied to the integer day count rather than to the float,
because adding 730485 to a shifted datenum in floating point would move it from a
spacing of 39 ns to one of 10 microseconds -- discarding precisely the precision
the shift exists to provide.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime

import numpy as np

__all__ = [
    "TRITON_EPOCH_OFFSET_YEARS",
    "from_matlab_datenum",
    "to_matlab_datenum",
    "from_triton_datenum",
    "to_triton_datenum",
    "parse_filename_time",
    "FILENAME_PATTERNS",
]

#: Years subtracted from the true date before it is written to a header.
TRITON_EPOCH_OFFSET_YEARS = 2000

_NS_PER_DAY = 86_400_000_000_000
_UNIX_EPOCH_64 = np.datetime64("1970-01-01T00:00:00.000000000", "ns")


# ------------------------------------------------------------------ datenum <-> instant

#: 2000 years is **exactly** 730485 days in the proleptic Gregorian calendar,
#: because 2000 is a multiple of the 400-year cycle of 146097 days. Verified
#: across ordinary dates and leap days. So Triton's 2000-year shift is a pure
#: integer day offset, with no calendar arithmetic and no 29-February edge case.
_TRITON_DAY_OFFSET = 730485

#: MATLAB datenum 1 is 0000-01-01; ``date.toordinal()`` puts 1 at 0001-01-01.
_DATENUM_TO_ORDINAL = 366


def _split(dnum: float) -> tuple[int, int]:
    """Split a datenum into whole days and nanoseconds within the day.

    The day count is taken as an integer and the offset applied to *it*, never to
    the float. Adding 730485 to a shifted datenum in floating point would move it
    from a spacing of 39 ns to one of 10 microseconds and throw away exactly the
    precision the shift exists to provide (timebase.md section 2). Splitting first
    keeps the fractional part's magnitude below 1, where its spacing is negligible.
    """
    days = math.floor(dnum)
    frac_ns = int(round((dnum - days) * _NS_PER_DAY))
    if frac_ns >= _NS_PER_DAY:          # rounding carried into the next day
        days += 1
        frac_ns -= _NS_PER_DAY
    return days, frac_ns


def _join(days: int, frac_ns: int) -> float:
    """Inverse of :func:`_split`.  Lossy: the result is a float."""
    return days + frac_ns / _NS_PER_DAY


def _days_to_datetime64(days: int, frac_ns: int) -> np.datetime64:
    """Whole days since MATLAB's epoch, plus ns within the day, as an instant."""
    ordinal = days - _DATENUM_TO_ORDINAL
    if ordinal < 1:
        raise ValueError(
            f"datenum {days} is before 0001-01-01 and cannot be represented; "
            "if this is Triton's shifted timebase, use from_triton_datenum"
        )
    d = date.fromordinal(ordinal)
    day_start = np.datetime64(d.isoformat(), "ns")
    return day_start + np.timedelta64(frac_ns, "ns")


def _datetime64_to_days(t: np.datetime64) -> tuple[int, int]:
    total_ns = int((np.datetime64(t, "ns") - _UNIX_EPOCH_64) / np.timedelta64(1, "ns"))
    days, frac_ns = divmod(total_ns, _NS_PER_DAY)   # floor division: frac_ns >= 0
    return days + _UNIX_EPOCH_DATENUM, frac_ns


#: ``datenum(1970,1,1)``
_UNIX_EPOCH_DATENUM = 719529


def _to_datetime64(t: datetime, extra_ns: int = 0) -> np.datetime64:
    """A ``datetime`` plus extra nanoseconds, as a nanosecond ``datetime64``.

    Used where a time is built from integer calendar fields -- header timestamps
    and filename timestamps -- so no float ever enters the value.
    """
    whole = np.datetime64(t.replace(microsecond=0).isoformat(), "ns")
    return whole + np.timedelta64(t.microsecond * 1000 + extra_ns, "ns")


# ------------------------------------------------------------------------------ public

def from_matlab_datenum(dnum: float) -> np.datetime64:
    """Convert a true (unshifted) MATLAB datenum to a nanosecond ``datetime64``."""
    days, frac_ns = _split(float(dnum))
    return _days_to_datetime64(days, frac_ns)


def to_matlab_datenum(t: np.datetime64) -> float:
    """Convert a ``datetime64`` to a true MATLAB datenum.  Lossy; boundary use only."""
    return _join(*_datetime64_to_days(t))


def from_triton_datenum(dnum: float) -> np.datetime64:
    """Convert one of Triton's shifted datenums to a true instant.

    The 2000-year offset is added here, once, and as an integer number of days so
    the shifted value's precision survives the conversion.
    """
    days, frac_ns = _split(float(dnum))
    return _days_to_datetime64(days + _TRITON_DAY_OFFSET, frac_ns)


def to_triton_datenum(t: np.datetime64) -> float:
    """Convert a true instant to one of Triton's shifted datenums.

    Lossy -- the result is a float -- but less lossy than the unshifted form, which
    is the point of the shift.  Boundary use only: writing headers, exporting to
    MATLAB, and legacy pick files.
    """
    days, frac_ns = _datetime64_to_days(t)
    return _join(days - _TRITON_DAY_OFFSET, frac_ns)


# --------------------------------------------------------------------------- filenames

#: The patterns ``wavname2dnum.m`` tries, in its order.
#:
#: Taken from ``wavname2dnum.m`` itself, **not** from ``datepatterns.m``. Those are
#: two different sets and only one is used here: ``wavname2dnum`` hardcodes the
#: sequence below and never calls ``datepatterns``, which serves other callers.
#: Implementing from the wrong file silently loses three of the six styles.
#:
#: Order matters, and not only for tidiness: ``\d{12}`` will match inside a longer
#: run of digits, so a filename can satisfy more than one pattern and the first
#: match must win (timebase.md section 7).
FILENAME_PATTERNS: list[tuple[str, str, str]] = [
    ("Triton default",     r"\d{6}-\d{6}",   "yymmdd-HHMMSS"),
    ("ISO8601-ish",        r"\d{6}T\d{6}Z",  "yymmddTHHMMSSZ"),
    ("Avisoft, SoundTrap", r"\d{12}",        "yymmddHHMMSS"),
    ("PAMGuard",           r"\d{8}_\d{6}",   "yyyymmdd_HHMMSS"),
    ("underscore",         r"\d{6}_\d{6}",   "yymmdd_HHMMSS"),
    ("AMAR",               r"\d{8}T\d{6}",   "yyyymmddTHHMMSS"),
]


def parse_filename_time(name: str) -> np.datetime64 | None:
    """Extract a start time from a filename, or ``None`` if no pattern matches.

    Returns a **true** instant built from the integer fields, so no float enters
    the value. A two-digit year is read as 20xx, matching MATLAB ``datenum``'s
    pivot as ``wavname2dnum.m`` relies on it.

    ``None`` rather than an exception: a file whose name carries no timestamp is
    ordinary, and Triton's own behaviour is to ask the user.
    """
    for _label, pattern, fmt in FILENAME_PATTERNS:
        m = re.search(pattern, name)
        if not m:
            continue
        digits = re.sub(r"\D", "", m.group(0))
        n_year = 4 if fmt.startswith("yyyy") else 2
        if len(digits) != n_year + 10:
            continue
        try:
            year = int(digits[:n_year])
            if n_year == 2:
                year += 2000
            month, day, hour, minute, sec = (
                int(digits[n_year + 2 * i: n_year + 2 * i + 2]) for i in range(5)
            )
            t = datetime(year, month, day, hour, minute, sec)
        except ValueError:
            # The pattern matched digits that are not a valid date -- keep trying,
            # which is what MATLAB's sequential attempts amount to.
            continue
        return _to_datetime64(t)
    return None
