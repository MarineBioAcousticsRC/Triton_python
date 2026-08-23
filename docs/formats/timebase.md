# Timebase: representation, precision, and conversion at the boundaries

This is the single most consequential representation decision in the port. It is written up
separately because getting it wrong produces errors that are small enough to pass review and large
enough to invalidate results.

---

## 1. What Triton does today

Internally, every time is a MATLAB `datenum` (floating-point days since year 0) **shifted back by
2000 years**:

- The x.wav raw-file header stores `year` as one `uint8` holding `year - 2000`.
- `rdxwavhd.m:151` builds `datenum([11 01 30 08 45 00])` directly from those bytes — year 11 AD.
- `get_headers.m:62` and `motion.m:258` explicitly subtract `datenum([2000 0 0 0 0 0])` when
  deriving times from filenames, to match.
- Only `timestr.m` (line 25, `yoffset = 2000`) adds the 2000 back, at display time.

So `PARAMS.plot.dnum`, `PARAMS.start.dnum`, `PARAMS.raw.dnumStart/End`, `PARAMS.ltsa.dnumStart/End`
are all offset datenums. Any code that hands one of these to `datestr` gets a year-11 date.

## 2. Why the offset is load-bearing

A `double` has 52 bits of mantissa, so the spacing of representable values grows with magnitude.
Measured in MATLAB R2025b at the 2011-01-30 fixture date (`fixtures/reference/datenum_precision.json`,
produced by `dump_reference.m` rather than asserted here):

| Epoch | `eps(datenum)` in days | in time |
|---|---:|---:|
| Real 2011 (`datenum([2011 1 30 …])`) | 1.164e-10 | **10.06 µs** |
| Triton's shifted 2011 (`datenum([11 1 30 …])`) | 4.547e-13 | **0.039 µs** |

One sample at 200 kHz is **5 µs**. **A float datenum at the true epoch cannot resolve a single
sample at HARP rates** — 10.06 µs of granularity for a 5 µs sample. The 2000-year shift buys
about eight bits, taking the granularity to 1/128 of a sample, and is why sample-accurate seeking
works at all.

Whether this was deliberate or a side effect of the one-byte year field, the consequence is the
same: "cleaning it up" to real datenums, or to any float-days representation, silently breaks
timing on high-rate data. Anyone proposing that change should be shown this table first.

## 3. MATLAB `datetime` — the right fix if we were staying in MATLAB

Correct: `datetime` is the modern replacement and does not have this problem. It stores time as an
integer millisecond count plus a fractional part rather than as a single float-days value, so its
resolution does not degrade with distance from the epoch (documented as resolving to roughly a
nanosecond, and it accepts up to nine fractional-second digits in format strings).

Worth confirming empirically before relying on a specific figure — this settles it in one line:

```matlab
t = datetime(2020,1,1,'Format','yyyy-MM-dd HH:mm:ss.SSSSSSSSS');
[t, t + seconds(1e-9)]     % do these display, and compare, as distinct?
```

If Triton stays partly in MATLAB during the transition, moving the MATLAB side to `datetime` and
the Python side to the representation below means the two agree without either carrying the
2000-year hack.

## 4. Python equivalents

| Type | Backing | Resolution | Range | Verdict |
|---|---|---|---|---|
| `datetime.datetime` (stdlib) | — | 1 µs | year 1–9999 | **Too coarse.** 1 µs < 5 µs but leaves no headroom for arithmetic, and 1 MHz recorders exist |
| `numpy.datetime64[us]` | int64 | 1 µs | ±290,000 yr | Same objection |
| `numpy.datetime64[ns]` | int64 | **1 ns** | 1678–2262 | **Recommended.** Range covers any plausible recording |
| `pandas.Timestamp` | int64 ns | 1 ns | 1678–2262 | Same as above; convenient, adds a dependency to core |
| `decimal.Decimal` / `fractions.Fraction` | arbitrary | exact | unlimited | Exact but slow; use only for rate arithmetic |

`numpy.datetime64[ns]` (equivalently a plain `int64` nanosecond count) is the recommendation: it
is exact integer arithmetic, vectorises, and has direct MATLAB `datetime` interop. It is 200× finer
than a sample at 200 kHz and 5000× finer than what a real-epoch datenum can express.

## 5. Recommended internal representation

**Never store a time as a float.** Store position as an integer sample index within a known
segment, and derive wall-clock time only when needed:

```python
@dataclass(frozen=True)
class Segment:
    """One contiguous run of samples — an x.wav raw file, or a whole wav file."""
    index: int
    start: np.datetime64        # ns precision, true epoch (year 2011, not 11)
    n_samples: int              # per channel
    sample_rate: int            # Hz, integer -- the raw-file table value for x.wav
    byte_loc: int               # for x.wav; None otherwise
    gain: float

@dataclass(frozen=True)
class SamplePos:
    """The canonical cursor. Exact, orderable, arithmetic-safe."""
    segment: int
    offset: int                 # samples from the start of that segment
```

Conversion is exact in both directions:

```python
def to_time(pos, seg) -> np.datetime64:
    # integer math: no float ever appears
    return seg.start + np.timedelta64(pos.offset * 1_000_000_000 // seg.sample_rate, 'ns')

def to_pos(t, seg) -> SamplePos:
    delta_ns = int((t - seg.start) / np.timedelta64(1, 'ns'))
    return SamplePos(seg.index, delta_ns * seg.sample_rate // 1_000_000_000)
```

Note `offset * 1e9 // rate` truncates by at most 1 ns and is exact whenever `1e9 % rate == 0`
(true for 10 kHz, 200 kHz, 320 kHz, 500 kHz — the rates that matter here). Where it is not exact,
the *sample index* remains the source of truth; the timestamp is the derived quantity, which is
the correct direction of dependency.

## 6. Conversion at the boundaries — and only there

Legacy datenums must appear in exactly four places:

1. reading x.wav / LTSA headers (integer date fields → `datetime64`, adding the 2000 offset once);
2. writing x.wav / LTSA headers (subtracting it once);
3. `session.as_params()` and `.mat` export, for MATLAB interop;
4. reading and writing legacy pick/label files.

```python
TRITON_EPOCH_OFFSET_YEARS = 2000

def from_triton_datenum(dnum: float) -> np.datetime64: ...
def to_triton_datenum(t: np.datetime64) -> float: ...      # lossy; boundary use only
```

Both must be round-trip tested against MATLAB output in the parity suite, and
`to_triton_datenum` should carry a docstring saying plainly that it loses precision and must not
be used internally.

## 7. Filename timestamp parsing

`wavname2dnum.m` recognises five patterns, tried in this order (the order matters — `\d{12}` will
match inside a longer digit run):

| Order | Pattern | MATLAB format | Origin |
|---:|---|---|---|
| 1 | `\d{6}-\d{6}` | `yymmdd-HHMMSS` | Triton default |
| 2 | `\d{6}T\d{6}Z` | `yymmddTHHMMSSZ` | ISO8601-ish |
| 3 | `\d{12}` | `yymmddHHMMSS` | Avisoft, SoundTrap |
| 4 | `\d{8}_\d{6}` | `yyyymmdd_HHMMSS` | PAMGuard |
| 5 | `\d{6}_\d{6}` | `yymmdd_HHMMSS` | |
| 6 | `\d{8}T\d{6}` | `yyyymmddTHHMMSS` | AMAR |

Two-digit years are resolved by MATLAB `datenum`'s pivot (2000-based here). The result then has
`datenum([2000 0 0 0 0 0])` subtracted by the caller. Reproduce the **order** as well as the
patterns; a filename can match more than one.
