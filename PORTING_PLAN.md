# Triton → Python: architecture review and porting plan

Scope of this review: the `.m` files in the Triton-master **base folder** (~18k lines, of
which ~3.5k is vendored third-party code). Subfolders `io/`, `Remoras/`, `sigproc/`,
`stat/`, `gui/`, `whistle_extraction/` were excluded per instruction; see
[§1.4](#14-verdict-on-the-ignored-subfolders) for the verification of that call and the one
exception worth knowing about.

---

## 1. What is actually in the base folder

### 1.1 Startup chain

`triton.m` is 50 lines and does six things:

```
triton.m
  clear global; close all force; warning off
  PARAMS.ver = ...
  check_path      % locate Triton root, create Settings/Extras/Remoras, addpath installed Remoras
  initparams      % populate ~120 default fields of the PARAMS global
  initwins        % create 3 figures: main plot, control, message (+ 1 hidden bug-workaround fig)
  initcontrol     % 2,076 lines; builds ~119 uicontrols into the control figure
  init_coorddisp  % 385 lines; builds the message/coordinate-readout window
  initpulldowns   % File/Settings/Tools/Remoras/Help menus; then runs each Remora's initialize.m
```

There is no application object. The application *is* the three figures plus four globals.

### 1.2 Functional layers (as they exist today)

| Layer | Files |
|---|---|
| **Global state** | `PARAMS` (data + view + config), `HANDLES` (all widgets), `DATA` (current sample block), `REMORA` (plugin state) |
| **Audio I/O** | `rdxwavhd`, `rdwavhd`, `readseg`, `wrxwavhd`, `initdata`, `decimatewav(_dir)` |
| **LTSA I/O** | `read_ltsahead`, `read_ltsadata`, `write_ltsahead`, `calc_ltsa`, `get_headers`, `get_ltsadir`, `get_ltsaparams`, `init_ltsaparams`, `ck_ltsaparams`, `mk_ltsa`, `init_ltsadata`, `getIndexBin` |
| **Timebase** | `timestr`, `timenum`, `wavname2dnum`, `check_time`, `check_ltsa_time`, `sectohhmmss`, `dateregexp`, `datepatterns` |
| **DSP** | `mkspecgram`, `logfmap`, `amp_spec_scaling`, `loadTF`, `retrieveTransferFunction`, filter code inline in `control.m` |
| **Rendering** | `plot_triton` (layout dispatcher), `plot_specgram`, `plot_timeseries`, `plot_spectra`, `plot_ltsa`, `ltsa_delimiter` |
| **Interaction** | `control` (976 lines, 50 string-dispatched actions), `control_ltsa`, `motion`, `motion_ltsa`, `displaybut`, `coorddisp`, `pickxyz`, `pickxwav`, `handleKeypress`, `zoomChangeTime`, `stepPlotTimeLTSA`, `init_tslider` |
| **Menus** | `filepd` (15 actions), `toolpd`, `miscpd`, `remorapd`, `dtpd` |
| **Legacy conversion** | `hrp2xwav` (809 lines), `decompressRawHRP` (423), `little2big_2byte` |
| **Plugin system** | `check_path` + `Settings/InstalledRemoras.cnf` + `Remoras/*/initialize.m` + `REMORA` global |
| **Vendored** | `findjobj` (2,938 lines, Java handle hack), `xml_read` (550 lines) |

### 1.3 Files that appear to be dead

Zero inbound references from anywhere in the repo (base **or** subfolders):

- `batch_write_ltsahead.m`
- `bringToFront.m`
- `check_for_duplicate.m`
- `slider_change.m`
- `initialize.m` and `initialize_copy.m` — **these are not base files.** They are a copy of the
  Detector Remora's initializer sitting in the Triton root, referencing `REMORA.dt.*` and
  calling `dt_initparams`/`dt_initwins`/`dt_initcontrol`. Since `initpulldowns` loads Remoras by
  `cd(remorapath); eval('initialize')`, a stray `initialize.m` on the path is a genuine shadowing
  hazard. Recommend deleting from the MATLAB tree as well.
- `spFrameIndices.m` — duplicate of `sigproc/spFrameIndices.m`; the base copy is what creates the
  only apparent base→`sigproc` dependency, and nothing calls it.
- `findjobj.m` — used once, by `disp_pick`, purely to auto-scroll the picks listbox. Does not
  port and is not needed.

Confirm with the team before deleting, but none of these need a Python counterpart.

### 1.4 Verdict on the ignored subfolders

The instinct was right, with one caveat worth acting on.

- `gui/`, `stat/`, `whistle_extraction/`, `io/` — **zero** inbound references from base. Correct
  to exclude. (`gui/` and `whistle_extraction/` are Detector-Remora support code that ended up in
  the root tree.)
- `sigproc/` — the only base→sigproc edge is via the orphaned `spFrameIndices.m` above. Correct to
  exclude.
- **`io/` deserves a second look — not as a dependency, but as a specification.** It contains
  `ioReadXWAVHeader`, `ioReadXWAV`, `ioReadRIFFCk_harp`, `ioReadWavHeader`, `ioWriteLTSAHeader`
  etc.: a *struct-returning, non-global* reimplementation of exactly the parsing that
  `rdxwavhd.m`/`read_ltsahead.m` do against globals. It is unused by base Triton but is
  copy-pasted into **at least four Remoras** (`Detector/io/`, `SPICE-Detector/io/`,
  `BlueWhaleBcall-Detector/Detection/`, plus root `io/`), which have since diverged.

  Two conclusions. First, somebody already solved the "get PARAMS out of the I/O layer" problem
  fifteen years ago and the ecosystem voted for it with its feet — that's a data point for
  [§3](#3-the-params-question). Second, when the Python `read_xwav_header` exists, those divergent
  copies are the compatibility test set: if the Python reader reproduces all of them, Remora
  porting is unblocked.

---

## 2. Behaviour that must be preserved deliberately (the traps)

These are the things that will silently produce *slightly* wrong numbers if reimplemented from
intuition rather than from the source. Each needs a parity test.

### 2.1 The year-2000 datenum offset

Triton does not store real datenums. `timestr.m` line 25 is `yoffset = 2000`, and
`get_headers.m`/`motion.m` do `PARAMS.start.dnum = dnums - datenum([2000 0 0 0 0 0])`. The x.wav
raw-file header stores year as a single `uchar` (i.e. `yy`), and `rdxwavhd` builds
`datenum([11 01 30 ...])` — year 11 *AD*. Every internal time is a MATLAB datenum shifted back by
2000 years, and only the display layer adds the offset back.

This is easy to dismiss as a wart. It is not — it is load-bearing:

| Epoch | datenum magnitude | double spacing (`eps`) |
|---|---|---|
| Real 2020 | ≈ 737,800 days | ≈ 2.3e-10 d ≈ **20 µs** |
| Triton's shifted 2020 | ≈ 7,300 days | ≈ 1.8e-12 d ≈ **0.16 µs** |

At 200 kHz one sample is 5 µs. A float datenum at the true epoch **cannot resolve a single
sample**; the 2000-year shift buys about seven bits of headroom and is (accidentally or not) why
timing works at all. Any Python port that "cleans this up" by moving to real datetimes-as-floats
will quietly destroy sample-accurate timing on high-rate data.

**Recommendation:** do not carry floats at all. Internally represent time as
`(segment_index, integer_sample_offset)` plus an exact segment start time
(`numpy.datetime64[ns]` or int nanoseconds). Provide `to_triton_datenum()` /
`from_triton_datenum()` *only* at the boundaries where legacy files and MATLAB interop demand it,
and unit-test the round trip.

### 2.2 Sample rate comes from the raw-file table, not the `fmt ` chunk

`rdxwavhd.m:188`: `PARAMS.fs = PARAMS.xhd.sample_rate(1)` with the comment that the `fmt ` chunk
rate "could be fake". Any reader that trusts `soundfile`/`wave` on an x.wav gets the wrong answer.
This is the single most important reason the x.wav path cannot be folded into a generic audio
reader.

### 2.3 `readseg` byte arithmetic

```matlab
fseek(fid, PARAMS.xhd.byte_loc(index) + skip*PARAMS.nch*PARAMS.samp.byte, 'bof');
DATA = fread(fid, [PARAMS.nch, PARAMS.tseg.samp], dtype)';
```

`skip` is derived from `(plot.dnum − raw.dnumStart(index))`, i.e. seek is relative to the
*current raw file's* start, but the read then runs contiguously off the end of that raw file into
the next one. When the recording is duty-cycled, the samples after the boundary are from a
different wall-clock time; Triton draws a red dashed delimiter line
(`PARAMS.raw.delimit_time`) rather than inserting a gap. **This is the existing contract and
analysts read plots assuming it.** Reproduce it exactly, and make "splice across gaps" vs
"honour gaps" an explicit, documented option rather than an accident.

Also note `check_time.m` handles three distinct pathologies — pointer in a gap, pointer past the
last raw file, pointer in *two* raw files at once (overlapping headers from a wrong sample rate).
That triage logic is real-world hardening and should be ported, not simplified away.

### 2.4 Window function definition

`mkspecgram` and `calc_ltsa` both use MATLAB `hanning(N)`. MATLAB's `hanning` is the
**symmetric window with the zero endpoints removed**, which is *not* `scipy.signal.windows.hann(N)`
(includes zeros) and *not* `hann(N, sym=False)` (periodic). The exact equivalent is:

```python
w = scipy.signal.windows.hann(N + 2, sym=True)[1:-1]   # == MATLAB hanning(N)
```

Getting this wrong shifts every LTSA and spectrogram value by a fraction of a dB — small enough to
pass a smell test, large enough to break reproducibility against a decade of published figures.

### 2.5 Spectral scaling chain

- Spectrogram: `[~,f,t,ps] = spectrogram(x, win, noverlap, nfft, fs)` (4th output = one-sided PSD),
  then `PARAMS.pwr = 10*log10(abs(ps))` → dB re counts²/Hz. Note the older, differently-normalised
  formula is commented out directly above in `mkspecgram.m:26-28`; don't resurrect it by accident.
- LTSA: `pwelch(data, hanning(nfft), 0, nfft, fs)` → `10*log10` → **written to disk as `int8`**
  (`calc_ltsa.m:162`). LTSA values are therefore integer dB clipped to [−128, 127]. Preserve the
  int8 storage for file compatibility, including MATLAB's round-and-saturate behaviour.
- Transfer function (`plot_specgram.m:64-80`): dedupe TF frequencies, `interp1(...,'linear','extrap')`
  onto the plot frequency vector, add `bwdb = 10*log10(nfft/fs)`. Reproduce exactly, including the
  extrapolation.
- Display mapping: `c = (contrast/100) * dB + bright`, then `caxis([1,65])` — hard-coded, in both
  `plot_specgram` and `plot_ltsa`. Brightness/contrast are a colour-axis transform, not a data
  transform; keep them separable in the port.

### 2.6 Other items

- **x.wav header v2** adds per-channel `drate` (float32 × nch) in the harp chunk and per-channel
  `dt` (float32 × nch, time offset relative to channel 1) in each raw-file record, and changes the
  expected subchunk size formula. Both v0/v1 and v2 must be supported.
- **LTSA versions 1–4** differ in field widths (`uint16` vs `uint32` for `nrftot`/`nave`) and
  filename field length (40 vs 80 bytes), with different padding skips. All four appear in the wild.
- `PARAMS.xgain` division on the x.wav path only (`readseg.m:116-118`); wav path returns native
  ints via `audioread(...,'native')`.
- 24-bit is declared in `readseg` (`dtype='int24'`) but MATLAB `fread` has no `int24` — that branch
  is broken today. Python should actually implement it.
- `disp_msg`/`disp_pick` are the application log *and* the annotation store; picks are saved by
  scraping strings out of the listbox widget (`filepd.m:471-489`). See [§4.4](#44-annotations).

---

## 3. The PARAMS question

The two positions are both right about something, which is why the argument has lasted. The
engineers want a single, always-available, inspectable description of the loaded data — and they
are correct that this is essential. The objection to globals is about *coupling and mutation*, not
about *inspectability*. Those can be separated.

### What's actually wrong today

It isn't the global-ness in the abstract; it's five specific consequences:

1. `PARAMS` mixes four unrelated things: file metadata (`xhd`, `raw`, `fs`, `nch`), view state
   (`freq0`, `bright`, `cmap`, `tseg`), app config (`path.*`, `ioft`, `iocq`) and scratch buffers
   (`pwr`, `t`, `f`, `cb`, `cbb` — colorbar *handles* live in PARAMS).
2. Exactly one dataset can be open per MATLAB session. No second window, no A/B comparison, no
   "open the previous deployment alongside this one".
3. Nothing is testable without booting the GUI. Batch Remoras currently start Triton to get a
   populated `PARAMS`.
4. Plugins mutate `PARAMS` and `REMORA` freely — e.g. the HelloWorld example overwrites
   `PARAMS.keypress` wholesale, so two Remoras with keymaps silently clobber each other.
5. Change propagation is manual: every one of the ~50 actions in `control.m` hand-rolls its own
   "update field → call `plot_triton` → toggle 20 widget `Enable` states" sequence. Most GUI bugs
   in this codebase live in that pattern.

### Proposal: one session object, globally *reachable*, not globally *scoped*

```python
@dataclass
class TritonSession:
    audio:       AudioState        # source, header, sample clock, current segment
    view:        ViewState         # tseg, freq limits, nfft, overlap, bright/contrast, cmap, channel
    ltsa:        LtsaState         # ltsa file, tseg.hr, its own freq/bright/contrast
    calibration: CalibrationState  # transfer function, gain
    config:      AppConfig         # paths, export defaults, window layout
    plugins:     dict[str, Any]    # per-remora namespace, replaces the REMORA global
```

with:

- **`triton.current_session()`** — one process-level accessor, for the console and for plugin
  convenience. This is deliberately *a* global seam, but exactly one, documented and typed,
  instead of forty `global PARAMS` declarations.
- **`session.as_params()`** — returns a nested plain dict shaped like today's `PARAMS`, so existing
  mental models, docs, and `export_params`-to-`.mat` all keep working. This is the compatibility
  promise that should win over the pro-globals camp.
- **An embedded Python console in the app** (Qt `qtconsole` / embedded IPython kernel) with `S`
  pre-bound to the live session. This is *strictly better* than typing `PARAMS.xhd` at the MATLAB
  prompt: tab completion, docstrings, type hints, and it works while the app is running and idle,
  which is the stated requirement.
- **A "Session Inspector" dock panel** — live tree view of the whole session, filterable,
  copy-as-JSON, plus `session.snapshot()` to dump state + provenance for bug reports.
- **Change notification** — mutations go through the session and emit `changed(path)`; the UI
  subscribes. This deletes most of `control.m`.

What this buys that the global cannot: multiple open datasets, headless/batch use of the same code
paths, unit tests without a GUI, undo, and validation at assignment time.

The honest cost: plugin authors must write `session.view.nfft` instead of `PARAMS.nfft`, and must
receive the session rather than reach for it. That is a small, mechanical, one-time change, and it
is the change that makes everything else in this document possible.

---

## 4. Target architecture

```
triton/
  io/          format layer — pure functions, zero app state
    xwav.py        XWavHeader, read_xwav_header, write_xwav_header  (v0/v1/v2)
    wav.py         wav via soundfile
    flac.py        flac via soundfile  (near-free once the source protocol exists)
    ltsa.py        LtsaFile reader/writer, versions 1-4
    source.py      AudioSource protocol + open_audio() dispatch
    picks.py       legacy .pik.txt, label/annotation formats
  timing/      SampleClock, segment table, datenum interop, filename→time parsing
  dsp/         spectrogram, welch/LTSA, transfer function, filters, equalization, display mapping
  session/     TritonSession + state dataclasses + change signals
  plugins/     Remora API, discovery, registry
  ui/          Qt views; reads session, emits intents; no DSP, no I/O
  tools/       mk_ltsa, decimate, hrp2xwav (CLI + GUI wrappers over the core)
  api.py       the stable public surface plugins import
```

### 4.1 The `AudioSource` protocol — where x.wav fidelity lives

```python
class AudioSource(Protocol):
    sample_rate: int          # x.wav: from the raw-file table, NOT the fmt chunk
    n_channels: int
    dtype: np.dtype
    segments: list[Segment]   # contiguous runs: start_time, n_samples, byte_loc, gain
    def read_samples(self, segment: int, offset: int, n: int) -> np.ndarray: ...
    def read_at(self, t: Time, duration: float, *,
                splice_gaps: bool = True) -> tuple[np.ndarray, list[GapMarker]]: ...
```

`XWavSource` implements `read_samples` with the exact `byte_loc + offset*nch*bytes_per_sample`
seek. `WavSource`/`FlacSource` implement it over `soundfile` with a single synthetic segment whose
start time comes from the filename. **Keep the separate implementations** — as the brief says, the
apparent inefficiency is the point; a common protocol at the top is what prevents a fifth divergent
copy of the header parser, without forcing x.wav through a generic path that would lose byte-level
control.

`GapMarker` replaces `PARAMS.raw.delimit_time` and carries enough information for the renderer to
draw the delimiter lines.

### 4.2 GUI technology

**Recommendation: PySide6 (Qt, LGPL) + pyqtgraph** for interactive plots, matplotlib retained only
for publication-quality export.

Rationale: the auto-advance loop, LTSA panning, and the cursor readout on a large image all need
fast blitted updates that matplotlib does not deliver comfortably; pyqtgraph gives image display,
linked axes (the existing "Lock Axes" multichannel feature), ROI/rubber-band selection, and mouse
coordinate mapping natively. PySide6's LGPL avoids the GPL entanglement PyQt would create with the
existing UC copyright terms. Audio playback via `sounddevice` — and the fs > 200 kHz decimation
kludge in `audvidplayer.m` becomes proper resampling.

A web front end (Dash/Bokeh/Panel) was considered and is not recommended for v1: byte-level local
file access, low-latency playback, and in-process Python plugins all argue for a desktop app. The
layering above leaves the door open to add a web viewer over the same core later.

### 4.3 Remora / plugin API

Current contract: a directory containing `initialize.m`, registered by absolute path in
`Settings/InstalledRemoras.cnf`, loaded by `cd(dir); eval('initialize')`, which mutates
`HANDLES`/`REMORA`/`PARAMS` and installs string-eval callbacks. Requires a restart. No versioning,
no dependency declaration, no isolation, no API boundary — which is why I/O code got copy-pasted
into four Remoras.

Proposed:

```python
class Remora:
    name = "labelvis"
    version = "2.0"
    requires_triton = ">=1.0"

    def menu(self) -> list[MenuItem]: ...
    def panel(self) -> QWidget | None: ...           # optional dockable control panel
    def keymap(self) -> dict[str, Command]: ...      # conflicts detected, not clobbered
    def on_segment_loaded(self, session, segment): ...
    def on_pick(self, session, pick): ...            # replaces REMORA.pick.fcn eval-cell
    def overlay(self, kind: AxesKind, painter, session): ...  # replaces ltsa_plot_lVis_lab{n}
    def batch(self) -> Callable | None: ...          # headless entry; no GUI needed
```

- Discovery by Python entry point (`triton.remoras` group) or a plugins directory — pip-installable,
  versioned, no absolute-path config file, no restart required.
- Plugin state lives in `session.plugins["labelvis"]`; the shared `REMORA` global disappears.
- `triton.api` is the *only* import surface for plugins, so `read_xwav_header` never needs to be
  copied again.
- `batch()` means batch Remoras stop needing a GUI to obtain state — this alone removes a large
  class of current awkwardness.

Design this API in Phase 1–2 and validate it in Phase 4; it is what determines whether the
17-Remora ecosystem can follow.

### 4.4 Annotations — the one place to change the model, not just the language

Today: picks are formatted strings appended to a listbox, persisted by scraping the widget into
`*.pik.txt`. Detections and labels each live in per-Remora formats. Given that annotation is half
the stated purpose of the tool, this is the highest-value structural change available, and it is
cheap now and expensive later:

```python
@dataclass
class Annotation:
    start: Time; end: Time | None
    f_low: float | None; f_high: float | None
    channel: int
    label: str
    author: str; created: datetime
    confidence: float | None
    source: str                # "manual" | remora name
    attrs: dict                # free-form, plugin-owned
```

with a documented on-disk format, readers for legacy `.pik.txt` and the common Remora label files,
and the message pane re-backed by the model rather than being the model. Everything else in the
port is a translation; this is the one deliberate upgrade recommended out of the gate.

---

## 5. Phased plan

**Phase 0 — Specification and golden fixtures** *(do this while MATLAB is still the reference)*
- Write byte-exact format specs for x.wav (v0/v1/v2) and LTSA (v1–4) from `rdxwavhd`, `wrxwavhd`,
  `read_ltsahead`, `write_ltsahead`, cross-checked against `io/ioReadXWAVHeader`.
- Assemble a small fixture corpus: x.wav v1 and v2, 16/24/32-bit, single- and multi-channel,
  continuous and duty-cycled; plain wav; flac; matching `.ltsa` files.
- Write a MATLAB script that dumps reference outputs for each fixture: header as JSON, a
  spectrogram matrix, an LTSA read block, a set of timestamps, a `readseg` byte range. **These
  become the parity test suite.** Without this the port cannot be shown to be correct.

**Phase 1 — Core library, no GUI**
`triton.io` + `triton.timing` + `triton.dsp`, tested against Phase 0 fixtures.
Milestone: `python -m triton.tools.mkltsa <dir>` produces an `.ltsa` **byte-identical** to
MATLAB's for the fixture set. That is an unambiguous, checkable proof the numerics are right.

**Phase 2 — Session layer + headless viewer API**
`TritonSession`, change signals, `as_params()`, snapshot/inspection. Milestone: open a file, seek
to a time, retrieve a spectrogram tile and an LTSA tile — entirely in a test, no GUI.

**Phase 3 — GUI v1** *(the long pole)*
Main plot window with the four stacked panel types and their layout rules; control panel
(motion, time entry, tseg/step, freq limits, nfft/overlap, brightness/contrast/colormap, channel,
filter, TF, equalization); message/pick window; cursor readout; keymap; playback; LTSA→x.wav
expand-on-click. Port `check_time`'s gap/overlap triage verbatim.

**Phase 4 — Plugin API + first real Remora**
Port HelloWorld, then one substantive Remora (LabelVis or the Detector STS batch) as the API
shakedown. Do this *before* the tool ports — the API shape is the highest-risk unknown.

**Phase 5 — Tools**
mk_ltsa GUI/CLI, decimate, transfer-function loading, exports (wav/x.wav/mat/figure).
`hrp2xwav` + `decompressRawHRP` are ~1,200 lines of legacy HARP disk handling — port as a separate
`triton-hrp` CLI package, not core.

**Phase 6 — Migration and distribution**
Side-by-side running against real deployments, docs, `pip install triton` for Python users plus a
frozen build (PyInstaller/conda-forge) to replace Triton-Compiled.

### Rough effort

| Phase | Estimate (FTE) |
|---|---|
| 0 — spec + fixtures | 2–3 weeks |
| 1 — core library | 4–6 weeks |
| 2 — session layer | 2 weeks |
| 3 — GUI v1 | 8–12 weeks |
| 4 — plugin API + 1 Remora | 3–4 weeks |
| 5 — tools | 3–4 weeks |
| 6 — migration | 3–4 weeks |

≈ 6 months FTE to reach parity with base Triton. Porting the 17-Remora ecosystem is a separate and
larger ongoing effort — which is exactly why the plugin API deserves disproportionate design
attention early.

---

## 6. Decisions — all accepted 2026-08-19

> **Status: all seven were reviewed and accepted.** Amendments made during that review, and the
> requirements that emerged with them, are recorded in [HANDOFF.md §2–3](HANDOFF.md). Phase 0 is
> complete; see [README.md](README.md). The list below is kept in its original form as the
> rationale.

1. **Timebase representation** — integer samples + exact segment start times, with datenum
   conversion confined to file/MATLAB boundaries. *(Strong recommendation; see §2.1.)*
2. **PARAMS** — single `TritonSession` object, one documented `current_session()` accessor, plus
   an inspector panel and an embedded console. *(See §3.)*
3. **GUI toolkit** — PySide6 + pyqtgraph, matplotlib for export only. *(See §4.2.)*
4. **Gap handling** — reproduce the current splice-and-delimit behaviour by default; make
   gap-honouring an explicit option.
5. **Annotation model** — introduce a first-class `Annotation` type now rather than porting the
   listbox-as-database. *(See §4.4.)*
6. **Plugin API** — entry-point discovery, typed hooks, per-plugin session namespace, `triton.api`
   as the only import surface. *(See §4.3.)*
7. **What not to port** — the dead files in §1.3, `findjobj`, `xml_read`, the `cd()`-everywhere
   behaviour, `C:` path defaults, and `clear global` at startup.
