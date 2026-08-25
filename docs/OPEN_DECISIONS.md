# Open decisions

A register of things found in Triton that **should not be decided by whoever happens to be editing
the code next**. Mostly these are MATLAB behaviours that look like mistakes, where changing them
would alter numbers that analysts have already published, or questions that need data or a
particular person's memory rather than more reading.

They were previously scattered across `docs/formats/*.md` section 6, `HANDOFF.md`, and commit
messages. Scattered is how they get lost, hence this file.

**How to use it.** Each entry says what the behaviour is, the evidence, what would change if it
were "fixed", and who can settle it. Nothing here is a to-do list for one person. When an entry is
resolved, move it to [§4](#4-settled-for-the-record) with the decision and the commit — do not
delete it, so the same argument does not get had twice.

**A note on the general shape of these.** Almost every entry in §1 has the same structure: a
one-line quirk, invisible in normal use, that has silently been part of every result the lab has
produced for years. That is exactly why they are not obvious bug fixes. Reproducing a quirk keeps
old and new results comparable; fixing it makes new results more correct but no longer comparable.
Both are defensible and the choice belongs to the people whose papers depend on it.

| # | Item | Would change | Who decides |
|---|---|---|---|
| [1.1](#11-the-last-spectral-average-of-each-raw-file-re-reads-earlier-data) | `calc_ltsa` last-average overlap | LTSA bytes | Lab — affects published LTSAs |
| [1.2](#12-decimatex-uses-two-different-anti-aliasing-corner-frequencies) | `decimateX` filter corner, 0.8 vs 0.84 | decimated wav bytes | Whoever owns the decimated products |
| [1.3](#13-only-the-last-raw-files-dt-and-padding-survive) | `rdxwavhd` `dt`/`padding` overwrite | v2 multichannel timing | Needs the Remora audit first |
| [1.4](#14-gain-is-applied-to-one-channel-only) | multichannel gain-corrected data | Lab — has this ever bitten? |
| [1.5](#15-two-end-of-file-times-are-one-sample-short) | which raw file a time resolves to | Low stakes, but pick one |
| [1.6](#16-the-display-filter-is-applied-once-per-visible-panel) | Display filter compounds per visible panel | on-screen dB near the filter corners | Fix, but tell filter users |
| [2.1](#21-is-audioformat-3-or-int32-for-32-bit-xwav) | 32-bit `AudioFormat` | how 32-bit files are read | Needs one real 32-bit file |
| [2.2](#22-do-v2-header-xwav-files-exist) | severity of the `io/` v2 blindness | Needs an archive search |
| [2.3](#23-do-xwav-files-with-other-than-1-or-4-channels-exist) | whether `blksz` needs more cases | Needs an archive search |
| [2.4](#24-blksz-assumes-16-bit-samples) | `write_length` for 32-bit | Coupled to 2.1 |
| [2.5](#25-is-the-4-channel-multidir-fork-superseded) | HARPproc merge scope | HARPproc authors |
| [2.6](#26-what-are-harplabs-25-unique-lines-in-hrp2xwav_multidir) | HARPproc merge scope | HARPLab author |
| [2.7](#27-is-extrasmsdxfer-still-used) | whether to keep it | Lab |
| [3.1](#31-an-ltsa-can-name-at-most-65535-source-files) | nothing — limit, with a fix | Settled: split |
| [3.2](#32-an-ltsa-cannot-exceed-4-gb-of-spectra) | nothing — limit | Settled: split |

---

## 1. Behaviour changes that would alter output

### 1.1 The last spectral average of each raw file re-reads earlier data

**What.** `calc_ltsa.m:95-100` advances the input file pointer by *this* spectral average's sample
count, but the distance it needs to move is the *previous* average's length:

```matlab
if n == 1
    xi = PARAMS.ltsahd.byte_loc(m);
else
    xi = xi + (nsamp * (PARAMS.ltsa.nBits/8) * PARAMS.ltsa.nch);
end
```

`nsamp` equals `sampPerAve` for every average except the last one of a raw file that does not
divide evenly, where it is short (`calc_ltsa.m:81`). So on that last average the pointer
under-advances by `(sampPerAve − nsamp) × bytesPerSample × nch` and the average is computed from
data the previous average already covered.

**Evidence.** Reproduced exactly in `tools/mkltsa.py` (marked `BUG-COMPATIBLE`), which is how the
byte-identical parity test passes. Worked example, the 10 kHz fixture — 25,000 samples per raw
file, `sampPerAve` 10,000, so `nave` is 3 against an exact 2.5:

| average | reads from sample | should read from |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 10,000 | 10,000 |
| 3 | **15,000** | 20,000 |

The third average is short (5,000 samples), so it covers 15,000–20,000 where it should cover
20,000–25,000. It never reads outside the raw file, so nothing looks corrupt.

**Scope.** Any LTSA whose `tave` does not divide its raw files evenly, in the last time bin of
every raw file. For a typical HARP deployment that is one bin in every 75 s write, so on the order
of 1% of bins — and always the same bin, so it is a systematic bias rather than noise. The affected
bin duplicates a *portion* of the preceding data, so its values are plausible and correlated with
their neighbour, which is why this has never looked like anything.

**If fixed.** Every LTSA made afterwards differs from every LTSA made before, in ~1% of bins. Old
and new LTSAs of the same data would no longer be byte-identical, which matters because that
identity is currently the regression check. Detectors that run on LTSA values would see a small
change in a systematic position.

**Options.** (a) Leave it and document. (b) Fix it and re-make LTSAs, treating it as a data
version bump. (c) Fix it behind a flag defaulting to the old behaviour, so new work can opt in.
(c) has the pleasant property that the parity test keeps working while the correct path is
available.

**Who decides.** The lab, not the port. Anyone whose results depend on LTSA values should weigh
in — this is small but it is real, and it is in published data.

**Status.** Reproduced faithfully in the Python writer. Undecided in MATLAB. Raised 2026-08-23.

### 1.2 `decimateX` uses two different anti-aliasing corner frequencies

**What.** The entire difference between the two `decimateX.m` versions in the source trees is a
toggled comment on the IIR filter's corner frequency:

```matlab
Triton-HARPLab, Triton/     fc = 0.84/r;   % try corner nearer to Nyquist
DataProcessing              fc = 0.8/r;    % original corner frequency
```

Someone was experimenting, left both lines in, and the trees ended up on opposite sides. HARPLab
and the old `Triton/` copy are otherwise byte-identical.

**Scope.** It is on the **default** path — the filter type defaults to IIR and almost every call
uses the two-argument `decimateX(data, PARAMS.df)` form. Only `decimatewav.m:174` asks for FIR. So
data decimated with HARPLab is **not** bit-identical to the same input decimated with
DataProcessing, and which filter produced a given file is invisible in the output.

**Trade-off.** `0.8/r` keeps more margin against aliasing and rolls off more of the top of the
passband; `0.84/r` preserves more usable bandwidth with less margin. This is a signal-processing
call about whether analysts use the top few percent of the decimated band.

**Whichever is chosen**, it should stop being a commented-out toggle — a named parameter, or at
minimum a documented default recorded somewhere the output can be traced to.

**Who decides.** Whoever owns the decimated products. **Status.** Open, raised 2026-08-23.

### 1.3 Only the last raw file's `dt` and `padding` survive

**What.** `rdxwavhd.m:135,137` assign `PARAMS.xhd.dt` and `PARAMS.xhd.padding` **unsubscripted**
inside the per-raw-file loop, so each iteration overwrites the last. For a v2 multichannel file the
per-channel time offsets of every raw file but the final one are discarded.

**Why it is not simply a bug.** It may be deliberate, and some Remoras are believed to compensate
for the last-value-wins behaviour. Changing what *consumers* see could therefore break code that
has been quietly correcting for it.

**Where the port stands.** `io/xwav.py` exposes the full per-raw-file `dt` array. That is strictly
more information, so nothing can break by having it, and it does not commit to a decision.

**Blocked on.** Finding and auditing the compensating Remora code. Until then, do not change what
MATLAB hands to callers. **Status.** Open, deferred deliberately since Phase 0.

### 1.4 Gain is applied to one channel only

**What.** `readseg.m:117-119`:

```matlab
if PARAMS.xgain > 0
    DATA(:,PARAMS.ch) = DATA(:,PARAMS.ch) ./ PARAMS.xgain(1);
end
```

Two things at once. Only `xgain(1)` is used although the header carries one gain per raw file. And
the division is applied to column `PARAMS.ch` alone — the channel being *displayed* — so on a
multichannel file the other channels come back unscaled.

**Scope.** Only bites on a multichannel file recorded with a gain other than 0. Unverified whether
any exist: no fixture has both, so the port reproduces the behaviour without an oracle for it (a
recorded coverage gap). **If a 4-channel deployment ran with gain, three of its four channels have
always come back unscaled.**

**Who decides.** The lab — and the first question is factual: has any multichannel deployment used
a non-zero gain? If not, this is theoretical and can be fixed freely.

**Status.** Open. Reproduced in `io/audio.py::_apply_gain`.

### 1.5 Two end-of-file times are one sample short

**What.** Two independent places subtract a sample period when computing where something ends:

- `rdxwavhd.m:160` — a raw file's end is `start + (byte_length − 2) / ByteRate`, i.e. one 16-bit
  sample short, and it divides by the **`fmt` chunk** `ByteRate` rather than anything derived from
  that raw file's own `sample_rate`.
- `read_ltsahead.m:159` — an LTSA entry's end is `start + (dur − 1/fs)`, subtracting one *sample*
  period from a duration measured in *time bins*.

**Why it matters at all.** These end times decide which raw file a requested time resolves to
(`check_time.m:65`, `read_ltsadata.m:23`), so they are not merely cosmetic. The `− 2` also makes
duty-cycle gaps report as very slightly longer than they are — the fixtures show 7.5001 s where the
truth is 7.5 s.

**Low stakes**, but worth settling rather than leaving two different conventions in place.
**Status.** Open. Both reproduced exactly.

### 1.6 The display filter is applied once per visible panel

**What.** `plot_specgram.m:27`, `plot_timeseries.m:28` and `plot_spectra.m:27` each do the same
thing to the *shared global*:

```matlab
if PARAMS.filter
    DATA(:,PARAMS.ch) = display_filter(DATA(:,PARAMS.ch), PARAMS.fs, PARAMS.ff1, PARAMS.ff2);
end
```

`DATA` is set once by `readseg` before `plot_triton` runs, and `plot_triton` then calls each
enabled panel in turn. So with the display filter on and all three of those panels shown, the data
is band-passed **three times cumulatively** — and because the assignment writes back, the panel
drawn *first* sees single-filtered data while the panel drawn *last* sees triple-filtered data.
Which panels are enabled changes what each one shows.

This is original behaviour, not a regression: before the shared `display_filter.m` was factored
out, each routine did `DATA(:,PARAMS.ch) = filter(b,a,DATA(:,PARAMS.ch))` with its own elliptical
design. The write-back has always been there.

**Measured**, 1 s at 10 kHz, band-pass 200–3000 Hz, comparing the spectra panel (3 passes) against
the spectrogram panel (1 pass):

| | difference |
|---|---|
| total in-band power | +0.07 dB |
| median across the passband | +0.01 dB |
| **worst, at the upper corner (3000 Hz)** | **−21.8 dB** |
| median over 200–300 Hz, just inside the lower corner | −3.1 dB |

So the panels agree in the middle of the band and disagree by up to ~20 dB at the corners. Each
`filtfilt` pass multiplies the frequency response, so a flat mid-band stays flat while the roll-off
skirts are effectively cubed. An analyst reading levels near their own chosen filter corners off
the spectra panel is reading numbers well below what the spectrogram immediately above it shows.

**The same routines also decrement `PARAMS.ch`** (`plot_specgram.m:22` and friends) when the LTSA
panel is shown and multichannel mode is on — once per enabled panel. So in that configuration
three panels display three *different channels* while all being labelled the same.

**Recommendation: fix rather than reproduce**, unlike the other entries in this section. The
behaviour is order-dependent and panel-count-dependent, so nobody can be deliberately relying on
it; nothing written to a file depends on it; and the correct behaviour is unambiguous. It does
change what is on screen for anyone using the display filter, so it is worth telling those users
rather than doing it silently.

**Status.** Not reproduced in the Python port: `session.frame()` computes the filtered samples
**once** and hands the same array to every panel, which is the structural fix — panels receive data
rather than mutating shared state. Open on the MATLAB side. Found 2026-08-23 while scoping Phase 3.

---

## 2. Questions needing data or someone's knowledge

### 2.1 Is `AudioFormat` 3 or int32 for 32-bit x.wav?

`wrxwavhd.m:107` writes `AudioFormat = 3` (IEEE float) for 32-bit files; `readseg.m:100` reads them
as `int32`. One of the two is wrong. **Needs one real 32-bit HARP x.wav to settle.** The synthetic
fixture writes `AudioFormat = 1` with int32 samples, and the Python reader matches the reader.
**Status.** Open since Phase 0.

### 2.2 Do v2-header x.wav files exist?

Determines the severity of a real bug: **four Remora copies of `ioReadXWAVHeader.m`** — under
`Remoras/Detector/io/`, `Remoras/SPICE-Detector/io/`,
`Remoras/BlueWhaleBcall-Detector/Detection/`, and the root `io/` — have no v2 branch at all. They
read the 8-byte `Reserved` field where `drate` sits and use fixed 32-byte directory entries, so on
a v2 file they return wrong `byte_loc`/`byte_length` **without warning**. If v2 files exist,
results those Remoras produced from them are suspect.

No v2 file has been seen: all 123 x.wavs in `ExampleData` are v1. **Needs an archive search.**
**Status.** Open, and the highest-value of these questions.

### 2.3 Do x.wav files with other than 1 or 4 channels exist?

`ck_ltsaparams.m:48-57` only defines sector geometry (`blksz`) for 1 and 4 channels and prints an
error otherwise. If 2- or 3-channel files exist, LTSA creation cannot handle them.
**Needs an archive search.** **Status.** Open since Phase 0.

### 2.4 `blksz` assumes 16-bit samples

`ck_ltsaparams.m:50` computes `(512 − 12)/2`, where the `/2` is bytes per sample. For 32-bit data
the sector arithmetic — and therefore `write_length` bookkeeping and `nave` — is wrong. Coupled to
[2.1](#21-is-audioformat-3-or-int32-for-32-bit-xwav): both need the same real file.
**Status.** Open since Phase 0.

### 2.5 Is the 4-channel multidir fork superseded?

`Triton-SpotCheck/.../hrp2xwav_multidir_4chTimeHeader.m` is 1,056 lines, **82% line-similar** to
the newest plain `hrp2xwav_multidir.m` (HARPproc_260304, 1,083 lines) with 179 lines not present
there. Its header comment is a verbatim copy of the plain version's and documents nothing about
what differs.

Four-channel support clearly exists in the mainline — the `3B*` firmware rows are all `nch=4` and
are in the current table. So the narrow question is whether this fork's **time-header** handling was
folded in or still lives only here. **Needs the author.** **Status.** Open, for the HARPproc merge.

### 2.6 What are HARPLab's 25 unique lines in `hrp2xwav_multidir`?

Relative to HARPproc_260304, HARPLab's copy has **25 lines not present there** (and 260304 has 87
that HARPLab lacks). So the merge task is reviewing 25 specific lines, not reconciling two forks.
**Needs the HARPLab author.** **Status.** Open, for the HARPproc merge.

### 2.7 Is `Extras/msdxfer` still used?

All three copies of `msdxfer.exe` are **byte-identical** — 56,035 bytes, same checksum, same 2021
setup notes. SpotCheck simply lacks it. So retaining it costs nothing and there is no version to
reconcile; the only question is whether to carry it forward at all. **Status.** Open, low stakes.

---

## 3. Format limits, not bugs

Recorded so nobody spends time looking for a bug that is not there.

### 3.1 An LTSA can name at most 65,535 source files

`nxwav` is **uint16** while `nrftot` was widened to uint32 in version 4 (`write_ltsahead.m:63,83`).
So an LTSA can describe four billion raw files but only 65,535 source audio files, regardless of
how little data each holds. This is the "maximum reference limit" hit when a deployment is many
small files rather than a few large ones — the limit is on the *file count*, not the data volume,
so it bites exactly when each file contributes least.

**Cannot be widened without a version 5**, which every existing reader would have to learn. So
splitting the work across several LTSAs is the fix, not a workaround.

**Status.** Settled. `write_ltsahead.m` now refuses instead of letting `fwrite` saturate the field
and silently record the wrong count. `tools/mkltsa.py::plan` refuses with an explanation and
`split_inputs()` chunks a list to fit; the CLI's `--split` writes several LTSAs.

### 3.2 An LTSA cannot exceed 4 GB of spectra

`byteloc` is uint32, and `write_ltsahead.m:131-135` aborts past `2^32`. Same shape as 3.1 and the
same answer: split. **Status.** Settled, guard reproduced in `tools/mkltsa.py::plan`.

---

## 4. Settled, for the record

So the same arguments are not had twice.

| Item | Decision | Where |
|---|---|---|
| `WavVersionNumber` stored as ASCII `'1'` by `mk_SpotCheck` and as integer `1` by `write_XWAVhead`; 112 of 123 `ExampleData` files use the ASCII form | Normalise 48/49/50 to 0/1/2 on read, so the `== 2` tests can fire whichever way the byte was written. Verified: 451/451 harness cases unchanged | `Triton_remoras` `60a7f02` |
| `check_ltsa_time` commented out, breaking the LTSA motion buttons (issue #129) | Restored. It had been disabled by the PR #116 squash, which carried a `Revert "Merge branch 'master'"` | `Triton_remoras` `c6a15dc` |
| `write_ltsahead`'s `exist('PARAMS.ltsa.outfile','var')` guard — `exist` cannot test a dotted field name, so the Save dialog always opened and the writer could not run headless | Changed to `~isfield(...)`. This is what let the LTSA fixtures be built from `matlab -batch` | `Triton_remoras` |
| 24-bit x.wav cannot be read (`readseg.m:98` asks `fread` for a nonexistent `'int24'`) | Confirmed with the team: no 24-bit x.wav exists. Implemented correctly in Python anyway, since it was nearly free; needs no fixture | xwav.md 6.2 |
| `as_params()`'s purpose | **Inspection and debugging convenience for expert users**, not a compatibility layer. Remoras get modified to use the session object. This frees it from having to reproduce `PARAMS`'s quirks faithfully | PORTING_PLAN §3, decided 2026-08-23 |
| Phase 2 change notification | **Framework-agnostic** callback registry, not Qt signals, so the session layer stays importable without a GUI | PORTING_PLAN §3, decided 2026-08-23 |

## 5. Pending the HARPproc merge

Not decisions — known fixes with no home yet, because the files they touch are not in
`Triton_remoras` and will arrive with the HARPproc consolidation.

1. **`mk_SpotCheck.m:391`** assigns `PARAMS.xhd.WavVersionNumber = '1'` where every neighbouring
   line genuinely is a string. Drop the quotes. Fixes new files only; the 112 existing ones keep
   their `49` regardless, which is why the reader-side fix was the one that mattered.
2. **`ckFirmware.m:29`** matches firmware with `any(findstr(vnum, fwCell{i,1}))` — a *substring*
   test — and does not `break`, so the **last** matching row wins. A firmware string that is a
   prefix of several entries (`V2.9` against `V2.98`, `V2.97`, `V2.96`, `V2.95`) silently takes
   whichever is last in the CSV, with no warning. Wants an exact match plus a warn-on-multiple.
   (`findstr` is also long deprecated in favour of `strfind`.)
3. **Three of the five `table_ckFirmware.csv` copies are three years stale** — HARPLab, SpotCheck
   and the old `Triton/` copy top out at `3A01220318` (2022) while DataProcessing's reach
   `3A01251028`. Verified that adopting the newest is **purely additive**: no copy holds a row the
   newest lacks, and the only differences in shared rows are in the trailing comment column, with
   all eleven functional columns identical. Consequence of not doing it: those three checkouts
   cannot process single-channel compressed HARP data from 2024 onward at all.
