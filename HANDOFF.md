# Handoff — state of the port as of 2026-08-23 (revised, LTSA)

Written so this project can be picked up in a fresh session (or by a different person) without
losing anything that currently lives only in conversation. Everything here is either *not*
recorded elsewhere in the repo, or is a pointer to where it is.

Read order for someone starting cold: [README.md](README.md) → [PORTING_PLAN.md](PORTING_PLAN.md)
→ this file → `docs/formats/`.

---

## 1. Where the project stands

**Phase 0 is complete.** **Phase 1's read side is complete**: timebase, x.wav, spectra, and LTSA.
The LTSA fixtures have now been built and the parity suite has **no skips left** — 40 passing and 1
xfail, that xfail being the Phase 1 milestone itself (a Python LTSA *writer* whose output is
byte-identical to MATLAB's). Everything Triton reads, this reads, and provably identically.

| Phase | Status |
|---|---|
| 0 — spec + golden fixtures | **Done.** 12 fixtures, 190 reference artefacts, tests passing |
| 1 — core library (`io`, `timing`, `dsp`) | **Read side complete.** 40 passing, 0 skipped, 1 xfail (the writer milestone) |
| 2 — session layer | Not started |
| 3 — GUI | Not started |
| 4 — plugin API + first Remora | Not started |
| 5 — tools | Not started |
| 6 — migration | Not started |

What exists under `src/triton/`:

| Module | Ports | State |
|---|---|---|
| `timebase.py` | the 2000-year datenum shift, `wavname2dnum.m` | Done. All 6 filename patterns parse |
| `io/xwav.py` | `rdxwavhd.m` | Done, including v2, which MATLAB's `io/` readers cannot do |
| `io/audio.py` | `readseg.m`, `check_time.m` | Done. Byte-exact against all 27 reference cases |
| `dsp.py` | `hanning`, `mkspecgram.m`, `pwelch` + int8, TF interpolation | Done. Agreement is machine precision — see below |
| `io/ltsa.py` | `read_ltsahead.m`, `read_ltsadata.m` | Done. Data blocks byte-identical on all 3 fixtures |

Nothing is skipped. The single xfail is `test_python_mkltsa_is_byte_identical`, which needs
`triton.tools.mkltsa` — the LTSA *writer*, and the whole point of Phase 1.

Measured agreement for `dsp`, because "it passes" understates it and a future regression should
have a number to fall short of:

| quantity | agreement | test tolerance |
|---|---|---|
| `hanning` | 6e-16 absolute | 1e-15 |
| spectrogram dB | **7.3e-12 dB absolute**, 3.2e-13 relative | 1e-9 |
| spectrogram `f`, `t` | exact (0.0) | 1e-12 |
| `pwelch` dB | 9e-16 relative | 1e-9 |
| int8 quantisation | exact, all 3 cases | exact |
| TF interpolation | 2e-16 relative | 1e-12 |

Two things about `dsp` that took measuring rather than reading:

* **State the spectrogram tolerance in absolute dB, not relative.** A dB value passes through zero
  legitimately — `specgram_nfft1000_ol50` has a bin at −3.25e−4 dB — and a pure relative bound
  there measures its own noise. That one bin produced a 2.2e−10 relative error while its absolute
  error was 7.3e−14 dB, making the test look 4.5× from failing when it was eleven orders off. The
  assertion now carries both a relative bound and an absolute floor.
* **No pwelch fixture lands on a rounding tie**, so the one part of `to_int8_db` that needed custom
  code was passing untested. Closest approach across all three cases is 0.0054. MATLAB's `int8()`
  rounds half *away from zero*; numpy's `round()` rounds half to *even*, so they disagree on 0.5,
  2.5 and 126.5. There is now a probe vector in `dump_reference.m` and a test that checks the rule
  against documented behaviour immediately and upgrades itself to the MATLAB dump once one exists.

## 2. Decisions that are settled

All seven recommendations at the end of [PORTING_PLAN.md](PORTING_PLAN.md) were reviewed and
**accepted on 2026-08-19**, with these amendments from that conversation:

| # | Decision | Amendment |
|---|---|---|
| 1 | Integer samples + exact segment start times; datenum only at boundaries | Confirmed as the direction the group was already moving. If any MATLAB remains, move it to `datetime`, which does not degrade with epoch distance |
| 2 | Single `TritonSession`, one `current_session()` accessor, inspector panel + embedded console | Accepted as-is |
| 3 | PySide6 + pyqtgraph, matplotlib for export | Accepted as a starting point, explicitly revisitable if a limitation shows up |
| 4 | Splice-and-delimit by default; honouring gaps as an option | Accepted; gap display was called out as a genuinely useful new feature, not just a compatibility switch |
| 5 | First-class `Annotation` model | Accepted, with a wider mandate — see §3 |
| 6 | Entry-point plugin discovery, typed hooks, `triton.api` as the only import surface | Accepted, with a low-barrier-to-entry constraint — see §3 |
| 7 | Don't port the dead files | Accepted enthusiastically; housecleaning is wanted, not risky |

## 3. Requirements that emerged in conversation and shape later phases

These are not in the plan document and would be lost otherwise.

**Annotation is a strategic capability, not a compatibility feature.** It was central when Triton
was used for large-scale manual annotation and is underused now. The revamp should lay groundwork
for: dragging boxes rather than only xy picks; zoom; saving selected acoustic events for
classifier training or similarity search ("find me more like this"); and computing standard
metrics on a selection (amplitude variants, SNR). The `Annotation` dataclass in
[PORTING_PLAN.md §4.4](PORTING_PLAN.md) should be checked against those use cases before it is
frozen — in particular it needs a time *and* frequency extent (it has one), a stable id, and
somewhere to hang derived metrics.

**Remoras are written by students and junior research staff.** Low barrier to entry is a hard
requirement, not a nice-to-have: adding a plugin must stay easy and the instructions must be
followable by someone who is not a software engineer. Balance this against the API in
[PORTING_PLAN.md §4.3](PORTING_PLAN.md), which is richer than what most plugins need.

The resolution to aim for: **most existing Remoras are only nominally integrated** — they open
from a Triton dropdown and are standalone thereafter. That path must stay trivial (one entry
point, one callable, done). The typed hooks (`overlay`, `on_pick`, `on_segment_loaded`) should be
strictly opt-in for the minority who want real integration. Design the tutorial around the
trivial case and document the hooks as a second tier.

**Long-term goal is branch unification.** Several divergent Triton versions exist across the
group. The user wants everyone back on one main tool carrying everyone's features, with minimal
regressions from version mixing. The Python port is the vehicle. This makes §4 below more urgent
than it would otherwise be.

## 4. The source-tree problem — real, but measured, and smaller than it looks

> **Measured 2026-08-19: the two trees produce byte-identical output on every path Phase 0
> covers.** `dump_reference` was run against both `Triton-master` and `Triton_remoras` and the
> two dumps compared: **74/74 binary artefacts and every JSON identical.** Header parsing,
> `readseg` byte arithmetic, derived timestamps, `mkspecgram` and `pwelch` are unaffected by the
> 19-file divergence.
>
> Not covered by that comparison, and still genuinely open: **the LTSA generation path**
> (`calc_ltsa`, `write_ltsahead`, `ck_ltsaparams`), because the `.ltsa` fixtures do not exist yet
> (§6.1). The known `calc_ltsa` difference affects only the final short time bin of **wav/flac**
> input, so even once fixtures exist the x.wav path may well come out identical too.
>
> **Revised priority: this is not a Phase 1 blocker.** Reconcile it as the port pulls on each
> file, using the harness as the arbiter (recipe below). The earlier "resolve before Phase 1"
> framing was written before the measurement and was too strong.

### Using the parity harness as the arbiter

This is the useful by-product of Phase 0 for the consolidation effort: "which version is right?"
becomes a measurement instead of a diff-reading exercise. About three minutes per tree.

```bash
matlab -batch "addpath('D:/Code/<CANDIDATE_TREE>'); addpath('D:/Code/Triton_python/tools/matlab'); \
  dump_reference('triton','D:/Code/<CANDIDATE_TREE>','out','D:/Code/Triton_python/fixtures/_compare')"
# then diff fixtures/_compare against fixtures/reference; ignore _environment.json
```

Any file whose differences do not move a single byte of that dump is a cosmetic merge — take
either version. Files that *do* move bytes are the short list that deserves human judgement, and
each one should gain a fixture case before the merge is finalised.

### The divergence itself

**All Phase 0 specs were written against `D:\Code\Triton-master`.** There is at least one other
full checkout on the same machine, and MATLAB's saved path actually points at the *other* one:

| Path | What it is |
|---|---|
| `D:\Code\Triton-master` | What the specs and the reference dump were built from |
| `D:\Code\Triton_remoras` | A second full checkout — same 90 base `.m` filenames, **19 differ in content**. This is what the MATLAB path pointed to when `dump_reference` ran (visible as `Triton_remoras\...` warnings in its output) |
| `D:\Code\Triton\triton1.95.20230315` | A versioned release drop |

Ignoring line endings, the 19 differing base files are:

```
calc_ltsa      check_path      ck_ltsaparams   decimatewav    decimatewav_dir
filepd         get_headers     get_ltsadir     get_ltsaparams initdata
initpulldowns  plot_ltsa       plot_triton     read_ltsadata  read_ltsahead
triton         wavname2dnum    write_ltsahead  xml_read
```

That is most of the LTSA pipeline. **Neither tree is a superset of the other.** Two confirmed
examples, both spot-checked:

- `write_ltsahead.m` — Triton-master has the un-scriptable `~exist('PARAMS.ltsa.outfile','var')`
  guard; **Triton_remoras already has the fix** (`~isfield(PARAMS.ltsa, 'outfile')`), added for
  the BatchLTSA remora, and identical to the fix independently proposed here.
- `calc_ltsa.m` — Triton_remoras branches on vector orientation when zero-padding the short final
  time bin; Triton-master does not, and is wrong for wav/flac column-vector input. **This changes
  numeric output**, so archived `.ltsa` files may contain either behaviour.
- Going the other way, Triton-master has flac support (`ftype == 3`) that Triton_remoras lacks.

**Action:** not a gate on Phase 1. Two things worth doing early and cheaply:

1. **Get every uncommitted tree into version control now.** Pure risk reduction, no scope creep,
   and `Triton_remoras` has already been shown to hold the authoritative version of at least two
   files. Losing it would lose real work.
2. **Build the `.ltsa` fixtures** (§6.1), which closes the one measurement gap above and settles
   the `calc_ltsa` question with data rather than argument.

Everything else — deciding a canonical version per file — can be pulled by the port, one file at
a time, arbitrated by the recipe above. Details in
[ltsa.md §7](docs/formats/ltsa.md#7-known-issues).

## 5. Provenance rule for resolving MATLAB disagreements

Established in conversation, now also recorded in [xwav.md](docs/formats/xwav.md):

> Files carrying `% Do not modify the following line, maintained by CVS` / `$Id: ... $` were
> written by a **different developer** from the core reader/writer set. They often re-express the
> same logic more clearly but are re-implementations and can carry errors. The un-marked core
> files (`rdxwavhd`, `wrxwavhd`, `readseg`, `read_ltsahead`, …) were written by **the author of
> the XWAV format** and are the authority when in doubt.

60 files in Triton-master carry the CVS marker, including all of `io/` and `prefix.m` in the base
folder. This rule already decided one case: the v2 header bug in §6 below is in a CVS-marked file,
so `rdxwavhd` wins.

## 5a. Phase 1 findings — things that were not in the specs

Four things turned up while implementing, each of which changes something outside the port.

### `WavVersionNumber` is written two ways, and MATLAB's v2 test cannot fire on one of them

Two writers disagree by one character, and **both kinds of file are in the archive**:

| writer | code | byte written |
|---|---|---|
| `HARPproc/write_XWAVhead.m:42` | `PARAMS.xhd.WavVersionNumber = 1;` | `0x01` |
| `HARPproc/mk_SpotCheck.m:391` | `PARAMS.xhd.WavVersionNumber = '1';` | `0x31`, ASCII `'1'` |

Both reach `wrxwavhd.m:145`, which writes with `fwrite(..., 'uchar')`; that converts a char to its
code point. In `mk_SpotCheck.m` every neighbouring assignment (`InstrumentID = 'DLXX'`, `SiteName
= 'sitX'`, `DiskSerialNumber = '12345678'`) genuinely *is* a string, so the quotes read as
copy-paste consistency rather than intent.

Measured over `Triton_remoras/ExampleData` — 123 x.wav files, all of them real:

| version byte | firmware | instrument | files |
|---:|---|---|---:|
| `1` | `V2.02S` | `DL37` | 5 |
| `1` | `V2.87` | `D102` | 4 |
| `1` | `V2.98` | `D111` | 2 |
| `49` = `'1'` | `3A01240501` | `DLXX` | 112 |

The split is by writer exactly as predicted; the 112 are `mk_SpotCheck` output, and their scrubbed
instrument/site/serial values are that function's literals rather than a separate anonymising step.

**Latent, not cosmetic.** v0 and v1 have identical layout and MATLAB's only consumer is
`WavVersionNumber == 2`, false either way, so nothing misreads *today*. But a v2 file written
through the `mk_SpotCheck` path would carry `'2'` = 50, `== 2` would be false, `rdxwavhd.m:106`
would size the harp chunk with the v1 formula, and every `byte_loc` in the raw-file table would be
read from the wrong offset — after emitting only its generic "SubchunkSize and NumOfRawFiles
discrepancy" warning. This compounds the four Remora copies of `io/ioReadXWAVHeader.m`, which have
no v2 branch at all (xwav.md 6.8).

Two MATLAB-side actions, neither done:

1. Drop the quotes in `mk_SpotCheck.m:391`. **`mk_SpotCheck.m` is in HARPproc, which is not merged
   into `Triton_remoras` yet**, so this belongs to that merge. It fixes new files only; the 112
   existing ones keep their `49`.
2. Normalise the field in `rdxwavhd.m` right after the `fread`, so the `== 2` tests work whichever
   way the byte was written. This is the change that protects existing and future data, and it is
   the one worth running through the MATLAB regression harness. No header has version 48 or higher,
   so mapping 48/49/50 to 0/1/2 is unambiguous.

`io/xwav.py` already accepts both. Full detail in [xwav.md §3](docs/formats/xwav.md).

### A time cannot be mapped to a sample bit-for-bit, and it does not matter

`readseg.m:111` computes `floor((plot.dnum - dnumStart) * 86400 * fs)` from float datenums
carrying about 40 ns of representation error. Forty nanoseconds is a small fraction of a sample at
any rate Triton handles, so it changes the answer **only when the requested time lands within that
distance of a sample boundary** — and there it decides the `floor`. Of the 27 reference cases, 7
land exactly on a boundary, and MATLAB falls one sample low in all 7:

```
offset 0.625 s at 10 kHz -> exact 6250, MATLAB 6249  (product 6249.999889)
offset 0.37  s at 10 kHz -> exact 3700, MATLAB 3699  (product 3699.999812)
```

Reproducing it is not merely hard, it is **ill-defined**: `plot.dnum` is not recomputed from the
requested time, it is *accumulated* — the file's start plus however many forward and backward steps
the analyst took. Two routes to the same nominal time carry different accumulated error, so the
same time in the same file can read different samples depending on how the analyst got there.
There is no "the MATLAB answer" to match.

So the port computes exactly and may differ from any given MATLAB session by at most one sample,
only at exact sample boundaries. That is 100 µs at 10 kHz and 5 µs at 200 kHz — immaterial for
spectrograms and detection, and it is the *correct* sample for anything that cares, such as
cross-correlation time-of-arrival work.

This shapes the test design, and the shape matters more than the finding. `read_samples(index,
skip, n)` is public API precisely so the parity test can ask for *exactly* the samples a given
MATLAB session read; that comparison is then byte-exact with no tolerance at all. The one-sample
question is isolated in `test_time_addressed_read_is_within_one_sample_of_matlab`, which asserts
both halves — never more than one sample out, and never out at all unless the exact product really
is within a datenum quantum of an integer. A genuine indexing bug fails the second half
immediately. **Do not collapse these back into one fuzzy-tolerance test.**

### Where the two sample rates must not be confused

`XwavSource` deliberately keeps both. `sample_rate` is the raw-file table's, which is what
`skip` uses (`rdxwavhd.m:188`, whose own comment says the `fmt ` value "could be fake").
`byte_rate` is the `fmt ` chunk's, which is what raw-file *end times* and delimiter spacing divide
by (`rdxwavhd.m:160`, `readseg.m:129`). On the `fakefs` fixture these are genuinely different
numbers and using one for the other shifts every delimiter. Both are reproduced as written.

### A field the reference knows and the test ignores is a field with no oracle

`test_ltsa_header_matches` originally asserted six of the fourteen fields the reference dumps.
`ch` was not among them, and the LTSA reader read it from the wrong offset — v3/v4 store it at 38
and the reader looked at 39. Offset 39 is padding, padding is zero, and zero is a perfectly
plausible channel number, so the only symptom was a quietly wrong value on every v4 file. The
test passed.

It surfaced only because a hand-written sanity script printed the field next to the value the
fixture was built with (`ch = 1`) and they disagreed. Two changes followed:

* the test now asserts **every** scalar the reference dumps, plus `rfileid` and `fnames` per entry,
  plus that the `byte_loc` chain reproduces from `nave × nf` (ltsa.md 3.2) — which is checkable
  without MATLAB and would catch a wrong directory stride;
* `io/ltsa.py` has a `_check_layout()` that runs at import and asserts the header and entry
  geometry of all four versions sums to its declared size, so an offset edit that breaks the
  arithmetic fails loudly instead of returning zeros.

The general form is worth keeping in mind for the remaining phases: **a dumped reference field
that no assertion touches is not coverage.** Prefer asserting the whole record.

### The int8 tie-breaking rule is now verified, not assumed

The probe added to `dump_reference.m` has been run. MATLAB's `int8()` output on the 15 probe values
matches `to_int8_db` exactly, and numpy's default `round()` would have differed on 4 of them
(−0.5, 0.5, 2.5, 126.5). That gap is closed with evidence.

### check_time.m conditions 2b and 3 do not warn, they crash

`check_time.m` leaves `PARAMS.raw.currentIndex` empty when the time is past the last raw file, and
of length 2 when it falls inside two at once. `readseg.m:108` then indexes `byte_loc` with it and
hands `fseek` a non-scalar offset, which errors out with nothing that names the cause. The comments
present both as warnings; they are not. The port raises `TimeNotInData` / `AmbiguousTime` from the
place that knows why, and the message says which raw files and what to do.

## 6. Open items, carried forward

### Blocking nothing, but must be done to finish Phase 0

1. ~~**LTSA fixtures are not built.**~~ **Done, 2026-08-23.** The first of the three options
   worked: pointed `make_ltsa_fixture` at `Triton_remoras`, whose `write_ltsahead` already tests
   `~isfield(PARAMS.ltsa,'outfile')` rather than the broken `exist('PARAMS.ltsa.outfile','var')`,
   and it ran headless with no dialogs. Three v4 LTSAs built, `dump_reference('sections',
   {'ltsa','dsp'})` captured, all four previously-skipped tests now pass.

   Two things had to be fixed in `dump_reference.m` first, both consequences of the MATLAB side
   having improved since Phase 0 was written:
   - Its `ltsa` section hand-builds `PARAMS` and called `read_ltsadata` directly. That worked when
     `check_ltsa_time` was commented out at `read_ltsadata.m:11` — which is what issue #129 turned
     out to be — and threw `Unrecognized field name "step"` once it was restored. It now sets
     `tseg.step`, `plotStartRawIndex` and `plotStartBin`, which is what `init_ltsadata` would have
     contributed. **Note the ordering trap:** `read_ltsadata` does derive `plotStartRawIndex`
     itself, but only *after* `check_ltsa_time` has already read it, so omitting it is an immediate
     error rather than a latent one.
   - No real caller was affected — `initparams.m:131`, `mk_ltsa.m:50` and `sm_mk_ltsa.m:46` all set
     `tseg.step` — so this was fixture tooling, not a regression. Checked before assuming.

   Then `dump_reference('sections',{'ltsa'})`. Until this happens,
   `test_ltsa_reference_present_or_explained` and the LTSA parity tests skip.
2. **Real deployment data.** `fixtures/real/` **is no longer empty** — it now holds
   `GOM_AC_05_disk01_212287.x.wav` (30 MB), its 15 MB raw counterpart, and
   `GOM_AC_01_example.ltsa`. Correctly gitignored; only `.gitkeep` and `README.md` are tracked.

   It earned its keep immediately: the `WavVersionNumber` finding in §5a came out of the first
   real file, which the reader rejected outright. Synthetic fixtures prove we implemented the spec;
   real files prove we implemented reality, and here the spec was wrong.

   The x.wav now reads end to end — 200 kHz, 1 channel, 16-bit, one raw file, 75 s, and
   `byte_loc + byte_length` equals the file size exactly. Two follow-ups:

   - `fixtures/real/MANIFEST.md` still says "(none yet)". Worth filling in, and note that
     `.gitignore` whitelists only `README.md` under `fixtures/real/`, so the manifest is currently
     untracked. Add `!fixtures/real/MANIFEST.md` if it should be committed — the point of a
     manifest is that it outlives the data.
   - The remaining wish list in [fixtures/README.md](fixtures/README.md) still stands, and one
     entry is now sharper: a **v2-header** file would settle both the `== 2` question in §5a and
     the severity of the `io/` reader bug.

### Parity coverage gaps in what is already written

The Phase 1 io tests pass, but three behaviours have **no MATLAB reference to check against**.
They are implemented from the source and hand-verified; that is weaker than the rest of the suite,
and worth closing when MATLAB is next in front of someone.

1. **The `check_time.m` triage is untested.** All 27 `readseg_index` reference cases have
   `currentIndex == 1`, so conditions 2a (a time in a duty-cycle gap, and the direction-dependent
   snap), 2b (past the last raw file) and 3 (inside two raw files at once) are exercised by nothing.
   Closing it needs three new fixture cases: an offset landing inside a gap on
   `xwav_v1_duty_1ch_16b_10k.x.wav`, an offset past the end, and a file written with overlapping
   raw files. The gap case is the one that matters — it is how analysts page through duty-cycled
   data. Hand-verified meanwhile: the backward snap lands at `end - (tseg - 1/fs)`, which for raw
   file 0 of the duty fixture is `08:45:02.4999 - 0.4999 = 08:45:02.000000000` exactly.
2. **Gain on a multichannel file is untested.** No fixture has `nch > 1` *and* `gain > 0`, so the
   quirk in `readseg.m:117-119` — the division is applied to column `PARAMS.ch` alone, leaving the
   other channels unscaled — is reproduced but unverified. If any real 4-channel deployment ran
   with gain, its other three channels have always come back unscaled.
3. **`splice_gaps=False` has no oracle by construction.** No MATLAB path lays a window out on true
   wall-clock time, so the NaN-filled mode can only be checked for self-consistency. It is: on the
   duty fixture a 0.5 s window opened at +2.25 s gives 2500 real samples then 2500 NaN, the switch
   falling exactly where raw file 0's 2.5 s of data ends, and the real samples are identical to the
   spliced mode's.

The 24-bit decode path is also unverified, but that is expected and harmless — no 24-bit x.wav
exists (§6, Resolved).

### Open questions

| Question | Status |
|---|---|
| 32-bit `AudioFormat`: `wrxwavhd` writes 3 (float), `readseg` reads int32 | **Open.** Needs one real 32-bit x.wav. Fixture currently writes `AudioFormat = 1` with int32 samples |
| `blksz` hard-codes the 16-bit assumption (`ck_ltsaparams.m:50`), so `write_length` bookkeeping is questionable for 32-bit | **Open**, coupled to the above |
| Do x.wav files with channel counts other than 1 or 4 exist? Triton's sector geometry only defines those two | **Open** |
| Do v2-header x.wav files exist in the archive, and in what quantity? | **Open**, and it determines the severity of the `io/` bug below |
| Which tree is canonical (§4) | **Open, highest priority** |

### Resolved since the specs were written

- **24-bit x.wav.** `readseg.m` asks `fread` for a nonexistent `'int24'`, so it cannot read them.
  Confirmed with the team: **no 24-bit x.wav exist; the archive is all 16-bit.** Implement it in
  Python anyway (nearly free) but it needs no fixture and blocks nothing. 24-bit plain *wav* does
  occur and has a fixture.

### Deferred deliberately

- **`rdxwavhd` per-raw-file `dt`/`padding` overwrite.** Lines 135/137 assign unsubscripted inside
  the loop, so only the last raw file's values survive. **This may be intentional, and some
  Remoras are believed to compensate for it.** Plan: the Python reader exposes the full
  per-raw-file `dt` array (strictly more information, so nothing breaks by reading it), but no
  change to what consumers see until the compensating Remora code is found and audited. Do that
  audit during Phase 4, not before.
- **`io/ioReadXWAVHeader.m` has no v2 branch** — it skips `drate`/`dt` and returns wrong
  `byte_loc`/`byte_length` silently. Copies live in four Remoras (`Detector/io/`,
  `SPICE-Detector/io/`, `BlueWhaleBcall-Detector/Detection/`, root `io/`). Agreed to **fix when
  Remora work begins**, not now. If v2 files turn out to be common, results those Remoras produced
  from them are suspect and this becomes urgent. Per §5, `rdxwavhd` is the correct reference.
- **`write_ltsahead` scriptability** — agreed to make LTSA creation scriptable in the future. The
  fix already exists in `Triton_remoras`; folding it in is part of §4.

## 7. Environment facts worth not rediscovering

- **MATLAB R2025b** at `C:\Program Files\MATLAB\R2025b\bin\matlab.exe`. R2013b, R2016b, R2023a and
  R2024a are also installed. The reference dump records which release produced it in
  `fixtures/reference/_environment.json` — numerics can shift between releases.
- Headless dump command that works:
  ```bash
  matlab -batch "addpath('D:/Code/Triton-master'); addpath('D:/Code/Triton_python/tools/matlab'); dump_reference"
  ```
  Takes a couple of minutes, mostly MATLAB startup. `dump_reference('sections',{...})` runs a
  subset.
- **`tr_headless_handles.m` is the trick that makes the oracle trustworthy.** It builds the ~15
  `HANDLES` that core functions touch, so `readseg`, `check_time`, `mkspecgram` and
  `read_ltsadata` run *for real* headlessly instead of being re-implemented in the dump script.
  If a new dump section needs another Triton function, extend that shim rather than transcribing
  logic.
- **Use the project venv, not system Python.** System Python 3.12 has a broken `anyio`/pytest
  plugin (missing `typing_extensions`) that makes any `pytest` invocation fail with an unrelated
  traceback. `.venv/` (numpy 2.5.1, scipy 1.18.0, pytest 9.1.1) is clean.
- Fixture generation uses `numpy.random.RandomState`, **not** `default_rng` — NEP 19 guarantees
  the former's stream is stable across numpy versions forever. Verified byte-identical output
  under numpy 1.26.2 and 2.5.1. Do not "modernise" this.
- Committed data sizes: `fixtures/generated/` 2.3 MB, `fixtures/reference/` 3.7 MB. The reference
  dump stores `readseg` blocks as int32 where lossless and caps windows at 25k samples to keep
  that number down; if you add dump sections, keep an eye on it.
- The repo is **not a git repository yet**. `git init` and an initial commit is a reasonable first
  action in the new environment.

## 8. Concrete next actions, in order

Done since the last revision: `git init`; `triton.timebase`; `triton.io.xwav`;
`triton.io.audio` (`readseg` + `check_time`). Remaining, in order:

1. **`triton.tools.mkltsa`** — the LTSA writer, and the Phase 1 milestone. Everything it needs
   now exists and is verified: `dsp.welch_db` and `dsp.to_int8_db` for the values,
   `io/ltsa.py`'s `_layout` for the byte geometry, `io/audio.py` for reading the source. The
   parameter derivation is in ltsa.md 5.2 (`nfft = floor(fs/dfreq)`, `cfact = tave*fs/nfft`,
   `nave = ceil(Nsamp/(nfft*cfact))`) and the `tave` clamp is real — the 200 kHz fixture hit it
   and came out at 0.25 s. `test_python_mkltsa_is_byte_identical` is the target; drop its xfail
   when it passes.
2. **Resolve §4** — pick the canonical MATLAB tree, or merge the two. Re-run `dump_reference`
   afterwards if anything in the pipeline changed.
3. **Close the parity coverage gaps in §6** while MATLAB is open anyway; the duty-cycle gap case
   is the one worth the effort.
4. Then the session layer (Phase 2).
5. Phase 1 milestone: `test_python_mkltsa_is_byte_identical` goes from xfail to pass.

Carried over to the MATLAB side, not this repo: the two `WavVersionNumber` fixes in §5a.

## 9. Files a fresh session should read first

| File | Why |
|---|---|
| [PORTING_PLAN.md](PORTING_PLAN.md) | Architecture, phases, the PARAMS/session design, the plugin API sketch |
| [docs/formats/xwav.md](docs/formats/xwav.md) | Byte tables + 8 known MATLAB inconsistencies |
| [docs/formats/ltsa.md](docs/formats/ltsa.md) | Byte tables + the two-tree divergence + 5 known issues |
| [docs/formats/timebase.md](docs/formats/timebase.md) | The year-2000 offset and why it must not be "cleaned up" |
| [tests/test_parity_phase1.py](tests/test_parity_phase1.py) | Phase 1's definition of done, already written |
| [tools/matlab/dump_reference.m](tools/matlab/dump_reference.m) | How the oracle is produced |
