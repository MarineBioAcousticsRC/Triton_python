# Triton — Python port

Python reimplementation of [Triton](https://github.com/MarineBioAcousticsRC/Triton), the MATLAB
passive-acoustic analysis package developed at the Scripps Whale Acoustics Lab.

**Status: Phase 0 (specification and golden fixtures) — complete. No implementation code yet.**

See [PORTING_PLAN.md](PORTING_PLAN.md) for the architecture, the phase plan, and the reasoning
behind the design decisions, and **[HANDOFF.md](HANDOFF.md) for current state, open questions, and
next actions** — start there if you are picking this up fresh.

---

## What Phase 0 produced

| | |
|---|---|
| [`docs/formats/xwav.md`](docs/formats/xwav.md) | Byte-exact x.wav spec (v0/v1/v2), the semantics that matter, and eight known inconsistencies in the MATLAB implementation |
| [`docs/formats/ltsa.md`](docs/formats/ltsa.md) | Byte-exact LTSA spec (v1–v4), how the values are computed, four known issues |
| [`docs/formats/timebase.md`](docs/formats/timebase.md) | The year-2000 datenum offset, why it is load-bearing, and the Python representation that replaces it |
| [`tools/make_fixtures.py`](tools/make_fixtures.py) | Generates 12 synthetic fixtures covering the format matrix; deterministic and byte-reproducible |
| [`tools/matlab/dump_reference.m`](tools/matlab/dump_reference.m) | Runs unmodified Triton over the fixtures and dumps ground truth |
| [`tests/test_fixture_integrity.py`](tests/test_fixture_integrity.py) | The fixtures satisfy every rule Triton's reader enforces |
| [`tests/test_reference_dump.py`](tests/test_reference_dump.py) | The MATLAB dump is complete and self-consistent |
| [`tests/test_parity_phase1.py`](tests/test_parity_phase1.py) | **Phase 1's definition of done** — written now, skipped until the code exists |

The reference dump is committed (`fixtures/reference/`, ~3.7 MB) so the parity suite runs without
a MATLAB licence.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -e ".[dev]"
python tools/make_fixtures.py
pytest
```

Expect the Phase 1 parity tests to skip — they are the target, not a regression.

## Regenerating the MATLAB ground truth

Needs MATLAB and a checkout of Triton-master. No Triton GUI is started;
`tools/matlab/tr_headless_handles.m` supplies the few `HANDLES` that core functions touch, so
`readseg`, `check_time`, `mkspecgram` and `read_ltsadata` run for real rather than being
re-implemented in the dump script.

```matlab
addpath('D:\Code\Triton-master')
addpath('D:\Code\Triton_python\tools\matlab')
make_ltsa_fixture      % builds the .ltsa fixtures (one Save dialog each -- see its help)
dump_reference         % writes fixtures/reference/
```

Or from a shell:

```bash
matlab -batch "addpath('D:/Code/Triton-master'); addpath('D:/Code/Triton_python/tools/matlab'); dump_reference"
```

`_environment.json` records the MATLAB release and toolbox list. Numeric output can shift between
releases; when it does, we want the diff to be visible.

---

## Things Phase 0 turned up in the MATLAB code

Found by running the existing code against controlled inputs. All are documented in the format
specs with file:line references; the notable ones:

1. **`io/ioReadXWAVHeader.m` has no v2 branch.** It skips the per-channel `drate` and `dt` fields
   and returns wrong `byte_loc`/`byte_length` on a v2 file, silently. Copies of this reader live
   in four Remoras, so they inherit it. If v2 files exist in the archive, results those Remoras
   produced from them are suspect. *Deferred to the Remora phase by agreement.*
   ([xwav.md §6.8](docs/formats/xwav.md#6-known-inconsistencies-in-the-matlab-implementation))
2. **Two local Triton checkouts disagree in 19 base files**, including most of the LTSA pipeline,
   and neither is a superset of the other. Measured impact so far: **none** — running the
   reference dump against both trees gives byte-identical output on every path Phase 0 covers.
   The LTSA generation path is not yet covered. [HANDOFF.md §4](HANDOFF.md).
3. **`rdxwavhd.m` keeps only the last raw file's `dt`.** Lines 135/137 assign `PARAMS.xhd.dt` and
   `.padding` unsubscripted inside the loop. *May be intentional; some Remoras are believed to
   compensate, so not to be changed until those are audited.*
4. **`write_ltsahead.m` can never be called non-interactively** in `Triton-master`. Its guard is
   `~exist('PARAMS.ltsa.outfile','var')`; `exist` cannot test a dotted field name, so it is always
   true and `uiputfile` always opens. `Triton_remoras` already carries the fix.
   ([ltsa.md §7.1](docs/formats/ltsa.md#7-known-issues))
5. **24-bit x.wav cannot be read at all** — `readseg.m:99` asks `fread` for a nonexistent
   `'int24'` type. Latent only: no 24-bit x.wav exist.
6. **The datenum precision argument, measured.** At the real epoch a float datenum resolves
   10.06 µs; one sample at 200 kHz is 5 µs. With Triton's 2000-year shift it resolves 0.039 µs.
   Numbers come from MATLAB itself (`fixtures/reference/datenum_precision.json`), not from our
   arithmetic.

## What we need from the team

Real deployment files, to sit in `fixtures/real/` (gitignored). Synthetic fixtures prove we
implemented the spec; real files prove we implemented reality. The list, and the three open
questions they would settle, is in [`fixtures/README.md`](fixtures/README.md).

## Layout

```
docs/formats/     byte-exact format specifications
tools/            make_fixtures.py + matlab/ reference-dump scripts
fixtures/
  generated/      synthetic corpus (committed)
  real/           real deployment data (gitignored)
  reference/      MATLAB ground truth (committed)
tests/            fixture integrity, reference sanity, Phase 1 parity
src/triton/       implementation -- Phase 1
```
