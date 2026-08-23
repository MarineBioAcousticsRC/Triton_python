# Real-data fixtures

Not committed — `.gitignore` keeps everything here out of the repo except this file, the README
and `.gitkeep`. One row per file: where it came from, why it is interesting, and anything unusual
about it. See [../README.md](../README.md) for what we are still looking for.

The point of these: synthetic fixtures prove we implemented the spec, real files prove we
implemented reality. That earned itself back on the first file — see "What these have already
settled" below.

| File | Size | What it is | Why it is here |
|---|---:|---|---|
| `GOM_AC_05_disk01_212287.x.wav` | 30,000,140 B | Converted x.wav. 200 kHz, 1 ch, 16-bit, **1 raw file**, 75 s, starting 2024-08-21T23:06:15.000 | First real x.wav in the suite. Immediately falsified the spec's `WavVersionNumber` claim |
| `GOM_AC_05_disk01_212287` | 15,380,480 B | The **raw HARP disk file** the x.wav was converted from — the pre-`hrp2xwav` format | A matched raw/converted pair. The only fixture that can test the conversion path at all |
| `GOM_AC_01_example.ltsa` | 18,083 B | LTSA, **format version 4** | The only real LTSA here, and v4 is the version with `nrftot` widened to uint32 |

## Provenance and scrubbing

The x.wav's identity fields are placeholders, not the deployment's: `InstrumentID = DLXX`,
`SiteName = sitX`, `ExperimentName = XXXXXXXX`, `DiskSerialNumber = 12345678`, and latitude,
longitude and depth all zero. These are **`mk_SpotCheck.m`'s own literals** (lines 393-400), not a
separate anonymising step — the file is SpotCheck output. Worth knowing before treating any of
those fields as data, and worth knowing that no scrubbing tool needs to be found or trusted.

The firmware and timestamps are real: firmware `3A01240501`, start 2024-08-21T23:06:15.

## The raw/x.wav pair

The extensionless file is genuinely the raw HARP format, not a stray copy. Its first eight bytes
are `24, 8, 21, 23, 6, 15, 0, 0` — year, month, day, hour, minute, second, and a uint16 of ticks —
which is exactly the x.wav's start time. That is a per-sector time header, so the sector geometry
should hold:

```
15,380,480 / 512            = 30,040 sectors exactly
30,040 x 250 samples/sector =  7,510,000 samples   (nsampPerSect for the 3A01 family)
x.wav data 30,000,000 / 2   = 15,000,000 samples
```

Ratio 1.9973, consistent with `compressionFactor = 2`. The 20,000-sample shortfall against an exact
doubling is 80 sectors' worth, plausibly `tailblk` zero-padding or a dropped incomplete raw file —
**worth confirming rather than assuming** before this pair is used as a conversion oracle.

## A firmware-table problem this file exposes

`3A01240501` decodes as the `3A01` family, 2024-05-01. It is **present in only two of the five
`table_ckFirmware.csv` copies in the source tree**:

| tree | rows | has `3A01240501` | newest `3A01` row |
|---|---:|:---:|---|
| `Triton1.95.20230315-DataProcessing` (both HARPproc copies) | 64 | **yes** | `3A01251028` |
| `Triton-HARPLab` | 59 | no | `3A01220318` |
| `Triton-SpotCheck` | 58 | no | `3A01220318` |
| `Triton/triton1.95.20230315` | 58 | no | `3A01220318` |

`ckFirmware.m` fails cleanly on an unknown firmware — `success = 0` and a message naming the file
to update, not silently wrong parameters — so the consequence is simply that **three of the four
checkouts cannot process this file at all**, or any single-channel compressed HARP data from 2024
onward. This is not hypothetical; the file is sitting in this directory.

Two things follow, both for the MATLAB side rather than the port:

1. It is a concrete argument for consolidating on one HARPproc with a maintained table, rather than
   a copy per working style. The divergence here is not features, it is *whether the lab's current
   data can be read*.
2. `ckFirmware.m:29` matches with `any(findstr(vnum, fwCell{i,1}))` — a **substring** test — and
   does not `break`, so the *last* matching row wins. A firmware string that is a prefix of several
   table entries (`V2.9` against `V2.98`, `V2.97`, `V2.96`, `V2.95`) silently takes whichever is
   last in the CSV, with no warning. Worth an exact match plus a warn-on-multiple when HARPproc is
   merged. (`findstr` is also long deprecated in favour of `strfind`.)

## What these have already settled

- **`WavVersionNumber` is stored as an ASCII digit by some writers.** The reader rejected this file
  outright with "unknown WavVersionNumber 49". 49 is ASCII `'1'`. Tracked in
  [xwav.md §3](../../docs/formats/xwav.md) and HANDOFF §5a; the consequence is that MATLAB's
  `WavVersionNumber == 2` test cannot fire on such a file.
- **The header parse is self-consistent on real data.** `byte_loc + byte_length` = 30,000,140 =
  the file size exactly, and the harp chunk's declared size matches its raw-file count.

## Still wanted

The ranked list is in [../README.md](../README.md). Two entries are sharper now:

- A **v2-header** x.wav would settle both the `== 2` question above and the severity of the
  `io/ioReadXWAVHeader.m` v2 blindness (xwav.md 6.8). Still no v2 file has been seen.
- A **multichannel file recorded with gain > 0** would settle whether `readseg.m`'s
  divide-only-the-displayed-column behaviour has ever affected real results. Every 4-channel
  fixture here has gain 0, so the quirk is reproduced but unverified.
