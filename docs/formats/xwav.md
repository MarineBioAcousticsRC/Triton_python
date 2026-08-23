# XWAV (`*.x.wav`) format specification

Derived from the MATLAB implementation in Triton-master:
[`rdxwavhd.m`](../../../Triton-master/rdxwavhd.m) (read),
[`wrxwavhd.m`](../../../Triton-master/wrxwavhd.m) (write),
[`get_headers.m`](../../../Triton-master/get_headers.m) (fast partial read),
[`readseg.m`](../../../Triton-master/readseg.m) (data access),
cross-checked against the independent implementation in `Triton-master/io/ioReadXWAVHeader.m`.

> **Which MATLAB file wins when two disagree.** Files carrying a
> `% Do not modify the following line, maintained by CVS` / `$Id: ... $` header were written by a
> different developer from the core reader/writer set. They often re-express the same logic more
> clearly, but they are re-implementations and can carry errors. The un-marked core files
> (`rdxwavhd`, `wrxwavhd`, `readseg`, `read_ltsahead`, …) were written by the author of the XWAV
> format itself and are the authority when in doubt. 60 files in Triton-master carry the CVS
> marker, including all of `io/`. §6.8 below is a case where that rule decides the answer.

XWAV is a standard RIFF/WAVE file with one extra chunk, `harp`, inserted between `fmt ` and
`data`. Standard WAVE readers that skip unknown chunks can read the audio; they will get the
wrong sample rate and no timing (see [§5](#5-critical-semantics)).

**Byte order is little-endian throughout. All offsets below are absolute from the start of file.**

---

## 1. Layout overview

```
+--------------------------------------------------+ 0
| RIFF chunk header                    12 bytes     |
+--------------------------------------------------+ 12
| 'fmt ' subchunk                      24 bytes     |
+--------------------------------------------------+ 36
| 'harp' subchunk header + fields                   |
|   v0/v1: 64 bytes  (36..99)                       |
|   v2:    64 + 4*nch bytes                         |
+--------------------------------------------------+ 100  (v1)
| raw-file table: NumOfRawFiles entries             |
|   v0/v1: 32 bytes each                            |
|   v2:    32 + 4*nch bytes each                    |
+--------------------------------------------------+
| 'data' subchunk header                8 bytes     |
+--------------------------------------------------+ = byte_loc[0]
| interleaved PCM samples                           |
+--------------------------------------------------+
```

`header_size = 108 + 32 * NumOfRawFiles` for v0/v1, and `byte_loc[0] == header_size`
(`wrxwavhd.m:27,42`).

## 2. RIFF and `fmt ` chunks

| Offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 4 | char | `ChunkID` | `"RIFF"` |
| 4 | 4 | uint32 | `ChunkSize` | filesize − 8. Triton **warns but continues** on mismatch (`rdxwavhd.m:34`) |
| 8 | 4 | char | `Format` | `"WAVE"` |
| 12 | 4 | char | `fSubchunkID` | `"fmt "` |
| 16 | 4 | uint32 | `fSubchunkSize` | must be 16, else Triton aborts (`rdxwavhd.m:57`) |
| 20 | 2 | uint16 | `AudioFormat` | 1 = PCM. `wrxwavhd.m:104-108` writes **3** for 32-bit — see [§6](#6-known-inconsistencies) |
| 22 | 2 | uint16 | `NumChannels` | `get_headers.m:78` seeks directly to 22 |
| 24 | 4 | uint32 | `SampleRate` | **nominal / possibly fake** — see [§5.1](#51-sample-rate) |
| 28 | 4 | uint32 | `ByteRate` | `SampleRate * BlockAlign` |
| 32 | 2 | uint16 | `BlockAlign` | `NumChannels * BitsPerSample / 8` |
| 34 | 2 | uint16 | `BitsPerSample` | 16, 24 or 32. `get_headers.m:81` seeks directly to 34 |

## 3. `harp` subchunk

| Offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 36 | 4 | char | `hSubchunkID` | `"harp"`. If this reads `"data"`, the file is a plain WAV and Triton falls back (`rdxwavhd.m:80`) |
| 40 | 4 | uint32 | `hSubchunkSize` | see formula below |
| 44 | 1 | uint8 | `WavVersionNumber` | 0, 1 or 2 -- **or the ASCII digit for it**; see below |
| 45 | 10 | char | `FirmwareVersionNumber` | |
| 55 | 4 | char | `InstrumentID` | |
| 59 | 4 | char | `SiteName` | |
| 63 | 8 | char | `ExperimentName` | |
| 71 | 1 | uint8 | `DiskSequenceNumber` | |
| 72 | 8 | char | `DiskSerialNumber` | |
| 80 | 2 | uint16 | `NumOfRawFiles` | `get_headers.m:94` seeks directly to 80 |
| 82 | 4 | int32 | `Longitude` | degrees × 100000, ±180 |
| 86 | 4 | int32 | `Latitude` | degrees × 100000, ±90 |
| 90 | 2 | int16 | `Depth` | positive = down |
| 92 | 4·nch | float32[nch] | `drate` | **v2 only** |
| 92 or 92+4·nch | 8 | uint8[8] | `Reserved` | padding to 64-byte chunk |

### `WavVersionNumber` is written two different ways

Two writers disagree, and **both kinds of file are in the archive**:

| writer | code | byte written |
|---|---|---|
| `HARPproc/write_XWAVhead.m:42` | `PARAMS.xhd.WavVersionNumber = 1;` | `0x01` |
| `HARPproc/mk_SpotCheck.m:391` | `PARAMS.xhd.WavVersionNumber = '1';` | `0x31` (ASCII `'1'`) |

Both reach `wrxwavhd.m:145`, which writes with `fwrite(..., 'uchar')`; that converts a char to
its code point, so the quotes on `'1'` are the whole difference. In `mk_SpotCheck.m` every
neighbouring assignment (`InstrumentID = 'DLXX'`, `SiteName = 'sitX'`, `DiskSerialNumber =
'12345678'`) genuinely *is* a string, so this reads as copy-paste consistency rather than intent.

Measured over `ExampleData` (123 x.wav files), the split is by writer, exactly as predicted:

| version byte | firmware | instrument | files |
|---:|---|---|---:|
| `1` | `V2.02S` | `DL37` | 5 |
| `1` | `V2.87` | `D102` | 4 |
| `1` | `V2.98` | `D111` | 2 |
| `49` = `'1'` | `3A01240501` | `DLXX` | 112 |

The 112 are `mk_SpotCheck` output; the scrubbed instrument/site/serial values are that function's
literals, not a separate anonymising step.

**A reader must accept both.** `io/xwav.py` normalises 48/49/50 to 0/1/2 -- no real header has
version 48 or higher, so the mapping is unambiguous.

**This is latent, not cosmetic.** v0 and v1 have identical layout, and MATLAB's only consumer is
`WavVersionNumber == 2`, false either way, so nothing misreads today. But a v2 file written
through the `mk_SpotCheck` path would carry `'2'` = 50, `== 2` would be false, `rdxwavhd.m:106`
would size the chunk with the v1 formula, and every `byte_loc` in the raw-file table would be read
from the wrong offset -- after emitting only its generic "SubchunkSize and NumOfRawFiles
discrepancy" warning. That compounds item 8 below: the four Remora copies of the `io/` reader have
no v2 branch *at all*.

Two things follow for the MATLAB side, both tracked in [HANDOFF.md](../../HANDOFF.md):

* Drop the quotes in `mk_SpotCheck.m:391` when HARPproc is merged into master. This fixes new
  files only; the 112 existing ones keep their `49`.
* Normalise the field in `rdxwavhd.m` right after the `fread`, so the `== 2` tests work whichever
  way the byte was written. This is the change that actually protects existing and future data.

`hSubchunkSize` (validated, warning only, `rdxwavhd.m:106-116`):

```
v0/v1:  64 - 8 + NumOfRawFiles * 32
v2:     (64 + 4*nch) - 8 + NumOfRawFiles * (32 + 4*nch)
```

## 4. Raw-file table

One entry per buffer flush by the recorder. `NumOfRawFiles` entries starting at offset 100
(v0/v1). Each entry:

| Rel. offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 1 | uint8 | `year` | **year − 2000** (see [§5.2](#52-timebase)) |
| 1 | 1 | uint8 | `month` | 1–12 |
| 2 | 1 | uint8 | `day` | |
| 3 | 1 | uint8 | `hour` | |
| 4 | 1 | uint8 | `minute` | |
| 5 | 1 | uint8 | `secs` | |
| 6 | 2 | uint16 | `ticks` | **milliseconds**, despite the name |
| 8 | 4 | uint32 | `byte_loc` | absolute byte offset of this raw file's samples |
| 12 | 4 | uint32 | `byte_length` | bytes of sample data (all channels) |
| 16 | 4 | uint32 | `write_length` | number of 512-byte disk sectors |
| 20 | 4 | uint32 | `sample_rate` | **the true rate** — see [§5.1](#51-sample-rate) |
| 24 | 1 | uint8 | `gain` | 1 = no change; data is divided by this |
| 25 | 7 | uint8[7] | `padding` | |
| 32 | 4·nch | float32[nch] | `dt` | **v2 only**: time offset of each channel relative to ch 1 |

### 4.1 `write_length` and sector geometry

`write_length` counts 512-byte disk sectors, each carrying a 12-byte timing header. From
`ck_ltsaparams.m:48-57`, samples per sector (`blksz`, summed over channels, **assumes 16-bit**):

| Channels | `blksz` |
|---:|---:|
| 1 | (512 − 12) / 2 = 250 |
| 4 | (512 − 12 − 4) / 2 = 248 |

so `samples_per_channel_in_raw_file = write_length * blksz / nch`, and correspondingly
`write_length = byte_length / (bytes_per_sample * blksz)`. `wrxwavhd.m:28` writes the equivalent
`byte_length / (512 - 12)`, valid for 16-bit only.

Only 1 and 4 channels are handled; anything else prints an error in `ck_ltsaparams`.

## 5. Critical semantics

### 5.1 Sample rate

**`PARAMS.fs = sample_rate[0]` from the raw-file table, not the `fmt ` chunk**
(`rdxwavhd.m:188`, whose comment notes the `fmt ` value "could be fake"). A generic WAVE reader
gets the nominal rate and is wrong.

**Exception, and it is a real one:** LTSA creation does *not* follow this rule.
`ck_ltsaparams.m:20-21` seeks to offset 24 and takes the **`fmt ` chunk** rate as `PARAMS.ltsa.fs`,
then verifies every raw file's `sample_rate` matches it and aborts if not. So the two rates must
agree for LTSA creation to succeed, but the *display* path always trusts the raw-file table. A
Python `mk_ltsa` must reproduce this or it will accept files MATLAB rejects (and vice versa).

### 5.2 Timebase

The `year` byte holds `year − 2000`. `rdxwavhd.m:151` builds
`datenum([year month day hour minute secs + ticks/1000])` **without** adding 2000, so a raw file
recorded 2011-01-30 becomes a datenum in year 11 AD. Only the display layer
(`timestr.m:25`, `yoffset = 2000`) adds the offset back.

This is load-bearing for precision, not a cosmetic bug. See
[timebase.md](timebase.md) for the full argument and the Python representation.

Per raw file, Triton derives:

```
dnumStart[i] = datenum([year month day hour minute secs + ticks/1000])
dnumEnd[i]   = dnumStart[i] + (byte_length[i] - 2) / ByteRate   seconds
```

Note the `- 2` in `rdxwavhd.m:160` (one 16-bit sample), and that it divides by the **`fmt ` chunk
`ByteRate`**, not by anything derived from the raw file's own `sample_rate`.

### 5.3 Reading a segment

`readseg.m:108-118`, for `ftype == 2` (xwav):

```
index = currentIndex                          # raw file containing the requested time
skip  = floor((plot.dnum - raw.dnumStart[index]) * 86400 * fs)   # samples
seek  = byte_loc[index] + skip * nch * bytes_per_sample
data  = fread(fid, [nch, tseg.samp], dtype).'   # tseg.samp = ceil(tseg.sec * fs)
data[:, ch] /= gain[0]                          # if gain > 0
```

**The read is byte-contiguous and runs off the end of the raw file into the next one.** When the
recording is duty-cycled, samples past the boundary come from a later wall-clock time; Triton does
not insert a gap. It computes `PARAMS.raw.delimit_time` (seconds from the left edge of the plot to
each raw-file boundary) and draws red dashed lines there (`plot_specgram.m:116-122`).

This splice-and-delimit behaviour is the contract analysts are used to and is the **default** in
the Python port; honouring gaps is an option, not a replacement.

`check_time.m` resolves which raw file a requested time falls in and handles three pathologies
that occur in real data:

1. time falls inside exactly one raw file — normal;
2. time falls in a gap between raw files — snap forward or backward depending on travel direction
   (`check_time.m:83-93`);
3. time falls inside **two** raw files at once — happens when `dnumEnd[i] > dnumStart[i+1]`, i.e.
   the sample rate or header times are wrong. Triton warns and lists the candidates.

Port this triage verbatim; it encodes years of field-data experience.

## 6. Known inconsistencies in the MATLAB implementation

Flagged for a decision rather than blind reproduction. The ones that need an actual decision --
because fixing them would change numbers already published -- are tracked with their evidence and
consequences in **[OPEN_DECISIONS.md](../OPEN_DECISIONS.md)**. This list stays as the
format-level reference.

1. **32-bit `AudioFormat`.** `wrxwavhd.m:107` writes `AudioFormat = 3` (IEEE float) for 32-bit
   files, but `readseg.m:100` reads them as `int32`. One of the two is wrong. Needs a real 32-bit
   HARP file to settle. *Open question for the team.*
2. **24-bit is broken, but latent.** `readseg.m:98-99` sets `dtype = 'int24'`, which MATLAB
   `fread` does not support, so 24-bit x.wav cannot be read. Confirmed with the team: **no 24-bit
   x.wav files exist — the archive is all 16-bit.** Implement it correctly in Python because it
   is nearly free, but it needs no fixture and blocks nothing. (24-bit plain *wav* is a different
   matter and does occur; there is a fixture for it.)
3. **`blksz` assumes 16-bit** (`ck_ltsaparams.m:50`), so `write_length` bookkeeping is wrong for
   32-bit data.
4. **`dnumEnd` uses the `fmt ` `ByteRate`** while `skip` uses the raw-file `sample_rate`
   (§5.2/§5.3). Consistent only when the two rates agree.
5. **`wrxwavhd` never writes v2.** No writer exists for the v2 per-channel `drate`/`dt` fields.
6. `PARAMS.xgain` is set from the whole `gain` vector (`rdxwavhd.m:194`) but only element 1 is ever
   applied (`readseg.m:117`).
7. **`PARAMS.xhd.dt` and `PARAMS.xhd.padding` hold only the *last* raw file's values.**
   `rdxwavhd.m:135,137` assign them unsubscripted inside the per-raw-file loop, so each iteration
   overwrites the previous one. For v2 multichannel files the per-channel time offsets of every
   raw file but the last are discarded. Confirmed against the v2 fixture.

   **Do not "fix" this silently.** It may be deliberate, and some Remoras are believed to
   compensate for the last-value-wins behaviour. The Python reader should expose the full
   per-raw-file `dt` array (strictly more information, so nothing breaks by reading it), but any
   change to what *consumers* see must wait until the compensating Remora code has been found and
   audited. Tracked in [HANDOFF.md](../../HANDOFF.md).
8. **`io/ioReadXWAVHeader.m` does not implement v2 at all.** It reads the 8-byte `Reserved` field
   immediately after `Depth` (skipping `drate`) and uses fixed 32-byte raw-file entries (skipping
   `dt`), so on a v2 file it returns wrong `byte_loc`/`byte_length` values without warning.
   Verified: on the v0/v1 fixtures it agrees with `rdxwavhd` exactly; on the v2 fixture it does
   not (`tests/test_reference_dump.py::test_io_reader_disagrees_only_on_v2`).

   This matters beyond the port: **four Remoras carry copies of that reader**
   (`Remoras/Detector/io/`, `Remoras/SPICE-Detector/io/`,
   `Remoras/BlueWhaleBcall-Detector/Detection/`, and the root `io/`), so they all inherit the
   same v2 blindness. Worth checking whether any v2 files exist in the archive — if they do,
   results those Remoras produced from them are suspect.

9. **`WavVersionNumber` is stored as an ASCII digit by one writer** and as an integer by
   another, so `rdxwavhd.m:101,106,136`'s `== 2` tests cannot fire on a file from the first. See
   the dedicated subsection in section 3; 112 of the 123 `ExampleData` x.wav files take the ASCII
   form. Currently latent -- there are no v2 files -- and it compounds item 8.

## 7. Reference values

Fixtures generated by [`tools/make_fixtures.py`](../../tools/make_fixtures.py) exercise: v1 and v2
headers; 1 and 4 channels; 16 and 32 bit; continuous and duty-cycled; whole-second and
millisecond (`ticks`) start times; and a raw-file boundary inside a read window.

MATLAB ground truth for each is produced by
[`tools/matlab/dump_reference.m`](../../tools/matlab/dump_reference.m).
