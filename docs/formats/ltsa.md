# LTSA (`*.ltsa`) format specification

Long-Term Spectral Average. Derived from
[`read_ltsahead.m`](../../../Triton-master/read_ltsahead.m),
[`write_ltsahead.m`](../../../Triton-master/write_ltsahead.m),
[`read_ltsadata.m`](../../../Triton-master/read_ltsadata.m),
[`calc_ltsa.m`](../../../Triton-master/calc_ltsa.m),
[`ck_ltsaparams.m`](../../../Triton-master/ck_ltsaparams.m),
[`get_headers.m`](../../../Triton-master/get_headers.m).

Little-endian throughout.

---

## 1. Layout

```
+-----------------------------------------------+ 0
| file header                          64 bytes |
+-----------------------------------------------+ 64 = dirStartLoc - 1
| directory: one entry per raw file             |
|   v1-v3: 64 bytes each                        |
|   v4:   104 bytes each                        |
|   ...padded out to maxNrawfiles entries       |
+-----------------------------------------------+ dataStartLoc
| spectral averages, int8, column-major         |
+-----------------------------------------------+
```

`maxNrawfiles = nrftot + 100` (`write_ltsahead.m:41`) — the directory is over-allocated by 100
entries, zero-filled, so a file can in principle be extended. `dataStartLoc = 104 * maxNrawfiles + 64`
for v4.

Note `dirStartLoc` is written as `64 + 1 = 65` — a **1-based** MATLAB index, whereas
`dataStartLoc` is a 0-based byte count used directly in `fseek`. Do not assume they use the same
convention.

## 2. File header (64 bytes)

| Offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 4 | char | type | `"LTSA"` |
| 4 | 1 | uint8 | `ver` | 1, 2, 3, or 4. Writer only emits 4 |
| 5 | 3 | char | spare | writer emits `"xxx"` |
| 8 | 4 | uint32 | `dirStartLoc` | = 65 (1-based; see above) |
| 12 | 4 | uint32 | `dataStartLoc` | 0-based byte offset of first spectrum |
| 16 | 4 | float32 | `tave` | seconds per time bin |
| 20 | 4 | float32 | `dfreq` | Hz per frequency bin |
| 24 | 4 | uint32 | `fs` | sample rate |
| 28 | 4 | uint32 | `nfft` | samples per FFT |
| 32 | 2 or 4 | uint16/uint32 | `nrftot` | total raw files. **uint16 for v1/v2, uint32 for v3/v4** |
| 34 or 36 | 2 | uint16 | `nxwav` | number of source audio files |
| 36 or 38 | 1 | uint8 | `ch` | which channel was averaged |
| — | 27 or 25 | uint8 | padding | `sk = 27` (v1/v2) or `25` (v3/v4) — pads to 64 |

## 3. Directory entries

| Rel. offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 1 | uint8 | `year` | year − 2000, as in x.wav |
| 1 | 1 | uint8 | `month` | |
| 2 | 1 | uint8 | `day` | |
| 3 | 1 | uint8 | `hour` | |
| 4 | 1 | uint8 | `minute` | |
| 5 | 1 | uint8 | `secs` | |
| 6 | 2 | uint16 | `ticks` | milliseconds |
| 8 | 4 | uint32 | `byteloc` | absolute offset of this raw file's first spectrum |
| 12 | 2 or 4 | uint16/uint32 | `nave` | number of time bins. **uint16 for v1/v2, uint32 for v3/v4** |
| … | 40 or 80 | char | `fname` | source filename. **40 bytes for v1–v3, 80 for v4** |
| … | 1 or 4 | uint8/uint32 | `rfileid` | index of the raw file within its x.wav. **uint8 v1–v3, uint32 v4** |
| … | 9 / 7 / 4 | uint8 | padding | `sk = 9` (v1/v2), `7` (v3), `4` (v4) |

Total entry size: 64 bytes for v1–v3, 104 bytes for v4.

### 3.1 Derived per-entry values

```
dnumStart[k] = datenum([year month day hour minute secs + ticks/1000])   # year-2000 offset
dur[k]       = tave * nave[k]                                            # seconds
dnumEnd[k]   = dnumStart[k] + (dur[k] - 1/fs)                            # seconds
```

Note `dnumEnd` subtracts `1/fs`, not `1/tave` (`read_ltsahead.m:159`).

### 3.2 Byte location chain (writer)

```
byteloc[0] = dataStartLoc
byteloc[k] = byteloc[k-1] + nave[k-1] * nfreq * 1      # 1 byte per value
```

with a guard that aborts if `byteloc > 2^32` (`write_ltsahead.m:118-123`).

## 4. Data section

A contiguous run of **`int8`** values: for each raw file, `nave[k]` spectra of `nfreq` values each,
stored spectrum-major (all frequencies of bin 0, then all of bin 1, …), read back as
`fread(fid, [nf, nbin], 'int8')` (`read_ltsadata.m:52`).

```
nfreq = nfft/2 + 1        if nfft even
        (nfft + 1)/2      if nfft odd
```

**Values are dB, quantised to signed 8-bit integers.** Anything below −128 dB or above 127 dB is
clipped by MATLAB's `fwrite`, which also rounds to nearest. This is a lossy on-disk format and the
quantisation must be reproduced exactly for byte-identical output.

### 4.1 Random access

`read_ltsadata.m:41-42`:

```
skip = byteloc[plotStartRawIndex] + (plotStartBin - 1) * nf
```

i.e. the offset within a raw file is counted in **bytes** as `bins * nf`, valid because each value
is 1 byte. `nbin = floor(tseg.hr * 3600 / tave)` bins are read at once and the reader deliberately
runs across raw-file boundaries (same splice behaviour as x.wav; see
[xwav.md §5.3](xwav.md#53-reading-a-segment)).

## 5. How the values are computed

`calc_ltsa.m`:

```matlab
window   = hanning(nfft);        % NOT scipy hann(nfft) -- see below
noverlap = 0;
[p, f]   = pwelch(data, window, noverlap, nfft, fs);
ltsa     = 10*log10(p);          % dB re counts^2/Hz
fwrite(fod, ltsa, 'int8');
```

Per time bin, `sampPerAve = tave * fs` samples are read; the last bin of each raw file is short and
is **zero-padded to `nfft`** if needed (`calc_ltsa.m:146-154`).

### 5.1 The window trap

MATLAB `hanning(N)` is the symmetric Hann window **with its zero endpoints removed**. It is
neither `scipy.signal.windows.hann(N)` (which includes the zeros) nor `hann(N, sym=False)` (the
periodic variant). The exact equivalent is:

```python
w = scipy.signal.windows.hann(N + 2, sym=True)[1:-1]     # == MATLAB hanning(N)
```

Using the wrong one shifts every value by a fraction of a dB — invisible in a spectrogram, fatal
for reproducing published numbers. The same applies to `mkspecgram.m`.

### 5.2 Parameter derivation

From `ck_ltsaparams.m`, given user-chosen `tave` (s) and `dfreq` (Hz):

```
nfft  = floor(fs / dfreq)
cfact = tave * fs / nfft                       # compression factor
nave  = ceil(Nsamp / (nfft * cfact))           # time bins per raw file
```

where `Nsamp` is the raw file's per-channel sample count:
`write_length * blksz / nch` for x.wav, or `nsamp` from `audioinfo` for wav/flac.

`tave` is clamped to `write_length[0] * blksz / fs` for x.wav input.

## 6. `ftype` / `dtype` codes

Two separate enumerations, easy to confuse:

| `PARAMS.ltsa.ftype` | meaning |
|---:|---|
| 1 | wav |
| 2 | xwav |
| 3 | flac |

| `PARAMS.ltsa.dtype` | meaning |
|---:|---|
| 1 | HARP (512-byte sectors, 12-byte headers) |
| 2 | ARP *(commented out)* |
| 3 | OBS *(commented out)* |
| 4 | wav/flac — no sector geometry |

Note `PARAMS.ftype` (the display path, `1 = wav`, `2 = xwav`) uses the same numbering as
`PARAMS.ltsa.ftype` but is a *different field*.

## 7. Known issues

> **Source-tree warning.** This spec was written against `D:\Code\Triton-master`. A second local
> checkout, `D:\Code\Triton_remoras`, differs in 19 base files — **including most of the LTSA
> pipeline** (`calc_ltsa`, `write_ltsahead`, `get_headers`, `ck_ltsaparams`, `read_ltsahead`,
> `read_ltsadata`). Neither tree is a superset of the other.
>
> Running the reference dump against both trees gives **byte-identical output on every path that
> has fixtures today**, so the divergence is confined. But the LTSA *generation* path has no
> fixtures yet, and items 1–2 below are exactly where the two trees differ — so this section is
> the one place where "which tree?" is still an open numeric question. See
> [HANDOFF.md §4](../../HANDOFF.md).

1. **`write_ltsahead` always prompts — in Triton-master.** The guard at line 18 is
   `if ~exist('PARAMS.ltsa.outfile','var')`; `exist` cannot test a dotted field name, so this is
   always true and `uiputfile` always opens, even when the caller has already set the output
   filename. This blocks non-interactive LTSA creation (see
   [`tools/matlab/make_ltsa_fixture.m`](../../tools/matlab/make_ltsa_fixture.m)).

   **`Triton_remoras` already has the fix** — `if ~isfield(PARAMS.ltsa, 'outfile')`, added for
   the BatchLTSA remora. Identical to the fix independently proposed here. Port that version.
2. **`calc_ltsa` zero-padding differs between the two trees**, and it changes numeric output.
   Triton-master pads the short final bin with an unconditional `data = [data,dz'];`
   (line 149), which is only correct for the row-vector shape x.wav reading produces; for
   wav/flac input `data` is a column vector and the concatenation builds a matrix instead of a
   padded vector. `Triton_remoras` branches on orientation and is correct:

   ```matlab
   if size(data,1) > 1        % column vector, typical for wav/flac
       data = [data;dz];
   elseif size(data,2) > 1    % row vector, typical for xwav
       data = [data,dz'];
   end
   ```

   Affects the last time bin of every raw file whose sample count is not a whole multiple of
   `nfft * cfact`. **Decide which behaviour the Python `mkltsa` must reproduce before writing
   it**, and note that existing archived `.ltsa` files may contain either.

   Conversely, Triton-master is *ahead* on flac support (`ftype == 3` throughout `get_headers`
   and `calc_ltsa`), which `Triton_remoras` lacks. The trees are genuinely mixed.
3. **The writer only emits v4**, but the reader supports v1–v4, so all four must be implemented on
   the read side.
4. `read_ltsahead` mixes I/O with GUI state — it enables pick buttons and installs figure
   callbacks at lines 190-200. Pure read/parse must be separated in the port.
5. `PARAMS.ltsa.dur` is a per-raw-file vector in `read_ltsahead` but is overwritten with a scalar
   in `calc_ltsa.m:85` on the short-final-bin path.
