# Export plan — audio clips and images

What MATLAB Triton's File menu offers, what it actually does, and how to build the
equivalent in the Python viewer. Written before the code so the decisions are visible
and arguable.

## 1. What MATLAB has

From `initpulldowns.m:12-52`. Both submenus are disabled until a file is open.

| Menu item | Implementation | Port? |
|---|---|---|
| Export Plotted Data → **normalized WAV** | `filepd.m:227` | Yes, with fixes |
| Export Plotted Data → **WAV** | `filepd.m:263` | Yes |
| Export Plotted Data → **XWAV** | `filepd.m:297` + `wrxwavhd(1)` | Yes — the most valuable one |
| Export Plotted Data → **MATLAB `.mat`** | `miscpd.m:171`, `save('DATA')` | Yes, plus `.npz` |
| Save Plot Window As → **JPEG** | `print(fig,'-djpeg100','-r300')` | Yes, as PNG and JPEG |
| Save Plot Window As → **PDF** | `print(fig,'-dpdf')` | Yes, with a caveat (§4) |
| Save Plot Window As → **`.fig`** | `hgsave` | **No** — MATLAB-only format |
| **Export PARAMS as `.mat`** | `miscpd.m:178`, `save('PARAMS')` | Yes, as JSON and `.mat` |
| **Save Spectrogram As Image** | `filepd.m:403` | Yes — and it is *disabled* in MATLAB |

The last one is worth noticing: `initpulldowns.m:46` creates it with `'Visible','off'`
and `'Enable','off'`, so nobody can reach it. It is the only item that writes the
spectrogram as **raw pixels** — one pixel per time bin per frequency bin, no axes, no
labels, colour-mapped — which is a genuinely different artefact from a screenshot and
the right thing for figure assembly or feeding another tool. Reviving it is free.

## 2. Bugs in the MATLAB export path

Found by reading; each would be inherited by a faithful port.

1. **"Normalized WAV" only normalises 16-bit data.** `filepd.m:252-259` computes the
   scale factor in the `nBits == 16` branch only; the 24- and 32-bit branches do a bare
   `int32(DATA)` with no scaling. A 32-bit file exported as "normalized" is not
   normalised, silently.
2. **Normalisation is broken for multichannel.** `mxd = max(abs(DATA))` on an
   *n*×*nch* matrix returns a 1×*nch* row, so `sf * DATA` is a matrix multiply rather
   than a scaling, and errors for any *nch* ≠ *n*.
3. **No mean removal before normalising.** Hydrophone data commonly carries a DC
   offset, so the peak is asymmetric about zero and one side clips first. Same issue
   already fixed in `triton/playback.py`.
4. **The x.wav header hardcodes 16-bit.** `wrxwavhd.m:20` sets
   `data_bytes = num_samples * 2` regardless of `nBits`, so a 32-bit export gets a
   header claiming half the bytes actually written.
5. **Multichannel x.wav export writes the wrong bytes in the wrong order.**
   `wrxwavhd(1)` sizes the header from `length(DATA(:,PARAMS.ch))` — one channel — while
   `filepd.m:334` writes `fwrite(fod, DATA, dtype)`, i.e. *all* channels. MATLAB's
   `fwrite` walks a matrix column-major, so the samples come out channel-major
   (de-interleaved) where the format requires interleaved frames. **Multichannel x.wav
   export is therefore broken today**, and the lab has 4-channel deployments.
6. **`dtype = 'int24'`** at `filepd.m:307` — a type `fwrite` does not have, the same
   bug as `readseg.m:98`. 24-bit export cannot work.

Items 4–6 are worth reporting regardless of the port. 1–3 are conveniences; nothing
measured depends on export gain, so they should be fixed rather than reproduced.

## 3. What to build

The established split holds: arithmetic and file writing in the core where it can be
tested without a display, Qt only for grabbing rendered pixels.

### `triton/export.py` — new, no Qt

```python
def write_wav(path, frame, *, normalise=False, channel=None) -> Path
def write_xwav(path, frame, session) -> Path
def write_arrays(path, frame, session) -> Path        # .npz
def write_mat(path, frame, session) -> Path           # scipy.io.savemat, free
def spectrogram_rgb(tile, colormap, *, flip=True) -> np.ndarray   # (h, w, 3) uint8
def provenance(session, frame) -> dict                # the sidecar / stamp contents
```

Notes on each:

* **`write_wav`** writes all channels interleaved, via `wave` from the standard library
  so no dependency is added. `normalise=True` removes the mean, scales the peak to full
  scale **at every bit depth**, and does it per-file rather than per-channel so the
  inter-channel balance survives. Fixes §2.1–2.3.
* **`write_xwav`** is the valuable one, because the format carries the clip's absolute
  start time and the deployment's identity — instrument, site, experiment, position,
  gain, firmware — so an exported clip stays traceable to the deployment it came from.
  Everything needed is already in `XwavHeader`, and `tools/mkltsa.py` already writes
  x.wav-adjacent headers, so the byte layout is proven. One raw file, correct
  `byte_length` for the actual bit depth, interleaved samples. Fixes §2.4–2.6.
  Refuses when the source is a plain wav, since there is no harp metadata to carry.
* **`write_arrays`** replaces `export_mat` for Python users: samples, the spectrogram
  dB, frequency and time axes, and the provenance dict, in one `.npz`.
* **`write_mat`** keeps MATLAB users working. `scipy.io.savemat` costs nothing — scipy
  is already a core dependency.
* **`spectrogram_rgb`** revives the disabled `saveimageas`, using `dsp`'s colour-map
  LUT so the exported pixels match what is on screen. Returns an array rather than
  writing a file, so the core needs no image library and the caller chooses the format.
* **`provenance`** is the same content as the on-plot stamps, from `session.snapshot()`.

### `triton/gui/` — the Qt half

* **Save plot window** → PNG or JPEG via `QWidget.grab()`, and PDF via `QPdfWriter`.
* **Save spectrogram image** → `spectrogram_rgb` into a `QImage`, saved at native
  resolution: one pixel per bin, not a screenshot.
* **File menu** wiring, mirroring MATLAB's grouping so the muscle memory carries:

```
File
  Open audio…                          Ctrl+O
  Open LTSA…                           Ctrl+L
  ──
  Export plotted data ▸   WAV… / normalized WAV… / x.wav… / NumPy .npz… / MATLAB .mat…
  Save plot window as ▸   PNG… / JPEG… / PDF…
  Save spectrogram image…                        (raw pixels, no axes)
  Export session metadata ▸  JSON… / MATLAB .mat…
  ──
  Quit
```

Both submenus disabled until something is open, as in MATLAB.

### Two additions MATLAB does not have

**A provenance sidecar.** Every export writes `<name>.json` beside it with the source
file, the exact window, and every parameter that shaped the output. An exported wav clip
with no record of where it came from is a liability — it is the artefact most likely to
be emailed around and least likely to carry context. For x.wav the timestamp is *in* the
file, which is the strongest argument for preferring that format.

**A timestamped default filename.** MATLAB's hidden `saveimageas` already does this
(`filepd.m:407`: `infile@HH-MM-SS_mmm`), and it is a good idea the other exports do not
use — they default to `data.wav`. Proposed: `{source stem}@{ISO time, colons to dashes}`,
so exports are self-identifying and sort chronologically.

## 4. Where the port will be worse than MATLAB

Stated plainly rather than discovered later.

**PDF will not be truly vector.** MATLAB's `print -dpdf` emits vector axes with the
spectrogram as an embedded raster. Qt's `QPdfWriter` will do the same for the axes, but
pyqtgraph's `SVGExporter` handles `ImageItem` poorly, so a fully vector spectrogram is
not on offer either way. In practice a 300 dpi PNG is the better artefact for a figure,
and PDF is there for people who want a page. If someone needs genuinely vector output
for publication, that is `matplotlib` and the `export` extra — which is what that extra
was reserved for.

**No `.fig` equivalent.** Nothing portable corresponds to a saved MATLAB figure. The
`.npz` plus the provenance JSON is the substitute: it re-creates the *data* rather than
the *figure*, which is more useful and less brittle.

## 5. Order of work

1. `triton/export.py` with `write_wav`, `provenance`, and the filename helper — the
   smallest useful slice, testable with no GUI.
2. `write_xwav`, with a round-trip test: export a clip, read it back with
   `read_xwav_header`, and assert the start time, gain and identity fields survived.
   That test is the real proof this is worth having.
3. `spectrogram_rgb` plus `write_arrays` / `write_mat`.
4. The File menu and the Qt image/PDF writers.
5. A `triton-export` CLI over the same functions, so a whole deployment can be clipped
   in batch. Not part of this piece, but the split above means it costs almost nothing
   later.

## 6. Questions worth settling first

1. **Does anyone still need `.mat`?** It is nearly free via scipy, so the answer only
   changes whether it appears in the menu. Default: include it.
2. **Should "export plotted data" write all channels or the displayed one?** MATLAB
   intends one and writes a broken version of all. Proposal: all channels, correctly
   interleaved, for wav and x.wav — a multichannel clip is what a multichannel
   recording is. The displayed channel alone stays available as an option.
3. **Is the `.wav` clip expected to be openable by Audacity, Raven, or similar?** If
   so the sample values must be plain counts (the `WAV` option), because normalising
   changes the numbers and any level read downstream would be wrong. Worth being
   explicit that `normalized WAV` is for listening and `WAV` is for measuring.
