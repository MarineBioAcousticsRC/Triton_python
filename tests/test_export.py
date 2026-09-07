"""Exporting the viewed window.

The important tests here are round trips: write a file, read it back with the same
readers the viewer uses, and assert what survived. That is the only way to know an
export is worth having, and it is what would have caught the MATLAB bugs catalogued in
docs/EXPORT_PLAN.md §2 -- most of which are about channel count and bit depth, so most
of these tests vary those deliberately.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from triton import export
from triton import io as tio
from triton.session import TritonSession

XWAV = "xwav_v1_cont_1ch_16b_10k.x.wav"
XWAV4 = "xwav_v1_cont_4ch_16b_10k.x.wav"
XWAV32 = "xwav_v1_cont_1ch_32b_10k.x.wav"
PLAINWAV = "wav_110130-084500_1ch_16b_10k.wav"


def _session(generated_dir: Path, name: str, tseg: float = 0.5) -> TritonSession:
    s = TritonSession()
    s.open_audio(generated_dir / name)
    s.view.tseg_sec = tseg
    return s


# ------------------------------------------------------------------------ naming


def test_default_stem_carries_the_timestamp():
    """MATLAB's reachable exports all default to `data.wav`; only its unreachable
    saveimageas timestamps the name. Exports should be self-identifying."""
    stem = export.default_stem(Path("/x/SOCAL_E_63_EN_180315_234230.x.wav"),
                               np.datetime64("2018-03-15T23:57:02.500"))
    assert stem == "SOCAL_E_63_EN_180315_234230@2018-03-15T23-57-02.500"
    assert ":" not in stem, "Windows forbids colons in filenames"


def test_default_stem_survives_no_source():
    assert export.default_stem(None, np.datetime64("2011-01-30T08:45:00")).startswith(
        "clip@")


# --------------------------------------------------------------------- plain wav


def test_wav_round_trips_the_counts_unchanged(generated_dir: Path, tmp_path: Path):
    """The unnormalised export must preserve sample values, or any level measured
    downstream is wrong. This is the whole reason both options exist."""
    s = _session(generated_dir, XWAV)
    frame = s.frame()
    out = export.write_wav(tmp_path / "clip.wav", frame)

    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == frame.fs
        assert w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    back = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    np.testing.assert_array_equal(back, frame.samples[:, 0])


def test_wav_writes_every_channel_interleaved(generated_dir: Path, tmp_path: Path):
    """MATLAB writes all channels column-major, i.e. de-interleaved, under a header
    that describes one (EXPORT_PLAN §2.5). Interleaving is what the format means."""
    s = _session(generated_dir, XWAV4, tseg=0.2)
    frame = s.frame()
    assert frame.samples.shape[1] == 4, "fixture assumption"

    out = export.write_wav(tmp_path / "four.wav", frame)
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 4
        raw = w.readframes(w.getnframes())
    back = np.frombuffer(raw, dtype="<i2").reshape(-1, 4).astype(np.float64)
    np.testing.assert_array_equal(back, frame.samples)


def test_wav_can_export_one_channel_of_many(generated_dir: Path, tmp_path: Path):
    s = _session(generated_dir, XWAV4, tseg=0.2)
    frame = s.frame()
    out = export.write_wav(tmp_path / "ch3.wav", frame, channel=3)
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    back = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    np.testing.assert_array_equal(back, frame.samples[:, 2])


def test_asking_for_a_channel_that_is_not_there_is_refused(generated_dir: Path,
                                                           tmp_path: Path):
    s = _session(generated_dir, XWAV)
    with pytest.raises(ValueError, match="out of range"):
        export.write_wav(tmp_path / "no.wav", s.frame(), channel=4)


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_normalisation_works_at_every_bit_depth(generated_dir: Path, tmp_path: Path,
                                                bits: int):
    """MATLAB scales only 16-bit and silently leaves 24- and 32-bit untouched
    (EXPORT_PLAN §2.1). 24-bit it cannot write at all -- fwrite has no 'int24'."""
    s = _session(generated_dir, XWAV)
    frame = s.frame()
    out = export.write_wav(tmp_path / f"n{bits}.wav", frame, normalise=True, bits=bits)

    with wave.open(str(out), "rb") as w:
        assert w.getsampwidth() == bits // 8
        raw = w.readframes(w.getnframes())
    if bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        back = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        back = np.where(back >= 1 << 23, back - (1 << 24), back)
    else:
        back = np.frombuffer(raw, dtype=f"<i{bits // 8}").astype(np.int64)

    full = 2 ** (bits - 1) - 1
    peak = int(np.max(np.abs(back)))
    assert peak > full * 0.99, f"{bits}-bit was not actually normalised: peak {peak}"


def test_normalisation_removes_the_mean(generated_dir: Path, tmp_path: Path):
    """Hydrophone data carries a DC offset; without removing it one side clips first."""
    s = _session(generated_dir, XWAV)
    frame = s.frame()
    assert abs(float(frame.samples.mean())) > 1, "fixture assumption: there is an offset"

    out = export.write_wav(tmp_path / "n.wav", frame, normalise=True)
    with wave.open(str(out), "rb") as w:
        back = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)
    assert abs(back.mean()) < 0.02 * np.max(np.abs(back))


def test_normalisation_keeps_the_balance_between_channels(generated_dir: Path,
                                                          tmp_path: Path):
    """One factor for the file, not one per channel: normalising each channel to its
    own peak would destroy the comparison a multichannel recording exists for."""
    s = _session(generated_dir, XWAV4, tseg=0.2)
    frame = s.frame()
    out = export.write_wav(tmp_path / "n4.wav", frame, normalise=True)
    with wave.open(str(out), "rb") as w:
        back = np.frombuffer(w.readframes(w.getnframes()),
                             dtype="<i2").reshape(-1, 4).astype(float)

    before = np.ptp(frame.samples - frame.samples.mean(axis=0), axis=0)
    after = np.ptp(back, axis=0)
    np.testing.assert_allclose(after / after.max(), before / before.max(),
                               rtol=0.02, atol=0.01)


def test_unnormalised_is_the_default(generated_dir: Path, tmp_path: Path):
    """Because measurement is the default use and normalising changes the numbers."""
    s = _session(generated_dir, XWAV)
    frame = s.frame()
    plain = export.write_wav(tmp_path / "a.wav", frame)
    with wave.open(str(plain), "rb") as w:
        back = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)
    np.testing.assert_array_equal(back, frame.samples[:, 0])


# -------------------------------------------------------------------------- x.wav


def test_xwav_round_trip_keeps_time_and_deployment(generated_dir: Path,
                                                   tmp_path: Path):
    """The test that justifies x.wav export existing at all.

    A wav clip is a sound. An x.wav clip still knows which moment and which deployment
    it came from, and the proof is reading it back with the viewer's own reader.
    """
    s = _session(generated_dir, XWAV)
    s.seek_text("@2.25")                     # start mid-file, not at a raw-file edge
    frame = s.frame()
    src_header = s.audio.source.header

    out = export.write_xwav(tmp_path / "clip.x.wav", frame, s)
    back = tio.read_xwav_header(out)

    assert back.n_raw_files == 1
    assert back.n_channels == frame.samples.shape[1]
    assert back.bits_per_sample == src_header.bits_per_sample
    assert back.sample_rate == frame.fs
    # The moment, to the millisecond the header can hold.
    assert back.start == np.datetime64(frame.start, "ms")
    # The deployment.
    assert back.instrument_id == src_header.instrument_id
    assert back.site_name == src_header.site_name
    assert back.experiment_name == src_header.experiment_name
    assert back.firmware_version == src_header.firmware_version
    assert back.latitude == pytest.approx(src_header.latitude, abs=1e-5)
    assert back.longitude == pytest.approx(src_header.longitude, abs=1e-5)
    assert back.depth == src_header.depth


def test_xwav_round_trip_returns_the_same_samples(generated_dir: Path, tmp_path: Path):
    s = _session(generated_dir, XWAV)
    frame = s.frame()
    out = export.write_xwav(tmp_path / "clip.x.wav", frame, s)

    reopened = tio.open_audio(out)
    got = reopened.read_samples(0, 0, frame.samples.shape[0])
    np.testing.assert_array_equal(got, frame.samples)


def test_xwav_round_trip_is_correct_for_four_channels(generated_dir: Path,
                                                      tmp_path: Path):
    """The case that is broken in MATLAB, and the lab has 4-channel deployments.

    Its header is sized from one channel while all four are written, column-major, so
    the file's header and body disagree *and* the samples are de-interleaved.
    """
    s = _session(generated_dir, XWAV4, tseg=0.2)
    frame = s.frame()
    out = export.write_xwav(tmp_path / "four.x.wav", frame, s)

    back = tio.read_xwav_header(out)
    assert back.n_channels == 4
    n = frame.samples.shape[0]
    assert back.raw_files[0].byte_length == n * 4 * 2, "header must describe all four"

    reopened = tio.open_audio(out)
    np.testing.assert_array_equal(reopened.read_samples(0, 0, n), frame.samples)


def test_xwav_byte_length_follows_the_real_bit_depth(generated_dir: Path,
                                                     tmp_path: Path):
    """wrxwavhd.m:20 hardcodes two bytes per sample, so a 32-bit export gets a header
    claiming half the bytes actually present (EXPORT_PLAN §2.4)."""
    s = _session(generated_dir, XWAV32)
    frame = s.frame()
    out = export.write_xwav(tmp_path / "b32.x.wav", frame, s)

    back = tio.read_xwav_header(out)
    assert back.bits_per_sample == 32
    n = frame.samples.shape[0]
    assert back.raw_files[0].byte_length == n * 4
    assert out.stat().st_size == back.raw_files[0].byte_loc + n * 4


def test_xwav_data_starts_where_a_real_harp_file_starts(generated_dir: Path,
                                                        tmp_path: Path):
    """One raw file at version 1 puts the data at byte 140, which is exactly the
    byte_loc in the real single-raw-file HARP file in fixtures/real."""
    s = _session(generated_dir, XWAV)
    out = export.write_xwav(tmp_path / "c.x.wav", s.frame(), s)
    assert tio.read_xwav_header(out).raw_files[0].byte_loc == 140


def test_xwav_export_refuses_a_plain_wav_source(generated_dir: Path, tmp_path: Path):
    """There is no harp metadata to carry, and a file with invented identity fields is
    worse than no file."""
    s = _session(generated_dir, PLAINWAV)
    with pytest.raises(ValueError, match="needs an x.wav source"):
        export.write_xwav(tmp_path / "no.x.wav", s.frame(), s)


# --------------------------------------------------------------- arrays and .mat


def test_npz_carries_the_arrays_and_its_own_provenance(generated_dir: Path,
                                                       tmp_path: Path):
    """The substitute for a saved .fig: it re-creates the data, not the figure."""
    s = _session(generated_dir, XWAV)
    frame = s.frame()
    out = export.write_arrays(tmp_path / "w.npz", frame, s)
    assert out.exists()

    with np.load(out, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["samples"], frame.samples)
        np.testing.assert_array_equal(z["spectrogram_db"], frame.spectrogram.db)
        np.testing.assert_array_equal(z["spectrogram_f_hz"], frame.spectrogram.f)
        assert int(z["sample_rate"][0]) == frame.fs
        meta = json.loads(str(z["provenance"]))
    assert meta["source"].endswith(XWAV)
    assert meta["nfft"] == s.view.nfft


def test_mat_is_readable_by_scipy(generated_dir: Path, tmp_path: Path):
    """Free via scipy, which is already a core dependency, so MATLAB analysis scripts
    keep working."""
    from scipy.io import loadmat

    s = _session(generated_dir, XWAV)
    frame = s.frame()
    out = export.write_mat(tmp_path / "w.mat", frame, s)

    m = loadmat(str(out))
    np.testing.assert_array_equal(m["samples"], frame.samples)
    np.testing.assert_array_equal(m["spectrogram_db"], frame.spectrogram.db)


# ------------------------------------------------------------------ provenance


def test_provenance_records_what_shaped_the_output(generated_dir: Path):
    s = _session(generated_dir, XWAV)
    s.view.nfft = 512
    s.view.filter_on = True
    s.view.filter_low, s.view.filter_high = 200.0, 3000.0
    meta = export.provenance(s, s.frame())

    assert meta["nfft"] == 512
    assert meta["display_filter_hz"] == [200.0, 3000.0]
    assert meta["deployment"]["site"]
    assert meta["window_start"].startswith("2011-01-30T08:45")


def test_provenance_omits_the_filter_when_it_is_off(generated_dir: Path):
    s = _session(generated_dir, XWAV)
    assert "display_filter_hz" not in export.provenance(s, s.frame())


def test_sidecar_is_written_beside_the_export(generated_dir: Path, tmp_path: Path):
    """A wav clip is the artefact most likely to be emailed and least likely to arrive
    with any record of where it came from."""
    s = _session(generated_dir, XWAV)
    frame = s.frame()
    out = export.write_wav(tmp_path / "clip.wav", frame,
                           sidecar=export.provenance(s, frame))
    # Beside the export, named after it, so the two cannot be separated by accident.
    side = out.with_suffix(out.suffix + ".json")
    assert side == tmp_path / "clip.wav.json"
    assert side.exists()
    meta = json.loads(side.read_text())
    assert meta["source"].endswith(XWAV)
    assert meta["normalised"] is False and meta["bits"] == 16


# ------------------------------------------------------------------- raw image


def test_spectrogram_rgb_is_one_pixel_per_bin(generated_dir: Path):
    """Reviving filepd.m:403, which MATLAB creates disabled and invisible. It is the
    only export that writes the data as an image rather than a picture of a plot."""
    s = _session(generated_dir, XWAV)
    s.view.nfft = 256
    tile = s.frame().spectrogram
    rgb = export.spectrogram_rgb(tile, "jet")

    assert rgb.shape == (tile.db.shape[0], tile.db.shape[1], 3)
    assert rgb.dtype == np.uint8


def test_spectrogram_rgb_puts_low_frequencies_at_the_bottom(generated_dir: Path):
    """Row 0 of an array is the top of an image, so it has to be flipped to match the
    screen."""
    s = _session(generated_dir, XWAV)
    tile = s.frame().spectrogram
    unflipped = export.spectrogram_rgb(tile, "jet", flip=False)
    flipped = export.spectrogram_rgb(tile, "jet")
    np.testing.assert_array_equal(flipped, unflipped[::-1])


def test_spectrogram_rgb_uses_the_viewers_colour_map(generated_dir: Path):
    """Same lookup table the image panel uses, so exported pixels match the screen.
    MATLAB's version used dB values directly as colour indices, so its output did not
    match its own display."""
    from triton.colormaps import colormap_lut

    s = _session(generated_dir, XWAV)
    tile = s.frame().spectrogram
    rgb = export.spectrogram_rgb(tile, "jet", flip=False)

    lut = colormap_lut("jet")
    lo, hi = tile.clim
    i, j = 3, tile.db.shape[1] // 2
    frac = (tile.db[i, j] - lo) / (hi - lo)
    expect = lut[int(np.clip(round(frac * 255), 0, 255))]
    np.testing.assert_array_equal(rgb[i, j], expect)


def test_spectrogram_rgb_handles_a_non_finite_bin(generated_dir: Path):
    """A silent bin is -inf, and an LTSA gap is NaN. Neither should raise."""
    s = _session(generated_dir, XWAV)
    tile = s.frame().spectrogram
    tile.db[0, 0] = -np.inf
    tile.db[1, 1] = np.nan
    rgb = export.spectrogram_rgb(tile, "jet", flip=False)
    assert rgb.dtype == np.uint8

    # Painted as a gap, not as the quietest bin: an exported image must not make a
    # hole in the data look like data.
    np.testing.assert_array_equal(rgb[0, 0], (128, 128, 128))
    np.testing.assert_array_equal(rgb[1, 1], (128, 128, 128))
    assert not np.array_equal(rgb[0, 0], colormap_lut_bottom()), \
        "a gap must not be the bottom of the colour scale"


def colormap_lut_bottom():
    from triton.colormaps import colormap_lut
    return colormap_lut("jet")[0]


def test_an_unknown_colour_map_falls_back(generated_dir: Path):
    s = _session(generated_dir, XWAV)
    tile = s.frame().spectrogram
    np.testing.assert_array_equal(
        export.spectrogram_rgb(tile, "no such map"),
        export.spectrogram_rgb(tile, "jet"),
    )
