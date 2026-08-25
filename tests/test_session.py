"""Phase 2: the session layer.

Two kinds of test here, and the distinction matters.

**Equivalence.** The session must change no numbers. It composes the Phase 1 code that
was verified against MATLAB byte-for-byte, so anything obtained through a session must
be identical to the same thing obtained by calling `triton.io` and `triton.dsp`
directly. Without these tests the session could quietly become a second implementation,
and the parity suite would not notice -- it never goes through the session.

**Behaviour.** Notification, validation, and the exactness properties the session
exists to provide. These have no MATLAB oracle because they are new: MATLAB has no
change notification and validates by failing later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import triton
from triton import dsp
from triton import io as tio
from triton.session import TritonSession, ValidationError

XWAV = "xwav_v1_cont_1ch_16b_10k.x.wav"
DUTY = "xwav_v1_duty_1ch_16b_10k.x.wav"
LTSA = "xwav_v1_cont_1ch_16b_10k__tave1_df100.ltsa"


@pytest.fixture
def s(generated_dir: Path) -> TritonSession:
    sess = TritonSession()
    sess.open_audio(generated_dir / XWAV)
    return sess


# --------------------------------------------------------------- THE MILESTONE


def test_milestone_open_seek_and_retrieve_both_tiles(generated_dir: Path):
    """Phase 2's definition of done, from PORTING_PLAN §5.

    Open a file, seek to a time, retrieve a spectrogram tile and an LTSA tile --
    entirely in a test, with no GUI.

    The last clause is not decoration. `triton.session` must not import a GUI toolkit,
    which is why change notification is a callback registry rather than Qt signals
    (docs/OPEN_DECISIONS.md §4). The assertion at the end of this test is what keeps
    that true, because the natural drift is for someone to reach for Qt signals when
    wiring Phase 3.
    """
    sess = triton.open_session(
        audio=generated_dir / XWAV, ltsa=generated_dir / LTSA, make_current=False
    )
    sess.view.tseg_sec = 0.5
    sess.view.nfft = 256

    sess.seek(sess.audio.source.segments[0].start + np.timedelta64(1500, "ms"))

    tile = sess.spectrogram_tile()
    assert tile.db.ndim == 2
    assert tile.db.shape[0] == tile.f.size
    assert tile.db.shape[1] == tile.t.size
    assert tile.clim[0] < tile.clim[1]
    assert tile.start == sess.audio.time

    lt = sess.ltsa_tile()
    assert lt.db.ndim == 2
    assert lt.db.shape[0] == lt.f.size
    assert lt.db.size > 0

    # Checked by scanning the module's own imports, NOT by looking at sys.modules.
    #
    # This started out doing both. The sys.modules half was wrong twice over: a toolkit
    # that is not installed cannot appear there, so it passed for the wrong reason on a
    # machine without Qt -- and once tests/test_gui.py existed it started *failing* for
    # the wrong reason too, because pytest runs both in one process and any GUI test
    # legitimately puts PySide6 in sys.modules. It could never distinguish "the session
    # imported Qt" from "something else in this process did".
    #
    # The AST scan answers the actual question and is unaffected by what else has run.
    import ast

    gui = {"PySide6", "PyQt5", "PyQt6", "matplotlib", "tkinter", "pyqtgraph"}

    import triton.session as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not (gui & imported), (
        f"triton.session imports a GUI toolkit: {sorted(gui & imported)}. Phase 3 "
        f"adapts the callback registry to Qt; it must not be the other way round."
    )


# ------------------------------------------------------------------ equivalence


def test_spectrogram_tile_equals_calling_dsp_directly(s: TritonSession):
    """The session must add no arithmetic of its own."""
    s.view.tseg_sec = 0.5
    s.view.nfft = 256
    s.view.overlap_pct = 50
    s.seek(s.audio.source.segments[0].start + np.timedelta64(250, "ms"))

    tile = s.spectrogram_tile()

    src = s.audio.source
    data, bounds = src.read_at(s.audio.time, 0.5, splice_gaps=True)
    expect = dsp.spectrogram(data[:, 0], fs=src.sample_rate, nfft=256,
                             overlap_pct=50, freq0=0.0, freq1=src.sample_rate / 2)

    np.testing.assert_array_equal(tile.db, expect.db)
    np.testing.assert_array_equal(tile.f, expect.f)
    np.testing.assert_array_equal(tile.t, expect.t)
    assert [b.offset_sec for b in tile.boundaries] == [b.offset_sec for b in bounds]


def test_ltsa_tile_equals_calling_io_directly(generated_dir: Path):
    sess = TritonSession()
    sess.open_ltsa(generated_dir / LTSA)
    sess.ltsa.tseg_hr = 1 / 60

    tile = sess.ltsa_tile()
    direct = tio.open_ltsa(generated_dir / LTSA)
    expect = direct.read_block(direct.start_time, hours=1 / 60)

    np.testing.assert_array_equal(tile.db, expect)
    np.testing.assert_array_equal(tile.f, direct.f)


def test_brightness_and_contrast_are_the_only_transform(s: TritonSession):
    """Contrast scales, brightness shifts, in that order -- and nothing else happens."""
    s.view.tseg_sec = 0.25
    plain = s.spectrogram_tile().db.copy()

    s.view.clim = None                      # let it re-derive
    s.view.contrast = 200.0
    s.view.brightness = -10.0
    shifted = s.spectrogram_tile().db

    np.testing.assert_allclose(shifted, plain * 2.0 - 10.0, rtol=0, atol=1e-12)


def test_transfer_function_is_added_per_frequency(s: TritonSession):
    s.view.tseg_sec = 0.25
    plain = s.spectrogram_tile().db.copy()

    s.view.clim = None
    s.calibration.tf_freq = np.array([10.0, 100.0, 1000.0, 5000.0])
    s.calibration.tf_value = np.array([5.0, 12.0, 30.0, 45.0])
    corrected = s.spectrogram_tile()

    curve = dsp.apply_transfer_function_curve(
        s.calibration.tf_freq, s.calibration.tf_value, corrected.f
    )
    np.testing.assert_allclose(corrected.db, plain + curve[:, None], rtol=0, atol=1e-12)


# ----------------------------------------------------- position is exact, not drifting


def test_stepping_forward_and_back_returns_to_the_same_sample(s: TritonSession):
    """The property the integer-sample position exists to provide.

    MATLAB accumulates `PARAMS.plot.dnum` as a float datenum, so many steps forward
    and back need not land where they started -- and at a sample boundary the float
    noise decides which sample is read (`AudioSource.skip_for`). Here the arithmetic is
    in samples, so the round trip is exact by construction.

    The step count is chosen to stay **inside** the file, and that is checked rather
    than assumed. An earlier version of this test used 500 steps in a 100,000-sample
    file: both directions clamped at the ends, so it passed while demonstrating
    clamping rather than exactness. Clamping is deliberately *not* reversible -- once
    clamped, how far past the end the step went is gone -- so a round-trip test that
    clamps proves nothing about drift.
    """
    src = s.audio.source
    total = sum(seg.n_samples for seg in src.segments)
    s.view.tseg_sec = 0.1
    per_step = int(0.1 * src.sample_rate)
    n_steps = (total // per_step) // 2                  # halfway, so no clamping

    start = (s.audio.segment, s.audio.offset)
    for _ in range(n_steps):
        s.step(+1)
    moved = (s.audio.segment, s.audio.offset)

    assert not s.at_end(), "the test must not clamp, or it proves nothing"
    assert moved != start, f"{n_steps} steps should have moved somewhere"

    for _ in range(n_steps):
        s.step(-1)
    assert not s.at_end()
    assert (s.audio.segment, s.audio.offset) == start


def test_many_small_steps_do_not_drift(s: TritonSession):
    """The same property under a step size that is not a whole number of samples.

    0.0301 s at 10 kHz is 301 samples exactly, but 0.03005 s is 300.5 -- so `step`
    rounds, and the question is whether the rounding accumulates. It cannot, because
    each step is computed from the current integer offset rather than from a running
    float total. This is the failure mode MATLAB has and this design does not.
    """
    s.view.tseg_sec = 0.03005
    per_step = int(round(0.03005 * s.audio.source.sample_rate))
    for i in range(1, 51):
        s.step(+1)
        assert s.audio.offset == i * per_step, f"drifted at step {i}"


def test_step_carries_across_segment_boundaries(s: TritonSession):
    s.view.tseg_sec = 1.0
    n = s.audio.source.segments[0].n_samples
    fs = s.audio.source.sample_rate

    steps = n // fs + 1                      # enough to leave segment 0
    for _ in range(steps):
        s.step(+1)
    assert s.audio.segment == 1
    # ...and back again lands exactly where it started
    for _ in range(steps):
        s.step(-1)
    assert (s.audio.segment, s.audio.offset) == (0, 0)


def test_step_clamps_rather_than_raising(s: TritonSession):
    """An analyst holding a motion button expects it to stop, not to throw.

    Both ends, and the offset as well as the segment. Clamping only the segment would
    leave `offset` running arbitrarily far past the last segment's length -- a position
    that still looks valid and reads as nothing.
    """
    src = s.audio.source
    s.view.tseg_sec = 1.0

    for _ in range(10_000):
        s.step(-1)
    assert (s.audio.segment, s.audio.offset) == (0, 0)
    assert s.at_start()

    for _ in range(10_000):
        s.step(+1)
    last = len(src.segments) - 1
    assert s.audio.segment == last
    assert s.audio.offset < src.segments[last].n_samples, (
        f"offset {s.audio.offset} ran past the segment's {src.segments[last].n_samples} "
        f"samples"
    )
    assert s.at_end()
    # and the clamped position is still readable, which is the point of clamping
    assert s.spectrogram_tile().db.size > 0


def test_time_is_derived_from_the_sample_index(s: TritonSession):
    s.seek_samples(1, 2500)
    seg = s.audio.source.segments[1]
    expect = seg.start + np.timedelta64(int(2500 * 1e9 / s.audio.source.sample_rate), "ns")
    assert s.audio.time == expect


def test_seek_snaps_out_of_a_duty_cycle_gap(generated_dir: Path):
    sess = TritonSession()
    sess.open_audio(generated_dir / DUTY)
    src = sess.audio.source

    gap_mid = np.datetime64(
        src.segments[0].end_ns
        + (src.segments[1].start_ns - src.segments[0].end_ns) // 2, "ns"
    )
    sess.seek(gap_mid)                       # forward, from the file start
    assert sess.audio.segment == 1
    assert sess.audio.offset == 0


# -------------------------------------------------------------------- notification


def test_notification_reports_dotted_paths(s: TritonSession):
    seen: list[str] = []
    s.subscribe(seen.append)

    s.view.nfft = 512
    s.view.brightness = 3.0
    s.ltsa.tseg_hr = 2.0

    assert seen == ["view.nfft", "view.brightness", "ltsa.tseg_hr"]


def test_notification_is_skipped_when_the_value_does_not_change(s: TritonSession):
    seen: list[str] = []
    s.subscribe(seen.append)
    s.view.nfft = 512
    s.view.nfft = 512
    assert seen == ["view.nfft"]


def test_prefix_filters_subscriptions(s: TritonSession):
    view: list[str] = []
    ltsa: list[str] = []
    s.subscribe(view.append, prefix="view")
    s.subscribe(ltsa.append, prefix="ltsa")

    s.view.nfft = 512
    s.ltsa.tseg_hr = 3.0

    assert view == ["view.nfft"]
    assert ltsa == ["ltsa.tseg_hr"]


def test_unsubscribe(s: TritonSession):
    seen: list[str] = []
    off = s.subscribe(seen.append)
    s.view.nfft = 512
    off()
    s.view.nfft = 256
    assert seen == ["view.nfft"]


def test_batch_collapses_notifications(s: TritonSession):
    """Without batching, a subscriber redraws on an intermediate state nobody asked for."""
    seen: list[str] = []
    s.subscribe(seen.append)

    with s.batch():
        s.view.nfft = 512
        s.view.overlap_pct = 50
        s.view.nfft = 1024
        assert seen == [], "nothing should fire until the batch closes"

    assert seen == ["view.nfft", "view.overlap_pct"], "repeats collapse, order kept"


def test_open_audio_notifies_once_per_field_not_per_intermediate(generated_dir: Path):
    sess = TritonSession()
    seen: list[str] = []
    sess.subscribe(seen.append)
    sess.open_audio(generated_dir / XWAV)
    assert "audio.source" in seen
    assert len(seen) == len(set(seen)), f"duplicate notifications: {seen}"


# ---------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "field,bad",
    [
        ("nfft", 0), ("nfft", -1), ("nfft", 2.5), ("nfft", True),
        ("tseg_sec", 0), ("tseg_sec", -1),
        ("overlap_pct", 100), ("overlap_pct", -1),
        ("freq0", -1), ("channel", 0), ("contrast", 0),
    ],
)
def test_invalid_view_values_are_refused_at_assignment(s: TritonSession, field, bad):
    with pytest.raises(ValidationError):
        setattr(s.view, field, bad)


def test_a_refused_assignment_leaves_the_old_value(s: TritonSession):
    s.view.nfft = 512
    with pytest.raises(ValidationError):
        s.view.nfft = -1
    assert s.view.nfft == 512


def test_overlap_of_100_percent_is_refused_with_a_reason(s: TritonSession):
    """100% overlap advances zero samples per frame -- an infinite loop, not a plot."""
    with pytest.raises(ValidationError, match="advance zero samples"):
        s.view.overlap_pct = 100


def test_seek_to_an_out_of_range_segment_is_refused(s: TritonSession):
    with pytest.raises(ValidationError, match="out of range"):
        s.seek_samples(99, 0)


# ------------------------------------------------------------- inspection surfaces


def test_as_params_has_the_shape_people_expect(s: TritonSession, generated_dir: Path):
    s.open_ltsa(generated_dir / LTSA)
    p = s.as_params()

    assert p["fs"] == s.audio.source.sample_rate
    assert p["nch"] == 1
    assert p["ftype"] == 2                    # xwav
    assert p["nfft"] == s.view.nfft
    assert p["raw"]["currentIndex"] == s.audio.segment + 1     # 1-based, as MATLAB
    assert p["xhd"]["NumOfRawFiles"] == len(s.audio.source.segments)
    assert p["ltsa"]["nf"] == s.ltsa.source.header.n_freq
    assert len(p["xhd"]["byte_loc"]) == len(s.audio.source.segments)


def test_as_params_is_a_copy_not_a_handle(s: TritonSession):
    """Mutating the inspection view must not reach the session.

    The complaint about PARAMS was never that it was inspectable -- it was that forty
    functions could mutate it. Handing out a live handle would reintroduce exactly that.
    """
    p = s.as_params()
    p["nfft"] = 99999
    p["tseg"]["sec"] = 99999
    assert s.view.nfft != 99999
    assert s.view.tseg_sec != 99999
    assert s.as_params()["nfft"] == s.view.nfft


def test_as_params_exposes_per_raw_file_dt(s: TritonSession):
    """Unlike rdxwavhd.m, which keeps only the last (OPEN_DECISIONS §1.3)."""
    p = s.as_params()
    assert len(p["xhd"]["dt"]) == len(s.audio.source.segments)


def test_snapshot_is_json_safe(s: TritonSession, generated_dir: Path):
    import json
    s.open_ltsa(generated_dir / LTSA)
    s.calibration.tf_freq = np.array([10.0, 100.0])
    snap = s.snapshot()
    json.dumps(snap)                          # must not raise
    assert snap["triton"] == triton.__version__
    assert snap["position"]["segment"] == 0
    assert snap["numpy"] == np.__version__


def test_walk_enumerates_every_field(s: TritonSession):
    paths = dict(s.walk())
    assert "view.nfft" in paths
    assert "audio.offset" in paths
    assert "ltsa.tseg_hr" in paths
    assert all("." in p for p in paths)


# ------------------------------------------------------------------ session scope


def test_current_session_is_one_seam(generated_dir: Path):
    triton.set_current_session(None)
    a = triton.current_session()
    assert triton.current_session() is a, "must be the same object each call"

    b = triton.open_session(audio=generated_dir / XWAV)
    assert triton.current_session() is b
    triton.set_current_session(None)


def test_two_sessions_are_independent(generated_dir: Path):
    """The thing PARAMS structurally cannot do: two datasets open at once."""
    a = TritonSession()
    b = TritonSession()
    a.open_audio(generated_dir / XWAV)
    b.open_audio(generated_dir / DUTY)

    a.view.nfft = 256
    b.view.nfft = 1024

    assert a.view.nfft == 256
    assert a.audio.path.name == XWAV
    assert b.audio.path.name == DUTY
    assert a.spectrogram_tile().db.shape != b.spectrogram_tile().db.shape


def test_plugins_get_their_own_namespace(s: TritonSession):
    """Replaces the REMORA global, where two plugins can clobber each other."""
    s.plugins["hello"] = {"keymap": {"h": "greet"}}
    s.plugins["other"] = {"keymap": {"o": "other"}}
    assert s.plugins["hello"]["keymap"] == {"h": "greet"}


# ------------------------------------------------------------------- clim holding


def test_clim_is_derived_once_then_held(s: TritonSession):
    """Issue #111's requirement: levels stay comparable while the analyst scrolls."""
    s.view.tseg_sec = 0.25
    first = s.spectrogram_tile().clim
    s.step(+1)
    assert s.spectrogram_tile().clim == first
    s.step(+1)
    assert s.spectrogram_tile().clim == first


def test_opening_a_file_resets_clim_but_stepping_does_not(generated_dir: Path):
    sess = TritonSession()
    sess.open_audio(generated_dir / XWAV)
    sess.view.tseg_sec = 0.25
    first = sess.spectrogram_tile().clim

    sess.step(+1)
    assert sess.spectrogram_tile().clim == first, "stepping must not move the range"

    sess.open_audio(generated_dir / DUTY)
    assert sess.view.clim is None, "an explicit open re-derives"


def test_manual_clim_survives_stepping(s: TritonSession):
    s.view.tseg_sec = 0.25
    s.spectrogram_tile()
    s.view.clim = (-20.0, 40.0)
    s.step(+1)
    assert s.spectrogram_tile().clim == (-20.0, 40.0)


# ------------------------------------------------- cursor readout and LTSA click-through


def test_time_at_is_piecewise_across_a_duty_cycle_gap(generated_dir: Path):
    """The thing a naive `window start + x` gets wrong, and by a lot.

    The window is read byte-contiguously across raw-file boundaries, so x is continuous
    while wall-clock time jumps. On the duty fixture the gap is 7.5 s, so a readout
    using the naive form reports a time seven and a half seconds early for any point
    past the boundary -- which is not a rounding error, it is the wrong recording.
    """
    sess = TritonSession()
    sess.open_audio(generated_dir / DUTY)
    src = sess.audio.source
    sess.view.tseg_sec = 0.5
    sess.seek_samples(0, int(2.25 * src.sample_rate))

    # Before the boundary the two agree...
    naive = sess.audio.time + np.timedelta64(100_000_000, "ns")
    got, seg = sess.time_at(0.1)
    assert got == naive and seg == 0

    # ...and after it they do not.
    got, seg = sess.time_at(0.4)
    naive = sess.audio.time + np.timedelta64(400_000_000, "ns")
    assert seg == 1, "past the boundary is the next raw file"
    assert got != naive
    assert got == src.segments[1].start + np.timedelta64(150_000_000, "ns")


def test_probe_reports_the_value_under_the_cursor(generated_dir: Path):
    sess = TritonSession()
    sess.open_audio(generated_dir / XWAV)
    sess.view.tseg_sec = 1.0
    sess.view.nfft = 256
    frame = sess.frame()

    sg = sess.probe(frame, "specgram", 0.5, 2000.0)
    assert sg.panel == "specgram"
    assert sg.frequency is not None and abs(sg.frequency - 2000.0) < 100
    # the reported level must be the array value at the reported bin, not interpolated
    j = int(np.argmin(np.abs(frame.spectrogram.f - sg.frequency)))
    i = int(np.argmin(np.abs(frame.spectrogram.t - 0.5)))
    assert sg.value == frame.spectrogram.db[j, i]

    ts = sess.probe(frame, "timeseries", 0.5, 0.0)
    assert ts.value_label == "Counts"
    assert ts.value == frame.samples[int(0.5 * frame.fs), 0]


def test_probe_does_not_run_off_the_end(generated_dir: Path):
    """Hover fires for every pixel, including the last one."""
    sess = TritonSession()
    sess.open_audio(generated_dir / XWAV)
    sess.view.tseg_sec = 1.0
    frame = sess.frame()
    for x in (0.0, 0.99999, 1.0, 1.5):
        for panel in ("specgram", "timeseries", "spectra"):
            sess.probe(frame, panel, x, 1000.0)      # must not raise


def test_ltsa_locate_walks_entries(generated_dir: Path):
    """getIndexBin.m's walk: x maps to a bin, bins are consumed entry by entry."""
    sess = TritonSession()
    sess.open_ltsa(generated_dir / LTSA)
    src = sess.ltsa.source
    tave, nave = src.header.tave, src.header.n_ave[0]
    assert nave == 3 and tave == 1.0, "fixture assumption"

    # 3 bins of 1 s per entry, so 3.6 s into the plot is entry 1, bin 0.
    idx, b, t = src.locate(src.start_time, 3.6 / 3600, 1 / 60)
    assert (idx, b) == (1, 0)
    assert t == src.header.entries[1].start + np.timedelta64(500, "ms"), "bin centre"

    idx, b, _ = src.locate(src.start_time, 7.2 / 3600, 1 / 60)
    assert (idx, b) == (2, 1)


def test_ltsa_locate_clamps_past_the_end(generated_dir: Path):
    """A click a pixel beyond the data is an ordinary thing for a user to do."""
    sess = TritonSession()
    sess.open_ltsa(generated_dir / LTSA)
    src = sess.ltsa.source
    idx, b, _ = src.locate(src.start_time, 999.0, 1 / 60)
    assert idx == len(src.header.entries) - 1
    assert b == src.header.entries[idx].n_ave - 1


def test_open_from_ltsa_opens_the_named_file_and_seeks(generated_dir: Path):
    """pickxwav.m's job. The LTSA names its own source files, so nothing is searched."""
    sess = TritonSession()
    sess.open_ltsa(generated_dir / LTSA)
    sess.ltsa.tseg_hr = 1 / 60
    assert not sess.audio.is_open

    t = sess.open_from_ltsa(3.6 / 3600)

    assert sess.audio.is_open
    assert sess.audio.path.name == XWAV
    assert sess.audio.time == t
    assert sess.view.show_specgram, "landing on a blank window reads as a failed click"
    assert sess.frame().spectrogram.db.size > 0, "and the position must be readable"


def test_open_from_ltsa_names_the_file_it_could_not_find(tmp_path: Path,
                                                         generated_dir: Path):
    """The error has to name the file, because the caller offers a file dialog next."""
    import shutil
    moved = tmp_path / LTSA
    shutil.copy(generated_dir / LTSA, moved)          # ...without the .x.wav beside it

    sess = TritonSession()
    sess.open_ltsa(moved)
    with pytest.raises(FileNotFoundError, match=XWAV.replace(".", r"\.")):
        sess.open_from_ltsa(0.0)
