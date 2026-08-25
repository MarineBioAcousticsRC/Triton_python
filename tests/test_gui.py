"""Phase 3: the viewer.

GUI code has a reputation for being untestable, which is mostly a reputation earned by
GUIs that keep their state in their widgets. This one keeps it in the session, so the
interesting things are all checkable without a human: that the bridge classifies and
coalesces changes correctly, that controls and session stay in step in both directions,
that a display-only change does not trigger a recompute, and that the panels shown match
the flags.

Runs on Qt's ``offscreen`` platform. Note that offscreen has **no fonts** on a bare
Windows box, so anything rendered comes out as boxes -- rendered *text* therefore cannot
be asserted on, but widget properties can, and those are what carry the meaning.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="GUI extra not installed")
pytest.importorskip("pyqtgraph", reason="GUI extra not installed")

from PySide6.QtWidgets import QApplication  # noqa: E402

from triton.gui.bridge import Dirty, SessionBridge, classify  # noqa: E402
from triton.gui.panels import PANEL_ORDER, colormap_lut  # noqa: E402
from triton.gui.window import MainWindow  # noqa: E402
from triton.session import TritonSession  # noqa: E402

XWAV = "xwav_v1_cont_1ch_16b_10k.x.wav"
LTSA = "xwav_v1_cont_1ch_16b_10k__tave1_df100.ltsa"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, generated_dir: Path):
    w = MainWindow()
    # A modal dialog with nobody to dismiss it blocks the test run forever, and because
    # it blocks inside a C call the traceback points at whatever opened it rather than
    # at the dialog. This is not paranoia -- it happened.
    w.modal_errors = False
    w.resize(1000, 700)
    w.show()
    w.open_audio(generated_dir / XWAV)
    app.processEvents()
    yield w
    w.close()


# ------------------------------------------------------------------- the seam holds


def test_gui_package_is_the_only_qt_importer():
    """`triton` and `triton.session` must stay importable with no Qt installed.

    `tests/test_session.py` asserts the session's own imports; this asserts that
    importing the package root does not drag the GUI in behind it, which is the other
    way the property gets lost.
    """
    import ast

    import triton
    tree = ast.parse(Path(triton.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        if isinstance(node, ast.ImportFrom) and node.level:
            names |= {a.name for a in node.names}
    assert "gui" not in names, "triton/__init__.py must not import the gui package"


# --------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "path,expect",
    [
        ("view.brightness", Dirty.DISPLAY),
        ("view.contrast", Dirty.DISPLAY),
        ("view.colormap", Dirty.DISPLAY),
        ("view.nfft", Dirty.DATA),
        ("audio.offset", Dirty.DATA),
        ("view.step_sec", Dirty.NOTHING),
        ("config.last_audio_dir", Dirty.NOTHING),
    ],
)
def test_classification(path, expect):
    assert classify(path) == expect


def test_unknown_paths_default_to_the_expensive_answer():
    """A field nobody classified should repaint too often, not too rarely.

    Too often is a performance bug someone notices and fixes; too rarely looks like the
    data being wrong, which is much worse and much harder to attribute.
    """
    assert classify("view.some_field_added_later") is Dirty.DATA
    assert classify("brand.new.group") is Dirty.DATA


def test_ltsa_paths_are_classified_as_ltsa():
    assert classify("ltsa.tseg_hr") is Dirty.LTSA
    assert classify("ltsa.brightness") is Dirty.LTSA


# ---------------------------------------------------------------------- coalescing


def test_notifications_coalesce_into_one_repaint(app, generated_dir: Path):
    session = TritonSession()
    session.open_audio(generated_dir / XWAV)
    bridge = SessionBridge(session)
    seen: list[Dirty] = []
    bridge.repaint_needed.connect(seen.append)

    for n in (256, 512, 1024, 2048):
        session.view.nfft = n
    assert seen == [], "nothing should fire before the event loop turns"

    app.processEvents()
    assert len(seen) == 1, f"four changes should give one repaint, got {len(seen)}"
    assert Dirty.DATA in seen[0]
    bridge.detach()


def test_coalesced_flags_are_the_union(app, generated_dir: Path):
    session = TritonSession()
    session.open_audio(generated_dir / XWAV)
    bridge = SessionBridge(session)
    seen: list[Dirty] = []
    bridge.repaint_needed.connect(seen.append)

    session.view.nfft = 512          # DATA
    session.view.brightness = 3.0    # DISPLAY
    app.processEvents()

    assert len(seen) == 1
    assert Dirty.DATA in seen[0] and Dirty.DISPLAY in seen[0]
    bridge.detach()


def test_detach_stops_notifications(app, generated_dir: Path):
    session = TritonSession()
    session.open_audio(generated_dir / XWAV)
    bridge = SessionBridge(session)
    seen: list[Dirty] = []
    bridge.repaint_needed.connect(seen.append)
    bridge.detach()
    session.view.nfft = 512
    app.processEvents()
    assert seen == []
    bridge.detach()                  # idempotent


# ------------------------------------------------------- display vs data, the fast path


def test_a_display_change_does_not_recompute_the_frame(win, app, monkeypatch):
    """The branch that decides whether a brightness slider is usable.

    Measured on a real file: a 30 s window at 200 kHz costs ~280 ms to compute and
    ~8 ms to re-map. If this test fails, the slider is 35x slower than it needs to be
    and nothing else will tell you.
    """
    calls = {"frame": 0, "remap": 0}
    real_frame = win.session.frame
    real_remap = win.session.remap

    def counted_frame():
        calls["frame"] += 1
        return real_frame()

    def counted_remap(f):
        calls["remap"] += 1
        return real_remap(f)

    monkeypatch.setattr(win.session, "frame", counted_frame)
    monkeypatch.setattr(win.session, "remap", counted_remap)

    win.session.view.brightness = -6.0
    app.processEvents()
    assert calls == {"frame": 0, "remap": 1}, "a display change must not re-read or re-FFT"

    win.session.view.nfft = 512
    app.processEvents()
    assert calls["frame"] == 1, "a data change must recompute"


def test_remap_agrees_with_a_full_recompute(win):
    """The fast path must not be a different answer, only a cheaper one."""
    frame = win.session.frame()
    win.session.view.contrast = 180.0
    win.session.view.brightness = -4.0
    win.session.view.clim = None

    remapped = win.session.remap(frame)
    win.session.view.clim = None
    recomputed = win.session.frame()

    np.testing.assert_allclose(remapped.spectrogram.db, recomputed.spectrogram.db,
                               rtol=0, atol=1e-12)


# ------------------------------------------------------------------ controls <-> session


def test_control_edit_reaches_the_session(win, app):
    spin = win.controls._bindings["view.nfft"][0]
    spin.set(512)
    spin._widget_changed()
    assert win.session.view.nfft == 512


def test_session_change_reaches_the_control(win, app):
    win.session.view.nfft = 2048
    app.processEvents()
    assert win.controls._bindings["view.nfft"][0].get() == 2048


def test_a_refused_value_snaps_the_widget_back(win, app):
    """Validation lives in the session; the widget defers to it and re-syncs.

    Exercised through a stand-in widget rather than the real spin box, because a
    `QSpinBox` clamps to its own range and so *cannot* hand the session an invalid
    value -- the ranges mirror the validators by design. That makes the real path
    hard to reach and the mechanism still worth testing, since it is easy to get
    wrong: `refresh()` is guarded by the same re-entrancy flag `_widget_changed`
    sets, so calling it before clearing that flag silently does nothing and leaves
    the widget showing a value the session rejected.
    """
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget

    from triton.gui.controls import _Binding

    class Stubborn(QWidget):
        edited = Signal()

        def __init__(self):
            super().__init__()
            self.value = 1024

    w = Stubborn()
    binding = _Binding(win.session, "view.nfft", w,
                       lambda: w.value, lambda v: setattr(w, "value", v), w.edited)
    win.session.view.nfft = 1024
    binding.refresh()
    assert w.value == 1024

    w.value = -5                          # something no spin box would offer
    w.edited.emit()
    assert win.session.view.nfft == 1024, "the session must have refused it"
    assert w.value == 1024, "and the widget must have snapped back"


def test_no_feedback_loop_between_control_and_session(win, app):
    """One edit must produce one session write, not an oscillation."""
    writes: list[str] = []
    win.session.subscribe(writes.append, prefix="view")
    binding = win.controls._bindings["view.contrast"][0]
    binding.set(150.0)
    binding._widget_changed()
    app.processEvents()
    assert writes.count("view.contrast") == 1, writes


# ----------------------------------------------------------------------- panel layout


def test_panels_are_stacked_in_matlab_order(win):
    """plot_triton.m:44-70 stacks LTSA, spectrogram, time series, spectra -- always in
    that order, whatever order they were switched on in."""
    assert PANEL_ORDER == ("ltsa", "specgram", "timeseries", "spectra")
    order = [win.splitter.widget(i) for i in range(win.splitter.count())]
    assert order == [win.panels[n] for n in PANEL_ORDER]


def test_only_enabled_panels_are_visible(win, app):
    with win.session.batch():
        win.session.view.show_specgram = True
        win.session.view.show_timeseries = True
        win.session.view.show_spectra = False
    app.processEvents()
    assert win.panels["specgram"].isVisible()
    assert win.panels["timeseries"].isVisible()
    assert not win.panels["spectra"].isVisible()


def test_the_ltsa_panel_stays_hidden_until_an_ltsa_is_open(win, app, generated_dir):
    win.session.view.show_ltsa = True
    app.processEvents()
    assert not win.panels["ltsa"].isVisible(), "no LTSA open, so nothing to show"

    win.session.open_ltsa(generated_dir / LTSA)
    app.processEvents()
    assert win.panels["ltsa"].isVisible()


def test_no_panels_says_so_rather_than_going_blank(win, app):
    with win.session.batch():
        for f in ("show_ltsa", "show_specgram", "show_timeseries", "show_spectra"):
            setattr(win.session.view, f, False)
    app.processEvents()
    assert not win.splitter.isVisible()
    assert win.empty_label.isVisible()
    assert "No plot type selected" in win.empty_label.text()


# --------------------------------------------------------------------------- rendering


def test_a_frame_actually_reaches_the_panels(win, app):
    with win.session.batch():
        win.session.view.show_specgram = True
        win.session.view.show_timeseries = True
        win.session.view.show_spectra = True
        win.session.view.nfft = 256
    app.processEvents()
    win.bridge.force()
    app.processEvents()

    assert win._frame is not None
    img = win.panels["specgram"].image.image
    assert img is not None and img.shape == win._frame.spectrogram.db.shape
    xs, _ = win.panels["timeseries"].curve.getData()
    assert xs is not None and len(xs) > 0


def test_raw_file_boundaries_are_drawn_once(win, app):
    """Red dashed delimiters, and only the ones inside the window.

    readseg.m:129-131 extrapolates one boundary past the right edge, so the list
    routinely contains one that does not belong on the plot.
    """
    win.session.view.tseg_sec = 0.5
    win.session.seek_samples(0, int(2.25 * win.session.audio.source.sample_rate))
    app.processEvents()
    win.bridge.force()
    app.processEvents()

    offsets = [b.offset_sec for b in win._frame.spectrogram.boundaries]
    assert len(offsets) > 1, "this window should span a boundary and extrapolate one"
    inside = [x for x in offsets if 0 < x < 0.5]
    assert len(win.panels["specgram"]._boundary_lines) == len(inside)


def test_delimiters_can_be_turned_off(win, app):
    win.session.view.tseg_sec = 0.5
    win.session.seek_samples(0, int(2.25 * win.session.audio.source.sample_rate))
    win.session.view.show_delimiters = False
    app.processEvents()
    win.bridge.force()
    app.processEvents()
    assert win.panels["specgram"]._boundary_lines == []


# ----------------------------------------------------------------------- colour maps


def test_colormap_lut_shape_and_ends():
    lut = colormap_lut("jet")
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    # jet runs dark blue to dark red
    assert lut[0, 2] > lut[0, 0], "low end should be blue-dominant"
    assert lut[-1, 0] > lut[-1, 2], "high end should be red-dominant"


def test_unknown_colormap_falls_back_rather_than_raising(win):
    """A colour map name from a settings file should never take the window down."""
    np.testing.assert_array_equal(colormap_lut("no such map"), colormap_lut("jet"))


def test_grey_is_monotonic():
    lut = colormap_lut("grey")
    assert np.all(np.diff(lut[:, 0].astype(int)) >= 0)


# ------------------------------------------------------------------------- errors


def test_an_unopenable_file_reports_and_does_not_raise(win, generated_dir):
    """The reader's messages are written for people; show them, do not crash."""
    win.open_audio(generated_dir / LTSA)          # an LTSA is not audio
    assert win.statusBar().currentMessage() != ""


def test_an_unreadable_window_keeps_the_last_good_frame(win, app):
    """Too few samples for the chosen nfft is a legitimate state to pass through."""
    win.bridge.force()
    app.processEvents()
    good = win._frame
    assert good is not None

    win.session.view.tseg_sec = 0.001         # far fewer samples than nfft
    win.session.view.nfft = 4096
    app.processEvents()

    assert win._frame is good, "the last good frame should still be on screen"
    assert win.statusBar().currentMessage() != ""
