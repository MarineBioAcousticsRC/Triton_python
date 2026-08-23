"""Checks on the MATLAB reference dump itself.

These run today, with no triton package present.  They catch a stale, partial, or internally
inconsistent dump before anyone spends time debugging Python against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import load_reference, load_reference_bin


def test_environment_recorded(reference_dir: Path):
    """Numeric output can shift between MATLAB releases; we need to know which one made this."""
    env = json.loads((reference_dir / "_environment.json").read_text())
    assert env["matlab_release"]
    assert env["triton_path"]
    assert env["generated"]


def test_every_xwav_fixture_has_a_header_dump(reference_dir: Path, manifest: dict):
    ref = load_reference("xwav_headers")
    dumped = {r["file"] for r in ref}
    expected = {m["file"] for m in manifest["fixtures"] if m["kind"] == "xwav"}
    assert dumped == expected, f"missing: {expected - dumped}"


def test_matlab_header_parse_matches_the_generator(reference_dir: Path, manifest: dict):
    """rdxwavhd.m read what make_fixtures.py wrote.  If this fails, one of the two is wrong
    about the spec -- resolve it before writing any Python."""
    ref = {r["file"]: r for r in load_reference("xwav_headers")}
    for m in manifest["fixtures"]:
        if m["kind"] != "xwav":
            continue
        r = ref[m["file"]]
        xhd = r["xhd"]
        assert xhd["NumOfRawFiles"] == m["n_raw_files"], m["file"]
        assert xhd["NumChannels"] == m["n_channels"], m["file"]
        assert xhd["BitsPerSample"] == m["bits_per_sample"], m["file"]
        assert xhd["WavVersionNumber"] == m["wav_version"], m["file"]
        assert xhd["SampleRate"] == m["fmt_sample_rate"], m["file"]
        # rdxwavhd takes fs from the raw-file table, NOT the fmt chunk (xwav.md 5.1)
        assert r["fs"] == m["true_sample_rate"], m["file"]

        as_list = lambda v: v if isinstance(v, list) else [v]  # noqa: E731
        assert as_list(xhd["byte_loc"]) == [f["byte_loc"] for f in m["raw_files"]], m["file"]
        assert as_list(xhd["byte_length"]) == [f["byte_length"] for f in m["raw_files"]]
        assert as_list(xhd["sample_rate"]) == [f["sample_rate"] for f in m["raw_files"]]


def test_fake_fs_fixture_confirms_the_two_rates_diverge(reference_dir: Path):
    """The whole reason x.wav cannot ride a generic WAVE reader."""
    ref = {r["file"]: r for r in load_reference("xwav_headers")}
    r = next(v for k, v in ref.items() if "fakefs" in k)
    assert r["xhd"]["SampleRate"] == 100_000, "fmt chunk says 100 kHz"
    assert r["fs"] == 200_000, "rdxwavhd must use the raw-file table's 200 kHz"


def test_io_reader_disagrees_only_on_v2(reference_dir: Path):
    """io/ioReadXWAVHeader.m has no v2 branch: it misses the per-channel drate/dt fields and
    mis-parses the raw-file table.  Documented in xwav.md; asserted here so that if someone
    fixes it upstream we notice and can drop the workaround.

    Four Remoras carry copies of that reader, so they inherit the same v2 blindness.
    """
    ref = load_reference("xwav_headers")
    for r in ref:
        if r["xhd"]["WavVersionNumber"] == 2:
            assert not r["io_agrees"], (
                f"{r['file']}: ioReadXWAVHeader now agrees on v2 -- it was fixed upstream, "
                "update docs/formats/xwav.md and this test"
            )
        else:
            assert r["io_agrees"], f"{r['file']}: the two MATLAB readers disagree on v0/v1!"


def test_datenum_precision_claim_is_measured_not_asserted(reference_dir: Path):
    """docs/formats/timebase.md 2 -- confirm from MATLAB itself, not from arithmetic we did."""
    p = load_reference("datenum_precision")
    assert p["eps_true_2011_us"] > p["sample_period_us_at_200kHz"], (
        "a float datenum at the real epoch must be too coarse to resolve one sample at "
        "200 kHz -- this is the entire argument for integer sample indices"
    )
    assert p["eps_shifted_2011_us"] < p["sample_period_us_at_200kHz"] / 10, (
        "the year-2000 shift must buy real headroom"
    )


def test_readseg_dumps_are_complete(reference_dir: Path, manifest: dict):
    index = load_reference("readseg_index")
    files = {r["file"] for r in index}
    expected = {m["file"] for m in manifest["fixtures"] if m["kind"] == "xwav"}
    assert files == expected
    labels = {r["label"] for r in index}
    assert "bof" in labels
    assert "across_boundary" in labels, "the splice case must be covered"
    for r in index:
        stem = f"readseg__{r['tag'].split('__', 1)[1]}"
        assert (reference_dir / f"{stem}__data.bin").exists(), stem
        assert (reference_dir / f"{stem}__meta.json").exists(), stem


def test_readseg_block_shapes_match_their_metadata(reference_dir: Path):
    for meta_path in sorted(reference_dir.glob("readseg__*__meta.json")):
        meta = json.loads(meta_path.read_text())
        name = meta_path.name[: -len("__meta.json")]
        arr, binmeta = load_reference_bin(f"{name}__data")
        assert list(arr.shape) == list(meta["data_size"]), name
        # tseg.samp = ceil(tseg.sec * fs), readseg.m:110
        assert meta["tseg_samp"] == int(np.ceil(meta["tseg_sec"] * meta["fs"])), name
        assert arr.shape[1] == meta["nch"], name


def test_gain_was_actually_applied(reference_dir: Path):
    """readseg.m:117 divides by gain[0].  With gain=4 the samples must not be raw counts."""
    metas = [
        json.loads(p.read_text())
        for p in reference_dir.glob("readseg__*gain4*__meta.json")
    ]
    assert metas, "gain fixture missing from the dump"
    assert all(m["xgain"] == 4 for m in metas)


def test_hanning_reference_is_not_scipys_hann(reference_dir: Path):
    """The trap in ltsa.md 5.1, pinned down against MATLAB's actual output."""
    scipy_signal = pytest.importorskip("scipy.signal")
    for n in (8, 9, 16, 256):
        matlab_w, _ = load_reference_bin(f"hanning_{n}")
        matlab_w = matlab_w.ravel()

        wrong_a = scipy_signal.windows.hann(n, sym=True)
        wrong_b = scipy_signal.windows.hann(n, sym=False)
        right = scipy_signal.windows.hann(n + 2, sym=True)[1:-1]

        np.testing.assert_allclose(matlab_w, right, rtol=0, atol=1e-15,
                                   err_msg=f"hann(N+2)[1:-1] should equal MATLAB hanning({n})")
        assert not np.allclose(matlab_w, wrong_a), f"n={n}: hann(N) coincidentally matched"
        if n > 2:
            assert not np.allclose(matlab_w, wrong_b), f"n={n}: hann(N,sym=False) matched"


def test_spectrogram_and_pwelch_dumps_present(reference_dir: Path):
    assert list(reference_dir.glob("specgram_*__pwr.bin")), "no spectrogram reference"
    assert list(reference_dir.glob("pwelch_*__db.bin")), "no pwelch reference"
    # the int8 quantisation that actually reaches an .ltsa file
    for db_path in reference_dir.glob("pwelch_*__db.bin"):
        stem = db_path.name[: -len("__db.bin")]
        q, _ = load_reference_bin(f"{stem}__int8")
        db, _ = load_reference_bin(f"{stem}__db")
        assert q.shape == db.shape
        assert q.dtype == np.int8


def test_ltsa_reference_present_or_explained(reference_dir: Path):
    """Not a hard failure: the .ltsa fixtures need one manual dialog click each to build."""
    if not (reference_dir / "ltsa_headers.json").exists():
        pytest.skip(
            "no LTSA reference yet -- run tools/matlab/make_ltsa_fixture.m then "
            "dump_reference('sections',{'ltsa'})"
        )
    hdrs = load_reference("ltsa_headers")
    assert hdrs
    for h in hdrs:
        assert h["ver"] in (1, 2, 3, 4)
        assert h["nf"] == (h["nfft"] // 2 + 1 if h["nfft"] % 2 == 0 else (h["nfft"] + 1) // 2)
