"""Phase 1 parity tests: the Python implementation vs MATLAB ground truth.

Every test here skips until the corresponding piece of `triton` exists.  They are written now,
before the implementation, so that Phase 1 has an unambiguous definition of done: this file
goes green.

Do not weaken a tolerance to make one of these pass.  The tolerances are deliberate:

    headers, byte offsets, sample data   exact
    derived timestamps                   exact to the nanosecond
    spectrogram / pwelch dB              1e-9 relative
    int8-quantised LTSA values           exact

A fractional-dB disagreement is a real bug -- see docs/formats/ltsa.md 5.1 for the window
definition that causes exactly that symptom.
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import pytest
from conftest import load_reference, load_reference_bin

triton_io = pytest.importorskip("triton.io", reason="Phase 1 not implemented yet")


# ---------------------------------------------------------------------------- headers


def test_read_xwav_header_matches_matlab(generated_dir: Path, reference_dir: Path):
    for r in load_reference("xwav_headers"):
        hdr = triton_io.read_xwav_header(generated_dir / r["file"])
        xhd = r["xhd"]

        assert hdr.n_channels == xhd["NumChannels"], r["file"]
        assert hdr.bits_per_sample == xhd["BitsPerSample"], r["file"]
        assert hdr.wav_version == xhd["WavVersionNumber"], r["file"]
        assert hdr.fmt_sample_rate == xhd["SampleRate"], r["file"]
        assert hdr.n_raw_files == xhd["NumOfRawFiles"], r["file"]

        # the rate that matters: raw-file table, not fmt chunk (xwav.md 5.1)
        assert hdr.sample_rate == r["fs"], r["file"]

        as_list = lambda v: v if isinstance(v, list) else [v]  # noqa: E731
        assert [s.byte_loc for s in hdr.raw_files] == as_list(xhd["byte_loc"]), r["file"]
        assert [s.byte_length for s in hdr.raw_files] == as_list(xhd["byte_length"])
        assert [s.sample_rate for s in hdr.raw_files] == as_list(xhd["sample_rate"])
        assert [s.write_length for s in hdr.raw_files] == as_list(xhd["write_length"])


def test_v2_header_fields_are_read(generated_dir: Path, reference_dir: Path):
    """Unlike io/ioReadXWAVHeader.m, the Python reader must handle v2 (xwav.md 6.8).

    It must also expose per-raw-file `dt`, which rdxwavhd.m loses by overwriting (xwav.md 6.7),
    so this checks more than MATLAB can.
    """
    r = next(x for x in load_reference("xwav_headers") if x["xhd"]["WavVersionNumber"] == 2)
    hdr = triton_io.read_xwav_header(generated_dir / r["file"])
    assert hdr.wav_version == 2
    assert hdr.drate is not None and len(hdr.drate) == hdr.n_channels
    for raw in hdr.raw_files:
        assert raw.dt is not None and len(raw.dt) == hdr.n_channels


# ---------------------------------------------------------------------------- timing


def test_segment_start_times_match_matlab_datenums(generated_dir: Path, reference_dir: Path):
    """MATLAB's shifted datenums, converted at the boundary, must land on the same instant."""
    for r in load_reference("xwav_headers"):
        dnums, _ = load_reference_bin(f"dnums__{_safe(r['file'])}")
        hdr = triton_io.read_xwav_header(generated_dir / r["file"])
        for i, raw in enumerate(hdr.raw_files):
            expect = triton_io.from_triton_datenum(dnums[i, 0])
            # Tolerance is one quantum of the float datenum being compared against,
            # not a fixed nanosecond count. `raw.start` is built from the header's
            # integer fields and is exact; `expect` comes from a MATLAB double,
            # whose spacing at a shifted datenum is about 39 ns (timebase.md 2).
            # Requiring closer agreement than the reference can express would be
            # asking our exact value to reproduce someone else's rounding. A real
            # error would be microseconds or worse, so this still catches one.
            quantum_ns = np.spacing(float(dnums[i, 0])) * 86400e9
            drift_ns = abs(int((raw.start - expect) / np.timedelta64(1, "ns")))
            assert drift_ns <= quantum_ns, (
                f"{r['file']} raw {i}: {drift_ns} ns drift, "
                f"float quantum {quantum_ns:.1f} ns"
            )


def test_triton_datenum_round_trip(reference_dir: Path):
    dnums, _ = load_reference_bin("timestr_dnums")
    for d in dnums.ravel():
        t = triton_io.from_triton_datenum(float(d))
        back = triton_io.to_triton_datenum(t)
        assert abs(back - float(d)) < 1e-9


def test_filename_timestamp_patterns(reference_dir: Path):
    """All six wavname2dnum.m patterns, in the order MATLAB tries them (timebase.md 7)."""
    names = load_reference("wavname2dnum")["names"]
    dnums, _ = load_reference_bin("wavname2dnum_dnums")
    for name, d in zip(names, dnums.ravel(), strict=True):
        got = triton_io.parse_filename_time(name)
        if np.isnan(d):
            assert got is None, name
        else:
            expect = triton_io.from_matlab_datenum(float(d))
            # Same reasoning as the raw-file starts above, but the quantum is far
            # larger: these are *true* datenums near 734533, where a double resolves
            # only about 10 microseconds. Our value is built from the filename's
            # integer fields and is the more accurate of the two.
            quantum_ns = np.spacing(float(d)) * 86400e9
            drift_ns = abs(int((got - expect) / np.timedelta64(1, "ns")))
            assert drift_ns <= quantum_ns, (
                f"{name}: {drift_ns} ns drift, float quantum {quantum_ns:.1f} ns"
            )


# ---------------------------------------------------------------------------- readseg


def _matlab_skip(meta: dict) -> int:
    """The sample offset MATLAB actually seeked to, from its own float datenums.

    `readseg.m:111` is `floor((plot.dnum - dnumStart(index)) * 86400 * fs)`, and both
    operands are the doubles MATLAB held, dumped verbatim.  Recomputing it here rather
    than trusting our own exact arithmetic is the point: it makes the sample comparison
    below exact, with no tolerance, and isolates the one-sample float question into the
    dedicated test that follows.
    """
    plot_dnum = struct.unpack(">d", bytes.fromhex(meta["plot_dnum_hex"]))[0]
    stem = re.sub(r"\W", "_", meta["file"])
    dnums, _ = load_reference_bin(f"dnums__{stem}")
    d_start = float(dnums[meta["currentIndex"] - 1, 0])
    return math.floor((plot_dnum - d_start) * 86400 * meta["fs"])


def test_read_segment_matches_matlab_byte_for_byte(generated_dir: Path, reference_dir: Path):
    """The core contract: same samples out, exactly.

    Addressed by *sample*, not by time.  `read_samples` is given the offset MATLAB
    seeked to, so any disagreement here is a real defect -- wrong byte_loc, wrong dtype,
    wrong channel interleave, wrong gain, wrong splice across the raw-file boundary.
    Whether a *time* maps to that same sample is tested separately below, because that
    part cannot be exact and should not be hidden inside a tolerance here.
    """
    seen = 0
    for meta_path in sorted(reference_dir.glob("readseg__*__meta.json")):
        meta = json.loads(meta_path.read_text())
        expected, _ = load_reference_bin(meta_path.name[: -len("__meta.json")] + "__data")

        src = triton_io.open_audio(generated_dir / meta["file"])
        got = src.read_samples(
            meta["currentIndex"] - 1, _matlab_skip(meta), meta["tseg_samp"]
        )

        assert got.shape == tuple(meta["data_size"]), meta["file"]
        np.testing.assert_array_equal(got, expected, err_msg=f"{meta['file']} {meta['label']}")
        seen += 1
    assert seen == 27, f"expected 27 readseg cases, found {seen}"


def test_time_addressed_read_is_within_one_sample_of_matlab(
    generated_dir: Path, reference_dir: Path
):
    """`read_at` may differ from MATLAB by at most one sample, and only at boundaries.

    Not a weakened tolerance -- a documented limit with a proof obligation attached.
    MATLAB derives its sample offset from float datenums carrying about 40 ns of
    representation error (timebase.md 2).  Forty nanoseconds is a small fraction of a
    sample at any rate Triton handles, so it can only change `floor`'s answer when the
    requested time sits within that distance of a sample boundary.

    So this asserts both halves: never more than one sample out, and *never out at all*
    unless the exact product really is that close to an integer.  A genuine indexing
    bug violates the second half immediately.

    See `AudioSource.skip_for` for why matching MATLAB bit-for-bit is not merely hard
    but ill-defined: `plot.dnum` accumulates along the analyst's path through the file,
    so the same time reached two ways can read different samples.
    """
    off_by_one = 0
    for meta_path in sorted(reference_dir.glob("readseg__*__meta.json")):
        meta = json.loads(meta_path.read_text())
        src = triton_io.open_audio(generated_dir / meta["file"])
        index = meta["currentIndex"] - 1

        t0 = src.segments[index].start + np.timedelta64(
            int(round(meta["requested_offset_sec"] * 1e9)), "ns"
        )
        delta = src.skip_for(index, t0) - _matlab_skip(meta)

        assert abs(delta) <= 1, (
            f"{meta['file']} {meta['label']}: {delta} samples from MATLAB, which is "
            f"more than float datenum noise can explain"
        )
        if delta:
            off_by_one += 1
            # The exact product must be within a datenum quantum of an integer, or the
            # discrepancy has another cause and this test should fail.
            exact = meta["requested_offset_sec"] * meta["fs"]
            plot_dnum = struct.unpack(">d", bytes.fromhex(meta["plot_dnum_hex"]))[0]
            quantum_samples = np.spacing(plot_dnum) * 86400 * meta["fs"]
            assert abs(exact - round(exact)) <= quantum_samples, (
                f"{meta['file']} {meta['label']}: off by {delta} sample(s) but the "
                f"requested time is {abs(exact - round(exact)):.3g} samples from a "
                f"boundary, more than the {quantum_samples:.3g} float noise allows"
            )
    assert off_by_one == 7, (
        f"{off_by_one} cases differ by a sample; 7 expected.  A change here means the "
        f"time-to-sample mapping moved -- investigate before updating this number."
    )


def test_boundaries_match_matlab_delimiters(generated_dir: Path, reference_dir: Path):
    """`PARAMS.raw.delimit_time`: where Triton draws the red dashed lines.

    Tolerance is one quantum of the float datenums MATLAB subtracted to get these, not
    a fixed second count.  `delimit_time` is `(dnumEnd - plot.dnum) * 86400`
    (readseg.m:121), a difference of two doubles near 4048 where an ulp is 2**-41 days
    = 39.3 ns.  Both operands carry up to half an ulp of error, so the difference
    carries up to one; our exact integer-nanosecond value is the more accurate of the
    two and cannot be asked to reproduce MATLAB's rounding.  A real error -- wrong
    ByteRate, wrong raw file, dropping the -2 sample -- moves these by milliseconds.
    """
    for meta_path in sorted(reference_dir.glob("readseg__*__meta.json")):
        meta = json.loads(meta_path.read_text())
        src = triton_io.open_audio(generated_dir / meta["file"])
        index = meta["currentIndex"] - 1
        t0 = src.segments[index].start + np.timedelta64(
            int(round(meta["requested_offset_sec"] * 1e9)), "ns"
        )
        _, bounds = src.read_at(t0, meta["tseg_sec"], splice_gaps=True)

        expect = np.atleast_1d(np.asarray(meta["delimit_time"], dtype=float))
        if not expect.size or np.isnan(expect).all():
            continue
        got = np.array([b.offset_sec for b in bounds])
        assert got.size == expect.size, (
            f"{meta['file']} {meta['label']}: {got.size} delimiters, expected {expect.size}"
        )
        plot_dnum = struct.unpack(">d", bytes.fromhex(meta["plot_dnum_hex"]))[0]
        atol = np.spacing(plot_dnum) * 86400.0
        np.testing.assert_allclose(
            got, expect, rtol=0, atol=atol,
            err_msg=f"{meta['file']} {meta['label']} (tol {atol * 1e9:.1f} ns)",
        )


def test_current_raw_file_index_matches(generated_dir: Path, reference_dir: Path):
    """check_time.m's three-way triage (xwav.md 5.3) must produce the same answer.
    MATLAB indices are 1-based; ours are 0-based."""
    for r in load_reference("readseg_index"):
        meta = json.loads((Path(reference_dir) / f"{r['tag']}__meta.json").read_text())
        src = triton_io.open_audio(generated_dir / Path(meta["file"]).name)
        t0 = src.segments[0].start + np.timedelta64(
            int(round(meta["requested_offset_sec"] * 1e9)), "ns"
        )
        assert src.segment_containing(t0) == meta["currentIndex"] - 1, r["tag"]


# ---------------------------------------------------------------------------- dsp


def test_hanning_window(reference_dir: Path):
    triton_dsp = pytest.importorskip("triton.dsp")
    for n in (8, 9, 16, 256, 1000, 1024):
        expected, _ = load_reference_bin(f"hanning_{n}")
        np.testing.assert_allclose(triton_dsp.hanning(n), expected.ravel(), rtol=0, atol=1e-15)


def test_spectrogram_matches_mkspecgram(generated_dir: Path, reference_dir: Path):
    triton_dsp = pytest.importorskip("triton.dsp")
    for meta_path in sorted(reference_dir.glob("specgram_*__meta.json")):
        meta = json.loads(meta_path.read_text())
        stem = meta_path.name[: -len("__meta.json")]
        pwr, _ = load_reference_bin(f"{stem}__pwr")
        f, _ = load_reference_bin(f"{stem}__f")
        t, _ = load_reference_bin(f"{stem}__t")
        x, _ = load_reference_bin("dsp_input_samples")

        got = triton_dsp.spectrogram(
            x.ravel()[: int(meta["duration_sec"] * meta["fs"])],
            fs=meta["fs"], nfft=meta["nfft"], overlap_pct=meta["overlap"],
            freq0=meta["freq0"], freq1=meta["freq1"],
        )
        np.testing.assert_allclose(got.f, f.ravel(), rtol=0, atol=1e-12, err_msg=stem)
        np.testing.assert_allclose(got.t, t.ravel(), rtol=0, atol=1e-12, err_msg=stem)
        # Relative *plus* a small absolute floor, and the floor is the point.  A dB
        # value can legitimately pass through zero -- specgram_nfft1000_ol50 has a bin
        # at -3.25e-4 dB -- and a pure relative bound on a near-zero value measures
        # nothing but its own noise.  That bin alone accounted for a 2.2e-10 relative
        # error while its absolute error was 7.3e-14 dB.  Measured agreement across
        # all four cases is 7.3e-12 dB absolute, and 3.2e-13 relative once bins below
        # 1 dB are excluded, so this is not slack: it is the same tightness expressed
        # in a form a zero crossing cannot trip.
        np.testing.assert_allclose(got.db, pwr, rtol=1e-9, atol=1e-9, err_msg=stem)


def test_pwelch_and_int8_quantisation(reference_dir: Path):
    """The LTSA path, including the lossy int8 write (ltsa.md 4)."""
    triton_dsp = pytest.importorskip("triton.dsp")
    x, _ = load_reference_bin("dsp_input_samples")
    for meta_path in sorted(reference_dir.glob("pwelch_*__meta.json")):
        meta = json.loads(meta_path.read_text())
        stem = meta_path.name[: -len("__meta.json")]
        db, _ = load_reference_bin(f"{stem}__db")
        q8, _ = load_reference_bin(f"{stem}__int8")

        got = triton_dsp.welch_db(x.ravel()[: meta["nsamples"]],
                                  fs=meta["fs"], nfft=meta["nfft"], noverlap=0)
        np.testing.assert_allclose(got, db.ravel(), rtol=1e-9, atol=0, err_msg=stem)
        np.testing.assert_array_equal(triton_dsp.to_int8_db(got), q8.ravel().astype(np.int8))


def test_int8_quantisation_ties_and_saturation(reference_dir: Path):
    """The rounding rule at a .5 tie, which no pwelch fixture reaches.

    Measured: the closest any pwelch reference value comes to a tie is 0.0054, so
    `test_pwelch_and_int8_quantisation` passing tells us nothing about tie-breaking --
    and tie-breaking is the one part of `to_int8_db` that needed custom code.  MATLAB's
    `int8()` and `fwrite(...,'int8')` round half **away from zero**; numpy's `round()`
    rounds half to **even**, so they disagree on 0.5, 2.5 and 126.5.  Out-of-range
    values saturate rather than wrapping (ltsa.md 4).

    Prefers a MATLAB dump when one exists.  Until `dump_reference('sections',{'dsp'})`
    is next run, it checks against the documented rule instead, which is weaker
    evidence but still catches the numpy default creeping back in.
    """
    triton_dsp = pytest.importorskip("triton.dsp")

    # Always check the rule, whether or not a MATLAB dump exists.  load_reference_bin
    # skips the whole test when a dump is missing, which would be exactly the wrong
    # behaviour here: this test's value is that it runs *before* the dump is produced.
    probe = np.array([-200.5, -128.5, -127.5, -1.5, -0.5, -0.4, 0.0, 0.4, 0.5, 1.5,
                      2.5, 126.5, 127.4, 127.5, 200.5])
    expected = np.array([-128, -128, -128, -2, -1, 0, 0, 0, 1, 2,
                         3, 127, 127, 127, 127], dtype=np.int8)
    np.testing.assert_array_equal(triton_dsp.to_int8_db(probe), expected)

    # Guard the actual trap: numpy's default rounding gets several of these wrong, so
    # if this ever stops being true the probe has lost its point.
    assert not np.array_equal(
        np.clip(np.round(probe), -128, 127).astype(np.int8), expected
    ), "probe no longer distinguishes half-away-from-zero from numpy's half-to-even"

    # Then, if MATLAB has since confirmed it, check against that instead of the doc.
    if (reference_dir / "int8_quant__in.bin").exists():
        ref_in, _ = load_reference_bin("int8_quant__in")
        ref_out, _ = load_reference_bin("int8_quant__out")
        np.testing.assert_array_equal(
            triton_dsp.to_int8_db(ref_in.ravel()), ref_out.ravel().astype(np.int8)
        )


def test_transfer_function_interpolation(reference_dir: Path):
    triton_dsp = pytest.importorskip("triton.dsp")
    tf, _ = load_reference_bin("tf_interp__in")
    fq, _ = load_reference_bin("tf_interp__f")
    expected, _ = load_reference_bin("tf_interp__out")
    got = triton_dsp.apply_transfer_function_curve(tf[:, 0], tf[:, 1], fq.ravel())
    np.testing.assert_allclose(got, expected.ravel(), rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------- ltsa


def test_ltsa_header_matches(generated_dir: Path, reference_dir: Path):
    if not (Path(reference_dir) / "ltsa_headers.json").exists():
        pytest.skip("no LTSA reference; see fixtures/README.md")
    for r in load_reference("ltsa_headers"):
        lt = triton_io.read_ltsa_header(Path(generated_dir) / r["file"])

        # Every scalar the reference dumps, not a chosen few.  This test originally
        # checked six fields and `ch` was not among them, which let a one-byte offset
        # error through: v3/v4 store `ch` at 38 and the reader looked at 39, which is
        # padding.  Padding is zero and zero is a plausible channel number, so the
        # only symptom was a quietly wrong value.  A field the reference knows and
        # the test ignores is a field with no oracle.
        assert lt.version == r["ver"], r["file"]
        assert lt.data_start_loc == r["dataStartLoc"], r["file"]
        assert lt.n_freq == r["nf"], r["file"]
        assert lt.n_raw_total == r["nrftot"], r["file"]
        assert lt.n_xwav == r["nxwav"], r["file"]
        assert lt.channel == r["ch"], r["file"]
        assert lt.nfft == r["nfft"], r["file"]
        assert lt.fs == r["fs"], r["file"]
        assert lt.tave == pytest.approx(r["tave"], rel=0, abs=0), r["file"]
        assert lt.dfreq == pytest.approx(r["dfreq"], rel=0, abs=0), r["file"]

        assert list(lt.byte_loc) == np.atleast_1d(r["byteloc"]).tolist(), r["file"]
        assert list(lt.n_ave) == np.atleast_1d(r["nave"]).tolist(), r["file"]
        assert [e.raw_file_id for e in lt.entries] == \
            np.atleast_1d(r["rfileid"]).tolist(), r["file"]
        assert [e.filename for e in lt.entries] == \
            [n.strip() for n in np.atleast_1d(r["fnames"]).tolist()], r["file"]

        # The byte-location chain is derivable from nave and nf (ltsa.md 3.2), so it
        # can be checked without MATLAB -- and disagreement would mean the directory
        # is being parsed at the wrong stride.
        chain = [lt.data_start_loc]
        for e in lt.entries[:-1]:
            chain.append(chain[-1] + e.n_ave * lt.n_freq)
        assert chain == list(lt.byte_loc), f"{r['file']}: byte_loc chain"


def test_ltsa_data_block_matches(generated_dir: Path, reference_dir: Path):
    if not (Path(reference_dir) / "ltsa_headers.json").exists():
        pytest.skip("no LTSA reference; see fixtures/README.md")
    for meta_path in sorted(Path(reference_dir).glob("ltsa_block__*__meta.json")):
        meta = json.loads(meta_path.read_text())
        stem = meta_path.name[: -len("__meta.json")]
        expected, _ = load_reference_bin(stem)
        lt = triton_io.open_ltsa(Path(generated_dir) / meta["file"])
        got = lt.read_block(lt.start_time, hours=meta["tseg_hr"])
        np.testing.assert_array_equal(got, expected)


def test_python_mkltsa_is_byte_identical(tmp_path: Path, generated_dir: Path,
                                         reference_dir: Path):
    """THE Phase 1 milestone, and it passes.

    A Python-generated .ltsa is byte-for-byte identical to the one MATLAB's calc_ltsa
    produced from the same inputs and parameters -- header, directory, spare-slot zero
    fill, and every quantised spectrum.

    This is the strongest test in the suite by some distance.  It is the only one that
    exercises the parameter *derivation* (blksz, the tave clamp, nfft, cfact, nave):
    every other test reads those values out of a header instead of computing them.  It
    also depends on the window definition, the PSD normalisation and the int8 quantiser
    being simultaneously right, with no tolerance to hide behind.

    If it breaks, suspect the derivation before the arithmetic -- the arithmetic has its
    own tests and the derivation does not."""
    if not (Path(reference_dir) / "ltsa_headers.json").exists():
        pytest.skip("no LTSA reference; see fixtures/README.md")
    from triton.tools import mkltsa

    for r in load_reference("ltsa_headers"):
        matlab_bytes = (Path(generated_dir) / r["file"]).read_bytes()
        out = tmp_path / r["file"]

        # `fnames` is dumped from PARAMS.ltsahd.fname, which get_headers.m:113 fills in
        # **per raw file** -- so a single four-raw-file x.wav appears four times.  The
        # input list is the unique names in order; passing the raw list would build an
        # LTSA of four copies of the same file, which is a different (valid) LTSA.
        names, seen = [], set()
        for n in np.atleast_1d(r["fnames"]).tolist():
            n = n.strip()
            if n not in seen:
                seen.add(n)
                names.append(n)

        mkltsa.build(
            inputs=[Path(generated_dir) / n for n in names],
            out=out, tave=r["tave"], dfreq=r["dfreq"], channel=r["ch"],
        )
        got = out.read_bytes()
        if got != matlab_bytes:
            # Locate the disagreement rather than dumping two 11 kB blobs, since the
            # offset alone usually names the field (ltsa.md 2 and 3).
            n = min(len(got), len(matlab_bytes))
            first = next((i for i in range(n) if got[i] != matlab_bytes[i]), n)
            raise AssertionError(
                f"{r['file']}: sizes {len(got)} vs {len(matlab_bytes)}, first "
                f"difference at byte {first}\n"
                f"  ours   {got[max(0, first - 8):first + 16].hex(' ')}\n"
                f"  MATLAB {matlab_bytes[max(0, first - 8):first + 16].hex(' ')}"
            )


def _safe(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", name)
