"""Self-checks on the generated fixture corpus.

These do NOT test the (not yet written) triton.io package.  They verify that the fixtures
themselves are well formed and satisfy every consistency rule that Triton's MATLAB reader
enforces -- so that when MATLAB rejects a fixture we know it is a spec misunderstanding, not a
generator bug.

The parser here is a deliberately minimal, independent re-read of docs/formats/xwav.md.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

YEAR_OFFSET = 2000


def parse_xwav_header(path: Path) -> dict:
    """Minimal x.wav header parser, straight from docs/formats/xwav.md."""
    b = path.read_bytes()
    h: dict = {"filesize": len(b)}

    h["ChunkID"] = b[0:4].decode("ascii")
    h["ChunkSize"] = struct.unpack_from("<I", b, 4)[0]
    h["Format"] = b[8:12].decode("ascii")

    h["fSubchunkID"] = b[12:16].decode("ascii")
    h["fSubchunkSize"] = struct.unpack_from("<I", b, 16)[0]
    (h["AudioFormat"], h["NumChannels"], h["SampleRate"], h["ByteRate"],
     h["BlockAlign"], h["BitsPerSample"]) = struct.unpack_from("<HHIIHH", b, 20)

    h["hSubchunkID"] = b[36:40].decode("ascii")
    h["hSubchunkSize"] = struct.unpack_from("<I", b, 40)[0]
    h["WavVersionNumber"] = b[44]
    h["FirmwareVersionNumber"] = b[45:55].decode("ascii")
    h["InstrumentID"] = b[55:59].decode("ascii")
    h["SiteName"] = b[59:63].decode("ascii")
    h["ExperimentName"] = b[63:71].decode("ascii")
    h["DiskSequenceNumber"] = b[71]
    h["DiskSerialNumber"] = b[72:80].decode("ascii")
    h["NumOfRawFiles"] = struct.unpack_from("<H", b, 80)[0]
    h["Longitude"] = struct.unpack_from("<i", b, 82)[0]
    h["Latitude"] = struct.unpack_from("<i", b, 86)[0]
    h["Depth"] = struct.unpack_from("<h", b, 90)[0]

    nch = h["NumChannels"]
    off = 92
    if h["WavVersionNumber"] == 2:
        h["drate"] = list(struct.unpack_from(f"<{nch}f", b, off))
        off += 4 * nch
    off += 8  # Reserved

    entry = 32 + (4 * nch if h["WavVersionNumber"] == 2 else 0)
    raws = []
    for _ in range(h["NumOfRawFiles"]):
        (yr, mo, dy, hh, mm, ss, ticks, byte_loc, byte_length, write_length,
         sample_rate, gain) = struct.unpack_from("<BBBBBBHIIIIB", b, off)
        r = dict(year=yr, month=mo, day=dy, hour=hh, minute=mm, secs=ss, ticks=ticks,
                 byte_loc=byte_loc, byte_length=byte_length, write_length=write_length,
                 sample_rate=sample_rate, gain=gain)
        if h["WavVersionNumber"] == 2:
            r["dt"] = list(struct.unpack_from(f"<{nch}f", b, off + 32))
        raws.append(r)
        off += entry
    h["raw_files"] = raws

    h["dSubchunkID"] = b[off:off + 4].decode("ascii")
    h["dSubchunkSize"] = struct.unpack_from("<I", b, off + 4)[0]
    h["data_offset"] = off + 8
    return h


def xwav_fixtures(manifest: dict):
    return [m for m in manifest["fixtures"] if m["kind"] == "xwav"]


# ------------------------------------------------------------------------------------------
# reproducibility
# ------------------------------------------------------------------------------------------


def test_fixtures_are_byte_reproducible(generated_dir: Path, manifest: dict):
    """Regenerating must produce identical bytes; the manifest hashes are the contract."""
    for m in manifest["fixtures"]:
        data = (generated_dir / m["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == m["sha256"], f"{m['file']} changed"
        assert len(data) == m["bytes"]


# ------------------------------------------------------------------------------------------
# rules Triton's reader enforces (rdxwavhd.m)
# ------------------------------------------------------------------------------------------


def test_riff_and_format_chunks(generated_dir: Path, manifest: dict):
    for m in xwav_fixtures(manifest):
        h = parse_xwav_header(generated_dir / m["file"])
        assert h["ChunkID"] == "RIFF"
        assert h["Format"] == "WAVE"
        # rdxwavhd.m:34 warns if this fails
        assert h["ChunkSize"] == h["filesize"] - 8, m["file"]
        # rdxwavhd.m:57 aborts if either fails
        assert h["fSubchunkID"] == "fmt "
        assert h["fSubchunkSize"] == 16
        assert h["BlockAlign"] == h["NumChannels"] * h["BitsPerSample"] // 8
        assert h["ByteRate"] == h["SampleRate"] * h["BlockAlign"]


def test_harp_subchunk_size_formula(generated_dir: Path, manifest: dict):
    """rdxwavhd.m:106-116 -- the formula differs between v1 and v2."""
    for m in xwav_fixtures(manifest):
        h = parse_xwav_header(generated_dir / m["file"])
        nch, nrf = h["NumChannels"], h["NumOfRawFiles"]
        if h["WavVersionNumber"] == 2:
            expect = (64 + 4 * nch) - 8 + nrf * (32 + 4 * nch)
        else:
            expect = 64 - 8 + nrf * 32
        assert h["hSubchunkSize"] == expect, m["file"]
        assert h["hSubchunkID"] == "harp"


def test_data_chunk_and_byte_locations(generated_dir: Path, manifest: dict):
    for m in xwav_fixtures(manifest):
        h = parse_xwav_header(generated_dir / m["file"])
        assert h["dSubchunkID"] == "data", m["file"]
        # first raw file starts immediately after the header (wrxwavhd.m:42)
        assert h["raw_files"][0]["byte_loc"] == h["data_offset"], m["file"]
        # raw files are contiguous on disk even when duty-cycled in time
        for a, b in zip(h["raw_files"], h["raw_files"][1:], strict=False):
            assert b["byte_loc"] == a["byte_loc"] + a["byte_length"], m["file"]
        total = sum(r["byte_length"] for r in h["raw_files"])
        assert h["dSubchunkSize"] == total, m["file"]
        assert h["data_offset"] + total == h["filesize"], m["file"]


def test_write_length_sector_arithmetic(generated_dir: Path, manifest: dict):
    """ck_ltsaparams.m:48-57 -- nave arithmetic goes non-integer if this doesn't hold."""
    for m in xwav_fixtures(manifest):
        h = parse_xwav_header(generated_dir / m["file"])
        nch = h["NumChannels"]
        blksz = (512 - 12) // 2 if nch == 1 else (512 - 12 - 4) // 2
        bps = h["BitsPerSample"] // 8
        for r in h["raw_files"]:
            assert r["write_length"] * blksz * bps == r["byte_length"], m["file"]


def test_year_byte_is_offset_by_2000(generated_dir: Path, manifest: dict):
    """The header stores year-2000; a value >= 100 means someone wrote a full year."""
    for m in xwav_fixtures(manifest):
        h = parse_xwav_header(generated_dir / m["file"])
        for r in h["raw_files"]:
            assert 0 <= r["year"] < 100, m["file"]
            assert 1 <= r["month"] <= 12
            assert 1 <= r["day"] <= 31
            assert 0 <= r["ticks"] < 1000


def test_manifest_matches_headers(generated_dir: Path, manifest: dict):
    """The manifest is what the parity tests read; it must agree with the actual bytes."""
    for m in xwav_fixtures(manifest):
        h = parse_xwav_header(generated_dir / m["file"])
        assert h["NumOfRawFiles"] == m["n_raw_files"], m["file"]
        assert h["NumChannels"] == m["n_channels"]
        assert h["BitsPerSample"] == m["bits_per_sample"]
        assert h["WavVersionNumber"] == m["wav_version"]
        assert h["SampleRate"] == m["fmt_sample_rate"]
        assert h["data_offset"] == m["header_size"]
        for r in h["raw_files"]:
            assert r["sample_rate"] == m["true_sample_rate"]
            assert r["gain"] == m["gain"]


# ------------------------------------------------------------------------------------------
# the specific conditions each fixture exists to exercise
# ------------------------------------------------------------------------------------------


def test_corpus_covers_the_intended_cases(manifest: dict):
    xs = xwav_fixtures(manifest)
    assert {x["wav_version"] for x in xs} >= {1, 2}, "need both v1 and v2 headers"
    assert {x["n_channels"] for x in xs} >= {1, 4}, "need both sector geometries"
    assert {x["bits_per_sample"] for x in xs} >= {16, 32}
    assert any(x["duty_cycled"] for x in xs), "need a duty-cycled file"
    assert any(not x["duty_cycled"] for x in xs)
    assert any(x["gain"] != 1 for x in xs), "need a non-unity gain file"
    assert any(x["true_sample_rate"] >= 200_000 for x in xs), "need a HARP-rate file"
    assert any(x["raw_files"][0]["ticks"] != 0 for x in xs), "need sub-second start times"
    assert any(x["fmt_sample_rate"] != x["true_sample_rate"] for x in xs), \
        "need a file where the fmt chunk rate is a lie (xwav.md 5.1)"
    wavs = [m for m in manifest["fixtures"] if m["kind"] == "wav"]
    assert any(w["bits_per_sample"] == 24 for w in wavs), "need a 24-bit file"


def test_duty_cycled_fixture_really_has_gaps(generated_dir: Path, manifest: dict):
    """A duty-cycled file must be contiguous in bytes but discontinuous in time -- the exact
    condition that makes readseg splice across a boundary."""
    import datetime as dt

    m = next(x for x in xwav_fixtures(manifest) if x["duty_cycled"])
    h = parse_xwav_header(generated_dir / m["file"])
    fs = h["raw_files"][0]["sample_rate"]
    bps = h["BitsPerSample"] // 8
    nch = h["NumChannels"]

    def start(r):
        return dt.datetime(r["year"] + YEAR_OFFSET, r["month"], r["day"],
                           r["hour"], r["minute"], r["secs"],
                           r["ticks"] * 1000)

    saw_gap = False
    for a, b in zip(h["raw_files"], h["raw_files"][1:], strict=False):
        dur = a["byte_length"] / (bps * nch) / fs
        gap = (start(b) - start(a)).total_seconds() - dur
        if gap > 0:
            saw_gap = True
        assert b["byte_loc"] == a["byte_loc"] + a["byte_length"], "must stay byte-contiguous"
    assert saw_gap, "duty-cycled fixture has no time gaps"


@pytest.mark.parametrize("field", ["true_sample_rate", "n_channels", "bits_per_sample"])
def test_no_zero_valued_key_fields(manifest: dict, field: str):
    for m in xwav_fixtures(manifest):
        assert m[field] > 0
