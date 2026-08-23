"""Shared pytest fixtures for the Triton-Python parity suite.

Layout:
    fixtures/generated/   synthetic files from tools/make_fixtures.py (committed, ~2.3 MB)
    fixtures/real/        real deployment files contributed by the team (NOT committed)
    fixtures/reference/   MATLAB ground-truth dumps from tools/matlab/dump_reference.m

Tests that need MATLAB reference output are skipped (not failed) when the dump is absent, so the
suite is usable on a machine without MATLAB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "fixtures"
GENERATED = FIXTURES / "generated"
REAL = FIXTURES / "real"
REFERENCE = FIXTURES / "reference"


@pytest.fixture(scope="session")
def generated_dir() -> Path:
    if not (GENERATED / "manifest.json").exists():
        pytest.skip("run `python tools/make_fixtures.py` first")
    return GENERATED


@pytest.fixture(scope="session")
def manifest(generated_dir: Path) -> dict:
    return json.loads((generated_dir / "manifest.json").read_text())


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    """MATLAB ground truth.  Skips the test if it hasn't been generated."""
    if not REFERENCE.exists() or not any(REFERENCE.iterdir()):
        pytest.skip(
            "no MATLAB reference dump present; run tools/matlab/dump_reference.m in MATLAB "
            "and commit fixtures/reference/"
        )
    return REFERENCE


def load_reference(name: str) -> dict:
    """Read one reference JSON dump by name, e.g. 'xwav_headers'."""
    p = REFERENCE / f"{name}.json"
    if not p.exists():
        pytest.skip(f"reference dump {p.name} not present")
    return json.loads(p.read_text())


def load_reference_bin(name: str, dtype: str = "float64"):
    """Read one reference binary array dump plus its JSON sidecar.

    dump_reference.m writes raw little-endian arrays alongside a .json describing shape/class,
    because JSON floats are lossy and we need bit-exact comparison.
    """
    import numpy as np

    meta_p = REFERENCE / f"{name}.json"
    bin_p = REFERENCE / f"{name}.bin"
    if not (meta_p.exists() and bin_p.exists()):
        pytest.skip(f"reference dump {name} not present")
    meta = json.loads(meta_p.read_text())
    arr = np.fromfile(bin_p, dtype=meta.get("dtype", dtype))
    # MATLAB writes column-major
    return arr.reshape(tuple(meta["shape"]), order="F"), meta
