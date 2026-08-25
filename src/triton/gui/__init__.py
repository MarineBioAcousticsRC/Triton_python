"""The Qt viewer.

Import-guarded: ``triton`` itself must stay usable with no GUI toolkit installed, so
nothing here is imported by the top-level package. ``triton.session`` is checked by a
test that scans its imports and fails if Qt appears in them.

Run it with ``python -m triton.gui [file...]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["launch", "main"]


def launch(audio: str | Path | None = None, ltsa: str | Path | None = None):
    """Open the window.  Returns ``(app, window)`` without entering the event loop.

    Split from :func:`main` so a test can build the window, drive it, and inspect it
    without blocking -- which is the only way GUI code gets tested at all.
    """
    from PySide6.QtWidgets import QApplication

    from ..session import TritonSession
    from .window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    session = TritonSession()
    win = MainWindow(session)
    if audio is not None:
        win.open_audio(audio)
    if ltsa is not None:
        session.open_ltsa(ltsa)
        session.view.show_ltsa = True
    return app, win


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m triton.gui", description="Triton viewer")
    ap.add_argument("files", nargs="*", type=Path,
                    help="audio and/or .ltsa files to open")
    a = ap.parse_args(argv)

    audio = next((f for f in a.files if not f.name.lower().endswith(".ltsa")), None)
    ltsa = next((f for f in a.files if f.name.lower().endswith(".ltsa")), None)

    app, win = launch(audio, ltsa)
    win.show()
    return app.exec()
