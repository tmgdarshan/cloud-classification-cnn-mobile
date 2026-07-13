"""Test configuration.

Makes the pre-package ``src`` modules importable during tests. This shim will be
removed once the source tree becomes an installable package (a later phase).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
