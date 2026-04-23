"""
Launch the Streamlit UI.

Usage:
    ./venv/bin/python scripts/run_ui.py

Equivalent to:
    ./venv/bin/streamlit run src/app/ui.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ui_path = root / "src" / "app" / "ui.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            "--server.headless=false",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
