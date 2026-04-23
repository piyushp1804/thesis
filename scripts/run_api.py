"""
Launch the FastAPI server.

Usage:
    ./venv/bin/python scripts/run_api.py

Equivalent to:
    ./venv/bin/uvicorn src.app.api:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    import uvicorn
    uvicorn.run(
        "src.app.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
