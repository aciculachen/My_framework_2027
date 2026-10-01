#!/usr/bin/env python3
"""raw csv (data/raw/v3) -> data/processed. Thin driver for data.preproc."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.preproc import main  # noqa: E402

if __name__ == "__main__":
    main()
