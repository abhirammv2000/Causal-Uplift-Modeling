"""Download the full Criteo uplift dataset (~290 MB compressed) into data/raw/.

Usage:
    python scripts/download_data.py [--force]

The file is large; the committed sample (data/sample/) is enough to run the
notebooks, tests, and pipeline end-to-end without this download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config, data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    print(f"Downloading Criteo uplift v2.1 from {config.CRITEO_URL}")
    path = data.download_criteo(force=args.force)
    size_mb = path.stat().st_size / 1e6
    print(f"Saved to {path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
