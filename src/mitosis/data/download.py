"""Download single MIDOG++ images from figshare (with retries)."""

import time
import urllib.request
from pathlib import Path

FIGSHARE_URL = "https://ndownloader.figshare.com/files/{}"


def download(fid: str, dest: Path, retries: int = 4):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(FIGSHARE_URL.format(fid), part)
            part.rename(dest)
            return
        except Exception as e:  # network hiccups are common on long runs
            print(f"  ! download {dest.name} failed ({e}), attempt {attempt}/{retries}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download {dest.name}")
