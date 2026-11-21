"""Download the Low Carbon London files from the London Datastore (dataset vqm0d, CC BY 4.0) and verify them.

Every file is checked against the size recorded in the Datastore metadata (and the zip structure / md5);
a file that fails the check is reported (HTTP status, final URL, content type, first bytes) and deleted,
and the next source is tried. Sources, in order:
  1. Partitioned LCL Data.zip  (168 CSVs of 1M rows)
  2. LCL-FullData.zip          (one ~10 GB CSV; lcl_prepare.py reads it in chunks)

Usage:  python scripts/lcl_download.py --raw /content/lcl/raw
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://data.london.gov.uk/download/vqm0d"
TARIFF = dict(name="Tariffs.xlsx", url=f"{BASE}/14855047-44c2-4856-8a48-e5649200e6ce/Tariffs.xlsx",
              size=245384, md5="d64694233f1e3b7241fb1db02ff063f2")
ZIPS = [
    dict(name="Partitioned_LCL_Data.zip", url=f"{BASE}/04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip",
         size=795722689),
    dict(name="LCL-FullData.zip", url=f"{BASE}/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip", size=801674949),
]
UA = {"User-Agent": "covattn-data-fetch/0.1 (research; python urllib)"}


def fetch(url: str, dest: Path, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, headers=UA)
    info: dict = {"url": url}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        info.update(status=r.status, final_url=r.geturl(), content_type=r.headers.get("Content-Type"),
                    content_length=r.headers.get("Content-Length"))
        print(f"  HTTP {r.status}, final URL {info['final_url']}, type {info['content_type']}, "
              f"length {info['content_length']}", flush=True)
        n, last = 0, 0.0
        with open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                n += len(chunk)
                if time.time() - last > 30:
                    print(f"  {n / 1e6:.0f} MB, {n / 1e6 / max(time.time() - t0, 1e-9):.1f} MB/s", flush=True)
                    last = time.time()
    info["bytes"] = n
    return info


def check(path: Path, spec: dict) -> str | None:
    """None if fine, else the reason."""
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size != spec["size"]:
        return f"size {size} != expected {spec['size']}"
    if "md5" in spec and hashlib.md5(path.read_bytes()).hexdigest() != spec["md5"]:
        return "md5 mismatch"
    if path.suffix == ".zip":
        if not zipfile.is_zipfile(path):
            return "not a zip file"
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            bad = z.testzip() if len(names) < 400 else None  # CRC check of every member (reads the whole file)
        if bad is not None:
            return f"corrupt member {bad}"
        print(f"  zip ok: {len(names)} members, e.g. {names[:3]}")
    return None


def report_bad(path: Path, why: str) -> None:
    head = path.read_bytes()[:400] if path.exists() else b""
    print(f"  BAD: {why}; first bytes: {head!r}")
    path.unlink(missing_ok=True)


def get(spec: dict, raw: Path) -> bool:
    dest = raw / spec["name"]
    why = check(dest, spec)
    if why is None:
        print(f"{spec['name']}: already present and verified")
        return True
    if dest.exists():
        report_bad(dest, f"existing file: {why}")
    print(f"{spec['name']}: downloading {spec['url']}", flush=True)
    try:
        fetch(spec["url"], dest)
    except Exception as e:  # noqa: BLE001 - report and try the next source
        print(f"  download error: {type(e).__name__}: {e}")
        dest.unlink(missing_ok=True)
        return False
    why = check(dest, spec)
    if why is not None:
        report_bad(dest, why)
        return False
    print(f"{spec['name']}: verified")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/lcl/raw")
    args = ap.parse_args()
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    if not get(TARIFF, raw):
        sys.exit("tariff file could not be downloaded")
    for spec in ZIPS:
        if get(spec, raw):
            print(f"USING {spec['name']}")
            return
    sys.exit("no consumption zip could be downloaded and verified (see messages above)")


if __name__ == "__main__":
    main()
