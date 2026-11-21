"""Download the Low Carbon London files and verify them.

Tariffs.xlsx always comes from the London Datastore (dataset vqm0d, CC BY 4.0; size + md5 checked).
Consumption data, by --source:
  kaggle (default)  Kaggle mirror jeanmidev/smart-meters-in-london, a refactor of the same Datastore data:
                    informations_households.csv (LCLid, stdorToU, Acorn, Acorn_grouped) and the 112
                    halfhourly_dataset blocks (LCLid, tstp, energy(kWh/hh)), each a zip of one CSV, checked
                    against the uncompressed sizes in the Kaggle file listing (+ CRC). Anonymous download.
  datastore         Partitioned LCL Data.zip, then LCL-FullData.zip. As of 2026-11-21 the Datastore origin
                    times out for both (Cloudflare HTTP 524 after ~127 s, from a local machine and from Colab),
                    while small files from the same host download fine.
A file that fails its check is reported (HTTP status, final URL, content type, first bytes) and deleted.

Usage:  python scripts/lcl_download.py --raw /content/lcl/raw [--source kaggle|datastore]
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
KAGGLE = "https://www.kaggle.com/api/v1/datasets"
KAGGLE_REF = "jeanmidev/smart-meters-in-london"
KAGGLE_DIR = "kaggle"  # must match lcl_prepare.KAGGLE_DIR


def kaggle_listing() -> dict[str, int]:
    """file path -> uncompressed bytes, from the public Kaggle API (paginated)."""
    import json
    import urllib.parse

    out, tok = {}, None
    while True:
        url = f"{KAGGLE}/list/{KAGGLE_REF}?pageSize=200" + (f"&pageToken={urllib.parse.quote(tok)}" if tok else "")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            d = json.load(r)
        out.update({f["name"]: int(f["totalBytes"]) for f in d["datasetFiles"]})
        tok = d.get("nextPageTokenNullable")
        if not tok:
            return out


def check_kaggle(path: Path, size: int) -> str | None:
    """Kaggle serves large files as a zip of one member and small ones raw; check the uncompressed size."""
    if not path.exists():
        return "missing"
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            if len(infos) != 1 or infos[0].file_size != size:
                return f"zip members {[(i.filename, i.file_size) for i in infos]} != one file of {size} bytes"
            if z.testzip() is not None:
                return "CRC error"
        return None
    return None if path.stat().st_size == size else f"size {path.stat().st_size} != {size}"


def get_kaggle(raw: Path, retries: int = 3) -> bool:
    import urllib.parse

    files = kaggle_listing()
    want = {"informations_households.csv": raw / KAGGLE_DIR / "informations_households.csv"}
    blocks = sorted((n for n in files if n.startswith("halfhourly_dataset/") and n.endswith(".csv")),
                    key=lambda n: int(Path(n).stem.split("_")[1]))
    for n in blocks:
        want[n] = raw / KAGGLE_DIR / "halfhourly" / (Path(n).stem + ".zip")
    print(f"Kaggle {KAGGLE_REF}: {len(blocks)} half-hourly blocks, "
          f"{sum(files[n] for n in blocks) / 1e9:.2f} GB uncompressed")
    ok = 0
    for i, (name, dest) in enumerate(want.items()):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if check_kaggle(dest, files[name]) is None:
            ok += 1
            continue
        for attempt in range(retries):
            try:
                fetch(f"{KAGGLE}/download/{KAGGLE_REF}/{urllib.parse.quote(name, safe='')}", dest, verbose=False)
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: {type(e).__name__}: {e} (attempt {attempt + 1})")
                dest.unlink(missing_ok=True)
                continue
            why = check_kaggle(dest, files[name])
            if why is None:
                ok += 1
                break
            report_bad(dest, f"{name}: {why}")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(want)} files", flush=True)
    # informations_households.csv may arrive zipped; store it as plain CSV for lcl_prepare.py
    info = want["informations_households.csv"]
    if info.exists() and zipfile.is_zipfile(info):
        with zipfile.ZipFile(info) as z:
            data = z.read(z.infolist()[0])
        info.write_bytes(data)
    print(f"Kaggle: {ok}/{len(want)} files verified")
    return ok == len(want)


def fetch(url: str, dest: Path, timeout: float = 120, verbose: bool = True) -> dict:
    req = urllib.request.Request(url, headers=UA)
    info: dict = {"url": url}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        info.update(status=r.status, final_url=r.geturl(), content_type=r.headers.get("Content-Type"),
                    content_length=r.headers.get("Content-Length"))
        if verbose:
            print(f"  HTTP {r.status}, final URL {info['final_url']}, type {info['content_type']}, "
                  f"length {info['content_length']}", flush=True)
        n, last = 0, (0.0 if verbose else float("inf"))
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
    ap.add_argument("--source", choices=("kaggle", "datastore"), default="kaggle")
    args = ap.parse_args()
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    if not get(TARIFF, raw):
        sys.exit("tariff file could not be downloaded")
    if args.source == "kaggle":
        if get_kaggle(raw):
            print("USING kaggle")
            return
        sys.exit("Kaggle download incomplete (see messages above); re-run to resume")
    for spec in ZIPS:
        if get(spec, raw):
            print(f"USING {spec['name']}")
            return
    sys.exit("no consumption zip could be downloaded and verified (see messages above)")


if __name__ == "__main__":
    main()
