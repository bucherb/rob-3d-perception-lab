#!/usr/bin/env python3
"""Download the Lab 1 dataset.

    python lab1/fetch_data.py --dest data/

Downloads the archive, verifies its checksum, and unpacks it to data/lab1/ --
the directory you then pass to run_lab1.py as --data.

RUN THIS THE DAY THE LAB IS RELEASED, not the night before it is due. The
ScanNet++ data requires you to have accepted the dataset terms, which takes time
to approve, and the download is several gigabytes.

The download URL and checksum are distributed on Piazza rather than hard-coded
here, so that a re-hosted dataset does not silently break everyone's checkout.
Fill in DATA_URL below from the Piazza post, or set ROB498_LAB1_URL.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = os.environ.get("ROB498_LAB1_URL", "")
SHA256 = os.environ.get("ROB498_LAB1_SHA256", "")


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while (block := f.read(chunk)):
            h.update(block)
    return h.hexdigest()


def unpack(archive: Path, target: Path) -> Path:
    """Unpack `archive` so that `target` holds segmentation/ and eval_scene/.

    An archive with a single top-level folder is unwrapped. Members that would
    land outside `target` are refused.
    """
    tmp = target.with_name(target.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    root = tmp.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():
                if not (root / m).resolve().is_relative_to(root):
                    raise RuntimeError(f"refusing archive member outside target: {m}")
            z.extractall(tmp)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not (root / m.name).resolve().is_relative_to(root) or m.issym() or m.islnk():
                    raise RuntimeError(f"refusing archive member: {m.name}")
            t.extractall(tmp)
    else:
        raise RuntimeError(f"{archive} is neither a zip nor a tar archive")
    entries = [e for e in tmp.iterdir() if not e.name.startswith((".", "__MACOSX"))]
    src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))
    shutil.rmtree(tmp, ignore_errors=True)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("data"))
    ap.add_argument("--url", default=DATA_URL)
    ap.add_argument("--sha256", default=SHA256,
                    help="expected checksum (or set ROB498_LAB1_SHA256)")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--no-unpack", action="store_true",
                    help="download and verify only")
    args = ap.parse_args()

    if not args.url:
        print("error: no download URL.\n"
              "Get it from the Lab 1 Piazza post and either pass --url or set\n"
              "  export ROB498_LAB1_URL=...", file=sys.stderr)
        return 2

    args.dest.mkdir(parents=True, exist_ok=True)
    # Name from the URL *path*, so a signed URL's ?query does not end up in it.
    out = args.dest / (Path(urllib.parse.urlparse(args.url).path).name or "lab1_data.zip")

    if out.exists():
        print(f"{out} already exists; delete it to re-download.")
    else:
        print(f"downloading {args.url}\n  -> {out}")

        def hook(blocks, bs, total):
            if total > 0:
                pct = min(100.0, blocks * bs * 100.0 / total)
                print(f"\r  {pct:5.1f}%  ({blocks * bs / 1e9:.2f} / {total / 1e9:.2f} GB)",
                      end="", flush=True)

        urllib.request.urlretrieve(args.url, out, reporthook=hook)
        print()

    if args.sha256 and not args.skip_verify:
        print("verifying checksum...")
        got = sha256sum(out)
        if got != args.sha256.lower():
            print(f"CHECKSUM MISMATCH\n  expected {args.sha256}\n  got      {got}\n"
                  "The download is corrupt or truncated. Delete it and retry.",
                  file=sys.stderr)
            return 1
        print("  ok")

    if args.no_unpack:
        print(f"\ndone: {out}")
        return 0
    target = unpack(out, args.dest / "lab1")
    missing = [d for d in ("segmentation", "eval_scene") if not (target / d).is_dir()]
    if missing:
        print(f"warning: {target} has no {', '.join(missing)}/ -- see lab1/dataset.py "
              f"for the expected layout", file=sys.stderr)
    print(f"\ndone: unpacked to {target}\n  python lab1/run_lab1.py --data {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
