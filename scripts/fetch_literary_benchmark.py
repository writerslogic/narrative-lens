# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Acquire a pinned public development corpus; production analysis never downloads it."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import urllib.request


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check(data, entry):
    git_blob = b"blob " + str(len(data)).encode() + b"\0" + data
    if len(data) != entry["size"] or hashlib.sha1(git_blob, usedforsecurity=False).hexdigest() != entry["gitBlobSha1"]:
        raise ValueError("pinned Git blob does not match: " + entry["path"])
    if entry.get("sha256") and sha256(data) != entry["sha256"]:
        raise ValueError("pinned SHA-256 does not match: " + entry["path"])


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=root / "docs/benchmarks/litbank-source.json")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    cache = (args.cache_dir or root / spec["cache"]).resolve()
    target = (root / "target").resolve()
    if cache.is_relative_to(root) and cache != target and not cache.is_relative_to(target):
        raise ValueError("corpus cache must stay under ignored target or outside the repository")
    if args.write_lock and all("sha256" in item for item in spec["files"]):
        raise ValueError("corpus lock already exists; do not overwrite frozen evidence")
    expected = {}
    for entry in spec["files"]:
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise ValueError("invalid corpus path")
        if entry["size"] > spec["limits"]["memberBytes"] or entry["path"] in expected:
            raise ValueError("invalid corpus index")
        expected[entry["path"]] = entry
    if sum(item["size"] for item in expected.values()) > spec["limits"]["selectedBytes"]:
        raise ValueError("selected corpus exceeds resource limit")
    missing = []
    for name, entry in expected.items():
        path = cache / name
        if not path.resolve().is_relative_to(cache):
            raise ValueError("cache path escapes its root")
        if path.exists():
            if path.stat().st_size != entry["size"]:
                raise ValueError("cached file length does not match: " + name)
            check(path.read_bytes(), entry)
        else:
            missing.append(name)
    archive_digest = None
    if missing:
        if not args.download:
            raise FileNotFoundError("missing corpus; opt into its bounded public download with --download")
        cache.mkdir(parents=True, exist_ok=True)
        url = f"https://codeload.github.com/{spec['repository']}/tar.gz/{spec['revision']}"
        descriptor, temporary = tempfile.mkstemp(prefix="litbank-", suffix=".tar.gz", dir=cache)
        try:
            total = 0
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > spec["limits"]["archiveBytes"]:
                        raise ValueError("archive exceeds resource limit")
                    digest.update(chunk)
                    output.write(chunk)
            archive_digest = digest.hexdigest()
            if spec.get("archiveSha256") and archive_digest != spec["archiveSha256"]:
                raise ValueError("pinned archive digest mismatch")
            seen = set()
            with tarfile.open(temporary, "r:gz") as archive:
                for member in archive:
                    parts = PurePosixPath(member.name).parts
                    name = "/".join(parts[1:])
                    if name not in expected:
                        continue
                    if name in seen or not member.isfile() or member.size != expected[name]["size"]:
                        raise ValueError("unexpected archive member")
                    seen.add(name)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError("unreadable archive member")
                    data = source.read(spec["limits"]["memberBytes"] + 1)
                    check(data, expected[name])
                    path = cache / name
                    if not path.resolve().is_relative_to(cache):
                        raise ValueError("cache path escapes its root")
                    atomic_write(path, data)
            if seen != expected.keys():
                raise ValueError("archive is missing pinned files")
        finally:
            Path(temporary).unlink(missing_ok=True)
    if args.write_lock:
        for name, entry in expected.items():
            entry["sha256"] = sha256((cache / name).read_bytes())
        if archive_digest:
            spec["archiveSha256"] = archive_digest
        atomic_write(args.spec, (json.dumps(spec, indent=2, ensure_ascii=False) + "\n").encode())
    print(json.dumps({"repository": spec["repository"], "revision": spec["revision"], "files": len(expected), "selectedBytes": sum(e["size"] for e in expected.values()), "cache": str(cache), "sha256Locked": all("sha256" in e for e in expected.values())}))


if __name__ == "__main__":
    main()
