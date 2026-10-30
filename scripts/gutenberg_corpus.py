"""Unlabelled public-domain novels for teacher-student self-training.

`select` reads Project Gutenberg's catalogue, keeps English prose fiction that is not a
LitBank source novel (by Gutenberg id; the LitBank holdout must stay unseen), takes the
lowest ids first (the earliest digitised classics), downloads each text, strips the
Gutenberg header and footer, keeps novel-length books, and pins id, title, author, word
count, and sha256 in `docs/benchmarks/gutenberg-source.json`. `fetch` re-downloads the
pinned list, verifies the hashes, and writes
`target/external-validation/narrative-multitask/gutenberg.jsonl` with text-only documents
(`token_labels: false`, `unlabeled: true`) that `multitask_train.py` structures with the
teacher's own predictions.

    uv run --no-project scripts/gutenberg_corpus.py select --novels 100
    uv run --no-project scripts/gutenberg_corpus.py fetch
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "target/external-validation/gutenberg"
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
SPEC = ROOT / "docs/benchmarks/gutenberg-source.json"
OUTPUT = ROOT / "target/external-validation/narrative-multitask/gutenberg.jsonl"
LITBANK_INPUTS = ROOT / "target/external-validation/litbank-prepared-v4/inputs.json"
EXCLUDED_SUBJECTS = ("juvenile", "short stories", "poetry", "drama", "science fiction", "fairy tales", "humor", "periodicals")
MIN_WORDS, MAX_WORDS = 40_000, 260_000
START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)


def download(url, maximum=8 * 1024 * 1024):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "narrative-lens corpus builder"}), timeout=120) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"{url} exceeds {maximum} bytes")
    return data


def body(raw):
    text = raw.decode("utf-8-sig", errors="replace")
    start = START.search(text)
    end = END.search(text)
    if not start or not end or end.start() <= start.end():
        return None
    return text[start.end():end.start()].strip().replace("\r\n", "\n")


def litbank_ids():
    if not LITBANK_INPUTS.exists():
        raise SystemExit(f"{LITBANK_INPUTS} missing; the LitBank ids (not labels) are needed to exclude its novels")
    return {case["id"] for case in json.loads(LITBANK_INPUTS.read_text())}


def candidates():
    path = CACHE / "pg_catalog.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(download(CATALOG_URL, 64 * 1024 * 1024))
    excluded = litbank_ids()
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            subjects = row["Subjects"].lower()
            if row["Type"] != "Text" or row["Language"] != "en" or row["Text#"] in excluded:
                continue
            if "fiction" not in subjects or any(bad in subjects for bad in EXCLUDED_SUBJECTS):
                continue
            rows.append(row)
    rows.sort(key=lambda row: int(row["Text#"]))
    return rows


def select(args):
    chosen = []
    for row in candidates():
        if len(chosen) >= args.novels:
            break
        book = int(row["Text#"])
        try:
            raw = download(TEXT_URL.format(id=book))
        except Exception as error:  # noqa: BLE001 - a missing text is skipped, not fatal
            print(f"skip {book}: {error}", file=sys.stderr)
            continue
        text = body(raw)
        words = len(text.split()) if text else 0
        if text is None or not MIN_WORDS <= words <= MAX_WORDS:
            print(f"skip {book}: {words} words", file=sys.stderr)
            continue
        (CACHE / f"pg{book}.txt").write_bytes(raw)
        chosen.append({"id": book, "title": row["Title"].split("\n")[0][:120], "author": row["Authors"].split(";")[0].strip()[:80], "words": words, "sha256": hashlib.sha256(raw).hexdigest()})
        print(f"{len(chosen):3d} {book:6d} {words:7d} {chosen[-1]['author'][:28]:28s} {chosen[-1]['title'][:60]}", flush=True)
        time.sleep(args.delay)
    SPEC.write_text(json.dumps({"source": "Project Gutenberg", "license": "Public domain in the United States (Project Gutenberg License applies to the files)", "textUrl": TEXT_URL, "selection": "English prose fiction from pg_catalog.csv, lowest ids first, LitBank source novels excluded by id, 40k-260k words after stripping the Gutenberg header and footer", "novels": chosen}, indent=2) + "\n")
    print(f"pinned {len(chosen)} novels in {SPEC}")
    write_corpus(chosen)


def fetch(args):
    spec = json.loads(SPEC.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    for novel in spec["novels"]:
        path = CACHE / f"pg{novel['id']}.txt"
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != novel["sha256"]:
            raw = download(TEXT_URL.format(id=novel["id"]))
            if hashlib.sha256(raw).hexdigest() != novel["sha256"]:
                raise SystemExit(f"pg{novel['id']}.txt does not match the pinned sha256")
            path.write_bytes(raw)
            time.sleep(args.delay)
    write_corpus(spec["novels"])


def write_corpus(novels):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as out:
        for novel in novels:
            text = body((CACHE / f"pg{novel['id']}.txt").read_bytes())
            surname = novel["author"].split(",")[0].strip().lower()
            out.write(json.dumps({"id": str(novel["id"]), "source": "gutenberg", "group": surname, "text": text, "proper": [], "nominal": [], "pronoun": [], "clusters": [], "quotes": [], "token_labels": False, "unlabeled": True}) + "\n")
    print(f"wrote {len(novels)} documents to {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["select", "fetch"])
    parser.add_argument("--novels", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between downloads")
    args = parser.parse_args()
    (select if args.mode == "select" else fetch)(args)


if __name__ == "__main__":
    main()
