"""Human review of the narrative model's decisions: evaluation excerpts and hard-case sheets.

    uv run --no-project scripts/review_sheet.py evalset --count 10 --words 2000
        Picks dialogue-dense excerpts from the pinned Gutenberg novels (authors outside
        LitBank and PDNC) and writes review/evalset-inputs.json in the evaluator's input
        format. Run the evaluator on it, then:
    uv run --no-project scripts/review_sheet.py sheet --inputs review/evalset-inputs.json --predictions review/evalset.jsonl --output review/evalset-sheet.csv
        One row per quotation and per pronoun with the surrounding text, the model's
        answer, and empty `correct` / `corrected_answer` columns to fill in (Y, or N plus
        the right character name; leave the name empty for "no antecedent" / "unknown").
    uv run --no-project scripts/review_sheet.py sheet ... --max-confidence 0.6 --limit 500
        The same sheet restricted to the model's least confident decisions (targeted
        correction batches).
    uv run --no-project scripts/review_sheet.py score --sheet review/evalset-sheet.csv --predictions new-run.jsonl
        Accuracy of any later run against the reviewed answers.
    uv run --no-project scripts/review_sheet.py import --sheet review/evalset-sheet.csv --inputs review/evalset-inputs.json --output target/external-validation/narrative-multitask/review.jsonl
        Reviewed answers as a gold document source for multitask_train.py (names become
        clusters; reviewed pronouns and quotations carry their cluster).
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "target/external-validation/review"
GUTENBERG = ROOT / "target/external-validation/narrative-multitask/gutenberg.jsonl"
CORPUS = ROOT / "target/external-validation/narrative-multitask"
QUOTE_MARKS = re.compile(r"[“\"]")


def normalized(text):
    return " ".join(text.split()).lower()


def surnames(source):
    path = CORPUS / f"{source}.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["group"].split(",")[0].strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def best_window(text, words):
    """The `words`-word stretch with the most quotation marks, starting at a paragraph break."""
    starts = [0] + [m.end() for m in re.finditer(r"\n\s*\n", text)]
    best, best_count = None, -1
    for start in starts:
        end = start
        for _ in range(words):
            m = re.compile(r"\S+\s*").match(text, end)
            if not m:
                break
            end = m.end()
        if end - start < words * 3:
            continue
        count = len(QUOTE_MARKS.findall(text[start:end]))
        if count > best_count:
            best, best_count = (start, end), count
    return best


def titled_novels(titles):
    """Gutenberg texts for requested titles (lowest matching English id), downloaded on demand."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import gutenberg_corpus as gc  # noqa: E402

    rows = []
    with (gc.CACHE / "pg_catalog.csv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["Type"] == "Text" and r["Language"] == "en"]
    novels = []
    for request in titles:
        title, _, author = (part.strip() for part in request.partition("|"))
        wanted = normalized(title)
        candidates = [r for r in rows if not author or normalized(author) in normalized(r["Authors"])]
        exact = [r for r in candidates if normalized(r["Title"]).split("\n")[0].rstrip(".") == wanted.rstrip(".")]
        matches = sorted(exact or (r for r in candidates if normalized(r["Title"]).startswith(wanted)), key=lambda r: int(r["Text#"]))
        if not matches:
            print(f"  no Gutenberg text titled {title!r}")
            continue
        row = matches[0]
        path = gc.CACHE / f"pg{row['Text#']}.txt"
        if not path.exists():
            try:
                path.write_bytes(gc.download(gc.TEXT_URL.format(id=row["Text#"])))
            except Exception as error:  # noqa: BLE001
                print(f"  {title!r} ({row['Text#']}): {error}")
                continue
        text = gc.body(path.read_bytes())
        if not text:
            print(f"  {title!r} ({row['Text#']}): header/footer not found")
            continue
        novels.append({"id": row["Text#"], "group": row["Authors"].split(",")[0].strip().lower(), "title": row["Title"].split("\n")[0], "text": text})
    return novels


def local_texts(folder):
    """Your own excerpts (one .txt each; not stored in the repository) as evaluation cases."""
    novels = []
    for path in sorted(Path(folder).glob("*.txt")):
        novels.append({"id": path.stem, "group": path.stem.split("-")[0].lower(), "title": path.stem, "text": path.read_text(encoding="utf-8")})
    return novels


def evalset(args):
    taken = surnames("litbank") | surnames("pdnc")
    requested = (titled_novels(args.titles) if args.titles else []) + (local_texts(args.texts) if args.texts else [])
    novels = [json.loads(line) for line in GUTENBERG.read_text(encoding="utf-8").splitlines() if line.strip()] if GUTENBERG.exists() else []
    ranked = []
    for novel in novels:
        if novel["group"] in taken:
            continue
        density = len(QUOTE_MARKS.findall(novel["text"])) / max(1, len(novel["text"].split()))
        ranked.append((density, novel))
    ranked.sort(key=lambda pair: -pair[0])
    cases, authors = [], set()
    for novel in requested + [novel for _, novel in ranked]:
        if novel["group"] in authors:
            continue
        window = best_window(novel["text"], args.words) if len(novel["text"].split()) > args.words * 1.5 else (0, len(novel["text"]))
        if window is None:
            continue
        excerpt = novel["text"][window[0]:window[1]].strip()
        density = round(len(QUOTE_MARKS.findall(excerpt)) / max(1, len(excerpt.split())), 4)
        note = " (author also in LitBank/PDNC training data)" if novel["group"] in taken else ""
        cases.append({"id": str(novel["id"]), "split": "development", "entityText": excerpt, "quotationText": excerpt, "suppliedCast": [], "group": novel["group"], "quoteDensity": density, "title": novel.get("title", ""), "note": note})
        authors.add(novel["group"])
        if len(cases) >= args.count:
            break
    REVIEW.mkdir(parents=True, exist_ok=True)
    out = REVIEW / "evalset-inputs.json"
    out.write_text(json.dumps([{k: v for k, v in c.items() if k in ("id", "split", "entityText", "quotationText", "suppliedCast")} for c in cases], indent=1))
    (REVIEW / "evalset-selection.json").write_text(json.dumps([{k: c[k] for k in ("id", "group", "title", "quoteDensity", "note")} for c in cases], indent=1) + "\n")
    print(f"{len(cases)} excerpts -> {out}")
    for c in cases:
        print(f"  {c['id']:>8} {c['group']:16s} {c['title'][:40]:40s} quote density {c['quoteDensity']}{c['note']}")


def normalize_space(text):
    """Collapse any run of whitespace (Gutenberg's hard-wrapped line breaks included) to one space."""
    return re.sub(r"\s+", " ", text).strip()


def context(text, start, end, width=160):
    lo, hi = max(0, start - width), min(len(text), end + width)
    return normalize_space(text[lo:start]) + " ⟦" + normalize_space(text[start:end]) + "⟧ " + normalize_space(text[end:hi])


def rows_from_predictions(inputs, predictions, key_suffix):
    rows = []
    for p in predictions:
        case = inputs[p["id"]]
        text = case["entityText"]
        for line in p.get("dialogueExplicit" + key_suffix, {}).get("lines", []):
            r = line["range"]
            start, end = r["byteStart"], r["byteEnd"]
            bstart, bend = len(text.encode()[:start].decode("utf-8", "ignore")), len(text.encode()[:end].decode("utf-8", "ignore"))
            speaker = normalize_space(line["line"]["speaker"])
            rows.append({"excerpt": p["id"], "kind": "quote", "start": start, "end": end, "text": normalize_space(text[bstart:bend])[:120], "context": context(text, bstart, bend),
                         "model_answer": "" if speaker in ("Unknown", "") else speaker, "confidence": f"{line['line']['confidence']:.3f}", "method": line["line"]["method"]})
        for mention in p.get("coreference" + key_suffix, {}).get("mentions", []):
            if not mention.get("isPronoun"):
                continue
            start = mention["start"]
            end = start + len(mention["text"].encode())
            bstart = len(text.encode()[:start].decode("utf-8", "ignore"))
            bend = bstart + len(mention["text"])
            rows.append({"excerpt": p["id"], "kind": "pronoun", "start": start, "end": end, "text": normalize_space(text[bstart:bend]), "context": context(text, bstart, bend),
                         "model_answer": normalize_space(mention.get("resolvedTo") or ""), "confidence": f"{mention.get('confidence', 0):.3f}", "method": mention.get("resolutionBasis") or ""})
    return rows


def sheet(args):
    inputs = {c["id"]: c for c in json.loads(Path(args.inputs).read_text())}
    predictions = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    rows = rows_from_predictions(inputs, predictions, "DetectedCast")
    if args.max_confidence is not None:
        rows = [r for r in rows if float(r["confidence"]) <= args.max_confidence]
        rows.sort(key=lambda r: float(r["confidence"]))
    if args.limit:
        rows = rows[: args.limit]
    fields = ["excerpt", "kind", "start", "end", "text", "context", "model_answer", "confidence", "method", "correct", "corrected_answer"]
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "correct": "", "corrected_answer": ""})
    kinds = Counter(r["kind"] for r in rows)
    print(f"{len(rows)} rows ({kinds['quote']} quotations, {kinds['pronoun']} pronouns) -> {args.output}")


def gold_from_sheet(path):
    """(excerpt, kind, start, end) -> gold name ('' means no answer) for every reviewed row."""
    gold = {}
    skipped = 0
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            verdict = r["correct"].strip().upper()
            if verdict == "Y":
                answer = r["model_answer"]
            elif verdict == "N":
                answer = r["corrected_answer"].strip()
            else:
                skipped += 1
                continue
            gold[(r["excerpt"], r["kind"], int(r["start"]), int(r["end"]))] = answer
    return gold, skipped


def score(args):
    gold, skipped = gold_from_sheet(args.sheet)
    inputs = {c["id"]: c for c in json.loads(Path(args.inputs).read_text())} if args.inputs else None
    predictions = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    if inputs is None:
        inputs = {p["id"]: {"entityText": ""} for p in predictions}
    answers = {(r["excerpt"], r["kind"], r["start"], r["end"]): r["model_answer"] for r in rows_from_predictions(inputs, predictions, "DetectedCast")}
    report = {}
    for kind in ("quote", "pronoun"):
        keys = [k for k in gold if k[1] == kind]
        assessed = [k for k in keys if gold[k]]
        decided = [k for k in assessed if answers.get(k)]
        correct = sum(normalized(answers[k]) == normalized(gold[k]) for k in decided)
        report[kind] = {"reviewed": len(keys), "withGoldAnswer": len(assessed), "decided": len(decided), "correct": correct,
                        "precision": correct / len(decided) if decided else None, "recall": correct / len(assessed) if assessed else None,
                        "abstained": len(assessed) - len(decided), "falseAnswersWhereGoldHasNone": sum(bool(answers.get(k)) for k in keys if not gold[k])}
    report["unreviewedRows"] = skipped
    print(json.dumps(report, indent=2))


def import_gold(args):
    """Reviewed names become clusters; only reviewed pronouns and quotations carry gold."""
    gold, _ = gold_from_sheet(args.sheet)
    inputs = {c["id"]: c for c in json.loads(Path(args.inputs).read_text())}
    by_excerpt = defaultdict(list)
    for (excerpt, kind, start, end), answer in gold.items():
        by_excerpt[excerpt].append((kind, start, end, answer))
    docs = []
    for excerpt, items in by_excerpt.items():
        text = inputs[excerpt]["entityText"]
        raw = text.encode()
        to_char = lambda b: len(raw[:b].decode("utf-8", "ignore"))
        names = sorted({normalized(a) for _, _, _, a in items if a})
        index = {name: i for i, name in enumerate(names)}
        clusters = defaultdict(list)
        proper, pronoun, quotes = [], [], []
        for name in names:
            for m in re.finditer(re.escape(name), text, flags=re.IGNORECASE):
                proper.append([m.start(), m.end()])
                clusters[index[name]].append([m.start(), m.end()])
        for kind, start, end, answer in items:
            s, e = to_char(start), to_char(end)
            cluster = index[normalized(answer)] if answer else -1
            if kind == "pronoun":
                pronoun.append([s, e])
                if cluster >= 0:
                    clusters[cluster].append([s, e])
            else:
                quotes.append({"start": s, "end": e, "speaker": cluster})
        docs.append({"id": f"review-{excerpt}", "source": "review", "group": f"review-{excerpt}", "text": text, "proper": proper, "nominal": [], "pronoun": pronoun,
                     "clusters": [clusters[i] for i in range(len(names))], "quotes": quotes, "token_labels": False})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as out:
        for doc in docs:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"{len(docs)} reviewed documents, {sum(len(d['quotes']) for d in docs)} quotations, {sum(len(d['pronoun']) for d in docs)} pronouns -> {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["evalset", "sheet", "score", "import"])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--words", type=int, default=2000)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-confidence", type=float)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--titles", nargs="*", help="evalset: Gutenberg titles to include first (public domain)")
    parser.add_argument("--texts", type=Path, help="evalset: folder of your own .txt excerpts to include (kept out of the repository)")
    args = parser.parse_args()
    {"evalset": evalset, "sheet": sheet, "score": score, "import": import_gold}[args.mode](args)


if __name__ == "__main__":
    main()
