"""Build the unified person-mention corpus the multi-task model trains on.

One JSON document per work in `target/external-validation/narrative-multitask/<source>.jsonl`:

    {"id", "source", "group", "text",
     "proper": [[start, end], ...], "nominal": [[start, end], ...], "pronoun": [[start, end], ...],
     "clusters": [[[start, end], ...], ...],
     "quotes": [{"start", "end", "speaker": <cluster index or -1>}, ...]}

Offsets are character offsets into `text`. `group` is the unit that must stay on one side
of a train/validation split (author for LitBank, novel for PDNC, document for GUM).
LitBank contributes only its development split; the holdout works are never read.

    uv run --no-project scripts/narrative_corpus.py litbank
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import literary_benchmark as lb  # noqa: E402

OUT = ROOT / "target/external-validation/narrative-multitask"


def litbank():
    spec = json.loads((ROOT / "docs/benchmarks/litbank-source.json").read_text())
    cache = ROOT / spec["cache"]
    files = {entry["path"]: entry for entry in spec["files"]}
    gold_split = {g["id"]: g["split"] for g in json.loads((ROOT / "target/external-validation/litbank-prepared-v4/gold.json").read_text())}
    docs, stats = [], defaultdict(int)
    for work in sorted(spec["works"], key=lambda w: int(w["id"])):
        if gold_split[work["id"]] != "development":
            continue
        prefix = work["id"] + "_"

        def source(folder, suffix):
            selected = [v for key, v in files.items() if key.startswith(folder + "/" + prefix) and key.endswith(suffix)]
            assert len(selected) == 1, work["id"]
            return lb.read_source(cache, selected[0])

        text = source("coref/brat", ".txt")
        mentions = [m for m in lb.mentions(text, source("coref/brat", ".ann")) if len(m["spans"]) == 1]
        groups = lb.coreference(text, source("coref/conll", ".conll"))
        kinds = {"PROP_PER": "proper", "NOM_PER": "nominal", "PRON_PER": "pronoun"}
        doc = {"id": work["id"], "source": "litbank", "group": work["author"], "text": text, "proper": [], "nominal": [], "pronoun": [], "clusters": [], "quotes": []}
        by_group, cluster_of = defaultdict(list), {}
        for m in mentions:
            kind = kinds.get(m["label"])
            if kind is None:
                continue
            span = tuple(m["spans"][0])
            doc[kind].append(list(span))
            gids = sorted(groups.get(span, set()))
            if gids:
                by_group[gids[0]].append(list(span))
                cluster_of[span] = gids[0]
            stats[kind] += 1
        group_index = {gid: i for i, gid in enumerate(sorted(by_group))}
        doc["clusters"] = [by_group[gid] for gid in sorted(by_group)]
        assert source("quotations/tsv", ".txt") == text
        for quote in lb.quotations(text, source("quotations/tsv", ".ann")):
            suffix = quote["speaker"].rsplit("-", 1)[-1]
            speaker = group_index.get(suffix, -1)
            doc["quotes"].append({"start": quote["start"], "end": quote["end"], "speaker": speaker})
            stats["quotes"] += 1
            stats["quotes_with_cluster"] += int(speaker >= 0)
        docs.append(doc)
    return docs, dict(stats)


PDNC_REPOSITORY = "https://github.com/Priya22/project-dialogism-novel-corpus"
PDNC_REVISION = "e507a27b89a85c1e9710a897ad0b3f5a9ad73dc4"
PRONOUNS = {"he", "she", "they", "i", "we", "you", "it", "him", "her", "them", "his", "hers", "their", "its"}
SAID_WORDS = {"said", "says", "cried", "asked", "replied", "answered", "exclaimed", "whispered", "muttered", "shouted", "continued", "added", "returned", "observed", "remarked", "went", "on", "began", "called", "inquired", "repeated", "interrupted", "demanded", "declared", "murmured", "sighed", "laughed"}


def fetch_pdnc():
    """Download the pinned PDNC revision once, verify (or record) its archive digest, and unpack it."""
    import hashlib
    import tarfile
    import urllib.request

    spec_path = ROOT / "docs/benchmarks/pdnc-source.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {
        "repository": PDNC_REPOSITORY, "revision": PDNC_REVISION,
        "license": "CC BY 4.0 as stated in the repository ReadMe; novel texts are Project Gutenberg / gutenberg.ca downloads",
        "citation": "Vishnubhotla, Hammond, Hirst. The Project Dialogism Novel Corpus. LREC 2022.",
        "archive": f"https://codeload.github.com/Priya22/project-dialogism-novel-corpus/tar.gz/{PDNC_REVISION}",
        "cache": "target/external-validation/pdnc",
        "holdoutPolicy": "Novels whose LitBank excerpt is in the final_holdout split are excluded from every use so the holdout stays unseen.",
    }
    cache = ROOT / spec["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "pdnc.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(spec["archive"], archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if spec.get("archiveSha256") and spec["archiveSha256"] != digest:
        raise ValueError("PDNC archive digest changed")
    spec["archiveSha256"] = digest
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    root = cache / "src"
    if not root.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
    (top,) = [p for p in root.iterdir() if p.is_dir()]
    return spec, top


def find_all(text, needle):
    at = 0
    while True:
        at = text.find(needle, at)
        if at < 0:
            return
        before = text[at - 1] if at else " "
        after = text[at + len(needle)] if at + len(needle) < len(text) else " "
        if not before.isalnum() and not after.isalnum():
            yield at, at + len(needle)
        at += 1


def pdnc():
    import ast
    import csv

    spec, top = fetch_pdnc()
    authors = {row["Author Code"]: f"{row['Surname(s)']}, {row['Given Name(s)']}" for row in csv.DictReader((top / "PDNC-Author-Index.csv").open(encoding="utf-8"))}
    litbank_spec = json.loads((ROOT / "docs/benchmarks/litbank-source.json").read_text())
    splits = {g["id"]: g["split"] for g in json.loads((ROOT / "target/external-validation/litbank-prepared-v4/gold.json").read_text())}
    holdout_titles = {lb.normalized(w["title"]) for w in litbank_spec["works"] if splits[w["id"]] == "final_holdout"}
    docs, stats, excluded = [], defaultdict(int), []
    for row in csv.DictReader((top / "PDNC-Novel-Index.csv").open(encoding="utf-8")):
        title, folder = row["Novel Title"], row["Folder Name"]
        if lb.normalized(title) in holdout_titles:
            excluded.append(title)
            continue
        base = top / "data" / folder
        text = (base / "novel_text.txt").read_text(encoding="utf-8")
        characters = {}
        for c in csv.DictReader((base / "character_info.csv").open(encoding="utf-8")):
            aliases = {c["Main Name"]} | set(ast.literal_eval(c["Aliases"]) if c["Aliases"].strip() else set())
            characters[c["Main Name"]] = (int(c["Character ID"]), {a for a in aliases if len(a) > 1})
        # alias occurrences become proper mentions; longer matches win on overlap
        candidates = sorted(((s, e, cid) for cid, aliases in characters.values() for a in aliases for s, e in find_all(text, a)), key=lambda x: (x[0], -(x[1] - x[0])))
        proper, last_end = [], -1
        for s, e, cid in candidates:
            if s >= last_end:
                proper.append((s, e, cid)); last_end = e
        cum = [0]
        for ch in text:
            cum.append(cum[-1] + len(ch.encode()))
        import bisect

        def to_char(b):
            return bisect.bisect_left(cum, b)

        quotes, pronoun = [], []
        rows = list(csv.DictReader((base / "quotation_info.csv").open(encoding="utf-8")))
        byte_offsets = None
        for q in rows:
            spans = ast.literal_eval(q["quoteByteSpans"])
            if not spans:
                continue
            if byte_offsets is None:
                s, e = spans[0]
                piece = q["quoteText"].strip()[:20]
                byte_offsets = text.encode()[s:e].decode(errors="ignore").strip().startswith(piece[:10]) and not text[s:e].strip().startswith(piece[:10])
            s = to_char(spans[0][0]) if byte_offsets else spans[0][0]
            e = to_char(spans[-1][1]) if byte_offsets else spans[-1][1]
            speaker = characters.get(q["speaker"], (None, None))[0]
            cluster = speaker if speaker is not None else -1
            quotes.append({"start": s, "end": e, "speaker": cluster})
            stats["quotes"] += 1
            stats["quotes_with_speaker"] += int(cluster >= 0)
            expression = q["referringExpression"].strip()
            if cluster >= 0 and expression and q["quoteType"] in ("Explicit", "Anaphoric"):
                lo, hi = max(0, s - 400), min(len(text), e + 400)
                hits = [p for p in find_all(text[lo:hi], expression)]
                if hits:
                    hs, he = min(hits, key=lambda p: min(abs(lo + p[0] - e), abs(lo + p[1] - s)))
                    hs, he = lo + hs, lo + he
                    words = [(m.start() + hs, m.end() + hs) for m in re.finditer(r"[A-Za-z']+", text[hs:he])]
                    aliases = characters[q["speaker"]][1]
                    if not any(hs <= ps and pe <= he and cid == cluster for ps, pe, cid in proper):
                        pron = [(ws, we) for ws, we in words if text[ws:we].lower() in PRONOUNS]
                        if pron:
                            pronoun.append((pron[0][0], pron[0][1], cluster))
                            stats["anaphoric_speaker_pronouns"] += 1
        by_cluster = defaultdict(list)
        for s, e, cid in proper + pronoun:
            by_cluster[cid].append([s, e])
        index = {cid: i for i, cid in enumerate(sorted(by_cluster))}
        for q in quotes:
            q["speaker"] = index.get(q["speaker"], -1)
        docs.append({"id": row["Novel Code"], "source": "pdnc", "group": authors[row["Author Code"]], "text": text,
                     "proper": [[s, e] for s, e, _ in proper], "nominal": [], "pronoun": [[s, e] for s, e, _ in pronoun],
                     "clusters": [by_cluster[cid] for cid in sorted(by_cluster)], "quotes": quotes, "token_labels": False})
        stats["proper_alias_mentions"] += len(proper)
    spec["excludedNovels"] = excluded
    (ROOT / "docs/benchmarks/pdnc-source.json").write_text(json.dumps(spec, indent=2) + "\n")
    stats["excluded_holdout_overlap"] = len(excluded)
    return docs, dict(stats)


def write(name, docs, stats):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(json.dumps({"source": name, "documents": len(docs), "groups": len({d["group"] for d in docs}), "clusters": sum(len(d["clusters"]) for d in docs), **stats}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["litbank", "pdnc", "gum"])
    args = parser.parse_args()
    builders = {"litbank": litbank, "pdnc": pdnc}
    if args.source not in builders:
        raise SystemExit(f"{args.source}: loader not written yet")
    write(args.source, *builders[args.source]())


if __name__ == "__main__":
    main()
