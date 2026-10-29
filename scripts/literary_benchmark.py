# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prepare and score pinned LitBank labels without changing them or training a model."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PRONOUNS = set("he him his himself she her hers herself they them their theirs themselves".split())


def normalized(text):
    return " ".join(text.split()).casefold()


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


def read_source(cache, entry):
    path = cache / entry["path"]
    if path.stat().st_size != entry["size"]:
        raise ValueError("source length mismatch: " + entry["path"])
    data = path.read_bytes()
    if fingerprint(data) != entry["sha256"]:
        raise ValueError("source digest mismatch: " + entry["path"])
    return data.decode("utf-8")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mentions(text, annotation):
    result = []
    for line in annotation.splitlines():
        if not line.startswith("T"):
            continue
        identifier, fields, quote = line.split("\t", 2)
        label, positions = fields.split(" ", 1)
        spans = [tuple(map(int, part.split())) for part in positions.split(";")]
        if any(not 0 <= start < end <= len(text) for start, end in spans):
            raise ValueError("annotation range invalid")
        if normalized(" ".join(text[start:end] for start, end in spans)) != normalized(quote):
            raise ValueError("annotation quote mismatch")
        result.append({"id": identifier, "label": label, "spans": spans, "quote": quote})
    return result


def coreference(text, annotation):
    tokens = list(re.finditer(r"\S+", text))
    position = 0
    opened = defaultdict(list)
    spans = defaultdict(set)
    for line in annotation.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 13 or position >= len(tokens) or fields[3] != tokens[position].group():
            raise ValueError("CoNLL token/source alignment mismatch")
        token = tokens[position]
        for label in fields[-1].split("|"):
            if not label or label == "_" or label == "-":
                continue
            if label.startswith("(") and label.endswith(")"):
                spans[(token.start(), token.end())].add(label[1:-1])
            elif label.startswith("("):
                opened[label[1:]].append(token.start())
            elif label.endswith(")"):
                group = label[:-1]
                if not opened[group]:
                    raise ValueError("unbalanced coreference annotation")
                spans[(opened[group].pop(), token.end())].add(group)
            else:
                raise ValueError("invalid coreference annotation")
        position += 1
    if position != len(tokens) or any(opened.values()):
        raise ValueError("incomplete coreference alignment")
    return spans


def quotations(text, annotation):
    rows = []
    offset = 0
    for row in text.splitlines(keepends=True):
        rows.append([(offset + m.start(), offset + m.end()) for m in re.finditer(r"\S+", row)])
        offset += len(row)
    quotes, speakers = {}, {}
    for line in annotation.splitlines():
        values = line.split("\t")
        if values[0] == "QUOTE":
            if len(values) != 7:
                raise ValueError("invalid quote row")
            identifier = values[1]
            first, a, last, b = map(int, values[2:6])
            start, end = rows[first][a][0], rows[last][b][1]
            if identifier in quotes or normalized(text[start:end]) != normalized(values[6]):
                raise ValueError("quote/source alignment mismatch")
            inner = text[start:end].strip()
            if len(inner) >= 2 and inner[0] in '\"\u201c\u2018\'' and inner[-1] in '\"\u201d\u2019\'':
                inner = inner[1:-1].strip()
            quotes[identifier] = {"start": start, "end": end, "quote": normalized(inner),
                                  "quotationBytes": [len(text[:start].encode()), len(text[:end].encode())]}
        elif values[0] == "ATTRIB":
            if len(values) != 3 or values[1] in speakers:
                raise ValueError("invalid speaker row")
            speakers[values[1]] = values[2]
        elif line.strip():
            raise ValueError("unsupported quotation row")
    if set(quotes) != set(speakers):
        raise ValueError("quotation speaker references differ")
    for identifier, quote in quotes.items():
        quote["speaker"] = speakers[identifier]
    return list(quotes.values())


def prepare(spec, cache, output):
    files = {entry["path"]: entry for entry in spec["files"]}
    cases, gold = [], []
    split_authors = defaultdict(set)
    for work in sorted(spec["works"], key=lambda x: int(x["id"])):
        prefix = work["id"] + "_"
        def source(folder, suffix):
            selected = [v for key, v in files.items() if key.startswith(folder + "/" + prefix) and key.endswith(suffix)]
            if len(selected) != 1:
                raise ValueError("nonunique source path for " + work["id"])
            return read_source(cache, selected[0])
        text = source("coref/brat", ".txt")
        annotated = mentions(text, source("coref/brat", ".ann"))
        groups = coreference(text, source("coref/conll", ".conll"))
        alias_groups = defaultdict(set)
        names = set()
        for mention in annotated:
            if mention["label"] != "PROP_PER" or len(mention["spans"]) != 1:
                continue
            start, end = mention["spans"][0]
            name = " ".join(text[start:end].split())
            names.add(normalized(name))
            alias_groups[normalized(name)].update(groups.get((start, end), set()))
        forms = defaultdict(Counter)
        for mention in annotated:
            name = " ".join(mention["quote"].split())
            if mention["label"] == "PROP_PER" and len(mention["spans"]) == 1 and len(name.encode()) <= 128 and len(alias_groups[normalized(name)]) == 1:
                forms[normalized(name)][name] += 1
        supplied = sorted(sorted(values, key=lambda name: (-values[name], name))[0] for values in forms.values())
        truncated = len(supplied) > 256
        supplied = supplied[:256]
        proper_groups = {group for mention in annotated if mention["label"] == "PROP_PER" and len(mention["spans"]) == 1 for group in groups.get(tuple(mention["spans"][0]), set())}
        for mention in annotated:
            if mention["label"] == "NOM_PER" and len(mention["spans"]) == 1:
                alias_groups[normalized(mention["quote"])].update(groups.get(tuple(mention["spans"][0]), set()))
        pronouns = []
        for mention in annotated:
            if mention["label"] == "PRON_PER" and len(mention["spans"]) == 1 and normalized(mention["quote"]) in PRONOUNS:
                start, end = mention["spans"][0]
                if not groups.get((start, end)):
                    raise ValueError("person pronoun missing identity cluster")
                pronouns.append({"start": len(text[:start].encode()), "end": len(text[:end].encode()), "groups": sorted(groups[(start, end)])})
        quote_text = source("quotations/tsv", ".txt")
        quote_gold = quotations(quote_text, source("quotations/tsv", ".ann"))
        for quote in quote_gold:
            speaker_name = quote["speaker"].rsplit("-", 1)[0].replace("_", " ")
            groups_for_name = sorted(alias_groups.get(normalized(speaker_name), set()))
            quote["speakerGroups"] = groups_for_name if len(groups_for_name) == 1 else []
            quote["speakerMapping"] = "exact_unique_canonical_name_alias" if quote["speakerGroups"] else "unassessed_identity"
        split = "final_holdout" if int(hashlib.sha256((spec["split"]["seed"] + "\n" + work["author"]).encode()).hexdigest()[:8], 16) % 5 == 0 else "development"
        split_authors[split].add(work["author"])
        cases.append({"id": work["id"], "split": split, "entityText": text, "quotationText": quote_text, "suppliedCast": supplied})
        proper_spans = [[len(text[:start].encode()), len(text[:end].encode())]
                        for mention in annotated if mention["label"] == "PROP_PER" and len(mention["spans"]) == 1
                        for start, end in mention["spans"]]
        gold.append({"id": work["id"], "split": split, "properPersonNames": sorted(names), "properPersonGroups": sorted(proper_groups), "properPersonSpans": proper_spans, "aliasGroups": {k: sorted(v) for k, v in sorted(alias_groups.items())}, "pronouns": pronouns, "quotes": quote_gold, "suppliedCastTruncated": truncated})
    if split_authors["development"] & split_authors["final_holdout"]:
        raise ValueError("author leakage between splits")
    write(output / "inputs.json", cases)
    write(output / "gold.json", gold)
    manifest = {"protocolVersion":4,"revision": spec["revision"], "license": spec["license"], "inputsSha256": fingerprint((output / "inputs.json").read_bytes()), "goldSha256": fingerprint((output / "gold.json").read_bytes()), "works": len(cases), "splits": dict(Counter(c["split"] for c in cases)), "authors": {k: len(v) for k, v in split_authors.items()}, "protocol": "Author-disjoint; original tokenized annotation texts. Supplied-cast mode uses unambiguous proper-name aliases from human labels,never gold pronoun/speaker assignments or gender. Detected-cast mode uses only library-extracted names. Unique-surface names are diagnostic;canonical inventory is scored by unambiguous named-person identity groups,with nominal-only predictions explicitly unassessed. Coreference covers supported personal pronouns with exact original byte ranges. Quotation speaker numeric suffix IDs are a separate annotation namespace;gold speaker canonical names link through exact unique aliases only,otherwise identity is explicitly unassessed with coverage and all-gold lower-bound recall. Legacy quotes use one-to-one normalized content multiset with repeated-content speaker ambiguity unassessed. Checked explicit-only quotes additionally use exact original delimiter boundaries and source-validated UTF8/UTF16 anchors,including repeated content. Pinned model NER is scored separately on contiguous proper-person spans;nominal and nested entity semantics differ from CoNLL/news. No data edits or model training."}
    write(output / "manifest.json", manifest)
    print(json.dumps(manifest))


def counts(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def metric_summary(values):
    result = {**counts(values["tp"], values["fp"], values["fn"]), **{key: value for key, value in values.items() if key not in {"tp", "fp", "fn"}}}
    if values["total"]:
        assessed = values["total"] - values["unassessedRepeatedText"] - values["unassessedIdentity"]
        decided = assessed - values["abstained"]
        result["assessmentCoverage"] = assessed / values["total"]
        result["recallOverAllGoldLowerBound"] = values["tp"] / values["total"]
        result["decisionCoverage"] = decided / values["total"]
        result["decisionCoverageWithinAssessed"] = decided / assessed if assessed else None
    return result


def speaker_groups(gold, quote):
    if "speakerGroups" in quote:
        return quote["speakerGroups"]
    name = quote["speaker"].rsplit("-", 1)[0].replace("_", " ")
    groups = gold["aliasGroups"].get(normalized(name), [])
    return groups if len(groups) == 1 else []


def validate_range(text, location, quote=None):
    start, end = location["byteStart"], location["byteEnd"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text.encode()):
        raise ValueError("invalid predicted byte range")
    raw = text.encode()
    try:
        before, selected = raw[:start].decode(), raw[start:end].decode()
    except UnicodeDecodeError as error:
        raise ValueError("predicted range splits UTF8") from error
    if location["utf16Start"] != len(before.encode("utf-16-le")) // 2 or location["utf16End"] != len((before + selected).encode("utf-16-le")) // 2:
        raise ValueError("predicted UTF16 coordinates disagree")
    if quote is not None and selected != quote:
        raise ValueError("predicted range does not select original quote")
    return start, end


def exact_dialogue_metrics(gold, prediction, text, suffix, metrics):
    key = "dialogueExplicit" + suffix
    if key not in prediction:
        return
    available = defaultdict(list)
    for quote in gold["quotes"]:
        available[tuple(quote["quotationBytes"])].append(quote)
    detected, speakers = metrics[key + "Exact"], metrics["speakerExplicit" + suffix + "Exact"]
    for located in prediction[key]["lines"]:
        if located["kind"] != "direct":
            raise ValueError("explicit-only benchmark emitted indirect speech")
        line = located["line"]
        validate_range(text, located["range"], line["quote"])
        outer = located["quotationRange"] or located["range"]
        bounds = validate_range(text, outer)
        if not (bounds[0] <= located["range"]["byteStart"] < located["range"]["byteEnd"] <= bounds[1]):
            raise ValueError("predicted quote is outside its delimiters")
        if located.get("attributionRange"):
            validate_range(text, located["attributionRange"])
        if not available[bounds]:
            detected.update(fp=1)
            if normalized(line["speaker"]) not in {"", "unknown", "unattributed"}:
                speakers.update(fp=1, unmatchedPredictedAssignments=1)
            continue
        quote = available[bounds].pop(0)
        detected.update(tp=1)
        expected = speaker_groups(gold, quote)
        if not expected:
            speakers.update(total=1, unassessedIdentity=1)
            continue
        assigned = normalized(line["speaker"]) not in {"", "unknown", "unattributed"}
        correct = assigned and gold["aliasGroups"].get(normalized(line["speaker"]), []) == expected
        speakers.update(tp=int(correct), fp=int(assigned and not correct), fn=int(not correct), total=1, abstained=int(not assigned))
    missed = sum(map(len, available.values()))
    detected.update(fn=missed)
    unassessed = sum(not speaker_groups(gold, quote) for values in available.values() for quote in values)
    speakers.update(fn=missed-unassessed, total=missed, abstained=missed-unassessed, unassessedIdentity=unassessed)


def score(folder, predictions, split, output):
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in ["inputs", "gold"]:
        if fingerprint((folder / (name + ".json")).read_bytes()) != manifest[name + "Sha256"]:
            raise ValueError("frozen corpus preparation changed")
    gold = [g for g in json.loads((folder / "gold.json").read_text()) if g["split"] == split]
    inputs = {case["id"]: case for case in json.loads((folder / "inputs.json").read_text())}
    predicted = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    by_id = {p["id"]: p for p in predicted}
    if len(by_id) != len(predicted) or any(g["id"] not in by_id for g in gold):
        raise ValueError("predictions are duplicate or missing works")
    metrics = defaultdict(Counter)
    failures = 0
    for g in gold:
        p = by_id[g["id"]]
        if p["split"] != split:
            raise ValueError("prediction split mismatch")
        failures += int("failure" in p)
        expected, actual = set(g["properPersonNames"]), {normalized(name) for name, _ in p.get("personNames", [])}
        metrics["properPersonNames"].update(tp=len(actual & expected), fp=len(actual - expected), fn=len(expected - actual))
        expected_groups, matched_groups = set(g["properPersonGroups"]), set()
        inventory = metrics["namedCharacterInventory"]
        for name in sorted(actual):
            candidates = set(g["aliasGroups"].get(name, []))
            if candidates and not candidates & expected_groups:
                inventory.update(unassessedNominalOnly=1)
            elif len(candidates) == 1 and not candidates & matched_groups:
                matched_groups.update(candidates)
                inventory.update(tp=1)
            else:
                inventory.update(fp=1)
        inventory.update(fn=len(expected_groups - matched_groups))
        if p.get("modelIdentity"):
            expected_spans = Counter(map(tuple, g["properPersonSpans"]))
            actual_spans = Counter((start, end) for _, label, start, end in p.get("nerSpans", []) if label in {"PER", "B-PER", "I-PER"})
            matches = sum((expected_spans & actual_spans).values())
            metrics["modelProperPersonSpansExact"].update(tp=matches, fp=sum(actual_spans.values()) - matches, fn=sum(expected_spans.values()) - matches)
        for suffix in ["SuppliedCast", "DetectedCast"]:
            if "dialogueExplicit" + suffix in p:
                exact_dialogue_metrics(g, p, inputs[g["id"]]["quotationText"], suffix, metrics)
            actual_mentions = defaultdict(list)
            for mention in p.get("coreference" + suffix, {}).get("mentions", []):
                if mention["isPronoun"]:
                    actual_mentions[(mention["start"], mention["start"] + len(mention["text"].encode()))].append(mention)
            coref = metrics["coreference" + suffix]
            for pronoun in g["pronouns"]:
                decisions = actual_mentions.get((pronoun["start"], pronoun["end"]), [])
                resolved = [m for m in decisions if m.get("resolvedTo")]
                correct = sum(len(g["aliasGroups"].get(normalized(m["resolvedTo"]), [])) == 1 and set(g["aliasGroups"][normalized(m["resolvedTo"])]) == set(pronoun["groups"]) for m in resolved)
                coref.update(tp=int(correct > 0), fp=len(resolved) - int(correct > 0), fn=int(correct == 0), total=1, abstained=int(not resolved))
            available = defaultdict(list)
            for quote in g["quotes"]:
                available[quote["quote"]].append(quote)
            dialogue = metrics["dialogue" + suffix]
            speaker = metrics["speaker" + suffix]
            repeated = {key for key, values in available.items() if len(values) > 1}
            for line in p.get("dialogue" + suffix, []):
                if line.get("method", "").startswith("indirect"):
                    dialogue.update(unassessedIndirect=1)
                    continue
                key = normalized(line["quote"])
                if not available[key]:
                    dialogue.update(fp=1)
                    continue
                quote = available[key].pop(0)
                dialogue.update(tp=1)
                if key in repeated:
                    speaker.update(total=1, unassessedRepeatedText=1)
                    continue
                expected = speaker_groups(g, quote)
                if not expected:
                    speaker.update(total=1, unassessedIdentity=1)
                    continue
                assigned = bool(line["speaker"]) and normalized(line["speaker"]) not in {"unknown", "unattributed"}
                correct = assigned and g["aliasGroups"].get(normalized(line["speaker"]), []) == expected
                speaker.update(tp=int(correct), fp=int(assigned and not correct), fn=int(not correct), total=1, abstained=int(not assigned))
            missed = sum(len(q) for q in available.values())
            dialogue.update(fn=missed)
            unassessed = sum(len(values) for key, values in available.items() if key in repeated)
            unknown_identity = sum(not speaker_groups(g, quote) for key, values in available.items() if key not in repeated for quote in values)
            speaker.update(fn=missed-unassessed-unknown_identity, total=missed, abstained=missed-unassessed-unknown_identity, unassessedRepeatedText=unassessed, unassessedIdentity=unknown_identity)
    modes = sorted({p.get("modelMode", "unreported legacy mode") for p in predicted if p["split"] == split})
    report = {"dataset": manifest, "split": split, "works": len(gold), "executionFailures": failures, "predictionSha256": fingerprint(predictions.read_bytes()), "modelModes": modes, "metrics": {key: metric_summary(values) for key,values in metrics.items()}, "limits": ["Independent published human labels,not blinded fresh editorial annotation or proof of all narrative semantics.", "Unique proper-name and supported-person-pronoun tasks are narrower than full nested NER/CoNLL coreference.", "Supplied cast is an oracle inventory;detected-cast mode measures its separate effect.", "Legacy quotation-content multiset metrics are relaxed;checked explicit-only Exact metrics validate boundaries,source anchors and repeated-content speaker identities separately.", "Model metadata records the actual supplied weights;confidence values remain uncalibrated rule strengths."]}
    write(output, report)
    print(json.dumps({"split": split, "works": len(gold), "executionFailures": failures, "metrics": report["metrics"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "score"])
    parser.add_argument("--folder", type=Path, default=ROOT / "target/external-validation/litbank-prepared")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--split", choices=["development", "final_holdout"], default="development")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "target/evaluation/literary-development.json")
    args = parser.parse_args()
    if args.mode == "prepare":
        spec = json.loads((ROOT / "docs/benchmarks/litbank-source.json").read_text())
        prepare(spec, ROOT / spec["cache"], args.folder)
    else:
        if args.split == "final_holdout" and not args.finalize:
            raise ValueError("final heldout scoring requires a frozen candidate and --finalize")
        if args.predictions is None:
            raise ValueError("--predictions required")
        score(args.folder, args.predictions, args.split, args.output)


if __name__ == "__main__":
    main()
