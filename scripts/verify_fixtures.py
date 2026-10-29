# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Verify annotation provenance, frozen corpus content, and split identity."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1] / "tests/fixtures"
manifest = json.loads((root / "provenance.json").read_text())
corpus_bytes = (root / manifest["corpus"]).read_bytes()
assert hashlib.sha256(corpus_bytes).hexdigest() == manifest["sha256"]
corpus = json.loads(corpus_bytes)
assert len(corpus["cases"]) == manifest["caseCount"]
ids = set()
splits = {"calibration": set(), "held_out": set()}
for case in corpus["cases"]:
    assert case["id"] not in ids
    ids.add(case["id"])
    text = "\n\x1e\n".join(case["scenes"])
    assert hashlib.sha256(text.encode()).hexdigest() == case["provenance"]["textSha256"]
    assert "agent-authored" in case["provenance"]["annotation"]
    splits[case["split"]].add(text)
assert not (splits["calibration"] & splits["held_out"]), "exact passage reuse across splits"
for split, count in manifest["splits"].items():
    assert sum(case["split"] == split for case in corpus["cases"]) == count
print(f"Verified {len(ids)} annotated cases, hashes, and disjoint split texts")

finding_meta = manifest["findingCorpus"]
finding_bytes = (root / finding_meta["corpus"]).read_bytes()
assert hashlib.sha256(finding_bytes).hexdigest() == finding_meta["sha256"]
finding_corpus = json.loads(finding_bytes)
assert len(finding_corpus["cases"]) == finding_meta["caseCount"]
texts = {"calibration": set(), "held_out": set()}
case_ids = set()
for case in finding_corpus["cases"]:
    assert case["id"] not in case_ids
    case_ids.add(case["id"])
    assert "agent-authored" in case["annotationOrigin"]
    scenes = {s["id"]: s["text"] for s in case["manuscript"]["scenes"]}
    text = "\n\x1e\n".join(scenes.values())
    assert hashlib.sha256(text.encode()).hexdigest() == case["textSha256"]
    if text.strip():
        texts[case["split"]].add(text)
    for expected in case["expected"]:
        if expected["quote"] is not None:
            source = scenes[expected["sceneId"]]
            span = expected["range"]
            assert source.encode()[span["byteStart"]:span["byteEnd"]].decode() == expected["quote"]
            utf16 = source.encode("utf-16-le")
            assert utf16[2 * span["utf16Start"]:2 * span["utf16End"]].decode("utf-16-le") == expected["quote"]
        for related in expected["related"]:
            assert related["quote"] in scenes[related["sceneId"]]
assert not (texts["calibration"] & texts["held_out"])
for split, count in finding_meta["splits"].items():
    assert sum(c["split"] == split for c in finding_corpus["cases"]) == count
assert hashlib.sha256((root.parents[1] / finding_meta["baseline"]).read_bytes()).hexdigest() == finding_meta["baselineSha256"]
assert hashlib.sha256((root / "sarif-schema-2.1.0.json").read_bytes()).hexdigest() == manifest["sarifSchema"]["sha256"]
models = root / "models"
model_meta = json.loads((models / "provenance.json").read_text())
for name, expected in model_meta["files"].items():
    data = (models / name).read_bytes()
    assert len(data) == expected["bytes"]
    assert hashlib.sha256(data).hexdigest() == expected["sha256"]
print(f"Verified {len(case_ids)} finding-level cases and original UTF-8/UTF-16 labels, frozen baseline, official SARIF schema, and {len(model_meta['files'])} local ONNX models")

assert hashlib.sha256((root / finding_meta["baselineCorpus"]).read_bytes()).hexdigest() == finding_meta["baselineCorpusSha256"]
