# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prepare one pinned local NER export; no paid inference or implicit production download."""

import argparse
import json
from pathlib import Path

import component_real_model_reference as resources


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    spec = json.loads((root / "docs/benchmarks/ner-reference.json").read_text())
    cache = root / "target/external-validation/ner-model" / spec["revision"]
    cache.mkdir(parents=True, exist_ok=True)
    source = {"repository":spec["repository"],"revision":spec["revision"]}
    model = resources.fetch(cache, spec["modelPath"], "model.onnx", spec["modelSha256"], 70 * 1024 * 1024, args.download,**source)
    tokenizer = cache / "tokenizer.json"
    if not tokenizer.exists() or resources.digest(tokenizer) != spec["tokenizerSha256"]:
        resources.fetch(cache, "tokenizer.json", "tokenizer.json", spec["tokenizerSha256"], 2 * 1024 * 1024, args.download,**source)
    original = json.loads(tokenizer.read_text())
    derived = dict(original, padding=None, truncation=None)
    adapted = cache / "tokenizer.adapted.json"
    resources.write_json(adapted, derived)
    manifest = {"schemaVersion": 1, "kind": "ner", "profile": "hugging_face_bio9_v1", "modelSha256": spec["modelSha256"], "tokenizerSha256": resources.digest(adapted), "maxTokens": spec["maxTokens"], "paddingTokenId": spec["paddingTokenId"], "labels": spec["labels"]}
    resources.write_json(cache / "manifest.json", manifest)
    report = {"spec": spec, "modelFetch": model, "cache": str(cache), "originalTokenizerSettings": {key: original[key] for key in ["padding", "truncation"]}, "adaptation": "Explicit cache-only padding/truncation-null copy; declared token limit preserved,original source unchanged.", "manifest": manifest}
    report_path = root / "target/evaluation/program-ner-reference-preparation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    resources.write_json(report_path, report)
    print(json.dumps({"cache": str(cache), "bytes": spec["modelBytes"], "modelSha256": spec["modelSha256"], "adaptedTokenizerSha256": manifest["tokenizerSha256"]}))


if __name__ == "__main__":
    main()
