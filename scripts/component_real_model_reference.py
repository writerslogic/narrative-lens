# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==2.4.6", "onnxruntime==1.27.0", "tokenizers==0.23.3"]
# ///
"""Optional, pinned MiniLM kernel/profile check; no editorial-quality claim.

CARGO_TARGET_DIR=/your/ignored/target uv run --python 3.11 \
  scripts/component_real_model_reference.py --download --prepare-only

Run the Rust example on the printed cache directory, then use --native-results
and optionally --wasi-results to compare. Model weights remain in the cache.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time
import urllib.request

REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
MODEL_BYTES = 90_405_214
MAX_DOWNLOAD_BYTES = 120 * 1024 * 1024
MAX_TOKENS = 128
TOLERANCES = {
    "native": {"atol": 2e-5, "rtol": 2e-4, "maxCosineDistance": 1e-6},
    "wasi": {"atol": 2e-4, "rtol": 2e-3, "maxCosineDistance": 1e-5},
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def fetch(cache, source_name, destination, expected, size_limit, allow_download, *, repository=None, revision=None):
    repository = repository or REPOSITORY
    revision = revision or REVISION
    path = cache / destination
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"cached {destination} digest mismatch")
        return {"source": "existing verification cache", "downloadedBytes": 0}
    upstream_cache = (
        Path.home() / ".cache/huggingface/hub"
        / ("models--" + repository.replace("/", "--")) / "snapshots"
        / revision / source_name
    )
    if upstream_cache.exists() and digest(upstream_cache) == expected:
        shutil.copyfile(upstream_cache, path)
        return {"source": "verified existing Hugging Face cache", "downloadedBytes": 0}
    if not allow_download:
        raise FileNotFoundError(f"missing {destination}; opt into bounded download with --download")
    temporary = path.with_suffix(path.suffix + ".download")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{source_name}"
    received = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            if int(response.headers.get("Content-Length", "0")) > size_limit:
                raise ValueError("upstream artifact exceeds download limit")
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > size_limit:
                    raise ValueError("upstream artifact exceeds download limit")
                output.write(chunk)
        if digest(temporary) != expected:
            raise ValueError(f"downloaded {destination} digest mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source": url, "downloadedBytes": received}



def proto_fields(data):
    """Bounded field inspection, not an ONNX validator; ORT validates the model."""
    position = 0
    def varint():
        nonlocal position
        result = 0
        for shift in range(0, 70, 7):
            if position >= len(data):
                raise ValueError("truncated protobuf varint")
            byte = data[position]
            position += 1
            result |= (byte & 127) << shift
            if byte < 128:
                return result
        raise ValueError("oversized protobuf varint")
    while position < len(data):
        tag = varint()
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError("invalid protobuf field")
        if wire == 0:
            value = varint()
        else:
            size = varint() if wire == 2 else {1: 8, 5: 4}.get(wire)
            if size is None or position + size > len(data):
                raise ValueError("invalid protobuf wire field")
            value = data[position:position + size]
            position += size
        yield number, wire, value


def model_metadata(path):
    # Field numbers follow the primary ONNX v1.17.0 onnx.proto3 schema.
    data = memoryview(path.read_bytes())
    graph = None
    metadata = {"schemaSource": "https://raw.githubusercontent.com/onnx/onnx/v1.17.0/onnx/onnx.proto3", "opsets": []}
    for field, wire, value in proto_fields(data):
        if field == 1:
            metadata["irVersion"] = value
        elif field in [2, 3]:
            metadata[{2: "producer", 3: "producerVersion"}[field]] = bytes(value).decode()
        elif field == 7:
            graph = value
        elif field == 8:
            fields = {f: v for f, _, v in proto_fields(value)}
            metadata["opsets"].append({"domain": bytes(fields.get(1, b"")).decode(), "version": fields.get(2)})
    if graph is None:
        raise ValueError("model has no graph")
    dtypes, stored, operations = Counter(), Counter(), Counter()
    external = 0
    for field, _, value in proto_fields(graph):
        if field == 5:
            dtype = None
            byte_count = 0
            is_external = False
            for tensor_field, _, tensor_value in proto_fields(value):
                if tensor_field == 2:
                    dtype = {1: "float32", 7: "int64"}.get(tensor_value, f"onnx_dtype_{tensor_value}")
                elif tensor_field in [4, 5, 7, 9, 10, 11] and not isinstance(tensor_value, int):
                    byte_count += len(tensor_value)
                elif tensor_field == 13 or (tensor_field == 14 and tensor_value != 0):
                    is_external = True
            dtypes[dtype] += 1
            stored[dtype] += byte_count
            external += int(is_external)
        elif field == 1:
            fields = {f: v for f, _, v in proto_fields(value) if f in [4, 7]}
            operations[bytes(fields.get(7, b"")).decode() + ":" + bytes(fields[4]).decode()] += 1
    if external or set(dtypes) - {"float32", "int64"} or not dtypes["float32"]:
        raise ValueError(f"selected graph has external/unsupported initializer data: {dtypes}, external={external}")
    metadata.update({"initializerDtypeCounts": dict(dtypes), "initializerStoredBytesByDtype": dict(stored), "externalInitializers": external, "operatorCounts": dict(sorted(operations.items())), "scope": "Graph initializer types/external flags and node op names inspected; full ONNX validity/inference independently checked by Python ORT."})
    return metadata


def prepare(cache, allow_download):
    cache.mkdir(parents=True, exist_ok=True)
    model_fetch = fetch(cache, "onnx/model.onnx", "model.onnx", MODEL_SHA256, MODEL_BYTES, allow_download)
    tokenizer_fetch = fetch(cache, "tokenizer.json", "tokenizer.upstream.json", TOKENIZER_SHA256, 1024 * 1024, allow_download)
    if model_fetch["downloadedBytes"] + tokenizer_fetch["downloadedBytes"] > MAX_DOWNLOAD_BYTES:
        raise ValueError("download budget exceeded")
    original = json.loads((cache / "tokenizer.upstream.json").read_text(encoding="utf-8"))
    adapted = dict(original)
    adapted["truncation"] = None
    adapted["padding"] = None
    write_json(cache / "tokenizer.adapted.json", adapted)
    adapted_sha = digest(cache / "tokenizer.adapted.json")
    manifests = {}
    for variant, sha in [("upstream", TOKENIZER_SHA256), ("adapted", adapted_sha)]:
        manifest = {"schemaVersion": 1, "kind": "embedding", "profile": "hugging_face_masked_mean_v1", "modelSha256": MODEL_SHA256, "tokenizerSha256": sha, "maxTokens": MAX_TOKENS, "paddingTokenId": 0, "labels": []}
        write_json(cache / f"manifest.{variant}.json", manifest)
        manifests[variant] = manifest
    cases = {
        "texts": [
            "The small dog rests beside the warm fire.",
            "A little dog sleeps near a bright fireplace.",
            "The telescope measured distant galaxies.",
            "Café naïve résumé — Straße, İstanbul, 你好 👩🏽‍🚀.",
            "First line.\r\nSecond line\tkeeps its spacing.",
            "",
            "Moonlight.",
            "At dawn the patient gardener opened the greenhouse. She watered the young plants, checked the soil, and wrote a short note about the weather before walking home through the quiet village.",
            " ".join(["cat"] * 125),
            " ".join(["cat"] * 126),
        ],
        "batches": [[0, 1, 2], [3, 4, 5, 6, 7], [0], [7], [0, 7], [6, 7], [8], [9]],
        "overLimitText": " ".join(["cat"] * 127),
    }
    write_json(cache / "cases.json", cases)
    identity = {
        "repository": REPOSITORY, "revision": REVISION,
        "license": "apache-2.0 (pinned upstream model-card metadata; not a separate license audit)",
        "primaryModelCard": f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md",
        "modelSource": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/onnx/model.onnx",
        "tokenizerSource": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/tokenizer.json",
        "modelBytes": (cache / "model.onnx").stat().st_size,
        "modelSha256": MODEL_SHA256, "upstreamTokenizerSha256": TOKENIZER_SHA256,
        "adaptedTokenizerSha256": adapted_sha,
        "upstreamTokenizerSettings": {key: original[key] for key in ["padding", "truncation"]},
        "adaptation": "Only set tokenizer JSON padding and truncation to null; reserialize UTF-8 JSON. Unmodified upstream tokenizer remains incompatible with strict v1. Both manifests use maxTokens128, matching upstream tokenizer limit; overflow must error, not truncate.",
        "fetches": {"model": model_fetch, "tokenizer": tokenizer_fetch},
        "onnxMetadata": model_metadata(cache / "model.onnx"),
        "manifests": manifests,
    }
    write_json(cache / "identity.json", identity)
    return cases, identity


def reference(cache, cases):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    started = time.perf_counter()
    session = ort.InferenceSession(str(cache / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - started) * 1000
    inputs = [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_inputs()]
    outputs = [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_outputs()]
    if len(outputs) != 1 or outputs[0]["type"] != "tensor(float)":
        raise ValueError("selected primary export is not the one-output float32 embedding profile")
    variants = {}
    for variant in ["upstream", "adapted"]:
        tokenizer = Tokenizer.from_file(str(cache / f"tokenizer.{variant}.json"))
        batches = []
        for indices in cases["batches"]:
            encodings = [tokenizer.encode(cases["texts"][index], add_special_tokens=True) for index in indices]
            length = max(len(e.ids) for e in encodings)
            ids = np.array([e.ids + [0] * (length - len(e.ids)) for e in encodings], dtype=np.int64)
            masks = np.array([e.attention_mask + [0] * (length - len(e.ids)) for e in encodings], dtype=np.int64)
            types = np.array([e.type_ids + [0] * (length - len(e.ids)) for e in encodings], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": masks, "token_type_ids": types}
            started = time.perf_counter()
            hidden = session.run(None, {item["name"]: feed[item["name"]] for item in inputs})[0]
            if hidden.dtype != np.float32 or hidden.shape != (len(indices), length, 384):
                raise ValueError(f"unexpected output {hidden.dtype} {hidden.shape}")
            pooled = (hidden.astype(np.float64) * masks[:, :, None]).sum(axis=1) / masks.sum(axis=1)[:, None]
            batches.append({"indices": indices, "ids": ids.tolist(), "masks": masks.tolist(), "types": types.tolist(), "vectors": pooled.tolist(), "runtimeMs": (time.perf_counter() - started) * 1000})
        variants[variant] = batches
    padding_deltas = [float(np.max(np.abs(np.array(a["vectors"]) - np.array(b["vectors"])))) for a, b in zip(variants["upstream"], variants["adapted"])]
    result = {"modelSha256": MODEL_SHA256, "adaptedTokenizerSha256": digest(cache / "tokenizer.adapted.json"), "casesSha256": digest(cache / "cases.json"), "runtime": "Python ONNX Runtime CPUExecutionProvider", "python": platform.python_version(), "versions": {name: importlib.metadata.version(name) for name in ["onnxruntime", "tokenizers", "numpy"]}, "buildInfo": ort.get_build_info(), "providers": session.get_providers(), "loadMs": load_ms, "inputs": inputs, "outputs": outputs, "adaptedBatches": variants["adapted"], "upstreamFixedPaddingVsAdaptedMaxAbs": max(padding_deltas)}
    write_json(cache / "reference.json", result)
    return result


def compare(result, reference_result, name):
    import numpy as np
    tolerance = TOLERANCES[name]
    summary = {"status": result["status"], "runtime": result["runtime"], "runtimeVersion": result.get("runtimeVersion"), "ortCrateVersions": result.get("ortCrateVersions"), "tractOnnxVersions": result.get("tractOnnxVersions"), "tokenizersLockVersions": result.get("tokenizersLockVersions"), "inputObservationScope": result.get("inputObservationScope"), "buildInfo": result.get("buildInfo"), "loadMs": result.get("loadMs"), "identity": result.get("identity"), "failure": result.get("failure"), "diagnostic": result.get("diagnostic"), "tolerance": tolerance}
    if result["status"] != "ok":
        summary["passed"] = False
        return summary
    token_equal = True
    finite = True
    max_error = 0.0
    max_cosine_distance = 0.0
    all_close = True
    runtimes = []
    if len(result["batches"]) != len(reference_result["adaptedBatches"]):
        raise ValueError("result batch count mismatch")
    for actual, expected in zip(result["batches"], reference_result["adaptedBatches"]):
        token_equal &= all(actual[key] == expected[key] for key in ["indices", "ids", "masks", "types"])
        a, b = np.array(actual["vectors"]), np.array(expected["vectors"])
        if a.shape != b.shape:
            raise ValueError("result vector shape mismatch")
        finite &= bool(np.isfinite(a).all())
        max_error = max(max_error, float(np.max(np.abs(a - b))))
        cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        max_cosine_distance = max(max_cosine_distance, float(np.max(1 - cosine)))
        all_close &= bool(np.allclose(a, b, atol=tolerance["atol"], rtol=tolerance["rtol"]))
        runtimes.append(actual["runtimeMs"])
    # Same short text embedded alone and with a longer companion.
    singles = {tuple(batch["indices"]): np.array(batch["vectors"]) for batch in result["batches"]}
    batch_delta = float(np.max(np.abs(singles[(0,)][0] - singles[(0, 7)][0])))
    batch_close = bool(np.allclose(singles[(0,)][0], singles[(0, 7)][0], atol=tolerance["atol"], rtol=tolerance["rtol"]))
    vectors = np.array(result["batches"][0]["vectors"])
    cosine = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    boundary = result.get("overLimitFailure", {})
    passed = token_equal and finite and all_close and batch_close and max_cosine_distance <= tolerance["maxCosineDistance"] and boundary.get("code") == "model_profile_token_limit"
    summary.update({"passed": passed, "tokenIdsMasksAndTypesExactlyMatch": token_equal, "allVectorsFinite": finite, "vectorsCompared": sum(len(batch["vectors"]) for batch in result["batches"]), "dimensions": 384, "maxAbsoluteError": max_error, "maxCosineDistance": max_cosine_distance, "allClose": all_close, "shortTextAloneVsLongerBatchMaxAbs": batch_delta, "batchPaddingAllClose": batch_close, "runtimeMsByBatch": runtimes, "overLimitFailure": boundary, "engineeringSanityOnly": {"relatedCosine": cosine(vectors[0], vectors[1]), "unrelatedCosine": cosine(vectors[0], vectors[2]), "notEditorialValidation": True}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "component-real-model" / REVISION)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-reference", action="store_true")
    parser.add_argument("--native-results", type=Path)
    parser.add_argument("--wasi-results", type=Path)
    parser.add_argument("--upstream-native-results", type=Path)
    parser.add_argument("--upstream-wasi-results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cache = args.cache_dir.resolve()
    if repo in cache.parents and repo / "target" not in cache.parents:
        raise ValueError("keep weights under ignored target directories, outside production sources/docs")
    cases, identity = prepare(cache, args.download)
    if args.prepare_only:
        print(json.dumps({"cacheDirectory": str(cache), "identity": identity}, indent=2))
        return
    if args.reuse_reference:
        ref = json.loads((cache / "reference.json").read_text(encoding="utf-8"))
        expected = {"modelSha256": MODEL_SHA256, "adaptedTokenizerSha256": digest(cache / "tokenizer.adapted.json"), "casesSha256": digest(cache / "cases.json")}
        if any(ref.get(key) != value for key, value in expected.items()):
            raise ValueError("cached reference identity/cases mismatch; rerun without --reuse-reference")
    else:
        ref = reference(cache, cases)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(cache / "tokenizer.adapted.json"))
    counts = [len(tokenizer.encode(text, add_special_tokens=True).ids) for text in cases["texts"]]
    overflow = len(tokenizer.encode(cases["overLimitText"], add_special_tokens=True).ids)
    if counts[-2:] != [127, 128] or overflow != 129:
        raise ValueError("probe token-boundary assumptions changed")
    original = json.loads((cache / "tokenizer.upstream.json").read_text(encoding="utf-8"))
    adapted = json.loads((cache / "tokenizer.adapted.json").read_text(encoding="utf-8"))
    only_settings_changed = adapted == dict(original, padding=None, truncation=None)
    if not only_settings_changed:
        raise ValueError("adaptation changed tokenizer semantics beyond declared settings")
    preprocessing = {"allCaseTokenCounts": counts, "overLimitTokenCount": overflow, "maxTokens": MAX_TOKENS, "onlyDeclaredTokenizerSettingsChanged": only_settings_changed, "specialTokensIncludedInMaskedMean": True, "poolingAccumulation": "float64 over float32 token embeddings; no L2 normalization", "batchShapes": [{"indices": batch["indices"], "shape": [len(batch["ids"]), len(batch["ids"][0])], "nonpaddingTokens": [sum(row) for row in batch["masks"]]} for batch in ref["adaptedBatches"]]}
    summary = {"identity": identity, "preprocessing": preprocessing, "reference": {key: value for key, value in ref.items() if key != "adaptedBatches"}, "predeclaredTolerances": TOLERANCES, "comparisons": {}, "upstreamStrictRejections": {}, "scope": "One official float32 MiniLM ONNX export; kernel/tokenizer/masking/profile checks only. Adapted tokenizer is separately pinned and never substituted silently. No editorial-accuracy or broad model-compatibility claim."}
    for name, result_path in [("native", args.native_results), ("wasi", args.wasi_results)]:
        if result_path:
            summary["comparisons"][name] = compare(json.loads(result_path.read_text(encoding="utf-8")), ref, name)
    for name, result_path in [("native", args.upstream_native_results), ("wasi", args.upstream_wasi_results)]:
        if result_path:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            summary["upstreamStrictRejections"][name] = {"status": result["status"], "failure": result.get("failure"), "runtime": result["runtime"]}
    kernel_ok = all(item["passed"] for item in summary["comparisons"].values()) if summary["comparisons"] else None
    upstream_ok = all(item["status"] == "load_failed" and item["failure"] == {"kind": "incompatible", "code": "model_profile_tokenizer_settings"} for item in summary["upstreamStrictRejections"].values()) if summary["upstreamStrictRejections"] else None
    summary["kernelComparisonsPassed"] = kernel_ok
    summary["expectedUpstreamRejectionsConfirmed"] = upstream_ok
    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if kernel_ok is False or upstream_ok is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
