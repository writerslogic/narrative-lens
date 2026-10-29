"""Fine-tune the pinned DistilBERT NER export on LitBank proper-person spans.

Reads only the development split of a prepared LitBank folder (see
`literary_benchmark.py prepare`); the holdout split is never loaded. `cv` trains
author-disjoint folds and reports exact span precision/recall/F1 for the untouched
base model and the fine-tuned one on each held-out fold. `export` trains on every
development work and writes a model directory in the layout `evaluate_literary
--ner-cache` expects (model.onnx, tokenizer files, config.json, manifest.json with
the same `hugging_face_bio9_v1` label set).

    uv run --no-project --with torch --with transformers --with onnx --with onnxruntime scripts/ner_finetune.py cv
    uv run --no-project --with torch --with transformers --with onnx --with onnxruntime scripts/ner_finetune.py export

The export is dynamically quantized to int8 weights like the pinned file (the loader
caps models at 70 MiB), and the graph is traced with eager attention so it uses the
same operator set the wasm (tract) runtime already handles.
"""

import argparse
import bisect
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "dslim/distilbert-NER"
PINNED = ROOT / "target/external-validation/ner-model/3a19fe9404a4469d91aa3d551558a97f68872f67"
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
MAX_TOKENS = 200
IGNORE = -100


def load_works(folder):
    gold = {g["id"]: g for g in json.loads((folder / "gold.json").read_text())}
    inputs = {c["id"]: c for c in json.loads((folder / "inputs.json").read_text())}
    authors = {w["id"]: w["author"] for w in json.loads((ROOT / "docs/benchmarks/litbank-source.json").read_text())["works"]}
    works = []
    for wid, g in gold.items():
        if g["split"] != "development":
            continue
        assert inputs[wid]["split"] == "development"
        works.append({"id": wid, "author": authors[wid], "text": inputs[wid]["entityText"], "spans": [tuple(s) for s in g["properPersonSpans"]]})
    return works


def byte_to_char_table(text):
    cum = [0]
    for ch in text:
        cum.append(cum[-1] + len(ch.encode()))
    return cum


def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_file=str(PINNED / "tokenizer.json"),
        cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]",
    )


def encode_work(tok, work):
    """Pack the one-sentence-per-line text into sequences of at most MAX_TOKENS tokens.

    Returns sequences of (input_ids, labels, char_offsets) where labels use LABELS
    indices for real tokens and IGNORE for [CLS]/[SEP]."""
    text, cum = work["text"], byte_to_char_table(work["text"])
    inside = sorted((bisect.bisect_left(cum, s), bisect.bisect_left(cum, e)) for s, e in work["spans"])
    starts = {s for s, _ in inside}
    sequences, ids, labels, offsets = [], [], [], []

    def flush():
        if ids:
            sequences.append(([tok.cls_token_id] + ids + [tok.sep_token_id], [IGNORE] + labels + [IGNORE], [(0, 0)] + offsets + [(0, 0)]))
        ids.clear(); labels.clear(); offsets.clear()

    pos = 0
    for line in text.split("\n"):
        enc = tok(line, add_special_tokens=False, return_offsets_mapping=True)
        line_ids = enc["input_ids"]
        if len(ids) + len(line_ids) > MAX_TOKENS:
            flush()
        for tid, (a, b) in zip(line_ids, enc["offset_mapping"]):
            a, b = a + pos, b + pos
            label = 0
            for s, e in inside:
                if a >= s and b <= e and b > a:
                    label = 1 if a in starts and a == s else 2
                    break
                if s > a:
                    break
            ids.append(tid); labels.append(label); offsets.append((a, b))
        pos += len(line) + 1
    flush()
    return sequences, cum


def spans_from_labels(label_ids, offsets, cum):
    """Maximal runs of PER-labelled tokens, as byte ranges (mirrors the crate's decoder)."""
    out, start, end = [], None, None
    for label, (a, b) in zip(label_ids, offsets):
        if label in (1, 2) and b > a:
            if start is None:
                start = a
            end = b
        else:
            if start is not None:
                out.append((cum[start], cum[end]))
            start = None
    if start is not None:
        out.append((cum[start], cum[end]))
    return out


def batches(sequences, size, shuffle):
    order = list(range(len(sequences)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [sequences[j] for j in order[i:i + size]]
        width = max(len(s[0]) for s in chunk)
        ids = torch.tensor([s[0] + [0] * (width - len(s[0])) for s in chunk])
        mask = torch.tensor([[1] * len(s[0]) + [0] * (width - len(s[0])) for s in chunk])
        labels = torch.tensor([s[1] + [IGNORE] * (width - len(s[1])) for s in chunk])
        yield ids, mask, labels


def train(model, sequences, device, epochs, lr, batch_size):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(sequences) + batch_size - 1) // batch_size)
    warmup = max(1, steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))
    for _ in range(epochs):
        for ids, mask, labels in batches(sequences, batch_size, shuffle=True):
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); schedule.step(); optimizer.zero_grad()
    model.eval()


@torch.no_grad()
def evaluate(model, encoded, device):
    tp = fp = fn = 0
    for sequences, cum, gold in encoded:
        predicted = set()
        for seq in sequences:
            ids = torch.tensor([seq[0]]).to(device)
            pred = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.argmax(-1)[0].cpu().tolist()
            predicted.update(spans_from_labels(pred, seq[2], cum))
        gold = set(gold)
        tp += len(predicted & gold); fp += len(predicted - gold); fn += len(gold - predicted)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def folds(works, k):
    by_author = defaultdict(list)
    for w in works:
        by_author[w["author"]].append(w)
    authors = sorted(by_author)
    random.Random(7).shuffle(authors)
    return [[w for a in authors[i::k] for w in by_author[a]] for i in range(k)]


def fresh_model(device):
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, attn_implementation="eager")
    assert [model.config.id2label[i] for i in range(len(LABELS))] == LABELS, model.config.id2label
    return model.to(device)


def cv(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = tokenizer()
    works = load_works(args.folder)
    encoded = {w["id"]: (*encode_work(tok, w), w["spans"]) for w in works}
    report = {"base": [], "tuned": [], "epochs": args.epochs, "lr": args.lr, "folds": args.folds, "works": len(works)}
    for i, held in enumerate(folds(works, args.folds)):
        held_ids = {w["id"] for w in held}
        train_seqs = [s for wid, (seqs, _, _) in encoded.items() if wid not in held_ids for s in seqs]
        held_enc = [encoded[wid] for wid in sorted(held_ids)]
        model = fresh_model(device)
        base = evaluate(model, held_enc, device)
        train(model, train_seqs, device, args.epochs, args.lr, args.batch_size)
        tuned = evaluate(model, held_enc, device)
        report["base"].append(base); report["tuned"].append(tuned)
        print(f"fold {i}: {len(held)} works base F1 {base['f1']:.3f} -> tuned F1 {tuned['f1']:.3f} (P {tuned['precision']:.3f} R {tuned['recall']:.3f})", flush=True)
    for key in ("base", "tuned"):
        tp = sum(f["tp"] for f in report[key]); fp = sum(f["fp"] for f in report[key]); fn = sum(f["fn"] for f in report[key])
        p, r = tp / (tp + fp), tp / (tp + fn)
        report[key + "Pooled"] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r)}
        print(f"{key} pooled: P {p:.3f} R {r:.3f} F1 {report[key + 'Pooled']['f1']:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def export(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = tokenizer()
    works = load_works(args.folder)
    sequences = [s for w in works for s in encode_work(tok, w)[0]]
    model = fresh_model(device)
    train(model, sequences, device, args.epochs, args.lr, args.batch_size)
    model = model.cpu()
    staging = args.output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    ids = torch.tensor([[tok.cls_token_id] + tok("Alice met Mr. Darcy.", add_special_tokens=False)["input_ids"] + [tok.sep_token_id]])
    torch.onnx.export(
        model, (ids, torch.ones_like(ids)), str(staging / "model-fp32.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"}, "logits": {0: "batch_size", 1: "sequence_length"}},
        opset_version=14, dynamo=False,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(staging / "model-fp32.onnx"), str(staging / "model.onnx"), weight_type=QuantType.QInt8)
    (staging / "model-fp32.onnx").unlink()
    size = (staging / "model.onnx").stat().st_size
    assert size <= 70 * 1024 * 1024, f"quantized model is {size} bytes, over the loader's 70 MiB cap"
    model_sha = hashlib.sha256((staging / "model.onnx").read_bytes()).hexdigest()
    final = args.output / model_sha
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    for name in ("tokenizer.json", "tokenizer.adapted.json"):
        shutil.copy(PINNED / name, final / name)
    config = json.loads((PINNED / "config.json").read_text())
    config["_name_or_path"] = f"{BASE_MODEL} fine-tuned on LitBank development proper-person spans"
    (final / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = json.loads((PINNED / "manifest.json").read_text())
    manifest["modelSha256"] = model_sha
    manifest["tokenizerSha256"] = hashlib.sha256((final / "tokenizer.adapted.json").read_bytes()).hexdigest()
    (final / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (final / "README.md").write_text(
        f"# distilbert-NER fine-tuned on LitBank (development split)\n\nBase: {BASE_MODEL} (the pinned export). Trained by `scripts/ner_finetune.py export` "
        f"on {len(works)} development-split works, {args.epochs} epochs, lr {args.lr}, seed {args.seed}. Labels unchanged; only PER spans carry training signal.\n"
    )
    print(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cv", "export"])
    parser.add_argument("--folder", type=Path, default=ROOT / "target/external-validation/litbank-prepared-v4")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("target/evaluation/ner-litbank-cv.json" if args.mode == "cv" else "target/external-validation/ner-model-litbank")
    (cv if args.mode == "cv" else export)(args)


if __name__ == "__main__":
    main()
