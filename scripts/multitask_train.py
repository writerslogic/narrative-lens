"""One encoder, four heads: the narrative multi-task model.

Trains on the documents written by `narrative_corpus.py` (LitBank development split, and
PDNC / GUM when present). A DistilBERT encoder initialised from the pinned
`dslim/distilbert-NER` export feeds:

- `entity`: 9-label BIO token classifier (the pinned NER manifest's label set, so the
  head is a drop-in for the existing loader; only PER carries training signal)
- `nominal`: 3-label BIO token classifier for nominal person mentions ("her mother")
- `coref`: antecedent scorer over person mentions (proper, nominal, pronoun) in a window
- `attribution`: speaker scorer over the person mentions near each quotation, with a
  "speaker not in window" option

`cv` reports every head on held-out groups (authors / novels / documents); `export`
trains on all documents and writes one ONNX graph with four outputs plus a manifest.

    uv run --no-project --with torch --with transformers --with onnx --with onnxruntime scripts/multitask_train.py cv
    uv run --no-project --with torch --with transformers --with onnx --with onnxruntime scripts/multitask_train.py export
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
from torch import nn
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "dslim/distilbert-NER"
PINNED = ROOT / "target/external-validation/ner-model/3a19fe9404a4469d91aa3d551558a97f68872f67"
CORPUS = ROOT / "target/external-validation/narrative-multitask"
ENTITY_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
NOMINAL_LABELS = ["O", "B-NOM", "I-NOM"]
WINDOW = 400
IGNORE = -100
HIDDEN = 768
SPAN = 256


def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_file=str(PINNED / "tokenizer.json"),
        cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]",
    )


def load_documents(sources):
    docs = []
    for source in sources:
        path = CORPUS / f"{source}.jsonl"
        if path.exists():
            docs.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not docs:
        raise SystemExit(f"no documents under {CORPUS}; run scripts/narrative_corpus.py first")
    return docs


def windows(tok, doc):
    """Cut a document into token windows carrying mention, cluster, and quote indices.

    A window is (ids, entity_labels, nominal_labels, mentions, quotes, offsets) where
    mentions are (start_tok, end_tok, cluster, kind) with inclusive token positions in
    the window (position 0 is [CLS]) and quotes are (start_tok, end_tok, speaker_cluster)."""
    enc = tok(doc["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]

    def token_span(s, e):
        ts = bisect.bisect_right(ends, s)  # first token ending after s
        te = bisect.bisect_left(starts, e) - 1  # last token starting before e
        return (ts, te) if 0 <= ts <= te < len(ids) else None

    cluster_of = {tuple(m): i for i, cluster in enumerate(doc["clusters"]) for m in cluster}
    next_singleton = [len(doc["clusters"])]
    mentions = []
    for kind in ("proper", "nominal", "pronoun"):
        for s, e in doc[kind]:
            span = token_span(s, e)
            if span is None:
                continue
            cluster = cluster_of.get((s, e))
            if cluster is None:
                cluster = next_singleton[0]
                next_singleton[0] += 1
            mentions.append((span[0], span[1], cluster, kind))
    mentions.sort()
    quotes = []
    for q in doc["quotes"]:
        span = token_span(q["start"], q["end"])
        if span is not None:
            quotes.append((span[0], span[1], q["speaker"]))
    out = []
    for a in range(0, len(ids), WINDOW):
        b = min(a + WINDOW, len(ids))
        # Sources without span annotations (PDNC) must not teach the token heads that everything is O.
        fill = 0 if doc.get("token_labels", True) else IGNORE
        ent = [fill] * (b - a)
        nom = [fill] * (b - a)
        inside = []
        for ts, te, cluster, kind in mentions:
            if ts >= a and te < b:
                inside.append((ts - a + 1, te - a + 1, cluster, kind))
                labels = ent if kind == "proper" else nom if kind == "nominal" else None
                if labels is not None and fill == 0:
                    labels[ts - a] = 1
                    for t in range(ts + 1, te + 1):
                        labels[t - a] = 2
        qs = [(ts - a + 1, te - a + 1, speaker) for ts, te, speaker in quotes if ts >= a and te < b]
        out.append(([tok.cls_token_id] + ids[a:b] + [tok.sep_token_id], [IGNORE] + ent + [IGNORE], [IGNORE] + nom + [IGNORE], inside, qs, [(0, 0)] + offsets[a:b] + [(0, 0)]))
    return out


class MultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        base = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, attn_implementation="eager")
        assert [base.config.id2label[i] for i in range(len(ENTITY_LABELS))] == ENTITY_LABELS
        self.encoder = base.distilbert
        self.dropout = nn.Dropout(0.1)
        self.entity = base.classifier
        self.nominal = nn.Linear(HIDDEN, len(NOMINAL_LABELS))
        self.span = nn.Sequential(nn.Linear(HIDDEN * 3, SPAN), nn.ReLU())
        self.coref = nn.Sequential(nn.Linear(SPAN * 3, SPAN), nn.ReLU(), nn.Linear(SPAN, 1))
        self.quote = nn.Sequential(nn.Linear(HIDDEN, SPAN), nn.ReLU())
        self.attrib = nn.Sequential(nn.Linear(SPAN * 3, SPAN), nn.ReLU(), nn.Linear(SPAN, 1))
        self.attrib_dummy = nn.Linear(SPAN, 1)

    @staticmethod
    def span_mean(hidden, starts, ends):
        positions = torch.arange(hidden.size(0), device=hidden.device)
        mask = ((positions[None, :] >= starts[:, None]) & (positions[None, :] <= ends[:, None])).to(hidden.dtype)
        return (mask @ hidden) / mask.sum(1, keepdim=True).clamp(min=1.0)

    def span_reprs(self, hidden, starts, ends):
        return self.span(torch.cat([hidden[starts], hidden[ends], self.span_mean(hidden, starts, ends)], -1))

    def coref_scores(self, spans):
        a = spans[:, None, :].expand(spans.size(0), spans.size(0), SPAN)
        b = spans[None, :, :].expand(spans.size(0), spans.size(0), SPAN)
        return self.coref(torch.cat([a, b, a * b], -1)).squeeze(-1)

    def attrib_scores(self, hidden, spans, qstarts, qends):
        quotes = self.quote(self.span_mean(hidden, qstarts, qends))
        q = quotes[:, None, :].expand(quotes.size(0), spans.size(0), SPAN)
        s = spans[None, :, :].expand(quotes.size(0), spans.size(0), SPAN)
        pair = self.attrib(torch.cat([q, s, q * s], -1)).squeeze(-1)
        return torch.cat([self.attrib_dummy(quotes), pair], 1)

    def forward(self, input_ids, attention_mask, mention_starts, mention_ends, quote_starts, quote_ends):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]
        hidden = self.dropout(hidden)
        spans = self.span_reprs(hidden, mention_starts, mention_ends)
        return (
            self.entity(hidden)[None],
            self.nominal(hidden)[None],
            self.coref_scores(spans),
            self.attrib_scores(hidden, spans, quote_starts, quote_ends),
        )


def marginal_loss(scores, gold, valid):
    """-log of the probability mass on gold columns, with invalid columns removed; rows without a gold column are skipped."""
    scores = scores.masked_fill(~valid, float("-inf"))
    rows = gold.any(1)
    if not rows.any():
        return scores.sum() * 0.0
    total = torch.logsumexp(scores[rows], 1)
    positive = torch.logsumexp(scores[rows].masked_fill(~gold[rows], float("-inf")), 1)
    return (total - positive).mean()


def window_tensors(window, device):
    ids, ent, nom, mentions, quotes, _ = window
    t = lambda x, dtype=torch.long: torch.tensor(x, dtype=dtype, device=device)
    ms = t([m[0] for m in mentions] or [0])
    me = t([m[1] for m in mentions] or [0])
    qs = t([q[0] for q in quotes] or [0])
    qe = t([q[1] for q in quotes] or [0])
    return t([ids]), torch.ones(1, len(ids), dtype=torch.long, device=device), t([ent]), t([nom]), ms, me, qs, qe


def coref_targets(mentions, device):
    m = len(mentions)
    clusters = torch.tensor([x[2] for x in mentions], device=device)
    lower = torch.tril(torch.ones(m, m, dtype=torch.bool, device=device), -1)
    same = (clusters[:, None] == clusters[None, :]) & lower
    dummy_gold = ~same.any(1)
    valid = torch.cat([torch.ones(m, 1, dtype=torch.bool, device=device), lower], 1)
    gold = torch.cat([dummy_gold[:, None], same], 1)
    return valid, gold


def attrib_targets(mentions, quotes, device):
    q, m = len(quotes), len(mentions)
    valid = torch.ones(q, m + 1, dtype=torch.bool, device=device)
    gold = torch.zeros(q, m + 1, dtype=torch.bool, device=device)
    for i, (qs, qe, speaker) in enumerate(quotes):
        if speaker < 0:
            continue
        hit = False
        for j, (ms, me, cluster, _) in enumerate(mentions):
            if ms >= qs and me <= qe:
                valid[i, j + 1] = False
            elif cluster == speaker:
                gold[i, j + 1] = True
                hit = True
        gold[i, 0] = not hit
    return valid, gold


def losses(model, window, device):
    ids, mask, ent, nom, ms, me, qs, qe = window_tensors(window, device)
    mentions, quotes = window[3], window[4]
    ent_logits, nom_logits, coref, attrib = model(ids, mask, ms, me, qs, qe)
    ce = nn.functional.cross_entropy
    out = {"entity": ce(ent_logits[0], ent[0], ignore_index=IGNORE), "nominal": ce(nom_logits[0], nom[0], ignore_index=IGNORE)}
    if len(mentions) > 1:
        valid, gold = coref_targets(mentions, device)
        out["coref"] = marginal_loss(torch.cat([torch.zeros(len(mentions), 1, device=device), coref], 1), gold, valid)
    if quotes and mentions:
        valid, gold = attrib_targets(mentions, quotes, device)
        out["attribution"] = marginal_loss(attrib, gold, valid)
    return out


def train(model, train_windows, device, epochs, lr, accumulate):
    model.train()
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": enc, "lr": lr}, {"params": heads, "lr": lr * 10}], weight_decay=0.01)
    steps = epochs * ((len(train_windows) + accumulate - 1) // accumulate)
    warmup = max(1, steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))
    for _ in range(epochs):
        order = list(range(len(train_windows)))
        random.shuffle(order)
        for i, j in enumerate(order):
            loss = sum(losses(model, train_windows[j], device).values()) / accumulate
            loss.backward()
            if (i + 1) % accumulate == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); schedule.step(); optimizer.zero_grad()
    model.eval()


def bio_spans(labels):
    """Spans from BIO ids: B opens a span (closing any open one), I continues or opens one, O closes."""
    out, start = [], None
    for t, label in enumerate(labels):
        if label == 1 or (label == 2 and start is None):
            if start is not None:
                out.append((start, t - 1))
            start = t
        elif label == 0 and start is not None:
            out.append((start, t - 1))
            start = None
    if start is not None:
        out.append((start, len(labels) - 1))
    return set(out)


def gold_spans(labels):
    return {(s, e) for s, e in bio_spans([0 if x == IGNORE else x for x in labels])}


@torch.no_grad()
def evaluate(model, eval_windows, device):
    c = defaultdict(int)
    for window in eval_windows:
        ids, mask, ent, nom, ms, me, qs, qe = window_tensors(window, device)
        mentions, quotes = window[3], window[4]
        ent_logits, nom_logits, coref, attrib = model(ids, mask, ms, me, qs, qe)
        for name, logits, labels in (("entity", ent_logits, window[1]), ("nominal", nom_logits, window[2])):
            pred = [p if l != IGNORE else 0 for p, l in zip(logits[0].argmax(-1).tolist(), labels)]
            predicted, gold = bio_spans(pred), gold_spans(labels)
            c[name + "_tp"] += len(predicted & gold); c[name + "_fp"] += len(predicted - gold); c[name + "_fn"] += len(gold - predicted)
        if len(mentions) > 1:
            valid, gold = coref_targets(mentions, device)
            scores = torch.cat([torch.zeros(len(mentions), 1, device=device), coref], 1).masked_fill(~valid, float("-inf"))
            pred = scores.argmax(1)
            correct = gold[torch.arange(len(mentions), device=device), pred]
            for (ms_, me_, _, kind), ok in zip(mentions, correct.tolist()):
                c["coref_" + kind + "_n"] += 1; c["coref_" + kind + "_ok"] += int(ok)
        if quotes and mentions:
            valid, gold = attrib_targets(mentions, quotes, device)
            pred = attrib.masked_fill(~valid, float("-inf")).argmax(1)
            for i, (_, _, speaker) in enumerate(quotes):
                if speaker >= 0:
                    c["attribution_n"] += 1; c["attribution_ok"] += int(gold[i, pred[i]].item())
    report = {}
    for name in ("entity", "nominal"):
        tp, fp, fn = c[name + "_tp"], c[name + "_fp"], c[name + "_fn"]
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        report[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    for kind in ("proper", "nominal", "pronoun"):
        n = c["coref_" + kind + "_n"]
        report["coref_" + kind + "_antecedent_accuracy"] = {"n": n, "accuracy": c["coref_" + kind + "_ok"] / n if n else None}
    report["attribution_accuracy"] = {"n": c["attribution_n"], "accuracy": c["attribution_ok"] / c["attribution_n"] if c["attribution_n"] else None}
    return report


def group_key(doc):
    """Author surname: LitBank excerpts and PDNC novels by the same author (often the same novel) must share a fold."""
    return doc["group"].split(",")[0].strip().lower()


def folds(docs, k):
    by_group = defaultdict(list)
    for d in docs:
        by_group[group_key(d)].append(d)
    groups = sorted(by_group)
    random.Random(7).shuffle(groups)
    return [[d for g in groups[i::k] for d in by_group[g]] for i in range(k)]


def device_():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def select_windows(doc, ws, limit):
    """Full novels (PDNC) are capped at `limit` windows each, preferring windows that hold quotations."""
    if limit <= 0 or len(ws) <= limit or doc["source"] == "litbank":
        return ws
    with_quotes = [w for w in ws if w[4]]
    pool = with_quotes if len(with_quotes) >= limit else ws
    return random.Random(hash(doc["id"]) & 0xFFFF).sample(pool, limit)


def cv(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device, tok = device_(), tokenizer()
    docs = load_documents(args.sources)
    encoded = {d["id"] + "/" + d["source"]: select_windows(d, windows(tok, d), args.novel_windows) for d in docs}
    key = lambda d: d["id"] + "/" + d["source"]
    report = {"sources": args.sources, "documents": len(docs), "folds": args.folds, "epochs": args.epochs, "lr": args.lr, "window": WINDOW, "perFold": []}
    for i, held in enumerate(folds(docs, args.folds)):
        if args.only_fold is not None and i != args.only_fold:
            continue
        held_keys = {key(d) for d in held}
        train_windows = [w for k, ws in encoded.items() if k not in held_keys for w in ws]
        eval_windows = [w for k in sorted(held_keys) for w in encoded[k]]
        model = MultiTask().to(device)
        train(model, train_windows, device, args.epochs, args.lr, args.accumulate)
        result = evaluate(model, eval_windows, device)
        sources_held = sorted({k.rsplit("/", 1)[1] for k in held_keys})
        if len(sources_held) > 1:
            result["bySource"] = {src: evaluate(model, [w for k in sorted(held_keys) if k.endswith("/" + src) for w in encoded[k]], device) for src in sources_held}
        report["perFold"].append(result)
        print(f"fold {i}: {len(held)} docs entity F1 {result['entity']['f1']:.3f} nominal F1 {result['nominal']['f1']:.3f} "
              f"pronoun antecedent {result['coref_pronoun_antecedent_accuracy']['accuracy']:.3f} attribution {result['attribution_accuracy']['accuracy'] or 0:.3f}", flush=True)
    pooled = {}
    for name in ("entity", "nominal"):
        tp = sum(f[name]["tp"] for f in report["perFold"]); fp = sum(f[name]["fp"] for f in report["perFold"]); fn = sum(f[name]["fn"] for f in report["perFold"])
        p, r = tp / (tp + fp), tp / (tp + fn)
        pooled[name] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r)}
    for name in ("coref_proper_antecedent_accuracy", "coref_nominal_antecedent_accuracy", "coref_pronoun_antecedent_accuracy", "attribution_accuracy"):
        n = sum(f[name]["n"] for f in report["perFold"]); ok = sum(round(f[name]["accuracy"] * f[name]["n"]) for f in report["perFold"] if f[name]["accuracy"] is not None)
        pooled[name] = {"n": n, "accuracy": ok / n if n else None}
    report["pooled"] = pooled
    print("pooled:", json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in pooled.items()}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def export(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device, tok = device_(), tokenizer()
    docs = load_documents(args.sources)
    train_windows = [w for d in docs for w in select_windows(d, windows(tok, d), args.novel_windows)]
    model = MultiTask().to(device)
    train(model, train_windows, device, args.epochs, args.lr, args.accumulate)
    model = model.cpu().eval()
    staging = args.output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    ids = torch.tensor([[tok.cls_token_id] + tok("\"Come in,\" said Alice to her sister.", add_special_tokens=False)["input_ids"] + [tok.sep_token_id]])
    example = (ids, torch.ones_like(ids), torch.tensor([5, 7]), torch.tensor([5, 8]), torch.tensor([1]), torch.tensor([4]))
    torch.onnx.export(
        model, example, str(staging / "model-fp32.onnx"),
        input_names=["input_ids", "attention_mask", "mention_starts", "mention_ends", "quote_starts", "quote_ends"],
        output_names=["entity_logits", "nominal_logits", "coref_scores", "attribution_scores"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "mention_starts": {0: "mentions"}, "mention_ends": {0: "mentions"},
                      "quote_starts": {0: "quotes"}, "quote_ends": {0: "quotes"}, "entity_logits": {0: "batch", 1: "sequence"}, "nominal_logits": {0: "batch", 1: "sequence"},
                      "coref_scores": {0: "mentions", 1: "mentions"}, "attribution_scores": {0: "quotes", 1: "mentions_plus_dummy"}},
        opset_version=14, dynamo=False,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(staging / "model-fp32.onnx"), str(staging / "model.onnx"), weight_type=QuantType.QInt8)
    (staging / "model-fp32.onnx").unlink()
    sha = hashlib.sha256((staging / "model.onnx").read_bytes()).hexdigest()
    final = args.output / sha
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    for name in ("tokenizer.json", "tokenizer.adapted.json"):
        shutil.copy(PINNED / name, final / name)
    manifest = {
        "schemaVersion": 1, "kind": "narrative_multitask", "profile": "narrative_multitask_v1", "modelSha256": sha,
        "tokenizerSha256": hashlib.sha256((final / "tokenizer.adapted.json").read_bytes()).hexdigest(), "maxTokens": 512, "paddingTokenId": 0,
        "entityLabels": ENTITY_LABELS, "nominalLabels": NOMINAL_LABELS,
        "inputs": ["input_ids", "attention_mask", "mention_starts", "mention_ends", "quote_starts", "quote_ends"],
        "outputs": ["entity_logits", "nominal_logits", "coref_scores", "attribution_scores"],
        "notes": "coref_scores[i][j] scores mention j as antecedent of mention i (j < i); 0 is the no-antecedent score. attribution_scores[q][0] is 'speaker not in window', column m+1 is mention m.",
        "training": {"sources": args.sources, "documents": len(docs), "epochs": args.epochs, "lr": args.lr, "seed": args.seed, "base": BASE_MODEL},
    }
    (final / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(final, (final / "model.onnx").stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cv", "export"])
    parser.add_argument("--sources", nargs="+", default=["litbank", "pdnc", "gum"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--novel-windows", type=int, default=40, help="windows kept per full novel (PDNC); 0 keeps all")
    parser.add_argument("--only-fold", type=int, help="run a single fold (for comparing data mixes on one held-out set)")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("target/evaluation/narrative-multitask-cv.json" if args.mode == "cv" else "target/external-validation/narrative-multitask-model")
    (cv if args.mode == "cv" else export)(args)


if __name__ == "__main__":
    main()
