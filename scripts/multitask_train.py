"""One encoder, four heads: the narrative multi-task model.

Trains on the documents written by `narrative_corpus.py` (LitBank development split, and
PDNC when present). Any HuggingFace encoder with a fast tokenizer works as `--base`;
a token-classification checkpoint with the pinned NER label set (the default
`dslim/distilbert-NER`) also seeds the entity head. The heads:

- `entity`: 9-label BIO token classifier (the pinned NER manifest's label set, so the
  head is a drop-in for the existing loader; only PER carries training signal)
- `nominal`: 3-label BIO token classifier for nominal person mentions ("her mother")
- `coref`: antecedent scorer over person mentions (proper, nominal, pronoun) in a window,
  with a learned token-distance feature
- `attribution`: speaker scorer over the person mentions near each quotation, with a
  "speaker not in window" option and the same distance feature

`cv` reports every head on held-out groups (authors / novels); `export` trains on all
documents and writes one ONNX graph with four outputs plus a manifest. With
`--teacher-base` a larger teacher is trained first on the same training documents, its
predictions label every training window (PDNC carries no span labels of its own), and
the student is trained on gold plus the teacher's distributions.

    uv run --no-project --with torch --with transformers --with onnx --with onnxruntime --with sentencepiece --with protobuf scripts/multitask_train.py cv
    uv run --no-project --with torch --with transformers --with onnx --with onnxruntime --with sentencepiece --with protobuf scripts/multitask_train.py export --sources litbank pdnc
"""

import argparse
import bisect
import contextlib
import hashlib
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "dslim/distilbert-NER"
CORPUS = ROOT / "target/external-validation/narrative-multitask"
ENTITY_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
NOMINAL_LABELS = ["O", "B-NOM", "I-NOM"]
IGNORE = -100
SPAN = 256
DISTANCE_BUCKETS = 18
DISTANCE_DIM = 32


def tokenizer(base):
    tok = AutoTokenizer.from_pretrained(base)
    if not tok.is_fast or tok.cls_token_id is None or tok.sep_token_id is None:
        raise SystemExit(f"{base} needs a fast tokenizer with CLS/SEP tokens")
    return tok


def load_documents(sources, limit=0):
    docs = []
    for source in sources:
        path = CORPUS / f"{source}.jsonl"
        if path.exists():
            loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            docs.extend(loaded[:limit] if limit else loaded)
    if not docs:
        raise SystemExit(f"no documents under {CORPUS}; run scripts/narrative_corpus.py first")
    return docs


def windows(tok, doc, window, offset=0):
    """Cut a document into token windows carrying mention, cluster, and quote indices.

    A window is (ids, entity_labels, nominal_labels, mentions, quotes, offsets) where
    mentions are (start_tok, end_tok, cluster, kind) with inclusive token positions in
    the window (position 0 is [CLS]) and quotes are (start_tok, end_tok, speaker_cluster).
    `offset` shifts the window grid so the same text is also seen with other boundaries."""
    enc = tok(doc["text"], add_special_tokens=False, return_offsets_mapping=True, verbose=False)
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
            mentions.append((span[0], span[1], cluster, kind, s, e))
    mentions.sort()
    quotes = []
    for q in doc["quotes"]:
        span = token_span(q["start"], q["end"])
        if span is not None:
            quotes.append((span[0], span[1], q["speaker"], q["start"], q["end"]))
    bounds, a = [], 0
    while a < len(ids):
        b = min(offset if (a == 0 and offset) else a + window, len(ids))
        bounds.append((a, b))
        a = b
    out = []
    for a, b in bounds:
        # Sources without span annotations (PDNC) must not teach the token heads that everything is O.
        fill = 0 if doc.get("token_labels", True) else IGNORE
        ent = [fill] * (b - a)
        nom = [fill] * (b - a)
        inside, mention_chars = [], []
        for ts, te, cluster, kind, s, e in mentions:
            if ts >= a and te < b:
                inside.append((ts - a + 1, te - a + 1, cluster, kind))
                mention_chars.append((s, e))
                labels = ent if kind == "proper" else nom if kind == "nominal" else None
                if labels is not None and fill == 0:
                    labels[ts - a] = 1
                    for t in range(ts + 1, te + 1):
                        labels[t - a] = 2
        qs = [(ts - a + 1, te - a + 1, speaker) for ts, te, speaker, _, _ in quotes if ts >= a and te < b]
        quote_chars = [(s, e) for ts, te, _, s, e in quotes if ts >= a and te < b]
        meta = {"key": key(doc), "chars": (offsets[a][0], offsets[b - 1][1]), "mentions": mention_chars, "quotes": quote_chars, "labelled": not doc.get("unlabeled", False)}
        out.append(([tok.cls_token_id] + ids[a:b] + [tok.sep_token_id], [IGNORE] + ent + [IGNORE], [IGNORE] + nom + [IGNORE], inside, qs, [(0, 0)] + offsets[a:b] + [(0, 0)], meta))
    return out


def teacher_view(teacher_tok, text, window):
    """The same window (text span, mentions, quotes) tokenized for the teacher; None when a span does not map."""
    meta = window[6]
    cs, ce = meta["chars"]
    enc = teacher_tok(text[cs:ce], add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if not ids:
        return None
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]

    def token_span(s, e):
        ts = bisect.bisect_right(ends, s - cs)
        te = bisect.bisect_left(starts, e - cs) - 1
        return (ts + 1, te + 1) if 0 <= ts <= te < len(ids) else None

    mentions = [token_span(s, e) for s, e in meta["mentions"]]
    quotes = [token_span(s, e) for s, e in meta["quotes"]]
    if any(m is None for m in mentions) or any(q is None for q in quotes):
        return None
    return [teacher_tok.cls_token_id] + ids + [teacher_tok.sep_token_id], mentions, quotes, ends


def distance_bucket(d):
    """Signed log-scale bucket of a token distance: 0-8 for d >= 0, 9-17 for d < 0."""
    magnitude = torch.log2(d.abs().float() + 1.0).floor().clamp(max=8).long()
    return torch.where(d >= 0, magnitude, magnitude + 9)


class MultiTask(nn.Module):
    def __init__(self, base, grad_checkpoint=False):
        super().__init__()
        config = AutoConfig.from_pretrained(base)
        id2label = getattr(config, "id2label", None) or {}
        source_labels = [id2label.get(i) for i in range(len(id2label))]
        if sorted(source_labels) == sorted(ENTITY_LABELS):
            # A CoNLL-style NER head seeds the entity head; its rows are permuted into ENTITY_LABELS order.
            classifier = AutoModelForTokenClassification.from_pretrained(base, attn_implementation="eager")
            self.encoder = classifier.base_model
            order = [source_labels.index(label) for label in ENTITY_LABELS]
            self.entity = nn.Linear(classifier.classifier.in_features, len(ENTITY_LABELS))
            with torch.no_grad():
                self.entity.weight.copy_(classifier.classifier.weight[order])
                self.entity.bias.copy_(classifier.classifier.bias[order])
        else:
            self.encoder = AutoModel.from_pretrained(base, attn_implementation="eager")
            self.entity = nn.Linear(config.hidden_size, len(ENTITY_LABELS))
        self.encoder.float(); self.entity.float()  # checkpoints stored in half precision must not mix with the fp32 heads
        if grad_checkpoint:
            self.encoder.gradient_checkpointing_enable()
        hidden = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.nominal = nn.Linear(hidden, len(NOMINAL_LABELS))
        self.span = nn.Sequential(nn.Linear(hidden * 3, SPAN), nn.ReLU())
        self.coref_distance = nn.Embedding(DISTANCE_BUCKETS, DISTANCE_DIM)
        self.coref = nn.Sequential(nn.Linear(SPAN * 3 + DISTANCE_DIM, SPAN), nn.ReLU(), nn.Linear(SPAN, 1))
        self.quote = nn.Sequential(nn.Linear(hidden, SPAN), nn.ReLU())
        self.attrib_distance = nn.Embedding(DISTANCE_BUCKETS, DISTANCE_DIM)
        self.attrib = nn.Sequential(nn.Linear(SPAN * 3 + DISTANCE_DIM, SPAN), nn.ReLU(), nn.Linear(SPAN, 1))
        self.attrib_dummy = nn.Linear(SPAN, 1)

    @staticmethod
    def span_mean(hidden, starts, ends):
        positions = torch.arange(hidden.size(0), device=hidden.device)
        mask = ((positions[None, :] >= starts[:, None]) & (positions[None, :] <= ends[:, None])).to(hidden.dtype)
        return (mask @ hidden) / mask.sum(1, keepdim=True).clamp(min=1.0)

    def span_reprs(self, hidden, starts, ends):
        return self.span(torch.cat([hidden[starts], hidden[ends], self.span_mean(hidden, starts, ends)], -1))

    def coref_scores(self, spans, starts, ends):
        m = spans.size(0)
        a = spans[:, None, :].expand(m, m, SPAN)
        b = spans[None, :, :].expand(m, m, SPAN)
        distance = self.coref_distance(distance_bucket(starts[:, None] - ends[None, :]))
        return self.coref(torch.cat([a, b, a * b, distance], -1)).squeeze(-1)

    def attrib_scores(self, hidden, spans, qstarts, qends, mstarts):
        quotes = self.quote(self.span_mean(hidden, qstarts, qends))
        q = quotes[:, None, :].expand(quotes.size(0), spans.size(0), SPAN)
        s = spans[None, :, :].expand(quotes.size(0), spans.size(0), SPAN)
        distance = self.attrib_distance(distance_bucket(mstarts[None, :] - qstarts[:, None]))
        pair = self.attrib(torch.cat([q, s, q * s, distance], -1)).squeeze(-1)
        return torch.cat([self.attrib_dummy(quotes), pair], 1)

    def forward(self, input_ids, attention_mask, mention_starts, mention_ends, quote_starts, quote_ends):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]
        hidden = self.dropout(hidden)
        spans = self.span_reprs(hidden, mention_starts, mention_ends)
        return (
            self.entity(hidden)[None],
            self.nominal(hidden)[None],
            self.coref_scores(spans, mention_starts, mention_ends),
            self.attrib_scores(hidden, spans, quote_starts, quote_ends, mention_starts),
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


def distill_rows(scores, teacher_probs, valid):
    """KL(teacher || student) over the valid columns of every row with a choice to make."""
    rows = valid.sum(1) > 1
    if not rows.any():
        return scores.sum() * 0.0
    log_student = torch.log_softmax(scores[rows].masked_fill(~valid[rows], float("-inf")), 1)
    p = teacher_probs[rows].float()
    return (p * (torch.log(p.clamp(min=1e-6)) - log_student)).masked_fill(~valid[rows], 0.0).sum(1).mean()


def window_tensors(window, device):
    ids, ent, nom, mentions, quotes = window[:5]
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


def autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_model(model, window, device, bf16):
    ids, mask, ent, nom, ms, me, qs, qe = window_tensors(window, device)
    with autocast(device, bf16):
        outputs = model(ids, mask, ms, me, qs, qe)
    return (ent, nom), tuple(x.float() for x in outputs)


def losses(model, window, device, bf16=False, teacher=None, distill_weight=1.0):
    (ent, nom), (ent_logits, nom_logits, coref, attrib) = run_model(model, window, device, bf16)
    mentions, quotes = window[3], window[4]
    ce = nn.functional.cross_entropy
    out = {}
    if any(label != IGNORE for label in window[1]):
        out["entity"] = ce(ent_logits[0], ent[0], ignore_index=IGNORE)
        out["nominal"] = ce(nom_logits[0], nom[0], ignore_index=IGNORE)
    elif teacher is not None:
        for name, logits in (("entity", ent_logits), ("nominal", nom_logits)):
            probs = teacher[name].to(device).float()
            out[name] = -(probs * torch.log_softmax(logits[0], -1)).sum(-1)[1:-1].mean()
    labelled = window[6]["labelled"]
    if len(mentions) > 1:
        valid, gold = coref_targets(mentions, device)
        scores = torch.cat([torch.zeros(len(mentions), 1, device=device), coref], 1)
        if labelled:
            out["coref"] = marginal_loss(scores, gold, valid)
        if teacher is not None and "coref" in teacher:
            out["coref_distill"] = distill_weight * distill_rows(scores, teacher["coref"].to(device), valid)
    if quotes and mentions:
        valid, gold = attrib_targets(mentions, quotes, device)
        if labelled:
            out["attribution"] = marginal_loss(attrib, gold, valid)
        if teacher is not None and "attribution" in teacher:
            out["attribution_distill"] = distill_weight * distill_rows(attrib, teacher["attribution"].to(device), valid)
    return out


@torch.no_grad()
def teacher_outputs(model, teacher_tok, texts, windows_, device, bf16):
    """The teacher's distributions for every student window, kept on the CPU in half precision.

    The teacher tokenizes the window's text itself; token distributions are projected onto
    the student's tokens by character position and the pair matrices align through the
    shared mention and quotation spans. Windows the teacher cannot align get no entry."""
    model.eval()
    out = []
    skipped = 0
    for window in windows_:
        view = teacher_view(teacher_tok, texts[window[6]["key"]], window)
        if view is None:
            out.append(None); skipped += 1
            continue
        ids, mention_spans, quote_spans, ends = view
        t = lambda x: torch.tensor(x, dtype=torch.long, device=device)
        ms = t([a for a, _ in mention_spans] or [0]); me = t([b for _, b in mention_spans] or [0])
        qs = t([a for a, _ in quote_spans] or [0]); qe = t([b for _, b in quote_spans] or [0])
        with autocast(device, bf16):
            ent_logits, nom_logits, coref, attrib = model(t([ids]), torch.ones(1, len(ids), dtype=torch.long, device=device), ms, me, qs, qe)
        ent_probs, nom_probs = torch.softmax(ent_logits[0].float(), -1), torch.softmax(nom_logits[0].float(), -1)
        cs = window[6]["chars"][0]
        index = [0] + [min(bisect.bisect_right(ends, s - cs), len(ends) - 1) + 1 for s, _ in window[5][1:-1]] + [len(ids) - 1]
        gather = torch.tensor(index, device=device)
        mentions, quotes = window[3], window[4]
        entry = {"entity": ent_probs[gather].half().cpu(), "nominal": nom_probs[gather].half().cpu()}
        if len(mentions) > 1:
            valid, _ = coref_targets(mentions, device)
            scores = torch.cat([torch.zeros(len(mentions), 1, device=device), coref.float()], 1)
            entry["coref"] = torch.softmax(scores.masked_fill(~valid, float("-inf")), 1).half().cpu()
        if quotes and mentions:
            valid, _ = attrib_targets(mentions, quotes, device)
            entry["attribution"] = torch.softmax(attrib.float().masked_fill(~valid, float("-inf")), 1).half().cpu()
        out.append(entry)
    if skipped:
        print(f"  teacher could not align {skipped} of {len(windows_)} windows", flush=True)
    return out


PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves"}


def quotation_spans(text):
    """Paired quotation marks (curly pairs, or straight quotes taken in turn), at most 1,000 characters each."""
    spans, open_at, straight = [], None, None
    for i, ch in enumerate(text):
        if ch == "\u201c":
            open_at = i
        elif ch == "\u201d" and open_at is not None:
            if i + 1 - open_at <= 1000:
                spans.append((open_at, i + 1))
            open_at = None
        elif ch == '"':
            if straight is None:
                straight = i
            else:
                if i + 1 - straight <= 1000:
                    spans.append((straight, i + 1))
                straight = None
    return spans


def bio_char_spans(labels, offsets):
    """Character spans of BIO-tagged tokens (label 1 opens, 2 continues), skipping specials with (0, 0) offsets."""
    out, start, end = [], None, None
    for label, (a, b) in zip(labels, offsets):
        if (a, b) == (0, 0):
            label = 0
        if label == 1 or (label == 2 and start is None):
            if start is not None:
                out.append((start, end))
            start, end = a, b
        elif label == 2:
            end = b
        elif start is not None:
            out.append((start, end))
            start = None
    if start is not None:
        out.append((start, end))
    return [(a, b) for a, b in out if b > a]


@torch.no_grad()
def pseudo_structure(model, tok, docs, window, device, bf16):
    """Give unlabelled documents the mention and quotation spans the teacher's token heads
    predict (plus the pronoun inventory and paired quotation marks); clusters stay empty
    so only the teacher's distributions, never gold, supervise their pair heads."""
    import re

    model.eval()
    for doc in docs:
        text = doc["text"]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        proper, nominal = [], []
        for a in range(0, len(ids), window):
            b = min(a + window, len(ids))
            batch = torch.tensor([[tok.cls_token_id] + ids[a:b] + [tok.sep_token_id]], device=device)
            zero = torch.zeros(1, dtype=torch.long, device=device)
            with autocast(device, bf16):
                ent_logits, nom_logits, _, _ = model(batch, torch.ones_like(batch), zero, zero, zero, zero)
            span_offsets = [(0, 0)] + offsets[a:b] + [(0, 0)]
            proper.extend(bio_char_spans(ent_logits[0].argmax(-1).tolist(), span_offsets))
            nominal.extend(bio_char_spans(nom_logits[0].argmax(-1).tolist(), span_offsets))
        taken = set()
        for span in proper:
            taken.add(span)
        pronoun = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", text) if m.group().lower() in PRONOUNS]
        doc["proper"] = [list(s) for s in proper]
        doc["nominal"] = [list(s) for s in nominal if s not in taken]
        doc["pronoun"] = [list(s) for s in pronoun]
        doc["quotes"] = [{"start": s, "end": e, "speaker": -1} for s, e in quotation_spans(text)]
        doc["clusters"] = []
    return docs


def train(model, train_windows, device, epochs, lr, accumulate, weights=None, bf16=False, teacher=None, distill_weight=1.0, label=""):
    weights = weights or {}
    model.train()
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": enc, "lr": lr}, {"params": heads, "lr": lr * 10}], weight_decay=0.01)
    steps = epochs * ((len(train_windows) + accumulate - 1) // accumulate)
    warmup = max(1, steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))
    started = time.time()
    for epoch in range(epochs):
        order = list(range(len(train_windows)))
        random.shuffle(order)
        running = defaultdict(float)
        for i, j in enumerate(order):
            parts = losses(model, train_windows[j], device, bf16, teacher[j] if teacher else None, distill_weight)
            if not parts:  # an unlabelled window with nothing to score
                continue
            loss = sum(weights.get(name, 1.0) * value for name, value in parts.items()) / accumulate
            loss.backward()
            for name, value in parts.items():
                running[name] += value.item()
            if (i + 1) % accumulate == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); schedule.step(); optimizer.zero_grad()
            if (i + 1) % 2000 == 0:
                print(f"  {label}epoch {epoch + 1} window {i + 1}/{len(order)} {time.time() - started:.0f}s "
                      + " ".join(f"{k} {v / (i + 1):.3f}" for k, v in sorted(running.items())), flush=True)
        print(f"  {label}epoch {epoch + 1} done {time.time() - started:.0f}s " + " ".join(f"{k} {v / len(order):.3f}" for k, v in sorted(running.items())), flush=True)
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
def evaluate(model, eval_windows, device, bf16=False):
    model.eval()
    c = defaultdict(int)
    for window in eval_windows:
        _, (ent_logits, nom_logits, coref, attrib) = run_model(model, window, device, bf16)
        mentions, quotes = window[3], window[4]
        for name, logits, labels in (("entity", ent_logits, window[1]), ("nominal", nom_logits, window[2])):
            pred = [p if l != IGNORE else 0 for p, l in zip(logits[0].argmax(-1).tolist(), labels)]
            predicted, gold = bio_spans(pred), gold_spans(labels)
            c[name + "_tp"] += len(predicted & gold); c[name + "_fp"] += len(predicted - gold); c[name + "_fn"] += len(gold - predicted)
        if len(mentions) > 1:
            valid, gold = coref_targets(mentions, device)
            scores = torch.cat([torch.zeros(len(mentions), 1, device=device), coref], 1).masked_fill(~valid, float("-inf"))
            pred = scores.argmax(1)
            correct = gold[torch.arange(len(mentions), device=device), pred]
            for (_, _, _, kind), ok in zip(mentions, correct.tolist()):
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


def summary(result):
    return (f"entity F1 {result['entity']['f1']:.3f} nominal F1 {result['nominal']['f1']:.3f} "
            f"pronoun antecedent {result['coref_pronoun_antecedent_accuracy']['accuracy'] or 0:.3f} attribution {result['attribution_accuracy']['accuracy'] or 0:.3f}")


def group_key(doc):
    """Author surname: LitBank excerpts and PDNC novels by the same author (often the same novel) must share a fold."""
    return doc["group"].split(",")[0].strip().lower()


def folds(docs, k):
    """Author-disjoint folds over the labelled documents; unlabelled ones always train."""
    by_group = defaultdict(list)
    for d in docs:
        if d.get("unlabeled"):
            continue
        by_group[group_key(d)].append(d)
    groups = sorted(by_group)
    random.Random(7).shuffle(groups)
    return [[d for g in groups[i::k] for d in by_group[g]] for i in range(k)]


def device_():
    """CUDA, else MPS, else CPU; NARRATIVE_DEVICE overrides (DeBERTa's matmuls hit an MPS assertion, so use cpu there)."""
    import os

    if os.environ.get("NARRATIVE_DEVICE"):
        return torch.device(os.environ["NARRATIVE_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def select_windows(doc, ws, limit):
    """Full novels (PDNC) are capped at `limit` windows each, preferring windows that hold quotations; 0 keeps all."""
    if limit <= 0 or len(ws) <= limit or doc["source"] == "litbank":
        return ws
    with_quotes = [w for w in ws if w[4]]
    pool = with_quotes if len(with_quotes) >= limit else ws
    return random.Random(hash(doc["id"]) & 0xFFFF).sample(pool, limit)


def key(doc):
    return doc["id"] + "/" + doc["source"]


def encode(tok, docs, args, offsets):
    """Per document: evaluation windows on the plain grid and training windows per offset grid."""
    evaluation, training = {}, {}
    for d in docs:
        evaluation[key(d)] = select_windows(d, windows(tok, d, args.window), args.novel_windows)
        training[key(d)] = {offset: select_windows(d, windows(tok, d, args.window, offset), args.novel_windows) for offset in sorted(set(offsets))}
    return evaluation, training


def encode_all(docs, args):
    """Student grids, teacher grids (the teacher's own tokenizer), and the texts both index.
    Unlabelled documents are structured later by the teacher, so they get no grids here."""
    tok = tokenizer(args.base)
    labelled = [d for d in docs if not d.get("unlabeled")]
    evaluation, training = encode(tok, labelled, args, args.offsets)
    teacher_training = encode(tokenizer(args.teacher_base), labelled, args, args.teacher_offsets)[1] if args.teacher_base else training
    return tok, evaluation, training, teacher_training, {key(d): d["text"] for d in docs}


def training_windows(docs, training, held_keys, offsets, litbank_repeat):
    out = []
    for d in docs:
        if key(d) in held_keys or d.get("unlabeled"):
            continue
        for offset in offsets:
            out.extend(training[key(d)][offset] * (litbank_repeat if d["source"] == "litbank" else 1))
    return out


def loss_weights(args):
    return {"entity": args.entity_weight, "nominal": args.nominal_weight}


def save_checkpoint(model, base, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"base": base, "state": model.state_dict()}, path)


def load_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu")
    model = MultiTask(saved["base"])
    model.load_state_dict(saved["state"])
    return model.to(device).eval()


def fit(args, train_windows, teacher_windows, texts, device, teacher_checkpoint=None, label="", unlabelled=(), tok=None):
    """Train the student on `train_windows`, optionally behind a teacher trained on `teacher_windows`
    (the same documents, usually only the plain window grid) that then labels every student window.
    A teacher checkpoint path is loaded when it exists and written after training otherwise."""
    teacher = None
    if args.teacher_base or teacher_checkpoint:
        if teacher_checkpoint and teacher_checkpoint.exists():
            teacher_model = load_checkpoint(teacher_checkpoint, device)
            print(f"{label}teacher loaded from {teacher_checkpoint}", flush=True)
        else:
            teacher_model = MultiTask(args.teacher_base, args.grad_checkpoint).to(device)
            print(f"{label}training teacher {args.teacher_base} on {len(teacher_windows)} windows", flush=True)
            train(teacher_model, teacher_windows, device, args.teacher_epochs, args.teacher_lr, args.accumulate, loss_weights(args), args.bf16, label=label + "teacher ")
            if teacher_checkpoint:
                save_checkpoint(teacher_model, args.teacher_base, teacher_checkpoint)
        teacher_tok = tokenizer(args.teacher_base)
        if unlabelled:
            pseudo_structure(teacher_model, teacher_tok, list(unlabelled), args.window, device, args.bf16)
            pool = [w for d in unlabelled for w in windows(tok, d, args.window) if w[3] or w[4]]
            random.Random(args.seed).shuffle(pool)
            extra = pool[: args.unlabeled_windows] if args.unlabeled_windows else pool
            print(f"{label}{len(extra)} of {len(pool)} teacher-structured windows from {len(unlabelled)} unlabelled documents join the student's training set", flush=True)
            train_windows = list(train_windows) + extra
        teacher = teacher_outputs(teacher_model, teacher_tok, texts, train_windows, device, args.bf16)
        teacher_model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        teacher_model = None
    model = MultiTask(args.base).to(device)  # base-size students fit without checkpointing
    print(f"{label}training {args.base} on {len(train_windows)} windows" + (" with distillation" if teacher else ""), flush=True)
    train(model, train_windows, device, args.epochs, args.lr, args.accumulate, loss_weights(args), args.bf16, teacher, args.distill_weight, label=label)
    return model, teacher_model


def report_header(args, docs):
    return {"base": args.base, "teacherBase": args.teacher_base, "sources": args.sources, "documents": len(docs), "folds": args.folds, "epochs": args.epochs, "lr": args.lr,
            "teacherEpochs": args.teacher_epochs, "teacherLr": args.teacher_lr, "distillWeight": args.distill_weight, "lossWeights": loss_weights(args),
            "novelWindows": args.novel_windows, "litbankRepeat": args.litbank_repeat, "window": args.window, "offsets": args.offsets, "teacherOffsets": args.teacher_offsets,
            "unlabeledWindows": args.unlabeled_windows, "unlabeledDocuments": sum(1 for d in docs if d.get("unlabeled")), "seed": args.seed}


def cv(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = device_()
    docs = load_documents(args.sources, args.limit_docs)
    tok, evaluation, training, teacher_training, texts = encode_all(docs, args)
    report = {**report_header(args, docs), "perFold": []}
    for i, held in enumerate(folds(docs, args.folds)):
        if args.only_fold is not None and i != args.only_fold:
            continue
        held_keys = {key(d) for d in held}
        train_windows = training_windows(docs, training, held_keys, args.offsets, args.litbank_repeat)
        teacher_windows = training_windows(docs, teacher_training, held_keys, args.teacher_offsets, args.litbank_repeat)
        eval_windows = [w for k in sorted(held_keys) for w in evaluation[k]]
        checkpoint = args.teacher_cache / f"fold{i}.pt" if args.teacher_cache else None
        unlabelled = [d for d in docs if d.get("unlabeled")]
        model, teacher_model = fit(args, train_windows, teacher_windows, texts, device, checkpoint, label=f"fold {i} ", unlabelled=unlabelled, tok=tok)
        result = evaluate(model, eval_windows, device, args.bf16)
        sources_held = sorted({k.rsplit("/", 1)[1] for k in held_keys})
        if len(sources_held) > 1:
            result["bySource"] = {src: evaluate(model, [w for k in sorted(held_keys) if k.endswith("/" + src) for w in evaluation[k]], device, args.bf16) for src in sources_held}
        if teacher_model is not None:
            teacher_eval = [w for k in sorted(held_keys) for w in encode(tokenizer(args.teacher_base), [d for d in held if key(d) == k], args, [0])[0][k]]
            result["teacher"] = evaluate(teacher_model.to(device), teacher_eval, device, args.bf16)
            teacher_model.cpu()
            print(f"fold {i} teacher: {summary(result['teacher'])}", flush=True)
        report["perFold"].append(result)
        print(f"fold {i}: {len(held)} docs {summary(result)}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del model, teacher_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pooled = {}
    for name in ("entity", "nominal"):
        tp = sum(f[name]["tp"] for f in report["perFold"]); fp = sum(f[name]["fp"] for f in report["perFold"]); fn = sum(f[name]["fn"] for f in report["perFold"])
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        pooled[name] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    for name in ("coref_proper_antecedent_accuracy", "coref_nominal_antecedent_accuracy", "coref_pronoun_antecedent_accuracy", "attribution_accuracy"):
        n = sum(f[name]["n"] for f in report["perFold"]); ok = sum(round(f[name]["accuracy"] * f[name]["n"]) for f in report["perFold"] if f[name]["accuracy"] is not None)
        pooled[name] = {"n": n, "accuracy": ok / n if n else None}
    report["pooled"] = pooled
    print("pooled:", json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in pooled.items()}))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def strip_shape_annotations(path):
    """Drop value_info so tract infers intermediate shapes from the inputs instead of unifying stale symbols."""
    import onnx

    graph = onnx.load(str(path))
    del graph.graph.value_info[:]
    onnx.save(graph, str(path))


def export(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = device_()
    docs = load_documents(args.sources, args.limit_docs)
    tok, _, training, teacher_training, texts = encode_all(docs, args)
    train_windows = training_windows(docs, training, set(), args.offsets, args.litbank_repeat)
    teacher_windows = training_windows(docs, teacher_training, set(), args.teacher_offsets, args.litbank_repeat)
    checkpoint = args.teacher_checkpoint or (args.teacher_cache / "all.pt" if args.teacher_cache else None)
    model, _ = fit(args, train_windows, teacher_windows, texts, device, checkpoint, unlabelled=[d for d in docs if d.get("unlabeled")], tok=tok)
    if args.save:
        save_checkpoint(model, args.base, args.save)
    model = model.cpu().eval()
    staging = args.output / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    ids = torch.tensor([[tok.cls_token_id] + tok("\"Come in,\" said Alice to her sister.", add_special_tokens=False)["input_ids"] + [tok.sep_token_id]])
    example = (ids, torch.ones_like(ids), torch.tensor([5, 7]), torch.tensor([5, 8]), torch.tensor([1]), torch.tensor([4]))
    # The span heads index one window's hidden states, so the batch axis stays 1; a
    # symbolic batch also breaks tract (wasm), which cannot shape a Flatten over
    # batch x sequence. Intermediate shape annotations are dropped for the same reason.
    torch.onnx.export(
        model, example, str(staging / "model-fp32.onnx"),
        input_names=["input_ids", "attention_mask", "mention_starts", "mention_ends", "quote_starts", "quote_ends"],
        output_names=["entity_logits", "nominal_logits", "coref_scores", "attribution_scores"],
        dynamic_axes={"input_ids": {1: "sequence"}, "attention_mask": {1: "sequence"}, "mention_starts": {0: "mentions"}, "mention_ends": {0: "mentions"},
                      "quote_starts": {0: "quotes"}, "quote_ends": {0: "quotes"}, "entity_logits": {1: "sequence"}, "nominal_logits": {1: "sequence"},
                      "coref_scores": {0: "mentions", 1: "mentions"}, "attribution_scores": {0: "quotes", 1: "mentions_plus_dummy"}},
        opset_version=14, dynamo=False,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(staging / "model-fp32.onnx"), str(staging / "model.onnx"), weight_type=QuantType.QInt8)
    (staging / "model-fp32.onnx").unlink()
    strip_shape_annotations(staging / "model.onnx")
    sha = hashlib.sha256((staging / "model.onnx").read_bytes()).hexdigest()
    final = args.output / sha
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    backend = tok.backend_tokenizer
    backend.save(str(final / "tokenizer.json"))
    backend.no_padding(); backend.no_truncation()
    backend.save(str(final / "tokenizer.adapted.json"))
    # manifest.json is the crate's strict ModelManifest (profile narrative_multitask_v1); provenance goes beside it.
    manifest = {
        "schemaVersion": 1, "kind": "narrative_multitask", "profile": "narrative_multitask_v1", "modelSha256": sha,
        "tokenizerSha256": hashlib.sha256((final / "tokenizer.adapted.json").read_bytes()).hexdigest(), "maxTokens": args.window + 2, "paddingTokenId": tok.pad_token_id or 0,
        "labels": ENTITY_LABELS, "nominalLabels": NOMINAL_LABELS, "windowTokens": args.window,
    }
    provenance = {
        "window": args.window,
        "inputs": ["input_ids", "attention_mask", "mention_starts", "mention_ends", "quote_starts", "quote_ends"],
        "outputs": ["entity_logits", "nominal_logits", "coref_scores", "attribution_scores"],
        "notes": "coref_scores[i][j] scores mention j as antecedent of mention i (j < i); 0 is the no-antecedent score. attribution_scores[q][0] is 'speaker not in window', column m+1 is mention m.",
        "training": {**report_header(args, docs), "teacherCheckpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None},
    }
    (final / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (final / "training.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(final, (final / "model.onnx").stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cv", "export"])
    parser.add_argument("--sources", nargs="+", default=["litbank", "pdnc"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", default=DEFAULT_BASE, help="HuggingFace encoder for the exported model")
    parser.add_argument("--teacher-base", help="larger encoder trained first on the same windows; its distributions supervise the student")
    parser.add_argument("--teacher-checkpoint", type=Path, help="export only: teacher checkpoint to load (or to write after training)")
    parser.add_argument("--teacher-cache", type=Path, help="directory of per-fold teacher checkpoints (fold<i>.pt, all.pt) loaded when present and written otherwise")
    parser.add_argument("--save", type=Path, help="export only: where to save the trained student checkpoint")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--teacher-epochs", type=int, default=3)
    parser.add_argument("--teacher-lr", type=float, default=2e-5)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--entity-weight", type=float, default=1.0, help="multiplier on the entity head's loss")
    parser.add_argument("--nominal-weight", type=float, default=1.0, help="multiplier on the nominal head's loss")
    parser.add_argument("--window", type=int, default=400, help="content tokens per window (plus CLS and SEP)")
    parser.add_argument("--offsets", type=int, nargs="+", default=[0], help="window grid offsets used for training (0 = plain grid)")
    parser.add_argument("--teacher-offsets", type=int, nargs="+", default=None, help="window grid offsets for teacher training (default: --offsets)")
    parser.add_argument("--novel-windows", type=int, default=40, help="windows kept per full novel (PDNC); 0 keeps all")
    parser.add_argument("--litbank-repeat", type=int, default=1, help="how many times LitBank windows recur per epoch")
    parser.add_argument("--only-fold", type=int, help="run a single fold (for comparing data mixes on one held-out set)")
    parser.add_argument("--limit-docs", type=int, default=0, help="documents per source (smoke tests)")
    parser.add_argument("--unlabeled-windows", type=int, default=30000, help="teacher-structured windows sampled from unlabelled sources per run; 0 keeps all")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="bfloat16 autocast on CUDA")
    args = parser.parse_args()
    if args.teacher_offsets is None:
        args.teacher_offsets = args.offsets
    if args.output is None:
        args.output = ROOT / ("target/evaluation/narrative-multitask-cv.json" if args.mode == "cv" else "target/external-validation/narrative-multitask-model")
    (cv if args.mode == "cv" else export)(args)


if __name__ == "__main__":
    main()
