# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "onnxruntime", "tokenizers"]
# ///
"""Reference decode of the narrative multi-task model on the passage used by tests/component_narrative_model.rs.

    uv run scripts/narrative_model_reference.py target/external-validation/narrative-multitask-model/<sha>

Prints the onnxruntime argmax antecedent and speaker per mention/quote so a Rust run can be compared by hand."""
import bisect
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

directory = sys.argv[1]
tokenizer_file = sys.argv[2] if len(sys.argv) > 2 else "tokenizer.adapted.json"
TEXT = ("Alice Bennet walked into the parlour and sat by the fire. \"Come in, Jane,\" said Alice to her sister. "
        "Jane Bennet looked up from her book. \"I will,\" she replied, closing it.")


def find(needle, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = TEXT.index(needle, start + 1)
    return start, start + len(needle)


mentions = [find("Alice Bennet"), find("Jane"), find("Alice", 1), find("her"), find("her sister"), find("Jane Bennet"), find("her", 1), find("she")]
quotes = [find("\"Come in, Jane,\""), find("\"I will,\"")]
tok = Tokenizer.from_file(f"{directory}/{tokenizer_file}")
enc = tok.encode(TEXT, add_special_tokens=False)
ids, offsets = enc.ids, enc.offsets
print("tokens:", list(zip(enc.tokens, offsets))[:12], "...", len(ids))
starts = [a for a, _ in offsets]
ends = [b for _, b in offsets]


def token_span(s, e):
    ts = bisect.bisect_right(ends, s)
    te = bisect.bisect_left(starts, e) - 1
    assert 0 <= ts <= te < len(ids), (s, e)
    return ts + 1, te + 1


ms = [token_span(s, e) for s, e in mentions]
qs = [token_span(s, e) for s, e in quotes]
print("mention token spans:", ms)
cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
window = [cls] + ids + [sep]
session = ort.InferenceSession(f"{directory}/model.onnx", providers=["CPUExecutionProvider"])
feeds = {
    "input_ids": np.array([window], dtype=np.int64),
    "attention_mask": np.ones((1, len(window)), dtype=np.int64),
    "mention_starts": np.array([a for a, _ in ms], dtype=np.int64),
    "mention_ends": np.array([b for _, b in ms], dtype=np.int64),
    "quote_starts": np.array([a for a, _ in qs], dtype=np.int64),
    "quote_ends": np.array([b for _, b in qs], dtype=np.int64),
}
entity, nominal, coref, attrib = session.run(None, feeds)
labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
print("entity:", [(t, labels[l]) for t, l in zip(enc.tokens, entity[0, 1:-1].argmax(-1)) if l])
print("nominal:", [(t, l) for t, l in zip(enc.tokens, nominal[0, 1:-1].argmax(-1)) if l])
m = len(ms)
for i in range(m):
    row = np.concatenate([[0.0], coref[i, :i]])
    best = int(row.argmax())
    p = np.exp(row - row.max()); p /= p.sum()
    print(f"mention {i} {TEXT[slice(*mentions[i])]!r}: antecedent {best - 1 if best else None} p={p[best]:.3f} row={np.round(row, 2).tolist()}")
for q in range(len(qs)):
    row = attrib[q].copy()
    for j, (s, e) in enumerate(mentions):
        if s >= quotes[q][0] and e <= quotes[q][1]:
            row[j + 1] = -np.inf
    best = int(row.argmax())
    p = np.exp(row - row.max()); p /= p.sum()
    print(f"quote {q}: speaker {best - 1 if best else None} p={p[best]:.3f} row={np.round(row, 2).tolist()}")
