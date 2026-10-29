# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==2.4.6", "onnxruntime==1.27.0", "tokenizers==0.23.3"]
# ///
"""Compare verified native NER spans to independent Python ORT execution,locally."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
TEXTS = [
    "Alice met Bob in Paris. Microsoft hired John Smith.",
    "😀 Zoë spoke with José in São Paulo.",
    "Dr. Watson and Sherlock Holmes visited London on Tuesday.",
    "Alice said , ' I do n't know , Bob . ' Mary waited .",
    "No proper names appear here. A person watched the garden.",
]
CANONICAL = "O B-MISC I-MISC B-PER I-PER B-ORG I-ORG B-LOC I-LOC".split()
HONORIFICS = set("mr mrs miss ms mme mlle dr sir lady lord madam madame monsieur colonel captain major general sergeant professor aunt uncle king queen prince princess judge reverend rev st saint".split())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def reference_spans(text, tokenizer, session, source_labels):
    encoding = tokenizer.encode(text, add_special_tokens=True)
    values = {"input_ids": encoding.ids, "attention_mask": encoding.attention_mask,
              "token_type_ids": encoding.type_ids}
    inputs = {value.name: np.array([values[value.name]], dtype=np.int64) for value in session.get_inputs()}
    logits = session.run(None, inputs)[0][0]
    if logits.shape != (len(encoding.ids), 9) or not np.isfinite(logits).all():
        raise ValueError('reference NER shape or values invalid')
    order = [source_labels.index(label) for label in CANONICAL]
    predictions = np.argmax(logits[:, order], axis=-1)
    spans = []
    active = None
    def flush():
        nonlocal active
        if active is not None:
            kind, start, end = active
            if any(ch.isalnum() for ch in text[start:end]):
                spans.append([kind, start, end])
        active = None
    previous_word = None
    for index, (start, end), word in zip(predictions, encoding.offsets, encoding.word_ids):
        label = CANONICAL[index]
        if start == end:
            flush()
            previous_word = None
            continue
        continuation = word is not None and word == previous_word
        previous_word = word
        if continuation:
            if active is not None:
                active = (active[0], active[1], end)
        elif label == 'O':
            flush()
        elif label.startswith('I-') and active is not None and active[0] == label[2:]:
            active = (active[0], active[1], end)
        else:
            flush()
            active = (label[2:], start, end)
    flush()
    floor = 0
    for span in spans:
        if span[0] == 'PER':
            start = span[1] + next((i for i, ch in enumerate(text[span[1]:span[2]]) if ch.isalnum()), 0)
            head = text[floor:start]
            gap = head.rstrip('. ')
            separator = head[len(gap):]
            index = len(gap)
            while index > 0 and gap[index - 1].isalpha():
                index -= 1
            word = gap[index:]
            attach = (bool(separator) and separator.count('.') <= 1 and separator.count(' ') <= 1
                      and word[:1].isupper() and word.lower() in HONORIFICS)
            span[1] = floor + index if attach else start
        floor = span[2]
    spans = [[text[start:end], kind, len(text[:start].encode()), len(text[:end].encode())] for kind, start, end in spans]
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / 'docs/benchmarks/ner-reference.json').read_text())
    cache = ROOT / 'target/external-validation/ner-model' / spec['revision']
    manifest = json.loads((cache / 'manifest.json').read_text())
    if digest(cache / 'model.onnx') != manifest['modelSha256'] or digest(cache / 'tokenizer.adapted.json') != manifest['tokenizerSha256']:
        raise ValueError('cached model identity changed')
    cases = [{'id': str(i), 'split': 'development', 'entityText': text, 'quotationText': text, 'suppliedCast': []}
             for i, text in enumerate(TEXTS)]
    inputs = cache / 'parity-inputs.json'
    inputs.write_text(json.dumps(cases, ensure_ascii=False) + '\n')
    command = [str(args.predictor.resolve()), str(inputs), '--ner-cache', str(cache), '--split', 'development']
    native = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    predictions = [json.loads(line) for line in native.stdout.splitlines() if line.strip()]
    if len(predictions) != len(cases) or [row['id'] for row in predictions] != [row['id'] for row in cases]:
        raise ValueError('native reference prediction cardinality/order mismatch')
    tokenizer = Tokenizer.from_file(str(cache / 'tokenizer.adapted.json'))
    session = ort.InferenceSession(str(cache / 'model.onnx'), providers=['CPUExecutionProvider'])
    comparisons = []
    for case, native_row in zip(cases, predictions):
        if 'failure' in native_row or native_row['modelIdentity']['manifest'] != manifest:
            raise ValueError('native model failed or loaded a different identity')
        expected = reference_spans(case['entityText'], tokenizer, session, manifest['labels'])
        comparisons.append({'id': case['id'], 'text': case['entityText'], 'referenceSpans': expected,
                            'nativeSpans': native_row['nerSpans'], 'exactMatch': expected == native_row['nerSpans']})
    report = {'scope': 'Engineering execution/tokenization/BIO fidelity only;authored cases are not independent accuracy labels.',
              'manifest': manifest, 'command': command, 'predictorSha256': digest(args.predictor),
              'inputsSha256': digest(inputs), 'versions': {name: importlib.metadata.version(name) for name in ['onnxruntime', 'tokenizers', 'numpy']},
              'provider': session.get_providers(), 'comparisons': comparisons,
              'pass': all(row['exactMatch'] for row in comparisons)}
    path = ROOT / 'docs/audit-2026-10-27/program-ner-parity.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'cases': len(comparisons), 'pass': report['pass'], 'report': str(path)}))
    if not report['pass']:
        raise SystemExit('native NER spans differ from independent Python ORT')


if __name__ == '__main__':
    main()
