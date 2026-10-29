"""Generate tiny untrained ONNX graphs using the official protobuf wire schema.

Schema: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
These constant/Cast graphs test runtime contracts, not language accuracy.
"""
import hashlib
import json
import struct
import math
from pathlib import Path


def varint(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    return bytes(out) + bytes([value])


def integer(field, value):
    return varint(field << 3) + varint(value)


def blob(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((field << 3) | 2) + varint(len(value)) + value


def value_info(name, element_type, dimensions):
    shape = b"".join(blob(1, integer(1, d) if isinstance(d, int) else blob(2, d)) for d in dimensions)
    tensor_type = integer(1, element_type) + blob(2, shape)
    return blob(1, name) + blob(2, blob(1, tensor_type))


def model(classes, failing=False, dimensions=None):
    dimensions = dimensions or [1, classes]
    inputs = [value_info("input_ids", 7, ["batch", "tokens"]), value_info("attention_mask", 7, ["batch", "tokens"])]
    if failing:
        inputs.append(value_info("position_ids", 7, ["batch", "tokens"]))
        attribute = blob(1, "to") + integer(3, 1) + integer(20, 2)
        node = blob(1, "position_ids") + blob(2, "logits") + blob(4, "Cast") + blob(5, attribute)
        output = value_info("logits", 1, ["batch", "tokens"])
    else:
        size = math.prod(dimensions)
        tensor = b"".join(integer(1, d) for d in dimensions) + integer(2, 1) + blob(9, struct.pack("<" + "f" * size, *([3.0] + [0.0] * (size - 1))))
        attribute = blob(1, "value") + blob(5, tensor) + integer(20, 4)
        node = blob(2, "logits") + blob(4, "Constant") + blob(5, attribute)
        output = value_info("logits", 1, dimensions)
    graph = blob(1, node) + blob(2, "narrative-lens-contract-fixture") + b"".join(blob(11, i) for i in inputs) + blob(12, output)
    return integer(1, 8) + blob(2, "narrative-lens engineering test") + blob(7, graph) + blob(8, integer(2, 13))


root = Path(__file__).resolve().parents[1] / "tests/fixtures/models"
root.mkdir(exist_ok=True)
manifest = {"origin": "Agent-authored untrained constant/Cast ONNX graphs; no downloaded weights or editorial labels", "license": "Apache-2.0", "generator": "scripts/generate_model_fixtures.py", "files": {}}
for capability, dimensions, incompatible in [
    ("nli", [1, 3], [1, 2]),
    ("emotion", [1, 7], [1, 6]),
    ("ner", [1, 1, 9], [1, 1, 8]),
    ("embedding", [1, 1, 2], [1, 2]),
]:
    for status in ["valid", "incompatible", "failing"]:
        dims = incompatible if status == "incompatible" else dimensions
        data = model(dims[-1], status == "failing", dims)
        name = f"{status}-{capability}.onnx"
        (root / name).write_bytes(data)
        manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
(root / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
