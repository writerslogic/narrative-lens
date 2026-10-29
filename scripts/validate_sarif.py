# /// script
# requires-python = ">=3.10"
# dependencies = ["jsonschema==4.25.1"]
# ///
"""Validate official SARIF output or generated manuscript contracts."""
import hashlib
import json
import sys
from pathlib import Path

from jsonschema import Draft4Validator, validators

if sys.argv[1:] in (["--contracts"], ["--standalone-contracts"]):
    filename = "standalone.schema.json" if sys.argv[1] == "--standalone-contracts" else "manuscript.schema.json"
    schemas = json.loads((Path(__file__).resolve().parents[1] / "docs" / filename).read_text())
    bundle = json.load(sys.stdin)
    for name, value in bundle.items():
        schema = schemas[name]
        validator_type = validators.validator_for(schema)
        validator_type.check_schema(schema)
        validator_type(schema).validate(value)
        print(f"Valid generated contract: {name}")
    raise SystemExit(0)

schema_path = Path(__file__).resolve().parents[1] / "tests/fixtures/sarif-schema-2.1.0.json"
assert hashlib.sha256(schema_path.read_bytes()).hexdigest() == "c3b4bb2d6093897483348925aaa73af03b3e3f4bd4ca38cef26dcb4212a2682e"
schema = json.loads(schema_path.read_text())
Draft4Validator.check_schema(schema)
validator = Draft4Validator(schema)
for filename in sys.argv[1:]:
    validator.validate(json.loads(Path(filename).read_text()))
    print(f"Valid SARIF 2.1.0: {filename}")
if len(sys.argv) < 2:
    raise SystemExit("provide at least one SARIF output file")
