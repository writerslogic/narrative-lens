# /// script
# requires-python = ">=3.10"
# ///
"""Check pinned readability data without network access."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
record = json.loads((root / "docs/readability-resources.json").read_text())
for filename, expected in [
    (record["resource"], record["resourceSha256"]),
    (record["licenseFile"], record["licenseSha256"]),
]:
    content = (root / filename).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest() == expected, filename
words = (root / record["resource"]).read_text().splitlines()
assert len(words) == record["wordCount"] == 2942
assert words == sorted(set(words))
assert (root / record["licenseFile"]).read_text().strip() in (root / "THIRD_PARTY_NOTICES").read_text()
print(f"Verified {len(words)} pinned Dale-Chall entries and the distributed MIT notice")
