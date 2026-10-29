import json
import subprocess
from pathlib import Path

repository = Path(__file__).resolve().parents[2]
evidence = Path(__file__).resolve().parent
metadata = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=repository
))
cli = Path(metadata["target_directory"]) / "debug" / "narrative-lens"
scene = evidence / "ci-probe.txt"
scene.write_text("She saw saw the bird.")
command = [str(cli), "--format", "json", "--rules", "NL-GRAMMAR-REPETITION", str(scene)]
before = subprocess.run(command, capture_output=True, text=True, check=True)
baseline = evidence / "ci-baseline.json"
baseline.write_text(before.stdout)
configuration = evidence / "limit.json"
configuration.write_text(json.dumps({"limits": {"maxFindings": 1}}))
scene.write_text("She saw saw the bird. He ran ran through town.")
after = subprocess.run(command + ["--baseline", str(baseline), "--new-only", "--fail-on", "warning", "--config", str(configuration)], capture_output=True, text=True)
report = json.loads(after.stdout)
(evidence / "ci-reproduction.json").write_text(after.stdout)
print(json.dumps({"probe": "new-only-omission", "exit": after.returncode, "summary": report["report"]["summary"], "comparison": {key: len(value) for key, value in report["comparison"].items()}}))
regions = []
for bom in [False, True]:
    scene = evidence / ("bom.txt" if bom else "no-bom.txt")
    scene.write_text(("\ufeff" if bom else "") + "😀 She saw saw the bird.")
    output = subprocess.run([str(cli), "--format", "sarif", "--rules", "NL-GRAMMAR-REPETITION", str(scene)], capture_output=True, text=True, check=True)
    document = json.loads(output.stdout)
    regions.append({"bom": bom, "region": document["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]})
(evidence / "sarif-bom-reproduction.json").write_text(json.dumps(regions, indent=2) + "\n")
print(json.dumps({"probe": "sarif-bom", "values": regions}))
