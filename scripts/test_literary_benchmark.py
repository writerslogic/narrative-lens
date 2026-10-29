# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import literary_benchmark as benchmark


class ScoringContracts(unittest.TestCase):
    def fixture(self, directory, *, resolved="Al", aliases=None):
        aliases = aliases or {"alice": ["1"], "al": ["1"], "bob": ["2"]}
        gold = [{"id": "one", "split": "development", "properPersonNames": ["alice", "bob"], "properPersonGroups": ["1", "2"], "aliasGroups": aliases, "pronouns": [{"start": 0, "end": 3, "groups": ["1"]}], "quotes": [{"quote": "hello.", "speaker": "Alice-1"}, {"quote": "yes.", "speaker": "Bob-2"}, {"quote": "yes.", "speaker": "Bob-2"}]}]
        prediction = {"id": "one", "split": "development", "personNames": [["Al", 1], ["Bob", 1]], "coreferenceSuppliedCast": {"mentions": [{"isPronoun": True, "start": 0, "text": "She", "resolvedTo": resolved}]}, "dialogueSuppliedCast": [{"quote": "Hello.", "speaker": "Alice"}, {"quote": "Hello.", "speaker": "Alice"}, {"quote": "Yes.", "speaker": "Bob"}, {"quote": "she heard it", "speaker": "Alice", "method": "indirect_speech"}]}
        benchmark.write(directory / "gold.json", gold)
        benchmark.write(directory / "inputs.json", [])
        benchmark.write(directory / "manifest.json", {"inputsSha256": benchmark.fingerprint((directory / "inputs.json").read_bytes()), "goldSha256": benchmark.fingerprint((directory / "gold.json").read_bytes())})
        (directory / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
        return directory / "predictions.jsonl"

    def score(self, directory, predictions):
        with contextlib.redirect_stdout(io.StringIO()):
            benchmark.score(directory, predictions, "development", directory / "report.json")
        return json.loads((directory / "report.json").read_text())

    def test_alias_inventory_and_quote_multiset_use_distinct_units(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            report = self.score(directory, self.fixture(directory))
            metrics = report["metrics"]
            self.assertEqual([metrics["properPersonNames"][k] for k in ["tp", "fp", "fn"]], [1, 1, 1])
            self.assertEqual([metrics["namedCharacterInventory"][k] for k in ["tp", "fp", "fn"]], [2, 0, 0])
            self.assertEqual([metrics["dialogueSuppliedCast"][k] for k in ["tp", "fp", "fn"]], [2, 1, 1])
            self.assertEqual(metrics["dialogueSuppliedCast"]["unassessedIndirect"], 1)
            speaker = metrics["speakerSuppliedCast"]
            self.assertEqual(speaker["unassessedRepeatedText"], 2)
            self.assertEqual(speaker["total"], 3)
            self.assertAlmostEqual(speaker["assessmentCoverage"], 1 / 3)
            self.assertAlmostEqual(speaker["recallOverAllGoldLowerBound"], 1 / 3)

    def test_ambiguous_alias_cannot_get_coreference_credit(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            report = self.score(directory, self.fixture(directory, aliases={"alice": ["1"], "al": ["1", "2"], "bob": ["2"]}))
            self.assertEqual([report["metrics"]["coreferenceSuppliedCast"][k] for k in ["tp", "fp", "fn"]], [0, 1, 1])

    def test_abstention_does_not_become_perfect_accuracy(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            report = self.score(directory, self.fixture(directory, resolved=None))
            result = report["metrics"]["coreferenceSuppliedCast"]
            self.assertIsNone(result["precision"])
            self.assertEqual(result["recall"], 0)
            self.assertEqual(result["decisionCoverage"], 0)

    def test_tampered_gold_and_missing_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            predictions = self.fixture(directory)
            (directory / "gold.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "changed"):
                self.score(directory, predictions)
            self.fixture(directory)
            predictions.write_text("")
            with self.assertRaisesRegex(ValueError, "missing"):
                self.score(directory, predictions)

    def exact_fixture(self, directory):
        text='😀 "Yes." "Yes."'
        def location(start, end):
            return {"byteStart":len(text[:start].encode()),"byteEnd":len(text[:end].encode()),
                    "utf16Start":len(text[:start].encode('utf-16-le'))//2,"utf16End":len(text[:end].encode('utf-16-le'))//2}
        quotes=[]
        located=[]
        for start,speaker in [(2,'Alice-1'),(9,'Bob-2')]:
            outer=location(start,start+6)
            quotes.append({"quotationBytes":[outer['byteStart'],outer['byteEnd']],"speaker":speaker})
            located.append({"kind":"direct","line":{"quote":"Yes.","speaker":"Alice"},"range":location(start+1,start+5),"quotationRange":outer,"attributionRange":None})
        return text,{"quotes":quotes,"aliasGroups":{"alice":["1"],"bob":["2"]}},{"dialogueExplicitSuppliedCast":{"lines":located}}

    def test_exact_spans_disambiguate_repeated_quote_speakers(self):
        with tempfile.TemporaryDirectory() as path:
            text,gold,prediction=self.exact_fixture(Path(path))
            metrics=benchmark.defaultdict(benchmark.Counter)
            benchmark.exact_dialogue_metrics(gold,prediction,text,'SuppliedCast',metrics)
            self.assertEqual(metrics['dialogueExplicitSuppliedCastExact']['tp'],2)
            speaker=metrics['speakerExplicitSuppliedCastExact']
            self.assertEqual([speaker[key] for key in ['tp','fp','fn']],[1,1,1])
            self.assertEqual(speaker['unassessedRepeatedText'],0)

    def test_predicted_byte_utf16_and_source_text_are_independently_validated(self):
        with tempfile.TemporaryDirectory() as path:
            text,gold,prediction=self.exact_fixture(Path(path))
            located=prediction['dialogueExplicitSuppliedCast']['lines'][0]
            for key,value,pattern in [('byteStart',1,'UTF8'),('utf16Start',0,'UTF16')]:
                saved=located['range'][key]
                located['range'][key]=value
                with self.assertRaisesRegex(ValueError,pattern):
                    benchmark.exact_dialogue_metrics(gold,prediction,text,'SuppliedCast',benchmark.defaultdict(benchmark.Counter))
                located['range'][key]=saved
            located['line']['quote']='No.'
            with self.assertRaisesRegex(ValueError,'original quote'):
                benchmark.exact_dialogue_metrics(gold,prediction,text,'SuppliedCast',benchmark.defaultdict(benchmark.Counter))

    def test_quotation_suffix_ids_do_not_share_coreference_namespace(self):
        text,gold,prediction=self.exact_fixture(Path('.'))
        gold['quotes'][0]['speaker']='Alice-917'
        metrics=benchmark.defaultdict(benchmark.Counter)
        benchmark.exact_dialogue_metrics(gold,prediction,text,'SuppliedCast',metrics)
        self.assertEqual(metrics['speakerExplicitSuppliedCastExact']['tp'],1)
        gold['aliasGroups']['alice']=['1','3']
        metrics=benchmark.defaultdict(benchmark.Counter)
        benchmark.exact_dialogue_metrics(gold,prediction,text,'SuppliedCast',metrics)
        result=metrics['speakerExplicitSuppliedCastExact']
        self.assertEqual(result['unassessedIdentity'],1)
        self.assertAlmostEqual(benchmark.metric_summary(result)['assessmentCoverage'],0.5)


if __name__ == "__main__":
    unittest.main()
