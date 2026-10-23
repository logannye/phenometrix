import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phenometrix_prosody.data import (BASE, SEED, authorize_split, digest,
    download_record, inventory_audit, parse_listing, parse_prompts, read_reference,
    select_records, speaker_split, split_assignment)


class DataContracts(unittest.TestCase):
    def test_frozen_group_split(self):
        self.assertEqual(split_assignment(), {"train": [4,8,7,9,10,2], "development": [1,3], "sealed_test": [5,6]})
        for pair in range(1, 11):
            self.assertEqual(speaker_split(f"F{pair:02}"), speaker_split(f"M{pair:02}"))
        with self.assertRaises(ValueError):
            speaker_split("F11")

    def test_listing_identity_and_prompt_exclusion(self):
        text = '<table><tr><td><a href="mic_F04_sa1.wav">sa1</a></td></tr><tr><td><a href="mic_F04_si1020.wav">si1020</a></td></tr></table>'
        result = parse_listing(text, BASE + "SPEECH%20DATA/FEMALE/MIC/F04/", "F04", "mic")
        self.assertEqual([r["primary_eligible"] for r in result], [False, True])
        self.assertEqual(result[1]["split"], "train")
        with self.assertRaises(ValueError):
            parse_listing(text, BASE, "F03", "mic")

    def test_normalized_prompt_overlap_fails_gate(self):
        records = []
        for speaker, prompt in [("F04", "si1020"), ("F01", "si453")]:
            for role in ("mic", "ref", "lar"):
                records.append({"recording_id": f"{speaker}_{prompt}", "speaker": speaker, "prompt": prompt,
                    "role": role, "split": speaker_split(speaker), "primary_eligible": True})
        audit = inventory_audit(records, {"si1020": "same text", "si453": "same text"})
        self.assertFalse(audit["no_shared_prompt_gate"])
        self.assertEqual(audit["primary_prompt_intersections"]["train__development"]["normalized_text_count"], 1)

    def test_parser_rejects_duplicate_or_nonfinite_targets(self):
        with self.assertRaises(ValueError):
            parse_prompts("Text. (sa1)\nOther. (sa1)")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ref.f0"
            path.write_text("100 1 3 0.9\n0 0 2 0.1\n")
            self.assertEqual(len(read_reference(path)), 2)
            path.write_text("nan 1 2 3\n")
            with self.assertRaises(ValueError):
                read_reference(path)
            path.write_text("1 2 3\n")
            with self.assertRaises(ValueError):
                read_reference(path)

    def test_sealed_test_requires_bound_freeze_before_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory)/"inventory.json"
            inventory.write_text("{}")
            with self.assertRaises(ValueError):
                authorize_split("sealed_test", inventory)
            freeze = Path(directory)/"freeze.json"
            freeze.write_text(json.dumps({"schema": "phenometrix-prosody-freeze-v1", "sealed_test_authorized": True,
                "source_revision": "abc", "protocol_sha256": "a", "model_sha256": "b", "configuration_sha256": "c",
                "inventory_sha256": "wrong"}))
            with self.assertRaises(ValueError):
                authorize_split("sealed_test", inventory, freeze)
            value = json.loads(freeze.read_text()); value["inventory_sha256"] = digest(inventory.read_bytes())
            freeze.write_text(json.dumps(value))
            authorize_split("sealed_test", inventory, freeze)

    def test_default_selection_has_no_universal_or_test_content(self):
        rows = []
        for speaker in ("F04", "M04", "F05"):
            for prompt in ("sa1", "sx138", "si1020", "si1021"):
                for role in ("mic", "ref", "lar"):
                    rows.append({"speaker": speaker, "prompt": prompt, "role": role,
                                 "split": speaker_split(speaker), "primary_eligible": prompt != "sa1"})
        inventory = {"audit": {"no_shared_prompt_gate": True}, "records": rows}
        selected = select_records(inventory)
        self.assertEqual(len(selected), 12)
        self.assertEqual({r["speaker"] for r in selected}, {"F04", "M04"})
        self.assertNotIn("sa1", {r["prompt"] for r in selected})
        self.assertEqual({r["prompt"] for r in selected if r["speaker"] == "F04"}, {r["prompt"] for r in selected if r["speaker"] == "M04"})

    def test_download_rejects_wrong_source_and_budget_before_network(self):
        record = {"url": "https://evil.example/mic_F04_si1020.wav", "filename": "mic_F04_si1020.wav",
                  "speaker": "F04", "prompt": "si1020", "split": "train", "role": "mic"}
        with tempfile.TemporaryDirectory() as directory, patch("phenometrix_prosody.data.urlopen") as fetch:
            with self.assertRaises(ValueError):
                download_record(record, directory)
            record["url"] = BASE + "SPEECH%20DATA/FEMALE/MIC/F04/" + record["filename"]
            with self.assertRaises(ValueError):
                download_record(record, directory, max_bytes=0)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
