"""Deterministic tests for transcript edit-distance comparison."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location(
    "compare_asr_transcripts",
    SCRIPTS_DIR / "compare-asr-transcripts.py",
)
assert SPEC is not None and SPEC.loader is not None
COMPARISON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARISON
SPEC.loader.exec_module(COMPARISON)


class TranscriptComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self.temporary.name)
        self.manifest = self.directory / "manifest.json"
        self.reference = self.directory / "reference"
        self.candidate = self.directory / "candidate"
        self.reference.mkdir()
        self.candidate.mkdir()
        self.videos = [
            {"url": "https://example.test/watch?v=one"},
            {"url": "https://example.test/watch?v=two"},
        ]
        self.manifest.write_text(
            json.dumps({"videos": self.videos}),
            encoding="utf-8",
        )
        for index, video in enumerate(self.videos):
            name = COMPARISON.transcript_path(
                self.reference,
                index,
                video["url"],
            ).name
            (self.reference / name).write_text(
                "Reference text\n",
                encoding="utf-8",
            )
            (self.candidate / name).write_text(
                "Candidate text\n",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validation(self, name: str, invalid: set[int]) -> pathlib.Path:
        path = self.directory / name
        path.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "index": index,
                            "issues": (
                                ["incomplete"] if index in invalid else []
                            ),
                        }
                        for index in range(len(self.videos))
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_uses_only_the_complete_validation_intersection(self) -> None:
        reference_validation = self.validation(
            "reference-validation.json",
            {1},
        )
        candidate_validation = self.validation(
            "candidate-validation.json",
            set(),
        )
        arguments = types.SimpleNamespace(
            manifest=self.manifest,
            reference_dir=self.reference,
            candidate_dir=self.candidate,
            reference_validation=reference_validation,
            candidate_validation=candidate_validation,
            reference_label="reference",
            candidate_label="candidate",
        )

        result = COMPARISON.compare(arguments)

        self.assertEqual(result["manifest_source_count"], 2)
        self.assertEqual(result["source_count"], 1)
        self.assertEqual(result["excluded_source_count"], 1)
        self.assertEqual(result["sources"][0]["index"], 0)
        self.assertNotIn("Reference text", json.dumps(result))
        self.assertNotIn("Candidate text", json.dumps(result))

    def test_rejects_incomplete_validation_data(self) -> None:
        path = self.directory / "incomplete-validation.json"
        path.write_text(
            json.dumps({"sources": [{"index": 0, "issues": []}]}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "complete manifest"):
            COMPARISON.validated_source_indexes(path, self.videos)


if __name__ == "__main__":
    unittest.main()
