"""Deterministic tests for the ASR timestamp artifact analyzer."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from typing import Any


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location(
    "analyze_asr_timestamps",
    SCRIPTS_DIR / "analyze-asr-timestamps.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)


def wrapper(index: int, url: str, event: dict[str, Any]) -> dict[str, Any]:
    return {"source_index": index, "source_url": url, "event": event}


def results_event(transcript: str, words: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "Results",
        "is_final": True,
        "channel": {
            "alternatives": [
                {
                    "transcript": transcript,
                    "words": words,
                }
            ]
        },
    }


class TimestampArtifactAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self.temporary.name)
        self.urls = [
            "https://example.test/watch?v=alpha",
            "https://example.test/watch?v=silent",
        ]
        self.manifest = self.directory / "manifest.json"
        self.baseline = self.directory / "baseline.ndjson"
        self.timestamp = self.directory / "timestamp.ndjson"
        self.manifest.write_text(
            json.dumps(
                {
                    "videos": [
                        {"url": self.urls[0], "duration_seconds": 10.0},
                        {"url": self.urls[1], "duration_seconds": 1.0},
                    ]
                }
            ),
            encoding="utf-8",
        )

        baseline_records = [
            wrapper(
                0,
                self.urls[0],
                {"type": "ResponseHead", "status": 503},
            ),
            wrapper(
                0,
                self.urls[0],
                {"type": "ResponseHead", "status": 200},
            ),
            wrapper(
                0,
                self.urls[0],
                {"type": "Metadata", "request_id": "baseline-alpha"},
            ),
            wrapper(
                0,
                self.urls[0],
                results_event(
                    "Hello world world",
                    [
                        {"word": "Hello", "start": 0.0, "end": 0.4},
                        {"word": "world", "start": 0.5, "end": 0.9},
                        {"word": "world", "start": 5.2, "end": 5.2},
                    ],
                ),
            ),
            wrapper(
                1,
                self.urls[1],
                {"type": "ResponseHead", "status": 200},
            ),
            wrapper(
                1,
                self.urls[1],
                {"type": "Metadata", "request_id": "baseline-silent"},
            ),
            wrapper(1, self.urls[1], results_event("", [])),
        ]
        timestamp_records = [
            wrapper(
                0,
                self.urls[0],
                {"type": "ResponseHead", "status": 200},
            ),
            wrapper(
                0,
                self.urls[0],
                {"type": "Metadata", "request_id": "timestamp-alpha"},
            ),
            wrapper(
                0,
                self.urls[0],
                results_event(
                    "hello world world",
                    [
                        {"word": "hello", "start": 0.1, "end": 0.5},
                        {"word": "world", "start": 0.6, "end": 1.0},
                        {"word": "world", "start": 5.22, "end": 5.62},
                    ],
                ),
            ),
            wrapper(
                1,
                self.urls[1],
                {"type": "ResponseHead", "status": 200},
            ),
            wrapper(
                1,
                self.urls[1],
                {"type": "Metadata", "request_id": "timestamp-silent"},
            ),
            wrapper(1, self.urls[1], results_event("", [])),
        ]
        self.write_ndjson(self.baseline, baseline_records)
        self.write_ndjson(self.timestamp, timestamp_records)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write_ndjson(
        path: pathlib.Path,
        records: list[dict[str, Any]],
    ) -> None:
        path.write_text(
            "".join(f"{json.dumps(record)}\n" for record in records),
            encoding="utf-8",
        )

    def arguments(self, allow_missing: bool = False) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            manifest=self.manifest,
            baseline_progress=self.baseline,
            timestamp_progress=self.timestamp,
            baseline_label="baseline",
            timestamp_label="timestamp",
            stride_seconds=5.2,
            stride_tolerance_seconds=0.15,
            allow_missing_sources=allow_missing,
        )

    def test_analyzes_text_timestamps_retries_and_silence(self) -> None:
        result = ANALYZER.analyze(self.arguments())

        self.assertEqual(result["compared_source_count"], 2)
        self.assertEqual(
            result["text_invariance"]["raw"]["edit_distance"],
            1,
        )
        self.assertEqual(
            result["text_invariance"]["normalized"]["edit_distance"],
            0,
        )
        baseline = result["baseline_word_timestamps"]
        self.assertEqual(
            baseline["timestamp_validity"]["nonpositive_durations"],
            1,
        )
        self.assertEqual(
            baseline["duplicate_indicators"]["adjacent_normalized_repeats"],
            1,
        )
        self.assertEqual(
            baseline["duplicate_indicators"][
                "adjacent_repeats_near_stride_grid"
            ],
            1,
        )
        timing = result["timing_comparison"]
        self.assertEqual(timing["matched_normalized_words"], 3)
        self.assertEqual(timing["matched_words_with_valid_intervals"], 2)
        self.assertAlmostEqual(
            timing["signed_delta_seconds"]["midpoint"]["mean"],
            0.1,
        )
        first = result["sources"][0]
        self.assertEqual(
            first["baseline_artifact"]["attempt_ordinal"],
            1,
        )
        self.assertEqual(first["pairing_method"], "manifest_source_index")

    def test_output_has_no_transcript_or_word_text(self) -> None:
        serialized = json.dumps(ANALYZER.analyze(self.arguments()))

        self.assertNotIn("Hello", serialized)
        self.assertNotIn("hello world", serialized)
        self.assertNotIn('"word":', serialized)
        self.assertFalse(
            ANALYZER.analyze(self.arguments())["transcript_text_included"]
        )

    def test_requires_full_manifest_by_default(self) -> None:
        records = [
            json.loads(line)
            for line in self.timestamp.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["source_index"] == 0
        ]
        self.write_ndjson(self.timestamp, records)

        with self.assertRaisesRegex(ValueError, "full manifest"):
            ANALYZER.analyze(self.arguments())

        partial = ANALYZER.analyze(self.arguments(allow_missing=True))
        self.assertEqual(partial["compared_source_count"], 1)
        self.assertEqual(
            partial["missing_sources"]["timestamp_indexes"],
            [1],
        )

    def test_command_writes_json_without_printing_text(self) -> None:
        output = self.directory / "analysis.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPTS_DIR / "analyze-asr-timestamps.py"),
                "--manifest",
                str(self.manifest),
                "--baseline-progress",
                str(self.baseline),
                "--timestamp-progress",
                str(self.timestamp),
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(output.is_file())
        self.assertNotIn("Hello", completed.stdout)
        self.assertNotIn("hello world", output.read_text(encoding="utf-8"))

    def test_ordered_matching_handles_repeated_words(self) -> None:
        matches = ANALYZER.ordered_token_matches(
            ["the", "the", "result"],
            ["the", "result"],
        )
        self.assertEqual(matches, [(0, 0), (2, 1)])

    def test_counts_order_overlap_duration_and_grid_failures(self) -> None:
        accumulator = ANALYZER.WordMetricAccumulator()
        accumulator.add_source(
            "a b b",
            [
                ANALYZER.WordObservation("a", 1.0, 2.0, "finite", "finite"),
                ANALYZER.WordObservation("b", 0.5, 1.5, "finite", "finite"),
                ANALYZER.WordObservation("b", 5.2, 5.2, "finite", "finite"),
            ],
            4.0,
            5.2,
            0.15,
        )
        metrics = accumulator.to_json()

        self.assertEqual(
            metrics["ordering"]["start_monotonicity_violations"],
            1,
        )
        self.assertEqual(
            metrics["ordering"]["end_monotonicity_violations"],
            1,
        )
        self.assertEqual(metrics["ordering"]["overlapping_intervals"], 1)
        self.assertEqual(
            metrics["timestamp_validity"]["nonpositive_durations"],
            1,
        )
        self.assertEqual(
            metrics["timestamp_validity"]["ends_after_source_duration"],
            1,
        )
        self.assertEqual(
            metrics["duplicate_indicators"][
                "adjacent_repeats_near_stride_grid"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
