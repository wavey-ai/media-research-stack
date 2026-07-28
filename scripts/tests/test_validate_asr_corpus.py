"""Deterministic tests for the ASR corpus validator."""

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
SPEC = importlib.util.spec_from_file_location(
    "validate_asr_corpus",
    SCRIPTS_DIR / "validate-asr-corpus.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def wrapper(index: int, url: str, event: dict[str, Any]) -> dict[str, Any]:
    return {"source_index": index, "source_url": url, "event": event}


def result_event(
    request_id: str,
    transcript: str,
    start: float,
    end: float,
) -> dict[str, Any]:
    words = []
    cursor = start
    tokens = transcript.split()
    for token_index, token in enumerate(tokens):
        token_end = end if token_index == len(tokens) - 1 else cursor + 0.4
        words.append({"word": token, "start": cursor, "end": token_end})
        cursor = token_end
    return {
        "type": "Results",
        "is_final": True,
        "start": start,
        "duration": end,
        "metadata": {"request_id": request_id},
        "channel": {
            "alternatives": [
                {
                    "transcript": transcript,
                    "words": words,
                }
            ]
        },
    }


def completion_event(request_id: str, duration: float) -> dict[str, Any]:
    return {
        "type": "Results",
        "is_final": True,
        "speech_final": True,
        "from_finalize": True,
        "start": 0.0,
        "duration": duration,
        "metadata": {"request_id": request_id},
        "channel": {
            "alternatives": [
                {
                    "transcript": "",
                    "words": [],
                }
            ]
        },
    }


class CorpusValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self.temporary.name)
        self.manifest = self.directory / "manifest.json"
        self.report = self.directory / "report.jsonl"
        self.progress = self.directory / "progress.ndjson"
        self.transcripts = self.directory / "transcripts"
        self.transcripts.mkdir()
        self.urls = [
            "https://example.test/watch?v=alpha",
            "https://example.test/watch?v=beta",
        ]
        self.manifest.write_text(
            json.dumps(
                {
                    "videos": [
                        {
                            "source_index": 0,
                            "url": self.urls[0],
                            "duration_seconds": 10.0,
                        },
                        {
                            "source_index": 1,
                            "url": self.urls[1],
                            "duration_seconds": 20.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.write_valid_artifacts()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arguments(self, **overrides: Any) -> types.SimpleNamespace:
        values = {
            "manifest": self.manifest,
            "report": self.report,
            "progress": self.progress,
            "transcripts": self.transcripts,
            "output": self.directory / "validation.json",
            "trailing_silence_seconds": 1.0,
            "max_service_rtfx": 256.0,
            "early_completion_min_seconds": 300.0,
            "ring_stride_seconds": 4096.0,
            "ring_stride_tolerance_seconds": 10.0,
            "duration_tolerance_seconds": 1.0,
            "timestamp_regression_tolerance_seconds": 0.05,
            "allow_progress_retries": False,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    @staticmethod
    def write_jsonl(
        path: pathlib.Path,
        records: list[dict[str, Any]],
    ) -> None:
        path.write_text(
            "".join(f"{json.dumps(record)}\n" for record in records),
            encoding="utf-8",
        )

    def write_valid_artifacts(self) -> None:
        transcripts = ["first complete", "second complete"]
        report_records = []
        progress_records = []
        durations = [10.0, 20.0]
        for index, (url, transcript, duration) in enumerate(
            zip(self.urls, transcripts, durations)
        ):
            name = VALIDATOR.transcript_name(index, url)
            (self.transcripts / name).write_text(
                f"{transcript}\n",
                encoding="utf-8",
            )
            report_records.append(
                {
                    "status": "ok",
                    "index": index,
                    "source_url": url,
                    "audio_seconds": duration,
                    "asr_wall_seconds": duration / 10.0,
                    "transcript_words": len(transcript.split()),
                    "transcript_chars": len(transcript),
                    "transcript_path": f"/remote/transcripts/{name}",
                }
            )
            request_id = f"request-{index}"
            progress_records.extend(
                [
                    wrapper(
                        index,
                        url,
                        {"type": "ResponseHead", "status": 200},
                    ),
                    wrapper(
                        index,
                        url,
                        {"type": "Metadata", "request_id": request_id},
                    ),
                    wrapper(
                        index,
                        url,
                        result_event(
                            request_id,
                            transcript,
                            0.0,
                            duration - 0.25,
                        ),
                    ),
                    wrapper(
                        index,
                        url,
                        completion_event(request_id, duration),
                    ),
                ]
            )
        self.write_jsonl(self.report, report_records)
        self.write_jsonl(self.progress, progress_records)

    def test_accepts_complete_consistent_artifacts(self) -> None:
        result = VALIDATOR.validate(self.arguments())

        self.assertTrue(result["valid"])
        self.assertEqual(result["summary"]["invalid_source_count"], 0)
        self.assertFalse(result["transcript_text_included"])

    def test_accepts_redacted_progress_counts(self) -> None:
        records = [
            json.loads(line)
            for line in self.progress.read_text(encoding="utf-8").splitlines()
        ]
        for record in records:
            alternative = VALIDATOR.first_alternative(record["event"])
            if alternative is None:
                continue
            transcript = alternative.pop("transcript")
            words = alternative.pop("words")
            alternative["transcript_chars"] = len(transcript.strip())
            alternative["transcript_words"] = len(words)
        self.write_jsonl(self.progress, records)

        result = VALIDATOR.validate(self.arguments())

        self.assertTrue(result["valid"])
        self.assertEqual(result["summary"]["invalid_source_count"], 0)

    def test_requires_processed_audio_completion_marker(self) -> None:
        records = [
            json.loads(line)
            for line in self.progress.read_text(encoding="utf-8").splitlines()
            if not (
                json.loads(line)["source_index"] == 0
                and json.loads(line)["event"].get("from_finalize") is True
            )
        ]
        self.write_jsonl(self.progress, records)

        result = VALIDATOR.validate(self.arguments())

        self.assertIn(
            "missing_processed_audio_marker",
            result["sources"][0]["issues"],
        )
        self.assertEqual(
            result["sources"][0]["coverage_basis"],
            "speech_timestamps",
        )

    def test_detects_coverage_early_completion_and_ring_stride(self) -> None:
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["videos"][0]["duration_seconds"] = 4106.0
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        records = [
            json.loads(line)
            for line in self.report.read_text(encoding="utf-8").splitlines()
        ]
        records[0]["audio_seconds"] = 4106.0
        records[0]["asr_wall_seconds"] = 1.0
        self.write_jsonl(self.report, records)

        result = VALIDATOR.validate(
            self.arguments(early_completion_min_seconds=300.0)
        )
        issues = result["sources"][0]["issues"]

        self.assertIn("source_duration_not_covered", issues)
        self.assertIn("implausibly_early_response_completion", issues)
        self.assertIn("ring_stride_shortfall", issues)
        self.assertEqual(
            result["sources"][0]["ring_stride_match"]["multiple"],
            1,
        )

    def test_detects_duplicate_attempts_and_request_ids(self) -> None:
        records = [
            json.loads(line)
            for line in self.progress.read_text(encoding="utf-8").splitlines()
        ]
        first_source_end = next(
            index
            for index, record in enumerate(records)
            if record["source_index"] == 1
        )
        retry = [
            wrapper(0, self.urls[0], {"type": "ResponseHead", "status": 200}),
            wrapper(
                0,
                self.urls[0],
                {"type": "Metadata", "request_id": "request-1"},
            ),
            wrapper(
                0,
                self.urls[0],
                result_event(
                    "request-1",
                    "first complete",
                    0.0,
                    9.75,
                ),
            ),
        ]
        records[first_source_end:first_source_end] = retry
        self.write_jsonl(self.progress, records)

        result = VALIDATOR.validate(self.arguments())

        self.assertIn(
            "duplicate_progress_attempts",
            result["sources"][0]["issues"],
        )
        self.assertIn("duplicate_request_id", result["sources"][0]["issues"])
        self.assertIn("duplicate_request_id", result["sources"][1]["issues"])
        self.assertEqual(
            result["sources"][0]["progress_attempts"][1]["request_ids"],
            ["request-1"],
        )

    def test_detects_missing_and_duplicate_source_records(self) -> None:
        report_records = [
            json.loads(line)
            for line in self.report.read_text(encoding="utf-8").splitlines()
        ]
        report_records.insert(1, dict(report_records[0]))
        self.write_jsonl(self.report, report_records)

        progress_records = [
            json.loads(line)
            for line in self.progress.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["source_index"] == 0
        ]
        self.write_jsonl(self.progress, progress_records)

        result = VALIDATOR.validate(self.arguments())

        self.assertIn(
            "duplicate_successful_reports",
            result["sources"][0]["issues"],
        )
        self.assertIn(
            "missing_progress_attempt",
            result["sources"][1]["issues"],
        )

    def test_can_accept_retry_history_after_validating_last_attempt(self) -> None:
        records = [
            wrapper(0, self.urls[0], {"type": "ResponseHead", "status": 503}),
            *[
                json.loads(line)
                for line in self.progress.read_text(
                    encoding="utf-8"
                ).splitlines()
            ],
        ]
        self.write_jsonl(self.progress, records)

        strict = VALIDATOR.validate(self.arguments())
        relaxed = VALIDATOR.validate(
            self.arguments(allow_progress_retries=True)
        )

        self.assertIn(
            "duplicate_progress_attempts",
            strict["sources"][0]["issues"],
        )
        self.assertTrue(relaxed["valid"])

    def test_detects_transcript_and_report_path_issues(self) -> None:
        records = [
            json.loads(line)
            for line in self.report.read_text(encoding="utf-8").splitlines()
        ]
        records[0]["transcript_words"] = 99
        records[1]["transcript_path"] = "/remote/transcripts/wrong.txt"
        self.write_jsonl(self.report, records)
        (self.transcripts / "orphan.txt.part").write_text(
            "partial output",
            encoding="utf-8",
        )

        result = VALIDATOR.validate(self.arguments())

        self.assertIn(
            "transcript_file_word_count_mismatch",
            result["sources"][0]["issues"],
        )
        self.assertIn(
            "report_transcript_path_mismatch",
            result["sources"][1]["issues"],
        )
        self.assertIn("partial_transcript_files", result["global_issues"])

    def test_command_fails_without_printing_transcript_text(self) -> None:
        name = VALIDATOR.transcript_name(0, self.urls[0])
        (self.transcripts / name).unlink()
        output = self.directory / "command-output.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPTS_DIR / "validate-asr-corpus.py"),
                "--manifest",
                str(self.manifest),
                "--report",
                str(self.report),
                "--progress",
                str(self.progress),
                "--transcripts",
                str(self.transcripts),
                "--trailing-silence-seconds",
                "1",
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertNotIn("first complete", completed.stdout)
        self.assertNotIn("first complete", completed.stderr)
        serialized = output.read_text(encoding="utf-8")
        self.assertNotIn("first complete", serialized)
        self.assertIn("missing_transcript_file", serialized)


if __name__ == "__main__":
    unittest.main()
