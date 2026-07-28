"""Deterministic tests for ASR benchmark metadata."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_asr_benchmark_matrix",
    SCRIPTS_DIR / "run-asr-benchmark-matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class BenchmarkMetadataTests(unittest.TestCase):
    def test_records_effective_default_settings(self) -> None:
        settings = RUNNER.effective_runtime_settings({})

        self.assertEqual(
            settings["ASR_COHERE_TIMESTAMP_BACKEND"],
            "token-frequency",
        )
        self.assertEqual(settings["UPLOAD_RESPONSE_TIMEOUT_MS"], "21600000")
        self.assertEqual(settings["UPLOAD_RESPONSE_WORKER_POLL_MS"], "2")

    def test_records_legacy_ctc_backend(self) -> None:
        settings = RUNNER.effective_runtime_settings(
            {"ASR_CTC_ALIGN_MODEL_DIR": "/models/parakeet"}
        )

        self.assertEqual(
            settings["ASR_COHERE_TIMESTAMP_BACKEND"],
            "parakeet-ctc",
        )

    def test_explicit_timestamp_backend_takes_precedence(self) -> None:
        settings = RUNNER.effective_runtime_settings(
            {
                "ASR_COHERE_TIMESTAMP_BACKEND": "parakeet-ctc-direct",
                "ASR_CTC_ALIGN_MODEL_DIR": "/models/parakeet",
            }
        )

        self.assertEqual(
            settings["ASR_COHERE_TIMESTAMP_BACKEND"],
            "parakeet-ctc-direct",
        )


if __name__ == "__main__":
    unittest.main()
