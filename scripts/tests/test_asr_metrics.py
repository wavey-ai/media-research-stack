"""Deterministic tests for shared ASR text metrics."""

from __future__ import annotations

import itertools
import pathlib
import sys
import unittest


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from asr_metrics import levenshtein_distance, normalize_transcript, normalize_word


def dynamic_distance(reference: str, candidate: str) -> int:
    previous = list(range(len(candidate) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for candidate_index, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    previous[candidate_index - 1]
                    + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


class AsrMetricTests(unittest.TestCase):
    def test_bit_parallel_distance_matches_dynamic_programming(self) -> None:
        strings = [""]
        for length in range(1, 5):
            strings.extend(
                "".join(characters)
                for characters in itertools.product("ab", repeat=length)
            )

        for reference in strings:
            for candidate in strings:
                self.assertEqual(
                    levenshtein_distance(reference, candidate),
                    dynamic_distance(reference, candidate),
                    (reference, candidate),
                )

    def test_normalization_is_stable(self) -> None:
        self.assertEqual(normalize_transcript("  ＨELLO\nWorld  "), "hello world")
        self.assertEqual(normalize_word("Can’t!"), "cant")


if __name__ == "__main__":
    unittest.main()
