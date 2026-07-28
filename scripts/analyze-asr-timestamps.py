#!/usr/bin/env python3
"""Compare text and word timestamps in two ASR progress artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

from asr_metrics import (
    distance_ratio,
    levenshtein_distance,
    normalize_transcript,
    normalize_word,
)


PERCENTILES = (
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
    ("p99", 0.99),
)
MIDPOINT_THRESHOLDS_SECONDS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a baseline ASR progress artifact with a timestamp-side-model "
            "artifact. The output does not contain transcript or word text."
        )
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-progress", type=pathlib.Path, required=True)
    parser.add_argument("--timestamp-progress", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--timestamp-label", default="timestamp-side-model")
    parser.add_argument("--stride-seconds", type=float, default=5.2)
    parser.add_argument("--stride-tolerance-seconds", type=float, default=0.15)
    parser.add_argument(
        "--allow-missing-sources",
        action="store_true",
        help="Analyze the source intersection instead of requiring the full manifest.",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def video_id(url: str) -> str:
    parsed = urlparse(url)
    candidate = parse_qs(parsed.query).get("v", [""])[0]
    if not candidate:
        candidate = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    normalized = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "-_")
        else "_"
        for character in candidate
    )
    return normalized or "source"


def load_videos(path: pathlib.Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = value.get("videos") if isinstance(value, dict) else value
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest JSON must contain a nonempty videos array")

    videos = []
    for entry in entries:
        if isinstance(entry, str):
            videos.append({"url": entry})
        elif isinstance(entry, dict) and isinstance(entry.get("url"), str):
            videos.append(entry)
        else:
            raise ValueError("each manifest entry must contain a URL")
    return videos


@dataclasses.dataclass
class Attempt:
    source_index: int
    source_url: str
    ordinal: int
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    request_id: str | None = None
    response_status: int | None = None


@dataclasses.dataclass(frozen=True)
class WordObservation:
    normalized: str
    start: float | None
    end: float | None
    start_state: str
    end_state: str

    @property
    def valid_interval(self) -> bool:
        return (
            self.start is not None
            and self.end is not None
            and self.start >= 0.0
            and self.end > self.start
        )


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    attempt: Attempt
    transcript: str
    words: list[WordObservation]
    result_events: int
    final_result_events: int


def event_type(event: dict[str, Any]) -> str | None:
    value = event.get("type")
    return value if isinstance(value, str) else None


def load_attempts(
    path: pathlib.Path,
    videos: list[dict[str, Any]],
) -> dict[int, list[Attempt]]:
    attempts: dict[int, list[Attempt]] = defaultdict(list)
    current: dict[int, Attempt] = {}
    with path.open("r", encoding="utf-8") as progress:
        for line_number, line in enumerate(progress, start=1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} has invalid JSON at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(wrapper, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            source_index = wrapper.get("source_index")
            source_url = wrapper.get("source_url")
            event = wrapper.get("event")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or not 0 <= source_index < len(videos)
            ):
                raise ValueError(
                    f"{path} line {line_number} has an invalid source_index"
                )
            if not isinstance(source_url, str):
                raise ValueError(f"{path} line {line_number} has no source URL")
            if source_url != videos[source_index]["url"]:
                raise ValueError(
                    f"{path} line {line_number} does not match the manifest URL"
                )
            if not isinstance(event, dict):
                raise ValueError(f"{path} line {line_number} has no event object")

            kind = event_type(event)
            request_id = event.get("request_id")
            request_id = request_id if isinstance(request_id, str) else None
            active = current.get(source_index)
            starts_attempt = (
                active is None
                or (kind == "ResponseHead" and bool(active.events))
                or (
                    request_id is not None
                    and active.request_id is not None
                    and request_id != active.request_id
                )
            )
            if starts_attempt:
                active = Attempt(
                    source_index=source_index,
                    source_url=source_url,
                    ordinal=len(attempts[source_index]),
                )
                attempts[source_index].append(active)
                current[source_index] = active

            active.events.append(event)
            if request_id is not None:
                active.request_id = request_id
            if kind == "ResponseHead":
                status = event.get("status")
                if (
                    not isinstance(status, bool)
                    and isinstance(status, int)
                ):
                    active.response_status = status
    return dict(attempts)


def alternative_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    channel = event.get("channel")
    if isinstance(channel, dict):
        alternatives = channel.get("alternatives")
        if (
            isinstance(alternatives, list)
            and alternatives
            and isinstance(alternatives[0], dict)
        ):
            return alternatives[0]

    results = event.get("results")
    if not isinstance(results, dict):
        return None
    channels = results.get("channels")
    if (
        not isinstance(channels, list)
        or not channels
        or not isinstance(channels[0], dict)
    ):
        return None
    alternatives = channels[0].get("alternatives")
    if (
        isinstance(alternatives, list)
        and alternatives
        and isinstance(alternatives[0], dict)
    ):
        return alternatives[0]
    return None


def parse_time(word: dict[str, Any], field: str) -> tuple[float | None, str]:
    if field not in word:
        return None, "missing"
    value = word[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "invalid"
    number = float(value)
    if not math.isfinite(number):
        return None, "invalid"
    return number, "finite"


def parse_word(value: Any) -> WordObservation:
    if not isinstance(value, dict):
        return WordObservation("", None, None, "invalid", "invalid")
    text = value.get("word")
    normalized = normalize_word(text) if isinstance(text, str) else ""
    start, start_state = parse_time(value, "start")
    end, end_state = parse_time(value, "end")
    return WordObservation(normalized, start, end, start_state, end_state)


def result_events(attempt: Attempt) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    events = []
    for event in attempt.events:
        if event_type(event) != "Results":
            continue
        alternative = alternative_from_event(event)
        if alternative is not None:
            events.append((event, alternative))
    return events


def extract_attempt(attempt: Attempt) -> AttemptResult:
    results = result_events(attempt)
    if not results:
        raise ValueError(
            f"source {attempt.source_index} attempt {attempt.ordinal} has no Results event"
        )
    finals = [
        pair for pair in results if pair[0].get("is_final") is True
    ]
    selected = finals if finals else [results[-1]]

    segments = []
    words = []
    for _, alternative in selected:
        if "transcript" not in alternative:
            raise ValueError(
                "progress artifacts must retain transcripts; "
                f"source {attempt.source_index} has redacted Results data"
            )
        transcript = alternative.get("transcript")
        if not isinstance(transcript, str):
            raise ValueError(
                f"source {attempt.source_index} has an invalid transcript field"
            )
        transcript = transcript.strip()
        if transcript:
            segments.append(transcript)
        raw_words = alternative.get("words", [])
        if not isinstance(raw_words, list):
            raise ValueError(
                f"source {attempt.source_index} has an invalid words field"
            )
        words.extend(parse_word(word) for word in raw_words)

    return AttemptResult(
        attempt=attempt,
        transcript=" ".join(segments),
        words=words,
        result_events=len(results),
        final_result_events=len(finals),
    )


def select_attempt(attempts: Iterable[Attempt]) -> AttemptResult | None:
    candidates = [
        attempt
        for attempt in attempts
        if result_events(attempt)
        and (
            attempt.response_status is None
            or 200 <= attempt.response_status < 300
        )
    ]
    if not candidates:
        return None
    return extract_attempt(candidates[-1])


def numeric_distribution(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            **{name: None for name, _ in PERCENTILES},
            "maximum": None,
        }
    result: dict[str, Any] = {
        "count": len(ordered),
        "minimum": ordered[0],
        "mean": statistics.fmean(ordered),
    }
    for name, percentile in PERCENTILES:
        index = max(0, math.ceil(len(ordered) * percentile) - 1)
        result[name] = ordered[index]
    result["maximum"] = ordered[-1]
    return result


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def filtered_tokens(values: Iterable[str]) -> tuple[list[str], list[int]]:
    tokens = []
    indexes = []
    for index, value in enumerate(values):
        if value:
            tokens.append(value)
            indexes.append(index)
    return tokens, indexes


def ordered_token_matches(
    reference: Iterable[str],
    candidate: Iterable[str],
) -> list[tuple[int, int]]:
    reference_tokens, reference_indexes = filtered_tokens(reference)
    candidate_tokens, candidate_indexes = filtered_tokens(candidate)
    if reference_tokens == candidate_tokens:
        return list(zip(reference_indexes, candidate_indexes))

    prefix_length = 0
    shared_length = min(len(reference_tokens), len(candidate_tokens))
    while (
        prefix_length < shared_length
        and reference_tokens[prefix_length] == candidate_tokens[prefix_length]
    ):
        prefix_length += 1

    suffix_length = 0
    while (
        suffix_length < shared_length - prefix_length
        and reference_tokens[-1 - suffix_length]
        == candidate_tokens[-1 - suffix_length]
    ):
        suffix_length += 1

    reference_end = len(reference_tokens) - suffix_length
    candidate_end = len(candidate_tokens) - suffix_length
    reference_middle = reference_tokens[prefix_length:reference_end]
    candidate_middle = candidate_tokens[prefix_length:candidate_end]
    matcher = difflib.SequenceMatcher(
        None,
        reference_middle,
        candidate_middle,
        autojunk=max(len(reference_middle), len(candidate_middle)) >= 200,
    )
    matches = [
        (reference_indexes[index], candidate_indexes[index])
        for index in range(prefix_length)
    ]
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            matches.append(
                (
                    reference_indexes[prefix_length + block.a + offset],
                    candidate_indexes[prefix_length + block.b + offset],
                )
            )
    if suffix_length:
        matches.extend(
            (
                reference_indexes[reference_end + offset],
                candidate_indexes[candidate_end + offset],
            )
            for offset in range(suffix_length)
        )
    return matches


def stride_residual(timestamp: float, stride_seconds: float) -> float:
    multiple = round(timestamp / stride_seconds)
    return abs(timestamp - multiple * stride_seconds)


@dataclasses.dataclass
class WordMetricAccumulator:
    counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    samples: dict[str, list[float]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )

    def add_source(
        self,
        transcript: str,
        words: list[WordObservation],
        duration_seconds: float | None,
        stride_seconds: float,
        stride_tolerance_seconds: float,
    ) -> None:
        self.counts["sources"] += 1
        transcript_tokens = [
            normalize_word(token) for token in transcript.split()
        ]
        transcript_tokens = [token for token in transcript_tokens if token]
        self.counts["transcript_words"] += len(transcript_tokens)
        self.counts["emitted_words"] += len(words)
        self.counts["adjacent_word_pairs"] += max(len(words) - 1, 0)

        valid_word_indexes: set[int] = set()
        previous_start: float | None = None
        previous_end: float | None = None
        previous_valid_interval: WordObservation | None = None
        for index, word in enumerate(words):
            if word.normalized:
                self.counts["words_with_normalized_text"] += 1
            self.counts[f"start_{word.start_state}"] += 1
            self.counts[f"end_{word.end_state}"] += 1
            if word.start is not None:
                self.counts["words_with_finite_start"] += 1
                if word.start < 0.0:
                    self.counts["negative_starts"] += 1
                if previous_start is not None:
                    self.counts["evaluated_start_order_pairs"] += 1
                    if word.start < previous_start:
                        self.counts["start_monotonicity_violations"] += 1
                previous_start = word.start
            if word.end is not None:
                self.counts["words_with_finite_end"] += 1
                if word.end < 0.0:
                    self.counts["negative_ends"] += 1
                if previous_end is not None:
                    self.counts["evaluated_end_order_pairs"] += 1
                    if word.end < previous_end:
                        self.counts["end_monotonicity_violations"] += 1
                previous_end = word.end

            if word.start is not None and word.end is not None:
                self.counts["words_with_finite_start_and_end"] += 1
                if word.end <= word.start:
                    self.counts["nonpositive_durations"] += 1
                else:
                    self.samples["positive_duration_seconds"].append(
                        word.end - word.start
                    )
            if duration_seconds is not None:
                if word.start is not None and word.start > duration_seconds:
                    self.counts["starts_after_source_duration"] += 1
                if word.end is not None and word.end > duration_seconds:
                    self.counts["ends_after_source_duration"] += 1

            if word.valid_interval:
                self.counts["valid_intervals"] += 1
                valid_word_indexes.add(index)
                if previous_valid_interval is not None:
                    previous_interval_end = previous_valid_interval.end
                    assert previous_interval_end is not None
                    self.counts["evaluated_interval_pairs"] += 1
                    if word.start < previous_interval_end:
                        self.counts["overlapping_intervals"] += 1
                        self.samples["overlap_seconds"].append(
                            previous_interval_end - word.start
                        )
                    else:
                        self.samples["gap_seconds"].append(
                            word.start - previous_interval_end
                        )
                previous_valid_interval = word

            if index == 0:
                continue
            previous = words[index - 1]
            if (
                not word.normalized
                or word.normalized != previous.normalized
            ):
                continue
            self.counts["adjacent_normalized_repeats"] += 1
            if word.start is not None and word.start >= 0.0:
                residual = stride_residual(word.start, stride_seconds)
                self.samples["adjacent_repeat_stride_grid_residual_seconds"].append(
                    residual
                )
                if residual <= stride_tolerance_seconds:
                    self.counts["adjacent_repeats_near_stride_grid"] += 1
            if (
                word.start is not None
                and previous.start is not None
                and word.start > previous.start
            ):
                separation = word.start - previous.start
                multiple = round(separation / stride_seconds)
                if multiple >= 1:
                    residual = abs(separation - multiple * stride_seconds)
                    self.samples[
                        "adjacent_repeat_stride_separation_residual_seconds"
                    ].append(residual)
                    if residual <= stride_tolerance_seconds:
                        self.counts["adjacent_repeats_stride_separated"] += 1

        matches = ordered_token_matches(
            transcript_tokens,
            (word.normalized for word in words),
        )
        self.counts["transcript_words_matched_to_emitted_words"] += len(matches)
        self.counts["valid_timed_transcript_words"] += sum(
            candidate_index in valid_word_indexes
            for _, candidate_index in matches
        )

    def merge(self, other: WordMetricAccumulator) -> None:
        for key, value in other.counts.items():
            self.counts[key] += value
        for key, values in other.samples.items():
            self.samples[key].extend(values)

    def to_json(self) -> dict[str, Any]:
        counts = self.counts
        emitted = counts["emitted_words"]
        transcript_words = counts["transcript_words"]
        interval_pairs = counts["evaluated_interval_pairs"]
        adjacent_repeats = counts["adjacent_normalized_repeats"]
        return {
            "sources": counts["sources"],
            "transcript_words": transcript_words,
            "emitted_words": emitted,
            "words_with_normalized_text": counts["words_with_normalized_text"],
            "timestamp_coverage": {
                "words_with_finite_start": counts["words_with_finite_start"],
                "finite_start_rate": safe_ratio(
                    counts["words_with_finite_start"], emitted
                ),
                "words_with_finite_end": counts["words_with_finite_end"],
                "finite_end_rate": safe_ratio(
                    counts["words_with_finite_end"], emitted
                ),
                "words_with_finite_start_and_end": counts[
                    "words_with_finite_start_and_end"
                ],
                "finite_start_and_end_rate": safe_ratio(
                    counts["words_with_finite_start_and_end"], emitted
                ),
                "valid_intervals": counts["valid_intervals"],
                "valid_interval_rate": safe_ratio(
                    counts["valid_intervals"], emitted
                ),
                "transcript_words_matched_to_emitted_words": counts[
                    "transcript_words_matched_to_emitted_words"
                ],
                "transcript_word_match_rate": safe_ratio(
                    counts["transcript_words_matched_to_emitted_words"],
                    transcript_words,
                ),
                "valid_timed_transcript_words": counts[
                    "valid_timed_transcript_words"
                ],
                "valid_timed_transcript_word_rate": safe_ratio(
                    counts["valid_timed_transcript_words"],
                    transcript_words,
                ),
            },
            "timestamp_validity": {
                "missing_starts": counts["start_missing"],
                "invalid_starts": counts["start_invalid"],
                "missing_ends": counts["end_missing"],
                "invalid_ends": counts["end_invalid"],
                "negative_starts": counts["negative_starts"],
                "negative_ends": counts["negative_ends"],
                "nonpositive_durations": counts["nonpositive_durations"],
                "starts_after_source_duration": counts[
                    "starts_after_source_duration"
                ],
                "ends_after_source_duration": counts[
                    "ends_after_source_duration"
                ],
            },
            "ordering": {
                "evaluated_start_pairs": counts[
                    "evaluated_start_order_pairs"
                ],
                "start_monotonicity_violations": counts[
                    "start_monotonicity_violations"
                ],
                "start_monotonicity_violation_rate": safe_ratio(
                    counts["start_monotonicity_violations"],
                    counts["evaluated_start_order_pairs"],
                ),
                "evaluated_end_pairs": counts["evaluated_end_order_pairs"],
                "end_monotonicity_violations": counts[
                    "end_monotonicity_violations"
                ],
                "end_monotonicity_violation_rate": safe_ratio(
                    counts["end_monotonicity_violations"],
                    counts["evaluated_end_order_pairs"],
                ),
                "evaluated_interval_pairs": interval_pairs,
                "overlapping_intervals": counts["overlapping_intervals"],
                "overlap_rate": safe_ratio(
                    counts["overlapping_intervals"], interval_pairs
                ),
            },
            "interval_distributions_seconds": {
                "positive_duration": numeric_distribution(
                    self.samples["positive_duration_seconds"]
                ),
                "gap": numeric_distribution(self.samples["gap_seconds"]),
                "overlap": numeric_distribution(
                    self.samples["overlap_seconds"]
                ),
            },
            "duplicate_indicators": {
                "adjacent_normalized_repeats": adjacent_repeats,
                "adjacent_normalized_repeat_rate": safe_ratio(
                    adjacent_repeats, counts["adjacent_word_pairs"]
                ),
                "adjacent_repeats_near_stride_grid": counts[
                    "adjacent_repeats_near_stride_grid"
                ],
                "near_stride_grid_rate": safe_ratio(
                    counts["adjacent_repeats_near_stride_grid"],
                    adjacent_repeats,
                ),
                "adjacent_repeats_stride_separated": counts[
                    "adjacent_repeats_stride_separated"
                ],
                "stride_separated_rate": safe_ratio(
                    counts["adjacent_repeats_stride_separated"],
                    adjacent_repeats,
                ),
                "stride_grid_residual_seconds": numeric_distribution(
                    self.samples[
                        "adjacent_repeat_stride_grid_residual_seconds"
                    ]
                ),
                "stride_separation_residual_seconds": numeric_distribution(
                    self.samples[
                        "adjacent_repeat_stride_separation_residual_seconds"
                    ]
                ),
            },
        }


@dataclasses.dataclass
class TimingDeltaAccumulator:
    counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    samples: dict[str, list[float]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )

    def add_source(
        self,
        baseline: list[WordObservation],
        timestamp: list[WordObservation],
    ) -> None:
        self.counts["sources"] += 1
        self.counts["baseline_words"] += len(baseline)
        self.counts["timestamp_words"] += len(timestamp)
        matches = ordered_token_matches(
            (word.normalized for word in baseline),
            (word.normalized for word in timestamp),
        )
        self.counts["matched_words"] += len(matches)
        for baseline_index, timestamp_index in matches:
            baseline_word = baseline[baseline_index]
            timestamp_word = timestamp[timestamp_index]
            if not baseline_word.valid_interval or not timestamp_word.valid_interval:
                continue
            self.counts["matched_words_with_valid_intervals"] += 1
            assert baseline_word.start is not None
            assert baseline_word.end is not None
            assert timestamp_word.start is not None
            assert timestamp_word.end is not None
            start_delta = timestamp_word.start - baseline_word.start
            end_delta = timestamp_word.end - baseline_word.end
            midpoint_delta = (
                (timestamp_word.start + timestamp_word.end)
                - (baseline_word.start + baseline_word.end)
            ) / 2.0
            duration_delta = (
                timestamp_word.end
                - timestamp_word.start
                - (baseline_word.end - baseline_word.start)
            )
            for key, value in (
                ("start_delta_seconds", start_delta),
                ("end_delta_seconds", end_delta),
                ("midpoint_delta_seconds", midpoint_delta),
                ("duration_delta_seconds", duration_delta),
            ):
                self.samples[key].append(value)
                self.samples[f"absolute_{key}"].append(abs(value))
            for threshold in MIDPOINT_THRESHOLDS_SECONDS:
                if abs(midpoint_delta) <= threshold:
                    self.counts[
                        f"midpoint_within_{threshold:.3f}_seconds"
                    ] += 1

    def merge(self, other: TimingDeltaAccumulator) -> None:
        for key, value in other.counts.items():
            self.counts[key] += value
        for key, values in other.samples.items():
            self.samples[key].extend(values)

    def to_json(self) -> dict[str, Any]:
        counts = self.counts
        matched = counts["matched_words"]
        valid = counts["matched_words_with_valid_intervals"]
        within = {}
        for threshold in MIDPOINT_THRESHOLDS_SECONDS:
            key = f"midpoint_within_{threshold:.3f}_seconds"
            within[f"{threshold:.3f}"] = {
                "words": counts[key],
                "rate": safe_ratio(counts[key], valid),
            }
        return {
            "sources": counts["sources"],
            "baseline_words": counts["baseline_words"],
            "timestamp_words": counts["timestamp_words"],
            "matched_normalized_words": matched,
            "baseline_word_match_rate": safe_ratio(
                matched, counts["baseline_words"]
            ),
            "timestamp_word_match_rate": safe_ratio(
                matched, counts["timestamp_words"]
            ),
            "matched_words_with_valid_intervals": valid,
            "valid_interval_pair_rate": safe_ratio(valid, matched),
            "delta_direction": "timestamp-side-model minus baseline",
            "signed_delta_seconds": {
                "start": numeric_distribution(
                    self.samples["start_delta_seconds"]
                ),
                "end": numeric_distribution(
                    self.samples["end_delta_seconds"]
                ),
                "midpoint": numeric_distribution(
                    self.samples["midpoint_delta_seconds"]
                ),
                "duration": numeric_distribution(
                    self.samples["duration_delta_seconds"]
                ),
            },
            "absolute_delta_seconds": {
                "start": numeric_distribution(
                    self.samples["absolute_start_delta_seconds"]
                ),
                "end": numeric_distribution(
                    self.samples["absolute_end_delta_seconds"]
                ),
                "midpoint": numeric_distribution(
                    self.samples["absolute_midpoint_delta_seconds"]
                ),
                "duration": numeric_distribution(
                    self.samples["absolute_duration_delta_seconds"]
                ),
            },
            "absolute_midpoint_thresholds_seconds": within,
        }


def source_text_metrics(
    baseline: str,
    timestamp: str,
) -> dict[str, dict[str, int | float]]:
    normalized_baseline = normalize_transcript(baseline)
    normalized_timestamp = normalize_transcript(timestamp)
    raw_distance = levenshtein_distance(baseline, timestamp)
    normalized_distance = levenshtein_distance(
        normalized_baseline,
        normalized_timestamp,
    )
    return {
        "raw": {
            "baseline_characters": len(baseline),
            "timestamp_characters": len(timestamp),
            "edit_distance": raw_distance,
            "distance_ratio": distance_ratio(
                raw_distance, len(baseline), len(timestamp)
            ),
        },
        "normalized": {
            "baseline_characters": len(normalized_baseline),
            "timestamp_characters": len(normalized_timestamp),
            "edit_distance": normalized_distance,
            "distance_ratio": distance_ratio(
                normalized_distance,
                len(normalized_baseline),
                len(normalized_timestamp),
            ),
        },
    }


def aggregate_text_metrics(
    sources: list[dict[str, Any]],
    kind: str,
) -> dict[str, Any]:
    metrics = [source["text_invariance"][kind] for source in sources]
    distances = [int(metric["edit_distance"]) for metric in metrics]
    baseline_lengths = [
        int(metric["baseline_characters"]) for metric in metrics
    ]
    timestamp_lengths = [
        int(metric["timestamp_characters"]) for metric in metrics
    ]
    denominator = sum(
        max(baseline_length, timestamp_length)
        for baseline_length, timestamp_length in zip(
            baseline_lengths,
            timestamp_lengths,
        )
    )
    return {
        "baseline_characters": sum(baseline_lengths),
        "timestamp_characters": sum(timestamp_lengths),
        "edit_distance": sum(distances),
        "distance_ratio": sum(distances) / max(denominator, 1),
        "exact_match_sources": sum(distance == 0 for distance in distances),
        "source_distance_ratio_distribution": numeric_distribution(
            float(metric["distance_ratio"]) for metric in metrics
        ),
    }


def attempt_metadata(result: AttemptResult) -> dict[str, Any]:
    attempt = result.attempt
    return {
        "request_id": attempt.request_id,
        "attempt_ordinal": attempt.ordinal,
        "response_status": attempt.response_status,
        "event_count": len(attempt.events),
        "result_event_count": result.result_events,
        "final_result_event_count": result.final_result_events,
    }


def source_duration(video: dict[str, Any]) -> float | None:
    value = video.get("duration_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.stride_seconds <= 0.0:
        raise ValueError("--stride-seconds must be greater than zero")
    if args.stride_tolerance_seconds < 0.0:
        raise ValueError("--stride-tolerance-seconds cannot be negative")

    videos = load_videos(args.manifest)
    baseline_attempts = load_attempts(args.baseline_progress, videos)
    timestamp_attempts = load_attempts(args.timestamp_progress, videos)
    baseline_global = WordMetricAccumulator()
    timestamp_global = WordMetricAccumulator()
    timing_global = TimingDeltaAccumulator()
    sources = []
    missing_baseline = []
    missing_timestamp = []

    for index, video in enumerate(videos):
        baseline = select_attempt(baseline_attempts.get(index, []))
        timestamp = select_attempt(timestamp_attempts.get(index, []))
        if baseline is None:
            missing_baseline.append(index)
        if timestamp is None:
            missing_timestamp.append(index)
        if baseline is None or timestamp is None:
            continue

        duration = source_duration(video)
        baseline_metrics = WordMetricAccumulator()
        baseline_metrics.add_source(
            baseline.transcript,
            baseline.words,
            duration,
            args.stride_seconds,
            args.stride_tolerance_seconds,
        )
        timestamp_metrics = WordMetricAccumulator()
        timestamp_metrics.add_source(
            timestamp.transcript,
            timestamp.words,
            duration,
            args.stride_seconds,
            args.stride_tolerance_seconds,
        )
        timing_metrics = TimingDeltaAccumulator()
        timing_metrics.add_source(baseline.words, timestamp.words)
        baseline_global.merge(baseline_metrics)
        timestamp_global.merge(timestamp_metrics)
        timing_global.merge(timing_metrics)

        same_request = (
            baseline.attempt.request_id is not None
            and baseline.attempt.request_id == timestamp.attempt.request_id
        )
        sources.append(
            {
                "index": index,
                "id": video_id(video["url"]),
                "source_url": video["url"],
                "pairing_method": (
                    "request_id" if same_request else "manifest_source_index"
                ),
                "baseline_artifact": attempt_metadata(baseline),
                "timestamp_artifact": attempt_metadata(timestamp),
                "text_invariance": source_text_metrics(
                    baseline.transcript,
                    timestamp.transcript,
                ),
                "baseline_word_timestamps": baseline_metrics.to_json(),
                "timestamp_word_timestamps": timestamp_metrics.to_json(),
                "timing_comparison": timing_metrics.to_json(),
            }
        )

    if (
        (missing_baseline or missing_timestamp)
        and not args.allow_missing_sources
    ):
        raise ValueError(
            "progress artifacts do not cover the full manifest: "
            f"{len(missing_baseline)} baseline source(s) and "
            f"{len(missing_timestamp)} timestamp source(s) are missing"
        )
    if not sources:
        raise ValueError("the progress artifacts have no comparable sources")

    return {
        "schema_version": 1,
        "metric_scope": (
            "character text invariance and word timestamp artifact diagnostics"
        ),
        "transcript_text_included": False,
        "word_text_included": False,
        "baseline_label": args.baseline_label,
        "timestamp_label": args.timestamp_label,
        "manifest_source_count": len(videos),
        "compared_source_count": len(sources),
        "missing_sources": {
            "baseline_indexes": missing_baseline,
            "timestamp_indexes": missing_timestamp,
        },
        "attempt_selection": (
            "latest successful attempt with a Results event for each manifest source"
        ),
        "pairing": (
            "matching request_id when equal; otherwise manifest source_index"
        ),
        "character_unit": "Unicode code point",
        "transcript_normalization": (
            "NFKC, Unicode case folding, and whitespace collapse; punctuation kept"
        ),
        "word_normalization": (
            "NFKC, Unicode case folding, and Unicode alphanumeric characters only"
        ),
        "word_alignment": (
            "ordered exact normalized-word matching; common edges are fixed, "
            "and SequenceMatcher aligns the remaining region"
        ),
        "percentile_method": "nearest-rank",
        "stride_grid": {
            "stride_seconds": args.stride_seconds,
            "tolerance_seconds": args.stride_tolerance_seconds,
            "grid_anchor": "source time zero",
        },
        "text_invariance": {
            "metric": "character-level Levenshtein edit distance",
            "raw": aggregate_text_metrics(sources, "raw"),
            "normalized": aggregate_text_metrics(sources, "normalized"),
        },
        "baseline_word_timestamps": baseline_global.to_json(),
        "timestamp_word_timestamps": timestamp_global.to_json(),
        "timing_comparison": timing_global.to_json(),
        "sources": sources,
    }


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeError) as error:
        print(error, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.name}.part")
    partial.write_text(
        f"{json.dumps(result, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    partial.replace(args.output)
    print(
        json.dumps(
            {
                "compared_source_count": result["compared_source_count"],
                "normalized_edit_distance": result["text_invariance"][
                    "normalized"
                ]["edit_distance"],
                "timestamp_valid_interval_rate": result[
                    "timestamp_word_timestamps"
                ]["timestamp_coverage"]["valid_interval_rate"],
                "matched_words_with_valid_intervals": result[
                    "timing_comparison"
                ]["matched_words_with_valid_intervals"],
                "output": str(args.output),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
