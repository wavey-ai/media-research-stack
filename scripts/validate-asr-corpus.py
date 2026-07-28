#!/usr/bin/env python3
"""Validate an ASR corpus without emitting transcript text."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import parse_qs, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ASR report, progress, and transcript artifacts. "
            "The output contains counts and identifiers, but no transcript text."
        )
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--progress", type=pathlib.Path, required=True)
    parser.add_argument("--transcripts", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--trailing-silence-seconds",
        type=float,
        default=10.5,
        help="Permit this gap between the last observed word and source end.",
    )
    parser.add_argument(
        "--max-service-rtfx",
        type=float,
        default=256.0,
        help="Reject longer sources that report a higher service real-time factor.",
    )
    parser.add_argument(
        "--early-completion-min-seconds",
        type=float,
        default=300.0,
        help="Apply the service real-time-factor limit at or above this duration.",
    )
    parser.add_argument(
        "--ring-stride-seconds",
        type=float,
        default=4096.0,
        help="Detect a coverage shortfall near a multiple of this ring stride.",
    )
    parser.add_argument(
        "--ring-stride-tolerance-seconds",
        type=float,
        default=10.0,
        help="Permit this residual when matching a ring-stride shortfall.",
    )
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=1.0,
        help="Permit this report-to-manifest duration difference.",
    )
    parser.add_argument(
        "--timestamp-regression-tolerance-seconds",
        type=float,
        default=0.05,
        help="Permit this timestamp decrease between result events.",
    )
    parser.add_argument(
        "--allow-progress-retries",
        action="store_true",
        help="Accept retry history and validate only the last progress attempt.",
    )
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


def transcript_name(index: int, url: str) -> str:
    return f"{index + 1:04d}-{video_id(url)}.txt"


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def load_videos(path: pathlib.Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = value.get("videos") if isinstance(value, dict) else value
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest JSON must contain a nonempty videos array")

    videos = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise ValueError(f"manifest source {index} must contain a URL")
        duration = finite_number(entry.get("duration_seconds"))
        if duration is None or duration <= 0.0:
            raise ValueError(
                f"manifest source {index} must contain a positive duration"
            )
        source_index = entry.get("source_index")
        if source_index is not None and source_index != index:
            raise ValueError(
                f"manifest source {index} has source_index {source_index!r}"
            )
        videos.append(entry)
    return videos


def load_report(
    path: pathlib.Path,
    videos: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as report:
        for line_number, line in enumerate(report, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} has invalid JSON at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path} line {line_number} is not a JSON object"
                )
            source_index = integer(record.get("index"))
            if source_index is None or not 0 <= source_index < len(videos):
                raise ValueError(
                    f"{path} line {line_number} has an invalid source index"
                )
            if record.get("source_url") != videos[source_index]["url"]:
                raise ValueError(
                    f"{path} line {line_number} does not match the manifest URL"
                )
            records[source_index].append(record)
    return dict(records)


@dataclasses.dataclass
class Attempt:
    source_index: int
    source_url: str
    ordinal: int
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    request_ids: set[str] = dataclasses.field(default_factory=set)


def event_request_id(event: dict[str, Any]) -> str | None:
    direct = event.get("request_id")
    if isinstance(direct, str) and direct:
        return direct
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("request_id")
        if isinstance(nested, str) and nested:
            return nested
    return None


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
                raise ValueError(
                    f"{path} line {line_number} is not a JSON object"
                )
            source_index = integer(wrapper.get("source_index"))
            if source_index is None or not 0 <= source_index < len(videos):
                raise ValueError(
                    f"{path} line {line_number} has an invalid source_index"
                )
            source_url = wrapper.get("source_url")
            if source_url != videos[source_index]["url"]:
                raise ValueError(
                    f"{path} line {line_number} does not match the manifest URL"
                )
            event = wrapper.get("event")
            if not isinstance(event, dict):
                raise ValueError(
                    f"{path} line {line_number} has no event object"
                )

            kind = event.get("type")
            kind = kind if isinstance(kind, str) else None
            request_id = event_request_id(event)
            active = current.get(source_index)
            starts_attempt = (
                active is None
                or (kind == "ResponseHead" and bool(active.events))
                or (
                    request_id is not None
                    and bool(active.request_ids)
                    and request_id not in active.request_ids
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
                active.request_ids.add(request_id)
    return dict(attempts)


def first_alternative(event: dict[str, Any]) -> dict[str, Any] | None:
    channel = event.get("channel")
    if not isinstance(channel, dict):
        return None
    alternatives = channel.get("alternatives")
    if (
        not isinstance(alternatives, list)
        or not alternatives
        or not isinstance(alternatives[0], dict)
    ):
        return None
    return alternatives[0]


def text_counts(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return len(trimmed), len(trimmed.split())


def alternative_counts(alternative: dict[str, Any]) -> tuple[int, int] | None:
    counts = text_counts(alternative.get("transcript"))
    if counts is not None:
        return counts
    characters = integer(alternative.get("transcript_chars"))
    words = integer(alternative.get("transcript_words"))
    if (
        characters is None
        or words is None
        or characters < 0
        or words < 0
    ):
        return None
    return characters, words


def attempt_metrics(
    attempt: Attempt,
    regression_tolerance: float,
) -> dict[str, Any]:
    response_statuses: list[int] = []
    metadata_events = 0
    result_events = 0
    final_result_events = 0
    error_events = 0
    word_array_count = 0
    final_segments: list[tuple[int, int]] = []
    last_interim: tuple[int, int] | None = None
    coverage_candidates: list[float] = []
    completion_coverage_candidates: list[float] = []
    result_duration_max: float | None = None
    previous_result_endpoint: float | None = None
    timestamp_regressions = 0
    transcript_counts_available = False

    for event in attempt.events:
        kind = event.get("type")
        if kind == "ResponseHead":
            status = integer(event.get("status"))
            if status is not None:
                response_statuses.append(status)
        elif kind == "Metadata":
            metadata_events += 1
        elif kind == "UtteranceEnd":
            endpoint = finite_number(event.get("last_word_end"))
            if endpoint is not None:
                coverage_candidates.append(endpoint)
        elif kind == "Results":
            result_events += 1
            duration = finite_number(event.get("duration"))
            if duration is not None:
                result_duration_max = (
                    duration
                    if result_duration_max is None
                    else max(result_duration_max, duration)
                )
            alternative = first_alternative(event)
            result_endpoint: float | None = None
            if alternative is not None:
                words = alternative.get("words")
                if isinstance(words, list):
                    word_array_count += len(words)
                    for word in words:
                        if not isinstance(word, dict):
                            continue
                        endpoint = finite_number(word.get("end"))
                        if endpoint is not None:
                            coverage_candidates.append(endpoint)
                            result_endpoint = (
                                endpoint
                                if result_endpoint is None
                                else max(result_endpoint, endpoint)
                            )
                counts = alternative_counts(alternative)
                if counts is not None:
                    transcript_counts_available = True
                    if counts[0] > 0:
                        if event.get("is_final") is True:
                            final_result_events += 1
                            final_segments.append(counts)
                        else:
                            last_interim = counts
                    if (
                        counts == (0, 0)
                        and event.get("is_final") is True
                        and event.get("speech_final") is True
                        and event.get("from_finalize") is True
                    ):
                        start = finite_number(event.get("start"))
                        if (
                            duration is not None
                            and start is not None
                            and abs(start) <= regression_tolerance
                        ):
                            completion_coverage_candidates.append(duration)
            if (
                result_endpoint is not None
                and previous_result_endpoint is not None
                and result_endpoint + regression_tolerance
                < previous_result_endpoint
            ):
                timestamp_regressions += 1
            if result_endpoint is not None:
                previous_result_endpoint = result_endpoint

        if "error" in event:
            error_events += 1

    if final_segments:
        transcript_chars = sum(chars for chars, _ in final_segments)
        transcript_chars += len(final_segments) - 1
        transcript_words = sum(words for _, words in final_segments)
    elif last_interim is not None:
        transcript_chars, transcript_words = last_interim
    else:
        transcript_chars = 0
        transcript_words = 0

    return {
        "attempt_ordinal": attempt.ordinal,
        "event_count": len(attempt.events),
        "response_statuses": response_statuses,
        "metadata_events": metadata_events,
        "request_ids": sorted(attempt.request_ids),
        "result_events": result_events,
        "final_result_events": final_result_events,
        "error_events": error_events,
        "word_array_count": word_array_count,
        "transcript_chars": transcript_chars,
        "transcript_words": transcript_words,
        "transcript_counts_available": transcript_counts_available,
        "speech_coverage_seconds": max(coverage_candidates, default=0.0),
        "processed_audio_seconds": (
            max(completion_coverage_candidates)
            if completion_coverage_candidates
            else None
        ),
        "result_duration_max": result_duration_max,
        "timestamp_regressions": timestamp_regressions,
    }


def add_issue(issues: list[str], issue: str) -> None:
    if issue not in issues:
        issues.append(issue)


def report_count(record: dict[str, Any], field: str) -> int | None:
    value = integer(record.get(field))
    return value if value is not None and value >= 0 else None


def transcript_metrics(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    normalized = text.strip()
    return {
        "characters": len(normalized),
        "words": len(normalized.split()),
        "bytes": path.stat().st_size,
        "ends_with_newline": text.endswith("\n"),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    numeric_options = (
        ("trailing silence", args.trailing_silence_seconds, True),
        ("maximum service RTFx", args.max_service_rtfx, False),
        ("early completion duration", args.early_completion_min_seconds, True),
        ("ring stride", args.ring_stride_seconds, True),
        ("ring tolerance", args.ring_stride_tolerance_seconds, True),
        ("duration tolerance", args.duration_tolerance_seconds, True),
        (
            "timestamp regression tolerance",
            args.timestamp_regression_tolerance_seconds,
            True,
        ),
    )
    for label, value, allow_zero in numeric_options:
        if not math.isfinite(value) or value < 0.0 or (
            not allow_zero and value == 0.0
        ):
            raise ValueError(f"{label} must be a valid nonnegative number")

    videos = load_videos(args.manifest)
    report_records = load_report(args.report, videos)
    attempts = load_attempts(args.progress, videos)
    expected_names = {
        transcript_name(index, video["url"])
        for index, video in enumerate(videos)
    }
    actual_names = {
        path.name for path in args.transcripts.glob("*.txt") if path.is_file()
    }
    part_files = sorted(
        path.name for path in args.transcripts.glob("*.part") if path.is_file()
    )
    unexpected_names = sorted(actual_names - expected_names)
    global_issues = []
    if unexpected_names:
        global_issues.append("unexpected_transcript_files")
    if part_files:
        global_issues.append("partial_transcript_files")

    selected_request_ids: dict[str, list[int]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    for index, video in enumerate(videos):
        source_issues: list[str] = []
        source_reports = report_records.get(index, [])
        successful_reports = [
            record for record in source_reports if record.get("status") == "ok"
        ]
        if not successful_reports:
            add_issue(source_issues, "missing_successful_report")
            selected_report = None
        else:
            selected_report = successful_reports[-1]
            if len(successful_reports) > 1:
                add_issue(source_issues, "duplicate_successful_reports")

        source_attempts = attempts.get(index, [])
        progress_attempts = [
            attempt_metrics(
                attempt,
                args.timestamp_regression_tolerance_seconds,
            )
            for attempt in source_attempts
        ]
        if not source_attempts:
            add_issue(source_issues, "missing_progress_attempt")
            selected_attempt = None
            progress_metrics = None
        else:
            selected_attempt = source_attempts[-1]
            progress_metrics = progress_attempts[-1]
            if len(source_attempts) > 1 and not args.allow_progress_retries:
                add_issue(source_issues, "duplicate_progress_attempts")
            statuses = progress_metrics["response_statuses"]
            if len(statuses) != 1:
                add_issue(source_issues, "invalid_response_head_count")
            elif not 200 <= statuses[0] < 300:
                add_issue(source_issues, "unsuccessful_response_status")
            if progress_metrics["metadata_events"] != 1:
                add_issue(source_issues, "invalid_metadata_event_count")
            request_ids = progress_metrics["request_ids"]
            if len(request_ids) != 1:
                add_issue(source_issues, "invalid_request_id_count")
            else:
                selected_request_ids[request_ids[0]].append(index)
            if progress_metrics["result_events"] == 0:
                add_issue(source_issues, "missing_results_events")
            if not progress_metrics["transcript_counts_available"]:
                add_issue(source_issues, "missing_progress_transcript_counts")
            if progress_metrics["processed_audio_seconds"] is None:
                add_issue(source_issues, "missing_processed_audio_marker")
            if progress_metrics["error_events"] > 0:
                add_issue(source_issues, "progress_error_event")
            if progress_metrics["timestamp_regressions"] > 0:
                add_issue(source_issues, "timestamp_regression")

        duration = float(video["duration_seconds"])
        report_audio_seconds = None
        report_words = None
        report_chars = None
        service_rtfx = None
        if selected_report is not None:
            report_audio_seconds = finite_number(
                selected_report.get("audio_seconds")
            )
            if report_audio_seconds is None:
                add_issue(source_issues, "missing_report_audio_duration")
            elif (
                abs(report_audio_seconds - duration)
                > args.duration_tolerance_seconds
            ):
                add_issue(source_issues, "report_duration_mismatch")
            report_words = report_count(selected_report, "transcript_words")
            report_chars = report_count(selected_report, "transcript_chars")
            if report_words is None:
                add_issue(source_issues, "invalid_report_transcript_words")
            if report_chars is None:
                add_issue(source_issues, "invalid_report_transcript_chars")
            wall_seconds = finite_number(
                selected_report.get("asr_wall_seconds")
            )
            if wall_seconds is not None and wall_seconds > 0.0:
                service_rtfx = duration / wall_seconds
                if (
                    duration >= args.early_completion_min_seconds
                    and service_rtfx > args.max_service_rtfx
                ):
                    add_issue(
                        source_issues,
                        "implausibly_early_response_completion",
                    )

        processed_audio_seconds = (
            progress_metrics["processed_audio_seconds"]
            if progress_metrics is not None
            else None
        )
        speech_coverage_seconds = (
            progress_metrics["speech_coverage_seconds"]
            if progress_metrics is not None
            else 0.0
        )
        coverage_seconds = (
            processed_audio_seconds
            if processed_audio_seconds is not None
            else speech_coverage_seconds
        )
        coverage_basis = (
            "processed_audio_marker"
            if processed_audio_seconds is not None
            else "speech_timestamps"
        )
        coverage_gap = max(0.0, duration - coverage_seconds)
        coverage_ratio = coverage_seconds / duration
        ring_stride_match = None
        if coverage_gap > args.trailing_silence_seconds:
            add_issue(source_issues, "source_duration_not_covered")
            if args.ring_stride_seconds > 0.0:
                multiple = max(
                    1,
                    round(coverage_gap / args.ring_stride_seconds),
                )
                residual = abs(
                    coverage_gap - multiple * args.ring_stride_seconds
                )
                if residual <= args.ring_stride_tolerance_seconds:
                    add_issue(source_issues, "ring_stride_shortfall")
                    ring_stride_match = {
                        "multiple": multiple,
                        "residual_seconds": residual,
                    }

        expected_name = transcript_name(index, video["url"])
        transcript_path = args.transcripts / expected_name
        file_metrics = None
        if not transcript_path.is_file():
            add_issue(source_issues, "missing_transcript_file")
        else:
            file_metrics = transcript_metrics(transcript_path)
            if (
                report_words is not None
                and file_metrics["words"] != report_words
            ):
                add_issue(source_issues, "transcript_file_word_count_mismatch")
            if (
                report_chars is not None
                and file_metrics["characters"] != report_chars
            ):
                add_issue(
                    source_issues,
                    "transcript_file_character_count_mismatch",
                )
            if not file_metrics["ends_with_newline"]:
                add_issue(source_issues, "transcript_file_missing_newline")

        if selected_report is not None:
            recorded_path = selected_report.get("transcript_path")
            if not isinstance(recorded_path, str):
                add_issue(source_issues, "missing_report_transcript_path")
            elif pathlib.Path(recorded_path).name != expected_name:
                add_issue(source_issues, "report_transcript_path_mismatch")

        if progress_metrics is not None:
            if (
                report_words is not None
                and progress_metrics["transcript_words"] != report_words
            ):
                add_issue(
                    source_issues,
                    "progress_transcript_word_count_mismatch",
                )
            if (
                report_chars is not None
                and progress_metrics["transcript_chars"] != report_chars
            ):
                add_issue(
                    source_issues,
                    "progress_transcript_character_count_mismatch",
                )

        sources.append(
            {
                "index": index,
                "id": video_id(video["url"]),
                "source_url": video["url"],
                "duration_seconds": duration,
                "report_record_count": len(source_reports),
                "report_error_count": sum(
                    record.get("status") != "ok" for record in source_reports
                ),
                "successful_report_count": len(successful_reports),
                "progress_attempt_count": len(source_attempts),
                "progress_attempts": progress_attempts,
                "selected_attempt_ordinal": (
                    selected_attempt.ordinal
                    if selected_attempt is not None
                    else None
                ),
                "request_id": (
                    progress_metrics["request_ids"][0]
                    if progress_metrics is not None
                    and len(progress_metrics["request_ids"]) == 1
                    else None
                ),
                "coverage_seconds": coverage_seconds,
                "coverage_basis": coverage_basis,
                "processed_audio_seconds": processed_audio_seconds,
                "speech_coverage_seconds": speech_coverage_seconds,
                "coverage_gap_seconds": coverage_gap,
                "coverage_ratio": coverage_ratio,
                "ring_stride_match": ring_stride_match,
                "service_rtfx": service_rtfx,
                "report_transcript_words": report_words,
                "report_transcript_characters": report_chars,
                "progress_transcript_words": (
                    progress_metrics["transcript_words"]
                    if progress_metrics is not None
                    else None
                ),
                "progress_transcript_characters": (
                    progress_metrics["transcript_chars"]
                    if progress_metrics is not None
                    else None
                ),
                "progress_word_array_count": (
                    progress_metrics["word_array_count"]
                    if progress_metrics is not None
                    else None
                ),
                "transcript_file": expected_name,
                "transcript_file_metrics": file_metrics,
                "issues": source_issues,
            }
        )

    for request_id, indexes in selected_request_ids.items():
        if len(indexes) < 2:
            continue
        for index in indexes:
            add_issue(sources[index]["issues"], "duplicate_request_id")

    invalid_sources = [source for source in sources if source["issues"]]
    issue_counts = Counter(
        issue for source in invalid_sources for issue in source["issues"]
    )
    valid = not invalid_sources and not global_issues
    return {
        "schema_version": 1,
        "valid": valid,
        "transcript_text_included": False,
        "policy": {
            "trailing_silence_seconds": args.trailing_silence_seconds,
            "max_service_rtfx": args.max_service_rtfx,
            "early_completion_min_seconds": (
                args.early_completion_min_seconds
            ),
            "ring_stride_seconds": args.ring_stride_seconds,
            "ring_stride_tolerance_seconds": (
                args.ring_stride_tolerance_seconds
            ),
            "duration_tolerance_seconds": args.duration_tolerance_seconds,
            "timestamp_regression_tolerance_seconds": (
                args.timestamp_regression_tolerance_seconds
            ),
            "allow_progress_retries": args.allow_progress_retries,
        },
        "summary": {
            "manifest_source_count": len(videos),
            "invalid_source_count": len(invalid_sources),
            "valid_source_count": len(videos) - len(invalid_sources),
            "issue_counts": dict(sorted(issue_counts.items())),
            "global_issue_count": len(global_issues),
        },
        "global_issues": global_issues,
        "unexpected_transcript_files": unexpected_names,
        "partial_transcript_files": part_files,
        "rerun_source_indexes": [
            source["index"] for source in invalid_sources
        ],
        "invalid_sources": invalid_sources,
        "sources": sources,
    }


def write_output(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(
        f"{json.dumps(value, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    partial.replace(path)


def main() -> int:
    args = parse_args()
    try:
        result = validate(args)
        write_output(args.output, result)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeError) as error:
        print(error, file=sys.stderr)
        return 2

    summary = result["summary"]
    state = "passed" if result["valid"] else "failed"
    print(
        f"ASR corpus validation {state}: "
        f"{summary['invalid_source_count']} invalid source(s), "
        f"{summary['global_issue_count']} global issue(s); "
        f"details: {args.output}",
        file=sys.stderr,
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
