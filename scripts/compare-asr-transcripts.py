#!/usr/bin/env python3
"""Compare ASR transcripts with character-level Levenshtein distance."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

from asr_metrics import (
    distance_ratio,
    levenshtein_distance,
    normalize_transcript,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two transcript sets with raw and normalized character edit "
            "distance."
        )
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--reference-dir", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-dir", type=pathlib.Path, required=True)
    parser.add_argument("--reference-validation", type=pathlib.Path)
    parser.add_argument("--candidate-validation", type=pathlib.Path)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
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
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise ValueError("each manifest entry must contain a URL")
        videos.append(entry)
    return videos


def transcript_path(directory: pathlib.Path, index: int, url: str) -> pathlib.Path:
    return directory / f"{index + 1:04d}-{video_id(url)}.txt"


def read_transcript(path: pathlib.Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as transcript:
        return transcript.read()


def validated_source_indexes(
    path: pathlib.Path,
    videos: list[dict[str, Any]],
) -> set[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    sources = value.get("sources") if isinstance(value, dict) else None
    if not isinstance(sources, list):
        raise ValueError(f"{path} does not contain a validation sources array")
    indexes: set[int] = set()
    seen: set[int] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(f"{path} contains an invalid validation source")
        index = source.get("index")
        issues = source.get("issues")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(videos)
            or not isinstance(issues, list)
            or index in seen
        ):
            raise ValueError(f"{path} contains invalid validation source data")
        seen.add(index)
        if not issues:
            indexes.add(index)
    if seen != set(range(len(videos))):
        raise ValueError(f"{path} does not cover the complete manifest")
    return indexes


def ratio(distance: int, reference_length: int, candidate_length: int) -> float:
    return distance_ratio(distance, reference_length, candidate_length)


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def aggregate(
    sources: list[dict[str, Any]],
    key: str,
    reference_length_key: str,
    candidate_length_key: str,
) -> dict[str, Any]:
    distances = [int(source[key]) for source in sources]
    reference_lengths = [
        int(source[reference_length_key]) for source in sources
    ]
    candidate_lengths = [
        int(source[candidate_length_key]) for source in sources
    ]
    ratios = [
        ratio(distance, reference_length, candidate_length)
        for distance, reference_length, candidate_length in zip(
            distances,
            reference_lengths,
            candidate_lengths,
        )
    ]
    denominator = sum(
        max(reference_length, candidate_length)
        for reference_length, candidate_length in zip(
            reference_lengths,
            candidate_lengths,
        )
    )
    return {
        "reference_characters": sum(reference_lengths),
        "candidate_characters": sum(candidate_lengths),
        "edit_distance": sum(distances),
        "distance_ratio": sum(distances) / max(denominator, 1),
        "exact_matches": sum(distance == 0 for distance in distances),
        "source_ratio_distribution": distribution(ratios),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    videos = load_videos(args.manifest)
    included_indexes = set(range(len(videos)))
    validation_selection = {}
    for label, path in (
        ("reference", args.reference_validation),
        ("candidate", args.candidate_validation),
    ):
        if path is None:
            continue
        valid_indexes = validated_source_indexes(path, videos)
        included_indexes &= valid_indexes
        validation_selection[label] = {
            "path": str(path),
            "valid_source_count": len(valid_indexes),
        }
    if not included_indexes:
        raise ValueError("the validation intersection has no complete sources")

    missing: list[str] = []
    sources = []
    for index, video in enumerate(videos):
        if index not in included_indexes:
            continue
        url = video["url"]
        reference_path = transcript_path(args.reference_dir, index, url)
        candidate_path = transcript_path(args.candidate_dir, index, url)
        for label, path in (
            (args.reference_label, reference_path),
            (args.candidate_label, candidate_path),
        ):
            if not path.is_file():
                missing.append(f"{label}: {path.name}")
        if missing and (
            not reference_path.is_file() or not candidate_path.is_file()
        ):
            continue

        reference = read_transcript(reference_path)
        candidate = read_transcript(candidate_path)
        normalized_reference = normalize_transcript(reference)
        normalized_candidate = normalize_transcript(candidate)
        raw_distance = levenshtein_distance(reference, candidate)
        normalized_distance = levenshtein_distance(
            normalized_reference,
            normalized_candidate,
        )
        sources.append(
            {
                "index": index,
                "id": video_id(url),
                "source_url": url,
                "raw_reference_characters": len(reference),
                "raw_candidate_characters": len(candidate),
                "raw_edit_distance": raw_distance,
                "raw_distance_ratio": ratio(
                    raw_distance,
                    len(reference),
                    len(candidate),
                ),
                "normalized_reference_characters": len(normalized_reference),
                "normalized_candidate_characters": len(normalized_candidate),
                "normalized_edit_distance": normalized_distance,
                "normalized_distance_ratio": ratio(
                    normalized_distance,
                    len(normalized_reference),
                    len(normalized_candidate),
                ),
            }
        )
    if missing:
        raise ValueError(
            f"{len(missing)} transcript file(s) are missing: {', '.join(missing[:5])}"
        )

    return {
        "schema_version": 1,
        "metric": "character-level Levenshtein edit distance",
        "unit": "Unicode code point",
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "normalization": (
            "NFKC, Unicode case folding, and whitespace collapse; punctuation kept"
        ),
        "manifest_source_count": len(videos),
        "source_count": len(sources),
        "excluded_source_count": len(videos) - len(sources),
        "validation_selection": validation_selection,
        "raw": aggregate(
            sources,
            "raw_edit_distance",
            "raw_reference_characters",
            "raw_candidate_characters",
        ),
        "normalized": aggregate(
            sources,
            "normalized_edit_distance",
            "normalized_reference_characters",
            "normalized_candidate_characters",
        ),
        "sources": sources,
    }


def main() -> int:
    args = parse_args()
    try:
        result = compare(args)
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
                "source_count": result["source_count"],
                "raw": result["raw"],
                "normalized": result["normalized"],
                "output": str(args.output),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
