#!/usr/bin/env python3
"""Create a repeatable ASR benchmark set from a complete media cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select cached sources by duration and publish a numbered benchmark set."
        )
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--media-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-duration", type=int, default=300)
    parser.add_argument("--max-duration", type=int, default=900)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy media instead of using hard links when possible.",
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


def source_stem(index: int, url: str) -> str:
    return f"{index + 1:04d}-{video_id(url)}"


def load_videos(path: pathlib.Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = value.get("videos") if isinstance(value, dict) else value
    if not isinstance(entries, list):
        raise ValueError("manifest JSON must be an array or contain a videos array")
    videos = []
    for entry in entries:
        if isinstance(entry, str):
            videos.append({"url": entry})
        elif isinstance(entry, dict) and isinstance(entry.get("url"), str):
            videos.append(entry)
        else:
            raise ValueError("each manifest entry must contain a URL")
    return videos


def load_cached_source(
    media_dir: pathlib.Path,
    index: int,
    url: str,
) -> tuple[pathlib.Path, pathlib.Path, dict[str, Any]] | None:
    stem = source_stem(index, url)
    audio_path = media_dir / f"{stem}.audio"
    metadata_path = media_dir / f"{stem}.json"
    if not audio_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_url") != url:
        raise ValueError(f"cache metadata URL does not match {metadata_path}")
    file_size = audio_path.stat().st_size
    if file_size < 1 or metadata.get("content_length") != file_size:
        raise ValueError(f"cache media size does not match {metadata_path}")
    mime_type = metadata.get("source_mime_type") or metadata.get("content_type")
    if not isinstance(mime_type, str) or mime_type.split(";", 1)[0] != "audio/webm":
        raise ValueError(f"cache source is not audio/webm: {metadata_path}")
    return audio_path, metadata_path, metadata


def publish_file(source: pathlib.Path, destination: pathlib.Path, copy: bool) -> None:
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    if copy:
        shutil.copy2(source, partial)
    else:
        try:
            os.link(source, partial)
        except OSError:
            shutil.copy2(source, partial)
    partial.replace(destination)


def fingerprint(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["url"].encode())
        digest.update(b"\0")
        digest.update(str(entry["duration_seconds"]).encode())
        digest.update(b"\0")
        digest.update(str(entry["content_length"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        print("--count must be greater than zero", file=sys.stderr)
        return 2
    if args.min_duration < 0 or args.max_duration < args.min_duration:
        print("the duration range is invalid", file=sys.stderr)
        return 2

    output_media = args.output_dir / "media"
    output_media.mkdir(parents=True, exist_ok=True)
    selected = []
    for source_index, video in enumerate(load_videos(args.manifest)):
        url = video["url"]
        cached = load_cached_source(args.media_dir, source_index, url)
        if cached is None:
            continue
        audio_path, metadata_path, metadata = cached
        duration = int(metadata["duration_seconds"])
        if not args.min_duration <= duration <= args.max_duration:
            continue

        output_index = len(selected)
        stem = source_stem(output_index, url)
        publish_file(audio_path, output_media / f"{stem}.audio", args.copy)
        publish_file(metadata_path, output_media / f"{stem}.json", args.copy)
        selected.append(
            {
                "source_index": source_index,
                "id": video_id(url),
                "title": video.get("title"),
                "url": url,
                "duration_seconds": duration,
                "content_length": metadata["content_length"],
                "source_mime_type": metadata.get("source_mime_type"),
                "itag": metadata.get("itag"),
            }
        )
        if len(selected) == args.count:
            break

    if len(selected) != args.count:
        print(
            f"found {len(selected)} matching cached sources; expected {args.count}",
            file=sys.stderr,
        )
        return 1

    manifest = {
        "schema_version": 1,
        "selection": {
            "count": args.count,
            "minimum_duration_seconds": args.min_duration,
            "maximum_duration_seconds": args.max_duration,
            "total_duration_seconds": sum(
                entry["duration_seconds"] for entry in selected
            ),
            "sha256": fingerprint(selected),
        },
        "videos": selected,
    }
    manifest_path = args.output_dir / "manifest.json"
    partial_path = args.output_dir / "manifest.json.part"
    partial_path.write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    partial_path.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sources": len(selected),
                "audio_seconds": manifest["selection"]["total_duration_seconds"],
                "sha256": manifest["selection"]["sha256"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
