#!/usr/bin/env python3

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


class JsonlTail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.position = 0

    def read_new(self) -> list[dict]:
        if self.file is None:
            if not self.path.exists():
                return []
            self.file = self.path.open("r", encoding="utf-8")
            self.file.seek(self.position)

        records = []
        while line := self.file.readline():
            self.position = self.file.tell()
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")))
        output.write("\n")


def completed_metric(record: dict) -> dict:
    fields = (
        "index",
        "source_url",
        "audio_seconds",
        "asr_wall_seconds",
        "asr_rtfx",
        "wall_seconds",
        "rtfx",
        "asr_input_mib_per_second",
        "asr_transcript_words_per_second",
        "run_processed_sources",
        "run_audio_seconds",
        "run_asr_wall_seconds",
        "run_asr_rtfx",
        "run_pipeline_wall_seconds",
        "run_pipeline_rtfx",
        "run_transcript_words_per_second",
    )
    metric = {
        "type": "completed_source",
        "observed_at": utc_timestamp(),
    }
    metric.update({field: record.get(field) for field in fields})
    return metric


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record live and completed-source ASR throughput metrics."
    )
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")

    progress_tail = JsonlTail(args.progress)
    report_tail = JsonlTail(args.report)
    active_source = None
    audio_position = 0.0

    for record in progress_tail.read_new():
        event = record.get("event", {})
        if event.get("type") != "Results":
            continue
        duration = event.get("duration")
        if not isinstance(duration, (int, float)):
            continue
        active_source = record.get("source_index")
        audio_position = float(duration)

    report_tail.read_new()
    baseline_source = active_source
    baseline_audio = audio_position
    baseline_time = time.monotonic()
    keep_running = True

    def stop(_signum, _frame) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while keep_running:
        time.sleep(args.interval_seconds)

        for record in progress_tail.read_new():
            event = record.get("event", {})
            if event.get("type") != "Results":
                continue
            duration = event.get("duration")
            if not isinstance(duration, (int, float)):
                continue
            source_index = record.get("source_index")
            if source_index != active_source:
                active_source = source_index
                audio_position = 0.0
            audio_position = max(audio_position, float(duration))

        now = time.monotonic()
        if active_source != baseline_source:
            baseline_source = active_source
            baseline_audio = 0.0
            baseline_time = now

        sample_wall_seconds = now - baseline_time
        sample_audio_seconds = max(0.0, audio_position - baseline_audio)
        if (
            active_source is not None
            and sample_wall_seconds > 0
            and sample_audio_seconds > 0
        ):
            live_metric = {
                "type": "live_interval",
                "observed_at": utc_timestamp(),
                "source_index": active_source,
                "audio_position_seconds": audio_position,
                "sample_audio_seconds": sample_audio_seconds,
                "sample_wall_seconds": sample_wall_seconds,
                "sample_rtfx": sample_audio_seconds / sample_wall_seconds,
            }
            append_record(args.output, live_metric)
            print(json.dumps(live_metric, separators=(",", ":")), flush=True)

        baseline_audio = audio_position
        baseline_time = now

        for record in report_tail.read_new():
            if record.get("status") != "ok" or record.get("metrics_schema") not in (
                1,
                2,
            ):
                continue
            metric = completed_metric(record)
            append_record(args.output, metric)
            print(json.dumps(metric, separators=(",", ":")), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
