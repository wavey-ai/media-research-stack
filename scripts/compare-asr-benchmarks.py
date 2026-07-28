#!/usr/bin/env python3
"""Compare stable ASR benchmark summaries."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local and remote ASR matrix summary files."
    )
    parser.add_argument("--local", type=pathlib.Path, required=True)
    parser.add_argument("--remote", type=pathlib.Path, required=True)
    parser.add_argument("--expected-sources", type=int, default=10)
    parser.add_argument("--json-output", type=pathlib.Path)
    return parser.parse_args()


def load(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def stable(records: list[dict[str, Any]], expected: int) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("exit_code") == 0
        and record.get("completed_sources") == expected
        and record.get("failed_sources") == 0
        and isinstance(record.get("effective_rtfx"), (int, float))
    ]


def best(records: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    candidates = stable(records, expected)
    if not candidates:
        raise ValueError("summary does not contain a stable complete run")
    return max(candidates, key=lambda record: float(record["effective_rtfx"]))


def number(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> int:
    args = parse_args()
    try:
        local = best(load(args.local), args.expected_sources)
        remote_records = load(args.remote)
        remote = best(remote_records, args.expected_sources)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1

    speedup = float(remote["effective_rtfx"]) / max(
        float(local["effective_rtfx"]), 0.001
    )
    result = {
        "local": local,
        "remote_best": remote,
        "remote_stable_configurations": len(
            stable(remote_records, args.expected_sources)
        ),
        "remote_speedup_over_local": speedup,
    }
    print(
        "| Host | Provider | Sessions | Requests | Workers | Effective RTFx | "
        "Peak GPU MiB |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for host, record in (("Local", local), ("Linode", remote)):
        print(
            f"| {host} | {record['execution_provider']} | "
            f"{record['onnx_sessions']} | {record['asr_concurrency']} | "
            f"{record['worker_instances']} | {number(record['effective_rtfx'])} | "
            f"{number(record.get('gpu_memory_max_mib'), 0)} |"
        )
    print(f"\nLinode speedup: {speedup:.2f}x")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.json_output.with_name(f"{args.json_output.name}.part")
        partial.write_text(
            f"{json.dumps(result, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        partial.replace(args.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
