#!/usr/bin/env python3
"""Run repeatable ASR concurrency tests and collect host metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TextIO


DEFAULT_MATRIX = (
    "1:1:1,1:2:1,2:2:1,2:4:1,3:3:1,"
    "3:6:1,4:4:1,4:8:1"
)

RECORDED_RUNTIME_SETTINGS = (
    "ASR_COHERE_BACKEND",
    "ASR_COHERE_EXECUTION_PROVIDER",
    "ASR_COHERE_INTER_THREADS",
    "ASR_COHERE_INTRA_THREADS",
    "ASR_COHERE_MAX_NEW_TOKENS",
    "ASR_COHERE_PARALLEL_EXECUTION",
    "ASR_COHERE_CTC_TIMESTAMPS",
    "ASR_COHERE_TIMINGS",
    "ASR_COHERE_TIMESTAMP_BACKEND",
    "ASR_COHERE_TRT_COMPONENTS",
    "ASR_COHERE_TRT_FP16",
    "ASR_COHERE_TRT_PROFILE_MAX_S",
    "ASR_COHERE_TRT_PROFILE_MIN_S",
    "ASR_COHERE_TRT_PROFILE_OPT_S",
    "ASR_COHERE_TRT_WORKSPACE_BYTES",
    "ASR_CTC_ALIGN_DEVICE_ID",
    "ASR_CTC_ALIGN_EXECUTION_PROVIDER",
    "ASR_CTC_ALIGN_INTER_THREADS",
    "ASR_CTC_ALIGN_INTRA_THREADS",
    "ASR_CTC_ALIGN_MODEL_DIR",
    "ASR_CTC_ALIGN_OFFSET_MS",
    "ASR_CTC_ALIGN_ONNX_FILE",
    "ASR_CTC_ALIGN_PAD_TO",
    "ASR_CTC_ALIGN_SESSIONS",
    "ASR_CTC_ALIGN_TIMINGS",
    "ASR_CTC_ALIGN_TRT_FP16",
    "ASR_CTC_DIRECT_END_OFFSET_MS",
    "ASR_CTC_DIRECT_MAX_UNMATCHED_WORDS",
    "ASR_CTC_DIRECT_MIN_MATCHED_WORDS",
    "ASR_CTC_DIRECT_MIN_MATCH_RATIO",
    "ASR_CTC_DIRECT_START_OFFSET_MS",
    "ASR_DEVICE_IDS",
    "ASR_ONNX_SESSIONS",
    "MEDIA_RESEARCH_STACK_ASR_CONCURRENCY",
    "MEDIA_RESEARCH_STACK_CONTINUE_ON_ERROR",
    "MEDIA_RESEARCH_STACK_MIN_TRANSCRIPT_WORDS",
    "MEDIA_RESEARCH_STACK_STORE_TRANSCRIPTS",
    "MEDIA_RESEARCH_STACK_WORKER_INSTANCES",
    "UPLOAD_RESPONSE_MAX_INFLIGHT",
    "UPLOAD_RESPONSE_NUM_STREAMS",
    "UPLOAD_RESPONSE_RING_BYTES",
    "UPLOAD_RESPONSE_SLOT_SIZE_KB",
    "UPLOAD_RESPONSE_SLOTS_PER_STREAM",
    "UPLOAD_RESPONSE_TIMEOUT_MS",
    "UPLOAD_RESPONSE_WATCH_POLL_MS",
    "UPLOAD_RESPONSE_WORKER_POLL_MS",
)

RECORDED_RUNTIME_DEFAULTS = {
    "ASR_COHERE_MAX_NEW_TOKENS": "128",
    "ASR_CTC_ALIGN_SESSIONS": "1",
    "MEDIA_RESEARCH_STACK_CONTINUE_ON_ERROR": "false",
    "MEDIA_RESEARCH_STACK_STORE_TRANSCRIPTS": "false",
    "UPLOAD_RESPONSE_RING_BYTES": str(64 * 1024 * 1024),
    "UPLOAD_RESPONSE_SLOT_SIZE_KB": "32",
    "UPLOAD_RESPONSE_TIMEOUT_MS": str(6 * 60 * 60 * 1_000),
    "UPLOAD_RESPONSE_WATCH_POLL_MS": "1",
    "UPLOAD_RESPONSE_WORKER_POLL_MS": "2",
}


@dataclass(frozen=True)
class Configuration:
    sessions: int
    concurrency: int
    workers: int

    @property
    def name(self) -> str:
        return (
            f"sessions-{self.sessions}-concurrency-{self.concurrency}"
            f"-workers-{self.workers}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cached ASR benchmark for each session and request setting."
        )
    )
    parser.add_argument(
        "--matrix",
        default=DEFAULT_MATRIX,
        help="Comma-separated session:concurrency:worker values.",
    )
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        required=True,
        help="Cohere model directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("target/audiomovers/benchmark-10"),
        help="Directory that contains manifest.json and media/.",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=pathlib.Path("target/audiomovers/benchmark-10/runs"),
        help="Directory for reports, logs, metrics, and the summary.",
    )
    parser.add_argument(
        "--runtime-lib",
        type=pathlib.Path,
        help="Path to libonnxruntime.so on Linux.",
    )
    parser.add_argument(
        "--execution-provider",
        choices=("mlx", "cuda", "tensorrt"),
        default="tensorrt",
    )
    parser.add_argument(
        "--mlx-runtime",
        type=pathlib.Path,
        help="Path to the asr-mlx-transcribe executable on macOS.",
    )
    parser.add_argument(
        "--trt-cache-dir",
        type=pathlib.Path,
        default=pathlib.Path("target/trt-cache"),
    )
    parser.add_argument(
        "--test-binary",
        type=pathlib.Path,
        help="Compiled mastering_videos test binary. The script uses cargo by default.",
    )
    parser.add_argument(
        "--cargo-config",
        action="append",
        default=[],
        help="Pass one --config value to cargo. Repeat this option as necessary.",
    )
    parser.add_argument(
        "--minimum-transcript-words",
        type=int,
        default=5,
        help=(
            "Reject shorter transcripts. Use zero to retain silent and "
            "mostly-music sources."
        ),
    )
    return parser.parse_args()


def parse_matrix(value: str) -> list[Configuration]:
    configurations = []
    for entry in value.split(","):
        fields = entry.strip().split(":")
        if len(fields) != 3:
            raise ValueError(
                f"invalid matrix entry {entry!r}; expected sessions:concurrency:workers"
            )
        sessions, concurrency, workers = (int(field) for field in fields)
        if min(sessions, concurrency, workers) < 1:
            raise ValueError(f"matrix values must be greater than zero: {entry!r}")
        configurations.append(Configuration(sessions, concurrency, workers))
    if not configurations:
        raise ValueError("the matrix must contain one configuration")
    return configurations


def benchmark_command(args: argparse.Namespace) -> list[str]:
    if args.test_binary:
        return [
            str(args.test_binary.resolve()),
            "--exact",
            "transcribes_audio_mastering_videos",
            "--nocapture",
        ]
    cargo = os.environ.get("CARGO") or shutil.which("cargo")
    if not cargo:
        rustup_cargo = pathlib.Path.home() / ".cargo" / "bin" / "cargo"
        cargo = str(rustup_cargo) if rustup_cargo.is_file() else "cargo"
    command = [cargo]
    for value in args.cargo_config:
        command.extend(["--config", value])
    command.extend(
        [
            "test",
            "--locked",
            "--release",
            "--test",
            "mastering_videos",
            "--",
            "--exact",
            "transcribes_audio_mastering_videos",
            "--nocapture",
        ]
    )
    return command


def run_environment(
    args: argparse.Namespace,
    configuration: Configuration,
    report_path: pathlib.Path,
    progress_path: pathlib.Path,
    transcripts_dir: pathlib.Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MEDIA_RESEARCH_STACK_BENCH": "1",
            "MEDIA_RESEARCH_STACK_URLS_FILE": str(
                (args.dataset_dir / "manifest.json").resolve()
            ),
            "MEDIA_RESEARCH_STACK_MEDIA_DIR": str(
                (args.dataset_dir / "media").resolve()
            ),
            "MEDIA_RESEARCH_STACK_REQUIRE_CACHE": "1",
            "MEDIA_RESEARCH_STACK_CACHE_MIME_TYPE": "audio/webm",
            "MEDIA_RESEARCH_STACK_REPORT": str(report_path.resolve()),
            "MEDIA_RESEARCH_STACK_PROGRESS": str(progress_path.resolve()),
            "MEDIA_RESEARCH_STACK_TRANSCRIPTS_DIR": str(transcripts_dir.resolve()),
            "MEDIA_RESEARCH_STACK_STARTUP_GRACE_SECS": "0",
            "MEDIA_RESEARCH_STACK_ASR_CONCURRENCY": str(configuration.concurrency),
            "MEDIA_RESEARCH_STACK_WORKER_INSTANCES": str(configuration.workers),
            "MEDIA_RESEARCH_STACK_MIN_TRANSCRIPT_WORDS": str(
                args.minimum_transcript_words
            ),
            "ASR_MODEL_DIR": str(args.model_dir.resolve()),
            "ASR_MODEL_PROVIDER": "cohere",
            "ASR_COHERE_BACKEND": (
                "mlx" if args.execution_provider == "mlx" else "onnx"
            ),
            "ASR_DEVICE_IDS": "" if args.execution_provider == "mlx" else "0",
            "ASR_ONNX_SESSIONS": str(configuration.sessions),
            "UPLOAD_RESPONSE_MAX_INFLIGHT": str(configuration.concurrency),
            "UPLOAD_RESPONSE_NUM_STREAMS": str(
                max(2, configuration.concurrency)
            ),
            "RUST_LOG": environment.get("RUST_LOG", "warn"),
        }
    )
    if args.runtime_lib:
        environment["ASR_ONNX_RUNTIME_LIB"] = str(args.runtime_lib.resolve())
    if args.mlx_runtime:
        environment["ASR_MLX_TRANSCRIBE_BIN"] = str(args.mlx_runtime.resolve())
    if args.execution_provider == "tensorrt":
        environment.update(
            {
                "ASR_COHERE_TRT_COMPONENTS": "all",
                "ASR_COHERE_TRT_CACHE_DIR": str(args.trt_cache_dir.resolve()),
                "ASR_COHERE_TRT_FP16": "true",
                "ASR_COHERE_TRT_PROFILE_MIN_S": "1",
                "ASR_COHERE_TRT_PROFILE_OPT_S": "30",
                "ASR_COHERE_TRT_PROFILE_MAX_S": "35",
            }
        )
    elif args.execution_provider == "cuda":
        environment["ASR_COHERE_TRT_COMPONENTS"] = "none"
    return environment


def start_gpu_monitor(path: pathlib.Path) -> tuple[subprocess.Popen[str] | None, TextIO | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, None
    output = path.open("w", encoding="utf-8")
    fields = (
        "timestamp,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    process = subprocess.Popen(
        [
            executable,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "--loop-ms=500",
        ],
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, output


def stop_gpu_monitor(
    process: subprocess.Popen[str] | None,
    output: TextIO | None,
) -> None:
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    if output:
        output.close()


def read_json_lines(path: pathlib.Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def gpu_summary(path: pathlib.Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {
            "gpu_samples": 0,
            "gpu_utilization_mean_percent": None,
            "gpu_utilization_max_percent": None,
            "gpu_memory_max_mib": None,
            "gpu_temperature_max_c": None,
            "gpu_power_mean_w": None,
        }
    utilization = []
    memory = []
    temperature = []
    power = []
    with path.open(encoding="utf-8") as source:
        for row in csv.reader(source):
            if len(row) != 6:
                continue
            try:
                utilization.append(float(row[1]))
                memory.append(float(row[2]))
                temperature.append(float(row[4]))
                power.append(float(row[5]))
            except ValueError:
                continue
    return {
        "gpu_samples": len(utilization),
        "gpu_utilization_mean_percent": (
            sum(utilization) / len(utilization) if utilization else None
        ),
        "gpu_utilization_max_percent": max(utilization, default=None),
        "gpu_memory_max_mib": max(memory, default=None),
        "gpu_temperature_max_c": max(temperature, default=None),
        "gpu_power_mean_w": sum(power) / len(power) if power else None,
    }


def benchmark_metadata(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = manifest.get("selection", {})
    videos = manifest.get("videos", [])
    if not isinstance(selection, dict) or not isinstance(videos, list):
        raise ValueError("the benchmark manifest has an invalid structure")
    fingerprint = selection.get("sha256")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("the benchmark manifest does not contain selection.sha256")
    return {
        "model": os.environ.get(
            "ASR_COHERE_SOURCE_MODEL",
            "CohereLabs/cohere-transcribe-03-2026",
        ),
        "model_directory": str(args.model_dir.resolve()),
        "dataset_fingerprint": fingerprint,
        "dataset_manifest_sources": len(videos),
        "dataset_manifest_audio_seconds": selection.get(
            "total_duration_seconds"
        ),
        "minimum_transcript_words": args.minimum_transcript_words,
        "host": platform.node(),
        "architecture": native_architecture(),
        "runner_architecture": platform.machine(),
        "operating_system": platform.platform(),
    }


def native_architecture() -> str:
    try:
        result = subprocess.run(
            ["uname", "-m"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return platform.machine()
    return result.stdout.strip() or platform.machine()


def effective_runtime_settings(environment: dict[str, str]) -> dict[str, str]:
    settings = {
        name: environment.get(name, RECORDED_RUNTIME_DEFAULTS.get(name))
        for name in RECORDED_RUNTIME_SETTINGS
        if name in environment or name in RECORDED_RUNTIME_DEFAULTS
    }
    if "ASR_COHERE_TIMESTAMP_BACKEND" not in environment:
        legacy_ctc = bool(environment.get("ASR_CTC_ALIGN_MODEL_DIR", "").strip())
        legacy_ctc = legacy_ctc or environment.get(
            "ASR_COHERE_CTC_TIMESTAMPS",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        settings["ASR_COHERE_TIMESTAMP_BACKEND"] = (
            "parakeet-ctc" if legacy_ctc else "token-frequency"
        )
    return settings


def summarize(
    configuration: Configuration,
    execution_provider: str,
    return_code: int,
    process_seconds: float,
    report_path: pathlib.Path,
    gpu_path: pathlib.Path,
    metadata: dict[str, object],
    command: list[str],
    environment: dict[str, str],
) -> dict[str, object]:
    records = read_json_lines(report_path)
    successes = [record for record in records if record.get("status") == "ok"]
    errors = [record for record in records if record.get("status") == "error"]
    audio_seconds = sum(float(record["audio_seconds"]) for record in successes)
    asr_seconds = sum(float(record["asr_wall_seconds"]) for record in successes)
    final = max(
        successes,
        key=lambda record: int(record.get("run_processed_sources", 0)),
        default={},
    )
    result: dict[str, object] = {
        "configuration": configuration.name,
        "execution_provider": execution_provider,
        "onnx_sessions": configuration.sessions,
        "asr_concurrency": configuration.concurrency,
        "worker_instances": configuration.workers,
        "exit_code": return_code,
        "process_wall_seconds": process_seconds,
        "completed_sources": len(successes),
        "failed_sources": len(errors),
        "audio_seconds": audio_seconds,
        "asr_service_seconds": asr_seconds,
        "asr_service_rtfx": audio_seconds / max(asr_seconds, 0.001),
        "benchmark_elapsed_seconds": final.get("run_elapsed_seconds"),
        "effective_rtfx": final.get("run_effective_rtfx"),
        "command": command,
        "runtime_settings": effective_runtime_settings(environment),
    }
    result.update(metadata)
    result.update(gpu_summary(gpu_path))
    return result


def main() -> int:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args = parse_args()
    if args.minimum_transcript_words < 0:
        print("--minimum-transcript-words cannot be negative", file=sys.stderr)
        return 2
    try:
        configurations = parse_matrix(args.matrix)
    except (ValueError, TypeError) as error:
        print(error, file=sys.stderr)
        return 2

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.trt_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        metadata = benchmark_metadata(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2
    summary_path = args.results_dir / (
        f"summary-{args.execution_provider}-{int(time.time())}.jsonl"
    )
    command = benchmark_command(args)
    successful_configurations = 0

    with summary_path.open("w", encoding="utf-8") as summary_file:
        for configuration in configurations:
            run_dir = args.results_dir / (
                f"{args.execution_provider}-{configuration.name}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path = run_dir / "report.jsonl"
            progress_path = run_dir / "progress.ndjson"
            transcripts_dir = run_dir / "transcripts"
            gpu_path = run_dir / "gpu.csv"
            log_path = run_dir / "benchmark.log"
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            for path in (report_path, progress_path, gpu_path, log_path):
                path.unlink(missing_ok=True)

            environment = run_environment(
                args,
                configuration,
                report_path,
                progress_path,
                transcripts_dir,
            )
            print(f"starting {args.execution_provider} {configuration.name}", flush=True)
            monitor, monitor_output = start_gpu_monitor(gpu_path)
            started_at = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    command,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            process_seconds = time.monotonic() - started_at
            stop_gpu_monitor(monitor, monitor_output)

            summary = summarize(
                configuration,
                args.execution_provider,
                result.returncode,
                process_seconds,
                report_path,
                gpu_path,
                metadata,
                command,
                environment,
            )
            line = json.dumps(summary, separators=(",", ":"))
            summary_file.write(f"{line}\n")
            summary_file.flush()
            print(line, flush=True)
            if (
                summary["exit_code"] == 0
                and summary["completed_sources"]
                == metadata["dataset_manifest_sources"]
                and summary["failed_sources"] == 0
            ):
                successful_configurations += 1

    print(f"wrote {summary_path}", flush=True)
    if successful_configurations == 0:
        print("the matrix did not contain a stable complete run", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
