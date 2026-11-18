"""Start a new statistic run with updated code, retaining verified successful labels."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from legal_application.config import rooted
from legal_application.data import strict_json
from legal_application.storage import atomic_json, file_hash, read_jsonl
from resume_statistic import DEFAULT_PROJECT_ROOT, resume_environment


def prepare_retry(root: Path, output: Path, source_run: str, *, input_jsonl: Path | None = None,
                  run_id: str | None = None, batch_size: int | None = None, judge_concurrency: int | None = None,
                  judge_json_retries: int = 4, max_output_tokens: int | None = None,
                  judge_thinking_mode: str | None = None) -> dict:
    run_id = run_id or (datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("retry-%Y%m%dT%H%M%S-") + uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Provide a simple new RUN_ID")
    for name, value, minimum in [("batch_size", batch_size, 1), ("judge_concurrency", judge_concurrency, 1),
                                  ("judge_json_retries", judge_json_retries, 0), ("max_output_tokens", max_output_tokens, 1)]:
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{name} must be >= {minimum}")
    root, output = root.resolve(), output.resolve()
    source_dir = Path(source_run)
    source_dir = source_dir.resolve() if source_dir.is_absolute() else (output / source_dir).resolve()
    if (output / run_id).exists():
        raise ValueError("New RUN_ID already exists; choose a new one")
    with (source_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Source statistic run is active; stop it and wait for exit") from None
        saved = strict_json((source_dir / "config.json").read_text(encoding="utf-8"))
        if saved.get("pipeline_kind") != "data_completeness" or saved.get("mode") != "statistic":
            raise ValueError("Source must be a statistic run")
        source_input = rooted(root, str(input_jsonl)) if input_jsonl else Path(saved["input_jsonl"])
        if file_hash(source_input) != saved["input_sha256"]:
            raise ValueError("Input checksum differs from the source run; old labels cannot be reused")
        details = source_dir / "classification_details.jsonl"
        rows = read_jsonl(details)
        successes = {row["doc_name"] for row in rows}
        failures = {}
        for error in read_jsonl(source_dir / "errors.jsonl"):
            if error.get("stage") == "classification" and error.get("doc_name") not in successes:
                metadata = error.get("metadata") or {}
                reasons = metadata.get("validation_errors") or [error.get("error", "unknown")]
                failures[error.get("doc_name")] = reasons[-1]
        seed_dir = output / ".statistic-seeds" / run_id
        seed_dir.mkdir(parents=True, exist_ok=False)
        seed = seed_dir / "classification_details.jsonl"
        shutil.copyfile(details, seed)
        # Pipeline verifies every imported raw response, source identity/hash and Judge before any API call.
        atomic_json(seed_dir / "source.json", {"source_run": str(source_dir), "source_config": saved,
                                               "source_details_sha256": file_hash(seed), "imported_records": len(rows),
                                               "remaining_error_reasons": dict(Counter(failures.values()))})
    snapshot = {**saved, "project_root": str(root), "output_dir": str(output), "input_jsonl": str(source_input),
                "run_id": run_id, "statistic_seed_jsonl": str(seed), "judge_json_retries": judge_json_retries}
    for key, value in [("batch_size", batch_size), ("judge_concurrency", judge_concurrency),
                       ("max_output_tokens", max_output_tokens), ("judge_thinking_mode", judge_thinking_mode)]:
        if value is not None:
            snapshot[key] = value
    environment = resume_environment(snapshot)
    environment["RESUME"] = "0"
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", help="Existing statistic RUN_ID or absolute run directory")
    parser.add_argument("--input-jsonl", type=Path, help="Current location of the unchanged original statistic input")
    parser.add_argument("--run-id", help="New run ID; generated automatically if omitted")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--judge-concurrency", type=int)
    parser.add_argument("--judge-json-retries", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--judge-thinking-mode")
    args = parser.parse_args()
    root = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()
    output = rooted(root, os.environ.get("OUTPUT_DIR", "outputs"))
    try:
        environment = prepare_retry(root, output, **vars(args))
    except (ValueError, OSError) as exc:
        print(f"Retry aborted: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"new_run_id": environment["RUN_ID"], "seed": environment["STATISTIC_SEED_JSONL"],
                      "message": "Import successful labels; call Judge only for remaining selected samples"}), file=sys.stderr)
    report = strict_json(Path(environment["STATISTIC_SEED_JSONL"]).with_name("source.json").read_text())
    print(json.dumps({"imported_records": report["imported_records"], "remaining_error_reasons": report["remaining_error_reasons"]}, ensure_ascii=False), file=sys.stderr)
    os.chdir(root)
    os.execvpe(sys.executable, [sys.executable, "-m", "legal_application"], environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
