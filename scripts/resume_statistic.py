"""Resume a statistic run with explicitly changed scheduling only.

Kept outside src/legal_application so installing this helper does not change the
pipeline implementation hash of a run that is already in progress.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from dataclasses import fields
from pathlib import Path
from uuid import uuid4

from legal_application.config import Config, rooted
from legal_application.storage import append_jsonl, atomic_json, file_hash, implementation_hash, now
from legal_application.data import strict_json
from legal_application.thinking import resolve_thinking


def prepare_resume(root: Path, output_dir: Path, run_id: str, *, batch_size: int | None = None,
                   judge_concurrency: int | None = None) -> dict:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Provide the existing RUN_ID; do not generate a new one")
    for name, value in [("BATCH_SIZE", batch_size), ("JUDGE_CONCURRENCY", judge_concurrency)]:
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    root, output_dir = root.resolve(), output_dir.resolve()
    run_dir = output_dir / run_id
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Existing statistic config not found: {config_path}")
    with (run_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Run is still active. Stop it with Ctrl-C and wait for exit before resuming") from None
        saved = strict_json(config_path.read_text())
        if saved.get("pipeline_kind") != "data_completeness" or saved.get("mode") != "statistic":
            raise ValueError("This helper supports statistic runs only")
        if saved.get("run_id") != run_id or Path(saved["project_root"]).resolve() != root or Path(saved["output_dir"]).resolve() != output_dir:
            raise ValueError("PROJECT_ROOT, OUTPUT_DIR and RUN_ID must point to the original run")
        updated = dict(saved)
        if batch_size is not None:
            updated["batch_size"] = batch_size
        if judge_concurrency is not None:
            updated["judge_concurrency"] = judge_concurrency
        allowed = {field.name for field in fields(Config)}
        values = {key: value for key, value in updated.items() if key in allowed}
        for key in ["project_root", "input_jsonl", "output_dir"]:
            values[key] = Path(values[key])
        config = Config(**values)
        config.validate()
        expected = config.snapshot()
        for key in list(expected):
            if key.startswith("generator_") or key in {"generation_jsonl", "data_status_used"}:
                expected.pop(key)
        expected.update(pipeline_kind="data_completeness", artifact_version=1,
                        input_sha256=file_hash(config.input_jsonl), implementation_sha256=implementation_hash(),
                        judge_thinking=resolve_thinking(config.judge_thinking_mode))
        if expected != updated:
            raise ValueError("Input/code/config differs from the saved run. Only batch size and Judge concurrency may change")
        changes = {key: {"old": saved[key], "new": updated[key]}
                   for key in ["batch_size", "judge_concurrency"] if saved[key] != updated[key]}
        if changes:
            change_id = uuid4().hex
            backup = run_dir / f"config.before-resume-{change_id}.json"
            atomic_json(backup, saved)
            event = {"change_id": change_id, "timestamp": now(), "changes": changes, "backup": backup.name}
            append_jsonl(run_dir / "resume_config_changes.jsonl", {**event, "state": "prepared"})
            atomic_json(config_path, updated)
            append_jsonl(run_dir / "resume_config_changes.jsonl", {**event, "state": "applied"})
        return updated


def resume_environment(snapshot: dict) -> dict:
    environment = os.environ.copy()
    aliases = {"timeout_seconds": "API_TIMEOUT_SECONDS", "retry_backoff": "RETRY_BACKOFF_SECONDS"}
    allowed = {field.name for field in fields(Config)}
    for key, value in snapshot.items():
        if key not in allowed:
            continue
        if value is None:
            value = "all" if key == "limit" else ""
        environment[aliases.get(key, key.upper())] = str(value)
    # These fields do not participate in statistic requests/snapshots.
    environment.update(RESUME="1", MODE="statistic", DATA_STATUS_USED="all", GENERATION_JSONL="",
                       GENERATOR_THINKING_MODE="none", GENERATOR_CONCURRENCY="20")
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume statistic; change only batch size and Judge concurrency")
    parser.add_argument("run_id", nargs="?", default=os.environ.get("RUN_ID"))
    parser.add_argument("--batch-size", type=int, default=os.environ.get("BATCH_SIZE"))
    parser.add_argument("--judge-concurrency", type=int, default=os.environ.get("JUDGE_CONCURRENCY"))
    args = parser.parse_args()
    try:
        root = Path(os.environ.get("PROJECT_ROOT", "/workspace/storage-shared/nlp/minhpn19/LegalApplication")).resolve()
        output = rooted(root, os.environ.get("OUTPUT_DIR", "outputs"))
        snapshot = prepare_resume(root, output, args.run_id, batch_size=args.batch_size,
                                  judge_concurrency=args.judge_concurrency)
    except (ValueError, OSError) as exc:
        print(f"Resume aborted: {exc}", file=sys.stderr)
        return 2
    print(f"Resuming {snapshot['run_id']}: batch={snapshot['batch_size']}, Judge concurrency={snapshot['judge_concurrency']}; "
          "other settings restored from config.json", file=sys.stderr)
    os.chdir(root)
    os.execvpe(sys.executable, [sys.executable, "-m", "legal_application"], resume_environment(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
