"""Resume a statistic run after scheduling changes or an explicit project relocation.

Kept outside src/legal_application so installing this helper does not change the
pipeline implementation hash of a run that is already in progress.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import sys
from dataclasses import fields
from pathlib import Path
from uuid import uuid4

from legal_application.config import Config, rooted
from legal_application.storage import append_jsonl, atomic_json, now
from legal_application.data import strict_json
from legal_application.statistic import statistic_snapshot

LEGACY_PROJECT_ROOT = "/workspace/storage-shared/nlp/minhpn19/LegalApplication"
DEFAULT_PROJECT_ROOT = "/workspace/storage-shared/nlp/users/minhpn19/LegalApplication"


def path_default_compatibility_hashes() -> set[str]:
    """Recognize ONLY the previous default-path literal, not arbitrary code changes."""
    import legal_application.config as config_module

    package = Path(config_module.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(p for p in package.rglob("*") if p.suffix in {".py", ".txt", ".json"}):
        data = path.read_bytes()
        if path == package / "config.py":
            old, new = LEGACY_PROJECT_ROOT.encode(), DEFAULT_PROJECT_ROOT.encode()
            if data.count(new) != 1:
                return set()
            data = data.replace(new, old, 1)
        digest.update(str(path.relative_to(package)).encode())
        digest.update(data)
    return {digest.hexdigest()}


def prepare_resume(root: Path, output_dir: Path, run_id: str, *, batch_size: int | None = None,
                   judge_concurrency: int | None = None, relocate: bool = False,
                   input_jsonl: Path | None = None) -> dict:
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
        if saved.get("run_id") != run_id:
            raise ValueError("RUN_ID does not match config.json")
        old_root = Path(saved["project_root"]).resolve()
        if not relocate and (old_root != root or Path(saved["output_dir"]).resolve() != output_dir):
            raise ValueError("Project/run path changed. Use --relocate with the NEW PROJECT_ROOT and OUTPUT_DIR")
        if input_jsonl is not None and not relocate:
            raise ValueError("--input-jsonl requires --relocate; input checksum must remain unchanged")
        updated = dict(saved)
        if relocate:
            updated.update(project_root=str(root), output_dir=str(output_dir))
            old_input = Path(saved["input_jsonl"])
            if input_jsonl is not None:
                updated["input_jsonl"] = str(rooted(root, str(input_jsonl)))
            elif old_input.is_relative_to(old_root):
                updated["input_jsonl"] = str((root / old_input.relative_to(old_root)).resolve())
            if saved.get("statistic_seed_jsonl") and Path(saved["statistic_seed_jsonl"]).is_relative_to(old_root):
                updated["statistic_seed_jsonl"] = str((root / Path(saved["statistic_seed_jsonl"]).relative_to(old_root)).resolve())
            # Inputs outside the old project (e.g. LegalDataset) keep their original path.
        if batch_size is not None:
            updated["batch_size"] = batch_size
        if judge_concurrency is not None:
            updated["judge_concurrency"] = judge_concurrency
        allowed = {field.name for field in fields(Config)}
        values = {key: value for key, value in updated.items() if key in allowed}
        for key in ["project_root", "input_jsonl", "output_dir", "statistic_seed_jsonl"]:
            if values.get(key) is not None:
                values[key] = Path(values[key])
        config = Config(**values)
        config.validate()
        expected = statistic_snapshot(config)
        if updated.get("implementation_sha256") != expected["implementation_sha256"]:
            if relocate and updated.get("implementation_sha256") in path_default_compatibility_hashes():
                updated["implementation_sha256"] = expected["implementation_sha256"]
            else:
                raise ValueError("Pipeline code differs from the run. Relocation can migrate only the verified default-path change; use scripts/retry_statistic.sh to import successes into a new run with updated code")
        if expected != updated:
            raise ValueError("Input/config differs from the saved run. Input bytes, Judge and evaluation settings must remain unchanged")
        changes = {key: {"old": saved[key], "new": updated[key]}
                   for key in ["project_root", "output_dir", "input_jsonl", "statistic_seed_jsonl", "implementation_sha256", "batch_size", "judge_concurrency"]
                   if saved.get(key) != updated.get(key)}
        if changes:
            change_id = uuid4().hex
            backup = run_dir / f"config.before-resume-{change_id}.json"
            atomic_json(backup, saved)
            event = {"change_id": change_id, "timestamp": now(), "changes": changes, "backup": backup.name,
                     "relocate": relocate, "input_checksum_verified": True}
            append_jsonl(run_dir / "resume_config_changes.jsonl", {**event, "state": "prepared"})
            atomic_json(config_path, updated)
            append_jsonl(run_dir / "resume_config_changes.jsonl", {**event, "state": "applied"})
        return updated


def snapshot_environment(snapshot: dict) -> dict:
    environment = os.environ.copy()
    aliases = {"timeout_seconds": "API_TIMEOUT_SECONDS", "retry_backoff": "RETRY_BACKOFF_SECONDS"}
    allowed = {field.name for field in fields(Config)}
    for key, value in snapshot.items():
        if key not in allowed:
            continue
        if value is None:
            value = "all" if key == "limit" else ""
        environment[aliases.get(key, key.upper())] = str(value)
    return environment


def resume_environment(snapshot: dict) -> dict:
    environment = snapshot_environment(snapshot)
    # These fields do not participate in statistic requests/snapshots.
    environment.pop("DATA_STATUS_USED", None)
    environment.update(RESUME="1", MODE="statistic", GENERATION_JSONL="",
                       GENERATOR_THINKING_MODE="none", GENERATOR_CONCURRENCY="20")
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume statistic with checked scheduling changes or project relocation")
    parser.add_argument("run_id", nargs="?", default=os.environ.get("RUN_ID"))
    parser.add_argument("--batch-size", type=int, default=os.environ.get("BATCH_SIZE"))
    parser.add_argument("--judge-concurrency", type=int, default=os.environ.get("JUDGE_CONCURRENCY"))
    parser.add_argument("--relocate", action="store_true", help="Update saved project/output paths to the current location")
    parser.add_argument("--input-jsonl", type=Path, help="New input location, allowed only with --relocate and identical file checksum")
    args = parser.parse_args()
    try:
        root = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()
        output = rooted(root, os.environ.get("OUTPUT_DIR", "outputs"))
        snapshot = prepare_resume(root, output, args.run_id, batch_size=args.batch_size,
                                  judge_concurrency=args.judge_concurrency, relocate=args.relocate, input_jsonl=args.input_jsonl)
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
