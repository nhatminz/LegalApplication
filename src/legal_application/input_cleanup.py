"""Remove malformed input JSONL lines, preserving a durable recovery journal."""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .data import strict_json
from .storage import atomic_json, file_hash, now

log = logging.getLogger(__name__)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_journal(path: Path, additions: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".removed-journal-", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            for source_path in [path, additions] if path.exists() else [additions]:
                with source_path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def clean_input(path: Path, run_dir: Path, record_config: Callable[[str], None]) -> dict:
    """Check all lines, commit configuration for the cleaned hash, then remove errors.

    Only UTF-8/strict JSON errors are removed. Valid JSON with the wrong application
    schema is left intact for read_dataset to reject. The callback verifies resume
    BEFORE changing the input or its removal journal. Samples are indexed afterwards.
    """
    path = path.resolve()
    report_path = run_dir / "input_cleanup.json"
    journal_path = run_dir / "removed_input_lines.jsonl"
    lock_path = path.with_name(path.name + ".cleanup.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another process is cleaning input: {path}") from None
        kept_path = rejected_path = None
        try:
            before_digest, after_digest = hashlib.sha256(), hashlib.sha256()
            count = kept_count = removed_count = 0
            cleanup_id, timestamp = uuid4().hex, now()
            with path.open("rb") as source, tempfile.NamedTemporaryFile(
                mode="wb", prefix="." + path.name + ".clean-", dir=path.parent, delete=False
            ) as kept, tempfile.NamedTemporaryFile(
                mode="wb", prefix=".removed-input-", dir=run_dir, delete=False
            ) as rejected:
                kept_path, rejected_path = Path(kept.name), Path(rejected.name)
                os.fchmod(kept.fileno(), stat.S_IMODE(os.fstat(source.fileno()).st_mode))
                for index, line in enumerate(source):
                    count += 1
                    before_digest.update(line)
                    decoded = None
                    try:
                        decoded = line.decode("utf-8")
                        strict_json(decoded)
                    except (ValueError, UnicodeError) as exc:
                        removed_count += 1
                        entry = {"cleanup_id": cleanup_id, "timestamp": timestamp, "input_path": str(path),
                                 "original_input_index": index, "original_input_line": index + 1,
                                 "error_type": type(exc).__name__, "error": str(exc),
                                 "json_error_line": getattr(exc, "lineno", None),
                                 "json_error_column": getattr(exc, "colno", None),
                                 "raw_line": decoded, "raw_line_base64": base64.b64encode(line).decode("ascii"),
                                 "raw_line_sha256": hashlib.sha256(line).hexdigest(), "byte_length": len(line)}
                        rejected.write((json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n").encode())
                    else:
                        kept_count += 1
                        kept.write(line)
                        after_digest.update(line)
                for output in [kept, rejected]:
                    output.flush()
                    os.fsync(output.fileno())
            before, after = before_digest.hexdigest(), after_digest.hexdigest()
            if file_hash(path) != before:
                raise ValueError("Input changed during cleanup scan; no lines removed")
            # A changed valid record/config/code rejects resume here without modifying input.
            # New runs persist the CLEANED hash before the first input rewrite, so a crash
            # between config commit and cleanup commit can safely resume the same operation.
            record_config(after)
            report = strict_json(report_path.read_text()) if report_path.exists() else {
                "input_path": str(path), "log_file": str(journal_path),
                "initial_input_sha256": before, "transactions": []}
            if report.get("input_path") != str(path):
                raise ValueError("Input cleanup report refers to a different input path")
            transactions = report["transactions"]
            pending = transactions[-1] if transactions and transactions[-1]["status"] == "prepared" else None
            if pending and pending["input_sha256_after"] == before:
                # Input replacement committed before a crash, but final report did not.
                pending["status"] = "applied"
                pending = None
            if pending and (pending["input_sha256_before"], pending["input_sha256_after"]) != (before, after):
                raise ValueError("Input does not match the pending cleanup transaction")

            def save_report():
                report.update(input_sha256_after=after, cleaned_line_count=kept_count,
                              removed_count=sum(t["removed_count"] for t in transactions if t["status"] == "applied"),
                              updated_at=now())
                report["original_line_count"] = kept_count + report["removed_count"]
                atomic_json(report_path, report)
                _fsync_directory(run_dir)

            if removed_count:
                if pending:
                    transaction = pending
                else:
                    # Persist every removed byte BEFORE replacing the input. If this fails,
                    # the original input survives unchanged and no model request is made.
                    _commit_journal(journal_path, rejected_path)
                    transaction = {"cleanup_id": cleanup_id, "status": "prepared", "timestamp": timestamp,
                                   "input_sha256_before": before, "input_sha256_after": after,
                                   "original_line_count": count, "cleaned_line_count": kept_count,
                                   "removed_count": removed_count}
                    transactions.append(transaction)
                    save_report()
                if file_hash(path) != before:
                    raise ValueError("Input changed before cleanup commit; no lines removed")
                os.replace(kept_path, path)
                _fsync_directory(path.parent)
                transaction["status"] = "applied"
                log.warning("Removed malformed input JSON lines count=%d input=%s journal=%s", removed_count, path, journal_path)
            else:
                journal_path.touch(exist_ok=True)
            save_report()
            return report
        finally:
            for temporary in [kept_path, rejected_path]:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

