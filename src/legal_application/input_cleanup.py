"""Remove unusable input JSONL records, preserving a durable recovery journal."""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import logging
import os
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable
from uuid import uuid4

from tqdm import tqdm

from .data import parse_sample, strict_json
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

    UTF-8/JSON errors, invalid application records and duplicate doc_names are
    removed. No missing facts, issues or laws are invented. The callback verifies resume
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
            removal_counts = Counter()
            seen_docs = {}
            cleanup_id, timestamp = uuid4().hex, now()
            with path.open("rb") as source, tempfile.NamedTemporaryFile(
                mode="wb", prefix="." + path.name + ".clean-", dir=path.parent, delete=False
            ) as kept, tempfile.NamedTemporaryFile(
                mode="wb", prefix=".removed-input-", dir=run_dir, delete=False
            ) as rejected, tqdm(total=os.fstat(source.fileno()).st_size, desc="Scan input", unit="B",
                                unit_scale=True, dynamic_ncols=True, disable=None) as progress:
                kept_path, rejected_path = Path(kept.name), Path(rejected.name)
                os.fchmod(kept.fileno(), stat.S_IMODE(os.fstat(source.fileno()).st_mode))
                for index, line in enumerate(source):
                    count += 1
                    before_digest.update(line)
                    decoded = None
                    raw = None
                    validation_stage = "json"
                    try:
                        decoded = line.decode("utf-8")
                        raw = strict_json(decoded)
                        validation_stage = "schema"
                        sample = parse_sample(raw, kept_count)
                        if sample.doc_name in seen_docs:
                            validation_stage = "duplicate_doc_name"
                            raise ValueError(f"duplicate doc_name {sample.doc_name!r}; first valid occurrence at original line {seen_docs[sample.doc_name]}")
                        seen_docs[sample.doc_name] = index + 1
                    except (ValueError, UnicodeError) as exc:
                        removed_count += 1
                        removal_counts[validation_stage] += 1
                        entry = {"cleanup_id": cleanup_id, "timestamp": timestamp, "input_path": str(path),
                                 "original_input_index": index, "original_input_line": index + 1,
                                 "validation_stage": validation_stage,
                                 "doc_name": raw.get("doc_name") if isinstance(raw, dict) and isinstance(raw.get("doc_name"), str) else None,
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
                    progress.set_postfix(kept=kept_count, invalid=removed_count, refresh=False)
                    progress.update(len(line))
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
                reasons = Counter()
                for transaction in transactions:
                    if transaction["status"] == "applied":
                        reasons.update(transaction.get("removal_counts", {"json": transaction["removed_count"]}))
                report.update(input_sha256_after=after, cleaned_line_count=kept_count,
                              removed_count=sum(t["removed_count"] for t in transactions if t["status"] == "applied"),
                              removal_counts=dict(reasons),
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
                                   "removed_count": removed_count, "removal_counts": dict(removal_counts)}
                    transactions.append(transaction)
                    save_report()
                if file_hash(path) != before:
                    raise ValueError("Input changed before cleanup commit; no lines removed")
                os.replace(kept_path, path)
                _fsync_directory(path.parent)
                transaction["status"] = "applied"
                log.warning("Removed invalid input records count=%d reasons=%s input=%s journal=%s", removed_count, dict(removal_counts), path, journal_path)
                with rejected_path.open(encoding="utf-8") as rejected:
                    for rejected_line in rejected:
                        entry = strict_json(rejected_line)
                        log.warning("Input record removed original_line=%d doc_name=%s stage=%s error=%s",
                                    entry["original_input_line"], entry["doc_name"], entry["validation_stage"], entry["error"])
            else:
                journal_path.touch(exist_ok=True)
            save_report()
            return report
        finally:
            for temporary in [kept_path, rejected_path]:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

