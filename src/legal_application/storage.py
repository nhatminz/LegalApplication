from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from collections.abc import Iterable

from .config import Config
from .data import strict_json

log = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.suffix in {".py", ".txt", ".json"}):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp." + uuid4().hex)
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def append_jsonl(path: Path, value: dict) -> None:
    # Called synchronously on the event loop: no interleaved lines; fsync before next stage.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_jsonl(path: Path, records: Iterable[dict]) -> None:
    temp = path.with_name(path.name + ".tmp." + uuid4().hex)
    try:
        with temp.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict]:
    records = []
    with path.open("rb") as handle:
        offset = 0
        for line_number, line in enumerate(handle, 1):
            try:
                record = strict_json(line.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
            except (ValueError, UnicodeError) as exc:
                # A crash can tear only the final append. Preserve a backup before recovery.
                final_unterminated_line = not line.endswith(b"\n") and not handle.read(1)
                if repair_tail and final_unterminated_line:
                    backup = path.with_name(path.name + ".torn-tail." + uuid4().hex + ".bak")
                    backup.write_bytes(line)
                    with path.open("r+b") as output:
                        output.truncate(offset)
                        output.flush()
                        os.fsync(output.fileno())
                    log.warning("Recovered torn JSONL tail path=%s line=%d backup=%s", path, line_number, backup)
                    break
                raise ValueError(f"{path}:{line_number}: {exc}") from None
            if not line.endswith(b"\n"):
                if repair_tail:
                    # Complete JSON committed before newline: retain it and terminate the line.
                    with path.open("ab") as output:
                        output.write(b"\n")
                        output.flush()
                        os.fsync(output.fileno())
                    log.warning("Restored final JSONL newline path=%s line=%d", path, line_number)
                # Read-only external files may legitimately lack a final newline.
            records.append(record)
            offset += len(line)
            if not line.endswith(b"\n"):
                break
    return records


class RunStore:
    def __init__(self, config: Config):
        self.config = config
        self.path = config.run_dir
        self.lock = None

    def __enter__(self):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.resume:
            if not self.path.is_dir():
                raise ValueError("RESUME=1 requires an existing run directory")
        else:
            try:
                self.path.mkdir()
            except FileExistsError:
                raise ValueError(f"Run already exists: {self.path}; use RESUME=1 or a new RUN_ID") from None
        self.lock = (self.path / ".run.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError(f"Another process holds run lock: {self.path}") from None
        return self

    def __exit__(self, *args):
        if self.lock:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def configure(self, snapshot: dict, artifact_names=None) -> None:
        path = self.path / "config.json"
        if self.config.resume:
            if not path.exists() or strict_json(path.read_text()) != snapshot:
                raise ValueError("Resume configuration/input/code differs from config.json; use a new RUN_ID")
        else:
            atomic_json(path, snapshot)
        for name in artifact_names if artifact_names is not None else ["generations.jsonl", "evaluations.jsonl", "errors.jsonl"]:
            (self.path / name).touch(exist_ok=True)

    def load(self, name: str) -> list[dict]:
        return read_jsonl(self.path / name, repair_tail=self.config.resume)

    def append(self, name: str, record: dict) -> None:
        append_jsonl(self.path / name, record)

    def summary(self, value: dict) -> None:
        atomic_json(self.path / "summary.json", value)

