from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from legal_application import input_cleanup
from legal_application.data import DatasetError, read_dataset
from legal_application.input_cleanup import clean_input
from legal_application.pipeline import Pipeline
from legal_application.storage import read_jsonl
from conftest import make_record


def invoke(config, transports):
    return asyncio.run(Pipeline(config, generator_transport=transports[0], judge_transport=transports[1]).run())


def test_middle_errors_removed_without_losing_following_samples_or_limit(config_factory, mock_transports):
    rows = [make_record(f"case-{i}") for i in range(3)]
    config = config_factory(rows, limit=2)
    valid = config.input_jsonl.read_bytes().splitlines(keepends=True)
    invalid = [b'{"broken":\n', b'not-json\n']
    original_lines = [valid[0], invalid[0], valid[1], invalid[1], valid[2]]
    config.input_jsonl.write_bytes(b"".join(original_lines))
    summary = invoke(config, mock_transports)
    assert config.input_jsonl.read_bytes() == b"".join(valid)
    assert summary["counts"]["input"] == 5
    assert summary["counts"]["input_valid"] == 3
    assert summary["counts"]["input_removed"] == 2
    assert summary["counts"]["selected"] == summary["counts"]["completed"] == 2
    generations = read_jsonl(config.run_dir / "generations.jsonl")
    assert [(r["doc_name"], r["input_index"]) for r in generations] == [("case-0", 0), ("case-1", 1)]
    entries = read_jsonl(config.run_dir / "removed_input_lines.jsonl")
    assert [e["original_input_line"] for e in entries] == [2, 4]
    assert [base64.b64decode(e["raw_line_base64"]) for e in entries] == invalid
    assert all(e["input_path"] == str(config.input_jsonl) for e in entries)
    # Deleted bytes plus surviving lines reconstruct the original input exactly.
    removed = {e["original_input_line"]: base64.b64decode(e["raw_line_base64"]) for e in entries}
    survivors = iter(valid)
    reconstructed = b"".join(removed[i] if i in removed else next(survivors) for i in range(1, 6))
    assert reconstructed == b"".join(original_lines)
    report = summary["input_cleanup"]
    assert report["transactions"][0]["status"] == "applied"
    assert report["initial_input_sha256"] == hashlib.sha256(reconstructed).hexdigest()
    assert report["input_sha256_after"] == hashlib.sha256(b"".join(valid)).hexdigest()
    saved_config = json.loads((config.run_dir / "config.json").read_text())
    assert saved_config["input_sha256"] == report["input_sha256_after"]
    resumed = invoke(replace(config, resume=True), mock_transports)
    assert resumed["counts"]["skipped"] == 2
    assert resumed["counts"]["input_removed"] == 2
    assert len(mock_transports[2]["generator"]) == 2
    # A new run reads the permanently cleaned file and encounters no old errors.
    fresh = invoke(replace(config, run_id="next-run", limit=None), mock_transports)
    assert fresh["counts"]["input_removed"] == 0
    assert fresh["counts"]["completed"] == 3


@pytest.mark.parametrize("bad", [b"\n", b"\xff\xfe\n", b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{"x":Infinity}\n'])
def test_invalid_encoding_blank_and_strict_json_logged_exactly(tmp_path, bad):
    source = tmp_path / "input.jsonl"
    good = (json.dumps(make_record(), ensure_ascii=False) + "\n").encode()
    source.write_bytes(bad + good)
    report = clean_input(source, tmp_path, lambda digest: None)
    assert source.read_bytes() == good
    assert report["removed_count"] == 1
    entry = read_jsonl(tmp_path / "removed_input_lines.jsonl")[0]
    assert base64.b64decode(entry["raw_line_base64"]) == bad
    assert entry["raw_line_sha256"] == hashlib.sha256(bad).hexdigest()
    assert entry["byte_length"] == len(bad)


def test_valid_json_with_wrong_schema_removed_and_run_continues(config_factory, mock_transports):
    config = config_factory()
    bad_schema = json.dumps({"doc_name": "missing-fields", "data_status": "full"}) + "\n"
    with config.input_jsonl.open("a") as output:
        output.write(bad_schema)
    summary = invoke(config, mock_transports)
    assert summary["counts"]["completed"] == 1
    assert summary["counts"]["input_removed"] == 1
    entries = read_jsonl(config.run_dir / "removed_input_lines.jsonl")
    assert entries[0]["validation_stage"] == "schema"
    assert entries[0]["doc_name"] == "missing-fields"
    assert entries[0]["raw_line"] == bad_schema
    assert len(read_dataset(config.input_jsonl, None).samples) == 1


@pytest.mark.parametrize("issues", [[], None, "", {}, [" "]])
def test_invalid_issues_between_valid_records_are_logged_deleted_and_not_generated(config_factory, mock_transports, issues):
    bad = make_record("invalid-issues")
    bad["issues"] = issues
    config = config_factory([make_record("before"), bad, make_record("after")], limit=2)
    summary = invoke(config, mock_transports)
    assert summary["counts"]["input"] == 3
    assert summary["counts"]["input_removed"] == 1
    assert summary["counts"]["completed"] == summary["counts"]["selected"] == 2
    generations = read_jsonl(config.run_dir / "generations.jsonl")
    assert [(row["doc_name"], row["input_index"]) for row in generations] == [("before", 0), ("after", 1)]
    entry = read_jsonl(config.run_dir / "removed_input_lines.jsonl")[0]
    assert entry["original_input_line"] == 2 and entry["doc_name"] == "invalid-issues"
    assert entry["validation_stage"] == "schema" and "issues" in entry["error"]
    assert json.loads(entry["raw_line"])["issues"] == issues
    assert summary["input_cleanup"]["removal_counts"] == {"schema": 1}
    resumed = invoke(replace(config, resume=True), mock_transports)
    assert resumed["counts"]["skipped"] == 2
    assert len(mock_transports[2]["generator"]) == 2


def test_reported_line_578_empty_issues_reproduction(config_factory, mock_transports):
    rows = [make_record(f"case-{i}") for i in range(579)]
    rows[577]["issues"] = []
    config = config_factory(rows, mode="validate-only")
    summary = invoke(config, mock_transports)
    assert summary["status"] == "validated"
    assert summary["counts"]["input"] == 579
    assert summary["counts"]["input_valid"] == summary["counts"]["selected"] == 578
    entry = read_jsonl(config.run_dir / "removed_input_lines.jsonl")[0]
    assert entry["original_input_line"] == 578
    assert entry["error"] == "issues must be a nonempty list"
    cleaned = read_dataset(config.input_jsonl, None)
    assert cleaned.samples[-1].doc_name == "case-578"
    assert cleaned.samples[-1].input_index == 577
    assert not mock_transports[2]["generator"] and not mock_transports[2]["judge"]


def test_duplicate_doc_names_keep_first_valid_record_and_log_later_one(config_factory, mock_transports):
    first = make_record("repeated")
    invalid_first = make_record("repeated")
    invalid_first["issues"] = []
    duplicate = make_record("repeated")
    duplicate["benchmark_seed"]["facts_clean"] = "A different later duplicate record"
    config = config_factory([invalid_first, first, duplicate, make_record("after")])
    summary = invoke(config, mock_transports)
    assert summary["counts"]["completed"] == 2
    entries = read_jsonl(config.run_dir / "removed_input_lines.jsonl")
    assert [e["validation_stage"] for e in entries] == ["schema", "duplicate_doc_name"]
    assert "original line 2" in entries[1]["error"]
    assert [s.doc_name for s in read_dataset(config.input_jsonl, None).samples] == ["repeated", "after"]


def test_resume_removes_new_schema_errors_without_regenerating(config_factory, mock_transports):
    config = config_factory([make_record("kept")])
    invoke(config, mock_transports)
    clean_bytes = config.input_jsonl.read_bytes()
    bad = make_record("bad")
    bad["issues"] = []
    config.input_jsonl.write_bytes((json.dumps(bad) + "\n").encode() + clean_bytes)
    summary = invoke(replace(config, resume=True), mock_transports)
    assert config.input_jsonl.read_bytes() == clean_bytes
    assert summary["counts"]["skipped"] == summary["counts"]["input_removed"] == 1
    assert len(mock_transports[2]["generator"]) == 1


def test_only_invalid_schema_records_produce_empty_input_report(config_factory, mock_transports):
    bad = make_record()
    bad["issues"] = []
    config = config_factory([bad])
    with pytest.raises(DatasetError):
        invoke(config, mock_transports)
    assert config.input_jsonl.read_bytes() == b""
    entry = read_jsonl(config.run_dir / "removed_input_lines.jsonl")[0]
    assert entry["validation_stage"] == "schema"
    assert not mock_transports[2]["generator"] and not mock_transports[2]["judge"]


def test_clean_input_unchanged_including_final_newline_and_permissions(tmp_path):
    source = tmp_path / "input.jsonl"
    before = json.dumps(make_record()).encode()  # deliberately no final newline
    source.write_bytes(before)
    source.chmod(0o640)
    stamp = source.stat().st_mtime_ns
    report = clean_input(source, tmp_path, lambda digest: None)
    assert source.read_bytes() == before
    assert source.stat().st_mtime_ns == stamp
    assert source.stat().st_mode & 0o777 == 0o640
    assert report["removed_count"] == 0


def test_validate_only_cleans_input_without_api(config_factory, mock_transports):
    config = config_factory(mode="validate-only")
    with config.input_jsonl.open("ab") as output:
        output.write(b'{"broken')
    summary = invoke(config, mock_transports)
    assert summary["status"] == "validated"
    assert summary["counts"]["input_removed"] == 1
    assert len(read_dataset(config.input_jsonl, None).samples) == 1
    assert not mock_transports[2]["generator"] and not mock_transports[2]["judge"]


def test_all_lines_bad_recorded_then_no_samples_fails(config_factory, mock_transports):
    config = config_factory()
    config.input_jsonl.write_bytes(b"invalid\n\xff\n")
    with pytest.raises(DatasetError):
        invoke(config, mock_transports)
    assert config.input_jsonl.read_bytes() == b""
    assert len(read_jsonl(config.run_dir / "removed_input_lines.jsonl")) == 2
    summary = json.loads((config.run_dir / "summary.json").read_text())
    assert summary["counts"]["input"] == summary["counts"]["input_removed"] == 2
    assert summary["counts"]["input_valid"] == 0
    assert not mock_transports[2]["generator"] and not mock_transports[2]["judge"]


def test_resume_removes_new_json_errors_but_never_regenerates(config_factory, mock_transports):
    config = config_factory([make_record("first"), make_record("second")])
    invoke(config, mock_transports)
    original = config.input_jsonl.read_bytes().splitlines(keepends=True)
    config.input_jsonl.write_bytes(b"broken at start\n" + original[0] + b"broken middle\n" + original[1])
    summary = invoke(replace(config, resume=True), mock_transports)
    assert config.input_jsonl.read_bytes() == b"".join(original)
    assert summary["counts"]["skipped"] == 2
    assert summary["counts"]["input_removed"] == 2
    assert len(mock_transports[2]["generator"]) == len(mock_transports[2]["judge"]) == 2


def test_rejected_resume_does_not_delete_or_log_input_lines(config_factory, mock_transports):
    config = config_factory()
    invoke(config, mock_transports)
    with config.input_jsonl.open("ab") as output:
        output.write(b"invalid\n")
        output.write((json.dumps(make_record("new-valid-record")) + "\n").encode())
    before = config.input_jsonl.read_bytes()
    with pytest.raises(ValueError, match="configuration/input/code differs"):
        invoke(replace(config, resume=True), mock_transports)
    assert config.input_jsonl.read_bytes() == before
    assert not read_jsonl(config.run_dir / "removed_input_lines.jsonl")


def test_journal_failure_preserves_original_input(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    original = b"bad\n" + (json.dumps(make_record()) + "\n").encode()
    source.write_bytes(original)

    def fail(*args):
        raise OSError("Cannot write removal log")

    monkeypatch.setattr(input_cleanup, "_commit_journal", fail)
    with pytest.raises(OSError, match="Cannot write removal log"):
        clean_input(source, tmp_path, lambda digest: None)
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".input.jsonl.clean-*"))


@pytest.mark.parametrize("crash_after_replace", [False, True])
def test_cleanup_transaction_recovers_without_duplicate_removal_logs(tmp_path, monkeypatch, crash_after_replace):
    source = tmp_path / "input.jsonl"
    good = (json.dumps(make_record()) + "\n").encode()
    source.write_bytes(b"bad\n" + good)
    replace_original = input_cleanup.os.replace
    json_original = input_cleanup.atomic_json

    def replacement(src, dst):
        if not crash_after_replace and Path(dst) == source:
            raise OSError("simulated input replace failure")
        return replace_original(src, dst)

    def save(path, value):
        if crash_after_replace and path.name == "input_cleanup.json" and value["transactions"][-1]["status"] == "applied":
            raise OSError("simulated final report failure")
        return json_original(path, value)

    monkeypatch.setattr(input_cleanup.os, "replace", replacement)
    monkeypatch.setattr(input_cleanup, "atomic_json", save)
    with pytest.raises(OSError):
        clean_input(source, tmp_path, lambda digest: None)
    assert len(read_jsonl(tmp_path / "removed_input_lines.jsonl")) == 1
    monkeypatch.setattr(input_cleanup.os, "replace", replace_original)
    monkeypatch.setattr(input_cleanup, "atomic_json", json_original)
    report = clean_input(source, tmp_path, lambda digest: None)
    assert source.read_bytes() == good
    assert report["removed_count"] == 1
    assert report["transactions"][0]["status"] == "applied"
    assert len(read_jsonl(tmp_path / "removed_input_lines.jsonl")) == 1
