from __future__ import annotations

import fcntl
import json
import runpy
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from legal_application.config import Config
from legal_application.storage import implementation_hash, read_jsonl
from test_statistic import handler_for, run, source

HELPER = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/resume_statistic.py"))
prepare = HELPER["prepare_resume"]
environment = HELPER["resume_environment"]


def test_increase_to_200_preserves_saved_results_and_original_settings(config_factory):
    records = [source("saved"), source("retry")]
    config = config_factory(records, mode="statistic", batch_size=50, judge_concurrency=50,
                            max_output_tokens=65536, api_max_retries=0)
    normal = handler_for(records)

    def broken(request):
        doc = json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"]
        return httpx.Response(503) if doc == "retry" else normal(request)

    run(config, broken)
    before = json.loads((config.run_dir / "config.json").read_text())
    details = (config.run_dir / "classification_details.jsonl").read_bytes()
    input_bytes = config.input_jsonl.read_bytes()
    updated = prepare(config.project_root, config.output_dir, config.run_id, batch_size=200, judge_concurrency=200)
    assert {k for k in before if before[k] != updated[k]} == {"batch_size", "judge_concurrency"}
    assert updated["implementation_sha256"] == before["implementation_sha256"] == implementation_hash()
    assert updated["max_output_tokens"] == 65536
    assert (config.run_dir / "classification_details.jsonl").read_bytes() == details
    backups = list(config.run_dir.glob("config.before-resume-*.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == before
    audit = read_jsonl(config.run_dir / "resume_config_changes.jsonl")
    assert [row["state"] for row in audit] == ["prepared", "applied"]
    assert audit[1]["changes"]["judge_concurrency"] == {"old": 50, "new": 200}
    called = []

    def recovered(request):
        called.append(json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"])
        return normal(request)

    summary = run(replace(config, resume=True, batch_size=200, judge_concurrency=200), recovered)
    assert called == ["retry"]
    assert summary["counts"]["skipped"] == 1 and summary["counts"]["processed"] == 2
    assert config.input_jsonl.read_bytes() == input_bytes


def test_restore_environment_from_snapshot_does_not_inherit_wrong_limits(config_factory, monkeypatch):
    config = config_factory([source()], mode="statistic", max_output_tokens=65536, limit=1)
    run(config, handler_for([source()]))
    snapshot = prepare(config.project_root, config.output_dir, config.run_id, batch_size=200, judge_concurrency=200)
    monkeypatch.setenv("LIMIT", "999")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "10")
    monkeypatch.setenv("JUDGE_MODEL", "wrong-model")
    restored = environment(snapshot)
    assert restored["LIMIT"] == "1" and restored["MAX_OUTPUT_TOKENS"] == "65536"
    assert restored["JUDGE_MODEL"] == config.judge_model and restored["RESUME"] == "1"
    assert restored["BATCH_SIZE"] == restored["JUDGE_CONCURRENCY"] == "200"
    for key, value in restored.items():
        monkeypatch.setenv(key, value)
    active = Config.from_env()
    assert active.run_id == config.run_id and active.input_jsonl == config.input_jsonl
    assert active.mode == "statistic" and active.limit == 1 and active.max_output_tokens == 65536


@pytest.mark.parametrize("kind", ["active", "input_changed", "code_changed", "invalid_value"])
def test_unsafe_resume_preparation_does_not_change_snapshot(config_factory, kind):
    raw = source()
    config = config_factory([raw], mode="statistic")
    run(config, handler_for([raw]))
    if kind == "input_changed":
        with config.input_jsonl.open("ab") as output:
            output.write(b"\n")
    if kind == "code_changed":
        value = json.loads((config.run_dir / "config.json").read_text())
        value["implementation_sha256"] = "old-other-code"
        (config.run_dir / "config.json").write_text(json.dumps(value))
    original = (config.run_dir / "config.json").read_bytes()
    with (config.run_dir / ".run.lock").open("a") as lock:
        if kind == "active":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError):
            prepare(config.project_root, config.output_dir, config.run_id,
                    batch_size=0 if kind == "invalid_value" else 200, judge_concurrency=200)
    assert (config.run_dir / "config.json").read_bytes() == original
    assert not list(config.run_dir.glob("config.before-resume-*.json"))
