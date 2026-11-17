from __future__ import annotations

import fcntl
import json
import runpy
import shutil
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from legal_application.config import Config
import legal_application.config as config_module
from legal_application.storage import implementation_hash, read_jsonl
from test_statistic import handler_for, run, source

HELPER = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/resume_statistic.py"))
prepare = HELPER["prepare_resume"]
environment = HELPER["resume_environment"]


def previous_default_hash(monkeypatch):
    """Hash the previous package through the pipeline's own hashing function."""
    config_path = Path(config_module.__file__)
    current_source = config_path.read_bytes()
    assert current_source.count(HELPER["DEFAULT_PROJECT_ROOT"].encode()) == 1
    previous_source = current_source.replace(HELPER["DEFAULT_PROJECT_ROOT"].encode(),
                                             HELPER["LEGACY_PROJECT_ROOT"].encode(), 1)
    read_bytes = Path.read_bytes
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_bytes", lambda path: previous_source if path == config_path else read_bytes(path))
        return implementation_hash()


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


@pytest.mark.parametrize("internal_input", [False, True])
@pytest.mark.parametrize("previous_default", [False, True])
def test_relocate_project_and_resume_only_pending_samples(config_factory, tmp_path, monkeypatch,
                                                         internal_input, previous_default):
    records = [source("saved"), source("retry")]
    original = config_factory(records, mode="statistic", api_max_retries=0, batch_size=50, judge_concurrency=50)
    old_root, new_root = tmp_path / "project-before", tmp_path / "project-after"
    old_root.mkdir()
    input_path = original.input_jsonl
    if internal_input:
        input_path = old_root / "input.jsonl"
        shutil.copyfile(original.input_jsonl, input_path)
    config = replace(original, project_root=old_root, output_dir=old_root / "outputs", input_jsonl=input_path)
    normal = handler_for(records)

    def broken(request):
        doc = json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"]
        return httpx.Response(503) if doc == "retry" else normal(request)

    run(config, broken)
    before = json.loads((config.run_dir / "config.json").read_text())
    if previous_default:
        before["implementation_sha256"] = previous_default_hash(monkeypatch)
        assert HELPER["path_default_compatibility_hashes"]() == {before["implementation_sha256"]}
        (config.run_dir / "config.json").write_text(json.dumps(before))
    artifacts = {name: (config.run_dir / name).read_bytes()
                 for name in ["classification_details.jsonl", "errors.jsonl"]}
    input_bytes = input_path.read_bytes()
    shutil.move(old_root, new_root)
    assert not old_root.exists()
    new_output = new_root / "outputs"
    run_dir = new_output / config.run_id
    new_input = new_root / "input.jsonl" if internal_input else input_path
    updated = prepare(new_root, new_output, config.run_id, relocate=True, batch_size=200, judge_concurrency=200)
    assert updated["project_root"] == str(new_root)
    assert updated["output_dir"] == str(new_output)
    assert updated["input_jsonl"] == str(new_input)
    assert updated["input_sha256"] == before["input_sha256"]
    assert updated["implementation_sha256"] == implementation_hash()
    assert updated["batch_size"] == updated["judge_concurrency"] == 200
    for name, content in artifacts.items():
        assert (run_dir / name).read_bytes() == content
    backups = list(run_dir.glob("config.before-resume-*.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == before
    audit = read_jsonl(run_dir / "resume_config_changes.jsonl")
    assert [row["state"] for row in audit] == ["prepared", "applied"]
    assert all(row["relocate"] and row["input_checksum_verified"] for row in audit)
    restored = environment(updated)
    assert restored["PROJECT_ROOT"] == str(new_root) and restored["OUTPUT_DIR"] == str(new_output)
    assert restored["INPUT_JSONL"] == str(new_input) and restored["RESUME"] == "1"
    called = []

    def recovered(request):
        called.append(json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"])
        return normal(request)

    summary = run(replace(config, project_root=new_root, output_dir=new_output, input_jsonl=new_input,
                          resume=True, batch_size=200, judge_concurrency=200), recovered)
    assert called == ["retry"]
    assert summary["counts"]["skipped"] == 1 and summary["counts"]["processed"] == 2
    assert summary["classified_output_complete"]
    assert new_input.read_bytes() == input_bytes


@pytest.mark.parametrize("changed_bytes", [False, True])
def test_relocated_input_override_requires_identical_checksum(config_factory, tmp_path, changed_bytes):
    raw = source()
    config = config_factory([raw], mode="statistic")
    run(config, handler_for([raw]))
    moved_input = tmp_path / "moved-dataset.jsonl"
    moved_input.write_bytes(config.input_jsonl.read_bytes() + (b"\n" if changed_bytes else b""))
    config.input_jsonl.unlink()
    config_path = config.run_dir / "config.json"
    before = config_path.read_bytes()
    if changed_bytes:
        with pytest.raises(ValueError, match="Input/config differs"):
            prepare(config.project_root, config.output_dir, config.run_id, relocate=True, input_jsonl=moved_input)
        assert config_path.read_bytes() == before
        assert not list(config.run_dir.glob("config.before-resume-*.json"))
    else:
        snapshot = prepare(config.project_root, config.output_dir, config.run_id, relocate=True, input_jsonl=moved_input)
        assert snapshot["input_jsonl"] == str(moved_input)
        resumed = replace(config, input_jsonl=moved_input, resume=True)
        summary = run(resumed, lambda request: pytest.fail("Saved record must be skipped after input move"))
        assert summary["counts"]["skipped"] == 1


@pytest.mark.parametrize("kind", ["no_flag", "active", "code_changed"])
def test_unsafe_relocation_preserves_config_and_results(config_factory, tmp_path, kind):
    raw = source()
    original = config_factory([raw], mode="statistic")
    old_root, new_root = tmp_path / "before", tmp_path / "after"
    old_root.mkdir()
    config = replace(original, project_root=old_root, output_dir=old_root / "outputs")
    run(config, handler_for([raw]))
    config_path = config.run_dir / "config.json"
    if kind == "code_changed":
        value = json.loads(config_path.read_text())
        value["implementation_sha256"] = "unrelated-code-change"
        config_path.write_text(json.dumps(value))
    before = config_path.read_bytes()
    details = (config.run_dir / "classification_details.jsonl").read_bytes()
    shutil.move(old_root, new_root)
    run_dir = new_root / "outputs" / config.run_id
    message = {"no_flag": "Use --relocate", "active": "Run is still active", "code_changed": "Pipeline code differs"}[kind]
    with (run_dir / ".run.lock").open("a") as lock:
        if kind == "active":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match=message):
            prepare(new_root, new_root / "outputs", config.run_id, relocate=kind != "no_flag")
    assert (run_dir / "config.json").read_bytes() == before
    assert (run_dir / "classification_details.jsonl").read_bytes() == details
    assert not list(run_dir.glob("config.before-resume-*.json"))


def test_input_override_requires_explicit_relocation(config_factory):
    raw = source()
    config = config_factory([raw], mode="statistic")
    run(config, handler_for([raw]))
    with pytest.raises(ValueError, match="--input-jsonl requires --relocate"):
        prepare(config.project_root, config.output_dir, config.run_id, input_jsonl=config.input_jsonl)
