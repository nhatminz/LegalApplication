from __future__ import annotations

import asyncio
import json
import os
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from conftest import envelope
from legal_application.config import Config
from legal_application.data import canonical_hash, parse_sample
from legal_application.filter_full import export_full, manifest_path
from legal_application.pipeline import Pipeline
from legal_application.storage import read_jsonl
from test_statistic import handler_for, labels, run, source


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("id_field", ["document_id", "doc_name"])
def test_filter_preserves_exact_source_bytes_and_application_uses_manifest(tmp_path, config_factory, mock_transports, id_field):
    rows = [source("partial"), source("full-first"), source("failed"), source("full-last")]
    for i, row in enumerate(rows):
        row["document_id"] = 1000 + i
    lines = [("  " + json.dumps(row, ensure_ascii=False) + " \r\n").encode() for row in rows]
    lines[-1] = lines[-1].rstrip(b"\r\n")
    raw = tmp_path / "original.jsonl"
    raw.write_bytes(lines[0] + b'{"doc_name":"bad","issues":[]}\n' + b"".join(lines[1:]))
    original = raw.read_bytes()
    details = tmp_path / "classification_details.jsonl"
    write_jsonl(details, [{id_field: row[id_field], "data_status": "partial" if i == 0 else "full",
                          "sample_sha256": canonical_hash(row), "classification_status": "completed"}
                         for i, row in reversed(list(enumerate(rows))) if i != 2])
    output = tmp_path / "input_application_clean_v1.jsonl"
    summary = export_full(raw, [details], output)
    assert output.read_bytes() == lines[1] + lines[3]
    assert read_jsonl(output) == [rows[1], rows[3]]
    assert summary["sample_count"] == 2 and summary["source_invalid"] == 1
    assert raw.read_bytes() == original
    base = config_factory()
    config = replace(base, input_jsonl=output, run_id="full-application")
    pipeline = lambda cfg: asyncio.run(Pipeline(cfg, generator_transport=mock_transports[0], judge_transport=mock_transports[1]).run())
    result = pipeline(config)
    assert result["counts"]["completed"] == 2
    assert result["data_status_counts"]["completed"] == {"full": 2, "partial": 0}
    assert [r["doc_name"] for r in read_jsonl(config.run_dir / "generations.jsonl")] == ["full-first", "full-last"]
    assert all("data_status" not in row for row in read_jsonl(output))
    assert pipeline(replace(config, resume=True))["counts"]["skipped"] == 2
    manifest_path(output).unlink()
    with pytest.raises(ValueError, match="data_status must"):
        pipeline(replace(config, run_id="missing-manifest"))


@pytest.mark.parametrize("problem", ["hash", "conflict", "unmatched", "duplicate", "no_full"])
def test_filter_refuses_ambiguous_or_changed_records_without_replacing_output(tmp_path, problem):
    raw, details, output = [tmp_path / name for name in ["source.jsonl", "details.jsonl", "filtered.jsonl"]]
    row = source()
    write_jsonl(raw, [row, row] if problem == "duplicate" else [row])
    label = {"doc_name": row["doc_name"], "data_status": "full", "sample_sha256": canonical_hash(row)}
    if problem == "hash":
        label["sample_sha256"] = "wrong-hash"
    if problem == "unmatched":
        label["doc_name"] = "not-in-source"
    if problem == "no_full":
        label["data_status"] = "partial"
    write_jsonl(details, [label, {**label, "data_status": "partial"}] if problem == "conflict" else [label])
    output.write_bytes(b"existing output must survive")
    before = raw.read_bytes()
    with pytest.raises(ValueError):
        export_full(raw, [details], output)
    assert output.read_bytes() == b"existing output must survive"
    assert raw.read_bytes() == before
    assert not manifest_path(output).exists()


def test_tampered_full_export_is_rejected_before_cleanup(config_factory, tmp_path, mock_transports):
    row = source()
    raw, details, output = [tmp_path / name for name in ["source.jsonl", "details.jsonl", "filtered.jsonl"]]
    write_jsonl(raw, [row])
    write_jsonl(details, [{"doc_name": row["doc_name"], "data_status": "full"}])
    export_full(raw, [details], output)
    with output.open("ab") as handle:
        handle.write(b"broken json\n")
    before = output.read_bytes()
    config = replace(config_factory(), input_jsonl=output)
    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        asyncio.run(Pipeline(config, generator_transport=mock_transports[0], judge_transport=mock_transports[1]).run())
    assert output.read_bytes() == before and not mock_transports[2]["generator"]


@pytest.mark.parametrize("change", ["whitespace", "fence", "hallucinated_amount", "malformed_source"])
def test_classification_format_repairs_keep_strict_evidence_checks(config_factory, change):
    row = source()
    row["benchmark_seed"]["facts_clean"] = row["benchmark_seed"]["facts_clean"].replace("văn bản", "văn\n\t bản")
    config = config_factory([row], mode="statistic", judge_json_retries=0)
    value = labels(row)
    if change == "whitespace":
        for analysis in value["issues_analysis"]:
            analysis["decisive_facts"][0]["facts_clean_evidence"] = [row["benchmark_seed"]["facts_clean"].replace("\n\t ", " ")]
    if change == "hallucinated_amount":
        value["issues_analysis"][0]["decisive_facts"][0]["facts_clean_evidence"] = ["Còn thiếu 999.982.000 đồng."]
    if change == "malformed_source":
        value["issues_analysis"][0]["decisive_facts"][0]["reference_evidence"][0]["source"] = []
    content = json.dumps(value, ensure_ascii=False)
    if change == "fence":
        content = "```json\n" + content + "\n```"
    result = run(config, lambda request: httpx.Response(200, json=envelope(content)))
    if change in {"whitespace", "fence"}:
        assert result["counts"]["processed"] == 1
        detail = read_jsonl(config.run_dir / "classification_details.jsonl")[0]
        assert detail["raw_judge_response"] == content
        assert detail["metadata"]["normalizations"]
        resumed = run(replace(config, resume=True), lambda request: pytest.fail("Normalized saved result must be reusable"))
        assert resumed["counts"]["skipped"] == 1
    else:
        assert result["counts"]["failed"] == 1
        error = read_jsonl(config.run_dir / "errors.jsonl")[0]
        assert "last_error=" in error["error"]
        assert error["metadata"]["last_validation_error"]
        assert error["metadata"]["invalid_responses"][0]["content_preview"]


def test_repair_receives_previous_parseable_response_and_specific_field_error(config_factory):
    row = source()
    config = config_factory([row], mode="statistic")
    calls = []

    def judge(request):
        body = json.loads(request.content)
        calls.append(body)
        value = labels(row)
        if len(calls) == 1:
            value["issues_analysis"][0]["decisive_facts"][0]["reference_evidence"][0]["quote"] = "not in source"
        return httpx.Response(200, json=envelope(json.dumps(value, ensure_ascii=False)))

    assert run(config, judge)["counts"]["processed"] == 1
    assert calls[1]["messages"][2]["role"] == "assistant"
    assert "reference_evidence[0].quote" in calls[1]["messages"][-1]["content"]


def test_retry_new_code_imports_successes_and_calls_only_failed_samples(config_factory, tmp_path, monkeypatch):
    rows = [source("full"), source("partial", partial=True), source("failed")]
    config = config_factory(rows, mode="statistic", api_max_retries=0)
    normal = handler_for(rows, partial_docs={"partial"})
    run(config, lambda request: httpx.Response(503) if json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"] == "failed" else normal(request))
    old_details = (config.run_dir / "classification_details.jsonl").read_bytes()
    old_config = json.loads((config.run_dir / "config.json").read_text())
    old_config["implementation_sha256"] = "old-code-version"
    # Reproduce an older snapshot with no seed setting and with a different project path.
    old_config.pop("statistic_seed_jsonl")
    old_config.pop("statistic_seed_sha256")
    old_config["project_root"] = "/previous/project/location"
    (config.run_dir / "config.json").write_text(json.dumps(old_config))
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    prepare = runpy.run_path(str(scripts / "retry_statistic.py"))["prepare_retry"]
    environment = prepare(config.project_root, config.output_dir, config.run_id, input_jsonl=config.input_jsonl,
                          run_id="recovered", batch_size=200, judge_concurrency=200, judge_json_retries=4)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    recovered = Config.from_env()
    called = []

    def judge(request):
        called.append(json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"])
        return normal(request)

    result = run(recovered, judge)
    assert called == ["failed"]
    assert result["counts"]["processed"] == 3 and result["counts"]["skipped"] == 2
    assert result["counts"]["full"] == 2 and result["counts"]["partial"] == 1
    assert (config.run_dir / "classification_details.jsonl").read_bytes() == old_details
    resumed = run(replace(recovered, resume=True), lambda request: pytest.fail("Every record already succeeded"))
    assert resumed["counts"]["skipped"] == 3


def test_seed_with_changed_source_identity_is_rejected_before_inference(config_factory):
    row = source()
    config = config_factory([row], mode="statistic")
    run(config, handler_for([row]))
    seed = config.run_dir / "classification_details.jsonl"
    bad = read_jsonl(seed)
    bad[0]["sample_sha256"] = "changed-content"
    write_jsonl(seed, bad)
    with pytest.raises(ValueError, match="identity/model mismatch"):
        run(replace(config, run_id="reject-bad-seed", statistic_seed_jsonl=seed),
            lambda request: pytest.fail("Invalid seed must be rejected before API requests"))


def test_filter_shell_entrypoint_exports_without_model_requests(tmp_path):
    root = Path(__file__).resolve().parents[1]
    raw, details, output = [tmp_path / name for name in ["source.jsonl", "details.jsonl", "input_application_clean_v1.jsonl"]]
    row = source()
    write_jsonl(raw, [row])
    write_jsonl(details, [{"doc_name": row["doc_name"], "data_status": "full"}])
    environment = {**os.environ, "PROJECT_ROOT": str(root), "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", str(root / "scripts/filter_full.sh"), "--input", str(raw),
                             "--classifications", str(details), "--output", str(output)],
                            env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["full"] == 1
    assert output.read_bytes() == raw.read_bytes() and manifest_path(output).is_file()


@pytest.mark.parametrize("kind", ["active", "input_changed"])
def test_recovery_rejects_active_run_or_changed_input(config_factory, monkeypatch, kind):
    import fcntl

    row = source()
    config = config_factory([row], mode="statistic")
    run(config, handler_for([row]))
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    prepare = runpy.run_path(str(scripts / "retry_statistic.py"))["prepare_retry"]
    if kind == "input_changed":
        with config.input_jsonl.open("ab") as handle:
            handle.write(b"\n")
    with (config.run_dir / ".run.lock").open("a") as lock:
        if kind == "active":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="is active|checksum differs"):
            prepare(config.project_root, config.output_dir, config.run_id, run_id="unsafe-recovery")
    assert not (config.output_dir / ".statistic-seeds" / "unsafe-recovery").exists()
