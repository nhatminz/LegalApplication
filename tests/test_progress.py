from __future__ import annotations

import asyncio
import io
import json
from dataclasses import replace

import httpx
import pytest
from tqdm import tqdm

from legal_application import data, input_cleanup, pipeline
from legal_application.pipeline import Pipeline
from conftest import envelope, make_labels, make_record


@pytest.fixture
def bars(monkeypatch):
    created = []

    def create(*args, **kwargs):
        # Render real tqdm bars into memory, even though pytest has no terminal.
        kwargs.update(file=io.StringIO(), disable=False, mininterval=0)
        bar = tqdm(*args, **kwargs)
        created.append(bar)
        return bar

    for module in [data, input_cleanup, pipeline]:
        monkeypatch.setattr(module, "tqdm", create)
    return created


def run(config, transports):
    return asyncio.run(Pipeline(config, generator_transport=transports[0], judge_transport=transports[1]).run())


def stages(bars):
    return [bar for bar in bars if bar.desc.startswith(("Generation", "Import generations", "Judge"))]


def test_global_progress_updates_across_batches_and_resume(config_factory, mock_transports, bars):
    config = config_factory([make_record(f"case-{i}") for i in range(5)], batch_size=2)
    run(config, mock_transports)
    gen, judge = stages(bars)
    assert (gen.n, gen.total, judge.n, judge.total) == (5, 5, 5, 5)
    assert "batch 3" in gen.desc and "batch 3" in judge.desc
    assert "ok=5" in gen.postfix and "ok=5" in judge.postfix
    assert all(bar.disable for bar in bars)  # context managers close every bar
    run(replace(config, resume=True), mock_transports)
    gen, judge = stages(bars)[-2:]
    assert gen.initial == judge.initial == 5
    assert gen.n == judge.n == 5  # never count persisted records twice
    assert "skipped=5" in gen.postfix and "skipped=5" in judge.postfix
    assert len(mock_transports[2]["generator"]) == len(mock_transports[2]["judge"]) == 5
    assert "\r" not in (config.run_dir / "run.log").read_text()


def test_cleanup_progress_counts_bytes_and_validate_only_has_no_inference_bars(config_factory, mock_transports, bars):
    config = config_factory(mode="validate-only")
    with config.input_jsonl.open("ab") as output:
        output.write(b"bad json\n")
    original_size = config.input_jsonl.stat().st_size
    summary = run(config, mock_transports)
    scan, validation = bars
    assert scan.desc == "Scan input" and scan.n == scan.total == original_size
    assert "invalid=1" in scan.postfix
    assert validation.desc == "Validate input" and validation.n == validation.total == config.input_jsonl.stat().st_size
    assert summary["counts"]["input_removed"] == 1
    assert not stages(bars)


def test_generation_failures_finish_processed_progress_with_judge_blocked(config_factory, mock_transports, bars):
    config = config_factory([make_record(f"case-{i}") for i in range(3)], api_max_retries=0)
    failed = httpx.MockTransport(lambda request: httpx.Response(503))
    summary = run(config, (failed, mock_transports[1]))
    gen, judge = stages(bars)
    assert gen.n == judge.n == 3
    assert "failed=3" in gen.postfix and "ok=0" in gen.postfix
    assert "blocked=3" in judge.postfix and "ok=0" in judge.postfix
    assert summary["counts"]["failed"] == 3
    assert not mock_transports[2]["judge"]


def test_judge_errors_and_review_are_shown_separately(config_factory, mock_transports, bars):
    config = config_factory([make_record("review"), make_record("bad")], api_max_retries=0)

    def judge(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        if payload["doc_name"] == "bad":
            return httpx.Response(401)
        return httpx.Response(200, json=envelope(json.dumps(make_labels(payload, review=True))))

    summary = run(config, (mock_transports[0], httpx.MockTransport(judge)))
    gen, judge_bar = stages(bars)
    assert gen.n == judge_bar.n == 2
    assert "failed=1" in judge_bar.postfix and "review=1" in judge_bar.postfix
    assert summary["counts"]["failed"] == summary["counts"]["needs_review"] == 1


def test_judge_only_import_progress_and_closed_bars_on_fatal_error(config_factory, mock_transports, bars, monkeypatch):
    config = config_factory()
    run(config, mock_transports)
    judge_config = replace(config, mode="judge-only", run_id="judge-progress", generation_jsonl=config.run_dir / "generations.jsonl")
    run(judge_config, mock_transports)
    imported, judged = stages(bars)[-2:]
    assert imported.desc.startswith("Import generations") and imported.n == judged.n == 1
    assert len(mock_transports[2]["generator"]) == 1
    original = pipeline.RunStore.append

    def fail(store, name, record):
        if name == "generations.jsonl":
            raise OSError("disk full")
        return original(store, name, record)

    monkeypatch.setattr(pipeline.RunStore, "append", fail)
    with pytest.raises(OSError, match="disk full"):
        run(replace(config, run_id="failed-progress"), mock_transports)
    gen, judge_bar = stages(bars)[-2:]
    assert gen.n == judge_bar.n == 0
    assert gen.disable and judge_bar.disable
