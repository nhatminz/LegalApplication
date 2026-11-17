from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from conftest import make_record
from legal_application.config import Config
from legal_application.data import DatasetError, read_dataset
from legal_application.pipeline import Pipeline
from legal_application.storage import read_jsonl


def run(config, transports):
    return asyncio.run(Pipeline(config, generator_transport=transports[0], judge_transport=transports[1]).run())


def mixed(n=6):
    rows = [make_record(f"case-{i}") for i in range(n)]
    for i, row in enumerate(rows):
        row["data_status"] = "partial" if i % 2 == 0 else "full"
    return rows


def test_filtering_precedes_limit_and_preserves_physical_index(config_factory):
    rows = [make_record(f"case-{i}") for i in range(270)]
    for i, row in enumerate(rows):
        row["data_status"] = "partial" if i < 150 else "full"
    config = config_factory(rows)
    dataset = read_dataset(config.input_jsonl, 100, "full_only")
    assert len(dataset.samples) == 100
    assert dataset.samples[0].doc_name == "case-150" and dataset.samples[0].input_index == 150
    assert dataset.samples[-1].doc_name == "case-249" and dataset.samples[-1].input_index == 249
    assert dataset.status_counts == {"full": 120, "partial": 150}
    assert dataset.eligible_count == 120 and dataset.filtered_out == 150


@pytest.mark.parametrize("limit,count", [(None, 1005), (1001, 1001), (0, 0)])
def test_no_hidden_filter_cap(config_factory, limit, count):
    config = config_factory([make_record(f"case-{i}") for i in range(1005)])
    assert len(read_dataset(config.input_jsonl, limit, "full_only").samples) == count


@pytest.mark.parametrize("selection,expected", [("all", [0, 1, 2]), ("full_only", [1, 3, 5])])
def test_pipeline_filter_mapping_summary_resume_and_judge_only(config_factory, mock_transports, selection, expected):
    config = config_factory(mixed(), data_status_used=selection, limit=3, batch_size=2)
    before = config.input_jsonl.read_bytes()
    summary = run(config, mock_transports)
    assert config.input_jsonl.read_bytes() == before
    assert [r["input_index"] for r in read_jsonl(config.run_dir / "generations.jsonl")] == expected
    groups = summary["data_status_counts"]
    assert groups["input"] == {"full": 3, "partial": 3}
    assert groups["filtered_out"] == (3 if selection == "full_only" else 0)
    assert groups["selected"] == groups["completed"]
    assert summary["counts"]["completed"] == 3
    if selection == "full_only":
        assert groups["completed"] == {"full": 3, "partial": 0}
    assert json.loads((config.run_dir / "config.json").read_text())["data_status_used"] == selection
    resumed = run(replace(config, resume=True), mock_transports)
    assert resumed["counts"]["skipped"] == 3
    assert len(mock_transports[2]["generator"]) == 3
    judge = replace(config, mode="judge-only", run_id="filtered-judge",
                    generation_jsonl=config.run_dir / "generations.jsonl")
    assert run(judge, mock_transports)["counts"]["completed"] == 3
    assert len(mock_transports[2]["generator"]) == 3
    other = "all" if selection == "full_only" else "full_only"
    with pytest.raises(ValueError, match="configuration/input/code differs"):
        run(replace(config, resume=True, data_status_used=other), mock_transports)


def test_validate_only_uses_same_filter_and_reports_zero_completed(config_factory, mock_transports):
    config = config_factory(mixed(), data_status_used="full_only", mode="validate-only", limit=2)
    summary = run(config, mock_transports)
    assert summary["counts"]["selected"] == 2
    assert summary["data_status_counts"]["completed"] == {"full": 0, "partial": 0}
    assert not mock_transports[2]["generator"] and not mock_transports[2]["judge"]


@pytest.mark.parametrize("selection", ["all", "full_only"])
@pytest.mark.parametrize("bad", [None, "FULL", "unknown", "", 1, [], {}])
def test_missing_or_invalid_labels_fail_without_guessing_or_deletion(config_factory, mock_transports, selection, bad):
    records = mixed()
    if bad is None:
        records[-1].pop("data_status")
    else:
        records[-1]["data_status"] = bad
    config = config_factory(records, data_status_used=selection, limit=1)
    before = config.input_jsonl.read_bytes()
    with pytest.raises(ValueError, match="data_status must be full or partial"):
        run(config, mock_transports)
    assert config.input_jsonl.read_bytes() == before
    assert not mock_transports[2]["generator"] and not mock_transports[2]["judge"]
    with pytest.raises(DatasetError):
        read_dataset(config.input_jsonl, 1, selection)


def test_invalid_filter_config_and_environment(config_factory, monkeypatch):
    with pytest.raises(ValueError, match="DATA_STATUS_USED"):
        config_factory(data_status_used="partial_only").validate()
    monkeypatch.setenv("DATA_STATUS_USED", "full_only")
    assert Config.from_env().data_status_used == "full_only"


def test_concurrent_filtered_results_do_not_shift_doc_names(config_factory):
    import httpx
    from conftest import envelope, generator_response, make_labels

    records = mixed(8)
    for i, row in enumerate(records):
        row["issues"] = [f"Issue for case {i}"]
    config = config_factory(records, data_status_used="full_only", batch_size=4, limit=3, generator_concurrency=3)

    async def generator(request):
        body = json.loads(request.content)
        issue = body["messages"][1]["content"].split("LEGAL ISSUES\n\n1. ")[1].split("\n\nAPPLICABLE LAWS")[0]
        await asyncio.sleep((8 - int(issue.rsplit(" ", 1)[-1])) * 0.001)
        return httpx.Response(200, json=envelope(generator_response([issue])))

    def judge(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert payload["issues"][0] in payload["evaluated_model_response"]
        return httpx.Response(200, json=envelope(json.dumps(make_labels(payload))))

    summary = run(config, (httpx.MockTransport(generator), httpx.MockTransport(judge)))
    assert summary["counts"]["completed"] == 3
    rows = read_jsonl(config.run_dir / "evaluations.jsonl")
    assert {r["input_index"] for r in rows} == {1, 3, 5}
    assert all(r["doc_name"] == f"case-{r['input_index']}" for r in rows)
