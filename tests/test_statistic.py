from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from conftest import envelope, make_record
from legal_application.completeness import COMPLETENESS_SCHEMA, completeness_messages, validate_classification
from legal_application.data import DatasetError, parse_sample
from legal_application.scoring import LabelError
from legal_application.statistic import StatisticPipeline
from legal_application.pipeline import Pipeline
from legal_application.storage import read_jsonl


def source(doc="case", *, partial=False, n=2):
    raw = make_record(doc, n=n)
    raw.pop("data_status")
    raw["benchmark_seed"] = {
        "facts_clean": "A yêu cầu B thanh toán 474.982.000 đồng; B không đồng ý." if partial else "B đã ký văn bản xác nhận còn thiếu A 474.982.000 đồng.",
        "facts": "B có giấy ký nhận nợ 474.982.000 đồng do chính B ký.",
        "application": "Giấy ký nhận nợ do B ký chứng minh nghĩa vụ thanh toán 474.982.000 đồng.",
        "conclusion": "Buộc B thanh toán cho A 474.982.000 đồng.",
    }
    return raw


def labels(raw, *, missing_issue=None, inferred=False):
    seed = raw["benchmark_seed"]
    rows = []
    for index, issue in enumerate(raw["issues"], 1):
        missing = index == missing_issue
        rows.append({"issue_index": index, "original_issue": issue,
                     "explanation": "Đối chiếu các tình tiết cần cho riêng issue này.",
                     "decisive_facts": [{"fact": "Có văn bản nhận nợ do B ký.", "decisive": True,
                                         "availability": "missing" if missing else "reasonably_inferred" if inferred else "present",
                                         "reference_evidence": [{"source": "application", "quote": seed["application"]}],
                                         "facts_clean_evidence": [] if missing else [seed["facts_clean"]],
                                         "raw_facts_evidence": [seed["facts"]],
                                         "explanation": "Chứng cứ quyết định nghĩa vụ; kiểm tra tương đương ý nghĩa, không so khớp câu chữ."}]})
    return {"doc_name": raw["doc_name"], "classification_status": "completed",
            "data_status": "partial" if missing_issue else "full", "reason": "Có/không thiếu chứng cứ quyết định theo từng issue.",
            "issues_analysis": rows}


def run(config, handler):
    return asyncio.run(StatisticPipeline(config, judge_transport=httpx.MockTransport(handler)).run())


def handler_for(records, *, partial_docs=()):
    records = {raw["doc_name"]: raw for raw in records}

    def handle(request):
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
        assert body["response_format"]["json_schema"]["schema"] == COMPLETENESS_SCHEMA
        payload = json.loads(body["messages"][1]["content"])
        raw = records[payload["doc_name"]]
        assert payload["benchmark_seed"] == raw["benchmark_seed"]
        assert payload["issues"] == raw["issues"]
        output = labels(raw, missing_issue=2 if raw["doc_name"] in partial_docs else None)
        return httpx.Response(200, json=envelope(json.dumps(output, ensure_ascii=False)))

    return handle


def test_full_paraphrase_and_partial_missing_decisive_evidence(config_factory):
    full, partial = source("paraphrase"), source("1801778", partial=True)
    assert full["benchmark_seed"]["facts_clean"] != full["benchmark_seed"]["application"]
    full["extra_nested"] = {"none": None, "items": [True, 3.14, "giữ nguyên\nVăn bản"], "flags": {"x": 1}}
    config = config_factory([full, partial], mode="statistic")
    before = config.input_jsonl.read_bytes()
    summary = run(config, handler_for([full, partial], partial_docs={"1801778"}))
    assert summary["counts"]["processed"] == 2
    assert summary["counts"]["full"] == summary["counts"]["partial"] == 1
    assert summary["ratios"] == {"full": 0.5, "partial": 0.5, "failed": 0.0}
    assert summary["classified_output_complete"] is True
    output = read_jsonl(config.run_dir / "classified_input.jsonl")
    assert output == [{**full, "data_status": "full"}, {**partial, "data_status": "partial"}]
    assert config.input_jsonl.read_bytes() == before
    assert not config.input_jsonl.with_name(config.input_jsonl.name + ".cleanup.lock").exists()
    detail = read_jsonl(config.run_dir / "classification_details.jsonl")[1]
    assert detail["issues_missing_facts"] == [{"issue_index": 2, "original_issue": partial["issues"][1]}]
    assert detail["decisive_missing_facts"][0]["reference_evidence"][0]["quote"] in partial["benchmark_seed"]["application"]
    assert not (config.run_dir / "generations.jsonl").exists()


def test_legal_reasoning_omission_and_reasonable_inference_do_not_force_partial():
    raw = source()
    sample = parse_sample(raw, 0)
    result = labels(raw, inferred=True)
    validate_classification(result, sample)
    prompt = completeness_messages(sample)[0]["content"]
    assert "Không coi thiếu diễn giải pháp luật" in prompt
    assert "không yêu cầu exact string matching" in prompt
    assert "Không tự động đánh dấu missing" in prompt
    assert "KHÔNG được đọc nguồn này" in prompt
    # A legal-only issue can need no additional decisive fact, with an explanation.
    result["issues_analysis"][0]["decisive_facts"] = []
    validate_classification(result, sample)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(data_status="full"),
    lambda r: r["issues_analysis"][1]["decisive_facts"][0].update(decisive=False),
    lambda r: r["issues_analysis"][1]["decisive_facts"][0]["reference_evidence"][0].update(quote="invented quote"),
    lambda r: r["issues_analysis"][0]["decisive_facts"][0].update(facts_clean_evidence=["invented facts quote"]),
    lambda r: r["issues_analysis"][0]["decisive_facts"][0].update(raw_facts_evidence=["invented private quote"]),
    lambda r: r["issues_analysis"].reverse(),
    lambda r: r["issues_analysis"].pop(),
    lambda r: r.update(doc_name="wrong"),
    lambda r: r.update(data_status="unknown"),
])
def test_invalid_evidence_labels_and_issue_mapping_rejected(mutate):
    raw = source(partial=True)
    result = labels(raw, missing_issue=2)
    mutate(result)
    with pytest.raises(LabelError):
        validate_classification(result, parse_sample(raw, 0))


def test_invalid_json_and_evidence_retry_with_original_context(config_factory):
    raw = source()
    config = config_factory([raw], mode="statistic", judge_json_retries=2)
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        output = labels(raw)
        if len(calls) == 1:
            content = "INVALID_DRAFT"
        else:
            if len(calls) == 2:
                output["issues_analysis"][0]["decisive_facts"][0]["reference_evidence"][0]["quote"] = "invented"
            content = json.dumps(output)
        return httpx.Response(200, json=envelope(content))

    assert run(config, handle)["counts"]["processed"] == 1
    assert len(calls) == 3
    assert all(json.loads(c["messages"][1]["content"])["benchmark_seed"] == raw["benchmark_seed"] for c in calls)
    assert "INVALID_DRAFT" not in json.dumps(calls[1:])
    assert read_jsonl(config.run_dir / "classification_details.jsonl")[0]["metadata"]["label_attempts"] == 3


def test_api_failure_no_fake_label_partial_file_then_resume(config_factory):
    records = [source("good"), source("fail")]
    config = config_factory(records, mode="statistic", api_max_retries=0)
    regular = handler_for(records)
    called = []

    def broken(request):
        doc = json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"]
        called.append(doc)
        return httpx.Response(503) if doc == "fail" else regular(request)

    summary = run(config, broken)
    assert summary["counts"]["processed"] == summary["counts"]["failed"] == 1
    assert summary["ratios"] == {"full": 1.0, "partial": 0.0, "failed": 0.5}
    assert not summary["classified_output_complete"]
    assert not (config.run_dir / "classified_input.jsonl").exists()
    assert [r["doc_name"] for r in read_jsonl(config.run_dir / "classified_input.partial.jsonl")] == ["good"]
    called.clear()

    def recovered(request):
        called.append(json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"])
        return regular(request)

    summary = run(replace(config, resume=True), recovered)
    assert called == ["fail"]
    assert summary["counts"]["processed"] == 2 and summary["counts"]["failed"] == 0
    assert summary["counts"]["skipped"] == 1
    assert len(read_jsonl(config.run_dir / "classification_details.jsonl")) == 2
    assert not (config.run_dir / "classified_input.partial.jsonl").exists()


def test_unclassifiable_is_error_not_partial(config_factory):
    raw = source()
    config = config_factory([raw], mode="statistic")
    result = labels(raw)
    result.update(classification_status="unclassifiable", data_status=None, reason="Nguồn tham chiếu không đủ rõ.")
    summary = run(config, lambda request: httpx.Response(200, json=envelope(json.dumps(result))))
    assert summary["counts"]["failed"] == 1 and summary["counts"]["partial"] == 0
    assert summary["counts"]["processed"] == 0
    assert summary["ratios"]["full"] is summary["ratios"]["partial"] is None
    assert not read_jsonl(config.run_dir / "classified_input.partial.jsonl")
    assert "could not classify reliably" in read_jsonl(config.run_dir / "errors.jsonl")[0]["error"]


@pytest.mark.parametrize("bad", [b"broken json\n", b'{"doc_name":"no-fields"}\n', b"\xff\n"])
def test_statistic_invalid_input_reports_and_preserves_source(config_factory, bad):
    raw = source()
    config = config_factory([raw], mode="statistic", limit=1)
    with config.input_jsonl.open("ab") as output:
        output.write(bad)
    original, stamp = config.input_jsonl.read_bytes(), config.input_jsonl.stat().st_mtime_ns
    with pytest.raises(DatasetError):
        run(config, lambda request: pytest.fail("No model requests for invalid statistic input"))
    assert config.input_jsonl.read_bytes() == original and config.input_jsonl.stat().st_mtime_ns == stamp
    assert not config.input_jsonl.with_name(config.input_jsonl.name + ".cleanup.lock").exists()
    assert not (config.run_dir / "classified_input.jsonl").exists()
    summary = json.loads((config.run_dir / "summary.json").read_text())
    assert summary["input_validation_errors"][0]["input_line"] == 2
    assert read_jsonl(config.run_dir / "errors.jsonl")[0]["stage"] == "input_validation"


def test_classification_concurrency_and_export_order(config_factory):
    records = [source(f"case-{i}") for i in range(7)]
    config = config_factory(records, mode="statistic", batch_size=4, judge_concurrency=3)
    active = maximum = 0
    regular = handler_for(records)

    async def handler(request):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        doc = json.loads(json.loads(request.content)["messages"][1]["content"])["doc_name"]
        await asyncio.sleep((7 - int(doc.split("-")[-1])) * 0.001)
        active -= 1
        return regular(request)

    summary = run(config, handler)
    assert maximum == 3
    assert summary["counts"]["processed"] == 7
    assert [r["doc_name"] for r in read_jsonl(config.run_dir / "classified_input.jsonl")] == [r["doc_name"] for r in records]
    assert [r["input_index"] for r in read_jsonl(config.run_dir / "classification_details.jsonl")] == list(range(7))
    assert (config.run_dir / "run.log").read_text().count("Classification batch start") == 2


@pytest.mark.parametrize("limit,expected", [(0, 0), (1, 1), (3, 3), (None, 4), (20, 4)])
def test_statistic_limit(config_factory, limit, expected):
    records = [source(f"case-{i}") for i in range(4)]
    config = config_factory(records, mode="statistic", limit=limit)
    summary = run(config, handler_for(records))
    assert summary["counts"]["input"] == 4
    assert summary["counts"]["selected"] == summary["counts"]["processed"] == expected
    assert len(read_jsonl(config.run_dir / "classified_input.jsonl")) == expected


def test_statistic_transport_retries_and_input_change_blocks_resume(config_factory):
    raw = source()
    config = config_factory([raw], mode="statistic")
    count = 0
    regular = handler_for([raw])

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(503) if count == 1 else regular(request)

    run(config, handler)
    assert count == 2
    with config.input_jsonl.open("ab") as output:
        output.write(b"invalid\n")
    before = config.input_jsonl.read_bytes()
    with pytest.raises(ValueError, match="configuration/input/code differs"):
        run(replace(config, resume=True), regular)
    assert config.input_jsonl.read_bytes() == before


def test_source_change_during_inference_never_publishes_complete_output(config_factory):
    raw = source()
    config = config_factory([raw], mode="statistic")
    regular = handler_for([raw])

    def handler(request):
        with config.input_jsonl.open("a") as output:
            output.write("\n")
        return regular(request)

    with pytest.raises(ValueError, match="Input changed during classification"):
        run(config, handler)
    assert not (config.run_dir / "classified_input.jsonl").exists()
    assert (config.run_dir / "classified_input.partial.jsonl").exists()
    assert not json.loads((config.run_dir / "summary.json").read_text())["classified_output_complete"]


def test_classification_export_feeds_application_filters_without_mutating_raw(config_factory, mock_transports):
    records = [source("partial-first", partial=True), source("full-second")]
    config = config_factory(records, mode="statistic")
    original = config.input_jsonl.read_bytes()
    run(config, handler_for(records, partial_docs={"partial-first"}))
    labelled = config.run_dir / "classified_input.jsonl"
    app = replace(config, mode="full", run_id="application-full-only", input_jsonl=labelled, data_status_used="full_only")
    summary = asyncio.run(Pipeline(app, generator_transport=mock_transports[0], judge_transport=mock_transports[1]).run())
    assert summary["counts"]["completed"] == 1
    generation = read_jsonl(app.run_dir / "generations.jsonl")[0]
    assert generation["doc_name"] == "full-second" and generation["input_index"] == 1
    assert summary["data_status_counts"]["input"] == {"full": 1, "partial": 1}
    assert summary["mean_score"] == 10
    assert config.input_jsonl.read_bytes() == original
    assert read_jsonl(labelled) == [{**records[0], "data_status": "partial"}, {**records[1], "data_status": "full"}]


def test_classification_limit_above_1000_has_no_hidden_cap(config_factory):
    records = [source(f"case-{i}", n=1) for i in range(1003)]
    config = config_factory(records, mode="statistic", limit=1001, batch_size=100)
    summary = run(config, handler_for(records))
    assert summary["counts"]["input"] == 1003
    assert summary["counts"]["selected"] == summary["counts"]["processed"] == 1001
    assert len(read_jsonl(config.run_dir / "classified_input.jsonl")) == 1001
