from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from conftest import envelope, make_record
from legal_application.data import DatasetError
from legal_application.pipeline import Pipeline
from legal_application.rubric_builder import RubricPipeline
from legal_application.rubrics import criteria_for, rubric_sample
from legal_application.scoring import LabelError, evaluation_plan, score_rubric_labels
from legal_application.storage import atomic_json, atomic_jsonl, file_hash, read_jsonl
from legal_application.rubric_taxonomy import TAXONOMY

ANSWER = "B ký nhận nợ. Do đó B phải trả 100 đồng."
REASONING = "Việc B xác nhận khoản nợ làm phát sinh nghĩa vụ hoàn trả đủ một trăm đồng."


def raw_record(doc="case-1", category="Civil", n=1):
    raw = make_record(doc, n)
    raw.update(category=category, instance_level="First-instance", code="DS-ST")
    raw["benchmark_seed"]["application"] = ANSWER
    return raw


def settings(config_factory, records=None, **changes):
    config = config_factory(records or [raw_record()], mode="build-rubrics", evaluation_profile="rubric")
    return replace(config, final_application_jsonl=config.project_root / "application.jsonl",
                   final_issue_spotting_jsonl=config.project_root / "spotting.jsonl", **changes)


def model_output(payload, *, group_id=None):
    chosen = group_id or next(iter(payload["taxonomy"]))
    criterion = {"description": "Giải thích nghĩa vụ trả khoản nợ đã xác nhận", "reference_reasoning": REASONING}
    return {"issues": [{"issue_index": issue["issue_index"], "groups": [{"group_id": chosen, "criteria": [criterion]}],
                        "reference_verdict": "B phải hoàn trả 100 đồng"} for issue in payload["gold_issues"]]}


def request_payload(request):
    body = json.loads(request.content)
    return body, json.loads(body["messages"][1]["content"])


def response(value, *, fence=False):
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if fence:
        content = "```json\n" + content + "\n```"
    return httpx.Response(200, json=envelope(content))


def build(config, handler):
    return asyncio.run(RubricPipeline(config, rubric_transport=httpx.MockTransport(handler)).run())


def verify_pair(config, expected_count):
    application = read_jsonl(config.final_application_jsonl)
    spotting = read_jsonl(config.final_issue_spotting_jsonl)
    assert len(application) == len(spotting) == expected_count
    assert len({r["doc_name"] for r in application}) == expected_count
    for i, (app, spot) in enumerate(zip(application, spotting)):
        rubric_sample(app, i)
        assert spot == {"input": app["input"]["facts_clean"], "expected_answer": app["input"]["issues"]}
    return application


def test_one_call_direct_final_auto_add_and_warnings(config_factory):
    raw = raw_record(category="Commercial", n=2)
    original = raw["issues"]
    config = settings(config_factory, [raw, raw_record("later")], batch_size=1, expected_count=2)
    raw_bytes = config.input_jsonl.read_bytes()
    calls = []

    def handler(request):
        body, payload = request_payload(request)
        calls.append(payload)
        assert body["model"] == config.rubric_model
        assert body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
        assert body["response_format"]["type"] == "json_schema"
        assert payload["expected_answer"] == ANSWER
        assert set(payload) == {"expected_answer", "gold_issues", "facts_clean", "rules", "category", "taxonomy"}
        if len(calls) == 2:
            verify_pair(config, 1)
            return response(model_output(payload))
        assert payload["category"] == "Commercial" and "contract" in payload["taxonomy"]
        result = model_output(payload, group_id="nhom-tu-de-xuat")
        result["issues"].reverse()
        result["issues"][1]["warnings"] = ["Cần đọc lại căn cứ tính lãi"]
        result["added_issues"] = [
            {"issue_text": original[0], "reason": "Trùng", "groups": []},
            {"issue_text": "Yêu cầu bồi thường độc lập được giải quyết thế nào?", "reason": "Ảnh hưởng kết quả riêng",
             "groups": [{"group_id": "remedies", "criteria": [{"description": "Phân tích yêu cầu độc lập", "reference_reasoning": "Bác bồi thường"}]}]},
        ]
        return response(result, fence=True)

    summary = build(config, handler)
    assert summary["status"] == "completed"
    assert summary["counts"]["completed"] == 2
    assert summary["counts"]["criteria"] == summary["counts"]["issues"] == 4
    assert len(calls) == summary["counts"]["api_calls"] == 2
    assert summary["counts"]["failed"] == 0 and summary["counts"]["added_issues"] == 1
    assert config.input_jsonl.read_bytes() == raw_bytes
    app, _ = verify_pair(config, 2)
    assert app["input"]["issues"][:2] == original
    assert app["rubrics"].keys() == {"version", "issues"}
    assert app["expected_answer"] == ANSWER
    first, second, added = app["rubrics"]["issues"]
    assert [c["criterion_id"] for issue in [first, second, added] for c in criteria_for(issue)] == ["I1_C1", "I2_C1", "I3_C1"]
    assert first["groups"][0]["group_id"] == "other_1"
    criterion = criteria_for(first)[0]
    assert set(criterion) == {"criterion_id", "description", "reference_reasoning"}
    assert criterion["reference_reasoning"] not in ANSWER
    log = read_jsonl(config.run_dir / "added_issues.jsonl")
    assert len(log) == 1 and log[0]["issue_text"] == app["input"]["issues"][-1]
    assert log[0]["reference_reasoning"] == ["Bác bồi thường"]
    assert read_jsonl(config.run_dir / "rubric_warnings.jsonl")
    assert not (config.run_dir / "rule_application_draft.jsonl").exists()


@pytest.mark.parametrize("bad", ["{invalid", {"issues": []}, {"issues": [{"issue_index": 1, "groups": []}]}])
def test_retry_recovers_invalid_json_or_missing_rubric(config_factory, bad):
    config = settings(config_factory, rubric_json_retries=1)
    calls = []

    def handler(request):
        body, payload = request_payload(request)
        calls.append(body)
        if len(calls) == 1:
            return response(bad)
        assert "Kết quả trước chưa đúng cấu trúc" in body["messages"][0]["content"]
        return response(model_output(payload))

    assert build(config, handler)["counts"]["completed"] == 1
    assert len(calls) == 2
    assert read_jsonl(config.run_dir / "rubric_records.jsonl")[0]["metadata"]["validation_errors"]


def test_resume_replays_torn_pair_and_only_calls_failed_sample(config_factory):
    config = settings(config_factory, [raw_record("ok"), raw_record("error", "Criminal", n=2)],
                      batch_size=1, expected_count=2, rubric_json_retries=0)
    calls = []

    def first(request):
        _, payload = request_payload(request)
        calls.append(payload["category"])
        return response(model_output(payload) if payload["category"] == "Civil" else {"issues": []})

    initial = build(config, first)
    assert initial["status"] == "completed_with_errors"
    assert initial["counts"]["completed"] == initial["counts"]["failed"] == 1
    assert initial["missing_doc_names"] == ["error"]
    verify_pair(config, 1)
    data = config.final_issue_spotting_jsonl.read_bytes()
    config.final_issue_spotting_jsonl.write_bytes(data[:10])
    config.final_application_jsonl.unlink()

    def second(request):
        assert config.final_issue_spotting_jsonl.read_bytes() == data
        verify_pair(config, 1)
        _, payload = request_payload(request)
        calls.append(payload["category"])
        return response(model_output(payload))

    resume = replace(config, resume=True)
    result = build(resume, second)
    assert result["status"] == "completed"
    assert result["counts"]["completed"] == 2 and result["counts"]["skipped"] == 1
    assert result["counts"]["failed"] == 0
    assert calls == ["Civil", "Criminal", "Criminal"]
    verify_pair(config, 2)

    def no_call(_):
        pytest.fail("Already completed")

    assert build(resume, no_call)["counts"]["skipped"] == 2


def test_output_conflict_does_not_overwrite_or_call_model(config_factory):
    config = settings(config_factory)
    app = config.final_application_jsonl
    app.write_text("existing data", encoding="utf-8")
    with pytest.raises(ValueError, match="Final dataset already exists"):
        build(config, lambda _: pytest.fail("Unexpected call"))
    assert app.read_text(encoding="utf-8") == "existing data"


def test_retry_into_new_pair_imports_successes(config_factory):
    source = settings(config_factory, [raw_record("ok"), raw_record("error", "Labor")],
                      rubric_json_retries=0)

    def first(request):
        _, payload = request_payload(request)
        return response(model_output(payload) if payload["category"] == "Civil" else {"issues": []})

    assert build(source, first)["counts"]["completed"] == 1
    retry = replace(source, run_id="retry", rubric_seed_run=source.run_dir,
                    final_application_jsonl=source.project_root / "retry-app.jsonl",
                    final_issue_spotting_jsonl=source.project_root / "retry-spot.jsonl")

    def second(request):
        _, payload = request_payload(request)
        assert payload["category"] == "Labor"
        assert verify_pair(retry, 1) == verify_pair(source, 1)
        return response(model_output(payload))

    result = build(retry, second)
    assert result["counts"]["skipped"] == 1 and result["counts"]["completed"] == 2
    verify_pair(retry, 2)
    verify_pair(source, 1)


def test_raw_preflight_count_validation(config_factory):
    config = settings(config_factory, expected_count=2)
    with pytest.raises(ValueError, match="found 1"):
        build(config, lambda _: pytest.fail("Unexpected call"))
    assert not config.final_application_jsonl.exists()


def test_unknown_raw_category_retains_line_report(config_factory):
    config = settings(config_factory, [raw_record(category="Unknown")])
    with pytest.raises(DatasetError, match="Input validation failed"):
        build(config, lambda _: pytest.fail("Unexpected call"))
    errors = json.loads((config.run_dir / "rubric_generation_summary.json").read_text())["input_validation_errors"]
    assert errors[0]["input_line"] == 1
    assert errors[0]["category"] == "Unknown"


def test_new_schema_eval_no_leak_and_issue_macro_average(config_factory):
    config = settings(config_factory, [raw_record(n=2)])

    def generate_rubric(request):
        _, payload = request_payload(request)
        result = model_output(payload)
        result["issues"][0]["groups"][0]["criteria"].append(
            {"description": "Phân tích hậu quả pháp lý", "reference_reasoning": "Buộc hoàn trả khoản nợ."})
        del result["issues"][1]["reference_verdict"]
        return response(result)

    build(config, generate_rubric)
    app, = verify_pair(config, 1)
    candidate = "Vì B đã thừa nhận nợ nên phải thanh toán."
    labels_sent = []

    def gen(request):
        prompt = json.loads(request.content)["messages"][1]["content"]
        for private in (ANSWER, REASONING, "Civil", "reference_reasoning", "DS-ST"):
            assert private not in prompt
        assert all(issue in prompt for issue in app["input"]["issues"])
        return response(candidate)

    def judge(request):
        _, payload = request_payload(request)
        assert payload["rubrics"] == app["rubrics"]
        assert payload["evaluated_model_response"] == candidate
        plan = payload["evaluation_plan"]
        assert not plan["needs_review"]
        assert "reference_reasoning" in payload["rubrics"]["issues"][0]["groups"][0]["criteria"][0]
        results = [
            {"issue_index": i, "evaluation_status": "scored",
             "criterion_scores": [{"criterion_id": criterion_id, "score": score,
                                   "evidence_from_model_answer": [], "explanation": "Đối chiếu nội dung."}
                                  for criterion_id, score in zip(issue["criterion_ids"], scores)],
             "outcome_correct": True if issue["outcome_scorable"] else None, "outcome_explanation": "Đánh giá riêng."}
            for i, (issue, scores) in enumerate(zip(plan["issues"], [[2, 1], [0]]), 1)]
        labels = {"doc_name": app["doc_name"], "category": app["category"], "issues_evaluation": results,
                  "needs_review": False, "evaluation_status": "completed"}
        labels_sent.append(labels)
        return response(labels)

    eval_config = replace(config, input_jsonl=config.final_application_jsonl, mode="full", run_id="eval")
    result = asyncio.run(Pipeline(eval_config, generator_transport=httpx.MockTransport(gen),
                                  judge_transport=httpx.MockTransport(judge)).run())
    assert result["application_score"] == 0.375
    assert result["outcome_accuracy"] == 1
    labels_sent[0]["issues_evaluation"][1]["criterion_scores"][0]["criterion_id"] = "invented"
    with pytest.raises(LabelError, match="eligible criterion"):
        score_rubric_labels(labels_sent[0], rubric_sample(app, 0), candidate)


def test_real_input_ten_rows_with_mock_endpoint(config_factory):
    sample_path = Path(__file__).resolve().parents[1] / "input" / "sample_v2.jsonl"
    if not sample_path.is_file():
        pytest.skip("The local sample_v2 dataset is unavailable")
    config = replace(settings(config_factory, expected_count=10), input_jsonl=sample_path)
    calls = []

    def handler(request):
        _, payload = request_payload(request)
        calls.append(payload["category"])
        return response(model_output(payload))

    result = build(config, handler)
    assert result["status"] == "completed"
    assert result["counts"]["completed"] == 10 and len(calls) == 10
    assert result["counts"]["failed"] == 0
    assert all(evaluation_plan(rubric_sample(row, i))["issues"] for i, row in enumerate(verify_pair(config, 10)))


def test_existing_v2_dataset_still_validates_with_manifest(config_factory):
    config = settings(config_factory)
    build(config, lambda request: response(model_output(request_payload(request)[1])))
    app, = verify_pair(config, 1)
    rubric = app["rubrics"]
    rubric.update(version=2, status="verified")
    issue, = rubric["issues"]
    issue.update(verdict_evidence=[ANSWER], outcome_review_reason=None, issue_review_reason=None)
    group, = issue["groups"]
    assert group["group_name"] == TAXONOMY["Civil"][group["group_id"]]
    criterion, = group["criteria"]
    criterion["criterion"] = criterion.pop("description")
    criterion.update(unit_id="U1", source_evidence=[ANSWER], relevant_facts=[], rule_refs=[],
                     needs_review=False, scorable=True, decisive=False, review_reason=None, missing_information=[],
                     reasoning_type="legal_consequence", legal_condition=None, application_link=None)
    atomic_jsonl(config.final_application_jsonl, [app])
    check = replace(config, mode="validate-only", input_jsonl=config.final_application_jsonl, run_id="missing-manifest")
    with pytest.raises(DatasetError, match="Input validation failed"):
        asyncio.run(Pipeline(check).run())
    path = config.final_application_jsonl.with_suffix(".jsonl.manifest.json")
    atomic_json(path, {"kind": "rule_application_rubrics", "complete": True, "sample_count": 1, "rubric_protocol": 2,
                       "application": {"sha256": file_hash(config.final_application_jsonl)}})
    assert asyncio.run(Pipeline(replace(check, run_id="with-manifest")).run())["status"] == "validated"
