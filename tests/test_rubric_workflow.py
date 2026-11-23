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


def model_output(payload):
    criterion = {"description": "Giải thích nghĩa vụ trả khoản nợ đã xác nhận", "reference_reasoning": REASONING}
    return {"issues": [{"issue_index": issue["issue_index"], "criteria": [criterion],
                        "reference_verdict": "B phải hoàn trả 100 đồng"} for issue in payload["gold_issues"]]}


def request_payload(request):
    body = json.loads(request.content)
    return body, json.loads(body["messages"][1]["content"])


def response(value, *, fence=False):
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if fence:
        content = "```json\n" + content + "\n```"
    return httpx.Response(200, json=envelope(content))


def quality_review(candidate, *, problems=(), reviews=None, approved=True, score=2):
    return {"approved": approved, "rubric_scores": dict.fromkeys(
        ["relevance", "coverage", "correctness", "input_feasibility", "granularity"], score),
        "rubric_problems": list(problems),
        "added_issue_reviews": reviews if reviews is not None else [
            {"issue_index": item["issue_index"], "decision": "keep", "reason": "Vấn đề độc lập.",
             "suggestion": "Giữ nguyên."} for item in candidate["added_issues"]],
        "overall_feedback": "Đã đối chiếu rubric với nguồn."}


def build(config, handler, *, classify=None, review=None):
    def dispatch(request):
        _, payload = request_payload(request)
        if "original_category" not in payload:
            return handler(request)
        if classify is not None:
            return classify(request)
        return response({"category": payload["original_category"], "confidence": "clear",
                         "reason": "Nội dung chính phù hợp nhãn gốc."})

    def quality(request):
        _, payload = request_payload(request)
        return review(request) if review else response(quality_review(payload["candidate"]))

    return asyncio.run(RubricPipeline(config, rubric_transport=httpx.MockTransport(dispatch),
                                      quality_transport=httpx.MockTransport(quality)).run())


def verify_pair(config, expected_count):
    application = read_jsonl(config.final_application_jsonl)
    spotting = read_jsonl(config.final_issue_spotting_jsonl)
    assert len(application) == len(spotting) == expected_count
    assert len({r["doc_name"] for r in application}) == expected_count
    for i, (app, spot) in enumerate(zip(application, spotting)):
        rubric_sample(app, i)
        assert spot == {"input": app["input"]["facts_clean"], "expected_answer": app["input"]["issues"]}
    return application


def test_direct_final_auto_add_and_warnings(config_factory):
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
        result = model_output(payload)
        result["issues"].reverse()
        result["issues"][1]["warnings"] = ["Cần đọc lại căn cứ tính lãi"]
        result["added_issues"] = [
            {"issue_text": original[0], "reason": "Trùng", "criteria": []},
            {"issue_text": "Yêu cầu bồi thường độc lập được giải quyết thế nào?", "reason": "Ảnh hưởng kết quả riêng",
             "criteria": [{"description": "Phân tích yêu cầu độc lập", "reference_reasoning": "Bác bồi thường"}]},
        ]
        return response(result, fence=True)

    summary = build(config, handler)
    assert summary["status"] == "completed"
    assert summary["counts"]["completed"] == 2
    assert summary["counts"]["criteria"] == summary["counts"]["issues"] == 4
    assert len(calls) == 2 and summary["counts"]["api_calls"] == 6
    assert summary["counts"]["failed"] == 0 and summary["counts"]["added_issues"] == 1
    assert config.input_jsonl.read_bytes() == raw_bytes
    app, _ = verify_pair(config, 2)
    assert app["input"]["issues"][:2] == original
    assert app["rubrics"].keys() == {"version", "issues"}
    assert app["expected_answer"] == ANSWER
    first, second, added = app["rubrics"]["issues"]
    assert [c["criterion_id"] for issue in [first, second, added] for c in criteria_for(issue)] == ["I1_C1", "I2_C1", "I3_C1"]
    assert set(first) == {"issue_index", "issue_text", "criteria", "reference_verdict"}
    assert summary["counts"]["category_corrections"] == 0
    assert read_jsonl(config.run_dir / "category_corrections.jsonl") == []
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


@pytest.mark.parametrize(
    "confidence, final_category", [("clear", "Marriage & Family"), ("uncertain", "Criminal")],
)
def test_category_check_precedes_generation(config_factory, confidence, final_category):
    raw = raw_record("1433613", "Criminal")
    raw["issues"] = ["Ai trực tiếp nuôi con sau ly hôn?"]
    raw["benchmark_seed"]["facts_clean"] = "Hai vợ chồng xin ly hôn và tranh chấp quyền nuôi con."
    raw["benchmark_seed"]["application"] = "Tòa chấp nhận ly hôn và giao con cho người mẹ trực tiếp nuôi."
    raw["laws"][0]["law_str"] = "Luật Hôn nhân và Gia đình"
    config = settings(config_factory, [raw])
    stages = []
    reason = "Yêu cầu chính là ly hôn và giải quyết quyền trực tiếp nuôi con."

    def classify(request):
        body, payload = request_payload(request)
        stages.append("category")
        assert set(TAXONOMY) == set(payload["allowed_categories"])
        assert payload["original_category"] == "Criminal"
        assert payload["facts_clean"] == raw["benchmark_seed"]["facts_clean"]
        assert payload["expected_answer"] == raw["benchmark_seed"]["application"]
        assert payload["rules"][0]["law_str"] == "Luật Hôn nhân và Gia đình"
        assert payload["gold_issues"] == [{"issue_index": 1, "issue_text": raw["issues"][0]}]
        assert "đối tượng tranh chấp hoặc truy tố chính" in body["messages"][0]["content"]
        return response({"category": "Marriage & Family", "confidence": confidence, "reason": reason})

    def generate(request):
        body, payload = request_payload(request)
        stages.append("rubric")
        assert payload["category"] == final_category
        assert payload["taxonomy"] == TAXONOMY[final_category]
        system = body["messages"][0]["content"]
        assert "Đừng hỏi Tòa án đã viết, xem xét, đề cập hoặc bỏ sót điều gì." in system
        return response({"issues": [{"issue_index": 1, "criteria": [
            {"description": "Đánh giá ai nên trực tiếp nuôi con sau ly hôn dựa trên tình tiết được cung cấp.",
             "reference_reasoning": "Người mẹ được giao trực tiếp nuôi con."}]}]})

    summary = build(config, generate, classify=classify)
    assert stages == ["category", "rubric"]
    app, = verify_pair(config, 1)
    assert app["category"] == final_category
    assert set(summary["by_category"]) == {final_category}
    assert json.loads(config.input_jsonl.read_text())["category"] == "Criminal"
    corrections = read_jsonl(config.run_dir / "category_corrections.jsonl")
    if confidence == "clear":
        assert corrections == [{"doc_name": "1433613", "category": final_category, "original_category": "Criminal",
                                "corrected_category": "Marriage & Family", "reason": reason}]
        assert summary["counts"]["category_corrections"] == 1
        (config.run_dir / "category_corrections.jsonl").unlink()
        no_call = lambda _: pytest.fail("Already completed")
        build(replace(config, resume=True), no_call, classify=no_call)
        assert read_jsonl(config.run_dir / "category_corrections.jsonl") == corrections
    else:
        assert corrections == []
        warnings = read_jsonl(config.run_dir / "rubric_warnings.jsonl")
        assert reason in warnings[0]["reason"]


def test_category_check_failure_keeps_original_and_generates(config_factory):
    config = settings(config_factory, [raw_record(category="Labor")], rubric_json_retries=0)

    def generate(request):
        _, payload = request_payload(request)
        assert payload["category"] == "Labor"
        return response(model_output(payload))

    result = build(config, generate, classify=lambda _: response("{invalid"))
    assert result["counts"]["completed"] == 1
    assert result["counts"]["category_corrections"] == 0
    notes = read_jsonl(config.run_dir / "rubric_warnings.jsonl")
    assert "bước kiểm tra gặp lỗi" in notes[0]["reason"]


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
    config = settings(config_factory, [raw_record(n=3)])

    def generate_rubric(request):
        _, payload = request_payload(request)
        result = model_output(payload)
        result["issues"][0]["criteria"].append(
            {"description": "Phân tích hậu quả pháp lý", "reference_reasoning": "Buộc hoàn trả khoản nợ."})
        del result["issues"][1]["reference_verdict"]
        result["issues"][2] = {"issue_index": 3, "criteria": [],
                               "missing_reference_reason": "Nhận định không giải quyết yêu cầu thứ ba."}
        return response(result)

    result = build(config, generate_rubric)
    assert result["counts"]["completed"] == 1
    assert result["counts"]["issues_without_criteria"] == 1
    notes = read_jsonl(config.run_dir / "rubric_warnings.jsonl")
    assert notes[0]["issue_index"] == 3 and "yêu cầu thứ ba" in notes[0]["reason"]
    app, = verify_pair(config, 1)
    assert app["rubrics"]["issues"][2]["criteria"] == []
    assert app["rubrics"]["issues"][2]["reference_verdict"] is None
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
        assert "reference_reasoning" in payload["rubrics"]["issues"][0]["criteria"][0]
        results = [
            {"issue_index": i, "evaluation_status": issue["evaluation_status"],
             "criterion_scores": [{"criterion_id": criterion_id, "score": score,
                                   "evidence_from_model_answer": [], "explanation": "Đối chiếu nội dung."}
                                  for criterion_id, score in zip(issue["criterion_ids"], scores)],
             "outcome_correct": True if issue["outcome_scorable"] else None, "outcome_explanation": "Đánh giá riêng."}
            for i, (issue, scores) in enumerate(zip(plan["issues"], [[2, 1], [0], []]), 1)]
        labels = {"doc_name": app["doc_name"], "category": app["category"], "issues_evaluation": results,
                  "needs_review": False, "evaluation_status": "completed"}
        labels_sent.append(labels)
        return response(labels)

    eval_config = replace(config, input_jsonl=config.final_application_jsonl, mode="full", run_id="eval")
    result = asyncio.run(Pipeline(eval_config, generator_transport=httpx.MockTransport(gen),
                                  judge_transport=httpx.MockTransport(judge)).run())
    assert result["application_score"] == 0.375
    assert result["outcome_accuracy"] == 1
    assert result["metrics"]["unscorable_issues"] == 1
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
    assert result["counts"]["api_calls"] == 30
    assert result["counts"]["failed"] == 0
    for index, row in enumerate(verify_pair(config, 10)):
        assert evaluation_plan(rubric_sample(row, index))["issues"]
        assert all("criteria" in issue and "groups" not in issue for issue in row["rubrics"]["issues"])


def test_existing_v2_dataset_still_validates_with_manifest(config_factory):
    config = settings(config_factory)
    build(config, lambda request: response(model_output(request_payload(request)[1])))
    app, = verify_pair(config, 1)
    rubric = app["rubrics"]
    rubric.update(version=2, status="verified")
    issue, = rubric["issues"]
    issue.update(verdict_evidence=[ANSWER], outcome_review_reason=None, issue_review_reason=None)
    group_id, group_name = next(iter(TAXONOMY["Civil"].items()))
    criterion, = issue.pop("criteria")
    issue["groups"] = [{"group_id": group_id, "group_name": group_name, "criteria": [criterion]}]
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


@pytest.mark.parametrize("quality_url", [None, "http://quality.test:8002/v1/"])
def test_quality_judge_reviews_added_issues_and_checks_revised_rubric_independently(config_factory, quality_url):
    raw = raw_record(category="Criminal")
    raw["issues"] = ["Trách nhiệm hình sự của bị cáo được xác định như thế nào?"]
    config = settings(config_factory, [raw], rubric_timeout_seconds=42, rubric_quality_timeout_seconds=23,
                      rubric_base_url="http://rubric-generator.test:8000/v1", rubric_quality_base_url=quality_url)
    requests = []
    initial = [
        ("Tiền bảo đảm phải xử lý như thế nào theo khoản 1 Điều 122 Bộ luật Tố tụng hình sự?", "rewrite"),
        ("Tịch thu tiêu hủy ma túy và trả lại điện thoại cho bị cáo.", "rewrite"),
        ("Có tình tiết giảm nhẹ nào khi xác định trách nhiệm hình sự của bị cáo?", "merge_or_remove"),
        ("Khoản tiền nộp làm bảo đảm cần được giải quyết ra sao?", "merge_or_remove"),
        ("Người bị hại có quyền yêu cầu bồi thường thiệt hại nào?", "keep"),
    ]
    problems = [
        {"issue_index": 1, "criterion_index": 1, "severity": "major",
         "problem": "Chỉ xác nhận thẩm quyền thông thường.", "suggestion": "Loại đoạn xác nhận thẩm quyền."},
        {"issue_index": 1, "criterion_index": 2, "severity": "major",
         "problem": "Đòi model biết Tòa đã viết gì.", "suggestion": "Hỏi cách áp dụng điều kiện cấu thành tội."},
        {"issue_index": 1, "severity": "major", "problem": "reference_verdict=null dù nhận định có kết luận rõ.",
         "suggestion": "Tóm tắt kết luận được nêu."},
        {"issue_index": 1, "criterion_index": 3, "severity": "major",
         "problem": "Cần loại tiêu chí giải thích dấu hiệu lỗi.", "suggestion": "Bỏ tiêu chí này."},
    ]
    replacements = {
        2: "Khoản tiền bảo đảm cần được xử lý như thế nào?",
        3: "Các vật chứng và tài sản liên quan cần được xử lý như thế nào?",
        4: "Gộp tình tiết giảm nhẹ có giá trị vào gold issue.",
        5: "Gộp nội dung có giá trị vào issue về tiền bảo đảm.",
        6: "Giữ issue độc lập.",
    }
    reasons = ["Đưa điều luật vào issue.", "Tiết lộ phán quyết.", "Đã nằm trong gold issue hình sự.",
               "Trùng ngữ nghĩa với issue về tiền bảo đảm.", "Trách nhiệm dân sự là vấn đề độc lập ở đây."]

    def remember(request, role, timeout):
        body, payload = request_payload(request)
        assert len(body["messages"]) == 2
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["model"] == config.rubric_model
        url = quality_url if role == "rubric_quality_judge" and quality_url else config.rubric_base_url
        assert str(request.url) == url.rstrip("/") + "/chat/completions"
        assert body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
        assert request.extensions["timeout"]["read"] == timeout
        assert body["response_format"]["json_schema"]["name"] == role
        assert "DO_NOT_SAVE_REASONING" not in json.dumps(body)
        requests.append(request.headers["X-Request-ID"])
        return body, payload

    def gen(request):
        body, payload = remember(request, "rubric_generator", 42)
        assert "Soạn rubric" in body["messages"][0]["content"]
        assert payload["expected_answer"] == ANSWER
        result = model_output(payload)
        criterion = {"description": "Giải thích dấu hiệu lỗi từ tình tiết.", "reference_reasoning": REASONING}
        issue = result["issues"][0]
        if "feedback" not in payload:
            issue["reference_verdict"] = None
            issue["criteria"] = [
                {**criterion, "description": "Tòa xác nhận thẩm quyền thụ lý."},
                {**criterion, "description": "Tòa đã viết gì về trách nhiệm hình sự?"},
                criterion,
            ]
            texts = [text for text, _ in initial]
        else:
            assert payload["feedback"]["assessment"]["rubric_problems"] == problems
            assert payload["feedback"]["citation_findings"][0]["issue_index"] == 2
            assert "đối chiếu" in body["messages"][0]["content"].casefold()
            issue["criteria"] = [criterion, {**criterion, "description": "Đánh giá tình tiết giảm nhẹ liên quan."}]
            texts = [replacements[2], replacements[3], initial[4][0]]
            result["revision_notes"] = [
                {"issue_index": 1, "action": "revised", "explanation": "Loại tiêu chí thẩm quyền và cập nhật verdict."},
                {"issue_index": 1, "action": "disagreed", "explanation": "Dấu hiệu lỗi thuộc nội dung cấu thành."},
                {"issue_index": 4, "action": "revised", "explanation": "Đã gộp giảm nhẹ vào gold issue."},
                {"issue_index": 5, "action": "revised", "explanation": "Bỏ issue trùng về tiền bảo đảm."},
            ]
        result["added_issues"] = [{"issue_text": text, "reason": "Vấn đề đề xuất.", "criteria": [criterion]}
                                  for text in texts]
        return response(result)

    def judge(request):
        body, payload = remember(request, "rubric_quality_judge", 23)
        assert "Tự đọc facts_clean" in body["messages"][0]["content"]
        assert "trùng nghĩa" in body["messages"][0]["content"]
        assert set(payload) == {"expected_answer", "facts_clean", "rules", "gold_issues", "category", "candidate"}
        candidate = payload["candidate"]
        if len(candidate["added_issues"]) == 5:
            assert candidate["issues"][0]["reference_verdict"] is None
            reviews = [{"issue_index": i, "decision": pair[1], "reason": reason, "suggestion": replacements[i]}
                       for i, (pair, reason) in enumerate(zip(initial, reasons), 2)]
            return response(quality_review(candidate, problems=problems, reviews=reviews, approved=False, score=1))
        descriptions = [c["description"] for c in candidate["issues"][0]["criteria"]]
        assert descriptions == ["Giải thích dấu hiệu lỗi từ tình tiết.", "Đánh giá tình tiết giảm nhẹ liên quan."]
        assert candidate["issues"][0]["reference_verdict"] is not None
        return response(quality_review(candidate))

    result = build(config, gen, review=judge)
    assert result["counts"]["feedback_rounds"] == 1
    assert len(requests) == len(set(requests)) == 4
    app, = verify_pair(config, 1)
    assert set(app["rubrics"]) == {"version", "issues"} and app["rubrics"]["version"] == 4
    assert app["input"]["issues"] == raw["issues"] + [replacements[2], replacements[3], initial[4][0]]
    additions = read_jsonl(config.run_dir / "added_issues.jsonl")
    assert [row["issue_text"] for row in additions] == app["input"]["issues"][1:]
    qa, = read_jsonl(config.run_dir / "rubric_quality_reviews.jsonl")
    assert [r["assessment"]["approved"] for r in qa["rounds"]] == [False, True]
    assert qa["rounds"][1]["revision_notes"][1]["action"] == "disagreed"
    assert qa["rounds"][0]["quality_judge"]["calls"][0]["usage"] == {"completion_tokens": 20}
    assert qa["selected_round"] == 1 and qa["final_quality_scores"]["coverage"] == 2
    assert qa["quality_warning"] is False
    assert "DO_NOT_SAVE_REASONING" not in json.dumps(qa)


def test_two_revision_limit_still_saves_best_rubric_and_filters_unsafe_additions(config_factory):
    config = settings(config_factory)
    generated, judged = [], []

    def gen(request):
        _, payload = request_payload(request)
        result = model_output(payload)
        idx = len(generated)
        generated.append(payload)
        issue = result["issues"][0]
        issue["reference_verdict"] = f"Phiên bản {idx}"
        criterion = issue["criteria"][0]
        if idx:
            result["revision_notes"] = [{"action": "disagreed", "explanation": "Giữ bản hiện tại."}]
        result["added_issues"] = [
            {"issue_text": "Có cần áp dụng Điều 51 không?", "reason": "Đề xuất.", "criteria": [criterion]},
            {"issue_text": "Buộc B trả tiền.", "reason": "Đề xuất.", "criteria": [criterion]},
            {"issue_text": "Khoản 5 triệu đồng được xử lý ra sao?", "reason": "Độc lập.", "criteria": [criterion]},
        ]
        return response(result)

    def judge(request):
        _, payload = request_payload(request)
        candidate = payload["candidate"]
        judged.append(candidate)
        review = quality_review(candidate, approved=False, score=[0, 2, 1][len(judged) - 1])
        rewrite = review["added_issue_reviews"][1]
        rewrite.update(decision="rewrite", reason="Nêu sẵn phán quyết.", suggestion="Hỏi trung lập.")
        # Simulate the Judge missing an explicit citation. The output guard still omits it.
        return response(review)

    summary = build(config, gen, review=judge)
    assert len(generated) == len(judged) == 3
    assert summary["counts"]["feedback_rounds"] == 2
    assert summary["counts"]["completed"] == summary["counts"]["quality_warnings"] == 1
    app, = verify_pair(config, 1)
    assert app["rubrics"]["issues"][0]["reference_verdict"] == "Phiên bản 1"
    assert app["input"]["issues"][1:] == ["Khoản 5 triệu đồng được xử lý ra sao?"]
    assert app["rubrics"]["issues"][1]["criteria"][0]["criterion_id"] == "I2_C1"
    qa, = read_jsonl(config.run_dir / "rubric_quality_reviews.jsonl")
    assert qa["quality_warning"] and qa["selected_round"] == 1
    assert [item["issue_index"] for item in qa["unresolved"]["omitted_added_issues"]] == [2, 3]
    assert [item["issue_index"] for item in read_jsonl(config.run_dir / "added_issues.jsonl")] == [2]


def test_resume_from_feedback_checkpoint(config_factory):
    config = settings(config_factory)
    tech_problem = {"severity": "major", "issue_index": 1, "problem": "Thiếu verdict.", "suggestion": "Bổ sung verdict."}

    def cancel_revision(request):
        _, payload = request_payload(request)
        if "feedback" in payload:
            raise asyncio.CancelledError
        return response(model_output(payload))

    def needs_fix(request):
        _, payload = request_payload(request)
        return response(quality_review(payload["candidate"], problems=[tech_problem], approved=False))

    with pytest.raises(asyncio.CancelledError):
        build(config, cancel_revision, review=needs_fix)
    verify_pair(config, 0)

    def revise(request):
        _, payload = request_payload(request)
        assert payload["feedback"]["assessment"]["rubric_problems"] == [tech_problem]
        result = model_output(payload)
        result["revision_notes"] = [{"issue_index": 1, "action": "revised", "explanation": "Đã kiểm tra."}]
        return response(result)

    assert build(replace(config, resume=True), revise,
                 classify=lambda _: pytest.fail("Category already checkpointed"))["counts"]["completed"] == 1
    assert [r["assessment"]["approved"] for r in read_jsonl(config.run_dir / "rubric_quality_reviews.jsonl")[0]["rounds"]] == [False, True]


def test_quality_technical_retries_are_independent_and_gold_still_saved(config_factory):
    config = settings(config_factory, rubric_api_max_retries=1, rubric_quality_api_max_retries=0,
                      rubric_quality_json_retries=0)
    gen_calls, judge_calls = [], []

    def gen(request):
        gen_calls.append(request.headers["X-Request-ID"])
        if len(gen_calls) == 1:
            return httpx.Response(503)
        _, payload = request_payload(request)
        result = model_output(payload)
        result["added_issues"] = [{"issue_text": "Có phải trả thêm tiền không?", "reason": "Độc lập.",
                                   "criteria": result["issues"][0]["criteria"]}]
        return response(result)

    def judge(request):
        judge_calls.append(request.headers["X-Request-ID"])
        return httpx.Response(503)

    result = build(config, gen, review=judge)
    assert len(gen_calls) == 2 and gen_calls[0] == gen_calls[1]
    assert len(judge_calls) == 1
    assert result["counts"]["completed"] == result["counts"]["quality_warnings"] == 1
    assert result["counts"]["added_issues"] == 0
    qa, = read_jsonl(config.run_dir / "rubric_quality_reviews.jsonl")
    assert qa["unresolved"]["technical_errors"][0]["role"] == "rubric_quality_judge"
    assert qa["final_quality_scores"] is None


def test_quality_json_retry_does_not_trigger_revision_for_minor_feedback(config_factory):
    config = settings(config_factory, rubric_json_retries=0, rubric_quality_json_retries=1)
    judged = []

    def generate(request):
        _, payload = request_payload(request)
        assert "feedback" not in payload
        return response(model_output(payload))

    def judge(request):
        body, payload = request_payload(request)
        judged.append(body)
        if len(judged) == 1:
            return response("{invalid")
        note = {"issue_index": 1, "criterion_index": 1, "severity": "minor",
                "problem": "Diễn đạt hơi dài.", "suggestion": "Có thể viết gọn hơn."}
        return response(quality_review(payload["candidate"], problems=[note]))

    result = build(config, generate, review=judge)
    assert len(judged) == 2 and result["counts"]["feedback_rounds"] == 0
    assert result["counts"]["api_calls_by_role"] == {"rubric_generator": 2, "rubric_quality_judge": 2}
    qa, = read_jsonl(config.run_dir / "rubric_quality_reviews.jsonl")
    assert qa["rounds"][0]["quality_judge"]["validation_errors"]
    assert qa["quality_warning"] is False
