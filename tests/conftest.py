from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from legal_application.config import Config


def make_record(doc="case-1", n=2):
    return {
        "doc_name": doc,
        "data_status": "full",
        "benchmark_seed": {"facts": "RAW_FACTS_SENTINEL", "facts_clean": "A yêu cầu B hoàn trả 100 đồng; B phủ nhận đã nhận tiền.",
                           "application": "PRIVATE_APPLICATION_SENTINEL: Chứng cứ xác định B nhận 100 đồng, phải hoàn trả.",
                           "conclusion": "PRIVATE_CONCLUSION_SENTINEL: Buộc B trả A 100 đồng.", "extra": {"keep": "FULL_SEED"}},
        "issues": [f"Vấn đề {i}: nghĩa vụ trả tiền?" for i in range(1, n + 1)],
        "laws": [
            {"law_str": "Bộ luật thử nghiệm", "private_metadata": "LAW_METADATA_MUST_NOT_LEAK",
             "provisions": [{"citation_str": "Điều 1", "law_content": {"context_text": "Toàn bộ nội dung điều 1.\nKhông cắt ngắn.", "exact_text": "EXACT_NOT_USED"}},
                            {"citation_str": "Điều 2", "law_content": {"context_text": "Toàn bộ nội dung điều 2."}}]},
            {"law_str": "Văn bản toàn văn", "provisions": [], "law_content": {"exact_text": "Toàn bộ văn bản fallback."}}
        ],
        "explain": ["PRIVATE_EXPLAIN_MUST_NOT_LEAK"], "markdown": "PRIVATE_MARKDOWN_MUST_NOT_LEAK"
    }


def make_labels(raw, *, covered=None, correct=None, order=True, review=False):
    n = len(raw["issues"])
    covered = covered if covered is not None else [True] * n
    correct = correct if correct is not None else [True] * n
    return {"doc_name": raw["doc_name"],
            "issues_evaluation": [{"issue_index": i + 1, "original_issue": issue, "covered": covered[i],
                                   "correct": None if review and i == 0 else correct[i],
                                   "extracted_model_verdict": "B trả A 100 đồng" if covered[i] else None,
                                   "inferred_reference_verdict": None if review and i == 0 else "B phải trả A 100 đồng",
                                   "explanation": "Đối chiếu nghĩa vụ trong application và conclusion.",
                                   "needs_review": review and i == 0}
                                  for i, issue in enumerate(raw["issues"])],
            "final_order_correct": None if review else order,
            "final_order_explanation": "Tập hợp các phán quyết theo issues phù hợp kết quả trong phạm vi được hỏi.",
            "needs_review": review, "evaluation_status": "needs_review" if review else "completed"}


def envelope(content, *, finish="stop"):
    return {"id": "mock-id", "model": "mock-served", "choices": [{"message": {"content": content, "reasoning_content": "DO_NOT_SAVE_REASONING"},
                                                               "finish_reason": finish}], "usage": {"completion_tokens": 20}}


def generator_response(issues):
    return "\n\n".join(f"ISSUE {i}: {issue}\nPhân tích và áp dụng pháp luật: Áp dụng Điều 1 vào chứng cứ được cung cấp.\nPhán quyết đối với Issue {i}: B trả A 100 đồng."
                        for i, issue in enumerate(issues, 1))


@pytest.fixture
def config_factory(tmp_path):
    def factory(records=None, **changes):
        path = tmp_path / "input.jsonl"
        records = records if records is not None else [make_record()]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        return replace(Config(project_root=tmp_path, input_jsonl=path, output_dir=tmp_path / "outputs", run_id="test-run",
                              generator_base_url="http://generator.test/v1", generator_model="evaluated-model",
                              judge_base_url="http://judge.test/v1", api_max_retries=1, judge_json_retries=1, retry_backoff=0), **changes)
    return factory


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch):
    async def fail(*args, **kwargs):
        raise AssertionError("Real network is forbidden in unit tests")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)


@pytest.fixture
def mock_transports():
    calls = {"generator": [], "judge": []}

    def gen(request):
        body = json.loads(request.content)
        calls["generator"].append(body)
        issues_block = body["messages"][1]["content"].split("LEGAL ISSUES\n\n")[1].split("\n\nAPPLICABLE LAWS")[0]
        issues = [item.split(". ", 1)[1] for item in issues_block.split("\n\n")]
        return httpx.Response(200, json=envelope(generator_response(issues)))

    def judge(request):
        body = json.loads(request.content)
        calls["judge"].append(body)
        payload = json.loads(body["messages"][1]["content"])
        return httpx.Response(200, json=envelope(json.dumps(make_labels(payload), ensure_ascii=False)))

    return httpx.MockTransport(gen), httpx.MockTransport(judge), calls
