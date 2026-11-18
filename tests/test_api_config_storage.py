from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from legal_application.api import APIClient, APIError, IncompleteResponse
from legal_application.config import Config
from legal_application.storage import RunStore, read_jsonl
from conftest import envelope


def call(client):
    async def invoke():
        try:
            return await client.complete([{"role": "user", "content": "test"}], doc_name="mock-case")
        finally:
            await client.close()
    return asyncio.run(invoke())


def test_xhigh_body_matches_legal_dataset(config_factory):
    config = config_factory(max_output_tokens=65536)
    client = APIClient(config, "judge", httpx.MockTransport(lambda request: httpx.Response(200, json=envelope("{}"))))
    body = client.body([])
    assert body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
    assert "reasoning_effort" not in body
    assert body["max_tokens"] == 65536
    assert body["response_format"]["type"] == "json_schema"
    assert body["model"] == "DeepSeek-V4.1-Flash-1017"
    call(client)


def test_free_text_generator_exact_raw_response_and_optional_thinking(config_factory):
    raw = " \nISSUE 1: text\n "
    config = config_factory()
    client = APIClient(config, "generator", httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(raw))))
    assert "response_format" not in client.body([])
    assert "chat_template_kwargs" not in client.body([])
    completion = call(client)
    assert completion.content == raw
    assert "DO_NOT_SAVE_REASONING" not in json.dumps(completion.metadata)
    client = APIClient(replace(config, generator_thinking_mode="deepseek_vllm_xhigh"), "generator", httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(raw))))
    assert client.body([])["chat_template_kwargs"]["reasoning_effort"] == 75
    call(client)


@pytest.mark.parametrize("generator_mode", ["none", "deepseek_vllm_xhigh"])
@pytest.mark.parametrize("judge_mode", ["none", "deepseek_vllm_xhigh"])
def test_thinking_modes_are_independent_per_role(config_factory, generator_mode, judge_mode):
    config = config_factory(generator_thinking_mode=generator_mode, judge_thinking_mode=judge_mode)
    for role, mode in [("generator", generator_mode), ("judge", judge_mode)]:
        def handle(request):
            body = json.loads(request.content)
            if mode == "none":
                assert "chat_template_kwargs" not in body and "reasoning_effort" not in body
            else:
                assert body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
            assert ("response_format" in body) == (role == "judge")
            return httpx.Response(200, json=envelope("{}"))

        completion = call(APIClient(config, role, httpx.MockTransport(handle)))
        actual = completion.metadata["actual_thinking_parameters"]
        assert actual == ({} if mode == "none" else {"chat_template_kwargs": {"thinking": True, "reasoning_effort": 75}})


def test_judge_thinking_mode_env_and_validation(config_factory, monkeypatch):
    monkeypatch.setenv("JUDGE_THINKING_MODE", "none")
    assert Config.from_env().judge_thinking_mode == "none"
    monkeypatch.delenv("JUDGE_THINKING_MODE")
    assert Config.from_env().judge_thinking_mode == "deepseek_vllm_xhigh"
    with pytest.raises(ValueError, match="JUDGE_THINKING_MODE"):
        config_factory(judge_thinking_mode="invalid").validate()


@pytest.mark.parametrize("first", [429, 503, "timeout", "connection", "envelope"])
def test_retry_then_success(config_factory, first):
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        if count == 1:
            if first == "timeout":
                raise httpx.ReadTimeout("mock timeout", request=request)
            if first == "connection":
                raise httpx.ConnectError("mock connection", request=request)
            if first == "envelope":
                return httpx.Response(200, json={"choices": []})
            return httpx.Response(first)
        return httpx.Response(200, json=envelope("raw"))

    completion = call(APIClient(config_factory(), "generator", httpx.MockTransport(handle)))
    assert count == completion.metadata["http_attempts"] == 2


def test_timeout_and_auth_headers_are_configured(config_factory):
    def handle(request):
        assert request.extensions["timeout"]["read"] == 17
        assert request.headers["Authorization"] == "Bearer secret-key"
        assert request.headers["X-Request-ID"]
        return httpx.Response(200, json=envelope("raw"))

    config = config_factory(timeout_seconds=17, generator_api_key="secret-key")
    assert "secret-key" not in json.dumps(config.snapshot())
    call(APIClient(config, "generator", httpx.MockTransport(handle)))


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_nonretryable_http_errors(config_factory, status):
    calls = []
    client = APIClient(config_factory(), "judge", httpx.MockTransport(lambda req: calls.append(req) or httpx.Response(status)))
    with pytest.raises(APIError, match=f"HTTP {status}"):
        call(client)
    assert len(calls) == 1


@pytest.mark.parametrize("finish,content", [("length", "partial"), ("length", "{}"), ("content_filter", "filtered"), ("stop", "")])
def test_incomplete_response_never_accepted(config_factory, finish, content):
    client = APIClient(config_factory(), "generator", httpx.MockTransport(lambda req: httpx.Response(200, json=envelope(content, finish=finish))))
    with pytest.raises(IncompleteResponse):
        call(client)


def test_no_http_retry_after_exhaustion(config_factory):
    calls = []
    client = APIClient(config_factory(api_max_retries=2), "generator", httpx.MockTransport(lambda req: calls.append(req) or httpx.Response(503)))
    with pytest.raises(APIError, match="exhausted API retries"):
        call(client)
    assert len(calls) == 3


def test_env_relative_paths_and_no_default_limit(monkeypatch, tmp_path):
    for key in ["LIMIT", "OUTPUT_DIR", "INPUT_JSONL", "RUN_ID", "MODE", "GENERATION_JSONL"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    config = Config.from_env()
    assert config.input_jsonl == tmp_path / "input/input_application_clean_v1.jsonl"
    assert config.output_dir == tmp_path / "outputs"
    assert config.limit is None
    monkeypatch.setenv("LIMIT", "0")
    monkeypatch.setenv("GENERATION_JSONL", "outputs/old/generations.jsonl")
    config = Config.from_env()
    assert config.limit == 0
    assert config.generation_jsonl == tmp_path / "outputs/old/generations.jsonl"


@pytest.mark.parametrize("kwargs", [{"limit": -1}, {"batch_size": 0}, {"run_id": "../escape"}, {"generator_model": ""},
                                    {"generator_concurrency": 0}, {"timeout_seconds": 0}, {"judge_json_retries": -1},
                                    {"mode": "judge-only", "generation_jsonl": None}])
def test_invalid_config(config_factory, kwargs):
    with pytest.raises(ValueError):
        config_factory(**kwargs).validate()


def test_run_lock_prevents_concurrent_writers(config_factory):
    config = config_factory()
    with RunStore(config):
        with pytest.raises(ValueError, match="Another process holds run lock"):
            with RunStore(replace(config, resume=True)):
                pass


def test_torn_tail_not_silently_ignored_for_external_files(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"valid":1}\n{"broken"')
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_complete_record_without_final_newline_retained_on_resume(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"valid":1}')
    assert read_jsonl(path, repair_tail=True) == [{"valid": 1}]
    assert path.read_bytes().endswith(b"\n")
