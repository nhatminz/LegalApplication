from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from legal_application.api import APIClient
from legal_application.thinking import resolve_thinking, thinking_parameters
from conftest import envelope


@pytest.mark.parametrize("mode,budget", [
    ("low", 25), ("high", 50), ("xhigh", 75), ("max", 100),
    ("deepseek_vllm_low", 25), ("deepseek_vllm_high", 50),
    ("deepseek_vllm_xhigh", 75), ("deepseek_vllm_max", 100),
    ("1", 1), ("63", 63), ("100", 100), ("deepseek_vllm_63", 63),
    (1, 1), (100, 100), (" HIGH ", 50),
])
@pytest.mark.parametrize("role", ["generator", "judge"])
def test_all_native_tiers_and_numeric_effort_reach_request(config_factory, role, mode, budget):
    config = config_factory(**{f"{role}_thinking_mode": mode})
    config.validate()

    def handle(request):
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": budget}
        assert "reasoning_effort" not in body  # Use numeric template effort, not hosted API tiers.
        assert ("response_format" in body) == (role == "judge")
        return httpx.Response(200, json=envelope("{}"))

    async def invoke():
        client = APIClient(config, role, httpx.MockTransport(handle))
        try:
            completion = await client.complete([], doc_name="mock-case")
            assert completion.metadata["actual_thinking_parameters"] == {
                "chat_template_kwargs": {"thinking": True, "reasoning_effort": budget}}
        finally:
            await client.close()

    asyncio.run(invoke())


@pytest.mark.parametrize("mode,parameters", [
    ("default", {}), ("none", {}),
    ("off", {"thinking": False}), ("on", {"thinking": True}),
    ("deepseek_vllm_off", {"thinking": False}), ("deepseek_vllm_on", {"thinking": True}),
    ("qwen3_on", {"enable_thinking": True}), ("qwen3_off", {"enable_thinking": False}),
])
def test_default_and_explicit_model_switches(config_factory, mode, parameters):
    config = config_factory(generator_thinking_mode=mode, judge_thinking_mode=mode)
    config.validate()
    expected = {"chat_template_kwargs": parameters} if parameters else {}
    assert thinking_parameters(mode) == expected
    assert resolve_thinking(mode)["effort_budget"] is None


@pytest.mark.parametrize("mode", ["minimal", "medium", "deepseek_vllm_medium", "0", "101", "-1", "25.5", "", True, None, 1.5, "qwen3_high"])
@pytest.mark.parametrize("role", ["generator", "judge"])
def test_unsupported_modes_fail_early_without_guessing(config_factory, role, mode):
    with pytest.raises(ValueError, match=f"{role.upper()}_THINKING_MODE"):
        config_factory(**{f"{role}_thinking_mode": mode}).validate()


def test_every_integer_budget_and_independent_parameter_dicts():
    for budget in range(1, 101):
        assert thinking_parameters(str(budget)) == {"chat_template_kwargs": {"thinking": True, "reasoning_effort": budget}}
    a = thinking_parameters("xhigh")
    a["chat_template_kwargs"]["reasoning_effort"] = 1
    assert thinking_parameters("xhigh")["chat_template_kwargs"]["reasoning_effort"] == 75
