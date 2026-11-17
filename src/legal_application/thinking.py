"""Explicit reasoning controls for the supported self-hosted model templates.

DeepSeek V4.1 vLLM: low=25/high=50/xhigh=75/max=100 or integer 1..100.
https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
These are NOT the hosted DeepSeek API's tier mappings.
"""
from __future__ import annotations

DEEPSEEK_EFFORTS = {"low": 25, "high": 50, "xhigh": 75, "max": 100}


def resolve_thinking(mode: str | int) -> dict:
    if isinstance(mode, bool) or not isinstance(mode, (str, int)):
        raise ValueError("Thinking mode must be a name or integer 1–100")
    value = str(mode).strip().lower()
    backend, enabled, effort, budget, parameters = "server_default", None, None, None, {}
    if value in {"default", "none"}:
        pass  # `none` retains its original meaning in this pipeline: omit parameters.
    elif value in {"qwen3_on", "qwen3_off"}:
        backend, enabled = "qwen3", value == "qwen3_on"
        parameters = {"chat_template_kwargs": {"enable_thinking": enabled}}
    else:
        native = value.removeprefix("deepseek_vllm_")
        backend = "deepseek_vllm"
        if native in {"on", "off"}:
            enabled = native == "on"
            parameters = {"chat_template_kwargs": {"thinking": enabled}}
        else:
            if native in DEEPSEEK_EFFORTS:
                effort, budget = native, DEEPSEEK_EFFORTS[native]
            elif native.isascii() and native.isdecimal() and 1 <= int(native) <= 100:
                effort = budget = int(native)
            else:
                raise ValueError(
                    "Supported thinking modes: default (legacy alias none), off, on, "
                    "low, high, xhigh, max, integer 1–100; deepseek_vllm_<level> aliases; "
                    "qwen3_on/qwen3_off. DeepSeek V4.1 vLLM does not support medium/minimal."
                )
            enabled = True
            parameters = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": budget}}
    return {"requested_mode": mode, "backend": backend, "thinking_enabled": enabled,
            "effort": effort, "effort_budget": budget, "parameters": parameters}


def thinking_parameters(mode: str | int) -> dict:
    return resolve_thinking(mode)["parameters"]
