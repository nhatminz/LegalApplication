from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from .thinking import resolve_thinking


def rooted(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def flag(value: str) -> bool:
    if value.lower() not in {"0", "1", "true", "false"}:
        raise ValueError(f"Expected boolean 0/1/true/false, got {value!r}")
    return value.lower() in {"1", "true"}


@dataclass(frozen=True)
class Config:
    project_root: Path
    input_jsonl: Path
    output_dir: Path
    run_id: str
    generator_base_url: str = ""
    generator_model: str = ""
    judge_base_url: str = "http://172.29.62.173:8000/v1"
    judge_model: str = "DeepSeek-V4.1-Flash-1017"
    mode: str = "full"
    generation_jsonl: Path | None = None
    limit: int | None = None
    batch_size: int = 20
    generator_concurrency: int = 20
    judge_concurrency: int = 20
    max_output_tokens: int = 32768
    timeout_seconds: float = 1800
    api_max_retries: int = 3
    judge_json_retries: int = 2
    retry_backoff: float = 2
    resume: bool = False
    generator_thinking_mode: str = "none"
    judge_thinking_mode: str = "deepseek_vllm_xhigh"
    judge_response_format: str = "json_schema"
    statistic_seed_jsonl: Path | None = None
    generator_api_key: str = field(default="EMPTY", repr=False)
    judge_api_key: str = field(default="EMPTY", repr=False)
    evaluation_profile: str = "rubric"
    expected_count: int | None = None
    final_application_jsonl: Path | None = None
    final_issue_spotting_jsonl: Path | None = None
    rubric_base_url: str = "http://172.29.116.48:8000/v1"
    rubric_model: str = "DeepSeek-V4.1-Flash-1017"
    rubric_api_key: str = field(default="EMPTY", repr=False)
    rubric_thinking_mode: str = "deepseek_vllm_xhigh"
    rubric_concurrency: int = 20
    rubric_max_output_tokens: int = 65536
    rubric_json_retries: int = 2
    rubric_response_format: str = "json_schema"

    @classmethod
    def from_env(cls) -> Config:
        e = os.environ
        root = Path(e.get("PROJECT_ROOT", "/workspace/storage-shared/nlp/users/minhpn19/LegalApplication")).expanduser().resolve()
        run_id = e.get("RUN_ID") or datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        limit = e.get("LIMIT", "").strip()
        source = e.get("GENERATION_JSONL", "")
        seed = e.get("STATISTIC_SEED_JSONL", "")
        mode = e.get("MODE", "full")
        profile = e.get("EVALUATION_PROFILE", "rubric")
        final_application = rooted(root, e.get("FINAL_APPLICATION_JSONL", "input/final/final_600_rule_application.jsonl"))
        default_input = {"statistic": "input/judgment_with_issues_sample.jsonl",
                         "build-rubrics": "input/final_raw_600_input_application.jsonl"}.get(mode)
        if default_input is None:
            default_input = str(final_application) if profile == "rubric" else "input/input_application_clean_v1.jsonl"
        input_key = "RAW_INPUT_JSONL" if mode == "build-rubrics" else "INPUT_JSONL"
        return cls(
            project_root=root,
            input_jsonl=rooted(root, e.get(input_key, default_input)),
            output_dir=rooted(root, e.get("OUTPUT_DIR", "outputs")),
            run_id=run_id,
            generator_base_url=e.get("GENERATOR_BASE_URL", ""), generator_model=e.get("GENERATOR_MODEL", ""),
            judge_base_url=e.get("JUDGE_BASE_URL", "http://172.29.62.173:8000/v1"),
            judge_model=e.get("JUDGE_MODEL", "DeepSeek-V4.1-Flash-1017"),
            mode=mode, generation_jsonl=rooted(root, source) if source else None,
            limit=None if limit.lower() in {"", "all", "none"} else int(limit),
            batch_size=int(e.get("BATCH_SIZE", "20")),
            generator_concurrency=int(e.get("GENERATOR_CONCURRENCY", "20")),
            judge_concurrency=int(e.get("JUDGE_CONCURRENCY", "20")),
            max_output_tokens=int(e.get("MAX_OUTPUT_TOKENS", "32768")),
            timeout_seconds=float(e.get("API_TIMEOUT_SECONDS", "1800")),
            api_max_retries=int(e.get("API_MAX_RETRIES", "3")),
            judge_json_retries=int(e.get("JUDGE_JSON_RETRIES", "2")),
            retry_backoff=float(e.get("RETRY_BACKOFF_SECONDS", "2")),
            resume=flag(e.get("RESUME", "0")),
            generator_thinking_mode=e.get("GENERATOR_THINKING_MODE", "none"),
            judge_thinking_mode=e.get("JUDGE_THINKING_MODE", "deepseek_vllm_xhigh"),
            judge_response_format=e.get("JUDGE_RESPONSE_FORMAT", "json_schema"),
            statistic_seed_jsonl=rooted(root, seed) if seed else None,
            generator_api_key=e.get("GENERATOR_API_KEY", "EMPTY"), judge_api_key=e.get("JUDGE_API_KEY", "EMPTY"),
            evaluation_profile=profile,
            expected_count=int(e["EXPECTED_COUNT"]) if e.get("EXPECTED_COUNT") else None,
            final_application_jsonl=final_application if mode == "build-rubrics" else None,
            final_issue_spotting_jsonl=rooted(root, e.get("FINAL_ISSUE_SPOTTING_JSONL", "input/final/final_600_issue_spotting.jsonl")) if mode == "build-rubrics" else None,
            rubric_base_url=e.get("RUBRIC_BASE_URL", "http://172.29.116.48:8000/v1"),
            rubric_model=e.get("RUBRIC_MODEL", "DeepSeek-V4.1-Flash-1017"),
            rubric_api_key=e.get("RUBRIC_API_KEY", "EMPTY"),
            rubric_thinking_mode=e.get("RUBRIC_THINKING_MODE", "deepseek_vllm_xhigh"),
            rubric_concurrency=int(e.get("RUBRIC_CONCURRENCY", "20")),
            rubric_max_output_tokens=int(e.get("RUBRIC_MAX_OUTPUT_TOKENS", "65536")),
            rubric_json_retries=int(e.get("RUBRIC_JSON_RETRIES", "2")),
            rubric_response_format=e.get("RUBRIC_RESPONSE_FORMAT", "json_schema"),
        )

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("RUN_ID must be a simple filename (letters, numbers, _, ., -)")
        if self.mode not in {"full", "judge-only", "validate-only", "statistic", "build-rubrics"}:
            raise ValueError("MODE must be full, judge-only, validate-only, statistic, or build-rubrics")
        if self.evaluation_profile not in {"rubric", "legacy"}:
            raise ValueError("EVALUATION_PROFILE must be rubric or legacy")
        if self.expected_count is not None and self.expected_count <= 0:
            raise ValueError("EXPECTED_COUNT must be positive")
        if self.mode == "build-rubrics":
            if not self.final_application_jsonl or not self.final_issue_spotting_jsonl:
                raise ValueError("Both final dataset paths are required")
            if not self.rubric_model.strip() or not re.fullmatch(r"https?://[^\s]+", self.rubric_base_url):
                raise ValueError("Set a valid RUBRIC_MODEL and RUBRIC_BASE_URL")
            resolve_thinking(self.rubric_thinking_mode)
            if min(self.rubric_concurrency, self.rubric_max_output_tokens) <= 0 or self.rubric_json_retries < 0:
                raise ValueError("Invalid rubric concurrency, token budget or retry count")
            if self.rubric_response_format not in {"json_schema", "json_object"}:
                raise ValueError("RUBRIC_RESPONSE_FORMAT must be json_schema or json_object")
        if self.statistic_seed_jsonl is not None and self.mode != "statistic":
            raise ValueError("STATISTIC_SEED_JSONL is only supported in statistic mode")
        if self.mode == "full" and not (self.generator_model.strip() and self.generator_base_url):
            raise ValueError("GENERATOR_MODEL and GENERATOR_BASE_URL are required in full mode")
        if self.mode == "judge-only" and self.generation_jsonl is None:
            raise ValueError("GENERATION_JSONL is required in judge-only mode")
        for name in ["batch_size", "generator_concurrency", "judge_concurrency", "max_output_tokens", "timeout_seconds"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ["api_max_retries", "judge_json_retries", "retry_backoff"]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.limit is not None and self.limit < 0:
            raise ValueError("LIMIT must be nonnegative, or empty/all for the entire input")
        for role in ["generator", "judge"]:
            try:
                resolve_thinking(getattr(self, f"{role}_thinking_mode"))
            except ValueError as exc:
                raise ValueError(f"{role.upper()}_THINKING_MODE: {exc}") from None
        if self.judge_response_format not in {"json_schema", "json_object"}:
            raise ValueError("JUDGE_RESPONSE_FORMAT must be json_schema or json_object")
        for role in ["generator", "judge"]:
            base = getattr(self, f"{role}_base_url")
            if base and not re.match(r"^https?://[^\s]+$", base):
                raise ValueError(f"Invalid {role} base URL")
        if not self.judge_model.strip():
            raise ValueError("JUDGE_MODEL is required")

    def snapshot(self) -> dict:
        result = asdict(self)
        for key in ["generator_api_key", "judge_api_key", "rubric_api_key", "resume"]:
            result.pop(key)
        if self.mode != "build-rubrics":
            result = {key: value for key, value in result.items()
                      if not key.startswith(("rubric_", "final_"))}
        return {k: str(v) if isinstance(v, Path) else v for k, v in result.items()}

