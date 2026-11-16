from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo


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
    judge_response_format: str = "json_schema"
    generator_api_key: str = field(default="EMPTY", repr=False)
    judge_api_key: str = field(default="EMPTY", repr=False)

    @classmethod
    def from_env(cls) -> Config:
        e = os.environ
        root = Path(e.get("PROJECT_ROOT", "/workspace/storage-shared/nlp/minhpn19/LegalApplication")).expanduser().resolve()
        run_id = e.get("RUN_ID") or datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        limit = e.get("LIMIT", "").strip()
        source = e.get("GENERATION_JSONL", "")
        return cls(
            project_root=root,
            input_jsonl=rooted(root, e.get("INPUT_JSONL", "input/judgment_with_issues_sample.jsonl")),
            output_dir=rooted(root, e.get("OUTPUT_DIR", "outputs")),
            run_id=run_id,
            generator_base_url=e.get("GENERATOR_BASE_URL", ""), generator_model=e.get("GENERATOR_MODEL", ""),
            judge_base_url=e.get("JUDGE_BASE_URL", "http://172.29.62.173:8000/v1"),
            judge_model=e.get("JUDGE_MODEL", "DeepSeek-V4.1-Flash-1017"),
            mode=e.get("MODE", "full"), generation_jsonl=rooted(root, source) if source else None,
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
            judge_response_format=e.get("JUDGE_RESPONSE_FORMAT", "json_schema"),
            generator_api_key=e.get("GENERATOR_API_KEY", "EMPTY"), judge_api_key=e.get("JUDGE_API_KEY", "EMPTY"),
        )

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("RUN_ID must be a simple filename (letters, numbers, _, ., -)")
        if self.mode not in {"full", "judge-only", "validate-only"}:
            raise ValueError("MODE must be full, judge-only, or validate-only")
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
        if self.generator_thinking_mode not in {"none", "deepseek_vllm_xhigh"}:
            raise ValueError("GENERATOR_THINKING_MODE must be none or deepseek_vllm_xhigh")
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
        for key in ["generator_api_key", "judge_api_key", "resume"]:
            result.pop(key)
        return {k: str(v) if isinstance(v, Path) else v for k, v in result.items()}

