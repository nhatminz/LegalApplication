from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm import tqdm


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f"Non-JSON numeric constant {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)


def text_field(value, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{path}: expected {'nonempty ' if nonempty else ''}string")
    return value


@dataclass
class Sample:
    input_index: int  # zero-based physical JSONL line; input_line = input_index + 1
    raw: dict
    law_blocks: list[dict]

    @property
    def doc_name(self) -> str:
        return self.raw["doc_name"]

    @property
    def issues(self) -> list[str]:
        return self.raw["issues"]

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.raw)

    def identity(self) -> dict:
        return {"doc_name": self.doc_name, "input_index": self.input_index, "input_line": self.input_index + 1,
                "sample_sha256": self.fingerprint}


def parse_sample(raw: object, input_index: int) -> Sample:
    if not isinstance(raw, dict):
        raise ValueError("record must be an object")
    text_field(raw.get("doc_name"), "doc_name")
    seed = raw.get("benchmark_seed")
    if not isinstance(seed, dict):
        raise ValueError("benchmark_seed must be an object")
    for key in ["facts", "facts_clean", "application", "conclusion"]:
        text_field(seed.get(key), f"benchmark_seed.{key}", nonempty=(key == "facts_clean"))
    issues = raw.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError("issues must be a nonempty list")
    for i, issue in enumerate(issues):
        text_field(issue, f"issues[{i}]")
    laws = raw.get("laws")
    if not isinstance(laws, list) or not laws:
        raise ValueError("laws must be a nonempty list")
    blocks = []
    for i, law in enumerate(laws):
        prefix = f"laws[{i}]"
        if not isinstance(law, dict):
            raise ValueError(f"{prefix}: expected object")
        name = text_field(law.get("law_str"), f"{prefix}.law_str")
        provisions = law.get("provisions")
        if provisions is not None and not isinstance(provisions, list):
            raise ValueError(f"{prefix}.provisions: expected list or null")
        if provisions:
            for j, provision in enumerate(provisions):
                loc = f"{prefix}.provisions[{j}]"
                if not isinstance(provision, dict):
                    raise ValueError(f"{loc}: expected object")
                citation = text_field(provision.get("citation_str"), f"{loc}.citation_str")
                content = provision.get("law_content")
                if not isinstance(content, dict):
                    raise ValueError(f"{loc}.law_content: expected object")
                body = text_field(content.get("context_text"), f"{loc}.law_content.context_text")
                blocks.append({"law_index": i, "provision_index": j, "law_str": name, "citation_str": citation,
                               "content": body, "source": f"{loc}.law_content.context_text", "fallback": False})
        else:
            content = law.get("law_content")
            if not isinstance(content, dict):
                raise ValueError(f"{prefix}: no provisions and no law_content fallback")
            key = "context_text" if content.get("context_text") is not None else "exact_text"
            body = text_field(content.get(key), f"{prefix}.law_content.{key}")
            blocks.append({"law_index": i, "provision_index": None, "law_str": name, "citation_str": None,
                           "content": body, "source": f"{prefix}.law_content.{key}", "fallback": True})
    return Sample(input_index, raw, blocks)


class DatasetError(ValueError):
    def __init__(self, errors: list[dict], input_count: int, valid_count: int):
        super().__init__(f"Input validation failed: {len(errors)} error(s); no API requests were made")
        self.errors, self.input_count, self.valid_count = errors, input_count, valid_count


@dataclass
class Dataset:
    samples: list[Sample]
    input_count: int
    sha256: str
    status_counts: dict[str, int] = field(default_factory=dict)
    eligible_count: int = 0
    filtered_out: int = 0
    valid_count: int = 0
    validation_errors: list[dict] = field(default_factory=list)


def read_dataset(path: Path, limit: int | None, data_status_used: str | None = None, *,
                 skip_invalid: bool = False, on_invalid: Callable[[dict, bytes], None] | None = None) -> Dataset:
    """Validate every line, including those beyond LIMIT; retain only selected samples."""
    selected, errors, seen = [], [], set()
    count = valid = 0
    status_counts = {"full": 0, "partial": 0}
    eligible = filtered_out = 0
    if data_status_used not in {None, "all", "full_only"}:
        raise ValueError("DATA_STATUS_USED must be all or full_only")
    digest = hashlib.sha256()
    with path.open("rb") as handle, tqdm(total=path.stat().st_size, desc="Validate input", unit="B",
                                         unit_scale=True, dynamic_ncols=True, disable=None) as progress:
        for index, line in enumerate(handle):
            count += 1
            digest.update(line)
            raw = None
            validation_stage = "json"
            try:
                raw = strict_json(line.decode("utf-8"))
                validation_stage = "schema"
                sample = parse_sample(raw, index)
                if sample.doc_name in seen:
                    validation_stage = "duplicate_doc_name"
                    raise ValueError(f"duplicate doc_name {sample.doc_name!r}")
                seen.add(sample.doc_name)
                status = sample.raw.get("data_status")
                if data_status_used is not None and (not isinstance(status, str) or status not in status_counts):
                    raise ValueError("data_status must be full or partial; run scripts/statistic.sh first")
                if isinstance(status, str) and status in status_counts:
                    status_counts[status] += 1
                valid += 1
                included = data_status_used != "full_only" or status == "full"
                eligible += int(included)
                filtered_out += int(not included)
                if included and (limit is None or len(selected) < limit):
                    selected.append(sample)
            except (ValueError, UnicodeError) as exc:
                error = {"input_index": index, "input_line": index + 1, "error": str(exc),
                         "validation_stage": validation_stage, "error_type": type(exc).__name__,
                         "doc_name": raw.get("doc_name") if isinstance(raw, dict) and isinstance(raw.get("doc_name"), str) else None}
                errors.append(error)
                if on_invalid is not None:
                    on_invalid(error, line)
            progress.set_postfix(valid=valid, errors=len(errors), refresh=False)
            progress.update(len(line))
    if not count:
        errors.append({"input_line": None, "error": "Input JSONL is empty"})
    if errors and (not skip_invalid or not valid):
        raise DatasetError(errors, count, valid)
    return Dataset(selected, count, digest.hexdigest(), status_counts, eligible, filtered_out, valid, errors)

