"""Export aligned issue spotting, application and conclusion datasets without API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from tqdm import tqdm

from legal_application.data import Sample, parse_sample, rule_inputs, strict_json
from legal_application.storage import file_hash

FILENAMES = {
    "issue_spotting": "input_issue_spotting.jsonl",
    "application": "input_application.jsonl",
    "conclusion": "input_conclusion.jsonl",
}


def task_records(sample: Sample) -> dict[str, dict]:
    seed = sample.raw["benchmark_seed"]
    rules = rule_inputs(sample)
    application_input = {"facts_clean": seed["facts_clean"], "issues": sample.issues, "rules": rules}
    return {
        "issue_spotting": {"input": seed["facts_clean"], "expected_answer": sample.issues},
        "application": {"input": application_input, "expected_answer": seed["application"]},
        "conclusion": {"input": {**application_input, "application": seed["application"]}, "expected_answer": seed["conclusion"]},
    }


def export_task_inputs(input_path: Path, output_dir: Path, *, expected_count: int | None = None) -> dict:
    input_path, output_dir = input_path.resolve(), output_dir.resolve()
    if expected_count is not None and (type(expected_count) is not int or expected_count <= 0):
        raise ValueError("--expected-count must be a positive integer")
    targets = [output_dir / name for name in [*FILENAMES.values(), "manifest.json"]]
    if input_path in [path.resolve() for path in targets]:
        raise ValueError("Output must not overwrite the source input")
    if any(path.exists() and input_path.samefile(path) for path in targets):
        raise ValueError("Output must not share a file with the source input")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    output_hashes = {task: hashlib.sha256() for task in FILENAMES}
    count, seen, mapping = 0, {}, []
    fallback_rules = 0
    # Validate and finish all three files before replacing any published output.
    with tempfile.TemporaryDirectory(prefix=".task-inputs-", dir=output_dir) as staging:
        stage = Path(staging)
        with ExitStack() as stack:
            writers = {task: stack.enter_context((stage / name).open("wb")) for task, name in FILENAMES.items()}
            source = stack.enter_context(input_path.open("rb"))
            progress = stack.enter_context(tqdm(total=os.fstat(source.fileno()).st_size, desc="Create task inputs", unit="B",
                                               unit_scale=True, dynamic_ncols=True, disable=None))
            for index, line in enumerate(source):
                source_hash.update(line)
                progress.update(len(line))
                try:
                    raw = strict_json(line.decode("utf-8"))
                    sample = parse_sample(raw, index)
                    if "data_status" in raw and raw["data_status"] != "full":
                        raise ValueError("Expected the full-only source; data_status must be full when present")
                    if sample.doc_name in seen:
                        raise ValueError(f"duplicate doc_name {sample.doc_name!r}; first seen at line {seen[sample.doc_name]}")
                    seen[sample.doc_name] = index + 1
                    for task, record in task_records(sample).items():
                        encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                        writers[task].write(encoded)
                        output_hashes[task].update(encoded)
                except ValueError as exc:
                    raise ValueError(f"{input_path}:{index + 1}: {exc}") from None
                count += 1
                fallback_rules += sum(block["fallback"] for block in sample.law_blocks)
                identity = {"output_line": count, "input_line": index + 1, "doc_name": sample.doc_name}
                if "document_id" in raw:
                    identity["document_id"] = raw["document_id"]
                mapping.append(identity)
                progress.set_postfix(samples=count, refresh=False)
            if not count:
                raise ValueError("Input JSONL is empty")
            if expected_count is not None and count != expected_count:
                raise ValueError(f"Expected {expected_count} samples, found {count}; no output files were replaced")
            for writer in writers.values():
                writer.flush()
                os.fsync(writer.fileno())
        if file_hash(input_path) != source_hash.hexdigest():
            raise ValueError("Source input changed during export; no output files were replaced")
        manifest = {"version": 1, "source_input": str(input_path), "source_sha256": source_hash.hexdigest(),
                    "sample_count": count, "facts_source": "benchmark_seed.facts_clean", "fallback_rule_count": fallback_rules,
                    "files": {task: {"name": name, "sample_count": count, "sha256": output_hashes[task].hexdigest()}
                              for task, name in FILENAMES.items()},
                    "samples": mapping}
        with (stage / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for target in targets:
            os.replace(stage / target.name, target)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=os.environ.get("TASK_SOURCE_JSONL"),
                        required=not os.environ.get("TASK_SOURCE_JSONL"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("TASK_INPUT_DIR"),
                        required=not os.environ.get("TASK_INPUT_DIR"))
    parser.add_argument("--expected-count", type=int, help="Require this count; never limit or silently drop samples")
    args = parser.parse_args()
    try:
        manifest = export_task_inputs(args.input, args.output_dir, expected_count=args.expected_count)
    except (ValueError, OSError) as exc:
        print(f"Task input export aborted: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"samples": manifest["sample_count"], "output_dir": str(args.output_dir.resolve()),
                      "files": [item["name"] for item in manifest["files"].values()]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
