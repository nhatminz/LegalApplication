"""Export proven-full records as their original JSONL bytes, with a checksum manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

from .data import canonical_hash, parse_sample, strict_json
from .storage import append_jsonl, atomic_json, file_hash, now

ID_FIELDS = ("document_id", "doc_name")


def identifiers(row: dict) -> set[tuple[str, str]]:
    result = set()
    for key in ID_FIELDS:
        value = row.get(key)
        if (isinstance(value, str) and value.strip()) or type(value) is int:
            result.add((key, str(value)))
    return result


def manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def verify_full_manifest(path: Path) -> dict | None:
    manifest = manifest_path(path)
    if not manifest.exists():
        return None
    value = strict_json(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "full_application_input" or value.get("version") != 1:
        raise ValueError(f"Invalid full-input manifest: {manifest}")
    if value.get("output_sha256") != file_hash(path):
        raise ValueError("Full-input manifest checksum mismatch; run scripts/filter_full.sh again")
    if type(value.get("sample_count")) is not int or value["sample_count"] <= 0:
        raise ValueError("Full-input manifest has no selected full records")
    return value


def export_full(input_path: Path, classifications: list[Path], output_path: Path) -> dict:
    input_path, output_path = input_path.resolve(), output_path.resolve()
    classifications = [path.resolve() for path in classifications]
    protected = [input_path, *classifications]
    if output_path in protected or (output_path.exists() and any(output_path.samefile(p) for p in protected)):
        raise ValueError("Filtered output must not overwrite input or classification details")
    if not classifications:
        raise ValueError("Provide at least one classification_details.jsonl")
    lookup, records, details_hashes = {}, [], {}
    for path in classifications:
        details_hashes[str(path)] = file_hash(path)
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                row = strict_json(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: classification must be an object")
                status = row.get("data_status", row.get("status"))
                if row.get("classification_status", "completed") != "completed" or not isinstance(status, str) or status not in {"full", "partial"}:
                    raise ValueError(f"{path}:{line_number}: expected a completed full/partial classification")
                aliases = identifiers(row)
                if not aliases:
                    raise ValueError(f"{path}:{line_number}: missing document_id/doc_name")
                for alias in aliases:
                    for previous in lookup.get(alias, []):
                        other = records[previous]
                        if (other["status"] != status or
                            any(key in row and key in other["row"] and row[key] != other["row"][key]
                                for key in [*ID_FIELDS, "sample_sha256"])):
                            raise ValueError(f"Conflicting classifications for {alias}")
                index = len(records)
                records.append({"row": row, "status": status})
                for alias in aliases:
                    lookup.setdefault(alias, []).append(index)
    full_indices = {i for i, value in enumerate(records) if value["status"] == "full"}
    if not full_indices:
        raise ValueError("No completed full classifications; no output was replaced")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output_path.with_name(output_path.name + ".filter_log.jsonl")
    export_id = uuid4().hex
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    selected = invalid = total = 0
    matched, seen_aliases, doc_names = set(), set(), []
    temporary = None
    try:
        with input_path.open("rb") as source, tempfile.NamedTemporaryFile(
            mode="wb", prefix=".full-input-", dir=output_path.parent, delete=False
        ) as target, tqdm(total=input_path.stat().st_size, desc="Filter full", unit="B", unit_scale=True,
                          dynamic_ncols=True, disable=None) as progress:
            temporary = Path(target.name)
            for index, line in enumerate(source):
                total += 1
                source_hash.update(line)
                progress.update(len(line))
                try:
                    raw = strict_json(line.decode("utf-8"))
                    sample = parse_sample(raw, index)
                except (ValueError, UnicodeError) as exc:
                    invalid += 1
                    append_jsonl(audit_path, {"export_id": export_id, "input_line": index + 1,
                                             "error": str(exc), "action": "skip_invalid", "timestamp": now()})
                    continue
                aliases = identifiers(raw)
                matches = {i for alias in aliases for i in lookup.get(alias, [])}
                if not matches:
                    continue  # Failed/unclassified samples never become full by default.
                statuses = {records[i]["status"] for i in matches}
                if len(statuses) != 1:
                    raise ValueError(f"Source line {index + 1}: document_id/doc_name have conflicting labels")
                for i in matches:
                    label = records[i]["row"]
                    if any(key in label and key in raw and str(label[key]) != str(raw[key]) for key in ID_FIELDS):
                        raise ValueError(f"Source line {index + 1}: document_id/doc_name mismatch")
                    if label.get("sample_sha256") and label["sample_sha256"] != canonical_hash(raw):
                        raise ValueError(f"Source line {index + 1}: sample checksum differs from classification")
                if statuses != {"full"}:
                    continue
                if "data_status" in raw and raw["data_status"] != "full":
                    raise ValueError(f"Source line {index + 1}: source data_status contradicts full classification")
                if aliases & seen_aliases:
                    raise ValueError(f"Duplicate full document at source line {index + 1}")
                seen_aliases.update(aliases)
                matched.update(matches)
                doc_names.append(sample.doc_name)
                target.write(line)  # Do not deserialize/re-serialize or add data_status.
                output_hash.update(line)
                selected += 1
                progress.set_postfix(full=selected, invalid=invalid, refresh=False)
            target.flush()
            os.fsync(target.fileno())
        if full_indices - matched:
            raise ValueError(f"{len(full_indices - matched)} full classifications were not found in the source input")
        if file_hash(input_path) != source_hash.hexdigest():
            raise ValueError("Source input changed during filtering; no output was replaced")
        if any(file_hash(Path(path)) != digest for path, digest in details_hashes.items()):
            raise ValueError("Classification details changed during filtering; stop the statistic run first")
        manifest = {"kind": "full_application_input", "version": 1, "timestamp": now(), "export_id": export_id,
                    "source_input": str(input_path), "source_sha256": source_hash.hexdigest(),
                    "classification_sha256": details_hashes, "output_sha256": output_hash.hexdigest(),
                    "sample_count": selected, "source_count": total, "source_invalid": invalid,
                    "full_doc_names": doc_names, "id_fields": list(ID_FIELDS)}
        os.replace(temporary, output_path)
        atomic_json(manifest_path(output_path), manifest)
        append_jsonl(audit_path, {**manifest, "action": "exported", "output": str(output_path)})
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=os.environ.get("INPUT_JSONL"), required=not os.environ.get("INPUT_JSONL"))
    parser.add_argument("--classifications", type=Path, action="append", help="Repeat to combine multiple statistic runs")
    parser.add_argument("--output", type=Path, default=os.environ.get("FILTERED_JSONL"), required=not os.environ.get("FILTERED_JSONL"))
    args = parser.parse_args()
    paths = args.classifications
    if not paths:
        if not os.environ.get("CLASSIFICATION_DETAILS"):
            parser.error("Provide --classifications or CLASSIFICATION_DETAILS")
        paths = [Path(os.environ["CLASSIFICATION_DETAILS"])]
    try:
        manifest = export_full(args.input, paths, args.output)
    except (ValueError, OSError) as exc:
        print(f"Filter aborted: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "full": manifest["sample_count"], "invalid": manifest["source_invalid"],
                      "manifest": str(manifest_path(args.output))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
