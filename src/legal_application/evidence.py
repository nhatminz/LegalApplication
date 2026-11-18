"""Source-backed evidence IDs for completeness; source text is never rewritten."""
from __future__ import annotations

import re
from copy import deepcopy

from .data import Sample
from .scoring import LabelError

EVIDENCE_FORMAT = "source_ids_v1"
SOURCES = ("facts_clean", "facts", "application", "conclusion")


def evidence_catalog(sample: Sample) -> dict[str, dict]:
    """Partition every source into contiguous spans. 500 is a span size, not an input cap."""
    catalog = {}
    for source in SOURCES:
        text = sample.raw["benchmark_seed"][source]
        start, number = 0, 1
        while start < len(text):
            end = min(start + 500, len(text))
            if end < len(text):
                window = text[start:end]
                # Prefer sentence/paragraph boundaries, then word boundaries.
                breaks = list(re.finditer(r"[.!?;](?=\s)|\n+", window))
                if not breaks:
                    breaks = list(re.finditer(r"\s+", window))
                if breaks:
                    end = start + breaks[-1].end()
            identifier = f"{source}:{number:06d}"
            catalog[identifier] = {"source": source, "start": start, "end": end, "text": text[start:end]}
            start, number = end, number + 1
    return catalog


def evidence_schema(schema: dict, catalog: dict[str, dict]) -> dict:
    """Constrain model output to known source IDs using ordinary JSON Schema enums."""
    wire = deepcopy(schema)
    wire["required"].append("evidence_format")
    wire["properties"]["evidence_format"] = {"enum": [EVIDENCE_FORMAT]}
    fact = wire["properties"]["issues_analysis"]["items"]["properties"]["decisive_facts"]["items"]["properties"]

    def ids(sources):
        return [key for key, span in catalog.items() if span["source"] in sources and span["text"].strip()]

    reference = fact["reference_evidence"]["items"]
    reference["required"] = ["source", "quote_id"]
    reference["properties"].pop("quote")
    reference["properties"]["quote_id"] = {"type": "string", "minLength": 1}
    references = ids({"application", "conclusion"})
    if references:
        reference["properties"]["quote_id"]["enum"] = references
    for field, source in [("facts_clean_evidence", "facts_clean"), ("raw_facts_evidence", "facts")]:
        choices = ids({source})
        if choices:
            fact[field]["items"]["enum"] = choices
        else:
            fact[field]["maxItems"] = 0
    return wire


def materialize_evidence(labels: dict, catalog: dict[str, dict]) -> tuple[dict, list[dict]]:
    """Expand model-selected IDs; never guess an ID or replace an invalid quote with a candidate."""
    labels = deepcopy(labels)
    labels.pop("evidence_format")
    resolved = []

    def resolve(identifier, source, path):
        if not isinstance(identifier, str) or identifier not in catalog:
            raise LabelError(f"{path}: unknown evidence ID {identifier!r}; select an ID from the supplied catalog")
        span = catalog[identifier]
        if span["source"] != source:
            raise LabelError(f"{path}: evidence ID {identifier!r} belongs to {span['source']}, not {source}")
        if not span["text"].strip():
            raise LabelError(f"{path}: evidence ID points to whitespace only")
        resolved.append({"path": path, "kind": "source_reference", "id": identifier,
                         "source": source, "start": span["start"], "end": span["end"]})
        return span["text"]

    for i, row in enumerate(labels["issues_analysis"]):
        for j, fact in enumerate(row["decisive_facts"]):
            path = f"issues_analysis[{i}].decisive_facts[{j}]"
            for k, reference in enumerate(fact["reference_evidence"]):
                reference["quote"] = resolve(reference.pop("quote_id"), reference["source"], f"{path}.reference_evidence[{k}].quote_id")
            for field, source in [("facts_clean_evidence", "facts_clean"), ("raw_facts_evidence", "facts")]:
                fact[field] = [resolve(identifier, source, f"{path}.{field}[{k}]") for k, identifier in enumerate(fact[field])]
    return labels, resolved
