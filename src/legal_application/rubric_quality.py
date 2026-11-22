"""Independent proposition accounting and the publication gate for rubric drafts."""
from __future__ import annotations

from .data import canonical_hash, text_field
from .rubrics import (AUDIT_SCHEMA, DECOMPOSITION_V2, normalized, require_schema, strict_wire_schema, unique,
                      unit_spans, validate_decomposition, validate_unit)

INVENTORY_SCHEMA = {
    **DECOMPOSITION_V2,
    "properties": {"propositions": DECOMPOSITION_V2["properties"]["units"],
                   "non_reasoning": DECOMPOSITION_V2["properties"]["non_reasoning"]},
    "required": ["propositions", "non_reasoning"],
}
AUDIT_V2 = strict_wire_schema(AUDIT_SCHEMA)


def validate_inventory(inventory, sources, taxonomy):
    require_schema(INVENTORY_SCHEMA, inventory, "independent inventory")
    return validate_decomposition({"units": inventory["propositions"], "non_reasoning": inventory["non_reasoning"]},
                                  sources, taxonomy=taxonomy, prefix="P")


def audit_coverage(inventory, decision, decomposition, audit, sources, taxonomy):
    """Check one-to-one links against a proposition list read independently of the candidate."""
    require_schema(AUDIT_V2, decision, "coverage audit")
    propositions = validate_inventory(inventory, sources, taxonomy)
    for proposition in decision["additional_propositions"]:
        pid = proposition["unit_id"]
        if not pid.startswith("P") or pid in propositions:
            raise ValueError(f"Supplemental proposition must have a fresh P ID: {pid}")
        validate_unit(proposition, sources, taxonomy)
        propositions[pid] = proposition
    unique([normalized(row["reasoning"]) for row in propositions.values()], "inventory propositions")
    units = {unit["unit_id"]: unit for unit in decomposition["units"]}
    candidate_spans = {uid: unit_spans(unit, sources) for uid, unit in units.items()}
    mapped = {row["unit_id"] for row in audit["mapping"]}
    excluded = {row["unit_id"] for row in audit["excluded_units"]}
    coverage_rows = decision["coverage"]
    ids = [row["proposition_id"] for row in coverage_rows]
    unique(ids, "coverage proposition IDs")
    if set(ids) != propositions.keys():
        raise ValueError("Coverage must account for every independent/supplemental proposition exactly once")
    links = [row["unit_id"] for row in coverage_rows if row["unit_id"] is not None]
    unique(links, "proposition links: one unit cannot cover multiple atomic propositions")
    counts = dict.fromkeys(["mapped", "excluded", "missing", "unverified"], 0)
    for row in coverage_rows:
        pid, status, uid = row["proposition_id"], row["status"], row["unit_id"]
        text_field(row["reason"], f"{pid}: coverage reason")
        if status in {"mapped", "excluded"}:
            if uid not in (mapped if status == "mapped" else excluded):
                raise ValueError(f"{pid}: {status} must link to a unit with the corresponding mapping or exclusion")
            if not any(a == b and max(start, left) < min(end, right)
                       and any(c.isalnum() for c in sources[a][max(start, left):min(end, right)])
                       for a, start, end in unit_spans(propositions[pid], sources)
                       for b, left, right in candidate_spans[uid]):
                raise ValueError(f"{pid}: linked unit {uid} must overlap verbatim evidence, not just share a source ID")
        elif uid is not None:
            raise ValueError(f"{pid}: missing or unverified coverage must use unit_id=null")
        counts[status] += 1
    if decision["approved"]:
        if counts["missing"] or counts["unverified"]:
            raise ValueError("Audit cannot approve with missing or unverified propositions")
        if set(links) != mapped | excluded:
            raise ValueError("Audit cannot approve unverified candidate units")
    return counts


def extend_inventory(inventory, additions):
    if not additions:
        return inventory
    assigned = {sid for row in additions for sid in row["source_ids"]}
    return {"propositions": inventory["propositions"] + additions,
            "non_reasoning": [row for row in inventory["non_reasoning"] if row["source_id"] not in assigned]}


def verified_rows(rows):
    """Promote rows whose rubrics passed semantic and coverage validation."""
    verified = []
    for row in rows:
        if row["semantic_audit"]["approved"] is not True:
            raise ValueError("Rubric must pass semantic audit before verified publication")
        draft = row["dataset"]
        dataset = {**draft, "rubrics": {**draft["rubrics"], "status": "verified"}}
        verified.append({**row, "dataset": dataset, "dataset_sha256": canonical_hash(dataset)})
    return verified
