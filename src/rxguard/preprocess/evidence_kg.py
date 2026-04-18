from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_KEYWORDS = (
    "translation",
    "permuted",
    "mapped",
    "isa",
    "inverse_isa",
    "same_as",
    "classified_as",
    "sibling",
    "child",
    "parent",
)

DEFAULT_INCLUDE_KEYWORDS = (
    "treat",
    "prevent",
    "manifestation",
    "finding",
    "cause",
    "associated",
    "ingredient",
    "mechanism",
    "site",
    "location",
    "contraind",
    "effect",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a UMLS evidence KG scoped to RxGuard concept nodes."
    )
    parser.add_argument("--patients-jsonl", type=Path, required=True)
    parser.add_argument("--mrrel-path", type=Path, required=True)
    parser.add_argument("--relation-inventory-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--relation-allowlist", type=Path, default=None)
    parser.add_argument("--relation-blocklist", type=Path, default=None)
    parser.add_argument("--include-relation", action="append", default=None)
    parser.add_argument("--exclude-relation", action="append", default=None)
    parser.add_argument("--min-relation-count", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke limit over MRREL rows.")
    return parser.parse_args()


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def _relation_name(record: dict) -> str:
    rela = (record.get("raw_relation") or "").strip()
    if rela:
        return rela
    samples = record.get("samples") or []
    if samples:
        sample = samples[0]
        if sample.get("rela"):
            return sample["rela"]
        if sample.get("rel"):
            return sample["rel"]
    return ""


def _default_relation_filter(record: dict, min_relation_count: int) -> bool:
    relation_name = _relation_name(record).lower()
    count = int(record.get("count", 0))
    if count < min_relation_count:
        return False
    if any(keyword in relation_name for keyword in DEFAULT_EXCLUDE_KEYWORDS):
        return False
    return any(keyword in relation_name for keyword in DEFAULT_INCLUDE_KEYWORDS)


def _select_relations(
    inventory: dict,
    relation_allowlist: list[str] | None,
    relation_blocklist: list[str] | None,
    include_relations: list[str] | None,
    exclude_relations: list[str] | None,
    min_relation_count: int,
) -> tuple[list[str], list[dict]]:
    relations = inventory.get("relations", [])
    selected = []
    selected_records = []

    explicit_allowlist = set(relation_allowlist or [])
    explicit_blocklist = set(relation_blocklist or [])
    explicit_include = set(include_relations or [])
    explicit_exclude = set(exclude_relations or [])

    for record in relations:
        relation_name = _relation_name(record)
        if not relation_name:
            continue
        if relation_name in explicit_blocklist or relation_name in explicit_exclude:
            continue
        if explicit_allowlist and relation_name not in explicit_allowlist:
            continue
        if explicit_include:
            if relation_name not in explicit_include:
                continue
        elif not _default_relation_filter(record, min_relation_count=min_relation_count):
            continue
        selected.append(relation_name)
        selected_records.append(
            {
                "relation": relation_name,
                "count": int(record.get("count", 0)),
                "sample": (record.get("samples") or [None])[0],
            }
        )

    return _unique_sorted(selected), selected_records


def _load_relation_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    values = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values.append(stripped)
    return _unique_sorted(values)


def _collect_scope_nodes(patients_jsonl: Path) -> tuple[set[str], dict[str, str]]:
    scope_nodes: set[str] = set()
    medication_rxcui_to_umls: dict[str, str] = {}
    with patients_jsonl.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for visit in row.get("visits", []):
                scope_nodes.update(visit.get("diagnoses", []))
                scope_nodes.update(visit.get("procedures", []))
                scope_nodes.update(visit.get("medication_umls_cuis", []))
                medication_rxcui_to_umls.update(visit.get("medication_rxcui_to_umls", {}))
    return scope_nodes, medication_rxcui_to_umls


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(args.relation_inventory_json.expanduser().resolve().read_text(encoding="utf-8"))
    relation_allowlist = _load_relation_list(args.relation_allowlist)
    relation_blocklist = _load_relation_list(args.relation_blocklist)
    allowed_relations, relation_records = _select_relations(
        inventory=inventory,
        relation_allowlist=relation_allowlist,
        relation_blocklist=relation_blocklist,
        include_relations=args.include_relation,
        exclude_relations=args.exclude_relation,
        min_relation_count=args.min_relation_count,
    )
    allowed_relation_set = set(allowed_relations)

    scope_nodes, medication_rxcui_to_umls = _collect_scope_nodes(args.patients_jsonl)
    edge_rows = []
    seen_edges = set()
    stats = Counter()

    with args.mrrel_path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if args.max_rows is not None and row_index >= args.max_rows:
                break
            parts = line.rstrip("\n").split("|")
            if len(parts) < 11:
                stats["malformed_rows"] += 1
                continue

            cui1, rel, cui2 = parts[0], parts[3], parts[4]
            rela = parts[7]
            sab = parts[10]
            relation_name = rela or rel
            stats["rows_seen"] += 1

            if relation_name not in allowed_relation_set:
                stats["rows_filtered_by_relation"] += 1
                continue
            if cui1 == cui2:
                stats["self_loops"] += 1
                continue
            if cui1 not in scope_nodes or cui2 not in scope_nodes:
                stats["rows_filtered_by_scope"] += 1
                continue

            edge_key = (cui1, relation_name, cui2)
            if edge_key in seen_edges:
                stats["duplicate_edges"] += 1
                continue
            seen_edges.add(edge_key)
            edge_rows.append(
                {
                    "source": cui1,
                    "relation": relation_name,
                    "target": cui2,
                    "rel": rel,
                    "rela": rela,
                    "sab": sab,
                }
            )
            stats["edges_kept"] += 1

    allowed_payload = {
        "allowed_relations": allowed_relations,
        "relation_records": relation_records,
    }
    (out_dir / "allowed_relations.json").write_text(
        json.dumps(allowed_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(out_dir / "evidence_edges.jsonl", edge_rows)
    (out_dir / "medication_rxcui_to_umls.json").write_text(
        json.dumps(medication_rxcui_to_umls, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "patients_jsonl": str(args.patients_jsonl.expanduser().resolve()),
        "mrrel_path": str(args.mrrel_path.expanduser().resolve()),
        "relation_inventory_json": str(args.relation_inventory_json.expanduser().resolve()),
        "relation_allowlist": str(args.relation_allowlist.expanduser().resolve()) if args.relation_allowlist else None,
        "relation_blocklist": str(args.relation_blocklist.expanduser().resolve()) if args.relation_blocklist else None,
        "max_rows": args.max_rows,
        "min_relation_count": args.min_relation_count,
        "num_scope_nodes": len(scope_nodes),
        "num_allowed_relations": len(allowed_relations),
        "num_edges": len(edge_rows),
        "stats": dict(stats),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
