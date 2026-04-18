from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


DRUGBANK_NS = {"db": "http://www.drugbank.ca"}


def _local(tag: str) -> str:
    return f"{{{DRUGBANK_NS['db']}}}{tag}"


def _iter_drugs(xml_path: Path, max_drugs: int | None = None) -> Iterator[ET.Element]:
    drug_depth = 0
    yielded = 0
    for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
        if event == "start" and elem.tag == _local("drug"):
            drug_depth += 1
            continue
        if event == "end" and elem.tag == _local("drug"):
            if drug_depth == 1:
                yield elem
                yielded += 1
                elem.clear()
                if max_drugs is not None and yielded >= max_drugs:
                    return
            drug_depth -= 1


def _child_text(parent: ET.Element, tag: str) -> Optional[str]:
    child = parent.find(f"db:{tag}", DRUGBANK_NS)
    return child.text.strip() if child is not None and child.text else None


def _drugbank_id(drug: ET.Element) -> Optional[str]:
    ids = drug.findall("db:drugbank-id", DRUGBANK_NS)
    for item in ids:
        if item.text and item.text.startswith("DB"):
            return item.text.strip()
    return None


def _extract_rxcui(drug: ET.Element) -> Optional[str]:
    ext_ids = drug.find("db:external-identifiers", DRUGBANK_NS)
    if ext_ids is None:
        return None
    for ext in ext_ids.findall("db:external-identifier", DRUGBANK_NS):
        resource = _child_text(ext, "resource")
        identifier = _child_text(ext, "identifier")
        if resource == "RxCUI" and identifier:
            return identifier
    return None


def _extract_synonyms(drug: ET.Element, limit: int = 20) -> List[str]:
    synonyms = []
    syn_root = drug.find("db:synonyms", DRUGBANK_NS)
    if syn_root is None:
        return synonyms
    for syn in syn_root.findall("db:synonym", DRUGBANK_NS):
        if syn.text:
            synonyms.append(syn.text.strip())
        if len(synonyms) >= limit:
            break
    return synonyms


def _extract_products(drug: ET.Element, limit: int = 20) -> List[str]:
    products = []
    product_root = drug.find("db:products", DRUGBANK_NS)
    if product_root is None:
        return products
    for product in product_root.findall("db:product", DRUGBANK_NS):
        name = _child_text(product, "name")
        if name:
            products.append(name)
        if len(products) >= limit:
            break
    return products


def _extract_interactions(drug: ET.Element) -> List[Tuple[str, Optional[str], Optional[str]]]:
    interactions_root = drug.find("db:drug-interactions", DRUGBANK_NS)
    if interactions_root is None:
        return []
    interactions = []
    for item in interactions_root.findall("db:drug-interaction", DRUGBANK_NS):
        other_id = _child_text(item, "drugbank-id")
        other_name = _child_text(item, "name")
        description = _child_text(item, "description")
        if other_id:
            interactions.append((other_id, other_name, description))
    return interactions


def build_drugbank_to_rxcui(xml_path: Path, max_drugs: int | None = None) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for drug in _iter_drugs(xml_path, max_drugs=max_drugs):
        drugbank_id = _drugbank_id(drug)
        if not drugbank_id:
            continue
        name = _child_text(drug, "name")
        rxcui = _extract_rxcui(drug)
        mapping[drugbank_id] = {
            "drugbank_id": drugbank_id,
            "name": name,
            "rxcui": rxcui,
            "synonyms": _extract_synonyms(drug),
            "products": _extract_products(drug),
        }
    return mapping


def build_ddi_pairs(
    xml_path: Path,
    mapping: Dict[str, dict],
    max_drugs: int | None = None,
) -> Tuple[List[dict], Dict[str, int]]:
    pair_index: Dict[Tuple[str, str], dict] = {}
    stats = Counter()

    for drug in _iter_drugs(xml_path, max_drugs=max_drugs):
        source_id = _drugbank_id(drug)
        if not source_id:
            stats["missing_source_drugbank_id"] += 1
            continue
        source_meta = mapping.get(source_id, {})
        source_rxcui = source_meta.get("rxcui")
        if not source_rxcui:
            stats["source_without_rxcui"] += 1
            continue

        for target_id, target_name, description in _extract_interactions(drug):
            stats["raw_interactions"] += 1
            target_meta = mapping.get(target_id, {})
            target_rxcui = target_meta.get("rxcui")
            if not target_rxcui:
                stats["target_without_rxcui"] += 1
                continue
            if source_rxcui == target_rxcui:
                stats["self_loop_after_rxcui"] += 1
                continue

            left, right = sorted((source_rxcui, target_rxcui))
            key = (left, right)
            if key not in pair_index:
                pair_index[key] = {
                    "rxcui_1": left,
                    "rxcui_2": right,
                    "drugbank_id_1": source_id if source_rxcui == left else target_id,
                    "drugbank_id_2": target_id if target_rxcui == right else source_id,
                    "name_1": source_meta.get("name") if source_rxcui == left else target_meta.get("name"),
                    "name_2": target_meta.get("name") if target_rxcui == right else source_meta.get("name"),
                    "descriptions": [],
                }
            if description:
                pair_index[key]["descriptions"].append(description)
            stats["mapped_interactions"] += 1

    pairs = sorted(pair_index.values(), key=lambda item: (item["rxcui_1"], item["rxcui_2"]))
    stats["unique_rxcui_pairs"] = len(pairs)
    return pairs, dict(stats)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_pairs_tsv(path: Path, rows: Iterable[dict]) -> None:
    fieldnames = [
        "rxcui_1",
        "rxcui_2",
        "drugbank_id_1",
        "drugbank_id_2",
        "name_1",
        "name_2",
        "example_description",
        "description_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            descriptions = row.get("descriptions", [])
            writer.writerow(
                {
                    "rxcui_1": row["rxcui_1"],
                    "rxcui_2": row["rxcui_2"],
                    "drugbank_id_1": row["drugbank_id_1"],
                    "drugbank_id_2": row["drugbank_id_2"],
                    "name_1": row["name_1"],
                    "name_2": row["name_2"],
                    "example_description": descriptions[0] if descriptions else "",
                    "description_count": len(descriptions),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RxCUI-level DDI pairs from DrugBank full XML.")
    parser.add_argument("--drugbank-xml", required=True, help="Path to DrugBank full database XML.")
    parser.add_argument("--out-dir", required=True, help="Output directory for extracted artifacts.")
    parser.add_argument("--max-drugs", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    xml_path = Path(args.drugbank_xml).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = build_drugbank_to_rxcui(xml_path, max_drugs=args.max_drugs)
    pairs, stats = build_ddi_pairs(xml_path, mapping, max_drugs=args.max_drugs)

    mapping_rows = sorted(mapping.values(), key=lambda item: item["drugbank_id"])
    _write_jsonl(out_dir / "drugbank_to_rxcui.jsonl", mapping_rows)
    _write_jsonl(out_dir / "ddi_pairs_rxcui.jsonl", pairs)
    _write_pairs_tsv(out_dir / "ddi_pairs_rxcui.tsv", pairs)

    summary = {
        "drugbank_xml": str(xml_path),
        "max_drugs": args.max_drugs,
        "num_drugbank_entries": len(mapping_rows),
        "num_drugbank_entries_with_rxcui": sum(1 for row in mapping_rows if row.get("rxcui")),
        "stats": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
