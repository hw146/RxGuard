from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator, Sequence


ForbiddenPair = frozenset[str]


@dataclass(frozen=True)
class PredictionRecord:
    instance_id: str
    target_medications: tuple[str, ...]
    predicted_medications: tuple[str, ...]


def _unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}))


def load_prediction_records(path: str | Path) -> list[PredictionRecord]:
    records = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records.append(
                PredictionRecord(
                    instance_id=row["instance_id"],
                    target_medications=_unique_sorted(row.get("target_medications", [])),
                    predicted_medications=_unique_sorted(row.get("predicted_medications", [])),
                )
            )
    return records


def load_ddi_pairs(path: str | Path) -> set[ForbiddenPair]:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() == ".jsonl":
        pairs = set()
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                left = row.get("rxcui_1")
                right = row.get("rxcui_2")
                if left and right and left != right:
                    pairs.add(frozenset((left, right)))
        return pairs

    if resolved.suffix.lower() == ".tsv":
        pairs = set()
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                left = row.get("rxcui_1")
                right = row.get("rxcui_2")
                if left and right and left != right:
                    pairs.add(frozenset((left, right)))
        return pairs

    raise ValueError(f"Unsupported DDI pair file format: {resolved}")


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _ddi_pair_counts(medications: Sequence[str], forbidden_pairs: set[ForbiddenPair]) -> tuple[int, int]:
    total_pairs = 0
    forbidden_count = 0
    for left, right in combinations(sorted(set(medications)), 2):
        total_pairs += 1
        if frozenset((left, right)) in forbidden_pairs:
            forbidden_count += 1
    return forbidden_count, total_pairs


def aggregate_metrics(
    records: Sequence[PredictionRecord],
    forbidden_pairs: set[ForbiddenPair] | None = None,
) -> dict:
    if not records:
        return {
            "num_instances": 0,
            "exact_match_rate": 0.0,
            "avg_jaccard": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "avg_predicted_size": 0.0,
            "avg_target_size": 0.0,
            "set_with_ddi_rate": 0.0,
            "ddi_pair_rate": 0.0,
            "num_sets_with_ddi": 0,
            "num_forbidden_pairs": 0,
            "num_total_pairs": 0,
        }

    totals = {
        "exact_match_rate": 0.0,
        "avg_jaccard": 0.0,
        "avg_precision": 0.0,
        "avg_recall": 0.0,
        "avg_f1": 0.0,
        "avg_predicted_size": 0.0,
        "avg_target_size": 0.0,
        "num_sets_with_ddi": 0,
        "num_forbidden_pairs": 0,
        "num_total_pairs": 0,
    }

    forbidden_pairs = forbidden_pairs or set()
    for record in records:
        predicted = set(record.predicted_medications)
        target = set(record.target_medications)
        intersection = predicted & target
        union = predicted | target

        precision = _safe_divide(len(intersection), len(predicted))
        recall = _safe_divide(len(intersection), len(target))
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        jaccard = _safe_divide(len(intersection), len(union))

        totals["exact_match_rate"] += float(predicted == target)
        totals["avg_jaccard"] += jaccard
        totals["avg_precision"] += precision
        totals["avg_recall"] += recall
        totals["avg_f1"] += f1
        totals["avg_predicted_size"] += len(predicted)
        totals["avg_target_size"] += len(target)

        forbidden_count, total_pairs = _ddi_pair_counts(record.predicted_medications, forbidden_pairs)
        totals["num_forbidden_pairs"] += forbidden_count
        totals["num_total_pairs"] += total_pairs
        if forbidden_count > 0:
            totals["num_sets_with_ddi"] += 1

    num_records = float(len(records))
    return {
        "num_instances": len(records),
        "exact_match_rate": round(totals["exact_match_rate"] / num_records, 6),
        "avg_jaccard": round(totals["avg_jaccard"] / num_records, 6),
        "avg_precision": round(totals["avg_precision"] / num_records, 6),
        "avg_recall": round(totals["avg_recall"] / num_records, 6),
        "avg_f1": round(totals["avg_f1"] / num_records, 6),
        "avg_predicted_size": round(totals["avg_predicted_size"] / num_records, 6),
        "avg_target_size": round(totals["avg_target_size"] / num_records, 6),
        "set_with_ddi_rate": round(totals["num_sets_with_ddi"] / num_records, 6),
        "ddi_pair_rate": round(
            _safe_divide(totals["num_forbidden_pairs"], totals["num_total_pairs"]),
            6,
        ),
        "num_sets_with_ddi": totals["num_sets_with_ddi"],
        "num_forbidden_pairs": totals["num_forbidden_pairs"],
        "num_total_pairs": totals["num_total_pairs"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RxGuard medication set predictions.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument(
        "--ddi-pairs",
        type=Path,
        default=None,
        help="Optional RxCUI DDI file in .jsonl or .tsv format.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_prediction_records(args.predictions_jsonl)
    forbidden_pairs = load_ddi_pairs(args.ddi_pairs) if args.ddi_pairs else None
    metrics = aggregate_metrics(records, forbidden_pairs=forbidden_pairs)
    metrics["predictions_jsonl"] = str(args.predictions_jsonl.expanduser().resolve())
    if args.ddi_pairs:
        metrics["ddi_pairs"] = str(args.ddi_pairs.expanduser().resolve())

    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.out_json:
        args.out_json.expanduser().resolve().write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
