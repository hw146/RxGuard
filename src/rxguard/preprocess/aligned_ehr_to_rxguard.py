from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 0

    def assign(self, patient_id: str) -> str:
        token = f"{patient_id}|{self.seed}".encode("utf-8")
        value = int(hashlib.sha1(token).hexdigest()[:12], 16) / float(16**12)
        train_cutoff = self.train_ratio
        valid_cutoff = self.train_ratio + self.valid_ratio
        if value < train_cutoff:
            return "train"
        if value < valid_cutoff:
            return "valid"
        return "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paper-aligned longitudinal EHR JSONL into RxGuard medication recommendation instances."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to a paper-aligned longitudinal EHR JSONL file.",
    )
    parser.add_argument("--dataset", required=True, choices=["mimic3", "mimic4"])
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for converted artifacts.")
    parser.add_argument("--min-history-visits", type=int, default=1)
    parser.add_argument("--require-target-medications", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-patients", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def _parse_timestamp(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").isoformat()
    except ValueError:
        return value


def _unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted({value for value in values if value})


def _extract_visit_concepts(visit: dict) -> dict:
    diagnoses = []
    procedures = []
    medications = []
    medication_umls_cuis = []
    medication_pairs = {}

    for concept in visit.get("aligned_concepts", []):
        source_type = concept.get("source_type")
        umls_cui = concept.get("umls_cui")
        rxcui = concept.get("rxnorm_rxcui")
        if source_type == "diagnosis" and umls_cui:
            diagnoses.append(umls_cui)
        elif source_type == "procedure" and umls_cui:
            procedures.append(umls_cui)
        elif source_type == "medication" and rxcui:
            medications.append(rxcui)
            if umls_cui:
                medication_umls_cuis.append(umls_cui)
                medication_pairs[rxcui] = umls_cui

    return {
        "visit_id": str(visit.get("visit_id")),
        "admittime": _parse_timestamp(visit.get("admittime")),
        "dischtime": _parse_timestamp(visit.get("dischtime")),
        "deathtime": _parse_timestamp(visit.get("deathtime")),
        "diagnoses": _unique_sorted(diagnoses),
        "procedures": _unique_sorted(procedures),
        "medications": _unique_sorted(medications),
        "medication_umls_cuis": _unique_sorted(medication_umls_cuis),
        "medication_rxcui_to_umls": {
            key: value for key, value in sorted(medication_pairs.items(), key=lambda item: item[0])
        },
    }


def _build_patient_record(row: dict) -> dict:
    visits = [_extract_visit_concepts(visit) for visit in row.get("visits", [])]
    return {
        "dataset": row["dataset"],
        "patient_id": str(row["patient_id"]),
        "num_visits": len(visits),
        "visits": visits,
    }


def _build_instances(
    patient_record: dict,
    split_name: str,
    min_history_visits: int,
    require_target_medications: bool,
) -> List[dict]:
    instances = []
    visits = patient_record["visits"]
    for target_index in range(min_history_visits, len(visits)):
        target = visits[target_index]
        if require_target_medications and not target["medications"]:
            continue

        instance_id = (
            f"{patient_record['dataset']}:{patient_record['patient_id']}:"
            f"visit:{target_index}:{target['visit_id']}"
        )
        instances.append(
            {
                "instance_id": instance_id,
                "dataset": patient_record["dataset"],
                "patient_id": patient_record["patient_id"],
                "split": split_name,
                "target_visit_index": target_index,
                "target_visit_id": target["visit_id"],
                "num_history_visits": target_index,
                "target_timestamp": target["admittime"],
                "target_diagnoses": target["diagnoses"],
                "target_procedures": target["procedures"],
                "target_medications": target["medications"],
            }
        )
    return instances


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    split_config = SplitConfig(seed=args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_rows: List[dict] = []
    instance_rows: Dict[str, List[dict]] = {"train": [], "valid": [], "test": []}
    summary = Counter()

    with args.input_jsonl.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if args.max_patients is not None and idx >= args.max_patients:
                break
            row = json.loads(line)
            patient_record = _build_patient_record(row)
            patient_rows.append(patient_record)

            split_name = split_config.assign(patient_record["patient_id"])
            instances = _build_instances(
                patient_record=patient_record,
                split_name=split_name,
                min_history_visits=args.min_history_visits,
                require_target_medications=args.require_target_medications,
            )

            summary["patients_total"] += 1
            summary[f"patients_{split_name}"] += 1
            summary["visits_total"] += patient_record["num_visits"]
            summary["instances_total"] += len(instances)
            summary[f"instances_{split_name}"] += len(instances)

            for visit in patient_record["visits"]:
                if visit["diagnoses"]:
                    summary["visits_with_diagnoses"] += 1
                if visit["procedures"]:
                    summary["visits_with_procedures"] += 1
                if visit["medications"]:
                    summary["visits_with_medications"] += 1

            instance_rows[split_name].extend(instances)

    patients_path = out_dir / "patients_visits_rxguard.jsonl"
    _write_jsonl(patients_path, patient_rows)
    for split_name, rows in instance_rows.items():
        _write_jsonl(out_dir / f"{split_name}_instances.jsonl", rows)

    sample_preview = {
        "patients_preview": patient_rows[:2],
        "train_instances_preview": instance_rows["train"][:2],
        "valid_instances_preview": instance_rows["valid"][:2],
        "test_instances_preview": instance_rows["test"][:2],
    }
    (out_dir / "sample_preview.json").write_text(
        json.dumps(sample_preview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_payload = {
        "dataset": args.dataset,
        "input_jsonl": str(args.input_jsonl.expanduser().resolve()),
        "output_dir": str(out_dir),
        "min_history_visits": args.min_history_visits,
        "require_target_medications": args.require_target_medications,
        "seed": args.seed,
        "max_patients": args.max_patients,
        "counts": dict(summary),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
