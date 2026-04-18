from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RxGuard vocabularies and dataset statistics from converted patient visit JSONL."
    )
    parser.add_argument("--patients-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_counter: Counter[str] = Counter()
    procedure_counter: Counter[str] = Counter()
    medication_counter: Counter[str] = Counter()
    patient_counter = 0
    visit_counter = 0
    meds_per_visit = []
    diags_per_visit = []
    procs_per_visit = []
    visits_with_history = 0

    with args.patients_jsonl.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            patient_counter += 1
            visits = row.get("visits", [])
            for visit_index, visit in enumerate(visits):
                visit_counter += 1
                diagnoses = visit.get("diagnoses", [])
                procedures = visit.get("procedures", [])
                medications = visit.get("medications", [])
                diagnosis_counter.update(diagnoses)
                procedure_counter.update(procedures)
                medication_counter.update(medications)
                diags_per_visit.append(len(diagnoses))
                procs_per_visit.append(len(procedures))
                meds_per_visit.append(len(medications))
                if visit_index > 0:
                    visits_with_history += 1

    diagnosis_vocab = sorted(diagnosis_counter)
    procedure_vocab = sorted(procedure_counter)
    medication_vocab = sorted(medication_counter)

    def serialize_vocab(name: str, values: list[str], counts: Counter[str]) -> None:
        rows = [
            {
                "token": token,
                "index": index,
                "frequency": counts[token],
            }
            for index, token in enumerate(values)
        ]
        _write_jsonl(out_dir / f"{name}_vocab.jsonl", rows)
        _write_json(
            out_dir / f"{name}_vocab_summary.json",
            {
                "name": name,
                "size": len(values),
                "top_20": rows[:20],
            },
        )

    serialize_vocab("diagnosis", diagnosis_vocab, diagnosis_counter)
    serialize_vocab("procedure", procedure_vocab, procedure_counter)
    serialize_vocab("medication", medication_vocab, medication_counter)

    summary = {
        "patients": patient_counter,
        "visits": visit_counter,
        "visits_with_history": visits_with_history,
        "diagnosis_vocab_size": len(diagnosis_vocab),
        "procedure_vocab_size": len(procedure_vocab),
        "medication_vocab_size": len(medication_vocab),
        "avg_diagnoses_per_visit": round(mean(diags_per_visit), 4) if diags_per_visit else 0.0,
        "avg_procedures_per_visit": round(mean(procs_per_visit), 4) if procs_per_visit else 0.0,
        "avg_medications_per_visit": round(mean(meds_per_visit), 4) if meds_per_visit else 0.0,
        "top_20_medications": medication_counter.most_common(20),
        "top_20_diagnoses": diagnosis_counter.most_common(20),
        "top_20_procedures": procedure_counter.most_common(20),
    }
    _write_json(out_dir / "dataset_statistics.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
