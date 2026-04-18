from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rxguard.model.rxguard import RxGuardConfig
from rxguard.runtime.artifacts import build_artifact_bundle, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RxGuard inference and export predictions/audit records.")
    parser.add_argument("--patients-jsonl", type=Path, required=True)
    parser.add_argument("--instances-jsonl", type=Path, required=True)
    parser.add_argument(
        "--patient-fraction",
        type=float,
        default=1.0,
        help="Subsample patients for this split (1.0 keeps all; 0.0625 keeps 1/16).",
    )
    parser.add_argument("--max-instances", type=int, default=None, help="Optional cap after subsampling.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for deterministic subsampling.")
    parser.add_argument("--diagnosis-vocab-jsonl", type=Path, required=True)
    parser.add_argument("--procedure-vocab-jsonl", type=Path, required=True)
    parser.add_argument("--medication-vocab-jsonl", type=Path, required=True)
    parser.add_argument("--ddi-pairs", type=Path, required=True)
    parser.add_argument("--evidence-edges-jsonl", type=Path, required=True)
    parser.add_argument("--medication-rxcui-to-umls-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--include-identifiers",
        action="store_true",
        help="Include patient_id and target_visit_id in exported rows.",
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location=args.device)
    config = RxGuardConfig(**checkpoint["config"])

    bundle = build_artifact_bundle(
        diagnosis_vocab_jsonl=args.diagnosis_vocab_jsonl,
        procedure_vocab_jsonl=args.procedure_vocab_jsonl,
        medication_vocab_jsonl=args.medication_vocab_jsonl,
        ddi_pairs_path=args.ddi_pairs,
        evidence_edges_path=args.evidence_edges_jsonl,
        medication_rxcui_to_umls_path=args.medication_rxcui_to_umls_json,
    )
    dataset = bundle.build_dataset(
        patients_jsonl=args.patients_jsonl,
        instances_jsonl=args.instances_jsonl,
        patient_fraction=args.patient_fraction,
        max_instances=args.max_instances,
        sample_seed=args.sample_seed,
    )

    model = build_model(bundle, config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prediction_rows = []
    audit_rows = []
    with torch.no_grad():
        for example in dataset:
            result = model(example.trajectory)
            target_tokens = [
                bundle.medication_vocab.index_to_token[index]
                for index in example.trajectory.target.target_medications
            ]
            prediction_rows.append(
                {
                    "instance_id": example.instance_id,
                    "dataset": example.dataset,
                    "target_medications": target_tokens,
                    "predicted_medications": list(result.selected_set),
                    "candidate_medications": list(result.canonical_candidates),
                    "calibrated_scores": result.calibrated_scores,
                }
            )
            if args.include_identifiers:
                prediction_rows[-1]["patient_id"] = example.patient_id
                prediction_rows[-1]["target_visit_id"] = example.target_visit_id
            audit_rows.append(
                {
                    "instance_id": example.instance_id,
                    "dataset": example.dataset,
                    "audit_record": {
                        "visit_context": list(result.audit_record.visit_context),
                        "candidate_set": list(result.audit_record.candidate_set),
                        "feasibility_interface": [sorted(list(pair)) for pair in result.audit_record.feasibility_interface],
                        "selected_set": list(result.audit_record.selected_set),
                        "inclusion_evidence": {
                            key: [list(edge) for edge in value]
                            for key, value in result.audit_record.inclusion_evidence.items()
                        },
                        "exclusion_rationales": result.audit_record.exclusion_rationales,
                    },
                }
            )
            if args.include_identifiers:
                audit_rows[-1]["patient_id"] = example.patient_id
                audit_rows[-1]["target_visit_id"] = example.target_visit_id

    out_jsonl = args.out_jsonl.expanduser().resolve()
    audit_jsonl = args.audit_jsonl.expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_jsonl, prediction_rows)
    _write_jsonl(audit_jsonl, audit_rows)
    print(json.dumps({"predictions": str(out_jsonl), "audits": str(audit_jsonl), "num_instances": len(prediction_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
