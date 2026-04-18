from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from rxguard.eval.metrics import PredictionRecord, aggregate_metrics, load_ddi_pairs
from rxguard.model.rxguard import RxGuardConfig
from rxguard.runtime.artifacts import build_artifact_bundle, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RxGuard on trajectory instances.")
    parser.add_argument("--patients-jsonl", type=Path, required=True)
    parser.add_argument("--train-instances-jsonl", type=Path, required=True)
    parser.add_argument("--valid-instances-jsonl", type=Path, default=None)
    parser.add_argument(
        "--train-patient-fraction",
        type=float,
        default=1.0,
        help="Subsample patients for train split (1.0 keeps all; 0.0625 keeps 1/16).",
    )
    parser.add_argument(
        "--valid-patient-fraction",
        type=float,
        default=1.0,
        help="Subsample patients for valid split (1.0 keeps all; 0.0625 keeps 1/16).",
    )
    parser.add_argument("--train-max-instances", type=int, default=None, help="Optional cap after subsampling.")
    parser.add_argument("--valid-max-instances", type=int, default=None, help="Optional cap after subsampling.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for deterministic subsampling; defaults to --seed when omitted.",
    )
    parser.add_argument("--diagnosis-vocab-jsonl", type=Path, required=True)
    parser.add_argument("--procedure-vocab-jsonl", type=Path, required=True)
    parser.add_argument("--medication-vocab-jsonl", type=Path, required=True)
    parser.add_argument("--ddi-pairs", type=Path, required=True)
    parser.add_argument("--evidence-edges-jsonl", type=Path, required=True)
    parser.add_argument("--medication-rxcui-to-umls-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs.")
    parser.add_argument(
        "--selection-metric",
        default="avg_jaccard",
        choices=[
            "avg_jaccard",
            "avg_f1",
            "avg_precision",
            "avg_recall",
            "exact_match_rate",
        ],
        help="Validation metric used to select checkpoint_best (paper uses avg_jaccard).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--candidate-size", type=int, default=50)
    parser.add_argument("--max-set-size", type=int, default=20)
    parser.add_argument("--recency-window", type=float, default=30.0)
    parser.add_argument("--recency-decay", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--ddi-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold for numerical stability.",
    )
    return parser.parse_args()


def _records_from_model(model, dataset, medication_vocab) -> list[PredictionRecord]:
    records = []
    model.eval()
    with torch.no_grad():
        for example in dataset:
            result = model(example.trajectory)
            target_tokens = tuple(medication_vocab.index_to_token[index] for index in example.trajectory.target.target_medications)
            records.append(
                PredictionRecord(
                    instance_id=example.instance_id,
                    target_medications=target_tokens,
                    predicted_medications=result.selected_set,
                )
            )
    return records


def _require_finite(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite tensor encountered: {label}")


def _first_nonfinite_grad(model: torch.nn.Module) -> str | None:
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            return name
    return None


def _sanitize_nonfinite_grads(model: torch.nn.Module) -> tuple[int, str | None]:
    replaced = 0
    first_bad = None
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        bad_mask = ~torch.isfinite(grad)
        bad_count = int(bad_mask.sum().item())
        if bad_count == 0:
            continue
        if first_bad is None:
            first_bad = name
        replaced += bad_count
        grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)
    return replaced, first_bad


def _sanitize_nonfinite_params(model: torch.nn.Module) -> tuple[int, str | None]:
    replaced = 0
    first_bad = None
    for name, param in model.named_parameters():
        data = param.data
        bad_mask = ~torch.isfinite(data)
        bad_count = int(bad_mask.sum().item())
        if bad_count == 0:
            continue
        if first_bad is None:
            first_bad = name
        replaced += bad_count
        param.data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return replaced, first_bad


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.patience < 0:
        raise SystemExit("--patience must be >= 0.")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_artifact_bundle(
        diagnosis_vocab_jsonl=args.diagnosis_vocab_jsonl,
        procedure_vocab_jsonl=args.procedure_vocab_jsonl,
        medication_vocab_jsonl=args.medication_vocab_jsonl,
        ddi_pairs_path=args.ddi_pairs,
        evidence_edges_path=args.evidence_edges_jsonl,
        medication_rxcui_to_umls_path=args.medication_rxcui_to_umls_json,
    )
    train_dataset = bundle.build_dataset(
        patients_jsonl=args.patients_jsonl,
        instances_jsonl=args.train_instances_jsonl,
        patient_fraction=args.train_patient_fraction,
        max_instances=args.train_max_instances,
        sample_seed=sample_seed,
    )
    valid_dataset = (
        bundle.build_dataset(
            patients_jsonl=args.patients_jsonl,
            instances_jsonl=args.valid_instances_jsonl,
            patient_fraction=args.valid_patient_fraction,
            max_instances=args.valid_max_instances,
            sample_seed=sample_seed,
        )
        if args.valid_instances_jsonl
        else None
    )

    config = RxGuardConfig(
        diagnosis_vocab_size=len(bundle.diagnosis_vocab.index_to_token),
        procedure_vocab_size=len(bundle.procedure_vocab.index_to_token),
        medication_vocab_size=len(bundle.medication_vocab.index_to_token),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_transformer_layers=args.num_transformer_layers,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        candidate_size=args.candidate_size,
        max_set_size=args.max_set_size,
        recency_window=args.recency_window,
        recency_decay=args.recency_decay,
        beta=args.beta,
        ddi_loss_weight=args.ddi_loss_weight,
    )
    model = build_model(bundle, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    forbidden_pairs = load_ddi_pairs(args.ddi_pairs)

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "device": args.device,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "selection_metric": args.selection_metric,
                "max_grad_norm": args.max_grad_norm,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    history = []
    best_valid_metric = None
    best_valid_epoch = None
    epochs_without_improve = 0
    best_checkpoint_path = out_dir / "checkpoint_best.pt"

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_prediction_loss = 0.0
        total_ddi_loss = 0.0
        num_seen = 0
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grad = 0
        sanitized_nonfinite_grad_values = 0
        sanitized_nonfinite_param_values = 0
        order = list(range(len(train_dataset)))
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(order):
            example = train_dataset[index]
            trajectory = example.trajectory
            loss, metrics = model.compute_loss(trajectory)

            # Keep long-running jobs alive even if a rare trajectory triggers numerical issues.
            # We log the offending instance for later investigation and skip the update.
            if not torch.isfinite(loss).all():
                skipped_nonfinite_loss += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    json.dumps(
                        {
                            "event": "skip_nonfinite_loss",
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "instance_id": example.instance_id,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    flush=True,
                )
                continue

            (loss / float(args.batch_size)).backward()

            total_loss += float(metrics["total_loss"])
            total_prediction_loss += float(metrics["prediction_loss"])
            total_ddi_loss += float(metrics["ddi_loss"])
            num_seen += 1

            if ((step + 1) % args.batch_size) == 0 or (step + 1) == len(order):
                replaced_grad_values, bad_grad_param = _sanitize_nonfinite_grads(model)
                if replaced_grad_values > 0:
                    sanitized_nonfinite_grad_values += replaced_grad_values
                    print(
                        json.dumps(
                            {
                                "event": "sanitize_nonfinite_grad",
                                "epoch": epoch + 1,
                                "step": step + 1,
                                "bad_param": bad_grad_param,
                                "num_values": replaced_grad_values,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        flush=True,
                    )

                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    if not torch.isfinite(grad_norm):
                        skipped_nonfinite_grad += 1
                        optimizer.zero_grad(set_to_none=True)
                        print(
                            json.dumps(
                                {
                                    "event": "skip_nonfinite_grad_norm",
                                    "epoch": epoch + 1,
                                    "step": step + 1,
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            ),
                            flush=True,
                        )
                        continue

                optimizer.step()

                replaced_param_values, bad_param = _sanitize_nonfinite_params(model)
                if replaced_param_values > 0:
                    sanitized_nonfinite_param_values += replaced_param_values
                    print(
                        json.dumps(
                            {
                                "event": "sanitize_nonfinite_param",
                                "epoch": epoch + 1,
                                "step": step + 1,
                                "bad_param": bad_param,
                                "num_values": replaced_param_values,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        flush=True,
                    )
                optimizer.zero_grad(set_to_none=True)

        epoch_record = {
            "epoch": epoch + 1,
            "train_total_loss": total_loss / max(num_seen, 1),
            "train_prediction_loss": total_prediction_loss / max(num_seen, 1),
            "train_ddi_loss": total_ddi_loss / max(num_seen, 1),
        }
        if skipped_nonfinite_loss:
            epoch_record["train_skipped_nonfinite_loss"] = skipped_nonfinite_loss
        if skipped_nonfinite_grad:
            epoch_record["train_skipped_nonfinite_grad"] = skipped_nonfinite_grad
        if sanitized_nonfinite_grad_values:
            epoch_record["train_sanitized_nonfinite_grad_values"] = sanitized_nonfinite_grad_values
        if sanitized_nonfinite_param_values:
            epoch_record["train_sanitized_nonfinite_param_values"] = sanitized_nonfinite_param_values

        if valid_dataset is not None:
            valid_records = _records_from_model(model, valid_dataset, bundle.medication_vocab)
            valid_metrics = aggregate_metrics(valid_records, forbidden_pairs=forbidden_pairs)
            for key, value in valid_metrics.items():
                epoch_record[f"valid_{key}"] = value
            metric_value = float(valid_metrics[args.selection_metric])
            if math.isnan(metric_value) or math.isinf(metric_value):
                raise RuntimeError(f"Non-finite validation metric: {args.selection_metric}={metric_value}")

            if best_valid_metric is None or metric_value > best_valid_metric:
                best_valid_metric = metric_value
                best_valid_epoch = epoch + 1
                epochs_without_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "epoch": epoch + 1,
                        "valid_metrics": valid_metrics,
                    },
                    best_checkpoint_path,
                )
            else:
                epochs_without_improve += 1

        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False, allow_nan=False), flush=True)

        if valid_dataset is not None and args.patience > 0 and epochs_without_improve >= args.patience:
            print(
                json.dumps(
                    {
                        "action": "early_stop",
                        "epoch": epoch + 1,
                        "best_valid_epoch": best_valid_epoch,
                        "best_valid_metric": best_valid_metric,
                        "selection_metric": args.selection_metric,
                        "patience": args.patience,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    final_checkpoint_path = out_dir / "checkpoint_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "epochs": args.epochs,
            "history": history,
        },
        final_checkpoint_path,
    )
    (out_dir / "train_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"history_path": str(out_dir / "train_history.json"), "checkpoint_last": str(final_checkpoint_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
