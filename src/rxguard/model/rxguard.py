from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rxguard.data.types import PatientTrajectory, RxGuardResult
from rxguard.model.decision_kg import DecisionKGCompiler
from rxguard.model.guardrails import GuardedSetSelector
from rxguard.model.scoring import TrajectoryAwareScorer


@dataclass(frozen=True)
class RxGuardConfig:
    diagnosis_vocab_size: int
    procedure_vocab_size: int
    medication_vocab_size: int
    embedding_dim: int = 64
    hidden_size: int = 128
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.1
    candidate_size: int = 50
    max_set_size: int = 20
    recency_window: float = 30.0
    recency_decay: float = 0.05
    beta: float = 0.2
    ddi_loss_weight: float = 0.1


class RxGuard(nn.Module):
    def __init__(self, config: RxGuardConfig, compiler: DecisionKGCompiler) -> None:
        super().__init__()
        self.config = config
        self.compiler = compiler
        self.scorer = TrajectoryAwareScorer(
            diagnosis_vocab_size=config.diagnosis_vocab_size,
            procedure_vocab_size=config.procedure_vocab_size,
            medication_vocab_size=config.medication_vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_transformer_layers=config.num_transformer_layers,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            candidate_size=config.candidate_size,
            recency_window=config.recency_window,
            recency_decay=config.recency_decay,
        )
        self.selector = GuardedSetSelector(max_set_size=config.max_set_size)

    def forward(self, trajectory: PatientTrajectory) -> RxGuardResult:
        scoring = self.scorer(trajectory)
        artifacts = self.compiler.compile(
            trajectory=trajectory,
            candidate_indices=scoring.candidate_indices,
            logits=scoring.logits,
        )
        selection = self.selector.decode(
            visit_context=artifacts.canonical_context,
            canonical_candidates=artifacts.canonical_candidates,
            calibrated_scores={k: float(v.detach().cpu()) for k, v in artifacts.calibrated_scores.items()},
            feasibility_interface=artifacts.feasibility_interface,
            evidence_edges=artifacts.evidence_edges,
        )
        return RxGuardResult(
            logits=scoring.logits,
            candidate_indices=scoring.candidate_indices,
            canonical_candidates=artifacts.canonical_candidates,
            calibrated_scores={k: float(v.detach().cpu()) for k, v in artifacts.calibrated_scores.items()},
            selected_set=selection.selected_set,
            audit_record=selection.audit_record,
            ddi_loss=float(artifacts.ddi_loss.detach().cpu()),
        )

    def compute_loss(self, trajectory: PatientTrajectory) -> Tuple[Tensor, dict]:
        scoring = self.scorer(trajectory)
        target = torch.zeros_like(scoring.logits)
        if trajectory.target.target_medications:
            target[list(trajectory.target.target_medications)] = 1.0

        prediction_loss = F.binary_cross_entropy_with_logits(scoring.logits, target)
        artifacts = self.compiler.compile(
            trajectory=trajectory,
            candidate_indices=scoring.candidate_indices,
            logits=scoring.logits,
        )
        total_loss = prediction_loss + self.config.ddi_loss_weight * artifacts.ddi_loss
        metrics = {
            "prediction_loss": float(prediction_loss.detach().cpu()),
            "ddi_loss": float(artifacts.ddi_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
        }
        return total_loss, metrics
