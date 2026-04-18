from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import Tensor, nn

from rxguard.data.types import PatientTrajectory, VisitRecord


def _mean_pool(embedding: nn.Embedding, token_ids: Iterable[int]) -> Tensor:
    ids = list(token_ids)
    if not ids:
        return embedding.weight.new_zeros(embedding.embedding_dim)
    index = torch.tensor(ids, dtype=torch.long, device=embedding.weight.device)
    return embedding(index).mean(dim=0)


class TimeGapEmbedding(nn.Module):
    """Learnable scalar-to-vector mapping phi(Delta)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, gap: float, device: torch.device) -> Tensor:
        gap_tensor = torch.tensor([[gap]], dtype=torch.float32, device=device)
        return self.net(gap_tensor).squeeze(0)


@dataclass
class ScoringOutput:
    logits: Tensor
    candidate_indices: Tuple[int, ...]
    trajectory_state: Tensor
    recency_memory: Tensor


class TrajectoryAwareScorer(nn.Module):
    def __init__(
        self,
        diagnosis_vocab_size: int,
        procedure_vocab_size: int,
        medication_vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_transformer_layers: int,
        num_attention_heads: int,
        dropout: float,
        candidate_size: int,
        recency_window: float,
        recency_decay: float,
    ) -> None:
        super().__init__()
        self.candidate_size = candidate_size
        self.recency_window = recency_window
        self.recency_decay = recency_decay

        self.diagnosis_embedding = nn.Embedding(diagnosis_vocab_size, embedding_dim)
        self.procedure_embedding = nn.Embedding(procedure_vocab_size, embedding_dim)
        self.medication_embedding = nn.Embedding(medication_vocab_size, embedding_dim)

        self.visit_projection = nn.Linear(3 * embedding_dim, hidden_size)
        self.time_gap_embedding = TimeGapEmbedding(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.query_projection = nn.Linear(2 * embedding_dim, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.recency_projection = nn.Linear(embedding_dim, hidden_size)
        self.output_projection = nn.Linear(2 * hidden_size, medication_vocab_size)

    def _encode_target_visit(self, trajectory: PatientTrajectory) -> Tensor:
        diagnosis_repr = _mean_pool(self.diagnosis_embedding, trajectory.target.diagnoses)
        procedure_repr = _mean_pool(self.procedure_embedding, trajectory.target.procedures)
        return torch.cat([diagnosis_repr, procedure_repr], dim=0)

    def _encode_history_visit(self, visit: VisitRecord) -> Tensor:
        diagnosis_repr = _mean_pool(self.diagnosis_embedding, visit.diagnoses)
        procedure_repr = _mean_pool(self.procedure_embedding, visit.procedures)
        medication_repr = _mean_pool(self.medication_embedding, visit.medications)
        return torch.cat([diagnosis_repr, procedure_repr, medication_repr], dim=0)

    def _build_history_tokens(self, trajectory: PatientTrajectory, device: torch.device) -> Tensor:
        tokens: List[Tensor] = []
        for visit in trajectory.history:
            base = self.visit_projection(self._encode_history_visit(visit))
            gap = max(trajectory.target.timestamp - visit.timestamp, 0.0)
            tokens.append(base + self.time_gap_embedding(gap, device))
        if not tokens:
            return self.output_projection.weight.new_zeros((1, 1, self.query_projection.out_features))
        return torch.stack(tokens, dim=0).unsqueeze(0)

    def _readout(self, target_representation: Tensor, contextualized_history: Tensor) -> Tensor:
        query = self.query_projection(target_representation)
        keys = self.key_projection(contextualized_history.squeeze(0))
        values = self.value_projection(contextualized_history.squeeze(0))
        scale = math.sqrt(keys.size(-1))
        weights = torch.softmax((keys @ query) / scale, dim=0)
        return torch.sum(values * weights.unsqueeze(-1), dim=0)

    def _recency_memory(self, trajectory: PatientTrajectory) -> Tensor:
        # Some upstream pipelines may contain slightly out-of-order timestamps.
        # Guard against negative gaps, which could otherwise lead to exp(+x) overflow
        # and NaNs that poison training.
        recent_visits = []
        for visit in trajectory.history:
            gap = trajectory.target.timestamp - visit.timestamp
            if gap < 0.0:
                gap = 0.0
            if gap < self.recency_window:
                recent_visits.append((visit, gap))
        if not recent_visits:
            return self.medication_embedding.weight.new_zeros(self.medication_embedding.embedding_dim)

        raw_weights = []
        medication_vectors = []
        for visit, gap in recent_visits:
            raw_weights.append(math.exp(-abs(self.recency_decay) * gap))
            medication_vectors.append(_mean_pool(self.medication_embedding, visit.medications))

        norm = sum(raw_weights)
        if norm == 0.0 or math.isinf(norm) or math.isnan(norm):
            # Fall back to uniform weighting if the exponentials under/overflow.
            norm = float(len(raw_weights))
            raw_weights = [1.0 for _ in raw_weights]
        weights = [weight / norm for weight in raw_weights]
        weighted = [vec * weight for vec, weight in zip(medication_vectors, weights)]
        return torch.stack(weighted, dim=0).sum(dim=0)

    def forward(self, trajectory: PatientTrajectory) -> ScoringOutput:
        device = self.output_projection.weight.device
        target_repr = self._encode_target_visit(trajectory)
        history_tokens = self._build_history_tokens(trajectory, device)
        contextualized = self.transformer(history_tokens)

        if trajectory.history:
            trajectory_state = self._readout(target_repr, contextualized)
        else:
            trajectory_state = self.output_projection.weight.new_zeros(self.query_projection.out_features)

        recency_memory = self._recency_memory(trajectory)
        logits = self.output_projection(
            torch.cat([trajectory_state, self.recency_projection(recency_memory)], dim=0)
        )
        candidate_count = min(self.candidate_size, logits.numel())
        candidate_indices = torch.topk(logits, k=candidate_count).indices.tolist()
        return ScoringOutput(
            logits=logits,
            candidate_indices=tuple(int(index) for index in candidate_indices),
            trajectory_state=trajectory_state,
            recency_memory=recency_memory,
        )
