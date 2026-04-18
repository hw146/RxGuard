from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import torch

from rxguard.data.types import (
    ContextNormalization,
    ForbiddenPair,
    KnowledgeEdge,
    KnowledgeGraph,
    MedicationNormalization,
    PatientTrajectory,
)


@dataclass
class DecisionArtifacts:
    canonical_candidates: Tuple[str, ...]
    canonical_context: Tuple[str, ...]
    feasibility_interface: Tuple[ForbiddenPair, ...]
    evidence_edges: Tuple[KnowledgeEdge, ...]
    calibrated_scores: Dict[str, torch.Tensor]
    ddi_loss: torch.Tensor


class DecisionKGCompiler:
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        medication_normalizer: MedicationNormalization,
        context_normalizer: ContextNormalization,
        beta: float,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.medication_normalizer = medication_normalizer
        self.context_normalizer = context_normalizer
        self.beta = beta

    def _canonical_base_scores(
        self,
        candidate_indices: Sequence[int],
        logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        grouped: Dict[str, List[torch.Tensor]] = {}
        for index in candidate_indices:
            canonical = self.medication_normalizer.raw_to_canonical.get(index)
            if canonical is None:
                continue
            grouped.setdefault(canonical, []).append(logits[index])
        return {key: torch.stack(values).max() for key, values in grouped.items()}

    def _compile_feasibility_interface(self, canonical_candidates: Tuple[str, ...]) -> Tuple[ForbiddenPair, ...]:
        candidate_set = set(canonical_candidates)
        edges = self.knowledge_graph.incident_edges(
            source_nodes=candidate_set,
            target_nodes=candidate_set,
            allowed_relations=set(self.knowledge_graph.constraint_relations),
        )
        forbidden = {frozenset((src, dst)) for src, _, dst in edges if src != dst}
        return tuple(sorted(forbidden, key=lambda item: tuple(sorted(item))))

    def _build_evidence_kg(
        self,
        canonical_candidates: Tuple[str, ...],
        canonical_context: Tuple[str, ...],
    ) -> Tuple[KnowledgeEdge, ...]:
        candidate_to_anchor = {
            candidate: self.medication_normalizer.kg_anchor(candidate)
            for candidate in canonical_candidates
        }
        anchor_to_candidates: Dict[str, List[str]] = {}
        for candidate, anchor in candidate_to_anchor.items():
            anchor_to_candidates.setdefault(anchor, []).append(candidate)

        anchor_set = set(candidate_to_anchor.values())
        node_scope = anchor_set | set(canonical_context)
        allowed_relations = set(self.knowledge_graph.constraint_relations | self.knowledge_graph.evidence_relations)
        edges = self.knowledge_graph.incident_edges(
            source_nodes=anchor_set,
            target_nodes=node_scope,
            allowed_relations=allowed_relations,
        )
        lifted: List[KnowledgeEdge] = []
        for src, rel, dst in edges:
            for candidate in anchor_to_candidates.get(src, [src]):
                lifted.append((candidate, rel, dst))
        undirected = list(lifted)
        undirected.extend([(dst, rel, src) for src, rel, dst in lifted])
        return tuple(undirected)

    def _neighbor_map(
        self,
        canonical_candidates: Tuple[str, ...],
        feasibility_interface: Tuple[ForbiddenPair, ...],
    ) -> Dict[str, Set[str]]:
        neighbors = {candidate: set() for candidate in canonical_candidates}
        for pair in feasibility_interface:
            left, right = tuple(pair)
            neighbors[left].add(right)
            neighbors[right].add(left)
        return neighbors

    def compile(
        self,
        trajectory: PatientTrajectory,
        candidate_indices: Sequence[int],
        logits: torch.Tensor,
    ) -> DecisionArtifacts:
        canonical_context = self.context_normalizer.canonicalize(
            trajectory.target.diagnoses,
            trajectory.target.procedures,
        )
        base_scores = self._canonical_base_scores(candidate_indices, logits)
        canonical_candidates = tuple(base_scores.keys())
        feasibility_interface = self._compile_feasibility_interface(canonical_candidates)
        evidence_edges = self._build_evidence_kg(canonical_candidates, canonical_context)
        neighbors = self._neighbor_map(canonical_candidates, feasibility_interface)

        calibrated_scores: Dict[str, torch.Tensor] = {}
        for candidate in canonical_candidates:
            risk = torch.zeros((), dtype=logits.dtype, device=logits.device)
            for neighbor in neighbors[candidate]:
                risk = risk + torch.sigmoid(base_scores[neighbor])
            calibrated_scores[candidate] = base_scores[candidate] - self.beta * risk

        ddi_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
        for pair in feasibility_interface:
            left, right = tuple(pair)
            ddi_loss = ddi_loss + (
                torch.sigmoid(calibrated_scores[left]) * torch.sigmoid(calibrated_scores[right])
            )

        return DecisionArtifacts(
            canonical_candidates=canonical_candidates,
            canonical_context=canonical_context,
            feasibility_interface=feasibility_interface,
            evidence_edges=evidence_edges,
            calibrated_scores=calibrated_scores,
            ddi_loss=ddi_loss,
        )
