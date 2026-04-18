from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from rxguard.data.types import AuditRecord, ForbiddenPair, KnowledgeEdge


@dataclass
class GuardedSelectionOutput:
    selected_set: Tuple[str, ...]
    audit_record: AuditRecord


class GuardedSetSelector:
    def __init__(self, max_set_size: int) -> None:
        self.max_set_size = max_set_size

    @staticmethod
    def _is_feasible(candidate: str, selected: Sequence[str], forbidden: set[ForbiddenPair]) -> bool:
        return all(frozenset((candidate, chosen)) not in forbidden for chosen in selected)

    def decode(
        self,
        visit_context: Tuple[str, ...],
        canonical_candidates: Tuple[str, ...],
        calibrated_scores: Dict[str, float],
        feasibility_interface: Tuple[ForbiddenPair, ...],
        evidence_edges: Tuple[KnowledgeEdge, ...],
    ) -> GuardedSelectionOutput:
        forbidden = set(feasibility_interface)
        ranked_candidates = sorted(
            canonical_candidates,
            key=lambda item: float(calibrated_scores[item]),
            reverse=True,
        )

        selected: List[str] = []
        for candidate in ranked_candidates:
            if len(selected) >= self.max_set_size:
                break
            if self._is_feasible(candidate, selected, forbidden):
                selected.append(candidate)

        inclusion_evidence: Dict[str, List[KnowledgeEdge]] = {}
        for medication in selected:
            inclusion_evidence[medication] = [
                edge for edge in evidence_edges if edge[0] == medication and edge[2] in set(visit_context)
            ]

        exclusion_rationales: Dict[str, List[str]] = {}
        for candidate in canonical_candidates:
            if candidate in selected:
                continue
            conflicts = [
                chosen for chosen in selected if frozenset((candidate, chosen)) in forbidden
            ]
            exclusion_rationales[candidate] = conflicts

        audit = AuditRecord(
            visit_context=visit_context,
            candidate_set=canonical_candidates,
            feasibility_interface=feasibility_interface,
            selected_set=tuple(selected),
            inclusion_evidence=inclusion_evidence,
            exclusion_rationales=exclusion_rationales,
        )
        return GuardedSelectionOutput(selected_set=tuple(selected), audit_record=audit)
