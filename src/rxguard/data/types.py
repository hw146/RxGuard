from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


NodeId = str
RelationId = str
ForbiddenPair = frozenset[str]
KnowledgeEdge = Tuple[NodeId, RelationId, NodeId]


@dataclass(frozen=True)
class VisitRecord:
    diagnoses: Tuple[int, ...]
    procedures: Tuple[int, ...]
    medications: Tuple[int, ...]
    timestamp: float


@dataclass(frozen=True)
class TargetVisit:
    diagnoses: Tuple[int, ...]
    procedures: Tuple[int, ...]
    timestamp: float
    target_medications: Tuple[int, ...]

    @property
    def context(self) -> Tuple[int, ...]:
        return tuple(dict.fromkeys(self.diagnoses + self.procedures))


@dataclass(frozen=True)
class PatientTrajectory:
    history: Tuple[VisitRecord, ...]
    target: TargetVisit


@dataclass(frozen=True)
class KnowledgeGraph:
    edges: Tuple[KnowledgeEdge, ...]
    constraint_relations: frozenset[RelationId]
    evidence_relations: frozenset[RelationId]

    def incident_edges(
        self,
        source_nodes: Set[NodeId],
        target_nodes: Set[NodeId],
        allowed_relations: Set[RelationId],
    ) -> List[KnowledgeEdge]:
        return [
            edge
            for edge in self.edges
            if edge[0] in source_nodes and edge[2] in target_nodes and edge[1] in allowed_relations
        ]


@dataclass(frozen=True)
class MedicationNormalization:
    raw_to_canonical: Mapping[int, str]
    canonical_to_kg: Mapping[str, str] | None = None

    def canonicalize(self, medication_ids: Iterable[int]) -> Tuple[str, ...]:
        canonical = []
        seen = set()
        for med_id in medication_ids:
            mapped = self.raw_to_canonical.get(med_id)
            if mapped is None or mapped in seen:
                continue
            canonical.append(mapped)
            seen.add(mapped)
        return tuple(canonical)

    def kg_anchor(self, canonical_id: str) -> str:
        if self.canonical_to_kg is None:
            return canonical_id
        return self.canonical_to_kg.get(canonical_id, canonical_id)


@dataclass(frozen=True)
class ContextNormalization:
    diagnosis_to_cui: Mapping[int, str]
    procedure_to_cui: Mapping[int, str]

    def canonicalize(self, diagnosis_ids: Iterable[int], procedure_ids: Iterable[int]) -> Tuple[str, ...]:
        context = []
        seen = set()
        for code in diagnosis_ids:
            mapped = self.diagnosis_to_cui.get(code)
            if mapped is None or mapped in seen:
                continue
            context.append(mapped)
            seen.add(mapped)
        for code in procedure_ids:
            mapped = self.procedure_to_cui.get(code)
            if mapped is None or mapped in seen:
                continue
            context.append(mapped)
            seen.add(mapped)
        return tuple(context)


@dataclass
class AuditRecord:
    visit_context: Tuple[str, ...]
    candidate_set: Tuple[str, ...]
    feasibility_interface: Tuple[ForbiddenPair, ...]
    selected_set: Tuple[str, ...]
    inclusion_evidence: Dict[str, List[KnowledgeEdge]] = field(default_factory=dict)
    exclusion_rationales: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RxGuardResult:
    logits: "TensorLike"
    candidate_indices: Tuple[int, ...]
    canonical_candidates: Tuple[str, ...]
    calibrated_scores: Dict[str, float]
    selected_set: Tuple[str, ...]
    audit_record: AuditRecord
    ddi_loss: float


class TensorLike:
    """A light forward reference helper for static readability."""
