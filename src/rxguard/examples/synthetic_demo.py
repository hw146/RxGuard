from __future__ import annotations

from pprint import pprint
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rxguard.data.types import (
    ContextNormalization,
    KnowledgeGraph,
    MedicationNormalization,
    PatientTrajectory,
    TargetVisit,
    VisitRecord,
)
from rxguard.model.decision_kg import DecisionKGCompiler
from rxguard.model.rxguard import RxGuard, RxGuardConfig


def build_demo_model() -> RxGuard:
    knowledge_graph = KnowledgeGraph(
        edges=(
            ("RXCUI:warfarin", "ddi", "RXCUI:aspirin"),
            ("RXCUI:warfarin", "ddi", "RXCUI:ibuprofen"),
            ("RXCUI:warfarin", "treats", "CUI:Atrial_fibrillation"),
            ("RXCUI:acetaminophen", "treats", "CUI:Pain"),
            ("RXCUI:metoprolol", "treats", "CUI:Atrial_fibrillation"),
        ),
        constraint_relations=frozenset({"ddi"}),
        evidence_relations=frozenset({"treats"}),
    )
    medication_normalizer = MedicationNormalization(
        raw_to_canonical={
            0: "RXCUI:warfarin",
            1: "RXCUI:metoprolol",
            2: "RXCUI:aspirin",
            3: "RXCUI:acetaminophen",
            4: "RXCUI:ibuprofen",
        }
    )
    context_normalizer = ContextNormalization(
        diagnosis_to_cui={
            10: "CUI:Atrial_fibrillation",
            11: "CUI:Hypertension",
        },
        procedure_to_cui={
            20: "CUI:Pain",
        },
    )
    compiler = DecisionKGCompiler(
        knowledge_graph=knowledge_graph,
        medication_normalizer=medication_normalizer,
        context_normalizer=context_normalizer,
        beta=0.2,
    )
    config = RxGuardConfig(
        diagnosis_vocab_size=32,
        procedure_vocab_size=32,
        medication_vocab_size=8,
        embedding_dim=16,
        hidden_size=32,
        num_transformer_layers=1,
        num_attention_heads=4,
        candidate_size=5,
        max_set_size=3,
        recency_window=30.0,
        recency_decay=0.05,
        beta=0.2,
        ddi_loss_weight=0.1,
    )
    return RxGuard(config=config, compiler=compiler)


def build_demo_trajectory() -> PatientTrajectory:
    return PatientTrajectory(
        history=(
            VisitRecord(
                diagnoses=(10,),
                procedures=(20,),
                medications=(0, 1),
                timestamp=1.0,
            ),
            VisitRecord(
                diagnoses=(11,),
                procedures=(),
                medications=(3,),
                timestamp=8.0,
            ),
        ),
        target=TargetVisit(
            diagnoses=(10,),
            procedures=(20,),
            timestamp=15.0,
            target_medications=(0, 1, 3),
        ),
    )


def main() -> None:
    model = build_demo_model()
    trajectory = build_demo_trajectory()

    loss, metrics = model.compute_loss(trajectory)
    result = model(trajectory)

    print("Loss metrics:")
    pprint(metrics)
    print()
    print("Selected set:")
    pprint(result.selected_set)
    print()
    print("Audit record:")
    pprint(result.audit_record)


if __name__ == "__main__":
    main()
