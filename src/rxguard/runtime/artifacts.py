from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from rxguard.data.dataset import RxGuardDataset, Vocabulary
from rxguard.data.types import ContextNormalization, KnowledgeEdge, KnowledgeGraph, MedicationNormalization
from rxguard.model.decision_kg import DecisionKGCompiler
from rxguard.model.rxguard import RxGuard, RxGuardConfig


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _load_ddi_edges(path: str | Path) -> list[KnowledgeEdge]:
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    rows = []
    if suffix == ".jsonl":
        rows = _load_jsonl(resolved)
    elif suffix == ".tsv":
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported DDI file format: {resolved}")

    edges: list[KnowledgeEdge] = []
    for row in rows:
        left = row.get("rxcui_1")
        right = row.get("rxcui_2")
        if not left or not right or left == right:
            continue
        edges.append((left, "ddi", right))
        edges.append((right, "ddi", left))
    return edges


def _load_evidence_edges(path: str | Path) -> list[KnowledgeEdge]:
    rows = _load_jsonl(path)
    return [
        (row["source"], row["relation"], row["target"])
        for row in rows
        if row.get("source") and row.get("relation") and row.get("target")
    ]


@dataclass(frozen=True)
class ArtifactBundle:
    diagnosis_vocab: Vocabulary
    procedure_vocab: Vocabulary
    medication_vocab: Vocabulary
    medication_normalizer: MedicationNormalization
    context_normalizer: ContextNormalization
    knowledge_graph: KnowledgeGraph

    def build_dataset(
        self,
        *,
        patients_jsonl: str | Path,
        instances_jsonl: str | Path,
        strict_vocab: bool = True,
        preencode_patients: bool = True,
        patient_fraction: float = 1.0,
        max_instances: int | None = None,
        sample_seed: int = 0,
    ) -> RxGuardDataset:
        return RxGuardDataset(
            patients_jsonl=patients_jsonl,
            instances_jsonl=instances_jsonl,
            diagnosis_vocab_jsonl=self._vocab_path(self.diagnosis_vocab),
            procedure_vocab_jsonl=self._vocab_path(self.procedure_vocab),
            medication_vocab_jsonl=self._vocab_path(self.medication_vocab),
            strict_vocab=strict_vocab,
            preencode_patients=preencode_patients,
            patient_fraction=patient_fraction,
            max_instances=max_instances,
            sample_seed=sample_seed,
        )

    @staticmethod
    def _vocab_path(vocab: Vocabulary) -> str:
        path = vocab.source_path
        if path is None:
            raise RuntimeError("Vocabulary is missing source_path metadata.")
        return str(path)


def build_artifact_bundle(
    *,
    diagnosis_vocab_jsonl: str | Path,
    procedure_vocab_jsonl: str | Path,
    medication_vocab_jsonl: str | Path,
    ddi_pairs_path: str | Path,
    evidence_edges_path: str | Path,
    medication_rxcui_to_umls_path: str | Path,
) -> ArtifactBundle:
    diagnosis_vocab = Vocabulary.from_jsonl(diagnosis_vocab_jsonl)
    procedure_vocab = Vocabulary.from_jsonl(procedure_vocab_jsonl)
    medication_vocab = Vocabulary.from_jsonl(medication_vocab_jsonl)

    ddi_edges = _load_ddi_edges(ddi_pairs_path)
    evidence_edges = _load_evidence_edges(evidence_edges_path)
    medication_rxcui_to_umls = _load_json(medication_rxcui_to_umls_path)

    medication_normalizer = MedicationNormalization(
        raw_to_canonical={index: token for index, token in enumerate(medication_vocab.index_to_token)},
        canonical_to_kg=medication_rxcui_to_umls,
    )
    context_normalizer = ContextNormalization(
        diagnosis_to_cui={index: token for index, token in enumerate(diagnosis_vocab.index_to_token)},
        procedure_to_cui={index: token for index, token in enumerate(procedure_vocab.index_to_token)},
    )
    knowledge_graph = KnowledgeGraph(
        edges=tuple(ddi_edges + evidence_edges),
        constraint_relations=frozenset({"ddi"}),
        evidence_relations=frozenset({edge[1] for edge in evidence_edges}),
    )
    return ArtifactBundle(
        diagnosis_vocab=diagnosis_vocab,
        procedure_vocab=procedure_vocab,
        medication_vocab=medication_vocab,
        medication_normalizer=medication_normalizer,
        context_normalizer=context_normalizer,
        knowledge_graph=knowledge_graph,
    )


def build_model(bundle: ArtifactBundle, config: RxGuardConfig) -> RxGuard:
    compiler = DecisionKGCompiler(
        knowledge_graph=bundle.knowledge_graph,
        medication_normalizer=bundle.medication_normalizer,
        context_normalizer=bundle.context_normalizer,
        beta=config.beta,
    )
    return RxGuard(config=config, compiler=compiler)
