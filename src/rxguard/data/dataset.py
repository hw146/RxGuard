from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, Iterable, Iterator, Mapping, Sequence, Tuple

from rxguard.data.types import PatientTrajectory, TargetVisit, VisitRecord


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


@dataclass(frozen=True)
class Vocabulary:
    token_to_index: Mapping[str, int]
    index_to_token: Tuple[str, ...]
    source_path: Path | None = None

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Vocabulary":
        resolved = Path(path).expanduser().resolve()
        token_to_index: Dict[str, int] = {}
        index_to_token: list[str] = []
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                token = row["token"]
                index = int(row["index"])
                token_to_index[token] = index
                while len(index_to_token) <= index:
                    index_to_token.append("")
                index_to_token[index] = token
        return cls(token_to_index=token_to_index, index_to_token=tuple(index_to_token), source_path=resolved)

    def encode(self, values: Iterable[str], strict: bool = True) -> Tuple[int, ...]:
        encoded = []
        for value in values:
            index = self.token_to_index.get(value)
            if index is None:
                if strict:
                    raise KeyError(f"Token not found in vocabulary: {value}")
                continue
            encoded.append(index)
        return tuple(encoded)


@dataclass(frozen=True)
class TrajectoryExample:
    instance_id: str
    dataset: str
    patient_id: str
    split: str
    target_visit_id: str
    trajectory: PatientTrajectory


class RxGuardDataset(Sequence[TrajectoryExample]):
    def __init__(
        self,
        patients_jsonl: str | Path,
        instances_jsonl: str | Path,
        diagnosis_vocab_jsonl: str | Path,
        procedure_vocab_jsonl: str | Path,
        medication_vocab_jsonl: str | Path,
        *,
        strict_vocab: bool = True,
        preencode_patients: bool = True,
        patient_fraction: float = 1.0,
        max_instances: int | None = None,
        sample_seed: int = 0,
    ) -> None:
        self.strict_vocab = strict_vocab
        self.diagnosis_vocab = Vocabulary.from_jsonl(diagnosis_vocab_jsonl)
        self.procedure_vocab = Vocabulary.from_jsonl(procedure_vocab_jsonl)
        self.medication_vocab = Vocabulary.from_jsonl(medication_vocab_jsonl)
        self.preencode_patients = preencode_patients

        instances = self._load_instances(instances_jsonl)
        if patient_fraction <= 0.0 or patient_fraction > 1.0:
            raise ValueError("patient_fraction must be in (0, 1].")
        if max_instances is not None and max_instances <= 0:
            raise ValueError("max_instances must be positive when provided.")

        if patient_fraction < 1.0:
            instances = self._subsample_instances_by_patient(
                instances,
                fraction=patient_fraction,
                seed=sample_seed,
            )
        if max_instances is not None and len(instances) > max_instances:
            rng = random.Random(sample_seed)
            rng.shuffle(instances)
            instances = instances[:max_instances]

        self.instances = instances
        needed_patient_keys = {(row["dataset"], row["patient_id"]) for row in self.instances}
        self.patients = self._load_patients_filtered(patients_jsonl, needed_patient_keys)

        # Training/inference iterates over many instances; rebuilding per-visit vocab encodings each time
        # is extremely slow on large cohorts (e.g., MIMIC-IV). Pre-encode visits once per patient.
        self.encoded_visits = self._preencode_patient_visits(self.patients) if preencode_patients else {}

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> TrajectoryExample:
        return self._build_example(self.instances[index])

    def __iter__(self) -> Iterator[TrajectoryExample]:
        for index in range(len(self)):
            yield self[index]

    def _load_patients_filtered(self, path: str | Path, keep: set[Tuple[str, str]]) -> Dict[Tuple[str, str], dict]:
        patients = {}
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["dataset"], row["patient_id"])
                if key in keep:
                    patients[key] = row
        missing = keep.difference(patients.keys())
        if missing:
            # This should never happen if preprocessing produced consistent instances/patients files.
            sample = sorted(list(missing))[:3]
            raise KeyError(f"Missing patient records for {len(missing)} instance keys (sample={sample}).")
        return patients

    def _load_instances(self, path: str | Path) -> list[dict]:
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    @staticmethod
    def _subsample_instances_by_patient(instances: list[dict], *, fraction: float, seed: int) -> list[dict]:
        keys = {(row["dataset"], row["patient_id"]) for row in instances}
        keys_sorted = sorted(keys)
        rng = random.Random(seed)
        rng.shuffle(keys_sorted)
        keep_n = max(1, int(round(len(keys_sorted) * fraction)))
        keep = set(keys_sorted[:keep_n])
        return [row for row in instances if (row["dataset"], row["patient_id"]) in keep]

    def _preencode_patient_visits(self, patients: Mapping[Tuple[str, str], dict]) -> Dict[Tuple[str, str], Tuple[VisitRecord, ...]]:
        encoded: Dict[Tuple[str, str], Tuple[VisitRecord, ...]] = {}
        for key, patient in patients.items():
            visits = patient.get("visits", [])
            time_axis = self._time_axis(visits)
            visit_records = tuple(
                self._build_visit_record(visits[index], timestamp=time_axis[index])
                for index in range(len(visits))
            )
            encoded[key] = visit_records
        return encoded

    def _time_axis(self, visits: Sequence[dict]) -> list[float]:
        datetimes = [
            _parse_timestamp(visit.get("admittime")) or _parse_timestamp(visit.get("dischtime"))
            for visit in visits
        ]
        origin = next((value for value in datetimes if value is not None), None)
        if origin is None:
            return [float(index) for index in range(len(visits))]

        axis = []
        for index, value in enumerate(datetimes):
            if value is None:
                axis.append(float(index))
                continue
            delta_days = (value - origin).total_seconds() / 86400.0
            axis.append(round(max(delta_days, 0.0), 6))
        return axis

    def _build_visit_record(self, visit: dict, timestamp: float) -> VisitRecord:
        return VisitRecord(
            diagnoses=self.diagnosis_vocab.encode(visit.get("diagnoses", []), strict=self.strict_vocab),
            procedures=self.procedure_vocab.encode(visit.get("procedures", []), strict=self.strict_vocab),
            medications=self.medication_vocab.encode(visit.get("medications", []), strict=self.strict_vocab),
            timestamp=timestamp,
        )

    def _build_example(self, instance: dict) -> TrajectoryExample:
        key = (instance["dataset"], instance["patient_id"])
        if self.preencode_patients:
            visits = self.encoded_visits[key]
        else:
            patient = self.patients[key]
            visits = patient["visits"]
        target_index = int(instance["target_visit_index"])
        if self.preencode_patients:
            history = tuple(visits[:target_index])
            target_visit = visits[target_index]
            target = TargetVisit(
                diagnoses=target_visit.diagnoses,
                procedures=target_visit.procedures,
                timestamp=target_visit.timestamp,
                target_medications=target_visit.medications,
            )
        else:
            time_axis = self._time_axis(visits)
            history = tuple(
                self._build_visit_record(visits[index], timestamp=time_axis[index])
                for index in range(target_index)
            )
            target_visit = visits[target_index]
            target = TargetVisit(
                diagnoses=self.diagnosis_vocab.encode(target_visit.get("diagnoses", []), strict=self.strict_vocab),
                procedures=self.procedure_vocab.encode(target_visit.get("procedures", []), strict=self.strict_vocab),
                timestamp=time_axis[target_index],
                target_medications=self.medication_vocab.encode(
                    target_visit.get("medications", []),
                    strict=self.strict_vocab,
                ),
            )
        return TrajectoryExample(
            instance_id=instance["instance_id"],
            dataset=instance["dataset"],
            patient_id=instance["patient_id"],
            split=instance["split"],
            target_visit_id=instance["target_visit_id"],
            trajectory=PatientTrajectory(history=history, target=target),
        )
