# RxGuard

Code for the paper: `RxGuard: Knowledge-Guided Safety Guardrails for Medication Recommendation`

## Requirements

Python 3.10+

```bash
pip install -r requirements.txt
pip install -e .
```

## Repository Scope

This public release contains the implementation of the RxGuard framework and the preprocessing/runtime utilities needed to reproduce the paper once the required licensed resources are available.

The release does not include:

- MIMIC patient data
- DrugBank full XML
- UMLS release files
- derived paper-aligned EHR files
- trained checkpoints or prediction outputs

## Code Organization

### 1. Core model

```text
src/rxguard/model/scoring.py
src/rxguard/model/decision_kg.py
src/rxguard/model/guardrails.py
src/rxguard/model/rxguard.py
```

### 2. Data structures and loading

```text
src/rxguard/data/types.py
src/rxguard/data/dataset.py
```

### 3. Preprocessing

```text
src/rxguard/preprocess/aligned_ehr_to_rxguard.py
src/rxguard/preprocess/build_vocab_and_stats.py
src/rxguard/preprocess/drugbank_ddi.py
src/rxguard/preprocess/evidence_kg.py
```

### 4. Runtime and evaluation

```text
src/rxguard/runtime/train.py
src/rxguard/runtime/predict.py
src/rxguard/eval/metrics.py
```

### 5. Lightweight examples

```text
src/rxguard/examples/synthetic_demo.py
```

## External Data and Assets

This release expects users to prepare the following external inputs separately:

- paper-aligned longitudinal EHR visits for medication recommendation
- DrugBank XML for DDI extraction
- UMLS MRREL.RRF for evidence-graph construction

See:

- `assets/README.md`
- `data/README.md`
- `data/input_schema.md`

## Minimal Workflow

Run the synthetic demo:

```bash
python -m rxguard.examples.synthetic_demo
```

Convert a paper-aligned longitudinal EHR JSONL file into RxGuard visit trajectories:

```bash
PYTHONPATH=src python -m rxguard.preprocess.aligned_ehr_to_rxguard \
  --input-jsonl /path/to/patients_visits_aligned.jsonl \
  --dataset mimic3 \
  --out-dir outputs/mimic3_rxguard \
  --min-history-visits 1 \
  --require-target-medications
```

Build vocabularies and dataset statistics:

```bash
PYTHONPATH=src python -m rxguard.preprocess.build_vocab_and_stats \
  --patients-jsonl /path/to/patients_visits_rxguard.jsonl \
  --out-dir outputs/mimic3_vocab
```

Extract RxNorm-level DDI pairs from DrugBank:

```bash
PYTHONPATH=src python -m rxguard.preprocess.drugbank_ddi \
  --drugbank-xml /path/to/full_database.xml \
  --out-dir outputs/drugbank_ddi
```

Build a scoped evidence KG from UMLS:

```bash
PYTHONPATH=src python -m rxguard.preprocess.evidence_kg \
  --patients-jsonl /path/to/patients_visits_rxguard.jsonl \
  --mrrel-path /path/to/MRREL.RRF \
  --relation-inventory-json /path/to/global_relation_inventory.json \
  --relation-allowlist configs/evidence_relations_core.txt \
  --out-dir outputs/evidence_kg
```

Train RxGuard:

```bash
PYTHONPATH=src python -m rxguard.runtime.train \
  --patients-jsonl /path/to/patients_visits_rxguard.jsonl \
  --train-instances-jsonl /path/to/train_instances.jsonl \
  --valid-instances-jsonl /path/to/valid_instances.jsonl \
  --diagnosis-vocab-jsonl /path/to/diagnosis_vocab.jsonl \
  --procedure-vocab-jsonl /path/to/procedure_vocab.jsonl \
  --medication-vocab-jsonl /path/to/medication_vocab.jsonl \
  --ddi-pairs /path/to/ddi_pairs_rxcui.tsv \
  --evidence-edges-jsonl /path/to/evidence_edges.jsonl \
  --medication-rxcui-to-umls-json /path/to/medication_rxcui_to_umls.json \
  --out-dir outputs/train_run
```

Predict with a saved checkpoint:

```bash
PYTHONPATH=src python -m rxguard.runtime.predict \
  --patients-jsonl /path/to/patients_visits_rxguard.jsonl \
  --instances-jsonl /path/to/test_instances.jsonl \
  --diagnosis-vocab-jsonl /path/to/diagnosis_vocab.jsonl \
  --procedure-vocab-jsonl /path/to/procedure_vocab.jsonl \
  --medication-vocab-jsonl /path/to/medication_vocab.jsonl \
  --ddi-pairs /path/to/ddi_pairs_rxcui.tsv \
  --evidence-edges-jsonl /path/to/evidence_edges.jsonl \
  --medication-rxcui-to-umls-json /path/to/medication_rxcui_to_umls.json \
  --checkpoint /path/to/checkpoint_best.pt \
  --out-jsonl outputs/test_predictions.jsonl \
  --audit-jsonl outputs/test_audits.jsonl
```

By default, exported prediction and audit rows omit patient-level identifiers. Add `--include-identifiers` only when you explicitly need them in a local, access-controlled workflow.

Evaluate predictions:

```bash
PYTHONPATH=src python -m rxguard.eval.metrics \
  --predictions-jsonl /path/to/predictions.jsonl \
  --ddi-pairs /path/to/ddi_pairs_rxcui.tsv
```

## Notes

- The public release keeps the framework code and paper-aligned interfaces, but does not redistribute licensed biomedical resources or patient-derived artifacts.
- The preprocessing pipeline assumes that diagnoses/procedures and medications have already been aligned into the identifier spaces used by the paper-aligned data format described in `data/input_schema.md`.
- The public release is organized around direct command-line workflows and does not depend on any project-specific remote cluster setup.
