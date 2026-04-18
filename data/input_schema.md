# RxGuard Input Schema

`src/rxguard/preprocess/aligned_ehr_to_rxguard.py` expects a longitudinal JSONL file, with one patient trajectory per line.

## Top-level fields

Each JSON object should contain:

- `dataset`: dataset name such as `mimic3` or `mimic4`
- `patient_id`: patient-level identifier used locally for grouping and split assignment
- `visits`: ordered list of visit objects

## Visit fields

Each visit object should contain:

- `visit_id`
- `admittime`
- `dischtime`
- `deathtime`
- `aligned_concepts`: list of aligned concept objects

The timestamp fields are optional. When present, they are converted to ISO format.

## Aligned concept fields

Each aligned concept object should contain:

- `source_type`: one of `diagnosis`, `procedure`, or `medication`
- `umls_cui`: UMLS CUI for diagnosis/procedure concepts, and optionally for medication concepts when available for evidence anchoring
- `rxnorm_rxcui`: RxNorm identifier for medication concepts

The preprocessing step assumes the identifier spaces used by RxGuard:

- diagnoses and procedures are read in the UMLS CUI space
- medications are read in the RxNorm space
- duplicate concepts within a visit are removed
- unmapped or empty identifiers are dropped

## Minimal example

```json
{
  "dataset": "mimic3",
  "patient_id": "12345",
  "visits": [
    {
      "visit_id": "v1",
      "admittime": "2101-02-03 10:00:00",
      "dischtime": "2101-02-05 12:00:00",
      "deathtime": null,
      "aligned_concepts": [
        {"source_type": "diagnosis", "umls_cui": "C0011849"},
        {"source_type": "procedure", "umls_cui": "C0184661"},
        {"source_type": "medication", "rxnorm_rxcui": "617314", "umls_cui": "C4270647"}
      ]
    }
  ]
}
```

This preprocessing entry point is intentionally independent of any other project-specific repository. It only assumes the identifier spaces described above.
