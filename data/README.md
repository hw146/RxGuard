# Data Policy

This public release does not include any patient-level data.

## Not Included

- Raw MIMIC-III tables
- Raw MIMIC-IV tables
- longitudinal EHR JSONL files prepared for RxGuard preprocessing
- train/validation/test instances
- vocabulary files
- patient-derived checkpoints, predictions, or audit outputs

## Expected Layout

```text
data/
└── raw/
    ├── mimic3/
    └── mimic4/
```

## Notes

- The public code expects a longitudinal JSONL input in the RxGuard preprocessing format.
- The expected input fields are documented in `data/input_schema.md`.
- If you derive these visit files from your own licensed pipeline, keep them outside version control and pass their paths through command-line arguments.
