# Data Policy

This public release does not include any patient-level data.

## Not Included

- Raw MIMIC-III tables
- Raw MIMIC-IV tables
- paper-aligned longitudinal EHR JSONL files
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

- The public code expects a paper-aligned longitudinal JSONL input for preprocessing into RxGuard trajectories.
- The expected input fields are documented in `data/input_schema.md`.
- If you derive aligned visits from your own licensed pipeline, keep those artifacts outside version control and pass their paths through command-line arguments or local config files.
