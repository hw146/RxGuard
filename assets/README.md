# External Assets

This repository does not include licensed biomedical resources.

## Not Included In GitHub

- DrugBank full XML
- UMLS Metathesaurus files
- any local terminology caches or model caches

## Expected Layout

```text
assets/
├── drugbank/
│   └── full_database.xml
└── umls/
    └── MRREL.RRF
```

## Notes

- `src/rxguard/preprocess/drugbank_ddi.py` uses DrugBank XML to extract RxNorm-level DDI pairs.
- `src/rxguard/preprocess/evidence_kg.py` uses `MRREL.RRF` together with a relation inventory and allowlist to build the scoped evidence graph.
- If these assets are stored outside the repository, update the corresponding paths in your local config files before running preprocessing.
