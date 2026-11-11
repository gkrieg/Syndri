# Syndri

### *For RECOMB Submission see [recomb branch](https://github.com/gkrieg/Syndri/tree/recomb)*

New structure for data on cluster:
```
/work/magroup/skrieger/Syndri/
├── data/          # Raw, interim, processed, and derived datasets
├── experiments/   # All methods, including Syndri and comparators
├── benchmarks/    # Cross-method evaluations
├── results/       # Final aggregated tables and figures
└── archive/
```

New structure for reorganization

```
Syndri/
  README.md
  environment.yml
  pyproject.toml              # optional, if Syndri becomes installable
  src/
    syndri/                   # reusable Syndri utilities from current code/
  heimdall/                   # renamed/moved from Heimdall/ if kept inside repo
  scripts/                    # runnable one-off or CLI scripts
  notebooks/
    preprocessing/
    evaluation/
    exploration/
  configs/
  tests/
  docs/
  data/
    raw/
    interim/
    processed/
    external/
  results/
    figures/
    tables/
    models/
    runs/
  logs/
  ```