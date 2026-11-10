# Pretrained Syndri models

These are copies of the trained checkpoints used to generate the paper results.

| Experiment | Checkpoint | Milestone | Parameters |
| --- | --- | --- | --- |
| Whole mouse brain MERFISH (BM) | `BM/model-130.pt` | 130 | `configs/models/BM.yaml` |
| NSCLC CosMx (NC) | `NC/model-68.pt` | 68 | `configs/models/NC.yaml` |
| Breast Xenium (BX), including context perturbation | `BX/model-50.pt` | 50 | `configs/models/BX.yaml` |
| Synthetic admixture benchmark (SIM) | `SIM/model-29.pt` | 29 | `configs/models/SIM.yaml` |

File sizes and SHA-256 digests are recorded in `checkpoints.json` and
`assets/manifest.json`. Inference loads these models by default; training saves
new models under `outputs/models/`.

The compatible transformer/tokenizer implementation and gene metadata are
included in this repository. Model configurations, cell-type token dictionaries,
and reference inputs are required alongside the weights.
