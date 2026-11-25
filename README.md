# CrisisQ

Code and recorded evidence for **CrisisQ: A decision-preservation audit of quantized hybrid crisis-language detection**.

CrisisQ combines linguistic rules with a DistilBERT classifier and development-constrained quantization. This repository examines the detection and resource costs of deploying that frozen system. It does not establish clinical validity, prospective workflow benefit or lossless compression.

## Start here

- **Read:** [Manuscript](paper/manuscript.pdf) · [Supplement](paper/supplementary.pdf)
- **Check the recorded findings:** install `requirements.txt`, then run `python scripts/verify_results.py`. This uses saved outputs, without model inference, training or dataset downloads.
- **Run on a phone:** follow [the device guide](docs/DEVICE.md). Exact frozen model files must be supplied separately.
- **Repeat a notebook:** use the numbered [notebook guide](docs/REPRODUCING.md). These are optional new executions, not required to verify the supplied results.

## Questions answered

| Question | Evidence in the paper | Recorded inputs |
|---|---|---|
| RQ1: Do fixed compressed artifacts preserve decisions across execution settings? | Section 5.5; Figure 3 | `results/runtime_matrix/` |
| RQ2: What detection, alerting and device costs accompany compression and rule routing? | Sections 5.1–5.4; Tables 1–3; Figure 2 | `results/external/`, `results/sdcnl/`, `results/device/` |
| RQ3: Can verified graph changes localize decision changes? | Section 5.6; Supplement S6 / Figure S5 | `results/fusion_localization/` |
| RQ4: What does selective precision recover, and at what cost? | Section 5.7; Supplement S10 / Figure S6 | `results/selective_precision/` |

The rejected single-graph pilot belongs only to Supplement S11. It is not a deployment model. Scope and reproduction limits are in [Evidence](docs/EVIDENCE.md).

## Repository layout

| Folder | One purpose |
|---|---|
| `paper/` | One clean manuscript, one supplement, and their LaTeX/figure sources |
| `notebooks/` | One numbered recipe per study |
| `scripts/` | Verification, saved-output analysis, asset preparation and plotting |
| `results/` | Recorded reports and unique summary tables; see [the results index](results/README.md) |
| `experiments/runtime_matrix/` | The one executed helper snapshot omitted from its original report |
| `app/` | Flutter application and platform integration |
| `manifests/`, `assets/` | Frozen protocols, model hashes and text-free reconstruction metadata |
| `docs/` | Reproduction, phone, evidence and publication instructions |
| `tests/` | Selection/routing and source-integrity checks |

Each completed report is stored once as `recorded_report.zip`. Its original bytes and hashes are retained. Analysis scripts reconstruct needed inputs under ignored `.cache/` and write checks under `.reproduced/`; neither folder belongs in GitHub. There are no duplicate extracted research reports, alternative paper versions, reviewer letters or private author packages. Standard Flutter platform files and license notices remain where their projects require them.

## Quick check

Use Python 3.11 or 3.12:

```sh
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -r requirements.txt
python scripts/verify_results.py
```

The check is offline after dependency installation. For device-data replay, figure generation, LaTeX compilation and required external inputs, see [Reproducing](docs/REPRODUCING.md). `SHA256SUMS.txt` verifies the distributed files; `python scripts/verify_release.py` checks it directly.

## Upload and citation

Extract this package and push the **contents of this folder** to one repository. Do not upload the ZIP as a single repository file. [Publishing instructions](docs/PUBLISHING.md) show the Git commands.

This is a public, author-attributed release. Use [CITATION.cff](CITATION.cff) for the author list and software citation. This public software release is version 1.4.12; the paper and numerical results are unchanged. Code is covered by [LICENSE](LICENSE); third-party models, data and manuscript rights have separate terms described in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Model weights and materialized source posts are not distributed here.
