# Reproducing CrisisQ

## Verify recorded results

Install `requirements.txt`, then run `python scripts/verify_results.py`.
This checks ordinary SHA-256 file checksums, frozen metadata, clean notebook syntax,
existing unit tests and saved runtime/fusion/SDCNL/fallback/pilot evidence. It also
recomputes device classification/paired counts and timing means/SD, keyword/probe
counts and saved-output audits. Device bootstrap intervals retain their original
checksummed bytes; the optional full replay below recomputes them as well.
No model inference, training, fitting, dataset download or network access occurs.

Each complete report is stored once. Needed original inputs are unpacked into
ignored `.cache/review_results/`; audit outputs go to `.reproduced/`. Individual
`verify_completed_*.py` scripts can be run separately. The runtime report omitted
its helper sources, so those are retained once in `experiments/runtime_matrix/`;
other studies keep their executed helpers inside the original report.

## Reconstruct device inputs and replay saved exports

```sh
python scripts/materialize_public_assets.py
python scripts/reproduce_results.py
python scripts/aggregate_device_runs.py
```

The materializer checks pinned Twitter/ESConv bytes and reconstructs the original
256-message timing and 1,700-message classification packets. Internet is needed for
missing sources/dependencies only. Existing source files can be supplied using
`--source-cache /path/to/cache`. It neither downloads a classifier nor changes
membership, labels, rules or thresholds. Replaying the eight exports is saved-data
analysis, not a new phone experiment. Do not commit reconstructed source posts.

## Rebuild figures and PDFs

```sh
python scripts/generate_paper_assets.py
python scripts/build_paper.py
```

LaTeX requires pdflatex/bibtex. Figures and tables use recorded numerical evidence;
no model is selected or fitted. Main Figures 1-3 cover framework, timing and runtime;
fusion/fallback are Supplementary Figures S5-S6. One clean paper and supplement are
supplied, without highlighted/title-page/response versions.

## Optional notebook executions

The ten source notebooks are byte-preserved, with simplified numbered filenames.
Their original output names remain unchanged for protocol compatibility. They are
recipes for optional new executions, not steps required to verify saved results.

| Number | Purpose and required input |
|---|---|
|01|Training/selection: authorized original training, augmentation and split CSVs, which are not distributed|
|02|Frozen external evaluation: whole model-containing training output|
|03|Device packet: pinned public source reconstruction|
|04|Practical/device handoff: preceding whole outputs; historical host peak-memory claims excluded|
|05|Controlled runtime matrix with frozen models|
|06|Verified graph/fusion interventions with frozen models|
|07|Pinned SDCNL download and frozen-threshold evaluation|
|08|Recorded selection-policy audit; saved tables only|
|09|Selective-precision recovery and host costs|
|10|Calibration-only single-graph pilot; recorded outcome rejected|

Read each notebook's input instructions before running. The original raw CSVs and
other seed weights cannot be reconstructed from reports. Exact retraining needs
the recorded GPU/software environment in `requirements-research-reference.txt`.
Available primary weights are imported by hash, without silent substitution.
