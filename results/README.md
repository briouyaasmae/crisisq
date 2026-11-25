# Recorded evidence

The paper/supplement are the reading entry points. This directory stores each
complete report once; scripts reconstruct required analysis inputs in ignored
`.cache/review_results/`. Original archive bytes and hashes are preserved.

| Directory | Contents |
|---|---|
| `runtime_matrix/` | One complete runtime/optimization/thread report |
| `fusion_localization/` | One complete verified-graph intervention report |
| `sdcnl/` | One complete frozen-threshold SDCNL report |
| `selective_precision/` | One complete recovery/cost report |
| `single_graph_precision_pilot/` | One complete rejected calibration pilot report |
| `device/raw/` | Six timing exports and two classification exports: separate measured runs, not duplicate releases |
| `device/consolidated/` | Cross-run summaries |
| `external/`, `replication/`, `practical/` | Unique saved training/source/host summary tables |
| `runtime_diagnostics/` | Unique recorded probe vectors; manifest aliases preserve the three equal-output settings |
| `additional_audit/`, `policy_audit/`, `posthoc/`, `followup_audit/` | Distinct derived checks, not additional model training |

`python scripts/verify_results.py` checks the recorded evidence without inference,
training or network access. Raw reports retain executed-source/provenance snapshots;
these are frozen evidence, not alternative editable implementations. Reused messages
and artifacts are not independent datasets or architecture replications.
