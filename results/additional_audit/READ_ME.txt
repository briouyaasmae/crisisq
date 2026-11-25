Additional evidence audit: explicitly post hoc

No training, inference, threshold change or new source data. This audit inventories
55 pre-existing public CSV tables and checks 1,191 applicable confusion/rate rows.
The original matrix, fusion and SDCNL archives remain unchanged. An additional
private coverage inventory lists 188 historical CSV files; these are not 188 studies.

Newly reported findings:
1. All 14 selected compressed graphs have net source-positive hybrid detection
   losses on the same SDCNL published test:1–11 posts,10 graphs with no gains.
   Each compares with its own FP32 counterpart at the same original threshold.
   These thresholds differ from the development acceptance reference at 0.70.
2. The primary neural AUROC estimates are 0.7447/0.7456, despite six lost hybrid
   source positives and identical overall source-label agreement. The paired
   AUROC difference interval includes zero; no equivalence claim is made.
3. The six routing-masked finite-suite disagreements are 2 probe positives and
   4 source-negative posts(3 binary main,1 keyword-enriched), all forced to alert.
   They must not be described as six corrected classifications.
4. Same-precision QUInt8 routing reduces binary-profile three-run mean latency
   by 23.9%HONOR and 25.1%Samsung. The no-exit support changes are within 0.6%.
   Fixed order, sampled thermal states and 64 unique messages/profile remain limits.
5. Three shared legacy seeds at fixed 0.70 separate the recorded 2 x 2 loss/augmentation
   controls. Loss raises main recall by 6.13/6.20 pp with FPR costs 5.54/5.21 pp;
   augmentation alone changes mean main recall by 0.27/0.33 pp. Legacy data were
   already inspected, and these are descriptive three-seed findings only.
6. All 7 selected SDCNL pairs have positive SAQS-minus-CE recall and negative-label
   alert-rate differences in both modes. Twitter recall is positive in 5 pairs,
   tied in 1 and negative in 1; enriched-source-negative alerting increases in all 7.

All conditional selection, source-label, near-duplicate/training-overlap and
one-backbone limitations continue to apply. Do not pool graphs or repeated
messages as independent clinical cases. See supplementary S8/Tables S6–S8 and
Figure S4. The script audit_additional_evidence.py reproduces these analyses
without model weights or network access. All 28 AUROC point estimates were
independently verified with sklearn; bootstrap estimates are author-cluster based.
