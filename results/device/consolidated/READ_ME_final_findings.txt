CrisisQ — completed two-device evidence package

Validated: three distinct timing runs per device, 30,720 total timings over the same 256 messages;
one complete classification run per device on the same 1,700 source-labeled messages.
Four configurations are derived from 3,400 neural model outputs per device.
All tests used frozen SAQS seed-42 weights, rules, threshold, CPU provider and four threads.
Timing includes on-device tokenization; full classification uses verified saved tokenizer IDs.

Three-run hybrid mean latencies (ratio = FP32 hybrid / QUInt8 hybrid):
HONOR HEY2-W09, binary_test: FP32 97.65 ms; QUInt8 35.33 ms; 2.76x.
HONOR HEY2-W09, labeled_hard_negative: FP32 86.26 ms; QUInt8 31.92 ms; 2.70x.
HONOR HEY2-W09, positive_probe: FP32 106.08 ms; QUInt8 38.11 ms; 2.78x.
HONOR HEY2-W09, support_unlabeled: FP32 130.00 ms; QUInt8 45.53 ms; 2.85x.
Samsung SM-A225F, binary_test: FP32 315.56 ms; QUInt8 143.48 ms; 2.20x.
Samsung SM-A225F, labeled_hard_negative: FP32 288.29 ms; QUInt8 131.05 ms; 2.20x.
Samsung SM-A225F, positive_probe: FP32 342.63 ms; QUInt8 156.00 ms; 2.20x.
Samsung SM-A225F, support_unlabeled: FP32 418.35 ms; QUInt8 189.73 ms; 2.20x.

Full-source main binary classification (500 positive, 500 negative), identical on both devices:
FP32 neural: recall 85.2%, FPR 14.6%, FN 74, FP 73.
FP32 hybrid: recall 85.8%, FPR 15.6%, FN 71, FP 78.
QUInt8 neural: recall 83.0%, FPR 13.2%, FN 85, FP 66.
QUInt8 hybrid: recall 84.2%, FPR 14.6%, FN 79, FP 73.

Recall tradeoff: QUInt8 hybrid loses 8 detected main-set positives, gains none, versus FP32 hybrid.
On the 400-case probe (300 positive), the corresponding additional misses are 9.
On 300 keyword-enriched source-negative messages, FP32 neural/hybrid alert counts are 207/223; QUInt8 184/209.
Linguistic routing increases the alert rate on keyword-enriched source-negative messages. Source labels were not independently audited; this is a stress test, not a real-traffic estimate.

All exported probabilities and neural/hybrid decisions on the 1700 messages matched exactly
between the two tested Android devices. QUInt8 still differs from Kaggle QUInt8: 1388 probabilities
and 12 threshold decisions. Within-device and cross-device repeatability do not erase that difference.

Scope and limitations:
- Three-run means are averaged equally within each device. Run-level SD is provided.
- The task order is the same frozen order across runs; it was not independently counterbalanced.
- P95 and medians are averaged run summaries, not quantiles of a pooled population.
- Memory is current whole-app PSS, not peak or isolated model memory; allocator effects remain.
- Samsung timing snapshots include none/light/moderate states; HONOR timing snapshots are none.
- All recorded full-classification thermal snapshots on both devices are none.
- Same inspected source rows on both devices are not independent datasets or additional people.
- Group bootstrap intervals condition on the original source groups/partitions and one frozen model.
- Historical training overlap remains incompletely auditable. Source labels are nonclinical.

Planned device experiment collection is complete. Next: revise the manuscript, tables and reviewer
response around measured device efficiency, reproducibility, routing/compression ablations and
explicit classification tradeoffs. Claims of lossless quantization, uniformly improved accuracy,
robust hard-negative handling, clinical utility or Kaggle-equivalent QUInt8 behavior are unsupported.
No additional same-device repetitions or retuning of these evaluated artifacts is recommended.
