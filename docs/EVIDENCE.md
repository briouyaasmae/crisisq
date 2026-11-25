# Evidence and limits

The unchanged manuscript/supplement are the primary account. Repository cleanup
adds no experiments or result changes.

- Runtime contrasts alter decisions in 13/14 selected compressed artifacts versus 0/14
  FP32 counterparts on 256 shared messages. These are conditional artifacts of one
  backbone, not independent architecture replications.
- Tested Android QUInt8 hybrid is 2.20-2.85 times faster than FP32 hybrid, with eight
  detected main positives lost and none gained. SDCNL shows six lost and none gained
  for the primary hybrid comparison.
- CE-SAQS comparisons are conditional recall/alerting trade-offs,
  not overall superiority. Routing can mask disagreement while adding alerts.
- Keyword-enriched source negatives support a stress-set alert rate, without
  adjudicated real-traffic false positives or causal keyword conclusions.
- Verified graph changes partially recover BASIC decisions without repairing runtime
  disagreement or uniquely identifying a faulty kernel.
- Fallback reaches zero reference-positive losses in 8/16 comparisons, adding alerts,
  memory and near-FP32 host latency. Phone mitigation costs remain unmeasured.
- The single-graph pilot was rejected at calibration; no accepted graph, external
  evaluation, timing or phone benefit is supplied.

Twitter partitions share a source; SDCNL labels are subreddit-derived. Missing
training texts prevent full overlap certification. Devices, seeds and repeated
messages are not independent source samples. Clinical validity, prospective benefit,
battery savings and reduced human-review workload are unmeasured.

The primary is SAQS42 / unsigned QUInt8 / threshold 0.6455836149936347. It differs
from the initial signed-QInt8/0.63 configuration, whose original headline checkpoint
is unavailable. Retraining needs missing authorized training/augmentation CSVs;
other-seed inference needs unavailable weights. Saved-output verification and checks
of available primary artifacts are the strongest reproduction paths.
