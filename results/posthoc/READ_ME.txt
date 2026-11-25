Alert rate on keyword-enriched source-negative messages, stratified by the
original frozen explicit_keyword_present flag. The split is post-hoc and
purely descriptive. It uses unchanged labels, membership and device decisions.
No text was relabeled, no annotator recruited, no model run or threshold tuned.
Both device exports give identical results; do not pool them as 600 messages.

QUInt8 hybrid: keyword present 130/141 (92.20%); absent 79/159 (49.69%).
Keyword-present messages account for 130/209 (62.20%) alerts and 141/300 (47%)
messages. Neural QUInt8: 105/141 and 79/159. Thus hybrid routing adds 25 net
alerts in the keyword-present group and zero net alerts in the absent group.
This is association and decision accounting, not proof of causal keyword
triggering, source-label error, or correctness of the resulting text flags.
The source labels were not independently audited. No estimate of alert rate
on naturally occurring traffic is justified by this enriched sample.
Original historical files and frozen metric field names remain unchanged.
