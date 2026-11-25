#!/usr/bin/env python3
"""Regenerate the registered six-run/device summaries after reproduce_results.py."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,io,json,zipfile,statistics,collections,shutil
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'.reproduced/consolidated';out.mkdir(exist_ok=True)
sources={'HONOR HEY2-W09':['125525025423','131856485987','134258709747'],'Samsung SM-A225F':['151645025366','162007409268','164748745397']}
records=[];memory=[];provenance=[];repeat_checks={}
def outcome(rows):return {(r['profile'],r['variant'],r['mode'],r['source_id'],r['repeat']):(r['probability'],r['decision']) for r in rows}
for device,stamps in sources.items():
 first=None;hashes=set();ids=set();differences=[]
 for index,stamp in enumerate(stamps,1):
  name=f'crisisq_device_results_2026-11-13T{stamp}Z.zip'
  with zipfile.ZipFile((_result_root(ROOT)/'device/raw')/name) as z:
   assert z.testzip() is None
   manifest=json.loads(z.read('run_manifest.json'));done=json.loads(z.read('completion.json'));context=manifest['configuration']
   assert done['complete'] and done['completed_tasks']==16 and context['build_mode']=='release' and context['threads']==4
   summ=list(csv.DictReader(io.StringIO(z.read('timing_summary.csv').decode())))
   rows=list(csv.DictReader(io.StringIO(z.read('device_results_QUINT8_PORTABILITY.csv').decode())))
   snapshots=list(csv.DictReader(io.StringIO(z.read('session_memory_thermal_snapshots.csv').decode())))
  assert len(rows)==5120 and len(summ)==16
  hashes.add(manifest['configuration_hash']);ids.add(manifest['run_id']);current=outcome(rows)
  if first is None:first=current
  differences.append(sum(current[k]!=first[k] for k in first))
  provenance.append({'device':device,'run_number':index,'file':name,'run_id':manifest['run_id'],'configuration_hash':manifest['configuration_hash'],'context':context,'thermal_snapshot_states':dict(collections.Counter(s['thermal_state'] for s in snapshots))})
  for r in summ:records.append({'device':device,'run_number':index,'run_id':manifest['run_id'],**r})
  for variant in ['fp32','q_quint8']:
   values=[float(s['memory_value'])/1024 for s in snapshots if s['variant']==variant and s['stage']=='after_warmup']
   inits=[float(s['session_init_ms']) for s in snapshots if s['variant']==variant and s['stage']=='after_session']
   memory.append({'device':device,'run_number':index,'variant':variant,'metric':'whole_app_current_PSS_MiB','median_post_warmup_pss_mib':statistics.median(values),'min_post_warmup_pss_mib':min(values),'max_post_warmup_pss_mib':max(values),'median_session_init_ms':statistics.median(inits)})
 assert len(hashes)==1 and len(ids)==3 and differences==[0,0,0]
 repeat_checks[device]={'distinct_timing_runs':3,'identical_configuration_within_device':True,'changed_exported_probability_or_decision_records_vs_first_run':differences}
def save(name,rows):
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save('timing_by_run.csv',records);save('process_memory_and_initialization_by_run.csv',memory)
aggregate=[];lookup={}
for device in sources:
 for profile in ['binary_test','labeled_hard_negative','positive_probe','support_unlabeled']:
  for variant in ['fp32','q_quint8']:
   for mode in ['neural','hybrid']:
    data=[r for r in records if (r['device'],r['profile'],r['variant'],r['mode'])==(device,profile,variant,mode)]
    assert len(data)==3
    means=[float(r['mean_total_ms']) for r in data]
    item={'device':device,'profile':profile,'variant':variant,'mode':mode,'runs':3,'unique_messages':64,'repeats_per_message_per_run':5,'timed_rows':960,'mean_of_run_mean_ms':statistics.mean(means),'sd_of_run_mean_ms':statistics.stdev(means),'min_run_mean_ms':min(means),'max_run_mean_ms':max(means),'mean_of_run_median_ms':statistics.mean(float(r['median_total_ms']) for r in data),'mean_of_run_p95_ms':statistics.mean(float(r['p95_total_ms']) for r in data),'neural_invocation_rate':float(data[0]['stage2_rate'])}
    aggregate.append(item);lookup[device,profile,variant,mode]=item
save('timing_three_run_summary.csv',aggregate)
quant=[];routing=[];combined=[]
for device in sources:
 for profile in ['binary_test','labeled_hard_negative','positive_probe','support_unlabeled']:
  for mode in ['neural','hybrid']:
   f=lookup[device,profile,'fp32',mode]['mean_of_run_mean_ms'];q=lookup[device,profile,'q_quint8',mode]['mean_of_run_mean_ms']
   quant.append({'device':device,'profile':profile,'mode':mode,'fp32_mean_ms':f,'quint8_mean_ms':q,'ratio_of_three_run_mean_latencies':f/q,'quint8_latency_reduction_percent':100*(1-q/f)})
  for variant in ['fp32','q_quint8']:
   n=lookup[device,profile,variant,'neural']['mean_of_run_mean_ms'];h=lookup[device,profile,variant,'hybrid']['mean_of_run_mean_ms']
   routing.append({'device':device,'profile':profile,'variant':variant,'neural_mean_ms':n,'hybrid_mean_ms':h,'ratio_of_three_run_mean_latencies':n/h,'routing_latency_reduction_percent':100*(1-h/n)})
  f=lookup[device,profile,'fp32','neural']['mean_of_run_mean_ms'];q=lookup[device,profile,'q_quint8','hybrid']['mean_of_run_mean_ms']
  combined.append({'device':device,'profile':profile,'fp32_neural_mean_ms':f,'quint8_hybrid_mean_ms':q,'ratio_of_three_run_mean_latencies':f/q,'latency_reduction_percent':100*(1-q/f)})
save('quantization_comparisons.csv',quant);save('routing_comparisons.csv',routing);save('combined_configuration_comparisons.csv',combined)
full=[];full_context=[]
for stamp in ['154043157364','171816679850']:
 name=f'crisisq_full_device_evaluation_2026-11-13T{stamp}Z.zip'
 with zipfile.ZipFile((_result_root(ROOT)/'device/raw')/name) as z:
  manifest=json.loads(z.read('run_manifest.json'));rows=list(csv.DictReader(io.StringIO(z.read('full_device_predictions.csv').decode())))
  assert len(rows)==3400 and json.loads(z.read('completion.json'))['complete'] is True
 full_context.append({'file':name,**manifest});full.append({(r['variant'],r['source_id']):(float(r['probability']),int(r['prediction_neural']),int(r['prediction_hybrid'])) for r in rows})
agreement={}
for variant in ['fp32','q_quint8']:
 keys=[k for k in full[0] if k[0]==variant]
 agreement[variant]={'unique_source_labeled_messages':1700,'exact_exported_probability_agreements':sum(full[0][k][0]==full[1][k][0] for k in keys),'neural_decision_agreements':sum(full[0][k][1]==full[1][k][1] for k in keys),'hybrid_decision_agreements':sum(full[0][k][2]==full[1][k][2] for k in keys),'max_probability_difference':max(abs(full[0][k][0]-full[1][k][0]) for k in keys)}
 assert agreement[variant]['exact_exported_probability_agreements']==1700
(out/'cross_device_agreement.json').write_text(json.dumps({'timing_repeat_checks':repeat_checks,'full_classification_agreement':agreement,'note':'Same 1700 labeled messages on two devices; do not treat as 3400 independent texts.'},indent=2))
(out/'run_provenance.json').write_text(json.dumps({'timing_runs':provenance,'full_classification_runs':full_context},indent=2))
for name in ['full_device_classification_metrics.csv','paired_quantization_and_routing_changes.csv','group_bootstrap_intervals.json','kaggle_to_device_drift.csv']:
 shutil.copy2((ROOT/'.reproduced/crisisq_full_device_evaluation_2026-11-13T154043157364Z')/name,out/name)
(out/'validation_HONOR_full.json').write_bytes((ROOT/'.reproduced/crisisq_full_device_evaluation_2026-11-13T154043157364Z/validation.json').read_bytes())
(out/'validation_Samsung_full.json').write_bytes((ROOT/'.reproduced/crisisq_full_device_evaluation_2026-11-13T171816679850Z/validation.json').read_bytes())
lines=['CrisisQ — completed two-device evidence package','',
'Validated: three distinct timing runs per device, 30,720 total timings over the same 256 messages;',
'one complete classification run per device on the same 1,700 source-labeled messages.',
'Four configurations are derived from 3,400 neural model outputs per device.',
'All tests used frozen SAQS seed-42 weights, rules, threshold, CPU provider and four threads.',
'Timing includes on-device tokenization; full classification uses verified saved tokenizer IDs.',
'', 'Three-run hybrid mean latencies (ratio = FP32 hybrid / QUInt8 hybrid):']
for row in quant:
 if row['mode']=='hybrid':lines.append(f"{row['device']}, {row['profile']}: FP32 {row['fp32_mean_ms']:.2f} ms; QUInt8 {row['quint8_mean_ms']:.2f} ms; {row['ratio_of_three_run_mean_latencies']:.2f}x.")
lines+=['','Full-source main binary classification (500 positive, 500 negative), identical on both devices:',
'FP32 neural: recall 85.2%, FPR 14.6%, FN 74, FP 73.',
'FP32 hybrid: recall 85.8%, FPR 15.6%, FN 71, FP 78.',
'QUInt8 neural: recall 83.0%, FPR 13.2%, FN 85, FP 66.',
'QUInt8 hybrid: recall 84.2%, FPR 14.6%, FN 79, FP 73.',
'', 'Recall tradeoff: QUInt8 hybrid loses 8 detected main-set positives, gains none, versus FP32 hybrid.',
'On the 400-case probe (300 positive), the corresponding additional misses are 9.',
'On 300 keyword-enriched source-negative messages, FP32 neural/hybrid alert counts are 207/223; QUInt8 184/209.',
'Linguistic routing increases the alert rate on keyword-enriched source-negative messages. Source labels were not independently audited; this is a stress test, not a real-traffic estimate.',
'', 'All exported probabilities and neural/hybrid decisions on the 1700 messages matched exactly',
'between the two tested Android devices. QUInt8 still differs from Kaggle QUInt8: 1388 probabilities',
'and 12 threshold decisions. Within-device and cross-device repeatability do not erase that difference.',
'', 'Scope and limitations:',
'- Three-run means are averaged equally within each device. Run-level SD is provided.',
'- The task order is the same frozen order across runs; it was not independently counterbalanced.',
'- P95 and medians are averaged run summaries, not quantiles of a pooled population.',
'- Memory is current whole-app PSS, not peak or isolated model memory; allocator effects remain.',
'- Samsung timing snapshots include none/light/moderate states; HONOR timing snapshots are none.',
'- All recorded full-classification thermal snapshots on both devices are none.',
'- Same inspected source rows on both devices are not independent datasets or additional people.',
'- Group bootstrap intervals condition on the original source groups/partitions and one frozen model.',
'- Historical training overlap remains incompletely auditable. Source labels are nonclinical.',
'', 'Planned device experiment collection is complete. Next: revise the manuscript, tables and reviewer',
'response around measured device efficiency, reproducibility, routing/compression ablations and',
'explicit classification tradeoffs. Claims of lossless quantization, uniformly improved accuracy,',
'robust hard-negative handling, clinical utility or Kaggle-equivalent QUInt8 behavior are unsupported.',
'No additional same-device repetitions or retuning of these evaluated artifacts is recommended.']
(out/'READ_ME_final_findings.txt').write_text('\n'.join(lines)+'\n')
archive=(ROOT/'.reproduced/CrisisQ_completed_two_device_evidence.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None;print('Verified final bundle:',len(z.namelist()),'files')
for r in combined:
 if r['profile']=='binary_test':print('COMBINED_BINARY',r['device'],r['ratio_of_three_run_mean_latencies'])
