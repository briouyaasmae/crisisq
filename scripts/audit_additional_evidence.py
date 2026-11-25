#!/usr/bin/env python3
"""Post-hoc saved-output audit. Never fits models or changes thresholds."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,json,collections,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RESULT=_result_root(ROOT);OUT=RESULT/'additional_audit'
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(name,data):
 if not data:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def auc(y,p,w=None):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);w=np.ones(len(y)) if w is None else np.asarray(w,dtype=float)
 _,ix=np.unique(p,return_inverse=True);a=np.bincount(ix,weights=w*y);b=np.bincount(ix,weights=w*(1-y));den=a.sum()*b.sum()
 return float(np.sum(a*(np.cumsum(b)-b+.5*b))/den) if den else float('nan')
def main():
 OUT.mkdir(parents=True,exist_ok=True);inventory=[];checks=0
 for p in sorted(RESULT.rglob('*.csv')):
  # Preserve the original audit's pre-follow-up scope; later studies have separate verifiers.
  if OUT in p.parents or RESULT/'policy_audit' in p.parents or RESULT/'selective_precision' in p.parents or RESULT/'single_graph_precision_pilot' in p.parents:continue
  data=rows(p);fields=list(data[0]) if data else [];inventory.append(dict(file=str(Path('results')/p.relative_to(RESULT)),rows=len(data),columns='|'.join(fields),sha256=_sha256_bytes(p.read_bytes())))
  for r in data:
   if all(k in r and r[k]!='' for k in ['n_positive','n_negative','tp','tn','fp','fn']):
    tp,tn,fp,fn,np_,nn=[int(float(r[k])) for k in ['tp','tn','fp','fn','n_positive','n_negative']];assert tp+fn==np_ and tn+fp==nn,(p,r)
    if 'recall' in r and r['recall']!='' and np_:assert abs(float(r['recall'])-tp/np_)<1e-12
    if 'fpr' in r and r['fpr']!='' and nn:assert abs(float(r['fpr'])-fp/nn)<1e-12
    checks+=1
 write('result_table_inventory.csv',inventory)
 # SDCNL: identical case identities/labels and original frozen operating thresholds.
 base=RESULT/'sdcnl';meta=json.loads((base/'case_metadata.json').read_text());lookup={r['source_id']:r for r in meta};ids=[r['source_id'] for r in meta];mask=np.array([r['authors_test'] and r['clean_eligible'] for r in meta]);y=np.array([r['label'] for r in meta]);gate=np.array([r['gate_v2'] for r in meta]);models=json.loads((base/'protocol_frozen.json').read_text())['protocol']['models'];pred=collections.defaultdict(dict)
 for r in rows(base/'predictions_text_free.csv'):
  assert r['text_hash']==lookup[r['source_id']]['text_hash'] and int(r['source_label'])==lookup[r['source_id']]['label'];pred[r['model']][r['source_id']]=r
 values={k:np.array([float(v[i]['probability']) for i in ids]) for k,v in pred.items()};table=[]
 for m in models:
  if m['variant']=='fp32':continue
  fp=f"{m['arm']}__{m['seed']}__fp32";a=values[fp];b=values[m['id']];tau=m['tau'];assert all(float(pred[k][i]['tau'])==tau for k in [fp,m['id']] for i in ids)
  n=a>=tau;q=b>=tau;h=np.where(gate==-1,n,gate==1);hq=np.where(gate==-1,q,gate==1);pos=mask&(y==1);neg=mask&(y==0)
  af,aq=auc(y[mask],a[mask]),auc(y[mask],b[mask]);lost=int(sum(pos&h&~hq));gained=int(sum(pos&~h&hq));added=int(sum(neg&~h&hq));removed=int(sum(neg&h&~hq))
  table.append(dict(arm=m['arm'],seed=m['seed'],variant=m['variant'],tau=tau,n_test=int(sum(mask)),n_source_positive=int(sum(pos)),n_source_negative=int(sum(neg)),fp32_auroc=af,quantized_auroc=aq,delta_auroc=aq-af,hybrid_source_positives_lost=lost,hybrid_source_positives_gained=gained,hybrid_net_positive_loss=lost-gained,hybrid_source_negative_alerts_added=added,hybrid_source_negative_alerts_removed=removed))
 table.sort(key=lambda r:(r['arm'],[42,123,777,1,2,3,4].index(r['seed'])));assert len(table)==14
 write('sdcnl_compression_decisions_and_ranking.csv',table)
 # Author-cluster resampling for primary ranking differences; no operating-point fitting.
 author=np.array([r['author_group'] for r in meta])[mask];names=sorted(set(author));group=np.array([names.index(v) for v in author]);rng=np.random.default_rng(20261114);boot=[]
 for _ in range(1000):
  sampled=rng.integers(0,len(names),len(names));counts=np.bincount(sampled,minlength=len(names));w=counts[group];af=auc(y[mask],values['saqs__42__fp32'][mask],w);aq=auc(y[mask],values['saqs__42__q_quint8'][mask],w)
  if np.isfinite(af+aq):boot.append((af,aq,aq-af))
 boot=np.asarray(boot);primary=next(r for r in table if r['arm']=='saqs' and r['seed']==42);point=[primary['fp32_auroc'],primary['quantized_auroc'],primary['delta_auroc']];ranking=[]
 for k,name in enumerate(['fp32_auroc','quantized_auroc','quantized_minus_fp32_auroc']):ranking.append(dict(metric=name,value=point[k],ci_low=float(np.quantile(boot[:,k],.025)),ci_high=float(np.quantile(boot[:,k],.975)),method='post-hoc1000 paired author-cluster percentile resamples; no ranking-equivalence test'))
 write('sdcnl_primary_ranking_intervals.csv',ranking)
 # Finite-suite stability masking: classify all six masked cases, not favorable examples.
 archive=RESULT/'runtime_matrix/raw/crisisq_runtime_matrix_v1_1_reports.zip'
 with zipfile.ZipFile(archive) as z:
  cases=json.loads(z.read('case_metadata.json'));jobs=json.loads(z.read('jobs_frozen.json'));jj=[j for j in jobs if j['model_id']=='saqs__42__q_quint8'];scores=[]
  for j in jj:
   import io
   with np.load(io.BytesIO(z.read('jobs/'+j['id']+'/predictions.npz')),allow_pickle=False) as p:scores.append(p['probabilities'].copy())
 suite=np.stack(scores);tau=jj[0]['tau'];unstable=(suite.min(0)<tau)&(suite.max(0)>=tau);masked=[]
 for i,c in enumerate(cases):
  if unstable[i] and c['gate_v2']!=-1:masked.append(dict(source_id=c['source_id'],text_hash=c['text_hash'],partition=c['partition'],source_label=c['label'],gate_decision=c['gate_v2'],agrees_with_source_label=c['label']==c['gate_v2'],p_min=float(suite[:,i].min()),p_max=float(suite[:,i].max()),tau=tau))
 assert len(masked)==6;write('masked_disagreement_source_labels.csv',masked)
 # Matched same-precision routing benefit, recomputed from three original run means.
 timing=rows(RESULT/'device/consolidated/timing_by_run.csv');routing=[]
 for device in sorted({v['device'] for v in timing}):
  for variant in ['fp32','q_quint8']:
   for profile in ['binary_test','labeled_hard_negative','positive_probe','support_unlabeled']:
    means=[];stage=[]
    for mode in ['neural','hybrid']:
     selected=[v for v in timing if v['device']==device and v['variant']==variant and v['profile']==profile and v['mode']==mode];assert len(selected)==3;means.append(float(np.mean([float(v['mean_total_ms']) for v in selected])));stage.append(float(np.mean([float(v['stage2_rate']) for v in selected])))
    percent=100*(1-means[1]/means[0]);routing.append(dict(device=device,variant=variant,profile=profile,independent_run_means=3,neural_ms=means[0],hybrid_ms=means[1],reduction_percent=percent,hybrid_gate_exit_fraction=1-stage[1],scope='same64messages/profile; fixed order; descriptive three-run means'))
 write('isolated_routing_device_effects.csv',routing)
 # Three shared seeds give the recorded2x2 loss/augmentation control, on inspected legacy sets.
 legacy=rows(RESULT/'replication/final_results.csv');contrasts=[]
 for description,basearm,target in [('loss_without_augmentation','ce','saqs_loss'),('loss_with_augmentation','ce_aug','saqs'),('augmentation_with_ce','ce','ce_aug'),('augmentation_with_saqs_loss','saqs_loss','saqs')]:
  for split in ['test','probe_final','hn_final']:
   for metric in ['recall','fpr']:
    per=[]
    for seed in ['42','123','777']:
     pair=[next(r for r in legacy if r['arm']==arm and r['seed']==seed and r['variant']=='fp32' and r['system']=='neural' and r['analysis']=='fixed_0.70' and r['split']==split) for arm in [basearm,target]]
     if all(r[metric]!='' for r in pair):per.append(float(pair[1][metric])-float(pair[0][metric]))
    if per:contrasts.append(dict(contrast=description,split=split,metric='source_negative_alert_rate' if split=='hn_final' and metric=='fpr' else metric,seeds=3,tau=.70,mean_delta=float(np.mean(per)),min_seed_delta=float(min(per)),max_seed_delta=float(max(per)),scope='post-hoc descriptive factorial contrasts on inspected legacy data; no generalization claim'))
 write('legacy_three_seed_factorial_contrasts.csv',contrasts)
 # Signs reported for every pair, including null/adverse Twitter recall differences.
 external=rows(RESULT/'external/external_results_per_seed.csv');sdcnl=rows(RESULT/'sdcnl/ce_saqs_per_seed.csv');signs=[]
 for corpus in ['Twitter','SDCNL']:
  for mode in ['neural','hybrid']:
   for metric in ['recall','source_negative_alert_rate']:
    deltas=[]
    for seed in [42,123,777,1,2,3,4]:
     if corpus=='SDCNL':
      r=next(r for r in sdcnl if r['kind']=='source_selected_quantized' and r['panel']=='authors_test_clean_primary' and r['mode']==mode and int(r['seed'])==seed);delta=float(r['delta_recall' if metric=='recall' else 'delta_source_negative_alert_rate'])
     else:
      part='binary_test' if metric=='recall' else 'labeled_hard_negative';key='recall' if metric=='recall' else 'alert_rate';system='neural' if mode=='neural' else 'hybrid_v2';pair=[next(r for r in external if r['arm']==arm and int(r['seed'])==seed and r['analysis']=='source_selected' and r['system']==system and r['partition']==part) for arm in ['ce','saqs']];delta=float(pair[1][key])-float(pair[0][key])
     deltas.append(delta)
    signs.append(dict(corpus=corpus,mode=mode,metric=metric,pairs=7,positive=sum(v>0 for v in deltas),zero=sum(v==0 for v in deltas),negative=sum(v<0 for v in deltas),mean_delta=float(np.mean(deltas)),scope='conditional jointly accepted selected systems; not isolated loss effect'))
 write('cross_corpus_pair_signs.csv',signs)
 summary=dict(result_tables_inventoried=len(inventory),confusion_rate_rows_checked=checks,sdcnl_quantized_graphs=14,sdcnl_net_positive_loss_graphs=sum(r['hybrid_net_positive_loss']>0 for r in table),sdcnl_zero_positive_gain_graphs=sum(r['hybrid_source_positives_gained']==0 for r in table),net_positive_loss_min=min(r['hybrid_net_positive_loss'] for r in table),net_positive_loss_max=max(r['hybrid_net_positive_loss'] for r in table),masked_source_positive_count=sum(r['source_label']==1 for r in masked),masked_source_negative_count=sum(r['source_label']==0 for r in masked),primary_ranking=ranking,scope='post-hoc reanalysis of previously inspected frozen predictions; no retraining, new inference, threshold change or new independent test')
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':main()
