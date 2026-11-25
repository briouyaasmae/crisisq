import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,hashlib,json,math,zipfile
import numpy as np
r=_result_root(Path(__file__).resolve().parents[1])/'selective_precision'
def h(path):
 if path.is_file():return _sha256_bytes(path.read_bytes())
 # Public source releases omit standalone .log files; the immutable report ZIP retains them.
 if path.suffix=='.log':
  with zipfile.ZipFile(r/'raw/crisisq_selective_precision_v1_1_reports.zip') as archive:
   return _sha256_bytes(archive.read(str(path.relative_to(r))))
 raise FileNotFoundError(path)

digest=lambda x:_sha256_bytes(json.dumps(x,sort_keys=True,default=str).encode())
p=json.loads((r/'protocol_frozen.json').read_text());rows=json.loads((r/'case_metadata.json').read_text());assert digest(rows)==p['case_metadata_hash']
y=np.array([v['label'] for v in rows]);g=np.array([v['gate_v2'] for v in rows]);cal=np.array(p['calibration_indices']);bands=json.loads((r/'routing_parameters_frozen.json').read_text())['parameters'];metrics=list(csv.DictReader((r/'metrics.csv').open()));ixs={part:np.array([i for i,v in enumerate(rows) if v['partition']==part and not v['calibration']]) for part in sorted({v['partition'] for v in rows})}
for name,value in p['implementation'].items():assert h(r/'source_code'/name)==value,name
for job in json.loads((r/'score_jobs.json').read_text()):
 dest=r/'scores'/job['id'];record=json.loads((dest/'complete.json').read_text());assert record['job_identity']==digest(job) and h(dest/'predictions.npz')==record['prediction_sha256'];assert record['repeat_decision_changes']==0
checks=0;overview=[]
for pair in p['models']:
 for version in p['runtime_versions']:
  arrays={}
  for kind in ['fp32','compressed']:
   with np.load(r/'scores'/f"{pair['id']}__{kind}__{version}"/'predictions.npz',allow_pickle=False) as z:
    assert np.array_equal(z['indices'],np.arange(len(rows)));arrays[kind]=z['probabilities']
  q=arrays['compressed'];f=arrays['fp32'];tau=pair['tau'];a=np.where(g>=0,g,q>=tau).astype(int);ref=np.where(g>=0,g,f>=tau).astype(int)
  if version=='1.31.1':
   pool=cal[g[cal]<0];e=pool[q[pool]<tau];k=math.floor(.05*len(pool));dist=np.sort(tau-q[e]);expected=0. if not k or not len(e) else np.nextafter(dist[-1],np.inf) if k>=len(e) else dist[k];assert expected==bands[pair['id']]['one_sided_05']['width']
  width=bands[pair['id']]['one_sided_05']['width'];use=(g<0)&(q<tau)&((tau-q)<width);b=np.where(use,ref,a)
  for part,ix in ixs.items():
   pos=y[ix]==1;neg=~pos
   for policy,pred,calls in [('always_compressed',a,0),('always_fp32',ref,0),('one_sided_05',b,int(use[ix].sum()))]:
    row=next(t for t in metrics if t['model']==pair['id'] and t['runtime']==version and t['partition']==part and t['policy']==policy)
    expected=dict(n=len(ix),n_positive=int(pos.sum()),n_negative=int(neg.sum()),positive_losses_vs_fp32=int(((y[ix]==1)&(ref[ix]==1)&(pred[ix]==0)).sum()),positive_gains_vs_quantized=int(((y[ix]==1)&(a[ix]==0)&(pred[ix]==1)).sum()),positive_losses_vs_quantized=int(((y[ix]==1)&(a[ix]==1)&(pred[ix]==0)).sum()),added_source_negative_alerts_vs_quantized=int(((y[ix]==0)&(a[ix]==0)&(pred[ix]==1)).sum()),fallback_calls=calls)
    for key,v in expected.items():assert int(row[key])==v,(pair['id'],version,part,policy,key)
    if pos.any():assert abs(float(row['source_positive_recall'])-pred[ix][pos].mean())<1e-12
    if neg.any():assert abs(float(row['source_negative_alert_rate'])-pred[ix][neg].mean())<1e-12
    checks+=1
   if part in ['binary_test','sdcnl_authors_test']:
    overview.append(dict(model=pair['id'],runtime=version,partition=part,baseline_losses=int(((y[ix]==1)&(ref[ix]==1)&(a[ix]==0)).sum()),remaining_losses=int(((y[ix]==1)&(ref[ix]==1)&(b[ix]==0)).sum()),added_negative_alerts=int(((y[ix]==0)&(a[ix]==0)&(b[ix]==1)).sum()),fallbacks=int(use[ix].sum()),n=len(ix)))
# Independent aggregation of every recorded fresh-process timing run.
summary=list(csv.DictReader((r/'timing_summary.csv').open()));run_means=[]
for job in json.loads((r/'timing_schedule_frozen.json').read_text()):
 dest=r/'timing'/job['id'];record=json.loads((dest/'complete.json').read_text());assert record['job_sha256']==h(r/'job_specs'/(job['id']+'.json'));assert record['timing_sha256']==h(dest/'timing.csv');assert record['live_vs_saved_decision_disagreements']==0
 raw=list(csv.DictReader((dest/'timing.csv').open()))
 for part in sorted({t['partition'] for t in raw}):
  subset=[t for t in raw if t['partition']==part];run_means.append((job['runtime'],job['policy'],part,sum(float(t['elapsed_ms']) for t in subset)/len(subset)))
for row in summary:
 vals=[v for rt,mode,part,v in run_means if (rt,mode,part)==(row['runtime'],row['policy'],row['partition'])];assert len(vals)==3 and abs(sum(vals)/3-float(row['mean_of_run_means_ms']))<1e-10


# Verify every declared policy row, including random control and sensitivities.
cache={};predictions={};panel_predictions={}
for pair in p['models']:
 for version in p['runtime_versions']:
  data={}
  for kind in ['fp32','compressed']:
   with np.load(r/'scores'/f"{pair['id']}__{kind}__{version}"/'predictions.npz',allow_pickle=False) as z:data[kind]=z['probabilities'].copy()
  q,f=data['compressed'],data['fp32'];tau=pair['tau'];qd=np.where(g>=0,g,q>=tau).astype(int);fd=np.where(g>=0,g,f>=tau).astype(int)
  cache[(pair['id'],version)]=(q,f,qd,fd,tau)
for row in metrics:
 q,f,qd,fd,tau=cache[(row['model'],row['runtime'])];ix=ixs[row['partition']];mode=row['policy']
 if mode=='always_compressed':use=np.zeros(len(g),bool);d=qd;qcalls=int((g[ix]<0).sum());fcalls=0
 elif mode=='always_fp32':use=g<0;d=fd;qcalls=0;fcalls=int(use[ix].sum())
 else:
  if mode=='random_one_sided_05_matched':
   spec=bands[row['model']]['one_sided_05'];regular=(g<0)&(q<tau)&(np.abs(q-tau)<spec['width']);k=int(regular[ix].sum());eligible=[int(i) for i in ix if g[i]<0 and q[i]<tau];eligible.sort(key=lambda i:_sha256_bytes(('CrisisQ-fallback-random-v1|'+rows[i]['source_id']).encode()));use=np.zeros(len(g),bool);use[eligible[:k]]=True
  else:
   spec=bands[row['model']][mode];use=(g<0)&(np.abs(q-tau)<spec['width'])
   if spec['side']=='one_sided':use&=q<tau
  d=np.where(use,fd,qd);qcalls=int((g[ix]<0).sum());fcalls=int(use[ix].sum())
  if mode!='random_one_sided_05_matched':predictions[(row['model'],row['runtime'],mode)]=d
 panel_predictions[(row['model'],row['runtime'],row['partition'],mode)]=d.copy()
 pos=y[ix]==1;neg=y[ix]==0
 expected=dict(n=len(ix),n_positive=int(pos.sum()),n_negative=int(neg.sum()),compressed_calls=qcalls,fp32_calls=fcalls,fallback_calls=0 if mode=='always_fp32' else fcalls,neural_route_count=int((g[ix]<0).sum()),positive_losses_vs_fp32=int((pos&(fd[ix]==1)&(d[ix]==0)).sum()),positive_gains_vs_fp32=int((pos&(fd[ix]==0)&(d[ix]==1)).sum()),positive_gains_vs_quantized=int((pos&(qd[ix]==0)&(d[ix]==1)).sum()),positive_losses_vs_quantized=int((pos&(qd[ix]==1)&(d[ix]==0)).sum()),added_source_negative_alerts_vs_quantized=int((neg&(qd[ix]==0)&(d[ix]==1)).sum()),removed_source_negative_alerts_vs_quantized=int((neg&(qd[ix]==1)&(d[ix]==0)).sum()),decisions_different_from_fp32=int((d[ix]!=fd[ix]).sum()))
 for key,value in expected.items():assert int(row[key])==value,(row['model'],mode,key)
 rates=dict(source_label_agreement=float((d[ix]==y[ix]).mean()),fallback_fraction_all_messages=0. if mode=='always_fp32' else float(use[ix].mean()))
 if pos.any():rates['source_positive_recall']=float(d[ix][pos].mean())
 if neg.any():rates['source_negative_alert_rate']=float(d[ix][neg].mean())
 for key,value in rates.items():assert abs(float(row[key])-value)<1e-12,(row['model'],mode,key)
 if mode=='always_compressed':predictions[(row['model'],row['runtime'],mode)]=qd
 if mode=='always_fp32':predictions[(row['model'],row['runtime'],mode)]=fd
for row in csv.DictReader((r/'cross_runtime_decisions.csv').open()):
 a=predictions[(row['model'],row['runtime_a'],row['policy'])];b=predictions[(row['model'],row['runtime_b'],row['policy'])];ix=ixs[row['partition']];assert int((a[ix]!=b[ix]).sum())==int(row['decision_disagreements'])
for row in csv.DictReader((r/'matched_random_reference.csv').open()):
 q,f,qd,fd,tau=cache[(row['model'],row['runtime'])];ix=ixs[row['partition']];e=ix[(g[ix]<0)&(q[ix]<tau)];loss=int(((y[e]==1)&(fd[e]==1)).sum());assert loss==int(row['baseline_positive_losses']);k=int(row['matched_fallbacks']);assert abs(float(row['random_expected_recovered'])-(k*loss/len(e) if len(e) else 0.))<1e-12
interval_rows=list(csv.DictReader((r/'paired_group_intervals.csv').open()))
for row in interval_rows:
 ix=ixs[row['partition']];valid=y[ix]==(1 if row['metric']=='recall' else 0);ix=ix[valid]
 if not len(ix):assert row['cluster_lo']==row['cluster_hi']=='';continue
 pred=panel_predictions[(row['model'],row['runtime'],row['partition'],row['policy'])];base=cache[(row['model'],row['runtime'])][2]
 group=np.array([rows[i]['group_id'] for i in ix]);unique=np.unique(group);delta=pred[ix]-base[ix]
 sums=np.array([delta[group==v].sum() for v in unique],float);counts=np.array([(group==v).sum() for v in unique]);sample=np.random.default_rng(20261115).integers(len(unique),size=(1000,len(unique)));distribution=sums[sample].sum(1)/counts[sample].sum(1);lo,hi=np.quantile(distribution,[.025,.975])
 assert abs(delta.mean()-float(row['delta_vs_compressed']))<1e-12
 assert abs(lo-float(row['cluster_lo']))<1e-12 and abs(hi-float(row['cluster_hi']))<1e-12
for line in (r/'SHA256SUMS.txt').read_text().splitlines():
 value,relative=line.split('  ',1);assert h(r/relative)==value,relative
completion=json.loads((r/'completion.json').read_text());assert completion['complete'] and completion['score_jobs']==16 and completion['timing_jobs']==30 and completion['timing_live_vs_saved_disagreements']==0
print(f'PASS:16score jobs/33,264outputs;{len(metrics)}metric rows;all policy/drift/random means and paired intervals;30timing jobs/20aggregates;original manifests and source hashes. No inference.')
