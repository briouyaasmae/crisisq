import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import json,hashlib,csv,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'.reproduced/followup_audit';out.mkdir(parents=True,exist_ok=True)
def digest(x):return _sha256_bytes(json.dumps(x,sort_keys=True,default=str).encode())
def sha(p):return _sha256_bytes(Path(p).read_bytes())

# Hash-exact archives; safe extraction, no model inference or external downloads.
import zipfile
def extract_checked(archive,destination,expected):
 assert sha(archive)==expected,'Raw archive checksum differs'
 destination.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for name in z.namelist():
   rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
   if name.endswith('/'):continue
   target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
extract_checked(_result_root(ROOT)/'fusion_localization/raw/crisisq_fusion_operator_localization_v1_reports.zip',ROOT/'.reproduced/crisisq_fusion_operator_localization_v1_received','5e33048eeb7671afe7ab3cd7019f5cbfd39e8f95a9c382141faec843f10f5ce8')
extract_checked(_result_root(ROOT)/'sdcnl/raw/crisisq_sdcnl_frozen_v1_reports.zip',ROOT/'.reproduced/crisisq_sdcnl_frozen_v1_received','14f7e5d90b6c957d6bdb8f2bea8a72b921939ac050f60ec87ed0104fba1c9985')
extract_checked(_result_root(ROOT)/'runtime_matrix/raw/crisisq_runtime_matrix_v1_1_reports.zip',ROOT/'.reproduced/runtime_matrix','4a7daa2a838199a88d5778a4e61739ca7c4c9e293731b41239207e66cbc13f99')

allresults={}
for stem,kind in [('crisisq_fusion_operator_localization_v1','fusion'),('crisisq_sdcnl_frozen_v1','sdcnl')]:
 root=ROOT/'.reproduced'/(stem+'_received');f=json.loads((root/'protocol_frozen.json').read_text());plan=f['protocol'];assert digest(plan)==f['plan_hash'];jobs=json.loads((root/'jobs_frozen.json').read_text());data={};maxrepeat=0.;maxreplay=0.
 hashes=plan.get('implementation_hashes',plan.get('implementation'))
 for name,value in hashes.items():assert sha(root/'source_code'/name)==value,(kind,name)
 for j in jobs:
  folder=root/'jobs'/j['id'];assert json.loads((folder/'job.json').read_text())==j;c=json.loads((folder/'complete.json').read_text())
  assert c['plan_hash']==f['plan_hash'] and c['job_identity']==digest(j) and c['complete']
  assert c['prediction_sha256']==sha(folder/'predictions.npz') and c['model_sha256']==j['model_sha256'] and c['inputs_sha256']==j['inputs_sha256']
  assert c['runtime']==j['runtime'] and c['numpy']=='2.1.3' and c['providers']==['CPUExecutionProvider']
  assert c['repeat_cases']==16 and c['repeat_decision_changes']==0;maxrepeat=max(maxrepeat,c['repeat_max_probability_difference'])
  with np.load(folder/'predictions.npz',allow_pickle=False) as a:
   ix=a['indices'];logit=a['logits'];p=a['probabilities'];assert list(ix)==j['indices'];v=np.exp(logit-np.max(logit,axis=1,keepdims=True));np.testing.assert_allclose(p,v[:,1]/v.sum(1),atol=1e-14,rtol=1e-14);data[j['id']]=p.copy()
  if kind=='fusion':
   assert c['graph_inventory_sha256']==sha(folder/'graph_inventory.json');assert c['graph_replay_cases']==16;maxreplay=max(maxreplay,c['graph_replay_max_logit_difference']);g=json.loads((folder/'graph_inventory.json').read_text());counts=collections.Counter(n['op'] for n in g['nodes']);assert dict(counts)==g['operator_counts']
 result=dict(jobs_verified=len(jobs),inference_rows=sum(len(p) for p in data.values()),maximum_within_session_repeat_delta=maxrepeat,maximum_saved_graph_replay_logit_delta=maxreplay if kind=='fusion' else None,plan_hash=f['plan_hash'])
 if kind=='sdcnl':
  cases=json.loads((root/'case_metadata.json').read_text());assert digest(cases)==plan['case_metadata_hash']
  y=np.array([r['label'] for r in cases]);g=np.array([r['gate_v2'] for r in cases]);clean=np.array([r['clean_eligible'] for r in cases]);test=np.array([r['authors_test'] for r in cases]);panels={'authors_test_clean_primary':clean&test,'all_posts_clean_secondary':clean,'authors_test_raw_sensitivity':test,'all_posts_raw_sensitivity':np.ones(len(cases),dtype=bool)}
  lookup={(j['model_id'],mode):np.where(g==-1,data[j['id']]>=j['tau'],g==1) if mode=='hybrid' else data[j['id']]>=j['tau'] for j in jobs for mode in ['neural','hybrid']}
  for r in csv.DictReader((root/'metrics.csv').open()):
   p=lookup[(r['model'],r['mode'])];mask=panels[r['panel']];pos=mask&(y==1);neg=mask&(y==0)
   checks=dict(n=sum(mask),source_positive=sum(pos),source_negative=sum(neg),detected_source_positive=sum(p&pos),missed_source_positive=sum(~p&pos),alerts_on_source_negative=sum(p&neg),nonalerts_on_source_negative=sum(~p&neg))
   for k,v in checks.items():assert int(r[k])==v
   for k,v in [('source_positive_recall',p[pos].mean()),('source_negative_alert_rate',p[neg].mean()),('source_label_agreement',(p[mask]==y[mask]).mean())]:assert abs(float(r[k])-v)<1e-14
  for r in csv.DictReader((root/'paired_decision_effects.csv').open()):
   mask=panels[r['panel']];a=lookup[tuple(r['baseline'].split('|'))];b=lookup[tuple(r['comparison'].split('|'))]
   checks=dict(changed_decisions=sum(mask&(a!=b)),source_positives_lost=sum(mask&(y==1)&a&~b),source_positives_gained=sum(mask&(y==1)&~a&b),source_negative_alerts_added=sum(mask&(y==0)&~a&b),source_negative_alerts_removed=sum(mask&(y==0)&a&~b))
   for k,v in checks.items():assert int(r[k])==v
  result['metrics_rows_verified']=sum(1 for _ in csv.DictReader((root/'metrics.csv').open()))
  result['paired_contrast_rows_verified']=sum(1 for _ in csv.DictReader((root/'paired_decision_effects.csv').open()))
 else:
  for r in csv.DictReader((root/'fusion_contrasts.csv').open()):
   job=next(j for j in jobs if j['model_id']==r['model'] and j['runtime']==r['runtime'] and j['name']==r['profile']);e=next(j for j in jobs if j['model_id']==r['model'] and j['runtime']==r['runtime'] and j['name']=='extended');b=next(j for j in jobs if j['model_id']==r['model'] and j['runtime']==r['runtime'] and j['name']=='basic');p=data[job['id']];ep=data[e['id']];bp=data[b['id']];tau=job['tau'];pa=p>=tau;pe=ep>=tau;pb=bp>=tau
   checks=dict(neural_decisions_changed_vs_extended=sum(pa!=pe),neural_disagreements_vs_basic=sum(pa!=pb),basic_extended_disagreements_recovered=sum((pb!=pe)&(pa==pb)),new_disagreements_with_basic=sum((pb==pe)&(pa!=pb)))
   for k,v in checks.items():assert int(r[k])==v
   assert float(r['max_probability_delta_vs_extended'])==float(abs(p-ep).max())
  # Compare the new baseline to the earlier completed controlled matrix, separately by runtime.
  old=ROOT/'.reproduced/runtime_matrix';comp=[]
  if (old/'jobs_frozen.json').exists():
   oldjobs=json.loads((old/'jobs_frozen.json').read_text())
   for j in jobs:
    if j['name'] not in ['basic','extended']:continue
    prior=next(x for x in oldjobs if x['model_id']==j['model_id'] and x['runtime']==j['runtime'] and x['optimization']==j['optimization'] and x['threads']==4)
    folder=old/'jobs'/prior['id']
    with np.load(folder/'predictions.npz',allow_pickle=False) as a:
     p=a['probabilities'];now=data[j['id']];comp.append(dict(model=j['model_id'],runtime=j['runtime'],level=j['name'],max_probability_delta=float(abs(p-now).max()),changed_decisions=int(sum((p>=j['tau'])!=(now>=j['tau'])))))
   result['comparison_to_prior_matrix']=comp
 allresults[kind]=result
(out/'audit.json').write_text(json.dumps(allresults,indent=2));print(json.dumps(allresults,indent=2))

# Recompute all paired-seed intervals; seeds remain conditional on joint acceptance.
p=ROOT/'.reproduced/crisisq_sdcnl_frozen_v1_received'
seed=list(csv.DictReader((p/'ce_saqs_per_seed.csv').open()))
intervals=list(csv.DictReader((p/'ce_saqs_paired_seed_intervals.csv').open()))
for r in intervals:
 a=np.array([float(v[r['metric']]) for v in seed if all(v[k]==r[k] for k in ['kind','mode','panel'])])
 rng=np.random.default_rng(20261114);b=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(1)
 assert abs(a.mean()-float(r['mean_delta']))<1e-12
 assert np.allclose(np.quantile(b,[.025,.975]),[float(r['ci_low']),float(r['ci_high'])],atol=1e-12,rtol=0)
print('PASS all32 paired-seed intervals independently recomputed.')
