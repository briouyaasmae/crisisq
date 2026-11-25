import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import zipfile
archive=_result_root(ROOT)/'runtime_matrix/raw/crisisq_runtime_matrix_v1_1_reports.zip'
assert _sha256_bytes(archive.read_bytes())=='4a7daa2a838199a88d5778a4e61739ca7c4c9e293731b41239207e66cbc13f99','Raw matrix archive changed'
R=ROOT/'.reproduced/runtime_matrix';R.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name in z.namelist():
  rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
  if name.endswith('/'):continue
  target=R/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
def sha(p):return _sha256_bytes(p.read_bytes())
def digest(o):return _sha256_bytes(json.dumps(o,sort_keys=True,default=str).encode())
frozen=json.loads((R/'protocol_frozen.json').read_text());p=frozen['protocol'];assert digest(p)==frozen['plan_hash']
jobs=json.loads((R/'jobs_frozen.json').read_text());assert digest(jobs)==json.loads((R/'execution_manifest.json').read_text())['jobs_hash']
cases=json.loads((R/'case_metadata.json').read_text());assert digest(cases)==p['case_metadata_hash']
assert len(jobs)==448 and len(p['models'])==28 and len(p['settings'])==16
source=ROOT/'experiments/runtime_matrix'
assert all(sha(ROOT/'scripts/build_full_device_packet.py' if n=='build_full_device_packet.py' else source/n)==h for n,h in p['implementation_hashes'].items()),'Executed implementation differs'
print('All five executed implementation hashes match.')
table={};max_repeat=0.;versions=collections.Counter();elapsed=[]
for j in jobs:
 d=R/'jobs'/j['id'];record=json.loads((d/'complete.json').read_text());assert record['complete'] and record['plan_hash']==frozen['plan_hash'] and record['job_identity']==digest(j)
 assert sha(d/'predictions.npz')==record['prediction_sha256'];assert record['runtime']==j['runtime'] and record['numpy']=='2.1.3' and record['model_sha256']==j['model_sha256']
 assert record['providers']==['CPUExecutionProvider'];assert record['repeat_decision_changes']==0
 max_repeat=max(max_repeat,record['repeat_max_probability_difference']);versions[record['runtime']]+=1
 with np.load(d/'predictions.npz',allow_pickle=False) as z:
  ix=z['indices'].copy();prob=z['probabilities'].copy();logits=z['logits'];assert list(ix)==j['indices'];assert np.isfinite(logits).all()
  ex=np.exp(logits-logits.max(axis=1,keepdims=True));assert np.allclose(prob,(ex/ex.sum(axis=1,keepdims=True))[:,1],rtol=1e-12,atol=1e-14);elapsed.append(float(z['elapsed_ms'].sum()))
 table[(j['model_id'],j['runtime'],j['optimization'],j['threads'])]=(ix,prob)
print('VERIFIED all448 hashes, logits/probabilities, case order, versions, CPU provider, repeat decisions. Max repeat delta',max_repeat,'versions',dict(versions))
print('Forward outputs',sum(len(j['indices']) for j in jobs),'core timed hours',sum(elapsed)/3.6e6)
models={m['id']:m for m in p['models']};panel=set(p['panel_indices']);stats=[]
for name,m in models.items():
 indices,first=table[(name,'1.31.1','ORT_ENABLE_ALL',4)];mask=np.array([int(i) in panel for i in indices]);assert sum(mask)==256
 values=np.stack([table[(name,s['runtime'],s['optimization'],s['threads'])][1] for s in p['settings']]);small=values[:,mask];unstable=(small.min(0)<m['tau'])&(small.max(0)>=m['tau'])
 runtime_changes=[];optimization_changes=[]
 for o in p['configuration']['optimization_levels']:
  a=table[(name,'1.31.1',o,1)][1][mask];b=table[(name,'1.28.0',o,1)][1][mask];runtime_changes.append(int(sum((a>=m['tau'])!=(b>=m['tau']))))
 for v in ['1.28.0','1.31.1']:
  a=table[(name,v,'ORT_ENABLE_BASIC',1)][1][mask];b=table[(name,v,'ORT_ENABLE_EXTENDED',1)][1][mask];optimization_changes.append(int(sum((a>=m['tau'])!=(b>=m['tau']))))
  assert np.array_equal(table[(name,v,'ORT_DISABLE_ALL',1)][1],table[(name,v,'ORT_ENABLE_BASIC',1)][1])
  assert np.array_equal(table[(name,v,'ORT_ENABLE_EXTENDED',1)][1],table[(name,v,'ORT_ENABLE_ALL',1)][1])
 for v in ['1.28.0','1.31.1']:
  for o in p['configuration']['optimization_levels']:assert np.array_equal(table[(name,v,o,1)][1],table[(name,v,o,4)][1])
 stats.append(dict(model=name,arm=m['arm'],seed=m['seed'],variant=m['variant'],n_common_panel=256,unstable_neural_messages=int(sum(unstable)),runtime_changed_decisions_max=max(runtime_changes),basic_to_extended_changed_decisions_max=max(optimization_changes)))
print('Exact DISABLED=BASIC and EXTENDED=ALL and thread1=4 for EVERY artifact/runtime (not just decisions).')
for kind in ['fp32','quantized']:
 rr=[r for r in stats if (r['variant']=='fp32')==(kind=='fp32')]
 print(kind,'artifacts',len(rr),'unstable',sum(r['unstable_neural_messages']>0 for r in rr),'runtime changing',sum(r['runtime_changed_decisions_max']>0 for r in rr),'optimization changing',sum(r['basic_to_extended_changed_decisions_max']>0 for r in rr),'ranges',min(r['unstable_neural_messages'] for r in rr),max(r['unstable_neural_messages'] for r in rr))
 # Per-artifact records are written to CSV below.
with (R/'independently_verified_common_panel.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
print('ENVIRONMENT',[(r['runtime'],r['numpy'],r['build'][:130]) for r in json.loads((R/'runtime_environments.json').read_text())])
