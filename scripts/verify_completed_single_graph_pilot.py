"""Independently audit saved calibration-only pilot output; no inference/network."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,hashlib,io,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=_result_root(ROOT)/'single_graph_precision_pilot'
def digest(x):return _sha256_bytes(json.dumps(x,sort_keys=True,default=str).encode())
def main():
 archive=R/'raw/crisisq_single_graph_precision_pilot_v1_1_reports.zip'
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for line in z.read('SHA256SUMS.txt').decode().splitlines():
   expected,name=line.split('  ',1);data=z.read(name);assert _sha256_bytes(data)==expected,name
   materialized=R/name
   if materialized.exists():assert materialized.read_bytes()==data,name
   else:assert materialized.suffix=='.log','Missing supplied artifact: '+name
  read=lambda name:json.loads(z.read(name));completion=read('completion.json');selected=read('selection_frozen.json');meta=read('calibration_metadata.json');pair=read('models_frozen.json')[0];cfg=read('configuration_frozen.json');history=read('calibration_search.json');table=list(csv.DictReader(io.StringIO(z.read('calibration_candidates.csv').decode())));g=np.array([r['gate_v2'] for r in meta]);tau=pair['tau'];versions=['1.31.1','1.28.0']
  assert len(meta)==256 and all('label' not in r for r in meta);assert digest(meta)==selected['calibration_metadata_hash'];assert cfg['max_groups']==3 and cfg['size_limit_mb']==100 and cfg['runtime_versions']==versions
  assert completion==dict(complete=True,status='calibration_rejected',accepted=False,evaluation_run=False,phone_benefit_measured=False)
  assert selected['accepted'] is False and selected['reason']=='no_strict_improvement' and selected['groups']==[] and selected['candidate'] is None and selected['tau']==tau
  assert tau==0.6455836149936347 and selected['constraints']==dict(max_size_mb=100,max_groups=3,required_calibration_reference_disagreements=0)
  assert not any(n.startswith(('timing/','evaluation/')) or n in ['metrics.csv','evaluation_metadata.json','timing_summary.csv','phone_candidate_manifest.json'] or n.endswith('.onnx') for n in z.namelist())
  arrays={};n_jobs=0;count=0
  for name in z.namelist():
   if not(name.startswith('scores/') and name.endswith('/complete.json')):continue
   record=read(name);assert record['complete'] and record['n']==256 and record['repeat_decision_changes']==0;job=read('job_specs/'+record['job_id']+'.json');assert record['job_identity']==digest(job);assert job['tau']==tau and job['indices']==list(range(256)) and job['inputs_sha256']==selected['calibration_inputs_sha256'];assert record['inputs_sha256']==job['inputs_sha256'];assert job['threads']==4 and job['optimization']=='ORT_ENABLE_ALL'
   blob=z.read(name.replace('/complete.json','/predictions.npz'));assert _sha256_bytes(blob)==record['prediction_sha256']
   with np.load(io.BytesIO(blob),allow_pickle=False) as a:
    logits=a['logits'];e=np.exp(logits-logits.max(axis=1,keepdims=True));p=e[:,1]/e.sum(axis=1);assert np.array_equal(a['indices'],np.arange(256)) and np.allclose(a['probabilities'],p,rtol=0,atol=1e-15);arrays[record['job_id']]=a['probabilities'].copy()
   n_jobs+=1;count+=record['n']
  assert n_jobs==68 and count==17408
  def probability(model,v):return arrays['cal__'+model+'__'+v]
  def decisions(p):return np.where(g<0,p>=tau,g)
  references={v:decisions(probability('saqs__42__fp32',v)) for v in versions};verified={}
  for row in table:
   pp={v:probability(row['id'],v) for v in versions};dd={v:decisions(pp[v]) for v in versions};loss=[];extra=[];total=[];mse=[];per={}
   build=read('builds/'+row['id']+'.json');assert build['id']==row['id'] and build['source_sha256']==pair['fp32']['sha256'];assert abs(float(row['size_mb'])-build['size_mb'])<1e-12;assert row['groups'].split('|')==build['groups'] or row['groups']=='' and build['groups']==[]
   checks=build['node_checks'];assert len(checks)==38 and all(r['verified'] for r in checks);expected_nodes=[r['name'] for r in checks if r['precision']=='float32'];assert sorted(expected_nodes)==build['excluded_nodes'];assert all(r['precision'] in ['float32','uint8'] for r in checks)
   for v in versions:
    ref=references[v];d=dd[v];a=int(((ref==1)&(d==0)).sum());b=int(((ref==0)&(d==1)).sum());c=int((ref!=d).sum());m=float(np.mean((pp[v]-probability('saqs__42__fp32',v))**2));loss.append(a);extra.append(b);total.append(c);mse.append(m);per[v]=dict(losses=a,added=b,total=c)
   cross=int((dd[versions[0]]!=dd[versions[1]]).sum());expected=dict(worst_reference_positive_losses=max(loss),worst_added_reference_negative_alerts=max(extra),worst_total_disagreements=max(total),cross_runtime_disagreements=cross,worst_score_mse=max(mse))
   for key,value in expected.items():assert abs(float(row[key])-value)<1e-14,(row['id'],key)
   verified[row['id']]=dict(**expected,size_mb=build['size_mb'],groups=build['groups'],per_runtime=per)
  assert len(verified)==32 and selected['evaluated_candidates']==32
  rank=lambda r:(r['worst_reference_positive_losses'],r['worst_total_disagreements'],r['cross_runtime_disagreements'],r['worst_score_mse'])
  current=verified['saqs__42__regenerated_quantized'];last=None;skipped=0
  for stage in [1,2,3]:
   proposals=[r for r in history if r['stage']==stage];assert len(proposals)==14-stage;eligible=[]
   for r in proposals:
    assert len(r['groups'])==stage and set(current['groups'])<set(r['groups'])
    if r.get('skipped'):
     assert r['skipped']=='over_size_limit' and r['size_mb']>100;skipped+=1;continue
    t=verified[r['id']];assert all(abs(r[k]-t[k])<1e-14 for k in ['size_mb','worst_reference_positive_losses','worst_total_disagreements','cross_runtime_disagreements','worst_score_mse']);assert t['size_mb']<=100 and t['worst_total_disagreements']>0
    if rank(t)<rank(current):eligible.append((r['id'],t))
   if eligible:
    assert stage<3;last,current=min(eligible,key=lambda kv:(rank(kv[1]),kv[1]['size_mb'],kv[0]))
   else:assert stage==3
  assert skipped==5 and current['groups']==['layer_1_ffn','layer_2_attention'];assert current['worst_reference_positive_losses']==2 and current['worst_added_reference_negative_alerts']==1 and current['worst_total_disagreements']==3 and current['cross_runtime_disagreements']==0
  baseline=verified['saqs__42__regenerated_quantized']
  for v in versions:assert np.array_equal(probability('saqs__42__original_quantized',v),probability('saqs__42__regenerated_quantized',v))
  original=[]
  for name,r in [('Original QUInt8',baseline),('Best intermediate allocation',current)]:
   original.append(dict(configuration=name,size_mb=r['size_mb'],losses_1_31_1=r['per_runtime']['1.31.1']['losses'],losses_1_28_0=r['per_runtime']['1.28.0']['losses'],added_reference_negative_alerts=r['worst_added_reference_negative_alerts'],worst_total_disagreements=r['worst_total_disagreements'],cross_runtime_disagreements=r['cross_runtime_disagreements'],scope='256 calibration messages; FP32-reference decisions, not clinical labels'))
 dest=ROOT/'.reproduced/single_graph_precision_pilot';dest.mkdir(parents=True,exist_ok=True)
 with (dest/'verified_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(original[0]));w.writeheader();w.writerows(original)
 text='PASS: immutable archive/checksums; 68 jobs/17,408 predictions; all 32 calibration allocations; 36 proposals/5 size exclusions; declared greedy ranks and rejection; no evaluation/timing/export.\nBest intermediate 88.588658 MB: losses 2/2 and added reference-negative alerts 1/1; worst disagreements 3; cross-runtime disagreements 0. Original 67.363342 MB: losses 4/5, added 2/2; worst disagreements 7; cross-runtime disagreements 1.\nNo zero-disagreement candidate under the tested bounded search. FP32 reference agreement is not source-label or clinical correctness. No phone or evaluation benefit established.\n'
 (dest/'VERIFIED_FINDINGS.txt').write_text(text);print(text)
if __name__=='__main__':main()
