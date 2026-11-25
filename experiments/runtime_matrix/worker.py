"""Isolated CPU inference worker. No training, quantization, or threshold search."""
from pathlib import Path
import argparse,hashlib,json,os,platform,sys,time
import numpy as np
import onnxruntime as ort

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(path,obj):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def run(job_path):
    job=json.loads(Path(job_path).read_text());dest=Path(job['output']);dest.mkdir(parents=True,exist_ok=True)
    assert ort.__version__==job['runtime'],(ort.__version__,job['runtime'])
    if job.get('require_isolated',True):assert Path(ort.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'ORT imported outside isolated environment'
    assert sha(__file__)==job['worker_sha256'],'Worker implementation changed'
    assert sha(job['model_path'])==job['model_sha256'],'Model bytes changed'
    assert sha(job['inputs_path'])==job['inputs_sha256'],'Input bytes changed'
    data=np.load(job['inputs_path'],allow_pickle=False);ix=np.asarray(job['indices'],dtype=np.int64)
    ids=data['input_ids'][ix];masks=data['attention_mask'][ix]
    identity=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest();checkpoint=dest/'checkpoint.npz'
    n=len(ix);logits=np.full((n,2),np.nan,dtype=np.float64);elapsed=np.full(n,np.nan);done=0
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as old:
            assert str(old['identity'].item())==identity,'Checkpoint belongs to another frozen job'
            logits=old['logits'];elapsed=old['elapsed_ms'];done=int(old['done'].item())
    opts=ort.SessionOptions();opts.intra_op_num_threads=job['threads'];opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level=getattr(ort.GraphOptimizationLevel,job['optimization'])
    opts.add_session_config_entry('session.intra_op.allow_spinning','0')
    init=time.perf_counter();session=ort.InferenceSession(job['model_path'],sess_options=opts,providers=['CPUExecutionProvider']);init_ms=(time.perf_counter()-init)*1000
    expected={x.name:x for x in session.get_inputs()};assert {'input_ids','attention_mask'}<=set(expected)
    assert set(expected)<={'input_ids','attention_mask','token_type_ids'}
    for v in expected.values():assert v.type=='tensor(int64)',(v.name,v.type)
    def infer(k):
        feed={'input_ids':ids[k:k+1],'attention_mask':masks[k:k+1]}
        if 'token_type_ids' in expected:feed['token_type_ids']=np.zeros_like(ids[k:k+1])
        t=time.perf_counter();value=np.asarray(session.run(None,feed)[0],dtype=np.float64).reshape(-1);ms=(time.perf_counter()-t)*1000
        assert value.shape==(2,) and np.isfinite(value).all(),'Expected two finite logits'
        return value,ms
    for k in range(min(10,n)):infer(k)
    for k in range(done,n):
        logits[k],elapsed[k]=infer(k)
        if (k+1)%64==0 or k+1==n:
            tmp=checkpoint.with_suffix('.partial.npz');np.savez_compressed(tmp,identity=np.array(identity),done=np.array(k+1),logits=logits,elapsed_ms=elapsed);tmp.replace(checkpoint)
            print(f"{job['id']}: {k+1}/{n}",flush=True)
    # A fixed subset checks within-session repeatability; it is not a full re-run.
    repeat_indices=sorted(range(n),key=lambda k:hashlib.sha256(str(int(ix[k])).encode()).hexdigest())[:min(16,n)]
    maximum=0.;changes=0
    def probability(z):
        z=z-np.max(z,axis=-1,keepdims=True);e=np.exp(z);return (e/e.sum(axis=-1,keepdims=True))[...,1]
    probabilities=probability(logits)
    for k in repeat_indices:
        z,_=infer(k);p=float(probability(z));maximum=max(maximum,abs(p-probabilities[k]));changes+=int((p>=job['tau'])!=(probabilities[k]>=job['tau']))
    result=dest/'predictions.npz';tmp=result.with_suffix('.partial.npz')
    np.savez_compressed(tmp,indices=ix,probabilities=probabilities,logits=logits,elapsed_ms=elapsed);tmp.replace(result)
    assert sha(job['model_path'])==job['model_sha256'],'Model changed during inference'
    record=dict(job_id=job['id'],plan_hash=job['plan_hash'],job_identity=identity,complete=True,n=n,runtime=ort.__version__,numpy=np.__version__,ort_build=ort.get_build_info(),ort_module=str(ort.__file__),providers=session.get_providers(),session_init_ms=init_ms,repeat_cases=len(repeat_indices),repeat_max_probability_difference=maximum,repeat_decision_changes=changes,model_sha256=job['model_sha256'],inputs_sha256=job['inputs_sha256'],prediction_sha256=sha(result),platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),cpu_count=os.cpu_count(),python=sys.version)
    write_json(dest/'complete.json',record)
    print('COMPLETE',job['id'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');run(p.parse_args().job)
