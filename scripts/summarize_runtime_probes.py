#!/usr/bin/env python3
"""Recompute supplementary diagnostic counts from text-free recorded outputs."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=_result_root(ROOT)/'runtime_diagnostics';probes=json.loads((folder/'local_probes.json').read_text())
    tau=json.loads((ROOT/'manifests/primary_model_files.json').read_text())['tau']
    ref={r['id']:r for r in probes[0]['rows']};assert len(ref)==256
    def record(runtime,setting,values):
        assert set(values)==set(ref)
        return dict(runtime=runtime,setting=setting,n=256,kaggle_probability_failures=sum(abs(p-ref[k]['kaggle'])>1e-5+1e-5*abs(ref[k]['kaggle']) for k,p in values.items()),android_probability_failures=sum(abs(p-ref[k]['android'])>1e-5+1e-5*abs(ref[k]['android']) for k,p in values.items()),kaggle_neural_decision_changes=sum((p>=tau)!=(ref[k]['kaggle']>=tau) for k,p in values.items()),max_kaggle_probability_difference=max(abs(p-ref[k]['kaggle']) for k,p in values.items()))
    rows=[]
    for file in sorted(folder.glob('ort128_*.tsv')):
        values={s:float(p) for s,p in csv.reader(file.open(),delimiter='\t')};r=record('1.28.0',file.stem.replace('ort128_',''),values)
        assert r['kaggle_probability_failures']==197 and r['android_probability_failures']==0 and r['kaggle_neural_decision_changes']==3
        rows.append(r)
    for probe in probes:rows.append(record(probe['runtime'],probe['setting'],{r['id']:r['p'] for r in probe['rows']}))
    with (folder/'probe_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Recomputed',len(rows),'small-packet diagnostic summaries; no inference or causal attribution.')
if __name__=='__main__':main()
