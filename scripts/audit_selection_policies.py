"""Post-hoc comparison of already-recorded policies; no selection or inference."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv, hashlib, json
import numpy as np

def read(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))
def sha(path): return _sha256_bytes(Path(path).read_bytes())
def write(path,rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    paths=[source/'final_results.csv',source/'development_acceptance.csv']
    records,dev=map(read,paths)
    required={'arm','seed','variant','split','analysis','system','tau','recall','fpr','missed','false_alarms','n_pos','n_neg','size_mb'}
    assert records and required<=records[0].keys(), 'Unrecognized final-results schema'
    assert dev and {'arm','seed','system','accepted'}<=dev[0].keys()
    all_seeds=sorted({(r['arm'],r['seed']) for r in dev})
    accepted={(r['arm'],r['seed']) for r in dev if r['system']=='hybrid' and r['accepted'].lower()=='true'}
    # All policies and comparisons fixed in this code before computation.
    configs={'selected':('dev_selected',None,'hybrid_v2'),
             'fixed_signed_070':('fixed_0.70','qint8','hybrid_v2'),
             'fixed_unsigned_070':('fixed_0.70','q_quint8','hybrid_v2'),
             'signed_calibration_R90':('calibration_R90','qint8','hybrid'),
             'fp32_070':('fixed_0.70','fp32','hybrid_v2')}
    index={}
    for policy,(analysis,variant,system) in configs.items():
        for r in records:
            if r['analysis']!=analysis or r['system']!=system or (variant and r['variant']!=variant):continue
            k=(r['arm'],r['seed'],r['split'],policy)
            assert k not in index, f'Duplicate policy record {k}'
            for field in ['tau','recall','fpr','size_mb']:float(r[field] or 'nan')
            index[k]=r
    coverage=[];paired=[];summary=[]
    for arm in sorted({a for a,s in all_seeds}):
        seeds=[s for a,s in all_seeds if a==arm];passed=[s for s in seeds if (arm,s) in accepted]
        observed={s for a,s,split,policy in index if a==arm and policy=='selected'}
        assert observed==set(passed), f'Acceptance/selected-report mismatch: {arm}'
        coverage.append(dict(arm=arm,all_seeds=len(seeds),accepted_seeds=len(passed),rejected_seeds=len(seeds)-len(passed),rejection_handling='no fallback; no zero-loss or zero-alert imputation'))
        for split in sorted({r['split'] for r in records}):
            for baseline in configs:
                if baseline=='selected':continue
                rows=[]
                for seed in passed:
                    a=index.get((arm,seed,split,'selected'));b=index.get((arm,seed,split,baseline))
                    if not a or not b:continue
                    assert (a['n_pos'],a['n_neg'],a['history'])==(b['n_pos'],b['n_neg'],b['history'])
                    row=dict(arm=arm,seed=seed,split=split,baseline=baseline,selected_variant=a['variant'],selected_tau=float(a['tau']),baseline_variant=b['variant'],baseline_tau=float(b['tau']),history=a['history'])
                    for field in ['recall','fpr','missed','false_alarms','size_mb']:
                        av,bv=float(a[field] or 'nan'),float(b[field] or 'nan');row['selected_'+field]=av;row['baseline_'+field]=bv;row['delta_'+field]=av-bv
                    rows.append(row);paired.append(row)
                if not rows:continue
                for metric in ['recall','fpr','missed','false_alarms','size_mb']:
                    vals=np.array([r['delta_'+metric] for r in rows],float);vals=vals[np.isfinite(vals)]
                    if not len(vals):continue
                    rng=np.random.default_rng(20261115)
                    boots=vals[rng.integers(len(vals),size=(10000,len(vals)))].mean(axis=1)
                    lo,hi=np.quantile(boots,[.025,.975]) if len(vals)>1 else (np.nan,np.nan)
                    summary.append(dict(arm=arm,split=split,baseline=baseline,metric='source_negative_alert_rate' if metric=='fpr' else metric,n_seed_pairs=len(vals),mean_selected_minus_baseline=float(vals.mean()),seed_bootstrap_lo=float(lo),seed_bootstrap_hi=float(hi),positive_pairs=int((vals>1e-12).sum()),negative_pairs=int((vals< -1e-12).sum()),tied_pairs=int((abs(vals)<=1e-12).sum()),scope='post hoc; accepted seeds only; fixed messages; no message/population uncertainty'))
    write(output/'coverage.csv',coverage);write(output/'paired_policies.csv',paired);write(output/'paired_seed_summary.csv',summary)
    manifest=dict(source_sha256={p.name:sha(p) for p in paths},policies=configs,bootstrap_seed=20261115,bootstrap_samples=10000,training=False,inference=False,threshold_fitting=False,scope='post-hoc existing-policy comparison; previously inspected legacy sets; conditional on constrained acceptance',warnings=['Rejected seeds have no deployed selected policy; never imputed as zero errors.','Comparators differ in thresholds and possibly quantizer: this is a policy comparison, not an isolated budget effect.','Accepted-seed comparisons do not estimate unconditional deployment benefit.','Seed bootstrap does not measure unseen-source uncertainty.','Source-negative stress alerts are not adjudicated false positives.'])
    (output/'provenance.json').write_text(json.dumps(manifest,indent=2))
    (output/'READ_ME.txt').write_text('Existing-policy audit. Read coverage.csv before paired_seed_summary.csv.\nAll deltas are constrained-selected minus baseline. Recall/alert-rate values are fractions; multiply by100 for percentage points. Missed/alert counts are per-seed counts; size is decimal MB.\nNo policy is selected from these outcomes. No thresholds are fitted or models run. Rejection is reported as abstention from selection, not a zero-error system. Previously inspected legacy sets supply exploratory evidence only.\n')
    return summary
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.source,a.output)
