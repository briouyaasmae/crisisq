#!/usr/bin/env python3
"""Plot frozen cross-environment outputs and tabulate both CE/SAQS modes."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,io,json,zipfile,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    all_runs=[]
    for file in sorted((_result_root(ROOT)/'device/raw').glob('crisisq_full_device_evaluation_*.zip')):
        with zipfile.ZipFile(file) as z:
            assert json.loads(z.read('completion.json'))['complete']
            rows=list(csv.DictReader(io.StringIO(z.read('full_device_predictions.csv').decode())))
        assert len(rows)==3400
        all_runs.append({(r['variant'],r['source_id']):r for r in rows})
    assert len(all_runs)==2
    first,second=all_runs
    assert set(first)==set(second)
    for key in first:
        assert all(first[key][field]==second[key][field] for field in ['text_hash','probability','prediction_neural','prediction_hybrid','reference_probability'])
    tau=json.loads((ROOT/'manifests/primary_model_files.json').read_text())['tau']
    fig,axes=plt.subplots(1,2,figsize=(6.3,3.5))
    summary=[]
    for ax,variant,name in zip(axes,['fp32','q_quint8'],['FP32','QUInt8']):
        rows=[r for (v,_),r in first.items() if v==variant];assert len(rows)==1700
        ref=np.array([float(r['reference_probability']) for r in rows]);device=np.array([float(r['probability']) for r in rows])
        mismatch=np.abs(device-ref)>(1e-5+1e-5*np.abs(ref));changed=(device>=tau)!=(ref>=tau)
        assert sum(mismatch)==(0 if variant=='fp32' else 1388)
        assert sum(changed)==(0 if variant=='fp32' else 12)
        assert all(bool(c)==(r['kaggle_decision_match']=='false') for c,r in zip(changed,rows))
        ax.plot([0,1],[0,1],color='#888888',lw=.8,zorder=1)
        ax.axhline(tau,color='#777777',ls='--',lw=.7);ax.axvline(tau,color='#777777',ls='--',lw=.7)
        ax.scatter(ref[~changed],device[~changed],s=9,c='#168275',alpha=.35,edgecolors='none',label='Same neural decision',zorder=2)
        ax.scatter(ref[changed],device[changed],s=30,c='#c43d3d',edgecolors='white',linewidths=.4,label='Changed neural decision',zorder=3)
        ax.text(.035,.97,f'{sum(mismatch):,}/1,700 outside tolerance\n{sum(changed)} changed neural decisions',transform=ax.transAxes,ha='left',va='top',fontsize=8.7,bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=2))
        ax.set(title=name,xlabel='Kaggle reference probability',ylabel='Android probability',xlim=(-.025,1.025),ylim=(-.025,1.025))
        ax.set_aspect('equal',adjustable='box');ax.spines[['top','right']].set_visible(False)
        summary.append(dict(variant=variant,n=1700,outside_probability_tolerance=int(sum(mismatch)),changed_neural_decisions=int(sum(changed)),max_absolute_probability_difference=float(np.max(np.abs(device-ref))),threshold=tau,devices_agree=True))
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.015),frameon=False,fontsize=9,markerscale=1.5)
    fig.subplots_adjust(left=.09,right=.99,bottom=.22,top=.90,wspace=.32)
    paper=ROOT/'paper';fig.savefig(paper/'figures/runtime_agreement.png',dpi=600);fig.savefig(paper/'figures/runtime_agreement.svg');plt.close(fig)
    target=_result_root(ROOT)/'posthoc';target.mkdir(exist_ok=True)
    with (target/'runtime_agreement_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    rows=list(csv.DictReader((_result_root(ROOT)/'external/ce_saqs_paired_seed_comparisons.csv').open()))
    lookup={(r['system'],r['partition'],r['metric']):r for r in rows if r['analysis']=='source_selected'}
    table=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{External Kaggle SAQS minus CE differences in percentage points, with 95\% paired-seed intervals. Both modes use the seven jointly development-accepted seed pairs and each arm\textquotesingle s selected quantizer/threshold. Conditioning on joint acceptance can bias the comparison; these are not unconditional arm or isolated loss effects.}\label{tab:arms}',r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrr@{}}',r'\toprule Outcome & Neural-only & Hybrid \\',r'\midrule']
    for part,metric,label in [('binary_test','recall','Main recall'),('binary_test','fpr','Main FPR'),('positive_probe','recall','Probe recall'),('positive_probe','fpr','Probe FPR'),('labeled_hard_negative','alert_rate','Keyword-enriched source-negative alert rate')]:
        values=[]
        for system in ['neural','hybrid_v2']:
            r=lookup[system,part,metric];assert int(r['paired_seeds'])==7
            values.append(f"{100*float(r['saqs_minus_ce']):+.2f} [{100*float(r['seed_delta_lo']):.2f}, {100*float(r['seed_delta_hi']):.2f}]")
        table.append(label+' & '+' & '.join(values)+r'\\')
    table += [r'\bottomrule\end{tabularx}',r'\end{table}']
    (paper/'tables/ce_saqs_external.tex').write_text('\n'.join(table)+'\n')
    print('Verified both device exports; generated runtime plot and both-mode CE–SAQS table.')
if __name__=='__main__':main()
