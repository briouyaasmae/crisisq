"""Independently verify the completed fixed matrix and generate its paper assets."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,json,subprocess,sys,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    with (cache/'completed_matrix_verification.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'scripts/verify_completed_runtime_matrix.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    received=ROOT/'.reproduced/runtime_matrix';paper=ROOT/'paper'
    artifacts=list(csv.DictReader((received/'independently_verified_common_panel.csv').open()))
    floats=[r for r in artifacts if r['variant']=='fp32'];quant=[r for r in artifacts if r['variant']!='fp32'];assert len(floats)==len(quant)==14
    counts=[sum(int(r[k])>0 for r in quant) for k in ['runtime_changed_decisions_max','basic_to_extended_changed_decisions_max']]+[0];assert counts==[13,12,0]
    cases=json.loads((received/'case_metadata.json').read_text());jobs=json.loads((received/'jobs_frozen.json').read_text());primary=[j for j in jobs if j['model_id']=='saqs__42__q_quint8'];assert len(primary)==16
    values=[]
    for j in primary:
        with np.load(received/'jobs'/j['id']/'predictions.npz',allow_pickle=False) as z:
            assert list(z['indices'])==list(range(1700));values.append(z['probabilities'].copy())
    suite=np.stack(values);tau=primary[0]['tau'];gate=np.array([r['gate_v2'] for r in cases]);unstable=(suite.min(0)<tau)&(suite.max(0)>=tau)
    neural=int(sum(unstable));hybrid=int(sum(unstable&(gate==-1)));assert (neural,hybrid)==(21,15)
    fig,axes=plt.subplots(1,2,figsize=(6.3,3.6),gridspec_kw={'width_ratios':[1.35,1]})
    ax=axes[0];x=np.arange(3);width=.32
    ax.bar(x-width/2,[0,0,0],width,color='#486b8e',label='FP32');bars=ax.bar(x+width/2,counts,width,color='#168275',label='Quantized')
    for at,val in zip(x+width/2,counts):ax.annotate(str(val),xy=(at,val),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    for at in x-width/2:ax.annotate('0',xy=(at,0),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9,color='#486b8e')
    ax.set_xticks(x,['Runtime\nwheel','BASIC →\nEXTENDED','1 → 4\nthreads'],fontsize=9);ax.set_ylim(0,16);ax.set_yticks([0,7,14]);ax.set_ylabel('Artifacts with changed decisions');ax.set_title('Common panel\n256 messages',fontsize=10,weight='bold');ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.015),ncol=2)
    ax=axes[1];bars=ax.bar([0,1],[neural,hybrid],color=['#168275','#d19d3f'],width=.55)
    for bar,value in zip(bars,[neural,hybrid]):ax.annotate(str(value),xy=(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xticks([0,1],['Neural','Hybrid']);ax.set_ylim(0,27);ax.set_ylabel('Messages with unstable decisions');ax.set_title('Primary QUInt8\n1,700 messages',fontsize=10,weight='bold')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
    fig.tight_layout(w_pad=1.4);fig.savefig(paper/'figures/controlled_runtime_matrix.png',dpi=600);fig.savefig(paper/'figures/controlled_runtime_matrix.svg');plt.close(fig)
    names={'q_quint8':'QUInt8','q_perchannel_rr':'PC-RR','qint8':'QInt8'};lookup={(r['arm'],int(r['seed'])):r for r in quant}
    text=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Common-panel neural decision instability across all sixteen settings: unique messages whose decisions vary, out of 256. Each seed has one CE and one SAQS checkpoint and their FP32/compressed artifacts. All fourteen FP32 counterparts have zero unstable decisions. Compression formats are development-selected, not randomized. PC-RR denotes per-channel reduced-range quantization.}\label{tab:matrix_artifacts}',r'\begin{tabular}{rlrlr}',r'\toprule Seed & CE format & CE unstable & SAQS format & SAQS unstable \\',r'\midrule']
    for seed in [42,123,777,1,2,3,4]:
        c=lookup['ce',seed];s=lookup['saqs',seed];text.append(f"{seed} & {names[c['variant']]} & {c['unstable_neural_messages']} & {names[s['variant']]} & {s['unstable_neural_messages']}\\\\")
    text += [r'\bottomrule\end{tabular}',r'\end{table}'];(paper/'tables/runtime_artifact_stability.tex').write_text('\n'.join(text)+'\n')
    hist=json.loads((ROOT/'experiments/runtime_matrix/historical_android_primary.json').read_text());android={r['source_id']:r['q_quint8'] for r in hist['rows']};ap=np.array([android[r['source_id']] for r in cases]);kp=np.array([r['reference']['q_quint8']['probability'] for r in cases]);y=np.array([r['label'] for r in cases]);part=np.array([r['partition'] for r in cases])
    text=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Primary QUInt8 numerical profiles at four threads. D/B denotes identical DISABLED/BASIC outputs; E/A denotes identical EXTENDED/ALL outputs. Columns K/A count neural decision disagreements against historical Kaggle/Android endpoints. Main recall and stress alerts use hybrid decisions. No row is selected as an Android fix.}\label{tab:matrix_profiles}',r'\begin{tabular}{llrrrr}',r'\toprule Runtime & Optimization & K & A & Main recall (\%) & Stress alerts/300 \\',r'\midrule']
    for version in ['1.28.0','1.31.1']:
        for optimization,label in [('ORT_DISABLE_ALL','D/B'),('ORT_ENABLE_ALL','E/A')]:
            j=next(j for j in primary if j['runtime']==version and j['optimization']==optimization and j['threads']==4)
            with np.load(received/'jobs'/j['id']/'predictions.npz',allow_pickle=False) as z:p=z['probabilities'].copy()
            h=np.where(gate==-1,p>=tau,gate==1);main=(part=='binary_test')&(y==1);stress=part=='labeled_hard_negative'
            text.append(f"{version} & {label} & {sum((p>=tau)!=(kp>=tau))} & {sum((p>=tau)!=(ap>=tau))} & {100*np.mean(h[main]):.1f} & {sum(h[stress])}\\\\")
    text += [r'\bottomrule\end{tabular}',r'\end{table}'];(paper/'tables/runtime_primary_profiles.tex').write_text('\n'.join(text)+'\n')
    print('Verified completed matrix; generated controlled figure and two supplementary tables.')
if __name__=='__main__':main()
