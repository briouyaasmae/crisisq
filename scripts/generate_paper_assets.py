#!/usr/bin/env python3
"""Generate manuscript tables and scientific plots from the released result tables."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
DATA=_result_root(ROOT)/'device/consolidated';PAPER=ROOT/'paper';(PAPER/'figures').mkdir(parents=True,exist_ok=True);(PAPER/'tables').mkdir(exist_ok=True)
def read(n):return list(csv.DictReader((DATA/n).open()))
def main():
    latency=read('timing_three_run_summary.csv');classification=read('full_device_classification_metrics.csv');interval=json.loads((DATA/'group_bootstrap_intervals.json').read_text())
    devices=['HONOR HEY2-W09','Samsung SM-A225F'];profiles=['binary_test','labeled_hard_negative','positive_probe','support_unlabeled'];labels=['Binary test','Enriched source negatives','Probe','Unlabeled support']
    lookup={(r['device'],r['profile'],r['variant'],r['mode']):r for r in latency}
    text=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Hybrid pipeline latency in milliseconds: mean of three run means. Each run uses 64 messages/profile and five repetitions/message. Ratios compare FP32 hybrid with QUInt8 hybrid.}\label{tab:latency}',r'\begin{tabular}{llrrr}',r'\toprule Device & Profile & FP32 & QUInt8 & Ratio \\',r'\midrule']
    for device in devices:
        for profile,label in zip(profiles,labels):
            f=float(lookup[device,profile,'fp32','hybrid']['mean_of_run_mean_ms']);q=float(lookup[device,profile,'q_quint8','hybrid']['mean_of_run_mean_ms'])
            text.append(f"{'HONOR' if device.startswith('HONOR') else 'Samsung'} & {label} & {f:.2f} & {q:.2f} & {f/q:.2f}\\\\")
        if device==devices[0]:text.append(r'\midrule')
    text += [r'\bottomrule\end{tabular}',r'\end{table}'];(PAPER/'tables/latency.tex').write_text('\n'.join(text)+'\n')
    text=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Full device classification, identical on both Android devices. Binary: 500 positives/500 negatives; probe: 300/100. FN and FP use original source labels and count unique messages. The separate stress panel reports alert rate on keyword-enriched source-negative messages (300 examples), whose labels were not independently audited.}\label{tab:classification}',r'\begin{tabular}{lllrrrr}',r'\toprule Partition & Model & Mode & Recall (\%) & FPR (\%) & FN & FP \\',r'\midrule']
    for part,label in [('binary_test','Binary'),('positive_probe','Probe')]:
        for r in classification:
            if r['partition']!=part:continue
            recall=f"{100*float(r['recall']):.1f}";fpr=f"{100*float(r['false_positive_rate']):.1f}"
            text.append(f"{label} & {'FP32' if r['variant']=='fp32' else 'QUInt8'} & {r['mode']} & {recall} & {fpr} & {r['fn']} & {r['fp']}\\\\")
        if part=='binary_test':text.append(r'\midrule')
    text += [r'\bottomrule\end{tabular}\par\medskip',r'\begin{tabular}{llrr}',r'\toprule Model & Mode & Alerts/300 & Alert rate (\%) \\',r'\midrule']
    for r in classification:
        if r['partition']=='labeled_hard_negative':text.append(f"{'FP32' if r['variant']=='fp32' else 'QUInt8'} & {r['mode']} & {r['fp']} & {100*float(r['false_positive_rate']):.1f}\\\\")
    text += [r'\bottomrule\end{tabular}',r'\end{table}'];(PAPER/'tables/classification.tex').write_text('\n'.join(text)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.9));x=np.arange(4);width=.26
    for ax,device in zip(axes,devices):
        for variant,offset,color,label in [('fp32',-.24,'#486b8e','FP32 hybrid'),('q_quint8',.24,'#168275','QUInt8 hybrid')]:
            means=[float(lookup[device,p,variant,'hybrid']['mean_of_run_mean_ms']) for p in profiles];sd=[float(lookup[device,p,variant,'hybrid']['sd_of_run_mean_ms']) for p in profiles]
            bars=ax.bar(x+offset,means,width,label=label,color=color,yerr=sd,capsize=3)
            for bar,value,error in zip(bars,means,sd):
                ax.annotate(f'{value:.1f}',xy=(bar.get_x()+bar.get_width()/2,value+error),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=8,clip_on=False)
        ax.set_title(device,weight='bold',pad=12);ax.set_xticks(x,['Binary','Enriched\nsource negatives','Probe','Unlabeled\nsupport']);ax.set_ylabel('Mean pipeline latency (ms)');ax.grid(axis='y',alpha=.2);ax.set_ylim(0,max(float(lookup[device,p,'fp32','hybrid']['mean_of_run_mean_ms']) +float(lookup[device,p,'fp32','hybrid']['sd_of_run_mean_ms']) for p in profiles)*1.24)
    handles,legend_labels=axes[0].get_legend_handles_labels();fig.legend(handles,legend_labels,frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.5,1),ncol=2);fig.subplots_adjust(left=.09,right=.99,bottom=.20,top=.76,wspace=.32);fig.savefig(PAPER/'figures/device_latency.png',dpi=600);fig.savefig(PAPER/'figures/device_latency.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(6.3,3.7),sharey=True);systems=[('fp32','neural'),('fp32','hybrid'),('q_quint8','neural'),('q_quint8','hybrid')];colors=['#486b8e','#91aeca','#168275','#d19d3f'];clook={(r['partition'],r['variant'],r['mode']):r for r in classification}
    for ax,(part,metric,title) in zip(axes,[('binary_test','recall','Main recall\n1,000 messages'),('positive_probe','recall','Probe recall\n400 messages'),('labeled_hard_negative','false_positive_rate','Keyword-enriched\nsource negatives\nAlert rate (n=300)')]):
        values=[100*float(clook[part,v,m][metric]) for v,m in systems];lo=[];hi=[]
        for value,(variant,mode) in zip(values,systems):
            ci=interval[part]['intervals'][variant+'/'+mode+'.'+metric];lo.append(value-100*ci['lower_95']);hi.append(100*ci['upper_95']-value)
        bars=ax.bar(np.arange(4),values,color=colors,yerr=[lo,hi],capsize=3)
        for bar,value,upper in zip(bars,values,hi):ax.text(bar.get_x()+bar.get_width()/2,value+upper+1.2,f'{value:.1f}',ha='center',fontsize=9.5)
        ax.set_title(title,fontsize=10,weight='bold');ax.set_xticks(np.arange(4),['FP32\nN','FP32\nH','Q8\nN','Q8\nH']);ax.set_ylabel('Percent');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(PAPER/'figures/classification_tradeoffs.png',dpi=600);fig.savefig(PAPER/'figures/classification_tradeoffs.svg');plt.close(fig)
    from generate_runtime_assets import main as runtime_assets
    runtime_assets()
    from generate_completed_matrix_assets import main as completed_assets
    completed_assets()
    from generate_followup_assets import main as followup_assets
    followup_assets()
    from generate_framework_asset import main as framework_asset
    framework_asset()
    from generate_additional_audit_assets import main as additional_assets
    additional_assets()
    from generate_selective_precision_assets import main as selective_assets
    selective_assets()
    from generate_single_graph_pilot_assets import main as single_graph_assets
    single_graph_assets()
    print('Generated framework, verified results, mitigation, compact pilot and post-hoc assets.')
if __name__=='__main__':main()
