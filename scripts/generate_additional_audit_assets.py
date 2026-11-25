#!/usr/bin/env python3
"""Generate supplementary assets from explicitly post-hoc saved-output checks."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rows(p):return list(csv.DictReader(p.open()))
def main():
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_additional_evidence.py')],check=True)
 source=_result_root(ROOT)/'additional_audit';tables=ROOT/'paper/tables';figures=ROOT/'paper/figures';end=chr(92)*2
 data=rows(source/'sdcnl_compression_decisions_and_ranking.csv')
 text=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{3pt}',r'\caption{Post-hoc SDCNL published-test comparison of every selected compressed graph with its own FP32 counterpart at the same frozen threshold. $L/G$ count lost/gained source-positive hybrid detections among 193 posts. AUROC uses original neural scores. Selection is conditional on joint acceptance; formats are not randomized. RR denotes per-channel reduced-range quantization.}\label{tab:additional_compression}',r'\begin{tabular}{llrrrr}',r'\toprule Arm/seed & Quantizer & FP32 AUROC & INT8 AUROC & $L/G$ & Net loss '+end,r'\midrule']
 variants={'q_quint8':'QUInt8','qint8':'QInt8','q_perchannel_rr':'RR'}
 for r in data:
  text.append(f"{r['arm'].upper()} {r['seed']} & {variants[r['variant']]} & {float(r['fp32_auroc']):.4f} & {float(r['quantized_auroc']):.4f} & {r['hybrid_source_positives_lost']}/{r['hybrid_source_positives_gained']} & {r['hybrid_net_positive_loss']} "+end)
 text += [r'\bottomrule\end{tabular}',r'\end{table}'];(tables/'additional_compression.tex').write_text('\n'.join(text)+'\n')
 timing=rows(source/'isolated_routing_device_effects.csv');text=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{3pt}',r'\caption{Isolated routing latency reduction in percent, holding precision fixed. Values compare the mean of three original neural run means with the mean of three hybrid run means on the same 64 messages/profile. Negative values denote a higher hybrid mean. Source-label costs are reported separately; support has no gate exits.}\label{tab:additional_routing}',r'\begin{tabular}{llrrrr}',r'\toprule Device & Precision & Binary & Enriched negatives & Probe & Support '+end,r'\midrule']
 for device in sorted({r['device'] for r in timing}):
  for variant,name in [('fp32','FP32'),('q_quint8','QUInt8')]:
   vals=[next(r for r in timing if r['device']==device and r['variant']==variant and r['profile']==p) for p in ['binary_test','labeled_hard_negative','positive_probe','support_unlabeled']]
   text.append(('HONOR' if device.startswith('HONOR') else 'Samsung')+' & '+name+' & '+' & '.join(f"{float(v['reduction_percent']):+.2f}" for v in vals)+' '+end)
 text += [r'\bottomrule\end{tabular}',r'\end{table}'];(tables/'additional_routing.tex').write_text('\n'.join(text)+'\n')
 controls=rows(source/'legacy_three_seed_factorial_contrasts.csv');text=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{3pt}',r'\caption{Post-hoc legacy factorial contrasts: mean differences in percentage points across three shared seeds, neural FP32 and fixed threshold 0.70. Each effect holds the other design factor fixed. Previously inspected data and three seeds limit generalization; these are descriptive means, not new independent tests.}\label{tab:additional_factorial}',r'\begin{tabular}{lrrrr}',r'\toprule Contrast & Main recall & Main FPR & Probe recall & Negative alert rate '+end,r'\midrule']
 for contrast,name in [('loss_without_augmentation','Loss, no augmentation'),('loss_with_augmentation','Loss, with augmentation'),('augmentation_with_ce','Augmentation, CE'),('augmentation_with_saqs_loss','Augmentation, SAQS loss')]:
  selected=[next(r for r in controls if r['contrast']==contrast and r['split']==split and r['metric']==metric) for split,metric in [('test','recall'),('test','fpr'),('probe_final','recall'),('hn_final','source_negative_alert_rate')]]
  text.append(name+' & '+' & '.join(f"{100*float(v['mean_delta']):+.2f}" for v in selected)+' '+end)
 text += [r'\bottomrule\end{tabular}',r'\end{table}'];(tables/'additional_factorial.tex').write_text('\n'.join(text)+'\n')
 plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(6.5,3.8),layout='constrained')
 for arm,color,marker in [('ce','#306b93','o'),('saqs','#c57430','s')]:
  a=[r for r in data if r['arm']==arm];ax.scatter([float(r['delta_auroc']) for r in a],[int(r['hybrid_net_positive_loss']) for r in a],c=color,marker=marker,s=45,label=arm.upper(),alpha=.9)
 primary=next(r for r in data if r['arm']=='saqs' and r['seed']=='42');x=float(primary['delta_auroc']);y=int(primary['hybrid_net_positive_loss']);ax.scatter([x],[y],s=125,facecolors='none',edgecolors='black',lw=1.2)
 ax.annotate('Prespecified primary',xy=(x,y),xytext=(-.0030,8.5),arrowprops=dict(arrowstyle='->',color='#444'),fontsize=9,ha='center')
 ax.axvline(0,color='#777',ls='--',lw=1);ax.axhline(0,color='#777',lw=.8);ax.set(xlabel='Neural AUROC change after quantization',ylabel='Net source-positive hybrid detections lost',ylim=(-.5,12.5),xlim=(-.014,.003));ax.grid(alpha=.16);ax.legend(frameon=False,loc='upper right')
 for suffix in ['png','pdf','svg']:fig.savefig(figures/('ranking_decision_tradeoff.'+suffix),dpi=300)
 plt.close(fig);print('Generated post-hoc Tables S6–S8 and saved-score Figure S4.')
if __name__=='__main__':main()
