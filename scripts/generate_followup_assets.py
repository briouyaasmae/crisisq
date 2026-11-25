#!/usr/bin/env python3
"""Generate publication figures/tables from unchanged completed reports."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import csv,json,subprocess,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'paper';FIG=PAPER/'figures';TABLE=PAPER/'tables';FIG.mkdir(exist_ok=True);TABLE.mkdir(exist_ok=True)
def rows(path):return list(csv.DictReader(path.open()))
def main():
 subprocess.run([sys.executable,str(ROOT/'scripts/verify_completed_followups.py')],check=True)
 f=rows(_result_root(ROOT)/'fusion_localization/fusion_contrasts.csv');r=rows(_result_root(ROOT)/'fusion_localization/matched_runtime_contrasts.csv')
 profiles=['extended','disable_DynamicQuantizeMatMulFusion__MatMulIntegerToFloatFusion','disable_SkipLayerNormFusion','disable_MatMulScaleFusion'];labels=['Baseline','Integer-MM\noff','Skip-LN\noff','Scale\noff']
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240,'font.family':'DejaVu Sans'})
 fig,ax=plt.subplots(1,2,figsize=(6.8,3.8),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained');x=np.arange(4);colors=['#28658c','#d27a2c']
 for i,version in enumerate(['1.28.0','1.31.1']):
  a=[next(v for v in f if v['model'].endswith('q_quint8') and v['runtime']==version and v['profile']==p) for p in profiles];values=[int(v['neural_disagreements_vs_basic']) for v in a]
  bars=ax[0].bar(x+(i-.5)*.35,values,.33,label='ORT '+version,color=colors[i]);ax[0].bar_label(bars,padding=3,fontsize=9)
 ax[0].set(xticks=x,xticklabels=labels,ylabel='Neural decisions differing from BASIC',ylim=(0,12),title='A. Recovery toward BASIC');ax[0].legend(frameon=False,loc='upper right',fontsize=8);ax[0].grid(axis='y',alpha=.16);ax[0].set_axisbelow(True)
 values=[int(next(v for v in r if v['model'].endswith('q_quint8') and v['profile']==p)['neural_decision_differences']) for p in profiles];bars=ax[1].bar(x,values,.62,color=['#7e8990','#7e8990','#538d82','#b37672']);ax[1].bar_label(bars,padding=3,fontsize=9)
 ax[1].set(xticks=x,xticklabels=labels,ylabel='Neural decisions differing across runtimes',ylim=(0,18),title='B. Runtime disagreement');ax[1].grid(axis='y',alpha=.16);ax[1].set_axisbelow(True)
 for a in ax:a.tick_params(axis='x',labelsize=8)
 fig.savefig(FIG/'fusion_localization.png');fig.savefig(FIG/'fusion_localization.pdf');plt.close(fig)
 a=rows(_result_root(ROOT)/'sdcnl/metrics.csv');selected=[next(r for r in a if r['panel']=='authors_test_clean_primary' and r['model']==m and r['mode']==mode) for m,mode in [('saqs__42__fp32','neural'),('saqs__42__fp32','hybrid'),('saqs__42__q_quint8','neural'),('saqs__42__q_quint8','hybrid')]]
 lines=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{3.5pt}',r'\caption{SDCNL published-test source-label outcomes: 193 SuicideWatch posts and 186 depression posts. Intervals are 95\% author-cluster percentile bootstrap intervals. Alerts on depression posts are not adjudicated false positives.}\label{tab:sdcnl}',r'\begin{tabular}{lrrrr}',r'\toprule System & Detected/193 & Alerts/186 & Recall (\%) [CI] & Alert rate (\%) [CI] \\',r'\midrule']
 for v in selected:
  name=('FP32' if v['model'].endswith('fp32') else 'QUInt8')+' '+v['mode'];rc,lo,hi=[100*float(v[k]) for k in ['source_positive_recall','recall_ci_low','recall_ci_high']];al,alo,ahi=[100*float(v[k]) for k in ['source_negative_alert_rate','negative_alert_ci_low','negative_alert_ci_high']]
  lines.append(f"{name} & {v['detected_source_positive']} & {v['alerts_on_source_negative']} & {rc:.1f} [{lo:.1f}, {hi:.1f}] & {al:.1f} [{alo:.1f}, {ahi:.1f}]" + " " + chr(92)*2)
 lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(TABLE/'sdcnl_primary.tex').write_text('\n'.join(lines)+'\n')
 # Distinguishable markers/legend avoid placing text labels over uncertainty bars.
 fig,ax=plt.subplots(figsize=(7.1,4.7),layout='constrained')
 for v,marker,color in zip(selected,['o','s','^','D'],['#306b93','#6b9dc0','#c57430','#e1a266']):
  xx,lo,hi=[100*float(v[k]) for k in ['source_negative_alert_rate','negative_alert_ci_low','negative_alert_ci_high']];yy,ylo,yhi=[100*float(v[k]) for k in ['source_positive_recall','recall_ci_low','recall_ci_high']];name=('FP32' if v['model'].endswith('fp32') else 'QUInt8')+' '+v['mode']
  ax.errorbar(xx,yy,xerr=[[xx-lo],[hi-xx]],yerr=[[yy-ylo],[yhi-yy]],fmt=marker,color=color,ms=7,capsize=3,lw=1.2,label=name)
 ax.set(xlabel='Alerts on source-negative posts (%)',ylabel='Source-positive recall (%)',xlim=(37,62),ylim=(74,94));ax.grid(alpha=.2);ax.legend(frameon=False,loc='lower right');fig.savefig(FIG/'sdcnl_tradeoff.png');fig.savefig(FIG/'sdcnl_tradeoff.pdf');plt.close(fig)
 lines=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{3.5pt}',r'\caption{Selected primary fusion interventions. $D_B$ counts neural disagreements with BASIC; $R/N$ are recovered/new disagreements relative to BASIC; $D_E^H$ counts hybrid changes relative to EXTENDED. Each profile scores 1,700 messages.}\label{tab:fusion_supp}',r'\begin{tabular}{llrrrr}',r'\toprule Runtime & EXTENDED profile & Nodes & $D_B$ & $R/N$ & $D_E^H$ \\',r'\midrule']
 short=['Baseline','Integer-MM disabled','Skip-layer norm disabled','MatMul scale disabled']
 for version in ['1.28.0','1.31.1']:
  for profile,name in zip(profiles,short):
   v=next(v for v in f if v['model'].endswith('q_quint8') and v['runtime']==version and v['profile']==profile)
   lines.append(f"{version} & {name} & {v['graph_nodes']} & {v['neural_disagreements_vs_basic']} & {v['basic_extended_disagreements_recovered']}/{v['new_disagreements_with_basic']} & {v['hybrid_decisions_changed_vs_extended']}" + " " + chr(92)*2)
 lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(TABLE/'fusion_interventions.tex').write_text('\n'.join(lines)+'\n')
 intervals=rows(_result_root(ROOT)/'sdcnl/ce_saqs_paired_seed_intervals.csv');lines=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{3.5pt}',r'\caption{SDCNL published-test SAQS-minus-CE differences in percentage points across seven jointly accepted available seed pairs. Intervals resample paired seeds; they are conditional on selection. Selected quantizers and thresholds may differ.}\label{tab:sdcnl_seeds}',r'\begin{tabular}{llrr}',r'\toprule Precision policy & Mode & Recall difference [CI] & Negative-alert difference [CI] \\',r'\midrule']
 for kind,name in [('fp32','FP32'),('source_selected_quantized','Selected compressed')]:
  for mode in ['neural','hybrid']:
   formatted=[]
   for metric in ['delta_recall','delta_source_negative_alert_rate']:
    r=next(v for v in intervals if v['kind']==kind and v['mode']==mode and v['panel']=='authors_test_clean_primary' and v['metric']==metric);mu,lo,hi=[100*float(r[k]) for k in ['mean_delta','ci_low','ci_high']];formatted.append(f'{mu:+.2f} [{lo:.2f}, {hi:.2f}]')
   lines.append(f'{name} & {mode} & '+ ' & '.join(formatted)+r' \\')
 lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(TABLE/'sdcnl_seed_tradeoffs.tex').write_text('\n'.join(lines)+'\n')
 print('Generated graph-localization and SDCNL figures/tables from verified completed reports.')
if __name__=='__main__':main()
