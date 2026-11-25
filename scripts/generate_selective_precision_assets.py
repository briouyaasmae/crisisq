"""Reproduce selective-precision figures/tables from verified completed reports."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=_result_root(ROOT)/'selective_precision';P=ROOT/'paper'
def read(n):return list(csv.DictReader((R/n).open()))
def main():
 subprocess.run([sys.executable,str(ROOT/'scripts/verify_completed_selective_precision.py')],check=True)
 metrics=read('metrics.csv');timing=read('timing_summary.csv');rand=read('matched_random_reference.csv')
 m={(r['model'],r['runtime'],r['partition'],r['policy']):r for r in metrics};t={(r['runtime'],r['partition'],r['policy']):r for r in timing}
 versions=['1.31.1','1.28.0'];parts=['binary_test','sdcnl_authors_test'];colors={'always_compressed':'#168275','always_fp32':'#486b8e','one_sided_05':'#d19d3f'}
 plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig=plt.figure(figsize=(7.0,5.6),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.05]);ax=fig.add_subplot(gs[0,:]);bx=fig.add_subplot(gs[1,0]);cx=fig.add_subplot(gs[1,1]);conditions=[(v,p) for v in versions for p in parts];x=np.arange(4);labels=['1.31.1\nTwitter','1.31.1\nSDCNL','1.28.0\nTwitter','1.28.0\nSDCNL']
 for policy,offset,label in [('always_compressed',-.16,'QUInt8 hybrid'),('one_sided_05',.16,'One-sided fallback')]:
  vals=[int(m['saqs__42',v,p,policy]['positive_losses_vs_fp32']) for v,p in conditions];bars=ax.bar(x+offset,vals,.3,color=colors[policy],label=label)
  for b,n in zip(bars,vals):ax.annotate(str(n),(b.get_x()+b.get_width()/2,n),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
 ax.set(xticks=x,xticklabels=labels,ylabel='Detected source positives lost\nrelative to FP32',ylim=(0,12),title='A. Primary-model recovery on evaluation messages');ax.legend(frameon=False,ncol=2,loc='upper right');ax.grid(axis='y',alpha=.18)
 for policy,label,marker in [('always_compressed','QUInt8 hybrid','o'),('one_sided_05','Fallback','s')]:
  vals=[float(t[v,p,policy]['speedup_vs_fp32']) for v,p in conditions];bx.plot(x,vals,marker=marker,linestyle="None",color=colors[policy],label=label)
 bx.axhline(1,color='#555555',lw=1,ls='--');bx.set(xticks=x,xticklabels=labels,ylabel='FP32 latency / policy latency',ylim=(.95,1.13),title='B. Host speed relative to FP32');bx.tick_params(axis='x',labelsize=8);bx.legend(frameon=False,fontsize=8);bx.grid(axis='y',alpha=.18)
 for policy,offset,label in [('always_compressed',-.25,'QUInt8'),('always_fp32',0,'FP32'),('one_sided_05',.25,'Fallback')]:
  vals=[float(t[v,'binary_test',policy]['mean_post_warm_pss_mib']) for v in versions];bars=cx.bar(np.arange(2)+offset,vals,.23,color=colors[policy],label=label)
  for b,n in zip(bars,vals):cx.annotate(f'{n:.0f}',(b.get_x()+b.get_width()/2,n),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
 cx.set(xticks=np.arange(2),xticklabels=versions,ylabel='Whole-process post-warm PSS (MiB)',ylim=(0,570),title='C. Fallback increases memory');cx.legend(frameon=False,fontsize=8,ncol=3,loc='upper left');cx.grid(axis='y',alpha=.18)
 for suffix in ['png','svg']:fig.savefig(P/'figures'/('selective_precision_recovery_cost.'+suffix),dpi=400)
 plt.close(fig)
 def table(name,caption,label,cols,head,body):
  text=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabularx}{\linewidth}{@{}'+cols+'@{}}',r'\toprule '+head+r'\\',r'\midrule']+body+[r'\bottomrule\end{tabularx}',r'\end{table}'];(P/'tables'/name).write_text('\n'.join(text)+'\n')
 body=[]
 for model in ['saqs__42','saqs__123','saqs__777','ce__42']:
  for version in versions:
   for part in parts:
    q=m[model,version,part,'always_compressed'];f=m[model,version,part,'one_sided_05'];label=model.replace('__',' ')
    body.append(f"{label} & {version} & {'Twitter' if part=='binary_test' else 'SDCNL'} & {q['positive_losses_vs_fp32']} $\\to$ {f['positive_losses_vs_fp32']} & +{f['added_source_negative_alerts_vs_quantized']} & {f['fallback_calls']}/{f['n']}"+r'\\')
 table('selective_recovery_all.tex',r'Primary one-sided policy across all checkpoints and runtimes. Losses are FP32-detected source positives lost under compression; alerts are added source-negative alerts. Twitter has 872 evaluation messages and SDCNL 379. SAQS777 uses reduced-range quantization.','tab:selective_all','llXrrr','Model & Runtime & Source & Losses & Added alerts & FP32 calls',body)
 body=[]
 for version in versions:
  for part in parts:
   for policy,label in [('always_compressed','QUInt8'),('always_fp32','FP32'),('one_sided_05','Fallback')]:
    a=t[version,part,policy];body.append(f"{version} & {'Twitter' if part=='binary_test' else 'SDCNL'} & {label} & {float(a['mean_of_run_means_ms']):.2f} ({float(a['sd_of_run_means_ms']):.2f}) & {float(a['speedup_vs_fp32']):.3f} & {float(a['mean_post_warm_pss_mib']):.1f}"+r'\\')
 table('selective_timing.tex','Primary-model host pipeline timing: mean (SD) of three independent process-run means, each with 64 messages per source. Ratios are FP32 latency divided by policy latency. PSS averages post-warm whole-process snapshots; each process measures both sources, so its PSS value is shared. Initialization, all control policies and peak RSS are retained in the CSV. These are not phone measurements.','tab:selective_timing','llXrrr','Runtime & Source & Policy & Latency, ms & Ratio & PSS, MiB',body)
 body=[]
 for version in versions:
  for part in parts:
   for policy,label in [('one_sided_05','One-sided'),('symmetric_05','Symmetric'),('random_one_sided_05_matched','Random')]:
    a=m['saqs__42',version,part,policy];body.append(f"{version} & {'Twitter' if part=='binary_test' else 'SDCNL'} & {label} & {a['positive_gains_vs_quantized']} & {a['positive_losses_vs_quantized']} & +{a['added_source_negative_alerts_vs_quantized']} & {a['fallback_calls']}"+r'\\')
 table('selective_controls.tex','Primary-model controls. Positive gains/losses and added source-negative alerts are relative to always-QUInt8 hybrid. Random routing matches the observed one-sided call count within each partition among eligible negative decisions. This is an experimental call-count control. Symmetric routing shares the nominal calibration allowance but can have a different evaluation call count.','tab:selective_controls','llXrrrr','Runtime & Source & Policy & Gains & Losses & Alerts & Calls',body)
 print('Generated selective-precision Figure5 and Supplement TablesS10–S12.')
if __name__=='__main__':main()
