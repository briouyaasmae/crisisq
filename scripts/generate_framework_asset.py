#!/usr/bin/env python3
"""Render the implemented development/inference paths as a reproducible schematic."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

def main():
 out=ROOT/'paper/figures';out.mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(6.8,4.25));ax.set(xlim=(0,10),ylim=(0,6.2));ax.axis('off')
 def box(x,y,w,h,text,fill='#edf4f8',edge='#365e75',size=9):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.08',lw=1,facecolor=fill,edgecolor=edge));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
 def arrow(a,b,label=None,at=None):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=1,color='#365e75'))
  if label:ax.text(*(at or ((a[0]+b[0])/2,(a[1]+b[1])/2+.2)),label,ha='center',va='center',fontsize=8)
 ax.text(.15,5.95,'Development: constrain selection before evaluation',fontsize=11,weight='bold')
 values=[(.15,'Trained FP32\ncheckpoint'),(2.65,'Export / quantize\ncandidate graphs'),(5.15,'Check development\ndecision budgets'),(7.65,'Freeze accepted\ngraph + threshold')]
 for x,t in values:box(x,4.85,2.15,.83,t,size=8.7)
 for x in [.15,2.65,5.15]:arrow((x+2.18,5.27),(x+2.45,5.27))
 ax.text(6.22,4.62,'Reject infeasible candidates',ha='center',fontsize=8,color='#90551e')
 ax.text(.15,4.05,'Classification: apply the fixed hybrid decision policy',fontsize=11,weight='bold')
 box(.15,2.75,1.25,.85,'Input text');box(2.0,2.75,2.05,.85,'Linguistic gate\n0 / 1 / uncertain');box(5.2,2.75,2.05,.85,'Selected neural\nclassifier');box(8.15,2.75,1.65,.85,'Binary\ntext flag')
 arrow((1.43,3.18),(1.97,3.18));arrow((4.08,3.18),(5.17,3.18),'Uncertain',(4.62,3.48));arrow((7.28,3.18),(8.12,3.18),'Score ≥ τ',(7.69,3.79))
 arrow((3.02,2.72),(3.02,2.13));arrow((3.02,2.13),(8.98,2.13));arrow((8.98,2.13),(8.98,2.72));ax.text(6.0,1.91,'Direct rule label 0 / 1',ha='center',fontsize=8.5)
 box(.15,.2,9.65,1.1,'Separate evaluation audits\nSource-label losses / gains and alerting • seed variation\nRuntime / graph checks and selective precision • device / host cost',fill='#edf5f0',edge='#42695a',size=9)
 fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
 for suffix in ['png','pdf','svg']:fig.savefig(out/('crisisq_framework.'+suffix),dpi=300)
 plt.close(fig)
 print('Rendered the framework schematic from the documented implementation paths.')
if __name__=='__main__':main()
