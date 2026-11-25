"""Generate the compact Supplement S11 table only after independent verification."""
from pathlib import Path
import csv,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 subprocess.run([sys.executable,str(ROOT/'scripts/verify_completed_single_graph_pilot.py')],check=True)
 rows=list(csv.DictReader((ROOT/'.reproduced/single_graph_precision_pilot/verified_summary.csv').open()))
 text=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Calibration-only single-graph pilot on 256 messages. Losses/additions refer to FP32-reference decisions, not source labels. Slash-separated counts are for runtimes 1.31.1/1.28.0. Total is the worst-runtime disagreement count; drift is the cross-runtime decision count. Neither row meets the required zero disagreement.}\label{tab:single_graph_pilot}',r'\begin{tabularx}{\linewidth}{@{}Xrrrrr@{}}',r'\toprule Configuration & MB & Losses & Added & Total & Drift\\\midrule']
 for r in rows:text.append(r['configuration']+f" & {float(r['size_mb']):.1f} & {r['losses_1_31_1']}/{r['losses_1_28_0']} & {r['added_reference_negative_alerts']}/{r['added_reference_negative_alerts']} & {r['worst_total_disagreements']} & {r['cross_runtime_disagreements']}"+r'\\')
 text +=[r'\bottomrule\end{tabularx}',r'\end{table}'];(ROOT/'paper/tables/single_graph_pilot.tex').write_text('\n'.join(text)+'\n')
 print('Generated Supplement Table S13; no new main figure.')
if __name__=='__main__':main()
