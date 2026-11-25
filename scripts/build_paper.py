from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]/'paper'
def run(*args): subprocess.run(args,cwd=root,check=True)
if __name__=='__main__':
    run('pdflatex','-interaction=nonstopmode','-halt-on-error','manuscript.tex')
    run('bibtex','manuscript')
    for _ in range(2): run('pdflatex','-interaction=nonstopmode','-halt-on-error','manuscript.tex')
    for _ in range(2): run('pdflatex','-interaction=nonstopmode','-halt-on-error','supplementary.tex')
