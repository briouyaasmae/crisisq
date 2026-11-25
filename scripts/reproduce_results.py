#!/usr/bin/env python3
"""Revalidate the eight original device exports without training or model inference."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import argparse,subprocess,sys,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'.reproduced');p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    if not (ROOT/'app/assets/study/full_evaluation_cases.jsonl').exists():raise SystemExit('Run python scripts/materialize_public_assets.py first.')
    outputs=[]
    for path in sorted((_result_root(ROOT)/'device/raw').glob('*.zip')):
        full=path.name.startswith('crisisq_full_device_evaluation_');tool='analyze_full_device_evaluation.py' if full else 'analyze_device_results.py';dest=a.out/path.stem
        cmd=[sys.executable,str(ROOT/'app/tools'/tool),str(path),'--out',str(dest)]
        if full:cmd+=['--bootstrap',str(a.bootstrap)]
        subprocess.run(cmd,check=True);outputs.append(str(dest))
    print('PASS: all six timing and two full classification exports revalidated. Outputs:',*outputs,sep='\n')
if __name__=='__main__':main()
