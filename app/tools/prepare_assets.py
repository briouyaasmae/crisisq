#!/usr/bin/env python3
"""Copy only the five byte-verified frozen study files into Flutter assets."""
from pathlib import Path
import argparse, hashlib, json, shutil, zipfile, sys
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=json.loads((ROOT/'assets/study/device_protocol.json').read_text())
EXPECTED={Path(k).name:v for model in PROTOCOL['models'].values() for k,v in model['all_frozen_file_hashes'].items()}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,help='Unpacked full Kaggle model output');p.add_argument('--zip',dest='archive',type=Path,help='crisisq_flutter_model_assets.zip from the exporter');p.add_argument('--check-only',action='store_true');a=p.parse_args()
 dest=ROOT/'assets/models';dest.mkdir(parents=True,exist_ok=True)
 if not a.check_only:
  if a.source and a.archive:p.error('Choose --source or --zip')
  if not a.source and not a.archive:
   candidate=Path.home()/'Downloads/crisisq_flutter_model_assets.zip'
   if candidate.exists():a.archive=candidate
   else:p.error('Pass --source /path/to/full_output or --zip /path/to/crisisq_flutter_model_assets.zip')
  if a.archive:
   with zipfile.ZipFile(a.archive) as z:
    for name,wanted in EXPECTED.items():
     matches=[n for n in z.namelist() if Path(n).name==name]
     if len(matches)!=1:raise SystemExit('Archive does not contain exactly one '+name)
     tmp=dest/(name+'.partial')
     with z.open(matches[0]) as src,tmp.open('wb') as target:shutil.copyfileobj(src,target,8*1024*1024)
     if sha(tmp)!=wanted:tmp.unlink();raise SystemExit('Wrong frozen bytes: '+name)
     tmp.replace(dest/name)
  else:
   source=a.source.resolve();candidates=[]
   if (source/'q_quint8.onnx').exists():candidates.append(source)
   candidates += [f.parent for f in source.rglob('q_quint8.onnx') if f.parent.name=='saqs__42']
   chosen=next((d for d in candidates if all((d/n).exists() for n in EXPECTED) and all(sha(d/n)==h for n,h in EXPECTED.items())),None)
   if chosen is None:raise SystemExit('No matching SAQS seed-42 files found. Use the original frozen output, not a retrained/re-exported model.')
   for name in EXPECTED:
    tmp=dest/(name+'.partial');shutil.copy2(chosen/name,tmp);tmp.replace(dest/name)
 for name,wanted in EXPECTED.items():
  f=dest/name
  if not f.exists() or sha(f)!=wanted:raise SystemExit('Missing or wrong frozen file: '+str(f))
  print('VERIFIED',name,f.stat().st_size)
 (dest/'asset_manifest.json').write_text(json.dumps({'evidence_plan_hash':PROTOCOL['evidence_plan_hash'],'sha256':EXPECTED,'prepared_by':'prepare_assets_v1'},indent=2))
 print('Ready. Run flutter pub get, then flutter run --release -d YOUR_DEVICE_ID.')
if __name__=='__main__':main()
