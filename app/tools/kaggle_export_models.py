# Paste this entire file into one Kaggle cell. Attach your FULL frozen model notebook output.
from pathlib import Path
import hashlib,json,zipfile,shutil
EXPECTED = {'config.json': '0b3083bf9cd1a27d1d060a02cb8450b2573cf56d34a1a7051f7b071ae4bc0a7a', 'fp32.onnx': '183e352bfba56cfe6f3d9c7b85823835de38d86642747977ceda9950572f754d', 'tokenizer.json': 'da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0', 'tokenizer_config.json': 'e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd', 'q_quint8.onnx': 'b56286e0de34799c321d08baa5e2e2be1fdf1429189206b012d34f637b2e5fe2'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
candidates=[p.parent for p in Path('/kaggle/input').rglob('q_quint8.onnx') if p.parent.name=='saqs__42']
source=next((d for d in candidates if all((d/n).exists() for n in EXPECTED) and all(sha(d/n)==h for n,h in EXPECTED.items())),None)
if source is None:raise FileNotFoundError('Attach the FULL frozen nonclinical output containing models/saqs__42, not just a reports ZIP.')
target=Path('/kaggle/working/crisisq_flutter_model_assets.zip')
size=sum((source/n).stat().st_size for n in EXPECTED)
if shutil.disk_usage(target.parent).free<size+128*1024**2:raise RuntimeError('Not enough space to package the two models; no training is required.')
tmp=target.with_suffix('.partial.zip')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for name in EXPECTED:z.write(source/name,'assets/models/'+name)
 z.writestr('assets/models/asset_manifest.json',json.dumps({'sha256':EXPECTED,'prepared_by':'kaggle_export_models_v1'},indent=2))
with zipfile.ZipFile(tmp) as z:
 if z.testzip() is not None:raise RuntimeError('Archive integrity error')
tmp.replace(target)
print('DOWNLOAD:',target,'MB:',target.stat().st_size/1e6)
print('Original frozen weights verified. Nothing was trained or converted.')
