#!/usr/bin/env python3
"""Check source syntax, frozen metadata, release boundaries and raw export coverage."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import ast,gzip,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def digest(x):return _sha256_bytes(json.dumps(x,sort_keys=True,default=str).encode())
def main():
    hashes={'training_protocol_frozen.json':None,'selection_frozen.json':'e976dbf922c37ce1406418ff9c5903a1ecc6a502aadb14d3d116ab18ef6ba626','external_evaluation_frozen.json':'fe7125e21acb826cd46fa837388b55ff75dfe1b4177a68cdebf133d14ab1dfbf'}
    protocol=json.loads((ROOT/'manifests/training_protocol_frozen.json').read_text());selection=json.loads((ROOT/'manifests/selection_frozen.json').read_text())
    assert selection['protocol']==digest(protocol)
    for name,wanted in hashes.items():
        if wanted:assert digest(json.loads((ROOT/'manifests'/name).read_text()))==wanted,name
    notebooks=list((ROOT/'notebooks').glob('*.ipynb'))
    for path in notebooks:
        nb=json.loads(path.read_text())
        for n,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            assert not c.get('outputs') and c.get('execution_count') is None
            s=''.join(c['source']);s='\n'.join('pass # notebook magic' if line.lstrip().startswith(('%','!')) else line for line in s.splitlines())
            compile(s,f'{path.name}:cell{n}','exec')
    for p in ROOT.rglob('*.py'):
        if not any(part in ['.cache','.venv','.reproduced'] for part in p.parts):ast.parse(p.read_text(),filename=str(p))
    raw=list((_result_root(ROOT)/'device/raw').glob('*.zip'));assert len(raw)==8
    timing_ids=set();full_count=0
    for path in raw:
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            assert not any(n.endswith('.onnx') for n in z.namelist())
            manifest=json.loads(z.read('run_manifest.json'));done=json.loads(z.read('completion.json'));assert done['complete'] is True
            if path.name.startswith('crisisq_device_results_'):timing_ids.add(manifest['run_id']);assert done['completed_tasks']==16
            else:full_count+=1;assert done['model_outputs']==3400
    assert len(timing_ids)==6 and full_count==2
    forbidden=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ['.onnx','.safetensors','.pt','.pth'] or p.name in ['misclassified_texts.csv','lexical_overlap_flags.csv'])]
    allowed = ROOT/'app/assets/models'
    expected = json.loads((ROOT/'manifests/primary_model_files.json').read_text())['sha256']
    for path in forbidden:
        assert path.parent == allowed and path.name in expected, 'Unexpected private/model file: '+str(path)
        assert _sha256_bytes(path.read_bytes()) == expected[path.name], 'Local model hash differs'

    print(f'PASS: {len(notebooks)} clean notebooks; Python syntax; frozen manifests; six timing runs; two full evaluations; optional local primary weights verified; no historical text tables.')
if __name__=='__main__':main()
