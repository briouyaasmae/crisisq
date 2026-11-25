#!/usr/bin/env python3
"""Reconstruct hash-exact public research assets; never download model weights."""
from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,io,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def fetch(record,path):
    if path.exists():data=path.read_bytes()
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        print('Downloading pinned public source:',record['repo'],flush=True)
        data=urllib.request.urlopen(urllib.request.Request(record['url'],headers={'User-Agent':'CrisisQ-reproducibility'}),timeout=120).read()
        if sha(data)!=record['sha256']:raise SystemExit('Source checksum mismatch')
        path.write_bytes(data)
    if sha(data)!=record['sha256']:raise SystemExit('Cached source checksum mismatch: '+str(path))
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-cache',type=Path,default=ROOT/'.cache/public_sources');a=p.parse_args()
    plan=json.loads((ROOT/'manifests/external_evaluation_frozen.json').read_text())
    twitter=fetch(plan['source_records']['twitter'],a.source_cache/'twitter.csv')
    esconv=fetch(plan['source_records']['esconv'],a.source_cache/'esconv.json')
    tweets=list(csv.DictReader(io.StringIO(twitter.decode('utf-8-sig'))));dialogs=json.loads(esconv)
    template=json.loads((ROOT/'assets/textless/benchmark_line_templates.json').read_text());lines=[]
    for item in template['lines']:
        source,*parts=item['source_id'].split(':')
        if source=='twitter':text=tweets[int(parts[0])]['tweet']
        elif source=='esconv':text=dialogs[int(parts[0])]['dialog'][int(parts[1])]['content']
        else:raise SystemExit('Unexpected frozen source ID')
        lines.append(item['prefix']+json.dumps(text,ensure_ascii=item['ensure_ascii'])+item['suffix'])
    content=('\n'.join(lines)+('\n' if template['trailing_newline'] else '')).encode()
    if sha(content)!=template['expected_sha256']:raise SystemExit('Reconstructed benchmark bytes differ')
    dest=ROOT/'app/assets/study';dest.mkdir(parents=True,exist_ok=True);(dest/'benchmark_cases.jsonl').write_bytes(content)
    spec=importlib.util.spec_from_file_location('packet_builder',ROOT/'scripts/build_full_device_packet.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    seed=json.loads(gzip.decompress((ROOT/'scripts/full_evaluation_seed.json.gz').read_bytes()))
    builder.main(seed,dest,a.source_cache/'twitter.csv')
    print('READY: exact 256-case timing packet and 1700-case full labeled packet. No model was trained or downloaded.')
if __name__=='__main__':main()
