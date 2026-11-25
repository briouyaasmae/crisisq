"""Rebuild exactly the frozen 1,700 labeled rows. No models are trained or selected."""
import argparse,base64,csv,gzip,hashlib,html,io,json,re,unicodedata,urllib.request,zipfile
from pathlib import Path

def sha(data):return hashlib.sha256(data).hexdigest()
def digest(obj):return sha(json.dumps(obj,sort_keys=True,default=str).encode())
def normal(text):
    s=unicodedata.normalize('NFKC',html.unescape(str(text))).casefold().replace('\u2019',"'")
    return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',s)).strip()
def main(seed,output,source=None):
    from tokenizers import Tokenizer
    import tokenizers
    if tokenizers.__version__!='0.23.3':raise RuntimeError('Use tokenizers==0.23.3 for this packet rebuild')
    if digest(seed['full_index'])!=seed['evaluation_rows_hash']:raise RuntimeError('Frozen membership digest differs')
    tokenizer_bytes=base64.b64decode(seed['tokenizer_base64'])
    if sha(tokenizer_bytes)!=seed['model_hashes']['tokenizer.json']:raise RuntimeError('Tokenizer bytes differ')
    if source is None:
        print('Downloading the exact pinned public Twitter CSV, not a changing dataset branch.',flush=True)
        data=urllib.request.urlopen(urllib.request.Request(seed['source']['url'],headers={'User-Agent':'CrisisQ-frozen-packet-builder'}),timeout=90).read()
    else:data=Path(source).read_bytes()
    if sha(data)!=seed['source']['sha256']:raise RuntimeError('Public source bytes differ from frozen source')
    public=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    tokenizer=Tokenizer.from_str(tokenizer_bytes.decode())
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128,pad_id=0,pad_token='[PAD]')
    rows=[]
    for index in seed['labeled_rows']:
        if not index['source_id'].startswith('twitter:'):raise RuntimeError('Unexpected labeled source')
        source_row=public[int(index['source_id'].split(':')[1])];text=source_row['tweet']
        if int(source_row['intention'])!=index['label'] or sha(normal(text).encode())!=index['text_hash']:raise RuntimeError('Source text or label differs: '+index['source_id'])
        encoding=tokenizer.encode(text)
        row={**index,'text':text,'input_ids':encoding.ids,'attention_mask':encoding.attention_mask}
        if len(row['input_ids'])!=128 or len(row['attention_mask'])!=128:raise RuntimeError('Token dimensions differ')
        prior=seed['token_checks'].get(index['source_id'])
        if prior and (row['input_ids']!=prior['input_ids'] or row['attention_mask']!=prior['attention_mask']):raise RuntimeError('Saved-token reference mismatch')
        rows.append(row)
    counts={p:sum(r['partition']==p for r in rows) for p in seed['counts']}
    if counts!=seed['counts'] or len(rows)!=1700:raise RuntimeError('Incomplete frozen labeled dataset')
    cases_bytes=('\n'.join(json.dumps(r,ensure_ascii=False,separators=(',',':')) for r in rows)+'\n').encode()
    manifest={'version':'full_labeled_device_evaluation_v1','source_external_hash':seed['external_plan_hash'],'source_evaluation_rows_hash':seed['evaluation_rows_hash'],'labeled_membership_hash':digest([{k:r[k] for k in ['source_id','text_hash','label','group_id','partition','role']} for r in rows]),'source_records':{'twitter':seed['source']},'counts':counts,'positive_count':sum(r['label']==1 for r in rows),'negative_count':sum(r['label']==0 for r in rows),'case_count':len(rows),'max_length':128,'tau':seed['tau'],'model_hashes':seed['model_hashes'],'rules_sha256':seed['rules_sha256'],'cases_sha256':sha(cases_bytes),'tokenizer_builder':'tokenizers 0.23.3','scope':'post-hoc full frozen-source device classification; no new independent dataset, model selection, threshold fitting, latency or clinical claims','historical_overlap':'Historical training overlap remains incompletely auditable.','inference':'one neural pass per case per model; neural and hybrid decisions derived from the same frozen probability and gate','unlabeled_support':'excluded; this packet includes source labels 0/1 only'}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'full_evaluation_cases.jsonl').write_bytes(cases_bytes)
    (output/'full_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    target=output.parent/'crisisq_full_labeled_device_packet.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['full_evaluation_manifest.json','full_evaluation_cases.jsonl']:z.write(output/name,'assets/study/'+name)
    print('READY:',len(rows),'labeled messages;',sum(r['label']==1 for r in rows),'positive;',sum(r['label']==0 for r in rows),'negative')
    print('DOWNLOAD:',target)
    print('Cases SHA256:',manifest['cases_sha256'])
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=Path,default=Path(__file__).with_name('full_evaluation_seed.json.gz'));p.add_argument('--output',type=Path,default=Path('/kaggle/working/full_device_packet'));p.add_argument('--source',type=Path);a=p.parse_args()
    main(json.loads(gzip.decompress(a.seed.read_bytes())),a.output,a.source)
