#!/usr/bin/env python3
"""Post-hoc descriptive keyword split, using frozen flags and text-free device exports."""
import sys as _sys
from pathlib import Path as _ReviewPath
_sys.path.insert(0, str(_ReviewPath(__file__).resolve().parent))
from review_store import result_root as _result_root, sha256_bytes as _sha256_bytes
from pathlib import Path
import csv,io,json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    index_path=_result_root(ROOT)/'external/evaluation_row_index.csv'
    index={r['source_id']:r for r in csv.DictReader(index_path.open()) if r['partition']=='labeled_hard_negative'}
    assert len(index)==300 and all(r['label']=='0' for r in index.values())
    assert all(r['explicit_keyword_present'] in ['True','False'] for r in index.values())
    records=[];signatures=[];source_files=[]
    for file in sorted((_result_root(ROOT)/'device/raw').glob('crisisq_full_device_evaluation_*.zip')):
        with zipfile.ZipFile(file) as z:
            completion=json.loads(z.read('completion.json'));assert completion['complete']
            rows=list(csv.DictReader(io.StringIO(z.read('full_device_predictions.csv').decode())))
        signature=[]
        for variant in ['fp32','q_quint8']:
            subset=[r for r in rows if r['variant']==variant and r['partition']=='labeled_hard_negative']
            assert len(subset)==300 and {r['source_id'] for r in subset}==set(index)
            for row in subset:
                assert row['text_hash']==index[row['source_id']]['text_hash']
                assert row['group_id']==index[row['source_id']]['group_id'] and row['label']=='0'
            for mode in ['neural','hybrid']:
                total=sum(int(r['prediction_'+mode]) for r in subset)
                for flag in ['True','False']:
                    group=[r for r in subset if index[r['source_id']]['explicit_keyword_present']==flag]
                    alerts=sum(int(r['prediction_'+mode]) for r in group)
                    record=dict(export=file.name,variant=variant,mode=mode,explicit_keyword_present=flag,n=len(group),alerts=alerts,alert_rate=alerts/len(group),share_of_configuration_alerts=alerts/total)
                    records.append(record);signature.append({k:v for k,v in record.items() if k!='export'})
        signatures.append(signature);source_files.append(dict(file=file.name,sha256=_sha256_bytes(file.read_bytes())))
    assert len(signatures)==2 and signatures[0]==signatures[1], 'Device keyword summaries differ'
    target=_result_root(ROOT)/'posthoc';target.mkdir(exist_ok=True)
    with (target/'stress_keyword_alerts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (target/'stress_keyword_provenance.json').write_text(json.dumps(dict(analysis='post-hoc descriptive split; no annotation, inference or model selection',index_sha256=_sha256_bytes(index_path.read_bytes()),exports=source_files,devices_agree=True,unit='unique message; devices not pooled',flag='original frozen explicit_keyword_present, not the broader enrichment pattern'),indent=2)+'\n')
    (target/'READ_ME.txt').write_text('''Alert rate on keyword-enriched source-negative messages, stratified by the
original frozen explicit_keyword_present flag. The split is post-hoc and
purely descriptive. It uses unchanged labels, membership and device decisions.
No text was relabeled, no annotator recruited, no model run or threshold tuned.
Both device exports give identical results; do not pool them as 600 messages.

QUInt8 hybrid: keyword present 130/141 (92.20%); absent 79/159 (49.69%).
Keyword-present messages account for 130/209 (62.20%) alerts and 141/300 (47%)
messages. Neural QUInt8: 105/141 and 79/159. Thus hybrid routing adds 25 net
alerts in the keyword-present group and zero net alerts in the absent group.
This is association and decision accounting, not proof of causal keyword
triggering, source-label error, or correctness of the resulting text flags.
The source labels were not independently audited. No estimate of alert rate
on naturally occurring traffic is justified by this enriched sample.
Original historical files and frozen metric field names remain unchanged.
''')
    print('Verified keyword split: 300 unique messages; identical on both devices; wrote',target)
if __name__=='__main__':main()
