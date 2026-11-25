#!/usr/bin/env python3
"""Validate full frozen-source device classification and compare four fixed systems."""
import argparse,csv,hashlib,io,json,math,random,statistics,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFEST_SHA='b0f6f391c1277d826658bc11339ac47317d9766f0ae8c4b96be54fae77886f19'
CASES_SHA='a24de8c2b432ddbe3880a579abc9106cde59d38b9f545eeae30f73ad955b1372'
def require(condition,message):
    if not condition:raise SystemExit('INVALID: '+message)
def sha(data):return hashlib.sha256(data).hexdigest()
def match(a,b):return math.isfinite(a) and abs(a-b)<=1e-5+1e-5*abs(b)
def ratios(c):
    tp,fn,fp,tn=c
    return {'recall':tp/(tp+fn) if tp+fn else None,'false_positive_rate':fp/(fp+tn) if fp+tn else None,
            'precision':tp/(tp+fp) if tp+fp else None,'specificity':tn/(tn+fp) if tn+fp else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'accuracy':(tp+tn)/sum(c) if sum(c) else None}
def quantile(x,p):
    x=sorted(x);pos=(len(x)-1)*p;low=math.floor(pos);high=math.ceil(pos)
    return x[low]+(x[high]-x[low])*(pos-low)
def grouped_intervals(part_cases,predictions,systems,B,seed):
    if B<=0:return {'bootstrap_repetitions':0,'intervals':{}}
    groups={}
    for case in part_cases:
        totals=groups.setdefault(case['group_id'],[0]*(4*len(systems)))
        for i,system in enumerate(systems):
            y=case['label'];p=predictions[system,case['source_id']];index=0 if y==1 and p==1 else 1 if y==1 else 2 if p==1 else 3
            totals[4*i+index]+=1
    blocks=list(groups.values());rng=random.Random(seed);samples={}
    for _ in range(B):
        totals=[0]*(4*len(systems))
        for _ in range(len(blocks)):
            block=blocks[rng.randrange(len(blocks))]
            for i,v in enumerate(block):totals[i]+=v
        stats={system:ratios(totals[4*i:4*i+4]) for i,system in enumerate(systems)}
        values={f'{system}.{metric}':value for system,result in stats.items() for metric,value in result.items()}
        for mode in ['neural','hybrid']:
            for metric in ['recall','false_positive_rate']:
                a,b=stats['q_quint8/'+mode][metric],stats['fp32/'+mode][metric]
                values[f'quantization/{mode}.{metric}_quint8_minus_fp32']=None if a is None or b is None else a-b
        for variant in ['fp32','q_quint8']:
            for metric in ['recall','false_positive_rate']:
                a,b=stats[variant+'/hybrid'][metric],stats[variant+'/neural'][metric]
                values[f'routing/{variant}.{metric}_hybrid_minus_neural']=None if a is None or b is None else a-b
        for key,value in values.items():
            if value is not None:samples.setdefault(key,[]).append(value)
    return {'bootstrap_repetitions':B,'group_count':len(blocks),'unit':'resampled original group_id clusters within this frozen source partition',
            'intervals':{k:{'lower_95':quantile(v,.025),'upper_95':quantile(v,.975),'valid_replicates':len(v)} for k,v in samples.items()}}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('export',type=Path);p.add_argument('--out',type=Path,default=Path('full_device_evaluation_analysis'));p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    manifest_bytes=(ROOT/'assets/study/full_evaluation_manifest.json').read_bytes();case_bytes=(ROOT/'assets/study/full_evaluation_cases.jsonl').read_bytes()
    require(sha(manifest_bytes)==MANIFEST_SHA and sha(case_bytes)==CASES_SHA,'local frozen packet differs')
    packet=json.loads(manifest_bytes);cases={c['source_id']:c for c in map(json.loads,case_bytes.decode().splitlines())}
    tau=packet['tau']
    with zipfile.ZipFile(a.export) as z:
        done=json.loads(z.read('completion.json'));run=json.loads(z.read('run_manifest.json'));ctx=run['configuration'];preflight=json.loads(z.read('preflight_report.json'))
        require(done.get('complete') is True and done.get('model_outputs')==3400 and done.get('required_model_outputs')==3400,'full evaluation is incomplete')
        require(json.loads(z.read('packet_manifest.json'))==packet,'export packet differs')
        require(ctx.get('protocol')=='full_labeled_device_evaluation_v1' and ctx.get('cases_sha256')==CASES_SHA and ctx.get('packet_manifest_sha256')==MANIFEST_SHA,'protocol or packet hash differs')
        require(ctx.get('source_external_hash')==packet['source_external_hash'] and ctx.get('source_evaluation_rows_hash')==packet['source_evaluation_rows_hash'],'source evaluation provenance differs')
        require(ctx.get('build_mode')=='release' and ctx.get('device',{}).get('physical') is True,'physical release device required')
        require(ctx.get('tau')==tau and ctx.get('model_hashes')==packet['model_hashes'] and ctx.get('rules_sha256')==packet['rules_sha256'],'models, rules or threshold changed')
        require(ctx.get('models')==['fp32','q_quint8'] and ctx.get('case_count')==1700,'configuration scope differs')
        require(preflight.get('threads')==ctx['threads'] and preflight.get('runtime')==ctx['runtime'] and preflight.get('session_configs',{})==ctx.get('session_configs',{}),'preflight settings differ')
        reference_cases={c['source_id']:c for c in map(json.loads,(ROOT/'assets/study/benchmark_cases.jsonl').read_text().splitlines())}
        checks=[r for r in preflight.get('models',[]) if r['variant']=='fp32']
        require(len(checks)==256 and {r['source_id'] for r in checks}==set(reference_cases),'FP32 preflight incomplete')
        for row in checks:
            ref=reference_cases[row['source_id']]['reference']['fp32']['probability'];actual=row['actual_probability']
            require(row['passed'] is True and match(actual,ref) and (actual>=tau)==(ref>=tau),'FP32 preflight did not pass')
        rows=list(csv.DictReader(io.StringIO(z.read('full_device_predictions.csv').decode())))
        thermal=json.loads(z.read('thermal_snapshots.json'))
    require(len(rows)==3400,'expected exactly 3,400 model outputs')
    seen=set();probabilities={};predictions={};drift=[]
    for row in rows:
        sid,variant=row['source_id'],row['variant'];require(sid in cases and variant in ['fp32','q_quint8'],'unknown source or model')
        require((variant,sid) not in seen,'duplicate model/source output');seen.add((variant,sid));case=cases[sid]
        for name in ['text_hash','group_id','partition']:require(row[name]==case[name],'row identity differs: '+name)
        require(int(row['label'])==case['label'] and int(row['gate_v2'])==case['gate_v2'],'label or gate changed')
        probability=float(row['probability']);reference=case['reference'][variant]['probability'];require(math.isfinite(probability) and 0<=probability<=1,'invalid probability')
        require(abs(float(row['reference_probability'])-reference)<=1e-12,'Kaggle reference changed')
        neural=int(probability>=tau);hybrid=neural if case['gate_v2']<0 else case['gate_v2']
        require(int(row['prediction_neural'])==neural and int(row['prediction_hybrid'])==hybrid,'decision does not follow frozen threshold and gate')
        same=match(probability,reference);same_decision=neural==int(reference>=tau)
        require(row['kaggle_probability_match']==str(same).lower() and row['kaggle_decision_match']==str(same_decision).lower(),'reference difference flags hide or invent drift')
        probabilities[variant,sid]=probability;predictions[variant+'/neural',sid]=neural;predictions[variant+'/hybrid',sid]=hybrid
        drift.append(dict(variant=variant,source_id=sid,partition=case['partition'],source_label=case['label'],kaggle_probability=reference,device_probability=probability,absolute_difference=abs(probability-reference),probability_match=same,decision_match=same_decision))
    require(len(seen)==len(cases)*2,'incomplete model/source coverage')
    a.out.mkdir(parents=True,exist_ok=True)
    def save(name,records):
        with (a.out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    systems=['fp32/neural','fp32/hybrid','q_quint8/neural','q_quint8/hybrid'];metrics=[];paired=[];intervals={}
    for partition in ['binary_test','labeled_hard_negative','positive_probe']:
        subset=[c for c in cases.values() if c['partition']==partition]
        for system in systems:
            c=[0,0,0,0]
            for case in subset:
                y=case['label'];v=predictions[system,case['source_id']];c[0 if y==1 and v==1 else 1 if y==1 else 2 if v==1 else 3]+=1
            variant,mode=system.split('/')
            metrics.append(dict(partition=partition,variant=variant,mode=mode,n=len(subset),n_positive=c[0]+c[1],n_negative=c[2]+c[3],tp=c[0],fn=c[1],fp=c[2],tn=c[3],**ratios(c),stage2_rate=1.0 if mode=='neural' else sum(x['gate_v2']<0 for x in subset)/len(subset)))
        for label,baseline,candidate in [('quantization_neural','fp32/neural','q_quint8/neural'),('quantization_hybrid','fp32/hybrid','q_quint8/hybrid'),('routing_fp32','fp32/neural','fp32/hybrid'),('routing_quint8','q_quint8/neural','q_quint8/hybrid')]:
            changes=[c for c in subset if predictions[baseline,c['source_id']]!=predictions[candidate,c['source_id']]]
            lost=sum(c['label']==1 and predictions[baseline,c['source_id']]==1 and predictions[candidate,c['source_id']]==0 for c in subset)
            gained=sum(c['label']==1 and predictions[baseline,c['source_id']]==0 and predictions[candidate,c['source_id']]==1 for c in subset)
            new_fp=sum(c['label']==0 and predictions[baseline,c['source_id']]==0 and predictions[candidate,c['source_id']]==1 for c in subset)
            removed_fp=sum(c['label']==0 and predictions[baseline,c['source_id']]==1 and predictions[candidate,c['source_id']]==0 for c in subset)
            detected=sum(c['label']==1 and predictions[baseline,c['source_id']]==1 for c in subset)
            paired.append(dict(partition=partition,comparison=label,baseline=baseline,candidate=candidate,n=len(subset),changed_decisions=len(changes),positive_lost=lost,positive_gained=gained,new_negative_alerts=new_fp,removed_negative_alerts=removed_fp,baseline_detected_positive=detected,conditional_positive_loss=lost/detected if detected else None))
        intervals[partition]=grouped_intervals(subset,predictions,systems,a.bootstrap,20261113+len(subset))
    save('full_device_classification_metrics.csv',metrics);save('paired_quantization_and_routing_changes.csv',paired);save('kaggle_to_device_drift.csv',drift)
    (a.out/'group_bootstrap_intervals.json').write_text(json.dumps(intervals,indent=2))
    summary={'complete':True,'run_id':run['run_id'],'model_outputs':3400,'unique_labeled_messages':1700,'derived_system_decisions':6800,'context':ctx,
      'kaggle_drift':{v:{'probability_mismatches':sum(not r['probability_match'] for r in drift if r['variant']==v),'changed_decisions':sum(not r['decision_match'] for r in drift if r['variant']==v),'max_absolute_difference':max(r['absolute_difference'] for r in drift if r['variant']==v)} for v in ['fp32','q_quint8']},
      'thermal_states':sorted({snap.get('thermal_state','unavailable') for chunk in thermal for snap in [chunk['before'],chunk['after']]}),
      'scope':'Post-hoc full frozen-source device evaluation. Same inspected source data and incomplete historical overlap audit; no new independent or clinical validation, no retraining or threshold tuning, no pipeline latency measurements.'}
    (a.out/'validation.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE: 1,700 unique source-labeled messages, 3,400 neural outputs, four derived configurations.')
    for row in metrics:
        if row['partition']=='binary_test':print(row['variant'],row['mode'],'TP',row['tp'],'FN',row['fn'],'FP',row['fp'],'TN',row['tn'],'recall',round(row['recall'],4))
    print('Tables:',a.out.resolve())
if __name__=='__main__':main()
