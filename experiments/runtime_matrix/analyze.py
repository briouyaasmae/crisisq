"""Analyze the entire declared suite, retaining null and adverse results."""
from pathlib import Path
import csv,hashlib,json,zipfile
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def save_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def compare(a,b,labels,tau):
    a=np.asarray(a);b=np.asarray(b);labels=np.asarray(labels);pa=a>=tau;pb=b>=tau
    return dict(n=len(a),probability_failures=int(np.sum(np.abs(a-b)>1e-5+1e-5*np.abs(a))),max_absolute_probability_difference=float(np.max(np.abs(a-b))),changed_neural_decisions=int(np.sum(pa!=pb)),n_positive=int(np.sum(labels==1)),positive_lost=int(np.sum((labels==1)&pa&~pb)),positive_gained=int(np.sum((labels==1)&~pa&pb)),negative_alerts_added=int(np.sum((labels==0)&~pa&pb)),negative_alerts_removed=int(np.sum((labels==0)&pa&~pb)))
def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);pos=y==1;neg=y==0
    return dict(n=len(y),positive=int(sum(pos)),negative=int(sum(neg)),recall=float(np.mean(p[pos])) if sum(pos) else '',negative_alert_rate=float(np.mean(p[neg])) if sum(neg) else '',alert_rate=float(np.mean(p)),missed_source_positives=int(np.sum(pos&~p)),source_negative_alerts=int(np.sum(neg&p)))
def main(root,bundle):
    root=Path(root);bundle=Path(bundle);frozen=json.loads((root/'protocol_frozen.json').read_text());protocol=frozen['protocol'];assert digest(protocol)==frozen['plan_hash']
    for name,value in protocol['implementation_hashes'].items():assert sha(bundle/name)==value,'Analysis/worker implementation differs from frozen protocol'
    jobs=json.loads((root/'jobs_frozen.json').read_text());assert digest(jobs)==json.loads((root/'execution_manifest.json').read_text())['jobs_hash']
    cases=json.loads((root/'case_metadata.json').read_text());assert digest(cases)==protocol['case_metadata_hash']
    table={};provenance=[];timing=[]
    for j in jobs:
        folder=Path(j['output']);assert (folder/'complete.json').exists(),'Incomplete declared suite: '+j['id']
        done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['plan_hash']==frozen['plan_hash'] and done['job_identity']==digest(j)
        assert sha(folder/'predictions.npz')==done['prediction_sha256']
        with np.load(folder/'predictions.npz',allow_pickle=False) as values:
            ix=values['indices'].copy();p=values['probabilities'].copy();times=values['elapsed_ms'].copy()
        model=next(m for m in protocol['models'] if m['id']==j['model_id'])
        expected=list(range(protocol['primary_n'])) if model['scope']=='primary_full_1700' else protocol['panel_indices']
        assert j['indices']==expected,'Job sample differs from frozen scope'
        assert list(ix)==j['indices'] and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        key=(j['model_id'],j['runtime'],j['optimization'],j['threads']);assert key not in table
        table[key]=(ix,p,j);provenance.append(done)
        timing.append(dict(model=j['model_id'],runtime=j['runtime'],optimization=j['optimization'],threads=j['threads'],n=len(p),session_init_ms=done['session_init_ms'],inference_mean_ms=float(np.mean(times)),inference_median_ms=float(np.median(times)),inference_p95_ms=float(np.quantile(times,.95)),repeat_cases=done['repeat_cases'],repeat_max_probability_difference=done['repeat_max_probability_difference'],repeat_decision_changes=done['repeat_decision_changes']))
    assert len(table)==len(protocol['models'])*len(protocol['settings']),'Incomplete or unexpected matrix'
    output=root/'tables';output.mkdir(exist_ok=True)
    model_map={m['id']:m for m in protocol['models']};anchor=protocol['reference_setting'];skey=lambda s:(s['runtime'],s['optimization'],s['threads'])
    anchor_comparisons=[];panel_comparisons=[];classification=[];stability=[];envelopes=[];interventions=[];compression=[];endpoint=[];panel_rows=[]
    historical=json.loads((bundle/'historical_android_primary.json').read_text());hist={r['source_id']:r for r in historical['rows']}
    for model in protocol['models']:
        model_id=model['id'];tau=model['tau'];base_ix,base_p,_=table[(model_id,*skey(anchor))]
        labels=np.array([cases[i]['label'] for i in base_ix]);gates=np.array([cases[i]['gate_v2'] for i in base_ix]);parts=np.array([cases[i]['partition'] for i in base_ix])
        suite=[]
        for setting in protocol['settings']:
            ix,p,j=table[(model_id,*skey(setting))];assert np.array_equal(ix,base_ix)
            suite.append(p)
            panel_mask=np.isin(base_ix,protocol['panel_indices']);assert int(sum(panel_mask))==len(protocol['panel_indices'])
            panel_comparisons.append(dict(model=model_id,**setting,**compare(base_p[panel_mask],p[panel_mask],labels[panel_mask],tau)))
            for partition in sorted(set(parts)):
                mask=parts==partition;info=dict(model=model_id,scope=model['scope'],partition=partition,**setting)
                anchor_comparisons.append({**info,**compare(base_p[mask],p[mask],labels[mask],tau)})
                neural=p>=tau;hybrid=np.where(gates==-1,neural,gates==1)
                for mode,pred in [('neural',neural),('hybrid',hybrid)]:classification.append({**info,'mode':mode,**metrics(labels[mask],pred[mask])})
            if model['scope']=='primary_full_1700':
                old=np.array([cases[i]['reference'][model['variant']]['probability'] for i in ix]);android=np.array([hist[cases[i]['source_id']][model['variant']] for i in ix])
                for name,reference in [('historical_kaggle',old),('historical_android',android)]:endpoint.append(dict(model=model_id,reference=name,**setting,**compare(reference,p,labels,tau)))
        suite=np.stack(suite);low=suite.min(axis=0);high=suite.max(axis=0);unstable=(low<tau)&(high>=tau);width=high-low
        delta=float(np.max(np.abs(suite-base_p[None,:])));margin=np.abs(base_p-tau);outside=margin>delta
        assert not np.any(outside&unstable),'Observed finite-suite bound violated'
        for partition in sorted(set(parts)):
            mask=parts==partition;stability.append(dict(model=model_id,scope=model['scope'],partition=partition,n=int(sum(mask)),settings=len(suite),neural_unstable_messages=int(sum(unstable&mask)),hybrid_unstable_messages=int(sum(unstable&mask&(gates==-1))),max_probability_range=float(np.max(width[mask])),mean_probability_range=float(np.mean(width[mask])),global_observed_max_delta=delta,messages_outside_global_observed_margin_band=int(sum(outside&mask))))
        envdir=root/'envelopes';envdir.mkdir(exist_ok=True);np.savez_compressed(envdir/(model_id+'.npz'),indices=base_ix,minimum=low,maximum=high,anchor=base_p,neural_unstable=unstable)
        panel_set=set(protocol['panel_indices'])
        for k,i in enumerate(base_ix):
            if int(i) in panel_set:panel_rows.append(dict(model=model_id,arm=model['arm'],seed=model['seed'],variant=model['variant'],source_id=cases[i]['source_id'],partition=cases[i]['partition'],neural_unstable=bool(unstable[k]),hybrid_unstable=bool(unstable[k] and gates[k]==-1),probability_range=float(width[k])))
        # Matched one-factor contrasts: version/wheel, optimizer level, thread count.
        for factor in ['runtime','optimization','threads']:
            for setting in protocol['settings']:
                if setting[factor]==anchor[factor]:continue
                reference={**setting,factor:anchor[factor]}
                # For each non-anchor factor level, hold all other settings fixed.
                a=table[(model_id,*skey(reference))][1];b=table[(model_id,*skey(setting))][1]
                interventions.append(dict(model=model_id,factor=factor,reference_value=reference[factor],candidate_value=setting[factor],held_runtime=setting['runtime'] if factor!='runtime' else 'matched host, different runtime wheels',held_optimization=setting['optimization'] if factor!='optimization' else 'varied',held_threads=setting['threads'] if factor!='threads' else 'varied',**compare(a,b,labels,tau)))
    # Same-setting compression effects; only intersect identical message IDs.
    for model in protocol['models']:
        if model['variant']=='fp32':continue
        fp=model['arm']+'__'+str(model['seed'])+'__fp32'
        if fp not in model_map:continue
        assert model_map[fp]['tau']==model['tau']
        for setting in protocol['settings']:
            ai,a,_=table[(fp,*skey(setting))];bi,b,_=table[(model['id'],*skey(setting))];lookup={int(i):p for i,p in zip(ai,a)}
            assert all(int(i) in lookup for i in bi);a=np.array([lookup[int(i)] for i in bi]);labels=np.array([cases[i]['label'] for i in bi]);parts=np.array([cases[i]['partition'] for i in bi]);g=np.array([cases[i]['gate_v2'] for i in bi])
            for mode in ['neural','hybrid']:
                aa=a.copy();bb=b.copy()
                if mode=='hybrid':aa=np.where(g==-1,aa,g==1);bb=np.where(g==-1,bb,g==1)
                for partition in sorted(set(parts)):
                    mask=parts==partition;compression.append(dict(model=model['id'],mode=mode,partition=partition,**setting,**compare(aa[mask],bb[mask],labels[mask],model['tau'])))
    for name,rows in [('reference_setting_comparisons.csv',anchor_comparisons),('common_panel_setting_comparisons.csv',panel_comparisons),('source_label_metrics.csv',classification),('finite_suite_stability.csv',stability),('matched_setting_contrasts.csv',interventions),('same_setting_compression_changes.csv',compression),('historical_endpoint_comparisons.csv',endpoint),('common_256_panel.csv',panel_rows),('host_execution_summary.csv',timing)]:save_csv(output/name,rows)
    (root/'job_provenance.json').write_text(json.dumps(provenance,indent=2))
    complete=dict(all_jobs_complete=True,plan_hash=frozen['plan_hash'],jobs=len(jobs),models=len(protocol['models']),settings=len(protocol['settings']),available_joint_pairs=sum(c['included'] for c in protocol['coverage']),primary_messages=protocol['primary_n'],multiseed_panel_messages=len(protocol['panel_indices']),new_training_runs=0,new_threshold_searches=0,new_independent_datasets=0,host_machine=protocol['system']['machine'],repeatability_failures=sum(r['repeat_decision_changes']>0 or r['repeat_max_probability_difference']>1e-5 for r in provenance))
    (root/'completion.json').write_text(json.dumps(complete,indent=2))
    lines=['CrisisQ controlled runtime matrix — completed fixed suite',json.dumps(complete,indent=2),'','Interpretation rules:',
    'All inputs are the previously inspected frozen Twitter messages. This is not a new independent dataset.',
    'Version contrasts hold observed host, NumPy, inputs, graph bytes, threshold, optimizer and threads fixed; runtime wheel builds still differ.',
    'An optimizer contrast identifies a tested setting effect, not a uniquely responsible operator.',
    'Historical Android endpoints are not a new controlled phone-runtime matrix. CPU/OS/build differences remain.',
    'The empirical min/max envelope covers only completed settings on this host. It is not certification for unseen devices or inputs.',
    'Stress outputs are alert rates on keyword-enriched source-negative messages; source labels are unaudited.',
    'Host one-pass timings are descriptive, not comparable to the published Flutter pipeline timing protocol.',
    'No result is selected to replace the frozen primary model. Null and adverse results remain reported.']
    if not complete['available_joint_pairs']:lines+=['','Only the primary pair is available: do not claim multiseed replication from this run.']
    for model in protocol['models']:
        if model['scope']!='primary_full_1700':continue
        found=[r for r in endpoint if r['model']==model['id'] and r['reference']=='historical_android']
        lines+=['',model['id']+': observed decision differences against historical Android range '+str(min(r['changed_neural_decisions'] for r in found))+'–'+str(max(r['changed_neural_decisions'] for r in found))+' across the declared host settings. This is descriptive, not a verified phone fix.']
    (root/'READ_ME_results.txt').write_text('\n'.join(lines)+'\n')
    create_handoff(root,bundle,protocol,cases,table)
    try:plots(root,protocol,stability,panel_comparisons)
    except ImportError:(root/'PLOTS_NOT_GENERATED.txt').write_text('Install matplotlib to regenerate optional plots. Numeric tables are complete.')
    archive=root.parent/(root.name+'_reports.zip');pack(root,archive)
    print('REPORTS:',archive);print('\n'.join(lines[-5:]));return complete

def create_handoff(root,bundle,protocol,cases,table):
    folder=root/'device_replay_handoff';folder.mkdir(exist_ok=True)
    primary=[m for m in protocol['models'] if m['scope']=='primary_full_1700'];anchor=protocol['reference_setting']
    # Tokens are stored separately: do not reconstruct text or alter published app assets.
    jobs=json.loads((root/'jobs_frozen.json').read_text());inputs=Path(jobs[0]['inputs_path'])
    import shutil
    shutil.copy2(inputs,folder/'runtime_inputs.npz')
    references=[]
    for m in primary:
        ix,p,_=table[(m['id'],anchor['runtime'],anchor['optimization'],anchor['threads'])]
        for i,value in zip(ix,p):references.append(dict(model=m['id'],source_id=cases[i]['source_id'],probability=float(value),neural_decision=int(value>=m['tau'])))
    save_csv(folder/'matched_host_reference.csv',references)
    (folder/'manifest.json').write_text(json.dumps(dict(plan_hash=json.loads((root/'protocol_frozen.json').read_text())['plan_hash'],models=primary,case_order=[r['source_id'] for r in cases],inputs_sha256=sha(folder/'runtime_inputs.npz'),scope='Handoff only, not a new phone result; keep model bytes and threshold unchanged. Graph files remain in the existing attached model output.'),indent=2))
    (folder/'READ_ME.txt').write_text('''Device replay handoff: ordered int64 inputs and matched-host reference outputs.
The models remain the exact existing frozen graphs; they are not duplicated here.
The published Flutter app has strict frozen manifests. Do not overwrite those
assets with this diagnostic matrix or treat its old reference checks as new
factorial evidence. A separate native/app harness must implement the declared
session options before recording new matched physical-device results. This
packet is preparation for that follow-up, not an executable phone application
or proof that a host setting fixes Android/iOS. Tokenized research data retain
original source terms; lack of raw text is not a claim of anonymization.
''')

def plots(root,protocol,stability,comparisons):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings=protocol['settings'];models=[m['id'] for m in protocol['models']]
    matrix=np.zeros((len(models),len(settings)))
    for i,m in enumerate(models):
        for j,s in enumerate(settings):matrix[i,j]=sum(r['changed_neural_decisions'] for r in comparisons if r['model']==m and all(r[k]==s[k] for k in ['runtime','optimization','threads']))
    fig,ax=plt.subplots(figsize=(12,max(3.5,.3*len(models))))
    im=ax.imshow(matrix,aspect='auto',cmap='magma',vmin=0,vmax=max(1,float(np.max(matrix))));ax.set_yticks(range(len(models)),models,fontsize=8)
    ax.set_xticks(range(len(settings)),[s['runtime']+'\n'+s['optimization'].replace('ORT_','')+'\nt='+str(s['threads']) for s in settings],fontsize=7,rotation=45,ha='right')
    ax.set_title('Neural decision differences vs each artifact’s declared reference setting\nCommon deterministic panel: n='+str(len(protocol['panel_indices']))+' for every artifact')
    fig.colorbar(im,ax=ax,label='Changed neural decisions on the common panel');fig.tight_layout();fig.savefig(root/'runtime_decision_matrix.png',dpi=220);plt.close(fig)
    primary=[r for r in stability if r['scope']=='primary_full_1700'];fig,ax=plt.subplots(figsize=(8,4))
    names=[r['model'].split('__')[-1]+'\n'+r['partition'] for r in primary];values=[100*r['neural_unstable_messages']/r['n'] for r in primary]
    ax.bar(range(len(values)),values,color='#168275');ax.set_xticks(range(len(values)),names,rotation=25,ha='right',fontsize=8);ax.set_ylabel('Messages whose neural decision changes (%)');ax.set_ylim(0,max(1,max(values,default=0)*1.2));ax.set_title('Instability across this finite host suite; not an unseen-device guarantee');fig.tight_layout();fig.savefig(root/'finite_suite_primary_instability.png',dpi=220);plt.close(fig)

def pack(root,archive):
    root=Path(root)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '.plot_cache' in p.parts or p.suffix in ['.onnx','.whl'] or p.name=='full_evaluation_cases.jsonl' or p.name.startswith('install_ort_'):continue
            z.write(p,p.relative_to(root))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
