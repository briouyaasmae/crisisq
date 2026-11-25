"""Frozen runtime-matrix orchestration; uses mounted models, not zip extraction."""
from pathlib import Path
import csv,gzip,hashlib,importlib.util,io,json,os,platform,random,shutil,subprocess,sys,time,urllib.request,venv,zipfile

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def atomic(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def prepare(config,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);bundle=Path(__file__).parent
    assert config['model_scope'] in ['primary_only','all_available_joint_pairs'],'Unknown model scope'
    assert len(set(config['runtime_versions']))==len(config['runtime_versions'])
    assert '1.31.1' in config['runtime_versions'] and 'ORT_ENABLE_ALL' in config['optimization_levels'] and 4 in config['threads'],'Declared anchor 1.31.1 / ALL / 4 must be present'
    assert all(type(t) is int and t>0 for t in config['threads'])
    seed=json.loads(gzip.decompress((bundle/'full_evaluation_seed.json.gz').read_bytes()));external=json.loads((bundle/'external_frozen.json').read_text());selection=json.loads((bundle/'selection_frozen.json').read_text())
    assert digest(external)==seed['external_plan_hash'],'Bundled source plan differs'
    assert digest(selection)==external['source_selection_hash']
    # Reuse only a hash-exact packet if it is mounted; otherwise rebuild pinned source.
    matches=sorted(Path(config['input_root']).rglob('full_evaluation_cases.jsonl'))
    expected='a24de8c2b432ddbe3880a579abc9106cde59d38b9f545eeae30f73ad955b1372'
    found=next((p for p in matches if sha(p)==expected),None)
    temp=Path(config['temporary_root']);temp.mkdir(parents=True,exist_ok=True)
    if found is None:
        import tokenizers
        assert tokenizers.__version__=='0.23.3','Use tokenizers 0.23.3 for exact preparation'
        builder=load_module(bundle/'build_full_device_packet.py','frozen_packet_builder');builder.main(seed,temp/'packet');found=temp/'packet/full_evaluation_cases.jsonl'
    assert sha(found)==expected
    cases=[json.loads(line) for line in found.read_text().splitlines()];assert len(cases)==1700
    import numpy as np
    inputs=temp/'runtime_inputs.npz'
    ids=np.array([r['input_ids'] for r in cases],dtype=np.int64);masks=np.array([r['attention_mask'] for r in cases],dtype=np.int64)
    if inputs.exists():
        with np.load(inputs,allow_pickle=False) as cached:assert np.array_equal(cached['input_ids'],ids) and np.array_equal(cached['attention_mask'],masks),'Cached inputs differ; use a new temporary folder'
    else:np.savez_compressed(inputs,input_ids=ids,attention_mask=masks)
    metadata=[{k:r[k] for k in ['source_id','text_hash','group_id','partition','label','gate_v2','reference']} for r in cases]
    atomic(root/'case_metadata.json',metadata)
    # The fixed 256-message diagnostic panel uses labels/partition only, not scores.
    sizes={('binary_test',1):64,('binary_test',0):64,('positive_probe',1):64,('positive_probe',0):32,('labeled_hard_negative',0):32}
    panel=[]
    for (partition,label),count in sizes.items():
        pool=[i for i,r in enumerate(cases) if r['partition']==partition and r['label']==label]
        pool.sort(key=lambda i:hashlib.sha256(('CrisisQ-runtime-panel-v1|'+cases[i]['source_id']).encode()).hexdigest());assert len(pool)>=count;panel.extend(pool[:count])
    panel=sorted(panel);assert len(set(panel))==256
    expected_models={}
    for job in external['jobs'].values():
        for name,value in job['files'].items():
            if name.endswith('.onnx'):expected_models[value]=Path(name).name
    discovered={};seen=[]
    paths=sorted(p for p in Path(config['input_root']).rglob('*.onnx') if p.name in set(expected_models.values()))
    print('Checking',len(paths),'mounted ONNX files. Hashing does not run a model.',flush=True)
    for i,p in enumerate(paths):
        value=sha(p)
        if value in expected_models:discovered.setdefault(value,p)
        seen.append(dict(path=str(p),sha256=value,recognized=value in expected_models))
        if (i+1)%10==0:print('Hashed',i+1,'files',flush=True)
    def model_info(arm,seed_number,variant,tau,scope):
        job=external['jobs'].get(f'{arm}__{seed_number}__{variant}')
        if job is None:return None
        values=[v for k,v in job['files'].items() if k.endswith('.onnx')];assert len(values)==1
        value=values[0]
        if value not in discovered:return None
        return dict(id=f'{arm}__{seed_number}__{variant}',arm=arm,seed=seed_number,variant=variant,tau=tau,sha256=value,path=str(discovered[value]),scope=scope)
    tau=external['primary']['tau'];models={};coverage=[]
    for variant in ['fp32','q_quint8']:
        m=model_info('saqs',42,variant,tau,'primary_full_1700')
        if m is None:raise RuntimeError('Missing frozen primary '+variant+'. Attach the latest WHOLE notebook output containing the models under /kaggle/input; report-only output is insufficient.')
        models[m['id']]=m
    original_seeds=[42,123,777,1,2,3,4,5,6,7]
    for seed_number in original_seeds:
        a=selection['models'].get(f'ce__{seed_number}',{}).get('selected');b=selection['models'].get(f'saqs__{seed_number}',{}).get('selected')
        if not a or not b:coverage.append(dict(seed=seed_number,included=False,reason='not jointly development-accepted'));continue
        wanted=[]
        for arm,s in [('ce',a),('saqs',b)]:
            for variant in ['fp32',s['variant']]:wanted.append(model_info(arm,seed_number,variant,s['tau'],'joint_pair_panel_256'))
        if config['model_scope']=='all_available_joint_pairs' and all(m is not None for m in wanted):
            for m in wanted:models.setdefault(m['id'],m)
            coverage.append(dict(seed=seed_number,included=True,reason='all four frozen artifacts available'))
        else:coverage.append(dict(seed=seed_number,included=False,reason='primary-only scope or missing one or more frozen artifacts'))
    settings=[dict(runtime=v,optimization=o,threads=t) for v in config['runtime_versions'] for o in config['optimization_levels'] for t in config['threads']]
    jobs=[]
    token_hash=digest([dict(source_id=r['source_id'],input_ids=r['input_ids'],attention_mask=r['attention_mask']) for r in cases])
    implementation={n:sha(bundle/n) for n in ['study.py','worker.py','analyze.py','build_full_device_packet.py','environment.py']}
    protocol=dict(implementation_hashes=implementation,version='crisisq_runtime_matrix_v1_1',study_kind='post-hoc controlled numerical audit on inspected frozen inputs; not independent external evaluation',configuration=config,models=list(models.values()),coverage=coverage,settings=settings,case_metadata_hash=digest(metadata),token_hash=token_hash,packet_sha256=expected,panel_indices=panel,primary_n=1700,panel_n=256,probability_tolerance=dict(atol=1e-5,rtol=1e-5),reference_setting=dict(runtime='1.31.1',optimization='ORT_ENABLE_ALL',threads=4),selection_hash=digest(selection),source_plan_hash=digest(external),system=dict(platform=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),python=sys.version))
    # Save hardware features without credentials or arbitrary environment variables.
    cpu=Path('/proc/cpuinfo')
    if cpu.exists():protocol['system']['cpu_summary']=[l for l in cpu.read_text().splitlines() if l.startswith(('model name','flags','Hardware','CPU architecture'))][:4]
    plan_hash=digest(protocol);plan_path=root/'protocol_frozen.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())['plan_hash']==plan_hash,'Plan differs from previous run. Restore its configuration or choose a NEW output folder; never overwrite an inspected study.'
    else:atomic(plan_path,dict(plan_hash=plan_hash,protocol=protocol))
    for model in models.values():
        for setting in settings:
            key=model['id']+'__'+setting['runtime']+'__'+setting['optimization']+'__t'+str(setting['threads'])
            jobs.append(dict(id=key,model_id=model['id'],model_path=model['path'],model_sha256=model['sha256'],tau=model['tau'],indices=list(range(1700)) if model['scope']=='primary_full_1700' else panel,inputs_path=str(inputs),inputs_sha256=sha(inputs),plan_hash=plan_hash,output=str(root/'jobs'/key),require_isolated=True,worker_sha256=implementation['worker.py'],**setting))
    random.Random(20261114).shuffle(jobs)
    jobs.sort(key=lambda j:0 if models[j['model_id']]['scope']=='primary_full_1700' else 1)
    execution=dict(plan_hash=plan_hash,jobs_hash=digest(jobs))
    if (root/'execution_manifest.json').exists():assert json.loads((root/'execution_manifest.json').read_text())==execution,'Execution schedule changed'
    else:atomic(root/'execution_manifest.json',execution)
    atomic(root/'jobs_frozen.json',jobs);atomic(root/'discovery.json',dict(files=seen,coverage=coverage))
    print('FROZEN before inference:',plan_hash,flush=True)
    print(len(models),'artifacts;',len(settings),'settings;',len(jobs),'jobs; jointly available pairs:',sum(x['included'] for x in coverage),flush=True)
    return protocol,jobs

def install_environments(config,root):
    temp=Path(config['temporary_root']);result={};records=[]
    envtools=load_module(Path(__file__).with_name('environment.py'),'crisisq_isolated_environments')
    for version in config['runtime_versions']:
        folder=temp/('ort_'+version);python=envtools.ensure_environment(folder);log=Path(root)/('install_ort_'+version+'.log')
        installed=False
        probe=subprocess.run([str(python),'-I','-c','import onnxruntime as o,numpy as n; print(o.__version__+"|"+n.__version__)'],capture_output=True,text=True)
        if probe.returncode==0 and probe.stdout.strip()==version+'|2.1.3':installed=True
        if not installed:
            print('Installing isolated CPU runtime',version,'(kernel packages are unchanged).',flush=True)
            with log.open('w') as f:
                process=subprocess.run(envtools.pip_command(python,'install','--only-binary=:all:','--disable-pip-version-check',f'onnxruntime=={version}','numpy==2.1.3'),stdout=f,stderr=subprocess.STDOUT)
            if process.returncode:raise RuntimeError('Cannot install pinned runtime '+version+'. Read '+str(log)+'. No runtime substitution or partial-suite success is allowed. Use a Kaggle Python version with matching wheels.')
        result[version]=python
        info=subprocess.check_output([str(python),'-I','-c','import json,sys,onnxruntime as o,numpy as n;print(json.dumps(dict(runtime=o.__version__,numpy=n.__version__,module=o.__file__,build=o.get_build_info(),python=sys.version)))'],text=True)
        records.append(json.loads(info));assert records[-1]['runtime']==version and records[-1]['numpy']=='2.1.3'
        native=Path(records[-1]['module']).parent
        records[-1]['native_library_hashes']={str(p.relative_to(native)):sha(p) for p in sorted(native.rglob('*.so'))}
        freeze=subprocess.check_output(envtools.pip_command(python,'freeze','--all'),text=True);(Path(root)/('packages_ort_'+version+'.txt')).write_text(freeze)
    environment_file=Path(root)/'runtime_environments.json'
    if environment_file.exists():assert json.loads(environment_file.read_text())==records,'Runtime environment changed; do not mix cached outputs from different wheels'
    else:atomic(environment_file,records)
    return result

def execute(config,root,jobs,environments):
    root=Path(root);worker=Path(__file__).with_name('worker.py');failures=[]
    for number,job in enumerate(jobs,1):
        folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True);complete=folder/'complete.json'
        if complete.exists():
            record=json.loads(complete.read_text())
            assert record['plan_hash']==job['plan_hash'] and record['job_identity']==digest(job) and record['prediction_sha256']==sha(folder/'predictions.npz')
            print(f'[{number}/{len(jobs)}] cached {job["id"]}',flush=True);continue
        atomic(folder/'job.json',job);print(f'[{number}/{len(jobs)}] running {job["id"]} ({len(job["indices"])} cases)',flush=True)
        start=time.time();last_heartbeat=start
        with (folder/'worker.log').open('a') as log:
            process=subprocess.Popen([str(environments[job['runtime']]),'-I',str(worker),str(folder/'job.json')],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':str(job['threads']),'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            while process.poll() is None:
                if time.time()-start>config['job_timeout_seconds']:process.kill();process.wait();break
                if time.time()-last_heartbeat>=30:
                    tail=(folder/'worker.log').read_text()[-250:];print('Progress:',tail.splitlines()[-1] if tail.splitlines() else 'initializing runtime',flush=True);last_heartbeat=time.time()
                time.sleep(1)
        if process.returncode or not complete.exists():
            failures.append(dict(id=job['id'],returncode=process.returncode,log=str(folder/'worker.log')));print('FAILED; retained with log',job['id'],flush=True)
            print((folder/'worker.log').read_text()[-1500:],flush=True)
        else:print('Completed in',round(time.time()-start,1),'seconds',flush=True)
        atomic(root/'execution_progress.json',dict(planned=len(jobs),completed=sum((Path(j['output'])/'complete.json').exists() for j in jobs),failures=failures))
    if failures:raise RuntimeError('Some fixed jobs failed. They are not dropped. Inspect execution_progress.json and rerun this cell to resume.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.prepare.read_text());prepare(cfg,Path(cfg['output_root']))
