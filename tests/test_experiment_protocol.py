import unittest
class ProtocolAudit(unittest.TestCase):
    def test_recorded_audit(self):
        import json,pathlib,ast,re,hashlib
        import numpy as np
        nb=json.loads((pathlib.Path(__file__).resolve().parents[1]/'notebooks/01_training_and_selection.ipynb').read_text())
        for c in nb['cells']:
            if isinstance(c.get('source'),list): c['source']=''.join(c['source'])
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code': compile(c['source'],f'cell{i}','exec')
        ns={'np':np,'re':re,'json':json,'hashlib':hashlib,'ABS_MIN_CAL_RECALL':.85,'N_BOOT':50,'DEFAULT_TAU':.70,'DEV':['cal','probe_dev','hn_dev']}
        for c in nb['cells']:
            if c['cell_type']=='code' and c['source'].startswith('def normalize_s1'): exec(c['source'],ns)
        for c in nb['cells']:
            if c['cell_type']=='code' and c['source'].startswith('def predict('): exec(c['source'],ns)
        for text in ['I was not suicidal yesterday, but now I want to die.','I am not suicidal but I want to end my life.']:
            assert ns['stage1_v1'](text)[0]==0
            assert ns['stage1_v2'](text)[0] is None
        assert ns['stage1_v2']('I want to end my life.')[0]==1
        assert ns['stage1_v2']('I am not suicidal.')[0]==0
        assert ns['stage1_v2']('I am not suicidal but I feel like a burden.')[0] is None
        b=dict(hpf_probe=0.,hpf_cal=.01,recall_drop=.02,cal_fpr_increase=.1,probe_fpr_increase=.05,hn_alert_rate=.15)
        y={'cal':np.array([1,1,0,0]),'probe_dev':np.array([1,1,0,0]),'hn_dev':np.array([0,0])}
        p={'cal':np.array([.9,.8,.2,.1]),'probe_dev':np.array([.9,.8,.2,.1]),'hn_dev':np.array([.2,.1])}
        select=ns['select_threshold']
        sel,_=select(y,p,p,{},b); assert sel['accepted'] and sel['tau'] is not None
        bad={s:np.full(len(y[s]),.5) for s in y}
        sel,rows=select(y,bad,p,{},b); assert not sel['accepted'] and sel['tau'] is None
        # Equal, weak models can pass original relative budgets but must fail the absolute guard.
        weak={s:np.full(len(y[s]),.1) for s in y}
        weak['cal']=np.array([.8,.1,.2,.1]);weak['probe_dev']=np.array([.8,.1,.2,.1])
        sel,_=select(y,weak,weak,{},b); assert not sel['accepted'] and sel['tau'] is None and sel['original_feasible']
        # Scores very near 1 remain separable: there is no .999 grid ceiling.
        high={'cal':np.array([.9999999,.9999998,.9999997,.9999996]),'probe_dev':np.array([.9999999,.9999998,.9999997,.9999996]),'hn_dev':np.array([.9999997,.9999996])}
        sel,_=select(y,high,high,{},b); assert sel['accepted'] and sel['tau']>.999
        # Group bootstrap samples independent groups, returns zero for identical predictions.
        a=np.array([1,1,0,0]); lo,hi=ns['cluster_delta'](y['cal'],a,a,np.array(['a','b','c','d']),B=50); assert lo==hi==0
        # Runtime final lock: no disk or inference is touched on forbidden final access.
        cell=next(c['source'] for c in nb['cells'] if c['cell_type']=='code' and 'def cache_split(' in c['source'])
        tree=ast.parse(cell); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cache_split')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'lock','exec'),dict(ns,FINAL=['test'],FINAL_OPEN=False),ns)
        try: ns['cache_split']('saqs',42,'qint8','test')
        except RuntimeError: pass
        else: raise AssertionError('Final lock failed')
        print('PASS: syntax, routing conflicts, feasible selection, strict rejection, absolute recall guard, saturated scores, group intervals, and final lock.')

if __name__=="__main__": unittest.main()
