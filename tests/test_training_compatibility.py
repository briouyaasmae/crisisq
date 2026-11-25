import unittest
class ProtocolAudit(unittest.TestCase):
    def test_recorded_audit(self):
        import ast,json,inspect,pathlib
        nb=json.loads((pathlib.Path(__file__).resolve().parents[1]/'notebooks/01_training_and_selection.ipynb').read_text())
        for c in nb['cells']:
            if isinstance(c.get('source'),list): c['source']=''.join(c['source'])
        source=next(c['source'] for c in nb['cells'] if c['cell_type']=='code' and 'def compatible_training_arguments(' in c['source'])
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='compatible_training_arguments')
        class OldArgs:
            def __init__(self,output_dir,warmup_ratio=0.): self.values=(output_dir,warmup_ratio)
        class NewArgs:
            def __init__(self,output_dir,warmup_steps=0): self.values=(output_dir,warmup_steps)
        ns={'inspect':inspect,'TrainingArguments':NewArgs}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'compat','exec'),ns)
        f=ns['compatible_training_arguments']; kw={'output_dir':'example','warmup_ratio':.06}
        assert f(kw,OldArgs).values==('example',.06)
        assert f(kw,NewArgs).values==('example',.06)
        assert kw=={'output_dir':'example','warmup_ratio':.06}
        try: f(dict(kw,unsupported_setting=True),NewArgs)
        except RuntimeError: pass
        else: raise AssertionError('Unexpected settings should not be silently discarded')
        print('PASS: old/new API signatures retain 6% warmup; input is preserved; unknown settings fail explicitly.')

if __name__=="__main__": unittest.main()
