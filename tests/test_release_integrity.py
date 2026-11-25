import importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ReleaseIntegrity(unittest.TestCase):
    def test_release(self):
        spec=importlib.util.spec_from_file_location('check_repository',ROOT/'scripts/check_repository.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main()
if __name__=='__main__':unittest.main()
