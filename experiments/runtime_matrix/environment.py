"""Create isolated environments without invoking ensurepip or changing the kernel."""
from pathlib import Path
import os,shutil,subprocess,sys,venv

def ensure_environment(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # Recreate the environment scaffold even if a prior ensurepip attempt left
    # a Python executable behind. Existing installed packages are not removed.
    venv.EnvBuilder(with_pip=False,system_site_packages=False,clear=False).create(folder)
    python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    probe=subprocess.run([str(python),'-I','-c','import sys;assert sys.prefix!=sys.base_prefix;print(sys.prefix)'],capture_output=True,text=True)
    if probe.returncode:raise RuntimeError('Isolated Python cannot start: '+probe.stderr[-1500:])
    assert Path(probe.stdout.strip()).resolve()==folder.resolve()
    return python

def pip_runner(controller_python=None):
    candidates=[[str(controller_python or sys.executable),'-m','pip']]
    executable=shutil.which('pip')
    if executable:candidates.append([executable])
    for command in candidates:
        try:probe=subprocess.run(command+['--help'],capture_output=True,text=True)
        except OSError:continue
        if probe.returncode==0 and '--python' in probe.stdout:return command
    raise RuntimeError('The notebook host needs an existing pip >=22.3 with --python support. No global package changes were made.')

def pip_command(python,*args,controller_python=None):
    # pip's documented --python mode can manage an environment without pip.
    return pip_runner(controller_python)+['--python',str(python),*map(str,args)]
