"""Create an isolated, pinned CUDA research environment on a disposable Colab.

Do not mutate the notebook's transformers/torchao environment. Optional packages
in a shared environment can break a plain AutoencoderKL import even when the
experiment never uses quantization or text models. This does not allocate GPU.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def bootstrap(root, accelerator='cuda'):
    if accelerator not in ('cuda', 'cpu'):
        raise ValueError('Expected cuda or cpu environment')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'UV_PYTHON_INSTALL_DIR': str(root/'python')}
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.22'],
                   check=True, timeout=120)
    uv = [sys.executable, '-m', 'uv']
    subprocess.run(uv+['python', 'install', '3.12'], env=env, check=True, timeout=180)
    target = root/'env'
    if not (target/'bin/python').is_file():
        if target.exists():
            raise ValueError('Incomplete environment exists; inspect before retry')
        subprocess.run(uv+['venv', '--python', '3.12', str(target)], env=env,
                       check=True, timeout=60)
    python = target/'bin/python'
    index = 'https://download.pytorch.org/whl/' + ('cu121' if accelerator == 'cuda' else 'cpu')
    subprocess.run(uv+['pip', 'install', '--python', str(python),
                       'torch==2.5.1', 'torchvision==0.20.1',
                       '--index-url', index],
                   env=env, check=True, timeout=600)
    requirements = Path(__file__).with_name('requirements.txt')
    subprocess.run(uv+['pip', 'install', '--python', str(python), '-r', str(requirements)],
                   env=env, check=True, timeout=180)
    probe = (
        "import sys,torch,timm,diffusers,json,importlib.metadata;from diffusers import AutoencoderKL;"
        "assert sys.version_info[:2]==(3,12) and torch.__version__.split('+')[0]=='2.5.1';"
        + ("assert torch.cuda.is_available(),'CUDA requested but unavailable';" if accelerator == 'cuda' else '')
        + f"path={str(root/'environment.json')!r};"
        "info={'python':sys.version,'torch':torch.__version__,'CUDA_available':torch.cuda.is_available(),"
        "'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}};"
        "open(path,'w').write(json.dumps(info,indent=2)+'\\n');"
        "print('ISOLATED_CODEC_READY',sys.version,torch.__version__,diffusers.__version__,torch.cuda.is_available())"
    )
    subprocess.run([str(python), '-c', probe], check=True, timeout=90)
    return python


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--accelerator', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    print(bootstrap(args.root, args.accelerator))
