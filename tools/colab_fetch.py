"""Interactive Drive mount followed immediately by a bounded verified file fetch.

No credentials are stored here. The user performs the normal Colab CLI consent
in their own terminal. Only the uniquely named CPU runtime created by this job
is stopped. Downloaded archives are never executed or extracted by this tool.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

REMOTE_WORKER = r'''
import hashlib,json,shutil,traceback
from pathlib import Path
cfg=CONFIG
root=Path(cfg['remote_root']);root.mkdir(exist_ok=True)
def state(value):
 p=root/'progress.tmp';p.write_text(json.dumps(value));p.replace(root/'progress.json')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(4*1024**2):h.update(block)
 return h.hexdigest()
try:
 source=Path(cfg['source']);assert source.is_file(),'Drive source not found'
 assert source.stat().st_size==cfg['bytes'],'Drive source size mismatch'
 target=root/'payload';count=0
 state({'stage':'copying','bytes':0})
 with source.open('rb') as src,target.open('wb') as dst:
  while block:=src.read(4*1024**2):
   dst.write(block);count+=len(block);state({'stage':'copying','bytes':count})
 assert digest(target)==cfg['sha256'],'Drive source checksum mismatch'
 parts=[]
 with target.open('rb') as stream:
  i=0
  while block:=stream.read(16*1024**2):
   name=f'part_{i:03d}';(root/name).write_bytes(block)
   parts.append({'name':name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()});i+=1
 state({'stage':'completed','bytes':count,'sha256':cfg['sha256'],'parts':parts})
except Exception as error:
 state({'stage':'failed','error':str(error)});raise
'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()


def scrub(text):
    return re.sub(r'(https?://[^\s?]+)\?[^\s]+', r'\1?[redacted]', text)


class Runner:
    def __init__(self, cli='colab-profile', timeout=45):
        self.cli, self.timeout = cli, timeout

    def call(self, profile, args, *, interactive=False, required=True):
        result = subprocess.run([self.cli, profile, *args], text=True,
                                capture_output=not interactive, timeout=None if interactive else self.timeout)
        text = '' if interactive else (result.stdout or '') + (result.stderr or '')
        if required and result.returncode:
            raise RuntimeError('Colab CLI failed: ' + scrub(text)[-1000:])
        return result.returncode, text


def fetch(config, runner=None, *, interval=5., deadline_seconds=300):
    """Allocate→probe→human mount→file child→verify→release, in one terminal."""
    runner = runner or Runner()
    profile, email = config['profile'], config['email']
    remote_source = config['source']
    path = PurePosixPath(remote_source)
    if '..' in path.parts or not remote_source.startswith('/content/drive/'):
        raise ValueError('Expected contained absolute Drive source')
    manifest = json.loads(Path(config['manifest']).read_text())
    expected, size = manifest['archive_sha256'], manifest['archive_bytes']
    if not re.fullmatch('[0-9a-f]{64}', expected) or type(size) is not int or size <= 0:
        raise ValueError('Invalid expected archive checksum/size')
    if sum(p['bytes'] for p in manifest['parts']) != size:
        raise ValueError('Archive manifest parts do not cover size')
    destination = Path(config['destination']).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == size and digest(destination) == expected:
            print('COLAB_FETCH_ALREADY_VERIFIED', destination, flush=True)
            return {'already_present': True, 'sha256': expected}
        raise ValueError('Destination exists with different contents; choose a new path')
    prefix = config.get('session_prefix', 'ctup-drive-fetch')
    if not re.fullmatch('[a-z0-9-]{1,42}', prefix):
        raise ValueError('Invalid session prefix')
    name = prefix + '-' + uuid.uuid4().hex[:8]
    remote_root = '/content/ctup_fetch_' + uuid.uuid4().hex[:12]
    state = {'session': name, 'profile': profile, 'email': email,
             'hardware': 'CPU', 'source': remote_source, 'sha256': expected,
             'destination': str(destination), 'verified': False}
    created = False
    print(f'CPU | {email} | {name}', flush=True)
    try:
        _, identity = runner.call(profile, ['whoami'])
        if not re.search(r'^Email:\s*' + re.escape(email) + r'\s*$', identity, re.M):
            raise ValueError('CLI profile email does not match requested account')
        created = True
        _, result = runner.call(profile, ['new', '--session', name])
        if 'Session READY' not in result:
            raise RuntimeError('Missing allocation marker')
        with tempfile.TemporaryDirectory(prefix='colab-fetch-', dir=destination.parent) as temporary:
            temporary = Path(temporary)
            probe = temporary / 'probe.py'
            probe.write_text("print('FETCH_KERNEL_READY')\n")
            _, result = runner.call(profile, ['exec', '--session', name, '--file', str(probe), '--timeout', '15'])
            if 'FETCH_KERNEL_READY' not in result:
                raise RuntimeError('Kernel probe did not complete')
            print(f'colab-profile {profile} drivemount --session {name}', flush=True)
            print(f'Cap quyen Google bang {email}; cho Mounted at /content/drive.', flush=True)
            runner.call(profile, ['drivemount', '--session', name], interactive=True)
            worker = temporary / 'worker.py'
            cfg = {'remote_root': remote_root, 'source': remote_source, 'sha256': expected, 'bytes': size}
            worker.write_text(REMOTE_WORKER.replace('CONFIG', repr(cfg), 1))
            runner.call(profile, ['upload', str(worker), remote_root + '.py', '--session', name])
            launcher = temporary / 'launcher.py'
            launcher.write_text("import subprocess,sys\n"
                                f"with open({(remote_root + '.log')!r},'ab') as log:\n"
                                f" p=subprocess.Popen([sys.executable,'-u',{(remote_root + '.py')!r}],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
                                "print('FETCH_WORKER_STARTED',p.pid)\n")
            _, result = runner.call(profile, ['exec', '--file', str(launcher), '--timeout', '15', '--session', name])
            if 'FETCH_WORKER_STARTED' not in result:
                raise RuntimeError('Missing file-child launch marker')
            limit, last = time.monotonic() + deadline_seconds, None
            progress_path = temporary / 'progress.json'
            while time.monotonic() < limit:
                rc, _ = runner.call(profile, ['download', remote_root + '/progress.json', str(progress_path), '--session', name], required=False)
                if rc == 0:
                    progress = json.loads(progress_path.read_text())
                    current = (progress['stage'], progress.get('bytes'))
                    if current != last:
                        print('FETCH_PROGRESS', *current, flush=True)
                        last = current
                    if progress['stage'] == 'failed':
                        raise RuntimeError('Remote fetch failed: ' + progress['error'])
                    if progress['stage'] == 'completed':
                        break
                time.sleep(interval)
            else:
                raise TimeoutError('File collector did not complete in bounded time')
            if progress['sha256'] != expected or progress['bytes'] != size:
                raise ValueError('Remote completion disagrees with expected archive')
            if not progress['parts'] or sum(x['bytes'] for x in progress['parts']) != size:
                raise ValueError('Remote parts do not cover expected archive')
            assembled = temporary / 'assembled'
            with assembled.open('wb') as output:
                for part in progress['parts']:
                    if not re.fullmatch('part_[0-9]{3}', part['name']) or not 0 < part['bytes'] <= 16 * 1024**2:
                        raise ValueError('Unsafe or oversized transfer part')
                    local = temporary / part['name']
                    runner.call(profile, ['download', remote_root + '/' + part['name'], str(local), '--session', name])
                    if local.stat().st_size != part['bytes'] or digest(local) != part['sha256']:
                        raise ValueError('Downloaded part checksum mismatch')
                    with local.open('rb') as source:
                        while block := source.read(4 * 1024**2):
                            output.write(block)
            if assembled.stat().st_size != size or digest(assembled) != expected:
                raise ValueError('Assembled archive checksum mismatch')
            # Both paths share a filesystem. Link publication is exclusive,
            # preserving any output created by a concurrent job since preflight.
            os.link(assembled, destination)
            assembled.unlink()
            state['verified'] = True
    except BaseException as error:
        state['error'] = scrub(type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if created:
            try:
                rc, _ = runner.call(profile, ['stop', '--session', name], required=False)
                state['stop_returncode'] = rc
            except Exception as error:
                state['cleanup_error'] = scrub(str(error))
        destination.with_name(destination.name + '.fetch.json').write_text(json.dumps(state, indent=2) + '\n')
    if state.get('stop_returncode') != 0:
        raise RuntimeError('Archive verified but runtime cleanup failed; see fetch status')
    print('COLAB_FETCH_COMPLETE', destination, flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Local JSON config; contains no credentials')
    args = parser.parse_args()
    fetch(json.loads(Path(args.config).read_text()))


if __name__ == '__main__':
    main()
