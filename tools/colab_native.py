"""Create CPU→human Drive consent→native crop archive→verify→stop, per owner.

Run in the user's terminal. Successful owners are reused on a later invocation.
The scientific workorder and original hashes stay local/Drive, outside Git.
"""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.colab_fetch import Runner, digest, scrub
from tools.native_roi_archive import job_digest, relative_path, validate


BUNDLE_FILES = ('backend/dataset_mosaic.py', 'backend/io_utils.py', 'backend/wsi_exporter.py',
                'backend/serial_pairing.py', 'tools/native_roi_archive.py')


def bundle(path):
    repo = Path(__file__).resolve().parents[1]
    # Code identity must not depend on gzip time/name or checkout ownership.
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for name in BUNDLE_FILES:
                    info = tar.gettarinfo(repo/name, arcname=name)
                    info.mtime, info.uid, info.gid = 0, 0, 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    with (repo/name).open('rb') as source:
                        tar.addfile(info, source)


def launcher(root, expected):
    """Notebook cell only starts a checked file/module child, then returns."""
    return f'''import hashlib,tarfile,subprocess,sys,os
from pathlib import Path
root=Path({root!r}); archive=root/'code.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={expected!r}
code=root/'code';code.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
 members=tar.getmembers();assert len(members)=={len(BUNDLE_FILES)}
 assert {{m.name for m in members}}=={set(BUNDLE_FILES)!r}
 for member in members:
  assert member.isfile();target=(code/member.name).resolve();assert target.is_relative_to(code.resolve())
  target.parent.mkdir(parents=True,exist_ok=True)
  with tar.extractfile(member) as src,target.open('wb') as dst:dst.write(src.read())
with (root/'worker.log').open('ab') as log:
 child=subprocess.Popen([sys.executable,'-u','-m','tools.native_roi_archive','--config',str(root/'config.json'),'--run-root',str(root/'result')],cwd=code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}})
print('NATIVE_WORKER_STARTED',child.pid)
'''


def run_account(config, email, output, code_archive, runner, *, interval=5., deadline=1200):
    """Collect only one owner, preserving checksums and owned-runtime cleanup."""
    relative_path(config['drive_destination'])
    validate(config)
    token = job_digest(config)
    destination = Path(output)/config['profile']/token[:20]
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/'native_windows.tar'
    status_path = destination/'transfer.json'
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if old.get('verified') and old.get('stop_returncode') == 0 and old.get('job_sha256') == token:
            if archive.stat().st_size == old['archive_bytes'] and digest(archive) == old['archive_sha256']:
                print('NATIVE_OWNER_ALREADY_VERIFIED', config['profile'], flush=True)
                return old
            raise ValueError('Previously verified local archive is corrupt')
    if archive.exists():
        raise ValueError('Uncommitted local archive exists; inspect transfer status before retry')
    name = 'ctup-native-' + config['profile'] + '-' + uuid.uuid4().hex[:8]
    if not re.fullmatch('[a-z0-9-]{1,63}', name):
        raise ValueError('Invalid profile/session name')
    root = '/content/ctup_native_' + uuid.uuid4().hex[:12]
    profile = config['profile']
    state = {'profile': profile, 'email': email, 'session': name, 'hardware': 'CPU',
             'job_sha256': token, 'verified': False, 'trainingAccepted': False}
    created = False
    print('CPU |', email, '|', name, flush=True)
    try:
        _, identity = runner.call(profile, ['whoami'])
        if not re.search(r'^Email:\s*'+re.escape(email)+r'\s*$', identity, re.M):
            raise ValueError('CLI email does not match requested owner')
        created = True
        _, result = runner.call(profile, ['new', '--session', name])
        if 'Session READY' not in result:
            raise RuntimeError('Missing CPU allocation marker')
        with tempfile.TemporaryDirectory(prefix='native-transfer-', dir=destination) as temp:
            temp = Path(temp)
            probe = temp/'probe.py'
            probe.write_text("import cv2,numpy,PIL,tifffile\nprint('NATIVE_KERNEL_READY')\n")
            _, result = runner.call(profile, ['exec', '--session', name, '--file', str(probe), '--timeout', '15'])
            if 'NATIVE_KERNEL_READY' not in result:
                raise RuntimeError('Kernel/dependency probe failed')
            print(f'colab-profile {profile} drivemount --session {name}', flush=True)
            print('Cap quyen Google bang '+email+'; giu terminal den khi xong.', flush=True)
            runner.call(profile, ['drivemount', '--session', name], interactive=True)
            cfg = temp/'config.json'
            cfg.write_text(json.dumps(config))
            # File API uploads may create directories implicitly; create root
            # in the same short launcher used for the actual checked child.
            prep = temp/'prepare.py'
            prep.write_text(f"from pathlib import Path\nPath({root!r}).mkdir()\nprint('NATIVE_ROOT_READY')\n")
            _, result = runner.call(profile, ['exec', '--session', name, '--file', str(prep), '--timeout', '15'])
            if 'NATIVE_ROOT_READY' not in result:
                raise RuntimeError('Remote directory preparation failed')
            runner.call(profile, ['upload', str(cfg), root+'/config.json', '--session', name])
            runner.call(profile, ['upload', str(code_archive), root+'/code.tar.gz', '--session', name])
            launch = temp/'launch.py'
            launch.write_text(launcher(root, config['bundle_sha256']))
            _, result = runner.call(profile, ['exec', '--session', name, '--file', str(launch), '--timeout', '15'])
            if 'NATIVE_WORKER_STARTED' not in result:
                raise RuntimeError('Native file child did not start')
            end, last = time.monotonic()+deadline, None
            while time.monotonic() < end:
                path = temp/'progress.json'
                rc, _ = runner.call(profile, ['download', root+'/result/progress.json', str(path), '--session', name], required=False)
                if rc == 0:
                    progress = json.loads(path.read_text())
                    value = (progress['stage'], progress.get('groups_done'), progress.get('regions_done'))
                    if value != last:
                        print('NATIVE_PROGRESS', profile, *value, flush=True)
                        last = value
                    if progress['stage'] == 'failed':
                        raise RuntimeError('Native collector: '+progress['error'])
                    if progress['stage'] == 'completed':
                        break
                time.sleep(interval)
            else:
                raise TimeoutError('Native collection exceeded bounded time')
            if progress['job_sha256'] != token or not progress['Drive_backup_verified']:
                raise ValueError('Completion/Drive backup mismatch')
            parts = progress['parts']
            if not parts or sum(p['bytes'] for p in parts) != progress['archive_bytes']:
                raise ValueError('Incomplete transfer part list')
            assembled = temp/'assembled.tar'
            with assembled.open('wb') as dst:
                for index, part in enumerate(parts):
                    if part['name'] != f'part_{index:03d}' or not 0 < part['bytes'] <= 16*1024**2:
                        raise ValueError('Unsafe transfer part')
                    local = temp/'part'
                    runner.call(profile, ['download', root+'/result/'+part['name'], str(local), '--session', name])
                    if local.stat().st_size != part['bytes'] or digest(local) != part['sha256']:
                        raise ValueError('Downloaded native part checksum mismatch')
                    with local.open('rb') as src:
                        shutil.copyfileobj(src, dst, 4*1024**2)
            if assembled.stat().st_size != progress['archive_bytes'] or digest(assembled) != progress['archive_sha256']:
                raise ValueError('Assembled native archive checksum mismatch')
            os.link(assembled, archive)
            state.update(progress)
            state['verified'] = True
            state['archive'] = str(archive)
    except BaseException as error:
        state['error'] = scrub(type(error).__name__+': '+str(error))
        raise
    finally:
        if created:
            try:
                rc, _ = runner.call(profile, ['stop', '--session', name], required=False)
                state['stop_returncode'] = rc
            except Exception as error:
                state['cleanup_error'] = scrub(str(error))
        status_path.write_text(json.dumps(state, indent=2)+'\n')
    if state.get('stop_returncode') != 0:
        raise RuntimeError('Native archive verified but CPU cleanup failed')
    print('NATIVE_OWNER_COMPLETE', profile, archive, flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    order = json.loads(Path(settings['workorder_path']).read_text())
    output = Path(settings['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'terminal.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='native-code-', dir=output) as temp:
            code = Path(temp)/'code.tar.gz'
            bundle(code)
            results = []
            for account in settings['accounts']:
                config = {'profile': account['profile'], 'workorder': order, 'bundle_sha256': digest(code),
                          'drive_destination': settings['drive_destination']+'/'+account['profile']}
                results.append(run_account(config, account['email'], output, code, Runner()))
            (output/'completed.json').write_text(json.dumps(results, indent=2)+'\n')
    print('NATIVE_COLLECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
