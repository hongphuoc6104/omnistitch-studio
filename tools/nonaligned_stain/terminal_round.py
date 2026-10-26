"""One-terminal, human-consented T4 screening with owned-runtime cleanup.

No runtime is allocated by importing or preparing the bundle. The user invokes
the finished command when available to consent, avoiding stale session names.
"""
import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

from tools.colab_fetch import Runner, digest, scrub

CODE_FILES = tuple('tools/paired_stain/'+n for n in
                   ('data.py', 'models.py', 'train.py', 'prepare.py', 'bootstrap_colab.py', 'requirements.txt')) + tuple(
                       'tools/nonaligned_stain/'+n for n in
                       ('data.py', 'objectives.py', 'bridge.py', 'train.py', 'diagnose.py', 'round_worker.py'))


def bundle_code(repo, destination):
    """Deterministic content identity, with no private images or checkpoints."""
    repo, destination = Path(repo), Path(destination)
    manifest = {name: digest(repo/name) for name in CODE_FILES}
    with destination.open('xb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w|') as tar:
            for name in CODE_FILES:
                content = (repo/name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(content), 0, 0o644
                tar.addfile(info, io.BytesIO(content))
    return {'bundle_sha256': digest(destination), 'code_files': manifest}


class BoundedRunner(Runner):
    def call(self, profile, args, *, interactive=False, required=True):
        if not interactive:
            return super().call(profile, args, required=required)
        result = subprocess.run([self.cli, profile, *args], timeout=300)
        if required and result.returncode:
            raise RuntimeError('Drive consent command failed')
        return result.returncode, ''


def run(config_path, *, runner=None, interval=5):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    cfg = json.loads(config_path.read_text())
    if (cfg.get('main_training') is not False or not 1 <= cfg['max_seconds'] <= 9000
            or not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', cfg['profile'])
            or not re.fullmatch(r'[^\s@]+@[^\s@]+', cfg['email'])
            or digest(base/'code.tar.gz') != cfg['bundle_sha256']):
        raise ValueError('Invalid screening configuration/bundle')
    runner = runner or BoundedRunner(timeout=120)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    with (base/'terminal.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = base/'state.json'
        if prior.is_file():
            previous = json.loads(prior.read_text())
            if previous.get('allocation_started') and previous.get('stop_returncode') != 0:
                session = previous.get('session', '')
                if (previous.get('profile') != cfg['profile'] or previous.get('main_training') is not False
                        or previous.get('job_fingerprint') != fingerprint
                        or not re.fullmatch('ctup-4x-round2-[0-9a-f]{8}', session)):
                    raise ValueError('Cannot establish ownership for cleanup retry')
                rc, message = runner.call(cfg['profile'], ['stop', '--session', session], required=False)
                missing = rc != 0 and 'not found' in message.lower()
                previous.update(stop_returncode=0 if missing else rc, cleanup_retried_without_retraining=True,
                                previous_runtime_already_missing=missing)
                prior.write_text(json.dumps(previous, indent=2)+'\n')
                (base/'attempts'/session/'state.json').write_text(json.dumps(previous, indent=2)+'\n')
                if rc != 0 and not missing:
                    raise RuntimeError('Owned runtime cleanup still unconfirmed; no new allocation')
            archive = base/'round2.tar.gz'
            if (previous.get('completed') and previous.get('Drive_backup_verified')
                    and archive.is_file()
                    and digest(archive) == previous.get('archive_sha256')
                    and previous.get('job_fingerprint') == fingerprint):
                print('ROUND2_ALREADY_VERIFIED', flush=True)
                return previous
        if (base/'round2.tar.gz').exists():
            raise ValueError('Existing archive/configuration conflict; use a new work directory')
        name = 'ctup-4x-round2-'+uuid.uuid4().hex[:8]
        remote = '/content/ctup_round2_'+uuid.uuid4().hex[:12]
        profile = cfg['profile']
        attempt = base/'attempts'/name
        attempt.mkdir(parents=True, exist_ok=False)
        state = {'session': name, 'profile': profile, 'email': cfg['email'], 'hardware': 'T4',
                 'main_training': False, 'completed': False, 'bundle_sha256': cfg['bundle_sha256'],
                 'job_fingerprint': fingerprint}
        created = False

        def persist():
            text = json.dumps(state, indent=2)+'\n'
            (attempt/'state.json').write_text(text)
            tmp = base/'state.tmp'
            tmp.write_text(text)
            tmp.replace(base/'state.json')

        def collect(progress):
            parts = progress['parts']
            if not 1 <= len(parts) <= 256 or sum(p['bytes'] for p in parts) != progress['archive_bytes']:
                raise ValueError('Invalid output transfer manifest')
            target = base/'round2.tar.gz'
            # Failed-attempt evidence is retained under its attempt directory.
            if progress['stage'] == 'failed':
                target = attempt/'failed_round2.tar.gz'
            with tempfile.TemporaryDirectory(dir=attempt) as temp:
                temp = Path(temp)
                archive = temp/'archive'
                with archive.open('wb') as dst:
                    for i, part in enumerate(parts):
                        if part['name'] != f'result_part_{i:03d}' or not 0 < part['bytes'] <= 8*1024**2:
                            raise ValueError('Unsafe result part')
                        if time.monotonic() >= end:
                            raise TimeoutError('Result transfer exceeded total GPU deadline')
                        local = temp/'part'
                        runner.call(profile, ['download', remote+'/'+part['name'], str(local), '--session', name])
                        if local.stat().st_size != part['bytes'] or digest(local) != part['sha256']:
                            raise ValueError('Result part checksum mismatch')
                        with local.open('rb') as src:
                            shutil.copyfileobj(src, dst, 1024*1024)
                if digest(archive) != progress['archive_sha256']:
                    raise ValueError('Complete result checksum mismatch')
                os.link(archive, target)
            return str(target)

        try:
            _, identity = runner.call(profile, ['whoami'])
            if not re.search(r'^Email:\s*'+re.escape(cfg['email'])+r'\s*$', identity, re.M):
                raise ValueError('Wrong profile Gmail')
            print('T4 screening |', cfg['email'], '|', name, '| NOT main training', flush=True)
            created = True
            started = time.monotonic()
            end = started+cfg['max_seconds']
            state['allocation_started'] = True
            persist()
            _, result = runner.call(profile, ['new', '--gpu', 'T4', '--session', name])
            if 'Session READY' not in result:
                raise RuntimeError('T4 allocation marker missing')
            with tempfile.TemporaryDirectory(dir=attempt) as temp:
                temp = Path(temp)
                probe = temp/'probe.py'
                probe.write_text("import subprocess\n"+f"from pathlib import Path\nPath({remote!r}).mkdir()\n"
                                 "print('ROUND2_KERNEL_READY',subprocess.check_output(['nvidia-smi','--query-gpu=name',"
                                 "'--format=csv,noheader'],text=True,timeout=10).strip())\n")
                _, result = runner.call(profile, ['exec', '--session', name, '--file', str(probe), '--timeout', '30'])
                if 'ROUND2_KERNEL_READY' not in result or 'T4' not in result:
                    raise RuntimeError('T4 kernel probe failed')
                state['kernel_ready'] = True
                persist()
                print('colab-profile '+profile+' drivemount --session '+name, flush=True)
                print('Cap quyen bang '+cfg['email']+'; giu terminal den ROUND2_COMPLETE.', flush=True)
                runner.call(profile, ['drivemount', '--session', name], interactive=True)
                mount = temp/'mount.py'
                sources = {k: {'path': '/content/drive/MyDrive/CTU_P_experiments/20261026_unpaired_4X/collected/'+v['filename'],
                               'bytes': v['bytes']} for k, v in cfg['archives'].items()}
                mount.write_text('from pathlib import Path\n'+f'items={sources!r}\n'+
                                 "assert Path('/content/drive/MyDrive').is_dir()\n"
                                 "for item in items.values():\n assert Path(item['path']).stat().st_size==item['bytes']\n"
                                 "print('ROUND2_DRIVE_SOURCES_READY')\n")
                _, result = runner.call(profile, ['exec', '--session', name, '--file', str(mount), '--timeout', '30'])
                if 'ROUND2_DRIVE_SOURCES_READY' not in result:
                    raise RuntimeError('Mounted Drive lacks verified input sources')
                state['Drive_sources_verified_by_size'] = True
                persist()
                cfg = dict(cfg, job_id=name, remaining_seconds=max(1, int(end-time.monotonic()-1800)))
                if cfg['remaining_seconds'] < 2700:
                    raise TimeoutError('Insufficient screening and transfer budget after consent')
                remote_cfg = temp/'config.json'
                remote_cfg.write_text(json.dumps(cfg, indent=2)+'\n')
                for local, filename in [(base/'code.tar.gz', 'code.tar.gz'), (remote_cfg, 'config.json')]:
                    runner.call(profile, ['upload', str(local), remote+'/'+filename, '--session', name])
                launcher = temp/'launch.py'
                launcher.write_text(f'''import hashlib,json,tarfile,subprocess,sys,shutil
from pathlib import Path
root=Path({remote!r});cfg=json.loads((root/'config.json').read_text())
assert hashlib.sha256((root/'code.tar.gz').read_bytes()).hexdigest()==cfg['bundle_sha256']
dst=root/'code';dst.mkdir()
with tarfile.open(root/'code.tar.gz') as tar:
 members=tar.getmembers()
 assert len(members)==len(cfg['code_files']) and {{m.name for m in members}}==set(cfg['code_files'])
 for m in members:
  target=(dst/m.name).resolve()
  assert m.isfile() and m.size<=2*1024**2 and target.is_relative_to(dst.resolve())
  content=tar.extractfile(m).read()
  assert hashlib.sha256(content).hexdigest()==cfg['code_files'][m.name]
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
with (root/'worker.log').open('ab') as log:
 child=subprocess.Popen([sys.executable,'-u','-m','tools.nonaligned_stain.round_worker','--root',str(root)],cwd=dst,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('ROUND2_WORKER_STARTED',child.pid)
''')
                _, result = runner.call(profile, ['exec', '--session', name, '--file', str(launcher), '--timeout', '30'])
                if 'ROUND2_WORKER_STARTED' not in result:
                    raise RuntimeError('Worker launch marker missing')
                state['launched'] = True
                persist()
                last = None
                while time.monotonic() < end:
                    path = attempt/'progress.json'
                    rc, _ = runner.call(profile, ['download', remote+'/progress.json', str(path), '--session', name], required=False)
                    if rc == 0:
                        progress = json.loads(path.read_text())
                        stage = progress['stage']
                        if stage != last:
                            print('ROUND2_PROGRESS', stage, flush=True)
                            last = stage
                        if stage in ('completed', 'failed'):
                            state.update(progress)
                            if progress.get('parts'):
                                state['local_archive'] = collect(progress)
                            persist()
                            if stage == 'failed':
                                raise RuntimeError(progress.get('error', 'Research worker failed'))
                            if not progress.get('Drive_backup_verified') or not state.get('local_archive'):
                                raise RuntimeError('Missing durable/local verified results')
                            break
                    time.sleep(interval)
                else:
                    raise TimeoutError('Owned T4 total deadline')
        except BaseException as error:
            state['error'] = scrub(type(error).__name__+': '+str(error))
            raise
        finally:
            if created:
                try:
                    runner.call(profile, ['download', remote+'/worker.log', str(attempt/'worker.log'), '--session', name], required=False)
                except Exception as error:
                    state['log_fetch_error'] = scrub(str(error))
                try:
                    state['stop_returncode'] = runner.call(profile, ['stop', '--session', name], required=False)[0]
                except Exception as error:
                    state['cleanup_error'] = scrub(str(error))
                state['allocated_seconds_including_cleanup'] = time.monotonic()-started
            persist()
        if state.get('stop_returncode') != 0:
            raise RuntimeError('Results verified but runtime cleanup not confirmed')
        print('ROUND2_COMPLETE', flush=True)
        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
