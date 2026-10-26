"""One owned Colab worker: diagnostic C3 and four bounded B3 screenings.

Runs on the stock interpreter, with research children in an isolated venv.
Inputs are immutable, human-mounted Drive archives verified before extraction.
No main training, seed selection or long continuation is performed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def stop_child(child):
    """Terminate only the newly created research process group, including pip/uv."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=5)
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def extract_selected(archive, destination, checkpoint_name, *, include_dataset=False):
    """Stream selected regular files; reject traversal/duplicates before writing."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seen, total = set(), 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            selected = member.name == 'output/checkpoint.pt' or (include_dataset and
                       (member.name == 'dataset/manifest.json' or
                        (member.name.startswith('dataset/') and path.suffix == '.png')))
            if not selected:
                continue
            if path.is_absolute() or '..' in path.parts or not member.isfile() or member.name in seen:
                raise ValueError('Unsafe/duplicate input archive member')
            if not 0 < member.size <= (256*1024**2 if member.name == 'output/checkpoint.pt' else 8*1024**2):
                raise ValueError('Oversized input member')
            seen.add(member.name)
            total += member.size
            if total > 384*1024**2 or len(seen) > 2050:
                raise ValueError('Input extraction budget exceeded')
            target = destination/(checkpoint_name if member.name == 'output/checkpoint.pt' else member.name)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Input archive path escape')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1024*1024)
    if 'output/checkpoint.pt' not in seen or (include_dataset and 'dataset/manifest.json' not in seen):
        raise ValueError('Missing required input archive members')
    return sorted(seen)


def archive_results(root, cfg, failed=False):
    root = Path(root)
    path = root/'round2.tar.gz'
    with tarfile.open(path, 'w:gz', compresslevel=1) as tar:
        for name in ('results', 'config.json', 'code'):
            tar.add(root/name, arcname=name)
        env = root/'environment/environment.json'
        if env.is_file():
            tar.add(env, arcname='environment.json')
        for name in ('swinir.py', 'swinir_LICENSE', 'dit.py', 'dit_LICENSE'):
            asset_path = root/'assets'/name
            if asset_path.is_file():
                tar.add(asset_path, arcname='backbone_assets/'+name)
    expected = digest(path)
    parts = []
    with path.open('rb') as stream:
        while block := stream.read(8*1024**2):
            name = f'result_part_{len(parts):03d}'
            (root/name).write_bytes(block)
            parts.append({'name': name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
    result = {'archive_sha256': expected, 'archive_bytes': path.stat().st_size,
              'parts': parts, 'Drive_backup_verified': False}
    if Path('/content/drive/MyDrive').is_dir():
        dest = Path('/content/drive/MyDrive/CTU_P_experiments/20261027_round2')/cfg['job_id']
        dest.mkdir(parents=True, exist_ok=False)
        tmp = dest/'round2.part'
        shutil.copyfile(path, tmp)
        if digest(tmp) != expected:
            raise ValueError('Drive output checksum mismatch')
        tmp.replace(dest/'round2.tar.gz')
        (dest/'verification.json').write_text(json.dumps({'archive_sha256': expected,
            'archive_bytes': path.stat().st_size, 'failed': failed,
            'verification': 'SHA readback complete; final DriveFS flush outcome is retained in controller state.'}, indent=2)+'\n')
        from google.colab import drive
        drive.flush_and_unmount(timeout_ms=60000)
        result.update(Drive_backup_verified=True, Drive_archive=str(dest/'round2.tar.gz'),
                      Drive_verification='SHA readback then DriveFS flush')
    return result



def run(root):
    root = Path(root).resolve()
    cfg = json.loads((root/'config.json').read_text())
    if cfg.get('main_training') is not False or not 1 <= cfg['max_seconds'] <= 9000:
        raise ValueError('This worker permits bounded screening only')
    output = root/'results'
    output.mkdir(exist_ok=False)
    deadline = time.time()+cfg['remaining_seconds']
    child = None
    state = {'completed': False, 'main_training': False, 'input_dataset_sha256': cfg['dataset_sha256']}

    def progress(stage, **kwargs):
        state.update(stage=stage, **kwargs)
        tmp = root/'progress.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(root/'progress.json')
        if time.time() >= deadline:
            raise TimeoutError('Owned GPU total deadline')

    def call(module, *args, seconds):
        nonlocal child
        remaining = deadline-time.time()
        if remaining < seconds+120:
            raise TimeoutError('Insufficient reserved job/evidence time')
        command = [str(python), '-u', '-m', module, *map(str, args)]
        child = subprocess.Popen(command, cwd=root/'code', start_new_session=True)
        try:
            child.wait(timeout=seconds+60)
        except subprocess.TimeoutExpired:
            stop_child(child)
            raise
        if child.returncode:
            raise RuntimeError('Research child failed: '+module+' exit '+str(child.returncode))

    try:
        progress('verify_Drive_inputs')
        source_root = Path('/content/drive/MyDrive/CTU_P_experiments/20261026_unpaired_4X/collected')
        for branch in ('C2', 'B2'):
            item = cfg['archives'][branch]
            source = source_root/item['filename']
            if source.stat().st_size != item['bytes'] or digest(source) != item['sha256']:
                raise ValueError('Original Drive archive checksum mismatch: '+branch)
            extract_selected(source, root/'inputs', branch+'.pt', include_dataset=branch == 'C2')
            if digest(root/'inputs'/(branch+'.pt')) != item['checkpoint_sha256']:
                raise ValueError('Extracted checkpoint checksum mismatch')
        dataset = root/'inputs/dataset'
        if digest(dataset/'manifest.json') != cfg['dataset_sha256']:
            raise ValueError('Extracted dataset identity mismatch')
        sys.path.insert(0, str(root/'code'))
        from tools.nonaligned_stain.data import validate
        validate(dataset, require_same_patient_style=True)
        progress('isolated_environment')
        child = subprocess.Popen([sys.executable, '-u', '-m', 'tools.paired_stain.bootstrap_colab',
                                  '--root', str(root/'environment')], cwd=root/'code', start_new_session=True)
        child.wait(timeout=600)
        if child.returncode:
            raise RuntimeError('Isolated bootstrap failed')
        python = root/'environment/env/bin/python'
        for branch in ('B', 'C'):
            progress('public_assets_'+branch)
            call('tools.paired_stain.prepare', 'assets', '--branch', branch,
                 '--assets', root/'assets', *(['--vae-cache', root/'vae'] if branch == 'C' else []), seconds=600)
        for branch in ('C2', 'B2'):
            progress('diagnostic_'+branch)
            seconds = 1200 if branch == 'C2' else 600
            call('tools.nonaligned_stain.diagnose', '--dataset', dataset, '--checkpoint', root/'inputs'/(branch+'.pt'),
                 '--checkpoint-sha256', cfg['archives'][branch]['checkpoint_sha256'], '--branch', branch,
                 '--assets', root/'assets', '--vae-cache', root/'vae', '--output', output/(branch+'_diagnostic'),
                 '--seconds', seconds, seconds=seconds)
        defaults = {'CX': 1., 'source_self_similarity': 10., 'palette': 5., 'same_HE_identity': 5., 'range': .1}
        experiments = [('B3-0', defaults), ('B3-S', dict(defaults, source_self_similarity=1.)),
                       ('B3-I', dict(defaults, same_HE_identity=.5))]
        reports = []
        for index in range(4):
            if index == 3:
                probe = json.loads((output/'B3-0/gradient_0020.json').read_text())
                norm = lambda key: probe['terms'][key]['weighted_gradient_norm']['residual_head']
                target = 'CX' if norm('CX') <= norm('palette') else 'palette'
                decision = {'selected_term': target, 'source': 'TRAIN step20 residual-head gradient',
                            'CX_weighted_norm': norm('CX'), 'palette_weighted_norm': norm('palette'),
                            'validation_used_for_choice': False}
                (output/'B3-T_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
                experiments.append(('B3-T', dict(defaults, **{target: defaults[target]*5})))
            name, weights = experiments[index]
            progress('training_'+name)
            call('tools.nonaligned_stain.train', '--dataset', dataset, '--assets', root/'assets',
                 '--output', output/name, '--branch', 'B2', '--steps', '1000', '--seconds', '900', '--seed', '2026',
                 '--loss-weights', json.dumps(weights), '--gradient-diagnostics', '--eval-all',
                 '--compact-evaluation', seconds=900)
            report = json.loads((output/name/'run.json').read_text())
            if not report['completed'] or report['final_validation_predictions'] != 81:
                raise ValueError('Missing complete B3 validation: '+name)
            reports.append({'name': name, 'updates': report['updates_this_run'], 'weights': weights,
                            'validation_predictions': report['final_validation_predictions'],
                            'wall_seconds': report['wall_seconds'], 'seed': 2026})
            (output/'B3_screening.json').write_text(json.dumps(reports, indent=2)+'\n')
        (output/'NEXT_REVIEW.json').write_text(json.dumps({'status': 'screening_needs_native_review',
            'seed_confirmation_started': False, 'C3_additional_training_started': False,
            'main_training_started': False, 'reason': 'Review diagnostic images before choosing confirmation or C3 intervention.'}, indent=2)+'\n')
        progress('archiving')
        result = archive_results(root, cfg)
        progress('completed', completed=True, **result)
    except BaseException as error:
        stop_child(child)
        failure = type(error).__name__+': '+str(error)
        try:
            result = archive_results(root, cfg, failed=True)
        except BaseException as backup_error:
            result = {'backup_error': type(backup_error).__name__+': '+str(backup_error)}
        # Write directly: a deadline must not hide the error/evidence state.
        state.update(stage='failed', error=failure, **result)
        (root/'progress.json').write_text(json.dumps(state, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    run(parser.parse_args().root)
