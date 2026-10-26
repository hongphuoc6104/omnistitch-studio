"""File/CLI-boundary lifecycle checks; no actual Colab or consent is simulated as real."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.nonaligned_stain.terminal_round import bundle_code, run
from tools.nonaligned_stain.round_worker import extract_selected, archive_results, digest, stop_child


class FakeRunner:
    def __init__(self, *, fail_mount=False, fail_first_stop=False, drive_verified=True):
        self.calls, self.fail_mount, self.fail_first_stop = [], fail_mount, fail_first_stop
        self.payload = b'fixture-verified-archive'
        self.drive_verified = drive_verified

    def call(self, profile, args, *, interactive=False, required=True):
        self.calls.append(args)
        command = args[0]
        if command == 'whoami':
            return 0, 'Email: fixture@example.invalid\n'
        if command == 'new':
            return 0, 'Session READY'
        if command == 'drivemount':
            if self.fail_mount:
                raise RuntimeError('fixture consent failure')
            return 0, ''
        if command == 'exec':
            code = Path(args[args.index('--file')+1]).read_text()
            if 'ROUND2_KERNEL_READY' in code:
                return 0, 'ROUND2_KERNEL_READY Tesla T4'
            if 'ROUND2_DRIVE_SOURCES_READY' in code:
                return 0, 'ROUND2_DRIVE_SOURCES_READY'
            return 0, 'ROUND2_WORKER_STARTED 123'
        if command == 'download':
            path = Path(args[2])
            if args[1].endswith('/progress.json'):
                sha = hashlib.sha256(self.payload).hexdigest()
                path.write_text(json.dumps({'stage': 'completed', 'completed': True,
                    'Drive_backup_verified': self.drive_verified, 'archive_sha256': sha, 'archive_bytes': len(self.payload),
                    'parts': [{'name': 'result_part_000', 'bytes': len(self.payload), 'sha256': sha}]}))
            else:
                path.write_bytes(self.payload)
            return 0, ''
        if command == 'stop' and self.fail_first_stop:
            self.fail_first_stop = False
            return 1, 'fixture transient stop failure'
        return 0, ''


class RoundTerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[1]
        meta = bundle_code(self.repo, self.root/'code.tar.gz')
        cfg = {**meta, 'main_training': False, 'max_seconds': 9000,
               'profile': 'fixture', 'email': 'fixture@example.invalid',
               'archives': {k: {'filename': k+'.tar.gz', 'bytes': 1} for k in ('B2', 'C2')}}
        self.config = self.root/'config.json'
        self.config.write_text(json.dumps(cfg))

    def tearDown(self):
        self.temp.cleanup()

    def test_deterministic_bundle_and_verified_reinvocation_does_not_allocate(self):
        first = hashlib.sha256((self.root/'code.tar.gz').read_bytes()).hexdigest()
        other = bundle_code(self.repo, self.root/'second.tar.gz')
        self.assertEqual(first, other['bundle_sha256'])
        runner = FakeRunner()
        result = run(self.config, runner=runner, interval=0)
        self.assertTrue(result['completed'])
        self.assertEqual(result['stop_returncode'], 0)
        self.assertEqual([x[0] for x in runner.calls].count('new'), 1)
        prior_calls = len(runner.calls)
        self.assertEqual(run(self.config, runner=runner, interval=0)['archive_sha256'], result['archive_sha256'])
        self.assertEqual(len(runner.calls), prior_calls)

    def test_mount_failure_retrieves_log_then_stops_only_owned_session(self):
        runner = FakeRunner(fail_mount=True)
        with self.assertRaisesRegex(RuntimeError, 'consent failure'):
            run(self.config, runner=runner, interval=0)
        self.assertEqual([x[0] for x in runner.calls][-2:], ['download', 'stop'])
        state = json.loads((self.root/'state.json').read_text())
        self.assertEqual(runner.calls[-1][-1], state['session'])
        self.assertEqual(state['stop_returncode'], 0)
        self.assertFalse(state['completed'])

    def test_cleanup_retry_does_not_reallocate_or_overwrite_verified_archive(self):
        runner = FakeRunner(fail_first_stop=True)
        with self.assertRaisesRegex(RuntimeError, 'cleanup'):
            run(self.config, runner=runner, interval=0)
        before = (self.root/'round2.tar.gz').read_bytes()
        result = run(self.config, runner=runner, interval=0)
        self.assertEqual(result['stop_returncode'], 0)
        self.assertEqual([x[0] for x in runner.calls].count('new'), 1)
        self.assertEqual((self.root/'round2.tar.gz').read_bytes(), before)

    def test_main_training_is_rejected_before_any_runtime(self):
        cfg = json.loads(self.config.read_text())
        cfg['main_training'] = True
        self.config.write_text(json.dumps(cfg))
        runner = FakeRunner()
        with self.assertRaisesRegex(ValueError, 'screening'):
            run(self.config, runner=runner)
        self.assertEqual(runner.calls, [])

    def test_failed_workload_cleanup_is_retried_before_new_allocation(self):
        runner = FakeRunner(fail_mount=True, fail_first_stop=True)
        with self.assertRaisesRegex(RuntimeError, 'consent failure'):
            run(self.config, runner=runner, interval=0)
        old = json.loads((self.root/'state.json').read_text())['session']
        count = len(runner.calls)
        runner.fail_mount = False
        result = run(self.config, runner=runner, interval=0)
        self.assertEqual(runner.calls[count], ['stop', '--session', old])
        self.assertTrue(result['completed'])

    def test_selected_archive_paths_and_required_members_are_verified(self):
        archive = self.root/'input.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name in ('output/checkpoint.pt', 'dataset/manifest.json', 'dataset/a.png'):
                value = b'fixture'
                info = tarfile.TarInfo(name)
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
        result = extract_selected(archive, self.root/'safe', 'C2.pt', include_dataset=True)
        self.assertEqual(len(result), 3)
        self.assertEqual((self.root/'safe/C2.pt').read_bytes(), b'fixture')
        with tarfile.open(self.root/'unsafe.tar.gz', 'w:gz') as tar:
            info = tarfile.TarInfo('dataset/../../escape.png')
            info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            extract_selected(self.root/'unsafe.tar.gz', self.root/'unsafe', 'C2.pt', include_dataset=True)
        self.assertFalse((self.root/'escape.png').exists())

    def test_full_result_archive_is_hashed_and_split_including_backbone_license(self):
        root = self.root/'worker'
        for name in ('results', 'code', 'assets'):
            (root/name).mkdir(parents=True, exist_ok=True)
        (root/'config.json').write_text('{}')
        (root/'results/run.json').write_text('{"completed": true}')
        (root/'assets/dit_LICENSE').write_text('fixture license')
        result = archive_results(root, {'job_id': 'fixture'})
        self.assertEqual(result['archive_sha256'], digest(root/'round2.tar.gz'))
        self.assertEqual(result['archive_bytes'], (root/'round2.tar.gz').stat().st_size)
        self.assertEqual(b''.join((root/p['name']).read_bytes() for p in result['parts']),
                         (root/'round2.tar.gz').read_bytes())
        with tarfile.open(root/'round2.tar.gz') as tar:
            self.assertIn('backbone_assets/dit_LICENSE', tar.getnames())

    def test_owned_process_group_is_terminated_and_reaped(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            stop_child(child)
            self.assertIsNotNone(child.poll())
            stop_child(child)
            self.assertIsNotNone(child.returncode)
        finally:
            stop_child(child)

    def test_drive_publication_failure_keeps_complete_local_transfer_manifest(self):
        root = self.root/'worker_failure'
        for name in ('results', 'code'):
            (root/name).mkdir(parents=True, exist_ok=True)
        (root/'config.json').write_text('{}')
        (root/'results/run.json').write_text('{"completed": true}')
        original_is_dir, original_mkdir = Path.is_dir, Path.mkdir
        def is_dir(path):
            return True if str(path) == '/content/drive/MyDrive' else original_is_dir(path)
        def mkdir(path, *args, **kwargs):
            if str(path).startswith('/content/drive/'):
                raise PermissionError('fixture Drive unavailable')
            return original_mkdir(path, *args, **kwargs)
        with patch.object(Path, 'is_dir', is_dir), patch.object(Path, 'mkdir', mkdir):
            result = archive_results(root, {'job_id': 'fixture'})
        self.assertFalse(result['Drive_backup_verified'])
        self.assertIn('PermissionError', result['Drive_error'])
        self.assertEqual(result['archive_sha256'], digest(root/'round2.tar.gz'))
        self.assertEqual(sum(p['bytes'] for p in result['parts']), result['archive_bytes'])

    def test_verified_local_input_mode_never_mounts_and_records_pending_Drive(self):
        cfg = json.loads(self.config.read_text())
        content = b'fixture input archive'
        part = self.root/'input_part_000'
        part.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        cfg.update(input_transport='local_parts', allow_local_verified_completion=True,
                   inputs_sha256=sha, inputs_bytes=len(content),
                   input_parts=[{'name':part.name,'bytes':len(content),'sha256':sha}])
        self.config.write_text(json.dumps(cfg))
        runner = FakeRunner(drive_verified=False)
        result = run(self.config, runner=runner, interval=0)
        self.assertTrue(result['completed'])
        self.assertTrue(result['Drive_backup_pending'])
        self.assertFalse(result['Drive_backup_verified'])
        self.assertNotIn('drivemount',[c[0] for c in runner.calls])
        self.assertTrue(any(c[0]=='upload' and 'input_part_000' in c[2] for c in runner.calls))
        count = len(runner.calls)
        run(self.config, runner=runner, interval=0)
        self.assertEqual(len(runner.calls),count)


if __name__ == '__main__':
    unittest.main()
