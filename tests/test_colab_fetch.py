"""File-child E2E of the fetch lifecycle without Google credentials or a VM."""
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.colab_fetch import fetch


class FileCli:
    def __init__(self, root, *, email='fixture@example.test', corrupt=False, fail_mount=False):
        self.root, self.email = Path(root), email
        self.corrupt, self.fail_mount = corrupt, fail_mount
        self.calls, self.child, self.worker = [], None, None
        self.race_destination, self.stop_returncode = None, 0

    def mapped(self, path):
        return self.root / path.removeprefix('/content/').lstrip('/')

    def call(self, profile, args, *, interactive=False, required=True):
        self.calls.append((profile, list(args), interactive))
        command = args[0]
        if command == 'whoami':
            return 0, 'Email: ' + self.email + '\n'
        if command == 'new':
            return 0, 'Session READY'
        if command == 'drivemount':
            if self.fail_mount:
                raise RuntimeError('fixture mount failure')
            return 0, 'Mounted at /content/drive'
        if command == 'upload':
            self.worker = self.mapped(args[2])
            self.worker.parent.mkdir(parents=True, exist_ok=True)
            self.worker.write_text(Path(args[1]).read_text().replace('/content/', str(self.root) + '/'))
            return 0, ''
        if command == 'exec':
            code = Path(args[args.index('--file') + 1]).read_text()
            if 'FETCH_WORKER_STARTED' in code:
                self.child = subprocess.Popen([sys.executable, str(self.worker)],
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return 0, 'FETCH_WORKER_STARTED ' + str(self.child.pid)
            p = subprocess.run([sys.executable, args[args.index('--file') + 1]],
                               capture_output=True, text=True, check=True)
            return 0, p.stdout
        if command == 'download':
            source = self.mapped(args[1])
            if not source.is_file():
                return 1, 'fixture file not present yet'
            shutil.copyfile(source, args[2])
            if self.corrupt and source.name == 'part_000':
                with Path(args[2]).open('ab') as f:
                    f.write(b'corruption')
            if self.race_destination and source.name == 'part_000':
                Path(self.race_destination).write_bytes(b'other job output')
            return 0, ''
        if command == 'stop':
            if self.child:
                try:
                    self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            return self.stop_returncode, ''
        raise AssertionError('Unexpected fixture command: ' + command)


class TestColabFetch(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        payload = bytes(range(256)) * 300
        source = root / 'remote/drive/source.bin'
        source.parent.mkdir(parents=True)
        source.write_bytes(payload)
        manifest = root / 'manifest.json'
        digest = hashlib.sha256(payload).hexdigest()
        manifest.write_text(json.dumps({'archive_sha256': digest, 'archive_bytes': len(payload),
                                        'parts': [{'bytes': len(payload)}]}))
        config = {'profile': 'fixture', 'email': 'fixture@example.test',
                  'source': '/content/drive/source.bin', 'manifest': str(manifest),
                  'destination': str(root / 'result.tar.gz'), 'session_prefix': 'fixture-fetch'}
        return config, payload, FileCli(root / 'remote')

    def execute(self, config, runner):
        with contextlib.redirect_stdout(io.StringIO()):
            return fetch(config, runner, interval=.01, deadline_seconds=10)

    def test_file_child_download_sha_and_only_owned_cpu_runtime_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            config, payload, cli = self.fixture(root)
            state = self.execute(config, cli)
            self.assertEqual(Path(config['destination']).read_bytes(), payload)
            self.assertTrue(state['verified'])
            self.assertEqual(state['stop_returncode'], 0)
            stops = [args for _, args, _ in cli.calls if args[0] == 'stop']
            self.assertEqual(stops, [['stop', '--session', state['session']]])
            mount = [x for x in cli.calls if x[1][0] == 'drivemount']
            self.assertTrue(mount[0][2])

    def test_corrupt_download_is_not_published_and_runtime_is_released(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, cli = self.fixture(root)
            cli.corrupt = True
            with self.assertRaisesRegex(ValueError, 'part checksum'):
                self.execute(config, cli)
            self.assertFalse(Path(config['destination']).exists())
            self.assertEqual(cli.calls[-1][1][0], 'stop')

    def test_source_checksum_error_propagates_even_if_cli_execution_succeeded(self):
        with tempfile.TemporaryDirectory() as root:
            config, payload, cli = self.fixture(root)
            cli.mapped(config['source']).write_bytes(b'X' * len(payload))
            with self.assertRaisesRegex(RuntimeError, 'source checksum'):
                self.execute(config, cli)
            self.assertEqual(cli.calls[-1][1][0], 'stop')
            self.assertFalse(Path(config['destination']).exists())

    def test_wrong_email_does_not_allocate_or_mount(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, cli = self.fixture(root)
            cli.email = 'different@example.test'
            with self.assertRaisesRegex(ValueError, 'email'):
                self.execute(config, cli)
            self.assertEqual([x[1][0] for x in cli.calls], ['whoami'])

    def test_mount_failure_releases_owned_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, cli = self.fixture(root)
            cli.fail_mount = True
            with self.assertRaisesRegex(RuntimeError, 'mount failure'):
                self.execute(config, cli)
            self.assertEqual(cli.calls[-1][1][0], 'stop')

    def test_verified_destination_is_reused_without_allocation_or_auth(self):
        with tempfile.TemporaryDirectory() as root:
            config, payload, cli = self.fixture(root)
            Path(config['destination']).write_bytes(payload)
            state = self.execute(config, cli)
            self.assertTrue(state['already_present'])
            self.assertEqual(cli.calls, [])

    def test_unsafe_source_or_manifest_is_rejected_before_any_cli_call(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, cli = self.fixture(root)
            config['source'] = '/content/drive/../secret'
            with self.assertRaises(ValueError):
                self.execute(config, cli)
            self.assertEqual(cli.calls, [])
            config['source'] = '/content/drive/source.bin'
            p = Path(config['manifest'])
            d = json.loads(p.read_text())
            d['archive_sha256'] = d['archive_sha256'][:-1]
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                self.execute(config, cli)
            self.assertEqual(cli.calls, [])

    def test_concurrent_destination_is_preserved_and_owned_runtime_released(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, cli = self.fixture(root)
            cli.race_destination = config['destination']
            with self.assertRaises(FileExistsError):
                self.execute(config, cli)
            self.assertEqual(Path(config['destination']).read_bytes(), b'other job output')
            self.assertEqual(cli.calls[-1][1][0], 'stop')

    def test_cleanup_failure_is_recorded_without_claiming_full_completion(self):
        with tempfile.TemporaryDirectory() as root:
            config, payload, cli = self.fixture(root)
            cli.stop_returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'cleanup'):
                self.execute(config, cli)
            self.assertEqual(Path(config['destination']).read_bytes(), payload)
            state = json.loads(Path(config['destination'] + '.fetch.json').read_text())
            self.assertEqual(state['stop_returncode'], 1)


if __name__ == '__main__':
    unittest.main()
