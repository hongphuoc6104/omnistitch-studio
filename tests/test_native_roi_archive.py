"""Real tiled TIFF→file child→Drive-like archive→verified download lifecycle.

CLI transport is simulated. Pixel reading, packaging, child execution, geometry,
checksums and cleanup are real; these tests do not certify Google OAuth.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import tifffile

from backend.serial_pairing import transform_points
from tools.colab_fetch import digest
from tools.colab_native import bundle, run_account
from tools.native_roi_archive import collect, validate


def fixture(root):
    root = Path(root)
    groups = {}
    for gid, domain in [('sample_THO', 'THO'), ('sample_HE', 'HE')]:
        folder = root/'drive/MyDrive/sources'/gid
        folder.mkdir(parents=True)
        yy, xx = np.indices((320, 320))
        rgb = np.stack([(xx*3+yy)%256, (yy*5)%256, (xx+yy*2)%256], -1).astype(np.uint8)
        mask = np.where(xx > 60, 255, 0).astype(np.uint8)
        tifffile.imwrite(folder/'mosaic.tif', rgb, tile=(32, 32), photometric='rgb')
        tifffile.imwrite(folder/'valid.tif', mask, tile=(32, 32))
        groups[gid] = {'group_id': gid, 'patient_id': 'fixture', 'split': 'train', 'domain': domain,
            'native_size': [320, 320], 'original_mosaic_sha256': digest(folder/'mosaic.tif'),
            'original_capture_mask_sha256': digest(folder/'valid.tif'),
            'source_reference': {'owner_profile': 'fixture', 'drive_relative_path': 'sources/'+gid}}
    h = np.array([[1., .03, 4.], [0., .99, 5.], [.0001, 0., 1.]])
    box = [30, 40, 158, 168]
    c = transform_points(np.linalg.inv(h), [[30, 40], [158, 40], [158, 168], [30, 168]])
    support = [max(0, int(np.floor(c[:, 0].min()))-2), max(0, int(np.floor(c[:, 1].min()))-2),
               min(320, int(np.ceil(c[:, 0].max()))+3), min(320, int(np.ceil(c[:, 1].max()))+3)]
    order = {'trainingAccepted': False, 'pixel_ground_truth': False, 'groups': groups,
        'candidates': [{'pair_id': 'fixture_pair', 'patient_id': 'fixture', 'split': 'train',
                       'fixed_group': 'sample_THO', 'moving_group': 'sample_HE',
                       'HE_native_to_THO_native': h.tolist(), 'trainingAccepted': False,
                       'ROI_search_windows': [{'search_window_id': 'center', 'fixed_native_box_xyxy': box,
                                               'moving_native_support_xyxy': support}]}]}
    return {'profile': 'fixture', 'workorder': order, 'drive_destination': 'native_fixture'}


class NativeCli:
    def __init__(self, root, *, corrupt=False, email='fixture@example.test', mount_error=False):
        self.root = Path(root)
        self.corrupt, self.email, self.mount_error = corrupt, email, mount_error
        self.calls, self.child = [], None
        self.stop_code, self.race_destination = 0, None

    def mapped(self, path):
        return self.root/path.removeprefix('/content/').lstrip('/')

    def call(self, profile, args, *, interactive=False, required=True):
        self.calls.append(list(args))
        cmd = args[0]
        if cmd == 'whoami':
            return 0, 'Email: '+self.email+'\n'
        if cmd == 'new':
            return 0, 'Session READY'
        if cmd == 'drivemount':
            if self.mount_error:
                raise RuntimeError('fixture mount failure')
            return 0, 'Mounted at /content/drive'
        if cmd == 'upload':
            target = self.mapped(args[2])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args[1], target)
            return 0, ''
        if cmd == 'exec':
            code = Path(args[args.index('--file')+1]).read_text()
            code = code.replace('/content/', str(self.root)+'/')
            code = code.replace("'--run-root',str(root/'result')", "'--run-root',str(root/'result'),'--content-root',"+repr(str(self.root)))
            namespace = {}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(code, '<simulated-notebook-cell-without-file>', 'exec'), namespace)
            if 'child' in namespace:
                self.child = namespace['child']
            return 0, output.getvalue()
        if cmd == 'download':
            source = self.mapped(args[1])
            if not source.is_file():
                return 1, 'file not ready'
            shutil.copyfile(source, args[2])
            if self.corrupt and source.name == 'part_000':
                with Path(args[2]).open('ab') as f:
                    f.write(b'corrupt')
            if self.race_destination and source.name == 'part_000':
                Path(self.race_destination).write_bytes(b'competing output')
            return 0, ''
        if cmd == 'stop':
            if self.child:
                try:
                    self.child.wait(timeout=10)
                except Exception:
                    self.child.kill()
                    self.child.wait()
            return self.stop_code, ''
        raise AssertionError('Unexpected CLI call '+cmd)


class TestNativeRoiArchive(unittest.TestCase):
    def test_integer_native_pixels_capture_and_projective_support_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = fixture(root)
            result = collect(cfg, root/'run', root)
            manifest = json.loads((root/'run/data/manifest.json').read_text())
            self.assertEqual(result['regions'], 2)
            for request in manifest['requests']:
                x1, y1, x2, y2 = request['box_xyxy']
                for original, output in [('mosaic.tif', 'RGB.png'), ('valid.tif', 'capture.png')]:
                    source = root/'drive/MyDrive/sources'/request['group_id']/original
                    expected = tifffile.imread(source)[y1:y2, x1:x2]
                    observed = np.array(Image.open(root/'run/data'/request['request_id']/output))
                    np.testing.assert_array_equal(expected, observed)
                    self.assertEqual(hashlib.sha256(observed.tobytes()).hexdigest(), request['pixels'][output]['raw_pixel_sha256'])
            self.assertFalse(manifest['trainingAccepted'])
            self.assertTrue(result['Drive_backup_verified'])

    def test_reuse_requires_matching_persisted_archive_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = fixture(root)
            first = collect(cfg, root/'run_a', root)
            second = collect(cfg, root/'run_b', root)
            self.assertEqual(first['archive_sha256'], second['archive_sha256'])
            Path(first['drive_destination'], 'native_windows.tar').write_bytes(b'corruption')
            with self.assertRaisesRegex(ValueError, 'persisted archive checksum'):
                collect(cfg, root/'run_c', root)

    def test_wrong_original_digest_never_publishes_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = fixture(root)
            cfg['workorder']['groups']['sample_HE']['original_mosaic_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'Original TIFF checksum'):
                collect(cfg, root/'run', root)
            self.assertEqual(list((root/'drive/MyDrive/native_fixture').rglob('completed.json')), [])

    def test_patient_split_and_test_leakage_are_rejected_before_io(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(temp)
            bad = copy.deepcopy(cfg)
            bad['workorder']['groups']['sample_HE']['patient_id'] = 'different'
            with self.assertRaisesRegex(ValueError, 'Patient'):
                validate(bad)
            bad = copy.deepcopy(cfg)
            bad['workorder']['groups']['sample_HE']['split'] = 'test'
            with self.assertRaisesRegex(ValueError, 'Test'):
                validate(bad)

    def test_source_containment_and_frozen_support_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(temp)
            bad = copy.deepcopy(cfg)
            bad['workorder']['groups']['sample_HE']['source_reference']['drive_relative_path'] = '../secret'
            with self.assertRaisesRegex(ValueError, 'contained'):
                validate(bad)
            bad = copy.deepcopy(cfg)
            bad['workorder']['candidates'][0]['ROI_search_windows'][0]['moving_native_support_xyxy'][0] += 1
            with self.assertRaisesRegex(ValueError, 'frozen 3x3'):
                validate(bad)

    def test_large_support_is_rejected_before_native_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(temp)
            cfg['workorder']['groups']['sample_HE']['native_size'] = [8192, 8192]
            pair = cfg['workorder']['candidates'][0]
            pair['HE_native_to_THO_native'] = [[.02, 0, 0], [0, .02, 0], [0, 0, 1]]
            pair['ROI_search_windows'][0]['moving_native_support_xyxy'] = [1498, 1998, 7903, 8192]
            with self.assertRaises(MemoryError):
                validate(cfg)

    def account_fixture(self, root, **kwargs):
        root = Path(root)
        remote = root/'remote'
        cfg = fixture(remote)
        code = root/'code.tar.gz'
        bundle(code)
        cfg['bundle_sha256'] = digest(code)
        return cfg, code, NativeCli(remote, **kwargs)

    def execute(self, cfg, code, cli, output):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_account(cfg, 'fixture@example.test', output, code, cli, interval=.02, deadline=20)

    def test_file_child_e2e_download_and_only_owned_cpu_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, code, cli = self.account_fixture(temp)
            result = self.execute(cfg, code, cli, Path(temp)/'output')
            self.assertTrue(result['verified'])
            self.assertEqual(digest(result['archive']), result['archive_sha256'])
            self.assertEqual([c for c in cli.calls if c[0] == 'stop'], [['stop', '--session', result['session']]])
            self.assertEqual(cli.child.poll(), 0)
            call_count = len(cli.calls)
            again = self.execute(cfg, code, cli, Path(temp)/'output')
            self.assertEqual(again['archive_sha256'], result['archive_sha256'])
            self.assertEqual(len(cli.calls), call_count)

    def test_corrupt_transfer_releases_runtime_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, code, cli = self.account_fixture(temp, corrupt=True)
            with self.assertRaisesRegex(ValueError, 'part checksum'):
                self.execute(cfg, code, cli, Path(temp)/'output')
            self.assertEqual(list((Path(temp)/'output').rglob('native_windows.tar')), [])
            self.assertEqual(cli.calls[-1][0], 'stop')
            self.assertIsNotNone(cli.child.poll())

    def test_identity_and_mount_failure_do_not_leave_owned_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, code, cli = self.account_fixture(temp, email='wrong@example.test')
            with self.assertRaisesRegex(ValueError, 'email'):
                self.execute(cfg, code, cli, Path(temp)/'output')
            self.assertEqual([x[0] for x in cli.calls], ['whoami'])
            cli.email = 'fixture@example.test'
            cli.mount_error = True
            with self.assertRaisesRegex(RuntimeError, 'mount failure'):
                self.execute(cfg, code, cli, Path(temp)/'output')
            self.assertEqual(cli.calls[-1][0], 'stop')

    def test_new_terminal_invocation_reuses_same_code_and_completed_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, code, cli = self.account_fixture(temp)
            with mock.patch('gzip.time.time', return_value=1000):
                bundle(code)
            cfg['bundle_sha256'] = digest(code)
            first = self.execute(cfg, code, cli, Path(temp)/'output')
            calls = len(cli.calls)
            with mock.patch('gzip.time.time', return_value=2000):
                bundle(code)
            cfg['bundle_sha256'] = digest(code)
            second = self.execute(cfg, code, cli, Path(temp)/'output')
            self.assertEqual(first['job_sha256'], second['job_sha256'])
            self.assertEqual(len(cli.calls), calls)

    def test_interrupted_drive_publication_does_not_expose_partial_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = fixture(root)
            def interrupted(source, target, *_):
                target.write(source.read(16))
                raise OSError('injected Drive write interruption')
            with mock.patch('tools.native_roi_archive.shutil.copyfileobj', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'interruption'):
                    collect(cfg, root/'run', root)
            self.assertEqual(list((root/'drive/MyDrive/native_fixture').rglob('completed.json')), [])
            self.assertEqual(list((root/'drive/MyDrive/native_fixture').rglob('native_windows.tar')), [])

    def test_cleanup_failure_is_separate_from_verified_data(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, code, cli = self.account_fixture(temp)
            cli.stop_code = 1
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                self.execute(cfg, code, cli, Path(temp)/'output')
            statuses = list((Path(temp)/'output').rglob('transfer.json'))
            self.assertEqual(len(statuses), 1)
            state = json.loads(statuses[0].read_text())
            self.assertTrue(state['verified'])
            self.assertEqual(state['stop_returncode'], 1)

    def test_concurrent_local_output_is_preserved_and_cpu_released(self):
        from tools.native_roi_archive import job_digest
        with tempfile.TemporaryDirectory() as temp:
            cfg, code, cli = self.account_fixture(temp)
            output = Path(temp)/'output'
            cli.race_destination = output/'fixture'/job_digest(cfg)[:20]/'native_windows.tar'
            with self.assertRaises(FileExistsError):
                self.execute(cfg, code, cli, output)
            self.assertEqual(cli.race_destination.read_bytes(), b'competing output')
            self.assertEqual(cli.calls[-1][0], 'stop')


if __name__ == '__main__':
    unittest.main()
