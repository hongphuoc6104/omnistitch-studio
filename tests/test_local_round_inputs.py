"""PNG/checkpoint→tar→parts→native release, without Drive or GPU."""
import hashlib
import io
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from test_nonaligned_stain import fixture
from tools.nonaligned_stain import inputs, data


class LocalRoundInputsTests(unittest.TestCase):
    def test_training_phase_transports_only_unchanged_dataset_without_unused_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root/'dataset')
            cfg = inputs.pack(root/'dataset', {}, root/'transport', required_checkpoints=())
            self.assertEqual(cfg['required_checkpoints'], [])
            self.assertFalse(any(name.endswith('.pt') for name in cfg['input_files']))
            dataset = inputs.stage(root/'transport', cfg)
            self.assertEqual(len(data.validate(dataset)['records']), 6)
            self.assertEqual(data.sha256(dataset/'manifest.json'), data.sha256(root/'dataset/manifest.json'))

    def test_original_file_transport_and_deterministic_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'dataset'
            fixture(source)
            checkpoints = {}
            for name in ('B2', 'C2'):
                path = root/(name+'.pt')
                path.write_bytes((name+' trusted fixture').encode())
                checkpoints[name] = {'path': str(path), 'sha256': inputs.digest(path)}
            first = inputs.pack(source, checkpoints, root/'sender')
            second = inputs.pack(source, checkpoints, root/'sender2')
            self.assertEqual(first['inputs_sha256'], second['inputs_sha256'])
            first['archives'] = {k: {'checkpoint_sha256': v['sha256']} for k, v in checkpoints.items()}
            receiver = root/'receiver'
            receiver.mkdir()
            for part in first['input_parts']:
                shutil.copyfile(root/'sender'/part['name'], receiver/part['name'])
            received = inputs.stage(receiver, first)
            self.assertEqual(len(data.validate(received)['records']), 6)
            self.assertEqual(data.sha256(received/'manifest.json'), data.sha256(source/'manifest.json'))
            self.assertEqual((receiver/'inputs/C2.pt').read_bytes(), (root/'C2.pt').read_bytes())
            for row in data.validate(received)['records']:
                self.assertEqual((received/row['file']).read_bytes(), (source/row['file']).read_bytes())
            (receiver/first['input_parts'][0]['name']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                inputs.verify_parts(receiver, first)

    def test_correctly_hashed_archive_still_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stream = io.BytesIO()
            name = 'dataset/../../escape.png'
            with tarfile.open(fileobj=stream, mode='w') as tar:
                info = tarfile.TarInfo(name)
                info.size = 1
                tar.addfile(info, io.BytesIO(b'x'))
            payload = stream.getvalue()
            path = root/'input_part_000'
            path.write_bytes(payload)
            metadata = {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
            cfg = {'inputs_bytes': len(payload), 'inputs_sha256': inputs.digest(path),
                   'input_parts': [{'name': path.name, 'bytes': len(payload), 'sha256': inputs.digest(path)}],
                   'input_files': {k: metadata for k in (name, 'B2.pt', 'C2.pt', 'dataset/manifest.json')}}
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                inputs.stage(root, cfg)
            self.assertFalse((root/'escape.png').exists())
            self.assertFalse((root.parent/'escape.png').exists())


if __name__ == '__main__':
    unittest.main()
