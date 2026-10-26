"""Checkpoint delivery survives a lost runtime; optional real CPU model resume."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from test_round2_terminal import FakeRunner
from tools.nonaligned_stain.terminal_round import bundle_code, run


class LostRuntimeRunner(FakeRunner):
    def __init__(self, publication, payload):
        super().__init__()
        self.publication, self.payload, self.polls = publication, payload, 0

    def call(self, profile, args, *, interactive=False, required=True):
        if args[0] == 'sessions':
            self.calls.append(args)
            return 0, 'No active sessions found on server.'
        if args[0] == 'download' and args[1].endswith('/progress.json'):
            self.calls.append(args)
            self.polls += 1
            if self.polls == 1:
                Path(args[2]).write_text(json.dumps({'stage': 'training_B3-0', 'publication': self.publication}))
                return 0, ''
            if self.polls >= 5:
                raise RuntimeError('fixture runtime disappeared')
            return 1, 'fixture transport missing'
        if args[0] == 'download' and 'publication_' in args[1]:
            self.calls.append(args)
            Path(args[2]).write_bytes(self.payload)
            return 0, ''
        return super().call(profile, args, interactive=interactive, required=required)


class CheckpointDeliveryTests(unittest.TestCase):
    def test_phase_scope_and_corrupt_delivery_are_rejected(self):
        from tools.nonaligned_stain import delivery
        from tools.nonaligned_stain.round_worker import selected_phases
        self.assertEqual(selected_phases({'phases':['B3-S']}), ['B3-S'])
        with self.assertRaisesRegex(ValueError, 'phase'):
            selected_phases({'phases':['B3-S','B3-S']})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'results/B3-0').mkdir(parents=True)
            cp = root/'results/B3-0/checkpoint.pt'
            cp.write_bytes(b'complete fixture checkpoint')
            meta = {'step':100, 'bytes':cp.stat().st_size, 'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}
            sidecar = cp.with_name('checkpoint_meta.json')
            sidecar.write_text(json.dumps(meta))
            p = delivery.snapshot(root,'B3-0')
            def bad_fetch(name,target):
                target.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                delivery.receive(p,root/'host',bad_fetch)
            self.assertFalse(list((root/'host').rglob('*.tar.gz')))
            meta['sha256']='0'*64
            sidecar.write_text(json.dumps(meta))
            self.assertIsNone(delivery.snapshot(root,'B3-0'))

    def test_file_controller_retains_checkpoint_before_runtime_disappears(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = {**bundle_code(Path(__file__).resolve().parents[1], root/'code.tar.gz'),
                   'main_training': False, 'max_seconds': 9000, 'profile': 'fixture',
                   'email': 'fixture@example.invalid',
                   'archives': {k: {'filename': k+'.tar.gz', 'bytes': 1} for k in ('B2', 'C2')}}
            (root/'config.json').write_text(json.dumps(cfg))
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w:gz') as tar:
                data = b'fixture checkpoint optimizer RNG'
                member = tarfile.TarInfo('checkpoint.pt')
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            payload = stream.getvalue()
            sha = hashlib.sha256(payload).hexdigest()
            publication = {'id': sha[:32], 'kind': 'checkpoint', 'phase': 'B3-0', 'step': 100,
                           'archive_sha256': sha, 'archive_bytes': len(payload),
                           'checkpoint_sha256': hashlib.sha256(data).hexdigest(),
                           'parts': [{'name': 'publication_'+sha[:32]+'_part_000',
                                      'bytes': len(payload), 'sha256': sha}]}
            runner = LostRuntimeRunner(publication, payload)
            with self.assertRaisesRegex(RuntimeError, 'runtime'):
                run(root/'config.json', runner=runner, interval=0)
            saved = list((root/'deliveries').rglob('*.tar.gz'))
            self.assertEqual(len(saved), 1, 'A published checkpoint must be local before provider loss')
            self.assertEqual(saved[0].read_bytes(), payload)
            self.assertLess(runner.polls, 5, 'A missing assignment must not poll until the two-hour deadline')
            self.assertTrue(any(c[0]=='upload' and 'checkpoint_ack_' in c[2] for c in runner.calls))


@unittest.skipUnless(os.environ.get('CTUP_TEST_TORCH') == '1', 'optional checkpoint model resume E2E')
class CheckpointModelResumeTests(unittest.TestCase):
    def test_native_fixture_checkpoint_snapshot_loss_and_exact_model_resume(self):
        import torch
        from test_nonaligned_stain import fixture
        from tools.paired_stain import models
        from tools.nonaligned_stain.train import train
        from tools.nonaligned_stain import delivery
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root/'dataset'
            fixture(dataset)
            assets = root/'assets'
            models.prepare_assets(assets, 'B')
            def args(output, steps, resume=None):
                return argparse.Namespace(dataset=str(dataset),output=str(output),branch='B2',assets=str(assets),
                                          vae_cache=None,steps=steps,seconds=60,accumulation=1,
                                          resume=resume,technical_smoke=True)
            train(args(root/'whole', 3))
            vm = root/'VM'
            train(args(vm/'results/B3-0', 1))
            publication = delivery.snapshot(vm, 'B3-0')
            self.assertEqual(publication['step'], 1)
            received = delivery.receive(publication, root/'host',
                                        lambda name,path: shutil.copyfile(vm/name,path))
            shutil.rmtree(vm)
            restored = delivery.restore(received, root/'restored')
            self.assertTrue(Path(restored).is_file())
            train(args(root/'resumed', 3, str(restored)))
            whole = torch.load(root/'whole/checkpoint.pt', weights_only=False)
            resumed = torch.load(root/'resumed/checkpoint.pt', weights_only=False)
            self.assertTrue(all(torch.equal(whole['models']['B2'][k], resumed['models']['B2'][k])
                                for k in whole['models']['B2']))
            self.assertEqual(whole['step'], resumed['step'])


if __name__ == '__main__':
    unittest.main()
