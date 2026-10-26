"""Image-only release E2E and optional real-backbone nonaligned loop checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.nonaligned_stain import data


def fixture(root):
    root.mkdir()
    rows = []
    for patient in range(3):
        for domain in ('THO', 'HE'):
            rgb = np.random.default_rng(patient*10+int(domain == 'HE')).integers(10, 240, (32, 32, 3), dtype=np.uint8)
            path = root/f'{patient}_{domain}.png'
            Image.fromarray(rgb).save(path)
            rows.append({'patient_id': 'fixture-'+str(patient), 'split': 'train' if patient < 2 else 'validation',
                         'domain': domain, 'file': path.name, 'sha256': data.sha256(path),
                         'pixel_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                         'source_sha256': data.sha256(path), 'box': [10, 20, 32, 32],
                         'native_laplacian_variance': 1., 'structured_fraction': 1.,
                         'coordinate_system': 'original_sensor_pixels'})
    doc = {'schema': 'nonaligned-stain-v1', 'paired': False, 'synthetic_fixture': True,
           'ready_for_unpaired_pilot': True, 'records': rows}
    (root/'manifest.json').write_text(json.dumps(doc))
    return doc


class NonalignedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base/'source'
        self.doc = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.root/'manifest.json').write_text(json.dumps(self.doc))

    def test_source_to_unpaired_release_keeps_native_pixels(self):
        out = self.base/'release'
        result = data.publish(self.root, out)
        self.assertEqual(len(data.validate(out)['records']), 6)
        self.assertFalse(result['paired'])
        self.assertFalse(result['trainingAccepted'])
        self.assertEqual(result['records'][0]['sha256'], data.sha256(out/result['records'][0]['file']))
        with self.assertRaises(FileExistsError):
            data.publish(self.root, out)

    def test_test_and_patient_leakage_rejected(self):
        self.doc['records'][-1]['patient_id'] = self.doc['records'][0]['patient_id']
        self.save()
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            data.validate(self.root)
        self.doc['records'][-1]['split'] = 'test'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Forbidden test'):
            data.validate(self.root)

    def test_corruption_and_mosaic_input_rejected(self):
        self.doc['records'][0]['sha256'] = '0'*64
        self.save()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            data.validate(self.root)
        self.doc['records'][0]['coordinate_system'] = 'mosaic'
        self.save()
        with self.assertRaisesRegex(ValueError, 'sensor'):
            data.validate(self.root)

    def test_missing_domain_and_false_paired_release_rejected(self):
        self.doc['records'] = [r for r in self.doc['records'] if r['domain'] != 'HE']
        self.save()
        with self.assertRaisesRegex(ValueError, 'domains'):
            data.validate(self.root)
        self.doc['paired'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'unpaired release'):
            data.validate(self.root)


@unittest.skipUnless(os.environ.get('CTUP_TEST_TORCH') == '1', 'optional nonaligned Torch E2E')
class NonalignedTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.paired_stain import models
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.assets = cls.base/'assets'
        for branch in ('B', 'C'):
            models.prepare_assets(cls.assets, branch)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, root, output, branch, steps=2, resume=None):
        return argparse.Namespace(dataset=str(root), output=str(output), branch=branch,
                                  assets=str(self.assets), vae_cache=None, steps=steps,
                                  seconds=60, accumulation=1, resume=resume, technical_smoke=True)

    def test_contextual_matching_ignores_target_permutation_and_backprops(self):
        import torch
        from tools.nonaligned_stain.objectives import contextual, self_similarity
        torch.manual_seed(2026)
        x, y = torch.randn(1, 8, 8, 8, requires_grad=True), torch.randn(1, 8, 8, 8)
        permutation = torch.randperm(64)
        shuffled = y.flatten(2)[:, :, permutation].reshape_as(y)
        loss, other = contextual(x, y), contextual(x, shuffled)
        self.assertTrue(torch.allclose(loss, other, atol=1e-6))
        self.assertLess(float(contextual(x, x)), float(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(float(self_similarity(x, x)), 0., places=6)

    def test_finite_ddim_roundtrip_with_constant_epsilon_oracle(self):
        import torch
        from tools.nonaligned_stain.bridge import transport
        class Constant(torch.nn.Module):
            def forward(self, x, t, labels):
                return torch.ones_like(x)*.12
        x = torch.randn(1, 4, 4, 4)
        encoded = transport(Constant(), x, reverse=True, steps=10)
        reconstructed = transport(Constant(), encoded, steps=10)
        self.assertTrue(torch.allclose(x, reconstructed, atol=2e-5))
        with self.assertRaises(ValueError):
            transport(Constant(), x, steps=1000)

    def test_two_file_to_train_loops_have_actual_optimizer_updates(self):
        from tools.nonaligned_stain.train import train
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            base = Path(temp)
            root = base/'data'
            fixture(root)
            for branch in ('B2', 'C2'):
                result = train(self.args(root, base/branch, branch))
                self.assertEqual(result['updates_this_run'], 2)
                self.assertGreater(result['final_validation_predictions'], 0)
                self.assertFalse(result['paired_ground_truth_used'])
                self.assertEqual(result['config']['precision'], 'fp32')
                self.assertEqual(result['actual_optimizer_updates_this_run'], 2 if branch == 'B2' else 4)

    def test_c2_can_use_a_patient_with_only_source_domain(self):
        from tools.nonaligned_stain.train import train
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            base = Path(temp)
            root = base/'data'
            doc = fixture(root)
            image = np.random.default_rng(77).integers(10, 240, (32, 32, 3), dtype=np.uint8)
            path = root/'source_only.png'
            Image.fromarray(image).save(path)
            row = dict(doc['records'][0], patient_id='source-only', file=path.name,
                       sha256=data.sha256(path), source_sha256=data.sha256(path),
                       pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            doc['records'].append(row)
            (root/'manifest.json').write_text(json.dumps(doc))
            result = train(self.args(root, base/'C2', 'C2', 1))
            self.assertEqual(result['actual_optimizer_updates_this_run'], 2)
            with self.assertRaisesRegex(ValueError, 'same-patient'):
                train(self.args(root, base/'B2', 'B2', 1))

    def test_resume_preserves_live_and_ema_models(self):
        import torch
        from tools.nonaligned_stain.train import train
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            base = Path(temp)
            root = base/'data'
            fixture(root)
            for branch in ('B2', 'C2'):
                a, b, c = [base/(branch+n) for n in ('whole', 'first', 'resumed')]
                train(self.args(root, a, branch, 3))
                train(self.args(root, b, branch, 1))
                train(self.args(root, c, branch, 3, str(b/'checkpoint.pt')))
                x = torch.load(a/'checkpoint.pt', weights_only=False)
                y = torch.load(c/'checkpoint.pt', weights_only=False)
                for key in ('models', 'ema'):
                    self.assertTrue(all(torch.equal(x[key][n][k], y[key][n][k]) for n in x[key] for k in x[key][n]))


if __name__ == '__main__':
    unittest.main()
