"""Synthetic file→QC→release E2E; optional real pinned-backbone CPU loops."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.paired_stain import data, models


def fixture(root, *, approved=True):
    root.mkdir()
    rows = []
    for index in range(3):
        folder = root/str(index)
        folder.mkdir()
        rgb = np.random.default_rng(index).integers(20, 230, (32, 32, 3), dtype=np.uint8)
        he = (rgb.astype(float)*[.9, .6, .8]).astype('uint8')
        arrays = {'THO': rgb, 'HE': he, 'capture': np.full((32, 32), 255, np.uint8),
                  'supervision': np.full((32, 32), 255 if approved else 0, np.uint8)}
        row = {'id': 'synthetic-'+str(index), 'patient_id': 'synthetic-'+str(index),
               'split': 'train' if index < 2 else 'validation',
               'THO_native_box_xyxy': [0, 0, 32, 32], 'HE_to_THO_3x3': np.eye(3).tolist(),
               'qc': {'status': 'approved', 'reviewer': 'synthetic_fixture_only',
                      'review_evidence': 'synthetic_known_transform', 'mask_reviewed': True,
                      'anatomical_identity_confirmed': True, 'supervision_scale': 'fine_spatial_correspondence',
                      'approved_branches': ['B', 'C'], 'fit_ids': ['F1', 'F2'],
                      'holdout_ids': ['V1', 'V2', 'V3'], 'heldout_regions_cover_patch': True,
                      'heldout_max_error_px': 0}}
        for name, array in arrays.items():
            path = folder/(name+'.png')
            Image.fromarray(array).save(path)
            row[name] = {'file': str(path.relative_to(root)), 'sha256': data.sha256(path)}
        row['source_evidence'] = {domain: {'patient_id': row['patient_id'], 'split': row['split'],
                                         'domain': domain,
                                         'original_mosaic_sha256': row[domain]['sha256'],
                                         'original_capture_mask_sha256': row['capture']['sha256'],
                                         'raw_crop_rgb_sha256': row[domain]['sha256']}
                                  for domain in ('THO', 'HE')}
        rows.append(row)
    doc = {'schema': 'paired-stain-v1', 'kind': 'approved_release' if approved else 'candidates',
           'trainingAccepted': approved, 'synthetic_fixture': True, 'records': rows}
    save(root, doc)
    return doc


def save(root, doc):
    (root/'manifest.json').write_text(json.dumps(doc))


class PairedReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base/'dataset'
        self.doc = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_release_e2e_explicit_review_and_unchanged_pixels(self):
        self.doc['kind'] = 'candidates'
        self.doc['trainingAccepted'] = False
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'Candidate'):
            data.validate(self.root)
        reviews = {'records': [{'id': r['id'], 'qc': r['qc'],
                               'supervision': {'file': 'dataset/'+r['supervision']['file'],
                                               'sha256': r['supervision']['sha256']}} for r in self.doc['records']]}
        path = self.base/'reviews.json'
        path.write_text(json.dumps(reviews))
        output = self.base/'approved'
        result = data.publish(self.root, path, output)
        self.assertEqual(len(data.validate(output)['records']), 3)
        self.assertEqual(result['records'][0]['THO']['sha256'], self.doc['records'][0]['THO']['sha256'])
        with self.assertRaises(FileExistsError):
            data.publish(self.root, path, output)

    def test_split_leakage_and_test_rejected(self):
        self.doc['records'][2]['patient_id'] = self.doc['records'][0]['patient_id']
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'split leakage'):
            data.validate(self.root)
        self.doc['records'][2]['split'] = 'test'
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'forbidden test'):
            data.validate(self.root)

    def test_pixel_leakage_even_when_file_names_differ(self):
        self.doc['records'][2]['THO'] = copy.deepcopy(self.doc['records'][0]['THO'])
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'Identical pixels'):
            data.validate(self.root)

    def test_original_source_provenance_required(self):
        del self.doc['records'][0]['source_evidence']
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            data.validate(self.root)

    def test_image_checksum_and_containment(self):
        row = self.doc['records'][0]
        row['THO']['sha256'] = '0'*64
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            data.validate(self.root)
        row['THO']['file'] = '../escape.png'
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'contained'):
            data.validate(self.root)

    def test_wrong_matrix_and_native_size(self):
        row = self.doc['records'][0]
        row['HE_to_THO_3x3'] = np.eye(3)[:2].tolist()
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, '3x3'):
            data.validate(self.root)
        row['HE_to_THO_3x3'] = np.eye(3).tolist()
        row['THO_native_box_xyxy'][2] += 1
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'Native size'):
            data.validate(self.root)

    def test_capture_is_not_supervision(self):
        row = self.doc['records'][0]
        mask = self.root/row['supervision']['file']
        Image.fromarray(np.zeros((32, 32), np.uint8)).save(mask)
        row['supervision']['sha256'] = data.sha256(mask)
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'fully reviewed'):
            data.validate(self.root)

    def test_supervision_outside_capture_rejected(self):
        row = self.doc['records'][0]
        mask = self.root/row['capture']['file']
        Image.fromarray(np.zeros((32, 32), np.uint8)).save(mask)
        row['capture']['sha256'] = data.sha256(mask)
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'outside observed'):
            data.validate(self.root)

    def test_coarse_qc_and_fitting_holdout_reuse_rejected(self):
        qc = self.doc['records'][0]['qc']
        qc['supervision_scale'] = 'large_lumens_only'
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'Fine-scale'):
            data.validate(self.root)
        qc['supervision_scale'] = 'fine_spatial_correspondence'
        qc['holdout_ids'][0] = 'F1'
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'Independent'):
            data.validate(self.root)

    def test_train_c_refuses_missing_codec_before_torch(self):
        from tools.paired_stain.train import required_codec
        with self.assertRaisesRegex(ValueError, 'VAE reconstruction'):
            required_codec(self.doc, None)

    def test_codec_review_coverage_and_reconstruction_integrity(self):
        from tools.paired_stain.train import required_codec
        report = {'VAE_revision': models.VAE_REV, 'synthetic_codec': False, 'records': []}
        for row in self.doc['records']:
            for domain in ('THO', 'HE'):
                report['records'].append({'input_sha256': row[domain]['sha256'],
                                           'reconstruction_file': 'dataset/'+row[domain]['file'],
                                           'reconstruction_sha256': row[domain]['sha256']})
        report_file = self.base/'report.json'
        report_file.write_text(json.dumps(report))
        approval = {'approved': True, 'reviewer': 'synthetic_test', 'review_evidence': 'fixture',
                    'report_file': report_file.name, 'report_sha256': data.sha256(report_file)}
        path = self.base/'approval.json'
        path.write_text(json.dumps(approval))
        self.assertEqual(required_codec(self.doc, path)['approval_sha256'], data.sha256(path))
        report['records'].pop()
        report_file.write_text(json.dumps(report))
        approval['report_sha256'] = data.sha256(report_file)
        path.write_text(json.dumps(approval))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            required_codec(self.doc, path)

    def test_header_bound_before_decode(self):
        row = self.doc['records'][0]
        path = self.root/row['THO']['file']
        Image.new('RGB', (1024, 1024)).save(path)
        row['THO']['sha256'] = data.sha256(path)
        save(self.root, self.doc)
        with self.assertRaisesRegex(ValueError, 'type/size'):
            data.validate(self.root)

    def test_candidate_stage_can_be_inspected_but_not_train(self):
        other = self.base/'candidate'
        fixture(other, approved=False)
        self.assertEqual(len(data.validate(other, require_release=False)['records']), 3)
        with self.assertRaisesRegex(ValueError, 'Candidate'):
            data.validate(other)

    def test_bootstrap_uses_isolated_python_and_matching_wheel_index(self):
        from unittest.mock import patch
        from tools.paired_stain.bootstrap_colab import bootstrap
        for accelerator in ('cpu', 'cuda'):
            with patch('tools.paired_stain.bootstrap_colab.subprocess.run') as run:
                bootstrap(self.base/('environment-'+accelerator), accelerator)
            commands = [c.args[0] for c in run.call_args_list]
            self.assertTrue(any('venv' in c and '3.12' in c for c in commands))
            self.assertFalse(any('--system-site-packages' in c for c in commands))
            index = 'https://download.pytorch.org/whl/'+('cpu' if accelerator == 'cpu' else 'cu121')
            self.assertTrue(any(index in c and 'torch==2.5.1' in c for c in commands))


@unittest.skipUnless(os.environ.get('CTUP_TEST_TORCH') == '1', 'optional pinned-backbone CPU E2E')
class PairedTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.assets = cls.base/'assets'
        for branch in ('B', 'C'):
            models.prepare_assets(cls.assets, branch)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def arguments(self, root, output, branch, steps, resume=None):
        return argparse.Namespace(dataset=str(root), output=str(output), branch=branch,
                                  assets=str(self.assets), vae_cache=None, codec_approval=None,
                                  steps=steps, seconds=60, accumulation=1, lr=.0001,
                                  seed=2026, technical_smoke=True, resume=resume)

    def test_both_upstream_backbones_file_to_train_and_eval(self):
        from tools.paired_stain.train import train
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            base = Path(temp)
            root = base/'data'
            fixture(root)
            for branch in ('B', 'C'):
                result = train(self.arguments(root, base/branch, branch, 2))
                self.assertEqual(result['updates_this_run'], 2)
                self.assertGreater(result['validation_predictions'], 0)
                self.assertTrue(np.isfinite(result['losses']).all())
                self.assertFalse(result['scientific_training_result'])

    def test_checkpoint_resume_matches_uninterrupted_updates(self):
        import torch
        from tools.paired_stain.train import train
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            base = Path(temp)
            root = base/'data'
            fixture(root)
            for branch in ('B', 'C'):
                whole, first, rest = [base/(branch+n) for n in ('whole', 'first', 'rest')]
                train(self.arguments(root, whole, branch, 3))
                train(self.arguments(root, first, branch, 1))
                train(self.arguments(root, rest, branch, 3, str(first/'checkpoint.pt')))
                a = torch.load(whole/'checkpoint.pt', weights_only=False)
                b = torch.load(rest/'checkpoint.pt', weights_only=False)
                self.assertEqual(a['step'], b['step'])
                self.assertTrue(all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']))

    def test_time_capped_training_reserves_heldout_evaluation(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from tools.paired_stain.train import train
        ticks = iter(i*.03 for i in range(1000))
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            base = Path(temp)
            root = base/'data'
            fixture(root)
            args = self.arguments(root, base/'timed', 'B', 8)
            args.seconds = 1
            with patch('tools.paired_stain.train.time', SimpleNamespace(monotonic=lambda: next(ticks))):
                run = train(args)
            self.assertLess(run['final_step'], 8)
            self.assertGreaterEqual(run['validation_predictions'], 1)

    def test_dit_condition_receives_gradient_and_changes_output(self):
        import torch
        torch.set_num_threads(1)
        model = models.build('C', 32, self.assets, tiny=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        source, noisy = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)
        t, target = torch.tensor([300]), torch.randn(1, 4, 4, 4)
        for _ in range(3):
            optimizer.zero_grad()
            loss = (models.predict_noise(model, noisy, source, t)-target).square().mean()
            loss.backward()
            optimizer.step()
        self.assertGreater(float(model.x_embedder.proj.weight.grad[:, 4:].abs().sum()), 0)
        with torch.no_grad():
            original = models.predict_noise(model, noisy, source, t)
            ablated = models.predict_noise(model, noisy, torch.zeros_like(source), t)
        self.assertFalse(torch.equal(original, ablated))


if __name__ == '__main__':
    unittest.main()
