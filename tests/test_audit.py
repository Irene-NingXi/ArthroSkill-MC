"""Regressions for procedure identity, fitted architectures and inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
from torch import nn
import yaml

from data.protocol import study_assignments, validate_assignments, require_adapted_backbone
from data.dataset import ArthroscopyDataset
from data.sais_dataset import FORMAT, load_manifest
from evaluate import model_from_checkpoint
from models.sais import LOCK_PATH

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    cfg['model']['appearance'].update(pretrained=False, checkpoint=None)
    return cfg


class AuditTests(unittest.TestCase):
    def test_motion_only_ablation_does_not_read_appearance(self):
        from scripts.run_ablation import AblationModel
        model = AblationModel(config(), 'motion_only').eval()
        with patch.object(model.base.appearance_branch, 'forward', side_effect=AssertionError('Appearance used')):
            outputs = model(None, torch.randn(2, 16, 6), torch.ones(2, 16))
        self.assertEqual(tuple(outputs['cls_logits'].shape), (2, 3))
        self.assertTrue(torch.isfinite(outputs['total_score']).all())

    def test_training_metrics_allow_tensor_only_checkpoint_loading(self):
        from utils.metrics import SkillMetrics
        metrics = SkillMetrics.compute_all(np.array([0, 1, 2]), np.array([0, 2, 2]),
                                           np.array([10., 18., 26.]), np.array([11., 20., 25.]),
                                           np.array([[.8, .1, .1], [.1, .3, .6], [.1, .1, .8]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            torch.save(dict(model_state_dict={'weight': torch.ones(1)}, metrics=metrics), path)
            restored = torch.load(path, weights_only=True)
            self.assertEqual(restored['metrics'], metrics)

    def test_same_counts_cannot_hide_swapped_assignments(self):
        records = [dict(video_id=v, **r) for v, r in study_assignments().items()]
        validate_assignments(records)
        first = next(r for r in records if r['split'] == 'train')
        second = next(r for r in records if r['split'] == 'val')
        first['split'], second['split'] = second['split'], first['split']
        with self.assertRaisesRegex(ValueError, 'assignment mismatch'):
            validate_assignments(records)

    def test_scores_grades_and_missing_procedures_are_rejected(self):
        original = [dict(video_id=v, **r) for v, r in study_assignments().items()]
        for field, value in [('label_cls', 99), ('total', 0), ('dimensions', [0] * 6), ('centre', 'Unknown')]:
            records = copy.deepcopy(original)
            records[0][field] = value
            with self.assertRaises(ValueError): validate_assignments(records)
        with self.assertRaisesRegex(ValueError, 'coverage'): validate_assignments(original[:-1])

    def test_sais_manifest_checks_actual_study_membership(self):
        lock = json.loads(LOCK_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for video_id, row in study_assignments().items():
                path = root / f'{video_id}.pt'
                path.touch()
                records.append(dict(video_id=video_id, split=row['split'], label_cls=row['label_cls'],
                                    path=path.name, source_frame_indices=list(range(16))))
            manifest = dict(format=FORMAT, provenance=dict(flow_step=15, sais_commit=lock['commit']), clips=records)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            load_manifest(root)
            first = next(r for r in records if r['split'] == 'train')
            second = next(r for r in records if r['split'] == 'test')
            first['split'], second['split'] = second['split'], first['split']
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'assignment mismatch'): load_manifest(root)

    def test_adaptation_is_required_for_internal_study_training(self):
        with self.assertRaisesRegex(ValueError, 'HeiChole'): require_adapted_backbone(config())

    def test_train_partition_inference_disables_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / 'train/clips'
            clips.mkdir(parents=True)
            frames = torch.randn(3, 16, 8, 8)
            torch.save(dict(frames=frames, motion=torch.zeros(16, 6), detection_conf=torch.ones(16),
                            label_cls=0, label_reg=torch.tensor([2.] * 6 + [12.]),
                            score_units='grs_points', video_id='fixture'), clips / 'clip.pt')
            dataset = ArthroscopyDataset(root, 'train', config=config(), augment=False)
            with patch.object(dataset, '_augment', side_effect=AssertionError('Unexpected augmentation')):
                torch.testing.assert_close(dataset[0]['appearance'], frames)

    def test_evaluation_restores_checkpoint_architecture_and_config(self):
        class Fixture(nn.Module):
            def __init__(self, cfg, *args, **kwargs):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                self.cfg = cfg
        cfg = config()
        cfg['training']['batch_size'] = 7
        for field, value, target in [(None, None, 'evaluate.ArthroSkillMC'),
                                     ('variant', 'unmasked', 'scripts.run_ablation.AblationModel'),
                                     ('model_type', 'resnet3d', 'models.baselines.ResNet3DBaseline'),
                                     ('model_type', 'swin', 'models.baselines.VideoSwinOnly')]:
            checkpoint = dict(config=cfg, model_state_dict={'weight': torch.tensor([3.])})
            if field: checkpoint[field] = value
            with patch(target, Fixture):
                model, restored = model_from_checkpoint(checkpoint, config())
            self.assertEqual(model.weight.item(), 3.)
            self.assertEqual(restored['training']['batch_size'], 7)
            self.assertFalse(restored['model']['appearance']['pretrained'])
            self.assertIsNone(restored['model']['appearance']['checkpoint'])
        self.assertEqual(cfg['training']['batch_size'], 7)


if __name__ == '__main__': unittest.main()
