"""Contract and actual-upstream integration checks for RGB/flow SAIS."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import yaml
from torch import nn
from data.sais_dataset import FORMAT, SAISFeatureDataset, collate_sais, load_manifest
from models.sais import SAISBaseline, TupleTransformerEncoder, LOCK_PATH, load_upstream
from utils.sais_features import encode_clip, flow_pairs
from utils.sais_training import aggregate_sais, train_sais, evaluate_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = yaml.safe_load((ROOT/'configs/config.yaml').read_text())
    cfg['training'].update(num_workers=0, batch_size=2, device='cpu')
    return cfg


def fixture_cache(root):
    lock = json.loads(LOCK_PATH.read_text())
    manifest = dict(format=FORMAT, provenance=dict(sais_commit=lock['commit'],
                    source_sha256=lock['source_sha256'], flow_step=15, crop_fraction=.8), clips=[])
    generator = torch.Generator().manual_seed(31)
    for split, grade in [('train', 0), ('val', 1), ('test', 2)]:
        for i in range(2):
            path = root/split/'clips'/f'{split}_{i}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
            indices = list(range(i*16, (i+1)*16))
            pairs = [[indices[0], indices[-1]]]
            payload = dict(format=FORMAT, rgb_features=torch.randn(16, 384, generator=generator),
                           flow_features=torch.randn(1, 384, generator=generator),
                           video_id=split, label_cls=grade, source_frame_indices=indices,
                           flow_frame_pairs=pairs)
            torch.save(payload, path)
            manifest['clips'].append(dict(path=path.relative_to(root).as_posix(), split=split,
                                           video_id=split, label_cls=grade,
                                           source_frame_indices=indices,
                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root/'manifest.json').write_text(json.dumps(manifest))
    return manifest


class SAISDataTests(unittest.TestCase):
    def test_pair_alignment(self):
        from PIL import Image
        indices = list(range(10, 42, 2)); frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(16)]; calls = []
        def raft(a, b):
            calls.append((int(a[0, 0, 0]), int(b[0, 0, 0]))); return Image.fromarray(a)
        def dino(images):
            return torch.tensor([np.asarray(i).mean() for i in images])[:, None].expand(-1, 384)
        rgb, flow, pairs = encode_clip(frames, indices, dino, raft, step=5)
        self.assertEqual(calls, [(0, 5), (5, 10), (10, 15)])
        self.assertEqual(pairs, [[10, 20], [20, 30], [30, 40]])
        self.assertEqual(tuple(rgb.shape), (16, 384)); self.assertEqual(tuple(flow.shape), (3, 384))
        with self.assertRaises(ValueError): flow_pairs(indices, 16)

    def test_probability_aggregation_is_classification_only(self):
        rows = [dict(video_id='A', true_cls=1, probabilities=[.6, .4, 0]),
                dict(video_id='A', true_cls=1, probabilities=[.1, .9, 0]),
                dict(video_id='B', true_cls=2, probabilities=[0, .1, .9])]
        result = aggregate_sais(rows)
        self.assertEqual(result['metrics']['accuracy'], 1.); self.assertEqual(result['metrics']['n'], 2)
        self.assertNotIn('mae', result['metrics']); self.assertNotIn('pearson_r', result['metrics'])
        with self.assertRaises(ValueError): aggregate_sais(rows+[dict(rows[0], true_cls=0)])

    def test_cache_integrity_pairing_and_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = fixture_cache(root)
            checked = load_manifest(root, require_study_counts=False); dataset = SAISFeatureDataset(root, 'train', checked)
            first, second = dataset[0], dataset[1]
            second['flow_features'] = second['flow_features'].repeat(2, 1)
            batch = collate_sais([first, second])
            self.assertEqual(batch['flow_lengths'].tolist(), [1, 2]); self.assertEqual(batch['rgb_lengths'].tolist(), [16, 16])
            with self.assertRaises(ValueError): load_manifest(root)
            bad = copy.deepcopy(manifest); bad['clips'][-1]['video_id'] = 'train'
            (root/'manifest.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'leakage'): load_manifest(root, False)
            (root/'train'/'clips'/'train_0.pt').write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError, 'checksum'): dataset[0]

    def test_transformer_compatibility_preserves_values_and_global_api(self):
        original_class = nn.TransformerEncoder.forward
        base = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, dropout=0), 2, enable_nested_tensor=False).eval()
        compatible = TupleTransformerEncoder.from_encoder(base).eval()
        x = torch.randn(4, 2, 8); mask = torch.tensor([[False]*4, [False, False, True, True]])
        expected = base(x, src_key_padding_mask=mask); actual, attention = compatible(x, src_key_padding_mask=mask)
        torch.testing.assert_close(actual, expected); self.assertIsNone(attention)
        self.assertIs(nn.TransformerEncoder.forward, original_class); self.assertEqual(set(base.state_dict()), set(compatible.state_dict()))

    def test_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'SAIS/scripts/prepare_model.py'
            path.parent.mkdir(parents=True);path.write_text('raise RuntimeError("must not execute")')
            with self.assertRaisesRegex(ValueError, 'pinned commit'):
                load_upstream(directory, 'SAIS/scripts/prepare_model.py')

    def test_raft_io_contract_for_both_released_interfaces(self):
        import types
        from unittest.mock import patch
        from utils.sais_features import PTLFlowRAFT
        class Model(nn.Module):
            output_stride=8
            def forward(self, inputs):
                h,w=inputs['images'].shape[-2:]
                return {'flows':torch.zeros(1,1,2,h,w)}
        seen=[]
        class LegacyIO:
            def __init__(self,model,input_size,cuda=False): pass
            def prepare_inputs(self,images):
                seen.append(images[0][0,0].tolist())
                return {'images':torch.zeros(1,2,3,*images[0].shape[:2])}
            def unpad_and_unscale(self,prediction): return prediction
        class ModernIO:
            def __init__(self,output_stride,input_size,cuda=False): pass
            prepare_inputs=LegacyIO.prepare_inputs
            unscale=LegacyIO.unpad_and_unscale
        modules={name:types.ModuleType(name) for name in ['ptlflow','ptlflow.utils','ptlflow.utils.io_adapter','ptlflow.utils.flow_utils']}
        modules['ptlflow'].get_model=lambda name:Model()
        modules['ptlflow.utils.flow_utils'].flow_to_rgb=lambda flows:torch.zeros(1,1,3,*flows.shape[-2:])
        with tempfile.TemporaryDirectory() as directory:
            weights=Path(directory)/'fixture.ckpt';torch.save({'state_dict':{}},weights)
            for interface in [LegacyIO,ModernIO]:
                modules['ptlflow.utils.io_adapter'].IOAdapter=interface
                with patch.dict('sys.modules',modules), patch('importlib.metadata.version',return_value='fixture'):
                    raft=PTLFlowRAFT(weights)
                    image=np.full((8,8,3),[10,20,30],dtype=np.uint8)
                    output=raft(image,image)
                    self.assertEqual(output.size,(8,8));self.assertEqual(output.mode,'RGB')
        self.assertEqual(seen,[[30,20,10],[30,20,10]])


class OfficialSAISIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('SAIS_ROOT', str(ROOT/'external'/'SAIS')))
        if not (cls.upstream/'SAIS/scripts/prepare_model.py').is_file():
            raise unittest.SkipTest('Set SAIS_ROOT or run scripts/setup_sais.py for actual-upstream integration tests')
        cls.model = SAISBaseline(config(), upstream_root=cls.upstream).eval()

    def test_loaded_official_dino_is_frozen(self):
        from PIL import Image
        from utils.sais_features import FrozenDINO
        import utils as project_utils
        import sys
        module=load_upstream(self.upstream,'SAIS/scripts/dino-main/vision_transformer.py')
        self.assertIs(sys.modules['utils'], project_utils)
        with tempfile.TemporaryDirectory() as directory:
            weights=Path(directory)/'synthetic_fixture.pth'
            torch.save(module.vit_small(patch_size=16,num_classes=0).state_dict(),weights)
            encoder=FrozenDINO(self.upstream,weights)
            result=encoder([Image.new('RGB',(640,480),(120,90,60))])
            self.assertEqual(tuple(result.shape),(1,384))
            self.assertFalse(result.requires_grad)
            self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))
            self.assertTrue(torch.isfinite(result).all())

    def test_both_modalities_and_prototypes_have_gradients(self):
        self.model.zero_grad(set_to_none=True)
        rgb = torch.randn(2, 16, 384, requires_grad=True); flow = torch.randn(2, 3, 384, requires_grad=True)
        original = rgb.detach().clone(); outputs = self.model(rgb, flow)
        self.model.compute_loss(outputs, torch.tensor([0, 2])).backward()
        self.assertGreater(rgb.grad.abs().sum().item(), 0); self.assertGreater(flow.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.prototypes['0'].grad.abs().sum().item(), 0)
        torch.testing.assert_close(rgb.detach(), original)
        with torch.no_grad(): changed = self.model(rgb, torch.zeros_like(flow))['cls_logits']
        self.assertFalse(torch.allclose(outputs['cls_logits'], changed))

    def test_prototype_loss_matches_released_implementation(self):
        misc = load_upstream(self.upstream, 'SAIS/scripts/prepare_miscellaneous.py')
        with torch.no_grad():
            outputs = self.model(torch.randn(2, 16, 384), torch.randn(2, 2, 384)); labels = torch.tensor([0, 2])
            expected = misc.calcNCELoss(0, outputs['embedding'], labels, ['P1', 'P2'], self.model.prototypes, ['ArthroSkill']*2)
            actual = self.model.compute_loss(outputs, labels)
        torch.testing.assert_close(actual, expected)

    def test_padding_cannot_change_a_clip_prediction(self):
        rgb = torch.randn(1, 16, 384); flow = torch.randn(1, 2, 384)
        with torch.no_grad():
            expected = self.model(rgb, flow)['cls_logits']; padded = torch.cat([flow, torch.randn(1, 2, 384)*100], dim=1)
            actual = self.model(rgb, padded, flow_lengths=[2])['cls_logits']
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_training_checkpoint_and_test_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root/'features'; data.mkdir(); fixture_cache(data)
            report = train_sais(config(), data, root/'run', upstream_root=self.upstream, epochs=2, require_study_counts=False)
            self.assertEqual([r['learning_rate'] for r in report['history']], [.0001, .0001])
            accuracies = [r['validation']['accuracy'] for r in report['history']]
            self.assertEqual(report['best_epoch'], int(np.argmax(accuracies))+1)
            result = evaluate_checkpoint(root/'run'/'sais_best_model.pth', data, root/'test', upstream_root=self.upstream, require_study_counts=False)
            self.assertEqual(result['split'], 'test'); self.assertEqual(result['metrics']['n'], 1); self.assertNotIn('mae', result['metrics'])
            self.assertEqual(len(result['per_clip']), 2)
            self.assertEqual(result['per_clip'][0]['source_frame_indices'], list(range(16)))
            self.assertEqual(result['per_clip'][0]['flow_frame_pairs'], [[0, 15]])
            self.assertEqual(len(result['per_clip'][0]['probabilities']), 3)
            checkpoint = torch.load(root/'run'/'sais_best_model.pth', weights_only=True)
            self.assertIn('prototypes.0', checkpoint['model_state_dict']); self.assertEqual(checkpoint['provenance']['scheduler'], 'constant')
            with self.assertRaisesRegex(FileExistsError, 'empty SAIS run directory'):
                train_sais(config(), data, root/'run', upstream_root=self.upstream, epochs=1, require_study_counts=False)
            with (data/'manifest.json').open('a') as stream: stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'differs from the selected training run'):
                evaluate_checkpoint(root/'run'/'sais_best_model.pth', data, root/'test', upstream_root=self.upstream, require_study_counts=False)


if __name__ == '__main__': unittest.main()
