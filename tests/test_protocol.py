"""Focused regressions for consequential protocol errors."""
import unittest
import numpy as np
import torch
from models.fusion import OcclusionAwareFusion
from models.motion_branch import MotionEncoder
from models.appearance_branch import AppearanceBranch
from models.heads import RegressionHead
from utils.inference import aggregate_clips


class ProtocolTests(unittest.TestCase):
    def test_mask_and_fallback_with_gradients(self):
        torch.manual_seed(12)
        module=OcclusionAwareFusion(appearance_dim=8,motion_dim=8,num_heads=2,dropout=0).eval()
        app=torch.randn(3,8,requires_grad=True)
        motion=torch.randn(3,4,8,requires_grad=True)
        mask=torch.tensor([[1,0,1,0],[0,0,0,0],[1,1,1,1]])
        out,weights=module(app,motion,mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(weights[0,0,1].item(),0)
        self.assertEqual(weights[0,0,3].item(),0)
        self.assertEqual(weights[1].abs().sum().item(),0)
        self.assertAlmostEqual(weights[0].sum().item(),1,places=6)
        residual=module.norm1(module.appearance_residual(app[1:2]))
        expected=module.norm2(residual+module.ffn(residual))
        torch.testing.assert_close(out[1:2],expected)
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(app.grad).all())
        self.assertTrue(torch.isfinite(motion.grad).all())
        self.assertEqual(motion.grad[1].abs().sum().item(),0)

    def test_probability_aggregation_and_procedure_weighting(self):
        rows=[dict(video_id='A',true_cls=1,true_reg=20.,pred_reg=18.,probabilities=[.51,.49,0.]),
              dict(video_id='A',true_cls=1,true_reg=20.,pred_reg=22.,probabilities=[.1,.9,0.]),
              dict(video_id='B',true_cls=2,true_reg=26.,pred_reg=25.,probabilities=[.1,.2,.7])]
        result=aggregate_clips(rows)
        self.assertEqual(result['metrics']['n'],2)
        self.assertEqual(result['metrics']['accuracy'],1)
        self.assertEqual(result['per_procedure'][0]['pred_reg'],20.)
        self.assertEqual(result['metrics']['mae'],.5)
        with self.assertRaises(ValueError):
            aggregate_clips(rows+[dict(rows[0],true_cls=0)])

    def test_motion_recurrent_parameters_receive_gradient(self):
        encoder=MotionEncoder(cnn_channels=[8,8,8],lstm_hidden=4,output_dim=8,dropout=0)
        tokens=encoder(torch.randn(2,16,6),return_sequence=True)
        self.assertEqual(tuple(tokens.shape),(2,16,8))
        tokens.square().mean().backward()
        self.assertGreater(encoder.lstm.weight_ih_l0.grad.abs().sum().item(),0)

    def test_regression_is_unbounded(self):
        reg=RegressionHead(input_dim=8,hidden_dim=16,dropout=0)
        with torch.no_grad():
            reg.total_regressor.weight.zero_();reg.total_regressor.bias.fill_(31)
        _,total=reg(torch.randn(2,8))
        torch.testing.assert_close(total,torch.full((2,1),31.))

    def test_backbone_has_temporal_attention(self):
        model=AppearanceBranch(pretrained=False).eval()
        self.assertEqual(model.backbone.patch_embed.proj.kernel_size[0],2)
        with torch.no_grad():
            result=model(torch.randn(1,3,4,32,32))
        self.assertEqual(tuple(result.shape),(1,768))
        self.assertTrue(torch.isfinite(result).all())

    def test_full_model_backward_uses_recurrent_motion(self):
        import yaml
        from models import ArthroSkillMC
        from pathlib import Path
        cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/config.yaml').read_text())
        cfg['model']['appearance'].update(pretrained=False,checkpoint=None)
        model=ArthroSkillMC(cfg).eval()
        appearance=torch.randn(2,3,4,32,32)
        motion=torch.randn(2,4,6)
        confidence=torch.tensor([[1.,0.,1.,1.],[0.,0.,0.,0.]])
        out=model(appearance,motion,confidence)
        targets=torch.tensor([[3.,3.,3.,3.,3.,3.,18.],[4.,4.,4.,4.,4.,4.,24.]])
        loss=model.compute_loss(out,torch.tensor([1,2]),targets)['total']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.motion_branch.encoder.lstm.weight_ih_l0.grad.abs().sum().item(),0)

    def test_statistics_adjustment_and_swaps(self):
        from scripts.analyze_predictions import holm, swaps_p
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        self.assertEqual(swaps_p(np.zeros(4),4,np.random.default_rng(1),100),1.)


if __name__=='__main__': unittest.main()
