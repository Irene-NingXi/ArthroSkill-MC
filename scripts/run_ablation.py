"""Run the five paper ablations on a fixed train/validation split.

Variants: appearance_only, motion_only, naive_concat, unmasked, full.  The
wrapper preserves the paper's shared training and loss protocol while changing
only the requested information path.  ``naive_concat`` uses a learned
concatenation projection; the other variants reuse the production branches.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader
from data import ArthroscopyDataset, collate_fn
from models import ArthroSkillMC
from data.protocol import validate_fixed_datasets, require_adapted_backbone

class AblationModel(nn.Module):
    def __init__(self, config, variant):
        super().__init__(); self.variant=variant
        if variant not in ['appearance_only', 'motion_only', 'naive_concat', 'unmasked', 'full']:
            raise ValueError('Unknown ablation variant')
        if variant == 'motion_only':
            import copy
            config = copy.deepcopy(config)
            config['model']['appearance'].update(pretrained=False, checkpoint=None)
        self.base=ArthroSkillMC(config); dim=config['model']['motion']['output_dim']; app=config['model']['appearance']['output_dim']
        self.concat=nn.Sequential(nn.Linear(app+dim,dim),nn.LayerNorm(dim),nn.ReLU()) if variant=='naive_concat' else None
    def forward(self, appearance, motion, detection_conf=None):
        b=self.base
        af=b.appearance_branch(appearance) if self.variant != 'motion_only' else None
        seq,mask=b.motion_branch(motion,detection_conf,return_sequence=True)
        mf=seq.mean(dim=1)
        if self.variant=='appearance_only': fused=b.fusion.appearance_residual(af); attn=None
        elif self.variant=='motion_only': fused=mf; attn=None
        elif self.variant=='naive_concat': fused=self.concat(torch.cat([af,mf],dim=-1)); attn=None
        else:
            if self.variant=='unmasked': mask=torch.ones_like(mask)
            fused,attn=b.fusion(af,seq,mask)
        cls, dims, total=b.head(fused)
        return {'cls_logits':cls,'dim_scores':dims,'total_score':total,'attn_weights':attn,'occlusion_mask':mask}
    def compute_loss(self,*args): return self.base.compute_loss(*args)

def main(a):
    c=yaml.safe_load(Path(a.config).read_text()); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c['training']['seed'] = a.seed
    c['training']['num_epochs'] = a.epochs
    if a.variant != 'motion_only': require_adapted_backbone(c)
    tr=DataLoader(ArthroscopyDataset(a.data,'train',a.config),batch_size=c['training']['batch_size'],shuffle=True,collate_fn=collate_fn,num_workers=0); va=DataLoader(ArthroscopyDataset(a.data,'val',a.config),batch_size=c['training']['batch_size'],shuffle=False,collate_fn=collate_fn,num_workers=0)
    validate_fixed_datasets({'train': tr.dataset, 'val': va.dataset})
    m=AblationModel(c,a.variant).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']); best=-1; rows=[]; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    from train import build_scheduler
    scheduler=build_scheduler(opt,a.epochs,c["training"]["warmup_epochs"])
    for ep in range(1,a.epochs+1):
        m.train()
        for batch in tr:
            o=m(batch['appearance'].to(dev),batch['motion'].to(dev),batch['detection_conf'].to(dev)); loss=m.compute_loss(o,batch['label_cls'].to(dev),batch['label_reg'].to(dev))['total']; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        from train import validate
        metrics=validate(m,va,dev)
        acc=metrics['accuracy']; mae=metrics['mae']
        scheduler.step()
        row={'epoch':ep,'accuracy':acc,'grs_mae':mae}; rows.append(row)
        print(f'{a.variant} epoch {ep}: accuracy={acc:.4f} mae={mae:.4f}')
        if acc>best: best=acc; torch.save({'model_state_dict':m.state_dict(),'variant':a.variant,'config':c,'epoch':ep,'metrics':row},out/f'{a.variant}_best.pth')
    (out/f'{a.variant}_history.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--variant',choices=['appearance_only','motion_only','naive_concat','unmasked','full'],required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--seed',type=int,default=3407); main(p.parse_args())
