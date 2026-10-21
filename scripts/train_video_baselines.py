"""Train/evaluate the paper's 3D-ResNet-18 or Video-Swin-only baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, json
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from data import ArthroscopyDataset, collate_fn
from models.baselines import ResNet3DBaseline, VideoSwinOnly

def main(a):
 c=yaml.safe_load(Path(a.config).read_text()); from_seed=c["training"]["seed"]; from train import set_seed; set_seed(from_seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); cls=ResNet3DBaseline if a.model=='resnet3d' else VideoSwinOnly; m=cls(c,pretrained=a.pretrained).to(dev) if a.model=='resnet3d' else cls(c).to(dev)
 tr=DataLoader(ArthroscopyDataset(a.data,'train',a.config),batch_size=c['training']['batch_size'],shuffle=True,collate_fn=collate_fn,num_workers=0); va=DataLoader(ArthroscopyDataset(a.data,'val',a.config),batch_size=c['training']['batch_size'],shuffle=False,collate_fn=collate_fn,num_workers=0); opt=torch.optim.AdamW(m.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 from train import build_scheduler, set_seed
 set_seed(c["training"]["seed"]); best=-1; scheduler=build_scheduler(opt,a.epochs,c["training"]["warmup_epochs"])
 for ep in range(1,a.epochs+1):
  m.train()
  for b in tr:
   o=m(b['appearance'].to(dev)); loss=torch.nn.functional.cross_entropy(o['cls_logits'],b['label_cls'].to(dev))+torch.nn.functional.smooth_l1_loss(o['total_score'],b['label_reg'][:,6:7].to(dev)); opt.zero_grad(); loss.backward(); opt.step()
  from utils.inference import predict_procedures
  metrics=predict_procedures(m,va,dev)['metrics']; acc=metrics['accuracy']; scheduler.step()
  print(f'epoch {ep}: val_accuracy={acc:.4f}')
  if acc>best:
   best=acc; torch.save({'model_state_dict':m.state_dict(),'config':c,'metrics':metrics},out/f'{a.model}_best.pth')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--model',choices=['resnet3d','swin'],required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--pretrained',action='store_true'); main(p.parse_args())
