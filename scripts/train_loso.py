"""Train site-held-out folds from independently prepared fold directories.

Each A/B/C directory contains train/clips, val/clips, test/clips and a
 detector_manifest.json recording the training procedure IDs of its detector.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import torch
import yaml
from torch.utils.data import DataLoader
from data import ArthroscopyDataset, collate_fn
from models import ArthroSkillMC
from data.protocol import require_adapted_backbone, validate_fixed_datasets
from train import set_seed, train_epoch, validate, build_optimizer, build_scheduler
from utils.inference import predict_procedures


def metadata(dataset):
    result = {}
    for sample in dataset.samples:
        record = torch.load(sample['clip_path'], map_location='cpu', weights_only=False)
        video_id = record['video_id']
        centre = record['centre'].replace('Centre ', '')
        if video_id in result and result[video_id] != centre:
            raise ValueError('Inconsistent procedure site')
        result[video_id] = centre
    if not result:
        raise ValueError('Empty LOSO partition')
    return result


def main(a):
    cfg = yaml.safe_load(Path(a.config).read_text()); tr = cfg['training']
    require_adapted_backbone(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results=[]; seen=set()
    for held in ['A','B','C']:
        root=Path(a.data)/held
        datasets={s:ArthroscopyDataset(root,s,a.config) for s in ['train','val','test']}
        validate_fixed_datasets(datasets, fixed_partition=False)
        # Held-out inference must never apply training augmentations.
        info={s:metadata(d) for s,d in datasets.items()}
        train_ids, val_ids, test_ids = (set(info[s]) for s in ['train','val','test'])
        if train_ids&val_ids or train_ids&test_ids or val_ids&test_ids:
            raise ValueError('Procedure leakage across fold partitions')
        if any(site==held for s in ['train','val'] for site in info[s].values()) or set(info['test'].values())!={held}:
            raise ValueError('Held-out hospital leaked into fitting or selection')
        manifest=json.loads((root/'detector_manifest.json').read_text())
        if not set(manifest['training_video_ids']) <= train_ids:
            raise ValueError('Detector annotations include non-training procedures')
        if not manifest.get('weights_sha256') or manifest.get('annotation_frames',0)<=0:
            raise ValueError('Detector manifest requires weights_sha256 and a positive annotation frame count')
        loaders={s:DataLoader(d,batch_size=tr['batch_size'],shuffle=s=='train',collate_fn=collate_fn,num_workers=0) for s,d in datasets.items()}
        set_seed(tr['seed']); model=ArthroSkillMC(cfg).to(device)
        optimizer=build_optimizer(model,tr['learning_rate'],tr['weight_decay'])
        scheduler=build_scheduler(optimizer,tr['num_epochs'],tr['warmup_epochs'])
        out=Path(a.output)/held;out.mkdir(parents=True,exist_ok=True);best=-1
        for epoch in range(1,tr['num_epochs']+1):
            train_epoch(model,loaders['train'],optimizer,device,epoch)
            metrics=validate(model,loaders['val'],device);scheduler.step()
            if metrics['accuracy']>best:
                best=metrics['accuracy'];torch.save(dict(model_state_dict=model.state_dict(),config=cfg,epoch=epoch,metrics=metrics),out/'best_model.pth')
        model.load_state_dict(torch.load(out/'best_model.pth',map_location=device,weights_only=False)['model_state_dict'])
        result=predict_procedures(model,loaders['test'],device);result['held_out']=held
        if seen&test_ids: raise ValueError('Repeated held-out procedure')
        seen.update(test_ids);results.append(result)
    if len(seen)!=196: raise ValueError('Expected all 196 procedures across LOSO tests')
    (Path(a.output)/'loso_results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
