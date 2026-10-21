"""Evaluate a selected checkpoint on a specified procedure partition."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from models import ArthroSkillMC
from data import ArthroscopyDataset, collate_fn
from utils.inference import predict_procedures


def evaluate(model, dataloader, device, config=None):
    return predict_procedures(model, dataloader, device)


def main(config_path, data_dir, checkpoint_path, output_dir, split='test'):
    config = yaml.safe_load(Path(config_path).read_text())
    config['model']['appearance'].update(pretrained=False, checkpoint=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ArthroscopyDataset(data_dir, split=split, config_path=config_path)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                        shuffle=False, collate_fn=collate_fn)
    model = ArthroSkillMC(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    result = evaluate(model, loader, device)
    result['split'] = split
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'evaluation_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    a = p.parse_args()
    main(a.config, a.data, a.checkpoint, a.output, a.split)
