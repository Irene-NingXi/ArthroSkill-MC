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
from data.protocol import validate_fixed_datasets


def model_from_checkpoint(checkpoint, fallback_config):
    """Recreate the fitted architecture without downloading initial weights."""
    import copy
    config = copy.deepcopy(checkpoint.get('config', fallback_config))
    config['model']['appearance'].update(pretrained=False, checkpoint=None)
    if 'variant' in checkpoint:
        from scripts.run_ablation import AblationModel
        model = AblationModel(config, checkpoint['variant'])
    elif checkpoint.get('model_type') in ['resnet3d', 'swin']:
        from models.baselines import ResNet3DBaseline, VideoSwinOnly
        model = (ResNet3DBaseline(config, pretrained=False) if checkpoint['model_type'] == 'resnet3d'
                 else VideoSwinOnly(config))
    else:
        model = ArthroSkillMC(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model, config


def evaluate(model, dataloader, device, config=None):
    return predict_procedures(model, dataloader, device)


def main(config_path, data_dir, checkpoint_path, output_dir, split='test'):
    config = yaml.safe_load(Path(config_path).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Selected local study checkpoints may contain legacy NumPy metric scalars.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model, config = model_from_checkpoint(checkpoint, config)
    model.to(device)
    dataset = ArthroscopyDataset(data_dir, split=split, config=config, augment=False)
    validate_fixed_datasets({split: dataset})
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                        shuffle=False, collate_fn=collate_fn)
    result = evaluate(model, loader, device)
    result['split'] = split
    result['checkpoint_epoch'] = checkpoint.get('epoch')
    result['model_type'] = checkpoint.get('variant', checkpoint.get('model_type', 'full'))
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
