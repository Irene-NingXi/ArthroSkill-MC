"""Fit external video baselines with Kinetics initialisation and fixed splits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import torch
import yaml
from torch.utils.data import DataLoader
from data import ArthroscopyDataset, collate_fn
from data.protocol import validate_fixed_datasets
from models.baselines import ResNet3DBaseline, VideoSwinOnly
from train import build_scheduler, set_seed
from utils.inference import predict_procedures
from utils.sais_training import device_from_config


def main(args):
    config = yaml.safe_load(Path(args.config).read_text())
    training = config['training']
    set_seed(training['seed'])
    device = device_from_config(config)
    # External baselines use public Kinetics weights, not internal adaptation.
    config['model']['appearance'].update(pretrained=True, checkpoint=None)
    model = (ResNet3DBaseline(config, pretrained=True) if args.model == 'resnet3d'
             else VideoSwinOnly(config)).to(device)
    datasets = {s: ArthroscopyDataset(args.data, s, config=config) for s in ['train', 'val']}
    validate_fixed_datasets(datasets)
    loaders = {s: DataLoader(dataset, batch_size=training['batch_size'], shuffle=s == 'train',
                             collate_fn=collate_fn, num_workers=training['num_workers'])
               for s, dataset in datasets.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                  weight_decay=training['weight_decay'])
    epochs = training['num_epochs'] if args.epochs is None else args.epochs
    if epochs < 1:
        raise ValueError('At least one epoch is required')
    config['training']['num_epochs'] = epochs
    scheduler = build_scheduler(optimizer, epochs, training['warmup_epochs'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders['train']:
            outputs = model(batch['appearance'].to(device))
            classification = torch.nn.functional.cross_entropy(outputs['cls_logits'], batch['label_cls'].to(device))
            regression = torch.nn.functional.smooth_l1_loss(outputs['total_score'], batch['label_reg'][:, 6:7].to(device))
            loss = classification + training['lambda_reg'] * regression
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = predict_procedures(model, loaders['val'], device)['metrics']
        history.append(dict(epoch=epoch, validation=metrics, learning_rate=optimizer.param_groups[0]['lr']))
        scheduler.step()
        print(f'Epoch {epoch}: procedure validation accuracy={metrics["accuracy"]:.4f}', flush=True)
        if metrics['accuracy'] > best:
            best = metrics['accuracy']
            torch.save(dict(model_state_dict=model.state_dict(), config=config, model_type=args.model,
                            epoch=epoch, metrics=metrics, initialisation='Kinetics-400'),
                       output / f'{args.model}_best.pth')
    (output / f'{args.model}_history.json').write_text(json.dumps(history, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['resnet3d', 'swin'], required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--pretrained', action='store_true', help='Compatibility option; Kinetics pretraining is always used')
    main(parser.parse_args())
