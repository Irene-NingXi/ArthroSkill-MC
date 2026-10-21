"""Training and classification-only procedure evaluation for official SAIS."""
import copy
import hashlib
import json
import importlib.metadata
from pathlib import Path
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from data.sais_dataset import SAISFeatureDataset, collate_sais, load_manifest
from models.sais import SAISBaseline, LOCK_PATH
from utils.metrics import SkillMetrics


def forward_batch(model, batch, device):
    return model(batch['rgb_features'].to(device), batch['flow_features'].to(device),
                 batch['rgb_lengths'].to(device), batch['flow_lengths'].to(device))


def aggregate_sais(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['video_id'], []).append(row)
    if not grouped:
        raise ValueError('No SAIS predictions')
    procedures = []
    for video_id, clips in sorted(grouped.items()):
        truth = {r['true_cls'] for r in clips}
        if len(truth) != 1:
            raise ValueError(f'Inconsistent grade within {video_id}')
        probs = np.asarray([r['probabilities'] for r in clips], dtype=float)
        if probs.shape != (len(clips), 3) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1):
            raise ValueError('Expected finite three-grade probability vectors')
        mean = probs.mean(0)
        procedures.append(dict(video_id=video_id, n_clips=len(clips),
                               true_cls=int(next(iter(truth))), pred_cls=int(mean.argmax()),
                               probabilities=mean.tolist()))
    y = np.array([r['true_cls'] for r in procedures])
    p = np.array([r['pred_cls'] for r in procedures])
    probs = np.asarray([r['probabilities'] for r in procedures])
    metrics = dict(n=len(y), correct=int((y == p).sum()), accuracy=float(accuracy_score(y, p)),
                   balanced_accuracy=float(balanced_accuracy_score(y, p)),
                   mean_grade_distance=float(np.abs(y-p).mean()),
                   confusion_matrix=confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
                   calibration_error=float(SkillMetrics.calibration_error(y, probs)),
                   error_structure=SkillMetrics.adjacent_grade_error_rate(y, p))
    metrics['accuracy_ci'] = list(binomtest(metrics['correct'], metrics['n']).proportion_ci(method='wilson'))
    # Kappa is undefined when every true and predicted grade is the same.
    if len(np.unique(np.concatenate([y, p]))) > 1:
        metrics['quadratic_grade_kappa'] = float(cohen_kappa_score(y, p, labels=[0, 1, 2], weights='quadratic'))
    return dict(unit='procedure', task='three_grade_classification', metrics=metrics,
                per_procedure=procedures)


@torch.no_grad()
def evaluate_sais(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        probs = forward_batch(model, batch, device)['cls_logits'].softmax(-1).cpu().tolist()
        rows.extend(dict(video_id=video_id, clip_id=batch['clip_ids'][i],
                         source_frame_indices=batch['source_frame_indices'][i],
                         flow_frame_pairs=batch['flow_frame_pairs'][i],
                         true_cls=int(batch['label_cls'][i]), probabilities=probs[i])
                    for i, video_id in enumerate(batch['video_ids']))
    result = aggregate_sais(rows)
    result['per_clip'] = rows
    return result


def device_from_config(config):
    name = config['training'].get('device', 'auto')
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def train_sais(config, data_dir, output_dir, upstream_root=None, seed=None, epochs=None,
               require_study_counts=True):
    config = copy.deepcopy(config)
    seed = int(config['training']['seed'] if seed is None else seed)
    config['training']['seed'] = seed
    epochs = int(config['training']['num_epochs'] if epochs is None else epochs)
    config['training']['num_epochs'] = epochs
    if epochs < 1:
        raise ValueError('At least one training epoch is required')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty SAIS run directory to preserve selected checkpoints and seed provenance')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    manifest = load_manifest(data_dir, require_study_counts=require_study_counts)
    lock = json.loads(LOCK_PATH.read_text())
    if manifest['provenance']['sais_commit'] != lock['commit'] or manifest['provenance']['source_sha256'] != lock['source_sha256']:
        raise ValueError('Feature cache and pinned official SAIS revision disagree')
    for field in ['flow_step', 'crop_fraction']:
        if manifest['provenance'][field] != config['sais'][field]:
            raise ValueError(f'Feature extraction and training configuration disagree: {field}')
    datasets = {s: SAISFeatureDataset(data_dir, s, manifest) for s in ['train', 'val']}
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(d, batch_size=int(config['training']['batch_size']),
                             shuffle=s == 'train', collate_fn=collate_sais,
                             num_workers=int(config['training'].get('num_workers', 0)),
                             generator=generator if s == 'train' else None)
               for s, d in datasets.items()}
    device = device_from_config(config)
    model = SAISBaseline(config, upstream_root=upstream_root).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']),
                                  weight_decay=float(config['training']['weight_decay']))
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(sais_commit=lock['commit'], source_sha256=lock['source_sha256'],
                      feature_manifest_sha256=hashlib.sha256((Path(data_dir)/'manifest.json').read_bytes()).hexdigest(),
                      feature_provenance=manifest['provenance'], optimizer='AdamW',
                      scheduler='constant', seed=seed, torch_version=str(torch.__version__),
                      timm_version=importlib.metadata.version('timm'))
    history = []; best = -1.; best_epoch = None
    for epoch in range(1, epochs+1):
        model.train(); total_loss = 0.; n_clips = 0
        for batch in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_batch(model, batch, device)
            loss = model.compute_loss(outputs, batch['label_cls'].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite SAIS prototype loss')
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach())*len(batch['label_cls']); n_clips += len(batch['label_cls'])
        validation = json.loads(json.dumps(evaluate_sais(model, loaders['val'], device)))
        row = dict(epoch=epoch, training_loss=total_loss/n_clips,
                   validation=validation['metrics'], learning_rate=optimizer.param_groups[0]['lr'])
        history.append(row)
        if validation['metrics']['accuracy'] > best:
            best = validation['metrics']['accuracy']; best_epoch = epoch
            torch.save(dict(format='official_sais_adapter_v1', model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(), config=config,
                            epoch=epoch, validation=validation, provenance=provenance),
                       out/'sais_best_model.pth')
        print(f'Epoch {epoch}/{epochs}: loss={row["training_loss"]:.5f}, procedure val accuracy={validation["metrics"]["accuracy"]:.4f}', flush=True)
    report = dict(best_epoch=best_epoch, best_validation_accuracy=best, history=history, provenance=provenance)
    (out/'sais_training.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def evaluate_checkpoint(checkpoint_path, data_dir, output_dir, upstream_root=None, split='test',
                        device='cpu', require_study_counts=True):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'official_sais_adapter_v1':
        raise ValueError('Expected an official RGB/flow SAIS adapter checkpoint, including prototypes')
    config = checkpoint['config']; manifest = load_manifest(data_dir, require_study_counts=require_study_counts)
    digest = hashlib.sha256((Path(data_dir)/'manifest.json').read_bytes()).hexdigest()
    if digest != checkpoint['provenance']['feature_manifest_sha256']:
        raise ValueError('Evaluation feature cache differs from the selected training run')
    if checkpoint['provenance']['sais_commit'] != json.loads(LOCK_PATH.read_text())['commit']:
        raise ValueError('Checkpoint official SAIS revision is incompatible')
    model = SAISBaseline(config, upstream_root=upstream_root).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    dataset = SAISFeatureDataset(data_dir, split, manifest)
    loader = DataLoader(dataset, batch_size=int(config['training']['batch_size']),
                        shuffle=False, collate_fn=collate_sais, num_workers=0)
    result = evaluate_sais(model, loader, device)
    result.update(split=split, checkpoint_epoch=checkpoint['epoch'], provenance=checkpoint['provenance'])
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'sais_evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
