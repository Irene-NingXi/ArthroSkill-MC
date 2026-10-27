"""Exercise official SAIS fitting and checkpoint evaluation on synthetic features."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import hashlib
import json
import time
import torch
import yaml
from data.sais_dataset import FORMAT
from models.sais import LOCK_PATH
from scripts.environment_report import environment
from utils.sais_training import train_sais, evaluate_checkpoint


def synthetic_cache(root):
    lock = json.loads(LOCK_PATH.read_text())
    manifest = dict(format=FORMAT, synthetic=True, provenance=dict(
        sais_commit=lock['commit'], source_sha256=lock['source_sha256'],
        flow_step=15, crop_fraction=.8, synthetic=True), clips=[])
    generator = torch.Generator().manual_seed(31)
    for split in ['train', 'val', 'test']:
        for grade in range(3):
            video_id = f'synthetic_{split}_{grade}'
            path = root / split / 'clips' / f'{video_id}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            indices = list(range(16))
            payload = dict(format=FORMAT, synthetic=True,
                           rgb_features=torch.randn(16, 384, generator=generator),
                           flow_features=torch.randn(1, 384, generator=generator),
                           video_id=video_id, label_cls=grade,
                           source_frame_indices=indices, flow_frame_pairs=[[0, 15]])
            torch.save(payload, path)
            manifest['clips'].append(dict(path=path.relative_to(root).as_posix(), split=split,
                                           video_id=video_id, label_cls=grade,
                                           source_frame_indices=indices,
                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run(output, upstream_root):
    start = time.perf_counter()
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty synthetic smoke-test directory')
    features = out / 'features'
    features.mkdir(parents=True)
    synthetic_cache(features)
    config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    config['training'].update(device='cpu', num_workers=0, batch_size=3, num_epochs=2)
    torch.set_num_threads(1)
    training = train_sais(config, features, out / 'training', upstream_root=upstream_root,
                           epochs=2, require_study_counts=False)
    result = evaluate_checkpoint(out / 'training/sais_best_model.pth', features,
                                 out / 'evaluation', upstream_root=upstream_root,
                                 device='cpu', require_study_counts=False)
    assert result['metrics']['n'] == 3
    assert len(result['per_clip']) == 3
    report = dict(passed=True, synthetic=True, study_performance=False,
                  description='Random feature fixture; no clinical videos or pretrained extraction weights used.',
                  best_epoch=training['best_epoch'], checkpoint_epoch=result['checkpoint_epoch'],
                  test_procedures=result['metrics']['n'],
                  elapsed_seconds=time.perf_counter() - start, environment=environment())
    (out / 'smoke_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Synthetic SAIS training and evaluation passed in {report["elapsed_seconds"]:.1f} seconds.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/smoke')
    parser.add_argument('--sais-root', default='external/SAIS')
    args = parser.parse_args()
    run(args.output, args.sais_root)
