"""Create paired official DINO/RAFT caches for the existing shared clip inventory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import importlib.metadata
import json
import torch
import yaml
from data.sais_dataset import FORMAT, load_manifest
from models.sais import LOCK_PATH
from utils.sais_features import FrozenDINO, PTLFlowRAFT, encode_clip, sha256_file


def inventory(metadata_path, clips_root):
    with Path(metadata_path).open(newline='', encoding='utf-8-sig') as f:
        metadata = list(csv.DictReader(f))
    if len({r['video_id'] for r in metadata}) != len(metadata):
        raise ValueError('Duplicate procedure in metadata')
    by_video = {r['video_id']: [] for r in metadata}
    expected = {r['video_id']: r for r in metadata}
    for split in ['train', 'val', 'test']:
        for path in sorted((Path(clips_root)/split/'clips').glob('*.pt')):
            # mmap avoids loading large appearance tensors merely to read IDs.
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            video_id = payload['video_id']
            row = expected[video_id]
            if row['split'] != split or int(row['label_cls']) != int(payload['label_cls']):
                raise ValueError('Source clip differs from the authoritative metadata')
            indices = payload.get('source_frame_indices')
            if indices is None:
                raise ValueError('Regenerate legacy clips without source_frame_indices')
            by_video[video_id].append(dict(source_path=path, source_frame_indices=list(indices)))
    if any(not clips for clips in by_video.values()):
        raise ValueError('Every metadata procedure must have shared input clips')
    return metadata, by_video


def generate_cache(config, metadata_path, clips_root, output, upstream_root, dino_weights,
                   raft_weights, device='cpu', feature_batch_size=16):
    import cv2
    cfg = config['sais']; step = int(cfg['flow_step'])
    metadata, clips = inventory(metadata_path, clips_root)
    out = Path(output)
    if out.exists():
        raise FileExistsError('Use a new output directory to avoid mixing feature caches')
    dino = FrozenDINO(upstream_root, dino_weights, device=device,
                      crop_fraction=float(cfg['crop_fraction']), batch_size=feature_batch_size)
    raft = PTLFlowRAFT(raft_weights, device=device)
    lock = json.loads(LOCK_PATH.read_text())
    manifest = dict(format=FORMAT, provenance=dict(
        sais_commit=lock['commit'], source_sha256=lock['source_sha256'],
        dino_weights_sha256=dino.weights_sha256, raft_weights_sha256=raft.weights_sha256,
        source_metadata_sha256=sha256_file(metadata_path), ptlflow_version=raft.version,
        torch_version=str(torch.__version__), torchvision_version=importlib.metadata.version('torchvision'),
        flow_step=step, crop_fraction=float(cfg['crop_fraction']),
        flow_image_encoding='PTLFlow flow_to_rgb; uint8 floor; JPEG quality 75',
        spatial_encoder='official DINO ViT-S/16; frozen; 384 dimensions',
        frame_size=config['data']['img_size'], source_fps=config['data']['fps']), clips=[])
    out.mkdir(parents=True)
    for row in metadata:
        cap = cv2.VideoCapture(row['video_path'])
        try:
            if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-config['data']['fps']) > 0.1:
                raise ValueError(f'Cannot open a matching-fps source video: {row["video_id"]}')
            frame_no = 0; used = set()
            for clip in sorted(clips[row['video_id']], key=lambda c: c['source_frame_indices'][0]):
                indices = clip['source_frame_indices']
                if len(indices) != 16 or any(b <= a for a, b in zip(indices, indices[1:])) or used.intersection(indices):
                    raise ValueError('Source clip inventory must be ordered and non-overlapping')
                used.update(indices); wanted = set(indices); frames = []
                while frame_no <= indices[-1]:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'Source video ended before a requested frame: {row["video_id"]}')
                    if frame_no in wanted:
                        h, w = config['data']['img_size']
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_no += 1
                rgb, flow, pairs = encode_clip(frames, indices, dino, raft, step)
                relative = Path(row['split'])/'clips'/clip['source_path'].name
                target = out/relative; target.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(format=FORMAT, rgb_features=rgb, flow_features=flow,
                               video_id=row['video_id'], label_cls=int(row['label_cls']),
                               source_frame_indices=indices, flow_frame_pairs=pairs)
                torch.save(payload, target)
                manifest['clips'].append(dict(path=relative.as_posix(), split=row['split'],
                                               video_id=row['video_id'], label_cls=int(row['label_cls']),
                                               source_frame_indices=indices, sha256=sha256_file(target),
                                               source_clip_sha256=sha256_file(clip['source_path'])))
        finally:
            cap.release()
        print(f'Extracted paired SAIS features: {row["video_id"]}', flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    load_manifest(out)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--metadata', required=True); p.add_argument('--clips', required=True)
    p.add_argument('--output', required=True); p.add_argument('--sais-root')
    p.add_argument('--dino-weights', required=True); p.add_argument('--raft-weights', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--feature-batch-size', type=int, default=16)
    a = p.parse_args(); cfg = yaml.safe_load(Path(a.config).read_text())
    generate_cache(cfg, a.metadata, a.clips, a.output, a.sais_root or cfg['sais']['upstream_root'],
                   a.dino_weights, a.raft_weights, a.device, a.feature_batch_size)
