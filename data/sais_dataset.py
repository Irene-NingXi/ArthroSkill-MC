"""Strict paired DINO-feature cache for SAIS on the shared procedure split."""
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

FORMAT = 'sais_rgb_flow_dino_v1'


def load_manifest(root, require_study_counts=True):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError('Unsupported SAIS feature-cache format')
    step = int(manifest['provenance']['flow_step'])
    if not 1 <= step < 16:
        raise ValueError('Invalid configured flow-pair step')
    records = manifest['clips']
    if not records:
        raise ValueError('Empty SAIS feature cache')
    ownership = {}; labels = {}; paths = set(); source_indices = {}
    for row in records:
        split = row['split']; video_id = row['video_id']; path = row['path']
        if split not in ['train', 'val', 'test'] or path in paths:
            raise ValueError('Invalid split or duplicate clip path')
        resolved = (root/path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f'Missing clip or path outside cache: {path}')
        if video_id in ownership and ownership[video_id] != split:
            raise ValueError(f'Procedure leakage across partitions: {video_id}')
        label = int(row['label_cls'])
        if label not in [0, 1, 2] or (video_id in labels and labels[video_id] != label):
            raise ValueError('Invalid or inconsistent procedure grade')
        indices = row['source_frame_indices']
        if len(indices) != 16 or any(type(i) is not int or i < 0 for i in indices) or any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError('Expected 16 strictly ordered source-frame indices')
        used = source_indices.setdefault(video_id, set())
        if used.intersection(indices):
            raise ValueError('Overlapping clips in a non-overlapping clip protocol')
        used.update(indices); ownership[video_id] = split; labels[video_id] = label; paths.add(path)
    counts = {s: sum(v == s for v in ownership.values()) for s in ['train', 'val', 'test']}
    if require_study_counts and counts != {'train': 117, 'val': 39, 'test': 40}:
        raise ValueError(f'Expected the fixed 117/39/40 procedure split, received {counts}')
    if require_study_counts:
        from .protocol import validate_assignments
        validate_assignments(records)
    return manifest


class SAISFeatureDataset(Dataset):
    def __init__(self, root, split, manifest=None):
        self.root = Path(root).resolve()
        self.manifest = manifest if manifest is not None else load_manifest(self.root)
        self.records = [r for r in self.manifest['clips'] if r['split'] == split]
        if not self.records:
            raise ValueError(f'No SAIS clips for {split}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        path = self.root/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Feature-cache checksum mismatch: {record["path"]}')
        row = torch.load(path, map_location='cpu', weights_only=True)
        if row.get('format') != FORMAT or row['video_id'] != record['video_id'] or int(row['label_cls']) != record['label_cls']:
            raise ValueError('Feature payload does not match its manifest')
        if row['source_frame_indices'] != record['source_frame_indices']:
            raise ValueError('Source-frame provenance mismatch')
        for field in ['rgb_features', 'flow_features']:
            x = row[field]
            if x.ndim != 2 or x.shape[1] != 384 or not 1 <= len(x) <= 2000 or not torch.isfinite(x).all():
                raise ValueError(f'Invalid paired DINO features: {field}')
        if len(row['rgb_features']) != 16:
            raise ValueError('The matched clip protocol requires 16 RGB tokens')
        pairs = row['flow_frame_pairs']
        if len(pairs) != len(row['flow_features']) or not pairs:
            raise ValueError('Flow tokens require recorded source-frame pairs')
        indices = set(record['source_frame_indices'])
        if any(len(pair) != 2 or pair[0] >= pair[1] or not set(pair) <= indices for pair in pairs):
            raise ValueError('Optical-flow pairs must lie inside the shared clip')
        step = int(self.manifest['provenance']['flow_step'])
        indices = record['source_frame_indices']
        expected_pairs = [[indices[i], indices[i+step]] for i in range(0, len(indices)-step, step)]
        if pairs != expected_pairs:
            raise ValueError('Flow endpoint pairs differ from the recorded extraction step')
        return dict(rgb_features=row['rgb_features'].float(),
                    flow_features=row['flow_features'].float(),
                    label_cls=int(row['label_cls']), video_id=row['video_id'],
                    clip_id=Path(record['path']).stem,
                    source_frame_indices=row['source_frame_indices'], flow_frame_pairs=pairs)


def collate_sais(batch):
    return dict(rgb_features=pad_sequence([r['rgb_features'] for r in batch], batch_first=True),
                flow_features=pad_sequence([r['flow_features'] for r in batch], batch_first=True),
                rgb_lengths=torch.tensor([len(r['rgb_features']) for r in batch]),
                flow_lengths=torch.tensor([len(r['flow_features']) for r in batch]),
                label_cls=torch.tensor([r['label_cls'] for r in batch], dtype=torch.long),
                video_ids=[r['video_id'] for r in batch], clip_ids=[r['clip_id'] for r in batch],
                source_frame_indices=[r['source_frame_indices'] for r in batch],
                flow_frame_pairs=[r['flow_frame_pairs'] for r in batch])
