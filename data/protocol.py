"""Validate fitted inputs against the released procedure assignments."""
from pathlib import Path

import numpy as np
import torch

from .source_data_io import read_public_tables


def study_assignments():
    tables, _ = read_public_tables(Path(__file__).parent / 'source_data')
    grades = ['novice', 'intermediate', 'expert']
    dimensions = {r['video_id']: [(r[f'rater_1_dimension_{i}'] + r[f'rater_2_dimension_{i}']) / 2
                                 for i in range(1, 7)] for r in tables['rater_scores']}
    return {r['video_id']: dict(split=r['dataset_split'],
                               label_cls=grades.index(r['skill_level']),
                               dimensions=dimensions[r['video_id']],
                               total=r['grs_score'],
                               centre=r['hospital'].replace('Centre ', ''))
            for r in tables['data_split']}


def validate_assignments(records, splits=('train', 'val', 'test'), fixed_partition=True):
    """Reject swapped IDs or grades even when partition counts are unchanged."""
    expected = study_assignments()
    seen = set()
    for row in records:
        video_id = row['video_id']
        if video_id not in expected:
            raise ValueError(f'Unknown study procedure: {video_id}')
        target = expected[video_id]
        if row['split'] not in splits or (fixed_partition and row['split'] != target['split']) or row['label_cls'] != target['label_cls']:
            raise ValueError(f'Public procedure assignment mismatch: {video_id}')
        if 'total' in row and not np.isclose(row['total'], target['total'], rtol=0, atol=1e-6):
            raise ValueError(f'Public consensus score mismatch: {video_id}')
        if 'centre' in row and row['centre'].replace('Centre ', '') != target['centre']:
            raise ValueError(f'Public centre mismatch: {video_id}')
        if 'dimensions' in row and not np.allclose(row['dimensions'], target['dimensions'], rtol=0, atol=1e-6):
            raise ValueError(f'Public dimension score mismatch: {video_id}')
        seen.add(video_id)
    required = {v for v, r in expected.items() if not fixed_partition or r['split'] in splits}
    if seen != required:
        raise ValueError(f'Incomplete study procedure coverage: missing {sorted(required-seen)}')


def validate_fixed_datasets(datasets, fixed_partition=True):
    records = []
    source_frames = {}
    for split, dataset in datasets.items():
        for sample in dataset.samples:
            row = torch.load(sample['clip_path'], map_location='cpu', weights_only=True, mmap=True)
            if tuple(row['frames'].shape) != (3, 16, 480, 640):
                raise ValueError('Study clips require RGB tensors with shape [3, 16, 480, 640]')
            indices = row.get('source_frame_indices', [])
            if len(indices) != 16 or any(type(i) is not int or i < 0 for i in indices) or any(b <= a for a, b in zip(indices, indices[1:])):
                raise ValueError('Study clips require 16 ordered original frame indices')
            used = source_frames.setdefault(row['video_id'], set())
            if used.intersection(indices):
                raise ValueError('Overlapping frames in the non-overlapping study clip inventory')
            used.update(indices)
            records.append(dict(video_id=row['video_id'], split=split,
                                label_cls=int(row['label_cls']),
                                total=float(row['label_reg'][6]), centre=row['centre'],
                                dimensions=row['label_reg'][:6].tolist()))
    validate_assignments(records, tuple(datasets), fixed_partition=fixed_partition)


def require_adapted_backbone(config):
    """Do not silently replace the study's surgical adaptation with Kinetics."""
    checkpoint = config['model']['appearance'].get('checkpoint')
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError('Study training requires model.appearance.checkpoint pointing to the HeiChole-adapted backbone')
