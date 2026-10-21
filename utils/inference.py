"""Average clip probabilities and GRS totals within each procedure."""
import numpy as np
import torch
from .metrics import SkillMetrics


def aggregate_clips(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['video_id'], []).append(row)
    if not grouped:
        raise ValueError('No clips supplied for procedure evaluation')
    procedures = []
    for video_id, clips in sorted(grouped.items()):
        truth = {(c['true_cls'], c['true_reg']) for c in clips}
        if len(truth) != 1:
            raise ValueError(f'Inconsistent labels within procedure {video_id}')
        true_cls, true_reg = truth.pop()
        probabilities = np.mean([c['probabilities'] for c in clips], axis=0)
        procedures.append(dict(video_id=video_id, n_clips=len(clips),
                               true_cls=true_cls, true_reg=true_reg,
                               pred_cls=int(probabilities.argmax()),
                               pred_reg=float(np.mean([c['pred_reg'] for c in clips])),
                               probabilities=probabilities.tolist()))
    metrics = SkillMetrics.compute_all(
        np.array([r['true_cls'] for r in procedures]),
        np.array([r['pred_cls'] for r in procedures]),
        np.array([r['true_reg'] for r in procedures]),
        np.array([r['pred_reg'] for r in procedures]),
        np.array([r['probabilities'] for r in procedures]))
    metrics['n'] = len(procedures)
    return {'unit': 'procedure', 'score_units': 'grs_points',
            'metrics': metrics, 'per_procedure': procedures}


@torch.no_grad()
def predict_procedures(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        outputs = model(batch['appearance'].to(device), batch['motion'].to(device),
                        batch['detection_conf'].to(device))
        probs = outputs['cls_logits'].softmax(-1).cpu().numpy()
        scores = outputs['total_score'].flatten().cpu().numpy()
        for i, video_id in enumerate(batch['video_ids']):
            rows.append(dict(video_id=video_id, true_cls=int(batch['label_cls'][i]),
                             true_reg=float(batch['label_reg'][i, 6]),
                             probabilities=probs[i].tolist(), pred_reg=float(scores[i])))
    return aggregate_clips(rows)
