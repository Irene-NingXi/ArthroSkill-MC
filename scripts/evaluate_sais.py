"""Evaluate the selected official SAIS checkpoint by averaging clip probabilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import torch
from utils.sais_training import evaluate_checkpoint


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sais-root')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    result = evaluate_checkpoint(a.checkpoint, a.data, a.output, upstream_root=a.sais_root,
                                 split=a.split, device=a.device)
    print(json.dumps(result['metrics'], indent=2))
