"""Train official RGB/flow SAIS with the ArthroSkill-MC fixed-split protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import yaml
from utils.sais_training import train_sais


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sais-root')
    p.add_argument('--seed', type=int)
    p.add_argument('--epochs', type=int)
    a = p.parse_args()
    train_sais(yaml.safe_load(Path(a.config).read_text()), a.data, a.output,
               upstream_root=a.sais_root, seed=a.seed, epochs=a.epochs)
