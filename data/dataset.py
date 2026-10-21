"""
Dataset definition.
Corresponds to paper Methods: "Each video was decoded at 25 fps, rescaled to 640x480,
and divided into non-overlapping 16-frame clips."
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
from pathlib import Path


class ArthroscopyDataset(Dataset):
    """
    Arthroscopic surgery video dataset.
    Input: preprocessed 16-frame clips (.pt)
    Output: {
        'appearance': [C, T, H, W] video tensor,
        'motion': [T, D] kinematic feature sequence,
        'label_cls': int skill level (0=novice, 1=intermediate, 2=expert),
        'label_reg': [7] 6 GRS dimensions + total score (native GRS points),
        'centre': str site identifier ("A", "B", or "C") when present
    }
    """
    def __init__(self, data_dir, split="train", config_path="configs/config.yaml", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.clip_len = self.cfg["data"]["clip_len"]
        self.img_size = self.cfg["data"]["img_size"]
        self.num_classes = self.cfg["data"]["num_classes"]

        self.samples = self._load_samples()
        print(f"[{split}] Loaded {len(self.samples)} samples")

    def _load_samples(self):
        split_dir = self.data_dir / self.split
        clips_dir = split_dir / "clips"

        samples = []
        if not clips_dir.exists():
            return samples

        for clip_file in sorted(clips_dir.glob("*.pt")):
            samples.append({
                "clip_path": str(clip_file),
                "clip_id": clip_file.stem
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.load(sample["clip_path"], weights_only=False)

        appearance = data["frames"]
        if appearance.dim() == 4 and appearance.shape[-1] == 3:
            appearance = appearance.permute(3, 0, 1, 2)

        motion = data["motion"]
        detection_conf = data["detection_conf"]
        if data.get("score_units") != "grs_points":
            raise ValueError("Clip requires score_units=grs_points; regenerate legacy normalised clips")
        label_cls = data["label_cls"]
        label_reg = data["label_reg"]
        centre = data.get("centre", None)

        if self.split == "train":
            appearance = self.transform(appearance) if self.transform else self._augment(appearance)

        return {
            "appearance": appearance.float(),
            "motion": motion.float(),
            "detection_conf": detection_conf.float(),
            "label_cls": torch.tensor(label_cls, dtype=torch.long),
            "label_reg": label_reg.float(),
            "clip_id": sample["clip_id"],
            "centre": centre,
            "video_id": data["video_id"]
        }

    def _augment(self, x):
        """Use one spatial crop, flip and colour transform for the full clip."""
        import torch.nn.functional as F
        cfg = self.cfg["training"].get("augmentation", {})
        if cfg.get("random_crop", False):
            h, w = x.shape[-2:]
            scale = float(torch.empty(1).uniform_(0.8, 1.0))
            ch, cw = max(1, int(h*scale)), max(1, int(w*scale))
            top = int(torch.randint(h-ch+1, (1,)))
            left = int(torch.randint(w-cw+1, (1,)))
            x = F.interpolate(x[:, :, top:top+ch, left:left+cw].permute(1,0,2,3),
                              size=(h,w), mode="bilinear", align_corners=False).permute(1,0,2,3)
        if cfg.get("horizontal_flip", False) and torch.rand(1) > 0.5:
            x = x.flip(-1)
        strength = float(cfg.get("color_jitter", 0))
        if strength:
            gain = torch.empty(3,1,1,1).uniform_(1-strength, 1+strength)
            x = x*gain + float(torch.empty(1).uniform_(-strength, strength))
        return x


def collate_fn(batch):
    """Custom collate for batching."""
    appearance = torch.stack([b["appearance"] for b in batch])
    motion = torch.stack([b["motion"] for b in batch])
    detection_conf = torch.stack([b["detection_conf"] for b in batch])
    label_cls = torch.stack([b["label_cls"] for b in batch])
    label_reg = torch.stack([b["label_reg"] for b in batch])
    clip_ids = [b["clip_id"] for b in batch]

    return {
        "appearance": appearance,
        "motion": motion,
        "detection_conf": detection_conf,
        "label_cls": label_cls,
        "label_reg": label_reg,
        "clip_ids": clip_ids,
        "video_ids": [b["video_id"] for b in batch],
        "centres": [b["centre"] for b in batch]
    }
