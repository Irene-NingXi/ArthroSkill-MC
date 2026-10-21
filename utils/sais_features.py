"""Frozen official DINO features and RAFT flow-image encoding for SAIS."""
from io import BytesIO
from pathlib import Path
import hashlib
import importlib.metadata
import inspect
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from models.sais import load_upstream


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def flow_pairs(indices, step):
    if step < 1 or step >= len(indices):
        raise ValueError('Flow step must be positive and smaller than the RGB clip length')
    return [(i, i+step) for i in range(0, len(indices)-step, step)]


def image_transform(crop_fraction=0.8):
    if not 0 < crop_fraction <= 1:
        raise ValueError('Crop fraction must lie in (0, 1]')
    def transform(image):
        width, height = image.size
        image = transforms.functional.center_crop(
            image, [round(height*crop_fraction), round(width*crop_fraction)])
        image = transforms.functional.resize(image, [224, 224], antialias=True)
        return transforms.functional.normalize(transforms.functional.to_tensor(image),
                                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transform


def flow_image(flow_rgb):
    """Match the released flow RGB quantisation and JPEG storage boundary."""
    image = Image.fromarray((np.asarray(flow_rgb)*255).astype(np.uint8), mode='RGB')
    buffer = BytesIO(); image.save(buffer, format='JPEG', quality=75); buffer.seek(0)
    return Image.open(buffer).convert('RGB')


class FrozenDINO:
    def __init__(self, upstream_root, weights, device='cpu', crop_fraction=0.8, batch_size=16):
        if not Path(weights).is_file():
            raise FileNotFoundError('Provide the pretrained DINO ViT-S/16 weights')
        module = load_upstream(upstream_root, 'SAIS/scripts/dino-main/vision_transformer.py')
        self.model = module.vit_small(patch_size=16, num_classes=0)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = device; self.transform = image_transform(crop_fraction)
        if batch_size < 1:
            raise ValueError('Feature batch size must be positive')
        self.batch_size = batch_size
        self.weights_sha256 = sha256_file(weights)

    @torch.no_grad()
    def __call__(self, images):
        if not images:
            raise ValueError('Empty feature-extraction image sequence')
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = torch.stack([self.transform(image) for image in images[start:start+self.batch_size]])
            chunks.append(self.model(batch.to(self.device)).cpu())
        return torch.cat(chunks)


class PTLFlowRAFT:
    """Use PTLFlow RAFT weights and colour encoding from the official pipeline."""
    def __init__(self, weights, device='cpu'):
        if not Path(weights).is_file():
            raise FileNotFoundError('Provide the PTLFlow RAFT things checkpoint')
        import ptlflow
        from ptlflow.utils.io_adapter import IOAdapter
        from ptlflow.utils.flow_utils import flow_to_rgb
        self.model = ptlflow.get_model('raft')
        # Load trusted local PTLFlow checkpoints, which may contain legacy
        # argparse metadata in addition to their tensor state dictionary.
        checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device); self.adapter_cls = IOAdapter
        self.flow_to_rgb = flow_to_rgb
        self.legacy_adapter = 'model' in inspect.signature(IOAdapter.__init__).parameters
        self.version = importlib.metadata.version('ptlflow')
        self.weights_sha256 = sha256_file(weights)

    @torch.no_grad()
    def __call__(self, first_rgb, second_rgb):
        kwargs = dict(input_size=first_rgb.shape[:2], cuda=False)
        if self.legacy_adapter:
            adapter = self.adapter_cls(self.model, **kwargs)
        else:
            adapter = self.adapter_cls(output_stride=self.model.output_stride, **kwargs)
        # The upstream PTLFlow input adapter expects OpenCV BGR arrays.
        inputs = adapter.prepare_inputs([first_rgb[:, :, ::-1].copy(), second_rgb[:, :, ::-1].copy()])
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        prediction = self.model(inputs)
        unscale = adapter.unpad_and_unscale if self.legacy_adapter else adapter.unscale
        prediction = unscale({'flows': prediction['flows']})
        rgb = self.flow_to_rgb(prediction['flows'])[0, 0].permute(1, 2, 0).cpu().numpy()
        return flow_image(rgb)


def encode_clip(frames_rgb, indices, dino, raft, step):
    if len(frames_rgb) != 16 or len(indices) != 16 or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError('Expected a shared 16-frame clip in source temporal order')
    pairs = flow_pairs(indices, step)
    flow_images = [raft(frames_rgb[i], frames_rgb[j]) for i, j in pairs]
    rgb_features = dino([Image.fromarray(frame, mode='RGB') for frame in frames_rgb])
    flow_features = dino(flow_images)
    return rgb_features, flow_features, [[indices[i], indices[j]] for i, j in pairs]
