"""Adapter for the external, pinned official RGB/flow SAIS prototype model.

Upstream source is loaded from a user-supplied checkout and is not vendored.
The two inputs are frozen DINO features, not instrument kinematics.
"""
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

LOCK_PATH = Path(__file__).resolve().parents[1]/'configs/sais_upstream.lock.json'


def load_upstream(root, relative_path):
    root = Path(root).resolve()
    lock = json.loads(LOCK_PATH.read_text())
    path = root/relative_path
    if not path.is_file():
        raise FileNotFoundError(f'Official SAIS source missing: {path}. Run scripts/setup_sais.py.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != lock['source_sha256'][relative_path]:
        raise ValueError(f'Official SAIS source differs from pinned commit: {relative_path}')
    name = '_sais_external_'+digest
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        if relative_path == 'SAIS/scripts/dino-main/vision_transformer.py':
            # Resolve the DINO-local helper without replacing this project's
            # utils package or changing global import paths or module aliases.
            helper = load_upstream(root, 'SAIS/scripts/dino-main/utils.py')
            def local_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == 'utils' and level == 0:
                    return helper
                return builtins.__import__(name, globals, locals, fromlist, level)
            module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


class TupleTransformerEncoder(nn.TransformerEncoder):
    """Supply the tuple expected by upstream without patching PyTorch files.

    Attention maps are not exported. The encoded tensor and parameters retain
    standard torch.nn.TransformerEncoder semantics and state-dict names.
    """
    @classmethod
    def from_encoder(cls, encoder):
        result = cls(copy.deepcopy(encoder.layers[0]), len(encoder.layers),
                     norm=copy.deepcopy(encoder.norm), enable_nested_tensor=False)
        result.load_state_dict(encoder.state_dict(), strict=True)
        return result

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        encoded = super().forward(src, mask=mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  is_causal=is_causal)
        return encoded, None


class SAISBaseline(nn.Module):
    """Official shared temporal encoder, RGB+flow sum and learned prototypes."""
    def __init__(self, config, upstream_root=None):
        super().__init__()
        cfg = config['sais']
        if cfg.get('modalities', 'RGB-Flow') != 'RGB-Flow' or cfg.get('feature_dim', 384) != 384 or cfg.get('prototype_dim', 256) != 256:
            raise ValueError('Official SAIS requires RGB-Flow, DINO dimension 384 and prototype dimension 256')
        upstream = load_upstream(upstream_root or cfg['upstream_root'],
                                 'SAIS/scripts/prepare_model.py')
        self.rep_dim = 384
        self.num_classes = int(config['data']['num_classes'])
        if self.num_classes != 3:
            raise ValueError('ArthroSkill-MC requires three ordered skill grades')
        self.core = upstream.fullModel(
            data_type='reps', nclasses=self.num_classes, domain='ArthroSkill',
            rep_dim=self.rep_dim, encoder_type='ViT', modalities='RGB-Flow',
            load_pretrained_params=False, freeze_encoder_params=True,
            self_attention=True, importance_loss=False)
        # The official reps pathway never uses this raw-image encoder. DINO
        # extraction is frozen and performed separately with verified weights.
        self.core.encoder = nn.Identity()
        self.core.transEncoderFrame = TupleTransformerEncoder.from_encoder(
            self.core.transEncoderFrame)
        self.prototypes = nn.ParameterDict({
            str(i): nn.Parameter(torch.rand(1, 256)) for i in range(self.num_classes)})
        self.upstream_commit = json.loads(LOCK_PATH.read_text())['commit']

    def forward(self, rgb_features, flow_features, rgb_lengths=None, flow_lengths=None):
        if rgb_features.ndim != 3 or flow_features.ndim != 3:
            raise ValueError('Expected RGB and flow features [B, tokens, 384]')
        if rgb_features.shape[0] != flow_features.shape[0] or any(
                x.shape[-1] != self.rep_dim for x in [rgb_features, flow_features]):
            raise ValueError('RGB/flow batch size and DINO feature dimensions must agree')
        if not all(torch.isfinite(x).all() for x in [rgb_features, flow_features]):
            raise ValueError('RGB and optical-flow features must be finite')
        batch = rgb_features.shape[0]
        lengths = []
        masks = []
        for features, count in [(rgb_features, rgb_lengths), (flow_features, flow_lengths)]:
            n = features.shape[1]
            if count is None:
                count = torch.full((batch,), n, dtype=torch.long, device=features.device)
            else:
                count = torch.as_tensor(count, dtype=torch.long, device=features.device)
            if count.shape != (batch,) or (count < 1).any() or (count > n).any() or n > 2000:
                raise ValueError('Each modality needs 1-2000 valid tokens per clip')
            # The upstream model prepends one valid classification token.
            invalid = torch.arange(n+1, device=features.device)[None, :] > count[:, None]
            lengths.append(count.tolist())
            masks.append(invalid.unsqueeze(1))
        embedding, _ = self.core(
            rgb_features.unsqueeze(1).clone(), flow_features.unsqueeze(1).clone(),
            lengths[0], lengths[1], 'Prototypes', masks[0], masks[1],
            ['ArthroSkill']*batch)
        prototypes = torch.cat([self.prototypes[str(i)] for i in range(self.num_classes)])
        # Cosine logits at temperature 1 give the released prototype InfoNCE
        # objective with numerically stable cross-entropy.
        logits = F.normalize(embedding, dim=-1) @ F.normalize(prototypes, dim=-1).T
        return {'cls_logits': logits, 'embedding': embedding}

    def compute_loss(self, outputs, labels_cls):
        return F.cross_entropy(outputs['cls_logits'], labels_cls)
