# Official RGB-and-flow SAIS integration

This integration invokes the released SAIS `fullModel` implementation from an external, pinned checkout. It replaces the earlier RGB-only proxy. Upstream source and pretrained model files are not redistributed in this repository.

## Sources and version

The upstream repository is [danikiyasseh/SAIS](https://github.com/danikiyasseh/SAIS), pinned to commit `3dd073492887fff99671df3bdb42cbc39066ab8c`. File checksums are recorded in [sais_upstream.lock.json](../configs/sais_upstream.lock.json). The adapter refuses mismatched model or DINO source files.

The method is described in [Kiyasseh et al., Nature Biomedical Engineering (2023)](https://www.nature.com/articles/s41551-023-01010-8). The relevant implementation paths are:

| Component | Released source |
| --- | --- |
| Shared temporal Transformer, RGB/flow addition and projection | [prepare_model.py](https://github.com/danikiyasseh/SAIS/blob/3dd073492887fff99671df3bdb42cbc39066ab8c/SAIS/scripts/prepare_model.py) |
| Prototype objective and probability calculation | [prepare_miscellaneous.py](https://github.com/danikiyasseh/SAIS/blob/3dd073492887fff99671df3bdb42cbc39066ab8c/SAIS/scripts/prepare_miscellaneous.py) |
| Frozen DINO spatial encoder and RAFT flow visualisation | [extract_representations.py](https://github.com/danikiyasseh/SAIS/blob/3dd073492887fff99671df3bdb42cbc39066ab8c/SAIS/scripts/extract_representations.py) |
| Released Custom flow spacing | [generate_paths.py](https://github.com/danikiyasseh/SAIS/blob/3dd073492887fff99671df3bdb42cbc39066ab8c/SAIS/scripts/generate_paths.py) |

SAIS has separate [license and use terms](https://github.com/danikiyasseh/SAIS#license-and-terms-of-use), including CC-BY-NC-ND 4.0 restrictions. This project's MIT license does not relicense SAIS, its weights or derived model artifacts. Check the upstream terms for the intended use and distribution. Local external checkouts and generated weights are ignored by Git.

## Model and objective

Both modalities use the same frozen official DINO ViT-S/16 spatial encoder, producing 384-dimensional features. The official SAIS model adds learned temporal positions and a classification token, processes each modality through its shared four-layer temporal Transformer, and adds the two modality representations. Its prototype pathway returns a 256-dimensional embedding. Three learned prototypes represent novice, intermediate and expert.

Logits are cosine similarities to the prototypes, at temperature 1. Softmax gives the class probabilities. Cross-entropy on these logits is the numerically stable form of the released prototype InfoNCE loss. No ordinary linear grade head or GRS regression head substitutes for that pathway.

The external model expects its Transformer to return an `(encoded, attention)` tuple. A local subclass supplies that interface while preserving encoded values and state-dict names. It does not modify PyTorch's installation or global Transformer API. DINO-local helper imports are resolved inside the external module without shadowing this project's `utils` package. Attention maps are not exported by this compatibility layer. The unused raw-image encoder is removed from the official `reps` pathway; spatial features are generated separately by the actual frozen DINO encoder.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/setup_sais.py --destination external/SAIS
```

The setup command checks out the pinned official source and verifies its file hashes. It refuses to replace an existing directory. For an existing matching checkout, pass `--sais-root /path/to/SAIS` to the commands below.

Feature extraction additionally needs OpenCV and PTLFlow:

```bash
python -m pip install -r requirements-sais-extraction.txt
```

The optional file pins PTLFlow 0.4.2 for the current API. The adapter also accepts the original SAIS PTLFlow 0.2.5 IO interface when run in a compatible environment. Versions and checkpoint hashes are written into the cache. Matching the historical extraction environment requires its original versions and weights; compatibility with a newer IO API is not a claim of bitwise identity to PTLFlow 0.2.5.

Obtain the [official pretrained DINO ViT-S/16 weights](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth) and a trusted local PTLFlow RAFT `things` checkpoint. The feature command requires both files and loads their parameters strictly. It does not silently generate random spatial or flow features. RAFT checkpoints from torchvision have a different contract and cannot be substituted directly.

## Shared clips and feature preparation

Prepare the existing study metadata and main-model clip inventory first, as described in [README.md](../README.md). The SAIS command reads each clip's `video_id`, grade, partition and exact `source_frame_indices`; it decodes those same source frames from the recording. It does not infer raw pixels from per-video standardised tensors, invent instrument motion as optical flow or resample a new procedure split.

```bash
python scripts/extract_sais_features.py --config configs/config.yaml --metadata metadata.csv --clips processed --output sais_features --sais-root external/SAIS --dino-weights weights/dino_deitsmall16_pretrain.pth --raft-weights weights/raft_things.ckpt --device cuda
```

Frames are decoded at the configured 25 fps and rescaled to 640 by 480. The released Custom DINO preprocessing uses an 80% centre crop, resize to 224 by 224 and ImageNet channel normalisation. Flow uses RAFT on the corresponding full-resolution frame pair, PTLFlow RGB colour encoding and the released flow-image JPEG boundary before the same DINO preprocessing.

`flow_step: 15` follows the released Custom helper's pair spacing. In the matched 16-frame clip protocol, it yields one flow map from retained-frame positions 0 and 15. Other positive steps below 16 can be configured before extraction. Original source indices of both endpoints are retained, including gaps caused by quality screening. The ArthroSkill-MC manuscript does not report the original comparator's flow stride or exact cache settings, so this explicit choice must not be presented as a recovered historical run configuration. Pairing is restricted to the shared input clip rather than using the upstream whole-video flow grid.

The output directory must be new. Each feature payload contains `rgb_features`, `flow_features`, `video_id`, `label_cls`, source indices and flow pairs. A manifest records clip checksums, source clip checksums, source-metadata checksum, source revision, preprocessing settings, dependency versions and weight checksums. Missing modalities, inconsistent grades, overlapping clips, changed payloads and cross-partition procedures raise errors.

Frozen deterministic features are cached once. Training does not apply the main model's image augmentation settings to those feature vectors. Additional temporal views or augmented feature caches would define a different protocol and require explicit preparation and documentation.

## Training and checkpoint selection

```bash
python scripts/train_sais.py --config configs/config.yaml --data sais_features --sais-root external/SAIS --output outputs/sais/seed3407 --seed 3407
```

The runner requires the exact released 117/39/40 procedure assignments and grades in the cache manifest, not only matching partition counts. A training output directory must be empty so one seed run cannot silently overwrite another. It fits only the training features and selects the earliest checkpoint attaining the highest validation procedure accuracy. AdamW uses learning rate 1e-4, weight decay 0.05, batch size 4 and 100 epochs. The learning rate stays constant, with no cosine schedule or warm-up, as specified for SAIS in the ArthroSkill-MC manuscript. These are the study's optimisation settings; the public SAIS generic runner instead constructs an SGD optimiser and uses different selection rules.

Run seeds 3433 and 2025 separately for the planned repeats. The selected checkpoint stores both the official temporal-model state and learned prototypes, together with its selected epoch, configuration and feature provenance. `sais_training.json` records per-epoch training loss, validation procedure metrics and learning rate. Test predictions are not computed during training.

## Procedure-level test evaluation

```bash
python scripts/evaluate_sais.py --data sais_features --sais-root external/SAIS --checkpoint outputs/sais/seed3407/sais_best_model.pth --split test --output outputs/sais/seed3407/test
```

Evaluation defaults to `test`. For each clip it computes the prototype softmax probabilities; for each procedure it averages those probabilities before argmax. Clips do not receive independent procedure-level votes. The output also retains individual clip probabilities, clip identifiers, original source-frame indices and flow endpoints for traceable paired evaluation. Procedure output contains paired procedure identifiers, three-grade probabilities, predictions, accuracy with a Wilson 95% interval, balanced accuracy, confusion matrix, ordinal error distances, grade kappa when defined, and calibration error. It reports classification only; no GRS MAE, RMSE or correlation is invented for SAIS.

The manifest checksum must match the selected training run. This prevents accidentally evaluating against a different feature cache. The `per_procedure` IDs and class coding match the main model's evaluation JSON, allowing later paired analyses on the same procedures.

## Verification and limits

```bash
python -m unittest discover -s tests -v
```

With the pinned external checkout present, integration tests exercise the actual official temporal model, compare the prototype loss with the released function, verify nonzero gradients through both input modalities and the learned prototypes, check padding invariance, and run a short synthetic training/checkpoint/test cycle. Additional tests check flow-pair alignment, cache corruption and procedure leakage. DINO loading and freezing are tested with an explicitly synthetic checkpoint. The legacy/current RAFT IO interfaces are checked with lightweight mock backends; pretrained RAFT inference on study recordings is not part of these tests. Without an external checkout, the official-model integration tests are explicitly skipped.

Synthetic fixtures test the software contract and do not establish study accuracy. Real-video extraction and a new 100-epoch study run still require authorised recordings, pretrained DINO/RAFT files and the intended compute environment. Existing source-data SAIS results in [RESULTS.md](../RESULTS.md) remain recorded study outputs, not results of this new integration.
