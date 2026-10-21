# ArthroSkill-MC

Occlusion-aware multimodal skill assessment across arthroscopic joints and hospitals.

This repository provides a PyTorch implementation of a Video Swin-T appearance encoder, instrument-motion encoder, confidence-masked cross-attention and separate grade/GRS heads. It also provides procedure-level evaluation and source-data statistical reanalysis.

The benchmark contains 196 knee and shoulder procedures from three hospitals. The fixed partition contains 117 training, 39 validation and 40 test procedures. Skill labels are novice, intermediate and expert. Six GRS dimensions range from 1 to 5; the total ranges from 6 to 30.

## Installation

```bash
python -m pip install -r requirements.txt
```

Use a compatible PyTorch/torchvision pair. CUDA is optional for statistical reanalysis; training full-resolution video requires an appropriate GPU.

## Source-data analysis

```bash
python scripts/analyze_predictions.py --xlsx /path/to/ArthroSkill-MC_Data_V5.xlsx --output outputs/source_data_analysis.json
```

The command recomputes results from procedure predictions, original ratings and annotated clip predictions. Defaults are 20,000 procedure-bootstrap samples, 99,999 procedure-level swaps and analysis seed 20261020. Visibility annotation-agreement intervals use seed 20261021. Seven fixed-test comparisons and six visibility comparisons receive separate Holm adjustments. Training repeats remain aggregate summaries of the same partition.

See [RESULTS.md](RESULTS.md) for source-data results and [the protocol](docs/protocol.md) for definitions and reproducibility boundaries. Statistical reanalysis does not train a model.

## Prepare training data

The source recordings are restricted under institutional ethics approvals. The workbook contains analysis tables rather than training videos.

```bash
python scripts/prepare_metadata.py --xlsx /path/to/ArthroSkill-MC_Data_V5.xlsx --videos /path/to/videos --output metadata.csv
python scripts/extract_motion.py --metadata metadata.csv --weights /path/to/training_only_detector.pt --output processed
```

Metadata export preserves the supplied procedure partition and averages the two raters' six dimension scores. Motion extraction checks 25 fps input, screens brightness below 30 and Laplacian variance below 100, performs per-video intensity normalisation and writes non-overlapping 16-frame clips with original frame indices. Train the arthroscopy detector on the eligible training procedures before extraction; generic object-detection weights do not establish instrument-detection validity.

Set `model.appearance.checkpoint` in `configs/config.yaml` to a HeiChole-adapted backbone state dictionary. A null checkpoint uses Kinetics-400 initialisation and omits the surgical adaptation stage. Checkpoints from the earlier 2D backbone are incompatible with the 3D backbone and require retraining.

## Training and evaluation

```bash
python train.py --config configs/config.yaml --data processed --output outputs/main
python evaluate.py --config configs/config.yaml --data processed --checkpoint outputs/main/best_model.pth --split test --output outputs/test
python scripts/run_ablation.py --config configs/config.yaml --data processed --variant unmasked --output outputs/unmasked
python scripts/train_video_baselines.py --config configs/config.yaml --data processed --model resnet3d --pretrained --output outputs/resnet3d
python scripts/train_loso.py --config configs/config.yaml --data processed_loso --output outputs/loso
```

Ablation choices are `appearance_only`, `motion_only`, `naive_concat`, `unmasked` and `full`. Shared internal appearance branches must use the same adapted checkpoint. External Video Swin-T uses Kinetics-400 weights. The legacy RGB-only SAIS proxy is not the RGB-and-optical-flow comparator; its training/evaluation commands stop with an explicit explanation. Use the official SAIS code and recipe for that comparison.

Validation selects the best checkpoint using procedure accuracy. Each procedure's clip softmax probabilities are averaged before argmax; continuous scores are averaged independently. Evaluation defaults to the test partition. All-invalid motion clips use the appearance residual pathway and never evaluate an all-masked softmax.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover masked attention, all-invalid fallback, gradients, probability aggregation, score units and the spatiotemporal backbone. They do not establish the reported trained-model performance.

## Data and code availability

Reproducing numerical tables requires the source-data workbook. Independent end-to-end reproduction additionally requires authorised recordings, detector annotations and weights, the HeiChole adaptation checkpoint and matching fitted model checkpoints/configurations. No surgical recording or individual source-data table is distributed here.

## Author and licence

Ning Xi. MIT licence; see [LICENSE](LICENSE). Third-party datasets, weights and implementations retain their respective terms.
