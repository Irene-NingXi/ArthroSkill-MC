# Results

Results below are computed from the [public English source tables](data/source_data/README.md), which preserve the source workbook's numerical records. They describe its recorded predictions and are not newly trained model results. The full aggregate reanalysis is in [source_data_summary.json](docs/source_data_summary.json).

## Fixed test partition (40 procedures)

| Model | Correct | Accuracy | GRS MAE |
| --- | ---: | ---: | ---: |
| ArthroSkill-MC | 35/40 | 87.50% | 1.537 |
| SAIS | 25/40 | 62.50% | Not reported |
| Video Swin-T | 26/40 | 65.00% | 2.639 |
| 3D-ResNet-18 | 24/40 | 60.00% | 2.468 |
| Appearance only | 28/40 | 70.00% | 1.954 |
| Motion only | 25/40 | 62.50% | 2.206 |
| Naive concat | 27/40 | 67.50% | 2.019 |
| Unmasked fusion | 26/40 | 65.00% | 2.211 |

SAIS regression fields present in the source workbook are retained in the public data for audit, but are not treated as validated comparator endpoints in the manuscript-facing analysis.

The complete model has Pearson r = 0.957, RMSE = 1.765 GRS points, and a Wilson accuracy interval of 73.9–94.5%. All five test errors involve adjacent grades. Full-cohort accuracy is 162/196 (82.65%) and includes training cases.

## Paired exploratory comparisons

| Comparator | Main-only / comparator-only correct | Exact McNemar p | Holm p |
| --- | ---: | ---: | ---: |
| SAIS | 13 / 3 | 0.02127 | 0.12762 |
| Video Swin-T | 13 / 4 | 0.04904 | 0.19617 |
| 3D-ResNet-18 | 13 / 2 | 0.00739 | 0.05170 |
| Appearance only | 10 / 3 | 0.09229 | 0.19617 |
| Motion only | 13 / 3 | 0.02127 | 0.12762 |
| Naive concat | 11 / 3 | 0.05737 | 0.19617 |
| Unmasked fusion | 13 / 4 | 0.04904 | 0.19617 |

None of the seven comparisons has Holm-adjusted p < 0.05. Marginal bootstrap effect intervals do not replace that multiplicity-adjusted inference.

## Visibility strata

| Severity | Clips / procedures | Complete | Naive concat | Unmasked fusion |
| --- | ---: | ---: | ---: | ---: |
| 0 | 910 / 40 | 89.8% | 84.8% | 84.0% |
| 1 | 411 / 40 | 88.3% | 73.5% | 72.7% |
| 2 | 206 / 39 | 71.8% | 54.4% | 46.1% |

Whole-procedure swaps retain within-procedure dependence. All six tier comparisons remain below 0.05 after Holm correction. The severe-minus-good mean gain contrast is 20.8 percentage points against unmasked fusion and 8.4 points against naive concatenation; only the former is significant after the separate two-test Holm adjustment. Absolute complete-model accuracy decreases in severe degradation. The workbook describes uniform temporal annotation sampling without image-quality selection; original frame endpoints for annotated clips are unavailable.

Monte Carlo p values and percentile interval endpoints may differ slightly from the manuscript because random-stream consumption and available score precision differ; exact counts, point estimates and exact McNemar tests are directly reproducible. The JSON records the calculation seed, resampling counts and workbook checksum.

## Independent ratings and annotation agreement

Rater ICC(A,1) = 0.822885; 32 procedures met the six-point review threshold. Original independent scores are retained. Visibility severity quadratic kappa = 0.937, linear kappa = 0.909; type agreement = 95.3%.

## Training repeats

| Model | Test accuracy mean | Sample SD |
| --- | ---: | ---: |
| ArthroSkill-MC | 88.33% | 3.82 percentage points |
| SAIS | 63.33% | 3.82 percentage points |

Seeds are 3407, 3433 and 2025 on one fixed test partition; seed 3407 is the designated main run. Repeats are aggregate records and do not supply case-level paired multi-seed inference.

## Leave-one-site-out

| Held-out hospital | Correct / procedures | Accuracy | GRS MAE |
| --- | ---: | ---: | ---: |
| Centre A | 56 / 90 | 62.2% | 4.38 |
| Centre B | 37 / 56 | 66.1% | 4.60 |
| Centre C | 33 / 50 | 66.0% | 4.15 |
| pooled | 126 / 196 | 64.3% | 4.38 |

LOSO uses a separate fitting protocol; its folds share training hospitals and are not independent training replications. Reproduction of new fitted models requires recordings, adapted and detector weights, and matching fold configurations.
