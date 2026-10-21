# ArthroSkill-MC public source data

This package provides the publicly shareable, de-identified source tables from the V5 study workbook. It supports reanalysis of ratings, fixed-split predictions, visibility annotations, aggregate training repeats and leave-one-site-out predictions. It does not contain surgical recordings or fitted checkpoints.

Release identifier: `v5-public-1`. The [manifest](manifest.json) records the original workbook checksum, exported row counts and file checksums. The [data dictionary](data_dictionary.json) defines every exported column, its type and source worksheet ordinal. All 22 original worksheets are represented through primary data or clearly labelled recorded summaries.

## Primary data

| File | Rows | Observation unit | Join key |
| --- | ---: | --- | --- |
| [data_split.csv](data_split.csv) | 196 | Included procedure | video_id |
| [rater_scores.csv](rater_scores.csv) | 196 | Procedure with two independent raters | video_id |
| [procedure_predictions.csv](procedure_predictions.csv) | 1,568 | Procedure and model | video_id, model |
| [visibility_annotations.csv](visibility_annotations.csv) | 1,527 | Annotated test clip | video_id, clip_idx |
| [clip_predictions.csv](clip_predictions.csv) | 4,581 | Annotated clip and model | video_id, clip_idx, model |
| [loso_predictions.csv](loso_predictions.csv) | 196 | Procedure tested with its hospital held out | video_id |
| [training_runs.csv](training_runs.csv) | 6 | Recorded model and training seed | model, seed |
| [exclusions.csv](exclusions.csv) | 4 | Recording excluded before annotation | video_id |

The fixed partition is 117 training, 39 validation and 40 test procedures. Model records cover eight models/variants and all 196 included procedures. Full-cohort summaries include fitted cases. The visibility records contain three paired model predictions per annotated clip and cover the same 40 test procedures. LOSO is a separate fitting protocol, not the fixed-split test output.

## Definitions and units

- `video_id` retains the study procedure pseudonym so tables can be joined. It is not a released patient-identity key. Hospitals remain Centre A, Centre B and Centre C, without a hospital-name mapping.
- Grades are `novice`, `intermediate` and `expert`, with ordinal codes 0, 1 and 2 for error-distance calculations. Grade predictions come from classification outputs and are not reconstructed from GRS predictions.
- Each independent GRS dimension is an integer from 1 to 5. Totals range from 6 to 30. Consensus totals are the mean of the two independent totals and retain 0.5-point granularity. Dimension numbers 1-6 retain the source workbook's ordering; this export does not invent an index-to-name mapping absent from that table.
- `rater_difference` is signed rater 1 minus rater 2. The absolute six-point review threshold identifies 32 procedures. The manuscript reports that review retained the independent scores unchanged; the numerical release preserves the recorded independent scores.
- `pred_grs` preserves the recorded prediction, usually rounded to three decimals. Recorded `residual` and `abs_error` fields often have greater precision. They are retained independently, rather than forced to equal arithmetic on the rounded prediction. Predictions outside the 6-30 reference range are not clipped.
- Visibility severity is 0 for good visibility, 1 for mild degradation and 2 for severe degradation. Final severity coincides with annotator A in every released clip. Annotator B enables sensitivity analysis. Degradation categories are `none`, `blood`, `blur`, `fog`, `out_of_view` and `overexposure`.
- `clip_idx` is a zero-based clip identifier within a procedure, not a source-video frame number. Visibility sampling ratios refer to quality-screened inference clips; they are not prevalence estimates in unfiltered recordings.
- Clip-table model `Complete` denotes ArthroSkill-MC. The other two clip models are `Naive concat` and `Unmasked fusion`.
- Accuracies and sampling ratios are fractions unless a field explicitly contains source percentage text. Percentages in `recorded_summaries/degradation_counts.csv` use the 0-100 scale.
- Empty CSV cells mean missing or not reported. Missing SAIS regression metrics in aggregate repeat records are not zeros. Boolean columns use literal `true` and `false`; the recorded LOSO correctness column retains its original numeric 0/1 coding.

## Recorded summaries and discrepancies

The 16 files in [recorded_summaries](recorded_summaries) preserve supplementary source snapshots, including test/validation details, model metrics, statistical tests, visibility summaries, scoring notes, ICC working values, LOSO summaries and the workbook overview/report. These are values-only exports with English labels and translations. They are retained for traceability and do not override the primary observations.

The following source differences are explicit:

1. The original statistical-test table flags several unadjusted p values as significant. None of the seven main-model comparisons is below 0.05 after Holm correction. Both the original flags and the current adjusted analysis remain available with their respective meanings.
2. Original visibility-summary p values have no test method documented in that sheet. They are not substituted for the whole-procedure bootstrap and paired-swap analysis in the manuscript protocol.
3. The cohort overview records ICC 0.822, the ICC calculation snapshot records 0.8229, and independent-score reanalysis gives about 0.822885, reported as 0.823. Source values are preserved rather than edited to agree.
4. The workbook contains SAIS total-score fields and aggregate regression metrics, while the manuscript comparator and official SAIS integration report classification only. Those source values remain available for audit. The manuscript-facing analysis does not treat them as validated SAIS regression endpoints.
5. The LOSO summary records Centre B MAE as 4.59. Reanalysis of its rounded procedure predictions yields approximately 4.60. Both records retain their provenance.
6. The experiment-report snapshot contains an approximately 48-million raw-frame claim, an approximately 99.5% filtering claim, an unadjusted significance statement and a broad severity-trend statement. These are source report text, not independently verified metadata or conclusions adopted by this release. The manuscript treats the raw recording total and retention fraction as unverified and uses procedure-level severity contrasts.
7. The source file is labelled V5 while its internal report contains a v6.0 label and administrative date/status fields. These are recorded source fields, not asserted publication dates or repository history.

No study numerical observation was changed during export. Repeated tables and notes were translated or relabelled, a surgeon-linkage field was removed, and missing-value/Boolean encodings were standardised as documented in the manifest.

## Reanalysis and validation

From the repository root:

```bash
python scripts/validate_source_data.py --data data/source_data
python scripts/analyze_predictions.py --data data/source_data --output outputs/source_data_analysis.json
```

The analysis defaults to 20,000 procedure-bootstrap draws, 99,999 whole-procedure swaps and seed 20261020. Visibility-annotation agreement intervals use seed 20261021. It performs the seven-comparison Holm family, the six visibility-tier comparison family and the separate two-comparator severe-minus-good contrast family. Clips are not independent resampling units.

To verify against an authorised local copy of the original workbook:

```bash
python scripts/validate_source_data.py --data data/source_data --xlsx /path/to/ArthroSkill-MC_Data_V5.xlsx
```

That check verifies the workbook checksum and exact primary-table equivalence after the documented surgeon-field omission, field translations and missing-value conversion. The original workbook is not modified. Export logic is available in [export_source_data.py](../../scripts/export_source_data.py).

Metadata for authorised model training can also be constructed without the workbook:

```bash
python scripts/prepare_metadata.py --data data/source_data --videos /path/to/authorised/videos --output metadata.csv
```

This preserves the released split and averages the paired dimension ratings. Access to these analysis tables does not provide access to the required training videos.

## Figure and table source mapping

| Manuscript component | Released numerical source |
| --- | --- |
| Cohort composition and annotation workflow | data_split.csv, rater_scores.csv, exclusions.csv |
| Fixed-test GRS regression | procedure_predictions.csv, filtered to ArthroSkill-MC and test |
| Model/ablation and subgroup comparison | procedure_predictions.csv joined to data_split.csv |
| Fixed-test confusion matrix and error flows | procedure_predictions.csv, filtered to ArthroSkill-MC and test |
| Visibility strata and annotation agreement | visibility_annotations.csv and clip_predictions.csv |
| Training-seed repeats | training_runs.csv; only aggregate repeat outputs were supplied |
| LOSO main-model results | loso_predictions.csv; other models' case-level LOSO outputs were not supplied |

Representative surgical images, original recordings, instrument boxes/trajectories, per-dimension model predictions, case-level outputs for additional seeds and fitted model checkpoints are not present in this workbook-derived package. They are not fabricated from the available tables.

## Withheld information and reuse

The source `surgeon_id` linkage column is not released. It is unnecessary for the reported procedure-level reanalysis, and its codes were unique per procedure in this workbook. No replacement grouping or real-identity mapping is invented. Original workbook properties, formatting and non-tabular internals are omitted. Surgical recordings and any patient, rater, surgeon or hospital identity keys are not distributed.

The tables are publicly downloadable. Data-specific reuse rights are not newly assigned by this export; see [TERMS.md](TERMS.md). The repository's MIT software licence does not override institutional or third-party rights. No DOI has been assigned to this GitHub data release. Cite a specific repository commit/version when using these tables, and cite the associated study when its bibliographic record is available.
