# Experimental protocol

## Cohort and partitions

The cohort contains 196 procedures (104 knee, 92 shoulder) from Centres A/B/C (90/56/50). Four of 200 recordings were excluded before annotation: P009, P019, P047 and P138. Each recording represents a different patient. The fixed 117/39/40 partition was stratified by grade and joint with split seed 20241012. Use the workbook's exact assignment rather than assuming a newly generated stratified split will match it.

Two raters independently scored six dimensions. Consensus dimensions and total are their means. GRS targets use native points, with no sigmoid or output clipping. Classification and total regression remain separate heads; the total is not constrained to equal the sum of predicted dimensions. Smooth-L1 losses are summed over six dimensions and total, then averaged across the batch and added to cross-entropy with weight 1.

## Representation and training

Appearance uses torchvision's [3D Swin-T](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.swin3d_t.html), retaining the clip's temporal dimension. An explicit checkpoint interface accepts an adapted backbone state dictionary with its classifier removed. The study's Kinetics-400 → 30-epoch HeiChole adaptation → arthroscopy transfer requires the corresponding data and weights. Adaptation is not implicitly supplied by a library download.

Motion features are normalised path length, mean displacement, reciprocal mean third-difference magnitude, idle proportion below 5% of peak displacement, nonnegative bimanual velocity correlation and reciprocal spatial centre variance. These image-coordinate proxies depend on detection, tracking and camera/instrument arrangement. The encoder uses CNN channels 64/128/256, kernels 7/5/3 and a bidirectional LSTM with hidden size 128. Recurrent sequence tokens provide fusion keys and values.

Appearance provides the queries. Motion tokens are eligible when frame-level maximum detection confidence is at least 0.5. Missing detections have confidence zero. All-invalid rows bypass attention and retain the appearance residual, normalisation and feed-forward path. Detection confidence is not a calibrated probability of kinematic correctness.

Fine-tuning uses AdamW, learning rate 1e-4, weight decay 0.05, batch size 4, 100 epochs, five linear warm-up epochs and cosine annealing. Crop/flip/colour transformations are coherent within a clip and applied only during training. The earliest highest-accuracy validation checkpoint is retained. Main seed is 3407; repeats are 3433 and 2025. Test procedures do not select checkpoints.

## Leave-one-site-out data contract

Prepare separate `processed_loso/A`, `B` and `C` directories. Each contains `train/clips`, `val/clips`, `test/clips` and `detector_manifest.json`. Split complete procedures within the two training hospitals into stratified fitting/validation subsets. The third hospital supplies the test procedures only. Refit the detector for each fold before extracting its clips. Do not reuse a detector trained with held-out-site annotations.

The manifest records `training_video_ids`, `annotation_frames` and `weights_sha256`. The runner checks disjoint procedures and sites, detector training eligibility, validation-based selection and complete pooled coverage of 196 held-out procedures. The manifest documents eligibility but cannot independently prove how an external detector was trained. Exact original fold validation assignments and fitted weights are needed to reproduce the original folds.

## Statistical analyses

Procedure accuracy uses Wilson intervals; Pearson correlation uses Fisher-z intervals. MAE, RMSE, residual bias, ordinal metrics and paired accuracy effects use percentile procedure bootstrap intervals. Ordinal labels are 0/1/2. Balanced accuracy averages class recall; quadratic grade kappa is distinct from rater ICC and visibility annotation kappa. The rater ICC(A,1) uses two-way random-effects absolute agreement, removing both procedure and rater means from its ANOVA residual. Ratings are retained unchanged after the 32 quality-control reviews.

Seven paired main-model comparisons form one exploratory Holm family. Exact two-sided McNemar tests use discordant procedure predictions. Marginal bootstrap intervals are not simultaneous intervals or inversions of those tests.

Visibility includes 1,527 annotated clips from 40 test procedures. The analysis joins clips by `(video_id, clip_idx)`. Bootstrap draws retain whole procedures; paired model-label swaps use one sign for all clips from each procedure. Tier accuracies and gains are clip-weighted. Good/mild/severe tiers contain 910/411/206 clips from 40/40/39 procedures. Six tier-by-comparator tests form one Holm family. Plus-one corrected two-sided Monte Carlo p values use 99,999 swaps.

Severe-minus-good contrasts average the within-procedure gain differences over the 39 procedures with both tiers. Two sign-flip tests form a separate Holm family. Annotator-B sensitivity repeats the visibility analysis with unchanged membership and predictions. Random-stream consumption may change Monte Carlo p values slightly between equivalent implementations; resampling units and multiplicity families must remain fixed.

Full-cohort summaries contain fitted cases and are descriptive. Three training seeds reuse one fixed test partition; aggregate repeats cannot support case-level paired multi-seed tests. Statistical reanalysis of released predictions does not establish independently reproduced training performance. Public English source tables and recorded-summary caveats are documented in [the data package](../data/source_data/README.md).

## Comparator boundaries

The external Video Swin and 3D ResNet runners use torchvision architectures and public initialisations. Numerical agreement with the original fitted baselines requires their exact weights and configurations. The SAIS adapter invokes the pinned official RGB-and-flow model, uses its shared temporal encoder and trains learned category prototypes. Frozen DINO features and RAFT flow images are prepared on the same source-frame clip inventory. The study-specific runner uses AdamW with constant learning rate, validation procedure accuracy for selection and procedure probability averaging. See [SAIS integration](sais.md) for exact source provenance, extraction settings, implementation boundaries and commands. The earlier RGB-only proxy is no longer used. Existing source-data comparisons do not constitute results from newly fitting this integration.
