# Aspect-ratio preprocessing sensitivity

Date: 2026-09-01

## Question

The principal three-seed baseline resizes every selected image directly to
224 x 224 pixels. This retains the full image but distorts rectangular relief
photographs. A matched sensitivity experiment tested whether aspect-preserving
resize-and-pad materially changes the result.

## Controlled change

Only the preprocessing geometry changed:

- **stretch baseline:** direct resize to 224 x 224;
- **pad sensitivity:** resize the longer edge to 224 and centre-pad the shorter
  dimension with RGB (124, 116, 104), approximately the ImageNet mean.

The frozen split, DINOv2 ViT-S/14 backbone, trainable head, optimiser,
validation-only checkpoint selection, and model seeds 24, 42, and 77 were held
constant. Test-set results were computed after the checkpoint for each seed had
been selected on validation macro-F1.

## Primary dependency-group classification

The primary test set contains 39 dependency groups. Image logits were averaged
within each group.

| Seed | Stretch accuracy | Pad accuracy | Difference | Stretch macro-F1 | Pad macro-F1 | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.821 | 0.795 | -0.026 | 0.824 | 0.800 | -0.023 |
| 42 | 0.821 | 0.872 | +0.051 | 0.814 | 0.874 | +0.060 |
| 77 | 0.846 | 0.795 | -0.051 | 0.841 | 0.801 | -0.040 |
| Mean +/- sample SD | 0.829 +/- 0.015 | 0.821 +/- 0.044 | -0.009 | 0.826 +/- 0.014 | 0.825 +/- 0.042 | -0.001 |

The pad-model percentile bootstrap intervals for accuracy were:

- seed 24: [0.667, 0.923];
- seed 42: [0.769, 0.974];
- seed 77: [0.667, 0.923].

The corresponding macro-F1 intervals were [0.655, 0.916], [0.749, 0.970],
and [0.657, 0.918].

## Secondary image and embedding diagnostics

| Evaluation | Stretch mean +/- SD | Pad mean +/- SD |
|---|---:|---:|
| Image accuracy (n=67) | 0.856 +/- 0.017 | 0.831 +/- 0.031 |
| Image macro-F1 (n=67) | 0.856 +/- 0.016 | 0.838 +/- 0.028 |
| Group nearest-centroid accuracy | 0.838 +/- 0.015 | 0.821 +/- 0.044 |
| Group nearest-centroid macro-F1 | 0.835 +/- 0.019 | 0.825 +/- 0.042 |
| Group retrieval Recall@1 | 0.872 +/- 0.051 | 0.803 +/- 0.039 |
| Group retrieval Recall@3 | 0.897 +/- 0.044 | 0.829 +/- 0.039 |
| Group retrieval Recall@5 | 0.897 +/- 0.044 | 0.863 +/- 0.015 |

## Interpretation

Aspect-preserving padding did not produce a consistent improvement. It helped
seed 42 but harmed seeds 24 and 77, leaving the mean group macro-F1 essentially
unchanged and increasing seed variability by roughly a factor of three. Its
retrieval scores were lower for every reported cutoff.

This is a sensitivity result, not a model-selection contest. The stretch
pipeline remains the principal baseline because it was the established
preprocessing protocol; choosing between the two after inspecting the test
results would introduce test-set selection. The defensible conclusion is that
the headline classification result is not driven solely by aspect-ratio
distortion, while individual model initialisations remain sensitive to the
preprocessing geometry. Neither transform is neutral: stretching distorts
shape, whereas padding introduces artificial borders and changes the scale of
relief detail.

## Machine-readable outputs

Pad runs:

- outputs/grouped_pad_seed24/eval/
- outputs/grouped_pad_seed42/eval/
- outputs/grouped_pad_seed77/eval/

Each directory contains image predictions, dependency-group predictions,
classification metrics with 10,000 bootstrap replicates, dependency-group
embeddings, centroid metrics, and retrieval results.
