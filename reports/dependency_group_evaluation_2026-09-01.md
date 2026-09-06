# Dependency-group evaluation audit (1 September 2026)

## Reason for the audit

The frozen experiment contains 189 filename-derived `Relief_ID` values, but
those identifiers are not uniformly equivalent to independent museum objects or
archaeological reliefs. The split construction joins linked objects and exact
duplicate imagery into 171 dependency groups. Evaluation by `Relief_ID` would
therefore count some linked observations more than once.

This audit makes the dependency group the primary statistical unit. It uses the
existing exported image logits and image embeddings; no model was retrained and
the frozen image split was not changed.

## Split composition

| Authority | Train groups (images) | Validation groups (images) | Test groups (images) | Total groups (images) |
|---|---:|---:|---:|---:|
| Ashurbanipal | 20 (194) | 15 (55) | 13 (28) | 48 (277) |
| Ashurnasirpal II | 35 (166) | 22 (48) | 16 (24) | 73 (238) |
| Sargon II | 25 (104) | 15 (30) | 10 (15) | 50 (149) |
| **Total** | **80 (464)** | **52 (133)** | **39 (67)** | **171 (664)** |

## Classification

Image logits were averaged within each dependency group. The class with the
largest mean logit was selected.

| Seed | Correct / 39 | Accuracy | Macro-F1 | Accuracy bootstrap 95% interval | Macro-F1 bootstrap 95% interval |
|---:|---:|---:|---:|---:|---:|
| 24 | 32/39 | 0.821 | 0.824 | [0.692, 0.923] | [0.681, 0.929] |
| 42 | 32/39 | 0.821 | 0.814 | [0.692, 0.923] | [0.672, 0.925] |
| 77 | 33/39 | 0.846 | 0.841 | [0.718, 0.949] | [0.703, 0.947] |
| **Mean +/- sample SD** | --- | **0.829 +/- 0.015** | **0.826 +/- 0.014** | --- | --- |

The bootstrap resamples dependency groups within one fixed test set. It
describes test-sample uncertainty conditional on that split; it is not a
confidence interval for all Neo-Assyrian reliefs.

## Embedding evaluation

Image embeddings were averaged and L2-normalised within each dependency group.
The training gallery contains 80 groups and the test query set contains 39.

| Seed | Nearest-centroid accuracy | Nearest-centroid macro-F1 | Recall@1 | Recall@3 | Recall@5 |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.846 | 0.850 | 0.923 | 0.949 | 0.949 |
| 42 | 0.821 | 0.814 | 0.821 | 0.872 | 0.872 |
| 77 | 0.846 | 0.841 | 0.872 | 0.872 | 0.872 |
| **Mean +/- sample SD** | **0.838 +/- 0.015** | **0.835 +/- 0.019** | **0.872 +/- 0.051** | **0.897 +/- 0.044** | **0.897 +/- 0.044** |

## Aggregation implementation correction

The branch named `mean_logits` in
`src/thesis_assyrian_relief/evaluation/relief_level.py` still selected the class
from mean probabilities. It now selects the class from mean logits and derives
confidence by applying softmax to the averaged logits. A regression test uses a
case where mean-logit and mean-probability predictions disagree.

## Reproducible outputs

For each seed, `outputs/grouped_seed<seed>/eval/` now contains:

- `test_dependency_group_predictions.csv`;
- `test_dependency_group_metrics.json`;
- `train_image_embeddings.pkl` and `test_image_embeddings.pkl`;
- `train_dependency_group_embeddings.pkl` and
  `test_dependency_group_embeddings.pkl`;
- `test_dependency_group_retrieval.csv`; and
- `test_dependency_group_embedding_metrics.json`.

The `outputs/` directory is intentionally Git-ignored because it contains model
checkpoints and generated arrays. This tracked report records the values used in
the thesis.
