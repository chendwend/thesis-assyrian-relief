# Submission controls audit - 6 September 2026

Recomputed from the frozen manifest and saved normalized dependency-group embeddings; no retraining.
Manifest SHA-256: `4942ec4984fa5a5b261cd9b2c7cc141e37a17ae84d82cf0e41becd83d518690e`.

## Retrieval

The evaluator uses all 80 distinct training groups and excludes none. All 39 test groups are queries.
The original `Recall@k` is a query-level same-label hit rate, also called Hit@k here; it is not the fraction of all relevant gallery objects retrieved.
Random rankings sample distinct gallery groups without replacement. For each query class c, the exact hit probability is 1 - C(80-n_train_c,k)/C(80,k), then weighted by test class frequency.

| k | Random Hit@k | Learned Hit@k, mean +/- sample SD | Random expected P@k | Learned P@k, mean +/- sample SD |
|---|---|---|---|---|
| 1 | 0.342949 | 0.872 +/- 0.051 | 0.342949 | 0.872 +/- 0.051 |
| 3 | 0.708433 | 0.897 +/- 0.044 | 0.342949 | 0.855 +/- 0.017 |
| 5 | 0.866318 | 0.897 +/- 0.044 | 0.342949 | 0.846 +/- 0.010 |

P@k is the proportion of the first k neighbours sharing the query label. Random probabilities are descriptive expectations, not significance tests. Seeds share the same split and queries.

## Comparable source-only baseline

Each training dependency group contributes one vote to its source prefix. Every group has exactly one source prefix. Unseen sources use the overall training-group majority; ties use lexical label order.
Training mapping: `{'AN': 'Ashurnasirpal II', 'AO': 'Sargon II', 'BM': 'Ashurnasirpal II', 'BOS': 'Ashurnasirpal II', 'BRK': 'Ashurnasirpal II', 'HM': 'Sargon II', 'MET': 'Ashurnasirpal II'}`. Fallback: Ashurnasirpal II.
Test result: 22/39; accuracy 0.564103; macro-F1 0.449495.
Mean full-model accuracy minus source baseline: 0.264957; paired group-bootstrap 95% interval [0.136752, 0.410256].
The bootstrap uses 10,000 replicates, seed 20260906, resampling 39 groups jointly across source predictions and the three fitted models. It is conditional on this split and these fitted models. It does not isolate carving style or establish transfer to new sources.

## Exact numeric correction

The saved seed-24 macro-F1 upper bootstrap bound is 0.9294947121034077, which rounds directly to 0.929 at three decimals. Rounding first to 0.9295 and then to three decimals would incorrectly produce 0.930.

## Reproduction and checks

`python scripts/audit_submission_controls.py --root .`
Assertions verify the frozen hash, group identities and labels, single-source membership, normalized vectors, saved top-one neighbour identities, and all saved Hit@k metrics. Full original-space rankings are saved for all queries, not just selected errors.
Companion files contain input hashes, all neighbour ranks/cosine similarities, paired predictions, and source-by-monarch group counts.
