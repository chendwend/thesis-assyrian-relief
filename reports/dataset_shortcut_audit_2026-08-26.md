# Frozen-split duplicate and shortcut audit

Date: 2026-08-26

Manifest: `data/splits/image_level_dataset_v2_grouped.csv`

Images: 664 (464 train, 133 validation, 67 test)

Reproduce with:

```bash
.venv/bin/python scripts/audit_dataset_shortcuts.py
```

The machine-readable outputs are written to `outputs/dataset_audit/`, which is
ignored by Git because the repository's `outputs/` tree also contains model
checkpoints.

## Cross-split duplicate result

- Exact SHA-256 matches: 0.
- DCT pHash pairs within Hamming distance 6: 0.
- Credible automated near-duplicate matches: 0.
- One review-only dHash flag compared `BM 124542-2.jpg` (train) with
  `BM 124583-2.jpg` (validation). The pHash distance was 28 and resized
  greyscale correlation was -0.114. Visual inspection showed unrelated scenes,
  so this is a hash collision rather than leakage.

These thresholds do not rule out partial-scene overlaps or substantially
different photographs of the same archaeological panel.

## Source association

The normalized mutual information between museum/source prefix and ruler label
is 0.350. The strongest imbalances are:

- 100 of 104 AO images are Sargon II;
- 273 of 277 Ashurbanipal images have a BM prefix; and
- 19 of 21 MET images are Ashurnasirpal II.

The source-majority baseline learns the most frequent training ruler for each
prefix and uses the global training majority for unseen prefixes.

| Baseline | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Source-prefix majority | 0.647 | 0.618 | 0.672 | 0.664 |
| Geometry and file size | 0.451 | 0.426 | 0.522 | 0.521 |
| Border statistics | 0.436 | 0.425 | 0.433 | 0.413 |
| Combined acquisition features | 0.534 | 0.538 | 0.552 | 0.551 |

The full seed-42 image classifier reaches 0.866 test accuracy, so it is not
reducible to these simple baselines. The baselines nevertheless demonstrate
that the frozen benchmark contains substantial non-stylistic label signal.

## Thesis-facing interpretation

The corrected grouped split has no detected exact or declared-threshold
perceptual leakage. Its accuracy still cannot be interpreted as a pure measure
of carving style because museum/source, framing, image geometry, and acquisition
statistics are predictive. Source-balanced or leave-one-source-out evaluation,
foreground segmentation, inscription masking, aspect-preserving preprocessing,
and partial-scene matching are the next robustness tests.
