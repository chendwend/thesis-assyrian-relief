"""Reproduce final group retrieval/source controls from frozen saved outputs.

No images are requested, no model is trained, and no split/checkpoint is changed.
Run: python scripts/audit_submission_controls.py --root .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / 'data/splits/image_level_dataset_v2_grouped.csv'
    components_path = root / 'data/splits/image_level_dataset_v2_grouped_components.csv'
    expected_hash = '4942ec4984fa5a5b261cd9b2c7cc141e37a17ae84d82cf0e41becd83d518690e'
    assert sha256(manifest) == expected_hash, 'Frozen manifest changed'
    components = pd.read_csv(components_path)
    assert len(components) == 171 and components.component_key.is_unique
    components['sources'] = components.filenames.map(
        lambda value: sorted({name.split()[0] for name in value.split(';')}))
    assert components.sources.map(len).eq(1).all(), 'Specify multi-source group treatment first'
    components['source'] = components.sources.map(lambda value: value[0])
    train = components.loc[components.split.eq('train')].copy()
    test = components.loc[components.split.eq('test')].sort_values('component_key').copy()
    assert len(train) == 80 and len(test) == 39
    assert set(train.component_key).isdisjoint(test.component_key)
    classes = sorted(components.Authority.unique())
    train_counts = train.Authority.value_counts().sort_index().to_dict()
    test_counts = test.Authority.value_counts().sort_index().to_dict()
    # Source-majority rule has one vote per TRAINING GROUP. Pandas mode breaks
    # any tie by lexical label order; unseen source uses training-group majority.
    fallback = str(train.Authority.mode().iat[0])
    mapping = train.groupby('source').Authority.agg(lambda s: str(s.mode().iat[0])).to_dict()
    source_pred = test.source.map(mapping).fillna(fallback).to_numpy()
    truth = test.Authority.to_numpy()
    source_correct = source_pred == truth
    source_metrics = {
        'n_test_groups': len(test), 'mapping': mapping, 'fallback': fallback,
        'correct': int(source_correct.sum()),
        'accuracy': float(accuracy_score(truth, source_pred)),
        'macro_f1': float(f1_score(truth, source_pred, labels=classes, average='macro')),
    }
    ks = (1, 3, 5)
    chance = {
        str(k): float(sum(test_counts[c] / len(test) *
             (1 - math.comb(len(train) - train_counts[c], k) / math.comb(len(train), k))
             for c in classes)) for k in ks
    }
    chance_precision = float(sum(test_counts[c] / len(test) * train_counts[c] / len(train)
                                 for c in classes))
    metrics_rows, rank_rows, comparisons, input_hashes, learned_correct = [], [], [], {}, []
    for seed in (24, 42, 77):
        eval_dir = root / f'outputs/grouped_seed{seed}/eval'
        gallery_path = eval_dir / 'train_dependency_group_embeddings.pkl'
        query_path = eval_dir / 'test_dependency_group_embeddings.pkl'
        pred_path = eval_dir / 'test_dependency_group_predictions.csv'
        metric_path = eval_dir / 'test_dependency_group_embedding_metrics.json'
        gallery, queries = pd.read_pickle(gallery_path), pd.read_pickle(query_path)
        assert len(gallery) == 80 and len(queries) == 39
        assert set(gallery.component_key) == set(train.component_key)
        assert set(queries.component_key) == set(test.component_key)
        assert gallery.relief_id.is_unique and queries.relief_id.is_unique
        for frame, expected in ((gallery, train), (queries, test)):
            aligned = expected.set_index('component_key').loc[frame.component_key]
            assert list(aligned.Authority) == list(frame.authority)
        gallery_emb = np.stack(gallery.embedding)
        assert np.allclose(np.linalg.norm(gallery_emb, axis=1), 1, atol=1e-5)
        assert np.allclose(np.linalg.norm(np.stack(queries.embedding), axis=1), 1, atol=1e-5)
        query_hits = {k: [] for k in ks}
        query_precision = {k: [] for k in ks}
        saved = json.loads(metric_path.read_text())
        saved_top1 = pd.read_csv(eval_dir / 'test_dependency_group_retrieval.csv').set_index('query_relief_id')
        # Match compute_retrieval_metrics exactly: all training groups; no
        # source/object exclusion; dot product of normalized vectors; argsort.
        for _, query in queries.iterrows():
            similarities = gallery_emb @ query.embedding
            order = np.argsort(-similarities)
            relevant = gallery.iloc[order].authority.to_numpy() == query.authority
            assert gallery.iloc[order[0]].relief_id == saved_top1.loc[query.relief_id, 'top1_relief_id']
            for k in ks:
                query_hits[k].append(bool(relevant[:k].any()))
                query_precision[k].append(float(relevant[:k].mean()))
            for rank, index in enumerate(order, 1):
                neighbour = gallery.iloc[index]
                rank_rows.append({
                    'seed': seed, 'query_group': query.component_key,
                    'query_relief_ids': query.relief_ids, 'query_authority': query.authority,
                    'rank': rank, 'neighbour_group': neighbour.component_key,
                    'neighbour_relief_ids': neighbour.relief_ids,
                    'neighbour_authority': neighbour.authority,
                    'cosine_similarity': float(similarities[index]),
                    'same_label': bool(neighbour.authority == query.authority),
                })
        for k in ks:
            hit = float(np.mean(query_hits[k]))
            assert math.isclose(hit, saved['retrieval'][f'recall@{k}'], abs_tol=1e-12)
            metrics_rows.append({'seed': seed, 'k': k, 'label_hit_rate': hit,
                                 'precision_at_k': float(np.mean(query_precision[k]))})
        pred = pd.read_csv(pred_path).set_index('component_key').loc[test.component_key]
        assert list(pred.true_authority) == list(truth)
        correct = pred.true_label.to_numpy() == pred.pred_label.to_numpy()
        learned_correct.append(correct)
        for i, (_, group) in enumerate(test.iterrows()):
            comparisons.append({'seed': seed, 'component_key': group.component_key,
                                'authority': group.Authority, 'source': group.source,
                                'source_prediction': source_pred[i],
                                'source_correct': bool(source_correct[i]),
                                'model_correct': bool(correct[i])})
        for path in (gallery_path, query_path, pred_path, metric_path):
            input_hashes[str(path.relative_to(root))] = sha256(path)
    metrics = pd.DataFrame(metrics_rows)
    # Paired bootstrap: resample the SAME 39 groups jointly for both predictors.
    # Average correctness across the three already fitted models; never treat
    # their 117 predictions as independent observations.
    difference = np.mean(np.stack(learned_correct), axis=0) - source_correct.astype(float)
    bootstrap_seed, n_bootstrap = 20260906, 10000
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(0, len(test), size=(n_bootstrap, len(test)))
    interval = np.quantile(difference[samples].mean(axis=1), [0.025, 0.975]).tolist()
    paired = {'mean_accuracy_difference': float(difference.mean()),
              'percentile_95_ci': interval, 'bootstrap_seed': bootstrap_seed,
              'bootstrap_replicates': n_bootstrap, 'unit': 'dependency group',
              'n_groups': 39, 'model_accuracy_mean': float(np.mean(learned_correct))}
    summary = {
        'manifest_sha256': expected_hash, 'components_sha256': sha256(components_path),
        'gallery': 'All 80 distinct training groups; no exclusions',
        'query': 'All 39 test groups, equal query weight',
        'train_group_counts': train_counts, 'test_group_counts': test_counts,
        'random_label_hit_probability': chance,
        'random_expected_precision_at_k': chance_precision,
        'learned_retrieval_by_seed': metrics_rows,
        'source_group_baseline': source_metrics, 'paired_accuracy_comparison': paired,
        'numeric_rounding': 'Round exact stored values once to nearest 0.001; do not round via 4 decimals',
        'input_sha256': input_hashes,
    }
    out = root / 'reports/submission_controls_2026-09-06'
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    pd.DataFrame(rank_rows).to_csv(out/'retrieval_all_ranks.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out/'paired_source_predictions.csv', index=False)
    source_table = pd.crosstab([components.source, components.split], components.Authority)
    source_table.to_csv(out/'source_by_monarch_groups.csv')
    lines = ['# Submission controls audit - 6 September 2026', '',
             'Recomputed from the frozen manifest and saved normalized dependency-group embeddings; no retraining.',
             f'Manifest SHA-256: `{expected_hash}`.', '',
             '## Retrieval', '',
             'The evaluator uses all 80 distinct training groups and excludes none. All 39 test groups are queries.',
             'The original `Recall@k` is a query-level same-label hit rate, also called Hit@k here; it is not the fraction of all relevant gallery objects retrieved.',
             'Random rankings sample distinct gallery groups without replacement. For each query class c, the exact hit probability is 1 - C(80-n_train_c,k)/C(80,k), then weighted by test class frequency.', '',
             '| k | Random Hit@k | Learned Hit@k, mean +/- sample SD | Random expected P@k | Learned P@k, mean +/- sample SD |',
             '|---|---|---|---|---|']
    for k in ks:
        rows = metrics.loc[metrics.k.eq(k)]
        lines.append(f'| {k} | {chance[str(k)]:.6f} | {rows.label_hit_rate.mean():.3f} +/- {rows.label_hit_rate.std():.3f} | {chance_precision:.6f} | {rows.precision_at_k.mean():.3f} +/- {rows.precision_at_k.std():.3f} |')
    lines += ['', 'P@k is the proportion of the first k neighbours sharing the query label. Random probabilities are descriptive expectations, not significance tests. Seeds share the same split and queries.', '',
              '## Comparable source-only baseline', '',
              'Each training dependency group contributes one vote to its source prefix. Every group has exactly one source prefix. Unseen sources use the overall training-group majority; ties use lexical label order.',
              f"Training mapping: `{mapping}`. Fallback: {fallback}.",
              f"Test result: {source_metrics['correct']}/39; accuracy {source_metrics['accuracy']:.6f}; macro-F1 {source_metrics['macro_f1']:.6f}.",
              f"Mean full-model accuracy minus source baseline: {paired['mean_accuracy_difference']:.6f}; paired group-bootstrap 95% interval [{interval[0]:.6f}, {interval[1]:.6f}].",
              'The bootstrap uses 10,000 replicates, seed 20260906, resampling 39 groups jointly across source predictions and the three fitted models. It is conditional on this split and these fitted models. It does not isolate carving style or establish transfer to new sources.', '',
              '## Exact numeric correction', '',
              'The saved seed-24 macro-F1 upper bootstrap bound is 0.9294947121034077, which rounds directly to 0.929 at three decimals. Rounding first to 0.9295 and then to three decimals would incorrectly produce 0.930.', '',
              '## Reproduction and checks', '',
              '`python scripts/audit_submission_controls.py --root .`',
              'Assertions verify the frozen hash, group identities and labels, single-source membership, normalized vectors, saved top-one neighbour identities, and all saved Hit@k metrics. Full original-space rankings are saved for all queries, not just selected errors.',
              'Companion files contain input hashes, all neighbour ranks/cosine similarities, paired predictions, and source-by-monarch group counts.']
    (root/'reports/submission_controls_2026-09-06.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()