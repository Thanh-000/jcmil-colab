"""Offline validation of prediction provenance and independent metric recomputation.

Hashes detect inconsistency, not fabrication. The producer must load the registered
checkpoint strictly; these tools cannot prove which model generated arbitrary logits.
Registry paths are relative to an explicit artifact root. No weights are deserialized.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from repro_contract import fingerprint, verify_manifest, score_snapshot, continual_metrics, file_digest


def validate_registry(registry, root):
    root = Path(root).resolve()
    records = registry['checkpoints']
    by_id = {}
    for r in records:
        content = {k: v for k, v in r.items() if k != 'record_id'}
        if fingerprint(content) != r['record_id'] or r['record_id'] in by_id:
            raise ValueError('Invalid/duplicate checkpoint record')
        path = (root / r['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Checkpoint path outside artifact root')
        if file_digest(path) != r['sha256']:
            raise ValueError('Checkpoint bytes mismatch')
        for field in ['experiment_id', 'manifest_id', 'seed', 'protocol', 'stage', 'method', 'regime', 'parents']:
            if field not in r:
                raise ValueError('Incomplete checkpoint identity')
        if type(r['seed']) is not int or type(r['stage']) is not int or r['stage'] not in range(5):
            raise ValueError('Invalid checkpoint seed/stage')
        if r['regime'] not in ('anchor', 'independent', 'sequential', 'merged'):
            raise ValueError('Unknown checkpoint regime')
        if len(set(r['parents'])) != len(r['parents']):
            raise ValueError('Duplicate parents')
        parents = []
        for parent in r['parents']:
            if parent not in by_id:
                raise ValueError('Parent missing or registry not topologically ordered')
            p = by_id[parent]
            if any(p[k] != r[k] for k in ('experiment_id', 'manifest_id', 'seed', 'protocol')):
                raise ValueError('Parent experiment/seed/protocol mismatch')
            parents.append(p)
        if r['regime'] == 'anchor':
            if r['stage'] != 0 or parents:
                raise ValueError('Invalid anchor')
        elif r['regime'] == 'independent':
            if r['stage'] == 0 or len(parents) != 1 or parents[0]['regime'] != 'anchor':
                raise ValueError('Independent expert must start from anchor')
        elif r['regime'] == 'sequential':
            if (r['stage'] == 0 or len(parents) != 1 or parents[0]['stage'] != r['stage'] - 1
                    or parents[0]['regime'] not in ('anchor', 'sequential')):
                raise ValueError('Sequential expert must follow fine-tuned predecessor')
        else:
            experts = [p for p in parents if p['regime'] != 'anchor']
            anchors = [p for p in parents if p['regime'] == 'anchor']
            if (r['stage'] == 0 or len(anchors) != 1
                    or sorted(p['stage'] for p in experts) != list(range(1, r['stage'] + 1))
                    or len({p['regime'] for p in experts}) != 1
                    or any(p['regime'] not in ('independent', 'sequential') for p in experts)):
                raise ValueError('Merge must reference anchor and complete homogeneous expert prefix')
        by_id[r['record_id']] = r
    return by_id


def score_bundle(manifest, bundle, records):
    verify_manifest(manifest)
    if bundle.get('schema') != 'prediction-bundle-v1':
        raise ValueError('Unknown prediction schema')
    if fingerprint({k: v for k, v in bundle.items() if k != 'bundle_id'}) != bundle['bundle_id']:
        raise ValueError('Prediction bundle content hash mismatch')
    meta = bundle['metadata']
    ckpt = records[meta['checkpoint_record_id']]
    for key in ('experiment_id', 'manifest_id', 'seed', 'protocol', 'stage', 'method', 'regime'):
        if meta[key] != ckpt[key]:
            raise ValueError('Prediction/checkpoint identity mismatch')
    if (meta['manifest_id'] != manifest['manifest_id'] or meta['target_policy'] != manifest['primary_target']
            or meta['class_order'] != manifest['classes'] or meta['split'] not in ('dev', 'test')):
        raise ValueError('Prediction label/split contract mismatch')
    if meta['score_format'] != 'raw_unmasked_logits' or meta['tie_policy'] != 'lowest_class_id':
        raise ValueError('Unsupported score/tie semantics')
    if type(meta['stage']) is not int or meta['stage'] not in range(1, 5):
        raise ValueError('Invalid prediction stage')
    stage = manifest['protocols'][meta['protocol']][meta['stage'] - 1]
    if meta['seen_classes'] != stage['seen_classes']:
        raise ValueError('Seen-class mask mismatch')
    seen = sorted(meta['seen_classes'])
    predictions = []
    for row in bundle['rows']:
        scores = row['logits']
        if (len(scores) != len(manifest['classes']) or
                any(type(x) not in (float, int) for x in scores) or not np.isfinite(scores).all()):
            raise ValueError('Invalid/nonfinite logits')
        # Mask by explicit selection; no magic -1e4 sentinel, no dependence on row target.
        predicted = seen[int(np.argmax(np.asarray(scores)[seen]))]
        if 'prediction' in row and (type(row['prediction']) is not int or row['prediction'] != predicted):
            raise ValueError('Stored prediction differs from masked argmax')
        predictions.append({'sample_id': row['sample_id'], 'prediction': predicted})
    result = score_snapshot(manifest, predictions, meta['protocol'], meta['stage'], meta['mode'], meta['split'])
    result['provenance'] = {k: meta[k] for k in ('experiment_id', 'seed', 'method', 'regime', 'checkpoint_record_id')}
    result['provenance']['bundle_id'] = bundle['bundle_id']
    return result


def score_series(manifest, bundles, registry, root):
    records = validate_registry(registry, root)
    if len(bundles) != 4:
        raise ValueError('Expected four stage snapshots')
    snapshots = [score_bundle(manifest, b, records) for b in bundles]
    first = snapshots[0]['provenance']
    for s in snapshots[1:]:
        if any(s['provenance'][k] != first[k] for k in ('experiment_id', 'seed', 'method', 'regime')):
            raise ValueError('Mixed experiment/seed/method series')
    # For chronological or merging series, enforce the specific chain, not just stage numbers.
    checkpoints = [records[s['provenance']['checkpoint_record_id']] for s in snapshots]
    for previous, current in zip(checkpoints, checkpoints[1:]):
        if current['regime'] == 'sequential' and current['parents'] != [previous['record_id']]:
            raise ValueError('Series follows different sequential branches')
        if current['regime'] == 'merged' and not set(previous['parents']) <= set(current['parents']):
            raise ValueError('Merge expert prefix changed between snapshots')
    return dict(status='ARTIFACT_CONSISTENCY_VALIDATED_NOT_MODEL_EXECUTION_PROOF', snapshots=snapshots,
                continual_accuracy=continual_metrics(snapshots),
                continual_macro_f1=continual_metrics(snapshots, 'macro_f1'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--bundles', nargs=4, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    result = score_series(read(args.manifest), [read(p) for p in args.bundles], read(args.registry), args.artifact_root)
    # Do not overwrite an existing evidence result.
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(result['status'])
