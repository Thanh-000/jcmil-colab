import copy
import hashlib
import json
from pathlib import Path

import pytest

from prediction_evidence import score_series, score_bundle, validate_registry
from repro_contract import fingerprint


def seal(value, key):
    value[key] = fingerprint({k: v for k, v in value.items() if k != key})
    return value


@pytest.fixture
def evidence(tmp_path):
    manifest = json.loads((Path(__file__).resolve().parent / 'manifest_common_label_v1.json').read_text())
    records, bundles = [], []
    for stage in range(5):
        path = tmp_path / f'stage{stage}.bin'
        path.write_bytes(f'fixture-only-not-model-stage{stage}'.encode())
        r = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            experiment_id='fixture-experiment', manifest_id=manifest['manifest_id'], seed=0,
            protocol='P3', stage=stage, method='seq_ft' if stage else 'anchor',
            regime='sequential' if stage else 'anchor', parents=[records[-1]['record_id']] if stage else [])
        records.append(seal(r, 'record_id'))
        if stage:
            meta = {k: r[k] for k in ('experiment_id', 'manifest_id', 'seed', 'protocol', 'stage', 'method', 'regime')}
            meta.update(checkpoint_record_id=r['record_id'], target_policy=manifest['primary_target'],
                class_order=manifest['classes'], split='test', mode='multimodal',
                score_format='raw_unmasked_logits', tie_policy='lowest_class_id',
                seen_classes=manifest['protocols']['P3'][stage - 1]['seen_classes'])
            rows = [dict(sample_id=row['sample_id'], logits=[1., 0., 0., 0., 0., 0., 0., 0.])
                    for row in manifest['rows']['test']]
            bundles.append(seal(dict(schema='prediction-bundle-v1', metadata=meta, rows=rows), 'bundle_id'))
    return manifest, {'checkpoints': records}, bundles, tmp_path


def test_independent_recomputation_and_row_order(evidence):
    m, registry, bundles, root = evidence
    result = score_series(m, bundles, registry, root)
    assert result['snapshots'][-1]['accuracy'] == pytest.approx(849 / 2237)
    assert result['continual_accuracy']['bwt'] == 0
    for bundle in bundles:
        bundle['rows'].reverse()
        seal(bundle, 'bundle_id')
    second = score_series(m, bundles, registry, root)
    assert second['continual_macro_f1'] == result['continual_macro_f1']
    assert second['snapshots'][-1]['confusion_matrix'] == result['snapshots'][-1]['confusion_matrix']


@pytest.mark.parametrize('change', ['seed', 'method', 'class_order', 'mask', 'duplicate', 'missing', 'prediction', 'hash', 'nan'])
def test_reject_invalid_bundles(evidence, change):
    m, registry, bundles, root = evidence
    b = bundles[-1]
    if change in ('seed', 'method'):
        b['metadata'][change] = 7 if change == 'seed' else 'other'
    elif change == 'class_order':
        b['metadata']['class_order'] = list(reversed(b['metadata']['class_order']))
    elif change == 'mask':
        b['metadata']['seen_classes'] = [0]
    elif change == 'duplicate':
        b['rows'].append(copy.deepcopy(b['rows'][0]))
    elif change == 'missing':
        b['rows'].pop()
    elif change == 'prediction':
        b['rows'][0]['prediction'] = 1
    else:
        b['rows'][0]['logits'][0] = float('nan') if change == 'nan' else 2.
    if change not in ('hash', 'nan'):
        seal(b, 'bundle_id')
    with pytest.raises(ValueError):
        score_series(m, bundles, registry, root)


def test_changed_checkpoint_bytes(evidence):
    m, registry, bundles, root = evidence
    (root / 'stage2.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes'):
        score_series(m, bundles, registry, root)


def test_checkpoint_parent_wrong_seed(evidence):
    m, registry, bundles, root = evidence
    registry['checkpoints'][1]['seed'] = 8
    seal(registry['checkpoints'][1], 'record_id')
    with pytest.raises(ValueError, match='seed'):
        validate_registry(registry, root)


def test_valid_but_different_experiment_cannot_mix(evidence):
    m, registry, bundles, root = evidence
    other = copy.deepcopy(registry['checkpoints'])
    mapping = {}
    for r in other:
        old = r['record_id']
        r['experiment_id'] = 'second-experiment'
        r['parents'] = [mapping[x] for x in r['parents']]
        seal(r, 'record_id')
        mapping[old] = r['record_id']
    registry['checkpoints'].extend(other)
    bundles[-1]['metadata']['experiment_id'] = 'second-experiment'
    bundles[-1]['metadata']['checkpoint_record_id'] = other[-1]['record_id']
    seal(bundles[-1], 'bundle_id')
    with pytest.raises(ValueError, match='Mixed'):
        score_series(m, bundles, registry, root)


def test_unseen_large_logits_and_tie_policy(evidence):
    m, registry, bundles, root = evidence
    bundle = bundles[0]
    # P3 stage1 permits 0 and 5. Huge future-class score must not win.
    for row in bundle['rows']:
        row['logits'] = [2., 999., 0., 0., 0., 2., 0., 0.]
    seal(bundle, 'bundle_id')
    result = score_bundle(m, bundle, validate_registry(registry, root))
    assert result['accuracy'] == pytest.approx(849 / 2237)
