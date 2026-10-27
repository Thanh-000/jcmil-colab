import copy
import hashlib
import json
from pathlib import Path

import pytest

from repro_contract import (build_manifest, score_snapshot, continual_metrics,
                           fingerprint, verify_cache, check_images)


@pytest.fixture(scope='module')
def manifest():
    return build_manifest(Path(__file__).resolve().parent.parent / 'audit_2026_10_26/sources/splits.zip')


def perfect(manifest):
    return [{'sample_id': r['sample_id'], 'prediction': r['label']} for r in manifest['rows']['test']]


def test_coverage_and_declared_repeat(manifest):
    for name, stages in manifest['protocols'].items():
        union = set().union(*(set(s['train_ids']) for s in stages))
        assert len(union) == 13608
        assert sum(len(s['train_ids']) for s in stages) == 13608 + (7409 if name in ('P1', 'P2') else 0)
        cohorts = [set(s['new_class_test_ids']) for s in stages]
        assert sum(map(len, cohorts)) == len(set().union(*cohorts)) == 2237


def test_targets_and_cohorts_independent_of_prediction(manifest):
    a = score_snapshot(manifest, perfect(manifest), 'P3', 4, 'multimodal')
    b = score_snapshot(manifest, [{'sample_id': r['sample_id'], 'prediction': 0}
                                for r in manifest['rows']['test']], 'P3', 4, 'multimodal')
    assert a['macro_f1_8'] == 1
    assert b['macro_f1_8'] < 1
    assert a['target_hash'] == b['target_hash']
    assert {g: c['id_hash'] for g, c in a['cohorts'].items()} == {g: c['id_hash'] for g, c in b['cohorts'].items()}


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'future', 'noninteger'])
def test_reject_bad_predictions(manifest, kind):
    pp = perfect(manifest)
    if kind == 'missing':
        pp.pop()
    elif kind == 'duplicate':
        pp.append(pp[0])
    elif kind == 'noninteger':
        pp[0]['prediction'] = '0'
    with pytest.raises(ValueError):
        score_snapshot(manifest, pp, 'P1', 1 if kind == 'future' else 4, 'text')


def test_manifest_tamper(manifest):
    m = copy.deepcopy(manifest)
    m['rows']['test'][0]['label'] = 99
    with pytest.raises(ValueError, match='hash'):
        score_snapshot(m, perfect(manifest), 'P1', 4, 'image')


def test_continual_metrics_hand_calculated():
    values = [[.8, .1, .1], [.9, .7, .1], [.6, .8, .5], [.7, .9, .8]]
    snapshots = [dict(manifest_id='m', protocol='P1', mode='text', split='test', stage=i,
                 target_hash='fixed', cohorts={str(g): dict(id_hash=f'group{g}', accuracy=v)
                 for g, v in zip([1, 3, 4], row)}) for i, row in enumerate(values, 1)]
    metrics = continual_metrics(snapshots)
    # First introductions at stages 1 and 3. Stage 2 is re-observation, not another task.
    assert metrics['n_old_disjoint_cohorts'] == 2
    assert metrics['bwt'] == pytest.approx(0)
    assert metrics['forgetting_signed'] == pytest.approx(.05)
    assert metrics['forgetting_positive_part'] == pytest.approx(.1)
    snapshots[-1]['cohorts']['1']['id_hash'] = 'changed'
    with pytest.raises(ValueError, match='membership'):
        continual_metrics(snapshots)


def test_cache_rejects_seed_parent_and_content_changes(tmp_path):
    ckpt = tmp_path / 'checkpoint.bin'
    ckpt.write_bytes(b'fixture-not-model')
    expected = dict(source_sha256='a', manifest_id='b', recipe={'lr': 1e-5}, seed=0,
                    parent_checkpoint_sha256='c', backbone_revision='d', tokenizer_revision='e',
                    environment_sha256='f', stage=1, regime='independent', method='magmax', protocol='P1')
    stored = dict(run_id=fingerprint(expected), checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())
    verify_cache(expected, stored, ckpt)
    for field, value in [('seed', 1), ('parent_checkpoint_sha256', 'different')]:
        with pytest.raises(ValueError, match='identity'):
            verify_cache({**expected, field: value}, stored, ckpt)
    ckpt.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        verify_cache(expected, stored, ckpt)


def test_missing_corrupt_duplicate_and_traversal_images(tmp_path):
    from PIL import Image
    Image.new('RGB', (8, 8)).save(tmp_path / 'valid.png')
    (tmp_path / 'bad.png').write_bytes(b'not image')
    rows = {'train': [dict(sample_id='1', image='valid.png'), dict(sample_id='2', image='bad.png')],
            'test': [dict(sample_id='3', image='valid.png'), dict(sample_id='4', image='../outside.png'),
                     dict(sample_id='5', image='missing.png')]}
    result = check_images({'rows': rows}, tmp_path)
    assert result['status'] == 'BLOCKED'
    assert len(result['failures']) == 3
    assert len(result['exact_cross_split_duplicates']) == 1


def test_builder_determinism(manifest):
    assert build_manifest(Path(__file__).resolve().parent.parent / 'audit_2026_10_26/sources/splits.zip') == manifest


def test_continual_rejects_mixed_mode_and_stage_order(manifest):
    pp = [{'sample_id': r['sample_id'], 'prediction': 0} for r in manifest['rows']['test']]
    snapshots = [score_snapshot(manifest, pp, 'P1', stage, 'text') for stage in range(1, 5)]
    with pytest.raises(ValueError, match='order'):
        continual_metrics(list(reversed(snapshots)))
    snapshots[-1]['mode'] = 'image'
    with pytest.raises(ValueError, match='identity'):
        continual_metrics(snapshots)
