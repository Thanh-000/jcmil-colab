import json
from pathlib import Path

import pytest

from build_run_plan import make_plan


@pytest.fixture(scope='module')
def plan():
    manifest = json.loads((Path(__file__).resolve().parent / 'manifest_common_label_v1.json').read_text(encoding='utf-8'))
    return make_plan(manifest)


def test_training_budget_and_topology(plan):
    assert plan['job_counts']['train'] == 145  # 4 protocols * 5 seeds * 7 experts + 5 Joint
    done = set()
    for job in plan['jobs']:
        assert set(job['parents']) <= done
        assert job['job_id'] not in done
        done.add(job['job_id'])
        if job['kind'] == 'train' and job['regime'] == 'sequential':
            parent = next(j for j in plan['jobs'] if j['job_id'] == job['parents'][0])
            assert parent['kind'] == 'train'  # never a merged model
            assert parent['stage'] == job['stage'] - 1
            assert parent['seed'] == job['seed']


def test_mergers_share_identical_experts(plan):
    for job in plan['jobs']:
        if job['kind'] == 'merge' and job['method'] == 'mean_delta':
            sibling = next(j for j in plan['jobs'] if j['job_id'] == job['job_id'].replace('mean_delta', 'maxabs_delta'))
            assert job['parents'] == sibling['parents']


def test_plan_is_not_execution_evidence(plan):
    assert all(j['status'] == 'PLANNED' for j in plan['jobs'])
    assert 'PENDING' in plan['status']
