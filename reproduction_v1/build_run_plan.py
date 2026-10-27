"""Produce a dependency plan, never launch training or claim a run completed."""
import argparse
import json
import math
from pathlib import Path

from repro_contract import fingerprint, verify_manifest


def make_plan(manifest, seeds=(0, 1, 2, 3, 4)):
    verify_manifest(manifest)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    jobs = []

    def add(job_id, kind, parents, **fields):
        known = {j['job_id'] for j in jobs}
        if job_id in known or not set(parents) <= known:
            raise ValueError('Duplicate job or unresolved dependency')
        jobs.append(dict(job_id=job_id, kind=kind, parents=parents, status='PLANNED', **fields))
        return job_id

    add('preflight', 'external_preflight', [],
        requirements=['real_images', 'pinned_pretrained_weights', 'preprocessing_parity',
                      'integrated_trainer', 'GPU_environment', 'recipe_frozen'])
    for seed in seeds:
        base = add(f'anchor/{seed}', 'initialize', ['preflight'], seed=seed,
                   note='Same initialized fusion/head within seed for all protocols/regimes')
        joint = add(f'joint/{seed}', 'train', [base], seed=seed, protocol='joint', stage=1,
                    regime='joint', n_samples=len(manifest['rows']['train']), epochs=10,
                    batch_size=32, updates=10 * math.ceil(len(manifest['rows']['train']) / 32))
        add(f'eval-joint/{seed}', 'evaluate', [joint], seed=seed, modes=['text', 'image', 'multimodal'],
            selection='dev_only', test_policy='locked_recipe_only')
        for protocol, stages in manifest['protocols'].items():
            independent, sequential = [], []
            for s in stages:
                stage = s['stage']
                spec = dict(seed=seed, protocol=protocol, stage=stage, epochs=10, batch_size=32,
                            n_samples=len(s['train_ids']), train_ids_hash=fingerprint(s['train_ids']),
                            updates=10 * math.ceil(len(s['train_ids']) / 32), modalities=s['modalities'])
                if stage == 1:
                    expert = add(f'{protocol}/{seed}/shared-s1', 'train', [base], regime='shared_stage1', **spec)
                    independent.append(expert)
                    sequential.append(expert)
                else:
                    independent.append(add(f'{protocol}/{seed}/ind-s{stage}', 'train', [base], regime='independent', **spec))
                    sequential.append(add(f'{protocol}/{seed}/seq-s{stage}', 'train', [sequential[-1]], regime='sequential', **spec))
                methods = [('seq_ft', sequential[-1])]
                for regime, experts in [('ind', independent), ('seq', sequential)]:
                    for method in ['mean_delta', 'maxabs_delta']:
                        merged = add(f'{protocol}/{seed}/{regime}-{method}-s{stage}', 'merge', [base, *experts],
                            seed=seed, protocol=protocol, stage=stage, method=method, regime=regime,
                            qualification='JCMIL baseline adaptation; explicit common head policy required')
                        methods.append((f'{regime}-{method}', merged))
                for method, checkpoint in methods:
                    add(f'{protocol}/{seed}/eval-{method}-s{stage}', 'evaluate', [checkpoint],
                        seed=seed, protocol=protocol, stage=stage, method=method,
                        modes=['text', 'image', 'multimodal'], selection='dev_only', test_policy='locked_recipe_only')
    counts = {kind: sum(j['kind'] == kind for j in jobs) for kind in {j['kind'] for j in jobs}}
    plan = dict(schema='jcmil-run-plan-v1', status='PLANNED_NOT_EXECUTABLE_TRAINER_PENDING',
                manifest_id=manifest['manifest_id'], target_policy='common_label_v1_proposal',
                seeds=list(seeds), job_counts=counts,
                total_training_updates=sum(j.get('updates', 0) for j in jobs),
                total_evaluation_mode_snapshots=sum(len(j.get('modes', [])) for j in jobs), jobs=jobs)
    plan['plan_id'] = fingerprint(plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    m = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    plan = make_plan(m, args.seeds)
    Path(args.output).write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in plan.items() if k != 'jobs'}, indent=2))
