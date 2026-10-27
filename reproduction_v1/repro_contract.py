"""Preparation/evaluation tools. This is NOT an integrated training runner.

Primary estimand: all input modes predict the fixed provided `label`.
Modality-specific and set-hit scores are secondary, never used for cohort routing.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

CLASSES = ['not_humanitarian', 'other_relevant_information',
           'rescue_volunteering_or_donation_effort', 'infrastructure_and_utility_damage',
           'affected_individuals', 'injured_or_dead_people', 'vehicle_damage', 'missing_or_found_people']
GROUPS = {'P1': [[0, 3, 5], [0, 3, 5], [1, 2, 6], [4, 7]],
          'P2': [[0, 3, 5], [0, 3, 5], [1, 2, 6], [4, 7]],
          'P3': [[0, 5], [1, 4], [2, 6], [3, 7]],
          'P4': [[0, 5], [1, 4], [2, 6], [3, 7]]}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_rows(archive):
    result = {}
    with zipfile.ZipFile(archive) as z:
        for split in ['train', 'dev', 'test']:
            names = [n for n in z.namelist() if not n.startswith('__MACOSX')
                     and n.endswith(f'task_humanitarian_text_img_{split}.tsv')]
            if len(names) != 1:
                raise ValueError(f'Ambiguous/missing annotation: {split}')
            rows = list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig')), delimiter='\t'))
            result[split] = []
            ids = set()
            for r in rows:
                uid = fingerprint([r['tweet_id'], r['image_id']])
                if uid in ids:
                    raise ValueError('Duplicate tweet/image pair in split')
                ids.add(uid)
                result[split].append(dict(sample_id=uid, tweet_id=r['tweet_id'], image_id=r['image_id'],
                    image=r['image'], text_sha256=hashlib.sha256(r['tweet_text'].encode()).hexdigest(),
                    label=CLASSES.index(r['label']), label_text=CLASSES.index(r['label_text']),
                    label_image=CLASSES.index(r['label_image'])))
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        if {r['sample_id'] for r in result[a]} & {r['sample_id'] for r in result[b]}:
            raise ValueError(f'Pair ID crosses splits: {a}/{b}')
    return result


def build_manifest(archive):
    rows = load_rows(archive)
    protocols = {}
    for name, groups in GROUPS.items():
        seen, stages = set(), []
        for stage, classes in enumerate(groups, 1):
            new = sorted(set(classes) - seen)
            seen.update(classes)
            train = [r['sample_id'] for r in rows['train'] if r['label'] in classes]
            cohort = [r['sample_id'] for r in rows['test'] if r['label'] in new]
            first = 'text' if name in ('P1', 'P3') else 'image'
            unimodal = stage == 1 or (name in ('P3', 'P4') and stage == 2)
            stages.append(dict(stage=stage, train_classes=classes, seen_classes=sorted(seen),
                               modalities=[first] if unimodal else ['text', 'image'],
                               train_ids=train, new_class_test_ids=cohort, new_classes=new))
        all_ids = set().union(*(set(s['train_ids']) for s in stages))
        if all_ids != {r['sample_id'] for r in rows['train']}:
            raise ValueError('Training coverage failure')
        for i in range(4):
            for j in range(i + 1, 4):
                overlap = set(stages[i]['train_ids']) & set(stages[j]['train_ids'])
                expected = set(stages[0]['train_ids']) if name in ('P1', 'P2') and (i, j) == (0, 1) else set()
                if overlap != expected:
                    raise ValueError('Undeclared stage re-observation')
        protocols[name] = stages
    adjacency = {i: set() for i in range(8)}
    for r in rows['train']:
        a, b = r['label_text'], r['label_image']
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining, components = set(range(8)), []
    while remaining:
        todo, component = [min(remaining)], set()
        while todo:
            v = todo.pop()
            if v not in component:
                component.add(v)
                todo.extend(adjacency[v] - component)
        remaining -= component
        components.append(sorted(component))
    manifest = dict(schema='jcmil-common-label-v1', status='PREPARATION_ONLY_NOT_TRAINED',
                    annotation_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                    classes=CLASSES, primary_target='label_for_all_modes', rows=rows,
                    protocols=protocols, train_label_graph_components=components)
    manifest['manifest_id'] = fingerprint(manifest)
    return manifest


def verify_manifest(manifest):
    content = {k: v for k, v in manifest.items() if k != 'manifest_id'}
    if fingerprint(content) != manifest['manifest_id']:
        raise ValueError('Manifest content hash mismatch')


def score_snapshot(manifest, predictions, protocol, stage, mode, split='test'):
    """One complete snapshot, predictions=[{sample_id,prediction}]. No target accepted."""
    verify_manifest(manifest)
    if mode not in ('text', 'image', 'multimodal') or stage not in range(1, 5):
        raise ValueError('Invalid mode/stage')
    rows = manifest['rows'][split]
    pred = {}
    seen = manifest['protocols'][protocol][stage - 1]['seen_classes']
    for p in predictions:
        uid, y = p['sample_id'], p['prediction']
        if uid in pred or type(y) is not int or y not in seen:
            raise ValueError('Duplicate sample, noninteger or out-of-seen prediction')
        pred[uid] = y
    if set(pred) != {r['sample_id'] for r in rows}:
        raise ValueError('Prediction coverage differs from fixed split')
    y = [r['label'] for r in rows]
    yp = [pred[r['sample_id']] for r in rows]
    result = dict(manifest_id=manifest['manifest_id'], protocol=protocol, stage=stage, mode=mode, split=split,
        n=len(rows), accuracy=float(accuracy_score(y, yp)),
        macro_f1_8=float(f1_score(y, yp, labels=list(range(8)), average='macro', zero_division=0)),
        confusion_matrix=confusion_matrix(y, yp, labels=list(range(8))).tolist(),
        target_hash=fingerprint([(r['sample_id'], r['label']) for r in rows]),
        set_hit_accuracy_secondary=float(np.mean([p in (r['label_text'], r['label_image']) for r, p in zip(rows, yp)])))
    if mode in ('text', 'image'):
        ym = [r['label_' + mode] for r in rows]
        result['modality_target_macro_f1_8_secondary'] = float(f1_score(ym, yp, labels=list(range(8)), average='macro', zero_division=0))
    result['cohorts'] = {}
    for s in manifest['protocols'][protocol]:
        if not s['new_classes']:
            continue
        rr = [r for r in rows if r['label'] in s['new_classes']]
        if not rr:
            raise ValueError('Empty evaluation cohort')
        yt, pp = [r['label'] for r in rr], [pred[r['sample_id']] for r in rr]
        result['cohorts'][str(s['stage'])] = dict(n=len(rr), id_hash=fingerprint([r['sample_id'] for r in rr]),
            accuracy=float(accuracy_score(yt, pp)), macro_f1=float(f1_score(yt, pp, labels=s['new_classes'], average='macro', zero_division=0)))
    return result


def continual_metrics(snapshots, metric='accuracy'):
    if len(snapshots) != 4 or metric not in ('accuracy', 'macro_f1'):
        raise ValueError('Require four ordered snapshots and explicit metric')
    if [s['stage'] for s in snapshots] != [1, 2, 3, 4]:
        raise ValueError('Snapshots are not in stage order')
    for snap in snapshots[1:]:
        if any(snap[k] != snapshots[0][k] for k in ('manifest_id', 'protocol', 'mode', 'split')):
            raise ValueError('Snapshot evaluation identity changed')
        if snap['target_hash'] != snapshots[0]['target_hash']:
            raise ValueError('Targets changed between snapshots')
        if snap['cohorts'].keys() != snapshots[0]['cohorts'].keys():
            raise ValueError('Cohort list changed')
        for g, cohort in snap['cohorts'].items():
            if cohort['id_hash'] != snapshots[0]['cohorts'][g]['id_hash']:
                raise ValueError('Cohort membership changed')
    bwt, forgetting = [], []
    for group in snapshots[0]['cohorts']:
        intro = int(group) - 1
        if intro >= 3:
            continue
        values = [s['cohorts'][group][metric] for s in snapshots]
        bwt.append(values[-1] - values[intro])
        forgetting.append(max(values[intro:-1]) - values[-1])
    return dict(metric=metric, bwt=float(np.mean(bwt)), forgetting_signed=float(np.mean(forgetting)),
                forgetting_positive_part=float(np.mean(np.maximum(forgetting, 0))),
                n_old_disjoint_cohorts=len(bwt))


def verify_cache(expected, stored, checkpoint):
    """Validate metadata before the caller deserializes any weights."""
    required = {'source_sha256', 'manifest_id', 'recipe', 'seed', 'parent_checkpoint_sha256',
                'backbone_revision', 'tokenizer_revision', 'environment_sha256', 'stage', 'regime',
                'method', 'protocol'}
    if not required <= expected.keys():
        raise ValueError('Incomplete run identity')
    if stored.get('run_id') != fingerprint(expected):
        raise ValueError('Cache run identity mismatch')
    if stored.get('checkpoint_sha256') != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
        raise ValueError('Checkpoint hash mismatch')


def check_images(manifest, image_root):
    from PIL import Image
    root = Path(image_root).resolve()
    hashes, failures = {}, []
    for split, rows in manifest['rows'].items():
        for r in rows:
            path = (root / r['image']).resolve()
            try:
                path.relative_to(root)
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert('RGB').load()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes.setdefault(digest, []).append([split, r['sample_id']])
            except (OSError, ValueError) as exc:
                failures.append(dict(split=split, sample_id=r['sample_id'], error=str(exc)))
    cross_split = {h: ids for h, ids in hashes.items() if len({x[0] for x in ids}) > 1}
    return dict(files_checked=sum(map(len, manifest['rows'].values())), failures=failures,
                exact_cross_split_duplicates=cross_split,
                status='PASS_EXACT_ONLY' if not failures and not cross_split else 'BLOCKED',
                note='Perceptual duplicates and event-disjoint generalization not certified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--annotations', required=True)
    build.add_argument('--output', required=True)
    images = sub.add_parser('images')
    images.add_argument('--manifest', required=True)
    images.add_argument('--image-root', required=True)
    images.add_argument('--output', required=True)
    score = sub.add_parser('score')
    score.add_argument('--manifest', required=True)
    score.add_argument('--predictions', required=True, help='JSON list for one complete snapshot')
    score.add_argument('--protocol', choices=list(GROUPS), required=True)
    score.add_argument('--stage', type=int, required=True)
    score.add_argument('--mode', choices=['text', 'image', 'multimodal'], required=True)
    score.add_argument('--split', choices=['dev', 'test'], required=True)
    score.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        result = build_manifest(args.annotations)
    else:
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        verify_manifest(manifest)
        if args.command == 'images':
            result = check_images(manifest, args.image_root)
        else:
            predictions = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
            result = score_snapshot(manifest, predictions, args.protocol, args.stage, args.mode, args.split)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'protocols', 'failures')}, ensure_ascii=False))
    if args.command == 'images' and result['status'] == 'BLOCKED':
        raise SystemExit(2)
