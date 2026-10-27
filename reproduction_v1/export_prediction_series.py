"""Link four actual sequential predictor exports to a checkpoint registry."""
import argparse
from pathlib import Path
from p3_runner import read, digest, write_new
from repro_contract import fingerprint
from prediction_evidence import score_series


def export(manifest, root, anchor, checkpoints, prediction_dirs, destination, mode, split):
    root, anchor = Path(root).resolve(), Path(anchor).resolve()
    destination=Path(destination)
    if len(checkpoints)!=4 or len(prediction_dirs)!=4:
        raise ValueError('Four checkpoints and predictions required')
    anchor_meta=read(anchor.parent/'anchor.json')
    if digest(anchor)!=anchor_meta['sha256']:
        raise ValueError('Anchor changed')
    exp=fingerprint(dict(anchor=anchor_meta, protocol='P3',target=manifest['primary_target']))
    common=dict(experiment_id=exp,manifest_id=manifest['manifest_id'],seed=anchor_meta['seed'],protocol='P3')
    record=dict(**common,path=str(anchor.relative_to(root)),sha256=digest(anchor),stage=0,method='anchor',regime='anchor',parents=[])
    record['record_id']=fingerprint(record)
    records=[record]; bundles=[]
    for stage,(checkpoint,pred_dir) in enumerate(zip(checkpoints,prediction_dirs),1):
        cp=Path(checkpoint).resolve(); meta=read(cp/'metadata.json'); ident=meta['identity']
        if (ident['parent_checkpoint_sha256']!=records[-1]['sha256'] or ident['stage']!=stage
                or ident['protocol']!='P3' or ident['seed']!=common['seed'] or ident['regime']!='sequential'
                or ident['manifest_id']!=manifest['manifest_id']):
            raise ValueError('Training parent identity mismatch')
        raw=read(Path(pred_dir)/f'{mode}.json')
        if raw['checkpoint_metadata']!=meta or raw['checkpoint_sha256']!=digest(cp/'state.pt') or raw['split']!=split or raw['mode']!=mode:
            raise ValueError('Export/checkpoint mismatch')
        record=dict(**common,path=str((cp/'state.pt').relative_to(root)),sha256=digest(cp/'state.pt'),
            stage=stage,method='seq_ft',regime='sequential',parents=[records[-1]['record_id']])
        record['record_id']=fingerprint(record); records.append(record)
        metadata={k:record[k] for k in ('experiment_id','manifest_id','seed','protocol','stage','method','regime')}
        metadata.update(checkpoint_record_id=record['record_id'],target_policy=manifest['primary_target'],
            class_order=raw['class_order'],split=split,mode=mode,score_format='raw_unmasked_logits',
            tie_policy='lowest_class_id',seen_classes=raw['seen_classes'])
        bundle=dict(schema='prediction-bundle-v1',metadata=metadata,rows=raw['rows'])
        bundle['bundle_id']=fingerprint(bundle); bundles.append(bundle)
    registry={'checkpoints':records}
    scores=score_series(manifest,bundles,registry,root)
    destination.mkdir(parents=True,exist_ok=False)
    write_new(destination/'registry.json',registry)
    for i,b in enumerate(bundles,1): write_new(destination/f'stage{i}.json',b)
    write_new(destination/'recomputed.json',scores)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--artifact-root',required=True)
    p.add_argument('--anchor',required=True);p.add_argument('--checkpoints',nargs=4,required=True)
    p.add_argument('--predictions',nargs=4,required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['text','image','multimodal'],required=True)
    p.add_argument('--split',choices=['dev','test'],required=True)
    a=p.parse_args()
    export(read(a.manifest),a.artifact_root,a.anchor,a.checkpoints,a.predictions,a.output,a.mode,a.split)
