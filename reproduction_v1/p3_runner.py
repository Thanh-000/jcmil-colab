"""Candidate common-label P3 runner. Real local CLIP snapshot; no fallback/subset.

Commands: preflight, train (one stage or Joint), predict (one checkpoint).
FP32, batch32, 10 epochs, num_workers=0; GPU required for training.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import time
from pathlib import Path
import zipfile

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from repro_contract import verify_manifest, fingerprint, check_images, score_snapshot
from experiment_runtime import (seed_everything, environment_record, FrozenRows,
    train_epoch, save_epoch_checkpoint, load_epoch_checkpoint)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


class PairedRows(Dataset):
    def __init__(self, manifest, annotations, image_root, processor, split, ids=None, images=True):
        if digest(annotations) != manifest['annotation_sha256']:
            raise ValueError('Annotation hash mismatch')
        self.root, self.processor, self.images = Path(image_root).resolve(), processor, images
        with zipfile.ZipFile(annotations) as z:
            names = [n for n in z.namelist() if not n.startswith('__MACOSX')
                     and n.endswith(f'task_humanitarian_text_img_{split}.tsv')]
            if len(names) != 1:
                raise ValueError('Ambiguous split')
            raw = list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig')), delimiter='\t'))
        by_id = {fingerprint([r['tweet_id'], r['image_id']]): r for r in raw}
        self.rows = [r for r in manifest['rows'][split] if ids is None or r['sample_id'] in ids]
        if ids is not None and {r['sample_id'] for r in self.rows} != set(ids):
            raise ValueError('Stage IDs not found')
        self.text = {}
        for r in self.rows:
            text = by_id[r['sample_id']]['tweet_text']
            if hashlib.sha256(text.encode()).hexdigest() != r['text_sha256']:
                raise ValueError('Text changed')
            self.text[r['sample_id']] = text
        # Tokenization is deterministic and unchanged across epochs. Cache only small token tensors.
        parts = [processor(text=[self.text[r['sample_id']] for r in self.rows[i:i+128]],
                 return_tensors='pt', padding='max_length', truncation=True, max_length=77)
                 for i in range(0,len(self.rows),128)]
        self.tokens = {k: torch.cat([part[k] for part in parts]) for k in parts[0]} if parts else {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        # Raw tweet policy deliberately explicit, no hidden cleaning/fallback.
        encoded = {k:v[index] for k,v in self.tokens.items()}
        if self.images:
            path = (self.root / r['image']).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Image path escaped root')
            with Image.open(path) as im:
                encoded.update({k:v.squeeze(0) for k,v in self.processor(
                    images=im.convert('RGB'), return_tensors='pt').items()})
        return {**encoded,
                'label': torch.tensor(r['label']), 'sample_id': r['sample_id']}


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702*x)


class Model(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip
        d = clip.config.projection_dim
        self.fusion = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, d),
                                    QuickGELU(), nn.Linear(d, d))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.gate = nn.Parameter(torch.tensor(.1))
        self.head = nn.Parameter(F.normalize(torch.randn(8, d), dim=-1))
        self.scale = nn.Parameter(torch.tensor(14.28).log())

    def features(self, batch, modes):
        result = {}
        if 'text' in modes or 'multimodal' in modes:
            result['text'] = F.normalize(self.clip.get_text_features(input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']), dim=-1)
        if 'image' in modes or 'multimodal' in modes:
            result['image'] = F.normalize(self.clip.get_image_features(pixel_values=batch['pixel_values']), dim=-1)
        if 'multimodal' in modes:
            t, i = result['text'], result['image']
            result['multimodal'] = F.normalize((t+i)/2 + .25*self.gate.tanh()*self.fusion(torch.cat([t,i], -1)), dim=-1)
        return {k: self.scale.exp().clamp(max=100) * (v @ F.normalize(self.head, dim=-1).T)
                for k, v in result.items()}


def configure_stage(model, modalities):
    for name, p in model.named_parameters():
        if name.startswith(('clip.text_model.', 'clip.text_projection.')):
            p.requires_grad_('text' in modalities)
        elif name.startswith(('clip.vision_model.', 'clip.visual_projection.')):
            p.requires_grad_('image' in modalities)
        elif name.startswith('clip.'):
            p.requires_grad_(False)  # unused CLIP contrastive scale
        elif name.startswith('fusion.') or name == 'gate':
            p.requires_grad_(len(modalities) == 2)
        else:
            p.requires_grad_(True)


def make_loss(modalities, seen, device):
    modes = ['text', 'image', 'multimodal'] if len(modalities) == 2 else list(modalities)
    weights = {'text': .2, 'image': .2, 'multimodal': .6} if len(modalities) == 2 else {modes[0]: 1.}
    mapping = torch.full((8,), -1, dtype=torch.long, device=device)
    mapping[seen] = torch.arange(len(seen), device=device)
    def loss_fn(model, batch):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        target = mapping[batch['label']]
        if (target < 0).any():
            raise ValueError('Target outside seen classes')
        logits = model.features(batch, modes)
        loss = sum(weights[m] * F.cross_entropy(logits[m][:, seen], target) for m in modes)
        return loss, len(target)
    return loss_fn


def load_backend(snapshot):
    from transformers import CLIPModel, CLIPProcessor
    snapshot = Path(snapshot)
    if not snapshot.is_dir():
        raise ValueError('Local CLIP snapshot required')
    processor = CLIPProcessor.from_pretrained(str(snapshot), local_files_only=True, use_fast=False)
    clip = CLIPModel.from_pretrained(str(snapshot), local_files_only=True)
    if (clip.config.projection_dim != 512 or clip.config.vision_config.patch_size != 16
            or clip.config.vision_config.hidden_size != 768):
        raise ValueError('Expected CLIP ViT-B/16')
    return Model(clip), processor


def source_id():
    return fingerprint({p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def prune_previous_state(output, epoch):
    """Drop only the previous epoch's tensor payload after verifying its successor."""
    if epoch < 2:
        return
    root = Path(output).resolve()
    current = root/f'epoch{epoch:02d}'
    previous = root/f'epoch{epoch-1:02d}'
    for directory in (current, previous):
        if directory.is_symlink() or directory.resolve().parent != root:
            raise ValueError('Checkpoint path escaped job directory')
    new = read(current/'metadata.json')
    if not new.get('complete') or new['epoch'] != epoch or digest(current/'state.pt') != new['checkpoint_sha256']:
        raise ValueError('Successor checkpoint is incomplete or corrupt')
    if not (previous/'state.pt').exists():
        return
    old = read(previous/'metadata.json')
    if (not old.get('complete') or old['epoch'] != epoch-1 or old['run_id'] != new['run_id']
            or (previous/'state.pt').is_symlink()):
        raise ValueError('Previous checkpoint identity mismatch')
    (previous/'state.pt').unlink()
    write_new(previous/'retention.json', dict(state_removed=True,
        reason='Replaced by verified completed successor', successor_epoch=epoch,
        successor_sha256=new['checkpoint_sha256']))


def validate_media_policy(current, approved=None):
    """Acknowledge only an exact, previously reviewed overlap inventory."""
    if current['failures']:
        return ['image_decode_or_access_failure']
    if not current['exact_cross_split_duplicates']:
        return []
    if approved is None:
        return ['cross_split_overlap_requires_reviewed_inventory']
    if (approved['failures'] or current['files_checked'] != approved['files_checked']
            or current['exact_cross_split_duplicates'] != approved['exact_cross_split_duplicates']):
        return ['reviewed_media_inventory_mismatch']
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'train', 'predict'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', type=int, choices=[1,2,3,4], default=1)
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--keep-latest', action='store_true', help='Keep latest epoch payload per job; preserve all epoch metadata and statistics')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--parent', help='Previous stage final epoch directory; stage1/Joint use anchor state file')
    parser.add_argument('--resume', help='Completed epoch directory of same job')
    parser.add_argument('--checkpoint', help='Completed epoch directory to predict')
    parser.add_argument('--preflight', help='Successful basic preflight JSON required for train/predict')
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    parser.add_argument('--reviewed-overlap-report', help='Prior preflight JSON acknowledging exact official split overlap; decode failures always block')
    args = parser.parse_args()
    manifest = read(args.manifest)
    verify_manifest(manifest)
    seed_everything(args.seed)
    out = Path(args.output)
    if args.command == 'preflight':
        out.mkdir(parents=True, exist_ok=False)
        result = dict(cuda_available=torch.cuda.is_available(), source_sha256=source_id(),
                      manifest_id=manifest['manifest_id'], image_root=str(Path(args.images).resolve()), errors=[])
        if digest(args.annotations) != manifest['annotation_sha256']:
            result['errors'].append('annotation_hash_mismatch')
        if not Path(args.images).is_dir():
            result['errors'].append('image_root_missing')
        else:
            result['images'] = check_images(manifest, args.images)
            approved = None
            if args.reviewed_overlap_report:
                review = read(args.reviewed_overlap_report)
                if review['manifest_id'] != manifest['manifest_id']:
                    raise ValueError('Reviewed overlap manifest mismatch')
                approved = review['images']
                result['reviewed_overlap_report_sha256'] = digest(args.reviewed_overlap_report)
            result['errors'].extend(validate_media_policy(result['images'], approved))
            result['media_policy'] = ('official_split_overlap_acknowledged' if approved is not None
                                      else 'exact_content_disjoint_required')
        try:
            model, processor = load_backend(args.snapshot)
            result['snapshot_files'] = {p.name:digest(p) for p in Path(args.snapshot).iterdir() if p.is_file()}
            model.eval()
            tokens = processor(text=['disaster rescue'], return_tensors='pt', padding=True)
            with torch.no_grad():
                encoded = model.clip.get_text_features(**tokens)
            result['text_probe_shape'] = list(encoded.shape)
            if not torch.isfinite(encoded).all():
                result['errors'].append('nonfinite_pretrained_features')
        except Exception as exc:
            result['errors'].append(f'backend: {type(exc).__name__}: {exc}')
        if not result['cuda_available']:
            result['errors'].append('GPU_training_unavailable')
        result['status'] = 'BLOCKED' if result['errors'] else 'PREFLIGHT_BASIC_PASS_PARITY_REVIEW_REQUIRED'
        write_new(out/'preflight.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('images','snapshot_files')}, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Full training/inference launcher requires CUDA; no CPU fallback')
    if not args.preflight:
        raise ValueError('--preflight required')
    proof=read(args.preflight)
    if (proof['status']!='PREFLIGHT_BASIC_PASS_PARITY_REVIEW_REQUIRED'
            or proof['manifest_id']!=manifest['manifest_id'] or proof['source_sha256']!=source_id()
            or proof['image_root']!=str(Path(args.images).resolve())
            or proof['snapshot_files']!={p.name:digest(p) for p in Path(args.snapshot).iterdir() if p.is_file()}):
        raise ValueError('Preflight stale or mismatched')
    model, processor = load_backend(args.snapshot)
    snapshot_id=fingerprint({p.name:digest(p) for p in Path(args.snapshot).iterdir() if p.is_file()})
    device = torch.device('cuda')
    model.to(device)
    stage = manifest['protocols']['P3'][args.stage-1]
    if args.joint:
        stage = dict(modalities=['text','image'], train_classes=list(range(8)), seen_classes=list(range(8)),
                     train_ids=[r['sample_id'] for r in manifest['rows']['train']])
    if args.command == 'predict':
        if not args.checkpoint:
            raise ValueError('--checkpoint required')
        cp = Path(args.checkpoint)
        meta = read(cp/'metadata.json')
        if (not meta.get('complete') or meta['identity']['backbone_revision']!=snapshot_id
                or meta['identity']['source_sha256']!=source_id()):
            raise ValueError('Incomplete checkpoint or source/backend mismatch')
        if digest(cp/'state.pt') != meta['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        expected_protocol = 'joint' if args.joint else 'P3'
        if any(meta['identity'][k] != v for k,v in [('manifest_id',manifest['manifest_id']),
                ('seed',args.seed),('protocol',expected_protocol),('stage',args.stage)]):
            raise ValueError('Prediction checkpoint identity mismatch')
        model.load_state_dict(torch.load(cp/'state.pt', map_location='cpu', weights_only=True, mmap=True)['model'], strict=True)
        model.eval()
        ds = PairedRows(manifest,args.annotations,args.images,processor,args.split)
        loader = DataLoader(ds,batch_size=32,shuffle=False,generator=torch.Generator().manual_seed(9000),num_workers=0,pin_memory=True)
        rows = {m:[] for m in ['text','image','multimodal']}
        with torch.inference_mode():
            for batch in loader:
                batch = {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
                logits = model.features(batch, list(rows))
                for mode in rows:
                    rows[mode].extend(dict(sample_id=uid,logits=l) for uid,l in zip(batch['sample_id'],logits[mode].cpu().tolist()))
        out.mkdir(parents=True,exist_ok=False)
        for mode, values in rows.items():
            write_new(out/f'{mode}.json',dict(schema='raw-predictor-export-v1',checkpoint_metadata=meta,
                checkpoint_sha256=meta['checkpoint_sha256'],source_sha256=source_id(),mode=mode,split=args.split,
                class_order=manifest['classes'],seen_classes=stage['seen_classes'],rows=values))
        return
    if not args.parent:
        raise ValueError('Explicit anchor/parent required; use prepare_anchor.py first')
    parent = Path(args.parent)
    parent_file = parent/'state.pt' if parent.is_dir() else parent
    if parent.is_dir():
        parent_meta=read(parent/'metadata.json')
        if (not parent_meta.get('complete') or parent_meta['epoch']!=10
                or parent_meta['identity']['backbone_revision']!=snapshot_id
                or parent_meta['identity']['source_sha256']!=source_id()):
            raise ValueError('Parent is incomplete or backend/source changed')
        if digest(parent_file)!=parent_meta['checkpoint_sha256']:
            raise ValueError('Parent hash mismatch')
        ident=parent_meta['identity']
        if (args.joint or args.stage==1 or ident['stage']!=args.stage-1 or ident['seed']!=args.seed
                or ident['protocol']!='P3' or ident['manifest_id']!=manifest['manifest_id']):
            raise ValueError('Invalid sequential parent')
    elif not (args.stage==1 or args.joint):
        raise ValueError('Later sequential stage requires prior epoch directory')
    else:
        anchor_meta=read(parent_file.parent/'anchor.json')
        if digest(parent_file)!=anchor_meta['sha256']:
            raise ValueError('Anchor bytes mismatch')
    parent_state = torch.load(parent_file,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(parent_state['model'],strict=True)
    if not parent.is_dir():
        if parent_state['manifest_id']!=manifest['manifest_id'] or parent_state['seed']!=args.seed:
            raise ValueError('Anchor identity mismatch')
        if parent_state['snapshot_id']!=snapshot_id:
            raise ValueError('Anchor backend mismatch')
    del parent_state
    configure_stage(model,stage['modalities'])
    groups=[]
    for kind, lr, wd in [('backbone',1e-5,.05),('fusion',1e-5,.01),('head',3e-5,0.)]:
        params=[p for n,p in model.named_parameters() if p.requires_grad and
                ('backbone' if n.startswith('clip.') else 'fusion' if n.startswith('fusion.') or n=='gate' else 'head')==kind]
        if params: groups.append(dict(params=params,lr=lr,weight_decay=wd))
    opt=torch.optim.AdamW(groups)
    generator=torch.Generator().manual_seed(args.seed*100+args.stage)
    ds=PairedRows(manifest,args.annotations,args.images,processor,'train',set(stage['train_ids']),'image' in stage['modalities'])
    loader=DataLoader(ds,batch_size=32,shuffle=True,generator=generator,num_workers=0,drop_last=False,pin_memory=True)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10*len(loader),eta_min=1e-7)
    recipe=dict(epochs=10,batch_size=32,target='common_label',text='raw_tweet',fusion='QuickGELU_MLP',
                precision='fp32',train_mask='seen',modalities=stage['modalities'],train_ids_hash=fingerprint(stage['train_ids']))
    identity=dict(source_sha256=source_id(),manifest_id=manifest['manifest_id'],recipe=recipe,seed=args.seed,
        parent_checkpoint_sha256=digest(parent_file),backbone_revision=snapshot_id,tokenizer_revision=snapshot_id,
        environment_sha256=fingerprint(environment_record()),stage=args.stage,regime='joint' if args.joint else 'sequential',
        method='joint' if args.joint else 'seq_ft',protocol='joint' if args.joint else 'P3')
    epoch,steps=0,0
    if args.resume:
        epoch,steps=load_epoch_checkpoint(args.resume,model,opt,scheduler,generator,identity)
    mask=torch.ones(8,dtype=torch.bool,device=device)
    mask[stage['train_classes']]=False
    guard=FrozenRows(model.head,mask)
    out.mkdir(parents=True,exist_ok=True)
    for e in range(epoch+1,11):
        # Conservative bound: model + two Adam moments, plus serialization reserve.
        needed = 3*sum(t.numel()*t.element_size() for t in model.state_dict().values()) + 2**30
        if shutil.disk_usage(out).free < needed:
            raise RuntimeError(f'Insufficient checkpoint space: need about {needed/2**30:.1f} GiB free before epoch {e}')
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        stats=train_epoch(model,loader,opt,scheduler,make_loss(stage['modalities'],stage['seen_classes'],device),[guard])
        torch.cuda.synchronize()
        stats.update(train_seconds=time.perf_counter()-started,
                     peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        steps+=stats['steps']
        save_epoch_checkpoint(out/f'epoch{e:02d}',model,opt,scheduler,generator,identity,e,steps)
        write_new(out/f'epoch{e:02d}'/'training_stats.json',stats)
        if args.keep_latest:
            prune_previous_state(out,e)
        print(json.dumps(dict(epoch=e,total_steps=steps,**stats)),flush=True)


if __name__ == '__main__':
    main()
