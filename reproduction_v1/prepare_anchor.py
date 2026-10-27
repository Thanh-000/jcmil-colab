"""Create one immutable real pretrained anchor for P3 and Joint."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from p3_runner import load_backend, read, digest, write_new, source_id
from repro_contract import verify_manifest, fingerprint
from experiment_runtime import seed_everything

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    seed_everything(a.seed)
    m=read(a.manifest)
    verify_manifest(m)
    out=Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    model,processor=load_backend(a.snapshot)
    model.eval()
    prompts=['A disaster report or photo of '+c.replace('_',' ') for c in m['classes']]
    encoded=processor(text=prompts,padding=True,truncation=True,max_length=77,return_tensors='pt')
    with torch.no_grad():
        model.head.copy_(F.normalize(model.clip.get_text_features(**encoded),dim=-1))
    meta=dict(manifest_id=m['manifest_id'],seed=a.seed,source_sha256=source_id(),
        snapshot_id=fingerprint({f.name:digest(f) for f in Path(a.snapshot).iterdir() if f.is_file()}),
        prompt_template='A disaster report or photo of {class_name}',future_vocabulary_known=True)
    out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(model=model.state_dict(),**meta),out/'anchor.pt')
    write_new(out/'anchor.json',dict(**meta,sha256=digest(out/'anchor.pt')))
    print('Created pretrained anchor, not a trained result:',out)
