"""Colab orchestration. Run setup, download_data, prepare, train, evaluate in order."""
import os, sys, subprocess, json, hashlib, zipfile, shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent
SRC=BASE/'reproduction_v1'
sys.path.insert(0,str(SRC))
from p3_runner import read, digest
from repro_contract import build_manifest
from export_prediction_series import export
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def setup():
    global RUN, ANN, manifest, run
    RUN=Path('/content/jcmil_p3_joint_seed0_github_v1')
    if not RUN.exists():
        free_gib=shutil.disk_usage('/content').free/2**30
        print(f'Ổ đĩa còn {free_gib:.1f} GiB; dự trù 25 GiB cho dữ liệu, 5 checkpoint cuối và xuất ZIP.')
        assert free_gib >=25, f'Chỉ còn {free_gib:.1f} GiB; cần dự trù 25 GiB cho lượt chạy đầy đủ.'
    RUN.mkdir(exist_ok=True)
    sys.path.insert(0,str(SRC))
    def run(args, logname):
        with (RUN/logname).open('a',encoding='utf-8') as log:
            log.write('\nCOMMAND '+json.dumps([str(a) for a in args])+'\n'); log.flush()
            proc=subprocess.Popen([str(a) for a in args],cwd=SRC,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in proc.stdout:
                    print(line,end=''); log.write(line); log.flush()
                if proc.wait(): raise RuntimeError(f'Lệnh thất bại: {logname}; xem log, không tiếp tục.')
            except BaseException:
                proc.terminate(); proc.wait(); raise
    # Tải annotations tại runtime thay vì nhúng hàng MB trong notebook.
    ANN=BASE/'audit_2026_10_26/sources/splits.zip'
    ANN.parent.mkdir(parents=True,exist_ok=True)
    if not ANN.exists():
        partial=ANN.with_suffix('.partial')
        run(['curl','--fail','--location','--retry','3','--output',partial,
             'https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_all.zip'],'annotations-download.log')
        assert hashlib.sha256(partial.read_bytes()).hexdigest()=='daee55291203af7ed6fd7f5bbb51153d5ec5b9c6662b36e3e289d9ce4a375a53', 'Annotations khác bản đã duyệt'
        partial.rename(ANN)
    assert hashlib.sha256(ANN.read_bytes()).hexdigest()=='daee55291203af7ed6fd7f5bbb51153d5ec5b9c6662b36e3e289d9ce4a375a53'
    from repro_contract import build_manifest
    manifest=build_manifest(ANN)
    assert manifest['manifest_id']=='81ad353ef6e667b96708a2197a165b099002e79fa472b5159be2b3d94c4e7bc1'
    (SRC/'manifest_common_label_v1.json').write_text(json.dumps(manifest,ensure_ascii=False))

    (RUN/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (RUN/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    (RUN/'git-commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True))

    (RUN/'execution-settings.json').write_text(json.dumps({'precision':'fp32','batch_size':32,'num_workers':0,'token_cache':True,'checkpoint_mmap':True},indent=2))


def download_data():
    global DATA, IMAGES, SNAPSHOT, COMMON
    # Tải ảnh chính thức (~1,8 GB); không cần upload ảnh từ máy cá nhân.
    import tarfile
    DATA=Path('/content/crisismmd_media'); DATA.mkdir(exist_ok=True)
    tarpath=DATA/'CrisisMMD_v2.0.tar.gz'
    if not (DATA/'EXTRACTED.ok').exists():
        if not tarpath.exists():
            if shutil.which('aria2c') is None:
                subprocess.run(['apt-get','update','-qq'],check=True,stdout=subprocess.DEVNULL)
                subprocess.run(['apt-get','install','-y','-qq','aria2'],check=True,stdout=subprocess.DEVNULL)
            assert shutil.which('aria2c'), 'aria2c chưa được cài'
            partial=tarpath.with_suffix('.partial')
            partial.touch(exist_ok=True)
            run(['aria2c','--continue=true','--max-connection-per-server=8','--split=8',
                 '--min-split-size=1M','--max-tries=0','--retry-wait=5','--connect-timeout=30',
                 '--timeout=60','--file-allocation=none','--summary-interval=30',
                 '--console-log-level=notice','--dir',DATA,'--out',partial.name,
                 'https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz'],'aria2-download.log')
            partial.rename(tarpath)
        with tarfile.open(tarpath) as archive:
            # Chỉ nhận file/thư mục, chặn link và đường dẫn thoát thư mục.
            for member in archive:
                assert (DATA/member.name).resolve().is_relative_to(DATA.resolve())
                assert member.isfile() or member.isdir(), member.name
                archive.extract(member,DATA,filter='data')
        (DATA/'EXTRACTED.ok').write_text('Extraction completed; image preflight still required')
    roots=list(DATA.rglob('data_image'))
    assert len(roots)==1, roots
    IMAGES=roots[0].parent
    from huggingface_hub import snapshot_download
    SNAPSHOT=snapshot_download('openai/clip-vit-base-patch16',
        revision='57c216476eefef5ab752ec549e440a49ae4ae5f3',
        allow_patterns=['config.json','merges.txt','preprocessor_config.json','pytorch_model.bin',
                        'special_tokens_map.json','tokenizer_config.json','tokenizer.json','vocab.json'])
    from p3_runner import digest, read
    assert digest(Path(SNAPSHOT)/'pytorch_model.bin')=='ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0'
    COMMON=['--manifest',SRC/'manifest_common_label_v1.json','--annotations',BASE/'audit_2026_10_26/sources/splits.zip',
            '--images',IMAGES,'--snapshot',SNAPSHOT]
    print('Images:',IMAGES,'CLIP:',SNAPSHOT)


def prepare():
    global PF, proof, ANCHOR, train_job
    # Preflight thực đọc/giải mã toàn bộ 18.082 dòng; không chỉ kiểm tra file tồn tại.
    PF=RUN/'preflight'
    if not PF.exists():
        run([sys.executable,'p3_runner.py','preflight',*COMMON,
             '--reviewed-overlap-report',SRC/'official_preflight_01/preflight.json','--output',PF],'preflight.log')
    proof=read(PF/'preflight.json')
    assert not proof['errors'], proof['errors']
    print({k:v for k,v in proof.items() if k not in ('images','snapshot_files')})
    ANCHOR=RUN/'anchor'
    if not ANCHOR.exists():
        run([sys.executable,'prepare_anchor.py','--manifest',SRC/'manifest_common_label_v1.json',
             '--snapshot',SNAPSHOT,'--seed','0','--output',ANCHOR],'anchor.log')
    assert digest(ANCHOR/'anchor.pt')==read(ANCHOR/'anchor.json')['sha256']
    def train_job(stage, joint=False):
        name='joint' if joint else f's{stage}'
        output=RUN/name
        parent=ANCHOR/'anchor.pt' if joint or stage==1 else RUN/f's{stage-1}'/'epoch10'
        checkpoints=sorted(cp for cp in output.glob('epoch*') if not (cp/'retention.json').exists()) if output.exists() else []
        for cp in checkpoints:
            assert (cp/'metadata.json').exists(), f'Checkpoint ghi dở: {cp}; giữ lại để kiểm tra trước khi khôi phục.'
            meta=read(cp/'metadata.json')
            assert meta['complete'] and digest(cp/'state.pt')==meta['checkpoint_sha256']
        if checkpoints and read(checkpoints[-1]/'metadata.json')['epoch']==10:
            print(name,'đã đủ 10 epoch'); return
        args=[sys.executable,'p3_runner.py','train',*COMMON,'--preflight',PF/'preflight.json',
              '--stage',str(stage),'--seed','0','--parent',parent,'--output',output,'--keep-latest']
        if joint: args+=['--joint']
        if checkpoints: args+=['--resume',checkpoints[-1]]
        run(args,name+'.log')

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from export_prediction_series import export

def evaluate(split='dev'):
    M=read(SRC/'manifest_common_label_v1.json')
    for stage,joint in [(1,False),(2,False),(3,False),(4,False),(1,True)]:
        name='joint' if joint else f's{stage}'
        output=RUN/f'pred-{split}-{name}'
        if not output.exists():
            args=[sys.executable,'p3_runner.py','predict',*COMMON,'--preflight',PF/'preflight.json',
                  '--stage',str(stage),'--seed','0','--checkpoint',RUN/name/'epoch10',
                  '--split',split,'--output',output]
            if joint: args+=['--joint']
            run(args,f'predict-{split}-{name}.log')
        for mode in ['text','image','multimodal']:
            raw=read(output/f'{mode}.json')
            assert raw['checkpoint_metadata']==read(RUN/name/'epoch10/metadata.json')
            assert raw['checkpoint_sha256']==digest(RUN/name/'epoch10/state.pt')
            assert raw['split']==split and raw['mode']==mode and raw['class_order']==M['classes']
    for mode in ['text','image','multimodal']:
        dest=RUN/f'scores-{split}-{mode}'
        if not dest.exists():
            export(M,RUN,ANCHOR/'anchor.pt',[RUN/f's{i}/epoch10' for i in range(1,5)],
                   [RUN/f'pred-{split}-s{i}' for i in range(1,5)],dest,mode,split)
    summary=[]
    for name in ['s4','joint']:
        for mode in ['text','image','multimodal']:
            raw=read(RUN/f'pred-{split}-{name}'/f'{mode}.json')
            assert raw['seen_classes']==list(range(8))
            byid={r['sample_id']:r['logits'] for r in raw['rows']}
            assert len(byid)==len(raw['rows'])==len(M['rows'][split])
            assert set(byid)=={r['sample_id'] for r in M['rows'][split]}
            logits=np.array([byid[r['sample_id']] for r in M['rows'][split]])
            assert logits.shape==(len(byid),8) and np.isfinite(logits).all()
            y=[r['label'] for r in M['rows'][split]]; p=logits.argmax(1)
            summary.append(dict(model=name,mode=mode,split=split,n=len(y),
                macro_f1_8=float(f1_score(y,p,labels=list(range(8)),average='macro',zero_division=0)),
                accuracy=float(accuracy_score(y,p)),confusion_matrix=confusion_matrix(y,p,labels=list(range(8))).tolist(),
                checkpoint_sha256=raw['checkpoint_sha256'],manifest_id=M['manifest_id']))
    (RUN/f'summary-{split}.json').write_text(json.dumps(summary,indent=2))
    for row in summary: print({k:v for k,v in row.items() if k not in ('confusion_matrix','checkpoint_sha256','manifest_id')})


def export_evidence():
    global RUN
    bundle=Path('/content/jcmil_evidence.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RUN.rglob('*')):
            if p.is_file() and p.suffix!='.pt': z.write(p,p.relative_to(RUN))
    return str(bundle)


def export_checkpoint(name='s4'):
    global RUN
    assert name in ['s1','s2','s3','s4','joint']
    cp=RUN/name/'epoch10'
    assert (cp/'metadata.json').exists()
    archive=Path('/content')/(name+'-epoch10.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
        for p in cp.iterdir():
            if p.is_file(): z.write(p,p.name)
    return str(archive)

