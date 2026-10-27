"""Small FP32, epoch-boundary training primitives; not a CLIP training application.

Exact resume is regression-tested on CPU with num_workers=0. GPU, distributed,
AMP, mid-epoch resume and persistent workers are outside the tested contract.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch

from repro_contract import fingerprint, file_digest


def seed_everything(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be uint32')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.Generator().manual_seed(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def environment_record():
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                pythonhashseed=os.environ.get('PYTHONHASHSEED'),
                gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def rng_state(generator):
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                torch=torch.get_rng_state(), loader=generator.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed')
        torch.cuda.set_rng_state_all(state['cuda'])


def optimizer_layout(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def require_identity(identity):
    required = {'source_sha256', 'manifest_id', 'recipe', 'seed', 'parent_checkpoint_sha256',
                'backbone_revision', 'tokenizer_revision', 'environment_sha256', 'stage',
                'regime', 'method', 'protocol'}
    if not required <= identity.keys() or not identity['recipe']:
        raise ValueError('Incomplete run identity')
    return fingerprint(identity)


def save_epoch_checkpoint(directory, model, optimizer, scheduler, generator, identity, epoch, steps):
    run_id = require_identity(identity)
    env = environment_record()
    if identity['environment_sha256'] != fingerprint(env):
        raise ValueError('Declared environment differs from runtime')
    if epoch < 1 or steps < 1:
        raise ValueError('Only completed training epochs may be saved')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)  # immutable; never overwrite a run
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), rng=rng_state(generator),
                   epoch=epoch, steps=steps, optimizer_layout=optimizer_layout(model, optimizer),
                   optimizer_class=type(optimizer).__qualname__, scheduler_class=type(scheduler).__qualname__)
    temporary = directory / 'state.pt.partial'
    torch.save(payload, temporary)
    state_path = directory / 'state.pt'
    temporary.replace(state_path)
    metadata = dict(schema='epoch-checkpoint-v1', run_id=run_id, identity=identity, environment=env,
                    checkpoint_sha256=file_digest(state_path),
                    epoch=epoch, steps=steps, complete=True)
    temp_meta = directory / 'metadata.json.partial'
    temp_meta.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    temp_meta.replace(directory / 'metadata.json')  # completion marker written last
    return metadata


def load_epoch_checkpoint(directory, model, optimizer, scheduler, generator, identity):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if (meta.get('schema') != 'epoch-checkpoint-v1' or not meta.get('complete')
            or meta['run_id'] != require_identity(identity)
            or meta['identity'] != identity):
        raise ValueError('Checkpoint identity mismatch or incomplete write')
    if fingerprint(environment_record()) != identity['environment_sha256']:
        raise ValueError('Exact resume requires the same recorded environment')
    state_path = directory / 'state.pt'
    if file_digest(state_path) != meta['checkpoint_sha256']:
        raise ValueError('Checkpoint bytes changed')
    state = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
    if (state['optimizer_layout'] != optimizer_layout(model, optimizer)
            or state['optimizer_class'] != type(optimizer).__qualname__
            or state['scheduler_class'] != type(scheduler).__qualname__):
        raise ValueError('Optimizer/scheduler layout mismatch')
    if (state['epoch'], state['steps']) != (meta['epoch'], meta['steps']):
        raise ValueError('Progress metadata mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'], generator)  # restore after object creation/load
    return state['epoch'], state['steps']


class FrozenRows:
    """Keep specified classifier rows bitwise fixed, including Adam momentum slots."""
    def __init__(self, parameter, frozen_mask):
        if frozen_mask.dtype != torch.bool or frozen_mask.shape != parameter.shape[:1]:
            raise ValueError('Expected a boolean row mask')
        self.parameter = parameter
        self.mask = frozen_mask.to(parameter.device)
        self.reference = parameter.detach().clone()

    def before_step(self):
        if self.parameter.grad is not None:
            self.parameter.grad[self.mask] = 0

    @torch.no_grad()
    def after_step(self, optimizer):
        self.parameter[self.mask] = self.reference[self.mask]
        for state in optimizer.state.get(self.parameter, {}).values():
            if torch.is_tensor(state) and state.shape == self.parameter.shape:
                state[self.mask] = 0


def train_epoch(model, loader, optimizer, scheduler, loss_fn, frozen_rows=()):
    """loss_fn(model,batch) -> (scalar loss, sample count). No evaluation or selection."""
    if loader.num_workers != 0:
        raise ValueError('Reference exact-resume engine currently requires num_workers=0')
    model.train()
    total, count, steps = 0., 0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, n = loss_fn(model, batch)
        if loss.ndim != 0 or not torch.isfinite(loss).item() or n <= 0:
            raise ValueError('Nonfinite/nonscalar loss or empty batch')
        loss.backward()
        for guard in frozen_rows:
            guard.before_step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        for guard in frozen_rows:
            guard.after_step(optimizer)
        scheduler.step()
        total += float(loss.detach()) * n
        count += n
        steps += 1
    if not steps:
        raise ValueError('Empty training loader')
    return dict(loss=total / count, samples=count, steps=steps)
