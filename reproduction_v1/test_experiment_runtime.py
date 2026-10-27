import copy
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from experiment_runtime import (FrozenRows, environment_record, load_epoch_checkpoint,
    save_epoch_checkpoint, seed_everything, train_epoch)
from repro_contract import fingerprint


class RandomizedFixture(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, i):
        # Exercise Python, NumPy, torch, shuffle and dropout RNGs on resume.
        x = torch.tensor([i / 12., random.random(), np.random.random(), torch.rand(()).item()])
        return x, torch.tensor(i % 2)


def setup(seed=17):
    generator = seed_everything(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=9)
    loader = DataLoader(RandomizedFixture(), batch_size=4, shuffle=True, generator=generator, num_workers=0)
    identity = dict(source_sha256='fixture-source', manifest_id='fixture-only', recipe={'epochs': 3},
        seed=seed, parent_checkpoint_sha256=None, backbone_revision='fixture', tokenizer_revision='none',
        environment_sha256=fingerprint(environment_record()), stage=1, regime='independent',
        method='train', protocol='fixture')
    return model, opt, sch, generator, loader, identity


def loss_fn(model, batch):
    x, y = batch
    return torch.nn.functional.cross_entropy(model(x), y), len(y)


def test_epoch_resume_bitwise_matches_uninterrupted(tmp_path):
    model, opt, sch, gen, loader, identity = setup()
    train_epoch(model, loader, opt, sch, loss_fn)
    save_epoch_checkpoint(tmp_path / 'epoch1', model, opt, sch, gen, identity, 1, 3)
    expected_logs = [train_epoch(model, loader, opt, sch, loss_fn) for _ in range(2)]
    expected_model = copy.deepcopy(model.state_dict())
    expected_opt = copy.deepcopy(opt.state_dict())
    expected_scheduler = copy.deepcopy(sch.state_dict())
    # Recreate objects with different random state; load must restore exact stream.
    model, opt, sch, gen, loader, _ = setup(999)
    assert load_epoch_checkpoint(tmp_path / 'epoch1', model, opt, sch, gen, identity) == (1, 3)
    actual_logs = [train_epoch(model, loader, opt, sch, loss_fn) for _ in range(2)]
    assert actual_logs == expected_logs
    assert all(torch.equal(v, expected_model[k]) for k, v in model.state_dict().items())
    for k, state in opt.state_dict()['state'].items():
        assert all(torch.equal(v, expected_opt['state'][k][name]) for name, v in state.items())
    assert sch.state_dict() == expected_scheduler


def test_frozen_rows_survive_adamw_decay_and_existing_momentum():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3, 4))
    opt = torch.optim.AdamW([p], lr=.1, weight_decay=.2)
    p.sum().backward()
    opt.step()  # establish nonzero momentum before freezing
    before = p.detach().clone()
    mask = torch.tensor([True, False, True])
    guard = FrozenRows(p, mask)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        p.square().sum().backward()
        guard.before_step()
        opt.step()
        guard.after_step(opt)
    assert torch.equal(p[mask], before[mask])
    assert not torch.equal(p[~mask], before[~mask])
    assert torch.count_nonzero(opt.state[p]['exp_avg'][mask]) == 0


@pytest.mark.parametrize('change', ['seed', 'bytes', 'optimizer_order', 'incomplete', 'overwrite'])
def test_checkpoint_rejects_invalid_resume(tmp_path, change):
    model, opt, sch, gen, loader, identity = setup()
    train_epoch(model, loader, opt, sch, loss_fn)
    path = tmp_path / 'checkpoint'
    save_epoch_checkpoint(path, model, opt, sch, gen, identity, 1, 3)
    if change == 'seed':
        identity = {**identity, 'seed': 18}
    elif change == 'bytes':
        with (path / 'state.pt').open('ab') as f:
            f.write(b'corrupt')
    elif change == 'optimizer_order':
        opt.param_groups[0]['params'].reverse()
    elif change == 'incomplete':
        (path / 'metadata.json').rename(path / 'metadata.json.partial')
    if change == 'overwrite':
        with pytest.raises(FileExistsError):
            save_epoch_checkpoint(path, model, opt, sch, gen, identity, 1, 3)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            load_epoch_checkpoint(path, model, opt, sch, gen, identity)


def test_nonfinite_loss_does_not_update_model():
    model, opt, sch, gen, loader, identity = setup()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='Nonfinite'):
        train_epoch(model, loader, opt, sch, lambda m, b: (m(b[0]).sum() * float('nan'), 4))
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert not opt.state
