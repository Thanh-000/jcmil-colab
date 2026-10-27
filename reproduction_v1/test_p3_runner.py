from types import SimpleNamespace
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytest
from p3_runner import Model, configure_stage, make_loss
from experiment_runtime import FrozenRows, train_epoch
from p3_runner import validate_media_policy
from p3_runner import prune_previous_state, digest, write_new


@pytest.mark.parametrize('failure', [None, 'corrupt_successor', 'different_run'])
def test_retention_only_removes_previous_payload_after_verified_save(tmp_path, failure):
    for epoch in [1, 2]:
        cp=tmp_path/f'epoch{epoch:02d}'; cp.mkdir()
        (cp/'state.pt').write_bytes(b'checkpoint'+bytes([epoch]))
        write_new(cp/'metadata.json', dict(complete=True, epoch=epoch,
            run_id='other' if failure=='different_run' and epoch==2 else 'same',
            checkpoint_sha256=digest(cp/'state.pt')))
        (cp/'training_stats.json').write_text('{}')
    if failure=='corrupt_successor':
        (tmp_path/'epoch02/state.pt').write_bytes(b'corrupt')
    if failure:
        with pytest.raises(ValueError): prune_previous_state(tmp_path,2)
        assert (tmp_path/'epoch01/state.pt').exists()
    else:
        prune_previous_state(tmp_path,2)
        assert not (tmp_path/'epoch01/state.pt').exists()
        assert (tmp_path/'epoch01/metadata.json').exists()
        assert (tmp_path/'epoch01/training_stats.json').exists()
        assert (tmp_path/'epoch01/retention.json').exists()
    assert (tmp_path/'epoch02/state.pt').exists()


def test_overlap_acknowledgement_never_bypasses_integrity_or_changed_inventory():
    report = dict(files_checked=3, failures=[], exact_cross_split_duplicates={'abc': [['train','a'],['dev','b']]})
    assert validate_media_policy(report)
    assert not validate_media_policy(report, copy.deepcopy(report))
    bad = copy.deepcopy(report)
    bad['failures'] = [{'sample_id':'c','error':'decode'}]
    assert validate_media_policy(bad, report)
    changed = copy.deepcopy(report)
    changed['exact_cross_split_duplicates']['abc'].append(['test','c'])
    assert validate_media_policy(changed, report)
    changed = copy.deepcopy(report)
    changed['files_checked'] = 2
    assert validate_media_policy(changed, report)


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(projection_dim=4)
        self.text_model=nn.Embedding(10,4)
        self.text_projection=nn.Linear(4,4,bias=False)
        self.vision_model=nn.Linear(4,4)
        self.visual_projection=nn.Linear(4,4,bias=False)
        self.logit_scale=nn.Parameter(torch.tensor(1.))
    def get_text_features(self,input_ids,attention_mask):
        return self.text_projection(self.text_model(input_ids).mean(1))
    def get_image_features(self,pixel_values):
        return self.visual_projection(self.vision_model(pixel_values))


def batch():
    return dict(input_ids=torch.tensor([[1,2],[2,3]]),attention_mask=torch.ones(2,2,dtype=torch.long),
                pixel_values=torch.randn(2,4),label=torch.tensor([0,5]))


def test_unavailable_modality_cannot_affect_unimodal_logits():
    torch.manual_seed(0)
    m=Model(TinyClip()).eval()
    b=batch()
    before=m.features(b,['text'])['text']
    b['pixel_values']*=1000
    assert torch.equal(before,m.features(b,['text'])['text'])
    before=m.features(b,['image'])['image']
    b['input_ids'].fill_(9)
    assert torch.equal(before,m.features(b,['image'])['image'])


def test_text_stage_freeze_and_future_negative_exclusion():
    torch.manual_seed(0)
    m=Model(TinyClip())
    configure_stage(m,['text'])
    b=batch()
    before=copy.deepcopy(m.state_dict())
    loss=make_loss(['text'],[0,5],'cpu')
    value,_=loss(m,b)
    with torch.no_grad():
        m.head[7].fill_(100)
    assert torch.equal(value,loss(m,b)[0])
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01,weight_decay=.1)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=1)
    mask=torch.ones(8,dtype=torch.bool); mask[[0,5]]=False
    head=m.head.detach().clone()
    guard=FrozenRows(m.head,mask)
    class OneBatch(list):
        num_workers=0
    train_epoch(m,OneBatch([b]),opt,sch,loss,[guard])
    assert torch.equal(head[mask],m.head[mask])
    assert torch.equal(before['clip.vision_model.weight'],m.state_dict()['clip.vision_model.weight'])
    assert torch.equal(before['fusion.1.weight'],m.state_dict()['fusion.1.weight'])
    assert not torch.equal(before['clip.text_model.weight'],m.state_dict()['clip.text_model.weight'])


def test_multimodal_objective_same_common_target():
    m=Model(TinyClip()); b=batch()
    loss,_=make_loss(['text','image'],list(range(8)),'cpu')(m,b)
    logits=m.features(b,['text','image','multimodal'])
    expected=sum(w*torch.nn.functional.cross_entropy(logits[k],b['label'])
                 for k,w in [('text',.2),('image',.2),('multimodal',.6)])
    assert torch.allclose(loss,expected)
    b['label']=torch.tensor([0,7])
    with pytest.raises(ValueError,match='outside'):
        make_loss(['text'],[0,5],'cpu')(m,b)
