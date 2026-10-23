import numpy as np
import torch
from phenometrix_prosody.train import training_labels, supervised_loss


def test_labels_use_delayed_target_and_preserve_uncovered_voiced_supervision():
    # end86ms -> target66ms/index5, distinct from endpoint86ms/index7.
    feature = {'sample_rate_hz': 48000, 'pcm_sample_count': 48000,
        'reference_sample': np.array([4128, 4608, 5088]), 'features': np.zeros((3, 57)),
        'candidate_hz': np.array([[100., 200.], [100., 200.], [100., 200.]]),
        'candidate_mask': np.ones((3, 2), bool)}
    refs = np.zeros((12, 4)); refs[5] = [200, 1, 0, 0]; refs[6] = [300, 1, 0, 0]
    voiced, valid, targets = training_labels(feature, refs, measurement_delay_ms=20)
    assert voiced.tolist() == [1, 1, 0]
    assert valid.all() and targets.tolist() == [1, -1, -1]
    assert training_labels(feature, refs, measurement_delay_ms=0)[0].tolist() == [0, 0, 0]


def test_uncovered_candidate_loss_and_padding_do_not_invent_class_targets():
    v = torch.tensor([[0., 0., 0.]], requires_grad=True)
    c = torch.tensor([[[0., 0.], [4., -2.], [1., 2.]]], requires_grad=True)
    loss = supervised_loss(v, c, torch.tensor([[1., 1., 0.]]), torch.tensor([[True, True, False]]), torch.tensor([[1, -1, -1]]))
    loss.backward()
    assert c.grad[0, 0, 1] < 0 and c.grad[0, 0, 0] > 0
    assert torch.equal(c.grad[0, 1:], torch.zeros_like(c.grad[0, 1:]))
    assert v.grad[0, 1] < 0 and v.grad[0, 2] == 0
