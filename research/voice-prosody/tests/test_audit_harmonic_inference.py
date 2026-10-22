import numpy as np
import pytest
import torch

from phenometrix_prosody.audit_harmonic_inference import (
    independent_logits, emit, initial_tensor_hash, original_probability, compare_arrays,
)
from phenometrix_prosody.harmonic_selector_model import HarmonicSelector
from phenometrix_prosody.selector_model import prepare_selector_batch
from phenometrix_prosody.model import VoiceCandidateModel, ModelConfig


def fixture():
    x = np.zeros((3, 57), np.float32); hz = np.full((3, 6), np.nan); mask = np.zeros((3, 6), bool)
    for row, frequencies in enumerate([[100, 200, 300], [150], []]):
        for i, frequency in enumerate(frequencies):
            scores = [.2, .3 + i*.05, 0, .1, .4]
            x[row,i*8:(i+1)*8] = [1,np.log2(frequency/50),*scores,max(scores)]
            hz[row,i],mask[row,i]=frequency,True
        x[row,48:]=[-3,.2,0,0,.1,.5,.6,len(frequencies)/6,1]
    return prepare_selector_batch(x,hz,mask,representation='shared-candidates')


@pytest.mark.parametrize('mode',['absolute','relative'])
def test_independent_tensor_equations_match_class_and_mask(mode):
    torch.manual_seed(25); model=HarmonicSelector(mode); batch=fixture()
    model.fit_scaling(batch,split='train')
    with torch.inference_mode():
        expected=model(batch)
        actual=independent_logits(model.state_dict(),mode,batch.features,batch.candidate_mask)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert torch.isfinite(actual).all()


def test_emission_has_independent_external_gate_mask_and_tie_policy():
    batch=fixture()
    logits=torch.tensor([[0.,0.,0.,100.,100.,100.],[0.]*6,[0.]*6])
    state,hz=emit(logits,batch.candidate_hz.numpy(),batch.candidate_mask.numpy(),np.array([.5,.49,.8]))
    assert state.tolist()==[1,0,2]
    assert hz[0]==100 and np.isnan(hz[1:]).all()
    good=compare_arrays((state,hz),state,hz,batch,np.array([.5,.49,.8]))
    assert good['state_mismatches']==good['hz_mismatches']==good['nonmember_emissions']==0
    changed=hz.copy();changed[0]=123
    bad=compare_arrays((state,changed),state,hz,batch,np.array([.5,.49,.8]))
    assert bad['nonmember_emissions']==1 and bad['hz_mismatches']==1


def test_initialization_reconstructs_seed_with_only_scaling_buffers():
    seed=20261022;torch.manual_seed(seed);model=HarmonicSelector('absolute','nearest')
    model.fit_scaling(fixture(),split='train')
    before=initial_tensor_hash('absolute','nearest',seed,model.state_dict())
    with torch.no_grad():
        for p in model.parameters():p.add_(1)
    assert initial_tensor_hash('relative','acceptable-set',seed,model.state_dict())==before
    assert initial_tensor_hash('relative','acceptable-set',seed+1,model.state_dict())!=before


def test_original_voice_equations_use_same_frozen_float64_sigmoid():
    torch.manual_seed(12);model=VoiceCandidateModel(ModelConfig(kind='framewise-mlp'))
    batch=fixture()
    with torch.inference_mode():
        expected=torch.sigmoid(model(batch.features,batch.candidate_mask,model.initial_state(3))[0][:,0].double()).numpy()
        got=original_probability(model.state_dict(),batch.features.numpy())
    np.testing.assert_array_equal(got,expected)
