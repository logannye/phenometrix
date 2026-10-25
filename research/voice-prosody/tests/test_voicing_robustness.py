import hashlib
import numpy as np
import pytest
from phenometrix_prosody.voicing_robustness_inputs import draw, construct, channel, seed_for, diagnostic_draw


def test_seed_and_draw_are_label_independent_and_repeatable():
    assert seed_for('F02_si1', 0) == int.from_bytes(hashlib.sha256(b'phenometrix-voicing-robustness-v1-20261025:F02_si1:0').digest()[:8], 'little')
    assert draw('x',3) == draw('x',3)
    assert draw('x',3) != draw('y',3)
    for slot in range(4):
        p = draw('x',slot)
        assert -18 <= p['gain_db'] <= 0
        assert p['noise'] == 'none' if slot < 3 else 5 <= p['snr_db'] <= 25


def test_filter_recurrences_and_prefix_causality():
    x = np.array([1., -.5, .2, 0.])
    for slot in (1,2):
        p = draw('x',slot); y = channel(x,p); gain=10**(p['gain_db']/20)
        if slot == 1:
            expected=(x-p['a']*np.r_[0,x[:-1]])/(1+p['a'])*gain
        else:
            expected=np.zeros(4)
            for i in range(4): expected[i]=(1-p['b'])*x[i]*gain+(p['b']*expected[i-1] if i else 0)
        np.testing.assert_allclose(y,expected,rtol=1e-15,atol=1e-15)
        np.testing.assert_array_equal(y[:2],channel(x[:2],p))


def test_noise_snr_zero_signal_and_no_saturation():
    x=np.sin(np.arange(4800)*.1).astype(np.float32)
    for name in ('white_snr8','ar_snr18'):
        p=diagnostic_draw('x',name); y,r=construct(x,p)
        assert abs(r['achieved_snr_db']-p['snr_db']) < 1e-5
        z,zr=construct(np.zeros(4800),p)
        assert not z.any() and zr['achieved_snr_db'] is None
        # Fixed realized noise and whole-recording scale, not prefix RMS.
        eps=np.random.default_rng(p['noise_seed']).standard_normal(len(x))
        if p['noise']=='ar1':
            for i in range(len(eps)): eps[i]=.6*eps[i]+(.8*eps[i-1] if i else 0)
        np.testing.assert_array_equal(y,(channel(x,p)+eps*r['noise_scale']).astype(np.float32))
    y,r=construct(np.full(10,10.),dict(filter='none',noise='none',gain_db=0.))
    assert r['outside_unit_count']==10 and y.max()==10


def test_reject_nonfinite():
    with pytest.raises(ValueError): construct([np.nan],draw('x',0))
    with pytest.raises(ValueError): draw('x',4)
