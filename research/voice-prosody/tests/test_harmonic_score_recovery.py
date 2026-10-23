import json
import numpy as np
import pytest
from phenometrix_prosody.recover_harmonic_scores import json_scalar

def test_numpy_gate_scalar_is_serialized_without_changing_value():
    values={'pass':np.bool_(True),'fail':np.float64(.1)>=.95,'count':np.int64(95),'rate':np.float32(.5)}
    with pytest.raises(TypeError):json.dumps(values)
    assert json.loads(json.dumps(values,default=json_scalar,allow_nan=False))=={'pass':True,'fail':False,'count':95,'rate':.5}

def test_recovery_does_not_accept_nonfinite_values_or_arrays():
    with pytest.raises(ValueError):json.dumps({'x':np.float64(np.nan)},default=json_scalar,allow_nan=False)
    with pytest.raises(TypeError):json.dumps({'x':np.array([1.])},default=json_scalar,allow_nan=False)
