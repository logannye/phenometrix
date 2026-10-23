import numpy as np

from phenometrix_prosody.pilot import references_from_array


def test_nominal_reference_grid_and_shared_support_mask():
    rows = np.array([[0, 0, 0, 0], [150, 1, .1, .9], [0, 1, 0, 0],
                     [150, 1, .1, .9], [150, 1, .1, .9]], dtype=float)
    references = references_from_array(rows, 4000)
    assert [row.sample for row in references] == [768, 1248, 1728, 2208, 2688]
    assert not references[0].voiced and references[0].f0_hz is None
    assert not references[2].valid
    assert [row.eligible for row in references] == [False, False, False, True, False]
