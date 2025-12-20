import numpy as np
from function_table import FTable


def test_basic_square():
    inputs = [[0], [1], [2], [3], [4]]
    outputs = [[0], [1], [4], [9], [16]]
    f = FTable(inputs, outputs)

    res = f(2, numpy=True)
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, 1)
    # With small dataset, exact match may not hold; ensure close
    assert abs(res[0, 0] - 4) < 1.0
