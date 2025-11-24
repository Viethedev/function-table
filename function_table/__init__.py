"""FunctionTable package

Provides `FunctionTable`, a callable object learned from input/output table using XGBoost.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import xgboost as xgb

__all__ = ["FunctionTable"]


class FunctionTable:
    """A callable object (function) defined by inputs and outputs (as table).

    Notes
    -----
    - Attempts to use GPU when available.
    - Immutable after construction.
    """

    __slots__ = ("_models", "_number_of_inputs", "_number_of_outputs")

    def __init__(self, inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]):
        X = np.array(inputs, dtype=float)
        Y = np.array(outputs, dtype=float)

        object.__setattr__(self, "_number_of_inputs", X.shape[1])
        object.__setattr__(self, "_number_of_outputs", Y.shape[1])
        object.__setattr__(self, "_models", [])

        # Auto-detect device internally
        try:
            # quick test to see if GPU method is available
            _ = xgb.train({"tree_method": "gpu_hist"}, xgb.DMatrix(np.array([[0.0]])), num_boost_round=1)
            tree_method = "gpu_hist"
        except Exception:
            tree_method = "hist"

        # Train one booster per output dimension
        for i in range(self.number_of_outputs):
            y = Y[:, i]
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                "objective": "reg:squarederror",
                "max_depth": 10,
                "eta": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "verbosity": 0,
                "tree_method": tree_method,
            }
            booster = xgb.train(params, dtrain, num_boost_round=250)
            self._models.append(booster)

    def __call__(self, *args, numpy: bool = False, round_digits: Optional[int] = None):
        # Accept either a single sequence/ndarray (batch or single-row) or multiple scalar args
        if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
            arr = np.atleast_2d(np.array(args[0], dtype=float))
        else:
            arr = np.array(args, dtype=float).reshape(1, -1)

        if arr.shape[1] != self.number_of_inputs:
            raise ValueError(
                f"Input shape mismatch: expected {self.number_of_inputs} features, got {arr.shape[1]}"
            )

        dtest = xgb.DMatrix(arr)
        predictions = np.array([booster.predict(dtest) for booster in self._models]).T

        if round_digits is not None:
            predictions = predictions.round(round_digits)

        return predictions if numpy else predictions.tolist()

    def __repr__(self):
        return f"{super().__repr__()} [inputs={self._number_of_inputs}, outputs={self._number_of_outputs}]"

    def __setattr__(self, key, value):
        raise AttributeError("FunctionTable objects are immutable.")
