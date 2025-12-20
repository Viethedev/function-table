"""FunctionTable package

Provides 'FTable', a callable object learned from input/output table using XGBoost
"""
from __future__ import annotations
from typing import Sequence
from numbers import Number
import warnings
import numpy as np
import xgboost as xgb

__all__ = ["FTable"]

# Table definition
Table = Sequence[Sequence[Number]]

# FunctionTable implementation
class FTable:
    """A callable object (function) defined by inputs and outputs (as table)

    Notes
    -----
    - Attempts to use GPU when available
    - Immutable after construction
    """

    __slots__ = ("_models", "_number_of_inputs", "_number_of_outputs", "_call")

    def __init__(self, inputs: Table, outputs: Table):
        """Initialize FTable object"""
        X = np.asarray(inputs, dtype=float)
        Y = np.asarray(outputs, dtype=float)

        # Normalize shapes:
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if Y.ndim == 1:
            # If length matches number of input rows, treat as column vector
            if Y.shape[0] == X.shape[0]:
                Y = Y.reshape(-1, 1)
            else:
                Y = Y.reshape(1, -1)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Number of input rows ({X.shape[0]}) does not match number of output rows ({Y.shape[0]})")

        if Y.shape[1] == 0:
            raise ValueError("Outputs table is empty")

        object.__setattr__(self, "_number_of_inputs", X.shape[1])
        object.__setattr__(self, "_number_of_outputs", Y.shape[1])
        
        if X.shape[1] == 0:
            warnings.warn("Zero-input FTable. Not caching XGBoost. Returning outputs", stacklevel=2)
            object.__setattr__(self, "_models", Y.setflags(write=False))
            object.__setattr__(self, "_call", self._return_constant)
            return
        
        # Train one booster per output dimension
        models = []
        params = {
            "objective": "reg:squarederror",
            "max_depth": 10,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "verbosity": 0,
            "device": "cuda",
            "predictor": "cpu_predictor"
        }

        with warnings.catch_warnings():
            # Hide harmless training chatter
            warnings.filterwarnings("ignore", module="xgboost")
            try:
                for i in range(self._number_of_outputs):
                    y = Y[:, i]
                    dtrain = xgb.DMatrix(X, label=y)
                    booster = xgb.train(params, dtrain, num_boost_round=250)
                    models.append(booster)
            except xgb.core.XGBoostError as e:
                error = str(e).splitlines()[-1]
                raise ValueError(error) from None

        object.__setattr__(self, "_models", tuple(models))
        object.__setattr__(self, "_call", self._predict)

    # Internal _call
    def _predict(self, args) -> np.ndarray:
        # Accept either a single sequence/ndarray (batch or single-row) or multiple scalar args
        if len(args) == 1 and isinstance(args[0], Sequence):
            arr = np.atleast_2d(np.array(args[0], dtype=float))
        else:
            arr = np.array(args, dtype=float).reshape(1, -1)

        if arr.shape[1] != self._number_of_inputs:
            raise ValueError(f"Input shape mismatch: expected {self._number_of_inputs} features, got {arr.shape[1]}")

        dtest = xgb.DMatrix(arr)
        return np.array([booster.predict(dtest) for booster in self._models]).T

    def _return_constant(self, args) -> np.ndarray:
        return self._models.copy()

    # Public attributes access
    @property
    def number_of_inputs(self) -> int:
        return self._number_of_inputs

    @property
    def number_of_outputs(self) -> int:
        return self._number_of_outputs

    # Constant table check
    @property
    def is_constant(self) -> bool:
        return isinstance(self._models, np.ndarray)

    # Utilities
    def __repr__(self) -> str:
        """Return friendly representation (with i/o shape)."""
        return f"{super().__repr__()} [inputs={self._number_of_inputs}, outputs={self._number_of_outputs}]"

    # Immutable enforcement
    def __setattr__(self, key, value):
        """Disabled due to immutability"""
        raise TypeError("'FTable' is immutable")

    def __getattr__(self, key):
        """Disabled due to internal protection"""
        raise TypeError("'FTable' is protected")

    # Public __call__ access
    def __call__(self, *args, numpy: bool = False, round_digits: int = None) -> Table:
        """
        Evaluate the FTable on the given input(s).

        Parameters
        ----------
        *args : scalar(s) or sequence(s)
            Single row or batch of input values.
        numpy : bool, optional
            If True, returns a NumPy array; otherwise, returns a list of lists.
        round_digits : int, optional
            Round predictions to this many decimal places.

        Returns
        -------
        Sequence[Sequence[float]] or np.ndarray
            Predicted output(s) for the given input(s).
        """
        predictions = self._call(args)
        if round_digits:
            predictions.round(round_digits, out=predictions)

        return predictions if numpy else predictions.tolist()
