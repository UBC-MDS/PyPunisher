"""
Criterion Tests
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from pypunisher.metrics.criterion import aic, bic
from sklearn.linear_model import LinearRegression
from tests._test_data import X_train

model = LinearRegression()


# -----------------------------------------------------------------------------
# `model` Param
# -----------------------------------------------------------------------------

def test_metric_model_parm():
    for kind in ("invalid", model):
        for metric in (aic, bic):
            if isinstance(kind, str):
                with pytest.raises(TypeError):
                    metric(kind, data=X_train)
            else:
                metric(kind, data=X_train)


# -----------------------------------------------------------------------------
# `data` Param
# -----------------------------------------------------------------------------


def test_metric_data_parm():
    for kind in ("invalid", X_train):
        for metric in (aic, bic):
            if isinstance(kind, str):
                with pytest.raises(TypeError):
                    metric(model, data=kind)
            else:
                metric(model, data=kind)


# -----------------------------------------------------------------------------
# Metric output
# -----------------------------------------------------------------------------


def test_metric_output():
    for metric in (aic, bic):
        assert isinstance(metric(model, data=X_train), float)
