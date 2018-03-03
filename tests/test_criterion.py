"""
Criterion Tests
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

import numpy as np
import statsmodels.api as sm
from pypunisher.metrics.criterion import aic, bic
from sklearn.linear_model import LinearRegression
from tests._test_data import X_train, y_train

COMP_TOLERANCE = 0.5  # comaprision tolerance between floats

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

sm_model = sm.OLS(y_train, X_train)
res = sm_model.fit()
sm_aic = res.aic
sm_bic = res.bic

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


# -----------------------------------------------------------------------------
# Output Value (Compare against the Stats Models Package).
# -----------------------------------------------------------------------------


def test_metric_output_value():
    for metric, comparision in zip((aic, bic), (sm_aic, sm_bic)):
        ours = metric(model, data=X_train)
        ours == pytest.approx(comparision, abs=COMP_TOLERANCE)
