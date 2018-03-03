"""

    Criterion Tests
    ~~~~~~~~~~~~~~~

"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

import statsmodels.api as sm
from pypunisher.metrics.criterion import aic, bic
from sklearn.linear_model import LinearRegression
from tests._test_data import X_train, y_train

COMP_TOLERANCE = 0.5  # comparision tolerance between floats

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

sm_model = sm.OLS(y_train, X_train)
res = sm_model.fit()
sm_aic = res.aic
sm_bic = res.bic

sk_model = LinearRegression()


# -----------------------------------------------------------------------------
# `model` Param
# -----------------------------------------------------------------------------


def test_metric_model_parm():
    """Test that the `model` params in `aic()` and `bic()`
    will raise a TypeError when passed something other
    than a sk-learn model."""
    for kind in ("invalid", sk_model):
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
    """Test that the `data` params in `aic()` and `bic()`
    will raise a TypeError when passed something other
    than an ndarray."""
    for kind in ("invalid", X_train):
        for metric in (aic, bic):
            if isinstance(kind, str):
                with pytest.raises(TypeError):
                    metric(sk_model, data=kind)
            else:
                metric(sk_model, data=kind)


# -----------------------------------------------------------------------------
# Metric output
# -----------------------------------------------------------------------------


def test_metric_output():
    """Test that both metrics (`aic()` and `bic()`) return
    floating point numbers."""
    for metric in (aic, bic):
        assert isinstance(metric(sk_model, data=X_train), float)


# -----------------------------------------------------------------------------
# Output Value (Compare against the Stats Models Package).
# -----------------------------------------------------------------------------


def test_metric_output_value():
    """Test that the actual AIC and BIC values computed by
    our functions match that computed by a well-respected
    statistical library in Python (StatsModels)."""
    for metric, comparision in zip((aic, bic), (sm_aic, sm_bic)):
        ours = metric(sk_model, data=X_train)
        ours == pytest.approx(comparision, abs=COMP_TOLERANCE)
