"""
Forward Selection Tests
"""

import os
import sys
import pytest
from copy import deepcopy

# sklearn helpers
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from pypunisher.selection_engines.forward import ForwardSelection
from tests._test_data import X_train, y_train, X_test, y_test

DEFAULT_FORWARD_SELECTION_PARAMS = {
    'model': LinearRegression(), 'X_train': X_train, 'y_train': y_train,
    'X_test': X_test, 'y_test': y_test, 'verbose': True, 'criterion': None
}


@pytest.fixture
def selection():
    return ForwardSelection(**DEFAULT_FORWARD_SELECTION_PARAMS)


# -----------------------------------------------------------------------------
# Test inputs
# -----------------------------------------------------------------------------

def test_input_types():
    """
    Check input types of ForwardSelection class
    """
    for k in ('X_train', 'y_train', 'X_test', 'y_test'):
        d = deepcopy(DEFAULT_FORWARD_SELECTION_PARAMS)
        d[k] = 12345
        with pytest.raises(TypeError):
            ForwardSelection(**d)


def test_sklearn_model_methods():
    """
    Check that an attribute error gets raised if the sklearn
    model does not have all 3 methods: fit, predict, and score
    """
    for method in ('fit', 'predict', 'score'):
        model = LinearRegression()
        delattr(model, method)
        with pytest.raises(AttributeError):
            ForwardSelection(model, method, verbose=True, )


def test_forward_attributes(selection):
    """
    Check attribute types of that attributes of backward methods are correct
    """
    msg = "Number of features must be a positive integer or float"
    with pytest.raises(TypeError, match=msg):
        selection.forward(n_features=-0.5)

    msg = "n_features must be None for min_change to work"
    with pytest.raises(TypeError, match=msg):
        selection.forward(n_features=0.5, min_change=0.5)


# -----------------------------------------------------------------------------
# Test outputs
# -----------------------------------------------------------------------------


def test_output_type(selection):
    """
    Test that output type is a list
    """
    assert isinstance(selection.forward, list)


def test_output_values(selection):
    """
    test that ForwardSelection selects best features
    """
    assert selection.forward == [1, 5, 7]
