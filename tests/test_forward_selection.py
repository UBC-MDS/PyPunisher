"""
Forward Selection Tests
"""
import os
import sys
import pytest
from copy import deepcopy

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from tests._test_data import TRUE_BEST_FEATURE
from tests._defaults import DEFAULT_FORWARD_SELECTION_PARAMS
from pypunisher.selection_engines.forward import ForwardSelection


def forward(**kwargs):
    fs = ForwardSelection(**DEFAULT_FORWARD_SELECTION_PARAMS)
    return fs.forward(**kwargs)


# -----------------------------------------------------------------------------
# Test inputs
# -----------------------------------------------------------------------------

def test_input_types():
    """
    Check input types of ForwardSelection class
    """
    for k in ('X_train', 'y_train', 'X_val', 'y_val'):
        d = deepcopy(DEFAULT_FORWARD_SELECTION_PARAMS)
        d[k] = 12345
        with pytest.raises(TypeError):
            ForwardSelection(**d)


def test_sklearn_model_methods():
    """
    Check that an attribute error gets raised if the sklearn
    model does not have all 3 methods: fit, predict, and score
    """
    required_methods = ('fit', 'predict', 'score')

    def gen_dummy_model(exclude):
        class Model(object):
            pass

        for r in required_methods:
            if r != exclude:
                setattr(Model, r, None)

        return Model

    for method_to_drop in required_methods:
        d = deepcopy(DEFAULT_FORWARD_SELECTION_PARAMS)
        d['model'] = gen_dummy_model(exclude=method_to_drop)
        with pytest.raises(AttributeError):
            ForwardSelection(**d)


def test_forward_params():
    """
    Check parameters to `forward()` raise when expected.
    """
    msg = "`min_change` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        forward(min_change=-0.5, max_features=None)

    msg = "`max_features` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        forward(min_change=None, max_features=-0.75)

    msg = "At least one of `min_change` and `max_features` must be None."
    with pytest.raises(TypeError, match=msg):
        forward(min_change=0.5, max_features=0.3)


# -----------------------------------------------------------------------------
# Test outputs
# -----------------------------------------------------------------------------

forward_output = forward()  # run the forward selection algorithm.


def test_output_type():
    """
    Test that output type is a list.
    """
    assert isinstance(forward_output, list)


def test_output_values():
    """
    Test that ForwardSelection selects the best feature
    in the contrived data.
    """
    assert TRUE_BEST_FEATURE in forward_output
