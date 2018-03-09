"""

    Run Tests Common to Forward and Backward Selection
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import sys
import pytest
from copy import deepcopy

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from pypunisher import Selection
from tests._wrappers import forward, backward
from tests._test_data import TRUE_BEST_FEATURE
from tests._defaults import DEFAULT_SELECTION_PARAMS


# -----------------------------------------------------------------------------
# Test Inputs: Types
# -----------------------------------------------------------------------------


def test_input_types():
    """
    Check input types when Initializing Selection().
    """
    for k in ('X_train', 'y_train', 'X_val', 'y_val'):
        d = deepcopy(DEFAULT_SELECTION_PARAMS)
        d[k] = 12345
        with pytest.raises(TypeError):
            Selection(**d)


# -----------------------------------------------------------------------------
# Test inputs: Model Attributes
# -----------------------------------------------------------------------------


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
        d = deepcopy(DEFAULT_SELECTION_PARAMS)
        d['model'] = gen_dummy_model(exclude=method_to_drop)
        with pytest.raises(AttributeError):
            Selection(**d)


# -----------------------------------------------------------------------------
# Outputs: Run the Forward and Backward Selection Algorithms
# -----------------------------------------------------------------------------

forward_output = forward()

# Force the backward selection algorithm to
# select the single feature it thinks is most predictive.
# If implemented correctly, `backward()` should be able to
# identify `TRUE_BEST_FEATURE` as predictive.
backward_output = backward(n_features=1)


# -----------------------------------------------------------------------------
# Test outputs: Type
# -----------------------------------------------------------------------------


def output_type(output):
    """
    Test that output type is a list.
    """
    msg = "Output from the algorithm was not a list."
    assert isinstance(output, list), msg


def test_fsel_output_type():
    output_type(forward_output)


def test_bsel_output_type():
    output_type(backward_output)


# -----------------------------------------------------------------------------
# Test outputs: Selection of the Predictive Feature
# -----------------------------------------------------------------------------


def output_values(output):
    """
    Test that ForwardSelection selects the best feature
    in the contrived data.
    """
    msg = "The algorithm failed to select the predictive feature."
    assert TRUE_BEST_FEATURE in output, msg


def test_fsel_output_values():
    output_values(forward_output)


def test_bsel_output_values():
    output_values(backward_output)
