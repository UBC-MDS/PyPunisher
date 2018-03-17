#!/usr/bin/env python

"""
Run Tests Common to Forward and Backward Selection
==================================================
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


def test_too_few_features():
    """
    Check that there are enough features for
    for selection to be a coherent goal (i.e., >= 2).
    """
    X_train = DEFAULT_SELECTION_PARAMS['X_train']
    X_train = X_train[:, 0:1]
    with pytest.raises(IndexError):
        forward(n_features=1, X_train=X_train)
    with pytest.raises(IndexError):
        backward(n_features=1, X_train=X_train)


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

# Run using the other parameter option
forward_output += forward(n_features=1, min_change=None)

# Force the backward selection algorithm to
# select the single feature it thinks is most predictive.
# If implemented correctly, `backward()` should be able to
# identify `TRUE_BEST_FEATURE` as predictive.
backward_output = backward(n_features=1)

# Run using the other parameter option
backward_output += backward(min_change=0.0001, n_features=None)


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
# Test `n_features`
# -----------------------------------------------------------------------------


def test_n_features():
    """
    Test various version of n_features
    """
    for n_features in (10000, -1.0):
        with pytest.raises(ValueError):  # should raise
            backward(n_features=n_features)
    backward(n_features=0.5)  # this should not raise


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


# -----------------------------------------------------------------------------
# Testing that forward selection works with 'aic' and 'bic' criterion
# -----------------------------------------------------------------------------

def test_fsel_aic_output():
    forward_output = forward(n_features=2, min_change=None, criterion='aic')
    assert len(forward_output) > 0


def test_fsel_bic_output():
    forward_output = forward(n_features=2, min_change=None, criterion='bic')
    assert len(forward_output) > 0


# -----------------------------------------------------------------------------
# Test that backward selection works with 'aic' and 'bic' criterion
# -----------------------------------------------------------------------------

def test_bsel_aic_output():
    backward_output = backward(n_features=2, min_change=None, criterion='aic')
    assert len(backward_output) >= 1


def test_bsel_bic_output():
    backward_output = backward(n_features=2, min_change=None, criterion='bic')
    assert len(backward_output) >= 1


# -----------------------------------------------------------------------------
# Test that forward and backward selection work with 'min_change' arg
# -----------------------------------------------------------------------------

def test_fsel_min_change_output():
    forward_output = forward(n_features=None, min_change=1, criterion=None)
    assert len(forward_output) >= 1


def test_bsel_min_change_output():
    backward_output = backward(n_features=None, min_change=10, criterion='aic')
    assert len(backward_output) >= 1


# -----------------------------------------------------------------------------
# Test that forward and backward selection work with verbose=True arg
# -----------------------------------------------------------------------------

def test_fsel_verbose_output():
    forward_output = forward(n_features=None, min_change=1, verbose=True)
    assert len(forward_output) >= 1


def test_bsel_verbose_output():
    backward_output = backward(n_features=2, min_change=None, verbose=True)
    assert len(backward_output) >= 1