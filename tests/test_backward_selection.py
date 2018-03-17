#!/usr/bin/env python

"""
Tests Specific to Backward Selection
====================================
"""
import os
import sys

import pytest
from numpy import ones

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from tests._wrappers import backward
from pypunisher._example_data import X_train


# -----------------------------------------------------------------------------
# Test `backward()` Params
# -----------------------------------------------------------------------------


def test_n_features_greater_than_zero_backward():
    """
    Check `backward()`'s `n_features` raises when
    not greater than zero
    """
    msg = "`n_features` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        backward(n_features=-0.5, min_change=None)


def test_min_change_greater_than_zero_backward():
    """
    Check `backward()`'s `min_change` raises when
    not greater than zero
    """
    msg = "`min_change` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        backward(n_features=None, min_change=-0.75)


def test_min_change_fails_on_string_backward():
    """
    Check that backward raises when passed a string
    for `min_change`.
    """
    msg = "`min_change` must be of type int or float."
    with pytest.raises(TypeError, match=msg):
        backward(min_change='-0.75', n_features=None)


def test_n_features_fails_on_string_backward():
    """
    Check that backward raises when passed a string
    for `n_features`.
    """
    msg = "`n_features` must be of type int or float."
    with pytest.raises(TypeError, match=msg):
        backward(min_change=None, n_features='-0.75')


def test_both_non_none_backward():
    """
    Check `backward()` raise when at least one
    of `min_change` or `n_features` are not None.
    """
    # Note: items in backticks (``) will be in alphabetical order.
    msg = "At least one of `min_change` and `n_features` must be None."
    with pytest.raises(TypeError, match=msg):
        backward(n_features=0.5, min_change=0.3)


def test_float_greater_than_one_raises_backward():
    """
    Test that float values not on (0, 1) raise.
    """
    msg = "^If a float, `n_features` must be on"
    with pytest.raises(ValueError, match=msg):
        backward(n_features=1.5)


def test_min_features_requirement_backward():
    """
    Check that the requirement that at least
    two features must be present.
    """
    msg = "less than 2 features present."
    with pytest.raises(IndexError, match=msg):
        backward(X_train=ones((501, 1)), X_val=ones((501, 1)))


# -----------------------------------------------------------------------------
# Test Exhausting loop
# -----------------------------------------------------------------------------

def test_loop_exhaust():
    """Text Exhausting backward()'s loop."""
    backward(n_features=X_train.shape[-1], min_change=None, _do_not_skip=False)
