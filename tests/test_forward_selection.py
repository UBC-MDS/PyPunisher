#!/usr/bin/env python

"""
Tests Specific to Forward Selection
===================================
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from tests._wrappers import forward
from pypunisher.example_data._example_data import X_train


# -----------------------------------------------------------------------------
# Test `forward()` Params
# -----------------------------------------------------------------------------

def test_n_features_greater_than_zero_forward():
    """
    Check that `n_features` is required to be > 0.
    """
    msg = "`n_features` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        forward(min_change=None, n_features=-0.75)


def test_min_change_greater_than_zero_forward():
    """
    Check that `min_change` is required to be > 0.
    """
    msg = "`min_change` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        forward(min_change=-0.5, n_features=None)


def test_n_features_fails_on_string_forward():
    """
    Check that forward raises when passed a string
    for `n_features`.
    """
    msg = "`n_features` must be of type int or float."
    with pytest.raises(TypeError, match=msg):
        forward(min_change=None, n_features='-0.75')


def test_min_change_fails_on_string_forward():
    """
    Check that forward raises when passed a string
    for `min_change`.
    """
    msg = "`min_change` must be of type int or float."
    with pytest.raises(TypeError, match=msg):
        forward(min_change='-0.75', n_features=None)


def test_both_non_none_forward():
    """
    Check `forward()` raise when at least one
    of `min_change` or `n_features` are not None.
    """
    # Note: items in backticks (``) will be in alphabetical order.
    msg = "At least one of `min_change` and `n_features` must be None."
    with pytest.raises(TypeError, match=msg):
        forward(min_change=0.5, n_features=0.3)


# -----------------------------------------------------------------------------
# Test Exhausting loop
# -----------------------------------------------------------------------------

def test_loop_exhaust():
    """Text Exhausting forwards()'s loop."""
    # Should not raise.
    forward(n_features=X_train.shape[-1], min_change=None, _do_not_skip=False)
