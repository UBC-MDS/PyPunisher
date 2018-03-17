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
from tests._test_data import X_train


def test_backward_params():
    """
    Check parameters to `backward()` raise when expected.
    """
    msg = "`n_features` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        backward(n_features=-0.5, min_change=None)

    msg = "`min_change` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        backward(n_features=None, min_change=-0.75)

    msg = "`min_change` must be of type int or float."
    with pytest.raises(TypeError, match=msg):
        backward(min_change='-0.75', n_features=None)

    # Note: items in backticks (``) will be in alphabetical order.
    msg = "At least one of `min_change` and `n_features` must be None."
    with pytest.raises(TypeError, match=msg):
        backward(n_features=0.5, min_change=0.3)

    msg = "^If a float, `n_features` must be on"
    with pytest.raises(ValueError, match=msg):
        backward(n_features=1.5)

    msg = "less than 2 features present."
    with pytest.raises(IndexError, match=msg):
        backward(X_train=ones((501, 1)), X_val=ones((501, 1)))


# -----------------------------------------------------------------------------
# Test Exhausting loop
# -----------------------------------------------------------------------------

def test_loop_exhaust():
    """Text Exhausting backward()'s loop."""
    backward(n_features=X_train.shape[-1], min_change=None, _do_skip=False)
