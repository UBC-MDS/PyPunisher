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
from tests._test_data import X_train


def test_forward_params():
    """
    Check parameters to `forward()` raise when expected.
    """
    msg = "`min_change` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        forward(min_change=-0.5, n_features=None)

    msg = "`n_features` must be greater than zero."
    with pytest.raises(ValueError, match=msg):
        forward(min_change=None, n_features=-0.75)

    msg = "`n_features` must be of type int or float."
    with pytest.raises(TypeError, match=msg):
        forward(min_change=None, n_features='-0.75')

    # Note: items in backticks (``) will be in alphabetical order.
    msg = "At least one of `min_change` and `n_features` must be None."
    with pytest.raises(TypeError, match=msg):
        forward(min_change=0.5, n_features=0.3)


# -----------------------------------------------------------------------------
# Test Exhausting loop
# -----------------------------------------------------------------------------

def test_loop_exhaust():
    """Text Exhausting forwards()'s loop."""
    forward(n_features=X_train.shape[-1], min_change=None, _do_skip=False)
