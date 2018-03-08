"""

    Tests Specific to Forward Selection
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from tests._wrappers import forward


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

    # Note: items in backticks (``) will be in alphabetical order.
    msg = "At least one of `max_features` and `min_change` must be None."
    with pytest.raises(TypeError, match=msg):
        forward(min_change=0.5, max_features=0.3)
