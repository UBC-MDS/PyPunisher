"""

    Tests Specific to Backward Selection
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from tests._wrappers import backward


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

    # Note: items in backticks (``) will be in alphabetical order.
    msg = "At least one of `min_change` and `n_features` must be None."
    with pytest.raises(TypeError, match=msg):
        backward(n_features=0.5, min_change=0.3)
    
    msg = "`criterion` must be one of: None, 'aic', 'bic'."
    with pytest.raises(ValueError, match=msg):
        backward(n_features=0.5, criterion='acc')
