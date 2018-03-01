"""
Backward Selection Tests
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# numpy testing tools
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_less
from numpy.testing import assert_approx_equal

# sklearn helpers
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

from pypunisher.selection_engines.backward import BackwardSelection

X = pd.DataFrame(np.random.randn(100, 40))
y = pd.DataFrame(np.random.randn(100, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


@pytest.fixture
def selection():
  return BackwardSelection(LinearRegression(), X_train, y_train, X_test, y_test, verbose=True)


# test inputs
def test_input_types():
  '''
  type checking class inputs
  '''
  msg = "X_train must be numpy array"
  with pytest.raises(TypeError, match=msg):
    BackwardSelection(LinearRegression(), 12345, y_train, X_test, y_test, verbose=True)


def test_sklearn_model_methods():
  '''
  test that error is returned if model doesn't have "fit", "predict", and "score" methods
  '''


# test output
def test_output_type(selection):
  '''
  test that output type is a list
  '''
  assert isinstance(selection, list)


def test_output_values(selection):
  '''
  test that output type is a list
  '''
  assert selection == [1,5,7]

