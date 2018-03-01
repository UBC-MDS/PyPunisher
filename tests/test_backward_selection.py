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

X = np.random.randn(100, 40)
y = np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


@pytest.fixture
def selection():
  return BackwardSelection(LinearRegression(), X_train, y_train, X_test, y_test, verbose=True)


# test inputs
def test_input_types():
  '''
  Check input types of BackwardSelection class
  '''
  with pytest.raises(TypeError):
    BackwardSelection(LinearRegression(), 12345, y_train, X_test, y_test)
  
  with pytest.raises(TypeError):
    BackwardSelection(LinearRegression(), X_train, 12345, X_test, y_test)

  with pytest.raises(TypeError):
    BackwardSelection(LinearRegression(), X_train, y_train, 12345, y_test)

  with pytest.raises(TypeError):
    BackwardSelection(LinearRegression(), X_train, y_train, X_test, 12345)

def test_sklearn_model_methods():
  '''
  Check that an attribute error gets raised if the sklearn model does not have all 3 methods: fit, predict, and score
  '''
  for method in ('fit', 'predict', 'score'):
      model = LinearRegression()
      delattr(model, method)
      with pytest.raises(AttributeError):
        BackwardSelection(model, method, verbose=True)

# test output
def test_output_type(selection):
  '''
  Test that output type is a list
  '''
  assert isinstance(selection.backward, list)


def test_output_values(selection):
  '''
  test that BackwardSelection selects best features
  '''
  assert selection.backward == [1,5,7]

