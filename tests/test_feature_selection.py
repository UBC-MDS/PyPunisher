"""
Feature Selection Tests
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
from sklearn import linear_model
from sklearn.utils.testing import assert_raise_message # helper function to test the message raised in an exception


sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

import pypunisher.feature_selection

df = pd.DataFrame(np.random.randn(100, 40))
split = np.random.rand(len(df)) < 0.7
X = df.iloc[:,1:]
y = df.iloc[:,0]
X_train = X[split]
X_test = X[~split]
y_train = y[split]
y_test = X[~split]
lin_reg = linear_model.LinearRegression()

@pytest.fixture
def selection():
  return Selection(lin_reg, X_train, y_train, X_test, y_test, verbose=True)


def test_forward_selection(selection):
  # TODO write unit tests for expected output of forward selection method
  print('test goes here')

def test_backward_selection(selection):
  # TODO write unit tests for expected output of backward selection method
  print('test goes here')

def test_error_handling(selection):
  # examples of error handling

  # TODO enhance error handling to make sure the sklearn model has "fit", "score", and "predict" methods
  msg = "Model must be an sklearn method"
  assert_raise_message(ValueError, msg, selection)

  # TODO enhance error handling to make sure that an error gets returned if Xtrain, ytrain, Xval, yval is not the correct format (i.e. not a numpy array)
  msg = "Training data must be a 2D array"
  assert_raise_message(ValueError, msg, selection)

  msg = "Test data must be a 2D array"
  assert_raise_message(ValueError, msg, selection)