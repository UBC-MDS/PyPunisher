"""
Criterion Tests
"""

import os
import sys
import pytest

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

import pypunisher.criterion


def test_aic(X_train, y_train, model):
  print('test goes here')

def test_aic_input_model_type():
  '''
  Check if the model passed into AIC is of correct type
  '''
  with pytest.raises(TypeError):
    aic(model, 1)


def test_aic_input_lambda_type():
  '''
  Check if the lambda parameter passed into AIC is an integer
  '''
  
  with pytest.raises(TypeError):
      aic(model, 1.5)


def test_aic_output_type():
  '''
  Check if the value returned by AIC is of type float.
  '''
  
  # assert isinstance(aic(model_1, 1), float)
  
  pass



def test_aic_model1_output_value():
  '''
  Check if the returned value is correct for specified sklearn model
  '''
  # assert aic(model_1, 1) == value
  
  pass


def test_aic_model2_output_value():
  '''
  Check if the returned value is correct for specified sklearn model
  '''
  # assert aic(model_2, 1) == value
  
  pass


def test_bic():
  print('test goes here')

def test_bic_input_model_type():
  '''
  Check if the model passed into BIC is of correct type
  '''
  
  with pytest.raises(TypeError):
      bic(model)


def test_bic_output_type():
  '''
  Check if the value returned by BIC is of type float.
  '''
  # assert isinstance(bic(model_1), float)
  
  pass


def test_bic_model1_output_value():
  '''
  Check if the returned value is correct for specified sklearn model
  '''
  # assert bic(model_1, 1) == value
  
  pass

def test_bic_model2_output_value():
  '''
  Check if the returned value is correct for specified sklearn model
  '''
  # assert bic(model_2, 1) == value
  
  pass

