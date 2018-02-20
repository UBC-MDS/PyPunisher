"""
Feature Selection Tests
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd


sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

import pypunisher.feature_selection


def generate_data():
  df = pd.DataFrame(np.random.randn(100, 40))
  split = np.random.rand(len(df)) < 0.7
  X = df.iloc[:,1:]
  y = df.iloc[:,0]
  Xtrain = X[split]
  Xtest = X[~split]
  ytrain = y[split]
  ytest = X[~split]

def test_forward_selection():
  print('test goes here')

def test_backward_selection():
  print('test goes here')