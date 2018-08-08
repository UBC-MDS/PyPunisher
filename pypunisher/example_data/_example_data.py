#!/usr/bin/env python

"""
Test Data
=========
Generate: y = x + e, where e ~ Uniform(0, 50) and
`x` is embedded as the middle column in a zero matrix.
That is, only ONE column is predictive of y, the rest are
trivial column vectors.

X_train : 2D array
    Training Features.
X_val : 2D array
    Validation Features.
y_train : 1D array
    Training labels.
y_val : 1D array
    Validation Labels
true_best_features : int, list
    Denotes the best features
    that is actually predictive of the response.
"""
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 99
FEATURES = 20
OBSERVATIONS = 501
sig_features = [7,13,18]

np.random.seed(SEED)
X = np.zeros((OBSERVATIONS, FEATURES))
y = np.arange(OBSERVATIONS)
for i in range(len(sig_features)):
    X[:, sig_features[i]] = y + np.random.uniform(0, 50, size=OBSERVATIONS)

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED)

true_best_features = sig_features