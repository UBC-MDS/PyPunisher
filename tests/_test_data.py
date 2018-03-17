#!/usr/bin/env python

"""
Test Data
=========
Generate: y = x + e, where e ~ Uniform(0, 50) and
`x` is embedded as the middle column in a zero matrix.
That is, only ONE column is predictive of y, the rest are
trivial column vectors.
"""
import numpy as np
from sklearn.model_selection import train_test_split


def test_data(seed=99, add_extra_col=False):
    """Generate training and testing data.

    Args:
        seed : int
            Random seed
        add_extra_col : bool
            If True, add another predictive column
            to the right of the middle column.

    Returns:
        X_train : 2D array
            Training Features.
        X_val : 2D array
            Validation Features.
        y_train : 1D array
            Training labels.
        y_val : 1D array
            Validation Labels
        true_best_feature : int, list
            List if `add_extra_col` is True
            Otherwise, an int.
            Denotes the best feature(s)
            that are actually predictive
            of the response.

    """
    features = 20
    obs = 501
    middle_feature = features // 2

    np.random.seed(seed)
    X = np.zeros((obs, features))
    y = np.arange(obs)
    X[:, middle_feature] = y + np.random.uniform(0, 50, size=obs)
    if add_extra_col:
        X[:, middle_feature + 1] = y + np.random.uniform(0, 50, size=obs)

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=seed)

    if add_extra_col:
        true_best_feature = [middle_feature, middle_feature + 1]
    else:
        true_best_feature = middle_feature
    return X_train, X_val, y_train, y_val, true_best_feature


X_train, X_val, y_train, y_val, true_best_feature = test_data()

# Visualize ---
# import matplotlib.pyplot as plt
# plt.scatter(X[:, middle_feature], y, s=1)
