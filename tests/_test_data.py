"""

    Test Data
    ~~~~~~~~~

    Generate: y = x + e, where e ~ Uniform(0, 50) and
    `x` is embedded as the middle column in a zero matrix.
    That is, only ONE column is predictive of y, the rest are
    trivial column vectors.

"""
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 99

features = 20
obs = 501
middle_feature = features // 2

np.random.seed(SEED)
X = np.zeros((obs, features))
y = np.arange(obs)
X[:, middle_feature] = y + np.random.uniform(0, 50, size=obs)

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED)

TRUE_BEST_FEATURE = middle_feature

# Visualize ---
# import matplotlib.pyplot as plt
# plt.scatter(X[:, middle_feature], y, s=1)
