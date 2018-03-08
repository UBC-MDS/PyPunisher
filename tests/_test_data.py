"""

    Test Data
    ~~~~~~~~~

    Generate y = x + e,

    where e ~ Uniform(0, 50).

"""
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 99

features = 20
obs = 501
middle_feature = features // 2

np.random.seed(SEED)
X = np.zeros((obs, features))
X[:, middle_feature] = np.arange(obs)
y = X[:, middle_feature] + np.random.uniform(0, 50, size=obs)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)

# Visualize ---
# import matplotlib.pyplot as plt
# plt.scatter(X[:, middle_feature], y, s=1)
