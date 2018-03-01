"""

    Test Data
    ~~~~~~~~~

"""
import numpy as np

features = 20
obs = 501

middle_feature = features // 2

X_train = np.zeros((obs, features))
X_train[:, middle_feature] = np.arange(obs)
y_train = X_train[:, middle_feature] + np.random.uniform(0, 50, size=obs)

X_test, y_test = X_train, y_train

# Visualize ---
# import matplotlib.pyplot as plt
# plt.scatter(X_train[:, middle_feature], y_train, s=1)
