"""

    Default Base for Testing Against
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from sklearn.linear_model import LinearRegression
from tests._test_data import X_train, y_train, X_val, y_val

DEFAULT_FORWARD_SELECTION_PARAMS = {
    'model': LinearRegression(), 'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val, 'verbose': False, 'criterion': None
}
