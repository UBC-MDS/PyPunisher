"""

     Information Criterion
     ~~~~~~~~~~~~~~~~~~~~~

"""


def aic(X_train, y_train, model):
    """Compute the Akaike Information Criterion (AIC)

    Args:
        model (sklearn model object): ...
        X_train (ndarray): a 2D numpy array of (observations, features).
        y_train (ndarray): a 1D array of target classes for X_train.

    Returns:
        aic value (float)

    References:
        * https://en.wikipedia.org/wiki/Akaike_information_criterion

    """
    pass


def bic(X_train, y_train, model):
    """Compute the Bayesian Information Criterion (BIC)

    Args:
        model (sklearn model object): ...
        X_train (ndarray): a 2D numpy array of (observations, features).
        y_train (ndarray): a 1D array of target classes for X_train.

    Returns:
        bic value (float)

    References:
        * https://en.wikipedia.org/wiki/Bayesian_information_criterion

    """
    pass
