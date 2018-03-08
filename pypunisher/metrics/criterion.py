"""

     Information Criterion
     ~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np

def aic(model, X_train, y_train):
    """Compute the Akaike Information Criterion (AIC)

    Args:
        model : sklearn model object)
            A sklearn model.
        data : ndarray
            The data used to train `model`.

    Returns:
        float: ...

    References:
        * https://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    k = model.coef_
    n = X_train.shape[0]
    y_pred = model.predict(X_train)
    rss = sum((y_train - y_pred)**2)
    return n*np.log(rss) + 2*k


def bic(model, X_train, y_train):
    """Compute the Bayesian Information Criterion (BIC)

    Args:
        model : sklearn model object)
            A sklearn model.
        data : ndarray
            The data used to train `model`.

    Returns:
        float: ...

    References:
        * https://en.wikipedia.org/wiki/Bayesian_information_criterion

    """
    k = X_train.shape[1]
    n = X_train.shape[0]
    
    y_pred = model.predict(X_train)
    rss = sum((y_train - y_pred)**2)
    return n*np.log(rss)+np.log(n)*k