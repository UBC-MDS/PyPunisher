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
        X_train : ndarray
            The data used to train `model`.
        y_train : 1d numpy array
            The response variable.

    Returns:
        float: ...

    References:
        * https://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    n = X_train.shape[0]
    k = X_train.shape[1]
    y_pred = model.predict(X_train)
    rss = sum((y_train - y_pred)**2)
    aic = n*np.log(rss/n) + 2*k

    if n/k < 40:
        # returns AICc if sample size is small wrt k
        return aic + 2*k*(k+1)/(n-k-1)
    else:
        return aic


def bic(model, X_train, y_train):
    """Compute the Bayesian Information Criterion (BIC)

    Args:
        model : sklearn model object)
            A sklearn model.
        X_train : ndarray
            The data used to train `model`.
        y_train : 1d numpy array
            The response variable. 

    Returns:
        float: ...

    References:
        * https://en.wikipedia.org/wiki/Bayesian_information_criterion

    """


    n = X_train.shape[0]
    k = X_train.shape[1]
    
    y_pred = model.predict(X_train)
    rss = sum((y_train - y_pred)**2)
    return n*np.log(rss/n)+np.log(n)*k
