"""

     Information Criterion
     ~~~~~~~~~~~~~~~~~~~~~

"""
from numpy import log, ndarray, pi
from pypunisher._checks import model_check

def _get_coeffs(model, X_train, y_train):
    """

    Args:
        model : sklearn model object
            A fitted sklearn model.
        X_train : ndarray
            The data used to train `model`.
        y_train : 1d numpy array
            The response variable.

    Returns:
        n : int
            Number of samples
        k : int
            Number of features
        llf : float
            Maximized value of log likelihood function
    """
    model_check(model)
    n = X_train.shape[0]
    k = X_train.shape[1]
    y_pred = model.predict(X_train)
    rss = sum((y_train - y_pred) ** 2)
    llf = -(n/2)*log(2*pi) - (n/2)*log(rss/n) - n/2
    return n, k, llf


def aic(model, X_train, y_train):
    """Compute the Akaike Information Criterion (AIC)

    AIC's objective is to prevent model overfitting by adding a penalty 
    term which penalizes more compelx models. Its formal definition is:
        -2ln(L)+2*k
    where L is the maximized value of the likelihood function. A smaller AIC value suggests that the model is a better fit for the data.

    Args:
        model : A fitted sklearn model object)
            A fitted sklearn model.
        X_train : ndarray
            The data used to train `model`.
        y_train : 1d numpy array
            The response variable.

    Returns:
        aic: float
            AIC value if sample size is sufficient. 
            If n/k < 40 where n is the number of observations and k is the number of features, AICc gets returned to adjust for small sample size.
                  
    References:
        * https://en.wikipedia.org/wiki/Akaike_information_criterion
    """

    if not isinstance(X_train, ndarray):
        raise TypeError("`X_train` must be an ndarray.")
    if not isinstance(y_train, ndarray):
        raise TypeError("`y_train` must be an ndarray.")

    n, k, llf = _get_coeffs(model, X_train=X_train, y_train=y_train)
    aic = -2*llf +2*k
    if n/k < 40:
        return aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        return aic


def bic(model, X_train, y_train):
    """Compute the Bayesian Information Criterion (BIC)

    BIC's objective is to prevent model over-fitting by adding a penalty
    term which penalizes more complex models. Its formal definition is:
        -2ln(L)+ln(n)k
    where L is the maximized value of the likelihood function. A smaller BIC value suggests that the model is a better
    fit for the data.

    Args:
        model : sklearn model object)
            A fitted sklearn linear regression model.
        X_train : ndarray
            The data used to train `model`.
        y_train : 1d numpy array
            The response variable. 

    Returns:
        bic: float
            Bayesian Information Criterion value.

    References:
        * https://en.wikipedia.org/wiki/Bayesian_information_criterion

    """
    if not isinstance(X_train, ndarray):
        raise TypeError("`X_train` must be an ndarray.")
    if not isinstance(y_train, ndarray):
        raise TypeError("`y_train` must be an ndarray.")

    n, k, llf = _get_coeffs(model, X_train=X_train, y_train=y_train)
    bic = -2*llf+log(n)*k

    return bic
