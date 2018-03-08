"""

     Information Criterion
     ~~~~~~~~~~~~~~~~~~~~~

"""
from numpy import log, ndarray


def _get_coeffs(model, X_train, y_train):
    """

    Args:
        model : sklearn model object)
            A sklearn model.
        X_train : ndarray
            The data used to train `model`.
        y_train : 1d numpy array
            The response variable.

    Returns:
        n : int
            Number of samples
        k : int
            Number of features
        rss : float
            Residual Sum of Squares

    """
    n = X_train.shape[0]
    k = X_train.shape[1]
    y_pred = model.predict(X_train)
    rss = sum((y_train - y_pred) ** 2)
    return n, k, rss


def aic(model, X_train, y_train):
    """Compute the Akaike Information Criterion (AIC)

    AIC's objective is to prevent model over-fitting by adding a penalty
    term which penalizes more complex models. Its formal definition is:
        -2ln(L) + 2*k
    where L is the maximized value of the likelihood function. We can approximate 
    -2ln(L) ≈ n*ln(RSS/n). A smaller AIC value suggests that the model is a better
    fit for the data.

    Args:
        model : sklearn model object)
            A sklearn model.
        X_train : ndarray
            The data used to train `model`.
        y_train : 1d numpy array
            The response variable.

    Returns:
        aic : float
            Akaike Information Criterion value if sample size is sufficient. 
            If n (number of observations)/k (number of features) < 40,
            AICc gets returned to adjust for small sample size.
                  
    References:
        * https://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    for method in ('fit', 'predict', 'score'):
        if not callable(getattr(model, method, None)):
            raise AttributeError(
                "Model does not have a callable method `{}`".format(method)
            )

    if not isinstance(X_train, ndarray):
        raise TypeError("`X_train` must be an ndarray.")
    if not isinstance(y_train, ndarray):
        raise TypeError("`y_train` must be an ndarray.")

    n, k, rss = _get_coeffs(model, X_train=X_train, y_train=y_train)
    aic = n * log(rss / n) + 2 * k

    if n / k < 40:
        # returns AICc for small sample sizes
        return aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        return aic


def bic(model, X_train, y_train):
    """Compute the Bayesian Information Criterion (BIC)

    BIC's objective is to prevent model over-fitting by adding a penalty
    term which penalizes more complex models. Its formal definition is:
        -2ln(L) + ln(n)k
    where L is the maximized value of the likelihood function. We can approximate 
    -2ln(L) ≈ n * ln(RSS/n). A smaller BIC value suggests that the model is a better
    fit for the data.

    Args:
        model : sklearn model object)
            A sklearn model.
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
    for method in ('fit', 'predict', 'score'):
        if not callable(getattr(model, method, None)):
            raise AttributeError

    if (not isinstance(X_train, ndarray)) or (not isinstance(y_train, ndarray)):
        raise TypeError

    n, k, rss = _get_coeffs(model, X_train=X_train, y_train=y_train)
    bic = n * log(rss / n) + log(n) * k
    return bic
