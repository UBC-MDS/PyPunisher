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
    n_features = model.coef_
    y_pred = model.predict(X_train)
    sse = sum((y_train - y_pred)**2)
    return 2*n_features - 2*np.log(sse)


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
    n_features = model.coef_
    n_observations = X_train.shape[0]
    
    y_pred = model.predict(X_train)
    sse = sum((y_train - y_pred)**2)
    return np.log(n_observations)*n_features-2*np.log(sse)