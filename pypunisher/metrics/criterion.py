"""

     Information Criterion
     ~~~~~~~~~~~~~~~~~~~~~

"""
from numpy import log, ndarray, pi

def aic(model, X_train, y_train):
    """Compute the Akaike Information Criterion (AIC)

    AIC's objective is to prevent model overfitting by adding a penalty 
    term which penalizes more compelx models. Its formal definition is:
        -2ln(L)+2*k
    where L is the maximized value of the likelihood function. A smaller AIC value suggests that the model is a better fit for the data.

    Args:
        model : sklearn model object)
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

    if (not isinstance(X_train, ndarray)) or (not isinstance(y_train,ndarray)):
        raise TypeError

    n = X_train.shape[0]
    k = X_train.shape[1]
    y_pred = model.predict(X_train)
    rss = sum((y_train - y_pred)**2)
    llf = -(n/2)*log(2*pi) - (n/2)*log(rss/n) - n/2
    aic = -2*log(llf)+2*k

    if n/k < 40:
        return aic + 2*k*(k+1)/(n-k-1)
    else:
        return aic


def bic(model, X_train, y_train):
    """Compute the Bayesian Information Criterion (BIC)

    BIC's objective is to prevent mdoel overfitting by adding a penalty 
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
    if (not isinstance(X_train, ndarray)) or (not isinstance(y_train,ndarray)):
        raise TypeError

    n = X_train.shape[0]
    k = X_train.shape[1]
    y_pred = model.predict(X_train)
    rss = sum((y_train - y_pred)**2)
    llf = -(n/2)*log(2*pi) - (n/2)*log(rss/n) - n/2
    bic = -2*log(llf)+log(n)*k
    return bic
