"""

    Unified Interface for Forward and Backward Selection Algorithms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from pypunisher.selection_engines.forward import ForwardSelection
from pypunisher.selection_engines.backward import BackwardSelection


class Selection(ForwardSelection, BackwardSelection):
    """Unified Forward and Backward Selection Class.

    Args:
        model : sklearn model
            any sklearn model with `.fit()`, `.predict()` and
            `.score()` methods.
        X_train ndarray:
            a 2D numpy array of (observations, features).
        y_train : ndarray:
            a 1D array of target classes for X_train.
        X_val : ndarray
            a 2D numpy array of (observations, features).
        y_val : ndarray
            a 1D array of target classes for X_validate.
        criterion : str or None
            model selection criterion.
            * 'aic': use Akaike Information Criterion.
            * 'bic': use Bayesian Information Criterion.
            * None: Use the default score built into the model
              object (i.e., call `.score()`).
        verbose : bool
            if True, print additional information as selection occurs.
            Defaults to True.

    Exposes:
        * pypunisher.selection_engines.forward.ForwardSelection.forward()
        * pypunisher.selection_engines.backward.BackwardSelection.backward()

    """

    def __init__(self, model, X_train, y_train,
                 X_val, y_val, criterion=None, verbose=True):
        super().__init__(model=model, X_train=X_train, y_train=y_train,
                         X_val=X_val, y_val=y_val, criterion=criterion,
                         verbose=verbose)
