"""

    Backward Selection Algorithm
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from pypunisher.selection_engines._utils import (get_n_features,
                                                 enforce_use_of_all_cpus,
                                                 worse_case_bar)


class BackwardSelection(object):
    """Backward Selection Algorithm.

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

    """

    def __init__(self, model, X_train, y_train,
                 X_val, y_val, criterion=None, verbose=True):
        self._model = enforce_use_of_all_cpus(model)
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        self._verbose = verbose
        self._criterion = criterion  # ToDo: Not Implemented Yet.
        self._n_features = get_n_features(X_train)

    def _fit_and_score_back(self, S, exclude):
        pass

    @staticmethod  # `self` captures the yield of **locals().
    def _backward_input_checks(self, S, n_features, min_change):
        pass

    def backward(self, n_features=0.5, min_change=None):
        """Run Backward Selection Algorithm.

        Args:
            n_features : int or float
                The number of features to select.
                Floats will be regarded as proportions of the total
                that must lie on (0, 1).
                `min_change` must be None for `n_features` to operate.
            min_change : int or float, optional
                The smallest change to be considered significant.
                `n_features` must be None for `min_change` to operate.

        Returns:
            S : list
              The column indices of `X_train` (and `X_val`) that denote the chosen features.

        Raises:
            if `n_features` and `min_change` are both non-None.

        """
        pass
