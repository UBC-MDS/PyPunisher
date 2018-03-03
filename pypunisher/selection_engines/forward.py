"""

    Forward Selection Algorithm
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np
from pypunisher.metrics.criterion import compute_score
from pypunisher.selection_engines._utils import (get_n_features,
                                                 enforce_use_of_all_cpus,
                                                 worse_case_bar)


class ForwardSelection(object):
    """Forward Selection Algorithm.

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
        self._criterion = criterion  # ToDO: Not Implemented Yet.
        self._n_features = get_n_features(X_train)

    def _fit_and_score_forward(self, S, include):
        pass

    def _forward_break_criteria(self, S, j_score_dict, min_change, max_features):
        pass

    @staticmethod  # `self` captures yield of **locals().
    def _forward_input_checks(self, max_features):
        pass

    def forward(self, min_change=0.5, max_features=None):
        """Perform forward selection on a Sklearn model.

        Args:
            min_change : int or float, optional
                The smallest change to be considered significant.
                `n_features` must be None for `min_change` to operate.
            max_features : int
                the max. number of features to allow.

        Returns:
            S : list
              The column indices of `X_train` (and `X_val`) that denote the chosen features.
        """
        pass
