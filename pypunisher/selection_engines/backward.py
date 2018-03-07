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
        features = [f for f in S if f != exclude] if exclude else S
        self._model.fit(self._X_train[:, features], self._y_train)
        return self._model.score(self._X_val[:, features], self._y_val)

    @staticmethod  # `self` captures the yield of **locals().
    def _backward_input_checks(self, S, n_features, min_change):
        if min_change is None and n_features is None:
            raise ValueError(
                "One, and only one, of `min_change` and `n_features` "
                "should be non-None."
            )
        if isinstance(n_features, int) and not 0 < n_features < len(S):
            raise ValueError(
                "If an int, `n_features` must be on (0, {}).".format(len(S))
            )
        if isinstance(n_features, float) and not 0 < n_features < 1:
            raise ValueError(
                "If a float, `n_features` must be on (0, 1)."
            )
        if min_change is not None and not isinstance(min_change, (int, float)):
            raise TypeError(
                "`min_change` must be of Type: None, int or float."
            )

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
        S = list(range(self._n_features))  # start with all features
        self._backward_input_checks(**locals())

        if isinstance(n_features, float):
            n_features = int(n_features * len(S))

        last_score = self._fit_and_score_back(S, exclude=None)
        worse_case = worse_case_bar(self._n_features, verbose=self._verbose)
        for _ in worse_case:
            worse_case.set_postfix(n_features=len(S), score=last_score)

            # 1. Hunt for the least predictive feature.
            best = None
            for j in S:
                score = self._fit_and_score_back(S, exclude=j)
                if score >= last_score and (best is None or score > best[1]):
                    best = (j, score)

            if not best:
                break  # Relent. Removing any `j` yielded a lower score.
            else:
                to_drop, best_new_score = best

            # 2a. Halting Based Blindly Based on `n_features`.
            if isinstance(n_features, int):
                S.remove(to_drop)  # blindly drop.
                last_score = best_new_score
                if len(S) == n_features:
                    break
                else:
                    continue  # i.e., ignore criteria below.

            # 2b. Halt if the change is not longer significant.
            if (best_new_score - last_score) < min_change:
                break
            else:
                S.remove(to_drop)
                last_score = best_new_score

            if len(S) == 1:  # continuing is futile.
                break

        return S
